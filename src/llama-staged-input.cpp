#include "llama-staged-input.h"

#include "llama-graph.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-model.h"
#include "llama-impl.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <system_error>

namespace {
bool readable(const ggml_tensor * table) {
    return table && table->buffer && table->data && ggml_backend_buffer_is_host(table->buffer) && ggml_is_contiguous(table) &&
        (table->type == GGML_TYPE_F32 || ggml_get_type_traits(table->type)->to_float);
}

void read_row(const ggml_tensor * table, int64_t row, float * dst) {
    if (row < 0 || row >= table->ne[1]) { throw std::runtime_error("staged embedding row is out of range"); }
    const auto * src = static_cast<const char *>(table->data) + row*table->nb[1];
    if (table->type == GGML_TYPE_F32) {
        std::memcpy(dst, src, table->ne[0]*sizeof(float));
    } else {
        ggml_get_type_traits(table->type)->to_float(src, dst, table->ne[0]);
    }
}

class staged_graph_input : public llm_graph_input_i {
public:
    staged_graph_input(llama_staged_inputs * stage, const llama_memory_hybrid_idx_context * memory, llm_graph_result * owner) : stage(stage), memory(memory), owner(owner), rows(stage->rows()) {}
    void set_input(const llama_ubatch * ubatch) override {
        stage->set_deferred(owner->get_source_program() != 0);
        stage->prepare(*ubatch, memory);
    }
    bool can_decode_sampled() const override { return stage->can_decode_sampled(); }
    bool can_reuse(const llm_graph_params & params) override {
        memory = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx);
        return params.staged_inputs == stage && params.ubatch.n_tokens == rows;
    }
private:
    llama_staged_inputs * stage;
    const llama_memory_hybrid_idx_context * memory;
    llm_graph_result * owner;
    const uint32_t rows;
};
}

std::unique_ptr<llama_staged_inputs> llama_staged_inputs::create(const llama_model & model, ggml_backend_t backend, bool prefetch, uint32_t n_tokens, bool sampled) {
    const auto & hp = model.hparams;
    if (n_tokens == 0 || (sampled && n_tokens != 1) || !readable(model.tok_embd) || !readable(model.per_layer_tok_embd) || hp.ple_ngram_size < 2 ||
        hp.ple_ngram_size > LLAMA_MAX_PLE_NGRAM || hp.ple_heads_per_ngram > LLAMA_MAX_PLE_HEADS / (hp.ple_ngram_size - 1) ||
        hp.ple_n_heads != (hp.ple_ngram_size - 1)*hp.ple_heads_per_ngram ||
        hp.ple_n_heads == 0 || hp.ple_n_heads > LLAMA_MAX_PLE_HEADS || model.per_layer_tok_embd->ne[0] != hp.ple_head_dim ||
        hp.n_embd_inp() != hp.n_embd || model.tok_embd->ne[0] != hp.n_embd || hp.f_embedding_scale != 0.0f ||
        uint64_t(n_tokens)*hp.n_embd > SIZE_MAX/sizeof(float) ||
        uint64_t(hp.ple_n_heads)*hp.ple_head_dim > SIZE_MAX/sizeof(float)/n_tokens) {
        return nullptr;
    }
    auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    auto get_api = reinterpret_cast<ggml_staged_input_get_api_t>(ggml_backend_reg_get_proc_address(reg, GGML_STAGED_INPUT_PROC));
    if (!get_api) { return nullptr; }
    auto result = std::unique_ptr<llama_staged_inputs>(new llama_staged_inputs(model, prefetch, n_tokens, sampled));
    result->consumed = reinterpret_cast<ggml_staged_input_consumed_t>(ggml_backend_reg_get_proc_address(reg, GGML_STAGED_INPUT_CONSUMED_PROC));
    result->api = get_api();
    if (!result->api) { return nullptr; }
    result->embedding = result->api->create(backend, size_t(n_tokens)*model.tok_embd->ne[0]*sizeof(float));
    result->ple = result->api->create(backend, size_t(n_tokens)*hp.ple_n_heads*hp.ple_head_dim*sizeof(float));
    if (!result->embedding || !result->ple) { return nullptr; }
    auto set_submit = reinterpret_cast<ggml_staged_input_set_submit_t>(ggml_backend_reg_get_proc_address(reg, GGML_STAGED_INPUT_SET_SUBMIT_PROC));
    if (set_submit) {
        const auto submit = +[](void * context) { static_cast<llama_staged_inputs *>(context)->submit(); };
        if (!set_submit(result->embedding, submit, result.get()) || !set_submit(result->ple, submit, result.get())) { return nullptr; }
        result->submit_supported = true;
    }
    try {
        result->worker = std::thread([stage = result.get()] { stage->run(); });
    } catch (const std::system_error &) {
        return nullptr;
    }
    return result;
}

llama_staged_inputs::~llama_staged_inputs() {
    submit();
    if (pending.valid()) { pending.wait(); }
    if (worker.joinable()) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        condition.notify_one();
        worker.join();
    }
    if (embedding) { api->destroy(embedding); }
    if (ple) { api->destroy(ple); }
}

void llama_staged_inputs::set_source(const llama_token * value, ggml_backend_event_t event) {
    GGML_ASSERT(sampled);
    finish();
    token = value;
    ready = event;
}

void llama_staged_inputs::set_rows(uint32_t rows) {
    finish();
    if (sampled || !rows || rows > capacity || !consumed || !consumed(embedding) || !consumed(ple)) {
        throw std::runtime_error("staged prefill backing is not available");
    }
    bounded_prefill = true;
    n_tokens = rows;
}

void llama_staged_inputs::set_lookahead(const llama_ubatch * ubatch, const llama_memory_hybrid_idx_context * memory) {
    finish();
    lookahead = {};
    if (!bounded_prefill || !ubatch || !memory || !ubatch->n_tokens || ubatch->n_tokens > capacity ||
            !ubatch->token || ubatch->is_mixed()) { return; }
    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        if (ubatch->n_seq_id[i] != 1 || ubatch->seq_id[i][0] != ubatch->seq_id[0][0] ||
                int64_t(ubatch->pos[i]) != int64_t(ubatch->pos[0]) + i) { return; }
    }
    known_chunk next;
    next.tokens.assign(ubatch->token, ubatch->token + ubatch->n_tokens);
    next.positions.assign(ubatch->pos, ubatch->pos + ubatch->n_tokens);
    next.sequences.assign(ubatch->n_tokens, ubatch->seq_id[0][0]);
    const uint32_t history = model.hparams.ple_ngram_size - 1;
    memory->get_attn()->get_prev_tokens(*ubatch, history, next.previous);
    // Future rows are not in KV yet. Use the known, consecutive prompt prefix.
    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        for (uint32_t j = 0; j < history; ++j) {
            const int64_t previous_row = int64_t(i) - (history - j);
            if (previous_row >= 0) { next.previous[size_t(i)*history + j] = next.tokens[previous_row]; }
        }
    }
    lookahead = std::move(next);
}

void llama_staged_inputs::collect_ple(const std::vector<llama_token> & tokens,
        const std::vector<llama_token> & previous, float * dst) const {
    const auto & hp = model.hparams;
    std::vector<int64_t> context(hp.ple_ngram_size, hp.ple_eos_token_id);
    std::vector<int64_t> rows(tokens.size()*hp.ple_n_heads);
    for (uint32_t i = 0; i < tokens.size(); ++i) {
        context[0] = tokens[i];
        bool cut = false;
        for (uint32_t s = 1; s < hp.ple_ngram_size; ++s) {
            const auto t = previous[size_t(i)*(hp.ple_ngram_size - 1) + hp.ple_ngram_size - 1 - s];
            cut = cut || t < 0 || t == int64_t(hp.ple_eos_token_id);
            context[s] = cut ? hp.ple_eos_token_id : t;
        }
        for (uint32_t n = 2; n <= hp.ple_ngram_size; ++n) {
            uint64_t mixed = uint64_t(context[0])*hp.ple_layer_multipliers[0];
            for (uint32_t j = 1; j < n; ++j) { mixed ^= uint64_t(context[j])*hp.ple_layer_multipliers[j]; }
            for (uint32_t g = 0; g < hp.ple_heads_per_ngram; ++g) {
                const auto h = (n - 2)*hp.ple_heads_per_ngram + g;
                const auto row = mixed % hp.ple_head_vocab_sizes[h] + hp.ple_head_offsets[h];
                if (row > INT64_MAX) { throw std::runtime_error("staged PLE row does not fit int64"); }
                rows[size_t(i)*hp.ple_n_heads + h] = int64_t(row);
            }
        }
    }
    if (prefetch) {
        std::vector<int32_t> indices(rows.size());
        for (size_t i = 0; i < rows.size(); ++i) {
            if (rows[i] > INT32_MAX) { throw std::runtime_error("staged PLE row does not fit int32"); }
            indices[i] = int32_t(rows[i]);
        }
        model.prefetch_rows(model.per_layer_tok_embd, indices.data(), indices.size());
    }
    for (size_t i = 0; i < rows.size(); ++i) { read_row(model.per_layer_tok_embd, rows[i], dst + i*hp.ple_head_dim); }
}

void llama_staged_inputs::prepare(const llama_ubatch & ubatch, const llama_memory_hybrid_idx_context * memory) {
    finish();
    GGML_ASSERT(ubatch.n_tokens == n_tokens && (sampled ? token && ready : ubatch.token && !ubatch.is_mixed()));
    for (uint32_t i = 0; i < n_tokens; ++i) { GGML_ASSERT(ubatch.n_seq_id[i] == 1); }
    if (bounded_prefill && (!consumed(embedding) || !consumed(ple))) {
        throw std::runtime_error("staged prefill reader has not consumed its input");
    }
    std::vector<llama_token> previous;
    memory->get_attn()->get_prev_tokens(ubatch, model.hparams.ple_ngram_size - 1, previous);
    std::vector<llama_token> values(n_tokens);
    if (!sampled) { std::copy_n(ubatch.token, n_tokens, values.data()); }
    const auto prepare_embedding = [this](const std::vector<llama_token> & tokens) {
        auto * embd = static_cast<float *>(api->data(embedding));
        for (size_t i = 0; i < tokens.size(); ++i) {
            read_row(model.tok_embd, tokens[i], embd + i*model.tok_embd->ne[0]);
        }
        api->publish(embedding);
    };
    bool cached = bounded_prefill && collected.tokens == values && collected.previous == previous &&
        collected.positions.size() == n_tokens && collected.sequences.size() == n_tokens &&
        collected.embedding.size() == size_t(n_tokens)*model.tok_embd->ne[0] &&
        collected.ple.size() == size_t(n_tokens)*model.hparams.ple_n_heads*model.hparams.ple_head_dim;
    for (uint32_t i = 0; cached && i < n_tokens; ++i) {
        cached = collected.positions[i] == ubatch.pos[i] && collected.sequences[i] == ubatch.seq_id[i][0];
    }
    if (bounded_prefill) {
        const size_t width = size_t(model.tok_embd->ne[0]) + size_t(model.hparams.ple_n_heads)*model.hparams.ple_head_dim;
        LLAMA_LOG_INFO("staged-prefill: rows=%u capacity=%u cached=%u next_rows=%zu pinned_bytes=%zu lookahead_bytes=%zu\n",
            n_tokens, capacity, unsigned(cached), lookahead.tokens.size(), size_t(capacity)*width*sizeof(float),
            lookahead.tokens.size()*width*sizeof(float));
    }
    std::exception_ptr embedding_failure;
    if (cached) {
        std::memcpy(api->data(embedding), collected.embedding.data(), collected.embedding.size()*sizeof(float));
        std::memcpy(api->data(ple), collected.ple.data(), collected.ple.size()*sizeof(float));
        api->publish(embedding);
        api->publish(ple);
    } else if (!sampled) {
        try { prepare_embedding(values); } catch (...) { embedding_failure = std::current_exception(); }
    }
    collected = {};
    auto collect = [this, previous = std::move(previous), values = std::move(values),
            next_chunk = std::move(lookahead), prepare_embedding, embedding_failure, cached]() mutable {
        bool embedding_published = cached || (!sampled && !embedding_failure);
        bool ple_published = cached;
        try {
            if (embedding_failure) { std::rethrow_exception(embedding_failure); }
            if (sampled) {
                ggml_backend_event_synchronize(ready);
                values[0] = *token;
                prepare_embedding(values);
                embedding_published = true;
            }
            if (!cached) {
                collect_ple(values, previous, static_cast<float *>(api->data(ple)));
                api->publish(ple);
                ple_published = true;
            }
            if (!next_chunk.tokens.empty()) {
                next_chunk.embedding.resize(next_chunk.tokens.size()*model.tok_embd->ne[0]);
                next_chunk.ple.resize(next_chunk.tokens.size()*model.hparams.ple_n_heads*model.hparams.ple_head_dim);
                for (size_t i = 0; i < next_chunk.tokens.size(); ++i) {
                    read_row(model.tok_embd, next_chunk.tokens[i], next_chunk.embedding.data() + i*model.tok_embd->ne[0]);
                }
                collect_ple(next_chunk.tokens, next_chunk.previous, next_chunk.ple.data());
                collected = std::move(next_chunk);
            }
        } catch (...) {
            // Release queued waits before propagating a staging failure to the caller.
            if (!embedding_published) { api->publish(embedding); }
            if (!ple_published) { api->publish(ple); }
            throw;
        }
    };
    auto next = std::packaged_task<void()>(std::move(collect));
    pending = next.get_future();
    const bool submit_now = bounded_prefill || !deferred || !submit_supported;
    {
        std::lock_guard<std::mutex> lock(mutex);
        task = std::move(next);
        task_submitted = submit_now;
    }
    if (submit_now) { condition.notify_one(); }
}

void llama_staged_inputs::submit() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (!task.valid() || task_submitted) { return; }
        task_submitted = true;
    }
    condition.notify_one();
}

void llama_staged_inputs::run() {
    for (;;) {
        std::packaged_task<void()> next;
        {
            std::unique_lock<std::mutex> lock(mutex);
            condition.wait(lock, [this] { return stopping || (task.valid() && task_submitted); });
            if (stopping) { return; }
            next = std::move(task);
            task_submitted = false;
        }
        next();
    }
}

void llama_staged_inputs::finish() {
    if (pending.valid()) { submit(); pending.get(); }
}

ggml_tensor * llama_staged_inputs::build_embedding(ggml_context * ctx, llm_graph_result * result) {
    auto inp = std::make_unique<llm_graph_input_embd>(model.hparams.n_embd_inp());
    inp->tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp->tokens, "inp_tokens");
    ggml_set_input(inp->tokens);
    inp->embd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, inp->n_embd, n_tokens);
    ggml_set_name(inp->embd, "inp_embd");
    ggml_set_input(inp->embd);
    result->t_inp_tokens = inp->tokens;
    auto * tensor = api->build(embedding, ctx, inp->tokens, int64_t(n_tokens)*model.tok_embd->ne[0]);
    if (n_tokens > 1) { tensor = ggml_reshape_2d(ctx, tensor, model.tok_embd->ne[0], n_tokens); }
    ggml_set_name(tensor, "staged_token_embedding");
    result->t_inp_embd = tensor;
    result->add_input(std::move(inp));
    return tensor;
}

ggml_tensor * llama_staged_inputs::build_ple(ggml_context * ctx, llm_graph_result * result, const llama_memory_hybrid_idx_context * memory) {
    result->add_input(std::make_unique<staged_graph_input>(this, memory, result));
    const int64_t width = int64_t(model.hparams.ple_n_heads)*model.hparams.ple_head_dim;
    auto * tensor = api->build(ple, ctx, nullptr, width*n_tokens);
    if (n_tokens > 1) { tensor = ggml_reshape_2d(ctx, tensor, width, n_tokens); }
    ggml_set_name(tensor, "staged_ple_embedding");
    return tensor;
}
