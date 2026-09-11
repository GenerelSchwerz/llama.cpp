#include "llama-staged-input.h"

#include "llama-graph.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-model.h"

#include <cstring>
#include <stdexcept>

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
    staged_graph_input(llama_staged_inputs * stage, const llama_memory_hybrid_idx_context * memory) : stage(stage), memory(memory) {}
    void set_input(const llama_ubatch * ubatch) override { stage->prepare(*ubatch, memory); }
    bool can_decode_sampled() const override { return true; }
    bool can_reuse(const llm_graph_params & params) override {
        memory = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx);
        return params.staged_inputs == stage && params.ubatch.n_tokens == 1;
    }
private:
    llama_staged_inputs * stage;
    const llama_memory_hybrid_idx_context * memory;
};
}

std::unique_ptr<llama_staged_inputs> llama_staged_inputs::create(const llama_model & model, ggml_backend_t backend, bool prefetch) {
    const auto & hp = model.hparams;
    if (!readable(model.tok_embd) || !readable(model.per_layer_tok_embd) || hp.ple_ngram_size < 2 ||
        hp.ple_ngram_size > LLAMA_MAX_PLE_NGRAM || hp.ple_heads_per_ngram > LLAMA_MAX_PLE_HEADS / (hp.ple_ngram_size - 1) ||
        hp.ple_n_heads != (hp.ple_ngram_size - 1)*hp.ple_heads_per_ngram ||
        hp.ple_n_heads == 0 || hp.ple_n_heads > LLAMA_MAX_PLE_HEADS || model.per_layer_tok_embd->ne[0] != hp.ple_head_dim ||
        hp.n_embd_inp() != hp.n_embd || model.tok_embd->ne[0] != hp.n_embd || hp.f_embedding_scale != 0.0f) {
        return nullptr;
    }
    auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    auto get_api = reinterpret_cast<ggml_staged_input_get_api_t>(ggml_backend_reg_get_proc_address(reg, GGML_STAGED_INPUT_PROC));
    if (!get_api) { return nullptr; }
    auto result = std::unique_ptr<llama_staged_inputs>(new llama_staged_inputs(model, prefetch));
    result->api = get_api();
    if (!result->api) { return nullptr; }
    result->embedding = result->api->create(backend, model.tok_embd->ne[0]*sizeof(float));
    result->ple = result->api->create(backend, size_t(hp.ple_n_heads)*hp.ple_head_dim*sizeof(float));
    if (!result->embedding || !result->ple) { return nullptr; }
    return result;
}

llama_staged_inputs::~llama_staged_inputs() {
    if (pending.valid()) { pending.wait(); }
    if (embedding) { api->destroy(embedding); }
    if (ple) { api->destroy(ple); }
}

void llama_staged_inputs::set_source(const llama_token * value, ggml_backend_event_t event) {
    finish();
    token = value;
    ready = event;
}

void llama_staged_inputs::prepare(const llama_ubatch & ubatch, const llama_memory_hybrid_idx_context * memory) {
    finish();
    GGML_ASSERT(ubatch.n_tokens == 1 && token && ready);
    std::vector<llama_token> previous;
    memory->get_attn()->get_prev_tokens(ubatch, model.hparams.ple_ngram_size - 1, previous);
    pending = std::async(std::launch::async, [this, previous = std::move(previous)]() {
        bool embedding_published = false;
        try {
            ggml_backend_event_synchronize(ready);
            const auto value = *token;
            read_row(model.tok_embd, value, static_cast<float *>(api->data(embedding)));
            api->publish(embedding);
            embedding_published = true;

            const auto & hp = model.hparams;
            std::vector<int64_t> context(hp.ple_ngram_size, hp.ple_eos_token_id);
            context[0] = value;
            bool cut = false;
            for (uint32_t s = 1; s < hp.ple_ngram_size; ++s) {
                const auto t = previous[hp.ple_ngram_size - 1 - s];
                cut = cut || t < 0 || t == int64_t(hp.ple_eos_token_id);
                if (!cut) { context[s] = t; }
            }
            auto * dst = static_cast<float *>(api->data(ple));
            if (prefetch) {
                int32_t rows[LLAMA_MAX_PLE_HEADS];
                for (uint32_t n = 2; n <= hp.ple_ngram_size; ++n) {
                    uint64_t mixed = uint64_t(context[0])*hp.ple_layer_multipliers[0];
                    for (uint32_t j = 1; j < n; ++j) { mixed ^= uint64_t(context[j])*hp.ple_layer_multipliers[j]; }
                    for (uint32_t g = 0; g < hp.ple_heads_per_ngram; ++g) {
                        const auto h = (n - 2)*hp.ple_heads_per_ngram + g;
                        const auto row = mixed % hp.ple_head_vocab_sizes[h] + hp.ple_head_offsets[h];
                        if (row > INT32_MAX) { throw std::runtime_error("staged PLE row does not fit int32"); }
                        rows[h] = int32_t(row);
                    }
                }
                model.prefetch_rows(model.per_layer_tok_embd, rows, hp.ple_n_heads);
                for (uint32_t h = 0; h < hp.ple_n_heads; ++h) {
                    read_row(model.per_layer_tok_embd, rows[h], dst + h*hp.ple_head_dim);
                }
            } else {
                for (uint32_t n = 2; n <= hp.ple_ngram_size; ++n) {
                    uint64_t mixed = uint64_t(context[0])*hp.ple_layer_multipliers[0];
                    for (uint32_t j = 1; j < n; ++j) { mixed ^= uint64_t(context[j])*hp.ple_layer_multipliers[j]; }
                    for (uint32_t g = 0; g < hp.ple_heads_per_ngram; ++g) {
                        const auto h = (n - 2)*hp.ple_heads_per_ngram + g;
                        const auto row = mixed % hp.ple_head_vocab_sizes[h] + hp.ple_head_offsets[h];
                        read_row(model.per_layer_tok_embd, row, dst + h*hp.ple_head_dim);
                    }
                }
            }
            api->publish(ple);
        } catch (...) {
            // Release queued waits before propagating a staging failure to the caller.
            if (!embedding_published) { api->publish(embedding); }
            api->publish(ple);
            throw;
        }
    });
}

void llama_staged_inputs::finish() {
    if (pending.valid()) { pending.get(); }
}

ggml_tensor * llama_staged_inputs::build_embedding(ggml_context * ctx, llm_graph_result * result) {
    auto inp = std::make_unique<llm_graph_input_embd>(model.hparams.n_embd_inp());
    inp->tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(inp->tokens, "inp_tokens");
    ggml_set_input(inp->tokens);
    inp->embd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, inp->n_embd, 1);
    ggml_set_name(inp->embd, "inp_embd");
    ggml_set_input(inp->embd);
    result->t_inp_tokens = inp->tokens;
    auto * tensor = api->build(embedding, ctx, inp->tokens, model.tok_embd->ne[0]);
    ggml_set_name(tensor, "staged_token_embedding");
    result->t_inp_embd = tensor;
    result->add_input(std::move(inp));
    return tensor;
}

ggml_tensor * llama_staged_inputs::build_ple(ggml_context * ctx, llm_graph_result * result, const llama_memory_hybrid_idx_context * memory) {
    result->add_input(std::make_unique<staged_graph_input>(this, memory));
    auto * tensor = api->build(ple, ctx, nullptr, int64_t(model.hparams.ple_n_heads)*model.hparams.ple_head_dim);
    ggml_set_name(tensor, "staged_ple_embedding");
    return tensor;
}
