#include "llama-draft-vocab.h"

#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-mmap.h"
#include "llama-vocab.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "gguf.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace {
const char * prefix = "mtp_draft_vocab.";

void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(std::string("MTP draft vocabulary: ") + message);
    }
}

std::vector<int32_t> specials(const llama_vocab & vocab) {
    return { vocab.token_bos(), vocab.token_eos(), vocab.token_eot(), vocab.token_eom(), vocab.token_unk(),
             vocab.token_sep(), vocab.token_nl(), vocab.token_pad(), vocab.token_mask(), vocab.token_prefix(),
             vocab.token_middle(), vocab.token_suffix(), vocab.token_fim_pre(), vocab.token_fim_suf(),
             vocab.token_fim_mid(), vocab.token_fim_pad(), vocab.token_fim_rep(), vocab.token_fim_sep() };
}

std::vector<int32_t> eogs(const llama_vocab & vocab) {
    std::vector<int32_t> result;
    for (uint32_t id = 0; id < vocab.n_tokens(); ++id) {
        if (vocab.is_eog(id)) {
            result.push_back(id);
        }
    }
    return result;
}

void check_ids(const llama_vocab & vocab, const int32_t * ids, size_t count) {
    require(vocab.n_tokens() > 0 && vocab.n_tokens() <= INT32_MAX, "unsupported token ID extent");
    require(ids && count > 0 && count <= vocab.n_tokens(), "empty or oversized selection");
    std::vector<bool> included(vocab.n_tokens());
    for (size_t i = 0; i < count; ++i) {
        require(ids[i] >= 0 && uint32_t(ids[i]) < vocab.n_tokens(), "selected token is out of range");
        require(!included[ids[i]], "duplicate selected token");
        included[ids[i]] = true;
    }
    for (auto id : eogs(vocab)) {
        require(included[id], "selection excludes an end-of-generation token");
    }
}

int64_t key(const gguf_context * file, const char * name, gguf_type type) {
    const auto id = gguf_find_key(file, (std::string(prefix) + name).c_str());
    require(id >= 0 && gguf_get_kv_type(file, id) == type, "missing key or incorrect key type");
    return id;
}

int64_t array(const gguf_context * file, const char * name, gguf_type type, size_t count) {
    const auto id = key(file, name, GGUF_TYPE_ARRAY);
    require(gguf_get_arr_type(file, id) == type && gguf_get_arr_n(file, id) == count, "incorrect array type or length");
    return id;
}

void check_array(const gguf_context * file, const char * name, const std::vector<int32_t> & values) {
    const auto id = array(file, name, GGUF_TYPE_INT32, values.size());
    require(values.empty() || !std::memcmp(gguf_get_arr_data(file, id), values.data(), values.size() * sizeof(int32_t)), "special token identity differs");
}

void set_array(gguf_context * file, const char * name, gguf_type type, const void * data, size_t count) {
    gguf_set_arr_data(file, (std::string(prefix) + name).c_str(), type, data, count);
}

void warn_head(const ggml_tensor * head, bool & warned, const char * reason) {
    if (!warned) {
        LLAMA_LOG_WARN("MTP draft vocabulary: full projection for %s: %s\n", head ? head->name : "output", reason);
        warned = true;
    }
}

ggml_backend_dev_t head_device(ggml_backend_buffer_type_t buft) {
    // The standard CPU buffer has no device pointer in GGML.
    return buft == ggml_backend_cpu_buffer_type() ? ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU) : ggml_backend_buft_get_device(buft);
}
}

bool llama_draft_vocab_write(const llama_vocab & vocab, const int32_t * ids, size_t count, const char * path) {
    check_ids(vocab, ids, count);
    require(path && *path, "missing output path");
    gguf_context_ptr file(gguf_init_empty());
    require(bool(file), "metadata allocation failed");
    gguf_set_val_u32(file.get(), "mtp_draft_vocab.version", 1);
    gguf_set_val_u32(file.get(), "mtp_draft_vocab.vocab_type", vocab.get_type());
    gguf_set_val_str(file.get(), "mtp_draft_vocab.tokenizer_model", vocab.get_tokenizer_model().c_str());
    gguf_set_val_str(file.get(), "mtp_draft_vocab.tokenizer_pre", vocab.get_tokenizer_pre().c_str());
    std::vector<uint8_t> bytes;
    std::vector<uint64_t> offsets = {0};
    std::vector<int32_t> attributes;
    for (uint32_t id = 0; id < vocab.n_tokens(); ++id) {
        const auto & token = vocab.get_token_data(id);
        bytes.insert(bytes.end(), token.text.begin(), token.text.end());
        offsets.push_back(bytes.size());
        attributes.push_back(token.attr);
    }
    const auto special = specials(vocab);
    const auto eog = eogs(vocab);
    set_array(file.get(), "token_bytes", GGUF_TYPE_UINT8, bytes.data(), bytes.size());
    set_array(file.get(), "token_offsets", GGUF_TYPE_UINT64, offsets.data(), offsets.size());
    set_array(file.get(), "token_attributes", GGUF_TYPE_INT32, attributes.data(), attributes.size());
    set_array(file.get(), "special_tokens", GGUF_TYPE_INT32, special.data(), special.size());
    set_array(file.get(), "eog_tokens", GGUF_TYPE_INT32, eog.data(), eog.size());
    set_array(file.get(), "selected_tokens", GGUF_TYPE_INT32, ids, count);
    return gguf_write_to_file(file.get(), path, true);
}

llama_draft_vocab::llama_draft_vocab(const llama_model & model, const char * path) {
    const auto & vocab = model.vocab;
    require(path && *path && vocab.n_tokens() > 0 && vocab.n_tokens() <= INT32_MAX, "invalid vocabulary or path");
    size_t token_bytes = 0;
    for (uint32_t id = 0; id < vocab.n_tokens(); ++id) {
        const size_t size = vocab.get_token_data(id).text.size();
        require(size <= SIZE_MAX - token_bytes, "token byte size overflow");
        token_bytes += size;
    }
    const auto special = specials(vocab);
    const auto eog = eogs(vocab);
    const uint64_t max_file_bytes = uint64_t(token_bytes) + uint64_t(vocab.n_tokens()) * 20 +
            vocab.get_tokenizer_model().size() + vocab.get_tokenizer_pre().size() + special.size() * sizeof(int32_t) + 1024;
    require(max_file_bytes >= token_bytes, "sidecar size overflow");
    llama_file input(path, "rb");
    require(input.size() <= max_file_bytes, "sidecar exceeds model-derived metadata bound");
    std::vector<uint8_t> contents(input.size());
    input.read_raw(contents.data(), contents.size());
    gguf_context_ptr file(gguf_init_from_buffer(contents.data(), contents.size(), {true, nullptr}));
    require(bool(file) && gguf_get_n_tensors(file.get()) == 0 && gguf_get_n_kv(file.get()) == 10, "cannot load versioned metadata-only GGUF sidecar");
    require(gguf_get_val_u32(file.get(), key(file.get(), "version", GGUF_TYPE_UINT32)) == 1, "unsupported sidecar version");
    require(gguf_get_val_u32(file.get(), key(file.get(), "vocab_type", GGUF_TYPE_UINT32)) == uint32_t(vocab.get_type()), "vocabulary type differs");
    require(gguf_get_val_str(file.get(), key(file.get(), "tokenizer_model", GGUF_TYPE_STRING)) == vocab.get_tokenizer_model(), "tokenizer model differs");
    require(gguf_get_val_str(file.get(), key(file.get(), "tokenizer_pre", GGUF_TYPE_STRING)) == vocab.get_tokenizer_pre(), "tokenizer preprocessor differs");
    const auto bytes_id = array(file.get(), "token_bytes", GGUF_TYPE_UINT8, token_bytes);
    const auto offsets_id = array(file.get(), "token_offsets", GGUF_TYPE_UINT64, size_t(vocab.n_tokens()) + 1);
    const auto attributes_id = array(file.get(), "token_attributes", GGUF_TYPE_INT32, vocab.n_tokens());
    const auto * bytes = static_cast<const uint8_t *>(gguf_get_arr_data(file.get(), bytes_id));
    const auto * offsets = static_cast<const uint64_t *>(gguf_get_arr_data(file.get(), offsets_id));
    const auto * attributes = static_cast<const int32_t *>(gguf_get_arr_data(file.get(), attributes_id));
    require(offsets[0] == 0 && offsets[vocab.n_tokens()] == token_bytes, "token byte extent differs");
    for (uint32_t id = 0; id < vocab.n_tokens(); ++id) {
        const auto & token = vocab.get_token_data(id);
        require(offsets[id] <= token_bytes && offsets[id + 1] >= offsets[id] && offsets[id + 1] <= token_bytes &&
                offsets[id + 1] - offsets[id] == token.text.size(), "token byte offsets differ");
        require(attributes[id] == int32_t(token.attr) && (token.text.empty() ||
                !std::memcmp(bytes + offsets[id], token.text.data(), token.text.size())), "ordered vocabulary differs");
    }
    check_array(file.get(), "special_tokens", special);
    check_array(file.get(), "eog_tokens", eog);
    const auto ids_id = key(file.get(), "selected_tokens", GGUF_TYPE_ARRAY);
    const auto count = gguf_get_arr_n(file.get(), ids_id);
    require(gguf_get_arr_type(file.get(), ids_id) == GGUF_TYPE_INT32 && count > 0 && count <= vocab.n_tokens(), "invalid selection array");
    const auto * ids = static_cast<const int32_t *>(gguf_get_arr_data(file.get(), ids_id));
    check_ids(vocab, ids, count);
    selected.assign(ids, ids + count);
    require(contents.size() == gguf_get_meta_size(file.get()), "sidecar has trailing or missing bytes");
    LLAMA_LOG_INFO("MTP draft vocabulary: opt-in selection %zu/%u; normal sampler/confidence policy\n", selected.size(), vocab.n_tokens());
    const auto prepare_head = [&](const ggml_tensor * head) {
        if (head && head->ne[1] == vocab.n_tokens()) {
            try {
                prepare(head);
            } catch (const std::bad_alloc &) {
                warn_head(head, heads[head].warned, "temporary logical export does not fit");
            }
        }
    };
    prepare_head(model.output);
    for (const auto & layer : model.layers) {
        prepare_head(layer.nextn.shared_head_head);
    }
}

llama_draft_vocab::head_state & llama_draft_vocab::prepare(const ggml_tensor * head) {
    const auto existing = heads.find(head);
    if (existing != heads.end()) {
        return existing->second;
    }
    auto & state = heads[head];
    if (!head || head->ne[0] <= 0 || head->ne[1] <= 0 || head->ne[2] != 1 || head->ne[3] != 1 ||
            !head->buffer || !head->buffer->iface.get_tensor || head->extra || head->view_src || !ggml_is_contiguous(head)) {
        warn_head(head, state.warned, "logical row export is unproved");
        return state;
    }
    const auto buft = ggml_backend_buffer_get_type(head->buffer);
    const auto device = head_device(buft);
    if (!device || ggml_backend_buft_is_meta(buft) || buft != ggml_backend_dev_buffer_type(device)) {
        warn_head(head, state.warned, "repacked or split storage is unproved");
        return state;
    }
    const size_t row_bytes = ggml_row_size(head->type, head->ne[0]);
    if (!row_bytes || head->nb[1] != row_bytes || uint64_t(head->ne[1]) > SIZE_MAX / row_bytes ||
            selected.size() > SIZE_MAX / row_bytes || ggml_nbytes(head) != size_t(head->ne[1]) * row_bytes) {
        warn_head(head, state.warned, "logical row size differs");
        return state;
    }
    ggml_context_ptr context(ggml_init({2 * ggml_tensor_overhead(), nullptr, true}));
    if (!context) {
        warn_head(head, state.warned, "head metadata does not fit");
        return state;
    }
    auto * weight = ggml_new_tensor_2d(context.get(), head->type, head->ne[0], selected.size());
    auto * ids = ggml_new_tensor_1d(context.get(), GGML_TYPE_I64, selected.size());
    ggml_set_name(weight, "mtp_compact_head");
    ggml_set_name(ids, "mtp_compact_ids");
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors_from_buft(context.get(), buft));
    if (!buffer) {
        warn_head(head, state.warned, "compact backing does not fit");
        return state;
    }
    std::vector<uint8_t> full(ggml_nbytes(head)), compact(selected.size() * row_bytes);
    std::vector<int64_t> map(selected.begin(), selected.end());
    ggml_backend_tensor_get(head, full.data(), 0, full.size());
    for (size_t i = 0; i < selected.size(); ++i) {
        std::memcpy(compact.data() + i * row_bytes, full.data() + size_t(selected[i]) * row_bytes, row_bytes);
    }
    ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(weight, compact.data(), 0, compact.size());
    ggml_backend_tensor_set(ids, map.data(), 0, map.size() * sizeof(int64_t));
    state.weight = weight;
    state.ids = ids;
    state.context = std::move(context);
    state.buffer = std::move(buffer);
    LLAMA_LOG_INFO("MTP draft vocabulary: head=%s type=%s selected=%zu compact_bytes=%zu backing_bytes=%zu\n",
            head->name, ggml_type_name(head->type), selected.size(), compact.size(), ggml_backend_buffer_get_size(state.buffer.get()));
    return state;
}

void llama_draft_vocab::apply(const llama_model & model, const llm_graph_params & params) {
    if (params.is_reserve || !params.n_outputs || params.cparams.ctx_type != LLAMA_CONTEXT_TYPE_MTP) {
        return;
    }
    auto * res = params.res;
    auto * logits = res->t_logits;
    const auto unsupported = [&](const char * reason) { warn_head(nullptr, boundary_warned, reason); };
    if (!logits || logits->op != GGML_OP_MUL_MAT || logits->type != GGML_TYPE_F32 || logits->ne[0] != model.vocab.n_tokens() ||
            logits->ne[2] != 1 || logits->ne[3] != 1 || logits->data || logits->buffer || logits->view_src ||
            logits == res->t_embd || logits == res->t_h_nextn || params.cparams.cb_eval || (params.loras && !params.loras->empty())) {
        unsupported("final projection or adapter equivalence is unproved");
        return;
    }
    auto * head = logits->src[0];
    bool identity = head && head == model.output;
    for (const auto & layer : model.layers) {
        identity |= head && head == layer.nextn.shared_head_head;
    }
    if (!identity || !logits->src[1] || head->ne[1] != model.vocab.n_tokens()) {
        unsupported("output head identity differs");
        return;
    }
    auto * graph = res->gf;
    bool found = false;
    for (int i = 0; i < graph->n_nodes; ++i) {
        const auto * node = graph->nodes[i];
        found |= node == logits;
        for (auto * input : node->src) {
            if (input == logits || node->view_src == logits) {
                unsupported("output projection already has a consumer");
                return;
            }
        }
    }
    if (!found || graph->n_nodes + 8 >= graph->size || graph->n_leafs + 3 >= graph->size) {
        unsupported("graph capacity or final root is unproved");
        return;
    }
    auto & state = prepare(head);
    if (!state.weight) {
        return;
    }
    auto * ctx = res->get_ctx();
    const int64_t rows = logits->ne[1];
    auto * projected = ggml_mul_mat(ctx, state.weight, logits->src[1]);
    std::memcpy(projected->op_params, logits->op_params, sizeof(logits->op_params));
    auto * mask = ggml_fill(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, logits->ne[0], rows), -INFINITY);
    auto * mapped = ggml_set_rows(ctx, mask, ggml_reshape_3d(ctx, projected, 1, selected.size(), rows),
            ggml_reshape_3d(ctx, state.ids, selected.size(), 1, 1));
    auto * output = ggml_reshape_2d(ctx, mapped, logits->ne[0], rows);
    const auto backend = ggml_backend_sched_get_tensor_backend(params.sched, logits);
    const auto device = backend ? ggml_backend_get_device(backend) : head_device(ggml_backend_buffer_get_type(state.buffer.get()));
    if (!device || !ggml_backend_dev_supports_op(device, projected) || !ggml_backend_dev_supports_op(device, mask) ||
            !ggml_backend_dev_supports_op(device, mapped)) {
        warn_head(head, state.warned, "projection or mapping is unsupported by this backend");
        return;
    }
    std::vector<ggml_tensor *> nodes(graph->nodes, graph->nodes + graph->n_nodes);
    std::vector<ggml_tensor *> leaves(graph->leafs, graph->leafs + graph->n_leafs);
    ggml_set_name(projected, "mtp_compact_projection");
    ggml_set_name(output, logits->name);
    if (backend) {
        ggml_backend_sched_set_tensor_backend(params.sched, projected, backend);
        ggml_backend_sched_set_tensor_backend(params.sched, mask, backend);
        ggml_backend_sched_set_tensor_backend(params.sched, mapped, backend);
    }
    // Explicit input leaves and order-only nodes must survive graph reconstruction.
    ggml_graph_clear(graph);
    for (auto * leaf : leaves) {
        ggml_build_forward_order(graph, leaf);
    }
    for (auto * node : nodes) {
        if (node != logits) {
            ggml_build_forward_order(graph, node);
        }
    }
    ggml_build_forward_expand(graph, output);
    res->t_logits = output;
}
