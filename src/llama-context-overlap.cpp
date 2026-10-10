#include "llama-context.h"

#include "llama-impl.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-model.h"

bool llama_context::can_decode_sampled_host() const {
    const auto * hybrid = dynamic_cast<const llama_memory_hybrid_idx *>(memory.get());
    return model.arch == LLM_ARCH_QWEN4EXP && n_seq_max() == 1 && !shared_workspace_peer() && hybrid &&
        hybrid->llama_memory_hybrid::can_decode_sampled() &&
        (!hybrid->get_mem_idx() || hybrid->get_mem_idx()->can_decode_sampled());
}

int32_t llama_context::decode_sampled_host(
        const llama_sampled_decode_item * items,
        int32_t n_items,
        const std::vector<ggml_tensor *> & sources,
        llama_token * previous) {
    synchronize();

    std::vector<llama_token> tokens(n_items);
    std::vector<llama_pos> positions(n_items);
    std::vector<llama_seq_id> seq_ids(n_items);
    for (int32_t i = 0; i < n_items; ++i) {
        ggml_backend_tensor_get(sources[i], &tokens[i], 0, sizeof(llama_token));
        previous[i] = tokens[i];
        positions[i] = items[i].pos;
        seq_ids[i] = items[i].seq_id;
    }

    // PLE hashes and KV token history need the real tokens before the next graph is queued.
    llama_batch_ext ext(this);
    for (int32_t i = 0; i < n_items; ++i) {
        const auto idx = ext.add_token(seq_ids[i]);
        if (idx < 0 || !ext.set_token_id(idx, tokens[i]) || !ext.set_token_pos(idx, &positions[i]) ||
                !ext.set_output(idx, true) || !ext.set_phase(idx, LLAMA_BATCH_PHASE_GENERATION)) { return -3; }
    }
    const int32_t ret = decode(ext);
    if (ret == 0) {
        LLAMA_LOG_DEBUG("%s: decode overlap with host token inputs, n_seqs = %d\n", __func__, n_items);
    }
    return ret == 0 ? 0 : (ret < 0 ? ret : -3);
}
