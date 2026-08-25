#pragma once

#include "llama.h"

#include <cstdint>
#include <vector>

#define LLAMA_MAX_SEQ 256

// Persistent per-layer scratch used by memory modules must also cover the
// synthetic graphs built while resolving fused operations.  Keep this value
// next to the probe definitions rather than duplicating their largest token
// geometry in individual memory implementations.
constexpr uint32_t LLAMA_MAX_FUSED_OP_PROBE_TOKENS_PER_SEQ = 16;

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_ctx_seq;       // context for a single sequence
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    uint32_t n_rs_seq;        // number of recurrent-state snapshots per seq for rollback
    uint32_t n_outputs_max;   // max outputs supported by the context
    uint32_t kv_gpu_layers;
    int32_t  n_threads;       // number of threads to use for generation
    int32_t  n_threads_batch; // number of threads to use for batch processing

    int32_t  nextn_layer_offset = 0;

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;

    bool embeddings;
    bool embeddings_nextn;        // also extract the hidden state before the final output norm
    bool embeddings_nextn_masked; // extract for only rows where batch.logits != 0
    bool causal_attn;
    bool offload_kqv;             // place attention KV storage on the accelerator
    bool offload_attn_compute;    // do not force attention compute to CPU solely because KV storage is on the host
    bool flash_attn;
    bool flash_attn_native_quants; // read a quantized K/V cache in place instead of casting it to F16
    bool auto_fa;
    bool fused_gdn_ar;       // use fused gated delta net (autoregressive)
    bool fused_gdn_ch;       // use fused gated delta net (chunked)
    bool auto_fgdn;
    bool fused_lid;          // use fused lightning indexer
    bool auto_flid;
    bool fused_dsv4_hc_pre;
    bool fused_dsv4_hc_comb;
    bool fused_dsv4_hc_post;
    bool auto_fhc;
    bool no_perf;
    bool warmup;             // TODO: remove [TAG_LLAMA_GRAPH_NO_WARMUP]
    bool op_offload;
    bool kv_unified;
    bool pipeline_parallel;
    bool kv_cpu_pinned;
    bool recurrent_state_offload;
    bool phase_aware_workspace;
    bool live_context_workspace;
    uint32_t kv_pipeline_depth;

    std::vector<bool> embeddings_layer_inp; // [n_layer()] extract input embeddings for layer

    enum llama_context_type ctx_type;
    enum llama_pooling_type pooling_type;

    // Structured KVarN cache settings.  Kept in the internal context params so
    // memory creation does not need to depend on the public params object.
    llama_kvarn_params kvarn;

    uint32_t  kv_tail_tokens = 0;
    uint32_t  kv_tail_tokens_swa = 0;
    uint32_t  kv_tail_tokens_requested = 0;
    uint32_t  kv_tail_tokens_swa_requested = 0;
    bool      kv_tail_native_exact = false;
    bool      kv_tail_native_exact_swa = false;
    uint32_t  kv_tail_rollback_tokens = 0;
    ggml_type kv_tail_type   = GGML_TYPE_COUNT;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;

    llama_context * ctx_other;
};
