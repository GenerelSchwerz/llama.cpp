#include "common.h"
#include "log.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"

// TODO: replace with #include "llama-ext.h" in the future
#include "../src/llama-arch.h"
#include "../src/llama-context.h"
#include "../src/llama-ext.h"
#include "../src/llama-model-saver.h"

#include <cinttypes>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// normalized mean squared error = mse(a, b) / mse(a, 0)
static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    GGML_ASSERT(a.size() == b.size());
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < a.size(); i++) {
        float a_i = a[i];
        float b_i = b[i];

        mse_a_b += (a_i - b_i) * (a_i - b_i);
        mse_a_0 += a_i * a_i;
    }

    return mse_a_b / mse_a_0;
}

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    size_t seed = *(const size_t *) userdata;
    std::hash<std::string> hasher;
    seed ^= hasher(tensor->name);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(0.0f, 1.0e-2f);

    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = dis(gen);
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = ggml_fp32_to_fp16(dis(gen));
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal error");
    }
}

static void usage(char ** argv) {
    printf("Usage: %s [-a/--arch arch] [-s/--seed seed] [-o/--out dir] [-v N] [-h/--help] [--test-phase-workspace] [--test-live-context-workspace] [--test-sched-copy-name]\n", argv[0]);
}

static std::vector<llama_token> get_tokens(const uint32_t n_tokens, const uint32_t n_vocab, const size_t seed){
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, n_vocab - 1);
    std::vector<llama_token> ret;
    ret.reserve(n_tokens);
    for (uint32_t i = 0; i < n_tokens; i++) {
        ret.push_back(dis(gen));
    }
    return ret;
}

static gguf_context_ptr get_gguf_ctx(const llm_arch arch, const bool moe, const bool mtp = false) {
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(arch, ret.get());
    const uint32_t n_ctx = 256;

    uint32_t n_vocab = 128;
    uint32_t n_embd  = 256;
    uint32_t n_head  = 2;
    uint32_t n_ff    = 384;
    uint32_t n_layer = 2;
    if (arch == LLM_ARCH_LLAMA4) {
        n_layer = 4; // hparams.n_no_rope_layer_step is hard-coded to 4
    } else if (arch == LLM_ARCH_GEMMA4) {
        n_embd = 128;
        n_head = 2;
        n_ff   = 192;
        n_layer = 5; // need at least 5 for swa_pattern (every 5th is full_attention)
    } else if (arch == LLM_ARCH_GEMMA3N) {
        n_embd = 64;
        n_head = 1;
        n_ff   = 96;
        n_layer = 22; // hparams.n_layer_kv_from_start = 20 is hardcoded
    } else if (arch == LLM_ARCH_DEEPSEEK4) {
        // head size 64 so that GPU flash attention kernels support the model
        n_embd  = 512;
        n_head  = 8;
        n_ff    = 1024;
        n_layer = 4;
    } else if (arch == LLM_ARCH_STEP35 || arch == LLM_ARCH_LAGUNA) {
        n_embd = 160; // exercise per-head tensor split granularity with head size 80
    } else if (arch == LLM_ARCH_QWEN3 || arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_AFMOE) {
        n_head = 4;
    } else if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_DEEPSEEK32
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_DOTS3NOTE
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_BAILINGMOE3
            || arch == LLM_ARCH_KIMI_K3
            || arch == LLM_ARCH_MISTRAL4
            || arch == LLM_ARCH_HY_V4) {
        n_embd = 128;
        n_head = 1;
        n_ff   = 192;
    } else if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        n_layer = 3;
    } else if (arch == LLM_ARCH_CHAMELEON) {
        n_vocab = 10240;
    } else if (arch == LLM_ARCH_QWEN3TTS) {
        n_vocab = 4096; // must be >= the hard-coded codec head size (3072)
    }

    GGML_ASSERT(!mtp || arch == LLM_ARCH_QWEN35);
    const uint32_t n_layer_all = n_layer + (mtp ? 1 : 0);

    uint32_t n_head_kv = n_head;
    if (arch == LLM_ARCH_QWEN3) {
        n_head_kv = 1; // MQA coverage
    } else if (arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_AFMOE) {
        n_head_kv = 2; // GQA coverage
    }
    const uint32_t n_embd_head = n_embd / n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llm_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,           n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               n_layer_all);
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(1));
    if (mtp) {
        ms.add_kv(LLM_KV_NEXTN_PREDICT_LAYERS, uint32_t(1));
    }

    if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        std::vector<uint32_t> n_ff_per_layer;
        n_ff_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_ff_per_layer.push_back(il <= 1 ? 0 : n_ff);
        }
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff_per_layer);
    } else {
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff);
    }

    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,   false);
    ms.add_kv(LLM_KV_LOGIT_SCALE,             1.0f);
    ms.add_kv(LLM_KV_TIME_MIX_EXTRA_DIM,      uint32_t(64));
    ms.add_kv(LLM_KV_TIME_DECAY_EXTRA_DIM,    uint32_t(128));
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL, uint32_t(2));

    if (arch == LLM_ARCH_PLAMO2 || arch == LLM_ARCH_JAMBA || arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE ||
            arch == LLM_ARCH_GRANITE_HYBRID || arch == LLM_ARCH_LFM2 || arch == LLM_ARCH_LFM2MOE || arch == LLM_ARCH_KIMI_LINEAR ||
            arch == LLM_ARCH_BAILINGMOE3 || arch == LLM_ARCH_KIMI_K3) {
        GGML_ASSERT(n_layer >= 2);
        std::vector<uint32_t> n_head_per_layer;
        n_head_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_head_per_layer.push_back(il == 1 ? 0 : n_head);
        }
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head_per_layer);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_per_layer);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, arch == LLM_ARCH_DEEPSEEK4 ? uint32_t(1) : n_head_kv);
    }

    ms.add_kv(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, 8.0f);
    if (arch == LLM_ARCH_DEEPSEEK4) {
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,   n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH, n_embd_head);
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,   n_embd_head/2);
    } else if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_DEEPSEEK32
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_DOTS3NOTE
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_BAILINGMOE3
            || arch == LLM_ARCH_KIMI_K3
            || arch == LLM_ARCH_MISTRAL4
            || arch == LLM_ARCH_HY_V4) {
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,       uint32_t(576));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,     uint32_t(512));
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       uint32_t(64));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   uint32_t(192));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, uint32_t(128));
        if (arch == LLM_ARCH_DOTS3NOTE) {
            // SWA layers reuse the same MLA geometry as the full layers in this fixture
            ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK_SWA,     uint32_t(512));
            ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_SWA,       uint32_t(576));
            ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,     uint32_t(512));
            ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA_SWA,   uint32_t(192));
            ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA_SWA, uint32_t(128));
            ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,             10000.0f);
            // indexer on the full-attention layers (inverse of the swa pattern)
            std::vector<uint32_t> indexer_types;
            indexer_types.reserve(n_layer);
            for (uint32_t il = 0; il < n_layer; il++) {
                indexer_types.push_back(il % 2 ? 0 : 1);
            }
            ms.add_kv(LLM_KV_ATTENTION_INDEXER_TYPES, indexer_types);
        }
    } else if (arch == LLM_ARCH_MINIMAX_M3) {
        // partial rotary: n_rot must not exceed the indexer key length (64)
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       uint32_t(64));
    }
    ms.add_kv(LLM_KV_ATTENTION_CLAMP_KQV,              1.0f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_GROUPS,       uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_Q_LORA_RANK,            arch == LLM_ARCH_DEEPSEEK4 ? uint32_t(64) : uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK,           uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,         n_ctx/8);

    if (arch == LLM_ARCH_GEMMA4) {
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,      n_embd/2);
        ms.add_kv(LLM_KV_ATTENTION_SHARED_KV_LAYERS,      uint32_t(0));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_SWA,        n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,      n_embd_head);
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,              10000.0f);
        // SWA pattern: every 5th layer is full attention (matches E2B layer_types)
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(5));
    } else if (arch == LLM_ARCH_COHERE2MOE || arch == LLM_ARCH_MIMO2 || arch == LLM_ARCH_STEP35 ||
            arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_GRANITE_SWA || arch == LLM_ARCH_DOTS3NOTE) {
        std::vector<uint32_t> pattern;
        pattern.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            pattern.push_back(il % 2);
        }
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, pattern);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(2));
    }

    // MSA requires one indexer head per GQA (KV) head, unlike the DSA archs where the
    // indexer head count is independent of the main attention head count.
    if (arch == LLM_ARCH_QWEN4EXP) {
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,    uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_LOW_RANK, uint32_t(8));
        // without this the QSA layers fall back to dense and go uncovered
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS, std::vector<uint32_t>(n_layer, 4));

        // has_cell_ext() needs ple_n_heads here: the indexer cache serializes no ext without it
        const uint32_t ple_ngram_size      = 3;
        const uint32_t ple_heads_per_ngram = 2;
        const uint32_t ple_n_heads         = (ple_ngram_size - 1)*ple_heads_per_ngram;
        GGML_ASSERT(n_embd % ple_n_heads == 0);
        const uint32_t ple_head_dim = n_embd/ple_n_heads;

        std::vector<uint64_t> ple_head_offsets(ple_n_heads);
        std::vector<uint64_t> ple_head_vocab_sizes(ple_n_heads, n_vocab);
        for (uint32_t h = 0; h < ple_n_heads; h++) {
            ple_head_offsets[h] = uint64_t(h)*n_vocab;
        }

        // the PLE history lives in the recurrent cache, so it must sit on a linear attention layer
        ms.add_kv(LLM_KV_PLE_LAYERS,                  std::vector<uint32_t>({ 0 }));
        ms.add_kv(LLM_KV_PLE_NGRAM_SIZE,              ple_ngram_size);
        ms.add_kv(LLM_KV_PLE_HEADS_PER_NGRAM,         ple_heads_per_ngram);
        ms.add_kv(LLM_KV_PLE_CONV_KERNEL,             uint32_t(4));
        ms.add_kv(LLM_KV_PLE_EOS_TOKEN_ID,            uint32_t(0));
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,  ple_head_dim);
        ms.add_kv(LLM_KV_PLE_LAYER_MULTIPLIERS,       std::vector<uint64_t>({ 1, 3, 5 }));
        ms.add_kv(LLM_KV_PLE_HEAD_OFFSETS,            ple_head_offsets);
        ms.add_kv(LLM_KV_PLE_HEAD_VOCAB_SIZES,        ple_head_vocab_sizes);
    }

    // minimax-m3 keeps one indexer head per GQA head; the rest use a fixed 64 to match the fused
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,   arch == LLM_ARCH_MINIMAX_M3 ? n_head : uint32_t(64));
    // qwen4exp ropes indexer keys with the main rotary width, so its head can't be < n_rot
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,
              arch == LLM_ARCH_QWEN4EXP ? n_embd_head : uint32_t(128));

    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,        uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE,   uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS, uint32_t(1));
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));

    if (arch == LLM_ARCH_HY_V4) {
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,     uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,   1.0e-6f);
        ms.add_kv(LLM_KV_HYPER_CONNECTION_MAGNITUDE, 2.0f);
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,           10.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,       1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,        true);
        // layer 0 must own an indexer, the odd layers share it
        std::vector<uint32_t> indexer_types;
        indexer_types.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            indexer_types.push_back(il % 2 ? 0 : 1);
        }
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_TYPES, indexer_types);
    }

    if (arch == LLM_ARCH_DEEPSEEK4) {
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,         uint32_t(8));
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,           uint32_t(32));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS,            std::vector<uint32_t>({0, 0, 4, 128}));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE,    160000.0f);
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,               uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,             1.0e-6f);
        ms.add_kv(LLM_KV_HASH_LAYER_COUNT,                      uint32_t(0));
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,                      10.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,                  1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,                   true);
    }
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,         "no_vocab");
    // ms.add_kv(LLM_KV_DENSE_2_FEAT_OUT,     n_embd);
    // ms.add_kv(LLM_KV_DENSE_3_FEAT_IN,      n_embd);

    if (moe) {
        ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, n_ff);
        ms.add_kv(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, n_ff / 2);  // distinct from n_ff so a saver key-clobber surfaces on reload
        ms.add_kv(LLM_KV_EXPERT_LATENT_LENGTH,       n_ff);
        ms.add_kv(LLM_KV_INTERLEAVE_MOE_LAYER_STEP,  uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_COUNT,               uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_USED_COUNT,          uint32_t(1));
        ms.add_kv(LLM_KV_EXPERT_SHARED_COUNT,        uint32_t(1));
        ms.add_kv(LLM_KV_EXPERT_GATING_FUNC,         arch == LLM_ARCH_DEEPSEEK4 ? uint32_t(4) : uint32_t(2)); // sqrtsoftplus : sigmoid
        ms.add_kv(LLM_KV_EXPERT_GROUP_SCALE,         1.0f);
        ms.add_kv(LLM_KV_EXPERTS_PER_GROUP,          uint32_t(1));
    }

    ms.add_kv(LLM_KV_POSNET_EMBEDDING_LENGTH,   n_embd);
    ms.add_kv(LLM_KV_POSNET_BLOCK_COUNT,        n_layer);
    ms.add_kv(LLM_KV_CONVNEXT_EMBEDDING_LENGTH, n_embd);
    ms.add_kv(LLM_KV_CONVNEXT_BLOCK_COUNT,      n_layer);
    ms.add_kv(LLM_KV_XIELU_ALPHA_N,             1.0f);
    ms.add_kv(LLM_KV_XIELU_ALPHA_P,             1.0f);
    ms.add_kv(LLM_KV_XIELU_BETA,                1.0f);
    ms.add_kv(LLM_KV_XIELU_EPS,                 1.0e-7f);
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,            arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE || arch == LLM_ARCH_QWEN4EXP ? 256 : 2*n_embd);
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,           uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,            uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,        n_head);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,           arch == LLM_ARCH_PLAMO2 ? 0 : uint32_t(2));
    ms.add_kv(LLM_KV_KDA_HEAD_DIM,              uint32_t(128));
    ms.add_kv(LLM_KV_KDA_SAFE_GATE,              true);
    ms.add_kv(LLM_KV_KDA_GATE_LOWER_BOUND,       -5.0f);
    if (arch == LLM_ARCH_BAILINGMOE3) {
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,   std::vector<float>({0.0f, 4.0f}));
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_SHEXP, std::vector<float>({0.0f, 5.0f}));
    }
    ms.add_kv(LLM_KV_WKV_HEAD_SIZE,             n_embd/n_head);
    ms.add_kv(LLM_KV_SHORTCONV_L_CACHE,         uint32_t(3));
    ms.add_kv(LLM_KV_RESIDUAL_SCALE,            3.5565588200778455f);
    ms.add_kv(LLM_KV_ATTN_RES_BLOCK_SIZE,       uint32_t(12));
    ms.add_kv(LLM_KV_ACTIVATION_SITU_BETA,      4.0f);
    ms.add_kv(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, 25.0f);
    ms.add_kv(LLM_KV_KDA_GATE_LOWER_BOUND,      -5.0f);

    for (uint32_t il = 0; il < n_layer; il++) {
        ggml_tensor t;
        memset(&t, 0, sizeof(ggml_tensor));
        t.type = GGML_TYPE_F16;
        ggml_format_name(&t, "conv%" PRIu32 "d.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv1.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv2.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "convnext.%" PRIu32 ".dw.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
    }
    return ret;
}

static bool silent_model_load_progress(float /*progress*/, void * /*user_data*/) {
    return true;
}

// with offload_kqv=false the cache lives in host memory
// n_seq_max > 1 gives the cache one stream per sequence
struct kv_config {
    bool     offload_kqv = true;
    uint32_t n_seq_max   = 1;
};

static std::pair<llama_model_ptr, llama_context_ptr> get_model_and_ctx(
        struct gguf_context * gguf_ctx, FILE * file, const size_t seed, const std::vector<ggml_backend_dev_t> & devs,
        const llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER, bool encode = false, const kv_config & kvc = {}) {
    GGML_ASSERT((gguf_ctx == nullptr) != (file == nullptr));
    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    std::vector<ggml_backend_dev_t> devs_copy = devs;
    devs_copy.push_back(nullptr);
    model_params.devices = devs_copy.data();
    model_params.split_mode = split_mode;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 0;
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;
    ctx_params.offload_kqv = kvc.offload_kqv;
    ctx_params.n_seq_max   = kvc.n_seq_max;
    if (!encode) {
        ctx_params.n_ubatch = 64;
    }

    size_t tmp = seed;
    llama_model_ptr model(gguf_ctx != nullptr ?
        llama_model_init_from_user(gguf_ctx, set_tensor_data, &tmp, model_params) :
        llama_model_load_from_file_ptr(file, model_params));
    if (!model) {
        throw std::runtime_error("failed to create llama model");
    }
    llama_context_ptr lctx(llama_init_from_model(model.get(), ctx_params));
    if (!lctx) {
        throw std::runtime_error("failed to create llama context");
    }
    return std::make_pair(std::move(model), std::move(lctx));
}

static bool phase_workspace_fail_alloc = false;
static bool phase_workspace_fail_once = false;
static int phase_workspace_alloc_failures = 0;
static ggml_backend_buffer_t (*phase_workspace_alloc_buffer)(ggml_backend_buffer_type_t, size_t) = nullptr;
static ggml_backend_t phase_workspace_sync_backends[2] = {};
static void (*phase_workspace_synchronize[2])(ggml_backend_t) = {};
static int phase_workspace_sync_calls[2] = {};

static ggml_backend_buffer_t phase_workspace_test_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    if (phase_workspace_fail_alloc) {
        phase_workspace_alloc_failures++;
        if (phase_workspace_fail_once) {
            phase_workspace_fail_alloc = false;
        }
        return nullptr;
    }
    return phase_workspace_alloc_buffer(buft, size);
}

static ggml_backend_t phase_workspace_get_buffer_backend(ggml_backend_sched_t sched) {
    ggml_backend_t result = nullptr;
    size_t result_size = 0;
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
        ggml_backend_t backend = ggml_backend_sched_get_backend(sched, i);
        const size_t size = ggml_backend_sched_get_buffer_size(sched, backend);
        if (size > result_size) {
            result = backend;
            result_size = size;
        }
    }
    return result;
}

static void phase_workspace_count_synchronize(ggml_backend_t backend) {
    for (int i = 0; i < 2; ++i) {
        if (phase_workspace_sync_backends[i] == backend) {
            phase_workspace_sync_calls[i]++;
            if (phase_workspace_synchronize[i] != nullptr) {
                phase_workspace_synchronize[i](backend);
            }
            return;
        }
    }
}

static llama_context_ptr make_phase_workspace_context(
        llama_model * model, llama_context_type type, llama_context * other = nullptr);

static llama_model_ptr make_live_context_workspace_model(llm_arch arch, size_t seed) {
    gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, false);
    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    ggml_backend_dev_t devices[] = { nullptr };
    model_params.devices = devices;

    size_t tensor_seed = seed;
    llama_model_ptr model(llama_model_init_from_user(gguf_ctx.get(), set_tensor_data, &tensor_seed, model_params));
    GGML_ASSERT(model);
    return model;
}

static llama_context_params make_live_context_workspace_params() {
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 1024;
    ctx_params.n_batch = 64;
    ctx_params.n_ubatch = 64;
    ctx_params.n_seq_max = 1;
    ctx_params.n_outputs_max = 1;
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    ctx_params.live_context_workspace = true;
    return ctx_params;
}

static void test_live_context_workspace_reserve(size_t seed) {
    llama_model_ptr model = make_live_context_workspace_model(LLM_ARCH_LLAMA, seed);
    llama_context_params ctx_params = make_live_context_workspace_params();

    llama_context_ptr ctx(llama_init_from_model(model.get(), ctx_params));
    GGML_ASSERT(ctx);
    GGML_ASSERT(!ctx->get_cparams().phase_aware_workspace);

    ggml_backend_sched_t sched = ctx->get_sched();
    ggml_backend_t backend_cpu = nullptr;
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
        ggml_backend_t backend = ggml_backend_sched_get_backend(sched, i);
        if (ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            backend_cpu = backend;
            break;
        }
    }
    GGML_ASSERT(backend_cpu != nullptr);

    const auto initial = ctx->make_sched_reserve_plan(1, 1);
    GGML_ASSERT(initial.live_kv);
    GGML_ASSERT(initial.n_kv_capacity == ctx_params.n_ctx);
    GGML_ASSERT(initial.n_kv == 256);
    const size_t size_256 = ggml_backend_sched_get_buffer_size(sched, backend_cpu);

    ctx->sched_reserve(1, 257);
    const size_t size_512 = ggml_backend_sched_get_buffer_size(sched, backend_cpu);
    GGML_ASSERT(size_512 > size_256);

    const auto half_bin = ctx->make_sched_reserve_plan(1, 256);
    GGML_ASSERT(half_bin.n_kv == 512);
    ctx->sched_reserve(1, 256);
    GGML_ASSERT(ggml_backend_sched_get_buffer_size(sched, backend_cpu) == size_512);

    ctx->sched_reserve(1, 513);
    const size_t size_1024 = ggml_backend_sched_get_buffer_size(sched, backend_cpu);
    GGML_ASSERT(size_1024 > size_512);

    const auto contraction = ctx->make_sched_reserve_plan(1, 256);
    GGML_ASSERT(contraction.n_kv == 256);
    ctx->sched_reserve(1, 256);
    GGML_ASSERT(ggml_backend_sched_get_buffer_size(sched, backend_cpu) == size_256);

    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    const std::vector<llama_token> tokens = get_tokens(10 * ctx_params.n_batch, n_vocab, seed);
    int32_t pos = 0;
    for (int32_t chunk = 0; chunk < 9; ++chunk) {
        llama_batch batch = llama_batch_init(ctx_params.n_batch, 0, 1);
        for (int32_t i = 0; i < (int32_t) ctx_params.n_batch; ++i) {
            common_batch_add(batch, tokens[pos], pos, { 0 }, i + 1 == (int32_t) ctx_params.n_batch);
            ++pos;
        }
        GGML_ASSERT(llama_decode(ctx.get(), batch) == 0);
        llama_synchronize(ctx.get());
        llama_batch_free(batch);
    }

    const auto runtime_high = ctx->make_sched_reserve_plan(ctx_params.n_batch);
    GGML_ASSERT(runtime_high.n_tokens == ctx_params.n_batch);
    GGML_ASSERT(runtime_high.n_kv == 1024);

    GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, 0, 512));
    llama_memory_seq_add(llama_get_memory(ctx.get()), 0, 512, pos, -512);

    llama_batch shifted = llama_batch_init(1, 0, 1);
    common_batch_add(shifted, tokens[pos], pos - 512, { 0 }, true);
    GGML_ASSERT(llama_decode(ctx.get(), shifted) == 0);
    llama_synchronize(ctx.get());
    llama_batch_free(shifted);

    // Logical positions are compacted, but live rows at the physical tail still require the high reserve.
    const auto physical_high = ctx->make_sched_reserve_plan(0);
    GGML_ASSERT(physical_high.n_tokens == ctx_params.n_batch);
    GGML_ASSERT(physical_high.n_kv == 1024);
    GGML_ASSERT(ggml_backend_sched_get_buffer_size(sched, backend_cpu) > size_256);

    llama_memory_clear(llama_get_memory(ctx.get()), false);
    llama_batch short_batch = llama_batch_init(1, 0, 1);
    common_batch_add(short_batch, tokens[0], 0, { 0 }, true);
    GGML_ASSERT(llama_decode(ctx.get(), short_batch) == 0);
    llama_synchronize(ctx.get());
    llama_batch_free(short_batch);

    const auto runtime_contraction = ctx->make_sched_reserve_plan(0);
    GGML_ASSERT(runtime_contraction.n_tokens == ctx_params.n_batch);
    GGML_ASSERT(runtime_contraction.n_kv == 256);
    GGML_ASSERT(ggml_backend_sched_get_buffer_size(sched, backend_cpu) == size_256);
    GGML_ASSERT(llama_trim_transient_memory(ctx.get()) == 0);
}

static void test_live_context_workspace_iswa_reserve(size_t seed) {
    llama_model_ptr model = make_live_context_workspace_model(LLM_ARCH_GEMMA4, seed);
    llama_context_params ctx_params = make_live_context_workspace_params();

    llama_context_ptr ctx(llama_init_from_model(model.get(), ctx_params));
    GGML_ASSERT(ctx);
    GGML_ASSERT(!ctx->get_cparams().phase_aware_workspace);

    const auto initial = ctx->make_sched_reserve_plan(1, 1);
    GGML_ASSERT(initial.live_kv);
    GGML_ASSERT(initial.n_kv_capacity == ctx_params.n_ctx);
    GGML_ASSERT(initial.n_kv == 256);

    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    const std::vector<llama_token> tokens = get_tokens(9 * ctx_params.n_batch, n_vocab, seed);
    int32_t pos = 0;
    for (int32_t chunk = 0; chunk < 9; ++chunk) {
        llama_batch batch = llama_batch_init(ctx_params.n_batch, 0, 1);
        for (int32_t i = 0; i < (int32_t) ctx_params.n_batch; ++i) {
            common_batch_add(batch, tokens[pos], pos, { 0 }, i + 1 == (int32_t) ctx_params.n_batch);
            ++pos;
        }
        GGML_ASSERT(llama_decode(ctx.get(), batch) == 0);
        llama_synchronize(ctx.get());
        llama_batch_free(batch);
    }

    const auto runtime = ctx->make_sched_reserve_plan(ctx_params.n_batch);
    GGML_ASSERT(runtime.n_kv == 1024);

    llama_memory_clear(llama_get_memory(ctx.get()), false);
    llama_batch short_batch = llama_batch_init(1, 0, 1);
    common_batch_add(short_batch, tokens[0], 0, { 0 }, true);
    GGML_ASSERT(llama_decode(ctx.get(), short_batch) == 0);
    llama_synchronize(ctx.get());
    llama_batch_free(short_batch);

    const auto contraction = ctx->make_sched_reserve_plan(0);
    GGML_ASSERT(contraction.n_kv == 256);
}

static void test_live_context_workspace_unsupported(size_t seed) {
    llama_model_ptr model = make_live_context_workspace_model(LLM_ARCH_MAMBA, seed);
    llama_context_params ctx_params = make_live_context_workspace_params();

    llama_context_ptr ctx(llama_init_from_model(model.get(), ctx_params));
    GGML_ASSERT(ctx);
    GGML_ASSERT(!ctx->get_cparams().live_context_workspace);
}

static void test_phase_workspace_runtime_reserve(size_t seed) {
    gguf_context_ptr gguf_ctx = get_gguf_ctx(LLM_ARCH_LLAMA, false);

    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    ggml_backend_dev_t devices[] = { nullptr };
    model_params.devices = devices;

    size_t tensor_seed = seed;
    llama_model_ptr model(llama_model_init_from_user(gguf_ctx.get(), set_tensor_data, &tensor_seed, model_params));
    GGML_ASSERT(model);

    phase_workspace_fail_alloc = false;
    phase_workspace_fail_once = false;
    phase_workspace_alloc_failures = 0;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 32;
    ctx_params.n_batch = 32;
    ctx_params.n_ubatch = 32;
    ctx_params.n_seq_max = 1;
    ctx_params.n_outputs_max = 0;
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;
    ctx_params.no_perf = false;
    ctx_params.phase_aware_workspace = true;

    llama_context_ptr ctx(llama_init_from_model(model.get(), ctx_params));
    GGML_ASSERT(ctx);
    GGML_ASSERT(llama_n_batch(ctx.get()) == ctx_params.n_batch);
    GGML_ASSERT(ctx->get_cparams().n_outputs_max == ctx_params.n_batch);

    ggml_backend_sched_t sched = ctx->get_sched();
    ggml_backend_t backend_workspace = phase_workspace_get_buffer_backend(sched);
    GGML_ASSERT(backend_workspace != nullptr);
    ggml_backend_buffer_type_t workspace_buft = ggml_backend_sched_get_buffer_type(sched, backend_workspace);
    phase_workspace_alloc_buffer = workspace_buft->iface.alloc_buffer;
    workspace_buft->iface.alloc_buffer = phase_workspace_test_alloc_buffer;
    const size_t size_small = ggml_backend_sched_get_buffer_size(sched, backend_workspace);

    llama_batch batch = llama_batch_init(16, 0, 1);
    const std::vector<llama_token> tokens = get_tokens(16, llama_vocab_n_tokens(llama_model_get_vocab(model.get())), seed);
    for (int32_t i = 0; i < 16; ++i) {
        common_batch_add(batch, tokens[i], i, { 0 }, true);
    }

    phase_workspace_fail_alloc = true;
    GGML_ASSERT(llama_decode(ctx.get(), batch) == -2);
    GGML_ASSERT(phase_workspace_alloc_failures > 0);
    GGML_ASSERT(ggml_backend_sched_get_buffer_size(sched, backend_workspace) == 0);

    phase_workspace_fail_alloc = false;
    const int64_t retry_start_us = ggml_time_us();
    GGML_ASSERT(llama_decode(ctx.get(), batch) == 0);
    llama_synchronize(ctx.get());
    const int64_t retry_elapsed_us = ggml_time_us() - retry_start_us;
    const llama_perf_context_data perf = llama_perf_context(ctx.get());
    GGML_ASSERT(perf.n_p_eval == 16);
    GGML_ASSERT(perf.t_p_eval_ms >= 0.0);
    GGML_ASSERT(perf.t_p_eval_ms <= retry_elapsed_us*1e-3 + 0.001);
    GGML_ASSERT(llama_get_logits_ith(ctx.get(), 0) != nullptr);
    GGML_ASSERT(llama_get_logits_ith(ctx.get(), 15) != nullptr);
    const size_t size_grown = ggml_backend_sched_get_buffer_size(sched, backend_workspace);
    GGML_ASSERT(size_grown > size_small);

    ctx->sched_reserve(ctx_params.n_ubatch);
    const size_t size_prompt = ggml_backend_sched_get_buffer_size(sched, backend_workspace);
    GGML_ASSERT(size_prompt == size_grown);

    llama_batch_free(batch);
    ctx.reset();

    ctx_params.n_outputs_max = 4;
    ctx.reset(llama_init_from_model(model.get(), ctx_params));
    GGML_ASSERT(ctx);
    sched = ctx->get_sched();
    backend_workspace = phase_workspace_get_buffer_backend(sched);
    GGML_ASSERT(backend_workspace != nullptr);
    const size_t size_declared = ggml_backend_sched_get_buffer_size(sched, backend_workspace);

    int32_t pos = 0;
    for (int32_t width = 1; width <= 4; width *= 2) {
        llama_batch declared = llama_batch_init(width, 0, 1);
        for (int32_t i = 0; i < width; ++i) {
            common_batch_add(declared, tokens[i], pos++, { 0 }, true);
        }
        GGML_ASSERT(llama_decode(ctx.get(), declared) == 0);
        llama_synchronize(ctx.get());
        GGML_ASSERT(ggml_backend_sched_get_buffer_size(sched, backend_workspace) == size_declared);
        llama_batch_free(declared);
    }

    ctx.reset();
    workspace_buft->iface.alloc_buffer = phase_workspace_alloc_buffer;
    phase_workspace_alloc_buffer = nullptr;
}

static void test_phase_workspace_late_pipeline_fallback(size_t seed) {
    std::vector<ggml_backend_dev_t> devices;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        if (props.caps.async && props.caps.events) {
            devices.push_back(dev);
        }
    }
    if (devices.size() < 2) {
        printf("test_phase_workspace_late_pipeline_fallback: skipped, two async GPU devices required\n");
        return;
    }

    gguf_context_ptr gguf_ctx = get_gguf_ctx(LLM_ARCH_LLAMA, false);
    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    model_params.n_gpu_layers = 99;
    ggml_backend_dev_t model_devices[] = { devices[0], devices[1], nullptr };
    model_params.devices = model_devices;

    size_t tensor_seed = seed;
    llama_model_ptr model(llama_model_init_from_user(gguf_ctx.get(), set_tensor_data, &tensor_seed, model_params));
    GGML_ASSERT(model);

    llama_context_ptr ctx = make_phase_workspace_context(model.get(), LLAMA_CONTEXT_TYPE_DEFAULT);
    GGML_ASSERT(ctx);
    GGML_ASSERT(ggml_backend_sched_get_n_copies(ctx->get_sched()) > 1);

    ggml_backend_buffer_type_t gpu_buft = nullptr;
    for (int i = 0; i < ggml_backend_sched_get_n_backends(ctx->get_sched()); ++i) {
        ggml_backend_t backend = ggml_backend_sched_get_backend(ctx->get_sched(), i);
        if (ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu_buft = ggml_backend_get_default_buffer_type(backend);
            break;
        }
    }
    GGML_ASSERT(gpu_buft != nullptr);

    phase_workspace_alloc_failures = 0;
    phase_workspace_alloc_buffer = gpu_buft->iface.alloc_buffer;
    phase_workspace_fail_once = true;
    phase_workspace_fail_alloc = true;
    gpu_buft->iface.alloc_buffer = phase_workspace_test_alloc_buffer;

    llama_batch batch = llama_batch_init(16, 0, 1);
    const std::vector<llama_token> tokens = get_tokens(16,
            llama_vocab_n_tokens(llama_model_get_vocab(model.get())), seed);
    for (int32_t i = 0; i < 16; ++i) {
        common_batch_add(batch, tokens[i], i, { 0 }, i == 15);
    }
    GGML_ASSERT(llama_decode(ctx.get(), batch) == 0);
    llama_synchronize(ctx.get());
    GGML_ASSERT(phase_workspace_alloc_failures == 1);
    GGML_ASSERT(ggml_backend_sched_get_n_copies(ctx->get_sched()) == 1);

    gpu_buft->iface.alloc_buffer = phase_workspace_alloc_buffer;
    phase_workspace_alloc_buffer = nullptr;
    phase_workspace_fail_alloc = false;
    phase_workspace_fail_once = false;
    llama_batch_free(batch);
}

static llama_batch make_mtp_batch(
        int32_t n_tokens, int32_t n_embd, int32_t pos, uint32_t n_vocab, size_t seed) {
    llama_batch batch = llama_batch_init(n_tokens, n_embd, 1);
    batch.token = (llama_token *) malloc(sizeof(llama_token) * n_tokens);
    GGML_ASSERT(batch.token != nullptr);

    const std::vector<llama_token> tokens = get_tokens(n_tokens, n_vocab, seed);
    for (int32_t i = 0; i < n_tokens; ++i) {
        common_batch_add(batch, tokens[i], pos + i, { 0 }, i == n_tokens - 1);
        for (int32_t j = 0; j < n_embd; ++j) {
            batch.embd[(size_t) i * n_embd + j] = (float) ((i + j) % 17) / 17.0f;
        }
    }
    return batch;
}

static llama_context_ptr make_phase_workspace_context(
        llama_model * model, llama_context_type type, llama_context * other) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = 32;
    params.n_batch = 32;
    params.n_ubatch = 32;
    params.n_seq_max = 1;
    params.n_outputs_max = type == LLAMA_CONTEXT_TYPE_MTP ? 1 : 4;
    params.n_threads = 4;
    params.n_threads_batch = 4;
    params.ctx_type = type;
    params.ctx_other = other;
    params.phase_aware_workspace = true;
    return llama_context_ptr(llama_init_from_model(model, params));
}

static void test_phase_workspace_mtp_lifecycle(size_t seed) {
    gguf_context_ptr gguf_ctx = get_gguf_ctx(LLM_ARCH_QWEN35, false, true);

    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    model_params.load_mtp = true;
    ggml_backend_dev_t devices[] = { nullptr };
    model_params.devices = devices;

    size_t tensor_seed = seed;
    llama_model_ptr model(llama_model_init_from_user(gguf_ctx.get(), set_tensor_data, &tensor_seed, model_params));
    GGML_ASSERT(model);

    llama_context_ptr target = make_phase_workspace_context(model.get(), LLAMA_CONTEXT_TYPE_DEFAULT);
    llama_context_ptr draft = make_phase_workspace_context(model.get(), LLAMA_CONTEXT_TYPE_MTP, target.get());
    GGML_ASSERT(target && draft);
    GGML_ASSERT(!llama_contexts_share_workspace(target.get(), draft.get()));

    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    const int32_t n_embd = llama_model_n_embd_out(model.get());
    llama_batch target_batch = llama_batch_init(4, 0, 1);
    const std::vector<llama_token> tokens = get_tokens(4, n_vocab, seed);
    for (int32_t i = 0; i < 4; ++i) {
        common_batch_add(target_batch, tokens[i], i, { 0 }, true);
    }
    llama_batch draft_batch = make_mtp_batch(4, n_embd, 0, n_vocab, seed + 1);

    GGML_ASSERT(llama_decode(target.get(), target_batch) == 0);
    GGML_ASSERT(llama_decode(draft.get(), draft_batch) == 0);
    llama_synchronize(draft.get());
    llama_synchronize(target.get());
    llama_batch_free(draft_batch);
    llama_batch_free(target_batch);

    ggml_backend_t target_workspace = phase_workspace_get_buffer_backend(target->get_sched());
    ggml_backend_t draft_workspace = phase_workspace_get_buffer_backend(draft->get_sched());
    GGML_ASSERT(target_workspace && draft_workspace);
    ggml_backend_buffer_type_t workspace_buft = ggml_backend_sched_get_buffer_type(
            target->get_sched(), target_workspace);
    phase_workspace_alloc_failures = 0;
    phase_workspace_alloc_buffer = workspace_buft->iface.alloc_buffer;
    phase_workspace_fail_once = true;
    phase_workspace_fail_alloc = true;
    workspace_buft->iface.alloc_buffer = phase_workspace_test_alloc_buffer;
    GGML_ASSERT(llama_attach_shared_workspace(draft.get(), target.get()) == -1);
    GGML_ASSERT(phase_workspace_alloc_failures == 1);
    GGML_ASSERT(draft->get_sched() == nullptr);
    GGML_ASSERT(ggml_backend_sched_get_buffer_size(target->get_sched(), target_workspace) == 0);
    workspace_buft->iface.alloc_buffer = phase_workspace_alloc_buffer;
    phase_workspace_alloc_buffer = nullptr;
    phase_workspace_fail_alloc = false;
    phase_workspace_fail_once = false;

    GGML_ASSERT(llama_attach_shared_workspace(draft.get(), target.get()) == 1);
    GGML_ASSERT(llama_contexts_share_workspace(target.get(), draft.get()));
    GGML_ASSERT(draft->make_sched_reserve_plan(4).n_tokens == 4);

    const size_t decode_size = ggml_backend_sched_get_buffer_size(target->get_sched(), target_workspace);
    GGML_ASSERT(decode_size > 0);
    GGML_ASSERT(ggml_backend_sched_get_buffer_size(draft->get_sched(), draft_workspace) == decode_size);
    draft->sched_reserve(4);
    GGML_ASSERT(ggml_backend_sched_get_buffer_size(draft->get_sched(), draft_workspace) == decode_size);

    phase_workspace_sync_backends[0] = target_workspace;
    phase_workspace_sync_backends[1] = draft_workspace;
    phase_workspace_synchronize[0] = target_workspace->iface.synchronize;
    phase_workspace_synchronize[1] = draft_workspace->iface.synchronize;
    phase_workspace_sync_calls[0] = 0;
    phase_workspace_sync_calls[1] = 0;
    target_workspace->iface.synchronize = phase_workspace_count_synchronize;
    draft_workspace->iface.synchronize = phase_workspace_count_synchronize;

    llama_batch target_catchup = llama_batch_init(1, 0, 1);
    common_batch_add(target_catchup, tokens[0], 4, { 0 }, true);
    llama_batch draft_catchup = make_mtp_batch(1, n_embd, 4, n_vocab, seed + 2);
    llama_batch target_reacquire = llama_batch_init(1, 0, 1);
    common_batch_add(target_reacquire, tokens[1], 5, { 0 }, true);
    GGML_ASSERT(llama_decode(target.get(), target_catchup) == 0);
    GGML_ASSERT(llama_decode(draft.get(), draft_catchup) == 0);
    GGML_ASSERT(phase_workspace_sync_calls[0] > 0);
    const int draft_syncs_before_reacquire = phase_workspace_sync_calls[1];
    GGML_ASSERT(llama_decode(target.get(), target_reacquire) == 0);
    GGML_ASSERT(phase_workspace_sync_calls[1] == draft_syncs_before_reacquire + 1);
    GGML_ASSERT(llama_get_logits_ith(target.get(), 0) != nullptr);
    target_workspace->iface.synchronize = phase_workspace_synchronize[0];
    draft_workspace->iface.synchronize = phase_workspace_synchronize[1];
    memset(phase_workspace_sync_backends, 0, sizeof(phase_workspace_sync_backends));
    memset(phase_workspace_synchronize, 0, sizeof(phase_workspace_synchronize));
    memset(phase_workspace_sync_calls, 0, sizeof(phase_workspace_sync_calls));
    llama_batch_free(target_reacquire);
    llama_batch_free(draft_catchup);
    llama_batch_free(target_catchup);

    llama_batch draft_failure = make_mtp_batch(1, n_embd, 5, n_vocab, seed + 3);
    GGML_ASSERT(llama_decode(draft.get(), draft_failure) == 0);
    phase_workspace_alloc_failures = 0;
    phase_workspace_alloc_buffer = workspace_buft->iface.alloc_buffer;
    phase_workspace_fail_once = true;
    phase_workspace_fail_alloc = true;
    workspace_buft->iface.alloc_buffer = phase_workspace_test_alloc_buffer;
    bool reserve_failed = false;
    try {
        target->sched_reserve(32);
    } catch (const std::exception &) {
        reserve_failed = true;
    }
    GGML_ASSERT(reserve_failed);
    GGML_ASSERT(phase_workspace_alloc_failures == 1);
    GGML_ASSERT(ggml_backend_sched_get_buffer_size(target->get_sched(), target_workspace) == 0);
    GGML_ASSERT(ggml_backend_sched_get_buffer_size(draft->get_sched(), draft_workspace) == 0);
    workspace_buft->iface.alloc_buffer = phase_workspace_alloc_buffer;
    phase_workspace_alloc_buffer = nullptr;
    phase_workspace_fail_alloc = false;
    phase_workspace_fail_once = false;
    llama_batch_free(draft_failure);

    target->sched_reserve(32);
    const size_t prompt_size = ggml_backend_sched_get_buffer_size(target->get_sched(), target_workspace);
    GGML_ASSERT(prompt_size > decode_size);
    GGML_ASSERT(ggml_backend_sched_get_buffer_size(draft->get_sched(), draft_workspace) == prompt_size);

    target->sched_reserve(0);
    draft->sched_reserve(0);
    GGML_ASSERT(ggml_backend_sched_get_buffer_size(target->get_sched(), target_workspace) == decode_size);
    target->sched_reserve(32);
    GGML_ASSERT(ggml_backend_sched_get_buffer_size(target->get_sched(), target_workspace) == prompt_size);

    draft.reset();
    draft = make_phase_workspace_context(model.get(), LLAMA_CONTEXT_TYPE_MTP, target.get());
    GGML_ASSERT(draft);
    GGML_ASSERT(!llama_contexts_share_workspace(target.get(), draft.get()));
    GGML_ASSERT(llama_attach_shared_workspace(draft.get(), target.get()) == 1);

    llama_batch teardown_batch = make_mtp_batch(1, n_embd, 0, n_vocab, seed + 4);
    GGML_ASSERT(llama_decode(draft.get(), teardown_batch) == 0);
    target.reset();
    llama_batch_free(teardown_batch);
    draft->sched_reserve(0);

    target = make_phase_workspace_context(model.get(), LLAMA_CONTEXT_TYPE_DEFAULT);
    GGML_ASSERT(target);
    GGML_ASSERT(llama_attach_shared_workspace(draft.get(), target.get()) == 1);
    target.reset();

    llama_batch retry_batch = make_mtp_batch(1, n_embd, 4, n_vocab, seed + 2);
    GGML_ASSERT(llama_decode(draft.get(), retry_batch) == 0);
    llama_synchronize(draft.get());
    llama_batch_free(retry_batch);
    draft.reset();
}

static void test_phase_workspace_mismatched_placement(size_t seed) {
    ggml_backend_dev_t gpu = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu = dev;
            break;
        }
    }
    if (gpu == nullptr) {
        printf("test_phase_workspace_mismatched_placement: skipped, GPU device required\n");
        return;
    }

    gguf_context_ptr gguf_ctx = get_gguf_ctx(LLM_ARCH_QWEN35, false, true);
    llama_model_params target_params = llama_model_default_params();
    target_params.progress_callback = silent_model_load_progress;
    target_params.n_gpu_layers = 99;
    target_params.load_mtp = true;
    ggml_backend_dev_t target_devices[] = { gpu, nullptr };
    target_params.devices = target_devices;

    llama_model_params draft_params = llama_model_default_params();
    draft_params.progress_callback = silent_model_load_progress;
    draft_params.load_mtp = true;
    ggml_backend_dev_t draft_devices[] = { nullptr };
    draft_params.devices = draft_devices;

    size_t target_seed = seed;
    size_t draft_seed = seed;
    llama_model_ptr target_model(llama_model_init_from_user(
            gguf_ctx.get(), set_tensor_data, &target_seed, target_params));
    llama_model_ptr draft_model(llama_model_init_from_user(
            gguf_ctx.get(), set_tensor_data, &draft_seed, draft_params));
    GGML_ASSERT(target_model && draft_model);

    llama_context_ptr target = make_phase_workspace_context(target_model.get(), LLAMA_CONTEXT_TYPE_DEFAULT);
    llama_context_ptr draft = make_phase_workspace_context(
            draft_model.get(), LLAMA_CONTEXT_TYPE_MTP, target.get());
    GGML_ASSERT(target && draft);
    GGML_ASSERT(!llama_contexts_share_workspace(target.get(), draft.get()));

    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(target_model.get()));
    const int32_t n_embd = llama_model_n_embd_out(draft_model.get());
    llama_batch target_batch = llama_batch_init(1, 0, 1);
    common_batch_add(target_batch, get_tokens(1, n_vocab, seed)[0], 0, { 0 }, true);
    llama_batch draft_batch = make_mtp_batch(1, n_embd, 0, n_vocab, seed + 1);
    GGML_ASSERT(llama_decode(target.get(), target_batch) == 0);
    GGML_ASSERT(llama_decode(draft.get(), draft_batch) == 0);
    llama_synchronize(draft.get());
    llama_synchronize(target.get());
    llama_batch_free(draft_batch);
    llama_batch_free(target_batch);

    const int32_t status = llama_attach_shared_workspace(draft.get(), target.get());
    GGML_ASSERT(status >= 0);
    GGML_ASSERT(llama_contexts_share_workspace(target.get(), draft.get()) == (status == 1));
}

// the meta backend recovers the source of a scheduler copy from its name, so both sides must agree
static void test_sched_copy_name() {
    const ggml_init_params params = {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    GGML_ASSERT(ctx);
    ggml_tensor * src  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
    ggml_tensor * copy = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);

    auto check = [&](const char * name, const char * backend_name, const char * expected) {
        char buf[GGML_MAX_NAME];
        ggml_set_name(src, name);
        ggml_backend_sched_name_copy(copy, backend_name, src, 0);
        ggml_backend_sched_copy_source_name(copy->name, buf, sizeof(buf));
        GGML_ASSERT(strcmp(buf, expected) == 0);
    };

    check("cache_k_l0", "CUDA0", "cache_k_l0");
    check("cache_k_l0 (view)", "CUDA0", "cache_k_l0");
    // the backend label is cut when the name does not fit, the source name survives
    check("cache_k_l31", "Meta(CUDA0,CUDA1,CUDA2,CUDA3,CUDA4,CUDA5,CUDA6,CUDA7)", "cache_k_l31");
}

// the tokens are spread over n_seq sequences, each of which starts at position 0
static std::vector<float> get_logits(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens, bool encode = false, uint32_t n_seq = 1) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_ctx    = llama_n_ctx(lctx);
    const uint32_t n_tokens = tokens.size();
    GGML_ASSERT(n_seq >= 1 && n_tokens % n_seq == 0);
    const uint32_t n_tokens_seq = n_tokens / n_seq;
    llama_batch batch = llama_batch_init(n_ctx, 0, n_seq);
    GGML_ASSERT(n_tokens <= n_ctx);
    for (uint32_t pos = 0; pos < n_tokens; pos++) {
        common_batch_add(batch, tokens[pos], pos % n_tokens_seq, {llama_seq_id(pos / n_tokens_seq)}, true);
    }
    batch.n_tokens = n_tokens;
    if (encode) {
        if (llama_encode(lctx, batch)) {
            llama_batch_free(batch);
            throw std::runtime_error("failed to encode batch");
        }
    }
    if (llama_decode(lctx, batch)) {
        llama_batch_free(batch);
        throw std::runtime_error("failed to decode batch");
    }

    std::vector<float> ret;
    ret.reserve(n_tokens*n_vocab);
    for (uint32_t i = 0; i < n_tokens; i++) {
        const float * logits_ith = llama_get_logits_ith(lctx, i);
        for (uint32_t j = 0; j < n_vocab; j++) {
            ret.push_back(logits_ith[j]);
        }
    }
    llama_batch_free(batch);
    return ret;
}

static bool moe_mandatory(const llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_LLAMA4:
        case LLM_ARCH_COHERE2MOE:
        case LLM_ARCH_GROK:
        case LLM_ARCH_QWEN2MOE:
        case LLM_ARCH_QWEN3MOE:
        case LLM_ARCH_QWEN3NEXT:
        case LLM_ARCH_QWEN3VLMOE:
        case LLM_ARCH_QWEN35MOE:
        case LLM_ARCH_QWEN4EXP:
        case LLM_ARCH_PHIMOE:
        case LLM_ARCH_DBRX:
        case LLM_ARCH_OLMOE:
        case LLM_ARCH_ARCTIC:
        case LLM_ARCH_DEEPSEEK:
        case LLM_ARCH_DEEPSEEK2:
        case LLM_ARCH_DEEPSEEK32:
        case LLM_ARCH_DOTS3NOTE:
        case LLM_ARCH_DEEPSEEK4:
        case LLM_ARCH_GLM4_MOE:
        case LLM_ARCH_GLM_DSA:
        case LLM_ARCH_EXAONE_MOE:
        case LLM_ARCH_BAILINGMOE:
        case LLM_ARCH_BAILINGMOE2:
        case LLM_ARCH_BAILINGMOE3:
        case LLM_ARCH_DOTS1:
        case LLM_ARCH_AFMOE:
        case LLM_ARCH_ERNIE4_5:
        case LLM_ARCH_ERNIE4_5_MOE:
        case LLM_ARCH_HUNYUAN_MOE:
        case LLM_ARCH_HY_V3:
        case LLM_ARCH_HY_V4:
        case LLM_ARCH_OPENAI_MOE:
        case LLM_ARCH_LFM2MOE:
        case LLM_ARCH_SMALLTHINKER:
        case LLM_ARCH_LLADA_MOE:
        case LLM_ARCH_GROVEMOE:
        case LLM_ARCH_MINIMAX_01:
        case LLM_ARCH_MINIMAX_M2:
        case LLM_ARCH_MINIMAX_M3:
        case LLM_ARCH_RND1:
        case LLM_ARCH_PADDLEOCR:
        case LLM_ARCH_MIMO2:
        case LLM_ARCH_KIMI_LINEAR:
        case LLM_ARCH_KIMI_K3:
        case LLM_ARCH_STEP35:
        case LLM_ARCH_MISTRAL4:
        case LLM_ARCH_MELLUM:
        case LLM_ARCH_LAGUNA:
            return true;
        default:
            return false;
    }
}

static bool moe_implemented(const llm_arch arch) {
    if (moe_mandatory(arch)) {
        return true;
    }
    switch (arch) {
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_REFACT:
        case LLM_ARCH_MINICPM:
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
        case LLM_ARCH_MISTRAL3:
        case LLM_ARCH_LLAMA_EMBED:
            return true;
        default:
            return false;
    }
}

static bool arch_supported(const llm_arch arch) {
    if (arch == LLM_ARCH_CLIP || arch == LLM_ARCH_GPTJ || arch == LLM_ARCH_UNKNOWN) {
        return false; // These models don't have usable implementations.
    }
    if (arch == LLM_ARCH_CHAMELEON) {
        return false; // Only half-implemented and to be removed in the future.
    }
    if (arch == LLM_ARCH_WAVTOKENIZER_DEC) {
        return false; // FIXME CUDA backend crashes.
    }
    if (arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
        return false; // FIXME @ngxson
    }
    if (arch == LLM_ARCH_GRANITE_SWITCH) {
        return false; // FIXME adapter fixture
    }
    if (arch == LLM_ARCH_LLAMA_EMBED || arch == LLM_ARCH_GEMMA_EMBEDDING || arch == LLM_ARCH_T5ENCODER) {
        return false; // FIXME Embedding (?) models produce inconsistent results.
    }
    if (arch == LLM_ARCH_RWKV6 || arch == LLM_ARCH_RWKV6QWEN2 || arch == LLM_ARCH_RWKV7 || arch == LLM_ARCH_ARWKV7) {
        return false; // FIXME RWKV models hang indefinitely.
    }
    if (arch == LLM_ARCH_BERT || arch == LLM_ARCH_MODERN_BERT || arch == LLM_ARCH_NOMIC_BERT || arch == LLM_ARCH_NOMIC_BERT_MOE ||
            arch == LLM_ARCH_NEO_BERT || arch == LLM_ARCH_JINA_BERT_V2 || arch == LLM_ARCH_JINA_BERT_V3 || arch == LLM_ARCH_EUROBERT) {
        return false; // TODO vocab
    }
    if (arch == LLM_ARCH_PLM) {
        return false; // TODO tensor shapes
    }
    if (arch == LLM_ARCH_DEEPSEEK2OCR) {
        return false;
    }
    // FIXME: these hit scheduler/view-backed-output issues with WebGPU on CI.
#ifdef GGML_USE_WEBGPU
    if (arch == LLM_ARCH_DEEPSEEK32 || arch == LLM_ARCH_GLM_DSA || arch == LLM_ARCH_DOTS3NOTE || arch == LLM_ARCH_QWEN4EXP) {
        return false;
    }
#endif // GGML_USE_WEBGPU

    // FIXME: jamba produces incorrect output (~0.55 NMSE vs CPU) on the HIP
    // backend on RDNA3.5 (gfx1151); the SSM kernels need investigation.
#ifdef GGML_USE_HIP
    if (arch == LLM_ARCH_JAMBA) {
        return false;
    }
#endif // GGML_USE_HIP

    return true;
}

static int save_models(const llm_arch target_arch, const size_t seed, const int verbosity, const std::string & dir) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } log_old;

        int verbosity;

        user_data_t(int verbosity) : verbosity(verbosity) {
            llama_log_get(&log_old.callback, &log_old.user_data);
        }
    };
    user_data_t ud(verbosity);

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        int verbosity = common_log_get_verbosity(level);
        if (verbosity <= ud->verbosity) {
            ud->log_old.callback(level, text, ud->log_old.user_data);
        }
    }, &ud);

    for (const llm_arch & arch : llm_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (target_arch != LLM_ARCH_UNKNOWN && arch != target_arch) {
            continue;
        }
        if (arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            continue; // FIXME: ISWA KV cache initialization needs more fixture params
        }
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH) {
            continue;
        }
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            if (!llama_model_saver_supports_arch(arch) || !arch_supported(arch)) {
                LOG_INF("%s: %s model (%s) is unsupported, skipping\n", __func__, llm_arch_name(arch), moe ? "MoE" : "dense");
                continue;
            }
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
            auto model_and_ctx = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, {});
            const std::string path = dir + "/" + llm_arch_name(arch) + (moe ? "-moe.gguf" : "-dense.gguf");
            LOG_INF("%s: Saving %s model (%s) to %s...\n", __func__, llm_arch_name(arch), moe ? "MoE" : "dense", path.c_str());
            llama_model_save_to_file(model_and_ctx.first.get(), path.c_str());
        }
    }
    llama_log_set(ud.log_old.callback, ud.log_old.user_data);
    return 0;
}

static int test_backends(const llm_arch target_arch, const size_t seed, const int verbosity) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } log_old;

        int verbosity;

        user_data_t(int verbosity) : verbosity(verbosity) {
            llama_log_get(&log_old.callback, &log_old.user_data);
        }
    };
    user_data_t ud(verbosity);

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        int verbosity = common_log_get_verbosity(level);
        if (verbosity <= ud->verbosity) {
            ud->log_old.callback(level, text, ud->log_old.user_data);
        }
    }, &ud);

    const std::vector<llama_token> tokens = get_tokens(128, 128, seed);

    struct device_config {
        std::vector<ggml_backend_dev_t> devs;
        std::string                     label;
        llama_split_mode                split_mode;
        kv_config                       kvc;

        device_config(std::vector<ggml_backend_dev_t> devs, std::string name, llama_split_mode split_mode, kv_config kvc = {})
            : devs(std::move(devs)), label(std::move(name)), split_mode(split_mode), kvc(std::move(kvc)) {}
    };

    std::vector<device_config> dev_configs;
    size_t max_device_label_length = 4;
    {
        std::vector<ggml_backend_dev_t> devices_meta;
        {
            const size_t device_count = ggml_backend_dev_count();
            for (size_t i = 0; i < device_count; i++) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                dev_configs.emplace_back(std::vector<ggml_backend_dev_t>{dev}, ggml_backend_dev_description(dev), LLAMA_SPLIT_MODE_LAYER);
                max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());

                // cpu-based devices cannot be used in tensor split mode
                if (ggml_backend_dev_buffer_type(dev) != ggml_backend_cpu_buffer_type()) {
                    devices_meta.push_back(dev);
                }
            }
        }

        dev_configs.emplace_back(devices_meta, "Meta", LLAMA_SPLIT_MODE_TENSOR);

        // a host-resident cache reaches attention as a scheduler copy that is split by head
        kv_config kvc_host;
        kvc_host.offload_kqv = false;
        dev_configs.emplace_back(devices_meta, "Meta -nkvo", LLAMA_SPLIT_MODE_TENSOR, kvc_host);

        // with more than one stream that copy is 4d, one stride per stream
        kv_config kvc_host_streams = kvc_host;
        kvc_host_streams.n_seq_max = 2;
        dev_configs.emplace_back(devices_meta, "Meta -nkvo -np 2", LLAMA_SPLIT_MODE_TENSOR, kvc_host_streams);

        for (const device_config & dc : dev_configs) {
            max_device_label_length = std::max(max_device_label_length, dc.label.length());
        }
    }

    size_t max_arch_name_length = 0;
    for (const llm_arch & arch : llm_arch_all()) {
        max_arch_name_length = std::max(max_arch_name_length, strlen(llm_arch_name(arch)));
    }

    const std::string template_header  = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|%15s|%9s|\n";
    const std::string template_row_cfg = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|";
    const std::string template_row_res = "%15s %10s|%20s|\n";

    bool all_ok = true;
    common_log_flush(common_log_main());
    printf(template_header.c_str(), "Model arch.", "Device", "Config", "NMSE vs. CPU", "Roundtrip");
    printf("|");
    for (size_t i = 0; i < max_arch_name_length; i++) {
        printf("-");
    }
    printf("|");
    for (size_t i = 0; i < max_device_label_length; i++) {
        printf("-");
    }
    printf("|------|---------------|---------|\n");
    for (const llm_arch & arch : llm_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (target_arch != LLM_ARCH_UNKNOWN && arch != target_arch) {
            continue;
        }
        if (arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            // FIXME: ISWA KV cache initialization needs more fixture params
            // this arch also loses accuracy with a host-resident cache under split mode tensor
            // (perplexity 235.03 versus 227.23), the other ISWA archs are clean
            continue;
        }
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH) {
            continue;
        }

        const bool encode = arch == LLM_ARCH_T5 || arch == LLM_ARCH_DREAM || arch == LLM_ARCH_LLADA || arch == LLM_ARCH_LLADA_MOE || arch == LLM_ARCH_RND1;
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            const std::string config_name = moe ? "MoE" : "Dense";
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
            if (arch == LLM_ARCH_BAILINGMOE3) {
                GGML_ASSERT(gguf_remove_key(gguf_ctx.get(), "bailingmoe3.kda.safe_gate") >= 0);
            }
            std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_cpu;
            std::map<uint32_t, std::vector<float>> logits_cpu_per_n_seq;
            for (device_config & dc : dev_configs) {
                // print test config first; should anything fail during model loading or inference, at least we know which test case caused it
                printf(template_row_cfg.c_str(),
                    llm_arch_name(arch), dc.label.c_str(), config_name.c_str());
                fflush(stdout);

                std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_dev;
                std::vector<float> logits_dev;
                std::string status_nmse      = "\033[1;33mSKIP\033[0m";
                std::string status_roundtrip = "\033[1;33mSKIP\033[0m";
                char nmse_str[12] = {0};

                // an encoder-decoder model needs its own batch layout, so it stays on one sequence
                bool skip = !arch_supported(arch) || (dc.split_mode == LLAMA_SPLIT_MODE_TENSOR && dc.devs.empty()) ||
                    (encode && dc.kvc.n_seq_max > 1);
                if (!skip) {
                    if (dc.split_mode != LLAMA_SPLIT_MODE_TENSOR || llm_arch_supports_sm_tensor(arch)) {
                        // the reference runs the same batch layout, one sequence per stream
                        std::vector<float> & logits_cpu = logits_cpu_per_n_seq[dc.kvc.n_seq_max];
                        if (logits_cpu.empty()) {
                            kv_config kvc_cpu;
                            kvc_cpu.n_seq_max = dc.kvc.n_seq_max;
                            model_and_ctx_cpu = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, {}, LLAMA_SPLIT_MODE_LAYER, encode, kvc_cpu);
                            logits_cpu = get_logits(model_and_ctx_cpu.first.get(), model_and_ctx_cpu.second.get(), tokens, encode, dc.kvc.n_seq_max);
                        }
                        model_and_ctx_dev = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, dc.devs, dc.split_mode, encode, dc.kvc);
                        logits_dev = get_logits(model_and_ctx_dev.first.get(), model_and_ctx_dev.second.get(), tokens, encode, dc.kvc.n_seq_max);
                        const double nmse_val = nmse(logits_cpu, logits_dev);
                        snprintf(nmse_str, sizeof(nmse_str), "(%.2e)", nmse_val);
                        status_nmse = "\033[1;32mOK\033[0m";
                        if (nmse_val > 1e-4) {
                            all_ok = false;
                            status_nmse = "\033[1;31mFAIL\033[0m";
                        }
                    }

                    FILE * file = tmpfile(); // Can be null on Windows without administrator privileges.
                    // FIXME: when adding a tensor to a gguf_context a copy is made, this changes the pointer which the meta backend
                    //     in turn uses to map the tensors to their simple equivalents - this is fundamentally incompatible
                    if (file != nullptr && llama_model_saver_supports_arch(arch) && dc.split_mode != LLAMA_SPLIT_MODE_TENSOR) {
                        GGML_ASSERT(model_and_ctx_dev.first && model_and_ctx_dev.second);
                        llama_model_saver ms = llama_model_saver(model_and_ctx_dev.first.get());
                        ms.add_kv_from_model();
                        ms.add_tensors_from_model();
                        ms.save(file);
                        rewind(file);

                        auto model_and_ctx_roundtrip = get_model_and_ctx(nullptr, file, seed, dc.devs, dc.split_mode, encode, dc.kvc);
                        const std::vector<float> logits_roundtrip = get_logits(
                            model_and_ctx_roundtrip.first.get(), model_and_ctx_roundtrip.second.get(), tokens, encode, dc.kvc.n_seq_max);
                        status_roundtrip = "\033[1;32mOK\033[0m";
                        GGML_ASSERT(logits_roundtrip.size() == logits_dev.size());
                        for (size_t i = 0; i < logits_roundtrip.size(); i++) {
                            if (logits_roundtrip[i] != logits_dev[i]) {
                                all_ok = false;
                                status_roundtrip = "\033[1;31mFAIL\033[0m";
                                break;
                            }
                        }
                    }
                }

                // log the results for this test case
                printf(template_row_res.c_str(),
                    status_nmse.c_str(), nmse_str, status_roundtrip.c_str());
            }
        }
    }
    llama_log_set(ud.log_old.callback, ud.log_old.user_data);
    return all_ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    // init the logger at max verbosity. filter with a custom callback respecting the user-configure verbosity
    common_log_set_verbosity_thold(LOG_LEVEL_DEBUG);
    common_init();

    std::random_device rd;

    llm_arch arch = LLM_ARCH_UNKNOWN;
    size_t seed = rd();
    std::string out;
    bool test_phase_workspace = false;
    bool test_live_context_workspace = false;
    bool test_copy_name = false;

    int verbosity = LOG_LEVEL_ERROR;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv);
            return 0;
        }
        if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--arch") == 0) {
            if (i + 1 < argc) {
                const std::string arch_name = argv[++i];
                arch = llm_arch_from_string(arch_name);
                if (arch == LLM_ARCH_UNKNOWN) {
                    LOG_ERR("%s: unkown LLM architecture: %s\n", __func__, arch_name.c_str());
                    return 1;
                }
            } else {
                usage(argv);
                return 1;
            }
        }
        if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--seed") == 0) {
            if (i + 1 < argc) {
                seed = std::stoull(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        }
        if (strcmp(argv[i], "-v") == 0) {
            if (i + 1 < argc) {
                verbosity = std::stoull(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        }
        if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--out") == 0) {
            if (i + 1 < argc) {
                out = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        }
        if (strcmp(argv[i], "--test-phase-workspace") == 0) {
            test_phase_workspace = true;
            continue;
        }
        if (strcmp(argv[i], "--test-live-context-workspace") == 0) {
            test_live_context_workspace = true;
            continue;
        }
        if (strcmp(argv[i], "--test-sched-copy-name") == 0) {
            test_copy_name = true;
            continue;
        }
    }
    printf("%s: using seed %zu\n", __func__, seed);

    try {
        if (test_phase_workspace) {
            test_phase_workspace_runtime_reserve(seed);
            test_phase_workspace_mtp_lifecycle(seed);
            test_phase_workspace_mismatched_placement(seed);
            test_phase_workspace_late_pipeline_fallback(seed);
            return 0;
        }
        if (test_live_context_workspace) {
            test_live_context_workspace_reserve(seed);
            test_live_context_workspace_iswa_reserve(seed);
            test_live_context_workspace_unsupported(seed);
            return 0;
        }
        if (test_copy_name) {
            test_sched_copy_name();
            return 0;
        }
        if (!out.empty()) {
            return save_models(arch, seed, verbosity, out);
        }
        return test_backends(arch, seed, verbosity);
    } catch (const std::exception & err) {
        fprintf(stderr, "encountered runtime error: %s\n", err.what());
        return -1;
    }
}
