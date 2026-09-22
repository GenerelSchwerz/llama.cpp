#include "../ggml/src/ggml-backend-impl.h"
#include "../ggml/src/ggml-backend-moe.h"
#include "common.h"
#include "fit.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"
#include "gguf.h"
#include "hash/hash.h"
#include "json.h"
#include "llama-cpp.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"

// TODO: replace with #include "llama-ext.h" in the future
#include "../src/llama-arch.h"
#include "../src/llama-context.h"
#include "../src/llama-ext.h"
#include "../src/llama-memory-hybrid-idx.h"
#include "../src/llama-model-saver.h"
#include "../src/llama-model.h"

#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <locale>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>
#if !defined(_WIN32)
#    include <sys/wait.h>
#endif

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
    printf(
        "Usage: %s [-a/--arch arch] [-s/--seed seed] [-o/--out dir] [-v N] [-h/--help] [--test-phase-workspace] "
        "[--test-live-context-workspace] [--test-speculative-limits] [--test-moe-cache-selector] "
        "[--test-moe-placement] [--test-moe-joint-measurement]\n",
        argv[0]);
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

static gguf_context_ptr get_gguf_ctx(
        const llm_arch arch, const bool moe, const bool mtp = false,
        uint32_t n_expert = 2, uint32_t n_expert_used = 2, int32_t n_layer_base = -1,
        uint32_t n_vocab_override = 0) {
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(arch, ret.get());
    const uint32_t n_ctx = 256;

    uint32_t n_vocab = 128;
    uint32_t n_embd  = 256;
    uint32_t n_head  = 2;
    uint32_t n_ff    = 384;
    uint32_t n_layer = 2;
    if (n_vocab_override != 0) {
        n_vocab = n_vocab_override;
    }
    if (n_layer_base >= 0) {
        n_layer = static_cast<uint32_t>(n_layer_base);
    }
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
        //n_vocab = 4096; // must be >= the hard-coded codec head size (3072)
        n_vocab = 3072; // TODO: should be 4096, but user code cannot get `n_vocab_out` yet [TAG_LLAMA_N_VOCAB_OUT]
    } else if (arch == LLM_ARCH_HRM_TEXT) {
        n_layer = 8; // 1 layer per stack x 2 h-cycles x (3 l-cycles + 1) cache slots
    }

    GGML_ASSERT(!mtp || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE);
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
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, std::min<uint32_t>(1, n_layer));
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
    } else if (arch == LLM_ARCH_COHERE2MOE || arch == LLM_ARCH_MIMO2 || arch == LLM_ARCH_STEP35 || arch == LLM_ARCH_SPARK2_5 ||
            arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_GRANITE_SWA || arch == LLM_ARCH_DOTS3NOTE ||
            arch == LLM_ARCH_MAPLE) {
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

    if (arch == LLM_ARCH_HRM_TEXT) {
        // 8 cache slots alias 2 physical blocks: 1 low-stack layer + 1 high-stack layer
        ms.add_kv(LLM_KV_HRM_LAYERS_PER_STACK, uint32_t(1));
        ms.add_kv(LLM_KV_HRM_H_CYCLES,         uint32_t(2));
        ms.add_kv(LLM_KV_HRM_L_CYCLES,         uint32_t(3));
    }

    if (arch == LLM_ARCH_MAPLE) {
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP, 7.0f);
    }

    // dummy tokenizer: token ids are derived from fixed-size chunks and detokenized as hex ids
    {
        std::vector<std::string> tokenizer_list(n_vocab);
        std::vector<float>       tokenizer_scores(n_vocab, 0.0f);

        ms.add_kv(LLM_KV_TOKENIZER_MODEL,         "test");
        for (uint32_t i = 0; i < n_vocab; i++) {
            tokenizer_list[i] = "tok_" + std::to_string(i);
        }
        ms.add_kv(LLM_KV_TOKENIZER_LIST,   tokenizer_list);
        ms.add_kv(LLM_KV_TOKENIZER_SCORES, tokenizer_scores);
    }

    // ms.add_kv(LLM_KV_DENSE_2_FEAT_OUT,     n_embd);
    // ms.add_kv(LLM_KV_DENSE_3_FEAT_IN,      n_embd);

    if (moe) {
        ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, n_ff);
        ms.add_kv(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, n_ff / 2);  // distinct from n_ff so a saver key-clobber surfaces on reload
        ms.add_kv(LLM_KV_EXPERT_LATENT_LENGTH,       n_ff);
        ms.add_kv(LLM_KV_INTERLEAVE_MOE_LAYER_STEP,  uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_COUNT,               n_expert);
        ms.add_kv(LLM_KV_EXPERT_USED_COUNT,          n_expert_used);
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

static std::pair<llama_model_ptr, llama_context_ptr> get_model_and_ctx(
        struct gguf_context * gguf_ctx, FILE * file, const size_t seed, const std::vector<ggml_backend_dev_t> & devs,
        const llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER, bool encode = false) {
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

struct mtp_finish_observation {
    int32_t n_embd;
    bool active = false;
    size_t n_batched_eh_proj = 0;
};

static bool observe_mtp_finish(ggml_tensor * tensor, bool ask, void * user_data) {
    auto * observation = static_cast<mtp_finish_observation *>(user_data);
    if (observation->active && ask && strcmp(tensor->name, "mtp_eh_proj-2") == 0 &&
            tensor->ne[0] == observation->n_embd && tensor->ne[1] == 2 && tensor->ne[2] == 1 && tensor->ne[3] == 1) {
        observation->n_batched_eh_proj++;
    }
    return false;
}

static llama_context_ptr make_phase_workspace_context(
        llama_model * model, llama_context_type type, llama_context * other = nullptr, uint32_t n_seq_max = 1,
        ggml_backend_sched_eval_callback cb_eval = nullptr, void * cb_eval_user_data = nullptr);

static llama_model_ptr make_live_context_workspace_model(llm_arch arch, size_t seed, bool moe = false) {
    gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
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

static void test_live_context_workspace_indexer_reserve(size_t seed) {
    llama_model_ptr model = make_live_context_workspace_model(LLM_ARCH_QWEN4EXP, seed, true);
    llama_context_params ctx_params = make_live_context_workspace_params();

    llama_context_ptr ctx(llama_init_from_model(model.get(), ctx_params));
    GGML_ASSERT(ctx);

    const auto initial = ctx->make_sched_reserve_plan(1, 1);
    GGML_ASSERT(initial.live_kv);
    GGML_ASSERT(initial.n_kv == 256);

    auto assert_reserve = [&](uint32_t n_kv) {
        llama_memory_context_ptr reserve = ctx->get_memory()->init_reserve(n_kv);
        auto * hybrid = dynamic_cast<llama_memory_hybrid_idx_context *>(reserve.get());
        GGML_ASSERT(hybrid != nullptr);
        GGML_ASSERT(hybrid->get_attn()->get_n_kv() == n_kv);
        GGML_ASSERT(hybrid->get_idx() != nullptr);
        GGML_ASSERT(hybrid->get_idx()->get_n_kv() == n_kv);
    };
    assert_reserve(256);

    ctx->sched_reserve(ctx_params.n_batch, 257);
    const auto grown = ctx->make_sched_reserve_plan(ctx_params.n_batch, 257);
    GGML_ASSERT(grown.n_kv == 512);
    assert_reserve(512);
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
        llama_model * model, llama_context_type type, llama_context * other, uint32_t n_seq_max,
        ggml_backend_sched_eval_callback cb_eval, void * cb_eval_user_data) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = 32;
    params.n_batch = 32;
    params.n_ubatch = 32;
    params.n_seq_max = n_seq_max;
    params.n_outputs_max = type == LLAMA_CONTEXT_TYPE_MTP ? n_seq_max : 4;
    params.n_threads = 4;
    params.n_threads_batch = 4;
    params.ctx_type = type;
    params.ctx_other = other;
    params.phase_aware_workspace = true;
    params.cb_eval = cb_eval;
    params.cb_eval_user_data = cb_eval_user_data;
    return llama_context_ptr(llama_init_from_model(model, params));
}

static void test_speculative_limits(size_t seed) {
    gguf_context_ptr gguf_ctx = get_gguf_ctx(LLM_ARCH_QWEN35, false, true);

    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    model_params.load_mtp = true;
    ggml_backend_dev_t devices[] = { nullptr };
    model_params.devices = devices;

    size_t tensor_seed = seed;
    llama_model_ptr model(llama_model_init_from_user(gguf_ctx.get(), set_tensor_data, &tensor_seed, model_params));
    GGML_ASSERT(model);

    const uint32_t n_seq = 2;
    llama_context_ptr target = make_phase_workspace_context(model.get(), LLAMA_CONTEXT_TYPE_DEFAULT, nullptr, n_seq);
    llama_context_ptr draft = make_phase_workspace_context(model.get(), LLAMA_CONTEXT_TYPE_MTP, target.get(), n_seq);
    GGML_ASSERT(target && draft);

    const llama_tokens prompt_match = { 99, 1, 2, 3, 4, 1 };
    const llama_tokens prompt_none  = { 99, 1, 2, 3, 4, 5 };

    auto make_spec = [&](int32_t n_max, int32_t n_min, std::vector<common_speculative_type> types) {
        common_params_speculative params;
        params.types = std::move(types);
        params.draft.n_max = n_max;
        params.draft.n_min = n_min;
        params.draft.p_min = 0.0f;
        params.draft.ctx_tgt = target.get();
        params.draft.ctx_dft = draft.get();
        params.ngram_simple.size_n = 2;
        params.ngram_simple.size_m = 2;
        return common_speculative_ptr(common_speculative_init(params, n_seq));
    };

    {
        auto spec = make_spec(2, 0, { COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, COMMON_SPECULATIVE_TYPE_DRAFT_MTP });
        llama_tokens result_zero;
        llama_tokens result_positive;
        common_speculative_get_draft_params(spec.get(), 0) = {
            true, 0, 0, 2, &prompt_match, &result_zero,
        };
        common_speculative_get_draft_params(spec.get(), 1) = {
            true, 1, 0, 6, &prompt_none, &result_positive,
        };

        common_speculative_draft(spec.get());

        GGML_ASSERT(result_zero.empty());
        GGML_ASSERT(result_positive.size() == 1);
        GGML_ASSERT(!common_speculative_get_draft_params(spec.get(), 0).drafting);
        GGML_ASSERT(!common_speculative_get_draft_params(spec.get(), 1).drafting);
        GGML_ASSERT(llama_memory_seq_pos_max(llama_get_memory(draft.get()), 0) == -1);
        GGML_ASSERT(llama_memory_seq_pos_max(llama_get_memory(draft.get()), 1) == 0);
        GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(draft.get()), 1, 0, -1));
    }

    {
        auto spec = make_spec(0, 0, { COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, COMMON_SPECULATIVE_TYPE_DRAFT_MTP });
        llama_tokens result_ngram;
        llama_tokens result_none;
        common_speculative_get_draft_params(spec.get(), 0) = {
            true, -1, 0, 2, &prompt_match, &result_ngram,
        };
        common_speculative_get_draft_params(spec.get(), 1) = {
            true, -1, 0, 6, &prompt_none, &result_none,
        };

        common_speculative_draft(spec.get());

        GGML_ASSERT(result_ngram == llama_tokens({ 3, 4 }));
        GGML_ASSERT(result_none.empty());
        GGML_ASSERT(llama_memory_seq_pos_max(llama_get_memory(draft.get()), 0) == -1);
        GGML_ASSERT(llama_memory_seq_pos_max(llama_get_memory(draft.get()), 1) == -1);
    }

    {
        auto spec = make_spec(2, 0, { COMMON_SPECULATIVE_TYPE_DRAFT_MTP });
        llama_tokens result_short;
        llama_tokens result_long;
        common_speculative_get_draft_params(spec.get(), 0) = {
            true, 1, 0, 2, &prompt_none, &result_short,
        };
        common_speculative_get_draft_params(spec.get(), 1) = {
            true, -1, 0, 6, &prompt_none, &result_long,
        };

        common_speculative_draft(spec.get());

        GGML_ASSERT(result_short.size() == 1);
        GGML_ASSERT(result_long.size() == 2);
        GGML_ASSERT(!common_speculative_retain_draft_state(spec.get(), 1));
        GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(draft.get()), 0, 0, -1));
        GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(draft.get()), 1, 0, -1));
    }

    {
        auto spec = make_spec(2, 2, { COMMON_SPECULATIVE_TYPE_DRAFT_MTP });
        llama_tokens result;
        common_speculative_get_draft_params(spec.get(), 0) = {
            true, 1, 0, 2, &prompt_none, &result,
        };

        common_speculative_draft(spec.get());

        GGML_ASSERT(result.empty());
        GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(draft.get()), 0, 0, -1));
    }

    {
        mtp_finish_observation observation = { llama_model_n_embd_out(model.get()) };
        llama_context_ptr target_retained = make_phase_workspace_context(
                model.get(), LLAMA_CONTEXT_TYPE_DEFAULT, nullptr, n_seq);
        llama_context_ptr draft_retained = make_phase_workspace_context(
                model.get(), LLAMA_CONTEXT_TYPE_MTP, nullptr, n_seq, observe_mtp_finish, &observation);
        GGML_ASSERT(target_retained && draft_retained);
        GGML_ASSERT(llama_get_ctx_other(draft_retained.get()) == nullptr);

        common_params_speculative params;
        params.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        params.draft.n_max = 1;
        params.draft.n_min = 0;
        params.draft.p_min = 0.0f;
        params.draft.ctx_tgt = target_retained.get();
        params.draft.ctx_dft = draft_retained.get();
        common_speculative_ptr spec(common_speculative_init(params, n_seq));
        GGML_ASSERT(spec);

        llama_tokens proposals[2];
        for (llama_seq_id seq_id = 0; seq_id < 2; ++seq_id) {
            common_speculative_get_draft_params(spec.get(), seq_id) = {
                true, 1, 0, 2 + seq_id, &prompt_none, &proposals[seq_id],
            };
        }
        common_speculative_draft(spec.get());
        GGML_ASSERT(proposals[0].size() == 1 && proposals[1].size() == 1);

        auto get_seq_state = [](llama_context * ctx, llama_seq_id seq_id) {
            const size_t size = llama_state_seq_get_size_ext(ctx, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE);
            GGML_ASSERT(size > 0);
            std::vector<uint8_t> data(size);
            GGML_ASSERT(llama_state_seq_get_data_ext(
                    ctx, data.data(), data.size(), seq_id, LLAMA_STATE_SEQ_FLAGS_NONE) == size);
            return data;
        };
        auto get_mtp_state = [](common_speculative * spec, llama_seq_id seq_id) {
            std::vector<uint8_t> data;
            GGML_ASSERT(common_speculative_get_mtp_state(spec, seq_id, data));
            return data;
        };

        std::vector<uint8_t> proposal_state[2];
        std::vector<uint8_t> pending_initial[2];
        for (llama_seq_id seq_id = 0; seq_id < 2; ++seq_id) {
            proposal_state[seq_id] = get_seq_state(draft_retained.get(), seq_id);
            pending_initial[seq_id] = get_mtp_state(spec.get(), seq_id);
        }

        llama_batch verify = llama_batch_init(4, 0, 1);
        for (llama_seq_id seq_id = 0; seq_id < 2; ++seq_id) {
            common_batch_add(verify, 2 + seq_id, 0, { seq_id }, true);
            common_batch_add(verify, proposals[seq_id][0], 1, { seq_id }, true);
        }
        GGML_ASSERT(llama_decode(target_retained.get(), verify) == 0);
        llama_synchronize(target_retained.get());

        auto * draft_mem = llama_get_memory(draft_retained.get());
        for (llama_seq_id seq_id = 0; seq_id < 2; ++seq_id) {
            GGML_ASSERT(llama_memory_seq_rm(draft_mem, seq_id, -1, -1));
            GGML_ASSERT(common_speculative_set_mtp_state(spec.get(), seq_id, pending_initial[seq_id]));
        }
        GGML_ASSERT(common_speculative_process(spec.get(), verify));

        std::vector<uint8_t> canonical_accepted_state[2];
        std::vector<uint8_t> canonical_accepted_pending[2];
        for (llama_seq_id seq_id = 0; seq_id < 2; ++seq_id) {
            canonical_accepted_state[seq_id] = get_seq_state(draft_retained.get(), seq_id);
            common_speculative_accept(spec.get(), seq_id, 1);
            canonical_accepted_pending[seq_id] = get_mtp_state(spec.get(), seq_id);
        }

        common_speculative_accept(spec.get(), 0, 0);
        GGML_ASSERT(llama_memory_seq_rm(draft_mem, 0, 1, -1));
        GGML_ASSERT(llama_memory_seq_rm(draft_mem, 1, 2, -1));

        std::vector<uint8_t> canonical_state[2];
        std::vector<uint8_t> canonical_pending[2];
        for (llama_seq_id seq_id = 0; seq_id < 2; ++seq_id) {
            canonical_state[seq_id] = get_seq_state(draft_retained.get(), seq_id);
            canonical_pending[seq_id] = get_mtp_state(spec.get(), seq_id);

            GGML_ASSERT(llama_memory_seq_rm(draft_mem, seq_id, -1, -1));
            GGML_ASSERT(llama_state_seq_set_data_ext(
                    draft_retained.get(), proposal_state[seq_id].data(), proposal_state[seq_id].size(),
                    seq_id, LLAMA_STATE_SEQ_FLAGS_NONE) == proposal_state[seq_id].size());
            GGML_ASSERT(common_speculative_set_mtp_state(spec.get(), seq_id, pending_initial[seq_id]));
            GGML_ASSERT(common_speculative_retain_draft_state(spec.get(), seq_id));
        }

        GGML_ASSERT(common_speculative_process(spec.get(), verify));
        common_speculative_accept(spec.get(), 0, 0);
        common_speculative_accept(spec.get(), 1, 1);
        GGML_ASSERT(common_speculative_has_deferred_accept(spec.get(), 0));
        GGML_ASSERT(common_speculative_has_deferred_accept(spec.get(), 1));
        GGML_ASSERT(common_speculative_finish_accept(spec.get(), {
            { 0, 0 },
            { 1, 1 },
        }));

        for (llama_seq_id seq_id = 0; seq_id < 2; ++seq_id) {
            GGML_ASSERT(!common_speculative_has_deferred_accept(spec.get(), seq_id));
            GGML_ASSERT(get_seq_state(draft_retained.get(), seq_id) == canonical_state[seq_id]);
            GGML_ASSERT(get_mtp_state(spec.get(), seq_id) == canonical_pending[seq_id]);

            GGML_ASSERT(llama_memory_seq_rm(draft_mem, seq_id, -1, -1));
            GGML_ASSERT(llama_state_seq_set_data_ext(
                    draft_retained.get(), proposal_state[seq_id].data(), proposal_state[seq_id].size(),
                    seq_id, LLAMA_STATE_SEQ_FLAGS_NONE) == proposal_state[seq_id].size());
            GGML_ASSERT(common_speculative_set_mtp_state(spec.get(), seq_id, pending_initial[seq_id]));
            GGML_ASSERT(common_speculative_retain_draft_state(spec.get(), seq_id));
        }

        GGML_ASSERT(common_speculative_process(spec.get(), verify));
        common_speculative_accept(spec.get(), 0, 1);
        common_speculative_accept(spec.get(), 1, 1);

        llama_synchronize(draft_retained.get());
        observation.active = true;
        GGML_ASSERT(common_speculative_finish_accept(spec.get(), {
            { 0, 1 },
            { 1, 1 },
        }));
        llama_synchronize(draft_retained.get());
        observation.active = false;
        GGML_ASSERT(observation.n_batched_eh_proj == 1);
        for (llama_seq_id seq_id = 0; seq_id < 2; ++seq_id) {
            GGML_ASSERT(!common_speculative_has_deferred_accept(spec.get(), seq_id));
            GGML_ASSERT(get_seq_state(draft_retained.get(), seq_id) == canonical_accepted_state[seq_id]);
            GGML_ASSERT(get_mtp_state(spec.get(), seq_id) == canonical_accepted_pending[seq_id]);

            GGML_ASSERT(llama_memory_seq_rm(draft_mem, seq_id, -1, -1));
            GGML_ASSERT(llama_state_seq_set_data_ext(
                    draft_retained.get(), proposal_state[seq_id].data(), proposal_state[seq_id].size(),
                    seq_id, LLAMA_STATE_SEQ_FLAGS_NONE) == proposal_state[seq_id].size());
            GGML_ASSERT(common_speculative_set_mtp_state(spec.get(), seq_id, pending_initial[seq_id]));
            GGML_ASSERT(common_speculative_retain_draft_state(spec.get(), seq_id));
        }

        GGML_ASSERT(common_speculative_process(spec.get(), verify));
        common_speculative_accept(spec.get(), 0, 1);
        common_speculative_accept(spec.get(), 1, 1);
        GGML_ASSERT(!common_speculative_finish_accept(spec.get(), {
            { 1, 2 },
            { 0, 1 },
        }));
        GGML_ASSERT(!common_speculative_has_deferred_accept(spec.get(), 0));
        GGML_ASSERT(!common_speculative_has_deferred_accept(spec.get(), 1));

        for (llama_seq_id seq_id = 0; seq_id < 2; ++seq_id) {
            GGML_ASSERT(llama_memory_seq_rm(draft_mem, seq_id, -1, -1));
            GGML_ASSERT(llama_state_seq_set_data_ext(
                    draft_retained.get(), proposal_state[seq_id].data(), proposal_state[seq_id].size(),
                    seq_id, LLAMA_STATE_SEQ_FLAGS_NONE) == proposal_state[seq_id].size());
            GGML_ASSERT(common_speculative_set_mtp_state(spec.get(), seq_id, pending_initial[seq_id]));
            GGML_ASSERT(common_speculative_retain_draft_state(spec.get(), seq_id));
        }

        GGML_ASSERT(common_speculative_process(spec.get(), verify));
        common_speculative_accept(spec.get(), 0, 1);
        common_speculative_accept(spec.get(), 1, 1);
        GGML_ASSERT(!common_speculative_finish_accept(spec.get(), {
            { 0, 2 },
        }));
        GGML_ASSERT(!common_speculative_has_deferred_accept(spec.get(), 0));
        GGML_ASSERT(common_speculative_has_deferred_accept(spec.get(), 1));
        GGML_ASSERT(common_speculative_finish_accept(spec.get(), {
            { 1, 0 },
        }));
        GGML_ASSERT(get_seq_state(draft_retained.get(), 1) == proposal_state[1]);

        GGML_ASSERT(llama_memory_seq_rm(draft_mem, 0, -1, -1));
        GGML_ASSERT(llama_state_seq_set_data_ext(
                draft_retained.get(), proposal_state[0].data(), proposal_state[0].size(),
                0, LLAMA_STATE_SEQ_FLAGS_NONE) == proposal_state[0].size());
        GGML_ASSERT(common_speculative_set_mtp_state(spec.get(), 0, pending_initial[0]));
        GGML_ASSERT(common_speculative_retain_draft_state(spec.get(), 0));
        GGML_ASSERT(common_speculative_has_deferred_accept(spec.get(), 0));
        GGML_ASSERT(common_speculative_set_mtp_state(spec.get(), 0, {}));
        GGML_ASSERT(!common_speculative_has_deferred_accept(spec.get(), 0));
        GGML_ASSERT(llama_memory_seq_rm(draft_mem, 0, -1, -1));

        llama_batch_free(verify);
    }
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

static std::vector<float> get_logits(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens, bool encode = false) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_ctx    = llama_n_ctx(lctx);
    const uint32_t n_tokens = tokens.size();
    llama_batch batch = llama_batch_init(n_ctx, 0, 1);
    GGML_ASSERT(n_tokens <= n_ctx);
    for (uint32_t pos = 0; pos < n_tokens; pos++) {
        common_batch_add(batch, tokens[pos], pos, {0}, true);
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

// decode two sequences in one batch, compare each with a decode of it alone
// logits_a: logits of tokens decoded alone with the same device config
static bool test_parallel_seqs(llama_model * model, const std::vector<llama_token> & tokens, const std::vector<float> & logits_a, bool encode) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_tokens = tokens.size();

    // same per-sequence n_ctx as the reference context
    const auto make_cparams = [&](uint32_t n_seq_max) {
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx           = n_seq_max*llama_model_n_ctx_train(model);
        cparams.n_threads       = 4;
        cparams.n_threads_batch = 4;
        cparams.n_seq_max       = n_seq_max;
        if (!encode) {
            cparams.n_ubatch = 64;
        }
        return cparams;
    };

    llama_context_ptr ctx_par(llama_init_from_model(model, make_cparams(2)));
    llama_context_ptr ctx_b(llama_init_from_model(model, make_cparams(1)));
    if (!ctx_par || !ctx_b) {
        return false;
    }

    std::vector<llama_token> tokens_b(n_tokens);
    for (uint32_t i = 0; i < n_tokens; i++) {
        tokens_b[i] = tokens[(i + 1) % n_tokens];
    }

    llama_batch batch_par = llama_batch_init(2*n_tokens, 0, 1);
    llama_batch batch_b   = llama_batch_init(n_tokens, 0, 1);
    for (uint32_t pos = 0; pos < n_tokens; pos++) {
        common_batch_add(batch_par, tokens[pos],   pos, {0}, true);
        common_batch_add(batch_par, tokens_b[pos], pos, {1}, true);
        common_batch_add(batch_b,   tokens_b[pos], pos, {0}, true);
    }
    bool ok = true;
    if (encode) {
        ok = llama_encode(ctx_par.get(), batch_par) == 0 && llama_encode(ctx_b.get(), batch_b) == 0;
    }
    ok = ok && llama_decode(ctx_par.get(), batch_par) == 0 && llama_decode(ctx_b.get(), batch_b) == 0;
    llama_batch_free(batch_par);
    llama_batch_free(batch_b);
    if (!ok) {
        return false;
    }

    std::vector<float> logits_par_a;
    std::vector<float> logits_par_b;
    std::vector<float> logits_b;
    for (uint32_t i = 0; i < n_tokens; i++) {
        const float * l_par_a = llama_get_logits_ith(ctx_par.get(), 2*i);
        const float * l_par_b = llama_get_logits_ith(ctx_par.get(), 2*i + 1);
        const float * l_b     = llama_get_logits_ith(ctx_b.get(), i);
        if (l_par_a == nullptr || l_par_b == nullptr || l_b == nullptr) {
            return false;
        }
        logits_par_a.insert(logits_par_a.end(), l_par_a, l_par_a + n_vocab);
        logits_par_b.insert(logits_par_b.end(), l_par_b, l_par_b + n_vocab);
        logits_b.insert(logits_b.end(), l_b, l_b + n_vocab);
    }

    // ubatch splits differ from the reference, so use the NMSE limit of the CPU comparison; NaN fails
    return nmse(logits_a, logits_par_a) <= 1e-4 && nmse(logits_b, logits_par_b) <= 1e-4;
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
        case LLM_ARCH_MAPLE:
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
    if (arch == LLM_ARCH_DEEPSEEK32 || arch == LLM_ARCH_GLM_DSA || arch == LLM_ARCH_DOTS3NOTE || arch == LLM_ARCH_QWEN4EXP ||
            arch == LLM_ARCH_HY_V4) {
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

static int test_moe_cache_selector_precedence(const size_t seed) {
    ggml_backend_dev_t         cache_dev  = nullptr;
    ggml_backend_buffer_type_t cache_buft = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto               get_cache_buft =
            reg != nullptr ? reinterpret_cast<ggml_backend_moe_cache_buffer_type_t>(
                                 ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_CACHE_BUFFER_TYPE_PROC_NAME)) :
                             nullptr;
        if (get_cache_buft != nullptr) {
            cache_dev  = dev;
            cache_buft = get_cache_buft();
            break;
        }
    }
    if (cache_dev == nullptr || cache_buft == nullptr) {
        fprintf(stderr, "test-moe-cache-selector: SKIP (no backend with MoE cache support)\n");
        return 77;
    }

    gguf_context_ptr              moe_metadata   = get_gguf_ctx(LLM_ARCH_QWEN3MOE, true);
    gguf_context_ptr              dense_metadata = get_gguf_ctx(LLM_ARCH_LLAMA, false);
    ggml_backend_dev_t            devices[]      = { cache_dev, nullptr };
    const llama_model_layer_range both_layers[]  = {
        { 0, 1 }
    };
    const llama_model_layer_range layer_zero[] = {
        { 0, 0 }
    };
    const llama_model_layer_range missing_layer[] = {
        { 2, 2 }
    };
    static const char * layer_zero_experts = "blk\\.0\\.ffn_(up|down|gate|gate_up)_(ch|)exps";
    static const char * layer_zero_up      = "blk\\.0\\.ffn_up_(ch|)exps";

    auto make_params = [&](const llama_model_layer_range * ranges, size_t n_ranges,
                           const llama_model_tensor_buft_override * overrides, int32_t slots = 4) {
        llama_model_params params              = llama_model_default_params();
        params.devices                         = devices;
        params.n_gpu_layers                    = 99;
        params.split_mode                      = LLAMA_SPLIT_MODE_LAYER;
        params.progress_callback               = silent_model_load_progress;
        params.moe_expert_cache_slots          = slots;
        params.moe_expert_cache_layer_ranges   = ranges;
        params.n_moe_expert_cache_layer_ranges = n_ranges;
        params.tensor_buft_overrides           = overrides;
        return params;
    };
    auto load = [&](gguf_context * metadata, llama_model_params params) {
        size_t tensor_seed = seed;
        return llama_model_ptr(llama_model_init_from_user(metadata, set_tensor_data, &tensor_seed, params));
    };
    const auto is_cached = [&](const ggml_tensor * tensor) {
        return tensor != nullptr && tensor->buffer != nullptr &&
               ggml_backend_buffer_get_type(tensor->buffer) == cache_buft;
    };
    const auto uses_buft = [&](const ggml_tensor * tensor, ggml_backend_buffer_type_t buft) {
        if (tensor == nullptr || tensor->buffer == nullptr) {
            return false;
        }
        return ggml_backend_buffer_get_type(tensor->buffer) == buft;
    };
    const auto layer_cached = [&](const llama_model & model, int layer) {
        const auto & block = model.layers.at(layer);
        return is_cached(block.ffn_gate_exps) && is_cached(block.ffn_up_exps) && is_cached(block.ffn_down_exps);
    };
    const auto layer_uses_buft = [&](const llama_model & model, int layer, ggml_backend_buffer_type_t buft) {
        const auto & block = model.layers.at(layer);
        return uses_buft(block.ffn_gate_exps, buft) && uses_buft(block.ffn_up_exps, buft) &&
               uses_buft(block.ffn_down_exps, buft);
    };
    const auto layer_is_host = [&](const llama_model & model, int layer) {
        const auto & block   = model.layers.at(layer);
        const auto   is_host = [](const ggml_tensor * tensor) {
            return tensor != nullptr && tensor->buffer != nullptr &&
                   ggml_backend_buft_is_host(ggml_backend_buffer_get_type(tensor->buffer));
        };
        return is_host(block.ffn_gate_exps) && is_host(block.ffn_up_exps) && is_host(block.ffn_down_exps);
    };
    const auto print_layer_bufts = [&](const llama_model & model, int layer) {
        const auto & block = model.layers.at(layer);
        const auto   name  = [](const ggml_tensor * tensor) {
            return tensor != nullptr && tensor->buffer != nullptr ?
                       ggml_backend_buft_name(ggml_backend_buffer_get_type(tensor->buffer)) :
                       "none";
        };
        fprintf(stderr, "test-moe-cache-selector: layer=%d gate=%s up=%s down=%s\n", layer, name(block.ffn_gate_exps),
                name(block.ffn_up_exps), name(block.ffn_down_exps));
    };

    bool       ok          = true;
    const auto expect_load = [&](const char * label, gguf_context * metadata, llama_model_params params,
                                 bool expected) {
        llama_model_ptr model = load(metadata, params);
        fprintf(stderr, "test-moe-cache-selector: %-28s loaded=%d expected=%d\n", label, model != nullptr, expected);
        if ((model != nullptr) != expected) {
            ok = false;
        }
        return model;
    };

    {
        auto model = expect_load("selected baseline", moe_metadata.get(), make_params(both_layers, 1, nullptr), true);
        if (model &&
            (!layer_cached(*model, 0) || !layer_cached(*model, 1) || model->moe_expert_cache_memory().empty())) {
            fprintf(stderr, "test-moe-cache-selector: baseline placement mismatch\n");
            ok = false;
        }
    }

    llama_model_tensor_buft_override cpu_override[] = {
        { layer_zero_experts, ggml_backend_cpu_buffer_type() },
        { nullptr,            nullptr                        }
    };
    {
        auto model =
            expect_load("selected explicit CPU", moe_metadata.get(), make_params(both_layers, 1, cpu_override), true);
        if (model && (layer_cached(*model, 0) || !layer_is_host(*model, 0) || !layer_cached(*model, 1) ||
                      model->moe_expert_cache_memory().empty())) {
            fprintf(stderr, "test-moe-cache-selector: explicit CPU placement mismatch\n");
            print_layer_bufts(*model, 0);
            print_layer_bufts(*model, 1);
            ok = false;
        }
    }

    std::vector<llama_model_tensor_buft_override> ncmoe_overrides;
    llm_add_n_cpu_ffn_overrides(1, LLM_FFN_EXPS_REGEX, ncmoe_overrides);
    ncmoe_overrides.push_back({ nullptr, nullptr });
    {
        auto model = expect_load("selected -ncmoe 1", moe_metadata.get(),
                                 make_params(both_layers, 1, ncmoe_overrides.data()), true);
        if (model && (layer_cached(*model, 0) || !layer_is_host(*model, 0) || !layer_cached(*model, 1))) {
            fprintf(stderr, "test-moe-cache-selector: -ncmoe placement mismatch\n");
            print_layer_bufts(*model, 0);
            print_layer_bufts(*model, 1);
            ok = false;
        }
    }

    llama_model_tensor_buft_override gpu_override[] = {
        { layer_zero_experts, ggml_backend_dev_buffer_type(cache_dev) },
        { nullptr,            nullptr                                 }
    };
    {
        auto model =
            expect_load("selected explicit GPU", moe_metadata.get(), make_params(both_layers, 1, gpu_override), true);
        if (model && (layer_cached(*model, 0) || !layer_uses_buft(*model, 0, ggml_backend_dev_buffer_type(cache_dev)) ||
                      !layer_cached(*model, 1))) {
            fprintf(stderr, "test-moe-cache-selector: explicit GPU placement mismatch\n");
            ok = false;
        }
    }

    static const char * layer_one_experts = "blk\\.1\\.ffn_(up|down|gate|gate_up)_(ch|)exps";
    static const char * both_layer_experts = "blk\\.[01]\\.ffn_(up|down|gate|gate_up)_(ch|)exps";
    llama_model_tensor_buft_override gpu_layer_one[] = {
        { layer_one_experts, ggml_backend_dev_buffer_type(cache_dev) },
        { nullptr,           nullptr                                 }
    };
    llama_model_tensor_buft_override gpu_both_layers[] = {
        { both_layer_experts, ggml_backend_dev_buffer_type(cache_dev) },
        { nullptr,            nullptr                                 }
    };
    llama_model_tensor_buft_override cpu_both_layers[] = {
        { both_layer_experts, ggml_backend_cpu_buffer_type() },
        { nullptr,            nullptr                        }
    };

    auto execution_vocab_model =
        expect_load("execution vocabulary", moe_metadata.get(), make_params(both_layers, 1, nullptr), true);
    if (!execution_vocab_model) {
        return 1;
    }
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(execution_vocab_model.get()));
    const auto execution_tokens = get_tokens(4, n_vocab, seed + 41);
    const auto run_execution = [&](const char * label, llama_model_params params,
                                   std::array<bool, 2> expected_cached) {
        auto model = expect_load(label, moe_metadata.get(), params, true);
        if (!model) {
            return std::vector<float>();
        }
        for (int layer = 0; layer < 2; ++layer) {
            if (layer_cached(*model, layer) != expected_cached[layer]) {
                fprintf(stderr, "test-moe-cache-selector: %s layer %d runtime placement mismatch\n", label, layer);
                ok = false;
            }
        }
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx                = 32;
        cparams.n_batch              = 8;
        cparams.n_ubatch             = 8;
        cparams.n_threads            = 4;
        cparams.n_threads_batch      = 4;
        llama_context_ptr context(llama_init_from_model(model.get(), cparams));
        if (!context) {
            fprintf(stderr, "test-moe-cache-selector: %s context creation failed\n", label);
            ok = false;
            return std::vector<float>();
        }
        auto logits = get_logits(model.get(), context.get(), execution_tokens);
        llama_synchronize(context.get());
        if (logits.empty() || !std::all_of(logits.begin(), logits.end(), [](float value) {
                return std::isfinite(value);
            })) {
            fprintf(stderr, "test-moe-cache-selector: %s produced invalid output\n", label);
            ok = false;
        }
        return logits;
    };

    const auto ordinary_device = run_execution(
        "execute entirely ordinary device", make_params(both_layers, 1, gpu_both_layers), { false, false });
    const auto ordinary_host = run_execution(
        "execute ordinary host control", make_params(both_layers, 1, cpu_both_layers), { false, false });
    const double control_nmse = ordinary_device.empty() || ordinary_host.empty() ? INFINITY :
        nmse(ordinary_device, ordinary_host);
    const double mixed_tolerance = std::max(1e-6, 4.0 * control_nmse);
    if (!std::isfinite(control_nmse) || control_nmse > 1e-4) {
        fprintf(stderr, "test-moe-cache-selector: ordinary control NMSE %.9g exceeds qualification bound\n",
                control_nmse);
        ok = false;
    }
    const auto check_execution = [&](const char * label, const std::vector<float> & actual) {
        const double error = ordinary_device.empty() || actual.empty() ? INFINITY : nmse(ordinary_device, actual);
        fprintf(stderr, "test-moe-cache-selector: %-28s nmse=%.9g tolerance=%.9g\n", label, error,
                mixed_tolerance);
        if (!std::isfinite(error) || error > mixed_tolerance) {
            ok = false;
        }
    };
    check_execution("ordinary/cache", run_execution(
        "execute ordinary/cache", make_params(both_layers, 1, gpu_override), { false, true }));
    check_execution("cache/ordinary", run_execution(
        "execute cache/ordinary", make_params(both_layers, 1, gpu_layer_one), { true, false }));
    check_execution("all cache", run_execution(
        "execute all cache", make_params(both_layers, 1, nullptr), { true, true }));
    check_execution("full-slot cache", run_execution(
        "execute full-slot cache", make_params(both_layers, 1, nullptr, 8), { true, true }));

    {
        gguf_context_ptr mtp_metadata = get_gguf_ctx(LLM_ARCH_QWEN35MOE, true, true, 8, 2, 2);
        const llama_model_layer_range routed_layers[] = {
            { 1, 2 },
        };
        static const char * target_experts = "blk\\.1\\.ffn_(up|down|gate|gate_up)_(ch|)exps";
        static const char * mtp_experts    = "blk\\.2\\.ffn_(up|down|gate|gate_up)_(ch|)exps";
        static const char * routed_experts = "blk\\.[12]\\.ffn_(up|down|gate|gate_up)_(ch|)exps";
        llama_model_tensor_buft_override target_ordinary[] = {
            { target_experts, ggml_backend_dev_buffer_type(cache_dev) },
            { nullptr,        nullptr                                 },
        };
        llama_model_tensor_buft_override mtp_ordinary[] = {
            { mtp_experts, ggml_backend_dev_buffer_type(cache_dev) },
            { nullptr,     nullptr                                 },
        };
        llama_model_tensor_buft_override all_ordinary[] = {
            { routed_experts, ggml_backend_dev_buffer_type(cache_dev) },
            { nullptr,        nullptr                                 },
        };

        auto make_mtp_params = [&](const llama_model_tensor_buft_override * overrides, int32_t slots = 4) {
            auto params              = make_params(routed_layers, 1, overrides, slots);
            params.load_mtp          = true;
            return params;
        };
        struct target_mtp_output {
            std::vector<float> target;
            std::vector<float> mtp;
        };
        const auto routed_layer_cached = [&](const llama_model & model, int layer) {
            const auto & block = model.layers.at(layer);
            const ggml_tensor * banks[] = {
                block.ffn_gate_exps, block.ffn_up_exps, block.ffn_down_exps, block.ffn_gate_up_exps,
            };
            bool saw_bank = false;
            for (const auto * bank : banks) {
                if (bank != nullptr) {
                    saw_bank = true;
                    if (!is_cached(bank)) {
                        return false;
                    }
                }
            }
            return saw_bank;
        };
        const auto run_target_mtp = [&](const char * label, llama_model_params params,
                                        std::array<bool, 2> expected_cached) {
            target_mtp_output output;
            auto model = expect_load(label, mtp_metadata.get(), params, true);
            if (!model) {
                return output;
            }
            if (routed_layer_cached(*model, 1) != expected_cached[0] ||
                routed_layer_cached(*model, 2) != expected_cached[1]) {
                fprintf(stderr,
                        "test-moe-cache-selector: %s target/MTP placement mismatch actual=%d,%d expected=%d,%d\n",
                        label, routed_layer_cached(*model, 1), routed_layer_cached(*model, 2),
                        expected_cached[0], expected_cached[1]);
                print_layer_bufts(*model, 1);
                print_layer_bufts(*model, 2);
                ok = false;
            }
            llama_context_ptr target = make_phase_workspace_context(model.get(), LLAMA_CONTEXT_TYPE_DEFAULT);
            llama_context_ptr mtp = make_phase_workspace_context(
                model.get(), LLAMA_CONTEXT_TYPE_MTP, target.get());
            if (!target || !mtp) {
                fprintf(stderr, "test-moe-cache-selector: %s target/MTP context creation failed\n", label);
                ok = false;
                return output;
            }
            const uint32_t vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
            output.target = get_logits(model.get(), target.get(), get_tokens(4, vocab, seed + 51));
            llama_batch mtp_batch = make_mtp_batch(1, llama_model_n_embd_out(model.get()), 0, vocab, seed + 52);
            if (llama_decode(mtp.get(), mtp_batch) != 0) {
                fprintf(stderr, "test-moe-cache-selector: %s MTP decode failed\n", label);
                ok = false;
            } else {
                llama_synchronize(mtp.get());
                const float * logits = llama_get_logits_ith(mtp.get(), 0);
                if (logits != nullptr) {
                    output.mtp.assign(logits, logits + vocab);
                }
            }
            llama_batch_free(mtp_batch);
            if (output.target.empty() || output.mtp.empty() ||
                !std::all_of(output.target.begin(), output.target.end(), [](float value) { return std::isfinite(value); }) ||
                !std::all_of(output.mtp.begin(), output.mtp.end(), [](float value) { return std::isfinite(value); })) {
                fprintf(stderr, "test-moe-cache-selector: %s target/MTP output invalid\n", label);
                ok = false;
            }
            return output;
        };

        const auto ordinary = run_target_mtp(
            "target+MTP all ordinary", make_mtp_params(all_ordinary), { false, false });
        const auto check_target_mtp = [&](const char * label, const target_mtp_output & actual) {
            const double target_error = ordinary.target.empty() || actual.target.empty() ? INFINITY :
                nmse(ordinary.target, actual.target);
            const double mtp_error = ordinary.mtp.empty() || actual.mtp.empty() ? INFINITY :
                nmse(ordinary.mtp, actual.mtp);
            fprintf(stderr, "test-moe-cache-selector: %-28s target_nmse=%.9g mtp_nmse=%.9g\n",
                    label, target_error, mtp_error);
            if (!std::isfinite(target_error) || !std::isfinite(mtp_error) ||
                target_error > 1e-6 || mtp_error > 1e-6) {
                ok = false;
            }
        };
        check_target_mtp("target ordinary/MTP cache", run_target_mtp(
            "target ordinary/MTP cache", make_mtp_params(target_ordinary), { false, true }));
        check_target_mtp("target cache/MTP ordinary", run_target_mtp(
            "target cache/MTP ordinary", make_mtp_params(mtp_ordinary), { true, false }));
        check_target_mtp("target+MTP all cache", run_target_mtp(
            "target+MTP all cache", make_mtp_params(nullptr), { true, true }));
        check_target_mtp("target+MTP full-slot", run_target_mtp(
            "target+MTP full-slot", make_mtp_params(nullptr, 8), { true, true }));
    }

    {
        auto model =
            expect_load("zero active cache groups", moe_metadata.get(), make_params(layer_zero, 1, cpu_override), true);
        ggml_backend_buffer_type_t device_buft = ggml_backend_dev_buffer_type(cache_dev);
        if (model && (layer_cached(*model, 0) || !layer_is_host(*model, 0) || layer_cached(*model, 1) ||
                      !layer_uses_buft(*model, 1, device_buft) || !model->moe_expert_cache_memory().empty())) {
            fprintf(stderr, "test-moe-cache-selector: zero-active placement mismatch\n");
            print_layer_bufts(*model, 0);
            print_layer_bufts(*model, 1);
            ok = false;
        }
    }

    llama_model_tensor_buft_override partial_override[] = {
        { layer_zero_up, ggml_backend_cpu_buffer_type() },
        { nullptr,       nullptr                        }
    };
    (void) expect_load("partial selected group", moe_metadata.get(), make_params(layer_zero, 1, partial_override),
                       false);
    (void) expect_load("nonexistent selected layer", moe_metadata.get(), make_params(missing_layer, 1, nullptr), false);
    (void) expect_load("dense-only selected layer", dense_metadata.get(), make_params(layer_zero, 1, nullptr), false);

    {
        auto model =
            expect_load("selector-free legacy", moe_metadata.get(), make_params(nullptr, 0, cpu_override), true);
        if (model && (!layer_cached(*model, 0) || !layer_cached(*model, 1))) {
            fprintf(stderr, "test-moe-cache-selector: legacy precedence changed\n");
            ok = false;
        }
    }

    fprintf(stderr, "test-moe-cache-selector: %s on %s\n", ok ? "PASS" : "FAIL",
            ggml_backend_dev_description(cache_dev));
    return ok ? 0 : 1;
}

static int test_moe_placement_report(const size_t seed) {
    ggml_backend_dev_t cache_dev = nullptr;
    std::vector<ggml_backend_dev_t> cache_devs;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto get_cache_buft = reg != nullptr ? reinterpret_cast<ggml_backend_moe_cache_buffer_type_t>(
            ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_CACHE_BUFFER_TYPE_PROC_NAME)) : nullptr;
        if (get_cache_buft != nullptr) {
            cache_devs.push_back(dev);
            if (cache_dev == nullptr) {
                cache_dev = dev;
            }
        }
    }
    if (cache_dev == nullptr) {
        fprintf(stderr, "test-moe-placement: SKIP (no backend with MoE cache support)\n");
        return 77;
    }

    bool ok = true;
    const auto check = [&](bool condition, const char * message) {
        if (!condition) {
            fprintf(stderr, "test-moe-placement: %s\n", message);
            ok = false;
        }
    };
    check(llama_moe_placement_context_mask(false, 1, -1, 2, 0) == 1 &&
              llama_moe_placement_context_mask(true, 1, 0, 2, 0) == 1,
          "ordinary/router context-use classification mismatch");
    check(llama_moe_placement_context_mask(true, 1, -1, 2, 0) == 1 &&
              llama_moe_placement_context_mask(true, 1, -1, 2, 2) == 2,
          "disjoint target/MTP context-use classification mismatch");
    check(llama_moe_placement_context_mask(true, 1, -1, 0, 0) == 3,
          "shared MTP-only context-use classification mismatch");
    ggml_backend_dev_t devices[] = { cache_dev, nullptr };
    const llama_model_layer_range both_layers[] = {{0, 1}};
    const llama_model_layer_range split_layers[] = {{0, 0}, {1, 1}};
    const llama_model_layer_range layer_zero[] = {{0, 0}};
    const llama_model_layer_range layer_one[] = {{1, 1}};
    auto make_params = [&](const llama_model_layer_range * ranges, size_t n_ranges,
                           const llama_model_tensor_buft_override * overrides, bool mtp = false,
                           size_t host_pin = 0, int32_t slots = 2) {
        llama_model_params params = llama_model_default_params();
        params.devices = devices;
        params.n_gpu_layers = 99;
        params.split_mode = LLAMA_SPLIT_MODE_LAYER;
        params.load_mode = LLAMA_LOAD_MODE_NONE;
        params.progress_callback = silent_model_load_progress;
        params.moe_expert_cache_slots = slots;
        params.moe_expert_cache_layer_ranges = ranges;
        params.n_moe_expert_cache_layer_ranges = n_ranges;
        params.tensor_buft_overrides = overrides;
        params.load_mtp = mtp;
        params.moe_expert_cache_host_pinned_size = host_pin;
        return params;
    };
    const auto load = [&](gguf_context * metadata, llama_model_params params, size_t tensor_seed) {
        return llama_model_ptr(llama_model_init_from_user(metadata, set_tensor_data, &tensor_seed, params));
    };
    const auto group_at = [&](const llama_moe_placement_report & report, int32_t layer) {
        return std::find_if(report.groups.begin(), report.groups.end(), [&](const auto & group) {
            return group.layer == layer && group.domain == GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY;
        });
    };
    const auto check_hashes = [&](const llama_moe_placement_report & report) {
        check(report.model_identity.size() == 64 && report.placement_id.size() == 64,
              "identity digest length mismatch");
        check(report.placement_identity_kind == "physical" ||
                  report.placement_identity_kind == "runtime_unverified",
              "placement identity stability is not explicit");
        check(report.direct_io_state == "none" || report.direct_io_state == "all" ||
                  report.direct_io_state == "mixed",
              "resolved DirectIO state is invalid");
        check(hash_sha256_hex(report.model_identity_record.data(), report.model_identity_record.size()) ==
                  report.model_identity,
              "model identity does not match retained record");
        check(hash_sha256_hex(report.placement_record.data(), report.placement_record.size()) ==
                  report.placement_id,
              "placement identity does not match retained record");
        check(report.model_identity_record.find("/home/") == std::string::npos &&
                  report.placement_record.find("0x") == std::string::npos,
              "identity contains a path or pointer-like value");
        try {
            (void) common_json::parse(report.model_identity_record);
            (void) common_json::parse(report.placement_record);
        } catch (const std::exception &) {
            check(false, "identity record is not valid JSON");
        }
    };

    llama_moe_placement_report retained;
    {
        gguf_context_ptr metadata = get_gguf_ctx(LLM_ARCH_QWEN3MOE, true);
        auto model = load(metadata.get(), make_params(both_layers, 1, nullptr), seed);
        check(model != nullptr, "baseline model failed to load");
        if (model) {
            retained = llama_model_moe_placement(model.get());
            check_hashes(retained);
            check(retained.model_identity_kind == "layout_unverified", "virtual model identity is not explicit");
            check(retained.groups.size() == 2 && retained.owners.size() == 1,
                  "baseline inventory cardinality mismatch");
            size_t group_fixed = 0;
            size_t group_per_slot = 0;
            for (const auto & group : retained.groups) {
                check(group.mode == LLAMA_MOE_PLACEMENT_RESIDUAL_CACHE && group.context_use_mask == 1,
                      "baseline group placement mismatch");
                for (const auto & bank : group.banks) {
                    if (bank.status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE) {
                        check(bank.reason == LLAMA_MOE_PLACEMENT_CACHE_SELECTOR &&
                                  bank.winning_override_index >= 0 && !bank.winning_override_pattern.empty() &&
                                  bank.requested_buft == bank.selected_buft && bank.selected_buft == bank.resolved_buft,
                              "cache-selector provenance mismatch");
                    }
                }
                group_fixed += group.cache_fixed_bytes;
                group_per_slot += group.cache_per_slot_bytes;
            }
            check(retained.owners[0].active_groups == 2 && retained.owners[0].active_context_mask == 1 &&
                      retained.owners[0].active_cache_groups == 2 &&
                      retained.owners[0].active_cache_context_mask == 1 &&
                      retained.owners[0].cache_group_fixed_bytes == group_fixed &&
                      retained.owners[0].cache_group_per_slot_bytes == group_per_slot &&
                      retained.owners[0].cache_fixed_bytes == group_fixed + retained.owners[0].cache_context_fixed_bytes,
                  "baseline owner ledger mismatch");
            check(!retained.model_allocations.empty() &&
                      std::all_of(retained.model_allocations.begin(), retained.model_allocations.end(), [](const auto & allocation) {
                          return allocation.bytes_available && allocation.current_allocation && allocation.bytes > 0 &&
                                 allocation.provenance == "backend_packed_buffer_current";
                      }),
                  "current packed model-buffer ledger is incomplete");
            check(std::any_of(retained.model_allocations.begin(), retained.model_allocations.end(), [](const auto & allocation) {
                      return allocation.resolved_class == "moe_cache_source_host" &&
                             allocation.owner_backend == "CPU";
                  }),
                  "host-backed MoE source storage was attributed to CUDA device memory");
            const auto report_mparams = make_params(both_layers, 1, nullptr);
            const auto report_cparams = llama_context_default_params();
            const std::string machine_text =
                common_moe_placement_report_json(retained, report_mparams, report_cparams);
            const common_json machine = common_json::parse(machine_text);
            const std::string configuration_id = machine.at("configuration_id").get<std::string>();
            check(configuration_id.size() == 64 && machine.at("placement_id").get<std::string>() == retained.placement_id &&
                      machine.at("measurement_completeness").get<std::string>() == "incomplete" &&
                      machine.at("capacity_assessment").get<std::string>() == "unknown",
                  "tool JSON report identity or unknown-state contract mismatch");
            check(machine_text.find(retained.groups[0].banks[0].winning_override_pattern) == std::string::npos,
                  "tool JSON leaked a raw override pattern");
            check(common_moe_placement_report_human(retained, report_mparams, report_cparams).find(configuration_id) !=
                      std::string::npos,
                  "human and JSON report views disagree on configuration identity");
            float split_a[] = {1.0f};
            float split_b[] = {2.0f};
            auto equivalent_mparams_a = report_mparams;
            auto equivalent_mparams_b = report_mparams;
            equivalent_mparams_a.tensor_split = split_a;
            equivalent_mparams_b.tensor_split = split_b;
            equivalent_mparams_b.n_gpu_layers = 1;
            const auto equivalent_config_a = common_json::parse(
                common_moe_placement_report_json(retained, equivalent_mparams_a, report_cparams));
            const auto equivalent_config_b = common_json::parse(
                common_moe_placement_report_json(retained, equivalent_mparams_b, report_cparams));
            check(equivalent_config_a.at("configuration_id").get<std::string>() ==
                      equivalent_config_b.at("configuration_id").get<std::string>(),
                  "raw split ratio or GPU-layer spelling changed resolved configuration identity");
            common_params runtime_a;
            common_params runtime_b;
            runtime_b.moe_early_router = true;
            const auto runtime_config_a = common_json::parse(
                common_moe_placement_report_json(retained, report_mparams, report_cparams, &runtime_a));
            const auto runtime_config_b = common_json::parse(
                common_moe_placement_report_json(retained, report_mparams, report_cparams, &runtime_b));
            check(runtime_config_a.at("configuration_id").get<std::string>() !=
                      runtime_config_b.at("configuration_id").get<std::string>(),
                  "effective early-router control was omitted from configuration identity");
            common_params runtime_spec = runtime_a;
            runtime_spec.speculative.draft.p_min += 0.125f;
            runtime_spec.speculative.draft.p_split += 0.125f;
            runtime_spec.speculative.draft.n_ubatch += 1;
            runtime_spec.speculative.draft.kv_gpu_layers += 1;
            const auto runtime_config_spec = common_json::parse(
                common_moe_placement_report_json(retained, report_mparams, report_cparams, &runtime_spec));
            check(runtime_config_a.at("configuration_id").get<std::string>() !=
                      runtime_config_spec.at("configuration_id").get<std::string>(),
                  "effective speculative controls were omitted from configuration identity");
#if !defined(_WIN32)
            const char * old_allreduce = std::getenv("GGML_CUDA_ALLREDUCE");
            const char * old_compute = std::getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE");
            const std::string saved_allreduce = old_allreduce != nullptr ? old_allreduce : "";
            const std::string saved_compute = old_compute != nullptr ? old_compute : "";
            const bool had_allreduce = old_allreduce != nullptr;
            const bool had_compute = old_compute != nullptr;
            unsetenv("GGML_CUDA_ALLREDUCE");
            unsetenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE");
            const auto env_default = common_json::parse(
                common_moe_placement_report_json(retained, report_mparams, report_cparams, &runtime_a));
            setenv("GGML_CUDA_ALLREDUCE",
#if defined(__linux__)
                "nccl",
#else
                "internal",
#endif
                1);
            setenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE", "auto", 1);
            const auto env_explicit_default = common_json::parse(
                common_moe_placement_report_json(retained, report_mparams, report_cparams, &runtime_a));
            setenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE", "F32", 1);
            const auto env_f32 = common_json::parse(
                common_moe_placement_report_json(retained, report_mparams, report_cparams, &runtime_a));
            setenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE", "fp32", 1);
            const auto env_fp32 = common_json::parse(
                common_moe_placement_report_json(retained, report_mparams, report_cparams, &runtime_a));
            check(env_default.at("configuration_id").get<std::string>() ==
                      env_explicit_default.at("configuration_id").get<std::string>(),
                  "implicit and explicit platform defaults produced different configuration identities");
            check(env_f32.at("configuration_id").get<std::string>() ==
                      env_fp32.at("configuration_id").get<std::string>() &&
                      env_f32.at("configuration_id").get<std::string>() !=
                          env_default.at("configuration_id").get<std::string>(),
                  "cuBLAS compute-type aliases were not normalized");
            had_allreduce ? setenv("GGML_CUDA_ALLREDUCE", saved_allreduce.c_str(), 1) :
                unsetenv("GGML_CUDA_ALLREDUCE");
            had_compute ? setenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE", saved_compute.c_str(), 1) :
                unsetenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE");
#endif
        }
    }
    check(retained.groups.size() == 2 && !retained.groups[0].banks.empty() &&
              !retained.groups[0].banks[0].name.empty() && !retained.model_identity_record.empty() &&
              !retained.placement_record.empty(),
          "owned report did not survive model and metadata destruction");

    {
        gguf_context_ptr metadata = get_gguf_ctx(LLM_ARCH_QWEN3MOE, true);
        auto equivalent = load(metadata.get(), make_params(split_layers, 2, nullptr), seed + 1);
        auto reseeded = load(metadata.get(), make_params(both_layers, 1, nullptr), seed + 2);
        check(equivalent && reseeded, "equivalent placement reload failed");
        if (equivalent && reseeded) {
            const auto equivalent_report = llama_model_moe_placement(equivalent.get());
            const auto reseeded_report = llama_model_moe_placement(reseeded.get());
            check(equivalent_report.placement_record == retained.placement_record &&
                      equivalent_report.placement_id == retained.placement_id,
                  "equivalent selectors changed configuration identity");
            check(reseeded_report.model_identity == retained.model_identity,
                  "virtual tensor payload seed changed explicitly layout-only identity");
        }
    }
    {
        llama_model_tensor_buft_override nonmatching[] = {
            {"/home/private/token=this_pattern_matches_nothing", ggml_backend_cpu_buffer_type()},
            {nullptr, nullptr},
        };
        gguf_context_ptr metadata = get_gguf_ctx(LLM_ARCH_QWEN3MOE, true);
        auto model = load(metadata.get(), make_params(both_layers, 1, nonmatching), seed);
        check(model != nullptr, "nonmatching override identity control failed to load");
        if (model) {
            const auto report = llama_model_moe_placement(model.get());
            check(report.placement_record == retained.placement_record && report.placement_id == retained.placement_id,
                  "raw nonmatching override changed resolved placement identity");
            check(report.placement_record.find("/home/private") == std::string::npos &&
                      report.placement_record.find("token=") == std::string::npos,
                  "raw override text leaked into placement identity");
        }
    }
    {
        gguf_context_ptr metadata = get_gguf_ctx(LLM_ARCH_QWEN3MOE, true);
        auto params = make_params(both_layers, 1, nullptr);
        params.no_alloc = true;
        auto model = load(metadata.get(), params, seed);
        check(model != nullptr, "allocation-free placement probe failed to load");
        if (model) {
            const auto report = llama_model_moe_placement(model.get());
            check(report.placement_record == retained.placement_record && report.placement_id == retained.placement_id,
                  "allocation-free probe changed resolved placement identity");
            for (const auto & group : report.groups) {
                for (const auto & bank : group.banks) {
                    check(!bank.actual_buft_available,
                          "allocation-free probe reported an actual runtime buffer");
                }
            }
            check(!report.model_allocations.empty() &&
                      std::all_of(report.model_allocations.begin(), report.model_allocations.end(), [](const auto & allocation) {
                          return allocation.bytes_available && !allocation.current_allocation && allocation.bytes > 0 &&
                                 allocation.provenance == "backend_packed_context_size_estimate";
                      }),
                  "allocation-free probe omitted its packed whole-buffer estimate");
        }
    }
    {
        gguf_context_ptr metadata = get_gguf_ctx(LLM_ARCH_QWEN4EXP, true);
        llama_model_params source_params = llama_model_default_params();
        source_params.progress_callback = silent_model_load_progress;
        auto source = load(metadata.get(), source_params, seed);
        const auto path = std::filesystem::temp_directory_path() /
            ("llama-moe-placement-lazy-" + std::to_string(reinterpret_cast<uintptr_t>(metadata.get())) + ".gguf");
        check(source != nullptr, "lazy strategy source failed to load");
        if (source) {
            llama_model_save_to_file(source.get(), path.string().c_str());
        }
        auto params = make_params(nullptr, 0, nullptr);
        params.lazy_mode = LLAMA_LAZY_MODE_ON;
        llama_model_ptr real(source ? llama_model_load_from_file(path.string().c_str(), params) : nullptr);
        auto probe_params = params;
        probe_params.no_alloc = true;
        llama_model_ptr probe(source ? llama_model_load_from_file(path.string().c_str(), probe_params) : nullptr);
        std::error_code remove_error;
        std::filesystem::remove(path, remove_error);
        check(real != nullptr && probe != nullptr, "lazy strategy identity fixture failed to load");
        if (real && probe) {
            const auto real_report = llama_model_moe_placement(real.get());
            const auto probe_report = llama_model_moe_placement(probe.get());
            check(real_report.has_lazy_tensors && probe_report.has_lazy_tensors &&
                      real_report.placement_id == probe_report.placement_id,
                  "real and allocation-free lazy strategy identities diverged");
        }
    }
    {
        gguf_context_ptr metadata = get_gguf_ctx(LLM_ARCH_QWEN3MOE, true);
        llama_model_params source_params = llama_model_default_params();
        source_params.progress_callback = silent_model_load_progress;
        auto source = load(metadata.get(), source_params, seed);
        const auto path = std::filesystem::temp_directory_path() /
            ("llama-moe-placement-mlock-" + std::to_string(reinterpret_cast<uintptr_t>(metadata.get())) + ".gguf");
        check(source != nullptr, "mlock strategy source failed to load");
        if (source) {
            llama_model_save_to_file(source.get(), path.string().c_str());
        }
        auto unlocked_params = make_params(both_layers, 1, nullptr);
        auto locked_params = unlocked_params;
        locked_params.load_mode = LLAMA_LOAD_MODE_MLOCK;
        llama_model_ptr unlocked(source ? llama_model_load_from_file(path.string().c_str(), unlocked_params) : nullptr);
        llama_model_ptr locked(source ? llama_model_load_from_file(path.string().c_str(), locked_params) : nullptr);
        std::error_code remove_error;
        std::filesystem::remove(path, remove_error);
        check(unlocked != nullptr && locked != nullptr, "mlock strategy identity fixture failed to load");
        if (unlocked && locked) {
            const auto unlocked_report = llama_model_moe_placement(unlocked.get());
            const auto locked_report = llama_model_moe_placement(locked.get());
            check(!unlocked_report.uses_mlock && locked_report.uses_mlock &&
                      locked_report.placement_id != unlocked_report.placement_id,
                  "mlock strategy was omitted from resolved placement identity");
        }
    }
    llama_moe_placement_report default_ordinary;
    {
        gguf_context_ptr metadata = get_gguf_ctx(LLM_ARCH_QWEN3MOE, true, false, 2, 1);
        auto model = load(metadata.get(), make_params(both_layers, 1, nullptr), seed);
        check(model != nullptr, "top-k identity control failed to load");
        if (model) {
            check(llama_model_moe_placement(model.get()).model_identity != retained.model_identity,
                  "top-k change did not change weak model identity");
        }
    }
    {
        gguf_context_ptr metadata = get_gguf_ctx(LLM_ARCH_QWEN3MOE, true, false, 2, 2, -1, 160);
        auto model = load(metadata.get(), make_params(both_layers, 1, nullptr), seed);
        check(model != nullptr, "non-MoE layout identity control failed to load");
        if (model) {
            check(llama_model_moe_placement(model.get()).model_identity != retained.model_identity,
                  "non-MoE tensor-layout change did not change weak model identity");
        }
    }
    {
        gguf_context_ptr metadata = get_gguf_ctx(LLM_ARCH_QWEN3MOE, true);
        auto model = load(metadata.get(), make_params(both_layers, 1, nullptr, false, 0, 3), seed);
        check(model != nullptr, "slot-count identity control failed to load");
        if (model) {
            check(llama_model_moe_placement(model.get()).placement_id != retained.placement_id,
                  "slot-count change did not change configuration identity");
        }
    }
    struct grouped_numpunct : std::numpunct<char> {
        char do_thousands_sep() const override { return '_'; }
        std::string do_grouping() const override { return "\3"; }
    };
    {
        const std::locale old_locale = std::locale();
        try {
            std::locale::global(std::locale(old_locale, new grouped_numpunct));
            gguf_context_ptr metadata = get_gguf_ctx(LLM_ARCH_QWEN3MOE, true);
            auto model = load(metadata.get(), make_params(both_layers, 1, nullptr), seed);
            check(model != nullptr, "locale identity control failed to load");
            if (model) {
                check(llama_model_moe_placement(model.get()).placement_record == retained.placement_record,
                      "canonical serialization depends on the process locale");
            }
            std::locale::global(old_locale);
        } catch (...) {
            std::locale::global(old_locale);
            throw;
        }
    }

    {
        gguf_context_ptr metadata = get_gguf_ctx(LLM_ARCH_QWEN3MOE, true);
        auto model = load(metadata.get(), make_params(layer_one, 1, nullptr), seed);
        check(model != nullptr, "default placement provenance failed to load");
        if (model) {
            default_ordinary = llama_model_moe_placement(model.get());
            const auto group0 = group_at(default_ordinary, 0);
            check(group0 != default_ordinary.groups.end() && group0->mode == LLAMA_MOE_PLACEMENT_ORDINARY_DEVICE,
                  "default ordinary placement summary mismatch");
            if (group0 != default_ordinary.groups.end()) {
                for (const auto & bank : group0->banks) {
                    if (bank.status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE) {
                        check(bank.reason == LLAMA_MOE_PLACEMENT_DEFAULT && bank.winning_override_index == -1 &&
                                  bank.winning_override_pattern.empty() && bank.requested_buft.empty() &&
                                  bank.selected_buft == bank.resolved_buft && bank.resolved_buft == bank.actual_buft,
                              "default bank provenance mismatch");
                    }
                }
            }
        }
    }

    {
        llama_model_tensor_buft_override same_device_override[] = {
            {"(?!/home/private/token=)blk\\.0\\.ffn_gate_exps", ggml_backend_dev_buffer_type(cache_dev)},
            {nullptr, nullptr},
        };
        gguf_context_ptr metadata = get_gguf_ctx(LLM_ARCH_QWEN3MOE, true);
        auto model = load(metadata.get(), make_params(layer_one, 1, same_device_override), seed);
        check(model != nullptr, "same-placement mixed-provenance control failed to load");
        if (model) {
            const auto report = llama_model_moe_placement(model.get());
            const auto group0 = group_at(report, 0);
            check(group0 != report.groups.end() && group0->mode == LLAMA_MOE_PLACEMENT_ORDINARY_DEVICE &&
                      group0->placement_reason == "mixed",
                  "provenance-only difference was misreported as mixed placement");
            check(report.placement_id == default_ordinary.placement_id,
                  "provenance-only difference changed resolved placement identity");
            check(report.placement_record.find("/home/private") == std::string::npos &&
                      report.placement_record.find("token=") == std::string::npos,
                  "winning override text leaked into placement identity");
            const auto cparams = llama_context_default_params();
            const std::string default_json = common_moe_placement_report_json(
                default_ordinary, make_params(layer_one, 1, nullptr), cparams);
            const std::string override_json = common_moe_placement_report_json(
                report, make_params(layer_one, 1, same_device_override), cparams);
            check(common_json::parse(default_json).at("configuration_id").get<std::string>() !=
                      common_json::parse(override_json).at("configuration_id").get<std::string>() &&
                      override_json.find("/home/private") == std::string::npos &&
                      override_json.find("token=") == std::string::npos,
                  "winning override provenance was omitted or leaked raw pattern text");
        }
    }

    const char * layer_zero_all = "blk\\.0\\.ffn_(gate|up|down)_exps";
    llama_model_tensor_buft_override cpu_layer_zero[] = {
        {layer_zero_all, ggml_backend_cpu_buffer_type()},
        {nullptr, nullptr},
    };
    llama_moe_placement_report cpu_ordinary;
    {
        gguf_context_ptr metadata = get_gguf_ctx(LLM_ARCH_QWEN3MOE, true);
        auto model = load(metadata.get(), make_params(layer_one, 1, cpu_layer_zero), seed);
        check(model != nullptr, "CPU override provenance failed to load");
        if (model) {
            cpu_ordinary = llama_model_moe_placement(model.get());
            const auto group0 = group_at(cpu_ordinary, 0);
            check(cpu_ordinary.model_identity == retained.model_identity,
                  "resolved placement changed weak model identity");
            check(group0 != cpu_ordinary.groups.end() && group0->mode == LLAMA_MOE_PLACEMENT_ORDINARY_CPU,
                  "CPU override placement summary mismatch");
            if (group0 != cpu_ordinary.groups.end()) {
                for (const auto & bank : group0->banks) {
                    if (bank.status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE) {
                        check(bank.reason == LLAMA_MOE_PLACEMENT_USER_OVERRIDE &&
                                  bank.winning_override_index == 0 &&
                                  bank.winning_override_pattern == cpu_layer_zero[0].pattern &&
                                  bank.requested_buft == ggml_backend_buft_name(ggml_backend_cpu_buffer_type()) &&
                                  bank.selected_buft == bank.resolved_buft &&
                                  bank.resolved_buft == bank.actual_buft,
                              "CPU override bank provenance mismatch");
                    }
                }
            }
        }
    }
    {
        gguf_context_ptr metadata = get_gguf_ctx(LLM_ARCH_QWEN3MOE, true);
        auto params = make_params(layer_one, 1, cpu_layer_zero);
        params.no_alloc = true;
        auto model = load(metadata.get(), params, seed);
        check(model != nullptr, "allocation-free CPU override probe failed to load");
        if (model) {
            const auto report = llama_model_moe_placement(model.get());
            const auto group0 = group_at(report, 0);
            check(report.placement_id == cpu_ordinary.placement_id &&
                      group0 != report.groups.end() && group0->mode == LLAMA_MOE_PLACEMENT_ORDINARY_CPU,
                  "allocation-free CPU override placement mismatch");
        }
    }
    {
        gguf_context_ptr metadata = get_gguf_ctx(LLM_ARCH_QWEN3MOE, true);
        auto model = load(metadata.get(), make_params(layer_zero, 1, cpu_layer_zero), seed);
        check(model != nullptr, "zero-active owner placement failed to load");
        if (model) {
            const auto report = llama_model_moe_placement(model.get());
            check(report.owners.size() == 1 && report.owners[0].active_cache_groups == 0 &&
                      report.owners[0].active_cache_context_mask == 0 && report.owners[0].cache_fixed_bytes == 0 &&
                      report.owners[0].cache_per_slot_bytes == 0,
                  "zero-active owner was omitted or charged cache resources");
        }
    }
    {
        gguf_context_ptr metadata = get_gguf_ctx(LLM_ARCH_QWEN3MOE, true);
        llama_model_params source_params = llama_model_default_params();
        source_params.progress_callback = silent_model_load_progress;
        auto source = load(metadata.get(), source_params, seed);
        const auto path = std::filesystem::temp_directory_path() /
            ("llama-moe-placement-mmap-" + std::to_string(reinterpret_cast<uintptr_t>(metadata.get())) + ".gguf");
        check(source != nullptr, "mmap provenance source failed to load");
        if (source) {
            llama_model_save_to_file(source.get(), path.string().c_str());
        }
        auto params = make_params(layer_one, 1, cpu_layer_zero);
        params.load_mode = LLAMA_LOAD_MODE_MMAP;
        llama_model_ptr model(source ? llama_model_load_from_file(path.string().c_str(), params) : nullptr);
        auto probe_params = params;
        probe_params.no_alloc = true;
        probe_params.load_mode = LLAMA_LOAD_MODE_AUTO;
        llama_model_ptr probe(source ? llama_model_load_from_file(path.string().c_str(), probe_params) : nullptr);
        std::error_code remove_error;
        std::filesystem::remove(path, remove_error);
        check(model != nullptr, "mmap CPU normalization failed to load");
        check(probe != nullptr, "mmap allocation-free placement probe failed to load");
        if (model) {
            const auto report = llama_model_moe_placement(model.get());
            const auto group0 = group_at(report, 0);
            if (group0 != report.groups.end()) {
                for (const auto & bank : group0->banks) {
                    if (bank.status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE) {
                        check(bank.reason == LLAMA_MOE_PLACEMENT_USER_OVERRIDE &&
                                  bank.requested_buft == ggml_backend_buft_name(ggml_backend_cpu_buffer_type()) &&
                                  bank.selected_buft != bank.resolved_buft &&
                                  bank.resolved_buft == ggml_backend_buft_name(ggml_backend_cpu_buffer_type()) &&
                                  bank.mode == LLAMA_MOE_PLACEMENT_ORDINARY_CPU && !bank.actual_buft.empty(),
                              "mmap CPU requested/selected/resolved provenance mismatch");
                    }
                }
            }
        }
        if (probe) {
            const auto probe_report = llama_model_moe_placement(probe.get());
            check(!probe_report.model_allocations.empty() &&
                      std::all_of(probe_report.model_allocations.begin(), probe_report.model_allocations.end(),
                          [](const auto & allocation) {
                              return allocation.bytes_available && !allocation.current_allocation && allocation.bytes > 0;
                          }),
                  "mmap allocation-free probe did not use packed-buffer estimates");
            if (model) {
                check(probe_report.placement_id == llama_model_moe_placement(model.get()).placement_id,
                      "mmap allocation-free probe changed normalized placement identity");
            }
        }
    }
    {
        gguf_context_ptr metadata = get_gguf_ctx(LLM_ARCH_QWEN3MOE, true);
        auto model = load(metadata.get(), make_params(nullptr, 0, cpu_layer_zero), seed);
        check(model != nullptr, "legacy cache provenance failed to load");
        if (model) {
            const auto report = llama_model_moe_placement(model.get());
            for (const auto & group : report.groups) {
                for (const auto & bank : group.banks) {
                    if (bank.status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE) {
                        check(bank.reason == LLAMA_MOE_PLACEMENT_CACHE_LEGACY &&
                                  bank.winning_override_index == 0 && !bank.winning_override_pattern.empty(),
                              "legacy cache provenance mismatch");
                    }
                }
            }
        }
    }

    const char * layer_zero_gate = "blk\\.0\\.ffn_gate_exps";
    const char * layer_zero_up_down = "blk\\.0\\.ffn_(up|down)_exps";
    llama_model_tensor_buft_override mixed_overrides[] = {
        {layer_zero_gate, ggml_backend_cpu_buffer_type()},
        {layer_zero_up_down, ggml_backend_dev_buffer_type(cache_dev)},
        {nullptr, nullptr},
    };
    {
        gguf_context_ptr metadata = get_gguf_ctx(LLM_ARCH_QWEN3MOE, true);
        auto model = load(metadata.get(), make_params(layer_one, 1, mixed_overrides), seed);
        check(model != nullptr, "mixed ordinary placement failed to load");
        if (model) {
            const auto report = llama_model_moe_placement(model.get());
            const auto group0 = group_at(report, 0);
            const auto group1 = group_at(report, 1);
            check(group0 != report.groups.end() && group0->mode == LLAMA_MOE_PLACEMENT_MIXED &&
                      group0->owner_index == -1 && group0->cache_fixed_bytes == 0 &&
                      group1 != report.groups.end() && group1->mode == LLAMA_MOE_PLACEMENT_RESIDUAL_CACHE,
                  "mixed group summary is misleading");
            if (group0 != report.groups.end()) {
                bool saw_host = false;
                bool saw_device = false;
                for (const auto & bank : group0->banks) {
                    if (bank.status != GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE) {
                        continue;
                    }
                    check(bank.reason == LLAMA_MOE_PLACEMENT_USER_OVERRIDE && bank.winning_override_index >= 0 &&
                              !bank.winning_override_pattern.empty() && !bank.requested_buft.empty() &&
                              !bank.selected_buft.empty() && bank.resolved_buft == bank.actual_buft,
                          "mixed bank provenance mismatch");
                    check(bank.allocation_estimate_available && bank.allocation_estimate > 0 &&
                              bank.allocation_provenance == "buffer_type_alloc_size_estimate",
                          "mixed bank allocation provenance mismatch");
                    saw_host = saw_host || bank.mode == LLAMA_MOE_PLACEMENT_ORDINARY_CPU;
                    saw_device = saw_device || bank.mode == LLAMA_MOE_PLACEMENT_ORDINARY_DEVICE;
                }
                check(saw_host && saw_device, "mixed fixture did not preserve both ordinary placements");
            }
        }
    }

    {
        const char * layer_zero_experts = "blk\\.0\\.ffn_(up|down|gate|gate_up)_(ch|)exps";
        llama_model_tensor_buft_override cpu_override[] = {
            {layer_zero_experts, ggml_backend_cpu_buffer_type()},
            {nullptr, nullptr},
        };
        const llama_model_layer_range layer_zero[] = {{0, 0}};
        gguf_context_ptr metadata = get_gguf_ctx(LLM_ARCH_QWEN3MOE, true);
        auto model = load(metadata.get(), make_params(layer_zero, 1, cpu_override), seed);
        check(model != nullptr, "zero-active-owner fixture failed to load");
        if (model) {
            const auto report = llama_model_moe_placement(model.get());
            check(report.owners.size() == 1 && report.owners[0].active_cache_groups == 0 &&
                      report.owners[0].active_cache_context_mask == 0 && report.owners[0].cache_fixed_bytes == 0 &&
                      report.owners[0].cache_per_slot_bytes == 0,
                  "zero-active owner was omitted or charged cache bytes");
        }
    }

    llama_model_tensor_buft_override overlapping[] = {
        {"blk\\.0\\.ffn_.*_exps", ggml_backend_cpu_buffer_type()},
        {layer_zero_gate, ggml_backend_dev_buffer_type(cache_dev)},
        {nullptr, nullptr},
    };
    {
        gguf_context_ptr metadata = get_gguf_ctx(LLM_ARCH_QWEN3MOE, true);
        auto model = load(metadata.get(), make_params(layer_one, 1, overlapping), seed);
        check(model != nullptr, "overlapping override fixture failed to load");
        if (model) {
            const auto report = llama_model_moe_placement(model.get());
            const auto group0 = group_at(report, 0);
            if (group0 != report.groups.end()) {
                for (const auto & bank : group0->banks) {
                    if (bank.status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE) {
                        check(bank.winning_override_index == 0 &&
                                  bank.winning_override_pattern == overlapping[0].pattern,
                              "first matching override was not retained");
                    }
                }
            }
        }
    }

    {
        gguf_context_ptr metadata = get_gguf_ctx(LLM_ARCH_QWEN3MOE, true);
        llama_model_params params = llama_model_default_params();
        ggml_backend_dev_t cpu_devices[] = {nullptr};
        params.devices = cpu_devices;
        params.progress_callback = silent_model_load_progress;
        auto model = load(metadata.get(), params, seed);
        check(model != nullptr, "CPU-only control failed to load");
        if (model) {
            const auto report = llama_model_moe_placement(model.get());
            check(report.owners.empty(), "CPU-only control exposed selected accelerator owners");
            for (const auto & group : report.groups) {
                check(group.cache_fixed_bytes == 0 && group.cache_per_slot_bytes == 0,
                      "CPU-only control exposed cache allocation");
            }
        }
    }

    {
        gguf_context_ptr metadata = get_gguf_ctx(LLM_ARCH_QWEN35MOE, true, true, 2, 2, 2);
        auto params = make_params(nullptr, 0, nullptr, true, 64u * 1024u * 1024u);
        auto model = load(metadata.get(), params, seed);
        check(model != nullptr, "disjoint MTP fixture failed to load");
        if (model) {
            const auto report = llama_model_moe_placement(model.get());
            bool saw_default = false;
            bool saw_mtp = false;
            bool saw_shared = false;
            size_t group_fixed = 0;
            size_t group_per_slot = 0;
            for (const auto & group : report.groups) {
                saw_default = saw_default || group.context_use_mask == 1;
                saw_mtp = saw_mtp || group.context_use_mask == 2;
                saw_shared = saw_shared || group.context_use_mask == 3;
                check(group.cache_fixed_bytes == group.cache_fixed_default_bytes + group.cache_fixed_mtp_bytes &&
                          group.cache_per_slot_bytes ==
                              group.cache_per_slot_default_bytes + group.cache_per_slot_mtp_bytes,
                      "MTP per-context group accounting mismatch");
                group_fixed += group.cache_fixed_bytes;
                group_per_slot += group.cache_per_slot_bytes;
            }
            check(report.owners.size() == 1 && report.owners[0].cache_group_fixed_bytes == group_fixed &&
                      report.owners[0].cache_group_per_slot_bytes == group_per_slot &&
                      report.owners[0].active_cache_context_mask == 3 &&
                      report.owners[0].mandatory_host_staging_available &&
                      report.owners[0].mandatory_host_staging_provenance == "backend_staging_size_v1_exact" &&
                      report.owners[0].mandatory_host_staging_default_bytes > 0 &&
                      report.owners[0].mandatory_host_staging_mtp_bytes > 0,
                  "MTP owner or staging ledger mismatch");
            if (report.owners.size() == 1) {
                check(report.mandatory_host_staging_bytes ==
                          report.owners[0].mandatory_host_staging_default_bytes +
                              report.owners[0].mandatory_host_staging_mtp_bytes,
                      "MTP report host-staging total mismatch");
            }
            check(saw_default && saw_mtp && !saw_shared, "disjoint MTP context masks missing");
        }
    }

    {
        gguf_context_ptr target_metadata = get_gguf_ctx(LLM_ARCH_QWEN35MOE, true);
        auto target = load(target_metadata.get(), make_params(nullptr, 0, nullptr), seed);
        check(target != nullptr, "shared-target fixture failed to load target");
        gguf_context_ptr full_draft_metadata = get_gguf_ctx(LLM_ARCH_QWEN35MOE, true, true);
        auto full_draft = load(full_draft_metadata.get(), make_params(nullptr, 0, nullptr, true), seed);
        check(full_draft != nullptr, "shared-target fixture failed to build full draft");
        const auto path = std::filesystem::temp_directory_path() /
            ("llama-moe-placement-shared-" +
             std::to_string(reinterpret_cast<uintptr_t>(full_draft_metadata.get())) + ".gguf");
        size_t omitted = 0;
        if (full_draft) {
            llama_model_saver saver(LLM_ARCH_QWEN35MOE, nullptr);
            gguf_set_kv(saver.gguf_ctx, full_draft_metadata.get());
            saver.add_kv(LLM_KV_NEXTN_SHARED_TARGET_TENSORS, true);
            for (const auto & [name, tensor] : full_draft->tensors_by_name) {
                if (name == "token_embd.weight" || name == "output.weight" || name == "output_norm.weight") {
                    ++omitted;
                    continue;
                }
                saver.add_tensor(tensor);
            }
            saver.save(path.string());
        }
        check(omitted == 3, "shared-target fixture did not identify the three borrowable tensors");
        auto draft_params = make_params(nullptr, 0, nullptr, true);
        draft_params.model_shared = target.get();
        llama_model_ptr draft(target && full_draft ?
            llama_model_load_from_file(path.string().c_str(), draft_params) : nullptr);
        auto cpu_target_params = make_params(nullptr, 0, nullptr);
        cpu_target_params.n_gpu_layers = 0;
        auto cpu_target = load(target_metadata.get(), cpu_target_params, seed);
        auto cpu_shared_draft_params = draft_params;
        cpu_shared_draft_params.model_shared = cpu_target.get();
        llama_model_ptr cpu_shared_draft(cpu_target && full_draft ?
            llama_model_load_from_file(path.string().c_str(), cpu_shared_draft_params) : nullptr);
        auto noalloc_target_params = make_params(nullptr, 0, nullptr);
        noalloc_target_params.no_alloc = true;
        auto noalloc_target = load(target_metadata.get(), noalloc_target_params, seed);
        auto noalloc_shared_draft_params = draft_params;
        noalloc_shared_draft_params.model_shared = noalloc_target.get();
        llama_model_ptr noalloc_shared_draft(noalloc_target && full_draft ?
            llama_model_load_from_file(path.string().c_str(), noalloc_shared_draft_params) : nullptr);
        std::error_code stamp_error;
        const auto initial_stamp = std::filesystem::last_write_time(path, stamp_error);
        if (!stamp_error) {
            std::filesystem::last_write_time(path, initial_stamp + std::chrono::seconds(1), stamp_error);
        }
        llama_model_ptr restamped_draft(!stamp_error && target && full_draft ?
            llama_model_load_from_file(path.string().c_str(), draft_params) : nullptr);
        std::error_code remove_error;
        std::filesystem::remove(path, remove_error);
        check(draft != nullptr && cpu_shared_draft != nullptr && noalloc_shared_draft != nullptr,
              "shared-target fixture failed to load GPU/CPU owner controls");
        if (draft) {
            const auto report = llama_model_moe_placement(draft.get());
            const auto full_report = llama_model_moe_placement(full_draft.get());
            check(report.model_identity_kind == "local_unverified" &&
                      report.model_identity_record.find("\"artifact_sources\":[{") != std::string::npos,
                  "file-backed draft identity omitted local artifact evidence");
            check(full_report.model_identity_kind == "layout_unverified",
                  "virtual full-draft control identity was mislabeled");
            check(report.placement_id != full_report.placement_id,
                  "shared storage did not change the resolved placement identity");
            check(!report.shared_tensors.empty(), "shared-target tensors were omitted from placement inventory");
            for (const auto & shared : report.shared_tensors) {
                check(!shared.name.empty() && shared.tensor_bytes > 0 && shared.resolved_storage_available &&
                          !shared.resolved_class.empty() && !shared.owner_canonical_id.empty() &&
                          shared.storage_relation == "borrowed_model_shared",
                      "shared-target tensor inventory is incomplete");
            }
            check(report.placement_record.find("shared_tensors") != std::string::npos,
                  "shared-target ownership is absent from placement identity");
            if (cpu_shared_draft) {
                const auto cpu_shared_report = llama_model_moe_placement(cpu_shared_draft.get());
                check(cpu_shared_report.model_identity == report.model_identity &&
                          cpu_shared_report.placement_id != report.placement_id,
                      "borrowed target storage owner did not affect resolved placement identity");
            }
            if (restamped_draft) {
                check(llama_model_moe_placement(restamped_draft.get()).model_identity != report.model_identity,
                      "file modification stamp did not affect local-unverified model identity");
            }
            if (noalloc_shared_draft) {
                const auto noalloc_shared_report = llama_model_moe_placement(noalloc_shared_draft.get());
                check(!noalloc_shared_report.shared_tensors.empty() &&
                          std::all_of(noalloc_shared_report.shared_tensors.begin(),
                              noalloc_shared_report.shared_tensors.end(), [](const auto & shared) {
                                  return shared.resolved_storage_available && !shared.current_storage_available &&
                                         shared.storage_provenance == "borrowed_model_shared_resolved_estimate";
                              }),
                      "no-allocation borrowed storage was mislabeled as a current backing");
            }
        }
    }

    if (cache_devs.size() >= 2) {
        ggml_backend_dev_t multi_devices[] = {cache_devs[0], cache_devs[1], nullptr};
        float tensor_split[] = {1.0f, 1.0f};
        const llama_model_layer_range multi_layers[] = {{0, 4}};
        llama_model_params params = llama_model_default_params();
        params.devices = multi_devices;
        params.tensor_split = tensor_split;
        params.n_gpu_layers = 99;
        params.split_mode = LLAMA_SPLIT_MODE_LAYER;
        params.load_mode = LLAMA_LOAD_MODE_NONE;
        params.progress_callback = silent_model_load_progress;
        params.moe_expert_cache_slots = 8;
        params.moe_expert_cache_layer_ranges = multi_layers;
        params.n_moe_expert_cache_layer_ranges = 1;
        gguf_context_ptr metadata = get_gguf_ctx(LLM_ARCH_QWEN3MOE, true, false, 8, 2, 5);
        auto model = load(metadata.get(), params, seed);
        check(model != nullptr, "multi-owner placement fixture failed to load");
        if (model) {
            const auto report = llama_model_moe_placement(model.get());
            std::unordered_set<std::string> active_owners;
            for (const auto & group : report.groups) {
                if (group.mode == LLAMA_MOE_PLACEMENT_RESIDUAL_CACHE) {
                    active_owners.insert(group.owner_canonical_id);
                    const char * owner_name = ggml_backend_dev_name(model->dev_layer(group.layer));
                    for (const auto & bank : group.banks) {
                        if (bank.status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE) {
                            check(bank.owner_name == owner_name,
                                  "cached expert bank owner differs from its layer execution device");
                        }
                    }
                }
            }
            check(report.owners.size() == 2 && active_owners.size() == 2,
                  "multi-owner resolved mapping did not cover both selected devices");
            check(model->dev_layer(2) == multi_devices[0] && model->dev_layer(3) == multi_devices[1],
                  "multi-owner fixture must use an asymmetric 3/2 layer split");
            for (size_t owner_index = 0; owner_index < report.owners.size(); ++owner_index) {
                size_t fixed = 0;
                size_t per_slot = 0;
                size_t groups = 0;
                for (const auto & group : report.groups) {
                    if (group.owner_index == static_cast<int32_t>(owner_index) &&
                        group.mode == LLAMA_MOE_PLACEMENT_RESIDUAL_CACHE) {
                        fixed += group.cache_fixed_bytes;
                        per_slot += group.cache_per_slot_bytes;
                        ++groups;
                    }
                }
                check(report.owners[owner_index].active_cache_groups == groups &&
                          report.owners[owner_index].cache_group_fixed_bytes == fixed &&
                          report.owners[owner_index].cache_group_per_slot_bytes == per_slot,
                      "multi-owner cache ledger mismatch");
            }

            const auto & memory = model->moe_expert_cache_memory();
            const auto first = memory.find(multi_devices[0]);
            const auto second = memory.find(multi_devices[1]);
            check(first != memory.end() && second != memory.end(),
                  "cache sizing used the host buffer provider instead of both layer owners");
            if (first != memory.end() && second != memory.end()) {
                const size_t budgets[] = {first->second.device_bytes(4), second->second.device_bytes(6)};
                auto budget_params = params;
                budget_params.moe_expert_cache_slots = 0;
                budget_params.moe_expert_cache_byte_budgets = budgets;
                budget_params.n_moe_expert_cache_byte_budgets = 2;
                // Exercise the implicit cache override as well as the layer selector above.
                budget_params.moe_expert_cache_layer_ranges = nullptr;
                budget_params.n_moe_expert_cache_layer_ranges = 0;
                auto budget_model = load(metadata.get(), budget_params, seed);
                check(budget_model != nullptr, "multi-owner byte-budget fixture failed to load");
                if (budget_model) {
                    check(budget_model->moe_expert_cache_slots(multi_devices[0]) == 4 &&
                              budget_model->moe_expert_cache_slots(multi_devices[1]) == 6,
                          "byte budgets did not produce owner-local slot counts");
                    auto probe_params = budget_params;
                    probe_params.no_alloc = true;
                    auto probe = load(metadata.get(), probe_params, seed);
                    check(probe && probe->moe_expert_cache_slots(multi_devices[0]) == 4 &&
                              probe->moe_expert_cache_slots(multi_devices[1]) == 6 &&
                              llama_model_moe_placement(probe.get()).placement_id ==
                                  llama_model_moe_placement(budget_model.get()).placement_id,
                          "allocation-free sizing disagreed with the live split owners");
                }
            }
        }
    }

    fprintf(stderr, "test-moe-placement: %s on %s\n", ok ? "PASS" : "FAIL",
            ggml_backend_dev_description(cache_dev));
    return ok ? 0 : 1;
}

struct joint_probe_sampler_counters {
    int clones      = 0;
    int clone_frees = 0;
};

struct joint_probe_sampler_state {
    joint_probe_sampler_counters * counters            = nullptr;
    bool                           backend_initialized = false;
    bool                           clone               = false;
};

static const char * joint_probe_sampler_name(const llama_sampler *) {
    return "joint-probe-ownership";
}

static void joint_probe_sampler_apply(llama_sampler *, llama_token_data_array *) {}

static llama_sampler * joint_probe_sampler_clone(const llama_sampler * sampler) {
    const auto * source = static_cast<const joint_probe_sampler_state *>(sampler->ctx);
    auto *       clone  = new joint_probe_sampler_state{ source->counters, false, true };
    ++clone->counters->clones;
    return llama_sampler_init(sampler->iface, clone);
}

static void joint_probe_sampler_free(llama_sampler * sampler) {
    auto * state = static_cast<joint_probe_sampler_state *>(sampler->ctx);
    if (state->clone) {
        ++state->counters->clone_frees;
    }
    delete state;
}

static bool joint_probe_sampler_backend_init(llama_sampler * sampler, ggml_backend_buffer_type_t, uint32_t) {
    static_cast<joint_probe_sampler_state *>(sampler->ctx)->backend_initialized = true;
    return true;
}

static void joint_probe_sampler_backend_apply(llama_sampler *, ggml_context *, ggml_cgraph *, llama_sampler_data *) {}

static llama_sampler_i joint_probe_sampler_i = {
    /* .name              = */ joint_probe_sampler_name,
    /* .accept            = */ nullptr,
    /* .apply             = */ joint_probe_sampler_apply,
    /* .reset             = */ nullptr,
    /* .clone             = */ joint_probe_sampler_clone,
    /* .free              = */ joint_probe_sampler_free,
    /* .backend_init      = */ joint_probe_sampler_backend_init,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ joint_probe_sampler_backend_apply,
    /* .backend_set_input = */ nullptr,
    /* .backend_reset     = */ nullptr,
    /* .copy_state        = */ nullptr,
};

static int test_moe_joint_measurement(const size_t seed, const std::string & fit_tool) {
    bool       ok    = true;
    const auto check = [&](bool condition, const char * message) {
        if (!condition) {
            fprintf(stderr, "test-moe-joint-measurement: %s\n", message);
            ok = false;
        }
    };

    const auto nonce        = std::to_string(seed) + "-" + std::to_string(reinterpret_cast<uintptr_t>(&ok));
    const auto temp         = std::filesystem::temp_directory_path();
    const auto target_path  = temp / ("llama-joint-target-" + nonce + ".gguf");
    const auto mtp_path     = temp / ("llama-joint-mtp-" + nonce + ".gguf");
    const auto partial_path = temp / ("llama-joint-partial-" + nonce + ".gguf");

    ggml_backend_dev_t cpu_devices[] = { nullptr };
    auto               load_params   = llama_model_default_params();
    load_params.devices              = cpu_devices;
    load_params.progress_callback    = silent_model_load_progress;

    auto            target_metadata = get_gguf_ctx(LLM_ARCH_QWEN35MOE, true);
    size_t          target_seed     = seed;
    llama_model_ptr target(
        llama_model_init_from_user(target_metadata.get(), set_tensor_data, &target_seed, load_params));
    auto mtp_metadata        = get_gguf_ctx(LLM_ARCH_QWEN35MOE, true, true);
    auto mtp_params          = load_params;
    mtp_params.load_mtp      = true;
    size_t          mtp_seed = seed;
    llama_model_ptr mtp(llama_model_init_from_user(mtp_metadata.get(), set_tensor_data, &mtp_seed, mtp_params));
    check(target != nullptr && mtp != nullptr, "failed to build source fixtures");

    const auto save_fixture = [&](const std::filesystem::path & path, gguf_context * metadata, llama_model * model,
                                  bool partial) {
        if (model == nullptr) {
            return;
        }
        llama_model_saver saver(LLM_ARCH_QWEN35MOE, nullptr);
        gguf_set_kv(saver.gguf_ctx, metadata);
        if (partial) {
            saver.add_kv(LLM_KV_NEXTN_SHARED_TARGET_TENSORS, true);
        }
        for (const auto & [name, tensor] : model->tensors_by_name) {
            if (partial && (name == "token_embd.weight" || name == "output.weight" || name == "output_norm.weight")) {
                continue;
            }
            saver.add_tensor(tensor);
        }
        saver.save(path.string());
    };
    save_fixture(target_path, target_metadata.get(), target.get(), false);
    save_fixture(mtp_path, mtp_metadata.get(), mtp.get(), false);
    save_fixture(partial_path, mtp_metadata.get(), mtp.get(), true);

    const auto make_component = [&](const std::filesystem::path & path, bool load_mtp) {
        common_joint_component_request component;
        component.path_model        = path.string();
        component.mparams           = llama_model_default_params();
        component.mparams.devices   = cpu_devices;
        component.mparams.load_mtp  = load_mtp;
        component.cparams           = llama_context_default_params();
        component.cparams.n_ctx     = 32;
        component.cparams.n_batch   = 32;
        component.cparams.n_ubatch  = 16;
        component.cparams.n_seq_max = 1;
        return component;
    };
    const auto make_request = [&](const std::filesystem::path & path, bool load_mtp) {
        common_joint_measurement_request request;
        request.target              = make_component(path, load_mtp);
        request.host_capacity_bytes = size_t(8) * 1024 * 1024 * 1024;
        request.host_margin_bytes   = 64 * 1024 * 1024;
        return request;
    };
    const auto find_component = [](const common_joint_measurement & measurement, common_joint_role role) {
        return std::find_if(measurement.components.begin(), measurement.components.end(),
                            [role](const auto & component) { return component.role == role; });
    };
    const auto component_memory = [](const common_joint_component_measurement & component) {
        std::map<std::string, std::array<size_t, 5>> result;
        for (const auto & owner : component.owners) {
            result[owner.canonical_id] = {
                owner.memory.model,
                owner.memory.context,
                owner.memory.compute,
                owner.memory.staging,
                owner.memory.staging_available ? size_t(1) : size_t(0),
            };
        }
        return result;
    };
    const auto sampler_count = [](const common_joint_component_measurement & component) {
        if (component.configuration_record.empty()) {
            return size_t(0);
        }
        return common_json::parse(component.configuration_record)
            .at("context").at("samplers").at("sequences").size();
    };

    joint_probe_sampler_counters sampler_counters;
    auto *            original_sampler_state = new joint_probe_sampler_state{ &sampler_counters, false, false };
    llama_sampler_ptr sampler_chain(llama_sampler_chain_init(llama_sampler_chain_default_params()));
    llama_sampler_chain_add(sampler_chain.get(), llama_sampler_init(&joint_probe_sampler_i, original_sampler_state));
    llama_sampler_seq_config sampler_config = { 0, sampler_chain.get() };

    auto target_only_request                            = make_request(target_path, false);
    target_only_request.target.sampler_configuration_id = "joint-probe-ownership-v1";
    target_only_request.target.cparams.samplers         = &sampler_config;
    target_only_request.target.cparams.n_samplers       = 1;
    std::vector<float> target_split(llama_max_devices(), 0.0f);
    target_split[0]                                 = 1.0f;
    target_only_request.target.mparams.tensor_split = target_split.data();
    const auto        target_devices_ptr            = target_only_request.target.mparams.devices;
    const auto        target_split_ptr              = target_only_request.target.mparams.tensor_split;
    const auto        target_ctx                    = target_only_request.target.cparams.n_ctx;
    ggml_log_callback logger_before                 = nullptr;
    void *            logger_data_before            = nullptr;
    llama_log_get(&logger_before, &logger_data_before);
    const auto        target_only_a     = common_measure_joint_configuration(target_only_request);
    const auto        target_only_b     = common_measure_joint_configuration(target_only_request);
    ggml_log_callback logger_after      = nullptr;
    void *            logger_data_after = nullptr;
    llama_log_get(&logger_after, &logger_data_after);
    check(target_only_a.completeness == COMMON_JOINT_COMPLETENESS_COMPLETE &&
              target_only_a.capacity == COMMON_JOINT_CAPACITY_WITHIN_LIMITS && target_only_a.admission_qualified,
          "target-only measurement did not qualify");
    check(!target_only_a.configuration_id.empty() && target_only_a.configuration_id == target_only_b.configuration_id &&
              target_only_a.configuration_record == target_only_b.configuration_record,
          "repeated target probes were not deterministic");
    check(target_only_request.target.mparams.devices == target_devices_ptr &&
              target_only_request.target.mparams.tensor_split == target_split_ptr && target_split[0] == 1.0f &&
              target_only_request.target.mparams.devices[0] == nullptr &&
              target_only_request.target.cparams.n_ctx == target_ctx,
          "joint measurement mutated borrowed input parameters");
    check(!original_sampler_state->backend_initialized && sampler_counters.clones == 2 &&
              sampler_counters.clone_frees == sampler_counters.clones,
          "joint measurement mutated the caller sampler or leaked a successful-probe clone");
    auto       no_sampler_request = make_request(target_path, false);
    const auto no_sampler         = common_measure_joint_configuration(no_sampler_request);
    check(no_sampler.completeness == COMMON_JOINT_COMPLETENESS_COMPLETE &&
              no_sampler.configuration_id != target_only_a.configuration_id,
          "sampler presence was omitted from configuration identity");
    auto unidentified_sampler_request = target_only_request;
    unidentified_sampler_request.target.sampler_configuration_id.clear();
    const auto unidentified_sampler = common_measure_joint_configuration(unidentified_sampler_request);
    check(unidentified_sampler.completeness == COMMON_JOINT_COMPLETENESS_INCOMPLETE &&
              !unidentified_sampler.admission_qualified && unidentified_sampler.configuration_id.empty() &&
              !original_sampler_state->backend_initialized && sampler_counters.clone_frees == sampler_counters.clones,
          "sampler-backed measurement without semantic identity was treated as resolved");
    auto changed_sampler_request                            = target_only_request;
    changed_sampler_request.target.sampler_configuration_id = "joint-probe-ownership-v2";
    check(
        common_measure_joint_configuration(changed_sampler_request).configuration_id != target_only_a.configuration_id,
        "sampler semantic identity was omitted from configuration identity");
    check(logger_before == logger_after && logger_data_before == logger_data_after,
          "joint measurement did not restore the global logger");

    common_params wrapped_target_params;
    wrapped_target_params.model.path               = target_path.string();
    wrapped_target_params.devices                  = { nullptr };
    wrapped_target_params.n_ctx                    = 32;
    wrapped_target_params.n_batch                  = 32;
    wrapped_target_params.n_ubatch                 = 16;
    wrapped_target_params.n_parallel               = 1;
    wrapped_target_params.n_gpu_layers             = 0;
    wrapped_target_params.sampling.backend_sampling = true;
    const auto wrapped_target_params_before = wrapped_target_params;
    const auto wrapped_target = common_measure_joint_configuration(wrapped_target_params);
    const auto wrapped_target_repeat = common_measure_joint_configuration(wrapped_target_params);
    const auto wrapped_target_component = find_component(wrapped_target, COMMON_JOINT_ROLE_TARGET);

    auto explicit_target_sampling = wrapped_target_params.sampling;
    common_params_sampling_prepare(target.get(), explicit_target_sampling);
    common_sampler_ptr explicit_target_sampler(common_sampler_init(target.get(), explicit_target_sampling));
    llama_sampler_seq_config explicit_target_config = { 0, common_sampler_get(explicit_target_sampler.get()) };
    auto explicit_target_request = make_request(target_path, false);
    explicit_target_request.target.cparams.samplers = &explicit_target_config;
    explicit_target_request.target.cparams.n_samplers = 1;
    explicit_target_request.target.sampler_configuration_id = "explicit-target-runtime-equivalent-v1";
    const auto explicit_target = common_measure_joint_configuration(explicit_target_request);
    const auto explicit_target_component = find_component(explicit_target, COMMON_JOINT_ROLE_TARGET);
    check(wrapped_target.completeness == COMMON_JOINT_COMPLETENESS_COMPLETE &&
              wrapped_target_component != wrapped_target.components.end() &&
              explicit_target_component != explicit_target.components.end() &&
              sampler_count(*wrapped_target_component) == 1 &&
              component_memory(*wrapped_target_component) == component_memory(*explicit_target_component),
          "common-params wrapper omitted or mismeasured target backend sampling");
    check(!wrapped_target.configuration_id.empty() &&
              wrapped_target.configuration_id == wrapped_target_repeat.configuration_id &&
              wrapped_target.configuration_record == wrapped_target_repeat.configuration_record,
          "wrapper-generated target sampler identity was not stable");
    auto changed_wrapped_target_params = wrapped_target_params;
    changed_wrapped_target_params.sampling.top_k++;
    const auto changed_wrapped_target = common_measure_joint_configuration(changed_wrapped_target_params);
    check(changed_wrapped_target.completeness == COMMON_JOINT_COMPLETENESS_COMPLETE &&
              changed_wrapped_target.configuration_id != wrapped_target.configuration_id,
          "target sampler semantics were omitted from wrapper configuration identity");
    check(wrapped_target_params.model.path == wrapped_target_params_before.model.path &&
              wrapped_target_params.devices == wrapped_target_params_before.devices &&
              wrapped_target_params.n_ctx == wrapped_target_params_before.n_ctx &&
              wrapped_target_params.n_batch == wrapped_target_params_before.n_batch &&
              wrapped_target_params.n_ubatch == wrapped_target_params_before.n_ubatch &&
              wrapped_target_params.n_parallel == wrapped_target_params_before.n_parallel &&
              wrapped_target_params.sampling.backend_sampling ==
                  wrapped_target_params_before.sampling.backend_sampling &&
              wrapped_target_params.sampling.top_k == wrapped_target_params_before.sampling.top_k &&
              wrapped_target_params.sampling.logit_bias.size() ==
                  wrapped_target_params_before.sampling.logit_bias.size() &&
              wrapped_target_params.sampling.logit_bias_eog.size() ==
                  wrapped_target_params_before.sampling.logit_bias_eog.size(),
          "common-params wrapper mutated caller target sampling parameters");
    auto invalid_wrapped_target_params = wrapped_target_params;
    invalid_wrapped_target_params.sampling.penalty_repeat = 0.0f;
    const auto invalid_wrapped_target = common_measure_joint_configuration(invalid_wrapped_target_params);
    check(invalid_wrapped_target.completeness == COMMON_JOINT_COMPLETENESS_ERROR &&
              !invalid_wrapped_target.admission_qualified && invalid_wrapped_target.configuration_id.empty() &&
              std::any_of(invalid_wrapped_target.diagnostics.begin(), invalid_wrapped_target.diagnostics.end(),
                          [](const std::string & diagnostic) {
                              return diagnostic.find("target backend sampler probe") != std::string::npos;
                          }),
          "target sampler preparation failure escaped its structured error result");
    try {
        const auto round_trip = common_json::parse(common_joint_measurement_json(target_only_a));
        check(round_trip.at("configuration_id").get<std::string>() == target_only_a.configuration_id &&
                  round_trip.at("admission_qualified").get<bool>(),
              "joint JSON did not round-trip");
    } catch (const std::exception &) {
        check(false, "joint JSON was not parseable");
    }
    const auto host_owner = std::find_if(target_only_a.owners.begin(), target_only_a.owners.end(),
                                         [](const auto & owner) { return owner.host; });
    check(host_owner != target_only_a.owners.end(), "target-only measurement omitted the host owner");
    if (host_owner != target_only_a.owners.end()) {
        auto at_upper                = target_only_request;
        at_upper.host_capacity_bytes = host_owner->required_upper_bytes + at_upper.host_margin_bytes;
        const auto within            = common_measure_joint_configuration(at_upper);
        check(within.capacity == COMMON_JOINT_CAPACITY_WITHIN_LIMITS && within.admission_qualified,
              "upper-bound capacity boundary was not admitted");
        if (host_owner->required_lower_bytes > 0) {
            auto below_lower                = target_only_request;
            below_lower.host_capacity_bytes = host_owner->required_lower_bytes + below_lower.host_margin_bytes - 1;
            const auto exceeds              = common_measure_joint_configuration(below_lower);
            check(exceeds.capacity == COMMON_JOINT_CAPACITY_EXCEEDS_LIMITS && !exceeds.admission_qualified,
                  "lower-bound capacity boundary was not rejected");
        }
        if (host_owner->required_lower_bytes < host_owner->required_upper_bytes) {
            auto between                = target_only_request;
            between.host_capacity_bytes = host_owner->required_lower_bytes + between.host_margin_bytes;
            const auto ambiguous        = common_measure_joint_configuration(between);
            check(ambiguous.capacity == COMMON_JOINT_CAPACITY_UNKNOWN && !ambiguous.admission_qualified,
                  "lower/upper capacity interval was not reported as unknown");
        }
        check(within.configuration_id != target_only_a.configuration_id,
              "explicit capacity envelope was omitted from configuration identity");
    }
    auto changed_context                   = target_only_request;
    changed_context.target.cparams.n_batch = 16;
    check(common_measure_joint_configuration(changed_context).configuration_id != target_only_a.configuration_id,
          "context geometry was omitted from configuration identity");

    auto shared_request                                 = make_request(mtp_path, true);
    shared_request.target.cparams.phase_aware_workspace = true;
    auto shared_mtp                                     = make_component(mtp_path, true);
    shared_mtp.role                                     = COMMON_JOINT_ROLE_MTP;
    shared_mtp.sharing                                  = COMMON_JOINT_SHARING_TARGET_MODEL;
    shared_request.extras.push_back(shared_mtp);
    const auto shared       = common_measure_joint_configuration(shared_request);
    const auto shared_extra = find_component(shared, COMMON_JOINT_ROLE_MTP);
    check(shared.completeness == COMMON_JOINT_COMPLETENESS_COMPLETE && shared.admission_qualified &&
              shared_extra != shared.components.end() && shared_extra->shared_model_bytes_deduplicated > 0,
          "fully shared target/MTP accounting failed");
    check(std::any_of(shared.owners.begin(), shared.owners.end(),
                      [](const auto & owner) {
                          return owner.bound == COMMON_JOINT_BOUND_UPPER_ESTIMATE &&
                                 owner.required_lower_bytes < owner.required_upper_bytes;
                      }),
          "phase-aware shared workspace did not retain a conservative bound");

    common_params wrapped_mtp_params;
    wrapped_mtp_params.model.path    = mtp_path.string();
    wrapped_mtp_params.devices       = { nullptr };
    wrapped_mtp_params.n_ctx         = 32;
    wrapped_mtp_params.n_batch       = 32;
    wrapped_mtp_params.n_ubatch      = 16;
    wrapped_mtp_params.n_parallel    = 1;
    wrapped_mtp_params.n_gpu_layers  = 0;
    wrapped_mtp_params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    wrapped_mtp_params.speculative.draft.n_max = 1;
    wrapped_mtp_params.speculative.draft.backend_sampling = true;
    const auto wrapped_mtp = common_measure_joint_configuration(wrapped_mtp_params);
    const auto wrapped_mtp_component = find_component(wrapped_mtp, COMMON_JOINT_ROLE_MTP);

    llama_sampler_ptr explicit_mtp_sampler(llama_sampler_chain_init(llama_sampler_chain_default_params()));
    llama_sampler_chain_add(explicit_mtp_sampler.get(), llama_sampler_init_top_k(10));
    llama_sampler_seq_config explicit_mtp_config = { 0, explicit_mtp_sampler.get() };
    auto explicit_mtp_target_params = wrapped_mtp_params;
    common_joint_measurement_request explicit_mtp_request;
    explicit_mtp_request.target.path_model = mtp_path.string();
    explicit_mtp_request.target.mparams = common_model_params_to_llama(explicit_mtp_target_params);
    explicit_mtp_request.target.cparams = common_context_params_to_llama(explicit_mtp_target_params);
    explicit_mtp_request.target.role = COMMON_JOINT_ROLE_TARGET;
    explicit_mtp_request.target.required = true;
    auto explicit_mtp_draft_params = common_base_params_to_speculative(explicit_mtp_target_params);
    common_joint_component_request explicit_mtp_component_request;
    explicit_mtp_component_request.path_model = mtp_path.string();
    explicit_mtp_component_request.mparams = common_model_params_to_llama(explicit_mtp_draft_params);
    explicit_mtp_component_request.cparams = common_context_params_to_llama(explicit_mtp_draft_params);
    explicit_mtp_component_request.cparams.n_rs_seq = 0;
    explicit_mtp_component_request.role = COMMON_JOINT_ROLE_MTP;
    explicit_mtp_component_request.sharing = COMMON_JOINT_SHARING_TARGET_MODEL;
    explicit_mtp_component_request.required = true;
    explicit_mtp_component_request.cparams.samplers = &explicit_mtp_config;
    explicit_mtp_component_request.cparams.n_samplers = 1;
    explicit_mtp_component_request.sampler_configuration_id = "explicit-mtp-top-k-10-v1";
    explicit_mtp_request.extras.push_back(explicit_mtp_component_request);
    const auto explicit_mtp = common_measure_joint_configuration(explicit_mtp_request);
    const auto explicit_mtp_component = find_component(explicit_mtp, COMMON_JOINT_ROLE_MTP);
    check(wrapped_mtp.completeness == COMMON_JOINT_COMPLETENESS_COMPLETE &&
              wrapped_mtp_component != wrapped_mtp.components.end() &&
              explicit_mtp_component != explicit_mtp.components.end() &&
              sampler_count(*wrapped_mtp_component) == 1 &&
              component_memory(*wrapped_mtp_component) == component_memory(*explicit_mtp_component),
          "common-params wrapper omitted or mismeasured default MTP backend sampling");
    const auto wrapped_mtp_repeat = common_measure_joint_configuration(wrapped_mtp_params);
    check(!wrapped_mtp.configuration_id.empty() &&
              wrapped_mtp.configuration_id == wrapped_mtp_repeat.configuration_id &&
              wrapped_mtp.configuration_record == wrapped_mtp_repeat.configuration_record,
          "wrapper-generated MTP sampler identity was not stable");

    auto unsupported_backend_draft = wrapped_mtp_params;
    unsupported_backend_draft.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3 };
    unsupported_backend_draft.speculative.draft.mparams.path = target_path.string();
    const auto unsupported_backend = common_measure_joint_configuration(unsupported_backend_draft);
    check(unsupported_backend.completeness == COMMON_JOINT_COMPLETENESS_ERROR &&
              !unsupported_backend.admission_qualified && unsupported_backend.configuration_id.empty() &&
              std::any_of(unsupported_backend.diagnostics.begin(), unsupported_backend.diagnostics.end(),
                          [](const std::string & diagnostic) {
                              return diagnostic.find("does not yet represent this draft backend sampler") !=
                                     std::string::npos;
                          }),
          "unrepresented draft backend sampler was treated as qualified");

    auto separate_request = make_request(target_path, false);
    auto separate_draft   = make_component(target_path, false);
    separate_draft.role   = COMMON_JOINT_ROLE_DRAFT;
    separate_request.extras.push_back(separate_draft);
    const auto separate       = common_measure_joint_configuration(separate_request);
    const auto separate_extra = find_component(separate, COMMON_JOINT_ROLE_DRAFT);
    check(separate.completeness == COMMON_JOINT_COMPLETENESS_COMPLETE && separate_extra != separate.components.end() &&
              separate_extra->shared_model_bytes_deduplicated == 0 && separate_extra->sharing_records.empty(),
          "separate draft accounting was deduplicated");

    auto partial_request = make_request(target_path, false);
    auto partial_mtp     = make_component(partial_path, true);
    partial_mtp.role     = COMMON_JOINT_ROLE_MTP;
    partial_mtp.sharing  = COMMON_JOINT_SHARING_BORROW_TARGET;
    partial_request.extras.push_back(partial_mtp);
    const auto partial       = common_measure_joint_configuration(partial_request);
    const auto partial_extra = find_component(partial, COMMON_JOINT_ROLE_MTP);
    const bool partial_ok = partial.completeness == COMMON_JOINT_COMPLETENESS_COMPLETE && partial.admission_qualified &&
                            partial_extra != partial.components.end() &&
                            partial_extra->shared_tensor_payload_bytes > 0 && !partial_extra->sharing_records.empty() &&
                            partial_extra->shared_model_bytes_deduplicated == 0 &&
                            std::all_of(partial_extra->sharing_records.begin(), partial_extra->sharing_records.end(),
                                        [](const auto & record) { return !record.owner_canonical_id.empty(); });
    check(partial_ok, "resolvable partial sharing did not produce a precise owned ledger");
    check(partial.configuration_id != separate.configuration_id,
          "sharing relationship was omitted from joint configuration identity");

    ggml_backend_dev_t gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu != nullptr) {
        ggml_backend_dev_t gpu_devices[] = { gpu, nullptr };
        auto               gpu_component = [&](const std::filesystem::path & path, bool load_mtp) {
            auto component                 = make_component(path, load_mtp);
            component.mparams.devices      = gpu_devices;
            component.mparams.n_gpu_layers = 99;
            return component;
        };
        auto gpu_partial_request                    = make_request(target_path, false);
        gpu_partial_request.target                  = gpu_component(target_path, false);
        gpu_partial_request.observe_device_capacity = true;
        auto gpu_partial_mtp                        = gpu_component(partial_path, true);
        gpu_partial_mtp.role                        = COMMON_JOINT_ROLE_MTP;
        gpu_partial_mtp.sharing                     = COMMON_JOINT_SHARING_BORROW_TARGET;
        gpu_partial_request.extras.push_back(gpu_partial_mtp);
        const auto gpu_partial       = common_measure_joint_configuration(gpu_partial_request);
        const auto gpu_partial_extra = find_component(gpu_partial, COMMON_JOINT_ROLE_MTP);

        auto gpu_full_request   = make_request(target_path, false);
        gpu_full_request.target = gpu_component(target_path, false);
        auto gpu_full_mtp       = gpu_component(mtp_path, true);
        gpu_full_mtp.role       = COMMON_JOINT_ROLE_MTP;
        gpu_full_request.extras.push_back(gpu_full_mtp);
        const auto gpu_full       = common_measure_joint_configuration(gpu_full_request);
        const auto gpu_full_extra = find_component(gpu_full, COMMON_JOINT_ROLE_MTP);
        const auto model_bytes    = [](const common_joint_component_measurement & component) {
            size_t result = 0;
            for (const auto & owner : component.owners) {
                result += owner.memory.model;
            }
            return result;
        };
        const bool gpu_partial_ok =
            gpu_partial.completeness == COMMON_JOINT_COMPLETENESS_COMPLETE &&
            gpu_partial_extra != gpu_partial.components.end() && gpu_full_extra != gpu_full.components.end() &&
            gpu_partial_extra->shared_tensor_payload_bytes > 0 && model_bytes(*gpu_partial_extra) > 0 &&
            model_bytes(*gpu_partial_extra) < model_bytes(*gpu_full_extra);
        check(gpu_partial_ok, "resolved partial sharing did not retain unique head bytes and exclude borrowed storage");
    }

    auto required_failure_request                            = make_request(target_path, false);
    required_failure_request.target.sampler_configuration_id = "joint-probe-ownership-v1";
    required_failure_request.target.cparams.samplers         = &sampler_config;
    required_failure_request.target.cparams.n_samplers       = 1;
    auto missing = make_component(temp / ("missing-joint-" + nonce + ".gguf"), false);
    missing.role = COMMON_JOINT_ROLE_DRAFT;
    required_failure_request.extras.push_back(missing);
    const auto required_failure = common_measure_joint_configuration(required_failure_request);
    check(required_failure.completeness == COMMON_JOINT_COMPLETENESS_ERROR && !required_failure.admission_qualified &&
              required_failure.configuration_id.empty(),
          "failed required extra became a target-only success");
    check(!original_sampler_state->backend_initialized && sampler_counters.clone_frees == sampler_counters.clones,
          "failed joint measurement mutated the caller sampler or leaked a probe clone");
    missing.required              = false;
    auto optional_failure_request = make_request(target_path, false);
    optional_failure_request.extras.push_back(missing);
    const auto optional_failure = common_measure_joint_configuration(optional_failure_request);
    check(optional_failure.completeness == COMMON_JOINT_COMPLETENESS_INCOMPLETE &&
              !optional_failure.admission_qualified && optional_failure.components.size() == 2 &&
              optional_failure.configuration_id.empty(),
          "failed optional extra was hidden or relabeled");
    llama_log_get(&logger_after, &logger_data_after);
    check(logger_before == logger_after && logger_data_before == logger_data_after,
          "failed joint probe did not restore the global logger");

#if !defined(_WIN32)
    if (!fit_tool.empty()) {
        const auto quote = [](const std::string & value) {
            std::string result = "'";
            for (const char c : value) {
                result += c == '\'' ? "'\\''" : std::string(1, c);
            }
            return result + "'";
        };
        const auto stdout_path = temp / ("llama-joint-tool-out-" + nonce + ".json");
        const auto stderr_path = temp / ("llama-joint-tool-err-" + nonce + ".log");
        const auto read_file   = [](const std::filesystem::path & path) {
            std::ifstream input(path, std::ios::binary);
            return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
        };
        const auto run_tool = [&](const std::string & args) {
            const std::string command =
                quote(fit_tool) + " " + args + " >" + quote(stdout_path.string()) + " 2>" + quote(stderr_path.string());
            const int status = std::system(command.c_str());
            return std::make_tuple(status == -1 ? -1 : WEXITSTATUS(status), read_file(stdout_path),
                                   read_file(stderr_path));
        };

        const std::string common_args = " -ngl 0 -c 32 -b 32 -ub 16 -fit off";
        const auto [qualified_status, qualified_stdout, qualified_stderr] =
            run_tool("--fit-moe-joint-report-json -m " + quote(mtp_path.string()) +
                     " --spec-type draft-mtp --spec-draft-n-max 1 -bs" + common_args);
        GGML_UNUSED(qualified_stderr);
        try {
            const auto output = common_json::parse(qualified_stdout);
            check(qualified_status == 0 && std::count(qualified_stdout.begin(), qualified_stdout.end(), '\n') == 1 &&
                      output.at("admission_qualified").get<bool>() && output.at("components").size() == 2 &&
                      output.at("components").at(0).at("configuration").at("context")
                          .at("samplers").at("sequences").size() == 1 &&
                      output.at("components").at(1).at("configuration").at("context")
                          .at("samplers").at("sequences").size() == 1,
                  "strict fit executable did not emit one qualified target/MTP JSON object");
        } catch (const std::exception &) {
            check(false, "strict fit executable stdout was not one JSON object");
        }

        const auto missing_path = temp / ("missing-tool-joint-" + nonce + ".gguf");
        const auto [failed_status, failed_stdout, failed_stderr] =
            run_tool("--fit-moe-joint-report-json -m " + quote(target_path.string()) + " -md " +
                     quote(missing_path.string()) + " --spec-type draft-mtp --spec-draft-n-max 1" + common_args);
        GGML_UNUSED(failed_stderr);
        try {
            const auto output = common_json::parse(failed_stdout);
            check(failed_status == 2 && !output.at("admission_qualified").get<bool>() &&
                      output.at("measurement_completeness").get<std::string>() == "error",
                  "failed required draft did not produce strict unqualified JSON");
        } catch (const std::exception &) {
            check(false, "failed strict fit executable stdout was not JSON");
        }

        const auto [legacy_status, legacy_stdout, legacy_stderr] =
            run_tool("-m " + quote(target_path.string()) + " -md " + quote(mtp_path.string()) +
                     " --spec-type draft-mtp --spec-draft-n-max 1" + common_args);
        check(legacy_status == 1 && legacy_stdout.empty() &&
                  legacy_stderr.find("require --fit-moe-joint-report-json") != std::string::npos,
              "legacy fit silently accepted model-backed speculative arguments");

        std::error_code error;
        std::filesystem::remove(stdout_path, error);
        std::filesystem::remove(stderr_path, error);
    }
#else
    GGML_UNUSED(fit_tool);
#endif

    for (const auto & path : { target_path, mtp_path, partial_path }) {
        std::error_code error;
        std::filesystem::remove(path, error);
    }
    fprintf(stderr, "test-moe-joint-measurement: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
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

    bool ok = true;
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

            const std::string path_q4_0 = dir + "/" + llm_arch_name(arch) + (moe ? "-moe-q4_0.gguf" : "-dense-q4_0.gguf");
            LOG_INF("%s: Quantizing %s model (%s) to %s...\n", __func__, llm_arch_name(arch), moe ? "MoE" : "dense", path_q4_0.c_str());
            llama_model_quantize_params qparams = llama_model_quantize_default_params();
            qparams.ftype   = LLAMA_FTYPE_MOSTLY_Q4_0;
            qparams.nthread = 1;
            if (llama_model_quantize(path.c_str(), path_q4_0.c_str(), &qparams) != 0) {
                LOG_ERR("%s: failed to quantize %s\n", __func__, path.c_str());
                // a failed quantization can leave a file with an invalid header
                std::filesystem::remove(path_q4_0);
                ok = false;
            }
        }
    }
    llama_log_set(ud.log_old.callback, ud.log_old.user_data);
    return ok ? 0 : 1;
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

        device_config(std::vector<ggml_backend_dev_t> devs, std::string name, llama_split_mode split_mode)
            : devs(std::move(devs)), label(std::move(name)), split_mode(split_mode) {}
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
    }

    size_t max_arch_name_length = 0;
    for (const llm_arch & arch : llm_arch_all()) {
        max_arch_name_length = std::max(max_arch_name_length, strlen(llm_arch_name(arch)));
    }

    const std::string template_header  = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|%15s|%9s|%9s|\n";
    const std::string template_row_cfg = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|";
    const std::string template_row_res = "%15s %10s|%20s|%20s|\n";

    bool all_ok = true;
    common_log_flush(common_log_main());
    printf(template_header.c_str(), "Model arch.", "Device", "Config", "NMSE vs. CPU", "Roundtrip", "Parallel");
    printf("|");
    for (size_t i = 0; i < max_arch_name_length; i++) {
        printf("-");
    }
    printf("|");
    for (size_t i = 0; i < max_device_label_length; i++) {
        printf("-");
    }
    printf("|------|---------------|---------|---------|\n");
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
            std::vector<float> logits_cpu;
            for (device_config & dc : dev_configs) {
                // print test config first; should anything fail during model loading or inference, at least we know which test case caused it
                printf(template_row_cfg.c_str(),
                    llm_arch_name(arch), dc.label.c_str(), config_name.c_str());
                fflush(stdout);

                std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_dev;
                std::vector<float> logits_dev;
                std::string status_nmse      = "\033[1;33mSKIP\033[0m";
                std::string status_roundtrip = "\033[1;33mSKIP\033[0m";
                std::string status_parallel  = "\033[1;33mSKIP\033[0m";
                char nmse_str[12] = {0};

                bool skip = !arch_supported(arch) || (dc.split_mode == LLAMA_SPLIT_MODE_TENSOR && dc.devs.empty());
                if (!skip) {
                    if (logits_cpu.empty()) {
                        model_and_ctx_cpu = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, {}, LLAMA_SPLIT_MODE_LAYER, encode);
                        logits_cpu = get_logits(model_and_ctx_cpu.first.get(), model_and_ctx_cpu.second.get(), tokens, encode);
                    }
                    if (dc.split_mode != LLAMA_SPLIT_MODE_TENSOR || llm_arch_supports_sm_tensor(arch)) {
                        model_and_ctx_dev = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, dc.devs, dc.split_mode, encode);
                        logits_dev = get_logits(model_and_ctx_dev.first.get(), model_and_ctx_dev.second.get(), tokens, encode);
                        const double nmse_val = nmse(logits_cpu, logits_dev);
                        snprintf(nmse_str, sizeof(nmse_str), "(%.2e)", nmse_val);
                        status_nmse = "\033[1;32mOK\033[0m";
                        if (nmse_val > 1e-4) {
                            all_ok = false;
                            status_nmse = "\033[1;31mFAIL\033[0m";
                        }

                        // FIXME: T5 kq_b does not broadcast over KV streams, so context init with n_seq_max > 1 aborts
                        if (arch != LLM_ARCH_T5) {
                            status_parallel = "\033[1;32mOK\033[0m";
                            if (!test_parallel_seqs(model_and_ctx_dev.first.get(), tokens, logits_dev, encode)) {
                                all_ok = false;
                                status_parallel = "\033[1;31mFAIL\033[0m";
                            }
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

                        auto model_and_ctx_roundtrip = get_model_and_ctx(nullptr, file, seed, dc.devs, dc.split_mode, encode);
                        const std::vector<float> logits_roundtrip = get_logits(
                            model_and_ctx_roundtrip.first.get(), model_and_ctx_roundtrip.second.get(), tokens, encode);
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
                    status_nmse.c_str(), nmse_str, status_roundtrip.c_str(), status_parallel.c_str());
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
    bool run_speculative_limits = false;
    bool run_moe_cache_selector = false;
    bool run_moe_placement = false;
    bool        run_moe_joint_measurement   = false;
    std::string fit_tool;

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
        if (strcmp(argv[i], "--test-speculative-limits") == 0) {
            run_speculative_limits = true;
            continue;
        }
        if (strcmp(argv[i], "--test-moe-cache-selector") == 0) {
            run_moe_cache_selector = true;
            continue;
        }
        if (strcmp(argv[i], "--test-moe-placement") == 0) {
            run_moe_placement = true;
            continue;
        }
        if (strcmp(argv[i], "--test-moe-joint-measurement") == 0) {
            run_moe_joint_measurement = true;
            continue;
        }
        if (strcmp(argv[i], "--fit-tool") == 0 && i + 1 < argc) {
            fit_tool = argv[++i];
            continue;
        }
    }
    if (test_phase_workspace || test_live_context_workspace || run_speculative_limits || run_moe_cache_selector ||
        run_moe_placement || run_moe_joint_measurement) {
        common_log_set_verbosity_thold(verbosity);
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
            test_live_context_workspace_indexer_reserve(seed);
            test_live_context_workspace_unsupported(seed);
            return 0;
        }
        if (run_speculative_limits) {
            test_speculative_limits(seed);
            return 0;
        }
        if (run_moe_cache_selector) {
            return test_moe_cache_selector_precedence(seed);
        }
        if (run_moe_placement) {
            return test_moe_placement_report(seed);
        }
        if (run_moe_joint_measurement) {
            return test_moe_joint_measurement(seed, fit_tool);
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
