#include "../ggml/src/ggml-backend-impl.h"
#include "../ggml/src/ggml-backend-moe.h"
#include "../ggml/src/ggml-moe-source-program.h"
#include "../ggml/src/ggml-cuda/moe-source-core.cuh"
#include "../ggml/src/ggml-impl.h"
#include "common.h"
#include "arg.h"
#include "fit.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#include "hash/hash.h"
#include "json.h"
#include "llama-cpp.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "ngram-map.h"

// TODO: replace with #include "llama-ext.h" in the future
#include "../src/llama-arch.h"
#include "../src/llama-batch.h"
#include "../src/llama-context.h"
#include "../src/llama-ext.h"
#include "../src/llama-memory-hybrid-idx.h"
#include "../src/llama-model-saver.h"
#include "../src/llama-model.h"

#include <array>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <locale>
#include <limits>
#include <random>
#include <regex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>
#if !defined(_WIN32)
#    include <sys/wait.h>
#endif

// These fixtures also test the legacy decode API and malformed batch certificates.
static void batch_add_compat(llama_batch & batch, llama_token id, llama_pos pos,
        const std::vector<llama_seq_id> & seq_ids, bool output) {
    GGML_ASSERT(batch.seq_id[batch.n_tokens]);
    batch.token[batch.n_tokens] = id;
    batch.pos[batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits[batch.n_tokens++] = output;
}

static bool arch_matches(const std::string & filter, llm_arch arch) {
    if (filter.empty()) {
        return true;
    }
    return std::regex_search(llm_arch_name(arch), std::regex(filter));
}

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

struct tensor_data_params {
    size_t seed;
    float  stdev;
};

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    const tensor_data_params & params = *(const tensor_data_params *) userdata;
    size_t seed = params.seed;
    std::hash<std::string> hasher;
    seed ^= hasher(tensor->name);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(0.0f, params.stdev);

    // TODO: refactor per-tensor initialization logic in a cleaner way

    // note: Mamba A must be negative (state decay)
    const bool is_ssm_a = strstr(tensor->name, "ssm_a") != nullptr;
    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            float val = dis(gen);
            tmp[i] = is_ssm_a ? -fabsf(val) : val;
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            float val = dis(gen);
            tmp[i] = ggml_fp32_to_fp16(is_ssm_a ? -fabsf(val) : val);
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal error");
    }
}

static void usage(char ** argv) {
    LOG("Usage: %s [options]\n\n", argv[0]);
    LOG("Options:\n");
    LOG("  -a, --arch <arch|regex>  Run only matching LLM architectures (default: all supported)\n");
    LOG("  -s, --seed <seed>        Set the random seed for tensor initialization and token generation\n");
    LOG("  -d, --stdev <stdev>      Set the standard deviation of the tensor initialization distribution (default: 0.1f)\n");
    LOG("  -o, --out <dir>          Save generated test models to <dir> instead of running backend tests\n");
    LOG("  -v <N>                   Set log verbosity level\n");
    LOG("  --test-phase-workspace   Run phase-aware workspace checks\n");
    LOG("  --test-live-context-workspace Run live-context workspace checks\n");
    LOG("  --test-speculative-limits Run speculative limit checks\n");
    LOG("  --test-mtp-draft-vocab   Run opt-in draft vocabulary checks\n");
    LOG("  --test-moe-cache-selector Run MoE cache selector checks\n");
    LOG("  --test-moe-placement    Run MoE placement checks\n");
    LOG("  --test-source-uses <model> Check loader source discovery with and without allocation\n");
    LOG("  --test-source-profile <model> Check GPU profile initialization of original routed sources\n");
    LOG("  --test-source-statistics Check source statistics identity, coverage and malformed inputs\n");
    LOG("  --collect-moe-profile <model> Collect a weighted native corpus profile (natural EOG)\n");
    LOG("  --profile-corpus <json>  Workload weights and calibration requests\n");
    LOG("  --profile-output <gguf>  New profile path; evidence goes in <gguf>.calibration\n");
    LOG("  --profile-threads <N>    Calibration CPU threads (default 4)\n");
    LOG("  --moe-hybrid on|off     Generic CPU/GPU execution (corpus collection defaults on)\n");
    LOG("  --moe-gpu-miss-fraction <F> GPU transfer share of distinct misses (default 0.17)\n");
    LOG("  --source-statistics-file <file> Write synthetic source statistics for the profile loading fixture\n");
    LOG("  --test-moe-replay <model> Record or replay identical-token full-logit rows\n");
    LOG("  --test-source-variants Check sequence-aware source captures against --replay-reference\n");
    LOG("  --test-source-auxiliary Check actual DRAFT/MTP shared source frontend against --replay-reference\n");
    LOG("  --auxiliary-uncached-reference Use ordinary uncached arithmetic for the auxiliary numerical control\n");
    LOG("  --auxiliary-no-output-first Start auxiliary execution without outputs, then grow the live graph\n");
    LOG("  --auxiliary-source-statistics Install supplied source statistics for actual auxiliary execution\n");
    LOG("  --auxiliary-source-ranks Install supplied source ranks for actual auxiliary execution\n");
    LOG("  --auxiliary-inherit-statistics Inherit parent statistics before replacing the parent configuration\n");
    LOG("  --auxiliary-different-model Use a distinct auxiliary model object with matching geometry\n");
    LOG("  --auxiliary-profile-file <path> Generate full profile files for automatic parent/child binding\n");
    LOG("  --auxiliary-explicit-profile-file <path> Bind opposite parent/child profiles through explicit context creation\n");
    LOG("  --auxiliary-sparse-outputs Request only the final output of a multi-sequence frame\n");
    LOG("  --auxiliary-parent-active Reject profile snapshots while the parent source call guard is active\n");
    LOG("  --replay-reference <file> Logit reference file; writes unless --replay-read is set\n");
    LOG("  --replay-read            Read reference tokens and report full-logit differences\n");
    LOG("  --replay-prompt-file <file> Bounded prompt text, up to64KiB/4096tokens\n");
    LOG("  --replay-rows <N>        Full-logit continuation rows, 1..4096 (default:32)\n");
    LOG("  --replay-split <name>    Provenance: diagnostic, calibration, development or held-out\n");
    LOG("  --replay-independent-sources Use routed operands without source fusion hints\n");
    LOG("  --replay-cache-slots <N> MoE replay residency capacity; 0 disables cache (default:32)\n");
    LOG("  --replay-cpu-oracle     Compare hybrid routed CPU outputs with ordinary CPU operators\n");
    LOG("  --replay-gpu-oracle     Observe sampled routed outputs with ordinary GPU operators\n");
    LOG("  --replay-load-mode <mode> Replay loading mode, none or mmap (default:none)\n");
    LOG("  --test-moe-joint-measurement Run MoE joint measurement checks\n");
    LOG("  --fit-tool <path>        Fit tool for joint measurement checks\n");
    LOG("  -b, --backend <backend>  Run only on the given backend device\n");
    LOG("  -h, --help               Show this help message\n\n");
    LOG("Examples:\n");
    LOG("  %s\n", argv[0]);
    LOG("  %s -a qwen35moe\n", argv[0]);
    LOG("  %s -a deepseek4 -o tests/test-models/\n", argv[0]);
    LOG("  %s -a cohere2moe -v 5\n", argv[0]);
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

// MoE archs that are also tested with the experts in host memory
static bool host_experts_test(const llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_DEEPSEEK2:
            return true;
        default:
            return false;
    }
}

static gguf_context_ptr get_gguf_ctx(
        const llm_arch arch, const bool moe, const bool mtp = false,
        uint32_t n_expert = 0, uint32_t n_expert_used = 2, int32_t n_layer_base = -1,
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
            || arch == LLM_ARCH_GLM5_NEXT
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
        n_layer = 6; // 1 layer per stack x 2 h-cycles x (2 l-cycles + 1) cache slots
    }

    GGML_ASSERT(!mtp || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE || arch == LLM_ARCH_QWEN4EXP);
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

    if (arch == LLM_ARCH_K2_HORIZON) {
        ms.add_kv(LLM_KV_ROPE_SCALING_YARN_BETA_FAST, 128.0f);
        ms.add_kv(LLM_KV_ROPE_SCALING_YARN_BETA_SLOW,   4.0f);
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
            arch == LLM_ARCH_BAILINGMOE3 || arch == LLM_ARCH_KIMI_K3 || arch == LLM_ARCH_GLM5_NEXT) {
        GGML_ASSERT(n_layer >= 2);
        std::vector<uint32_t> n_head_per_layer;
        n_head_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_head_per_layer.push_back(il == 1 ? 0 : n_head);
        }
        // GLM5 next KDA heads come from the uniform head count, only head_count_kv is per layer.
        if (arch == LLM_ARCH_GLM5_NEXT) {
            ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
        } else {
            ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head_per_layer);
        }
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
            || arch == LLM_ARCH_GLM5_NEXT
            || arch == LLM_ARCH_HY_V4
            || arch == LLM_ARCH_MISTRAL4) {
        // GLM5 next MLA is nope only, the cache row is the compressed latent alone.
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,       arch == LLM_ARCH_GLM5_NEXT ? uint32_t(512) : uint32_t(576));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,     uint32_t(512));
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       arch == LLM_ARCH_GLM5_NEXT ? uint32_t(0) : uint32_t(64));
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
        std::vector<uint32_t> pattern;
        pattern.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            pattern.push_back((il + 1) % 5 != 0);
        }
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, pattern);
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
    if (arch == LLM_ARCH_QWEN4EXP || arch == LLM_ARCH_GLM5_NEXT) {
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,    uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,  1.0e-6f);
        ms.add_kv(LLM_KV_HYPER_CONNECTION_LOW_RANK, uint32_t(8));
        // without this the QSA layers fall back to dense and go uncovered
        std::vector<uint32_t> compress_ratios(n_layer_all, 4);
        if (mtp) {
            compress_ratios.back() = 0;
        }
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS, compress_ratios);

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

    // note: using a realistic top-k here makes the results unstable and hard to match between CPU and GPU
    //       a large value makes things deterministic since all data is selected by the indexer
    //ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,        uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,        uint32_t(131072));

    ms.add_kv(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE,   uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KPOOL,        uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KPOOL_SELECT_TAIL, true);
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS, uint32_t(1));
    // mrope sections count rope pairs; Ling 3.0 VL files carry [t, h, w] sections
    // summing to n_rot / 2 (n_rot is 64 in this fixture)
    if (arch == LLM_ARCH_BAILINGMOE3) {
        ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({8, 12, 12, 0}));
    } else {
        ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));
    }

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
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,          uint32_t(8));
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,            uint32_t(32));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS,             std::vector<uint32_t>({0, 0, 4, 128}));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE,     160000.0f);
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,                uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS,  uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,              1.0e-6f);
        ms.add_kv(LLM_KV_HASH_LAYER_COUNT,                      uint32_t(0));
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,                      10.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,                  1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,                   true);
    }

    if (arch == LLM_ARCH_HRM_TEXT) {
        // 6 cache slots alias 2 physical blocks: 1 low-stack layer + 1 high-stack layer
        ms.add_kv(LLM_KV_HRM_LAYERS_PER_STACK, uint32_t(1));
        ms.add_kv(LLM_KV_HRM_H_CYCLES,         uint32_t(2));
        ms.add_kv(LLM_KV_HRM_L_CYCLES,         uint32_t(2));
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
        if (n_expert == 0) { n_expert = host_experts_test(arch) ? 64 : 2; }
        ms.add_kv(LLM_KV_EXPERT_COUNT,               n_expert);
        ms.add_kv(LLM_KV_EXPERT_USED_COUNT,          n_expert_used);
        ms.add_kv(LLM_KV_EXPERT_SHARED_COUNT,        uint32_t(1));
        ms.add_kv(LLM_KV_EXPERT_GATING_FUNC,         arch == LLM_ARCH_DEEPSEEK4 ? uint32_t(4) : uint32_t(2)); // sqrtsoftplus : sigmoid
        ms.add_kv(LLM_KV_EXPERT_GROUP_SCALE,         1.0f);
        ms.add_kv(LLM_KV_EXPERTS_PER_GROUP,          uint32_t(1));
        if (arch == LLM_ARCH_K2_HORIZON) {
            ms.add_kv(LLM_KV_ATTENTION_VALUE_EXPERT_COUNT,      uint32_t(2));
            ms.add_kv(LLM_KV_ATTENTION_VALUE_EXPERT_USED_COUNT, uint32_t(2));
        }
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
    ms.add_kv(LLM_KV_KDA_SAFE_GATE,             true);
    ms.add_kv(LLM_KV_KDA_GATE_LOWER_BOUND,      -5.0f);
    if (arch == LLM_ARCH_BAILINGMOE3) {
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,   std::vector<float>({0.0f, 4.0f}));
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_SHEXP, std::vector<float>({0.0f, 5.0f}));
    }
    ms.add_kv(LLM_KV_WKV_HEAD_SIZE,               n_embd/n_head);
    ms.add_kv(LLM_KV_SHORTCONV_L_CACHE,           uint32_t(3));
    ms.add_kv(LLM_KV_RESIDUAL_SCALE,              3.5565588200778455f);
    ms.add_kv(LLM_KV_ATTN_RES_BLOCK_SIZE,         uint32_t(12));
    ms.add_kv(LLM_KV_ACTIVATION_SITU_BETA,        4.0f);
    ms.add_kv(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, 25.0f);
    ms.add_kv(LLM_KV_KDA_GATE_LOWER_BOUND,        -5.0f);

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
        struct gguf_context * gguf_ctx, FILE * file, const size_t seed, const float stdev,
        const std::vector<ggml_backend_dev_t> & devs,
        const llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER, bool encode = false,
        const llama_model_tensor_buft_override * tensor_buft_overrides = nullptr, const size_t moe_cache_size = 0) {
    GGML_ASSERT((gguf_ctx == nullptr) != (file == nullptr));
    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    std::vector<ggml_backend_dev_t> devs_copy = devs;
    devs_copy.push_back(nullptr);
    model_params.devices = devs_copy.data();
    model_params.split_mode = split_mode;
    model_params.tensor_buft_overrides = tensor_buft_overrides;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 0;
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;
    if (!encode) {
        ctx_params.n_ubatch = 64;
    }
    if (moe_cache_size > 0) {
        // the MoE cache is only used for small ubatches
        ctx_params.moe_cache_size = moe_cache_size;
        ctx_params.n_ubatch = 2;
    }

    tensor_data_params tensor_params = { seed, stdev };
    llama_model_ptr model(gguf_ctx != nullptr ?
        llama_model_init_from_user(gguf_ctx, set_tensor_data, &tensor_params, model_params) :
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

static int test_model_source_uses(const char * path) {
    for (bool no_alloc : {true, false}) {
        auto params = llama_model_default_params();
        params.n_gpu_layers = 0;
        params.no_alloc = no_alloc;
        params.progress_callback = silent_model_load_progress;
        llama_model_ptr model(llama_model_load_from_file(path, params));
        if (!model) { return 1; }
        std::unordered_set<const ggml_tensor *> tensors;
        for (const auto & entry : model->tensors_by_name) { tensors.insert(entry.second); }
        std::unordered_set<const ggml_tensor *> recorded;
        std::map<const ggml_tensor *, std::vector<std::string>> source_names;
        size_t routed = 0;
        for (const auto & use : model->tensor_uses()) {
            if (!use.tensor || !tensors.count(use.tensor) || use.op < GGML_OP_NONE || use.op >= GGML_OP_COUNT || use.layer < -1 || use.name.empty()) {
                fprintf(stderr, "test-model-source-uses: invalid returned tensor/op/layer\n");
                return 1;
            }
            recorded.insert(use.tensor);
            source_names[use.tensor].push_back(use.name);
            routed += use.op == GGML_OP_MUL_MAT_ID;
        }
        for (auto & entry : source_names) {
            auto & names = entry.second;
            std::sort(names.begin(), names.end());
            names.erase(std::unique(names.begin(), names.end()), names.end());
        }
        if (recorded != tensors || routed == 0) {
            fprintf(stderr, "test-model-source-uses: coverage recorded=%zu tensors=%zu routed=%zu\n", recorded.size(), tensors.size(), routed);
            return 1;
        }
        for (const auto & group : model->moe_sources()) {
            for (const auto & bank : group.banks) {
                if (!recorded.count(bank.tensor) || bank.names != source_names.at(bank.tensor)) {
                    fprintf(stderr, "test-model-source-uses: source group tensor missing\n");
                    return 1;
                }
            }
        }
        const auto source_groups = model->moe_sources().size();
        std::map<ggml_tensor *, std::string> display_names;
        for (const auto & use : model->tensor_uses()) {
            if (use.op == GGML_OP_MUL_MAT_ID && !display_names.count(use.tensor)) {
                display_names.emplace(use.tensor, use.tensor->name);
                ggml_set_name(use.tensor, "shared-truncated-display-name");
            }
        }
        auto layers = std::move(model->layers);
        model->layers.clear();
        model->build_moe_sources();
        std::unordered_set<const ggml_tensor *> routed_tensors;
        for (const auto & use : model->tensor_uses()) {
            if (use.op == GGML_OP_MUL_MAT_ID) { routed_tensors.insert(use.tensor); }
        }
        std::unordered_set<const ggml_tensor *> catalog;
        for (const auto & group : model->moe_sources()) {
            if (group.layout != GGML_BACKEND_MOE_CANDIDATE_LAYOUT_ROUTED_MATRIX || group.banks.size() != 1 ||
                    group.banks[0].role != GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_ROUTED_WEIGHT ||
                    group.banks[0].status != GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE ||
                    group.banks[0].names != source_names.at(group.banks[0].tensor) ||
                    !catalog.insert(group.banks[0].tensor).second) {
                fprintf(stderr, "test-model-source-uses: invalid operation source catalog\n");
                return 1;
            }
        }
        if (catalog != routed_tensors) {
            fprintf(stderr, "test-model-source-uses: routed catalog does not cover returned operations\n");
            return 1;
        }
        for (const auto & entry : display_names) { ggml_set_name(entry.first, entry.second.c_str()); }
        model->layers = std::move(layers);
        model->build_moe_sources();
        if (model->moe_sources().size() != source_groups) { return 1; }
        fprintf(stderr, "test-model-source-uses: no_alloc=%d uses=%zu tensors=%zu routed=%zu exact returned-storage coverage OK\n",
            no_alloc, model->tensor_uses().size(), tensors.size(), routed);
        fprintf(stderr, "test-model-source-uses: no_alloc=%d independent routed sources=%zu without layer bank hints OK\n",
            no_alloc, catalog.size());
        fprintf(stderr, "test-model-source-uses: no_alloc=%d full loader identities survive colliding display names OK\n", no_alloc);
    }
    return 0;
}

static gguf_context_ptr make_source_profile_statistics(const std::vector<llama_moe_source_group> & sources, bool reverse = false, const llama_moe_profile_statistics * measured = nullptr) {
    llama_moe_profile_statistics synthetic;
    if (!measured) {
        synthetic.provenance = "Synthetic fixture occurrences; no calibrated quality or content identity claim";
        std::map<uint32_t, std::unordered_set<const ggml_tensor *>> seen;
        for (const auto & source : sources) {
            for (const auto & bank : source.banks) {
                if (bank.status != GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE || !seen[source.domain].insert(bank.tensor).second) { continue; }
                if (!bank.tensor || bank.tensor->ne[2] <= 0 || bank.tensor->ne[2] > (1u << 22)) { throw std::runtime_error("invalid statistics fixture geometry"); }
                llama_moe_profile_source_statistics item;
                item.tensor = bank.tensor; item.domain = source.domain;
                for (int64_t i = 0; i < bank.tensor->ne[2]; ++i) {
                    const uint64_t value = uint64_t(reverse ? bank.tensor->ne[2] - i : i + 1);
                    item.counts.push_back(value); item.observations += value;
                }
                synthetic.sources.push_back(std::move(item));
            }
        }
        measured = &synthetic;
    }
    const auto bytes = llama_moe_profile_statistics_serialize(*measured, sources);
    gguf_context_ptr metadata(gguf_init_from_buffer(bytes.data(), bytes.size(), {true, nullptr}));
    if (!metadata) { throw std::runtime_error("invalid serialized profile fixture"); }
    return metadata;
}

static std::vector<uint8_t> source_profile_metadata_bytes(const gguf_context * metadata) {
    std::vector<uint8_t> result(gguf_get_meta_size(metadata));
    gguf_get_meta_data(metadata, result.data());
    return result;
}

struct routed_profile_collector {
    llama_moe_profile_statistics statistics;
    std::map<std::pair<uint32_t, const ggml_tensor *>, size_t> source_indices;
    std::array<uint64_t, 3> ownership_counts = {};
    uint64_t projections = 0;
    uint64_t first_epoch = 0;
    uint64_t last_epoch = 0;

    bool initialize(const std::vector<llama_moe_source_group> & sources) {
        statistics = {}; source_indices.clear(); ownership_counts = {}; projections = first_epoch = last_epoch = 0;
        size_t entries = 0;
        for (const auto & source : sources) {
            for (const auto & bank : source.banks) {
                if (bank.status != GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE) { continue; }
                const auto identity = std::make_pair(source.domain, static_cast<const ggml_tensor *>(bank.tensor));
                const auto found = source_indices.find(identity);
                if (found != source_indices.end()) {
                    continue;
                }
                if (!bank.tensor || bank.names.empty() || bank.tensor->ne[2] <= 0 || uint64_t(bank.tensor->ne[2]) > (1u << 22) - entries ||
                        statistics.sources.size() >= GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS * GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS) { return false; }
                entries += size_t(bank.tensor->ne[2]);
                source_indices.emplace(identity, statistics.sources.size());
                statistics.sources.push_back({bank.tensor, source.domain, 0, std::vector<uint64_t>(bank.tensor->ne[2], 0)});
            }
        }
        return !statistics.sources.empty();
    }

    bool observe(uint64_t epoch, const ggml_backend_moe_source_access_v1 * probe) {
        if (!probe || !probe->source_witness || !probe->source_domain || !probe->source || !probe->route_indices || !probe->expert_ids || !probe->classes ||
                !probe->n_routes || probe->n_routes > (1u << 22) || !probe->n_distinct || probe->n_distinct > probe->n_routes ||
                !epoch || projections == UINT64_MAX) { return false; }
        const auto found = source_indices.find(std::make_pair(probe->source_domain, probe->source_witness));
        if (found == source_indices.end()) { return false; }
        auto & source = statistics.sources[found->second];
        if (probe->source->type != source.tensor->type || std::memcmp(probe->source->ne, source.tensor->ne, sizeof(source.tensor->ne)) ||
                std::memcmp(probe->source->nb, source.tensor->nb, sizeof(source.tensor->nb)) ||
                probe->n_distinct > source.counts.size() || probe->n_routes > UINT64_MAX - source.observations) { return false; }
        std::array<uint64_t, 3> classes = {};
        for (uint32_t i = 0; i < probe->n_routes; ++i) {
            const auto index = probe->route_indices[i];
            if (index >= probe->n_distinct || probe->expert_ids[index] < 0 || size_t(probe->expert_ids[index]) >= source.counts.size() ||
                    probe->classes[index] < 0 || probe->classes[index] > 2) { return false; }
            ++classes[probe->classes[index]];
        }
        for (size_t i = 0; i < classes.size(); ++i) { if (classes[i] > UINT64_MAX - ownership_counts[i]) { return false; } }
        for (uint32_t i = 0; i < probe->n_routes; ++i) { ++source.counts[probe->expert_ids[probe->route_indices[i]]]; }
        for (size_t i = 0; i < classes.size(); ++i) { ownership_counts[i] += classes[i]; }
        source.observations += probe->n_routes;
        if (!projections) { first_epoch = epoch; }
        last_epoch = epoch; ++projections;
        return true;
    }
};

// Reduce rounding error when adding requests with different lengths.
struct profile_rate_sum {
    double sum = 0, correction = 0;
    void add(double value) {
        const double next = sum + value;
        correction += std::fabs(sum) >= std::fabs(value) ? (sum - next) + value : (value - next) + sum;
        sum = next;
    }
    double value() const { return sum + correction; }
};

struct profile_corpus_family {
    double weight = 0;
    uint32_t requests = 0;
    std::vector<std::vector<profile_rate_sum>> rates;
};

struct profile_corpus_average {
    llama_moe_profile_statistics statistics;
    std::map<std::string, profile_corpus_family> families;

    void initialize(const llama_moe_profile_statistics & sources, const std::map<std::string, double> & weights) {
        size_t entries = 0;
        for (const auto & source : sources.sources) { entries += source.counts.size(); }
        if (weights.empty() || entries > (64 * 1024 * 1024 / sizeof(profile_rate_sum)) / weights.size()) {
            throw std::runtime_error("corpus rate storage exceeds 64 MiB");
        }
        statistics = sources;
        families.clear();
        profile_rate_sum total;
        for (const auto & weight : weights) {
            if (!std::isfinite(weight.second) || weight.second < 0) { throw std::runtime_error("invalid workload weight"); }
            total.add(weight.second);
        }
        if (!std::isfinite(total.value()) || total.value() <= 0) { throw std::runtime_error("empty workload weight"); }
        for (const auto & weight : weights) {
            auto & family = families[weight.first];
            family.weight = weight.second / total.value();
            for (const auto & source : sources.sources) { family.rates.emplace_back(source.counts.size()); }
        }
    }

    void add(const std::string & name, const llama_moe_profile_statistics & request, uint32_t rows) {
        const auto found = families.find(name);
        if (found == families.end() || !rows || request.sources.size() != statistics.sources.size()) { throw std::runtime_error("invalid corpus request"); }
        auto & family = found->second;
        for (size_t i = 0; i < request.sources.size(); ++i) {
            const auto & source = request.sources[i];
            auto & raw = statistics.sources[i];
            if (source.tensor != raw.tensor || source.domain != raw.domain || source.counts.size() != raw.counts.size() ||
                    source.observations > UINT64_MAX - raw.observations) { throw std::runtime_error("corpus source mismatch or overflow"); }
            for (size_t j = 0; j < source.counts.size(); ++j) {
                if (source.counts[j] > UINT64_MAX - raw.counts[j]) { throw std::runtime_error("corpus occurrence overflow"); }
                raw.counts[j] += source.counts[j];
                family.rates[i][j].add(double(source.counts[j]) / rows);
            }
            raw.observations += source.observations;
        }
        ++family.requests;
    }

    void finish() {
        for (const auto & family : families) { if (!family.second.requests) { throw std::runtime_error("workload has no measured requests"); } }
        for (size_t i = 0; i < statistics.sources.size(); ++i) {
            auto & source = statistics.sources[i];
            source.scores.resize(source.counts.size());
            for (size_t j = 0; j < source.counts.size(); ++j) {
                profile_rate_sum score;
                for (const auto & family : families) {
                    score.add((family.second.rates[i][j].value() / family.second.requests) * family.second.weight);
                }
                source.scores[j] = score.value();
            }
        }
    }
};

static int test_source_profile_learning() {
    uint32_t checks = 0;
    for (const uint32_t experts : {1u, 3u, 17u, 257u, 65536u, 65537u}) {
        for (const uint32_t seed_count : {1u, experts}) {
            std::vector<int32_t> seed;
            for (uint32_t i = 0; i < seed_count; ++i) { seed.push_back(int32_t(experts - i - 1)); }
            ggml_moe_source_profile_learning learning;
            if (!ggml_moe_source_profile_initialize(learning, seed.data(), seed.size(), experts)) { return 1; }
            std::vector<uint64_t> counts, expected_counts(experts, 0);
            std::vector<double> heat, expected_heat(experts, 0);
            std::vector<float> usage(experts, 0);
            std::vector<int32_t> ranks;
            uint64_t observations = 0;
            if (!ggml_moe_source_profile_snapshot(learning, counts, heat, ranks, observations) || observations || counts != expected_counts || heat != expected_heat || ranks.front() != seed.front() || ranks.size() != experts) { return 1; }
            for (uint32_t window = 1; window <= 17; ++window) {
                std::vector<int32_t> routes;
                for (uint32_t i = 0; i < 23; ++i) { routes.push_back(int32_t((i * 7 + window * 13) % experts)); }
                if (!ggml_moe_source_profile_observe(learning, routes.data(), routes.size())) { return 1; }
                for (const int32_t id : routes) { ++expected_counts[id]; usage[id] += 1; }
                if (window % 4 == 0) {
                    if (!ggml_moe_source_profile_accumulate(learning, usage)) { return 1; }
                    for (size_t i = 0; i < usage.size(); ++i) { expected_heat[i] += double(usage[i]); usage[i] *= 0.7f; }
                }
                if (!ggml_moe_source_profile_snapshot(learning, counts, heat, ranks, observations) || observations != uint64_t(window) * routes.size() || counts != expected_counts || heat != expected_heat || ranks.size() != experts) { return 1; }
                std::vector<int32_t> expected_ranks = seed;
                for (uint32_t i = 0; i < experts - seed_count; ++i) { expected_ranks.push_back(int32_t(i)); }
                std::stable_sort(expected_ranks.begin(), expected_ranks.end(), [&](int32_t a, int32_t b) { return expected_heat[a] > expected_heat[b]; });
                if (ranks != expected_ranks) { return 1; }
                const auto reject_unchanged = [&] {
                    std::vector<uint64_t> current_counts;
                    std::vector<double> current_heat;
                    std::vector<int32_t> current_ranks;
                    uint64_t current_observations = 0;
                    return ggml_moe_source_profile_snapshot(learning, current_counts, current_heat, current_ranks, current_observations) &&
                        counts == current_counts && heat == current_heat && ranks == current_ranks && observations == current_observations;
                };
                routes.back() = int32_t(experts);
                if (ggml_moe_source_profile_observe(learning, routes.data(), routes.size()) || !reject_unchanged()) { return 1; }
                routes.back() = -1;
                if (ggml_moe_source_profile_observe(learning, routes.data(), routes.size()) || !reject_unchanged()) { return 1; }
                auto invalid = usage;
                for (const float value : {-1.0f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
                    invalid.back() = value;
                    if (ggml_moe_source_profile_accumulate(learning, invalid) || !reject_unchanged()) { return 1; }
                    ++checks;
                }
                const int32_t duplicate[] = {0, 0};
                if (ggml_moe_source_profile_initialize(learning, duplicate, 2, experts) || !reject_unchanged() ||
                        ggml_moe_source_profile_initialize(learning, nullptr, 1, experts) || !reject_unchanged() ||
                        ggml_moe_source_profile_observe(learning, nullptr, 1) || !reject_unchanged() ||
                        ggml_moe_source_profile_accumulate(learning, {}) || !reject_unchanged()) { return 1; }
                ++checks;
            }
            // Repeated snapshots must not count or fold the tail again.
            if (!ggml_moe_source_profile_snapshot(learning, counts, heat, ranks, observations) || counts != expected_counts || heat != expected_heat) { return 1; }
        }
    }
    ggml_moe_source_profile_learning empty;
    std::vector<uint64_t> counts{9};
    std::vector<double> heat{8};
    std::vector<int32_t> ranks{7};
    uint64_t observations = 6;
    const int32_t id = 0;
    if (ggml_moe_source_profile_snapshot(empty, counts, heat, ranks, observations) || counts != std::vector<uint64_t>{9} || heat != std::vector<double>{8} || ranks != std::vector<int32_t>{7} || observations != 6 ||
            ggml_moe_source_profile_initialize(empty, &id, 1, 0) || ggml_moe_source_profile_initialize(empty, &id, 1, (1u << 22) + 1) || ggml_moe_source_profile_observe(empty, &id, 1)) { return 1; }
    fprintf(stderr, "test-source-profile-learning: full coverage, raw counts, decayed heat, prior ties, repeated snapshots and%u rejection/state checks OK\n", checks);
    return 0;
}

static int test_source_profile_statistics() {
    if (test_source_profile_learning()) { return 1; }
    {
        std::vector<ggml_moe_profile_capacity_group> groups{{10, 1, {9, 8, 7, 6}}, {20, 1, {5, 4, 3, 2}}};
        std::vector<uint32_t> capacities;
        uint64_t paid = 0;
        if (!ggml_moe_profile_plan_capacities(groups, 60, capacities, paid) || capacities != std::vector<uint32_t>{4, 1} || paid != 60) { return 1; }
        for (uint64_t budget = 30; budget <= 130; ++budget) {
            if (!ggml_moe_profile_plan_capacities(groups, budget, capacities, paid) || capacities.size() != groups.size() ||
                    capacities[0] < 1 || capacities[0] > 4 || capacities[1] < 1 || capacities[1] > 4 ||
                    paid != capacities[0] * 10 + capacities[1] * 20 || paid > budget) { return 1; }
        }
        const auto unchanged = capacities;
        const auto unchanged_paid = paid;
        for (uint32_t variant = 0; variant < 7; ++variant) {
            auto invalid = groups;
            uint64_t budget = 60;
            if (variant == 0) { invalid[0].per_slot_bytes = 0; }
            if (variant == 1) { invalid[0].minimum_slots = 0; }
            if (variant == 2) { invalid[0].minimum_slots = 5; }
            if (variant == 3) { invalid[0].priorities[1] = 10; }
            if (variant == 4) { invalid[0].priorities[0] = std::numeric_limits<long double>::infinity(); }
            if (variant == 5) { invalid[0].per_slot_bytes = UINT64_MAX; }
            if (variant == 6) { budget = 29; }
            if (ggml_moe_profile_plan_capacities(invalid, budget, capacities, paid) || capacities != unchanged || paid != unchanged_paid) { return 1; }
        }
    }
    const uint64_t small[] = {9, 1, 0}, scaled[] = {900, 100, 0}, other[] = {0, 10, 0}, zero[] = {0, 0, 0};
    std::vector<ggml_moe_profile_bank_statistics> banks{{small, 10, 9, 3}, {other, 10, 1, 3}};
    std::vector<int32_t> ranks;
    if (!ggml_moe_source_rank_statistics(banks, ranks) || ranks != std::vector<int32_t>{0, 1, 2}) { return 1; }
    banks[0].counts = scaled; banks[0].observations = 1000;
    if (!ggml_moe_source_rank_statistics(banks, ranks) || ranks != std::vector<int32_t>{0, 1, 2}) { return 1; }
    banks[0].payload_bytes = 1; banks[1].payload_bytes = 9;
    if (!ggml_moe_source_rank_statistics(banks, ranks) || ranks != std::vector<int32_t>{1, 0, 2}) { return 1; }
    for (auto & bank : banks) { bank.counts = zero; bank.observations = 0; }
    if (!ggml_moe_source_rank_statistics(banks, ranks) || ranks != std::vector<int32_t>{0, 1, 2}) { return 1; }
    const auto unchanged = ranks;
    const auto reject_ranking = [&](const std::vector<ggml_moe_profile_bank_statistics> & input) {
        if (ggml_moe_source_rank_statistics(input, ranks) || ranks != unchanged) { throw std::runtime_error("invalid canonical statistics changed ranking"); }
    };
    reject_ranking({});
    for (uint32_t variant = 0; variant < 8; ++variant) {
        auto invalid = banks;
        if (variant == 0) { invalid[0].counts = nullptr; }
        if (variant == 1) { invalid[0].observations = 1; }
        if (variant == 2) { invalid[0].payload_bytes = 0; }
        if (variant == 3) { invalid[0].payload_bytes = UINT64_MAX; }
        if (variant == 4) { invalid[0].n_experts = 2; }
        if (variant == 5) { invalid[0].n_experts = (1u << 22) + 1; }
        if (variant == 6) { invalid.resize(GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS + 1); }
        const uint64_t overflow[] = {UINT64_MAX, 1, 0};
        if (variant == 7) { invalid[0].counts = overflow; invalid[0].observations = UINT64_MAX; }
        reject_ranking(invalid);
    }
    fprintf(stderr, "test-source-statistics-ranking: normalized calibration,canonical bank byte costs,zero observations,stable ties,full ranks and invalid-input atomicity OK\n");
    const ggml_init_params params{ggml_tensor_overhead() * 4, nullptr, true};
    ggml_context_ptr context(ggml_init(params));
    auto * a = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, 8, 7, 3);
    auto * b = ggml_new_tensor_3d(context.get(), GGML_TYPE_F16, 8, 6, 5);
    ggml_set_name(a, "same-display-name"); ggml_set_name(b, "same-display-name");
    const std::string long_name(200, 'x');
    std::vector<llama_moe_source_group> sources{
        {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_ROUTED_MATRIX, 1, true,
            {{a, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_ROUTED_WEIGHT, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE, {long_name, "alias-a"}}}, 4},
        {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_ROUTED_MATRIX, 2, true,
            {{b, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_ROUTED_WEIGHT, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE, {"source-b"}}}, 4}};
    auto metadata = make_source_profile_statistics(sources);
    const auto bytes = source_profile_metadata_bytes(metadata.get());
    const auto bound = llama_moe_profile_statistics_parse(bytes.data(), bytes.size(), sources);
    if (bound.sources.size() != 2 || bound.sources[0].tensor != a || bound.sources[1].tensor != b ||
            bound.sources[0].counts != std::vector<uint64_t>{1, 2, 3} || bound.sources[1].counts != std::vector<uint64_t>{1, 2, 3, 4, 5} ||
            bound.sources[0].observations != 6 || bound.sources[1].observations != 15 || bound.provenance.empty()) { return 1; }
    auto learned = bound;
    learned.provenance = "Learned fixture: joined target windows, raw occurrences and pre-decay heat; no quality claim";
    for (auto & source : learned.sources) {
        source.heat.assign(source.counts.size(), 1);
        source.usage.resize(source.counts.size());
        source.prior.resize(source.counts.size());
        source.windows = 1;
        source.scores.resize(source.counts.size());
        for (size_t i = 0; i < source.counts.size(); ++i) {
            source.usage[i] = float(source.counts[i]) * 0.3f;
            source.prior[i] = int32_t(source.counts.size() - i - 1);
            source.scores[i] = double(i + 1) / source.counts.size();
        }
    }
    const auto learned_bytes = llama_moe_profile_statistics_serialize(learned, sources);
    std::vector<ggml_backend_moe_source_learning_v1> learning_views;
    for (const auto & source : learned.sources) {
        learning_views.push_back({sizeof(ggml_backend_moe_source_learning_v1), 1,
            {source.tensor, source.counts.data(), source.observations, uint32_t(source.counts.size()), source.domain},
            source.heat.data(), source.usage.data(), source.prior.data(), source.windows});
    }
    if (!ggml_moe_source_learning_valid(learning_views.data(), uint32_t(learning_views.size())) ||
            ggml_moe_source_learning_valid(nullptr, 1) || ggml_moe_source_learning_valid(learning_views.data(), 0) ||
            ggml_moe_source_learning_valid(learning_views.data(), GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS * GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS + 1)) { return 1; }
    for (uint32_t variant = 0; variant < 12; ++variant) {
        auto invalid_views = learning_views;
        auto & first = invalid_views.front();
        if (variant == 0) { --first.struct_size; }
        if (variant == 1) { ++first.abi_version; }
        if (variant == 2) { first.heat = nullptr; }
        if (variant == 3) { first.usage = nullptr; }
        if (variant == 4) { first.prior = nullptr; }
        if (variant == 5) { first.source.counts = nullptr; }
        if (variant == 6) { first.source.tensor = nullptr; }
        if (variant == 7) { first.source.n_experts = (1u << 22) + 1; }
        if (variant == 8) { first.windows = 0; }
        if (variant == 9) { first.windows = first.source.observations + 1; }
        if (variant == 10) { ++first.source.observations; }
        if (variant == 11) { invalid_views.push_back(first); }
        if (ggml_moe_source_learning_valid(invalid_views.data(), uint32_t(invalid_views.size()))) { return 1; }
    }
    fprintf(stderr, "test-source-learning-contract: descriptor-bound views, ABI bounds, null/count/cadence/identity rejection OK\n");
    const auto loaded = llama_moe_profile_statistics_parse(learned_bytes.data(), learned_bytes.size(), sources);
    if (llama_moe_profile_statistics_serialize(loaded, sources) != learned_bytes) { return 1; }
    for (size_t index = 0; index < learned.sources.size(); ++index) {
        const auto & original = learned.sources[index];
        const auto & saved = loaded.sources[index];
        if (saved.counts != original.counts || saved.heat != original.heat || saved.usage != original.usage ||
                saved.prior != original.prior || saved.windows != original.windows || saved.scores != original.scores) { return 1; }
        ggml_moe_source_profile_learning before, after;
        if (!ggml_moe_source_profile_restore(before, original.counts, original.heat, original.prior, original.observations) ||
                !ggml_moe_source_profile_restore(after, saved.counts, saved.heat, saved.prior, saved.observations)) { return 1; }
        const int32_t tail[] = {0, 0, 1};
        auto usage = original.usage;
        for (const int32_t id : tail) { usage[id] += 1; }
        for (auto * state : {&before, &after}) {
            if (!ggml_moe_source_profile_observe(*state, tail, 3) || !ggml_moe_source_profile_accumulate(*state, usage)) { return 1; }
        }
        std::vector<uint64_t> before_counts, after_counts;
        std::vector<double> before_heat, after_heat;
        std::vector<int32_t> before_ranks, after_ranks, before_prior, after_prior;
        uint64_t before_observations = 0, after_observations = 0;
        if (!ggml_moe_source_profile_snapshot(before, before_counts, before_heat, before_ranks, before_observations, &before_prior) ||
                !ggml_moe_source_profile_snapshot(after, after_counts, after_heat, after_ranks, after_observations, &after_prior) ||
                before_counts != after_counts || before_heat != after_heat || before_ranks != after_ranks ||
                before_prior != after_prior || before_observations != after_observations) { return 1; }
    }
    uint32_t rejected = 0;
    const auto invalid = [&](const std::vector<uint8_t> & input, const std::vector<llama_moe_source_group> & catalog) {
        try { (void) llama_moe_profile_statistics_parse(input.data(), input.size(), catalog); }
        catch (const std::runtime_error &) { ++rejected; return; }
        throw std::runtime_error("invalid statistics fixture accepted");
    };
    for (uint32_t variant = 0; variant < 19; ++variant) {
        auto bad = learned;
        auto & item = bad.sources.front();
        if (variant == 0) { item.heat[0] = -1; }
        if (variant == 1) { item.heat[0] = std::numeric_limits<double>::infinity(); }
        if (variant == 2) { item.heat[0] = std::numeric_limits<double>::quiet_NaN(); }
        if (variant == 3) { item.prior[0] = -1; }
        if (variant == 4) { item.prior[0] = int32_t(item.counts.size()); }
        if (variant == 5) { item.prior[0] = item.prior[1]; }
        if (variant == 6) { item.usage[0] = -1; }
        if (variant == 7) { item.usage[0] = std::numeric_limits<float>::infinity(); }
        if (variant == 8) { item.usage[0] = std::numeric_limits<float>::quiet_NaN(); }
        if (variant == 9) { item.usage[0] = float(item.counts[0] + 1); }
        if (variant == 10) { item.windows = 0; }
        if (variant == 11) { item.windows = item.observations + 1; }
        if (variant == 12) { item.heat.clear(); }
        if (variant == 13) { item.usage.clear(); }
        if (variant == 14) { item.prior.clear(); }
        if (variant == 15) { item.scores[0] = 0; }
        if (variant == 16) { item.heat[0] = item.heat[1] = std::numeric_limits<double>::max(); }
        if (variant == 17) { bad.sources.push_back(item); }
        if (variant == 18) { item.counts[0] = 0; --item.observations; }
        try { (void) llama_moe_profile_statistics_serialize(bad, sources); }
        catch (const std::runtime_error &) { ++rejected; continue; }
        throw std::runtime_error("invalid learned writer accepted");
    }
    for (uint32_t variant = 0; variant < 6; ++variant) {
        gguf_context_ptr malformed(gguf_init_from_buffer(learned_bytes.data(), learned_bytes.size(), {true, nullptr}));
        if (variant == 0) { gguf_set_val_u32(malformed.get(), "moe.profile.version", 2); }
        if (variant == 1) { gguf_set_val_u32(malformed.get(), "moe.profile.learning_policy", 2); }
        if (variant == 2) { gguf_set_arr_data(malformed.get(), "moe.profile.learned_heat", GGUF_TYPE_FLOAT32, learned.sources[0].usage.data(), 3); }
        if (variant == 3) { gguf_set_arr_data(malformed.get(), "moe.profile.prior_indices", GGUF_TYPE_UINT64, learned.sources[0].counts.data(), 3); }
        if (variant == 4) { gguf_set_arr_data(malformed.get(), "moe.profile.decayed_usage", GGUF_TYPE_FLOAT32, learned.sources[0].usage.data(), 3); }
        if (variant == 5) { gguf_set_arr_data(malformed.get(), "moe.profile.learning_windows", GGUF_TYPE_UINT64, &learned.sources[0].windows, 1); }
        invalid(source_profile_metadata_bytes(malformed.get()), sources);
    }
    auto unseen_learned = learned;
    for (auto & source : unseen_learned.sources) {
        std::fill(source.counts.begin(), source.counts.end(), 0);
        std::fill(source.heat.begin(), source.heat.end(), 0);
        std::fill(source.usage.begin(), source.usage.end(), 0);
        source.observations = source.windows = 0;
    }
    const auto unseen_learned_bytes = llama_moe_profile_statistics_serialize(unseen_learned, sources);
    const auto unseen_restored = llama_moe_profile_statistics_parse(unseen_learned_bytes.data(), unseen_learned_bytes.size(), sources);
    for (const auto & source : unseen_restored.sources) {
        std::vector<int32_t> projected;
        if (source.observations || source.windows || !ggml_moe_source_rank_statistics(
                {{source.counts.data(), 0, source.tensor->nb[2], uint32_t(source.counts.size()), source.scores.data()}}, projected)) { return 1; }
        for (size_t slots = 1; slots <= projected.size(); ++slots) {
            for (size_t i = 0; i < slots; ++i) { if (source.prior[projected[i]] != int32_t(i) || source.counts[projected[i]]) { return 1; } }
        }
    }
    fprintf(stderr, "test-source-profile-unobserved-prior: both bank geometries retain original rank at all8 capacities with zero occurrences OK\n");
    const auto empty_learning = llama_moe_profile_learning_baseline(sources, {}, {});
    const auto calibration_learning = llama_moe_profile_learning_baseline(sources, bound, {});
    const int32_t partial_order[] = {2};
    const std::vector<ggml_backend_moe_static_profile_v1> partial_profiles{{a, partial_order, 1}};
    const auto partial_learning = llama_moe_profile_learning_baseline(sources, {}, partial_profiles);
    if (empty_learning.sources.size() != 2 || calibration_learning.sources.size() != 2 || partial_learning.sources.size() != 2 ||
            empty_learning.sources[0].prior != std::vector<int32_t>{0, 1, 2} ||
            calibration_learning.sources[0].prior != std::vector<int32_t>{2, 1, 0} ||
            calibration_learning.sources[1].prior != std::vector<int32_t>{4, 3, 2, 1, 0} ||
            partial_learning.sources[0].prior != std::vector<int32_t>{1, 2, 0} ||
            partial_learning.sources[1].prior != std::vector<int32_t>{0, 1, 2, 3, 4}) { return 1; }
    for (const auto * baseline : {&empty_learning, &calibration_learning, &partial_learning}) {
        for (const auto & source : baseline->sources) {
            if (source.observations || source.windows || source.counts != std::vector<uint64_t>(source.counts.size(), 0) ||
                    source.heat != std::vector<double>(source.counts.size(), 0) || source.usage != std::vector<float>(source.counts.size(), 0)) { return 1; }
        }
        const auto metadata = llama_moe_profile_statistics_serialize(*baseline, sources);
        const auto reloaded = llama_moe_profile_statistics_parse(metadata.data(), metadata.size(), sources);
        if (reloaded.sources.size() != 2 || reloaded.sources[0].prior != baseline->sources[0].prior ||
                reloaded.sources[1].prior != baseline->sources[1].prior) { return 1; }
    }
    const auto resumed_baseline = llama_moe_profile_learning_baseline(sources, learned, {});
    const auto resumed_metadata = llama_moe_profile_statistics_serialize(resumed_baseline, sources);
    const auto resumed_loaded = llama_moe_profile_statistics_parse(resumed_metadata.data(), resumed_metadata.size(), sources);
    for (size_t i = 0; i < learned.sources.size(); ++i) {
        const auto & before = learned.sources[i]; const auto & after = resumed_loaded.sources[i];
        if (before.counts != after.counts || before.heat != after.heat || before.usage != after.usage || before.prior != after.prior ||
                before.windows != after.windows || before.observations != after.observations || before.scores != after.scores) { return 1; }
    }
    auto aliased_catalog = sources; aliased_catalog.push_back(sources.front());
    if (llama_moe_profile_learning_baseline(aliased_catalog, bound, {}).sources.size() != 2) { return 1; }
    uint32_t baseline_rejections = 0;
    for (uint32_t variant = 0; variant < 8; ++variant) {
        auto catalog = sources;
        auto initial = bound;
        auto profiles = partial_profiles;
        if (variant == 0) { initial.sources.clear(); profiles[0].experts = nullptr; }
        if (variant == 1) { initial.sources.clear(); profiles[0].n_experts = 4; }
        if (variant == 2) { initial.sources.clear(); profiles.push_back(profiles.front()); }
        if (variant == 3) { initial.sources.pop_back(); profiles.clear(); }
        if (variant == 4) { initial.sources[0].counts.pop_back(); profiles.clear(); }
        if (variant == 5) { profiles.clear(); catalog[0].banks[0].tensor = nullptr; }
        if (variant == 6) { initial.sources.clear(); profiles.clear(); catalog.clear(); }
        if (variant == 7) { initial.sources.clear(); profiles[0].down = nullptr; }
        try { (void) llama_moe_profile_learning_baseline(catalog, initial, profiles); return 1; }
        catch (const std::runtime_error &) { ++baseline_rejections; }
    }
    auto * wide = ggml_new_tensor_3d(context.get(), GGML_TYPE_Q5_K, 256, 7, 65537);
    const std::vector<llama_moe_source_group> wide_catalog{{GGML_BACKEND_MOE_CANDIDATE_LAYOUT_ROUTED_MATRIX, 1, true,
        {{wide, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_ROUTED_WEIGHT, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE, {"wide-quantized-source"}}}, 4}};
    const auto wide_learning = llama_moe_profile_learning_baseline(wide_catalog, {}, {});
    if (wide_learning.sources.size() != 1 || wide_learning.sources[0].counts.size() != 65537 ||
            wide_learning.sources[0].prior.back() != 65536 || wide_learning.sources[0].observations) { return 1; }
    fprintf(stderr, "test-source-profile-full-learning: full heterogeneous/quantized catalog, calibration/prior separation, partial seed completion, aliases, restored history,65537 experts and%u invalid cases OK\n", baseline_rejections);
    auto overlaid = empty_learning;
    std::vector<uint8_t> covered(overlaid.sources.size(), 0);
    if (!llama_moe_profile_learning_overlay(overlaid, covered, nullptr, 0)) { return 1; }
    for (const auto & view : learning_views) {
        if (!llama_moe_profile_learning_overlay(overlaid, covered, &view, 1)) { return 1; }
    }
    overlaid.provenance = learned.provenance;
    if (llama_moe_profile_statistics_serialize(overlaid, sources) != learned_bytes ||
            covered != std::vector<uint8_t>(overlaid.sources.size(), 1)) { return 1; }
    if (!llama_moe_profile_learning_overlay(overlaid, covered, learning_views.data(), uint32_t(learning_views.size())) ||
            llama_moe_profile_statistics_serialize(overlaid, sources) != learned_bytes) { return 1; }
    for (uint32_t variant = 0; variant < 7; ++variant) {
        auto conflict = learned.sources.front();
        auto view = learning_views.front();
        if (variant == 0) { ++conflict.counts.front(); ++view.source.observations; view.source.counts = conflict.counts.data(); }
        if (variant == 1) { ++conflict.heat.front(); view.heat = conflict.heat.data(); }
        if (variant == 2) { conflict.usage.front() = conflict.usage.front() ? 0 : 0.25f; view.usage = conflict.usage.data(); }
        if (variant == 3) { std::swap(conflict.prior[0], conflict.prior[1]); view.prior = conflict.prior.data(); }
        if (variant == 4) { view.windows = view.windows == 1 ? 2 : 1; }
        if (variant == 5) { ++view.source.domain; }
        auto rejected_coverage = covered;
        if (variant == 6) { rejected_coverage.pop_back(); }
        if (llama_moe_profile_learning_overlay(overlaid, rejected_coverage, &view, 1) ||
                llama_moe_profile_statistics_serialize(overlaid, sources) != learned_bytes ||
                (variant != 6 && rejected_coverage != covered)) { return 1; }
    }
    fprintf(stderr, "test-source-profile-context-overlay: disjoint actual views, identical-owner deduplication, complete codec and7 conflicting/identity/coverage rejections preserve output OK\n");
    fprintf(stderr, "test-source-profile-learning-codec: descriptor-bound full metadata, exact roundtrip, prior/heat/usage/cadence, resumed learning, zero coverage and25 malformed cases OK\n");
    gguf_context_ptr raw_metadata(gguf_init_from_buffer(bytes.data(), bytes.size(), {true, nullptr}));
    const auto unpadded_size = gguf_get_data_offset(raw_metadata.get());
    const auto unpadded = llama_moe_profile_statistics_parse(bytes.data(), unpadded_size, sources);
    if (unpadded.sources[0].counts != bound.sources[0].counts) { return 1; }
    for (size_t size = 0; size < unpadded_size; size += 17) { invalid(std::vector<uint8_t>(bytes.begin(), bytes.begin() + size), sources); }
    if (unpadded_size < bytes.size()) { auto bad_padding = bytes; bad_padding.back() = 1; invalid(bad_padding, sources); }
    for (size_t offset : {size_t(0), size_t(4), size_t(8), size_t(16)}) {
        auto input = bytes; input[offset] = 255; invalid(input, sources);
    }
    auto trailing = bytes; trailing.push_back(0); invalid(trailing, sources);
    const std::vector<uint32_t> wrong_domains{1, 1}, wrong_types{GGML_TYPE_F32, GGML_TYPE_F32};
    const std::vector<uint64_t> wrong_offsets{0, 3, UINT64_MAX}, wrong_observations{6, 16}, wrong_counts{UINT64_MAX, 2, 3, 1, 2, 3, 4, 5};
    const auto malformed = [&](const char * name, gguf_type type, const void * data, size_t count) {
        auto changed = make_source_profile_statistics(sources); gguf_set_arr_data(changed.get(), name, type, data, count);
        invalid(source_profile_metadata_bytes(changed.get()), sources);
    };
    malformed("moe.profile.source_domains", GGUF_TYPE_UINT32, wrong_domains.data(), wrong_domains.size());
    malformed("moe.profile.source_types", GGUF_TYPE_UINT32, wrong_types.data(), wrong_types.size());
    malformed("moe.profile.expert_offsets", GGUF_TYPE_UINT64, wrong_offsets.data(), wrong_offsets.size());
    malformed("moe.profile.expert_offsets", GGUF_TYPE_UINT32, wrong_domains.data(), wrong_domains.size());
    malformed("moe.profile.source_name_offsets", GGUF_TYPE_UINT64, wrong_offsets.data(), wrong_offsets.size());
    malformed("moe.profile.observations", GGUF_TYPE_UINT64, wrong_observations.data(), wrong_observations.size());
    malformed("moe.profile.expert_counts", GGUF_TYPE_UINT64, wrong_counts.data(), wrong_counts.size());
    malformed("moe.profile.expert_counts", GGUF_TYPE_UINT8, wrong_counts.data(), wrong_counts.size());
    auto names = std::vector<uint8_t>(long_name.begin(), long_name.end()); names.insert(names.end(), {'s','o','u','r','c','e','-','b'});
    names[1] = 0; malformed("moe.profile.source_name_bytes", GGUF_TYPE_UINT8, names.data(), names.size());
    auto omitted = make_source_profile_statistics({sources.front()}); invalid(source_profile_metadata_bytes(omitted.get()), sources);
    auto ambiguous = sources; ambiguous.push_back(sources.front()); ambiguous.back().banks.front().tensor = b; invalid(bytes, ambiguous);
    auto unavailable = sources; unavailable.front().banks.front().names.clear(); invalid(bytes, unavailable);
    auto geometry = sources; auto * c = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, 9, 7, 3); geometry.front().banks.front().tensor = c; invalid(bytes, geometry);
    auto alias_catalog = sources; alias_catalog.front().banks.front().names = {"alias-a", long_name};
    auto alias_metadata = make_source_profile_statistics(alias_catalog); auto alias_bytes = source_profile_metadata_bytes(alias_metadata.get());
    const auto aliases = llama_moe_profile_statistics_parse(alias_bytes.data(), alias_bytes.size(), sources);
    if (aliases.sources[0].tensor != a || aliases.sources[0].counts != bound.sources[0].counts) { return 1; }
    auto shared = make_source_profile_statistics(sources);
    const std::string shared_names = long_name + "alias-asource-b";
    const std::vector<uint64_t> shared_name_offsets{0, 200, 207, 215}, shared_offsets{0, 3, 6, 11}, shared_observations{6, 6, 15};
    const std::vector<uint32_t> shared_domains{1, 1, 2}, shared_types{GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F16};
    std::vector<int64_t> shared_shapes;
    for (const auto * tensor : {a, a, b}) { shared_shapes.insert(shared_shapes.end(), tensor->ne, tensor->ne + GGML_MAX_DIMS); }
    std::vector<uint64_t> shared_counts{1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5};
    gguf_set_arr_data(shared.get(), "moe.profile.source_name_bytes", GGUF_TYPE_UINT8, shared_names.data(), shared_names.size());
    gguf_set_arr_data(shared.get(), "moe.profile.source_name_offsets", GGUF_TYPE_UINT64, shared_name_offsets.data(), shared_name_offsets.size());
    gguf_set_arr_data(shared.get(), "moe.profile.source_domains", GGUF_TYPE_UINT32, shared_domains.data(), shared_domains.size());
    gguf_set_arr_data(shared.get(), "moe.profile.source_types", GGUF_TYPE_UINT32, shared_types.data(), shared_types.size());
    gguf_set_arr_data(shared.get(), "moe.profile.source_shapes", GGUF_TYPE_INT64, shared_shapes.data(), shared_shapes.size());
    gguf_set_arr_data(shared.get(), "moe.profile.expert_offsets", GGUF_TYPE_UINT64, shared_offsets.data(), shared_offsets.size());
    gguf_set_arr_data(shared.get(), "moe.profile.observations", GGUF_TYPE_UINT64, shared_observations.data(), shared_observations.size());
    gguf_set_arr_data(shared.get(), "moe.profile.expert_counts", GGUF_TYPE_UINT64, shared_counts.data(), shared_counts.size());
    const auto shared_bytes = source_profile_metadata_bytes(shared.get());
    const auto shared_bound = llama_moe_profile_statistics_parse(shared_bytes.data(), shared_bytes.size(), sources);
    if (shared_bound.sources.size() != 2 || shared_bound.sources[0].counts != bound.sources[0].counts) { return 1; }
    shared_counts[3] = 2; shared_counts[4] = 1;
    gguf_set_arr_data(shared.get(), "moe.profile.expert_counts", GGUF_TYPE_UINT64, shared_counts.data(), shared_counts.size());
    invalid(source_profile_metadata_bytes(shared.get()), sources);
    const std::vector<uint64_t> zero_counts{0, 0, 0, 1, 2, 3, 4, 5}, zero_observations{0, 15};
    auto unseen = make_source_profile_statistics(sources);
    gguf_set_arr_data(unseen.get(), "moe.profile.expert_counts", GGUF_TYPE_UINT64, zero_counts.data(), zero_counts.size());
    gguf_set_arr_data(unseen.get(), "moe.profile.observations", GGUF_TYPE_UINT64, zero_observations.data(), zero_observations.size());
    const auto unseen_bytes = source_profile_metadata_bytes(unseen.get());
    const auto unseen_bound = llama_moe_profile_statistics_parse(unseen_bytes.data(), unseen_bytes.size(), sources);
    if (unseen_bound.sources[0].counts.size() != 3 || unseen_bound.sources[0].observations != 0) { return 1; }
    routed_profile_collector collector;
    auto collected_catalog = sources;
    collected_catalog.push_back(sources.front());
    if (!collector.initialize(collected_catalog) || collector.statistics.sources.size() != 2) { return 1; }
    const uint32_t route_indices[] = {0, 1, 0};
    const int32_t expert_ids[] = {2, 0}, ownership[] = {0, 2};
    ggml_backend_moe_source_access_v1 probe = {};
    ggml_tensor copied_weight = *a;
    probe.source = &copied_weight; probe.source_witness = a; probe.source_domain = sources.front().domain;
    probe.route_indices = route_indices; probe.expert_ids = expert_ids; probe.classes = ownership;
    probe.n_routes = 3; probe.n_distinct = 2;
    if (!collector.observe(2, &probe) || collector.statistics.sources[0].counts != std::vector<uint64_t>{1, 0, 2} ||
            collector.statistics.sources[0].observations != 3 || collector.ownership_counts != std::array<uint64_t, 3>{2, 0, 1}) { return 1; }
    const auto counts_before = collector.statistics.sources[0].counts;
    const uint32_t invalid_routes[] = {0, 2, 0}; probe.route_indices = invalid_routes;
    if (collector.observe(3, &probe) || collector.statistics.sources[0].counts != counts_before) { return 1; }
    probe.route_indices = route_indices;
    const int32_t invalid_ids[] = {3, 0}; probe.expert_ids = invalid_ids;
    if (collector.observe(3, &probe) || collector.statistics.sources[0].counts != counts_before) { return 1; }
    probe.expert_ids = expert_ids; probe.source = nullptr;
    if (collector.observe(3, &probe)) { return 1; }
    probe.source = &copied_weight; collector.statistics.sources[0].observations = UINT64_MAX;
    if (collector.observe(3, &probe) || collector.statistics.sources[0].counts != counts_before) { return 1; }
    collector.statistics.sources[0].observations = 3;
    collector.statistics.provenance = "Synthetic routing observer fixture; complete source vectors, not calibrated model observations";
    auto collected_metadata = make_source_profile_statistics(collected_catalog, false, &collector.statistics);
    const auto collected_bytes = source_profile_metadata_bytes(collected_metadata.get());
    const auto collected = llama_moe_profile_statistics_parse(collected_bytes.data(), collected_bytes.size(), collected_catalog);
    if (collected.sources.size() != 2 || collected.sources[0].counts != counts_before || collected.sources[1].counts != std::vector<uint64_t>(5, 0) ||
            collected.sources[1].observations != 0) { return 1; }
    routed_profile_collector empty;
    if (!empty.initialize(sources)) { return 1; }
    profile_corpus_average mixture;
    mixture.initialize(empty.statistics, {{"general", 1}, {"code", 3}});
    auto request = empty.statistics;
    request.sources[0].counts = {1, 1, 0}; request.sources[0].observations = 2;
    mixture.add("general", request, 2);
    request.sources[0].counts = {0, 10, 0}; request.sources[0].observations = 10;
    mixture.add("general", request, 10);
    request.sources[0].counts = {8, 0, 2}; request.sources[0].observations = 10;
    mixture.add("code", request, 10);
    mixture.finish();
    mixture.statistics.provenance = "Synthetic unequal-length weighted corpus fixture";
    const auto & scored = mixture.statistics.sources[0];
    const std::vector<double> expected_scores{0.6625, 0.1875, 0.15};
    for (size_t i = 0; i < 3; ++i) { if (std::fabs(scored.scores[i] - expected_scores[i]) > 1e-15) { return 1; } }
    if (scored.counts != std::vector<uint64_t>{9, 11, 2} || scored.observations != 22) { return 1; }
    auto scored_metadata = make_source_profile_statistics(sources, false, &mixture.statistics);
    const auto scored_bytes = source_profile_metadata_bytes(scored_metadata.get());
    const auto score_roundtrip = llama_moe_profile_statistics_parse(scored_bytes.data(), scored_bytes.size(), sources);
    if (score_roundtrip.sources[0].scores != scored.scores || score_roundtrip.sources[1].scores != std::vector<double>(5, 0)) { return 1; }
    if (!ggml_moe_source_rank_statistics({{scored.counts.data(), scored.observations, 16, 3, scored.scores.data()}}, ranks) ||
            ranks != std::vector<int32_t>{0, 1, 2}) { return 1; }
    std::vector<double> flat_scores = scored.scores;
    flat_scores.resize(8, 0);
    for (uint32_t variant = 0; variant < 7; ++variant) {
        auto malformed = make_source_profile_statistics(sources, false, &mixture.statistics);
        auto changed = flat_scores;
        if (variant == 0) { changed[0] = -1; }
        if (variant == 1) { changed[0] = INFINITY; }
        if (variant == 2) { changed[0] = NAN; }
        if (variant == 3) { changed[3] = 1; }
        if (variant == 4) { changed[0] = changed[1] = std::numeric_limits<double>::max(); }
        gguf_set_arr_data(malformed.get(), "moe.profile.ranking_scores", GGUF_TYPE_FLOAT64, changed.data(), variant == 5 ? 7 : 8);
        if (variant == 6) { gguf_set_val_u32(malformed.get(), "moe.profile.version", 1); }
        invalid(source_profile_metadata_bytes(malformed.get()), sources);
    }
    auto mistyped = make_source_profile_statistics(sources, false, &mixture.statistics);
    gguf_set_arr_data(mistyped.get(), "moe.profile.ranking_scores", GGUF_TYPE_UINT64, flat_scores.data(), 8);
    invalid(source_profile_metadata_bytes(mistyped.get()), sources);
    auto missing_scores = make_source_profile_statistics(sources);
    gguf_set_val_u32(missing_scores.get(), "moe.profile.version", 2);
    invalid(source_profile_metadata_bytes(missing_scores.get()), sources);
    gguf_context_ptr score_aliases(gguf_init_from_buffer(shared_bytes.data(), shared_bytes.size(), {true, nullptr}));
    std::vector<double> alias_scores{0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0, 0, 0, 0, 0};
    gguf_set_val_u32(score_aliases.get(), "moe.profile.version", 2);
    gguf_set_arr_data(score_aliases.get(), "moe.profile.ranking_scores", GGUF_TYPE_FLOAT64, alias_scores.data(), alias_scores.size());
    const auto valid_alias_scores = source_profile_metadata_bytes(score_aliases.get());
    if (llama_moe_profile_statistics_parse(valid_alias_scores.data(), valid_alias_scores.size(), sources).sources[0].scores != std::vector<double>{0.1, 0.2, 0.3}) { return 1; }
    alias_scores[3] = 0.2;
    gguf_set_arr_data(score_aliases.get(), "moe.profile.ranking_scores", GGUF_TYPE_FLOAT64, alias_scores.data(), alias_scores.size());
    invalid(source_profile_metadata_bytes(score_aliases.get()), sources);
    ggml_backend_moe_source_statistics_v1 scored_view{scored.tensor, scored.counts.data(), scored.observations, 3, scored.domain};
    const double * score_views[] = {scored.scores.data()};
    if (!ggml_moe_source_scores_valid(&scored_view, score_views, 1)) { return 1; }
    score_views[0] = nullptr;
    if (ggml_moe_source_scores_valid(&scored_view, score_views, 1)) { return 1; }
    printf("test-source-corpus-mixture: %s\n", common_json({{"scores", scored.scores}, {"counts", scored.counts},
        {"ranks", ranks}, {"rows", common_json::array({2, 10, 10})}, {"weights", {{"general", 1}, {"code", 3}}}}).dump().c_str());
    auto ambiguous_domains = sources; ambiguous_domains.back().banks.front().tensor = a;
    if (!collector.initialize(ambiguous_domains) || collector.statistics.sources.size() != 2) { return 1; }
    probe.source_domain = ambiguous_domains.back().domain;
    if (!collector.observe(3, &probe) || collector.statistics.sources[0].observations || collector.statistics.sources[1].observations != 3) { return 1; }
    fprintf(stderr, "test-source-profile-collector: original tensor/domain identities,aliases,ownership,zero coverage,invalid route/ID/source/overflow rejection and codec roundtrip OK\n");
    fprintf(stderr, "test-source-profile-statistics: heterogeneous sources/domains,full200-byte names,aliases,complete occurrences and%u malformed/coverage cases OK\n", rejected);
    return 0;
}

static int test_model_source_learning(const char * path, const char * output_path, uint32_t slots, uint32_t context_size, uint32_t batch_size, const char * adaptation) {
    if (!output_path || !*output_path) { return 1; }
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;
    model_params.moe_expert_cache_slots = slots;
    model_params.moe_expert_cache_host_pinned_size = 0;
    model_params.load_mode = LLAMA_LOAD_MODE_NONE;
    model_params.progress_callback = silent_model_load_progress;
    llama_model_ptr model(llama_model_load_from_file(path, model_params));
    if (!model) { return 1; }
    const auto & sources = model->moe_sources();
    const auto seed = llama_moe_profile_learning_baseline(sources, {}, {});
    const auto seed_bytes = llama_moe_profile_statistics_serialize(seed, sources);
    auto seed_path = std::filesystem::u8path(output_path); seed_path += ".seed.gguf";
    if (!common_moe_profile_write(seed_path, seed_bytes.data(), seed_bytes.size())) { return 1; }
    auto params = llama_context_default_params();
    params.n_ctx = context_size; params.n_batch = params.n_ubatch = batch_size;
    params.n_threads = params.n_threads_batch = 4; params.phase_aware_workspace = true;
    const auto filename = fs_path_to_utf8(seed_path);
    llama_context_ptr context(llama_init_from_model_with_moe_profile(model.get(), params, filename.c_str(), adaptation));
    if (!context || !context->source_core_enabled()) { return 1; }
    const auto * vocab = llama_model_get_vocab(model.get());
    auto tokens = common_tokenize(vocab, "Hello there.", true, true);
    if (tokens.empty() || tokens.size() > params.n_batch) { return 1; }
    auto prompt_token = tokens.back();
    tokens.resize(params.n_batch, prompt_token);
    auto batch = llama_batch_get_one(tokens.data(), int32_t(tokens.size()));
    if (llama_decode(context.get(), batch)) { return 1; }
    for (int i = 0; i < 8; ++i) {
        const auto * logits = llama_get_logits_ith(context.get(), -1);
        if (!logits || !std::all_of(logits, logits + llama_vocab_n_tokens(vocab), [](float value) { return std::isfinite(value); })) { return 1; }
        llama_token token = llama_vocab_bos(vocab);
        if (token == LLAMA_TOKEN_NULL) { token = tokens.front(); }
        batch = llama_batch_get_one(&token, 1);
        if (llama_decode(context.get(), batch)) { return 1; }
    }
    llama_synchronize(context.get());
    std::vector<uint8_t> before;
    if (!context->snapshot_moe_learning(before, 5000) || !common_moe_profile_save(context.get(), std::filesystem::u8path(output_path))) { return 1; }
    const auto learned = llama_moe_profile_statistics_parse(before.data(), before.size(), sources);
    const auto observed = std::count_if(learned.sources.begin(), learned.sources.end(), [](const auto & source) { return source.observations > 0; });
    if (!observed) { return 1; }
    context.reset();
    model.reset();
    for (const uint32_t restored_slots : {slots, slots + 1}) {
        model_params.moe_expert_cache_slots = restored_slots;
        llama_model_ptr fresh_model(llama_model_load_from_file(path, model_params));
        if (!fresh_model) { return 1; }
        llama_context_ptr restored(llama_init_from_model_with_moe_profile(fresh_model.get(), params, output_path, adaptation));
        std::vector<uint8_t> after;
        if (!restored || !restored->snapshot_moe_learning(after, 5000) || before != after) { return 1; }
        batch = llama_batch_get_one(tokens.data(), int32_t(tokens.size()));
        if (llama_decode(restored.get(), batch)) { return 1; }
        for (uint32_t i = 0; i < 4; ++i) {
            batch = llama_batch_get_one(&prompt_token, 1);
            if (llama_decode(restored.get(), batch)) { return 1; }
            const auto * logits = llama_get_logits_ith(restored.get(), -1);
            if (!logits || !std::all_of(logits, logits + llama_vocab_n_tokens(llama_model_get_vocab(fresh_model.get())),
                    [](float value) { return std::isfinite(value); })) { return 1; }
        }
        llama_synchronize(restored.get());
        if (!restored->snapshot_moe_learning(after, 5000)) { return 1; }
        const auto continued = llama_moe_profile_statistics_parse(after.data(), after.size(), fresh_model->moe_sources());
        if (continued.sources.size() != learned.sources.size()) { return 1; }
        for (size_t i = 0; i < learned.sources.size(); ++i) {
            const auto & initial = learned.sources[i];
            const auto & current = continued.sources[i];
            if (current.observations < initial.observations || current.windows < initial.windows || current.prior != initial.prior ||
                    (initial.observations && (current.observations == initial.observations || current.windows < initial.windows + 4))) { return 1; }
            for (size_t j = 0; j < initial.counts.size(); ++j) {
                if (current.counts[j] < initial.counts[j] || current.heat[j] < initial.heat[j]) { return 1; }
            }
        }
        fprintf(stderr, "test-model-source-learning: fresh_model=1 slots=%u prompt_rows=%zu continued_windows=4 full_history_retained OK\n", restored_slots, tokens.size());
    }
    fprintf(stderr, "test-model-source-learning: full_sources=%zu observed_sources=%zu saved_bytes=%zu cold_snapshot=exact load_mode=none\n",
        learned.sources.size(), size_t(observed), before.size());
    return 0;
}

static int test_model_source_profile(const char * path, const char * statistics_file) {
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;
    model_params.moe_expert_cache_slots = 4;
    model_params.moe_expert_cache_host_pinned_size = 0;
    model_params.progress_callback = silent_model_load_progress;
    llama_model_ptr model(llama_model_load_from_file(path, model_params));
    if (!model) { return 1; }
    auto layers = std::move(model->layers);
    model->layers.clear();
    model->build_moe_sources();
    model->layers = std::move(layers);
    auto params = llama_context_default_params();
    params.n_ctx = 512;
    params.n_batch = params.n_ubatch = 128;
    params.n_threads = params.n_threads_batch = 4;
    params.phase_aware_workspace = true;
    const auto & sources = model->moe_sources();
    auto metadata = make_source_profile_statistics(sources);
    const auto bytes = source_profile_metadata_bytes(metadata.get());
    if (statistics_file) {
        const auto * configured = getenv("GGML_MOE_EXPERT_PROFILE");
        if (!configured || strcmp(configured, statistics_file)) { return 1; }
        std::ofstream output(statistics_file, std::ios::binary | std::ios::trunc);
        if (!output.write(reinterpret_cast<const char *>(bytes.data()), bytes.size())) { return 1; }
    }
    llama_context_ptr ctx(llama_init_from_model(model.get(), params));
    if (!ctx || !ctx->initialize_moe_profile()) { return 1; }
    const auto statistics = llama_moe_profile_statistics_parse(bytes.data(), bytes.size(), sources);
    std::vector<ggml_backend_moe_source_statistics_v1> views;
    for (const auto & source : statistics.sources) {
        views.push_back({source.tensor, source.counts.data(), source.observations, uint32_t(source.counts.size()), source.domain});
    }
    if (!ctx->initialize_moe_statistics(views)) { return 1; }
    std::vector<std::vector<int32_t>> source_ranks(statistics.sources.size());
    std::vector<ggml_backend_moe_static_profile_v1> profiles;
    for (size_t i = 0; i < statistics.sources.size(); ++i) {
        const auto & source = statistics.sources[i];
        const auto * tensor = source.tensor;
        if (tensor->ne[2] < 4 || tensor->ne[2] > 65536) { return 1; }
        auto & ranks = source_ranks[i];
        for (int64_t expert = 0; expert < tensor->ne[2]; ++expert) { ranks.push_back(int32_t(expert)); }
        std::sort(ranks.begin(), ranks.end(), [&](int32_t a, int32_t b) { return source.counts[a] > source.counts[b] || (source.counts[a] == source.counts[b] && a < b); });
        profiles.push_back({tensor, ranks.data(), uint32_t(ranks.size())});
    }
    if (sources.empty() || !ctx->initialize_moe_profile(profiles)) { return 1; }
    const auto * vocab = llama_model_get_vocab(model.get());
    auto tokens = common_tokenize(vocab, "Explain the sky.", true, true);
    if (tokens.empty() || tokens.size() > params.n_batch) { return 1; }
    auto batch = llama_batch_get_one(tokens.data(), int32_t(tokens.size()));
    if (llama_decode(ctx.get(), batch) != 0) { return 1; }
    const auto * logits = llama_get_logits_ith(ctx.get(), -1);
    if (!logits) { return 1; }
    for (int32_t i = 0; i < llama_vocab_n_tokens(vocab); ++i) {
        if (!std::isfinite(logits[i])) { return 1; }
    }
    for (uint32_t step = 0; step < 8; ++step) {
        auto token = tokens.back();
        auto decode = llama_batch_get_one(&token, 1);
        if (llama_decode(ctx.get(), decode) != 0) { return 1; }
        const auto * current = llama_get_logits_ith(ctx.get(), -1);
        if (!current) { return 1; }
        for (int32_t i = 0; i < llama_vocab_n_tokens(vocab); ++i) { if (!std::isfinite(current[i])) { return 1; } }
    }
    for (auto & ranks : source_ranks) { std::reverse(ranks.begin(), ranks.end()); }
    if (!ctx->initialize_moe_profile(profiles)) { return 1; }
    fprintf(stderr, "test-model-source-profile: groups=%zu original routed banks initialized before and after decode without layer/down-bank lookup OK\n", sources.size());
    fprintf(stderr, "test-model-source-profile: full source-indexed statistics bytes=%zu sources=%zu decoded and ranked independently of cache capacity OK\n", bytes.size(), statistics.sources.size());
    return 0;
}

struct routed_cpu_oracle {
    routed_profile_collector profile_collector;
    std::string profile_export_path;
    std::string profile_trace_path;
    llama_context * profile_context = nullptr;
    std::ofstream profile_trace;
    size_t profile_trace_bytes = 0;
    uint64_t profile_frame = 0;
    uint64_t profile_frames = 0;
    common_json records = common_json::array();
    uint64_t mismatched_routes = 0;
    ggml_backend_ptr backend;
    bool cpu_enabled = false;
    bool gpu_enabled = false;
    ggml_backend_ptr gpu_backend;
    common_json gpu_records = common_json::array();
    common_json projection_records = common_json::array();
    std::string projection_capture_path;
    bool projection_captured = false;
    std::string trace_path;
    size_t trace_bytes = 0;
    common_json trace_records = common_json::array();
    common_json trace_states = common_json::array();

    bool profile_trace_record(const common_json & record) {
        constexpr size_t limit = 256 * 1024 * 1024;
        const std::string line = record.dump();
        if (profile_trace_bytes >= limit || line.size() >= limit - profile_trace_bytes) { return false; }
        if (!(profile_trace << line << '\n')) { return false; }
        profile_trace_bytes += line.size() + 1;
        return true;
    }

    bool trace_profile_projection(uint64_t epoch, const ggml_backend_moe_source_access_v1 * probe) {
        const auto * frame = profile_context ? profile_context->get_moe_test_frame() : nullptr;
        if (!frame || !frame->ubatch || !frame->certificate || !probe || !probe->ids) { return false; }
        if (getenv("GGML_TEST_MOE_PROFILE_FAIL_FRAME")) { return false; }
        const auto & batch = *frame->ubatch;
        const auto & certificate = *frame->certificate;
        if (certificate.magic != GGML_GRAPH_EXECUTION_CERTIFICATE_MAGIC || certificate.struct_size != sizeof(certificate) ||
                certificate.n_rows != batch.n_tokens || !batch.n_tokens || !batch.n_pos || !batch.pos || !batch.n_seq_id || !batch.seq_id ||
                uint64_t(batch.n_tokens) * batch.n_pos > (1u << 22)) { return false; }
        if (profile_frame != frame->submission) {
            if (profile_frames >= 65536 || frame->submission <= profile_frame) { return false; }
            common_json rows = common_json::array();
            uint64_t cells = uint64_t(batch.n_tokens) * batch.n_pos;
            for (uint32_t row = 0; row < batch.n_tokens; ++row) {
                if (batch.n_seq_id[row] <= 0 || !batch.seq_id[row] || uint64_t(batch.n_seq_id[row]) > (1u << 22) - cells) { return false; }
                cells += batch.n_seq_id[row];
                std::vector<llama_pos> positions(batch.n_pos);
                for (uint32_t axis = 0; axis < batch.n_pos; ++axis) { positions[axis] = batch.pos[size_t(axis) * batch.n_tokens + row]; }
                common_json item = {{"row", row}, {"positions", positions},
                    {"sequences", std::vector<llama_seq_id>(batch.seq_id[row], batch.seq_id[row] + batch.n_seq_id[row])}};
                item["token"] = batch.token ? common_json(batch.token[row]) : common_json(nullptr);
                item["output"] = batch.output ? common_json(bool(batch.output[row])) : common_json(nullptr);
                item["original_batch_index"] = batch.data && batch.data->batch_idxs.size() == batch.n_tokens ?
                    common_json(batch.data->batch_idxs[row]) : common_json(nullptr);
                rows.push_back(std::move(item));
            }
            if (!profile_trace_record({{"kind", "frame"}, {"schema", 1}, {"submission", frame->submission},
                    {"batched", frame->batched}, {"owner_namespace", certificate.owner_namespace},
                    {"owner_generation", certificate.owner_generation}, {"execution_domain", certificate.domain},
                    {"row_semantics", certificate.row_semantics}, {"execution_flags", certificate.flags},
                    {"n_rows", certificate.n_rows}, {"n_sequences", certificate.n_sequences}, {"rows", rows},
                    {"acceptance", "unknown at source publication"}, {"request_id", nullptr}})) { return false; }
            profile_frame = frame->submission; ++profile_frames;
        }
        const auto source = profile_collector.source_indices.find(std::make_pair(probe->source_domain, probe->source_witness));
        if (source == profile_collector.source_indices.end()) { return false; }
        const auto * ids = probe->ids;
        const bool mapped = ids->ne[0] > 0 && ids->ne[1] == batch.n_tokens && ids->ne[2] == 1 && ids->ne[3] == 1 &&
            uint64_t(ids->ne[0]) * batch.n_tokens == probe->n_routes;
        std::vector<int32_t> experts(probe->n_routes), ownership(probe->n_routes);
        for (uint32_t route = 0; route < probe->n_routes; ++route) {
            experts[route] = probe->expert_ids[probe->route_indices[route]];
            ownership[route] = probe->classes[probe->route_indices[route]];
        }
        return profile_trace_record({{"kind", "projection"}, {"submission", frame->submission}, {"backend_epoch", epoch},
            {"source_index", source->second}, {"source_domain", probe->source_domain}, {"projection_index", probe->layer},
            {"projection_count", probe->n_layers}, {"ids_shape", std::vector<int64_t>(ids->ne, ids->ne + 4)},
            {"ids_strides", std::vector<size_t>(ids->nb, ids->nb + 4)}, {"row_mapping", mapped ? "flat route / ids_shape[0]" : "unbound geometry"},
            {"experts", experts}, {"ownership", ownership}});
    }

    bool trace_blob(const std::string & name, const void * data, size_t bytes) {
        constexpr size_t limit = 256 * 1024 * 1024;
        if (!data || !bytes || bytes > limit - trace_bytes) { return false; }
        std::ofstream file(std::filesystem::path(trace_path) / name, std::ios::binary);
        if (!file.write(static_cast<const char *>(data), bytes)) { return false; }
        file.close();
        if (!file) { return false; }
        trace_bytes += bytes;
        return true;
    }

    static common_json trace_geometry(const ggml_tensor * tensor) {
        return {{"type", uint32_t(tensor->type)}, {"shape", std::vector<int64_t>(tensor->ne, tensor->ne + GGML_MAX_DIMS)},
            {"strides", std::vector<size_t>(tensor->nb, tensor->nb + GGML_MAX_DIMS)}};
    }

    bool trace_projection(uint64_t epoch, const ggml_cuda_moe_source_gpu_probe * probe) {
        if (trace_path.empty() || epoch < 2 || epoch > 7) { return true; }
        if (!probe || !probe->operation || !probe->activation || !probe->weight || !probe->input_data ||
                !probe->route_indices || !probe->expert_ids || !probe->classes ||
                !probe->n_layers || probe->n_layers > 1024 || probe->layer >= probe->n_layers ||
                !probe->n_routes || probe->n_routes > 64 || !probe->n_distinct || probe->n_distinct > probe->n_routes ||
                probe->operation->op != GGML_OP_MUL_MAT_ID || probe->operation->type != GGML_TYPE_F32 ||
                !probe->operation->buffer || !probe->operation->data || probe->operation->ne[0] <= 0 ||
                probe->operation->ne[0] > 65536 || !ggml_is_contiguous(probe->operation) || trace_records.size() >= 6144) {
            return false;
        }
        const size_t input_bytes = ggml_nbytes(probe->activation), output_bytes = ggml_nbytes(probe->operation);
        if (!input_bytes || input_bytes > 8 * 1024 * 1024 || probe->input_bytes < input_bytes ||
                output_bytes != size_t(probe->operation->ne[0]) * probe->n_routes * sizeof(float)) { return false; }
        std::vector<int32_t> ids(probe->n_routes), classes(probe->n_routes);
        for (uint32_t route = 0; route < probe->n_routes; ++route) {
            const uint32_t index = probe->route_indices[route];
            if (index >= probe->n_distinct) { return false; }
            ids[route] = probe->expert_ids[index]; classes[route] = probe->classes[index];
            if (ids[route] < 0 || ids[route] >= probe->weight->ne[2] || classes[route] < 0 || classes[route] > 2) { return false; }
        }
        const std::string prefix = std::to_string(epoch) + "-" + std::to_string(probe->layer) + "-" + std::to_string(trace_records.size());
        std::vector<float> output(output_bytes / sizeof(float));
        ggml_backend_tensor_get(probe->operation, output.data(), 0, output_bytes);
        for (float value : output) { if (!std::isfinite(value)) { return false; } }
        if (!trace_blob(prefix + "-input.bin", probe->input_data, input_bytes) ||
                !trace_blob(prefix + "-output.bin", output.data(), output_bytes)) { return false; }
        trace_records.push_back({{"epoch", epoch}, {"layer", probe->layer}, {"n_layers", probe->n_layers},
            {"name", probe->operation->name}, {"weight", trace_geometry(probe->weight)},
            {"activation", trace_geometry(probe->activation)}, {"output", trace_geometry(probe->operation)},
            {"input_file", prefix + "-input.bin"}, {"output_file", prefix + "-output.bin"}, {"ids", ids}, {"classes", classes}});
        return true;
    }

    bool trace_state(llama_context * ctx, uint32_t step) {
        if (trace_path.empty() || step > 6) { return true; }
        const size_t bytes = llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (!bytes || bytes > 64 * 1024 * 1024 || bytes > 256 * 1024 * 1024 - trace_bytes) { return false; }
        std::vector<uint8_t> state(bytes);
        if (llama_state_seq_get_data_ext(ctx, state.data(), bytes, 0, LLAMA_STATE_SEQ_FLAGS_NONE) != bytes) { return false; }
        const std::string name = "state-" + std::to_string(step) + ".bin";
        if (!trace_blob(name, state.data(), bytes)) { return false; }
        trace_states.push_back({{"step", step}, {"file", name}, {"bytes", bytes}, {"sha256", hash_sha256_hex(state.data(), bytes)}});
        return true;
    }

    static bool cpu_reference(ggml_backend_t backend_cpu, const ggml_tensor * operation,
            const ggml_tensor * source_weight, const ggml_tensor * source_activation,
            const void * weight_data, const void * input_data, const int32_t * logical_ids, std::vector<float> & values) {
        const auto * source_ids = operation->src[2];
        ggml_context_ptr context(ggml_init({16 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false), nullptr, true}));
        if (!context || !backend_cpu || !source_ids) { return false; }
        auto * weight = ggml_dup_tensor(context.get(), source_weight);
        auto * activation = ggml_dup_tensor(context.get(), source_activation);
        auto * ids = ggml_new_tensor_2d(context.get(), GGML_TYPE_I32, source_ids->ne[0], source_ids->ne[1]);
        memcpy(weight->nb, source_weight->nb, sizeof(weight->nb));
        memcpy(activation->nb, source_activation->nb, sizeof(activation->nb));
        weight->data = const_cast<void *>(weight_data);
        activation->data = const_cast<void *>(input_data);
        ids->data = const_cast<int32_t *>(logical_ids);
        auto * output = ggml_mul_mat_id(context.get(), weight, activation, ids);
        memcpy(output->op_params, operation->op_params, sizeof(output->op_params));
        if (memcmp(output->ne, operation->ne, sizeof(output->ne)) ||
                memcmp(output->nb, operation->nb, sizeof(output->nb)) || !ggml_backend_supports_op(backend_cpu, output)) { return false; }
        values.resize(ggml_nelements(output));
        output->data = values.data();
        auto * graph = ggml_new_graph_custom(context.get(), 16, false);
        ggml_build_forward_expand(graph, output);
        ggml_backend_cpu_set_n_threads(backend_cpu, 4);
        return ggml_backend_graph_compute(backend_cpu, graph) == GGML_STATUS_SUCCESS;
    }

    static bool observe_gpu(routed_cpu_oracle & oracle, uint64_t epoch, void * view) {
        if (!oracle.gpu_enabled) { return true; }
        const auto * probe = static_cast<const ggml_cuda_moe_source_gpu_probe *>(view);
        if (!probe || !probe->operation || !probe->activation || !probe->weight || !probe->weight_data || !probe->input_data ||
                !probe->route_indices || !probe->expert_ids || !probe->classes || !probe->n_layers || probe->n_layers > 65536 || probe->layer >= probe->n_layers ||
                !epoch || !probe->n_routes || probe->n_routes > 65536 || !probe->n_distinct || probe->n_distinct > probe->n_routes ||
                probe->operation->op != GGML_OP_MUL_MAT_ID || probe->operation->type != GGML_TYPE_F32 ||
                !probe->operation->buffer || oracle.gpu_records.size() >= 65536) { return false; }
        const uint32_t start = uint32_t(((epoch - 1) % probe->n_layers) * 3 % probe->n_layers);
        const uint32_t distance = (probe->layer + probe->n_layers - start) % probe->n_layers;
        if (distance >= std::min(3u, probe->n_layers)) { return true; }
        const auto * source_ids = probe->operation->src[2];
        if (!source_ids || source_ids->type != GGML_TYPE_I32 || source_ids->ne[0] <= 0 || source_ids->ne[1] <= 0 ||
                source_ids->ne[2] != 1 || source_ids->ne[3] != 1 || uint64_t(source_ids->ne[0]) > probe->n_routes ||
                uint64_t(source_ids->ne[1]) > probe->n_routes || uint64_t(source_ids->ne[0]) * uint64_t(source_ids->ne[1]) != probe->n_routes ||
                probe->weight->ne[2] <= 0 || probe->weight->ne[3] != 1 || probe->input_bytes < ggml_nbytes(probe->activation) ||
                ggml_nbytes(probe->weight) > 1024 * 1024 * 1024 || ggml_nbytes(probe->activation) > 64 * 1024 * 1024 ||
                ggml_nbytes(probe->operation) > 64 * 1024 * 1024) { return false; }
        std::vector<int32_t> logical_ids(probe->n_routes);
        std::vector<int32_t> route_classes(probe->n_routes);
        for (uint32_t route = 0; route < probe->n_routes; ++route) {
            const uint32_t index = probe->route_indices[route];
            if (index >= probe->n_distinct) { return false; }
            const int32_t id = probe->expert_ids[index], kind = probe->classes[index];
            if (id < 0 || id >= probe->weight->ne[2] || kind < 0 || kind > 2) { return false; }
            logical_ids[route] = id;
            route_classes[route] = kind;
        }
        const auto device = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(probe->operation->buffer));
        if (!device) { return false; }
        if (!oracle.gpu_backend || ggml_backend_get_device(oracle.gpu_backend.get()) != device) {
            oracle.gpu_backend.reset(ggml_backend_dev_init(device, nullptr));
        }
        if (!oracle.gpu_backend) { return false; }
        ggml_context_ptr weight_context(ggml_init({2 * ggml_tensor_overhead(), nullptr, true}));
        ggml_context_ptr context(ggml_init({16 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false), nullptr, true}));
        if (!weight_context || !context) { return false; }
        auto * weight = ggml_dup_tensor(weight_context.get(), probe->weight);
        auto * activation = ggml_dup_tensor(context.get(), probe->activation);
        auto * ids = ggml_new_tensor_2d(context.get(), GGML_TYPE_I32, source_ids->ne[0], source_ids->ne[1]);
        memcpy(weight->nb, probe->weight->nb, sizeof(weight->nb));
        memcpy(activation->nb, probe->activation->nb, sizeof(activation->nb));
        auto * output = ggml_mul_mat_id(context.get(), weight, activation, ids);
        memcpy(output->op_params, probe->operation->op_params, sizeof(output->op_params));
        if (memcmp(output->ne, probe->operation->ne, sizeof(output->ne)) ||
                memcmp(output->nb, probe->operation->nb, sizeof(output->nb)) ||
                !ggml_backend_supports_op(oracle.gpu_backend.get(), output)) { return false; }
        ggml_backend_buffer_ptr weights(ggml_backend_alloc_ctx_tensors(weight_context.get(), oracle.gpu_backend.get()));
        if (!weights) { return false; }
        ggml_backend_buffer_set_usage(weights.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(context.get(), oracle.gpu_backend.get()));
        if (!buffer) { return false; }
        ggml_backend_tensor_set(weight, probe->weight_data, 0, ggml_nbytes(weight));
        ggml_backend_tensor_set(activation, probe->input_data, 0, ggml_nbytes(activation));
        ggml_backend_tensor_set(ids, logical_ids.data(), 0, logical_ids.size() * sizeof(int32_t));
        auto * graph = ggml_new_graph_custom(context.get(), 16, false);
        ggml_build_forward_expand(graph, output);
        if (ggml_backend_graph_compute(oracle.gpu_backend.get(), graph) != GGML_STATUS_SUCCESS) { return false; }
        std::vector<float> expected(ggml_nelements(output)), actual(expected.size());
        ggml_backend_tensor_get(output, expected.data(), 0, expected.size() * sizeof(float));
        ggml_backend_tensor_get(probe->operation, actual.data(), 0, actual.size() * sizeof(float));
        if (output->ne[0] <= 0 || uint64_t(output->ne[0]) * probe->n_routes != expected.size()) { return false; }
        std::vector<float> expected_cpu;
        if (!cpu_reference(oracle.backend.get(), probe->operation, probe->weight, probe->activation,
                probe->weight_data, probe->input_data, logical_ids.data(), expected_cpu) || expected_cpu.size() != expected.size()) { return false; }
        double cpu_gpu_error = 0, hybrid_gpu_error = 0, mixed_error = 0, gpu_energy = 0;
        double cpu_gpu_maximum = 0, hybrid_gpu_maximum = 0;
        uint64_t mixed_mismatches = 0;
        for (size_t column = 0; column < expected.size(); ++column) {
            const float gpu = expected[column], cpu = expected_cpu[column], observed = actual[column];
            if (!std::isfinite(gpu) || !std::isfinite(cpu) || !std::isfinite(observed)) { return false; }
            const float mixed = route_classes[column / size_t(output->ne[0])] == 2 ? cpu : gpu;
            const double cpu_delta = double(cpu) - gpu, hybrid_delta = double(observed) - gpu, mixed_delta = double(observed) - mixed;
            cpu_gpu_error += cpu_delta * cpu_delta;
            hybrid_gpu_error += hybrid_delta * hybrid_delta;
            mixed_error += mixed_delta * mixed_delta;
            gpu_energy += double(gpu) * gpu;
            cpu_gpu_maximum = std::max(cpu_gpu_maximum, std::fabs(cpu_delta));
            hybrid_gpu_maximum = std::max(hybrid_gpu_maximum, std::fabs(hybrid_delta));
            mixed_mismatches += memcmp(&mixed, &observed, sizeof(float)) != 0;
        }
        oracle.projection_records.push_back({ {"epoch", epoch}, {"layer", probe->layer}, {"name", probe->operation->name},
            {"type", ggml_type_name(weight->type)}, {"routes", probe->n_routes}, {"gpu_energy", gpu_energy},
            {"cpu_gpu_squared_error", cpu_gpu_error}, {"cpu_gpu_nmse", cpu_gpu_error / std::max(gpu_energy, 1e-30)},
            {"cpu_gpu_max_abs_error", cpu_gpu_maximum}, {"hybrid_gpu_squared_error", hybrid_gpu_error},
            {"hybrid_gpu_nmse", hybrid_gpu_error / std::max(gpu_energy, 1e-30)}, {"hybrid_gpu_max_abs_error", hybrid_gpu_maximum},
            {"mixed_squared_error", mixed_error}, {"mixed_bit_mismatches", mixed_mismatches} });
        if (mixed_mismatches) { return false; }
        if (!oracle.projection_capture_path.empty() && !oracle.projection_captured && cpu_gpu_error > 5e-4 * gpu_energy) {
            const std::filesystem::path directory(oracle.projection_capture_path);
            if (!std::filesystem::create_directory(directory)) { return false; }
            struct blob { const char * name; const void * data; size_t bytes; };
            const blob blobs[] = {
                {"weight.bin", probe->weight_data, ggml_nbytes(probe->weight)},
                {"activation.bin", probe->input_data, ggml_nbytes(probe->activation)},
                {"ids.bin", logical_ids.data(), logical_ids.size() * sizeof(int32_t)},
                {"classes.bin", route_classes.data(), route_classes.size() * sizeof(int32_t)},
                {"cpu.bin", expected_cpu.data(), expected_cpu.size() * sizeof(float)},
                {"gpu.bin", expected.data(), expected.size() * sizeof(float)},
                {"observed.bin", actual.data(), actual.size() * sizeof(float)},
            };
            common_json capture = oracle.projection_records.back();
            capture["blobs"] = common_json::object();
            for (const auto & blob : blobs) {
                std::ofstream file(directory / blob.name, std::ios::binary);
                if (!file.write(static_cast<const char *>(blob.data), blob.bytes)) { return false; }
                file.close();
                if (!file) { return false; }
                capture["blobs"][blob.name] = blob.bytes;
            }
            const auto geometry = [](const ggml_tensor * tensor) {
                return common_json{{"type", uint32_t(tensor->type)}, {"type_name", ggml_type_name(tensor->type)},
                    {"shape", std::vector<int64_t>(tensor->ne, tensor->ne + GGML_MAX_DIMS)},
                    {"strides", std::vector<size_t>(tensor->nb, tensor->nb + GGML_MAX_DIMS)}};
            };
            capture["weight"] = geometry(probe->weight);
            capture["activation"] = geometry(probe->activation);
            capture["ids"] = geometry(ids);
            capture["source_ids"] = geometry(source_ids);
            capture["output"] = geometry(probe->operation);
            capture["op_params"] = std::vector<int32_t>(std::begin(probe->operation->op_params), std::end(probe->operation->op_params));
            capture["criterion"] = "first sampled ordinary CPU/GPU MUL_MAT_ID above existing 5e-4 limit; no tolerance waiver";
            std::ofstream metadata(directory / "CAPTURE.json");
            if (!(metadata << capture.dump(2) << '\n')) { return false; }
            metadata.close();
            if (!metadata) { return false; }
            oracle.projection_captured = true;
            fprintf(stderr, "test-moe-replay: captured original projection epoch=%llu layer=%u bytes=%zu\n",
                (unsigned long long) epoch, probe->layer, ggml_nbytes(probe->weight));
        }
        for (int32_t kind = 0; kind < 3; ++kind) {
            uint32_t owned = 0, mismatched = 0;
            double squared_error = 0, squared_reference = 0, maximum = 0;
            for (uint32_t route = 0; route < probe->n_routes; ++route) {
                if (route_classes[route] != kind) { continue; }
                ++owned;
                bool exact = true;
                const size_t offset = size_t(route) * size_t(output->ne[0]);
                for (int64_t column = 0; column < output->ne[0]; ++column) {
                    const float reference = expected[offset + size_t(column)], observed = actual[offset + size_t(column)];
                    if (!std::isfinite(reference) || !std::isfinite(observed)) { return false; }
                    exact &= memcmp(&reference, &observed, sizeof(float)) == 0;
                    const double delta = double(observed) - reference;
                    squared_error += delta * delta;
                    squared_reference += double(reference) * reference;
                    maximum = std::max(maximum, std::fabs(delta));
                }
                mismatched += !exact;
            }
            if (!owned) { continue; }
            oracle.gpu_records.push_back({ {"epoch", epoch}, {"layer", probe->layer}, {"layers", probe->n_layers},
                {"name", probe->operation->name}, {"type", ggml_type_name(weight->type)}, {"class", kind},
                {"logical_ids", logical_ids}, {"route_classes", route_classes}, {"device", ggml_backend_dev_name(device)},
                {"owned_routes", owned}, {"mismatched_routes", mismatched},
                {"squared_error", squared_error}, {"reference_energy", squared_reference},
                {"nmse", squared_error / std::max(squared_reference, 1e-30)}, {"max_abs_error", maximum} });
        }
        return true;
    }

    static bool observe(void * opaque, uint32_t phase, uint64_t epoch, void * view) {
        auto & oracle = *static_cast<routed_cpu_oracle *>(opaque);
        if (phase == GGML_BACKEND_MOE_HYBRID_TEST_SOURCE_ACCESS) {
            if (oracle.profile_export_path.empty()) { return true; }
            const auto * access = static_cast<const ggml_backend_moe_source_access_v1 *>(view);
            return oracle.profile_collector.observe(epoch, access) && oracle.trace_profile_projection(epoch, access);
        }
        if (phase == GGML_BACKEND_MOE_HYBRID_TEST_SOURCE_ROUTED_GPU_PUBLICATION_PROBE) {
            return oracle.trace_projection(epoch, static_cast<const ggml_cuda_moe_source_gpu_probe *>(view)) && observe_gpu(oracle, epoch, view);
        }
        if (!oracle.cpu_enabled) { return true; }
        const bool publication = phase == GGML_BACKEND_MOE_HYBRID_TEST_SOURCE_ROUTED_PUBLICATION_PROBE;
        if (!publication && phase != GGML_BACKEND_MOE_HYBRID_TEST_SOURCE_ROUTED_CPU_PROBE) { return true; }
        const auto * probe = static_cast<const ggml_cuda_moe_source_cpu_probe *>(view);
        if (!probe || !probe->operation || !probe->activation || !probe->weight || !probe->weight_data ||
                !probe->input_data || !probe->ids || !probe->ownership || !probe->output ||
                probe->operation->op != GGML_OP_MUL_MAT_ID || probe->operation->type != GGML_TYPE_F32 ||
                probe->input_bytes < ggml_nbytes(probe->activation) || probe->output_bytes < ggml_nbytes(probe->operation) ||
                !probe->n_routes || probe->n_routes > 65536 || oracle.records.size() >= 65536 || !oracle.backend) { return false; }
        const auto * source_ids = probe->operation->src[2];
        if (!source_ids || source_ids->ne[0] <= 0 || source_ids->ne[1] <= 0 || source_ids->ne[2] != 1 || source_ids->ne[3] != 1 ||
                uint64_t(source_ids->ne[0]) > probe->n_routes || uint64_t(source_ids->ne[1]) > probe->n_routes ||
                uint64_t(source_ids->ne[0]) * uint64_t(source_ids->ne[1]) != probe->n_routes) { return false; }
        std::vector<float> values;
        if (publication) {
            if (!ggml_is_contiguous(probe->operation)) { return false; }
            values.resize(ggml_nelements(probe->operation));
            ggml_backend_tensor_get(probe->operation, values.data(), 0, values.size() * sizeof(float));
        } else if (!cpu_reference(oracle.backend.get(), probe->operation, probe->weight, probe->activation,
                probe->weight_data, probe->input_data, probe->ids, values)) { return false; }
        const auto * output = probe->operation;
        const auto * weight = probe->weight;
        uint32_t owned = 0, mismatched = 0;
        double squared_error = 0, squared_reference = 0, maximum = 0;
        for (uint32_t route = 0; route < probe->n_routes; ++route) {
            if (!probe->ownership[route]) { continue; }
            ++owned;
            const size_t offset = size_t(route) * size_t(output->ne[0]);
            bool exact = true;
            for (int64_t column = 0; column < output->ne[0]; ++column) {
                const float expected = values[offset + size_t(column)], actual = probe->output[offset + size_t(column)];
                if (!std::isfinite(expected) || !std::isfinite(actual)) { return false; }
                exact &= memcmp(&expected, &actual, sizeof(float)) == 0;
                const double delta = double(actual) - expected;
                squared_error += delta * delta;
                squared_reference += double(expected) * expected;
                maximum = std::max(maximum, std::fabs(delta));
            }
            mismatched += !exact;
        }
        oracle.mismatched_routes += mismatched;
        oracle.records.push_back({ {"epoch", epoch}, {"layer", probe->layer}, {"phase", publication ? "publication" : "cpu"}, {"name", probe->operation->name},
            {"type", ggml_type_name(weight->type)}, {"owned_routes", owned}, {"mismatched_routes", mismatched},
            {"nmse", squared_error / std::max(squared_reference, 1e-30)}, {"max_abs_error", maximum} });
        return true;
    }
};

static int test_model_moe_concurrent_replay(llama_model * model, llama_context_params params,
        const std::vector<llama_token> & prompt, std::ifstream & input, const char * mode, const char * output_path) {
    const bool staggered = !strcmp(mode, "staggered") || !strcmp(mode, "rollback");
    const bool rollback = !strcmp(mode, "rollback");
    if ((!staggered && strcmp(mode, "parallel")) || !output_path || !output_path[0] ||
            prompt.size() >= 128 || std::filesystem::exists(output_path)) { return 1; }
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<llama_token> tokens(32);
    std::vector<std::vector<float>> expected(32, std::vector<float>(n_vocab));
    for (uint32_t step = 0; step < 32; ++step) {
        input.read(reinterpret_cast<char *>(&tokens[step]), sizeof(llama_token));
        input.read(reinterpret_cast<char *>(expected[step].data()), size_t(n_vocab) * sizeof(float));
        if (!input || tokens[step] < 0 || tokens[step] >= n_vocab) { return 1; }
        for (const auto value : expected[step]) { if (!std::isfinite(value)) { return 1; } }
    }
    if (input.peek() != std::char_traits<char>::eof()) { return 1; }
    std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
    const uint32_t header[] = {0x43525031, uint32_t(n_vocab), 32, 2};
    output.write(reinterpret_cast<const char *>(header), sizeof(header));
    if (!output) { return 1; }
    params.n_ctx = 1024;
    params.n_seq_max = 2;
    llama_context_ptr ctx(llama_init_from_model(model, params));
    if (!ctx) { return 1; }
    llama_batch batch = llama_batch_init(128, 0, 1);
    common_json rows = common_json::array();
    uint32_t completed[2] = {};
    std::vector<uint8_t> checkpoint;
    common_json state = common_json::object();
    bool restored = false;
    int status = 0;
    struct route_trace {
        uint32_t frame = 0, requested_first_step = 0;
        bool repeated = false;
        common_json records = common_json::array();
        static bool observe(void * opaque, uint32_t phase, uint64_t epoch, void * data) {
            if (phase != GGML_BACKEND_MOE_HYBRID_TEST_SOURCE_ROUTED_GPU_PUBLICATION_PROBE) { return true; }
            auto & self = *static_cast<route_trace *>(opaque);
            const auto & probe = *static_cast<const ggml_cuda_moe_source_gpu_probe *>(data);
            const auto * ids = probe.operation->src[2];
            if (!ids || ids->ne[0] <= 0 || ids->ne[1] <= 0 || uint64_t(ids->ne[0]) * ids->ne[1] != probe.n_routes) { return false; }
            std::vector<int32_t> logical(probe.n_routes), classes(probe.n_routes);
            bool has_cpu = false;
            for (uint32_t route = 0; route < probe.n_routes; ++route) {
                const auto w = probe.route_indices[route];
                if (w >= probe.n_distinct) { return false; }
                logical[route] = probe.expert_ids[w];
                classes[route] = probe.classes[w];
                has_cpu = has_cpu || classes[route] == 2;
            }
            const size_t row_bytes = std::min(probe.input_bytes, probe.activation->nb[2]);
            self.records.push_back({{"frame", self.frame}, {"requested_first_step", self.requested_first_step},
                {"repeated", self.repeated}, {"epoch", epoch}, {"projection", probe.layer},
                {"name", probe.operation->name}, {"local_rows", ids->ne[1]}, {"routes_per_row", ids->ne[0]},
                {"ids", logical}, {"classes", classes}, {"cpu_input_available", has_cpu},
                {"first_row_input_hash", has_cpu ? hash_sha256_hex(probe.input_data, row_bytes) : std::string{}}});
            return true;
        }
    } trace;
    const auto hook = [&]() {
        if (getenv("GGML_TEST_MOE_REPLAY_ROUTING")) {
            ggml_backend_sched_moe_hybrid_set_test_hook_v1(ctx->get_sched(), route_trace::observe, &trace);
        }
    };
    const auto snapshot = [&](llama_seq_id seq) {
        const size_t size = llama_state_seq_get_size_ext(ctx.get(), seq, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (!size || size > size_t(1024) * 1024 * 1024) { return std::vector<uint8_t>{}; }
        std::vector<uint8_t> bytes(size);
        if (llama_state_seq_get_data_ext(ctx.get(), bytes.data(), size, seq, LLAMA_STATE_SEQ_FLAGS_NONE) != size) { bytes.clear(); }
        return bytes;
    };
    const auto record = [&](llama_seq_id seq, uint32_t step, int32_t index, bool repeated) {
        const float * actual = llama_get_logits_ith(ctx.get(), index);
        if (!actual) { return false; }
        double error = 0.0, norm = 0.0, maximum = 0.0;
        llama_token top = 0;
        for (int32_t i = 0; i < n_vocab; ++i) {
            if (!std::isfinite(actual[i])) { return false; }
            const double delta = double(actual[i]) - expected[step][i];
            error += delta * delta;
            norm += double(expected[step][i]) * expected[step][i];
            maximum = std::max(maximum, std::fabs(delta));
            if (actual[i] > actual[top]) { top = i; }
        }
        const uint32_t identity[] = {uint32_t(seq), step, uint32_t(repeated)};
        output.write(reinterpret_cast<const char *>(identity), sizeof(identity));
        output.write(reinterpret_cast<const char *>(actual), size_t(n_vocab) * sizeof(float));
        rows.push_back({{"sequence", seq}, {"step", step}, {"repeated", repeated},
            {"top", top}, {"reference_top", tokens[step]}, {"nmse", error / std::max(norm, 1e-30)},
            {"max_abs_error", maximum}, {"logit_hash", hash_sha256_hex(actual, size_t(n_vocab) * sizeof(float))}});
        return bool(output);
    };
    const auto prefill = [&](llama_seq_id seq) {
        batch.n_tokens = 0;
        for (size_t i = 0; i < prompt.size(); ++i) {
            batch_add_compat(batch, prompt[i], i, {seq}, i + 1 == prompt.size());
        }
        if (llama_decode(ctx.get(), batch) || !record(seq, 0, -1, false)) { return false; }
        completed[seq] = 1;
        return true;
    };
    if (!prefill(0) || (!staggered && !prefill(1))) { status = 1; }
    while (!status && (completed[0] < 32 || completed[1] < 32)) {
        batch.n_tokens = 0;
        const bool starts_second = staggered && completed[0] == 8 && completed[1] == 0;
        std::vector<std::pair<llama_seq_id, uint32_t>> outputs;
        for (llama_seq_id seq = 0; seq < 2; ++seq) {
            if (!completed[seq] || completed[seq] == 32) { continue; }
            const uint32_t step = completed[seq];
            batch_add_compat(batch, tokens[step - 1], prompt.size() + step - 1, {seq}, true);
            outputs.push_back({seq, step});
        }
        if (starts_second) {
            for (size_t i = 0; i < prompt.size(); ++i) {
                batch_add_compat(batch, prompt[i], i, {1}, i + 1 == prompt.size());
            }
            outputs.push_back({1, 0});
        }
        ++trace.frame;
        trace.requested_first_step = completed[0];
        trace.repeated = restored;
        hook();
        if (batch.n_tokens == 0 || llama_decode(ctx.get(), batch)) { status = 1; break; }
        hook();
        for (size_t i = 0; i < outputs.size(); ++i) {
            const auto seq = outputs[i].first;
            const auto step = outputs[i].second;
            if (!record(seq, step, int32_t(i) - int32_t(outputs.size()), restored && seq == 0 && step >= 17 && step <= 20)) { status = 1; break; }
            completed[seq] = step + 1;
        }
        if (rollback && !restored && completed[0] == 17 && checkpoint.empty()) {
            checkpoint = snapshot(0);
            if (checkpoint.empty()) { status = 1; }
            state["checkpoint_size"] = checkpoint.size();
            state["checkpoint_hash"] = hash_sha256_hex(checkpoint.data(), checkpoint.size());
        }
        if (!status && rollback && !restored && completed[0] == 21) {
            const auto before = snapshot(1);
            if (before.empty() || !llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, -1, -1) ||
                    llama_state_seq_set_data_ext(ctx.get(), checkpoint.data(), checkpoint.size(), 0,
                        LLAMA_STATE_SEQ_FLAGS_NONE) != checkpoint.size()) { status = 1; break; }
            const auto after = snapshot(1);
            const auto restored_state = snapshot(0);
            state["restored_state_exact"] = restored_state == checkpoint;
            state["restored_state_hash"] = hash_sha256_hex(restored_state.data(), restored_state.size());
            if (restored_state != checkpoint) { status = 1; break; }
            state["survivor_state_unchanged"] = before == after;
            state["survivor_step"] = completed[1];
            if (before != after) { status = 1; break; }
            completed[0] = 17;
            restored = true;
        }
    }
    llama_batch_free(batch);
    output.flush();
    if (!output || (rollback && !restored)) { status = 1; }
    ctx.reset();
    common_log_flush(common_log_main());
    printf("test-moe-concurrent-replay: %s\n", common_json({{"mode", mode}, {"status", status},
        {"completed", common_json::array({completed[0], completed[1]})}, {"n_vocab", n_vocab}, {"rows", rows}, {"state", state}, {"routing", trace.records}}).dump().c_str());
    fflush(stdout);
    return status;
}

static std::string profile_read_text(const std::filesystem::path & path, size_t limit) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw std::runtime_error("cannot read corpus input: " + path.string()); }
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    if (error || !size || size > limit) { throw std::runtime_error("corpus input exceeds extent: " + path.string()); }
    std::string text(size_t(size), '\0');
    file.read(text.data(), text.size());
    if (!file || file.peek() != EOF) { throw std::runtime_error("corpus input changed while reading: " + path.string()); }
    if (text.find('\0') != std::string::npos) { throw std::runtime_error("corpus input contains NUL"); }
    return text;
}

static void profile_write_bytes(const std::filesystem::path & path, const void * data, size_t size) {
    std::ofstream file(path, std::ios::binary);
    file.write(static_cast<const char *>(data), size);
    file.close();
    if (!file) { throw std::runtime_error("cannot write profile artifact: " + path.string()); }
}

static int collect_model_moe_corpus(const char * model_path, const char * corpus_path, const char * output_path,
        int32_t cache_slots, uint32_t threads, llama_load_mode load_mode) {
    if (!corpus_path || !output_path || cache_slots <= 0) { throw std::runtime_error("corpus, output and positive cache slots are required"); }
    if (getenv("GGML_MOE_EXPERT_PROFILE")) { throw std::runtime_error("unset GGML_MOE_EXPERT_PROFILE while collecting a fresh profile"); }
    const auto text = profile_read_text(corpus_path, 8 * 1024 * 1024);
    const auto corpus = common_json::parse(text);
    if (!corpus.is_object() || corpus.size() != 3 || !corpus.contains("version") || corpus.at("version") != 1 ||
            !corpus.contains("weights") || !corpus.at("weights").is_object() || corpus.at("weights").empty() || corpus.at("weights").size() > 32 ||
            !corpus.contains("requests") || !corpus.at("requests").is_array() || corpus.at("requests").empty() || corpus.at("requests").size() > 128) {
        throw std::runtime_error("invalid corpus schema; expected version1, weights and requests");
    }
    std::map<std::string, double> weights;
    std::map<std::string, size_t> family_requests;
    for (auto it = corpus.at("weights").begin(); it != corpus.at("weights").end(); ++it) {
        const auto & name = it.key();
        if (name.empty() || name.size() > 64 || name.find('\0') != std::string::npos || !it.value().is_number()) { throw std::runtime_error("invalid workload label or weight"); }
        const double weight = it.value().get<double>();
        if (!std::isfinite(weight) || weight < 0) { throw std::runtime_error("invalid workload weight"); }
        weights.emplace(name, weight);
    }
    struct request { std::string family, prompt; uint32_t max_tokens; };
    std::vector<request> requests;
    for (const auto & item : corpus.at("requests")) {
        if (!item.is_object() || item.size() != 3 || !item.contains("family") || !item.at("family").is_string() ||
                !item.contains("max_tokens") || !item.at("max_tokens").is_number_integer() ||
                item.contains("prompt") == item.contains("prompt_file")) { throw std::runtime_error("invalid corpus request schema"); }
        const auto family = item.at("family").get<std::string>();
        const int64_t tokens = item.at("max_tokens").get<int64_t>();
        if (!weights.count(family) || tokens < 2 || tokens > 4096) { throw std::runtime_error("unknown workload or continuation outside 2..4096"); }
        const auto & prompt_value = item.at(item.contains("prompt") ? "prompt" : "prompt_file");
        if (!prompt_value.is_string()) { throw std::runtime_error("invalid corpus prompt"); }
        auto prompt = prompt_value.get<std::string>();
        if (item.contains("prompt_file")) {
            if (prompt.empty() || prompt.find('\0') != std::string::npos) { throw std::runtime_error("invalid prompt file path"); }
            prompt = profile_read_text(std::filesystem::path(corpus_path).parent_path() / prompt, 64 * 1024);
        }
        if (prompt.empty() || prompt.size() > 64 * 1024 || prompt.find('\0') != std::string::npos) { throw std::runtime_error("prompt outside 1..65536 bytes"); }
        requests.push_back({family, prompt, uint32_t(tokens)});
        ++family_requests[family];
    }
    if (family_requests.size() != weights.size()) { throw std::runtime_error("a workload has no requests"); }
    profile_rate_sum weight_total;
    for (const auto & weight : weights) { weight_total.add(weight.second); }
    if (!std::isfinite(weight_total.value()) || weight_total.value() <= 0) { throw std::runtime_error("workload weights must have a positive finite sum"); }
    const std::filesystem::path output(output_path), evidence(std::string(output_path) + ".calibration"), partial(std::string(output_path) + ".partial");
    if (std::filesystem::exists(output) || std::filesystem::exists(partial) || !std::filesystem::create_directory(evidence)) {
        throw std::runtime_error("use fresh profile and evidence paths");
    }
    profile_write_bytes(evidence / "corpus.json", text.data(), text.size());
    const auto corpus_hash = hash_sha256_hex(text.data(), text.size());
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;
    model_params.split_mode = LLAMA_SPLIT_MODE_LAYER;
    model_params.load_mode = load_mode;
    model_params.moe_expert_cache_slots = cache_slots;
    model_params.moe_expert_cache_host_pinned_size = 0;
    model_params.progress_callback = silent_model_load_progress;
    llama_model_ptr model(llama_model_load_from_file(model_path, model_params));
    if (!model) { throw std::runtime_error("cannot load corpus model"); }
    routed_profile_collector catalog;
    if (!catalog.initialize(model->moe_sources())) { throw std::runtime_error("no eligible expert source catalog"); }
    profile_corpus_average average;
    average.initialize(catalog.statistics, weights);
    const auto * vocab = llama_model_get_vocab(model.get());
    const auto n_vocab = llama_vocab_n_tokens(vocab);
    if (n_vocab <= 0 || n_vocab > 1024 * 1024) { throw std::runtime_error("invalid model vocabulary extent"); }
    common_json weight_json = common_json::object();
    for (const auto & weight : weights) { weight_json[weight.first] = weight.second; }
    common_json report = {{"version", 1}, {"status", "incomplete"}, {"corpus_sha256", corpus_hash}, {"weights", weight_json},
        {"formula", "sum(normalized workload weight * mean(request expert counts / actual target decode rows))"},
        {"model", model_path}, {"cache_slots", cache_slots}, {"threads", threads}, {"load_mode", llama_load_mode_name(load_mode)},
        {"sampling", "greedy; natural EOG; serial fresh contexts; no draft or MTP calibration"}, {"requests", common_json::array()}};
    uint64_t total_rows = 0;
    for (size_t index = 0; index < requests.size(); ++index) {
        const auto & request = requests[index];
        const auto prompt = common_tokenize(vocab, request.prompt, true, true);
        if (prompt.empty() || prompt.size() > 4096) { throw std::runtime_error("prompt outside 1..4096 tokens"); }
        const auto prefix = evidence / std::to_string(index);
        routed_cpu_oracle observer;
        if (!observer.profile_collector.initialize(model->moe_sources())) { throw std::runtime_error("invalid request source catalog"); }
        observer.profile_export_path = prefix.string() + ".gguf";
        observer.profile_trace_path = prefix.string() + ".routes.jsonl";
        observer.profile_trace.open(observer.profile_trace_path, std::ios::binary);
        if (!observer.profile_trace) { throw std::runtime_error("cannot open routing evidence"); }
        auto params = llama_context_default_params();
        params.n_ctx = std::max<uint32_t>(512, uint32_t(prompt.size()) + request.max_tokens);
        params.n_batch = params.n_ubatch = std::max<uint32_t>(128, uint32_t(prompt.size()));
        params.n_threads = params.n_threads_batch = threads;
        params.phase_aware_workspace = true;
        params.no_perf = false;
        llama_context_ptr ctx(llama_init_from_model(model.get(), params));
        if (!ctx) { throw std::runtime_error("cannot create calibration context"); }
        auto prefill = llama_batch_get_one(const_cast<llama_token *>(prompt.data()), int32_t(prompt.size()));
        if (llama_decode(ctx.get(), prefill)) { throw std::runtime_error("calibration prefill failed"); }
        observer.profile_context = ctx.get();
        if (!ctx->set_moe_test_hook(routed_cpu_oracle::observe, &observer)) { throw std::runtime_error("corpus collection requires the generic source executor"); }
        std::vector<llama_token> emitted;
        uint32_t rows = 0;
        bool eog = false;
        for (uint32_t step = 0; step < request.max_tokens; ++step) {
            const auto * logits = llama_get_logits_ith(ctx.get(), -1);
            if (!logits) { throw std::runtime_error("missing calibration logits"); }
            llama_token token = 0;
            for (int32_t id = 0; id < n_vocab; ++id) {
                if (!std::isfinite(logits[id])) { throw std::runtime_error("nonfinite calibration logits"); }
                if (logits[id] > logits[token]) { token = id; }
            }
            if (llama_vocab_is_eog(vocab, token)) { eog = true; break; }
            emitted.push_back(token);
            if (step + 1 == request.max_tokens) { break; }
            auto decode = llama_batch_get_one(&token, 1);
            if (llama_decode(ctx.get(), decode) || ctx->get_moe_test_frame()) { throw std::runtime_error("calibration decode failed or caller frame retained"); }
            ++rows;
        }
        ctx->synchronize();
        if (!ctx->set_moe_test_hook(nullptr, nullptr) || !rows || observer.profile_frames != rows || !observer.profile_collector.projections) {
            throw std::runtime_error("request has no complete target decode observations");
        }
        observer.profile_trace.close();
        if (!observer.profile_trace) { throw std::runtime_error("cannot finish routing evidence"); }
        const auto trace = profile_read_text(observer.profile_trace_path, 256 * 1024 * 1024);
        auto & statistics = observer.profile_collector.statistics;
        statistics.provenance = common_json({{"producer", "native corpus source observer"}, {"corpus_sha256", corpus_hash},
            {"request", index}, {"workload", request.family}, {"actual_decode_rows", rows}, {"generated_tokens", emitted.size()},
            {"stop", eog ? "EOG" : "limit"}, {"routes_sha256", hash_sha256_hex(trace.data(), trace.size())}}).dump();
        const auto metadata = make_source_profile_statistics(model->moe_sources(), false, &statistics);
        const auto bytes = source_profile_metadata_bytes(metadata.get());
        const auto decoded = llama_moe_profile_statistics_parse(bytes.data(), bytes.size(), model->moe_sources());
        for (size_t i = 0; i < statistics.sources.size(); ++i) {
            if (decoded.sources[i].counts != statistics.sources[i].counts) { throw std::runtime_error("request profile roundtrip failed"); }
        }
        profile_write_bytes(observer.profile_export_path, bytes.data(), bytes.size());
        const auto perf = llama_perf_context(ctx.get());
        average.add(request.family, statistics, rows);
        total_rows += rows;
        report["requests"].push_back({{"request", index}, {"family", request.family}, {"prompt_tokens", prompt.size()},
            {"prompt_sha256", hash_sha256_hex(request.prompt.data(), request.prompt.size())},
            {"prompt_tokens_sha256", hash_sha256_hex(prompt.data(), prompt.size() * sizeof(llama_token))},
            {"max_tokens", request.max_tokens}, {"actual_decode_rows", rows}, {"generated_tokens", emitted}, {"stop", eog ? "EOG" : "limit"},
            {"profile_sha256", hash_sha256_hex(bytes.data(), bytes.size())}, {"routes_sha256", hash_sha256_hex(trace.data(), trace.size())},
            {"ownership_counts", std::vector<uint64_t>(observer.profile_collector.ownership_counts.begin(), observer.profile_collector.ownership_counts.end())}, {"projection_records", observer.profile_collector.projections},
            {"prompt_ms", perf.t_p_eval_ms}, {"decode_ms", perf.t_eval_ms}});
        ctx.reset();
        const auto progress = report.dump(2) + '\n';
        profile_write_bytes(evidence / "REPORT.json", progress.data(), progress.size());
        fprintf(stderr, "moe-profile-corpus: request=%zu/%zu workload=%s decode_rows=%u emitted=%zu stop=%s\n",
            index + 1, requests.size(), request.family.c_str(), rows, emitted.size(), eog ? "EOG" : "limit");
    }
    average.finish();
    average.statistics.provenance = common_json({{"producer", "test-llama-archs native corpus collector"},
        {"version", 1}, {"corpus_sha256", corpus_hash}, {"requests", requests.size()}, {"actual_decode_rows", total_rows},
        {"weights", weight_json}, {"formula", report.at("formula")}, {"sampling", report.at("sampling")},
        {"evidence", "Adjacent .calibration directory contains per-request raw GGUF statistics, routing JSONL and REPORT.json"}}).dump();
    const auto metadata = make_source_profile_statistics(model->moe_sources(), false, &average.statistics);
    const auto bytes = source_profile_metadata_bytes(metadata.get());
    const auto decoded = llama_moe_profile_statistics_parse(bytes.data(), bytes.size(), model->moe_sources());
    for (size_t i = 0; i < decoded.sources.size(); ++i) {
        if (decoded.sources[i].counts != average.statistics.sources[i].counts || decoded.sources[i].scores != average.statistics.sources[i].scores) {
            throw std::runtime_error("weighted profile roundtrip failed");
        }
    }
    profile_write_bytes(partial, bytes.data(), bytes.size());
    report["status"] = "complete";
    report["profile_sha256"] = hash_sha256_hex(bytes.data(), bytes.size());
    report["actual_decode_rows"] = total_rows;
    const auto complete = report.dump(2) + '\n';
    profile_write_bytes(evidence / "REPORT.json", complete.data(), complete.size());
    model.reset();
    std::filesystem::rename(partial, output);
    printf("moe-profile-corpus: wrote %s requests=%zu rows=%llu sources=%zu bytes=%zu\n", output_path,
        requests.size(), (unsigned long long) total_rows, decoded.sources.size(), bytes.size());
    return 0;
}

static int test_model_moe_replay(const char * path, const char * reference_path, bool replay, bool independent_sources, int32_t cache_slots, bool cpu_oracle, bool gpu_oracle, llama_load_mode load_mode, const char * prompt_path, uint32_t replay_rows, const std::string & corpus_split) {
    if (!reference_path || !replay_rows || replay_rows > 4096 ||
            (getenv("GGML_TEST_MOE_PROFILE_EXPORT") && replay_rows < 2) ||
            (getenv("GGML_TEST_MOE_PROFILE_LIFECYCLE") && replay_rows < 32) ||
            (getenv("GGML_TEST_MOE_REPLAY_TRACE") && replay_rows < 8) ||
            (getenv("GGML_TEST_MOE_REPLAY_CONCURRENCY") && replay_rows != 32)) { return 1; }
    std::string prompt_text = "Write a short explanation of why the sky is blue.";
    if (prompt_path) {
        std::ifstream file(prompt_path, std::ios::binary);
        if (!file) { return 1; }
        prompt_text.resize(64 * 1024 + 1);
        file.read(prompt_text.data(), prompt_text.size());
        const auto bytes = file.gcount();
        if (!file.eof() || file.bad() || bytes <= 0 || bytes > 64 * 1024) { return 1; }
        prompt_text.resize(size_t(bytes));
        if (prompt_text.find('\0') != std::string::npos) { return 1; }
    }
    const std::string prompt_hash = hash_sha256_hex(prompt_text.data(), prompt_text.size());
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;
    model_params.split_mode = LLAMA_SPLIT_MODE_LAYER;
    model_params.load_mode = load_mode;
    model_params.moe_expert_cache_slots = cache_slots;
    model_params.moe_expert_cache_host_pinned_size = 0;
    model_params.progress_callback = silent_model_load_progress;
    llama_model_ptr model(llama_model_load_from_file(path, model_params));
    if (!model) { return 1; }
    if (independent_sources) {
        auto layers = std::move(model->layers);
        model->layers.clear();
        model->build_moe_sources();
        model->layers = std::move(layers);
        std::unordered_set<const ggml_tensor *> routed;
        for (const auto & use : model->tensor_uses()) {
            if (use.op == GGML_OP_MUL_MAT_ID) { routed.insert(use.tensor); }
        }
        std::unordered_set<const ggml_tensor *> described;
        for (const auto & group : model->moe_sources()) {
            if (group.layout != GGML_BACKEND_MOE_CANDIDATE_LAYOUT_ROUTED_MATRIX || group.banks.size() != 1 ||
                    group.banks[0].role != GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_ROUTED_WEIGHT ||
                    !described.insert(group.banks[0].tensor).second) { return 1; }
        }
        if (routed.empty() || described != routed) { return 1; }
        fprintf(stderr, "test-moe-replay: independent source catalog covers%zu original operands\n", described.size());
    }
    const auto * vocab = llama_model_get_vocab(model.get());
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    if (n_vocab <= 0 || n_vocab > 1024 * 1024) { return 1; }
    const auto prompt = common_tokenize(vocab, prompt_text, true, true);
    if (prompt.empty() || prompt.size() > 4096 ||
            (getenv("GGML_TEST_MOE_REPLAY_CONCURRENCY") && prompt.size() > 128)) { return 1; }
    const uint32_t header[] = { 0x4d525031, uint32_t(n_vocab), replay_rows, uint32_t(prompt.size()) };
    const uint64_t reference_bytes = sizeof(header) + uint64_t(prompt.size()) * sizeof(llama_token) +
        uint64_t(replay_rows) * (sizeof(llama_token) + uint64_t(n_vocab) * sizeof(float));
    if (reference_bytes > 512 * 1024 * 1024) { return 1; }
    std::ifstream input;
    std::ofstream output;
    if (replay) {
        std::error_code error;
        if (std::filesystem::file_size(reference_path, error) != reference_bytes || error) { return 1; }
        input.open(reference_path, std::ios::binary);
        uint32_t saved_header[4] = {};
        input.read(reinterpret_cast<char *>(saved_header), sizeof(saved_header));
        std::vector<llama_token> saved_prompt(prompt.size());
        input.read(reinterpret_cast<char *>(saved_prompt.data()), saved_prompt.size() * sizeof(llama_token));
        if (!input || std::memcmp(header, saved_header, sizeof(header)) != 0 || saved_prompt != prompt) { return 1; }
    } else {
        output.open(reference_path, std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char *>(header), sizeof(header));
        output.write(reinterpret_cast<const char *>(prompt.data()), prompt.size() * sizeof(llama_token));
        if (!output) { return 1; }
    }
    routed_cpu_oracle oracle;
    oracle.cpu_enabled = cpu_oracle;
    oracle.gpu_enabled = gpu_oracle;
    if (const char * path = getenv("GGML_TEST_MOE_PROFILE_EXPORT")) {
        if (!*path || std::filesystem::exists(path) || !oracle.profile_collector.initialize(model->moe_sources())) { return 1; }
        oracle.profile_export_path = path;
        oracle.profile_trace_path = std::string(path) + ".routes.jsonl";
        if (std::filesystem::exists(oracle.profile_trace_path)) { return 1; }
        oracle.profile_trace.open(oracle.profile_trace_path, std::ios::binary);
        if (!oracle.profile_trace) { return 1; }
    }
    if (const char * trace = getenv("GGML_TEST_MOE_REPLAY_TRACE")) {
        if (!replay || !*trace || !std::filesystem::create_directory(trace)) { return 1; }
        oracle.trace_path = trace;
    }
    if (const char * capture = getenv("GGML_TEST_MOE_REPLAY_PROJECTION_CAPTURE")) {
        if (!gpu_oracle || !*capture || std::filesystem::exists(capture)) { return 1; }
        oracle.projection_capture_path = capture;
    }
    if (cpu_oracle || gpu_oracle) { oracle.backend.reset(ggml_backend_cpu_init()); }
    bool oracle_hooked = false;
    auto params = llama_context_default_params();
    params.n_ctx = std::max<uint32_t>(512, uint32_t(prompt.size()) + replay_rows);
    params.n_batch = params.n_ubatch = std::max<uint32_t>(128, uint32_t(prompt.size()));
    params.n_threads = params.n_threads_batch = 4;
    params.phase_aware_workspace = true;
    params.no_perf = false;
    if (const char * mode = getenv("GGML_TEST_MOE_REPLAY_CONCURRENCY")) {
        if (!replay || cpu_oracle || gpu_oracle || !oracle.profile_export_path.empty()) { return 1; }
        return test_model_moe_concurrent_replay(model.get(), params, prompt, input, mode,
            getenv("GGML_TEST_MOE_REPLAY_CONCURRENT_OUTPUT"));
    }
    llama_context_ptr ctx(llama_init_from_model(model.get(), params));
    if (!ctx) { return 1; }
    const bool profile_lifecycle = getenv("GGML_TEST_MOE_PROFILE_LIFECYCLE") != nullptr;
    uint64_t paused_projections = 0;
    if (profile_lifecycle && oracle.profile_export_path.empty()) { return 1; }
    if (!oracle.profile_export_path.empty()) {
        oracle.profile_context = ctx.get();
        if (ctx->get_moe_test_frame()) { return 1; }
        if (!ctx->begin_source_call()) { return 1; }
        const bool active_set = ctx->set_moe_test_hook(routed_cpu_oracle::observe, &oracle);
        ctx->end_source_call();
        if (active_set) { return 1; }
        oracle_hooked = ctx->set_moe_test_hook(routed_cpu_oracle::observe, &oracle);
        if (!oracle_hooked) { return 1; }
    }
    llama_batch batch = llama_batch_init(params.n_batch, 0, 1);
    for (size_t i = 0; i < prompt.size(); ++i) {
        batch_add_compat(batch, prompt[i], i, { 0 }, i + 1 == prompt.size());
    }
    std::vector<float> expected(n_vocab);
    common_json rows = common_json::array();
    common_json logit_hashes = common_json::array();
    std::vector<llama_token> emitted_tokens;
    double squared_error = 0.0;
    double squared_reference = 0.0;
    uint32_t agreements = 0;
    int status = 0;
    for (uint32_t step = 0; step < header[2]; ++step) {
        if (profile_lifecycle && step == 8) { ctx->sched_reserve(params.n_batch); }
        if (profile_lifecycle && step == 16) {
            paused_projections = oracle.profile_collector.projections;
            if (!ctx->set_moe_test_hook(nullptr, nullptr)) { status = 1; break; }
        }
        if (profile_lifecycle && step == 24 && !ctx->set_moe_test_hook(routed_cpu_oracle::observe, &oracle)) { status = 1; break; }
        if (llama_decode(ctx.get(), batch) != 0) {
            if (!oracle.profile_export_path.empty()) {
                fprintf(stderr, "test-moe-profile-frame: failed decode frame_cleared=%d\n", ctx->get_moe_test_frame() == nullptr);
            }
            status = 1; break;
        }
        if (!oracle.profile_export_path.empty() && ctx->get_moe_test_frame()) { status = 1; break; }
        if (profile_lifecycle && step >= 16 && step < 24 && oracle.profile_collector.projections != paused_projections) { status = 1; break; }
        if ((cpu_oracle || gpu_oracle || !oracle.trace_path.empty() || !oracle.profile_export_path.empty()) && !oracle_hooked) {
            oracle_hooked = ggml_backend_sched_moe_hybrid_set_test_hook_v1(ctx->get_sched(), routed_cpu_oracle::observe, &oracle);
        }
        if (!oracle.trace_state(ctx.get(), step)) { status = 1; break; }
        const float * actual = llama_get_logits_ith(ctx.get(), -1);
        if (!actual) { status = 1; break; }
        llama_token actual_top = 0;
        for (int32_t i = 0; i < n_vocab; ++i) {
            if (!std::isfinite(actual[i])) { status = 1; break; }
            if (actual[i] > actual[actual_top]) { actual_top = i; }
        }
        if (status) { break; }
        logit_hashes.push_back(hash_sha256_hex(actual, size_t(n_vocab) * sizeof(float)));
        llama_token token = actual_top;
        if (replay) {
            input.read(reinterpret_cast<char *>(&token), sizeof(token));
            input.read(reinterpret_cast<char *>(expected.data()), expected.size() * sizeof(float));
            if (!input || token < 0 || token >= n_vocab) { status = 1; break; }
            double error = 0.0;
            double norm = 0.0;
            double max_error = 0.0;
            float second = -INFINITY;
            for (int32_t i = 0; i < n_vocab; ++i) {
                if (!std::isfinite(expected[i]) || expected[i] > expected[token]) { status = 1; break; }
                const double delta = double(actual[i]) - expected[i];
                error += delta * delta;
                norm += double(expected[i]) * expected[i];
                max_error = std::max(max_error, std::fabs(delta));
                if (i != token) { second = std::max(second, expected[i]); }
            }
            if (status) { break; }
            squared_error += error;
            squared_reference += norm;
            agreements += actual_top == token;
            rows.push_back({ { "step", step }, { "reference_top", token }, { "actual_top", actual_top },
                { "reference_margin", expected[token] - second }, { "max_abs_error", max_error },
                { "nmse", norm > 0.0 ? error / norm : error } });
        } else {
            output.write(reinterpret_cast<const char *>(&token), sizeof(token));
            output.write(reinterpret_cast<const char *>(actual), n_vocab * sizeof(float));
            if (!output) { status = 1; break; }
        }
        emitted_tokens.push_back(token);
        batch.n_tokens = 0;
        batch_add_compat(batch, token, prompt.size() + step, { 0 }, true);
    }
    llama_batch_free(batch);
    if (replay && status == 0 && input.peek() != std::char_traits<char>::eof()) { status = 1; }
    if (!replay) { output.flush(); if (!output) { status = 1; } }
    if (cpu_oracle && (!oracle_hooked || oracle.records.empty() || oracle.mismatched_routes)) { status = 1; }
    if (gpu_oracle && (!oracle_hooked || oracle.gpu_records.empty())) { status = 1; }
    if (!oracle.projection_capture_path.empty() && !oracle.projection_captured) { status = 1; }
    if (!oracle.trace_path.empty()) {
        if (!oracle_hooked || oracle.trace_records.empty() || oracle.trace_states.size() != 7) { status = 1; }
        std::ofstream trace(std::filesystem::path(oracle.trace_path) / "TRACE.json");
        if (!(trace << common_json{{"records", oracle.trace_records}, {"states", oracle.trace_states},
                {"bytes", oracle.trace_bytes}, {"status", status}}.dump(2) << '\n')) { status = 1; }
        trace.close();
        if (!trace) { status = 1; }
    }
    common_json profile_collection;
    if (!oracle.profile_export_path.empty()) {
        auto & collector = oracle.profile_collector;
        oracle.profile_trace.close();
        if (!oracle_hooked || !collector.projections || !oracle.profile_frames || !oracle.profile_trace) { status = 1; }
        if (!status) {
            std::ifstream trace_file(oracle.profile_trace_path, std::ios::binary);
            std::vector<char> trace_bytes(std::istreambuf_iterator<char>{trace_file}, std::istreambuf_iterator<char>{});
            if (!trace_file || trace_bytes.size() != oracle.profile_trace_bytes) { status = 1; }
            const std::string trace_hash = hash_sha256_hex(trace_bytes.data(), trace_bytes.size());
            collector.statistics.provenance = common_json({{"producer", "test-llama-archs generic routed publication observer"},
                {"sampling", "all original source projection occurrences in decode epochs after prefill; all ownership classes"},
                {"backend_epoch_scope", "session-local; not a token or request position"}, {"lifecycle_test", profile_lifecycle},
                {"routing_sidecar", {{"format", "moe-routing-jsonl-v1"}, {"sha256", trace_hash}, {"bytes", oracle.profile_trace_bytes},
                    {"frames", oracle.profile_frames}, {"projection_records", collector.projections}}},
                {"prompt_token_count", prompt.size()}, {"prompt_sha256", prompt_hash},
                {"prompt_token_ids_sha256", hash_sha256_hex(prompt.data(), prompt.size() * sizeof(llama_token))}, {"corpus_split", corpus_split},
                {"teacher_rows", replay_rows}, {"teacher_replay", replay}, {"generation", replay ? "reference tokens" : "greedy original logits"},
                {"stop_policy", "fixed rows; EOG does not stop the diagnostic"}, {"first_epoch", collector.first_epoch},
                {"last_epoch", collector.last_epoch}, {"projections", collector.projections}, {"ownership_counts", std::vector<uint64_t>(collector.ownership_counts.begin(), collector.ownership_counts.end())},
                {"omissions", common_json::array({"prefill", "draft", "MTP acceptance", "representative corpus and held-out quality"})}}).dump();
            auto metadata = make_source_profile_statistics(model->moe_sources(), false, &collector.statistics);
            const auto bytes = source_profile_metadata_bytes(metadata.get());
            const auto decoded = llama_moe_profile_statistics_parse(bytes.data(), bytes.size(), model->moe_sources());
            if (decoded.sources.size() != collector.statistics.sources.size() || bytes.size() > 64 * 1024 * 1024) { status = 1; }
            for (size_t i = 0; !status && i < decoded.sources.size(); ++i) {
                const auto & source = collector.statistics.sources[i]; const auto & result = decoded.sources[i];
                if (result.tensor != source.tensor || result.domain != source.domain || result.observations != source.observations || result.counts != source.counts) { status = 1; }
            }
            if (!status) {
                std::ofstream profile(oracle.profile_export_path, std::ios::binary);
                if (!profile.write(reinterpret_cast<const char *>(bytes.data()), bytes.size())) { status = 1; }
                profile.close(); if (!profile) { status = 1; }
            }
            profile_collection = {{"file", oracle.profile_export_path}, {"sources", decoded.sources.size()}, {"bytes", bytes.size()},
                {"routing_sidecar", oracle.profile_trace_path}, {"routing_sha256", trace_hash}, {"routing_bytes", oracle.profile_trace_bytes}, {"frames", oracle.profile_frames},
                {"projections", collector.projections}, {"ownership_counts", std::vector<uint64_t>(collector.ownership_counts.begin(), collector.ownership_counts.end())}, {"first_epoch", collector.first_epoch}, {"last_epoch", collector.last_epoch}};
        }
    }
    if (!oracle.profile_export_path.empty() && !ctx->set_moe_test_hook(nullptr, nullptr)) { status = 1; }
    const auto perf = llama_perf_context(ctx.get());
    common_json report = { { "replay", replay }, { "independent_sources", independent_sources },
        { "source_groups", model->moe_sources().size() }, { "cache_slots", cache_slots }, { "status", status }, { "n_vocab", n_vocab },
        { "prompt_tokens", prompt }, { "prompt_sha256", prompt_hash }, { "corpus_split", corpus_split },
        { "requested_rows", replay_rows }, { "emitted_tokens", emitted_tokens }, { "top_agreements", agreements }, { "rows", rows },
        { "nmse", squared_reference > 0.0 ? squared_error / squared_reference : squared_error },
        { "decode_ms", perf.t_eval_ms }, { "decode_tokens", perf.n_eval }, { "logit_hashes", logit_hashes }, { "cpu_oracle", oracle.records }, { "cpu_oracle_mismatched_routes", oracle.mismatched_routes },
        { "gpu_oracle", oracle.gpu_records }, { "projection_references", oracle.projection_records },
        { "load_mode", llama_load_mode_name(load_mode) }, { "profile_collection", profile_collection } };
    ctx.reset();
    model.reset();
    common_log_flush(common_log_main());
    printf("test-moe-replay: %s\n", report.dump().c_str());
    fflush(stdout);
    return status;
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
        ggml_backend_sched_eval_callback cb_eval = nullptr, void * cb_eval_user_data = nullptr, uint32_t n_outputs_max = 4);

static llama_model_ptr make_live_context_workspace_model(llm_arch arch, size_t seed, float stdev, bool moe = false) {
    gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    ggml_backend_dev_t devices[] = { nullptr };
    model_params.devices = devices;

    tensor_data_params tensor_params = { seed, stdev };
    llama_model_ptr model(llama_model_init_from_user(gguf_ctx.get(), set_tensor_data, &tensor_params, model_params));
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

static void test_live_context_workspace_reserve(size_t seed, float stdev) {
    llama_model_ptr model = make_live_context_workspace_model(LLM_ARCH_LLAMA, seed, stdev);
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
            batch_add_compat(batch, tokens[pos], pos, { 0 }, i + 1 == (int32_t) ctx_params.n_batch);
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
    batch_add_compat(shifted, tokens[pos], pos - 512, { 0 }, true);
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
    batch_add_compat(short_batch, tokens[0], 0, { 0 }, true);
    GGML_ASSERT(llama_decode(ctx.get(), short_batch) == 0);
    llama_synchronize(ctx.get());
    llama_batch_free(short_batch);

    const auto runtime_contraction = ctx->make_sched_reserve_plan(0);
    GGML_ASSERT(runtime_contraction.n_tokens == ctx_params.n_batch);
    GGML_ASSERT(runtime_contraction.n_kv == 256);
    GGML_ASSERT(ggml_backend_sched_get_buffer_size(sched, backend_cpu) == size_256);
    GGML_ASSERT(llama_trim_transient_memory(ctx.get()) == 0);
}

static void test_live_context_workspace_iswa_reserve(size_t seed, float stdev) {
    llama_model_ptr model = make_live_context_workspace_model(LLM_ARCH_GEMMA4, seed, stdev);
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
            batch_add_compat(batch, tokens[pos], pos, { 0 }, i + 1 == (int32_t) ctx_params.n_batch);
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
    batch_add_compat(short_batch, tokens[0], 0, { 0 }, true);
    GGML_ASSERT(llama_decode(ctx.get(), short_batch) == 0);
    llama_synchronize(ctx.get());
    llama_batch_free(short_batch);

    const auto contraction = ctx->make_sched_reserve_plan(0);
    GGML_ASSERT(contraction.n_kv == 256);
}

static void test_live_context_workspace_indexer_reserve(size_t seed, float stdev) {
    llama_model_ptr model = make_live_context_workspace_model(LLM_ARCH_QWEN4EXP, seed, stdev, true);
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

static void test_live_context_workspace_unsupported(size_t seed, float stdev) {
    llama_model_ptr model = make_live_context_workspace_model(LLM_ARCH_MAMBA, seed, stdev);
    llama_context_params ctx_params = make_live_context_workspace_params();

    llama_context_ptr ctx(llama_init_from_model(model.get(), ctx_params));
    GGML_ASSERT(ctx);
    GGML_ASSERT(!ctx->get_cparams().live_context_workspace);
}

static void test_phase_workspace_runtime_reserve(size_t seed, float stdev) {
    gguf_context_ptr gguf_ctx = get_gguf_ctx(LLM_ARCH_LLAMA, false);

    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    ggml_backend_dev_t devices[] = { nullptr };
    model_params.devices = devices;

    tensor_data_params tensor_params = { seed, stdev };
    llama_model_ptr model(llama_model_init_from_user(gguf_ctx.get(), set_tensor_data, &tensor_params, model_params));
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
        batch_add_compat(batch, tokens[i], i, { 0 }, true);
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
            batch_add_compat(declared, tokens[i], pos++, { 0 }, true);
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

static void test_phase_workspace_late_pipeline_fallback(size_t seed, float stdev) {
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

    tensor_data_params tensor_params = { seed, stdev };
    llama_model_ptr model(llama_model_init_from_user(gguf_ctx.get(), set_tensor_data, &tensor_params, model_params));
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
        batch_add_compat(batch, tokens[i], i, { 0 }, i == 15);
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
        batch_add_compat(batch, tokens[i], pos + i, { 0 }, i == n_tokens - 1);
        for (int32_t j = 0; j < n_embd; ++j) {
            batch.embd[(size_t) i * n_embd + j] = (float) ((i + j) % 17) / 17.0f;
        }
    }
    return batch;
}

static llama_context_ptr make_phase_workspace_context(
        llama_model * model, llama_context_type type, llama_context * other, uint32_t n_seq_max,
        ggml_backend_sched_eval_callback cb_eval, void * cb_eval_user_data, uint32_t n_outputs_max) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = 32;
    params.n_batch = 32;
    params.n_ubatch = 32;
    params.n_seq_max = n_seq_max;
    params.n_outputs_max = type == LLAMA_CONTEXT_TYPE_MTP ? n_seq_max : n_outputs_max;
    params.n_threads = 4;
    params.n_threads_batch = 4;
    params.ctx_type = type;
    params.ctx_other = other;
    params.phase_aware_workspace = true;
    params.cb_eval = cb_eval;
    params.cb_eval_user_data = cb_eval_user_data;
    return llama_context_ptr(llama_init_from_model(model, params));
}


static int test_source_graph_variants(size_t seed, float stdev, const char * reference_path, bool read_reference) {
    if (!reference_path || !*reference_path) { throw std::runtime_error("source variants require --replay-reference"); }
    ggml_backend_dev_t device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!device) { throw std::runtime_error("source variants require a GPU"); }
    ggml_backend_dev_t devices[] = { device, nullptr };
    auto metadata = get_gguf_ctx(LLM_ARCH_LLAMA, true, false, 8, 2);
    auto model_params = llama_model_default_params();
    model_params.devices = devices;
    model_params.n_gpu_layers = 99;
    model_params.split_mode = LLAMA_SPLIT_MODE_LAYER;
    model_params.load_mode = LLAMA_LOAD_MODE_NONE;
    model_params.moe_expert_cache_slots = 4;
    model_params.moe_expert_cache_host_pinned_size = 0;
    model_params.progress_callback = silent_model_load_progress;
    tensor_data_params tensor_params = { seed, stdev };
    llama_model_ptr model(llama_model_init_from_user(metadata.get(), set_tensor_data, &tensor_params, model_params));
    if (!model) { throw std::runtime_error("source variant model initialization failed"); }
    const llama_adapter_loras adapters;
    const llama_moe_candidate_snapshot candidates(*model, adapters);
    const auto snapshot = candidates.get();
    fprintf(stderr, "test-source-descriptor: flags=%u slots=%u groups=%u tensors=%u\n",
        snapshot.flags, snapshot.n_slots, snapshot.n_groups, snapshot.n_tensors);
    for (uint32_t i = 0; i < snapshot.n_groups; ++i) {
        const auto & group = snapshot.groups[i];
        fprintf(stderr, "test-source-descriptor: group=%u layout=%u domain=%u flags=%u\n", i, group.layout, group.domain, group.flags);
    }
    for (uint32_t i = 0; i < snapshot.n_tensors; ++i) {
        const auto & record = snapshot.tensors[i];
        if (record.group_index == UINT32_MAX) { continue; }
        const auto * tensor = record.tensor;
        fprintf(stderr, "test-source-descriptor: tensor=%s group=%u role=%u status=%u flags=%u type=%s ne=%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 " nb=%zu,%zu,%zu,%zu\n",
            tensor->name, record.group_index, record.role, record.status, record.flags, ggml_type_name(tensor->type),
            tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3]);
    }

    const bool sampling = std::getenv("GGML_TEST_SOURCE_SAMPLER_VARIANTS") != nullptr;
    std::array<llama_sampler_ptr, 4> samplers;
    std::array<llama_sampler_seq_config, 4> sampler_configs;
    std::array<std::mt19937, 4> randoms;
    if (sampling) {
        for (llama_seq_id seq = 0; seq < 4; ++seq) {
            samplers[seq].reset(llama_sampler_chain_init(llama_sampler_chain_default_params()));
            llama_sampler_chain_add(samplers[seq].get(), llama_sampler_init_temp(10.0f));
            llama_sampler_chain_add(samplers[seq].get(), llama_sampler_init_dist(4242 + seq));
            sampler_configs[seq] = { seq, samplers[seq].get() };
            randoms[seq].seed(4242 + seq);
        }
    }
    auto params = llama_context_default_params();
    params.n_ctx = 256;
    params.n_batch = 32;
    params.n_ubatch = 4;
    params.n_seq_max = 4;
    params.n_outputs_max = 4;
    params.n_threads = params.n_threads_batch = 4;
    params.phase_aware_workspace = true;
    params.moe_source_graph_capacity = true;
    if (sampling) {
        params.samplers = sampler_configs.data();
        params.n_samplers = sampler_configs.size();
        params.n_outputs_max_per_seq = 4;
    }
    llama_context_ptr ctx(llama_init_from_model(model.get(), params));
    if (!ctx) { throw std::runtime_error("source variant context initialization failed"); }

    struct contract { int32_t rows, sequences, span; };
    const contract cases[] = {
        {4, 4, 0}, {4, 1, 4}, {4, 2, 2}, {4, 1, 4},
        {1, 1, 0}, {2, 1, 2}, {3, 1, 3}, {3, 3, 0},
        {2, 2, 0}, {4, 4, 0}, {4, 2, 2}, {4, 1, 4},
        {1, 1, 0}, {2, 2, 0}, {3, 3, 0}, {3, 1, 3},
        {2, 1, 2}, {4, 1, 4}, {4, 2, 2}, {4, 1, 4},
        {4, 1, -4}, {4, 1, -4}, {4, 2, 2},
    };
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    std::vector<uint8_t> output;
    auto append = [&](const void * data, size_t size) {
        const auto * bytes = static_cast<const uint8_t *>(data);
        output.insert(output.end(), bytes, bytes + size);
    };
    llama_pos positions[4] = {};
    int32_t total_rows = 0;
    for (size_t frame = 0; frame < sizeof(cases)/sizeof(cases[0]); ++frame) {
        const auto & c = cases[frame];
        const int32_t span = c.span ? std::abs(c.span) : 1;
        auto tokens = get_tokens(c.rows, n_vocab, seed + 1 + frame);
        llama_batch batch = llama_batch_init(c.rows, 0, 1);
        for (int32_t seq = 0; seq < c.sequences; ++seq) {
            for (int32_t i = 0; i < span; ++i) {
                batch_add_compat(batch, tokens[seq*span + i], positions[seq] + i, { seq }, true);
            }
        }
        llama_decode_execution_intent intent = {};
        intent.magic = LLAMA_DECODE_EXECUTION_INTENT_MAGIC;
        intent.abi_version = LLAMA_DECODE_EXECUTION_INTENT_VERSION;
        intent.struct_size = sizeof(intent);
        intent.type = LLAMA_DECODE_EXECUTION_INTENT_TARGET_VERIFICATION;
        intent.verification_span = c.span > 0 ? c.span : 0;
        const int status = c.span > 0 ? llama_decode_ext(ctx.get(), batch, &intent) : llama_decode(ctx.get(), batch);
        llama_batch_free(batch);
        if (status) { throw std::runtime_error("source variant decode failed at frame " + std::to_string(frame)); }
        llama_synchronize(ctx.get());
        auto expected_randoms = randoms;
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (int32_t row = 0; row < c.rows; ++row) {
            const float * logits = sampling ? llama_get_sampled_logits_ith(ctx.get(), row) : llama_get_logits_ith(ctx.get(), row);
            if (!logits || !std::all_of(logits, logits + n_vocab, [](float x) { return std::isfinite(x); })) {
                throw std::runtime_error("invalid source variant logits");
            }
            append(logits, n_vocab*sizeof(float));
            if (sampling) {
                const llama_seq_id seq = row / span;
                const auto token = llama_get_sampled_token_ith(ctx.get(), row);
                const auto * probs = llama_get_sampled_probs_ith(ctx.get(), row);
                if (!probs || token < 0 || token >= n_vocab || llama_get_sampled_probs_count_ith(ctx.get(), row) != uint32_t(n_vocab)) {
                    throw std::runtime_error("invalid source variant sampled output");
                }
                float before = 0.0f;
                for (int32_t id = 0; id < token; ++id) { before += probs[id]; }
                const float rnd = dist(expected_randoms[seq]);
                if (rnd < before - 1e-4f || rnd > before + probs[token] + 1e-4f) {
                    throw std::runtime_error("source variant used the wrong sampler random input");
                }
                append(&token, sizeof(token));
                append(probs, n_vocab*sizeof(float));
                if (row % span < (span + 1) / 2) {
                    llama_sampler_accept(samplers[seq].get(), token);
                    dist(randoms[seq]);
                }
            }
        }
        for (int32_t seq = 0; seq < c.sequences; ++seq) { positions[seq] += span; }
        for (int32_t seq = 0; seq < 4; ++seq) {
            if (llama_memory_seq_pos_max(llama_get_memory(ctx.get()), seq) != positions[seq] - 1) {
                throw std::runtime_error("source variant sequence state changed");
            }
        }
        ggml_backend_moe_hybrid_state_v1 state = {};
        state.struct_size = sizeof(state);
        const bool source_selected = ggml_backend_sched_moe_source_selected_v1(ctx->get_sched());
        const bool have_state = ggml_backend_sched_moe_hybrid_state_v1(ctx->get_sched(), &state);
        if ((c.span >= 0 ? !source_selected || !have_state || !state.window_launches ||
                !(state.resident_routes + state.transfer_routes) : false) ||
            (have_state && (state.cpu_routes || state.cpu_jobs || state.errors || state.capacity_errors ||
                state.dispatch_active || state.cpu_active_jobs))) {
            throw std::runtime_error("source variant did not finish a healthy all-GPU source execution");
        }
        fprintf(stderr, "test-source-variants: frame=%zu rows=%d sequences=%d span=%d windows=%" PRIu64 " resident=%" PRIu64 " transfer=%" PRIu64 "\n",
            frame, c.rows, c.sequences, c.span, state.window_launches, state.resident_routes, state.transfer_routes);
        total_rows += c.rows;
    }
    for (int32_t seq = 0; seq < 4; ++seq) {
        const size_t size = llama_state_seq_get_size_ext(ctx.get(), seq, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (!size || size > 16*1024*1024) { throw std::runtime_error("invalid source variant state size"); }
        std::vector<uint8_t> state(size);
        if (llama_state_seq_get_data_ext(ctx.get(), state.data(), size, seq, LLAMA_STATE_SEQ_FLAGS_NONE) != size) {
            throw std::runtime_error("source variant state serialization failed");
        }
        append(state.data(), size);
    }
    if (read_reference) {
        std::ifstream file(reference_path, std::ios::binary);
        std::vector<uint8_t> expected(output.size());
        if (!file.read(reinterpret_cast<char *>(expected.data()), expected.size()) || file.peek() != EOF || expected != output) {
            throw std::runtime_error("source variant full logits/state differ from the frozen control");
        }
    } else {
        if (std::filesystem::exists(reference_path)) { throw std::runtime_error("refusing to replace source variant reference"); }
        std::ofstream file(reference_path, std::ios::binary);
        if (!file.write(reinterpret_cast<const char *>(output.data()), output.size())) {
            throw std::runtime_error("cannot write source variant reference");
        }
    }
    fprintf(stderr, "test-source-variants: PASS frames=%zu rows=%d vocab=%d bytes=%zu reference=%s\n",
        sizeof(cases)/sizeof(cases[0]), total_rows, n_vocab, output.size(), read_reference ? "exact" : "written");
    return 0;
}

static int test_source_auxiliary_frontend(size_t seed, float stdev, const char * reference_path, bool read_reference, bool uncached_reference, bool no_output_first, bool source_statistics, bool source_ranks, bool inherit_statistics, bool different_model, const char * profile_file, bool parent_active, bool explicit_profile, bool sparse_outputs) {
    if (!reference_path || !*reference_path) { throw std::runtime_error("auxiliary source test requires --replay-reference"); }
    const auto * mode = getenv("GGML_MOE_HYBRID");
    if (!uncached_reference && (!mode || strcmp(mode, "required"))) { throw std::runtime_error("auxiliary source test requires GGML_MOE_HYBRID=required"); }
    auto * device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!device) { throw std::runtime_error("auxiliary source test requires a GPU"); }
    std::vector<uint8_t> recorded;
    const auto append = [&](const void * data, size_t bytes) {
        const auto * first = static_cast<const uint8_t *>(data);
        recorded.insert(recorded.end(), first, first + bytes);
    };
    const auto state_bytes = [](llama_context * ctx, llama_seq_id seq) {
        const size_t bytes = llama_state_seq_get_size_ext(ctx, seq, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (!bytes || bytes > 16 * 1024 * 1024) { throw std::runtime_error("invalid auxiliary state size"); }
        std::vector<uint8_t> result(bytes);
        if (llama_state_seq_get_data_ext(ctx, result.data(), bytes, seq, LLAMA_STATE_SEQ_FLAGS_NONE) != bytes) {
            throw std::runtime_error("auxiliary state serialization failed");
        }
        return result;
    };
    const auto record_state = [&](llama_context * ctx) {
        for (llama_seq_id seq = 0; seq < 2; ++seq) {
            const auto bytes = state_bytes(ctx, seq);
            const uint64_t extent = bytes.size();
            append(&extent, sizeof(extent)); append(bytes.data(), bytes.size());
        }
    };
    struct frame { int32_t rows, sequences; bool outputs; bool last_output = false; };
    struct caller_probe {
        llama_context * context = nullptr;
        const llama_batch * expected = nullptr;
        uint32_t domain = 0;
        uint64_t calls = 0;
        static bool observe(void * opaque, uint32_t phase, uint64_t, void *) {
            if (phase != GGML_BACKEND_MOE_HYBRID_TEST_SOURCE_COMPUTE_ENTERED) { return true; }
            auto & probe = *static_cast<caller_probe *>(opaque);
            const auto * current = probe.context->get_moe_test_frame();
            if (!current || !current->ubatch || !current->certificate || !probe.expected ||
                    current->certificate->domain != probe.domain || !current->certificate->owner_namespace) { return false; }
            const auto & batch = *current->ubatch;
            if (!batch.data || batch.data->batch_idxs.size() != batch.n_tokens ||
                    batch.n_tokens != current->certificate->n_rows || !batch.n_pos) { return false; }
            for (uint32_t row = 0; row < batch.n_tokens; ++row) {
                const auto original = batch.data->batch_idxs[row];
                if (original < 0 || original >= probe.expected->n_tokens ||
                        batch.pos[row] != probe.expected->pos[original] || batch.n_seq_id[row] != probe.expected->n_seq_id[original] ||
                        bool(batch.output[row]) != bool(probe.expected->logits[original])) { return false; }
                for (int32_t seq = 0; seq < batch.n_seq_id[row]; ++seq) {
                    if (batch.seq_id[row][seq] != probe.expected->seq_id[original][seq]) { return false; }
                }
            }
            ++probe.calls;
            return true;
        }
    };
    const bool observe_caller = getenv("GGML_TEST_MOE_CALLER_FRAME") != nullptr;
    uint64_t caller_frames = 0;
    const frame frames[] = {{1, 1, !no_output_first}, {2, 1, true}, {2, 2, true}, {3, 1, false},
        {4, 2, true, sparse_outputs}, {1, 1, false}, {4, 1, true}, {2, 2, true}};
    for (const auto type : {LLAMA_CONTEXT_TYPE_DRAFT, LLAMA_CONTEXT_TYPE_MTP}) {
        for (bool shared : {false, true}) {
            for (bool owner_first : {false, true}) {
                const bool mtp = type == LLAMA_CONTEXT_TYPE_MTP;
                const bool require_auxiliary_source = read_reference && !uncached_reference &&
                    llama_speculative_grouped_intent_test_access::hybrid_required(mode, type);
                auto metadata = get_gguf_ctx(mtp ? LLM_ARCH_QWEN35MOE : LLM_ARCH_LLAMA, true, mtp, 8, 2);
                ggml_backend_dev_t devices[] = {device, nullptr};
                auto model_params = llama_model_default_params();
                model_params.devices = devices;
                model_params.n_gpu_layers = 99;
                model_params.load_mode = LLAMA_LOAD_MODE_NONE;
                model_params.load_mtp = mtp;
                model_params.moe_expert_cache_slots = uncached_reference ? 0 : 4;
                model_params.moe_expert_cache_host_pinned_size = 0;
                model_params.progress_callback = silent_model_load_progress;
                tensor_data_params tensor_params = {seed, stdev};
                llama_model_ptr model(llama_model_init_from_user(metadata.get(), set_tensor_data, &tensor_params, model_params));
                if (!model) { throw std::runtime_error("auxiliary test model initialization failed"); }
                const int32_t vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
                const int32_t embd = llama_model_n_embd_out(model.get());
                auto params = llama_context_default_params();
                params.n_ctx = 128;
                params.n_batch = 16;
                params.n_ubatch = 4;
                params.n_seq_max = 2;
                params.n_outputs_max = 4;
                params.n_threads = params.n_threads_batch = 4;
                params.phase_aware_workspace = true;
                params.moe_source_graph_capacity = true;
                std::string saved_profile, saved_adapt;
                std::string selected_profile;
                const char * selected_adaptation = getenv("GGML_TEST_AUXILIARY_ADAPT");
                if (!selected_adaptation) { selected_adaptation = explicit_profile ? "off" : "occurrence"; }
                if (profile_file && read_reference && !uncached_reference) {
                    const auto * old_profile = getenv("GGML_MOE_EXPERT_PROFILE");
                    const auto * old_adapt = getenv("GGML_MOE_HYBRID_PROFILE_ADAPT");
                    if (old_profile) { saved_profile = old_profile; }
                    if (old_adapt) { saved_adapt = old_adapt; }
                    const auto statistics_metadata = make_source_profile_statistics(model->moe_sources(), explicit_profile);
                    const auto bytes = source_profile_metadata_bytes(statistics_metadata.get());
                    const std::string path = std::string(profile_file) + (explicit_profile ? ".parent." : ".") + std::to_string(int(type));
                    selected_profile = path;
                    if (std::filesystem::exists(path)) {
                        std::ifstream file(path, std::ios::binary);
                        std::vector<uint8_t> retained(bytes.size());
                        if (!file.read(reinterpret_cast<char *>(retained.data()), retained.size()) || file.peek() != EOF || retained != bytes) {
                            throw std::runtime_error("existing auxiliary profile fixture differs");
                        }
                    } else {
                        std::ofstream file(path, std::ios::binary);
                        if (!file.write(reinterpret_cast<const char *>(bytes.data()), bytes.size())) {
                            throw std::runtime_error("cannot write auxiliary profile fixture");
                        }
                    }
                    if (!explicit_profile) {
                        common_set_env("GGML_MOE_EXPERT_PROFILE", path);
                        common_set_env("GGML_MOE_HYBRID_PROFILE_ADAPT", selected_adaptation);
                    }
                }
                llama_context_ptr target(explicit_profile ?
                    llama_init_from_model_with_moe_profile(model.get(), params, selected_profile.c_str(), selected_adaptation) :
                    llama_init_from_model(model.get(), params));
                if (explicit_profile) { std::fill(selected_profile.begin(), selected_profile.end(), '!'); }
                if (profile_file && !explicit_profile && read_reference && !uncached_reference) {
                    common_set_env("GGML_MOE_EXPERT_PROFILE", saved_profile);
                    common_set_env("GGML_MOE_HYBRID_PROFILE_ADAPT", saved_adapt);
                }
                if (!target) { throw std::runtime_error("actual target context initialization failed"); }
                if (inherit_statistics && !profile_file && read_reference && !uncached_reference) {
                    const auto profile_metadata = make_source_profile_statistics(model->moe_sources());
                    const auto profile_bytes = source_profile_metadata_bytes(profile_metadata.get());
                    auto statistics = llama_moe_profile_statistics_parse(profile_bytes.data(), profile_bytes.size(), model->moe_sources());
                    std::vector<ggml_backend_moe_source_statistics_v1> views;
                    std::vector<const double *> scores;
                    for (auto & source : statistics.sources) {
                        if (getenv("GGML_TEST_MOE_SCORED_STATISTICS")) {
                            source.scores.resize(source.counts.size());
                            for (size_t i = 0; i < source.counts.size(); ++i) { source.scores[i] = double(source.counts[i]) / source.observations; }
                            scores.push_back(source.scores.data());
                        }
                        views.push_back({source.tensor, source.counts.data(), source.observations, uint32_t(source.counts.size()), source.domain});
                    }
                    if (!target->initialize_moe_statistics(views, scores.empty() ? nullptr : scores.data())) { throw std::runtime_error("parent supplied statistics installation failed"); }
                    for (auto & source : statistics.sources) { std::fill(source.counts.begin(), source.counts.end(), 0); std::fill(source.scores.begin(), source.scores.end(), -1); }
                }
                params.ctx_type = type;
                if (inherit_statistics) { params.ctx_other = target.get(); }
                llama_model_ptr auxiliary_model;
                if (different_model) {
                    auto auxiliary_metadata = get_gguf_ctx(mtp ? LLM_ARCH_QWEN35MOE : LLM_ARCH_LLAMA, true, mtp, 8, 2);
                    auxiliary_model.reset(llama_model_init_from_user(auxiliary_metadata.get(), set_tensor_data, &tensor_params, model_params));
                    if (!auxiliary_model) { throw std::runtime_error("distinct auxiliary model initialization failed"); }
                }
                if (parent_active && read_reference && !uncached_reference && !different_model) {
                    if (!target->begin_source_call()) { throw std::runtime_error("cannot enter parent source call guard"); }
                    llama_context_ptr unavailable(llama_init_from_model(model.get(), params));
                    target->end_source_call();
                    if (unavailable) { throw std::runtime_error("active parent source context allowed a profile snapshot"); }
                    fprintf(stderr, "test-source-auxiliary-parent: active snapshot rejected type=%d shared=%d owner_first=%d\n",
                        int(type), shared, owner_first);
                }
                auto * selected_model = auxiliary_model ? auxiliary_model.get() : model.get();
                if (explicit_profile && read_reference && !uncached_reference) {
                    const auto profile_metadata = make_source_profile_statistics(selected_model->moe_sources());
                    const auto bytes = source_profile_metadata_bytes(profile_metadata.get());
                    selected_profile = std::string(profile_file) + ".child." + std::to_string(int(type));
                    if (std::filesystem::exists(selected_profile)) {
                        std::ifstream file(selected_profile, std::ios::binary);
                        std::vector<uint8_t> retained(bytes.size());
                        if (!file.read(reinterpret_cast<char *>(retained.data()), retained.size()) || file.peek() != EOF || retained != bytes) {
                            throw std::runtime_error("existing explicit auxiliary profile differs");
                        }
                    } else {
                        std::ofstream file(selected_profile, std::ios::binary);
                        if (!file.write(reinterpret_cast<const char *>(bytes.data()), bytes.size())) {
                            throw std::runtime_error("cannot write explicit auxiliary profile");
                        }
                    }
                }
                llama_context_ptr auxiliary(explicit_profile ?
                    llama_init_from_model_with_moe_profile(selected_model, params, selected_profile.c_str(), selected_adaptation) :
                    llama_init_from_model(selected_model, params));
                if (explicit_profile) { std::fill(selected_profile.begin(), selected_profile.end(), '!'); }

                if (inherit_statistics && auxiliary && read_reference && !uncached_reference) {
                    const auto profile_metadata = make_source_profile_statistics(model->moe_sources());
                    const auto profile_bytes = source_profile_metadata_bytes(profile_metadata.get());
                    auto statistics = llama_moe_profile_statistics_parse(profile_bytes.data(), profile_bytes.size(), model->moe_sources());
                    std::vector<ggml_backend_moe_source_statistics_v1> views;
                    for (auto & source : statistics.sources) {
                        std::reverse(source.counts.begin(), source.counts.end());
                        views.push_back({source.tensor, source.counts.data(), source.observations, uint32_t(source.counts.size()), source.domain});
                    }
                    if (!profile_file && !target->initialize_moe_statistics(views)) { throw std::runtime_error("parent opposite statistics installation failed"); }
                    for (auto & source : statistics.sources) { std::fill(source.counts.begin(), source.counts.end(), 0); }
                    fprintf(stderr, "test-source-auxiliary-statistics: type=%d shared=%d owner_first=%d inherited_sources=%zu parent_replaced=%d profile_file=%d explicit=%d different_model=%d\n",
                        int(type), shared, owner_first, views.size(), !profile_file, profile_file != nullptr, explicit_profile, different_model);
                }
                if (!target || !auxiliary) { throw std::runtime_error("actual auxiliary context initialization failed"); }
                if ((source_statistics || source_ranks) && read_reference && !uncached_reference) {
                    const auto profile_metadata = make_source_profile_statistics(model->moe_sources());
                    const auto profile_bytes = source_profile_metadata_bytes(profile_metadata.get());
                    auto statistics = llama_moe_profile_statistics_parse(profile_bytes.data(), profile_bytes.size(), model->moe_sources());
                    std::vector<ggml_backend_moe_source_statistics_v1> views;
                    for (const auto & source : statistics.sources) {
                        views.push_back({source.tensor, source.counts.data(), source.observations, uint32_t(source.counts.size()), source.domain});
                    }
                    if (source_statistics) {
                        if (!auxiliary->initialize_moe_statistics(views)) {
                            throw std::runtime_error("auxiliary supplied statistics installation failed");
                        }
                        for (uint32_t variant = 0; variant < 5; ++variant) {
                            auto invalid = views;
                            if (variant == 0) { invalid.pop_back(); }
                            if (variant == 1) { invalid.front().domain = UINT32_MAX; }
                            if (variant == 2) { ++invalid.front().observations; }
                            if (variant == 3) { invalid.front().counts = nullptr; }
                            if (variant == 4) { invalid.back() = invalid.front(); }
                            if (auxiliary->initialize_moe_statistics(invalid)) {
                                throw std::runtime_error("invalid auxiliary statistics accepted");
                            }
                        }
                    } else {
                        std::vector<std::vector<int32_t>> ranks;
                        std::vector<ggml_backend_moe_static_profile_v1> profiles;
                        ranks.reserve(statistics.sources.size());
                        for (const auto & source : statistics.sources) {
                            ranks.emplace_back();
                            for (int32_t expert = int32_t(source.counts.size()) - 1; expert >= 0; --expert) { ranks.back().push_back(expert); }
                            profiles.push_back({source.tensor, ranks.back().data(), uint32_t(ranks.back().size())});
                        }
                        if (!auxiliary->initialize_moe_profile(profiles)) {
                            throw std::runtime_error("auxiliary supplied ranks installation failed");
                        }
                        for (uint32_t variant = 0; variant < 5; ++variant) {
                            auto invalid = profiles;
                            if (variant == 0) { invalid.front().down = nullptr; }
                            if (variant == 1) { invalid.front().experts = nullptr; }
                            if (variant == 2) { ++invalid.front().n_experts; }
                            if (variant == 3) { invalid.back() = invalid.front(); }
                            const int32_t saved = ranks.front()[1];
                            if (variant == 4) { ranks.front()[1] = ranks.front()[0]; }
                            const bool accepted = auxiliary->initialize_moe_profile(invalid);
                            ranks.front()[1] = saved;
                            if (accepted) {
                                throw std::runtime_error("invalid auxiliary ranks accepted");
                            }
                        }
                        for (auto & rank : ranks) { std::fill(rank.begin(), rank.end(), 0); }
                    }
                    for (auto & source : statistics.sources) { std::fill(source.counts.begin(), source.counts.end(), 0); }
                    fprintf(stderr, "test-source-auxiliary-statistics: type=%d shared=%d owner_first=%d supplied_then_mutated_sources=%zu\n",
                        int(type), shared, owner_first, views.size());
                }
                if (explicit_profile) {
                    for (const char * invalid_path : {static_cast<const char *>(nullptr), "", "missing-profile-fixture"}) {
                        llama_context_ptr invalid(llama_init_from_model_with_moe_profile(selected_model, params, invalid_path, "off"));
                        if (invalid) { throw std::runtime_error("invalid explicit profile path accepted"); }
                    }
                    selected_profile = std::string(profile_file) + ".child." + std::to_string(int(type));
                    llama_context_ptr invalid(llama_init_from_model_with_moe_profile(selected_model, params, selected_profile.c_str(), "invalid-adaptation"));
                    if (invalid) { throw std::runtime_error("invalid explicit adaptation accepted"); }
                    fprintf(stderr, "test-source-explicit-profile: bad paths and adaptation rejected; caller paths mutated\n");
                }
                std::array<llama_pos, 2> target_pos{}, auxiliary_pos{};
                const auto decode = [&](llama_context * ctx, bool embeddings, const frame & spec,
                        std::array<llama_pos, 2> & positions, size_t input_seed, bool require_source) {
                    llama_batch batch = llama_batch_init(spec.rows, embeddings ? embd : 0, 1);
                    if (embeddings) {
                        batch.token = static_cast<llama_token *>(malloc(sizeof(llama_token) * spec.rows));
                        GGML_ASSERT(batch.token);
                    }
                    const auto tokens = get_tokens(spec.rows, vocab, input_seed);
                    const int32_t per_sequence = spec.rows / spec.sequences;
                    for (int32_t row = 0; row < spec.rows; ++row) {
                        const llama_seq_id seq = row / per_sequence;
                        const llama_pos pos = positions[seq] + row % per_sequence;
                        batch_add_compat(batch, tokens[row], pos, {seq}, spec.outputs && (!spec.last_output || row == spec.rows - 1));
                        if (embeddings) {
                            for (int32_t feature = 0; feature < embd; ++feature) {
                                batch.embd[size_t(row) * embd + feature] = float((pos + feature) % 17) / 17.0f;
                            }
                        }
                    }
                    if (source_statistics || source_ranks || inherit_statistics) {
                        fprintf(stderr, "test-source-auxiliary-decode: type=%d auxiliary=%d rows=%d outputs=%d begin\n",
                            int(type), ctx == auxiliary.get(), spec.rows, spec.outputs);
                    }
                    caller_probe probe{ctx, &batch, ctx == target.get() ? GGML_GRAPH_EXECUTION_DOMAIN_MAIN :
                        (mtp ? GGML_GRAPH_EXECUTION_DOMAIN_MTP : GGML_GRAPH_EXECUTION_DOMAIN_DRAFT), 0};
                    if (observe_caller && require_source && !ctx->set_moe_test_hook(caller_probe::observe, &probe)) {
                        throw std::runtime_error("cannot bind actual auxiliary caller observer");
                    }
                    const int status = llama_decode(ctx, batch);
                    if (observe_caller && require_source) {
                        if (!probe.calls || ctx->get_moe_test_frame() || !ctx->set_moe_test_hook(nullptr, nullptr)) {
                            throw std::runtime_error("actual caller frame missing or escaped its graph call");
                        }
                        caller_frames += probe.calls;
                        fprintf(stderr, "test-source-caller-frame: domain=%u rows=%d sequences=%d outputs=%d calls=%llu original_rows=exact cleared=1\n",
                            probe.domain, spec.rows, spec.sequences, spec.outputs, (unsigned long long) probe.calls);
                    }
                    llama_batch_free(batch);
                    if (status) { throw std::runtime_error("actual auxiliary/target decode failed"); }
                    llama_synchronize(ctx);
                    for (int32_t seq = 0; seq < spec.sequences; ++seq) { positions[seq] += per_sequence; }
                    for (int32_t seq = 0; seq < 2; ++seq) {
                        if (llama_memory_seq_pos_max(llama_get_memory(ctx), seq) != positions[seq] - 1) {
                            throw std::runtime_error("actual auxiliary sequence positions changed");
                        }
                    }
                    if (spec.outputs) {
                        for (int32_t row = spec.last_output ? spec.rows - 1 : 0; row < spec.rows; ++row) {
                            const auto * logits = llama_get_logits_ith(ctx, row);
                            if (!logits || !std::all_of(logits, logits + vocab, [](float x) { return std::isfinite(x); })) {
                                throw std::runtime_error("invalid auxiliary full-vocabulary output");
                            }
                            append(logits, size_t(vocab) * sizeof(float));
                        }
                    }
                    if (require_source) {
                        ggml_backend_moe_hybrid_state_v1 state{}; state.struct_size = sizeof(state);
                        if (!ggml_backend_sched_moe_source_selected_v1(ctx->get_sched()) ||
                                !ggml_backend_sched_moe_hybrid_state_v1(ctx->get_sched(), &state) || !state.window_launches ||
                                !(state.resident_routes + state.transfer_routes) || state.cpu_routes || state.cpu_jobs ||
                                state.errors || state.capacity_errors || state.dispatch_active || state.cpu_active_jobs) {
                            throw std::runtime_error("actual auxiliary graph did not complete healthy all-GPU hybrid execution");
                        }
                    }
                    record_state(ctx);
                };
                decode(target.get(), false, {2, 2, true}, target_pos, seed, read_reference && !uncached_reference);
                if (shared) {
                    if (require_auxiliary_source) {
                        if (!target->begin_source_call()) { throw std::runtime_error("cannot enter workspace owner call guard"); }
                        const auto busy = llama_attach_shared_workspace(auxiliary.get(), target.get());
                        target->end_source_call();
                        if (busy != -1) { throw std::runtime_error("active source owner allowed workspace capacity mutation"); }
                    }
                    const auto attached = llama_attach_shared_workspace(auxiliary.get(), target.get());
                    if (attached < 0 || (require_auxiliary_source && attached != 0)) {
                        throw std::runtime_error("actual auxiliary shared workspace attachment failed");
                    }
                    fprintf(stderr, "test-source-auxiliary: type=%d requested_shared=1 actual_shared=%d control=%d\n",
                        int(type), attached, !read_reference);
                }
                for (size_t index = 0; index < sizeof(frames) / sizeof(frames[0]); ++index) {
                    const auto prior = index == 3 ? state_bytes(auxiliary.get(), 0) : std::vector<uint8_t>{};
                    const auto surviving = index == 3 ? state_bytes(auxiliary.get(), 1) : std::vector<uint8_t>{};
                    const auto prefix_pos = auxiliary_pos;
                    decode(auxiliary.get(), mtp, frames[index], auxiliary_pos, seed + 100 + index, require_auxiliary_source);
                    if (index == 3) {
                        if (llama_state_seq_set_data_ext(auxiliary.get(), prior.data(), prior.size(), 0,
                                LLAMA_STATE_SEQ_FLAGS_NONE) != prior.size()) {
                            throw std::runtime_error("auxiliary accepted-prefix restore failed");
                        }
                        if (state_bytes(auxiliary.get(), 1) != surviving) {
                            throw std::runtime_error("auxiliary restore changed surviving sequence state");
                        }
                        auxiliary_pos = prefix_pos;
                        decode(auxiliary.get(), mtp, {1, 1, false}, auxiliary_pos, seed + 100 + index, require_auxiliary_source);
                    }
                    decode(target.get(), false, {1, 1, true}, target_pos, seed + 200 + index, read_reference && !uncached_reference);
                    fprintf(stderr, "test-source-auxiliary: type=%d shared=%d owner_first=%d frame=%zu rows=%d sequences=%d outputs=%d OK last_output=%d\n",
                        int(type), shared, owner_first, index, frames[index].rows, frames[index].sequences, frames[index].outputs, frames[index].last_output);
                }
                if ((source_statistics || source_ranks) && read_reference && !uncached_reference) {
                    const auto profile_metadata = make_source_profile_statistics(model->moe_sources());
                    const auto profile_bytes = source_profile_metadata_bytes(profile_metadata.get());
                    const auto statistics = llama_moe_profile_statistics_parse(profile_bytes.data(), profile_bytes.size(), model->moe_sources());
                    std::vector<ggml_backend_moe_source_statistics_v1> views;
                    for (const auto & source : statistics.sources) {
                        views.push_back({source.tensor, source.counts.data(), source.observations, uint32_t(source.counts.size()), source.domain});
                    }
                    if (auxiliary->initialize_moe_statistics(views)) {
                        throw std::runtime_error("installed auxiliary source program allowed unchecked profile replacement");
                    }
                    fprintf(stderr, "test-source-auxiliary-setter: invalid inputs and installed-program replacement rejected\n");
                }
                if (owner_first) {
                    target.reset();
                    decode(auxiliary.get(), mtp, {1, 1, true}, auxiliary_pos, seed + 999, require_auxiliary_source);
                    auxiliary.reset();
                } else {
                    auxiliary.reset();
                    decode(target.get(), false, {1, 1, true}, target_pos, seed + 999, read_reference && !uncached_reference);
                    target.reset();
                }
            }
        }
    }
    if (read_reference) {
        std::ifstream file(reference_path, std::ios::binary);
        std::vector<uint8_t> expected(recorded.size());
        if (!file.read(reinterpret_cast<char *>(expected.data()), expected.size()) || file.peek() != EOF || expected != recorded) {
            throw std::runtime_error("actual auxiliary full logits/state differ from frozen control");
        }
    } else {
        if (std::filesystem::exists(reference_path)) { throw std::runtime_error("refusing to replace auxiliary control"); }
        std::ofstream file(reference_path, std::ios::binary);
        if (!file.write(reinterpret_cast<const char *>(recorded.data()), recorded.size())) {
            throw std::runtime_error("cannot write auxiliary control");
        }
    }
    fprintf(stderr, "test-source-auxiliary: PASS contexts=DRAFT,MTP shared/private both teardown orders complete logits/state/accepted-prefix bytes=%zu reference=%s\n",
        recorded.size(), read_reference ? "exact" : "written");
    if (observe_caller) { fprintf(stderr, "test-source-caller-frame: PASS callbacks=%llu\n", (unsigned long long) caller_frames); }
    return 0;
}

static void test_speculative_limits(size_t seed, float stdev) {
    {
        common_ngram_history history;
        int32_t match = 0;
        history.update({ 1, 2, 3, 4, 5, 6, 99, 1, 2 });
        GGML_ASSERT(history.propose({3}, 2, 3, match) == llama_tokens({4, 5}) && match == 3);
        GGML_ASSERT(history.propose({3, 4}, 2, 3, match) == llama_tokens({5, 6}) && match == 4);
        GGML_ASSERT(history.propose({3, 4}, 1, 3, match) == llama_tokens({5}));
        GGML_ASSERT(history.propose({3, 4}, 0, 3, match).empty());
        GGML_ASSERT(history.propose({3, 4}, 2, 5, match).empty());
        GGML_ASSERT(history.propose({3, 44}, 2, 3, match).empty());
        // Queries must not insert rejected pending tokens into the history index.
        GGML_ASSERT(history.propose({3, 4}, 2, 3, match) == llama_tokens({5, 6}));
        history.update({7, 8, 9, 10, 11, 99, 7, 8});
        GGML_ASSERT(history.propose({9}, 2, 3, match) == llama_tokens({10, 11}));
        GGML_ASSERT(history.propose({3, 4}, 2, 3, match).empty());
        history.update({7, 8});
        GGML_ASSERT(history.propose({9}, 2, 3, match).empty());
        history.clear();
        GGML_ASSERT(history.propose({7, 8, 9}, 2, 3, match).empty());
        history.update({4, 5, 6, 1, 8, 4, 5, 6, 2, 8, 4, 5});
        GGML_ASSERT(history.propose({6}, 2, 3, match) == llama_tokens({2, 8}));

        common_ngram_chain_policy policy(9);
        GGML_ASSERT(policy.choose(3, 6, 3) == 0);
        GGML_ASSERT(policy.choose(3, 6, 20) == 6);
        for (int i = 0; i < 20; ++i) {
            policy.observe(3, 0, 3, 0, 10.0);
            policy.observe(3, 6, 3, 3, 100.0);
        }
        GGML_ASSERT(policy.choose(3, 6, 3) == 0);
        GGML_ASSERT(policy.choose(3, 0, 3) == 0 && policy.choose(3, 6, 2) == 0);
        common_ngram_chain_policy beneficial(9);
        for (int i = 0; i < 20; ++i) {
            beneficial.observe(3, 0, 3, 0, 10.0);
            beneficial.observe(3, 6, 9, 20, 12.0);
        }
        GGML_ASSERT(beneficial.choose(3, 6, 20) == 6);

        common_ngram_chain_policy confidence(6);
        for (int i = 0; i < 20; ++i) {
            confidence.observe(3, 0, 3, 0, 10.0);
            confidence.observe(3, 3, 6, 20, 12.0);
        }
        GGML_ASSERT(confidence.choose(3, 3, 20, {1.0f, 1.0f, 1.0f}) == 3);
        GGML_ASSERT(confidence.choose(3, 3, 20, {0.1f, 0.1f, 0.1f}) == 0);
        GGML_ASSERT(confidence.choose_replacement(3, 3, 20, {0.1f, 0.1f, 0.1f}) == 3);
        GGML_ASSERT(confidence.choose_replacement(3, 2, 20, {0.1f}) == 0);
        GGML_ASSERT(confidence.choose_replacement(3, 6, 2, {0.1f}) == 0);
        common_ngram_chain_policy conditioned(6);
        for (int i = 0; i < 30; ++i) { conditioned.observe(3, 3, 0, 3, 12.0, true); }
        GGML_ASSERT(conditioned.choose_replacement(3, 6, 3, {1.0f, 1.0f, 1.0f}) == 0);
        GGML_ASSERT(conditioned.choose_replacement(3, 6, 20, {0.1f, 0.1f, 0.1f}) > 0);
        GGML_ASSERT(confidence.choose(3, 3, 20, {NAN, NAN, NAN}) == 0);

        common_params_speculative params;
        params.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
        params.draft.n_max = 3;
        params.lookup_chain = 6;
        GGML_ASSERT(common_speculative_n_max(&params) == 9 && params.need_n_rs_seq() == 9);
        common_validate_speculative_params(params, 16);
        params.mtp_rs_planes = 4;
        GGML_ASSERT(params.is_mtp_rs_capped() && params.need_n_rs_seq() == 3);
        bool rejected = false;
        try { common_validate_speculative_params(params, 9); } catch (const std::invalid_argument &) { rejected = true; }
        GGML_ASSERT(rejected);
        params.mtp_rs_planes = 0;
        params.types = {COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE};
        rejected = false;
        try { common_validate_speculative_params(params, 16); } catch (const std::invalid_argument &) { rejected = true; }
        GGML_ASSERT(rejected);
        params.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
        rejected = false;
        try { common_speculative_ptr missing(common_speculative_init(params, 1)); } catch (const std::invalid_argument &) { rejected = true; }
        GGML_ASSERT(rejected);
        params.draft.n_max = INT32_MAX;
        rejected = false;
        try { common_validate_speculative_params(params, 16); } catch (const std::invalid_argument &) { rejected = true; }
        GGML_ASSERT(rejected);
    }

    gguf_context_ptr gguf_ctx = get_gguf_ctx(LLM_ARCH_QWEN35, false, true);

    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    model_params.load_mtp = true;
    ggml_backend_dev_t devices[] = { nullptr };
    model_params.devices = devices;

    tensor_data_params tensor_params = { seed, stdev };
    llama_model_ptr model(llama_model_init_from_user(gguf_ctx.get(), set_tensor_data, &tensor_params, model_params));
    GGML_ASSERT(model);

    const uint32_t n_seq = 2;
    llama_context_ptr target = make_phase_workspace_context(model.get(), LLAMA_CONTEXT_TYPE_DEFAULT, nullptr, n_seq, nullptr, nullptr, 5);
    llama_context_ptr draft = make_phase_workspace_context(model.get(), LLAMA_CONTEXT_TYPE_MTP, target.get(), n_seq);
    GGML_ASSERT(target && draft);

    const llama_tokens prompt_match = { 99, 1, 2, 3, 4, 1 };
    const llama_tokens prompt_none  = { 99, 1, 2, 3, 4, 5 };

    auto make_spec = [&](int32_t n_max, int32_t n_min, std::vector<common_speculative_type> types, int32_t chain = 0) {
        common_params_speculative params;
        params.types = std::move(types);
        params.draft.n_max = n_max;
        params.draft.n_min = n_min;
        params.lookup_chain = chain;
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

    for (uint16_t accepted : {0, 1, 4}) {
        auto spec = make_spec(2, 0, { COMMON_SPECULATIVE_TYPE_DRAFT_MTP }, 2);
        GGML_ASSERT(common_speculative_n_max(spec.get()) == 4);
        llama_tokens result;
        common_speculative_get_draft_params(spec.get(), 0) = {true, 2, 0, 2, &prompt_none, &result};
        common_speculative_draft(spec.get());
        GGML_ASSERT(result.size() == 2);
        const llama_token alternate = (result[1] + 1) % llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        const llama_tokens prefix = {10, 11, 12, 13, 14, 15, 16, 17, 18};
        llama_tokens history = prefix;
        history.insert(history.end(), {2, result[0], alternate, 7, 8, 99});
        history.insert(history.end(), prefix.begin(), prefix.end());
        GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(draft.get()), 0, 0, -1));
        result.clear();
        common_speculative_begin(spec.get(), 0, history);
        common_speculative_get_draft_params(spec.get(), 0) = {true, 4, 0, 2, &history, &result};
        llama_tokens other_result;
        common_speculative_get_draft_params(spec.get(), 1) = {true, 2, 0, 6, &prompt_none, &other_result};
        common_speculative_draft(spec.get());
        GGML_ASSERT(other_result.size() == 2);
        GGML_ASSERT(result.size() == 4 && result[1] == alternate && result[2] == 7 && result[3] == 8);
        GGML_ASSERT(common_speculative_get_draft_params(spec.get(), 0).probabilities->size() == 2);
        GGML_ASSERT(!common_speculative_retain_draft_state(spec.get(), 0));
        GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(draft.get()), 0, 0, -1));

        common_batch verification(target.get());
        verification.add(2, 0, 0, true);
        for (size_t i = 0; i < result.size(); ++i) { verification.add(result[i], llama_pos(i + 1), 0, true); }
        GGML_ASSERT(llama_process(target.get(), LLAMA_PROCESS_TYPE_DECODE, verification.get()) == 0);
        GGML_ASSERT(common_speculative_process(spec.get(), verification));
        common_speculative_accept(spec.get(), 0, accepted);
        std::vector<uint8_t> state;
        GGML_ASSERT(common_speculative_get_mtp_state(spec.get(), 0, state));
        GGML_ASSERT(state.size() == 3 * sizeof(uint32_t) + size_t(llama_model_n_embd_out(model.get())) * sizeof(float));
        GGML_ASSERT(memcmp(state.data() + 3 * sizeof(uint32_t), llama_get_embeddings_nextn_ith(target.get(), accepted),
                size_t(llama_model_n_embd_out(model.get())) * sizeof(float)) == 0);
        common_speculative_print_stats(spec.get());
        llama_memory_clear(llama_get_memory(target.get()), true);
        llama_memory_clear(llama_get_memory(draft.get()), true);
    }

    {
        auto spec = make_spec(2, 0, { COMMON_SPECULATIVE_TYPE_DRAFT_MTP }, 2);
        llama_tokens result;
        common_speculative_get_draft_params(spec.get(), 0) = {true, 2, 0, 2, &prompt_none, &result};
        common_speculative_draft(spec.get());
        const llama_tokens original = result;
        const llama_tokens prefix = {10, 11, 12, 13, 14, 15, 16, 17, 18};
        llama_tokens history = prefix;
        history.insert(history.end(), {2, (result[0] + 1) % llama_vocab_n_tokens(llama_model_get_vocab(model.get())), 7, 8, 9, 99});
        history.insert(history.end(), prefix.begin(), prefix.end());
        GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(draft.get()), 0, 0, -1));
        common_speculative_begin(spec.get(), 0, history);
        result.clear();
        common_speculative_get_draft_params(spec.get(), 0) = {true, 4, 0, 2, &history, &result};
        common_speculative_draft(spec.get());
        GGML_ASSERT(result == original);
        llama_memory_clear(llama_get_memory(draft.get()), true);
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

        common_batch verify(target_retained.get());
        for (llama_seq_id seq_id = 0; seq_id < 2; ++seq_id) {
            verify.add( 2 + seq_id, 0, { seq_id }, true);
            verify.add( proposals[seq_id][0], 1, { seq_id }, true);
        }
        GGML_ASSERT(llama_process(target_retained.get(), LLAMA_PROCESS_TYPE_DECODE, verify.get()) == 0);
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

    }
}

static void test_mtp_state_dependencies() {
    for (int64_t width : { 4, 8 }) {
        llm_graph_result result(256);
        auto * ctx = result.get_ctx();
        auto * graph = result.get_gf();
        auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, 1);
        auto * required = ggml_add(ctx, input, input);
        auto * copy = ggml_cpy(ctx, required, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, 1));
        auto * store = ggml_set_rows(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, 2), required,
                ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1));
        auto * dead = ggml_mul(ctx, input, input);
        auto * empty_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 0);
        auto * empty_output = ggml_get_rows(ctx, dead, empty_ids);
        ggml_set_output(empty_output);
        auto * unclassified = ggml_arange(ctx, 0, float(width), 1);
        GGML_ASSERT(!ggml_op_is_pure(unclassified->op));
        auto * unknown_input = ggml_sqr(ctx, input);
        auto * unknown = ggml_map_custom1(ctx, ggml_get_rows(ctx, unknown_input, empty_ids),
                [](ggml_tensor *, const ggml_tensor *, int, int, void *) {}, 1, nullptr);
        auto * private_input = ggml_scale(ctx, input, 2.0f);
        auto * private_write = ggml_scale_inplace(ctx, private_input, 3.0f);
        auto * hidden = ggml_rms_norm(ctx, input, 1e-5f);
        auto * hidden_output = ggml_view_1d(ctx, hidden, width, 0);
        ggml_set_output(hidden_output);
        std::vector<float> backing(width);
        auto * external = ggml_sub(ctx, input, input);
        external->data = backing.data();
        for (auto * root : { copy, store, empty_output, unknown, unclassified, private_write, hidden_output, external }) {
            ggml_build_forward_expand(graph, root);
        }
        std::vector<ggml_tensor *> nodes;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) { nodes.push_back(ggml_graph_node(graph, i)); }
        result.retain_state_computation();
        GGML_ASSERT(!(dead->flags & GGML_TENSOR_FLAG_COMPUTE));
        for (auto * retained : { required, copy, store, unknown_input, unknown, unclassified, private_input, private_write, hidden, hidden_output, external }) {
            GGML_ASSERT(retained->flags & GGML_TENSOR_FLAG_COMPUTE);
        }
        GGML_ASSERT(empty_output->flags & GGML_TENSOR_FLAG_OUTPUT);
        GGML_ASSERT(hidden_output->flags & GGML_TENSOR_FLAG_OUTPUT);
        result.retain_state_computation();
        GGML_ASSERT(!(dead->flags & GGML_TENSOR_FLAG_COMPUTE));
        GGML_ASSERT(nodes.size() == size_t(ggml_graph_n_nodes(graph)));
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) { GGML_ASSERT(nodes[i] == ggml_graph_node(graph, i)); }
    }
}

static void test_mtp_graph_reactivation(size_t seed, float stdev) {
    for (bool gpu : { false, true }) {
        auto * device = gpu ? ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) : nullptr;
        if (gpu && !device) { continue; }
        for (bool moe : { false, true }) {
            const auto arch = moe ? LLM_ARCH_QWEN35MOE : LLM_ARCH_QWEN35;
            gguf_context_ptr metadata = get_gguf_ctx(arch, moe, true);
            llama_model_params model_params = llama_model_default_params();
            model_params.progress_callback = silent_model_load_progress;
            model_params.load_mtp = true;
            ggml_backend_dev_t devices[] = { device, nullptr };
            model_params.n_gpu_layers = gpu ? 99 : 0;
            model_params.devices = devices;
            tensor_data_params tensor_params = { seed, stdev };
            llama_model_ptr model(llama_model_init_from_user(metadata.get(), set_tensor_data, &tensor_params, model_params));
            GGML_ASSERT(model);
            llama_context_params params = llama_context_default_params();
            params.n_ctx = 32;
            params.n_batch = params.n_ubatch = 4;
            params.n_threads = params.n_threads_batch = 4;
            params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
            params.phase_aware_workspace = true;
            params.decode_boundary_overlap = true;
            params.moe_source_graph_capacity = false;
            llama_context_ptr control(llama_init_from_model(model.get(), params));
            params.moe_source_graph_capacity = true;
            llama_context_ptr retained(llama_init_from_model(model.get(), params));
            GGML_ASSERT(control && retained);
            const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
            for (int pos = 0; pos < 12; ++pos) {
                llama_batch batch = make_mtp_batch(1, llama_model_n_embd_out(model.get()), pos, n_vocab, seed + pos);
                batch.logits[0] = pos % 2 == 0;
                GGML_ASSERT(llama_decode(control.get(), batch) == 0);
                GGML_ASSERT(llama_decode(retained.get(), batch) == 0);
                llama_synchronize(control.get());
                llama_synchronize(retained.get());
                if (batch.logits[0]) {
                    const float * expected = llama_get_logits_ith(control.get(), -1);
                    const float * actual = llama_get_logits_ith(retained.get(), -1);
                    GGML_ASSERT(expected && actual);
                    for (uint32_t i = 0; i < n_vocab; ++i) {
                        GGML_ASSERT(std::isfinite(actual[i]) && std::fabs(expected[i] - actual[i]) < 1e-5f);
                    }
                }
                llama_batch_free(batch);
            }
            GGML_ASSERT(llama_perf_context(retained.get()).n_reused > llama_perf_context(control.get()).n_reused);
        }
    }
}

static void test_phase_workspace_mtp_lifecycle(size_t seed, float stdev) {
    gguf_context_ptr gguf_ctx = get_gguf_ctx(LLM_ARCH_QWEN35, false, true);

    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    model_params.load_mtp = true;
    ggml_backend_dev_t devices[] = { nullptr };
    model_params.devices = devices;

    tensor_data_params tensor_params = { seed, stdev };
    llama_model_ptr model(llama_model_init_from_user(gguf_ctx.get(), set_tensor_data, &tensor_params, model_params));
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
        batch_add_compat(target_batch, tokens[i], i, { 0 }, true);
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
    batch_add_compat(target_catchup, tokens[0], 4, { 0 }, true);
    llama_batch draft_catchup = make_mtp_batch(1, n_embd, 4, n_vocab, seed + 2);
    llama_batch target_reacquire = llama_batch_init(1, 0, 1);
    batch_add_compat(target_reacquire, tokens[1], 5, { 0 }, true);
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

static void test_mtp_draft_vocab(size_t seed, float stdev) {
    gguf_context_ptr metadata = get_gguf_ctx(LLM_ARCH_QWEN4EXP, true, true);
    gguf_set_val_u32(metadata.get(), "tokenizer.ggml.eos_token_id", 2);
    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    model_params.load_mtp = true;
    model_params.use_extra_bufts = false;
    model_params.no_host = true;
    ggml_backend_dev_t devices[] = { nullptr };
    model_params.devices = devices;
    tensor_data_params tensor_params = { seed, stdev };
    llama_model_ptr model(llama_model_init_from_user(metadata.get(), set_tensor_data, &tensor_params, model_params));
    GGML_ASSERT(model);
    const auto * vocab = llama_model_get_vocab(model.get());
    const uint32_t n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<int32_t> ids;
    std::vector<bool> included(n_vocab);
    for (int32_t id = n_vocab - 1; id >= 0; --id) {
        if (id % 3 == 0 || llama_vocab_is_eog(vocab, id)) {
            ids.push_back(id);
            included[id] = true;
        }
    }
    const auto path = std::filesystem::temp_directory_path() / ("mtp-draft-vocab-" + std::to_string(std::random_device{}()) + ".gguf");
    const auto bad_path = path.string() + ".bad";
    const auto filename = path.string();
    GGML_ASSERT(llama_write_mtp_draft_vocab(model.get(), ids.data(), ids.size(), filename.c_str()));
    auto draft = make_phase_workspace_context(model.get(), LLAMA_CONTEXT_TYPE_MTP);
    auto full = make_phase_workspace_context(model.get(), LLAMA_CONTEXT_TYPE_MTP);
    auto normal = make_phase_workspace_context(model.get(), LLAMA_CONTEXT_TYPE_DEFAULT);
    GGML_ASSERT(draft && full && normal);
    GGML_ASSERT(!llama_set_mtp_draft_vocab(normal.get(), filename.c_str()));
    const int32_t duplicate[] = { ids[0], ids[0] };
    const int32_t invalid[] = { int32_t(n_vocab) };
    GGML_ASSERT(!llama_write_mtp_draft_vocab(model.get(), duplicate, 2, bad_path.c_str()));
    GGML_ASSERT(!llama_write_mtp_draft_vocab(model.get(), invalid, 1, bad_path.c_str()));
    GGML_ASSERT(!llama_write_mtp_draft_vocab(model.get(), nullptr, 0, bad_path.c_str()));
    std::vector<int32_t> no_eog;
    for (auto id : ids) {
        if (!llama_vocab_is_eog(vocab, id)) {
            no_eog.push_back(id);
        }
    }
    GGML_ASSERT(!no_eog.empty() && !llama_write_mtp_draft_vocab(model.get(), no_eog.data(), no_eog.size(), bad_path.c_str()));
    for (const char * field : { "version", "selected_tokens", "token_attributes", "token_offsets", "special_tokens" }) {
        gguf_context_ptr bad(gguf_init_from_file(filename.c_str(), {true, nullptr}));
        GGML_ASSERT(bad);
        const auto name = std::string("mtp_draft_vocab.") + field;
        gguf_set_val_str(bad.get(), name.c_str(), "wrong type");
        GGML_ASSERT(gguf_write_to_file(bad.get(), bad_path.c_str(), true));
        GGML_ASSERT(!llama_set_mtp_draft_vocab(draft.get(), bad_path.c_str()));
    }
    GGML_ASSERT(llama_set_mtp_draft_vocab(draft.get(), filename.c_str()));
    GGML_ASSERT(!llama_set_mtp_draft_vocab(draft.get(), filename.c_str()));
    // The generated optional output scales must retain full projection.
    auto fallback = make_mtp_batch(1, llama_model_n_embd_out(model.get()), 0, n_vocab, seed);
    GGML_ASSERT(llama_decode(draft.get(), fallback) == 0 && llama_decode(full.get(), fallback) == 0);
    const auto * fallback_expected = llama_get_logits_ith(full.get(), -1);
    const auto * fallback_actual = llama_get_logits_ith(draft.get(), -1);
    for (uint32_t id = 0; id < n_vocab; ++id) {
        GGML_ASSERT(std::isfinite(fallback_actual[id]) && fallback_actual[id] == fallback_expected[id]);
    }
    llama_batch_free(fallback);
    draft.reset();
    full.reset();
    normal.reset();
    // A separate fixture proves the supported direct projection boundary.
    model->output_s = nullptr;
    for (auto & layer : model->layers) {
        layer.nextn.shared_head_head_s = nullptr;
    }
    const auto direct_context = [&]() {
        auto params = llama_context_default_params();
        params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        params.n_ctx = 32;
        params.n_batch = params.n_ubatch = 32;
        params.n_outputs_max = params.n_outputs_max_per_seq = 4;
        params.n_threads = params.n_threads_batch = 4;
        params.phase_aware_workspace = true;
        return llama_context_ptr(llama_init_from_model(model.get(), params));
    };
    draft = direct_context();
    full = direct_context();
    GGML_ASSERT(draft && full && llama_set_mtp_draft_vocab(draft.get(), filename.c_str()));
    int32_t pos = 0;
    for (int32_t rows : {1, 2, 3, 4, 1}) {
        auto batch = make_mtp_batch(rows, llama_model_n_embd_out(model.get()), pos, n_vocab, seed + pos);
        std::fill(batch.logits, batch.logits + rows, true);
        GGML_ASSERT(llama_decode(draft.get(), batch) == 0);
        GGML_ASSERT(llama_decode(full.get(), batch) == 0);
        for (int32_t row = 0; row < rows; ++row) {
            const auto * expected = llama_get_logits_ith(full.get(), row);
            const auto * actual = llama_get_logits_ith(draft.get(), row);
            GGML_ASSERT(expected && actual);
            for (uint32_t id = 0; id < n_vocab; ++id) {
                const bool equal = included[id] ? std::isfinite(actual[id]) && std::fabs(actual[id] - expected[id]) < 1e-5f : actual[id] == -INFINITY;
                if (!equal) {
                    fprintf(stderr, "MTP draft vocabulary: pos=%d row=%d id=%u selected=%u full=%g actual=%g\n", pos, row, id, unsigned(included[id]), expected[id], actual[id]);
                }
                GGML_ASSERT(equal);
            }
        }
        llama_batch_free(batch);
        pos += rows;
    }
    llama_perf_context_reset(full.get());
    GGML_ASSERT(!llama_set_mtp_draft_vocab(full.get(), filename.c_str()));
    draft.reset();
    full.reset();
    normal.reset();
    std::filesystem::remove(path);
    std::filesystem::remove(bad_path);
}

static void test_phase_workspace_qwen4exp_mtp_reserve(size_t seed, float stdev) {
    gguf_context_ptr metadata = get_gguf_ctx(LLM_ARCH_QWEN4EXP, true, true);
    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    model_params.load_mtp = true;
    ggml_backend_dev_t devices[] = { nullptr };
    model_params.devices = devices;
    tensor_data_params tensor_params = { seed, stdev };
    llama_model_ptr model(llama_model_init_from_user(metadata.get(), set_tensor_data, &tensor_params, model_params));
    GGML_ASSERT(model);

    llama_context_ptr draft = make_phase_workspace_context(model.get(), LLAMA_CONTEXT_TYPE_MTP);
    GGML_ASSERT(draft);
    GGML_ASSERT(dynamic_cast<llama_memory_hybrid_idx *>(draft->get_memory()));
    draft->sched_reserve(4);
    draft->sched_reserve(0);
    llama_batch batch = make_mtp_batch(4, llama_model_n_embd_out(model.get()), 0,
            llama_vocab_n_tokens(llama_model_get_vocab(model.get())), seed + 1);
    GGML_ASSERT(llama_decode(draft.get(), batch) == 0);
    llama_synchronize(draft.get());
    GGML_ASSERT(llama_get_logits_ith(draft.get(), 3));
    llama_batch_free(batch);

    llama_context_ptr rebuilt = make_phase_workspace_context(model.get(), LLAMA_CONTEXT_TYPE_MTP);
    batch = make_mtp_batch(4, llama_model_n_embd_out(model.get()), 0,
            llama_vocab_n_tokens(llama_model_get_vocab(model.get())), seed + 1);
    GGML_ASSERT(llama_decode(rebuilt.get(), batch) == 0);
    llama_batch_free(batch);
    auto * kept_memory = draft->get_memory();
    auto * full_memory = rebuilt->get_memory();
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    const auto compare = [&](int32_t pos, uint32_t tokens) {
        // A self-copy changes no KV values and requests a full pool layout rebuild.
        full_memory->seq_cp(0, 0, 0, -1);
        llama_batch next = make_mtp_batch(tokens, llama_model_n_embd_out(model.get()), pos, n_vocab, seed + pos + 2);
        GGML_ASSERT(llama_decode(draft.get(), next) == 0);
        GGML_ASSERT(llama_decode(rebuilt.get(), next) == 0);
        const float * expected = llama_get_logits_ith(rebuilt.get(), -1);
        const float * actual = llama_get_logits_ith(draft.get(), -1);
        for (uint32_t i = 0; i < n_vocab; ++i) {
            GGML_ASSERT(std::isfinite(actual[i]) && std::fabs(expected[i] - actual[i]) < 1e-5f);
        }
        llama_batch_free(next);
    };
    for (int32_t pos = 4; pos < 16; pos += 4) { compare(pos, 4); }
    for (int32_t pos : {13, 9, 0}) {
        GGML_ASSERT(kept_memory->seq_rm(0, pos, -1));
        GGML_ASSERT(full_memory->seq_rm(0, pos, -1));
        compare(pos, 4);
    }
}

static void (*mtp_input_set_async)(ggml_backend_t, ggml_tensor *, const void *, size_t, size_t) = nullptr;
static size_t mtp_hidden_uploads = 0;
static void (*mtp_device_props)(ggml_backend_dev_t, ggml_backend_dev_props *) = nullptr;

static void mtp_device_no_events(ggml_backend_dev_t device, ggml_backend_dev_props * props) {
    mtp_device_props(device, props);
    props->caps.events = false;
}

static void count_mtp_input_upload(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    mtp_hidden_uploads += strcmp(tensor->name, "mtp_h_input") == 0;
    mtp_input_set_async(backend, tensor, data, offset, size);
}

static void test_mtp_input_staging(size_t seed, float stdev) {
    auto * gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!gpu || strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(gpu)), "CUDA") != 0) {
        printf("test_mtp_input_staging: skipped, CUDA device required\n");
        return;
    }
    gguf_context_ptr gguf_ctx = get_gguf_ctx(LLM_ARCH_QWEN35, false, true);
    auto model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    model_params.n_gpu_layers = 99;
    model_params.load_mtp = true;
    ggml_backend_dev_t devices[] = { gpu, nullptr };
    model_params.devices = devices;
    tensor_data_params tensor_params = { seed, stdev };
    llama_model_ptr model(llama_model_init_from_user(gguf_ctx.get(), set_tensor_data, &tensor_params, model_params));
    GGML_ASSERT(model);

    mtp_device_props = gpu->iface.get_props;
    for (bool events : { true, false }) {
        gpu->iface.get_props = events ? mtp_device_props : mtp_device_no_events;
        for (auto kv_type : { GGML_TYPE_F16, GGML_TYPE_Q8_0 }) {
            auto params = llama_context_default_params();
            params.n_ctx = 64;
            params.n_batch = params.n_ubatch = 16;
            params.n_seq_max = params.n_outputs_max = 2;
            params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
            params.type_k = params.type_v = kv_type;
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            llama_context_ptr reference(llama_init_from_model(model.get(), params));
            params.decode_boundary_overlap = true;
            llama_context_ptr staged(llama_init_from_model(model.get(), params));
            GGML_ASSERT(reference && staged);
            auto * backend = ggml_backend_sched_get_backend(staged->get_sched(), 0);
            mtp_input_set_async = backend->iface.set_tensor_async;
            GGML_ASSERT(mtp_input_set_async);
            backend->iface.set_tensor_async = count_mtp_input_upload;
            mtp_hidden_uploads = 0;
            ggml_backend_dev_props props;
            ggml_backend_dev_get_props(gpu, &props);
            const size_t expected_uploads = props.caps.async && props.caps.events && ggml_backend_dev_host_buffer_type(gpu) ? 8 : 0;
            const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
            const int32_t n_embd = llama_model_n_embd_out(model.get());
            int32_t positions[2] = {};
            int step = 0;
            for (int n_tokens : { 1, 3, 1, 4, 2, 1, 3, 1 }) {
                const llama_seq_id seq_id = step % 2;
                const bool both_sequences = n_tokens == 4;
                llama_batch batch = make_mtp_batch(n_tokens, n_embd, positions[seq_id], n_vocab, seed + step++);
                for (int i = 0; i < n_tokens; ++i) {
                    const llama_seq_id row_seq = both_sequences && i >= 2 ? seq_id ^ 1 : seq_id;
                    batch.seq_id[i][0] = row_seq;
                    batch.pos[i] = positions[row_seq] + (both_sequences ? i % 2 : i);
                  }
                GGML_ASSERT(llama_decode(reference.get(), batch) == 0);
                const float * expected = llama_get_logits_ith(reference.get(), n_tokens - 1);
                GGML_ASSERT(expected);
                std::vector<float> logits_expected(expected, expected + n_vocab);
                GGML_ASSERT(llama_decode(staged.get(), batch) == 0);
                // Decode owns submitted inputs after returning, even while GPU work is pending.
                std::fill_n(batch.token, n_tokens, -1);
                std::fill_n(batch.embd, (size_t) n_tokens * n_embd, -123.0f);
                const float * actual = llama_get_logits_ith(staged.get(), n_tokens - 1);
                GGML_ASSERT(actual);
                GGML_ASSERT(nmse(logits_expected, std::vector<float>(actual, actual + n_vocab)) < 1e-7);
                for (llama_seq_id s = 0; s < 2; ++s) {
                    const size_t size = llama_state_seq_get_size(reference.get(), s);
                    std::vector<uint8_t> a(size), b(size);
                    GGML_ASSERT(llama_state_seq_get_data(reference.get(), a.data(), size, s) == size);
                    GGML_ASSERT(llama_state_seq_get_data(staged.get(), b.data(), size, s) == size);
                    GGML_ASSERT(a == b);
                  }
                if (both_sequences) {
                    positions[0] += 2;
                    positions[1] += 2;
                  } else {
                    positions[seq_id] += n_tokens;
                  }
                llama_batch_free(batch);
              }
            backend->iface.set_tensor_async = mtp_input_set_async;
            GGML_ASSERT(expected_uploads ? mtp_hidden_uploads >= expected_uploads : mtp_hidden_uploads == 0);
            fprintf(stderr, "test_mtp_input_staging: kv=%s events=%d hidden_uploads=%zu input lifetime and exact KV OK\n", ggml_type_name(kv_type), events, mtp_hidden_uploads);
        }
    }
    gpu->iface.get_props = mtp_device_props;
}

static void test_phase_workspace_mismatched_placement(size_t seed, float stdev) {
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

    tensor_data_params target_data = { seed, stdev };
    tensor_data_params draft_data = { seed, stdev };
    llama_model_ptr target_model(llama_model_init_from_user(
            gguf_ctx.get(), set_tensor_data, &target_data, target_params));
    llama_model_ptr draft_model(llama_model_init_from_user(
            gguf_ctx.get(), set_tensor_data, &draft_data, draft_params));
    GGML_ASSERT(target_model && draft_model);

    llama_context_ptr target = make_phase_workspace_context(target_model.get(), LLAMA_CONTEXT_TYPE_DEFAULT);
    llama_context_ptr draft = make_phase_workspace_context(
            draft_model.get(), LLAMA_CONTEXT_TYPE_MTP, target.get());
    GGML_ASSERT(target && draft);
    GGML_ASSERT(!llama_contexts_share_workspace(target.get(), draft.get()));

    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(target_model.get()));
    const int32_t n_embd = llama_model_n_embd_out(draft_model.get());
    llama_batch target_batch = llama_batch_init(1, 0, 1);
    batch_add_compat(target_batch, get_tokens(1, n_vocab, seed)[0], 0, { 0 }, true);
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
    common_batch batch(lctx);
    GGML_ASSERT(n_tokens <= n_ctx);
    for (uint32_t pos = 0; pos < n_tokens; pos++) {
        batch.add(tokens[pos], pos, 0, true);
    }
    if (encode) {
        if (llama_process(lctx, LLAMA_PROCESS_TYPE_ENCODE, batch.get())) {
            throw std::runtime_error("failed to encode batch");
        }
    }
    if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
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
        batch_add_compat(batch_par, tokens[pos],   pos, {0}, true);
        batch_add_compat(batch_par, tokens_b[pos], pos, {1}, true);
        batch_add_compat(batch_b,   tokens_b[pos], pos, {0}, true);
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

// entries [n/4, n/2) are embd rows, decoded either as token/embd/token chunks or as one mixed batch
// returns the llama_process() error code
static int32_t get_logits_mixed(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens, const std::vector<float> & embd, bool mixed,
        std::vector<float> & ret) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_embd   = llama_model_n_embd_inp(model);
    const uint32_t n_tokens = tokens.size();
    const uint32_t i_embd_0 = n_tokens/4;
    const uint32_t i_embd_1 = n_tokens/2;

    const std::vector<uint32_t> bounds = mixed
        ? std::vector<uint32_t>{0, n_tokens}
        : std::vector<uint32_t>{0, i_embd_0, i_embd_1, n_tokens};

    llama_memory_clear(llama_get_memory(lctx), true);
    llama_batch_ext_ptr batch(llama_batch_ext_init(lctx));

    ret.clear();
    ret.reserve(n_tokens*n_vocab);
    for (size_t c = 0; c + 1 < bounds.size(); c++) {
        llama_batch_ext_clear(batch.get());
        for (uint32_t i = bounds[c]; i < bounds[c + 1]; i++) {
            const bool is_embd = i >= i_embd_0 && i < i_embd_1;
            const int32_t idx = is_embd
                ? llama_batch_ext_add_embd(batch.get(), 0, { embd.data() + (size_t) (i - i_embd_0)*n_embd, 1, n_embd })
                : llama_batch_ext_add_token(batch.get(), 0, tokens[i]);
            GGML_ASSERT(idx >= 0);
            const llama_pos pos[4] = { (llama_pos) i, (llama_pos) i, (llama_pos) i, 0 };
            llama_batch_ext_set_pos(batch.get(), idx, pos);
            llama_batch_ext_set_output_logits(batch.get(), idx, true);
        }
        const int32_t err = llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (err != 0) {
            return err;
        }
        for (uint32_t i = 0; i < bounds[c + 1] - bounds[c]; i++) {
            const float * logits_ith = llama_get_logits_ith(lctx, i);
            ret.insert(ret.end(), logits_ith, logits_ith + n_vocab);
        }
    }
    return 0;
}

static bool check_causal_attn_toggle(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_past   = tokens.size();
    const uint32_t n_ubatch = llama_n_ubatch(lctx);

    GGML_ASSERT(n_past + n_ubatch/2 + n_ubatch <= llama_n_ctx(lctx));

    llama_set_causal_attn(lctx, false);

    common_batch batch(lctx);

    bool ok = true;
    uint32_t pos = n_past;
    for (const uint32_t n_tokens : { n_ubatch/2, n_ubatch }) {
        batch.clear();
        for (uint32_t i = 0; i < n_tokens; i++) {
            batch.add(tokens[i], pos++, 0, true);
        }

        const int32_t rc = llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (rc != 0) {
            LOG_ERR("%s: n_tokens=%u: llama_process returned %d\n", __func__, n_tokens, rc);
            ok = false;
            break;
        }

        const float * logits = llama_get_logits_ith(lctx, n_tokens - 1);
        if (logits == nullptr) {
            LOG_ERR("%s: n_tokens=%u: no logits\n", __func__, n_tokens);
            ok = false;
            break;
        }
        for (uint32_t j = 0; j < n_vocab; j++) {
            if (std::isnan(logits[j])) {
                LOG_ERR("%s: n_tokens=%u: nan logit\n", __func__, n_tokens);
                ok = false;
                break;
            }
        }
    }

    return ok;
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
        case LLM_ARCH_GLM5_NEXT:
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
        case LLM_ARCH_K2_HORIZON:
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
    if (arch == LLM_ARCH_LLAMA_EMBED || arch == LLM_ARCH_GEMMA_EMBEDDING || arch == LLM_ARCH_GEMMA_EMBEDDING2 || arch == LLM_ARCH_T5ENCODER) {
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
    if (arch == LLM_ARCH_CLEF) {
        return false; // TODO decision head tensors
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

static int test_moe_cache_selector_precedence(const size_t seed, const float stdev) {
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
        tensor_data_params tensor_params = { seed, stdev };
        return llama_model_ptr(llama_model_init_from_user(metadata, set_tensor_data, &tensor_params, params));
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
        if (model) {
            ggml_backend_moe_source_owner_v1 owner = {};
            if (!model->moe_source_owner_v1(&owner)) { return 1; }
            size_t certified = 0;
            for (const auto & group : model->moe_sources()) {
                for (const auto & bank : group.banks) {
                    if (bank.status != GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE) { continue; }
                    ggml_backend_moe_source_span_v1 span = {};
                    span.struct_size = sizeof(span);
                    span.abi_version = GGML_BACKEND_MOE_SOURCE_OWNER_V1_VERSION;
                    span.witness = bank.tensor;
                    span.data = bank.tensor->data;
                    span.bytes = ggml_nbytes(bank.tensor);
                    span.expert_stride = bank.tensor->nb[2];
                    span.type = bank.tensor->type;
                    std::copy(std::begin(bank.tensor->ne), std::end(bank.tensor->ne), std::begin(span.ne));
                    std::copy(std::begin(bank.tensor->nb), std::end(bank.tensor->nb), std::begin(span.nb));
                    const int status = owner.validate_span(&owner, owner.generation, &span);
                    if (status != GGML_BACKEND_MOE_SOURCE_STATUS_V1_OK) {
                        fprintf(stderr, "test-moe-cache-selector: callback source certificate rejected tensor=%s status=%d\n", bank.tensor->name, status);
                        return 1;
                    }
                    span.data = static_cast<const uint8_t *>(span.data) + 1;
                    if (owner.validate_span(&owner, owner.generation, &span) == GGML_BACKEND_MOE_SOURCE_STATUS_V1_OK) { return 1; }
                    span.data = bank.tensor->data;
                    if (owner.validate_span(&owner, owner.generation + 1, &span) != GGML_BACKEND_MOE_SOURCE_STATUS_V1_GENERATION_MISMATCH) { return 1; }
                    ++certified;
                }
            }
            if (!certified) { return 1; }
            fprintf(stderr, "test-moe-cache-selector: callback source certificates=%zu stale/data controls rejected\n", certified);
        }
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

static int test_moe_placement_report(const size_t seed, const float stdev) {
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
        tensor_data_params tensor_params = { tensor_seed, stdev };
        return llama_model_ptr(llama_model_init_from_user(metadata, set_tensor_data, &tensor_params, params));
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
        gguf_context_ptr metadata = get_gguf_ctx(LLM_ARCH_QWEN3MOE, true);
        auto enabled_params = make_params(both_layers, 1, nullptr);
        enabled_params.moe_early_router = true;
        enabled_params.moe_early_router_max_rows = 3;
        auto enabled = load(metadata.get(), enabled_params, seed);
        check(enabled != nullptr, "early-router placement fixture failed to load");
        if (enabled) {
            const auto report = llama_model_moe_placement(enabled.get());
            check(report.owners.size() == 1 && retained.owners.size() == 1 &&
                      report.owners[0].cache_fixed_bytes > retained.owners[0].cache_fixed_bytes,
                  "disabled early routing reserved speculative cache storage");
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

static int test_moe_joint_measurement(const size_t seed, const float stdev, const std::string & fit_tool) {
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
    tensor_data_params target_data = { seed, stdev };
    llama_model_ptr target(
        llama_model_init_from_user(target_metadata.get(), set_tensor_data, &target_data, load_params));
    auto mtp_metadata        = get_gguf_ctx(LLM_ARCH_QWEN35MOE, true, true);
    auto mtp_params          = load_params;
    mtp_params.load_mtp      = true;
    tensor_data_params mtp_data = { seed, stdev };
    llama_model_ptr mtp(llama_model_init_from_user(mtp_metadata.get(), set_tensor_data, &mtp_data, mtp_params));
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

static int save_models(const std::string & arch_filter, const size_t seed, const float stdev, const int verbosity, const std::string & dir) {
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
        if (!arch_matches(arch_filter, arch)) {
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
            auto model_and_ctx = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, stdev, {});
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

static int test_backends(const std::string & arch_filter, const size_t seed, const float stdev, const int verbosity, const char * target_backend) {
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
        bool                            host_experts; // keep the experts in host memory, see host_experts_test
        size_t                          moe_cache_size;

        device_config(std::vector<ggml_backend_dev_t> devs, std::string name, llama_split_mode split_mode, bool host_experts = false, size_t moe_cache_size = 0)
            : devs(std::move(devs)), label(std::move(name)), split_mode(split_mode), host_experts(host_experts), moe_cache_size(moe_cache_size) {}
    };

    const llama_model_tensor_buft_override host_experts_overrides[] = {
        { LLM_FFN_EXPS_REGEX, ggml_backend_cpu_buffer_type() },
        { nullptr, nullptr },
    };

    std::vector<device_config> dev_configs;
    size_t max_device_label_length = 4;
    {
        std::vector<ggml_backend_dev_t> devices_meta;
        {
            const size_t device_count = ggml_backend_dev_count();
            for (size_t i = 0; i < device_count; i++) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                if (target_backend != nullptr && strcmp(target_backend, ggml_backend_dev_name(dev)) != 0) {
                    continue;
                }
                dev_configs.emplace_back(std::vector<ggml_backend_dev_t>{dev}, ggml_backend_dev_description(dev), LLAMA_SPLIT_MODE_LAYER);
                max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());

                // cpu-based devices cannot be used in tensor split mode
                if (ggml_backend_dev_buffer_type(dev) != ggml_backend_cpu_buffer_type()) {
                    devices_meta.push_back(dev);
                }
            }
        }

        if (target_backend == nullptr) {
            dev_configs.emplace_back(devices_meta, "Meta", LLAMA_SPLIT_MODE_TENSOR);
        }

        // the ops that use the experts are offloaded to the first device and the scheduler copies the used experts
        if (!devices_meta.empty()) {
            dev_configs.emplace_back(devices_meta, "Host experts", LLAMA_SPLIT_MODE_LAYER, true);
            max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());
        }

        // the ops that use the host experts run on a GPU and read the experts from a cache
        // the cache has only a few slots (4 for 288 KiB experts), so the experts are evicted and uploaded again
        if (!devices_meta.empty()) {
            const enum ggml_backend_dev_type type = ggml_backend_dev_type(devices_meta[0]);
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                dev_configs.emplace_back(std::vector<ggml_backend_dev_t>{devices_meta[0]}, "MoE cache", LLAMA_SPLIT_MODE_LAYER, true, 1536*1024);
                max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());
                // each GPU caches the layers assigned to it
                if (devices_meta.size() > 1) {
                    dev_configs.emplace_back(devices_meta, "MoE cache, layer split", LLAMA_SPLIT_MODE_LAYER, true, 1536*1024);
                    max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());
                }
            }
        }
    }

    size_t max_arch_name_length = 0;
    for (const llm_arch & arch : llm_arch_all()) {
        max_arch_name_length = std::max(max_arch_name_length, strlen(llm_arch_name(arch)));
    }

    const std::string template_header  = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|%15s|%9s|%9s|%15s|\n";
    const std::string template_row_cfg = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|";
    const std::string template_row_res = "%15s %10s|%20s|%20s|%15s %10s|\n";

    bool all_ok = true;
    size_t n_tests = 0;
    size_t n_failed = 0;
    common_log_flush(common_log_main());
    LOG(template_header.c_str(), "Model arch.", "Device", "Config", "NMSE vs. CPU", "Roundtrip", "Parallel", "Mixed batch");
    LOG("|");
    for (size_t i = 0; i < max_arch_name_length; i++) {
        LOG("-");
    }
    LOG("|");
    for (size_t i = 0; i < max_device_label_length; i++) {
        LOG("-");
    }
    LOG("|------|---------------|---------|---------|---------------|\n");
    for (const llm_arch & arch : llm_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (!arch_matches(arch_filter, arch)) {
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
                if (dc.host_experts && (!moe || !host_experts_test(arch))) {
                    continue;
                }
                const llama_model_tensor_buft_override * overrides = dc.host_experts ? host_experts_overrides : nullptr;

                // print test config first; should anything fail during model loading or inference, at least we know which test case caused it
                LOG(template_row_cfg.c_str(), llm_arch_name(arch), dc.label.c_str(), config_name.c_str());
                fflush(stdout);

                std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_dev;
                std::vector<float> logits_dev;
                std::string status_nmse      = "\033[1;33mSKIP\033[0m";
                std::string status_roundtrip = "\033[1;33mSKIP\033[0m";
                std::string status_parallel  = "\033[1;33mSKIP\033[0m";
                std::string status_mixed     = "\033[1;33mSKIP\033[0m";
                char nmse_str[12] = {0};
                char mixed_str[12] = {0};

                bool skip = !arch_supported(arch) || (dc.split_mode == LLAMA_SPLIT_MODE_TENSOR && dc.devs.empty());
                bool test_executed = false;
                bool test_ok = true;
                if (!skip) {
                    if (logits_cpu.empty()) {
                        model_and_ctx_cpu = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, stdev, {}, LLAMA_SPLIT_MODE_LAYER, encode);
                        logits_cpu = get_logits(model_and_ctx_cpu.first.get(), model_and_ctx_cpu.second.get(), tokens, encode);
                    }
                    if (dc.split_mode != LLAMA_SPLIT_MODE_TENSOR || llm_arch_supports_sm_tensor(arch)) {
                        test_executed = true;
                        model_and_ctx_dev = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, stdev, dc.devs, dc.split_mode, encode, overrides, dc.moe_cache_size);
                        logits_dev = get_logits(model_and_ctx_dev.first.get(), model_and_ctx_dev.second.get(), tokens, encode);
                        const double nmse_val = nmse(logits_cpu, logits_dev);
                        snprintf(nmse_str, sizeof(nmse_str), "(%.2e)", nmse_val);
                        status_nmse = "\033[1;32mOK\033[0m";
                        if (nmse_val > 1e-4) {
                            test_ok = false;
                            status_nmse = "\033[1;31mFAIL\033[0m";
                        }

                        // FIXME: T5 kq_b does not broadcast over KV streams, so context init with n_seq_max > 1 aborts
                        if (arch != LLM_ARCH_T5) {
                            status_parallel = "\033[1;32mOK\033[0m";
                            if (!test_parallel_seqs(model_and_ctx_dev.first.get(), tokens, logits_dev, encode)) {
                                test_ok = false;
                                status_parallel = "\033[1;31mFAIL\033[0m";
                            }
                        }
                        // chunked decode matches a single batch only with causal attention over a memory
                        llama_context * lctx_dev = model_and_ctx_dev.second.get();
                        if (!encode && llama_get_memory(lctx_dev) != nullptr) {
                            std::vector<float> embd_mixed((size_t) llama_model_n_embd_inp(model_and_ctx_dev.first.get())*tokens.size()/4);
                            std::mt19937 gen(seed);
                            std::normal_distribution<float> dis(0.0f, stdev);
                            for (float & v : embd_mixed) {
                                v = dis(gen);
                            }
                            std::vector<float> logits_mixed;
                            std::vector<float> logits_chunks;
                            if (llm_arch_supports_mixed_batch(arch)) {
                                if (get_logits_mixed(model_and_ctx_dev.first.get(), lctx_dev, tokens, embd_mixed, false, logits_chunks) != 0 ||
                                    get_logits_mixed(model_and_ctx_dev.first.get(), lctx_dev, tokens, embd_mixed, true,  logits_mixed)  != 0) {
                                    throw std::runtime_error("failed to decode mixed batch");
                                }
                                const double nmse_mixed = nmse(logits_chunks, logits_mixed);
                                snprintf(mixed_str, sizeof(mixed_str), "(%.2e)", nmse_mixed);
                                status_mixed = "\033[1;32mOK\033[0m";
                                if (nmse_mixed > 1e-4) {
                                    test_ok = false;
                                    status_mixed = "\033[1;31mFAIL\033[0m";
                                }
                            } else {
                                // must be rejected as an invalid batch, mute the expected error log
                                ud.verbosity = LOG_LEVEL_OUTPUT;
                                const int32_t err = get_logits_mixed(model_and_ctx_cpu.first.get(), model_and_ctx_cpu.second.get(), tokens, embd_mixed, true, logits_mixed);
                                ud.verbosity = verbosity;
                                if (err != -1) {
                                    test_ok = false;
                                    status_mixed = "\033[1;31mFAIL\033[0m";
                                }
                            }
                        }

                        // runs after the mixed batch check, as it leaves the context with non-causal attention
                        if (!encode && !check_causal_attn_toggle(model_and_ctx_dev.first.get(), model_and_ctx_dev.second.get(), tokens)) {
                            if (test_ok) {
                                status_nmse = "\033[1;31mFAIL\033[0m (toggle)";
                            }
                            test_ok = false;
                        }
                    }

                    FILE * file = tmpfile(); // Can be null on Windows without administrator privileges.
                    // FIXME: when adding a tensor to a gguf_context a copy is made, this changes the pointer which the meta backend
                    //     in turn uses to map the tensors to their simple equivalents - this is fundamentally incompatible
                    if (file != nullptr && llama_model_saver_supports_arch(arch) && dc.split_mode != LLAMA_SPLIT_MODE_TENSOR) {
                        test_executed = true;
                        GGML_ASSERT(model_and_ctx_dev.first && model_and_ctx_dev.second);
                        llama_model_saver ms = llama_model_saver(model_and_ctx_dev.first.get());
                        ms.add_kv_from_model();
                        ms.add_tensors_from_model();
                        ms.save(file);
                        rewind(file);

                        auto model_and_ctx_roundtrip = get_model_and_ctx(nullptr, file, seed, stdev, dc.devs, dc.split_mode, encode, overrides, dc.moe_cache_size);
                        const std::vector<float> logits_roundtrip = get_logits(
                            model_and_ctx_roundtrip.first.get(), model_and_ctx_roundtrip.second.get(), tokens, encode);
                        status_roundtrip = "\033[1;32mOK\033[0m";
                        GGML_ASSERT(logits_roundtrip.size() == logits_dev.size());
                        for (size_t i = 0; i < logits_roundtrip.size(); i++) {
                            if (logits_roundtrip[i] != logits_dev[i]) {
                                test_ok = false;
                                status_roundtrip = "\033[1;31mFAIL\033[0m";
                                break;
                            }
                        }
                    }
                }

                if (test_executed) {
                    n_tests++;
                    if (!test_ok) {
                        n_failed++;
                        all_ok = false;
                    }
                }

                // log the results for this test case
                LOG(template_row_res.c_str(), status_nmse.c_str(), nmse_str, status_roundtrip.c_str(), status_parallel.c_str(), status_mixed.c_str(), mixed_str);
            }
        }
    }

    if (n_tests == 0) {
        LOG("Summary: no tests executed\n");
    } else if (n_failed == 0) {
        LOG("Summary: all %zu test(s) passed\n", n_tests);
    } else {
        LOG("Summary: %zu test(s) executed, %zu failed\n", n_tests, n_failed);
    }

    llama_log_set(ud.log_old.callback, ud.log_old.user_data);
    return all_ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    // init the logger at max verbosity. filter with a custom callback respecting the user-configure verbosity
    common_log_set_verbosity_thold(LOG_LEVEL_DEBUG);
    common_init();
    llama_backend_init();

    std::random_device rd;

    std::string arch_filter;
    size_t seed = rd();
    float stdev = 0.1f;
    std::string out;
    bool test_phase_workspace = false;
    bool test_live_context_workspace = false;
    bool run_speculative_limits = false;
    bool run_mtp_draft_vocab = false;
    bool run_source_variants = false;
    bool run_source_auxiliary = false;
    bool auxiliary_uncached_reference = false;
    bool auxiliary_no_output_first = false;
    bool auxiliary_source_statistics = false;
    bool auxiliary_source_ranks = false;
    bool auxiliary_inherit_statistics = false;
    bool auxiliary_different_model = false;
    const char * auxiliary_profile_file = nullptr;
    bool auxiliary_parent_active = false;
    bool auxiliary_explicit_profile = false;
    bool auxiliary_sparse_outputs = false;
    bool run_moe_cache_selector = false;
    bool run_moe_placement = false;
    bool        run_moe_joint_measurement   = false;
    std::string fit_tool;
    const char * source_uses_path = nullptr;
    const char * source_profile_path = nullptr;
    const char * source_learning_path = nullptr;
    const char * source_learning_output = nullptr;
    const char * source_learning_adaptation = "occurrence-sync";
    uint32_t source_learning_slots = 1;
    uint32_t source_learning_context = 512;
    uint32_t source_learning_batch = 128;
    const char * source_statistics_file = nullptr;
    bool source_statistics = false;
    const char * profile_model_path = nullptr;
    const char * profile_corpus_path = nullptr;
    std::vector<std::string> hybrid_args{"test-llama-archs"};
    const char * profile_output_path = nullptr;
    uint32_t profile_threads = 4;
    const char * moe_replay_path = nullptr;
    const char * replay_reference_path = nullptr;
    bool replay_read = false;
    const char * replay_prompt_path = nullptr;
    uint32_t replay_rows = 32;
    std::string replay_split = "diagnostic";
    bool replay_independent_sources = false;
    int32_t replay_cache_slots = 32;
    bool replay_cpu_oracle = false;
    bool replay_gpu_oracle = false;
    llama_load_mode replay_load_mode = LLAMA_LOAD_MODE_NONE;
    const char * target_backend = nullptr;

    int verbosity = LOG_LEVEL_ERROR;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv);
            return 0;
        } else if (strcmp(argv[i], "--test-phase-workspace") == 0) {
            test_phase_workspace = true;
        } else if (strcmp(argv[i], "--test-live-context-workspace") == 0) {
            test_live_context_workspace = true;
        } else if (strcmp(argv[i], "--test-source-variants") == 0) {
            run_source_variants = true;
        } else if (strcmp(argv[i], "--test-source-auxiliary") == 0) {
            run_source_auxiliary = true;
        } else if (strcmp(argv[i], "--auxiliary-explicit-profile-file") == 0) {
            if (++i >= argc) { return 1; }
            auxiliary_profile_file = argv[i];
            auxiliary_inherit_statistics = true;
            auxiliary_explicit_profile = true;
        } else if (strcmp(argv[i], "--auxiliary-profile-file") == 0) {
            if (++i >= argc) { return 1; }
            auxiliary_profile_file = argv[i];
            auxiliary_inherit_statistics = true;
        } else if (strcmp(argv[i], "--auxiliary-sparse-outputs") == 0) {
            auxiliary_sparse_outputs = true;
        } else if (strcmp(argv[i], "--auxiliary-parent-active") == 0) {
            auxiliary_parent_active = true;
        } else if (strcmp(argv[i], "--auxiliary-different-model") == 0) {
            auxiliary_different_model = true;
        } else if (strcmp(argv[i], "--auxiliary-inherit-statistics") == 0) {
            auxiliary_inherit_statistics = true;
        } else if (strcmp(argv[i], "--auxiliary-source-ranks") == 0) {
            auxiliary_source_ranks = true;
        } else if (strcmp(argv[i], "--auxiliary-source-statistics") == 0) {
            auxiliary_source_statistics = true;
        } else if (strcmp(argv[i], "--auxiliary-no-output-first") == 0) {
            auxiliary_no_output_first = true;
        } else if (strcmp(argv[i], "--auxiliary-uncached-reference") == 0) {
            auxiliary_uncached_reference = true;
        } else if (strcmp(argv[i], "--test-speculative-limits") == 0) {
            run_speculative_limits = true;
        } else if (strcmp(argv[i], "--test-mtp-draft-vocab") == 0) {
            run_mtp_draft_vocab = true;
        } else if (strcmp(argv[i], "--test-moe-cache-selector") == 0) {
            run_moe_cache_selector = true;
        } else if (strcmp(argv[i], "--test-source-uses") == 0 && i + 1 < argc) {
            source_uses_path = argv[++i];
        } else if (strcmp(argv[i], "--test-source-profile") == 0 && i + 1 < argc) {
            source_profile_path = argv[++i];
        } else if (strcmp(argv[i], "--test-source-learning") == 0 && i + 1 < argc) {
            source_learning_path = argv[++i];
        } else if (strcmp(argv[i], "--source-learning-output") == 0 && i + 1 < argc) {
            source_learning_output = argv[++i];
        } else if (strcmp(argv[i], "--source-learning-adaptation") == 0 && i + 1 < argc) {
            source_learning_adaptation = argv[++i];
            if (strcmp(source_learning_adaptation, "occurrence") && strcmp(source_learning_adaptation, "occurrence-sync")) { return 1; }
        } else if ((strcmp(argv[i], "--source-learning-cache-slots") == 0 || strcmp(argv[i], "--source-learning-context") == 0 ||
                strcmp(argv[i], "--source-learning-batch") == 0) && i + 1 < argc) {
            const char * option = argv[i];
            const char * value = argv[++i];
            char * end = nullptr;
            const long parsed = strtol(value, &end, 10);
            const long limit = strcmp(option, "--source-learning-cache-slots") == 0 ? 65535 :
                strcmp(option, "--source-learning-context") == 0 ? 32768 : 2048;
            if (!*value || *end || parsed < 1 || parsed > limit) { return 1; }
            if (strcmp(option, "--source-learning-cache-slots") == 0) { source_learning_slots = uint32_t(parsed); }
            else if (strcmp(option, "--source-learning-context") == 0) { source_learning_context = uint32_t(parsed); }
            else { source_learning_batch = uint32_t(parsed); }
        } else if (strcmp(argv[i], "--source-statistics-file") == 0 && i + 1 < argc) {
            source_statistics_file = argv[++i];
        } else if (strcmp(argv[i], "--test-source-statistics") == 0) {
            source_statistics = true;
        } else if (strcmp(argv[i], "--collect-moe-profile") == 0 && i + 1 < argc) {
            profile_model_path = argv[++i];
        } else if ((strcmp(argv[i], "--moe-hybrid") == 0 || strcmp(argv[i], "--moe-gpu-miss-fraction") == 0) && i + 1 < argc) {
            hybrid_args.push_back(argv[i]);
            hybrid_args.push_back(argv[++i]);
        } else if (strcmp(argv[i], "--profile-corpus") == 0 && i + 1 < argc) {
            profile_corpus_path = argv[++i];
        } else if (strcmp(argv[i], "--profile-output") == 0 && i + 1 < argc) {
            profile_output_path = argv[++i];
        } else if (strcmp(argv[i], "--profile-threads") == 0 && i + 1 < argc) {
            const char * value = argv[++i];
            char * end = nullptr;
            const long threads = strtol(value, &end, 10);
            if (!*value || *end || threads < 1 || threads > 1024) { return 1; }
            profile_threads = uint32_t(threads);
        } else if (strcmp(argv[i], "--test-moe-replay") == 0 && i + 1 < argc) {
            moe_replay_path = argv[++i];
        } else if (strcmp(argv[i], "--replay-reference") == 0 && i + 1 < argc) {
            replay_reference_path = argv[++i];
        } else if (strcmp(argv[i], "--replay-read") == 0) {
            replay_read = true;
        } else if (strcmp(argv[i], "--replay-prompt-file") == 0 && i + 1 < argc) {
            replay_prompt_path = argv[++i];
        } else if (strcmp(argv[i], "--replay-rows") == 0 && i + 1 < argc) {
            const char * value = argv[++i];
            if (!*value || strlen(value) > 4 || *value < '0' || *value > '9') { return 1; }
            char * end = nullptr;
            const long rows = strtol(value, &end, 10);
            if (*end || rows < 1 || rows > 4096) { return 1; }
            replay_rows = uint32_t(rows);
        } else if (strcmp(argv[i], "--replay-split") == 0 && i + 1 < argc) {
            replay_split = argv[++i];
            if (replay_split != "diagnostic" && replay_split != "calibration" && replay_split != "development" && replay_split != "held-out") { return 1; }
        } else if (strcmp(argv[i], "--replay-independent-sources") == 0) {
            replay_independent_sources = true;
        } else if (strcmp(argv[i], "--replay-gpu-oracle") == 0) {
            replay_gpu_oracle = true;
        } else if (strcmp(argv[i], "--replay-load-mode") == 0 && i + 1 < argc) {
            const char * value = argv[++i];
            if (strcmp(value, "none") && strcmp(value, "mmap")) { return 1; }
            replay_load_mode = llama_load_mode_from_str(value);
        } else if (strcmp(argv[i], "--replay-cpu-oracle") == 0) {
            replay_cpu_oracle = true;
        } else if (strcmp(argv[i], "--replay-cache-slots") == 0 && i + 1 < argc) {
            const char * value = argv[++i];
            if (!*value || strlen(value) > 5) { return 1; }
            char * end = nullptr;
            const long slots = strtol(value, &end, 10);
            if (*end || slots < 0 || slots > 65536) { return 1; }
            replay_cache_slots = int32_t(slots);
        } else if (strcmp(argv[i], "--test-moe-placement") == 0) {
            run_moe_placement = true;
        } else if (strcmp(argv[i], "--test-moe-joint-measurement") == 0) {
            run_moe_joint_measurement = true;
        } else if (strcmp(argv[i], "--fit-tool") == 0 && i + 1 < argc) {
            fit_tool = argv[++i];
        } else if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--arch") == 0) {
            if (i + 1 < argc) {
                const std::string arch_name = argv[++i];
                if (llm_arch_from_string(arch_name) != LLM_ARCH_UNKNOWN) {
                    // exact architecture name
                    arch_filter = "^" + arch_name + "$";
                } else {
                    try {
                        std::regex re(arch_name);
                        arch_filter = arch_name;
                    } catch (const std::regex_error & err) {
                        LOG_ERR("%s: invalid architecture regex: %s (%s)\n", __func__, arch_name.c_str(), err.what());
                        return 1;
                    }
                }
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--seed") == 0) {
            if (i + 1 < argc) {
                seed = std::stoull(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--stdev") == 0) {
            if (i + 1 < argc) {
                stdev = std::stof(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-v") == 0) {
            if (i + 1 < argc) {
                verbosity = std::stoull(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--out") == 0) {
            if (i + 1 < argc) {
                out = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--backend") == 0) {
            if (i + 1 < argc) {
                const char * backend_name = argv[++i];
                ggml_backend_dev_t dev = ggml_backend_dev_by_name(backend_name);
                if (dev == nullptr) {
                    LOG_ERR("%s: unknown backend device: %s\n", __func__, backend_name);
                    return 1;
                }
                target_backend = ggml_backend_dev_name(dev);
            } else {
                usage(argv);
                return 1;
            }
        } else {
            LOG_ERR("%s: unknown argument: %s\n", __func__, argv[i]);
            usage(argv);
            return 1;
        }
    }
    if (test_phase_workspace || test_live_context_workspace || run_speculative_limits || run_mtp_draft_vocab || run_moe_cache_selector ||
        run_moe_placement || run_moe_joint_measurement || profile_model_path || moe_replay_path || run_source_variants || run_source_auxiliary) {
        common_log_set_verbosity_thold(verbosity);
    }
    if (stdev <= 0.0f) {
        LOG_ERR("%s: stdev must be > 0\n", __func__);
        return 1;
    }
    LOG_INF("%s: using seed %zu, stdev %f\n", __func__, seed, stdev);

    try {
        if (!profile_model_path && (profile_corpus_path || profile_output_path)) { throw std::runtime_error("profile inputs require --collect-moe-profile"); }
        if (profile_model_path || hybrid_args.size() > 1) {
            common_params hybrid_params;
            std::vector<char *> hybrid_argv;
            for (auto & argument : hybrid_args) { hybrid_argv.push_back(argument.data()); }
            if (!common_params_parse(hybrid_argv.size(), hybrid_argv.data(), hybrid_params, LLAMA_EXAMPLE_SERVER)) { return 1; }
            if (profile_model_path && hybrid_params.moe_hybrid.empty()) { hybrid_params.moe_hybrid = "on"; }
            if (profile_model_path && hybrid_params.moe_hybrid == "off") { throw std::runtime_error("corpus collection requires --moe-hybrid on"); }
            common_moe_hybrid_configure(hybrid_params);
        }
        if (profile_model_path) { return collect_model_moe_corpus(profile_model_path, profile_corpus_path, profile_output_path, replay_cache_slots, profile_threads, replay_load_mode); }
        if (run_mtp_draft_vocab) {
            test_mtp_draft_vocab(seed, stdev);
            return 0;
        }
        if (test_phase_workspace) {
            test_mtp_state_dependencies();
            test_phase_workspace_runtime_reserve(seed, stdev);
            test_mtp_graph_reactivation(seed, stdev);
            test_phase_workspace_mtp_lifecycle(seed, stdev);
            test_phase_workspace_qwen4exp_mtp_reserve(seed, stdev);
            test_phase_workspace_mismatched_placement(seed, stdev);
            test_mtp_input_staging(seed, stdev);
            test_phase_workspace_late_pipeline_fallback(seed, stdev);
            return 0;
        }
        if (test_live_context_workspace) {
            test_live_context_workspace_reserve(seed, stdev);
            test_live_context_workspace_iswa_reserve(seed, stdev);
            test_live_context_workspace_indexer_reserve(seed, stdev);
            test_live_context_workspace_unsupported(seed, stdev);
            return 0;
        }
        if (run_source_variants) { return test_source_graph_variants(seed, stdev, replay_reference_path, replay_read); }
        if (run_source_auxiliary) { return test_source_auxiliary_frontend(seed, stdev, replay_reference_path, replay_read, auxiliary_uncached_reference, auxiliary_no_output_first, auxiliary_source_statistics, auxiliary_source_ranks, auxiliary_inherit_statistics, auxiliary_different_model, auxiliary_profile_file, auxiliary_parent_active, auxiliary_explicit_profile, auxiliary_sparse_outputs); }
        if (run_speculative_limits) {
            test_speculative_limits(seed, stdev);
            return 0;
        }
        if (run_moe_cache_selector) {
            return test_moe_cache_selector_precedence(seed, stdev);
        }
        if (source_uses_path) { return test_model_source_uses(source_uses_path); }
        if (source_profile_path) { return test_model_source_profile(source_profile_path, source_statistics_file); }
        if (source_learning_path) {
            if (source_learning_context < source_learning_batch + 16) { return 1; }
            return test_model_source_learning(source_learning_path, source_learning_output, source_learning_slots, source_learning_context, source_learning_batch, source_learning_adaptation);
        }
        if (source_statistics) { return test_source_profile_statistics(); }
        if (moe_replay_path) { return test_model_moe_replay(moe_replay_path, replay_reference_path, replay_read, replay_independent_sources, replay_cache_slots, replay_cpu_oracle, replay_gpu_oracle, replay_load_mode, replay_prompt_path, replay_rows, replay_split); }
        if (run_moe_placement) {
            return test_moe_placement_report(seed, stdev);
        }
        if (run_moe_joint_measurement) {
            return test_moe_joint_measurement(seed, stdev, fit_tool);
        }
        if (!out.empty()) {
            return save_models(arch_filter, seed, stdev, verbosity, out);
        }
        return test_backends(arch_filter, seed, stdev, verbosity, target_backend);
    } catch (const std::exception & err) {
        fprintf(stderr, "encountered runtime error: %s\n", err.what());
        return -1;
    }
}
