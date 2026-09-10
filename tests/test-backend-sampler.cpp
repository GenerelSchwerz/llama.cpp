#include "ggml.h"
#include "llama.h"
#include "llama-cpp.h"
#include "common.h"
#include "../src/llama-ext.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct test_args {
    std::string model;
    std::string test;
    std::string device = "auto";
    int32_t staged_input_cache = 0;
};

struct test_params {
    llama_model_ptr model;
    bool sampled_decode = false;
    bool sampled_decode_gpu = false;
    bool sampled_stochastic = false;
    bool sampled_host_inputs = false;
};

static llama_model_ptr load_model(const test_args & args) {
    auto mparams = llama_model_default_params();
    if (args.staged_input_cache > 0) {
        mparams.moe_expert_cache_slots = args.staged_input_cache;
        mparams.load_mode = LLAMA_LOAD_MODE_NONE;
        mparams.lazy_mode = LLAMA_LAZY_MODE_ON;
    }

    ggml_backend_dev_t devs[2] = { nullptr, nullptr };
    llama_model_tensor_buft_override overrides[2] = {};

    if (args.device != "auto") {
        if (args.device == "gpu") {
            devs[0] = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);

            if (devs[0] == nullptr) {
                fprintf(stderr, "Error: GPU requested but not available\n");
                return nullptr;
            }

            mparams.n_gpu_layers = 999;
            if (!args.staged_input_cache && (args.test == "decode_sampled" || args.test == "decode_sampled_batch" || args.test == "decode_sampled_lifecycle")) {
                overrides[0] = {"token_embd.weight", ggml_backend_dev_buffer_type(devs[0])};
                mparams.tensor_buft_overrides = overrides;
            }
        } else if (args.device == "cpu") {
            devs[0] = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);

            mparams.n_gpu_layers = 0;
        } else {
            fprintf(stderr, "Error: invalid device '%s'\n", args.device.c_str());
            return nullptr;
        }

        mparams.devices = devs;

        fprintf(stderr, "Using device: %s\n", ggml_backend_dev_name(devs[0]));
    }

    llama_model_ptr res;

    res.reset(llama_model_load_from_file(args.model.c_str(), mparams));

    if (!res) {
        fprintf(stderr, "Warning: failed to load model '%s', skipping test\n", args.model.c_str());
        return nullptr;
    }

    return res;
}

struct test_context {
    llama_context_ptr ctx;

    int n_vocab = 0;

    const llama_vocab * vocab = nullptr;

    std::unordered_map<llama_seq_id, int32_t> seq_positions;
    std::unordered_map<llama_seq_id, int32_t> last_batch_info;

    test_context(
            const test_params & params,
            std::vector<llama_sampler_seq_config> & configs,
            int32_t n_seq_max = -1,
            uint32_t n_outputs_max = 0,
            uint32_t n_ubatch = 0,
            uint32_t n_outputs_max_per_seq = 1,
            bool kv_unified = true,
            uint32_t n_ctx = 512) {
        auto * model = params.model.get();

        GGML_ASSERT(model);
        GGML_ASSERT(!ctx);

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = n_ctx;
        cparams.n_batch = 512;
        if (n_ubatch > 0) {
            cparams.n_ubatch = n_ubatch;
        }
        cparams.n_outputs_max = n_outputs_max;
        cparams.n_outputs_max_per_seq = n_outputs_max_per_seq;
        cparams.samplers = configs.data();
        cparams.n_samplers = configs.size();
        cparams.kv_unified = kv_unified;
        cparams.n_rs_seq = params.sampled_decode ? 1 : 0;
        if (params.sampled_host_inputs) {
            cparams.type_k = GGML_TYPE_Q8_0;
            cparams.type_v = GGML_TYPE_Q8_0;
            cparams.n_threads = 12;
            cparams.n_threads_batch = 12;
        }

        // If n_seq_max is not specified, calculate it from configs
        if (n_seq_max < 0) {
            int32_t max_seq_id = 0;
            for (const auto & config : configs) {
                max_seq_id = std::max(config.seq_id, max_seq_id);
            }
            cparams.n_seq_max = max_seq_id + 1;
        } else {
            cparams.n_seq_max = n_seq_max;
        }

        ctx.reset(llama_init_from_model(model, cparams));
        if (!ctx) {
            throw std::runtime_error("failed to create context");
        }

        vocab = llama_model_get_vocab(model);
        n_vocab = llama_vocab_n_tokens(vocab);
    }

    bool decode(const std::map<llama_seq_id, std::string> & prompts) {
        GGML_ASSERT(ctx);

        last_batch_info.clear();
        llama_batch batch = llama_batch_init(512, 0, prompts.size());

        for (const auto & [seq_id, prompt] : prompts) {
            std::vector<llama_token> tokens;
            tokens.push_back(llama_vocab_bos(vocab));

            std::vector<llama_token> prompt_tokens(32);
            int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.length(),
                                           prompt_tokens.data(), prompt_tokens.size(),
                                           false, false);
            if (n_tokens < 0) {
                fprintf(stderr, "Warning: tokenization failed for seq_id %d\n", seq_id);
                llama_batch_free(batch);
                return false;
            }

            for (int i = 0; i < n_tokens; i++) {
                tokens.push_back(prompt_tokens[i]);
            }

            if (seq_positions.find(seq_id) == seq_positions.end()) {
                seq_positions[seq_id] = 0;
            }

            int32_t start_pos = seq_positions[seq_id];
            for (size_t i = 0; i < tokens.size(); i++) {
                common_batch_add(batch, tokens[i], start_pos + i, { seq_id }, i == tokens.size() - 1);
            }

            seq_positions[seq_id] = start_pos + tokens.size();
        }


        printf("Batch contents:\n");
        printf("n_tokens: %d\n", batch.n_tokens);
        for (int i = 0; i < batch.n_tokens; i++) {
            printf("token[%d]: tok=%-5d, pos=%d, n_seq_id=%d, seq_ids=[", i, batch.token[i], batch.pos[i], batch.n_seq_id[i]);

            for (int j = 0; j < batch.n_seq_id[i]; j++) {
                printf("%d%s", batch.seq_id[i][j], j < batch.n_seq_id[i]-1 ? ", " : "");
            }
            printf("], logits=%d\n", batch.logits[i]);
        }

        if (llama_decode(ctx.get(), batch) != 0) {
            fprintf(stderr, "Warning: llama_decode failed\n");
            llama_batch_free(batch);
            return false;
        }

        // Build mapping from seq id to batch token idx
        for (int i = 0; i < batch.n_tokens; i++) {
            if (batch.logits[i]) {
                llama_seq_id seq_id = batch.seq_id[i][0];
                last_batch_info[seq_id] = i;
            }
        }

        llama_batch_free(batch);
        return true;
    }

    int32_t idx_for_seq(llama_seq_id seq_id) {
        auto it = last_batch_info.find(seq_id);
        if (it == last_batch_info.end()) {
            fprintf(stderr, "Error: no batch index found for seq_id %d\n", seq_id);
            return -1;
        }
        return it->second;
    }

    void update_batch_info(const llama_batch & batch) {
        last_batch_info.clear();
        for (int i = 0; i < batch.n_tokens; i++) {
            if (batch.logits[i]) {
                llama_seq_id cur_seq = batch.seq_id[i][0];
                last_batch_info[cur_seq] = i;
            }
        }
    }

    bool decode_token(llama_token token, llama_seq_id seq_id = 0) {
        GGML_ASSERT(ctx);

        llama_batch batch = llama_batch_init(1, 0, 1);
        int32_t pos = seq_positions[seq_id];
        common_batch_add(batch, token, pos, { seq_id }, true);

        if (llama_decode(ctx.get(), batch) != 0) {
            fprintf(stderr, "Warning: llama_decode failed for token %d in seq %d\n", token, seq_id);
            llama_batch_free(batch);
            return false;
        }

        update_batch_info(batch);

        seq_positions[seq_id]++;
        llama_batch_free(batch);

        return true;
    }

    bool decode_tokens(const std::map<llama_seq_id, llama_token> & seq_tokens) {
        GGML_ASSERT(ctx);

        llama_batch batch = llama_batch_init(seq_tokens.size(), 0, seq_tokens.size());

        for (const auto & [seq_id, token] : seq_tokens) {
            int32_t pos = seq_positions[seq_id];
            common_batch_add(batch, token, pos, { seq_id }, true);
        }

        if (llama_decode(ctx.get(), batch) != 0) {
            fprintf(stderr, "Warning: llama_decode failed for batch tokens\n");
            llama_batch_free(batch);
            return false;
        }

        for (const auto & [seq_id, _] : seq_tokens) {
            seq_positions[seq_id]++;
        }

        update_batch_info(batch);

        llama_batch_free(batch);

        return true;
    }

    std::string token_to_piece(llama_token token, bool special) const {
        std::string piece;
        piece.resize(piece.capacity());  // using string internal cache, 15 bytes + '\n'
        const int n_chars = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
        if (n_chars < 0) {
            piece.resize(-n_chars);
            int check = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
            GGML_ASSERT(check == -n_chars);
        } else {
            piece.resize(n_chars);
        }

        return piece;
    }
};

struct test_single_output_backend_sampler {
    bool backend_initialized = false;
    uint32_t backend_outputs_max_per_seq = 0;
    int backend_apply_count = 0;
    int apply_count = 0;
};

static const char * test_single_output_backend_sampler_name(const llama_sampler * /*smpl*/) {
    return "single-output-backend";
}

static void test_single_output_backend_sampler_apply(
        llama_sampler * smpl, llama_token_data_array * /*cur_p*/) {
    auto * ctx = (test_single_output_backend_sampler *) smpl->ctx;
    ctx->apply_count++;
}

static void test_single_output_backend_sampler_free(llama_sampler * smpl) {
    delete (test_single_output_backend_sampler *) smpl->ctx;
}

static bool test_single_output_backend_sampler_backend_init(
        llama_sampler * smpl, ggml_backend_buffer_type_t /*buft*/, uint32_t n_outputs_max_per_seq) {
    auto * ctx = (test_single_output_backend_sampler *) smpl->ctx;
    ctx->backend_outputs_max_per_seq = n_outputs_max_per_seq;
    if (n_outputs_max_per_seq > 1) {
        return false;
    }
    ctx->backend_initialized = true;
    return true;
}

static void test_single_output_backend_sampler_backend_apply(
        llama_sampler * smpl, ggml_context * /*ctx*/, ggml_cgraph * /*gf*/, llama_sampler_data * /*data*/) {
    auto * ctx = (test_single_output_backend_sampler *) smpl->ctx;
    ctx->backend_apply_count++;
}

static llama_sampler_i test_single_output_backend_sampler_i = {
    /* .name              = */ test_single_output_backend_sampler_name,
    /* .accept            = */ nullptr,
    /* .apply             = */ test_single_output_backend_sampler_apply,
    /* .reset             = */ nullptr,
    /* .clone             = */ nullptr,
    /* .free              = */ test_single_output_backend_sampler_free,
    /* .backend_init      = */ test_single_output_backend_sampler_backend_init,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ test_single_output_backend_sampler_backend_apply,
    /* .backend_set_input = */ nullptr,
    /* .backend_reset     = */ nullptr,
    /* .copy_state        = */ nullptr,
};

static llama_sampler * test_single_output_backend_sampler_init(
        test_single_output_backend_sampler ** sampler_ctx) {
    auto * ctx = new test_single_output_backend_sampler;
    *sampler_ctx = ctx;
    return llama_sampler_init(&test_single_output_backend_sampler_i, ctx);
}

static void test_backend_greedy_sampling(const test_params & params) {
    const int seq_id = 0;

    struct llama_sampler_chain_params backend_sampler_params = llama_sampler_chain_default_params();
    llama_sampler_ptr backend_sampler_chain(llama_sampler_chain_init(backend_sampler_params));

    llama_sampler_chain_add(backend_sampler_chain.get(), llama_sampler_init_greedy());
    std::vector<llama_sampler_seq_config> backend_sampler_configs = {{ seq_id, backend_sampler_chain.get() }};

    test_context test_ctx(params, backend_sampler_configs);

    if (!test_ctx.decode({{seq_id, "Some"}})) {
        GGML_ASSERT(false && "Failed to decode token");
    }

    int32_t batch_idx = test_ctx.idx_for_seq(seq_id);

    llama_token token = llama_get_sampled_token_ith(test_ctx.ctx.get(), batch_idx);
    printf("greedy sampled id:%d, string:'%s'\n", token, test_ctx.token_to_piece(token, false).c_str());
    GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);

    token = llama_get_sampled_token_ith(test_ctx.ctx.get(), -1);
    printf("greedy sampled id:%d, string:'%s'\n", token, test_ctx.token_to_piece(token, false).c_str());
    GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);

    for (int i = 0; i < 10; i++) {
        int32_t loop_idx = test_ctx.idx_for_seq(seq_id);
        llama_token token = llama_get_sampled_token_ith(test_ctx.ctx.get(), loop_idx);
        printf("Generation step %d: token id:%d, string: %s\n", i, token, test_ctx.token_to_piece(token, false).c_str());
        if (!test_ctx.decode_token(token, 0)) {
            GGML_ASSERT(false && "Failed to decode token");
        }
    }
}

static void test_backend_top_k_sampling(const test_params & params) {
    const int seq_id = 0;
    const int32_t k = 8;
    struct llama_sampler_chain_params backend_chain_params = llama_sampler_chain_default_params();
    llama_sampler_ptr backend_sampler_chain(llama_sampler_chain_init(backend_chain_params));
    llama_sampler_chain_add(backend_sampler_chain.get(), llama_sampler_init_top_k(k));
    std::vector<llama_sampler_seq_config> backend_sampler_configs = {{ seq_id, backend_sampler_chain.get() }};

    test_context test_ctx(params, backend_sampler_configs);

    if (!test_ctx.decode({{seq_id, "Hello"}})) {
        GGML_ASSERT(false && "Failed to decode token");
    }

    int32_t batch_idx = test_ctx.idx_for_seq(seq_id);

    float * logits = llama_get_sampled_logits_ith(test_ctx.ctx.get(), batch_idx);
    uint32_t n_logits = llama_get_sampled_logits_count_ith(test_ctx.ctx.get(), batch_idx);
    for (size_t i = 0; i < n_logits; ++i) {
        printf("top_k logit[%zu] = %.6f\n", i, logits[i]);
    }

    llama_token * candidates = llama_get_sampled_candidates_ith(test_ctx.ctx.get(), batch_idx);
    uint32_t n_candidates = llama_get_sampled_candidates_count_ith(test_ctx.ctx.get(), batch_idx);
    for (size_t i = 0; i < n_candidates; ++i) {
        printf("top_k candidate[%zu] = %d : %s\n", i, candidates[i],
               test_ctx.token_to_piece(candidates[i], false).c_str());
    }

    // Sample using CPU sampler for verification that it is possible to do hybrid
    // sampling, first top_k on the backend and then dist on the CPU.
    struct llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    llama_sampler_ptr chain(llama_sampler_chain_init(chain_params));
    GGML_ASSERT(chain->iface->backend_apply != nullptr);

    llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(18));
    llama_token token = llama_sampler_sample(chain.get(), test_ctx.ctx.get(), batch_idx);
    GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);

    printf("backend top-k hybrid sampling test PASSED\n");
}

static void test_backend_temp_sampling(const test_params & params) {
    {
        const float temp_0 = 0.8f;
        struct llama_sampler_chain_params backend_chain_params_0 = llama_sampler_chain_default_params();
        llama_sampler_ptr backend_sampler_chain_0(llama_sampler_chain_init(backend_chain_params_0));
        llama_sampler_chain_add(backend_sampler_chain_0.get(), llama_sampler_init_temp(temp_0));

        const float temp_1 = 0.1f;
        struct llama_sampler_chain_params backend_chain_params_1 = llama_sampler_chain_default_params();
        llama_sampler_ptr backend_sampler_chain_1(llama_sampler_chain_init(backend_chain_params_1));
        llama_sampler_chain_add(backend_sampler_chain_1.get(), llama_sampler_init_temp(temp_1));

        std::vector<llama_sampler_seq_config> backend_sampler_configs = {
            { 0, backend_sampler_chain_0.get() },
            { 1, backend_sampler_chain_1.get() }
        };

        test_context test_ctx(params, backend_sampler_configs);

        if (!test_ctx.decode({{0, "Some where over the"}, {1, "Once upon a"}})) {
            GGML_ASSERT(false && "Failed to decode token");
        }

        // Verify sequence 0
        {
            int32_t batch_idx = test_ctx.idx_for_seq(0);
            int n_logits = llama_get_sampled_logits_count_ith(test_ctx.ctx.get(), batch_idx);
            GGML_ASSERT(n_logits == test_ctx.n_vocab);

            // Sample from sequence 0 using CPU sampler
            struct llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
            llama_sampler_ptr chain(llama_sampler_chain_init(chain_params));
            llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(18));

            llama_token token = llama_sampler_sample(chain.get(), test_ctx.ctx.get(), batch_idx);
            const std::string token_str = test_ctx.token_to_piece(token, false);
            printf("Sequence 0 sampled token id:%d, string: '%s'\n", token, token_str.c_str());
            GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);
        }


        // Verify sequence 1
        {
            int32_t batch_idx = test_ctx.idx_for_seq(1);

            // Sample from sequence 1 using CPU sampler
            struct llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
            llama_sampler_ptr chain(llama_sampler_chain_init(chain_params));
            llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(18));

            llama_token token = llama_sampler_sample(chain.get(), test_ctx.ctx.get(), batch_idx);
            const std::string token_str = test_ctx.token_to_piece(token, false);
            printf("Sequence 1 sampled token id:%d, string: '%s'\n", token, token_str.c_str());
            GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);
        }
    }

    // lambda for testing non-positive temperature values.
    auto test_argmax_temp = [&](float temp) {
        printf("\nTesting temperature = %.1f\n", temp);

        int seq_id = 0;
        struct llama_sampler_chain_params backend_chain_params = llama_sampler_chain_default_params();
        llama_sampler_ptr backend_sampler_chain(llama_sampler_chain_init(backend_chain_params));
        llama_sampler_chain_add(backend_sampler_chain.get(), llama_sampler_init_temp(temp));

        std::vector<llama_sampler_seq_config> backend_sampler_configs = {
            { seq_id, backend_sampler_chain.get() },
        };

        test_context test_ctx(params, backend_sampler_configs);

        if (!test_ctx.decode({{seq_id, "Once"}})) {
            GGML_ASSERT(false && "Failed to decode token");
        }

        int32_t batch_idx = test_ctx.idx_for_seq(seq_id);

        uint32_t n_logits = llama_get_sampled_logits_count_ith(test_ctx.ctx.get(), batch_idx);
        GGML_ASSERT(n_logits == 1);
    };

    test_argmax_temp(0.0f);
    test_argmax_temp(-1.0f);

    printf("backend temp sampling test PASSED\n");
}

static void test_backend_temp_ext_sampling(const test_params & params) {
    {
        int seq_id = 0;
        const float temp = 0.8f;
        const float delta = 0.5f;
        const float exponent = 1.5f;
        struct llama_sampler_chain_params backend_chain_params = llama_sampler_chain_default_params();
        llama_sampler_ptr backend_sampler_chain(llama_sampler_chain_init(backend_chain_params));
        llama_sampler_chain_add(backend_sampler_chain.get(), llama_sampler_init_temp_ext(temp, delta, exponent));

        std::vector<llama_sampler_seq_config> backend_sampler_configs = {
            { seq_id, backend_sampler_chain.get() },
        };

        test_context test_ctx(params, backend_sampler_configs);

        if (!test_ctx.decode({{seq_id, "Once upon a"}})) {
            GGML_ASSERT(false && "Failed to decode token");
        }

        // Verify sequence 0
        {
            int32_t batch_idx = test_ctx.idx_for_seq(seq_id);
            int n_logits = llama_get_sampled_logits_count_ith(test_ctx.ctx.get(), batch_idx);
            GGML_ASSERT(n_logits == test_ctx.n_vocab);
        }
    }

    // lambda for testing non-positive temp/delta/exponent values.
    auto test_argmax_temp = [&](float temp, float delta, float exponent) {
        printf("\nTesting temperature = %.1f, delta = %1.f, exponent = %1.f\n", temp, delta, exponent);

        int seq_id = 0;
        struct llama_sampler_chain_params backend_chain_params = llama_sampler_chain_default_params();
        llama_sampler_ptr backend_sampler_chain(llama_sampler_chain_init(backend_chain_params));
        llama_sampler_chain_add(backend_sampler_chain.get(), llama_sampler_init_temp_ext(temp, delta, exponent));

        std::vector<llama_sampler_seq_config> backend_sampler_configs = {
            { seq_id, backend_sampler_chain.get() },
        };

        test_context test_ctx(params, backend_sampler_configs);

        if (!test_ctx.decode({{seq_id, "Once"}})) {
            GGML_ASSERT(false && "Failed to decode token");
        }

        int32_t batch_idx = test_ctx.idx_for_seq(seq_id);

        uint32_t n_logits = llama_get_sampled_logits_count_ith(test_ctx.ctx.get(), batch_idx);

        if (temp <= 0.0f && delta >= 0.0f) {
            GGML_ASSERT(n_logits == 1);
        } else {
            GGML_ASSERT(n_logits == (uint32_t) test_ctx.n_vocab);
        }
    };

    test_argmax_temp(0.0f,  0.3f, 1.0f); // Greedy (temp=0)
    test_argmax_temp(-1.0f, 0.3f, 2.0f); // Greedy (temp<0)
    test_argmax_temp(0.8f,  0.0f, 2.0f); // Temperature scaling

    printf("backend temp_ext sampling test PASSED\n");
}

static void test_backend_min_p_sampling(const test_params & params) {
    const int seq_id = 0;
    const float p = 0.1;
    struct llama_sampler_chain_params backend_chain_params = llama_sampler_chain_default_params();
    llama_sampler_ptr backend_sampler_chain(llama_sampler_chain_init(backend_chain_params));
    llama_sampler_chain_add(backend_sampler_chain.get(), llama_sampler_init_min_p(p, 0));
    std::vector<llama_sampler_seq_config> backend_sampler_configs = {{ seq_id, backend_sampler_chain.get() }};

    test_context test_ctx(params, backend_sampler_configs);

    if (!test_ctx.decode({{seq_id, "Hello"}})) {
        GGML_ASSERT(false && "Failed to decode token");
    }

    int32_t batch_idx = test_ctx.idx_for_seq(seq_id);

    float * logits = llama_get_sampled_logits_ith(test_ctx.ctx.get(), batch_idx);
    uint32_t n_logits = llama_get_sampled_logits_count_ith(test_ctx.ctx.get(), batch_idx);

    // Print the logits that are above the min-p threshold
    std::vector<float> filtered_logits;
    for (size_t i = 0; i < n_logits; ++i) {
        if (logits[i] > -1e9f) {
            filtered_logits.push_back(logits[i]);
            //printf("min_p logit[%zu] = %.6f\n", i, logits[i]);
        }
    }
    GGML_ASSERT(filtered_logits.size() < (size_t) test_ctx.n_vocab);

    // Sample using CPU sampler for verification to inspect they are reasonable
    struct llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    llama_sampler_ptr chain(llama_sampler_chain_init(chain_params));
    llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(88));

    llama_token token = llama_sampler_sample(chain.get(), test_ctx.ctx.get(), batch_idx);
    const std::string token_str = test_ctx.token_to_piece(token, false);
    printf("min-p cpu sampled token id:%d, string: '%s'\n", token, token_str.c_str());
    GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);

    // Decode and sample 10 more tokens
    for (int i = 0; i < 10; i++) {
        int32_t loop_idx = test_ctx.idx_for_seq(seq_id);
        llama_token token = llama_sampler_sample(chain.get(), test_ctx.ctx.get(), loop_idx);
        printf("min-p gen step %d: token id :%5.d, string: %s\n", i, token, test_ctx.token_to_piece(token, false).c_str());
        if (!test_ctx.decode_token(token, 0)) {
            GGML_ASSERT(false && "Failed to decode token");
        }
    }

    printf("min-p sampling test PASSED\n");
}

static void test_backend_top_p_sampling(const test_params & params) {
    const int seq_id = 0;
    const float p = 0.9;
    struct llama_sampler_chain_params backend_chain_params = llama_sampler_chain_default_params();
    llama_sampler_ptr backend_sampler_chain(llama_sampler_chain_init(backend_chain_params));
    llama_sampler_chain_add(backend_sampler_chain.get(), llama_sampler_init_top_p(p, 0));
    std::vector<llama_sampler_seq_config> backend_sampler_configs = {{ seq_id, backend_sampler_chain.get() }};

    test_context test_ctx(params, backend_sampler_configs);

    if (!test_ctx.decode({{seq_id, "Hello"}})) {
        return;
    }

    int32_t batch_idx = test_ctx.idx_for_seq(seq_id);

    float * logits = llama_get_sampled_logits_ith(test_ctx.ctx.get(), batch_idx);
    uint32_t n_logits = llama_get_sampled_logits_count_ith(test_ctx.ctx.get(), batch_idx);

    // Print the logits that are above the min-p threshold
    std::vector<float> filtered_logits;
    for (size_t i = 0; i < n_logits; ++i) {
        if (logits[i] > -1e9f) {
            filtered_logits.push_back(logits[i]);
        }
    }
    GGML_ASSERT(filtered_logits.size() < (size_t) test_ctx.n_vocab);
    GGML_ASSERT(filtered_logits.size() > 0);

    // Sample using CPU sampler for verification to inspect they are reasonable
    struct llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    llama_sampler_ptr chain(llama_sampler_chain_init(chain_params));
    llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(88));

    llama_token token = llama_sampler_sample(chain.get(), test_ctx.ctx.get(), batch_idx);
    const std::string token_str = test_ctx.token_to_piece(token, false);
    printf("top-p cpu sampled token id:%d, string: '%s'\n", token, token_str.c_str());
    GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);

    // Decode and sample 10 more tokens
    for (int i = 0; i < 10; i++) {
        int32_t loop_idx = test_ctx.idx_for_seq(seq_id);
        llama_token token = llama_sampler_sample(chain.get(), test_ctx.ctx.get(), loop_idx);
        printf("top-p gen step %d: token id :%5.d, string: %s\n", i, token, test_ctx.token_to_piece(token, false).c_str());
        test_ctx.decode_token(token, 0);
    }

    printf("top-p sampling test PASSED\n");
}

static void test_backend_multi_sequence_sampling(const test_params & params) {
    struct llama_sampler_chain_params chain_params_0 = llama_sampler_chain_default_params();
    llama_sampler_ptr sampler_chain_0(llama_sampler_chain_init(chain_params_0));
    llama_sampler_chain_add(sampler_chain_0.get(), llama_sampler_init_greedy());

    struct llama_sampler_chain_params chain_params_1 = llama_sampler_chain_default_params();
    llama_sampler_ptr sampler_chain_1(llama_sampler_chain_init(chain_params_1));
    llama_sampler_chain_add(sampler_chain_1.get(), llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(sampler_chain_1.get(), llama_sampler_init_greedy());

    std::vector<llama_sampler_seq_config> backend_sampler_configs = {
        { 0, sampler_chain_0.get() },
        { 1, sampler_chain_1.get() }
    };

    test_context test_ctx(params, backend_sampler_configs);

    std::map<llama_seq_id, std::string> prompts = {
        {0, "Hello"},
        {1, "Some"}
    };

    if (!test_ctx.decode(prompts)) {
        GGML_ASSERT(false && "Failed to decode token");
    }

    // Verify sequence 0
    {
        int32_t batch_idx = test_ctx.idx_for_seq(0);
        llama_token token = llama_get_sampled_token_ith(test_ctx.ctx.get(), batch_idx);
        const std::string token_str = test_ctx.token_to_piece(token, false);
        printf("Seq 0 sampled token id=%d, string='%s'\n", token, token_str.c_str());
        GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);
    }

    // Verify sequence 1
    {
        int32_t batch_idx= test_ctx.idx_for_seq(1);
        llama_token token = llama_get_sampled_token_ith(test_ctx.ctx.get(), batch_idx);
        const std::string token_str = test_ctx.token_to_piece(token, false);
        printf("Seq 1 sampled token id=%d, string='%s'\n", token, token_str.c_str());
        GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);
    }

    // Generate tokens for each sequence
    printf("\nMulti-sequence generation:\n");
    for (int step = 0; step < 4; step++) {
        std::map<llama_seq_id, llama_token> tokens;

        for (llama_seq_id seq_id : {0, 1}) {
            int32_t idx = test_ctx.idx_for_seq(seq_id);
            llama_token token = llama_get_sampled_token_ith(test_ctx.ctx.get(), idx);
            const std::string token_str = test_ctx.token_to_piece(token, false);
            printf("  Seq %d, step %d: token id=%d, string='%s'\n", seq_id, step, token, token_str.c_str());
            tokens[seq_id] = token;
        }

        // Decode all tokens in a single batch
        if (!test_ctx.decode_tokens(tokens)) {
            GGML_ASSERT(false && "Failed to decode token");
        }
    }

    printf("backend multi-sequence sampling test PASSED\n");
}

static void test_backend_dist_sampling(const test_params & params) {
    const int seq_id = 0;
    const int32_t seed = 88;

    struct llama_sampler_chain_params backend_chain_params = llama_sampler_chain_default_params();
    llama_sampler_ptr backend_sampler_chain(llama_sampler_chain_init(backend_chain_params));
    llama_sampler_chain_add(backend_sampler_chain.get(), llama_sampler_init_dist(seed));
    std::vector<llama_sampler_seq_config> backend_sampler_configs = {{ seq_id, backend_sampler_chain.get() }};

    test_context test_ctx(params, backend_sampler_configs);

    if (!test_ctx.decode({{seq_id, "Some"}})) {
        GGML_ASSERT(false && "Failed to decode token");
    }

    int32_t batch_idx = test_ctx.idx_for_seq(seq_id);
    llama_token token = llama_get_sampled_token_ith(test_ctx.ctx.get(), batch_idx);
    printf("dist sampled id:%d, string:'%s'\n", token, test_ctx.token_to_piece(token, false).c_str());
    GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);
    //GGML_ASSERT(llama_get_sampled_logits_ith(test_ctx.ctx.get(), batch_idx) == nullptr);

    token = llama_get_sampled_token_ith(test_ctx.ctx.get(), -1);
    printf("dist sampled id:%d, string:'%s'\n", token, test_ctx.token_to_piece(token, false).c_str());
    GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);

    printf("backend dist sampling test PASSED\n");
}

static void test_backend_dist_sampling_and_cpu(const test_params & params) {
    const int seq_id = 0;
    const int32_t seed = 88;

    struct llama_sampler_chain_params backend_chain_params = llama_sampler_chain_default_params();
    llama_sampler_ptr backend_sampler_chain(llama_sampler_chain_init(backend_chain_params));
    llama_sampler_chain_add(backend_sampler_chain.get(), llama_sampler_init_dist(seed));
    std::vector<llama_sampler_seq_config> backend_sampler_configs = {{ seq_id, backend_sampler_chain.get() }};

    test_context test_ctx(params, backend_sampler_configs);

    if (!test_ctx.decode({{seq_id, "Some"}})) {
        GGML_ASSERT(false && "Failed to decode token");
    }

    int32_t batch_idx = test_ctx.idx_for_seq(seq_id);

    // Sample using CPU sampler
    struct llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    llama_sampler_ptr chain(llama_sampler_chain_init(chain_params));
    llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(18));

    llama_token backend_token = llama_get_sampled_token_ith(test_ctx.ctx.get(), batch_idx);
    llama_token cpu_token = llama_sampler_sample(chain.get(), test_ctx.ctx.get(), batch_idx);
    printf("dist & cpu sampled id:%d, string:'%s'\n", cpu_token, test_ctx.token_to_piece(cpu_token, false).c_str());
    GGML_ASSERT(backend_token == cpu_token);

    printf("backend dist & cpu sampling test PASSED\n");
}

static void test_backend_logit_bias_sampling(const test_params & params) {
    const auto * model = params.model.get();
    const auto * vocab = llama_model_get_vocab(model);

    const int seq_id = 0;

    std::vector<llama_logit_bias> logit_bias;

    // Get the token for the piece "World".
    const std::string piece = "World";
    std::vector<llama_token> tokens(16);
    llama_tokenize(vocab, piece.c_str(), piece.size(), tokens.data(), tokens.size(), false, false);

    llama_token bias_token = tokens[0];
    // TODO: biasing too much here makes the Vulkan sampling fail - should be investigated further
    //       https://github.com/ggml-org/llama.cpp/actions/runs/20894267644/job/60030252675?pr=18753#step:3:23350
    //logit_bias.push_back({ bias_token, +100.0f });
    logit_bias.push_back({ bias_token, +10.0f });
    const llama_token suppressed = (bias_token + 1) % llama_vocab_n_tokens(vocab);
    logit_bias.push_back({ suppressed, +100.0f });
    logit_bias.push_back({ suppressed, -INFINITY });

    printf("biasing token piece '%s' -> token id %d\n", piece.c_str(), bias_token);

    struct llama_sampler_chain_params backend_chain_params = llama_sampler_chain_default_params();
    llama_sampler_ptr backend_sampler_chain(llama_sampler_chain_init(backend_chain_params));
    llama_sampler_chain_add(backend_sampler_chain.get(), llama_sampler_init_logit_bias(
                llama_vocab_n_tokens(vocab),
                logit_bias.size(),
                logit_bias.data()));
    llama_sampler_chain_add(backend_sampler_chain.get(), llama_sampler_init_dist(88));

    std::vector<llama_sampler_seq_config> backend_sampler_configs = {
        { seq_id, backend_sampler_chain.get() },
    };

    test_context test_ctx(params, backend_sampler_configs);

    if (!test_ctx.decode({{seq_id, "Hello"}})) {
        GGML_ASSERT(false && "Failed to decode token");
    }

    llama_token backend_token = llama_get_sampled_token_ith(test_ctx.ctx.get(), test_ctx.idx_for_seq(seq_id));
    printf("sampled token = %d, expected = %d\n", backend_token, bias_token);
    GGML_ASSERT(backend_token == bias_token);

    printf("backend logit bias sampling test PASSED\n");
}

static void accept_prompt(llama_sampler * smpl, const llama_vocab * vocab, const std::string & prompt) {
    const llama_token bos = llama_vocab_bos(vocab);
    if (bos != LLAMA_TOKEN_NULL) {
        llama_sampler_accept(smpl, bos);
    }

    std::vector<llama_token> tokens(64);
    int32_t n_tokens = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(),
            tokens.data(), (int32_t) tokens.size(), false, false);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(),
                tokens.data(), (int32_t) tokens.size(), false, false);
    }

    for (int32_t i = 0; i < n_tokens; ++i) {
        llama_sampler_accept(smpl, tokens[i]);
    }
}

static std::vector<float> decode_raw_logits(const test_params & params, const std::string & prompt) {
    const int seq_id = 0;
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(params.model.get()));
    std::vector<llama_sampler_seq_config> empty_configs;
    test_context ctx(params, empty_configs);

    GGML_ASSERT(ctx.decode({{ seq_id, prompt }}));

    float * logits = llama_get_logits_ith(ctx.ctx.get(), ctx.idx_for_seq(seq_id));
    GGML_ASSERT(logits != nullptr);
    return std::vector<float>(logits, logits + n_vocab);
}

static std::vector<llama_token_data> apply_cpu_sampler(
        const std::vector<float> & raw_logits,
        llama_sampler * sampler) {
    std::vector<llama_token_data> data;
    data.reserve(raw_logits.size());
    for (llama_token token = 0; token < (llama_token) raw_logits.size(); ++token) {
        data.push_back({ token, raw_logits[token], 0.0f });
    }

    llama_token_data_array cur_p = { data.data(), data.size(), -1, false };
    llama_sampler_apply(sampler, &cur_p);
    data.resize(cur_p.size);
    return data;
}

using sampler_setup_fn = std::function<void(llama_sampler *)>;
using sampler_init_fn = std::function<llama_sampler *()>;

enum class penalties_position {
    before_filter,
    after_filter,
};

static void add_filter_and_penalties(
        llama_sampler * chain,
        const sampler_init_fn & init_filter,
        int32_t n_vocab,
        int32_t penalty_last_n,
        float penalty_repeat,
        float penalty_freq,
        float penalty_present,
        penalties_position position) {
    const auto add_penalties = [&]() {
        llama_sampler_chain_add(chain, llama_sampler_init_penalties(
                    n_vocab, penalty_last_n, penalty_repeat, penalty_freq, penalty_present));
    };

    if (position == penalties_position::before_filter) {
        add_penalties();
        llama_sampler_chain_add(chain, init_filter());
    } else {
        llama_sampler_chain_add(chain, init_filter());
        add_penalties();
    }
}

static llama_sampler_ptr make_sampler_chain(
        const sampler_setup_fn & add_samplers,
        const sampler_setup_fn & accept_history) {
    llama_sampler_ptr chain(llama_sampler_chain_init(llama_sampler_chain_default_params()));
    add_samplers(chain.get());
    accept_history(chain.get());
    return chain;
}

struct backend_sampler_output {
    std::vector<float> logits;
    std::vector<llama_token> candidates;
};

static backend_sampler_output run_backend_sampler(
        const test_params & params,
        const std::string & prompt,
        llama_sampler * sampler) {
    const int seq_id = 0;
    std::vector<llama_sampler_seq_config> configs = {{ seq_id, sampler }};
    test_context ctx(params, configs);

    GGML_ASSERT(ctx.decode({{ seq_id, prompt }}));
    llama_synchronize(ctx.ctx.get());

    const int32_t idx = ctx.idx_for_seq(seq_id);
    const uint32_t n_logits = llama_get_sampled_logits_count_ith(ctx.ctx.get(), idx);
    const uint32_t n_candidates = llama_get_sampled_candidates_count_ith(ctx.ctx.get(), idx);
    float * logits = llama_get_sampled_logits_ith(ctx.ctx.get(), idx);
    llama_token * candidates = llama_get_sampled_candidates_ith(ctx.ctx.get(), idx);
    GGML_ASSERT(logits != nullptr);

    backend_sampler_output result;
    result.logits.assign(logits, logits + n_logits);
    result.candidates.resize(n_logits);

    if (n_candidates == 0) {
        for (uint32_t i = 0; i < n_logits; ++i) {
            result.candidates[i] = (llama_token) i;
        }
    } else {
        GGML_ASSERT(candidates != nullptr);
        GGML_ASSERT(n_candidates == n_logits);
        std::memcpy(result.candidates.data(), candidates, n_candidates * sizeof(llama_token));
    }

    return result;
}

struct sampler_comparison_output {
    std::vector<llama_token_data> expected;
    backend_sampler_output actual;
};

static sampler_comparison_output run_sampler_comparison(
        const test_params & params,
        const std::string & prompt,
        const std::vector<float> & raw_logits,
        const sampler_setup_fn & add_samplers,
        const sampler_setup_fn & accept_history) {
    llama_sampler_ptr cpu_chain = make_sampler_chain(add_samplers, accept_history);
    llama_sampler_ptr backend_chain = make_sampler_chain(add_samplers, accept_history);
    return {
        apply_cpu_sampler(raw_logits, cpu_chain.get()),
        run_backend_sampler(params, prompt, backend_chain.get()),
    };
}

static std::unordered_map<llama_token, float> map_logits(const std::vector<llama_token_data> & data) {
    std::unordered_map<llama_token, float> result;
    result.reserve(data.size());
    for (const auto & item : data) {
        result[item.id] = item.logit;
    }
    return result;
}

struct sampler_comparison_stats {
    int n_mismatch = 0;
    int n_masked = 0;
    float max_diff = 0.0f;
};

static sampler_comparison_stats compare_sampler_outputs(
        const char * name,
        const std::unordered_map<llama_token, float> & expected,
        const backend_sampler_output & actual,
        bool allow_extra_candidates = false) {
    GGML_ASSERT(actual.logits.size() == actual.candidates.size());

    sampler_comparison_stats result;
    std::unordered_set<llama_token> seen;
    seen.reserve(actual.candidates.size());

    for (size_t i = 0; i < actual.logits.size(); ++i) {
        const llama_token token = actual.candidates[i];
        const float logit = actual.logits[i];
        if (!seen.insert(token).second || std::isnan(logit)) {
            if (result.n_mismatch < 5) {
                printf("%s token %d has invalid backend output\n", name, token);
            }
            ++result.n_mismatch;
            continue;
        }

        const auto it = expected.find(token);
        if (it == expected.end()) {
            if (std::isinf(logit) && logit < 0.0f) {
                ++result.n_masked;
            } else if (!allow_extra_candidates) {
                if (result.n_mismatch < 5) {
                    printf("%s token %d was not masked\n", name, token);
                }
                ++result.n_mismatch;
            }
            continue;
        }

        const float diff = fabsf(it->second - logit);
        result.max_diff = std::max(result.max_diff, diff);
        if (!std::isfinite(logit) || diff > 1e-3f) {
            if (result.n_mismatch < 5) {
                printf("%s mismatch token %d: cpu=%.6f backend=%.6f diff=%.6f\n",
                        name, token, it->second, logit, diff);
            }
            ++result.n_mismatch;
        }
    }

    for (const auto & item : expected) {
        if (seen.find(item.first) == seen.end()) {
            if (result.n_mismatch < 5) {
                printf("%s missing backend token %d\n", name, item.first);
            }
            ++result.n_mismatch;
        }
    }

    printf("%s logits: max_diff=%.6f n_masked=%d n_mismatch=%d\n",
            name, result.max_diff, result.n_masked, result.n_mismatch);
    return result;
}

static float find_backend_logit(const backend_sampler_output & output, llama_token token) {
    for (size_t i = 0; i < output.candidates.size(); ++i) {
        if (output.candidates[i] == token) {
            return output.logits[i];
        }
    }
    GGML_ABORT("backend token not found");
}

static sampler_comparison_output run_penalties_comparison(
        const test_params & params,
        int32_t penalty_last_n,
        float penalty_repeat,
        float penalty_freq,
        float penalty_present,
        const std::string & prompt,
        const std::function<void(llama_sampler *)> & extra_accept = {}) {
    const auto * vocab = llama_model_get_vocab(params.model.get());
    const std::vector<float> raw_logits = decode_raw_logits(params, prompt);
    const auto add_samplers = [&](llama_sampler * chain) {
        llama_sampler_chain_add(chain, llama_sampler_init_penalties(
                    llama_vocab_n_tokens(vocab), penalty_last_n, penalty_repeat, penalty_freq, penalty_present));
    };
    const auto accept_history = [&](llama_sampler * chain) {
        accept_prompt(chain, vocab, prompt);
        if (extra_accept) {
            extra_accept(chain);
        }
    };

    return run_sampler_comparison(
            params, prompt, raw_logits, add_samplers, accept_history);
}

static void compare_penalties_logits(
        const test_params & params,
        int32_t penalty_last_n,
        float penalty_repeat,
        float penalty_freq,
        float penalty_present,
        const std::string & prompt,
        const std::function<void(llama_sampler *)> & extra_accept = {}) {
    const sampler_comparison_output output = run_penalties_comparison(
            params, penalty_last_n, penalty_repeat, penalty_freq, penalty_present, prompt, extra_accept);

    GGML_ASSERT(output.expected.size() == output.actual.logits.size());

    const sampler_comparison_stats stats = compare_sampler_outputs(
            "penalties", map_logits(output.expected), output.actual);
    GGML_ASSERT(stats.n_masked == 0);
    GGML_ASSERT(stats.n_mismatch == 0);
}

static void test_penalty_parameter_values(const test_params & params) {
    struct penalty_test_case {
        const char * name;
        float repeat;
        float frequency;
        float presence;
    };

    const penalty_test_case cases[] = {
        { "frequency -1",   1.0f, -1.0f,     0.0f },
        { "frequency 0",    1.0f,  0.0f,     0.0f },
        { "frequency 1",    1.0f,  1.0f,     0.0f },
        { "presence -1",    1.0f,  0.0f,    -1.0f },
        { "presence 0",     1.0f,  0.0f,     0.0f },
        { "presence 1",     1.0f,  0.0f,     1.0f },
        { "repeat 1",       1.0f,  0.0f,     0.0f },
    };

    int n_failed = 0;
    for (const auto & test : cases) {
        const sampler_comparison_output output = run_penalties_comparison(
                params, 64, test.repeat, test.frequency, test.presence, "Hello Hello world");
        GGML_ASSERT(output.expected.size() == output.actual.logits.size());
        const sampler_comparison_stats stats = compare_sampler_outputs(
                test.name, map_logits(output.expected), output.actual);
        n_failed += stats.n_mismatch != 0;
    }

    GGML_ASSERT(n_failed == 0);
}

static void compare_top_k_penalties_logits(
        const test_params & params,
        int32_t k,
        int32_t penalty_last_n,
        float penalty_repeat,
        float penalty_freq,
        float penalty_present,
        const std::string & prompt,
        penalties_position position) {
    const auto * vocab = llama_model_get_vocab(params.model.get());
    const std::vector<float> raw_logits = decode_raw_logits(params, prompt);
    const int n_vocab = (int) raw_logits.size();

    GGML_ASSERT(n_vocab > k);

    const sampler_init_fn init_top_k = [k]() {
        return llama_sampler_init_top_k(k);
    };
    llama_sampler_ptr top_k(init_top_k());
    const std::vector<llama_token_data> top_k_data = apply_cpu_sampler(raw_logits, top_k.get());
    GGML_ASSERT(top_k_data.size() == (size_t) k);
    const llama_token retained_history_token = top_k_data[0].id;

    llama_token excluded_history_token = LLAMA_TOKEN_NULL;
    for (llama_token token = 0; token < n_vocab; ++token) {
        const auto it = std::find_if(top_k_data.begin(), top_k_data.end(), [token](const llama_token_data & data) {
            return data.id == token;
        });
        if (it == top_k_data.end()) {
            excluded_history_token = token;
            break;
        }
    }
    GGML_ASSERT(excluded_history_token != LLAMA_TOKEN_NULL);

    const auto add_samplers = [&](llama_sampler * chain) {
        add_filter_and_penalties(chain, init_top_k, n_vocab,
                penalty_last_n, penalty_repeat, penalty_freq, penalty_present, position);
    };

    auto accept_history = [&](llama_sampler * smpl) {
        accept_prompt(smpl, vocab, prompt);
        llama_sampler_accept(smpl, excluded_history_token);
        llama_sampler_accept(smpl, excluded_history_token);
        llama_sampler_accept(smpl, retained_history_token);
        llama_sampler_accept(smpl, retained_history_token);
    };

    const sampler_comparison_output output = run_sampler_comparison(
            params, prompt, raw_logits, add_samplers, accept_history);

    GGML_ASSERT(output.expected.size() == (size_t) k);
    GGML_ASSERT(output.actual.logits.size() == (size_t) k);

    const std::unordered_map<llama_token, float> expected_logits = map_logits(output.expected);

    if (position == penalties_position::after_filter) {
        GGML_ASSERT(expected_logits.find(retained_history_token) != expected_logits.end());
        GGML_ASSERT(fabsf(expected_logits.at(retained_history_token) - raw_logits[retained_history_token]) > 1e-6f);
        GGML_ASSERT(expected_logits.find(excluded_history_token) == expected_logits.end());
        GGML_ASSERT(std::find(output.actual.candidates.begin(), output.actual.candidates.end(),
                    excluded_history_token) == output.actual.candidates.end());
    } else {
        const std::unordered_map<llama_token, float> unpenalized_logits = map_logits(top_k_data);
        bool changed = false;
        for (const auto & item : expected_logits) {
            const auto it = unpenalized_logits.find(item.first);
            if (it == unpenalized_logits.end() || fabsf(it->second - item.second) > 1e-6f) {
                changed = true;
                break;
            }
        }
        GGML_ASSERT(changed);
    }

    const char * name = position == penalties_position::before_filter
        ? "penalties top-k"
        : "top-k penalties";
    const sampler_comparison_stats stats = compare_sampler_outputs(
            name, expected_logits, output.actual);
    GGML_ASSERT(stats.n_masked == 0);
    GGML_ASSERT(stats.n_mismatch == 0);
}

static void compare_masking_penalties_logits(
        const test_params & params,
        const char * filter_name,
        const sampler_init_fn & init_filter,
        int32_t penalty_last_n,
        float penalty_repeat,
        float penalty_freq,
        float penalty_present,
        const std::string & prompt,
        penalties_position position,
        bool allow_extra_candidates,
        bool add_history = true) {
    const auto * vocab = llama_model_get_vocab(params.model.get());
    const std::vector<float> raw_logits = decode_raw_logits(params, prompt);
    const int n_vocab = (int) raw_logits.size();
    llama_sampler_ptr filter(init_filter());
    const std::vector<llama_token_data> filtered_data = apply_cpu_sampler(raw_logits, filter.get());
    GGML_ASSERT(!filtered_data.empty());
    GGML_ASSERT(filtered_data.size() < (size_t) n_vocab);

    const llama_token penalized_token = filtered_data[0].id;
    std::unordered_set<llama_token> retained_tokens;
    retained_tokens.reserve(filtered_data.size());
    for (const auto & data : filtered_data) {
        retained_tokens.insert(data.id);
    }

    llama_token masked_token = LLAMA_TOKEN_NULL;
    for (llama_token token = 0; token < n_vocab; ++token) {
        if (retained_tokens.find(token) == retained_tokens.end()) {
            masked_token = token;
            break;
        }
    }
    GGML_ASSERT(masked_token != LLAMA_TOKEN_NULL);

    const auto add_samplers = [&](llama_sampler * chain) {
        add_filter_and_penalties(chain, init_filter, n_vocab,
                penalty_last_n, penalty_repeat, penalty_freq, penalty_present, position);
    };
    auto accept_history = [&](llama_sampler * smpl) {
        if (!add_history) {
            return;
        }
        accept_prompt(smpl, vocab, prompt);
        llama_sampler_accept(smpl, penalized_token);
        llama_sampler_accept(smpl, penalized_token);
        llama_sampler_accept(smpl, masked_token);
        llama_sampler_accept(smpl, masked_token);
    };

    const sampler_comparison_output output = run_sampler_comparison(
            params, prompt, raw_logits, add_samplers, accept_history);

    GGML_ASSERT(output.actual.logits.size() == (size_t) n_vocab);

    const std::unordered_map<llama_token, float> expected_logits = map_logits(output.expected);

    GGML_ASSERT(expected_logits.find(masked_token) == expected_logits.end());
    if (add_history) {
        if (position == penalties_position::after_filter) {
            GGML_ASSERT(expected_logits.find(penalized_token) != expected_logits.end());
            GGML_ASSERT(fabsf(expected_logits.at(penalized_token) - raw_logits[penalized_token]) > 1e-6f);
        } else {
            llama_sampler_ptr penalties(llama_sampler_init_penalties(
                        n_vocab, penalty_last_n, penalty_repeat, penalty_freq, penalty_present));
            accept_history(penalties.get());
            const std::unordered_map<llama_token, float> penalized_logits =
                map_logits(apply_cpu_sampler(raw_logits, penalties.get()));
            GGML_ASSERT(fabsf(penalized_logits.at(penalized_token) - raw_logits[penalized_token]) > 1e-6f);
        }
    }

    const std::string name = position == penalties_position::before_filter
        ? "penalties " + std::string(filter_name)
        : std::string(filter_name) + " penalties";
    const sampler_comparison_stats stats = compare_sampler_outputs(
            name.c_str(), expected_logits, output.actual, allow_extra_candidates);
    const float masked_logit = find_backend_logit(output.actual, masked_token);
    GGML_ASSERT(stats.n_masked > 0);
    GGML_ASSERT(std::isinf(masked_logit) && masked_logit < 0.0f);
    GGML_ASSERT(stats.n_mismatch == 0);
}

static void test_backend_penalties_sampling(const test_params & params) {
    printf("Testing backend penalties (repeat + freq + presence)\n");
    compare_penalties_logits(params, 64, 1.1f, 0.5f, 0.25f, "Hello Hello world");

    printf("Testing backend penalties with penalty_last_n > 64\n");
    const auto * vocab = llama_model_get_vocab(params.model.get());
    std::vector<llama_token> tokens(8);
    int32_t n_tok = llama_tokenize(vocab, "a", 1, tokens.data(), (int32_t) tokens.size(), false, false);
    if (n_tok < 0) {
        tokens.resize(-n_tok);
        n_tok = llama_tokenize(vocab, "a", 1, tokens.data(), (int32_t) tokens.size(), false, false);
    }
    GGML_ASSERT(n_tok > 0);
    const llama_token tok = tokens[0];

    compare_penalties_logits(params, 80, 1.15f, 0.1f, 0.05f, "a", [tok](llama_sampler * smpl) {
        // accept_prompt already accepted BOS + one 'a'; fill the ring to n=80
        for (int i = 0; i < 78; ++i) {
            llama_sampler_accept(smpl, tok);
        }
    });

    printf("Testing backend penalties without filler entries\n");
    compare_penalties_logits(params, 64, 1.1f, 0.5f, 0.25f, "Hello", [](llama_sampler * smpl) {
        for (llama_token token = 0; token < 64; ++token) {
            llama_sampler_accept(smpl, token);
        }
    });

    printf("Testing backend top-k followed by penalties\n");
    compare_top_k_penalties_logits(params, 8, 64, 1.1f, 0.5f, 0.25f, "Hello",
            penalties_position::after_filter);

    printf("Testing backend penalties followed by top-k\n");
    compare_top_k_penalties_logits(params, 8, 64, 1.1f, 0.5f, 0.25f, "Hello",
            penalties_position::before_filter);

    printf("Testing backend top-p followed by penalties\n");
    compare_masking_penalties_logits(params, "top-p", []() {
        return llama_sampler_init_top_p(0.9f, 0);
    }, 64, 1.1f, 0.5f, 0.25f, "Hello", penalties_position::after_filter, true);

    printf("Testing backend top-p followed by penalties with a large history window\n");
    compare_masking_penalties_logits(params, "top-p large-window", []() {
        return llama_sampler_init_top_p(0.9f, 0);
    }, 4096, 1.1f, 0.5f, 0.25f, "Hello", penalties_position::after_filter, true);

    printf("Testing backend penalties followed by top-p\n");
    compare_masking_penalties_logits(params, "top-p", []() {
        return llama_sampler_init_top_p(0.9f, 0);
    }, 64, 1.1f, 0.5f, 0.25f, "Hello", penalties_position::before_filter, true);

    printf("Testing backend min-p followed by penalties\n");
    compare_masking_penalties_logits(params, "min-p", []() {
        return llama_sampler_init_min_p(0.1f, 0);
    }, 64, 1.1f, 0.5f, 0.25f, "Hello", penalties_position::after_filter, false);

    printf("Testing backend penalties followed by min-p\n");
    compare_masking_penalties_logits(params, "min-p", []() {
        return llama_sampler_init_min_p(0.1f, 0);
    }, 64, 1.1f, 0.5f, 0.25f, "Hello", penalties_position::before_filter, false);

    printf("Testing backend top-p followed by penalties with empty history\n");
    compare_masking_penalties_logits(params, "top-p empty", []() {
        return llama_sampler_init_top_p(0.9f, 0);
    }, 64, 1.1f, 0.5f, 0.25f, "Hello", penalties_position::after_filter, true, false);

    printf("Testing backend top-p followed by individual penalties\n");
    compare_masking_penalties_logits(params, "top-p repeat", []() {
        return llama_sampler_init_top_p(0.9f, 0);
    }, 64, 1.1f, 0.0f, 0.0f, "Hello", penalties_position::after_filter, true);
    compare_masking_penalties_logits(params, "top-p frequency", []() {
        return llama_sampler_init_top_p(0.9f, 0);
    }, 64, 1.0f, 0.5f, 0.0f, "Hello", penalties_position::after_filter, true);
    compare_masking_penalties_logits(params, "top-p presence", []() {
        return llama_sampler_init_top_p(0.9f, 0);
    }, 64, 1.0f, 0.0f, 0.25f, "Hello", penalties_position::after_filter, true);

    printf("Testing backend penalty parameter values\n");
    test_penalty_parameter_values(params);

    printf("backend penalties sampling test PASSED\n");
}

// This test verifies that it is possible to have two different backend samplers,
// one that uses the backend dist sampler, and another that uses CPU dist sampler.
static void test_backend_mixed_sampling(const test_params & params) {
    struct llama_sampler_chain_params chain_params_0 = llama_sampler_chain_default_params();
    llama_sampler_ptr sampler_chain_0(llama_sampler_chain_init(chain_params_0));
    llama_sampler_chain_add(sampler_chain_0.get(), llama_sampler_init_dist(88));

    int k = 40;
    struct llama_sampler_chain_params chain_params_1 = llama_sampler_chain_default_params();
    llama_sampler_ptr sampler_chain_1(llama_sampler_chain_init(chain_params_1));
    llama_sampler_chain_add(sampler_chain_1.get(), llama_sampler_init_top_k(k));

    std::vector<llama_sampler_seq_config> backend_sampler_configs = {
        { 0, sampler_chain_0.get() },
        { 1, sampler_chain_1.get() }
    };

    test_context test_ctx(params, backend_sampler_configs);

    std::map<llama_seq_id, std::string> prompts = {
        {0, "Hello"},
        {1, "Some"}
    };

    if (!test_ctx.decode(prompts)) {
        GGML_ASSERT(false && "Failed to decode token");
    }

    // Verify sequence 0 that used the dist backend sampler.
    {
        int32_t batch_idx = test_ctx.idx_for_seq(0);
        llama_token token = llama_get_sampled_token_ith(test_ctx.ctx.get(), batch_idx);
        const std::string token_str = test_ctx.token_to_piece(token, false);
        printf("sampled token id=%d, string='%s'\n", token, token_str.c_str());
        GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);
        //GGML_ASSERT(llama_get_sampled_logits_ith(test_ctx.ctx.get(), batch_idx) == nullptr);
        //GGML_ASSERT(llama_get_sampled_logits_count_ith(test_ctx.ctx.get(), batch_idx) == 0);
    }

    // Verify sequence 1 that used the top-k backend sampler.
    {
        int32_t batch_idx = test_ctx.idx_for_seq(1);
        float * logits = llama_get_sampled_logits_ith(test_ctx.ctx.get(), batch_idx);
        GGML_ASSERT(logits != nullptr);
        size_t n_logits = llama_get_sampled_logits_count_ith(test_ctx.ctx.get(), batch_idx);
        GGML_ASSERT(n_logits == (size_t) k);
        GGML_ASSERT(llama_get_sampled_token_ith(test_ctx.ctx.get(), batch_idx) == LLAMA_TOKEN_NULL);
    }

    printf("backend mixed sampling test PASSED\n");
}

static void test_backend_set_sampler(const test_params & params) {
    const int seq_id = 0;
    const int32_t seed = 88;

    struct llama_sampler_chain_params backend_chain_params = llama_sampler_chain_default_params();
    llama_sampler_ptr backend_sampler_chain(llama_sampler_chain_init(backend_chain_params));
    llama_sampler_chain_add(backend_sampler_chain.get(), llama_sampler_init_dist(seed));
    std::vector<llama_sampler_seq_config> backend_sampler_configs = {{ seq_id, backend_sampler_chain.get() }};

    test_context test_ctx(params, backend_sampler_configs);

    if (!test_ctx.decode({{seq_id, "Hello"}})) {
        GGML_ASSERT(false && "Failed to decode token");
    }

    int32_t batch_idx = test_ctx.idx_for_seq(seq_id);

    // Sample using backend sampler configured above
    llama_token backend_token = llama_get_sampled_token_ith(test_ctx.ctx.get(), batch_idx);
    const std::string backend_token_str = test_ctx.token_to_piece(backend_token, false);
    printf("dist sampled token = %d, string='%s'\n", backend_token, backend_token_str.c_str());

    // Now clear the backend sampler for this sequence.
    llama_set_sampler(test_ctx.ctx.get(), seq_id, nullptr);
    printf("Cleared backend sampler for seq_id %d\n", seq_id);

    // Sample using CPU sampler
    struct llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    llama_sampler_ptr chain(llama_sampler_chain_init(chain_params));
    llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(18));

    std::map<llama_seq_id, llama_token> tokens = { { seq_id, backend_token}, };
    if (!test_ctx.decode_tokens(tokens)) {
        GGML_ASSERT(false && "Failed to decode token");
    }

    // Should not have any sampled token or probs after clearing the backend sampler.
    const int32_t idx = test_ctx.idx_for_seq(seq_id);
    GGML_ASSERT(llama_get_sampled_token_ith(test_ctx.ctx.get(), idx) == LLAMA_TOKEN_NULL);
    GGML_ASSERT(llama_get_sampled_probs_ith(test_ctx.ctx.get(), idx) == nullptr);

    // Sample the token using the CPU sampler chain.
    llama_token token2 = llama_sampler_sample(chain.get(), test_ctx.ctx.get(), seq_id);
    const std::string token2_str = test_ctx.token_to_piece(token2, false);
    printf("CPU sampled token after clearing backend sampler: id=%d, string='%s'\n", token2, token2_str.c_str());
    std::map<llama_seq_id, llama_token> tokens2 = { { seq_id, token2}, };

    // Set a new backend sampler for the sequence.
    struct llama_sampler_chain_params new_backend_chain_params = llama_sampler_chain_default_params();
    llama_sampler_ptr new_backend_sampler_chain(llama_sampler_chain_init(new_backend_chain_params));
    llama_sampler_chain_add(new_backend_sampler_chain.get(), llama_sampler_init_top_k(20));
    llama_sampler_chain_add(new_backend_sampler_chain.get(), llama_sampler_init_dist(seed));
    llama_set_sampler(test_ctx.ctx.get(), seq_id, new_backend_sampler_chain.get());

    if (!test_ctx.decode_tokens(tokens2)) {
        GGML_ASSERT(false && "Failed to decode token");
    }

    llama_token new_backend_token = llama_get_sampled_token_ith(test_ctx.ctx.get(), test_ctx.idx_for_seq(seq_id));
    const std::string new_backend_token_str = test_ctx.token_to_piece(new_backend_token, false);
    printf("dist sampled token = %d, string='%s'\n", new_backend_token, new_backend_token_str.c_str());

    printf("backend set sampler test PASSED\n");
}

static void test_backend_cpu_mixed_batch(const test_params & params) {
    // Sequence 0 uses backend sampling
    struct llama_sampler_chain_params chain_params_0 = llama_sampler_chain_default_params();
    llama_sampler_ptr sampler_chain_0(llama_sampler_chain_init(chain_params_0));
    llama_sampler_chain_add(sampler_chain_0.get(), llama_sampler_init_dist(88));

    std::vector<llama_sampler_seq_config> backend_sampler_configs = {
        { 0, sampler_chain_0.get() },
    };

    // We need 2 sequences: seq 0 with backend sampling, seq 1 with CPU sampling
    test_context test_ctx(params, backend_sampler_configs, 2);

    std::map<llama_seq_id, std::string> prompts = {
        {0, "Hello"}, // Will use backend sampling
        {1, "Some"}   // Will use CPU sampling
    };

    if (!test_ctx.decode(prompts)) {
        GGML_ASSERT(false && "Failed to decode token");
    }

    // Verify sequence 0 (backend sampled)
    {
        int32_t batch_idx = test_ctx.idx_for_seq(0);
        llama_token token = llama_get_sampled_token_ith(test_ctx.ctx.get(), batch_idx);
        const std::string token_str = test_ctx.token_to_piece(token, false);
        printf("Seq 0 (backend) sampled token id=%d, string='%s'\n", token, token_str.c_str());
        GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);
    }

    // Verify sequence 1 (CPU sampled)
    {
        int32_t batch_idx = test_ctx.idx_for_seq(1);

        llama_token backend_token = llama_get_sampled_token_ith(test_ctx.ctx.get(), batch_idx);
        GGML_ASSERT(backend_token == LLAMA_TOKEN_NULL);

        struct llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
        llama_sampler_ptr chain(llama_sampler_chain_init(chain_params));
        llama_sampler_chain_add(chain.get(), llama_sampler_init_greedy());

        llama_token token = llama_sampler_sample(chain.get(), test_ctx.ctx.get(), batch_idx);
        const std::string token_str = test_ctx.token_to_piece(token, false);
        printf("Seq 1 (CPU) sampled token id=%d, string='%s'\n", token, token_str.c_str());
        GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);
    }

    // Clear/remove the backend sampler, and sample again
    {
        // clear the backend sampler for seq 0 so that there are no backend
        // samplers.
        llama_set_sampler(test_ctx.ctx.get(), 0, nullptr);

        // Create a CPU sampler and verify we can sample from it.
        struct llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
        llama_sampler_ptr chain(llama_sampler_chain_init(chain_params));
        llama_sampler_chain_add(chain.get(), llama_sampler_init_greedy());

        int32_t batch_idx = test_ctx.idx_for_seq(1);
        llama_token token = llama_sampler_sample(chain.get(), test_ctx.ctx.get(), batch_idx);
        if (!test_ctx.decode_token(token, 1)) {
            GGML_ASSERT(false && "Failed to decode token");
        }
    }

    // Set a backend sampler so that we can verify that it can be reset
    {
        struct llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
        llama_sampler_ptr sampler_chain(llama_sampler_chain_init(chain_params));
        llama_sampler_chain_add(sampler_chain.get(), llama_sampler_init_dist(88));

        llama_set_sampler(test_ctx.ctx.get(), 0, sampler_chain.get());

        if (!test_ctx.decode_token(3834, 0)) {
            GGML_ASSERT(false && "Failed to decode token");
        }

        int32_t batch_idx = test_ctx.idx_for_seq(0);
        llama_token token = llama_get_sampled_token_ith(test_ctx.ctx.get(), batch_idx);
        const std::string token_str = test_ctx.token_to_piece(token, false);
        printf("re-added backend sampled token id=%d, string='%s'\n", token, token_str.c_str());
        GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);
    }

    printf("backend-cpu mixed batch test PASSED\n");
}

static void test_backend_multi_output_limit(const test_params & params) {
    const llama_seq_id seq_id = 0;

    llama_sampler_ptr chain(llama_sampler_chain_init(llama_sampler_chain_default_params()));
    llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(88));
    std::vector<llama_sampler_seq_config> configs = {{ seq_id, chain.get() }};
    test_context test_ctx(params, configs, 1, 3, 0, 2);

    llama_batch batch = llama_batch_init(3, 0, 1);
    for (int i = 0; i < 3; ++i) {
        common_batch_add(batch, llama_vocab_bos(test_ctx.vocab), i, { seq_id }, true);
    }

    printf(">>> test_backend_multi_output_limit expected error start:\n");
    const int ret = llama_decode(test_ctx.ctx.get(), batch);
    GGML_ASSERT(ret != 0 && "llama_decode should reject outputs above the per-sequence limit");
    printf("<<< test_backend_multi_output_limit expected error end.\n");

    llama_batch_free(batch);

    printf("backend multi-output limit test PASSED\n");
}

static void test_backend_multi_sequence_multi_output_dist(const test_params & params) {
    const llama_vocab * vocab = llama_model_get_vocab(params.model.get());
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const uint32_t seeds[] = { 88, 1337 };
    // reduce the chance that swapped random inputs select the same token
    const float temp = 10.0f;

    llama_sampler_ptr chain_0(llama_sampler_chain_init(llama_sampler_chain_default_params()));
    llama_sampler_ptr chain_1(llama_sampler_chain_init(llama_sampler_chain_default_params()));
    llama_sampler_chain_add(chain_0.get(), llama_sampler_init_temp(temp));
    llama_sampler_chain_add(chain_0.get(), llama_sampler_init_dist(seeds[0]));
    llama_sampler_chain_add(chain_1.get(), llama_sampler_init_temp(temp));
    llama_sampler_chain_add(chain_1.get(), llama_sampler_init_dist(seeds[1]));
    std::vector<llama_sampler_seq_config> configs = {
        { 0, chain_0.get() },
        { 1, chain_1.get() },
    };
    test_context test_ctx(params, configs, 2, 4, 0, 2);

    std::vector<llama_sampler_seq_config> reference_configs;
    test_context reference_ctx(params, reference_configs, 2, 4);

    const llama_token seq_tokens[2][2] = {
        { llama_vocab_bos(vocab), llama_vocab_eos(vocab) },
        { llama_vocab_eos(vocab), llama_vocab_bos(vocab) },
    };

    llama_batch batch = llama_batch_init(4, 0, 1);
    for (int pos = 0; pos < 2; ++pos) {
        common_batch_add(batch, seq_tokens[0][pos], pos, { 0 }, true);
        common_batch_add(batch, seq_tokens[1][pos], pos, { 1 }, true);
    }

    GGML_ASSERT(llama_decode(test_ctx.ctx.get(), batch) == 0);
    GGML_ASSERT(llama_decode(reference_ctx.ctx.get(), batch) == 0);

    std::mt19937 reference_rngs[] = {
        std::mt19937(seeds[0]),
        std::mt19937(seeds[1]),
    };
    std::uniform_real_distribution<double> reference_dist(0.0, 1.0);

    for (int i = 0; i < batch.n_tokens; ++i) {
        const llama_seq_id seq_id = batch.seq_id[i][0];
        GGML_ASSERT(seq_id == 0 || seq_id == 1);

        llama_sampler * chain = seq_id == 0 ? chain_0.get() : chain_1.get();
        const llama_token backend_token = llama_sampler_sample(chain, test_ctx.ctx.get(), i);
        const float * sampled_logits = llama_get_sampled_logits_ith(test_ctx.ctx.get(), i);
        const float * sampled_probs = llama_get_sampled_probs_ith(test_ctx.ctx.get(), i);
        const uint32_t n_logits = llama_get_sampled_logits_count_ith(test_ctx.ctx.get(), i);
        const uint32_t n_probs = llama_get_sampled_probs_count_ith(test_ctx.ctx.get(), i);
        const float * reference_logits = llama_get_logits_ith(reference_ctx.ctx.get(), i);

        GGML_ASSERT(backend_token >= 0 && backend_token < n_vocab);
        GGML_ASSERT(sampled_logits != nullptr);
        GGML_ASSERT(sampled_probs != nullptr);
        GGML_ASSERT(reference_logits != nullptr);
        GGML_ASSERT(n_logits == (uint32_t) n_vocab);
        GGML_ASSERT(n_probs == (uint32_t) n_vocab);

        float prob_sum = 0.0f;
        float cumsum_before = 0.0f;
        for (llama_token token = 0; token < n_vocab; ++token) {
            const float expected_logit = reference_logits[token] / temp;
            const float tolerance = 1e-4f * std::max(1.0f, std::fabs(expected_logit));
            GGML_ASSERT(std::fabs(sampled_logits[token] - expected_logit) <= tolerance);
            GGML_ASSERT(std::isfinite(sampled_probs[token]));
            GGML_ASSERT(sampled_probs[token] >= 0.0f);

            prob_sum += sampled_probs[token];
            if (token < backend_token) {
                cumsum_before += sampled_probs[token];
            }
        }

        GGML_ASSERT(std::fabs(prob_sum - 1.0f) <= 1e-3f);

        const float rnd = reference_dist(reference_rngs[seq_id]);
        const float cumsum_sampled = cumsum_before + sampled_probs[backend_token];
        GGML_ASSERT(rnd >= cumsum_before - 1e-4f);
        GGML_ASSERT(rnd <= cumsum_sampled + 1e-4f);
    }

    llama_batch_free(batch);

    printf("backend multi-sequence multi-output dist test PASSED\n");
}

static void test_backend_multi_output_dist_transaction(const test_params & params) {
    const llama_seq_id seq_id = 0;
    const uint32_t seed = 95;
    const llama_vocab * vocab = llama_model_get_vocab(params.model.get());

    llama_sampler_ptr chain(llama_sampler_chain_init(llama_sampler_chain_default_params()));
    llama_sampler_chain_add(chain.get(), llama_sampler_init_temp(10.0f));
    llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(seed));
    std::vector<llama_sampler_seq_config> configs = {{ seq_id, chain.get() }};
    test_context test_ctx(params, configs, 1, 3, 2, 3);

    auto verify_random = [&](int32_t row, float rnd, bool accept = true) {
        const llama_token token = accept ?
            llama_sampler_sample(chain.get(), test_ctx.ctx.get(), row) :
            llama_get_sampled_token_ith(test_ctx.ctx.get(), row);
        const float * probs = llama_get_sampled_probs_ith(test_ctx.ctx.get(), row);

        GGML_ASSERT(token >= 0 && token < llama_vocab_n_tokens(vocab));
        GGML_ASSERT(probs != nullptr);

        float cumsum_before = 0.0f;
        for (llama_token i = 0; i < token; ++i) {
            cumsum_before += probs[i];
        }

        const float cumsum_sampled = cumsum_before + probs[token];
        GGML_ASSERT(rnd >= cumsum_before - 1e-4f);
        GGML_ASSERT(rnd <= cumsum_sampled + 1e-4f);
    };

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    float randoms[3];
    for (float & rnd : randoms) {
        rnd = dist(rng);
    }

    int32_t pos = 0;
    auto decode = [&]() {
        llama_batch batch = llama_batch_init(3, 0, 1);
        for (int32_t i = 0; i < 3; ++i) {
            common_batch_add(batch, llama_vocab_bos(vocab), pos++, { seq_id }, true);
        }
        GGML_ASSERT(llama_decode(test_ctx.ctx.get(), batch) == 0);
        return batch;
    };

    llama_batch batch = decode();
    verify_random(0, randoms[0], false);
    llama_batch_free(batch);

    batch = decode();
    verify_random(0, randoms[0]);
    verify_random(1, randoms[1]);
    llama_batch_free(batch);

    batch = decode();
    llama_sampler_ptr saved(llama_sampler_clone(chain.get()));
    verify_random(0, randoms[2]);
    llama_batch_free(batch);

    llama_sampler_copy(saved.get(), chain.get());

    batch = decode();
    verify_random(0, randoms[2]);
    llama_batch_free(batch);

    printf("backend multi-output dist transaction test PASSED\n");
}

static void test_backend_multi_output_sampling_chain(const test_params & params) {
    const llama_seq_id seq_id = 0;
    const uint32_t seed = 88;
    const float p = 0.9f;
    const float temp = 0.8f;
    const float cdf_epsilon = 1e-4f;
    const llama_vocab * vocab = llama_model_get_vocab(params.model.get());
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const uint32_t k = std::min<uint32_t>(512, n_vocab);
    const llama_logit_bias bias = { llama_vocab_bos(vocab), -0.1f };

    auto make_filter_chain = [&]() {
        llama_sampler_ptr result(llama_sampler_chain_init(llama_sampler_chain_default_params()));
        llama_sampler_chain_add(result.get(), llama_sampler_init_logit_bias(n_vocab, 1, &bias));
        llama_sampler_chain_add(result.get(), llama_sampler_init_top_k(k));
        llama_sampler_chain_add(result.get(), llama_sampler_init_top_p(p, 1));
        llama_sampler_chain_add(result.get(), llama_sampler_init_min_p(0.01f, 1));
        llama_sampler_chain_add(result.get(), llama_sampler_init_temp(temp));
        return result;
    };

    llama_sampler_ptr chain = make_filter_chain();
    llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(seed));
    std::vector<llama_sampler_seq_config> configs = {{ seq_id, chain.get() }};
    test_context test_ctx(params, configs, 1, 2, 2, 2);

    std::vector<llama_sampler_seq_config> reference_configs;
    test_context reference_ctx(params, reference_configs, 1, 2, 2);

    llama_sampler_ptr reference_bias(llama_sampler_init_logit_bias(n_vocab, 1, &bias));
    llama_sampler_ptr reference_top_k(llama_sampler_init_top_k(k));
    llama_sampler_ptr reference_top_p(llama_sampler_init_top_p(p, 1));
    llama_sampler_ptr reference_min_p(llama_sampler_init_min_p(0.01f, 1));
    llama_sampler_ptr reference_temp(llama_sampler_init_temp(temp));
    std::vector<llama_token_data> reference_data(n_vocab);

    auto make_batch = [&](int32_t pos) {
        llama_batch batch = llama_batch_init(2, 0, 1);
        for (int i = 0; i < 2; ++i) {
            common_batch_add(batch, llama_vocab_bos(vocab), pos + i, { seq_id }, true);
        }
        return batch;
    };

    llama_batch batch = make_batch(0);
    GGML_ASSERT(llama_decode(test_ctx.ctx.get(), batch) == 0);
    GGML_ASSERT(llama_decode(reference_ctx.ctx.get(), batch) == 0);

    for (int i = 0; i < batch.n_tokens; ++i) {
        const llama_token backend_token = llama_sampler_sample(chain.get(), test_ctx.ctx.get(), i);
        const float * sampled_logits = llama_get_sampled_logits_ith(test_ctx.ctx.get(), i);
        const float * sampled_probs = llama_get_sampled_probs_ith(test_ctx.ctx.get(), i);
        const llama_token * sampled_candidates = llama_get_sampled_candidates_ith(test_ctx.ctx.get(), i);
        const uint32_t n_logits = llama_get_sampled_logits_count_ith(test_ctx.ctx.get(), i);
        const uint32_t n_probs = llama_get_sampled_probs_count_ith(test_ctx.ctx.get(), i);
        const uint32_t n_candidates = llama_get_sampled_candidates_count_ith(test_ctx.ctx.get(), i);
        const float * reference_logits = llama_get_logits_ith(reference_ctx.ctx.get(), i);

        GGML_ASSERT(backend_token >= 0 && backend_token < n_vocab);
        GGML_ASSERT(sampled_logits != nullptr);
        GGML_ASSERT(sampled_probs != nullptr);
        GGML_ASSERT(sampled_candidates != nullptr);
        GGML_ASSERT(reference_logits != nullptr);
        GGML_ASSERT(n_logits == k);
        GGML_ASSERT(n_probs == n_logits);
        GGML_ASSERT(n_candidates == n_logits);

        for (llama_token token = 0; token < n_vocab; ++token) {
            reference_data[token] = { token, reference_logits[token], 0.0f };
        }

        llama_token_data_array reference = {
            /* .data     = */ reference_data.data(),
            /* .size     = */ reference_data.size(),
            /* .selected = */ LLAMA_TOKEN_NULL,
            /* .sorted   = */ false,
        };

        llama_sampler_apply(reference_bias.get(), &reference);
        llama_sampler_apply(reference_top_k.get(), &reference);
        llama_sampler_apply(reference_top_p.get(), &reference);
        GGML_ASSERT(reference.size > 0);

        float cdf = 0.0f;
        for (size_t j = 0; j < reference.size; ++j) {
            cdf += reference.data[j].p;
        }
        const float cdf_before = cdf - reference.data[reference.size - 1].p;
        const float boundary_distance = std::min(std::fabs(cdf_before - p), std::fabs(cdf - p));

        llama_sampler_apply(reference_min_p.get(), &reference);
        llama_sampler_apply(reference_temp.get(), &reference);

        std::unordered_map<llama_token, float> reference_by_id;
        for (size_t j = 0; j < reference.size; ++j) {
            reference_by_id.emplace(reference.data[j].id, reference.data[j].logit);
        }
        size_t n_backend_only = 0;
        int32_t sampled_index = -1;
        float prob_sum = 0.0f;

        for (uint32_t j = 0; j < n_logits; ++j) {
            GGML_ASSERT(sampled_candidates[j] >= 0 && sampled_candidates[j] < n_vocab);
            GGML_ASSERT(std::isfinite(sampled_probs[j]));
            GGML_ASSERT(sampled_probs[j] >= 0.0f);
            prob_sum += sampled_probs[j];

            if (sampled_candidates[j] == backend_token) {
                sampled_index = j;
            }
            if (!std::isfinite(sampled_logits[j])) {
                GGML_ASSERT(std::isinf(sampled_logits[j]) && sampled_logits[j] < 0.0f);
                GGML_ASSERT(sampled_probs[j] == 0.0f);
                continue;
            }

            const auto match = reference_by_id.find(sampled_candidates[j]);
            if (match == reference_by_id.end()) {
                ++n_backend_only;
                continue;
            }

            const float tolerance = 1e-4f * std::max(1.0f, std::fabs(match->second));
            GGML_ASSERT(std::fabs(sampled_logits[j] - match->second) <= tolerance);
            reference_by_id.erase(match);
        }

        const size_t n_reference_only = reference_by_id.size();

        if (n_backend_only != 0 || n_reference_only != 0) {
            GGML_ASSERT(n_backend_only <= 1);
            GGML_ASSERT(n_reference_only <= 1);
            GGML_ASSERT(boundary_distance <= cdf_epsilon);
        }

        GGML_ASSERT(sampled_index >= 0);
        GGML_ASSERT(std::isfinite(sampled_logits[sampled_index]));
        GGML_ASSERT(sampled_probs[sampled_index] > 0.0f);
        GGML_ASSERT(std::fabs(prob_sum - 1.0f) <= 1e-3f);
    }

    llama_batch_free(batch);

    batch = make_batch(2);
    GGML_ASSERT(llama_decode(test_ctx.ctx.get(), batch) == 0);
    llama_batch_free(batch);

    printf("backend multi-output sampling chain test PASSED\n");
}

static void test_backend_multi_output_cpu_suffix(const test_params & params) {
    const llama_seq_id seq_id = 0;
    const int32_t k = 8;
    const llama_vocab * vocab = llama_model_get_vocab(params.model.get());

    auto make_chain = [&](test_single_output_backend_sampler ** sampler_ctx) {
        llama_sampler_ptr result(llama_sampler_chain_init(llama_sampler_chain_default_params()));
        llama_sampler_chain_add(result.get(), llama_sampler_init_top_k(k));
        llama_sampler_chain_add(result.get(), test_single_output_backend_sampler_init(sampler_ctx));
        llama_sampler_chain_add(result.get(), llama_sampler_init_dist(88));
        return result;
    };

    {
        test_single_output_backend_sampler * sampler_ctx = nullptr;
        llama_sampler_ptr chain = make_chain(&sampler_ctx);
        std::vector<llama_sampler_seq_config> configs = {{ seq_id, chain.get() }};
        test_context test_ctx(params, configs, 1, 1, 0, 4);

        llama_batch batch = llama_batch_init(1, 0, 1);
        common_batch_add(batch, llama_vocab_bos(vocab), 0, { seq_id }, true);
        GGML_ASSERT(llama_decode(test_ctx.ctx.get(), batch) == 0);

        GGML_ASSERT(sampler_ctx->backend_initialized);
        GGML_ASSERT(sampler_ctx->backend_outputs_max_per_seq == 1);
        GGML_ASSERT(sampler_ctx->backend_apply_count > 0);
        GGML_ASSERT(sampler_ctx->apply_count == 0);
        GGML_ASSERT(llama_get_sampled_token_ith(test_ctx.ctx.get(), 0) != LLAMA_TOKEN_NULL);

        llama_batch_free(batch);
    }

    {
        test_single_output_backend_sampler * sampler_ctx = nullptr;
        llama_sampler_ptr chain = make_chain(&sampler_ctx);
        std::vector<llama_sampler_seq_config> configs = {{ seq_id, chain.get() }};
        test_context test_ctx(params, configs, 1, 2, 0, 0);

        llama_batch batch = llama_batch_init(2, 0, 1);
        for (int i = 0; i < 2; ++i) {
            common_batch_add(batch, llama_vocab_bos(vocab), i, { seq_id }, true);
        }
        GGML_ASSERT(llama_decode(test_ctx.ctx.get(), batch) == 0);

        GGML_ASSERT(!sampler_ctx->backend_initialized);
        GGML_ASSERT(sampler_ctx->backend_outputs_max_per_seq == 2);
        GGML_ASSERT(sampler_ctx->backend_apply_count == 0);
        for (int i = 0; i < batch.n_tokens; ++i) {
            GGML_ASSERT(llama_get_sampled_token_ith(test_ctx.ctx.get(), i) == LLAMA_TOKEN_NULL);
            GGML_ASSERT(llama_get_sampled_logits_count_ith(test_ctx.ctx.get(), i) == (uint32_t) k);
            GGML_ASSERT(llama_get_sampled_candidates_count_ith(test_ctx.ctx.get(), i) == (uint32_t) k);
            const llama_token token = llama_sampler_sample(chain.get(), test_ctx.ctx.get(), i);
            GGML_ASSERT(token >= 0 && token < llama_vocab_n_tokens(vocab));
        }
        GGML_ASSERT(sampler_ctx->apply_count == batch.n_tokens);

        llama_batch_free(batch);
    }

    printf("backend multi-output CPU suffix test PASSED\n");
}

static llama_sampler_ptr sampled_decode_sampler(const test_params & params, uint32_t seed = 12345) {
    const auto * model = params.model.get();
    const auto * vocab = llama_model_get_vocab(model);
    int32_t n_suppress = 0;
    const llama_token * suppress = llama_vocab_get_suppress_tokens(vocab, &n_suppress);
    std::vector<llama_logit_bias> biases;
    for (int32_t i = 0; i < n_suppress; ++i) {
        biases.push_back({suppress[i], -INFINITY});
    }
    llama_sampler_ptr result(llama_sampler_chain_init(llama_sampler_chain_default_params()));
    if (!biases.empty()) {
        llama_sampler_chain_add(result.get(), llama_sampler_init_logit_bias(llama_vocab_n_tokens(vocab), biases.size(), biases.data()));
    }
    if (params.sampled_stochastic) {
        llama_sampler_chain_add(result.get(), llama_sampler_init_top_k(40));
        llama_sampler_chain_add(result.get(), llama_sampler_init_top_p(0.9f, 1));
        llama_sampler_chain_add(result.get(), llama_sampler_init_min_p(0.05f, 1));
        llama_sampler_chain_add(result.get(), llama_sampler_init_temp(0.8f));
        llama_sampler_chain_add(result.get(), llama_sampler_init_dist(seed));
    } else {
        llama_sampler_chain_add(result.get(), llama_sampler_init_greedy());
    }
    return result;
}

static void test_backend_decode_sampled(const test_params & params) {
    const auto generate = [&](int device_input) {
        auto sampler = sampled_decode_sampler(params);
        llama_sampler_ptr checkpoint(llama_sampler_clone(sampler.get()));
        std::vector<llama_sampler_seq_config> configs = {{0, sampler.get()}};
        test_context tc(params, configs, 1);
        llama_token previous = LLAMA_TOKEN_NULL;
        GGML_ASSERT(llama_decode_sampled(tc.ctx.get(), 0, 0) == 1);
        GGML_ASSERT(llama_decode_sampled_async(tc.ctx.get(), 0, 0, &previous) == 1);
        GGML_ASSERT(tc.decode({{0, "Hello"}}));
        if (params.sampled_host_inputs) {
            GGML_ASSERT(tc.decode_token(llama_vocab_eos(tc.vocab)));
        }
        llama_synchronize(tc.ctx.get());

        auto * memory = llama_get_memory(tc.ctx.get());
        llama_token token = llama_get_sampled_token_ith(tc.ctx.get(), tc.idx_for_seq(0));
        GGML_ASSERT(token != LLAMA_TOKEN_NULL);
        const auto first_pos = llama_memory_seq_pos_max(memory, 0);
        GGML_ASSERT(llama_decode_sampled(tc.ctx.get(), 0, first_pos) == 1);
        GGML_ASSERT(llama_decode_sampled(tc.ctx.get(), 1, first_pos + 1) == 1);
        GGML_ASSERT(llama_decode_sampled_async(tc.ctx.get(), 0, first_pos, &previous) == 1);
        GGML_ASSERT(llama_decode_sampled_async(tc.ctx.get(), 1, first_pos + 1, &previous) == 1);
        GGML_ASSERT(previous == LLAMA_TOKEN_NULL);
        GGML_ASSERT(llama_memory_seq_pos_max(memory, 0) == first_pos);

        if (device_input == 2 && params.sampled_decode_gpu && llama_recurrent_set_sparse_snapshot_mode(tc.ctx.get(), true, 0)) {
            GGML_ASSERT(llama_decode_sampled_async(tc.ctx.get(), 0, first_pos + 1, &previous) == 1);
            GGML_ASSERT(previous == LLAMA_TOKEN_NULL);
            GGML_ASSERT(llama_memory_seq_pos_max(memory, 0) == first_pos);
            GGML_ASSERT(llama_get_sampled_token_ith(tc.ctx.get(), tc.idx_for_seq(0)) == token);
            GGML_ASSERT(llama_recurrent_set_sparse_snapshot_mode(tc.ctx.get(), false, -1));
        }

        if (!params.sampled_decode_gpu) {
            GGML_ASSERT(llama_decode_sampled(tc.ctx.get(), 0, first_pos + 1) == 1);
            GGML_ASSERT(llama_decode_sampled_async(tc.ctx.get(), 0, first_pos + 1, &previous) == 1);
            GGML_ASSERT(llama_memory_seq_pos_max(memory, 0) == first_pos);
            return std::vector<llama_token> {token};
        }

        std::vector<llama_token> result;
        if (device_input != 2) {
            result.push_back(token);
        }
        llama_token input = token;
        for (int i = 0; i < 288; ++i) {
            llama_sampler_copy(sampler.get(), checkpoint.get());
            input = token;
            if (device_input) {
                const llama_pos pos = llama_memory_seq_pos_max(memory, 0) + 1;
                if (device_input == 2) {
                    GGML_ASSERT(llama_decode_sampled_async(tc.ctx.get(), 0, pos, &input) == 0);
                    result.push_back(input);
                } else {
                    GGML_ASSERT(llama_decode_sampled(tc.ctx.get(), 0, pos) == 0);
                }
                tc.seq_positions[0] = pos + 1;
            } else {
                GGML_ASSERT(tc.decode_token(input));
            }
            if (device_input != 2) {
                token = llama_get_sampled_token_ith(tc.ctx.get(), 0);
                GGML_ASSERT(token != LLAMA_TOKEN_NULL);
                result.push_back(token);
            }
        }
        if (device_input == 2) {
            token = llama_get_sampled_token_ith(tc.ctx.get(), 0);
            GGML_ASSERT(token != LLAMA_TOKEN_NULL);
            result.push_back(token);
        }

        const llama_pos pos = llama_memory_seq_pos_max(memory, 0);
        GGML_ASSERT(llama_memory_seq_rm(memory, 0, pos, -1));
        GGML_ASSERT(llama_memory_seq_pos_max(memory, 0) == pos - 1);
        llama_sampler_copy(checkpoint.get(), sampler.get());
        tc.seq_positions[0] = pos;
        GGML_ASSERT(tc.decode_token(input));
        llama_synchronize(tc.ctx.get());
        GGML_ASSERT(llama_get_sampled_token_ith(tc.ctx.get(), 0) == token);
        if (device_input == 2) {
            llama_sampler_copy(sampler.get(), checkpoint.get());
            GGML_ASSERT(llama_decode_sampled_async(tc.ctx.get(), 0, pos + 1, &previous) == 0);
            GGML_ASSERT(previous == token);
            llama_synchronize(tc.ctx.get());
            GGML_ASSERT(llama_memory_seq_rm(memory, 0, pos + 1, -1));
            llama_sampler_copy(checkpoint.get(), sampler.get());
        }
        return result;
    };

    const auto expected = generate(0);
    if (!params.sampled_host_inputs) {
        GGML_ASSERT(expected == generate(1));
    }
    GGML_ASSERT(expected == generate(2));
    printf("sampled decode parity, admission, and one-token rollback PASSED (%s)\n", params.sampled_decode_gpu ? "CUDA" : "fallback");
}

static void test_backend_decode_sampled_batch(const test_params & params) {
    const auto generate = [&](bool overlap, bool unified) {
        std::vector<llama_sampler_ptr> samplers;
        llama_sampler_ptr checkpoint;
        std::vector<llama_sampler_seq_config> configs;
        for (llama_seq_id seq = 0; seq < 4; ++seq) {
            samplers.push_back(sampled_decode_sampler(params, 12345 + seq));
            configs.push_back({seq, samplers.back().get()});
        }
        test_context tc(params, configs, 4, 0, 0, 1, unified);
        GGML_ASSERT(tc.decode({{0, "Hello"}, {1, "Count to ten"}, {2, "The sky is"}, {3, "A small cat"}}));
        auto * memory = llama_get_memory(tc.ctx.get());
        std::map<llama_seq_id, llama_token> tokens;
        for (llama_seq_id seq = 0; seq < 4; ++seq) {
            tokens[seq] = llama_get_sampled_token_ith(tc.ctx.get(), tc.idx_for_seq(seq));
        }
        // Unequal prompts can leave their final outputs in different recurrent microbatches.
        GGML_ASSERT(tc.decode_tokens(tokens));
        for (llama_seq_id seq = 0; seq < 4; ++seq) {
            tokens[seq] = llama_get_sampled_token_ith(tc.ctx.get(), tc.idx_for_seq(seq));
        }
        llama_sampled_decode_item duplicate[] = {{0, tc.seq_positions[0]}, {0, tc.seq_positions[0]}};
        llama_token previous[] = {LLAMA_TOKEN_NULL, LLAMA_TOKEN_NULL};
        GGML_ASSERT(llama_decode_sampled_batch_async(tc.ctx.get(), duplicate, 2, previous) == 1);
        GGML_ASSERT(previous[0] == LLAMA_TOKEN_NULL && previous[1] == LLAMA_TOKEN_NULL);
        GGML_ASSERT(llama_memory_seq_pos_max(memory, 0) + 1 == tc.seq_positions[0]);

        std::map<llama_seq_id, std::vector<llama_token>> result;
        std::map<llama_seq_id, llama_token> last_inputs;
        for (int step = 0; step < 64; ++step) {
            if (!checkpoint) {
                checkpoint.reset(llama_sampler_clone(samplers[0].get()));
            } else {
                llama_sampler_copy(samplers[0].get(), checkpoint.get());
            }
            std::vector<llama_seq_id> active = {3, 1, 2, 0};
            if (unified && step >= 24) {
                active.erase(std::remove(active.begin(), active.end(), 1), active.end());
            }
            if (unified && step >= 40) {
                active.erase(std::remove(active.begin(), active.end(), 3), active.end());
            }
            if (step % 2) {
                std::reverse(active.begin(), active.end());
            }
            if (!unified) {
                std::sort(active.begin(), active.end());
            }
            std::vector<llama_sampled_decode_item> items;
            for (auto seq : active) {
                items.push_back({seq, tc.seq_positions[seq]});
            }
            if (overlap) {
                std::vector<llama_token> consumed(items.size(), LLAMA_TOKEN_NULL);
                GGML_ASSERT(llama_decode_sampled_batch_async(tc.ctx.get(), items.data(), items.size(), consumed.data()) == 0);
                for (size_t i = 0; i < items.size(); ++i) {
                    result[items[i].seq_id].push_back(consumed[i]);
                    last_inputs[items[i].seq_id] = consumed[i];
                }
            } else {
                llama_batch batch = llama_batch_init(items.size(), 0, 1);
                for (const auto & item : items) {
                    result[item.seq_id].push_back(tokens[item.seq_id]);
                    last_inputs[item.seq_id] = tokens[item.seq_id];
                    common_batch_add(batch, tokens[item.seq_id], item.pos, {item.seq_id}, true);
                }
                GGML_ASSERT(llama_decode(tc.ctx.get(), batch) == 0);
                llama_batch_free(batch);
            }
            for (size_t i = 0; i < items.size(); ++i) {
                tc.seq_positions[items[i].seq_id]++;
                tc.last_batch_info[items[i].seq_id] = i;
                if (!overlap || step == 63) {
                    tokens[items[i].seq_id] = llama_get_sampled_token_ith(tc.ctx.get(), i);
                }
            }
        }
        const llama_pos peer_pos = llama_memory_seq_pos_max(memory, 2);
        const llama_pos pos = llama_memory_seq_pos_max(memory, 0);
        GGML_ASSERT(llama_memory_seq_rm(memory, 0, pos, -1));
        llama_sampler_copy(checkpoint.get(), samplers[0].get());
        tc.seq_positions[0] = pos;
        GGML_ASSERT(tc.decode_token(last_inputs[0], 0));
        GGML_ASSERT(llama_get_sampled_token_ith(tc.ctx.get(), 0) == tokens[0]);
        GGML_ASSERT(llama_memory_seq_pos_max(memory, 2) == peer_pos);
        return result;
    };
    if (!params.sampled_decode_gpu) {
        return;
    }
    for (bool unified : {false, true}) {
        const auto expected = generate(false, unified);
        GGML_ASSERT(expected == generate(true, unified));
        printf("sampled batch decode parity PASSED (kv_unified = %d)\n", unified);
    }
    printf("sampled batch decode parity, reordered rows, shrinking batch, and independent rollback PASSED\n");
}

static void test_backend_decode_sampled_lifecycle(const test_params & params) {
    if (!params.sampled_decode_gpu) {
        return;
    }
    const auto run = [&](bool overlap, bool unified) {
        struct sequence {
            bool active = false;
            bool pending = false;
            int generation = 0;
            llama_pos pos = 0;
            llama_token last = LLAMA_TOKEN_NULL;
        };
        std::vector<llama_sampler_ptr> samplers(4);
        std::vector<llama_sampler_ptr> checkpoints(4);
        std::vector<llama_sampler_seq_config> configs;
        test_context tc(params, configs, 4, 0, 0, 1, unified, 256);
        auto * ctx = tc.ctx.get();
        auto * memory = llama_get_memory(ctx);
        std::array<sequence, 4> seqs;
        std::map<std::pair<llama_seq_id, int>, std::vector<llama_token>> result;
        int n_queued = 0;
        int n_discarded = 0;
        int n_boundaries = 0;
        int n_cancel_pending = 0;
        int n_gap_queued = 0;
        int n_gap_fallback = 0;
        int n_reuse_queued[2] = {};
        const llama_pos limit = 64;

        for (int step = 0; step < 96; ++step) {
            std::map<llama_seq_id, std::string> admissions;
            if (step == 0) {
                admissions = {{0, "Hello"}, {1, "The sky is"}};
            } else if (step == 8) {
                admissions = {{2, "Count to ten"}};
            } else if (step == 12) {
                admissions = {{3, "A small cat"}};
            } else if (step == 23) {
                admissions = {{1, "Write a simple loop"}};
            } else if (step == 40) {
                admissions = {{2, "A green tree"}};
            }
            const llama_seq_id cancel = step == 20 ? 1 : step == 36 ? 2 : -1;
            n_cancel_pending += cancel >= 0 && seqs[cancel].pending;
            bool reset = !admissions.empty() || cancel >= 0;
            for (const auto & seq : seqs) {
                reset |= seq.active && seq.pos >= limit - 1;
            }
            if (reset) {
                llama_synchronize(ctx);
                for (llama_seq_id id = 0; id < 4; ++id) {
                    auto & seq = seqs[id];
                    if (seq.pending) {
                        GGML_ASSERT(llama_memory_seq_rm(memory, id, seq.pos - 1, -1));
                        GGML_ASSERT(llama_memory_seq_pos_max(memory, id) == seq.pos - 2);
                        llama_sampler_copy(checkpoints[id].get(), samplers[id].get());
                        seq.pending = false;
                        ++n_discarded;
                    }
                }
            }
            if (cancel >= 0) {
                GGML_ASSERT(llama_set_sampler(ctx, cancel, nullptr));
                GGML_ASSERT(llama_memory_seq_rm(memory, cancel, 0, -1));
                seqs[cancel].active = false;
                GGML_ASSERT(llama_memory_seq_pos_max(memory, cancel) == -1);
            }
            for (llama_seq_id id = 0; id < 4; ++id) {
                auto & seq = seqs[id];
                if (!seq.active || seq.pos < limit - 1) {
                    continue;
                }
                if (llama_memory_can_shift(memory)) {
                    GGML_ASSERT(llama_memory_seq_rm(memory, id, 1, 33));
                    llama_memory_seq_add(memory, id, 33, -1, -32);
                    seq.pos -= 32;
                    GGML_ASSERT(llama_memory_seq_pos_max(memory, id) == seq.pos - 2);
                } else {
                    GGML_ASSERT(llama_memory_seq_rm(memory, id, 0, -1));
                    seq.active = false;
                    admissions[id] = "Begin a fresh answer";
                }
                ++n_boundaries;
            }

            llama_batch batch = llama_batch_init(512, 0, 1);
            std::map<llama_seq_id, int32_t> output_indices;
            for (llama_seq_id id = 0; id < 4; ++id) {
                auto & seq = seqs[id];
                const auto admission = admissions.find(id);
                if (admission != admissions.end()) {
                    GGML_ASSERT(llama_set_sampler(ctx, id, nullptr));
                    samplers[id] = sampled_decode_sampler(params, 12345 + id);
                    checkpoints[id].reset(llama_sampler_clone(samplers[id].get()));
                    GGML_ASSERT(llama_set_sampler(ctx, id, samplers[id].get()));
                    GGML_ASSERT(llama_memory_seq_rm(memory, id, 0, -1));
                    seq.active = true;
                    seq.pending = false;
                    seq.pos = 0;
                    ++seq.generation;
                    const auto prompt = common_tokenize(ctx, admission->second, true);
                    for (size_t i = 0; i < prompt.size(); ++i) {
                        common_batch_add(batch, prompt[i], seq.pos++, {id}, i + 1 == prompt.size());
                    }
                    output_indices[id] = batch.n_tokens - 1;
                } else if (seq.active && !seq.pending) {
                    common_batch_add(batch, seq.last, seq.pos - 1, {id}, true);
                    output_indices[id] = batch.n_tokens - 1;
                } else if (seq.active) {
                    output_indices[id] = tc.idx_for_seq(id);
                }
            }
            if (batch.n_tokens > 0) {
                GGML_ASSERT(llama_decode(ctx, batch) == 0);
            }
            llama_batch_free(batch);

            std::vector<llama_sampled_decode_item> items;
            bool can_queue = overlap && step != 30;
            for (llama_seq_id id = 0; id < 4; ++id) {
                if (seqs[id].active) {
                    items.push_back({id, seqs[id].pos});
                    can_queue &= seqs[id].pos + 2 < limit;
                    seqs[id].pending = false;
                }
            }
            std::vector<llama_token> tokens(items.size(), LLAMA_TOKEN_NULL);
            if (can_queue) {
                for (const auto & item : items) {
                    llama_sampler_copy(samplers[item.seq_id].get(), checkpoints[item.seq_id].get());
                }
            }
            const int ret = can_queue ? llama_decode_sampled_batch_async(ctx, items.data(), items.size(), tokens.data()) : 1;
            GGML_ASSERT(ret == 0 || ret == 1);
            if ((step >= 20 && step < 23) || (step >= 36 && step < 40)) {
                n_gap_queued += ret == 0;
                n_gap_fallback += ret == 1;
            }
            n_reuse_queued[0] += step >= 23 && step < 30 && ret == 0;
            n_reuse_queued[1] += step >= 40 && step < 50 && ret == 0;
            for (size_t i = 0; i < items.size(); ++i) {
                const llama_seq_id id = items[i].seq_id;
                auto & seq = seqs[id];
                if (ret == 1) {
                    tokens[i] = llama_get_sampled_token_ith(ctx, output_indices.at(id));
                } else {
                    seq.pending = true;
                    tc.last_batch_info[id] = i;
                    ++n_queued;
                }
                GGML_ASSERT(tokens[i] != LLAMA_TOKEN_NULL);
                seq.last = tokens[i];
                result[{id, seq.generation}].push_back(tokens[i]);
                ++seq.pos;
                GGML_ASSERT(llama_memory_seq_pos_max(memory, id) == seq.pos - (seq.pending ? 1 : 2));
            }
        }
        GGML_ASSERT(n_boundaries > 0);
        if (overlap) {
            GGML_ASSERT(n_queued > 0 && n_discarded > 0);
            GGML_ASSERT(n_cancel_pending == 2);
            GGML_ASSERT(n_reuse_queued[0] > 0 && n_reuse_queued[1] > 0);
            GGML_ASSERT(!unified || n_gap_queued > 0);
            printf("lifecycle coverage: gap queued=%d fallback=%d, reuse queued=%d/%d, pending cancels=%d, boundaries=%d\n",
                    n_gap_queued, n_gap_fallback, n_reuse_queued[0], n_reuse_queued[1], n_cancel_pending, n_boundaries);
        }
        return result;
    };
    for (bool unified : {false, true}) {
        const auto expected = run(false, unified);
        const auto actual = run(true, unified);
        for (const auto & entry : expected) {
            if (entry.second != actual.at(entry.first)) {
                const auto & got = actual.at(entry.first);
                size_t first = 0;
                while (first < entry.second.size() && first < got.size() && entry.second[first] == got[first]) {
                    ++first;
                }
                fprintf(stderr, "lifecycle mismatch: unified=%d seq=%d generation=%d index=%zu expected=%d actual=%d\n",
                        unified, entry.first.first, entry.first.second, first,
                        first < entry.second.size() ? entry.second[first] : LLAMA_TOKEN_NULL,
                        first < got.size() ? got[first] : LLAMA_TOKEN_NULL);
            }
        }
        GGML_ASSERT(expected == actual);
        printf("sampled decode joins, cancellation, sparse slots, reuse, fallback and context boundaries PASSED (kv_unified = %d)\n", unified);
    }
}

struct backend_test_case {
    std::string name;
    void (*fn)(const test_params &);
    bool enabled_by_default;
};

static const backend_test_case BACKEND_TESTS[] = {
    { "decode_sampled",  test_backend_decode_sampled,          false },
    { "decode_sampled_batch", test_backend_decode_sampled_batch, false },
    { "decode_sampled_lifecycle", test_backend_decode_sampled_lifecycle, false },
    { "greedy",          test_backend_greedy_sampling,         true  },
    { "logit_bias",      test_backend_logit_bias_sampling,     true  },
    { "penalties",       test_backend_penalties_sampling,      true  },
    { "temp",            test_backend_temp_sampling,           true  },
    { "temp_ext",        test_backend_temp_ext_sampling,       true  },
    { "top_k",           test_backend_top_k_sampling,          true  },
    { "multi_sequence",  test_backend_multi_sequence_sampling, true  },
    { "dist",            test_backend_dist_sampling,           true  },
    { "dist_and_cpu",    test_backend_dist_sampling_and_cpu,   true  },
    { "set_sampler",     test_backend_set_sampler,             true  },
    { "multi_output_limit",    test_backend_multi_output_limit,      true },
    { "multi_sequence_multi_output_dist", test_backend_multi_sequence_multi_output_dist, true },
    { "multi_output_dist_transaction", test_backend_multi_output_dist_transaction, true },
    { "multi_output_sampling_chain", test_backend_multi_output_sampling_chain, true },
    { "multi_output_cpu",      test_backend_multi_output_cpu_suffix, true },
    { "mixed",           test_backend_mixed_sampling,          true  },
    { "min_p",           test_backend_min_p_sampling,          true  },
    { "cpu_mixed",       test_backend_cpu_mixed_batch,         true  },
    { "top_p",           test_backend_top_p_sampling,          true  },
};

static test_args parse_cli(int argc, char ** argv) {
    test_args out;

    for (int i = 1; i < argc; ++i) {
        const char * arg = argv[i];

        if (std::strcmp(arg, "--staged-input-cache") == 0) {
            if (i + 1 >= argc || (out.staged_input_cache = std::atoi(argv[++i])) <= 0) {
                fprintf(stderr, "--staged-input-cache expects a positive slot count\n");
                exit(EXIT_FAILURE);
            }
            continue;
        }

        if (std::strcmp(arg, "--test") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--test expects a value\n");
                exit(EXIT_FAILURE);
            }
            out.test = argv[++i];
            continue;
        }
        if (std::strncmp(arg, "--test=", 7) == 0) {
            out.test = arg + 7;
            continue;
        }
        if (std::strcmp(arg, "--model") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--model expects a value\n");
                exit(EXIT_FAILURE);
            }
            out.model = argv[++i];
            continue;
        }
        if (std::strncmp(arg, "--model=", 8) == 0) {
            out.model = arg + 8;
            continue;
        }
        if (std::strcmp(arg, "--device") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--device expects a value (cpu or gpu)\n");
                exit(EXIT_FAILURE);
            }
            out.device = argv[++i];
            continue;
        }
        if (std::strncmp(arg, "--device=", 9) == 0) {
            out.device = arg + 9;
            continue;
        }
        if (out.model.empty()) {
            out.model = arg;
            continue;
        }

        fprintf(stderr, "Unexpected argument: %s\n", arg);
        exit(EXIT_FAILURE);
    }

    if (out.device != "cpu" && out.device != "gpu" && out.device != "auto") {
        fprintf(stderr, "Invalid device '%s'. Must be 'cpu', 'gpu' or 'auto'\n", out.device.c_str());
        exit(EXIT_FAILURE);
    }

    return out;
}

static std::vector<const backend_test_case *> collect_tests_to_run(const std::string & requested) {
    std::vector<const backend_test_case *> selected;

    if (!requested.empty()) {
        for (const auto & test : BACKEND_TESTS) {
            if (test.name == requested) {
                selected.push_back(&test);
                break;
            }
        }
        if (selected.empty()) {
            fprintf(stderr, "Unknown test '%s'. Available tests:\n", requested.c_str());
            for (const auto & test : BACKEND_TESTS) {
                fprintf(stderr, "  %s\n", test.name.c_str());
            }
            exit(EXIT_FAILURE);
        }
    } else {
        for (const auto & test : BACKEND_TESTS) {
            if (!test.enabled_by_default) {
                continue;
            }
#ifdef GGML_USE_HIP
            // TODO: remove this when https://github.com/ggml-org/llama.cpp/pull/26592 is merged
            if (test.name == "penalties" || test.name == "set_sampler" ||
                test.name == "mixed"     || test.name == "top_p"       ||
                test.name == "multi_output_sampling_chain" ||
                test.name == "multi_output_cpu") {
                fprintf(stderr, "Skipping test '%s' on HIP backend (no backend TOP_K support)\n", test.name.c_str());
                continue;
            }
#endif // GGML_USE_HIP
            selected.push_back(&test);
        }
    }

    if (selected.empty()) {
        fprintf(stderr, "No backend sampling tests selected. Use --test=<name> to pick one.\n");
    }

    return selected;
}

static void run_tests(const std::vector<const backend_test_case *> & tests, const test_params & args) {
    for (const auto & test : tests) {
        fprintf(stderr, "\n=== %s ===\n", test->name.c_str());
        try {
            test->fn(args);
        } catch (const std::exception & e) {
            fprintf(stderr, "Error running test '%s': %s\n", test->name.c_str(), e.what());
            exit(EXIT_FAILURE);
        }
    }
}

int main(int argc, char ** argv) {
    test_args args = parse_cli(argc, argv);

    if (args.model.empty()) {
        args.model = common_get_model_or_exit(1, argv);
    }

    {
        std::ifstream file(args.model);
        if (!file.is_open()) {
            fprintf(stderr, "no model '%s' found\n", args.model.c_str());
            return EXIT_FAILURE;
        }
    }

    fprintf(stderr, "using '%s'\n", args.model.c_str());

    llama_backend_init();

    test_params params = {
        /*.model =*/ load_model(args),
    };
    params.sampled_decode = (args.test == "decode_sampled" || args.test == "decode_sampled_batch" || args.test == "decode_sampled_lifecycle");
    params.sampled_host_inputs = args.staged_input_cache > 0;
    auto * gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    params.sampled_decode_gpu = args.device == "gpu" && gpu &&
            strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(gpu)), "CUDA") == 0;

    const std::vector<const backend_test_case *> tests = collect_tests_to_run(args.test);
    if (!tests.empty()) {
        run_tests(tests, params);
        if (params.sampled_decode) {
            params.sampled_stochastic = true;
            fprintf(stderr, "\n=== stochastic sampled decode ===\n");
            run_tests(tests, params);
        }
    }

    return 0;
}
