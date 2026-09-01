#include "arg.h"
#include "common.h"
#include "download.h"
#include "llama.h"
#include "speculative.h"

#include <limits>
#include <string>
#include <vector>
#include <sstream>
#include <unordered_set>

#undef NDEBUG
#include <cassert>

static void set_test_env(const char * name, const char * value) {
#ifdef _WIN32
    assert(_putenv_s(name, value) == 0);
#else
    assert(setenv(name, value, true) == 0);
#endif
}

static void unset_test_env(const char * name) {
#ifdef _WIN32
    assert(_putenv_s(name, "") == 0);
#else
    assert(unsetenv(name) == 0);
#endif
}

static void test(void) {
    common_params params;

    auto assert_output_limits = [](int32_t n_batch, int32_t n_parallel, int32_t n_draft,
                                   int32_t total, int32_t per_seq) {
        const auto limits = common_speculative_get_output_limits(n_batch, n_parallel, n_draft);
        assert(limits.total == total);
        assert(limits.per_seq == per_seq);
    };

    assert_output_limits(16, 2,  3, 8, 4);
    assert_output_limits(16, 2, -1, 2, 1);
    assert_output_limits( 6, 2,  3, 6, 4);
    assert_output_limits( 2, 1,  3, 2, 2);
    assert_output_limits(
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max());

    {
        common_params base;
        base.n_parallel = 4;
        base.n_outputs_max_per_seq = 8;

        const auto draft = common_base_params_to_speculative(base);
        assert(draft.n_outputs_max == 4);
        assert(draft.n_outputs_max_per_seq == 1);

        base.n_batch = 16;
        base.n_parallel = 2;
        base.speculative.draft.n_max = 3;
        base.phase_aware_workspace = true;
        const auto phase_draft = common_base_params_to_speculative(base);
        assert(phase_draft.n_outputs_max == 2);
        assert(phase_draft.n_outputs_max_per_seq == 1);
    }

    printf("test-arg-parser: make sure there is no duplicated arguments in any examples\n\n");
    for (int ex = 0; ex < LLAMA_EXAMPLE_COUNT; ex++) {
        try {
            auto ctx_arg = common_params_parser_init(params, (enum llama_example)ex);
            common_params_add_preset_options(ctx_arg.options);
            std::unordered_set<std::string> seen_args;
            std::unordered_set<std::string> seen_env_vars;
            for (const auto & opt : ctx_arg.options) {
                // check for args duplications
                for (const auto & arg : opt.get_args()) {
                    if (seen_args.find(arg) == seen_args.end()) {
                        seen_args.insert(arg);
                    } else {
                        fprintf(stderr, "test-arg-parser: found different handlers for the same argument: %s", arg.c_str());
                        exit(1);
                    }
                }
                // check for env var duplications
                for (const auto & env : opt.get_env()) {
                    if (seen_env_vars.find(env) == seen_env_vars.end()) {
                        seen_env_vars.insert(env);
                    } else {
                        fprintf(stderr, "test-arg-parser: found different handlers for the same env var: %s", env.c_str());
                        exit(1);
                    }
                }

                // exclude spec args from this check
                // ref: https://github.com/ggml-org/llama.cpp/pull/22397
                const bool skip = opt.is_spec;

                // ensure shorter argument precedes longer argument
                if (!skip && opt.args.size() > 1) {
                    const std::string first(opt.args.front());
                    const std::string last(opt.args.back());

                    if (first.length() > last.length()) {
                        fprintf(stderr, "test-arg-parser: shorter argument should come before longer one: %s, %s\n",
                                first.c_str(), last.c_str());
                        assert(false);
                    }
                }

                // same check for negated arguments
                if (opt.args_neg.size() > 1) {
                    const std::string first(opt.args_neg.front());
                    const std::string last(opt.args_neg.back());

                    if (first.length() > last.length()) {
                        fprintf(stderr, "test-arg-parser: shorter negated argument should come before longer one: %s, %s\n",
                                first.c_str(), last.c_str());
                        assert(false);
                    }
                }
            }
        } catch (std::exception & e) {
            printf("%s\n", e.what());
            assert(false);
        }
    }

    auto list_str_to_char = [](std::vector<std::string> & argv) -> std::vector<char *> {
        std::vector<char *> res;
        for (auto & arg : argv) {
            res.push_back(const_cast<char *>(arg.data()));
        }
        return res;
    };

    std::vector<std::string> argv;

    printf("test-arg-parser: test invalid usage\n\n");

    // missing value
    argv = {"binary_name", "-m"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // wrong value (int)
    argv = {"binary_name", "-ngl", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    argv = {"binary_name", "--kv-gpu-layers", "-1"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    argv = {"binary_name", "--spec-draft-kv-gpu-layers", "-1"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SPECULATIVE));

    // wrong value (enum)
    argv = {"binary_name", "-sm", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    {
        common_params penalty_params;
        assert(penalty_params.sampling.penalty_last_n == 64);
        assert(penalty_params.sampling.dry_penalty_last_n == 64);

        argv = {"binary_name", "--repeat-last-n", "-1"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--dry-penalty-last-n", "-1"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "0"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "-1"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "nan"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "inf"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "-inf"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        const char * penalty_options[] = {"--frequency-penalty", "--presence-penalty"};
        const char * nonfinite_values[] = {"nan", "inf", "-inf"};
        for (const char * option : penalty_options) {
            for (const char * value : nonfinite_values) {
                argv = {"binary_name", option, value};
                assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));
            }
        }
    }

    // non-existence arg in specific example (--draft cannot be used outside llama-speculative)
    argv = {"binary_name", "--draft", "123"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_EMBEDDING));

    argv = {"binary_name", "-lm", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    printf("test-arg-parser: test valid usage\n\n");

    argv = {"binary_name", "-m", "model_file.gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "model_file.gguf");

    argv = {"binary_name", "-t", "1234"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.cpuparams.n_threads == 1234);

    argv = {"binary_name", "--verbose"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.verbosity > 1);

    argv = {"binary_name", "-m", "abc.gguf", "--predict", "6789", "--batch-size", "9090"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "abc.gguf");
    assert(params.n_predict == 6789);
    assert(params.n_batch == 9090);

    // --draft cannot be used outside llama-speculative
    argv = {"binary_name", "--spec-draft-n-max", "123"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SPECULATIVE));
    assert(params.speculative.draft.n_max == 123);

    params = common_params();
    argv = {"binary_name", "-m", "model_file.gguf", "--spec-draft-ubatch-size", "64"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SPECULATIVE));
    assert(params.speculative.draft.n_ubatch == 64);

    params = common_params();
    argv = {"binary_name", "-m", "model_file.gguf", "--ubatch-size-draft", "32"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SPECULATIVE));
    assert(params.speculative.draft.n_ubatch == 32);

    params = common_params();
    argv = {"binary_name", "-m", "model_file.gguf", "-ubd", "16"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SPECULATIVE));
    assert(params.speculative.draft.n_ubatch == 16);

    params = common_params();
    argv = {"binary_name", "-m", "model_file.gguf", "--spec-draft-ubatch-size", "-1"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SPECULATIVE));

    struct mtp_parse_case {
        std::vector<std::string> args;
        bool valid;
        int32_t planes;
        int32_t n_rs_seq;
        int32_t capped;
    };
    const mtp_parse_case mtp_cases[] = {
        { { "--spec-type", "draft-mtp", "--ubatch-size", "512", "--spec-draft-ubatch-size", "128" }, false, -1, -1, -1 },
        { { "--spec-type", "draft-mtp", "-b", "256", "-ub", "512", "-ubd", "512" }, true, -1, -1, -1 },
        { { "--spec-type", "draft-mtp", "-b", "256", "-ub", "512", "-ubd", "256" }, false, -1, -1, -1 },
        { { "--spec-type", "draft-mtp", "--spec-draft-n-max", "256", "--spec-mtp-rs-planes", "2", "-b", "256", "-ub", "512", "-ubd", "512" }, false, -1, -1, -1 },
        { { "--spec-type", "draft-mtp", "--spec-draft-n-max", "256", "--spec-mtp-rs-planes", "2", "-b", "256", "-ub", "0" }, false, -1, -1, -1 },
        { { "--spec-type", "draft-mtp", "--spec-draft-n-max", "8" }, true, 0, 8, false },
        { { "--spec-type", "draft-mtp", "--spec-draft-n-max", "8", "--spec-mtp-rs-planes", "0" }, true, 0, 8, false },
        { { "--spec-type", "draft-mtp", "--spec-draft-n-max", "8", "--spec-mtp-rs-planes", "2" }, true, 2, 1, true },
        { { "--spec-type", "draft-mtp", "--spec-draft-n-max", "8", "--ubatch-size", "8", "--spec-mtp-rs-planes", "2" }, false, -1, -1, -1 },
        { { "--spec-type", "draft-mtp", "--spec-draft-n-max", "8", "--ubatch-size", "9", "--spec-mtp-rs-planes", "2" }, true, 2, 1, true },
        { { "--spec-type", "draft-mtp", "--spec-draft-n-max", "8", "--batch-size", "8", "--ubatch-size", "9", "--spec-mtp-rs-planes", "2" }, false, -1, -1, -1 },
        { { "--spec-type", "draft-mtp", "--spec-draft-n-max", "8", "--spec-mtp-rs-planes", "9" }, true, 9, 8, false },
        { { "--spec-type", "draft-mtp", "--spec-draft-n-max", "8", "--spec-mtp-rs-planes", "-1" }, false, -1, -1, -1 },
        { { "--spec-type", "draft-mtp", "--spec-draft-n-max", "8", "--spec-mtp-rs-planes", "1" }, false, -1, -1, -1 },
        { { "--spec-type", "draft-mtp", "--spec-draft-n-max", "8", "--spec-mtp-rs-planes", "10" }, false, -1, -1, -1 },
        { { "--spec-type", "draft-dflash", "--spec-draft-n-max", "8", "--spec-mtp-rs-planes", "4" }, false, -1, -1, -1 },
        { { "--spec-type", "draft-mtp,draft-eagle3", "--spec-draft-n-max", "8", "--spec-mtp-rs-planes", "4" }, false, -1, -1, -1 },
        { { "--spec-type", "draft-mtp,draft-eagle3", "--spec-draft-n-max", "8" }, false, -1, -1, -1 },
    };
    for (const auto & test_case : mtp_cases) {
        params = common_params();
        argv = { "binary_name" };
        argv.insert(argv.end(), test_case.args.begin(), test_case.args.end());
        assert(test_case.valid == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));
        if (test_case.planes >= 0) {
            assert(params.speculative.mtp_rs_planes == test_case.planes);
            assert(params.speculative.need_n_rs_seq() == (uint32_t) test_case.n_rs_seq);
            assert(params.speculative.is_mtp_rs_capped() == (bool) test_case.capped);
        }
    }

    params = common_params();
    params.model.path = "model_file.gguf";

    common_params draft_arg_params;
    assert(draft_arg_params.speculative.draft.kv_gpu_layers == -1);
    argv = {"binary_name", "-m", "model_file.gguf", "--kv-gpu-layers-draft", "0"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), draft_arg_params, LLAMA_EXAMPLE_SPECULATIVE));
    assert(draft_arg_params.speculative.draft.kv_gpu_layers == 0);

    {
        common_params synth_params;
        argv = {"binary_name", "--spec-synth-len", "3.4"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), synth_params, LLAMA_EXAMPLE_SERVER));
        assert(synth_params.speculative.synth_len == 3.4);
    }

    {
        common_params synth_params;
        argv = {"binary_name", "--spec-synth-rates", "0.8,0.6,0.2"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), synth_params, LLAMA_EXAMPLE_SERVER));
        assert(synth_params.speculative.synth_rates == std::vector<double>({0.8, 0.6, 0.2}));
    }

    {
        common_params synth_params;
        argv = {"binary_name", "--spec-synth-len", "3.4x"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), synth_params, LLAMA_EXAMPLE_SERVER));
    }

    argv = {"binary_name", "-lm", "none"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_NONE);

    argv = {"binary_name", "-lm", "mmap"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP);

    argv = {"binary_name", "-lm", "mlock"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MLOCK);

    argv = {"binary_name", "-lm", "mmap+mlock"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK);

    argv = {"binary_name", "-lm", "dio"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_DIRECT_IO);

    {
        const auto defaults = llama_context_default_params();
        assert(!defaults.kv_cpu_pinned);
        assert(!defaults.recurrent_state_offload);
        assert(defaults.kv_gpu_layers == 0);
        assert(!defaults.phase_aware_workspace);
        assert(!defaults.live_context_workspace);

        common_params placement_params;
        argv = {"binary_name", "-m", "model.gguf", "--no-kv-offload", "--kv-cpu-pinned", "--kv-gpu-layers", "4", "--recurrent-state-offload"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), placement_params, LLAMA_EXAMPLE_COMMON));
        assert(placement_params.no_kv_offload);
        assert(placement_params.kv_cpu_pinned);
        assert(placement_params.kv_gpu_layers == 4);
        assert(placement_params.recurrent_state_offload);

        const auto cparams = common_context_params_to_llama(placement_params);
        assert(!cparams.offload_kqv);
        assert(cparams.kv_cpu_pinned);
        assert(cparams.kv_gpu_layers == 4);
        assert(cparams.recurrent_state_offload);

        argv = {"binary_name", "--no-kv-cpu-pinned", "--no-recurrent-state-offload"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), placement_params, LLAMA_EXAMPLE_COMMON));
        assert(!placement_params.kv_cpu_pinned);
        assert(!placement_params.recurrent_state_offload);
    }

    {
        unset_test_env("LLAMA_ARG_PHASE_AWARE_WORKSPACE");
        common_params phase_params;
        argv = {"binary_name", "-m", "model.gguf"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), phase_params, LLAMA_EXAMPLE_SERVER));
        assert(!phase_params.phase_aware_workspace);

        phase_params = common_params();
        argv = {"binary_name", "-m", "model.gguf", "--phase-aware-workspace"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), phase_params, LLAMA_EXAMPLE_SERVER));
        assert(phase_params.phase_aware_workspace);
        assert(common_context_params_to_llama(phase_params).phase_aware_workspace);

        phase_params = common_params();
        argv = {"binary_name", "-m", "model.gguf", "--phase-aware-workspace", "--no-phase-aware-workspace"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), phase_params, LLAMA_EXAMPLE_SERVER));
        assert(!phase_params.phase_aware_workspace);

        set_test_env("LLAMA_ARG_PHASE_AWARE_WORKSPACE", "1");
        phase_params = common_params();
        argv = {"binary_name", "-m", "model.gguf"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), phase_params, LLAMA_EXAMPLE_SERVER));
        assert(phase_params.phase_aware_workspace);
        unset_test_env("LLAMA_ARG_PHASE_AWARE_WORKSPACE");
    }

    {
        unset_test_env("LLAMA_ARG_LIVE_CONTEXT_WORKSPACE");
        common_params live_params;
        argv = {"binary_name", "-m", "model.gguf"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), live_params, LLAMA_EXAMPLE_SERVER));
        assert(!live_params.live_context_workspace);
        assert(!common_context_params_to_llama(live_params).live_context_workspace);

        live_params = common_params();
        argv = {"binary_name", "-m", "model.gguf", "--live-context-workspace"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), live_params, LLAMA_EXAMPLE_SERVER));
        assert(live_params.live_context_workspace);
        assert(!live_params.phase_aware_workspace);
        assert(common_context_params_to_llama(live_params).live_context_workspace);

        live_params = common_params();
        argv = {"binary_name", "-m", "model.gguf", "--live-context-workspace", "--no-live-context-workspace"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), live_params, LLAMA_EXAMPLE_SERVER));
        assert(!live_params.live_context_workspace);

        set_test_env("LLAMA_ARG_LIVE_CONTEXT_WORKSPACE", "1");
        live_params = common_params();
        argv = {"binary_name", "-m", "model.gguf"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), live_params, LLAMA_EXAMPLE_SERVER));
        assert(live_params.live_context_workspace);
        assert(!live_params.phase_aware_workspace);
        unset_test_env("LLAMA_ARG_LIVE_CONTEXT_WORKSPACE");
    }

    {
        common_params draft_placement_params;
        draft_placement_params.no_kv_offload = true;
        draft_placement_params.kv_gpu_layers = 7;

        const auto inherited = common_base_params_to_speculative(draft_placement_params);
        assert(inherited.no_kv_offload);
        assert(inherited.kv_gpu_layers == 7);

        draft_placement_params.no_kv_offload = false;
        draft_placement_params.speculative.draft.kv_gpu_layers = 3;
        const auto overridden = common_base_params_to_speculative(draft_placement_params);
        assert(overridden.no_kv_offload);
        assert(overridden.kv_gpu_layers == 3);

        const auto cparams = common_context_params_to_llama(overridden);
        assert(!cparams.offload_kqv);
        assert(cparams.kv_gpu_layers == 3);

        draft_placement_params.speculative.draft.kv_gpu_layers = 0;
        const auto host_only = common_base_params_to_speculative(draft_placement_params);
        assert(host_only.no_kv_offload);
        assert(host_only.kv_gpu_layers == 0);

        assert(!draft_placement_params.no_kv_offload);
        assert(draft_placement_params.kv_gpu_layers == 7);
    }

    // multi-value args (CSV)
    argv = {"binary_name", "--lora", "file1.gguf,\"file2,2.gguf\",\"file3\"\"3\"\".gguf\",file4\".gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.lora_adapters.size() == 4);
    assert(params.lora_adapters[0].path == "file1.gguf");
    assert(params.lora_adapters[1].path == "file2,2.gguf");
    assert(params.lora_adapters[2].path == "file3\"3\".gguf");
    assert(params.lora_adapters[3].path == "file4\".gguf");

// skip this part on windows, because setenv is not supported
#ifdef _WIN32
    printf("test-arg-parser: skip on windows build\n");
#else
    printf("test-arg-parser: test environment variables (valid + invalid usages)\n\n");

    setenv("LLAMA_ARG_SPEC_DRAFT_UBATCH", "32", true);
    params = common_params();
    argv = {"binary_name", "-m", "model_file.gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SPECULATIVE));
    assert(params.speculative.draft.n_ubatch == 32);
    unsetenv("LLAMA_ARG_SPEC_DRAFT_UBATCH");

    setenv("LLAMA_ARG_SPEC_DRAFT_KV_GPU_LAYERS", "3", true);
    common_params draft_env_params;
    argv = {"binary_name", "-m", "model_file.gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), draft_env_params, LLAMA_EXAMPLE_SPECULATIVE));
    assert(draft_env_params.speculative.draft.kv_gpu_layers == 3);
    unsetenv("LLAMA_ARG_SPEC_DRAFT_KV_GPU_LAYERS");

    setenv("LLAMA_ARG_THREADS", "blah", true);
    argv = {"binary_name"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    setenv("LLAMA_ARG_MODEL", "blah.gguf", true);
    setenv("LLAMA_ARG_THREADS", "1010", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "blah.gguf");
    assert(params.cpuparams.n_threads == 1010);

    setenv("LLAMA_ARG_LOAD_MODE", "blah", true);
    argv = {"binary_name"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    setenv("LLAMA_ARG_LOAD_MODE", "mmap", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP);

    setenv("LLAMA_ARG_LOAD_MODE", "mlock", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MLOCK);

    setenv("LLAMA_ARG_LOAD_MODE", "mmap+mlock", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK);

    setenv("LLAMA_ARG_LOAD_MODE", "dio", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_DIRECT_IO);

    printf("test-arg-parser: test negated environment variables\n\n");

    setenv("LLAMA_ARG_LOAD_MODE", "none", true);
    setenv("LLAMA_ARG_NO_PERF", "1", true); // legacy format
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_NONE);
    assert(params.no_perf == true);

    printf("test-arg-parser: test environment variables being overwritten\n\n");

    setenv("LLAMA_ARG_MODEL", "blah.gguf", true);
    setenv("LLAMA_ARG_THREADS", "1010", true);
    argv = {"binary_name", "-m", "overwritten.gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "overwritten.gguf");
    assert(params.cpuparams.n_threads == 1010);
#endif // _WIN32

    printf("test-arg-parser: test download functions\n\n");
    const char * GOOD_URL = "http://ggml.ai/";
    const char * BAD_URL  = "http://ggml.ai/404";

    {
        printf("test-arg-parser: test good URL\n\n");
        auto res = common_remote_get_content(GOOD_URL, {});
        assert(res.first == 200);
        assert(res.second.size() > 0);
        std::string str(res.second.data(), res.second.size());
        assert(str.find("llama.cpp") != std::string::npos);
    }

    {
        printf("test-arg-parser: test bad URL\n\n");
        auto res = common_remote_get_content(BAD_URL, {});
        assert(res.first == 404);
    }

    {
        printf("test-arg-parser: test max size error\n");
        common_remote_params params;
        params.max_size = 1;
        try {
            common_remote_get_content(GOOD_URL, params);
            assert(false && "it should throw an error");
        } catch (std::exception & e) {
            printf("  expected error: %s\n\n", e.what());
        }
    }

    printf("test-arg-parser: all tests OK\n\n");
}

static void test_draft_ubatch_override() {
    common_params params;
    params.n_ubatch = 512;

    const common_params inherited = common_base_params_to_speculative(params);
    assert(inherited.n_ubatch == 512);

    params.speculative.draft.n_ubatch = 64;
    const common_params overridden = common_base_params_to_speculative(params);
    assert(overridden.n_ubatch == 64);

    const llama_context_params cparams = common_context_params_to_llama(overridden);
    assert(cparams.n_ubatch == 64);

    assert(params.n_ubatch == 512);
}

static void test_mtp_draft_ubatch_validation() {
    common_params_speculative params;
    params.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };

    common_validate_speculative_params(params, 512, 512);
    params.draft.n_ubatch = 512;
    common_validate_speculative_params(params, 512, 512);

    params.draft.n_ubatch = 128;
    bool rejected = false;
    try {
        common_validate_speculative_params(params, 512, 512);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);

    common_validate_speculative_params(params, 0, 512);

    params.types = { COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH };
    common_validate_speculative_params(params, 512, 512);

    params.draft.n_ubatch = 0;
    params.draft.n_max = 8;
    params.mtp_rs_planes = 2;
    rejected = false;
    try {
        common_validate_speculative_params(params, 512, 512);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);

    params.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    common_validate_speculative_params(params, 512, 512);
}

static void test_model_backed_speculative_validation() {
    const std::vector<common_speculative_type> draftless = {
        COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE,
        COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K,
        COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V,
        COMMON_SPECULATIVE_TYPE_NGRAM_MOD,
        COMMON_SPECULATIVE_TYPE_NGRAM_CACHE,
    };
    const common_speculative_type model_backed[] = {
        COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE,
        COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3,
        COMMON_SPECULATIVE_TYPE_DRAFT_MTP,
        COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH,
        COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK,
    };

    common_params_speculative params;
    params.types = draftless;
    common_validate_speculative_params(params, 512, 512);

    for (const common_speculative_type type : model_backed) {
        params.types = draftless;
        params.types.insert(params.types.begin() + 2, type);
        common_validate_speculative_params(params, 512, 512);

        params.types = { type, type };
        common_validate_speculative_params(params, 512, 512);
    }

    const size_t n_model_backed = sizeof(model_backed) / sizeof(model_backed[0]);
    for (size_t i = 0; i < n_model_backed; ++i) {
        for (size_t j = i + 1; j < n_model_backed; ++j) {
            params.types = {
                model_backed[i],
                COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE,
                model_backed[j],
            };
            bool rejected = false;
            try {
                common_validate_speculative_params(params, 512, 512);
            } catch (const std::invalid_argument &) {
                rejected = true;
            }
            assert(rejected);
        }
    }
}

static void test_mtp_state_boundaries() {
    std::vector<uint8_t> state;
    const std::vector<uint8_t> malformed = { 0x01, 0x02, 0x03 };

    assert(!common_speculative_get_mtp_state(nullptr, 0, state));
    assert(state.empty());
    assert(common_speculative_set_mtp_state(nullptr, 0, {}));
    assert(!common_speculative_set_mtp_state(nullptr, 0, malformed));
}

int main(void) {
    try {
        test();
        test_draft_ubatch_override();
        test_mtp_draft_ubatch_validation();
        test_model_backed_speculative_validation();
        test_mtp_state_boundaries();
    } catch (std::exception & e) {
        fprintf(stderr, "test-arg-parser: exception: %s\n", e.what());
        return 1;
    }
    return 0;
}
