// Tests llama_context_set_split_mode: the weights are placed again for another split mode while the
// context keeps its memory, its logits and its identity.
//
// The check that matters is that a context which switched into a mode continues exactly like a
// context that was in that mode all along and was handed the same memory state.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama-cpp.h"

#include <algorithm>
#include <clocale>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

struct llama_batch_ptr {
    llama_batch batch;

    llama_batch_ptr(int32_t n_tokens, int32_t embd, int32_t n_seq_max)
        : batch{llama_batch_init(n_tokens, embd, n_seq_max)} {}

    ~llama_batch_ptr() { llama_batch_free(batch); }

    llama_batch_ptr(const llama_batch_ptr &) = delete;
    llama_batch_ptr & operator=(const llama_batch_ptr &) = delete;

    llama_batch & get() { return batch; }
};

static bool has_gpu_device() {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            return true;
        }
    }
    return false;
}

static llama_model * load_model(const std::string & path, common_params & params, llama_split_mode split_mode) {
    auto mparams = common_model_params_to_llama(params);
    mparams.split_mode = split_mode;

    return llama_model_load_from_file(path.c_str(), mparams);
}

static llama_context * make_context(llama_model * model, const common_params & params) {
    auto cparams = common_context_params_to_llama(params);
    return llama_init_from_model(model, cparams);
}

static bool decode_tokens(llama_context * ctx, const llama_tokens & tokens, int & n_past, int32_t n_batch) {
    llama_batch_ptr batch(n_batch, 0, 1);

    for (size_t i = 0; i < tokens.size(); i += n_batch) {
        const size_t n = std::min<size_t>(n_batch, tokens.size() - i);

        common_batch_clear(batch.get());
        for (size_t j = 0; j < n; j++) {
            common_batch_add(batch.get(), tokens[i + j], n_past + (int) j, {0}, i + j == tokens.size() - 1);
        }

        if (llama_decode(ctx, batch.get())) {
            LOG_ERR("%s: failed to decode\n", __func__);
            return false;
        }
        n_past += (int) n;
    }

    return true;
}

static llama_tokens generate_greedy(llama_context * ctx, int & n_past, int32_t n_predict) {
    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_greedy());

    llama_tokens result;
    llama_batch_ptr batch(1, 0, 1);

    for (int i = 0; i < n_predict; i++) {
        const llama_token id = llama_sampler_sample(smpl.get(), ctx, -1);
        result.push_back(id);

        common_batch_clear(batch.get());
        common_batch_add(batch.get(), id, n_past, {0}, true);

        if (llama_decode(ctx, batch.get())) {
            LOG_ERR("%s: failed to decode\n", __func__);
            return {};
        }
        n_past++;
    }

    return result;
}

// the logits of the last decode live in a host buffer of the context and must survive the switch,
// otherwise a caller would have to decode a token again before it can sample
// compared as raw bytes, because a dummy model can produce NaN logits and those never compare equal
static std::vector<uint8_t> get_last_logits(llama_context * ctx, const llama_model * model) {
    const float * logits = llama_get_logits_ith(ctx, -1);
    if (logits == nullptr) {
        return {};
    }
    const size_t n_bytes = llama_vocab_n_tokens(llama_model_get_vocab(model)) * sizeof(float);
    return std::vector<uint8_t>((const uint8_t *) logits, (const uint8_t *) logits + n_bytes);
}

static std::vector<uint8_t> get_state(llama_context * ctx) {
    std::vector<uint8_t> state(llama_state_get_size(ctx));
    if (!state.empty() && llama_state_get_data(ctx, state.data(), state.size()) != state.size()) {
        return {};
    }
    return state;
}

// generate under `sm` from a memory state that was produced elsewhere, so that the result can be
// compared against a context that reached the same state by switching into `sm`
static llama_tokens generate_from_state(
        const std::string & path, common_params & params, llama_split_mode sm,
        const std::vector<uint8_t> & state, llama_token last, int n_past) {
    llama_model_ptr model{load_model(path, params, sm)};
    if (!model) {
        LOG_ERR("%s: failed to load the model under split mode %s\n", __func__, llama_split_mode_name(sm));
        return {};
    }

    llama_context_ptr ctx{make_context(model.get(), params)};
    if (!ctx) {
        LOG_ERR("%s: failed to create the context under split mode %s\n", __func__, llama_split_mode_name(sm));
        return {};
    }

    if (llama_state_set_data(ctx.get(), state.data(), state.size()) != state.size()) {
        LOG_ERR("%s: failed to restore the state\n", __func__);
        return {};
    }

    if (!decode_tokens(ctx.get(), {last}, n_past, params.n_batch)) {
        return {};
    }

    return generate_greedy(ctx.get(), n_past, params.n_predict);
}

struct test_ctx {
    std::string   path;
    common_params params;
    llama_tokens  prompt;

    size_t n_pass = 0;
    size_t n_fail = 0;
    size_t n_skip = 0;

    void pass(const std::string & name) {
        LOG_INF("  PASS  %s\n", name.c_str());
        n_pass++;
    }

    void fail(const std::string & name, const std::string & why) {
        LOG_ERR("  FAIL  %s: %s\n", name.c_str(), why.c_str());
        n_fail++;
    }

    void skip(const std::string & name, const std::string & why) {
        LOG_INF("  SKIP  %s: %s\n", name.c_str(), why.c_str());
        n_skip++;
    }
};

// the whole prompt but its last token goes in under `from`, then the context switches to `to` and
// the last token plus the generation run there
static void test_switch(test_ctx & t, llama_split_mode from, llama_split_mode to) {
    const std::string name = std::string("switch ") + llama_split_mode_name(from) + " -> " + llama_split_mode_name(to);

    llama_model_ptr model{load_model(t.path, t.params, from)};
    if (!model) {
        t.skip(name, "the model does not load under the first split mode");
        return;
    }

    llama_context_ptr ctx{make_context(model.get(), t.params)};
    if (!ctx) {
        t.skip(name, "the context does not build under the first split mode");
        return;
    }

    llama_tokens head(t.prompt.begin(), t.prompt.end() - 1);
    const llama_token last = t.prompt.back();

    int n_past = 0;
    if (!decode_tokens(ctx.get(), head, n_past, t.params.n_batch)) {
        t.fail(name, "failed to decode the prompt");
        return;
    }

    const std::vector<uint8_t> state_before  = get_state(ctx.get());
    const std::vector<uint8_t> logits_before = get_last_logits(ctx.get(), model.get());
    if (state_before.empty() || logits_before.empty()) {
        t.fail(name, "failed to read the state before the switch");
        return;
    }

    if (!llama_context_set_split_mode(ctx.get(), to, nullptr)) {
        t.skip(name, "the split mode was rejected");
        return;
    }

    if (llama_model_get_split_mode(model.get()) != to) {
        t.fail(name, "the model reports the wrong split mode after the switch");
        return;
    }

    const std::vector<uint8_t> state_after = get_state(ctx.get());
    if (state_after != state_before) {
        t.fail(name, "the state is not the same after the switch");
        return;
    }

    if (get_last_logits(ctx.get(), model.get()) != logits_before) {
        t.fail(name, "the logits of the last decode did not survive the switch");
        return;
    }

    int n_past_switch = n_past;
    if (!decode_tokens(ctx.get(), {last}, n_past_switch, t.params.n_batch)) {
        t.fail(name, "failed to decode the last prompt token after the switch");
        return;
    }

    const llama_tokens got = generate_greedy(ctx.get(), n_past_switch, t.params.n_predict);
    if (got.empty()) {
        t.fail(name, "failed to generate after the switch");
        return;
    }

    // free the switched model before the reference asks for the same device memory
    ctx.reset();
    model.reset();

    const llama_tokens want = generate_from_state(t.path, t.params, to, state_before, last, n_past);
    if (want.empty()) {
        t.fail(name, "failed to generate the reference");
        return;
    }

    if (got != want) {
        t.fail(name, "the generation after the switch differs from the reference");
        return;
    }

    t.pass(name);
}

// a switch out and back must leave the context exactly where it was
static void test_round_trip(test_ctx & t, llama_split_mode from, llama_split_mode via) {
    const std::string name = std::string("round trip ") + llama_split_mode_name(from) + " -> " + llama_split_mode_name(via) + " -> " + llama_split_mode_name(from);

    llama_model_ptr model{load_model(t.path, t.params, from)};
    if (!model) {
        t.skip(name, "the model does not load under the split mode");
        return;
    }

    llama_context_ptr ctx{make_context(model.get(), t.params)};
    if (!ctx) {
        t.skip(name, "the context does not build under the split mode");
        return;
    }

    llama_tokens head(t.prompt.begin(), t.prompt.end() - 1);
    const llama_token last = t.prompt.back();

    int n_past = 0;
    if (!decode_tokens(ctx.get(), head, n_past, t.params.n_batch)) {
        t.fail(name, "failed to decode the prompt");
        return;
    }

    const std::vector<uint8_t> state = get_state(ctx.get());
    if (state.empty()) {
        t.fail(name, "failed to read the state");
        return;
    }

    if (!llama_context_set_split_mode(ctx.get(), via, nullptr)) {
        t.skip(name, "the split mode was rejected");
        return;
    }

    if (!llama_context_set_split_mode(ctx.get(), from, nullptr)) {
        t.fail(name, "failed to switch back");
        return;
    }

    if (get_state(ctx.get()) != state) {
        t.fail(name, "the state is not the same after the round trip");
        return;
    }

    int n_past_rt = n_past;
    if (!decode_tokens(ctx.get(), {last}, n_past_rt, t.params.n_batch)) {
        t.fail(name, "failed to decode the last prompt token after the round trip");
        return;
    }

    const llama_tokens got = generate_greedy(ctx.get(), n_past_rt, t.params.n_predict);
    if (got.empty()) {
        t.fail(name, "failed to generate after the round trip");
        return;
    }

    ctx.reset();
    model.reset();

    const llama_tokens want = generate_from_state(t.path, t.params, from, state, last, n_past);
    if (want.empty()) {
        t.fail(name, "failed to generate the reference");
        return;
    }

    if (got != want) {
        t.fail(name, "the generation after the round trip differs from the reference");
        return;
    }

    t.pass(name);
}

// a switch to the split mode the context already has changes nothing, and a split mode that the
// model cannot take leaves a context that still works
static void test_no_change(test_ctx & t, llama_split_mode sm, llama_split_mode unsupported) {
    const std::string name = std::string("no change under ") + llama_split_mode_name(sm);

    llama_model_ptr model{load_model(t.path, t.params, sm)};
    if (!model) {
        t.skip(name, "the model does not load under the split mode");
        return;
    }

    llama_context_ptr ctx{make_context(model.get(), t.params)};
    if (!ctx) {
        t.skip(name, "the context does not build under the split mode");
        return;
    }

    int n_past = 0;
    if (!decode_tokens(ctx.get(), t.prompt, n_past, t.params.n_batch)) {
        t.fail(name, "failed to decode the prompt");
        return;
    }

    const std::vector<uint8_t> state = get_state(ctx.get());

    if (!llama_context_set_split_mode(ctx.get(), sm, nullptr)) {
        t.fail(name, "a switch to the current split mode was rejected");
        return;
    }

    if (unsupported != sm && llama_context_set_split_mode(ctx.get(), unsupported, nullptr)) {
        // the model took it after all, which is not a failure of this test
        LOG_INF("  note: %s was accepted, skipping the rejection check\n", llama_split_mode_name(unsupported));
        if (!llama_context_set_split_mode(ctx.get(), sm, nullptr)) {
            t.fail(name, "failed to switch back");
            return;
        }
    }

    if (llama_model_get_split_mode(model.get()) != sm) {
        t.fail(name, "the split mode changed although nothing should have");
        return;
    }

    if (get_state(ctx.get()) != state) {
        t.fail(name, "the state changed although nothing should have");
        return;
    }

    int n_past_check = n_past;
    const llama_tokens got = generate_greedy(ctx.get(), n_past_check, t.params.n_predict);
    if (got.empty()) {
        t.fail(name, "the context no longer generates");
        return;
    }

    t.pass(name);
}

// a switch may also give the new mode a tensor split of its own, because a share of the layers and
// a share of every tensor are not the same thing
static void test_tensor_split(test_ctx & t, llama_split_mode from, llama_split_mode to) {
    const std::string name = std::string("tensor split ") + llama_split_mode_name(from) + " -> " + llama_split_mode_name(to);

    llama_model_ptr model{load_model(t.path, t.params, from)};
    if (!model) {
        t.skip(name, "the model does not load under the split mode");
        return;
    }

    llama_context_ptr ctx{make_context(model.get(), t.params)};
    if (!ctx) {
        t.skip(name, "the context does not build under the split mode");
        return;
    }

    llama_tokens head(t.prompt.begin(), t.prompt.end() - 1);
    const llama_token last = t.prompt.back();

    int n_past = 0;
    if (!decode_tokens(ctx.get(), head, n_past, t.params.n_batch)) {
        t.fail(name, "failed to decode the prompt");
        return;
    }

    const std::vector<uint8_t> state = get_state(ctx.get());
    if (state.empty()) {
        t.fail(name, "failed to read the state");
        return;
    }

    std::vector<float> tensor_split(llama_max_devices(), 0.0f);
    tensor_split[0] = 0.7f;
    tensor_split[1] = 0.3f;

    if (!llama_context_set_split_mode(ctx.get(), to, tensor_split.data())) {
        t.skip(name, "the split mode was rejected");
        return;
    }

    if (get_state(ctx.get()) != state) {
        t.fail(name, "the state is not the same after the switch");
        return;
    }

    int n_past_switch = n_past;
    if (!decode_tokens(ctx.get(), {last}, n_past_switch, t.params.n_batch)) {
        t.fail(name, "failed to decode the last prompt token after the switch");
        return;
    }

    if (generate_greedy(ctx.get(), n_past_switch, t.params.n_predict).empty()) {
        t.fail(name, "failed to generate after the switch");
        return;
    }

    t.pass(name);
}

// the model level API on its own, without a context on top of it
static void test_model_only(test_ctx & t, llama_split_mode from, llama_split_mode to) {
    const std::string name = std::string("model only ") + llama_split_mode_name(from) + " -> " + llama_split_mode_name(to);

    llama_model_ptr model{load_model(t.path, t.params, from)};
    if (!model) {
        t.skip(name, "the model does not load under the split mode");
        return;
    }

    if (!llama_model_set_split_mode(model.get(), to, nullptr)) {
        t.skip(name, "the split mode was rejected");
        return;
    }

    if (llama_model_get_split_mode(model.get()) != to) {
        t.fail(name, "the model reports the wrong split mode");
        return;
    }

    llama_context_ptr ctx{make_context(model.get(), t.params)};
    if (!ctx) {
        t.fail(name, "the context does not build on the placed weights");
        return;
    }

    int n_past = 0;
    if (!decode_tokens(ctx.get(), t.prompt, n_past, t.params.n_batch)) {
        t.fail(name, "failed to decode the prompt on the placed weights");
        return;
    }

    const llama_tokens got = generate_greedy(ctx.get(), n_past, t.params.n_predict);

    ctx.reset();

    llama_tokens want;
    {
        llama_model_ptr model_ref{load_model(t.path, t.params, to)};
        if (!model_ref) {
            t.fail(name, "failed to load the reference model");
            return;
        }
        llama_context_ptr ctx_ref{make_context(model_ref.get(), t.params)};
        if (!ctx_ref) {
            t.fail(name, "failed to create the reference context");
            return;
        }
        int n_past_ref = 0;
        if (!decode_tokens(ctx_ref.get(), t.prompt, n_past_ref, t.params.n_batch)) {
            t.fail(name, "failed to decode the prompt on the reference");
            return;
        }
        want = generate_greedy(ctx_ref.get(), n_past_ref, t.params.n_predict);
    }

    if (got.empty() || got != want) {
        t.fail(name, "the placed weights do not generate like a model loaded under the same split mode");
        return;
    }

    t.pass(name);
}

static bool run_tests_for_model(const std::string & path, const common_params & base_params, size_t & n_pass, size_t & n_fail, size_t & n_skip) {
    test_ctx t;
    t.path   = path;
    t.params = base_params;
    t.params.model.path = path;

    {
        llama_model_ptr model{load_model(path, t.params, LLAMA_SPLIT_MODE_LAYER)};
        if (!model) {
            LOG_ERR("%s: failed to load '%s'\n", __func__, path.c_str());
            return false;
        }

        const auto * vocab = llama_model_get_vocab(model.get());
        const auto n_vocab = llama_vocab_n_tokens(vocab);

        std::mt19937 rng(t.params.sampling.seed);
        std::uniform_int_distribution<llama_token> dist(0, n_vocab - 1);
        for (int i = 0; i < t.params.n_batch; i++) {
            t.prompt.push_back(dist(rng));
        }
    }

    test_switch    (t, LLAMA_SPLIT_MODE_LAYER,  LLAMA_SPLIT_MODE_TENSOR);
    test_switch    (t, LLAMA_SPLIT_MODE_TENSOR, LLAMA_SPLIT_MODE_LAYER);
    test_round_trip(t, LLAMA_SPLIT_MODE_LAYER,  LLAMA_SPLIT_MODE_TENSOR);
    test_round_trip(t, LLAMA_SPLIT_MODE_TENSOR, LLAMA_SPLIT_MODE_LAYER);
    test_no_change (t, LLAMA_SPLIT_MODE_LAYER,  LLAMA_SPLIT_MODE_TENSOR);
    test_tensor_split(t, LLAMA_SPLIT_MODE_LAYER, LLAMA_SPLIT_MODE_TENSOR);
    test_model_only(t, LLAMA_SPLIT_MODE_LAYER,  LLAMA_SPLIT_MODE_TENSOR);

    n_pass += t.n_pass;
    n_fail += t.n_fail;
    n_skip += t.n_skip;

    return t.n_fail == 0;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.prompt        = "";
    params.n_batch       = 32;
    params.n_ctx         = 256;
    params.n_predict     = 8;
    params.n_parallel    = 1;
    params.kv_unified    = true;
    params.fit_params    = false;
    params.sampling.seed = 1234;

    common_init();

    // extract our own --models DIR option before handing the rest to the common arg parser
    std::string models_dir;
    std::vector<char *> filtered_argv;
    filtered_argv.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--models") == 0) {
            if (i + 1 >= argc) {
                LOG_ERR("%s: --models requires a directory argument\n", __func__);
                return 1;
            }
            models_dir = argv[i + 1];
            i++;
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }
    filtered_argv.push_back(nullptr);
    const int fargc = (int) filtered_argv.size() - 1;

    if (!models_dir.empty()) {
        params.model.path = models_dir;
    }

    if (!common_params_parse(fargc, filtered_argv.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    if (!has_gpu_device()) {
        LOG_INF("%s: no GPU device found, a split mode switch has nothing to move - skipping\n", __func__);
        return 0;
    }

    std::vector<std::string> models;
    if (!models_dir.empty()) {
        if (!std::filesystem::exists(models_dir) || !std::filesystem::is_directory(models_dir)) {
            LOG_ERR("%s: models directory '%s' does not exist\n", __func__, models_dir.c_str());
            return 1;
        }
        for (const auto & entry : std::filesystem::directory_iterator(models_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
                models.push_back(entry.path().string());
            }
        }
        std::sort(models.begin(), models.end());

        if (models.empty()) {
            LOG_ERR("%s: no .gguf models found in '%s'\n", __func__, models_dir.c_str());
            return 1;
        }
    } else {
        models.push_back(params.model.path);
    }

    size_t n_pass = 0;
    size_t n_fail = 0;
    size_t n_skip = 0;
    size_t n_models_failed = 0;

    for (const auto & model_path : models) {
        LOG("\n================================================================\n");
        LOG_INF("%s: model %s\n", __func__, model_path.c_str());

        if (!run_tests_for_model(model_path, params, n_pass, n_fail, n_skip)) {
            n_models_failed++;
        }
    }

    LOG("\n================================================================\n");
    LOG_INF("%s: summary: %zu passed, %zu failed, %zu skipped over %zu models\n",
            __func__, n_pass, n_fail, n_skip, models.size());

    return n_models_failed == 0 ? 0 : 1;
}
