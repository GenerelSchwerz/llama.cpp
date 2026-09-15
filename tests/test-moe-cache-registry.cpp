#include "test-moe-cache.h"

#include "../src/llama-batch.h"
#include "../src/llama-context.h"
#include "../src/llama-model.h"
#include "../src/llama-vocab.h"

void test_moe_tensor_split_rejection() {
    auto params = llama_model_default_params();
    params.split_mode = LLAMA_SPLIT_MODE_TENSOR;
    params.moe_expert_cache_slots = 8;
    bool rejected = false;
    try {
        llama_model_free(llama_model_create(LLM_ARCH_DEEPSEEK4, params));
    } catch (const std::runtime_error & error) {
        rejected = std::string(error.what()).find("MoE expert caching does not support tensor split") != std::string::npos;
    }
    CHECK(rejected);
    params.moe_expert_cache_slots = 0;
    llama_model * model = llama_model_create(LLM_ARCH_DEEPSEEK4, params);
    CHECK(model != nullptr);
    llama_model_free(model);
    fprintf(stderr, "test-moe-cache: unsupported tensor/cache combination rejected before weights OK\n");
}

struct mtp_batch_fixture {
    std::vector<float> embd;
    std::vector<llama_pos> pos;
    std::vector<int32_t> n_seq_id;
    std::vector<llama_seq_id> seq_id_data;
    std::vector<llama_seq_id *> seq_id;
    std::vector<int8_t> output;

    void add(llama_seq_id sequence, llama_pos position) {
        embd.push_back((float) embd.size());
        pos.push_back(position);
        n_seq_id.push_back(1);
        seq_id_data.push_back(sequence);
        output.push_back(1);
    }

    void add_span(llama_seq_id sequence, llama_pos first, uint32_t n_rows) {
        for (uint32_t row = 0; row < n_rows; ++row) {
            add(sequence, first + (llama_pos) row);
        }
    }

    llama_batch batch() {
        seq_id.resize(seq_id_data.size());
        for (size_t row = 0; row < seq_id.size(); ++row) {
            seq_id[row] = &seq_id_data[row];
        }
        llama_batch result = {};
        result.n_tokens = (int32_t) pos.size();
        result.embd = embd.data();
        result.pos = pos.data();
        result.n_seq_id = n_seq_id.data();
        result.seq_id = seq_id.data();
        result.logits = output.data();
        return result;
    }
};

static void check_speculative_sequential_splits(
        mtp_batch_fixture & fixture,
        llama_context_type context_type,
        uint32_t n_ubatch,
        bool equal,
        uint32_t expected_ubatches) {
    llama_vocab vocab;
    llama_batch batch = fixture.batch();
    const uint32_t row_semantics = llama_speculative_grouped_intent_test_access::classify_batch(batch);
    CHECK(row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL);

    llama_batch_allocr balloc(1);
    CHECK(balloc.init(batch, vocab, nullptr, 1, 4, false));
    uint32_t n_rows = 0;
    uint32_t n_ubatches = 0;
    while (true) {
        llama_ubatch ubatch = equal ? balloc.split_equal(n_ubatch, true, 0) : balloc.split_simple(n_ubatch);
        if (ubatch.n_tokens == 0) {
            break;
        }
        CHECK(llama_speculative_grouped_intent_test_access::matches_ubatch(
            context_type, ubatch, row_semantics));
        n_rows += ubatch.n_tokens;
        ++n_ubatches;
    }
    CHECK(n_rows == (uint32_t) batch.n_tokens && n_ubatches == expected_ubatches);
}

void test_speculative_grouped_intent_splits() {
    mtp_batch_fixture catch_up;
    catch_up.add_span(0, 40, 3);
    catch_up.add_span(1, 70, 1);
    catch_up.add_span(2, 15, 2);
    for (llama_context_type context_type : {LLAMA_CONTEXT_TYPE_DRAFT, LLAMA_CONTEXT_TYPE_MTP}) {
        check_speculative_sequential_splits(catch_up, context_type, 4, false, 2);
        check_speculative_sequential_splits(catch_up, context_type, 4, true, 3);
    }

    mtp_batch_fixture chain_head;
    for (llama_seq_id sequence = 0; sequence < 4; ++sequence) {
        chain_head.add_span(sequence, 100 + 10 * sequence, 3);
    }
    for (llama_context_type context_type : {LLAMA_CONTEXT_TYPE_DRAFT, LLAMA_CONTEXT_TYPE_MTP}) {
        check_speculative_sequential_splits(chain_head, context_type, 5, false, 3);
        check_speculative_sequential_splits(chain_head, context_type, 8, true, 2);
    }

    mtp_batch_fixture interleaved;
    interleaved.add(0, 10);
    interleaved.add(1, 20);
    interleaved.add(0, 11);
    CHECK(llama_speculative_grouped_intent_test_access::classify_batch(interleaved.batch()) ==
        GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INVALID);

    mtp_batch_fixture discontinuous;
    discontinuous.add(0, 10);
    discontinuous.add(0, 12);
    CHECK(llama_speculative_grouped_intent_test_access::classify_batch(discontinuous.batch()) ==
        GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INVALID);
    fprintf(stderr, "test-moe-cache: speculative intent split lifecycle OK\n");
}

void test_speculative_required_grouped_backend_capability(int device) {
    ggml_backend_ptr cuda_backend(ggml_backend_cuda_init(device));
    ggml_backend_ptr cpu_backend(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    CHECK(cuda_backend != nullptr && cpu_backend != nullptr);
    const bool cuda_supported = llama_mtp_grouped_intent_test_access::backend_supported(cuda_backend.get());
    const bool cpu_supported = llama_mtp_grouped_intent_test_access::backend_supported(cpu_backend.get());
    CHECK(cuda_supported && !cpu_supported);
    CHECK(llama_mtp_grouped_intent_test_access::flags(12, cuda_supported) ==
        GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED);
    CHECK(llama_mtp_grouped_intent_test_access::flags(12, cpu_supported) ==
        GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_NONE);
    CHECK(llama_mtp_grouped_intent_test_access::flags(0, true) ==
        GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_NONE);

    std::array<llama_token, 2> implicit_tokens = {0, 0};
    llama_batch implicit_batch = llama_batch_get_one(implicit_tokens.data(), 2);
    CHECK(llama_mtp_grouped_intent_test_access::classify_batch(implicit_batch) ==
        GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INVALID);

    llama_vocab implicit_vocab;
    llama_vocab_test_access::add_dummy_token(implicit_vocab);
    llama_mtp_execution_policy implicit_sequential;
    CHECK(llama_mtp_grouped_intent_test_access::policy_after_batch_init(
        implicit_batch, implicit_vocab, 12, cuda_supported, implicit_sequential));
    CHECK(implicit_sequential.preserve_intent && !implicit_sequential.fail_closed &&
        implicit_sequential.row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL &&
        implicit_sequential.flags == GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED);

    implicit_batch.n_tokens = 1;
    llama_mtp_execution_policy implicit_independent;
    CHECK(llama_mtp_grouped_intent_test_access::policy_after_batch_init(
        implicit_batch, implicit_vocab, 12, cuda_supported, implicit_independent));
    CHECK(implicit_independent.preserve_intent && !implicit_independent.fail_closed &&
        implicit_independent.row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT &&
        implicit_independent.flags == GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED);

    implicit_batch.n_tokens = 2;
    llama_speculative_execution_policy draft_decoder;
    CHECK(llama_speculative_grouped_intent_test_access::policy_after_batch_init(
        LLAMA_CONTEXT_TYPE_DRAFT, implicit_batch, implicit_vocab, 12, cuda_supported, draft_decoder));
    CHECK(draft_decoder.domain == GGML_GRAPH_EXECUTION_DOMAIN_DRAFT &&
        draft_decoder.preserve_intent && !draft_decoder.fail_closed &&
        draft_decoder.row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL &&
        draft_decoder.flags == GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED);

    std::array<float, 3> encoder_embd = {};
    llama_batch encoder_batch = {};
    encoder_batch.n_tokens = (int32_t) encoder_embd.size();
    encoder_batch.embd = encoder_embd.data();
    llama_speculative_execution_policy draft_encoder;
    CHECK(llama_speculative_grouped_intent_test_access::policy_after_batch_init(
        LLAMA_CONTEXT_TYPE_DRAFT, encoder_batch, implicit_vocab, 12, cuda_supported, draft_encoder, true));
    CHECK(draft_encoder.domain == GGML_GRAPH_EXECUTION_DOMAIN_DRAFT &&
        draft_encoder.preserve_intent && !draft_encoder.fail_closed &&
        draft_encoder.row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL &&
        draft_encoder.flags == GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED);

    mtp_batch_fixture sequential;
    sequential.add_span(0, 10, 2);
    sequential.add_span(1, 20, 2);
    const auto legacy_sequential = llama_mtp_grouped_intent_test_access::policy(
        sequential.batch(), 12, cpu_supported);
    CHECK(legacy_sequential.preserve_intent && !legacy_sequential.fail_closed &&
        legacy_sequential.row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL &&
        legacy_sequential.flags == GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_NONE);
    const auto draft_legacy_sequential = llama_speculative_grouped_intent_test_access::policy(
        LLAMA_CONTEXT_TYPE_DRAFT, sequential.batch(), 12, cpu_supported);
    CHECK(draft_legacy_sequential.domain == GGML_GRAPH_EXECUTION_DOMAIN_DRAFT &&
        draft_legacy_sequential.preserve_intent && !draft_legacy_sequential.fail_closed &&
        draft_legacy_sequential.row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL &&
        draft_legacy_sequential.flags == GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_NONE);

    mtp_batch_fixture malformed;
    malformed.add(0, 10);
    malformed.add(1, 20);
    malformed.add(0, 11);
    const auto legacy_malformed = llama_mtp_grouped_intent_test_access::policy(
        malformed.batch(), 12, cpu_supported);
    CHECK(!legacy_malformed.preserve_intent && !legacy_malformed.fail_closed &&
        legacy_malformed.row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INVALID &&
        legacy_malformed.flags == GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_NONE);
    const auto required_malformed = llama_mtp_grouped_intent_test_access::policy(
        malformed.batch(), 12, cuda_supported);
    CHECK(!required_malformed.preserve_intent && required_malformed.fail_closed &&
        required_malformed.row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INVALID &&
        required_malformed.flags == GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED);
    const auto draft_legacy_malformed = llama_speculative_grouped_intent_test_access::policy(
        LLAMA_CONTEXT_TYPE_DRAFT, malformed.batch(), 12, cpu_supported);
    CHECK(draft_legacy_malformed.domain == GGML_GRAPH_EXECUTION_DOMAIN_DRAFT &&
        !draft_legacy_malformed.preserve_intent && !draft_legacy_malformed.fail_closed &&
        draft_legacy_malformed.flags == GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_NONE);
    const auto draft_required_malformed = llama_speculative_grouped_intent_test_access::policy(
        LLAMA_CONTEXT_TYPE_DRAFT, malformed.batch(), 12, cuda_supported);
    CHECK(draft_required_malformed.domain == GGML_GRAPH_EXECUTION_DOMAIN_DRAFT &&
        !draft_required_malformed.preserve_intent && draft_required_malformed.fail_closed &&
        draft_required_malformed.flags == GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED);
    fprintf(stderr, "test-moe-cache: speculative required-grouped backend capability OK\n");
}

struct scheduler_certificate_probe {
    struct record {
        ggml_graph_execution_certificate certificate = {};
        uint64_t graph_uid = 0;
        uint32_t backend_index = UINT32_MAX;
    };

    std::array<ggml_backend_t, 2> backends = {};
    std::array<enum ggml_status (*)(ggml_backend_t, ggml_cgraph *), 2> delegates = {};
    std::array<void (*)(ggml_backend_t), 2> synchronize_delegates = {};
    std::array<uint32_t, 2> synchronize_calls = {};
    std::array<record, 24> records = {};
    uint32_t n_backends = 0;
    uint32_t calls = 0;
    uint32_t fail_backend = UINT32_MAX;
};

static scheduler_certificate_probe * scheduler_certificate_probe_current = nullptr;

static enum ggml_status scheduler_certificate_graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    CHECK(scheduler_certificate_probe_current != nullptr);
    auto & probe = *scheduler_certificate_probe_current;
    CHECK(probe.calls < probe.records.size());
    uint32_t backend_index = 0;
    while (backend_index < probe.n_backends && probe.backends[backend_index] != backend) {
        ++backend_index;
    }
    CHECK(backend_index < probe.n_backends && probe.delegates[backend_index] != nullptr);
    probe.records[probe.calls].certificate = graph->execution_certificate;
    probe.records[probe.calls].graph_uid = graph->uid;
    probe.records[probe.calls].backend_index = backend_index;
    ++probe.calls;
    if (probe.fail_backend == backend_index) {
        return GGML_STATUS_FAILED;
    }
    return probe.delegates[backend_index](backend, graph);
}

static void scheduler_certificate_synchronize(ggml_backend_t backend) {
    CHECK(scheduler_certificate_probe_current != nullptr);
    auto & probe = *scheduler_certificate_probe_current;
    uint32_t backend_index = 0;
    while (backend_index < probe.n_backends && probe.backends[backend_index] != backend) {
        ++backend_index;
    }
    CHECK(backend_index < probe.n_backends);
    ++probe.synchronize_calls[backend_index];
    if (probe.synchronize_delegates[backend_index] != nullptr) {
        probe.synchronize_delegates[backend_index](backend);
    }
}

static bool scheduler_certificate_eval_callback(ggml_tensor *, bool, void *) {
    return false;
}

static const ggml_backend_moe_candidate_tensor_v2 * candidate_tensor(
        const ggml_backend_moe_candidate_snapshot_v2 & snapshot,
        const ggml_tensor * tensor) {
    for (uint32_t i = 0; i < snapshot.n_tensors; ++i) {
        if (snapshot.tensors[i].tensor == tensor) {
            return &snapshot.tensors[i];
        }
    }
    return nullptr;
}

void test_candidate_graph_coverage_ledger() {
    candidate_test_fixture fixture;
    const int64_t gate_up_ne[] = {64, 64, 4};
    const int64_t down_ne[] = {32, 64, 4};
    ggml_tensor * gate_up = fixture.cached_tensor(GGML_TYPE_Q4_0, 3, gate_up_ne);
    ggml_tensor * down = fixture.cached_tensor(GGML_TYPE_Q4_0, 3, down_ne);
    ggml_tensor * unknown = fixture.cached_tensor(GGML_TYPE_Q4_0, 3, gate_up_ne);
    ggml_tensor * ordinary = fixture.tensor(GGML_TYPE_Q4_0, 3, gate_up_ne);
    std::array<ggml_backend_moe_candidate_bank_v1, 2> banks = {{
        {gate_up, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, 0},
        {down, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
    }};
    ggml_backend_moe_candidate_group_v1 group = {
        banks.data(), banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 0, 0,
    };
    const auto snapshot = candidate_snapshot(12, &group, 1);
    ggml_cuda_moe_grouped_context registry(&fixture.owner, 0);
    CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);

    const candidate_route route = candidate_top_k_route(fixture, 4, 2);
    ggml_tensor * gate_up_reader = candidate_mmid(fixture, gate_up, route.ids);
    ggml_tensor * down_reader = candidate_mmid(fixture, down, route.ids);
    ggml_tensor * unknown_reader = candidate_mmid(fixture, unknown, route.ids);
    ggml_tensor * ordinary_reader = candidate_mmid(fixture, ordinary, route.ids);
    ggml_cgraph * graph = candidate_graph(fixture, {
        route.root, route.ids, gate_up_reader, down_reader, unknown_reader, ordinary_reader,
    });

    ggml_cuda_moe_graph_plan plan;
    ggml_cuda_moe_graph_execution execution;
    registry.compile_graph_plan(graph, 901, &plan, &execution);
    const auto & diagnostics = plan.coverage_diagnostics();
    CHECK(diagnostics.cached_mmid == 3);
    CHECK(diagnostics.counts[GGML_CUDA_MOE_GRAPH_COVERAGE_REGISTERED] == 2);
    CHECK(diagnostics.counts[GGML_CUDA_MOE_GRAPH_COVERAGE_REVERSE_MAP_MISS] == 1);
    CHECK(diagnostics.counts[GGML_CUDA_MOE_GRAPH_COVERAGE_SOURCE_CHANGED] == 0);
    CHECK(diagnostics.counts[GGML_CUDA_MOE_GRAPH_COVERAGE_INVALID_REVERSE_MAP] == 0);
    CHECK(diagnostics.first_source[GGML_CUDA_MOE_GRAPH_COVERAGE_REGISTERED] == gate_up);
    CHECK(diagnostics.first_node_index[GGML_CUDA_MOE_GRAPH_COVERAGE_REGISTERED] == 2);
    CHECK(diagnostics.first_group_index[GGML_CUDA_MOE_GRAPH_COVERAGE_REGISTERED] == 0);
    CHECK(diagnostics.first_bank_index[GGML_CUDA_MOE_GRAPH_COVERAGE_REGISTERED] == 0);
    CHECK(diagnostics.first_source[GGML_CUDA_MOE_GRAPH_COVERAGE_REVERSE_MAP_MISS] == unknown);
    CHECK(diagnostics.first_node_index[GGML_CUDA_MOE_GRAPH_COVERAGE_REVERSE_MAP_MISS] == 4);
    CHECK(execution.size() == 1 && !execution.find(down_reader, nullptr));
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR);

    const int64_t saved_ne0 = down->ne[0];
    down->ne[0]--;
    registry.compile_graph_plan(graph, 902, &plan, &execution);
    CHECK(plan.coverage_diagnostics().cached_mmid == 3);
    CHECK(plan.coverage_diagnostics().counts[GGML_CUDA_MOE_GRAPH_COVERAGE_REGISTERED] == 1);
    CHECK(plan.coverage_diagnostics().counts[GGML_CUDA_MOE_GRAPH_COVERAGE_REVERSE_MAP_MISS] == 1);
    CHECK(plan.coverage_diagnostics().counts[GGML_CUDA_MOE_GRAPH_COVERAGE_SOURCE_CHANGED] == 1);
    CHECK(plan.coverage_diagnostics().first_source[GGML_CUDA_MOE_GRAPH_COVERAGE_SOURCE_CHANGED] == down);
    down->ne[0] = saved_ne0;

    ggml_cgraph view = ggml_graph_view(graph, 4, 6);
    registry.compile_graph_plan(&view, 903, &plan, &execution);
    CHECK(plan.coverage_diagnostics().cached_mmid == 1);
    CHECK(plan.coverage_diagnostics().counts[GGML_CUDA_MOE_GRAPH_COVERAGE_REVERSE_MAP_MISS] == 1);
    CHECK(plan.coverage_diagnostics().counts[GGML_CUDA_MOE_GRAPH_COVERAGE_REGISTERED] == 0);

    const auto disabled = candidate_snapshot(12, nullptr, 0);
    CHECK(registry.replace(&disabled) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    registry.compile_graph_plan(graph, 904, &plan, &execution);
    CHECK(plan.coverage_diagnostics().cached_mmid == 3);
    CHECK(plan.coverage_diagnostics().counts[GGML_CUDA_MOE_GRAPH_COVERAGE_REVERSE_MAP_MISS] == 3);
    CHECK(execution.size() == 0);
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_PREFILL_LEGACY);
    CHECK(!ggml_cuda_moe_grouped_context_test_access::has_device_resource(registry, {0, 0}));

    const int64_t ungated_up_ne[] = {64, 32, 4};
    ggml_tensor * ungated_up = fixture.cached_tensor(GGML_TYPE_Q4_0, 3, ungated_up_ne);
    ggml_tensor * ungated_down = fixture.cached_tensor(GGML_TYPE_Q4_0, 3, down_ne);
    ggml_tensor * chunk_gate_up = fixture.cached_tensor(GGML_TYPE_Q4_0, 3, gate_up_ne);
    ggml_tensor * chunk_down = fixture.cached_tensor(GGML_TYPE_Q4_0, 3, down_ne);
    ggml_tensor * lora_gate_up = fixture.cached_tensor(GGML_TYPE_Q4_0, 3, gate_up_ne);
    ggml_tensor * lora_down = fixture.cached_tensor(GGML_TYPE_Q4_0, 3, down_ne);
    ggml_tensor * override_gate_up = fixture.cached_tensor(GGML_TYPE_Q4_0, 3, gate_up_ne);
    ggml_tensor * override_down = fixture.cached_tensor(GGML_TYPE_Q4_0, 3, down_ne);
    ggml_tensor * incomplete_gate_up = fixture.cached_tensor(GGML_TYPE_Q4_0, 3, gate_up_ne);
    ggml_tensor * incomplete_down = fixture.cached_tensor(GGML_TYPE_Q4_0, 3, down_ne);
    ggml_tensor * unsupported_gate_up = fixture.cached_tensor(GGML_TYPE_I8, 3, gate_up_ne);
    ggml_tensor * unsupported_down = fixture.cached_tensor(GGML_TYPE_I8, 3, down_ne);
    ggml_tensor * excluded_cached = fixture.cached_tensor(GGML_TYPE_Q4_0, 3, gate_up_ne);
    ggml_tensor * opaque_cached = fixture.cached_tensor(GGML_TYPE_Q4_0, 3, gate_up_ne);
    ggml_tensor * missing_cached = fixture.cached_tensor(GGML_TYPE_Q4_0, 3, gate_up_ne);
    std::array<ggml_backend_moe_candidate_group_v2, 7> v2_groups = {{
        {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY, 0, 0},
        {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_UNGATED, GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY, 0, 0},
        {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_CHUNK, 0, 0},
        {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY,
            GGML_BACKEND_MOE_CANDIDATE_GROUP_V2_FLAG_ACTIVE_LORA, 0},
        {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY,
            GGML_BACKEND_MOE_CANDIDATE_GROUP_V2_FLAG_TENSOR_OVERRIDES, 0},
        {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY,
            GGML_BACKEND_MOE_CANDIDATE_GROUP_V2_FLAG_INCOMPLETE, 0},
        {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY, 0, 0},
    }};
    constexpr uint32_t cached = GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_CACHED_BUFFER;
    std::array<ggml_backend_moe_candidate_tensor_v2, 16> v2_tensors = {{
        {gate_up, 0, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE, cached, 0},
        {down, 0, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE, cached, 0},
        {ungated_up, 1, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE, cached, 0},
        {ungated_down, 1, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE, cached, 0},
        {chunk_gate_up, 2, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE, cached, 0},
        {chunk_down, 2, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE, cached, 0},
        {lora_gate_up, 3, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE,
            cached | GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_ACTIVE_LORA, 0},
        {lora_down, 3, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE, cached, 0},
        {override_gate_up, 4, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE,
            cached | GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_TENSOR_OVERRIDES, 0},
        {override_down, 4, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE, cached, 0},
        {incomplete_gate_up, 5, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE, cached, 0},
        {incomplete_down, 5, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE, cached, 0},
        {unsupported_gate_up, 6, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE, cached, 0},
        {unsupported_down, 6, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE, cached, 0},
        {excluded_cached, UINT32_MAX, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID,
            GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_EXCLUDED_SHARED, cached, 0},
        {opaque_cached, UINT32_MAX, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID,
            GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_UNCLASSIFIED, cached, 0},
    }};
    const auto v2_snapshot = candidate_snapshot_v2(12, v2_groups.data(), v2_groups.size(), v2_tensors.data(), v2_tensors.size());
    CHECK(registry.replace(&v2_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(registry.state().n_groups == 2 && registry.state().n_weights == 4);

    const candidate_route v2_route = candidate_top_k_route(fixture, 4, 2);
    std::array<ggml_tensor *, 12> v2_readers = {{
        candidate_mmid(fixture, gate_up, v2_route.ids),
        candidate_mmid(fixture, down, v2_route.ids),
        candidate_mmid(fixture, ungated_up, v2_route.ids),
        candidate_mmid(fixture, ungated_down, v2_route.ids),
        candidate_mmid(fixture, chunk_gate_up, v2_route.ids),
        candidate_mmid(fixture, lora_gate_up, v2_route.ids),
        candidate_mmid(fixture, override_gate_up, v2_route.ids),
        candidate_mmid(fixture, incomplete_gate_up, v2_route.ids),
        candidate_mmid(fixture, unsupported_gate_up, v2_route.ids),
        candidate_mmid(fixture, excluded_cached, v2_route.ids),
        candidate_mmid(fixture, opaque_cached, v2_route.ids),
        candidate_mmid(fixture, missing_cached, v2_route.ids),
    }};
    ggml_cgraph * v2_graph = ggml_new_graph_custom(fixture.ctx, 32, false);
    ggml_graph_add_node(v2_graph, v2_route.root);
    ggml_graph_add_node(v2_graph, v2_route.ids);
    for (ggml_tensor * reader : v2_readers) {
        ggml_graph_add_node(v2_graph, reader);
    }
    candidate_rebuild_graph_uses(v2_graph);
    registry.compile_graph_plan(v2_graph, 905, &plan, &execution);
    const auto & v2_diagnostics = plan.coverage_diagnostics();
    CHECK(v2_diagnostics.manifest_version == GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_VERSION);
    CHECK(v2_diagnostics.cached_mmid == v2_readers.size());
    CHECK(v2_diagnostics.counts[GGML_CUDA_MOE_GRAPH_COVERAGE_REGISTERED] == 4);
    CHECK(v2_diagnostics.counts[GGML_CUDA_MOE_GRAPH_COVERAGE_DORMANT_LAYOUT] == 1);
    CHECK(v2_diagnostics.counts[GGML_CUDA_MOE_GRAPH_COVERAGE_ACTIVE_LORA] == 1);
    CHECK(v2_diagnostics.counts[GGML_CUDA_MOE_GRAPH_COVERAGE_TENSOR_OVERRIDE] == 1);
    CHECK(v2_diagnostics.counts[GGML_CUDA_MOE_GRAPH_COVERAGE_INCOMPLETE] == 1);
    CHECK(v2_diagnostics.counts[GGML_CUDA_MOE_GRAPH_COVERAGE_UNSUPPORTED_DESCRIPTOR] == 1);
    CHECK(v2_diagnostics.counts[GGML_CUDA_MOE_GRAPH_COVERAGE_EXCLUDED] == 1);
    CHECK(v2_diagnostics.counts[GGML_CUDA_MOE_GRAPH_COVERAGE_UNCLASSIFIED] == 1);
    CHECK(v2_diagnostics.counts[GGML_CUDA_MOE_GRAPH_COVERAGE_REVERSE_MAP_MISS] == 1);
    CHECK(v2_diagnostics.first_domain[GGML_CUDA_MOE_GRAPH_COVERAGE_DORMANT_LAYOUT] == GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_CHUNK);
    CHECK(v2_diagnostics.first_rejection[GGML_CUDA_MOE_GRAPH_COVERAGE_UNSUPPORTED_DESCRIPTOR] ==
        GGML_CUDA_MOE_CANDIDATE_REJECT_UNSUPPORTED_TYPE);
    CHECK(execution.size() == 2 && !execution.find(v2_readers[1], nullptr));
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR);

    auto incomplete_snapshot = v2_snapshot;
    incomplete_snapshot.flags = GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_FLAG_INCOMPLETE;
    CHECK(registry.replace(&incomplete_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(registry.state().n_groups == 0);
    registry.compile_graph_plan(v2_graph, 906, &plan, &execution);
    const auto & incomplete_diagnostics = plan.coverage_diagnostics();
    CHECK(incomplete_diagnostics.cached_mmid == v2_readers.size());
    CHECK(incomplete_diagnostics.counts[GGML_CUDA_MOE_GRAPH_COVERAGE_REGISTERED] == 0);
    CHECK(incomplete_diagnostics.counts[GGML_CUDA_MOE_GRAPH_COVERAGE_INCOMPLETE] == v2_readers.size() - 3);
    CHECK(incomplete_diagnostics.counts[GGML_CUDA_MOE_GRAPH_COVERAGE_ACTIVE_LORA] == 1);
    CHECK(incomplete_diagnostics.counts[GGML_CUDA_MOE_GRAPH_COVERAGE_TENSOR_OVERRIDE] == 1);
    CHECK(incomplete_diagnostics.counts[GGML_CUDA_MOE_GRAPH_COVERAGE_DORMANT_LAYOUT] == 0);
    CHECK(incomplete_diagnostics.counts[GGML_CUDA_MOE_GRAPH_COVERAGE_REVERSE_MAP_MISS] == 1);
    CHECK(incomplete_diagnostics.first_source[GGML_CUDA_MOE_GRAPH_COVERAGE_INCOMPLETE] == gate_up);
    CHECK(execution.size() == 0);
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR);
    CHECK(execution.requires_dispatch());

    std::array<ggml_backend_moe_candidate_group_v2, 2> dormant_groups = {{
        {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_UNGATED, GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY, 0, 0},
        {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_CHUNK, 0, 0},
    }};
    std::array<ggml_backend_moe_candidate_tensor_v2, 4> dormant_tensors = {{
        {ungated_up, 0, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE, cached, 0},
        {ungated_down, 0, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE, cached, 0},
        {chunk_gate_up, 1, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE, cached, 0},
        {chunk_down, 1, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE, cached, 0},
    }};
    const auto dormant_snapshot = candidate_snapshot_v2(
        12, dormant_groups.data(), dormant_groups.size(), dormant_tensors.data(), dormant_tensors.size());
    CHECK(registry.replace(&dormant_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(registry.state().n_groups == 1 && registry.state().n_weights == 2);
    std::array<ggml_tensor *, 4> dormant_readers = {{
        candidate_mmid(fixture, ungated_up, v2_route.ids),
        candidate_mmid(fixture, ungated_down, v2_route.ids),
        candidate_mmid(fixture, chunk_gate_up, v2_route.ids),
        candidate_mmid(fixture, chunk_down, v2_route.ids),
    }};
    ggml_cgraph * dormant_graph = candidate_graph(fixture, {
        v2_route.root, v2_route.ids,
        dormant_readers[0], dormant_readers[1], dormant_readers[2], dormant_readers[3],
    });
    const auto dormant_coverage = candidate_certify_graph(registry, dormant_graph);
    registry.compile_graph_plan(
        dormant_graph, 907, &plan, &execution, dormant_coverage.epoch, dormant_coverage.nodes,
        dormant_coverage.mmid_count, dormant_coverage.mmid_fingerprint);
    CHECK(plan.coverage_diagnostics().cached_mmid == dormant_readers.size());
    CHECK(plan.coverage_diagnostics().counts[GGML_CUDA_MOE_GRAPH_COVERAGE_REGISTERED] == 2);
    CHECK(plan.coverage_diagnostics().counts[GGML_CUDA_MOE_GRAPH_COVERAGE_DORMANT_LAYOUT] == 2);
    CHECK(execution.size() == 1 && execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR);
    CHECK(execution.requires_dispatch() && execution.rejects_cached_mmid(dormant_readers[2]));
    CHECK(!registry.begin_graph_dispatch(&execution, GGML_CUDA_MOE_GRAPH_DISPATCH_DIRECT));
    ggml_cuda_moe_graph_execution dormant_reused;
    CHECK(!registry.bind_graph_plan(
        dormant_graph, 907, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, plan, &dormant_reused,
        dormant_coverage.epoch, dormant_coverage.nodes,
        dormant_coverage.mmid_count, dormant_coverage.mmid_fingerprint));
    CHECK(dormant_reused.size() == 0);

    std::array<ggml_backend_moe_candidate_group_v2, 3> mixed_dormant_groups = {{
        {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY, 0, 0},
        dormant_groups[0],
        dormant_groups[1],
    }};
    std::array<ggml_backend_moe_candidate_tensor_v2, 6> mixed_dormant_tensors = {{
        {gate_up, 0, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE, cached, 0},
        {down, 0, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE, cached, 0},
        {ungated_up, 1, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE, cached, 0},
        {ungated_down, 1, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE, cached, 0},
        {chunk_gate_up, 2, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE, cached, 0},
        {chunk_down, 2, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE, cached, 0},
    }};
    const auto mixed_dormant_snapshot = candidate_snapshot_v2(
        12, mixed_dormant_groups.data(), mixed_dormant_groups.size(), mixed_dormant_tensors.data(), mixed_dormant_tensors.size());
    CHECK(registry.replace(&mixed_dormant_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(registry.state().n_groups == 2 && registry.state().n_weights == 4);
    ggml_tensor * mixed_active_gate_up = candidate_mmid(fixture, gate_up, v2_route.ids);
    ggml_tensor * mixed_active_down = candidate_mmid(fixture, down, v2_route.ids);
    ggml_cgraph * mixed_dormant_graph = candidate_graph(fixture, {
        v2_route.root, v2_route.ids, mixed_active_gate_up, mixed_active_down,
        dormant_readers[0], dormant_readers[1], dormant_readers[2], dormant_readers[3],
    });
    const auto mixed_dormant_coverage = candidate_certify_graph(registry, mixed_dormant_graph);
    registry.compile_graph_plan(
        mixed_dormant_graph, 908, &plan, &execution, mixed_dormant_coverage.epoch, mixed_dormant_coverage.nodes,
        mixed_dormant_coverage.mmid_count, mixed_dormant_coverage.mmid_fingerprint);
    CHECK(execution.size() == 2 && execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR);
    CHECK(!execution.find(mixed_active_gate_up, nullptr) && !execution.find(mixed_active_down, nullptr));
    CHECK(execution.rejects_cached_mmid(dormant_readers[2]));
    CHECK(!registry.begin_graph_dispatch(&execution, GGML_CUDA_MOE_GRAPH_DISPATCH_DIRECT));
    fprintf(stderr, "test-moe-cache: inactive cached MMID coverage ledger OK\n");
}

void test_candidate_graph_inventory_reuse() {
    candidate_test_fixture fixture;
    const int64_t gate_up_ne[] = {64, 64, 4};
    const int64_t down_ne[] = {32, 64, 4};
    const int64_t second_gate_up_ne[] = {256, 512, 4};
    const int64_t second_down_ne[] = {256, 256, 4};
    std::array<ggml_tensor *, 4> weights = {{
        fixture.cached_tensor(GGML_TYPE_Q4_0, 3, gate_up_ne),
        fixture.cached_tensor(GGML_TYPE_Q4_0, 3, down_ne),
        fixture.cached_tensor(GGML_TYPE_Q4_K, 3, second_gate_up_ne),
        fixture.cached_tensor(GGML_TYPE_Q4_K, 3, second_down_ne),
    }};
    std::array<ggml_backend_moe_candidate_bank_v1, 2> first_banks = {{
        {weights[0], GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, 0},
        {weights[1], GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
    }};
    std::array<ggml_backend_moe_candidate_bank_v1, 2> second_banks = {{
        {weights[2], GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, 0},
        {weights[3], GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
    }};
    std::array<ggml_backend_moe_candidate_group_v1, 2> groups = {{
        {first_banks.data(), first_banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 0, 0},
        {second_banks.data(), second_banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 0, 0},
    }};
    const auto snapshot = candidate_snapshot(12, groups.data(), groups.size());
    ggml_cuda_moe_grouped_context registry(&fixture.owner, 0);
    CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);

    const candidate_route first_route = candidate_top_k_route(fixture, 4, 2);
    const candidate_route second_route = candidate_top_k_route(fixture, 4, 2);
    ggml_tensor * first_gate_up = candidate_mmid(fixture, weights[0], first_route.ids);
    ggml_tensor * first_down = candidate_mmid(fixture, weights[1], first_route.ids);
    ggml_tensor * second_gate_up = candidate_mmid(fixture, weights[2], second_route.ids);
    ggml_tensor * second_down = candidate_mmid(fixture, weights[3], second_route.ids);
    ggml_cgraph * graph = candidate_graph(fixture, {
        first_route.root, first_route.ids, first_gate_up, first_down,
        first_route.source, first_route.source, first_route.source, first_route.source,
    });
    const auto coverage = candidate_certify_graph(registry, graph);

    std::shared_ptr<ggml_cuda_moe_graph_plan> plan;
    ggml_cuda_moe_graph_execution execution;
    CHECK(registry.prepare_graph_execution(
        graph, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &plan, &execution,
        coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
        GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
    CHECK(execution.size() == 1 && execution.find(first_down, nullptr));
    const std::shared_ptr<ggml_cuda_moe_graph_plan> stale_plan = plan;

    graph->nodes[0] = second_route.root;
    graph->nodes[1] = second_route.ids;
    graph->nodes[2] = second_gate_up;
    graph->nodes[3] = second_down;
    candidate_rebuild_graph_uses(graph);
    CHECK(registry.prepare_graph_execution(
        graph, 1, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &plan, &execution,
        coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
        GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
    CHECK(execution.size() == 1 && execution.find(second_down, nullptr));
    const std::shared_ptr<ggml_cuda_moe_graph_plan> uncertified_plan = plan;
    CHECK(registry.prepare_graph_execution(
        graph, 2, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &plan, &execution,
        coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
        GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
    CHECK(plan != uncertified_plan && execution.find(second_down, nullptr));
    const auto replacement_coverage = candidate_certify_graph(registry, graph);
    CHECK(replacement_coverage.mmid_count == coverage.mmid_count &&
        replacement_coverage.mmid_fingerprint != coverage.mmid_fingerprint);
    CHECK(registry.prepare_graph_execution(
        graph, 3, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &plan, &execution,
        replacement_coverage.epoch, replacement_coverage.nodes,
        replacement_coverage.mmid_count, replacement_coverage.mmid_fingerprint) ==
        GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
    const std::shared_ptr<ggml_cuda_moe_graph_plan> certified_plan = plan;
    CHECK(registry.prepare_graph_execution(
        graph, 4, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &plan, &execution,
        replacement_coverage.epoch, replacement_coverage.nodes,
        replacement_coverage.mmid_count, replacement_coverage.mmid_fingerprint) ==
        GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
    CHECK(plan == certified_plan && execution.find(second_down, nullptr));

    graph->nodes[0] = first_route.root;
    graph->nodes[1] = first_route.ids;
    graph->nodes[2] = first_gate_up;
    graph->nodes[3] = first_down;

    graph->nodes[4] = second_route.root;
    graph->nodes[5] = second_route.ids;
    graph->nodes[6] = second_gate_up;
    graph->nodes[7] = second_down;
    candidate_rebuild_graph_uses(graph);
    const auto updated_coverage = candidate_certify_graph(registry, graph);
    CHECK(!registry.bind_graph_plan(
        graph, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, *stale_plan, &execution,
        updated_coverage.epoch, updated_coverage.nodes,
        updated_coverage.mmid_count, updated_coverage.mmid_fingerprint));
    CHECK(registry.prepare_graph_execution(
        graph, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &plan, &execution,
        updated_coverage.epoch, updated_coverage.nodes,
        updated_coverage.mmid_count, updated_coverage.mmid_fingerprint) ==
        GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
    CHECK(plan != stale_plan && execution.size() == 2 && execution.find(first_down, nullptr) && execution.find(second_down, nullptr));
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED);
    fprintf(stderr, "test-moe-cache: complete cached MMID inventory reuse OK\n");
}

static ggml_cuda_mmid_capability_query candidate_mmid_query(
        ggml_type type,
        int64_t n_tokens = 1,
        ggml_cuda_mmid_mapping mapping = GGML_CUDA_MMID_MAPPING_DIRECT,
        bool use_mmq = false,
        size_t smpbo = 64 * 1024) {
    ggml_cuda_mmid_capability_query query;
    query.source_type = type;
    query.input_type = GGML_TYPE_F32;
    query.output_type = GGML_TYPE_F32;
    query.source_ne[0] = 256;
    query.source_ne[1] = 128;
    query.source_ne[2] = 64;
    query.source_ne[3] = 1;
    query.source_nb[0] = ggml_type_size(type);
    query.source_nb[1] = query.source_nb[0] * query.source_ne[0] / ggml_blck_size(type);
    query.source_nb[2] = query.source_nb[1] * query.source_ne[1];
    query.source_nb[3] = query.source_nb[2] * query.source_ne[2];
    query.n_tokens = n_tokens;
    query.n_experts = query.source_ne[2];
    query.cc = 800;
    query.warp_size = 32;
    query.smpbo = smpbo;
    query.phase = n_tokens == 1 ? GGML_CUDA_MMID_PHASE_DECODE : GGML_CUDA_MMID_PHASE_PREFILL;
    query.mapping = mapping;
    query.use_mmq = use_mmq;
    return query;
}

void test_mmid_capabilities() {
    constexpr std::array<ggml_type, 27> advertised = {
        GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16,
        GGML_TYPE_Q1_0, GGML_TYPE_Q2_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0,
        GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_K,
        GGML_TYPE_IQ1_M, GGML_TYPE_IQ1_S, GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_XXS,
        GGML_TYPE_IQ3_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS,
        GGML_TYPE_MXFP4, GGML_TYPE_NVFP4,
    };
    uint32_t n_advertised = 0;
    uint32_t n_mmvq = 0;
    uint32_t n_mmq = 0;
    uint32_t n_mapped_mmq = 0;
    uint32_t n_scalar = 0;
    uint32_t n_generic = 0;
    for (int value = 0; value < GGML_TYPE_COUNT; ++value) {
        const auto type = static_cast<ggml_type>(value);
        const bool expected = std::find(advertised.begin(), advertised.end(), type) != advertised.end();
        CHECK(((ggml_cuda_mmid_source_capability_for(type).flags & GGML_CUDA_MMID_SOURCE_ADVERTISED) != 0) == expected);
    }
    for (ggml_type type : advertised) {
        const auto source = ggml_cuda_mmid_source_capability_for(type);
        CHECK(source.type == type);
        n_advertised += (source.flags & GGML_CUDA_MMID_SOURCE_ADVERTISED) != 0;
        n_mmvq += (source.flags & GGML_CUDA_MMID_SOURCE_MMVQ) != 0;
        n_mmq += (source.flags & GGML_CUDA_MMID_SOURCE_MMQ) != 0;
        n_mapped_mmq += (source.flags & GGML_CUDA_MMID_SOURCE_MAPPED_MMQ) != 0;
        n_scalar += (source.flags & GGML_CUDA_MMID_SOURCE_SCALAR) != 0;
        n_generic += (source.flags & GGML_CUDA_MMID_SOURCE_GENERIC) != 0;

        const auto capability = ggml_cuda_mmid_get_capability(candidate_mmid_query(type));
        if (type == GGML_TYPE_Q8_K) {
            CHECK(capability.selection == GGML_CUDA_MMID_CONSUMER_UNSUPPORTED);
            CHECK(capability.reason == GGML_CUDA_MMID_CAPABILITY_UNSUPPORTED_CONSUMER);
        } else if ((source.flags & GGML_CUDA_MMID_SOURCE_SCALAR) != 0) {
            CHECK(capability.selection == GGML_CUDA_MMID_CONSUMER_MMF || capability.selection == GGML_CUDA_MMID_CONSUMER_GENERIC);
            CHECK(capability.reason == GGML_CUDA_MMID_CAPABILITY_OK);
        } else {
            CHECK(capability.selection == GGML_CUDA_MMID_CONSUMER_MMVQ);
            CHECK(capability.reason == GGML_CUDA_MMID_CAPABILITY_OK);
        }
    }
    CHECK(n_advertised == 27 && n_mmvq == 23 && n_mmq == 22 && n_mapped_mmq == 20 && n_scalar == 3 && n_generic == 26);
    CHECK(ggml_cuda_mmid_source_capability_for(GGML_TYPE_Q8_1).flags == 0);
    CHECK(ggml_cuda_mmid_source_capability_for(GGML_TYPE_COUNT).flags == 0);

    auto query = candidate_mmid_query(GGML_TYPE_Q4_K, 16, GGML_CUDA_MMID_MAPPING_DIRECT, true);
    const auto direct = ggml_cuda_mmid_get_capability(query);
    CHECK((direct.selection == GGML_CUDA_MMID_CONSUMER_MMQ || direct.selection == GGML_CUDA_MMID_CONSUMER_GENERIC) &&
        direct.reason == GGML_CUDA_MMID_CAPABILITY_OK);
    query.mapping = GGML_CUDA_MMID_MAPPING_SOURCE_MAP;
    auto capability = ggml_cuda_mmid_get_capability(query);
    CHECK(capability.selection == direct.selection && capability.reason == GGML_CUDA_MMID_CAPABILITY_OK);
    for (ggml_type type : {GGML_TYPE_MXFP4, GGML_TYPE_NVFP4}) {
        query = candidate_mmid_query(type, 16, GGML_CUDA_MMID_MAPPING_DIRECT, true);
        capability = ggml_cuda_mmid_get_capability(query);
        CHECK((capability.selection == GGML_CUDA_MMID_CONSUMER_MMQ || capability.selection == GGML_CUDA_MMID_CONSUMER_GENERIC) &&
            capability.reason == GGML_CUDA_MMID_CAPABILITY_OK);
        query.mapping = GGML_CUDA_MMID_MAPPING_SOURCE_MAP;
        capability = ggml_cuda_mmid_get_capability(query);
        CHECK(capability.selection == GGML_CUDA_MMID_CONSUMER_GENERIC && capability.reason == GGML_CUDA_MMID_CAPABILITY_OK);
        query.use_mmq = false;
        capability = ggml_cuda_mmid_get_capability(query);
        CHECK(capability.selection == GGML_CUDA_MMID_CONSUMER_GENERIC && capability.reason == GGML_CUDA_MMID_CAPABILITY_OK);
    }
    query = candidate_mmid_query(GGML_TYPE_IQ1_M, 16, GGML_CUDA_MMID_MAPPING_DIRECT, true);
    CHECK(ggml_cuda_mmid_get_capability(query).selection == GGML_CUDA_MMID_CONSUMER_GENERIC);
    query = candidate_mmid_query(GGML_TYPE_Q4_K, 16, GGML_CUDA_MMID_MAPPING_DIRECT, true, 32 * 1024);
    CHECK(ggml_cuda_mmid_get_capability(query).selection == GGML_CUDA_MMID_CONSUMER_GENERIC);

    query = candidate_mmid_query(GGML_TYPE_Q4_K, 2);
    query.phase = GGML_CUDA_MMID_PHASE_DECODE;
    CHECK(ggml_cuda_mmid_get_capability(query).reason == GGML_CUDA_MMID_CAPABILITY_INVALID_PHASE);
    query.independent_rows = true;
    CHECK(ggml_cuda_mmid_get_capability(query).reason == GGML_CUDA_MMID_CAPABILITY_OK);
    query = candidate_mmid_query(GGML_TYPE_Q4_K);
    query.source_nb[0]++;
    CHECK(ggml_cuda_mmid_get_capability(query).reason == GGML_CUDA_MMID_CAPABILITY_INVALID_GEOMETRY);
    query = candidate_mmid_query(GGML_TYPE_Q4_K);
    query.mapping = static_cast<ggml_cuda_mmid_mapping>(2);
    CHECK(ggml_cuda_mmid_get_capability(query).reason == GGML_CUDA_MMID_CAPABILITY_INVALID_MAPPING);
    query = candidate_mmid_query(GGML_TYPE_Q4_K);
    query.smpbo = 0;
    CHECK(ggml_cuda_mmid_get_capability(query).reason == GGML_CUDA_MMID_CAPABILITY_INVALID_DEVICE);
    query = candidate_mmid_query(GGML_TYPE_Q4_K);
    query.input_type = GGML_TYPE_F16;
    CHECK(ggml_cuda_mmid_get_capability(query).reason == GGML_CUDA_MMID_CAPABILITY_INVALID_IO);
    query = candidate_mmid_query(GGML_TYPE_Q8_K, 16, GGML_CUDA_MMID_MAPPING_SOURCE_MAP, true);
    CHECK(ggml_cuda_mmid_get_capability(query).reason == GGML_CUDA_MMID_CAPABILITY_UNSUPPORTED_CONSUMER);

    query = candidate_mmid_query(GGML_TYPE_Q4_0, 4, GGML_CUDA_MMID_MAPPING_DIRECT, true);
    CHECK(ggml_cuda_mmid_can_use_compact_mmvq(query, 12));
    query = candidate_mmid_query(GGML_TYPE_Q4_0, 16, GGML_CUDA_MMID_MAPPING_DIRECT, true);
    CHECK(!ggml_cuda_mmid_can_use_compact_mmvq(query, 12));
    query = candidate_mmid_query(GGML_TYPE_Q4_0, 4, GGML_CUDA_MMID_MAPPING_SOURCE_MAP, true);
    CHECK(!ggml_cuda_mmid_can_use_compact_mmvq(query, 12));
}

void test_scheduler_execution_certificate() {
    ggml_backend_ptr first_backend(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    ggml_backend_ptr second_backend(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    CHECK(first_backend != nullptr && second_backend != nullptr);

    scheduler_certificate_probe probe;
    probe.backends = {first_backend.get(), second_backend.get()};
    probe.delegates = {first_backend->iface.graph_compute, second_backend->iface.graph_compute};
    probe.synchronize_delegates = {first_backend->iface.synchronize, second_backend->iface.synchronize};
    probe.n_backends = probe.backends.size();
    CHECK(probe.delegates[0] != nullptr && probe.delegates[1] != nullptr && scheduler_certificate_probe_current == nullptr);
    scheduler_certificate_probe_current = &probe;
    first_backend->iface.graph_compute = scheduler_certificate_graph_compute;
    second_backend->iface.graph_compute = scheduler_certificate_graph_compute;
    first_backend->iface.synchronize = scheduler_certificate_synchronize;
    second_backend->iface.synchronize = scheduler_certificate_synchronize;

    ggml_init_params params = {};
    params.mem_size = 24 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false);
    params.no_alloc = true;
    ggml_context_ptr ctx(ggml_init(params));
    CHECK(ctx != nullptr);
    ggml_tensor * first_input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
    ggml_tensor * second_input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
    ggml_set_input(first_input);
    ggml_set_input(second_input);
    ggml_tensor * first = ggml_add(ctx.get(), first_input, second_input);
    ggml_tensor * output = ggml_sqr(ctx.get(), first);
    ggml_set_output(output);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, output);

    ggml_backend_t backends[] = {first_backend.get(), second_backend.get()};
    ggml_backend_buffer_type first_buft = *ggml_backend_cpu_buffer_type();
    ggml_backend_buffer_type second_buft = *ggml_backend_cpu_buffer_type();
    ggml_backend_buffer_type_t bufts[] = {&first_buft, &second_buft};
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, bufts, 2, 16, false, false));
    CHECK(sched != nullptr);
    ggml_backend_sched_set_tensor_backend(sched.get(), first, first_backend.get());
    ggml_backend_sched_set_tensor_backend(sched.get(), output, second_backend.get());
    CHECK(ggml_backend_sched_alloc_graph(sched.get(), graph));
    CHECK(ggml_backend_sched_graph_compute(sched.get(), graph) == GGML_STATUS_SUCCESS);
    CHECK(ggml_backend_sched_get_n_splits(sched.get()) == 2);
    CHECK(probe.calls == 2);
    const ggml_graph_execution_certificate empty = {};
    const auto check_uncertified = [&](uint32_t first_record, bool uid_zero) {
        CHECK(probe.calls == first_record + 2);
        CHECK(probe.records[first_record].backend_index != probe.records[first_record + 1].backend_index);
        for (uint32_t record = first_record; record < first_record + 2; ++record) {
            CHECK(memcmp(&probe.records[record].certificate, &empty, sizeof(empty)) == 0);
            CHECK((probe.records[record].graph_uid == 0) == uid_zero);
        }
    };
    check_uncertified(0, false);

    ggml_graph_execution_certificate certificate = {};
    certificate.magic = GGML_GRAPH_EXECUTION_CERTIFICATE_MAGIC;
    certificate.abi_version = GGML_GRAPH_EXECUTION_CERTIFICATE_VERSION;
    certificate.struct_size = sizeof(certificate);
    certificate.domain = GGML_GRAPH_EXECUTION_DOMAIN_MAIN;
    certificate.row_semantics = GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT;
    certificate.n_rows = 1;
    certificate.n_sequences = 1;
    certificate.owner_namespace = 0x7363686564756c65ULL;
    certificate.owner_generation = 1;
    CHECK(ggml_backend_sched_graph_compute_ext(sched.get(), graph, &certificate) == GGML_STATUS_SUCCESS);
    CHECK(probe.calls == 4);
    const auto & first_stamped = probe.records[2];
    const auto & second_stamped = probe.records[3];
    CHECK(first_stamped.backend_index != second_stamped.backend_index);
    for (const auto * stamped : {&first_stamped, &second_stamped}) {
        CHECK(stamped->certificate.magic == GGML_GRAPH_EXECUTION_CERTIFICATE_MAGIC &&
            stamped->certificate.source_graph_uid == graph->uid &&
            stamped->certificate.split_graph_uid == stamped->graph_uid &&
            stamped->certificate.source_graph_uid != stamped->certificate.split_graph_uid);
    }
    CHECK(first_stamped.certificate.source_graph_uid == second_stamped.certificate.source_graph_uid &&
        first_stamped.certificate.split_graph_uid != second_stamped.certificate.split_graph_uid);
    CHECK(certificate.source_graph_uid == 0 && certificate.split_graph_uid == 0);

    certificate.abi_version++;
    CHECK(ggml_backend_sched_graph_compute_ext(sched.get(), graph, &certificate) == GGML_STATUS_SUCCESS);
    check_uncertified(4, false);
    certificate.abi_version = GGML_GRAPH_EXECUTION_CERTIFICATE_VERSION;

    ggml_cgraph callback_view = ggml_graph_view(graph, 0, graph->n_nodes);
    CHECK(ggml_backend_sched_graph_compute_ext(sched.get(), &callback_view, &certificate) == GGML_STATUS_SUCCESS);
    check_uncertified(6, false);

    ggml_backend_sched_set_eval_callback(sched.get(), scheduler_certificate_eval_callback, nullptr);
    CHECK(ggml_backend_sched_graph_compute_ext(sched.get(), graph, &certificate) == GGML_STATUS_SUCCESS);
    check_uncertified(8, true);
    ggml_backend_sched_set_eval_callback(sched.get(), nullptr, nullptr);

    CHECK(ggml_backend_sched_graph_compute(sched.get(), graph) == GGML_STATUS_SUCCESS);
    check_uncertified(10, false);

    certificate.flags = GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED;
    CHECK(ggml_backend_sched_graph_compute_ext(sched.get(), graph, &certificate) == GGML_STATUS_SUCCESS);
    CHECK(probe.calls == 14);
    for (uint32_t record = 12; record < 14; ++record) {
        CHECK(probe.records[record].certificate.flags == GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED &&
            probe.records[record].certificate.source_graph_uid == graph->uid &&
            probe.records[record].certificate.split_graph_uid == probe.records[record].graph_uid);
    }

    ggml_backend_sched_set_eval_callback(sched.get(), scheduler_certificate_eval_callback, nullptr);
    CHECK(ggml_backend_sched_graph_compute_ext(sched.get(), graph, &certificate) == GGML_STATUS_FAILED);
    CHECK(probe.calls == 14);
    ggml_backend_sched_set_eval_callback(sched.get(), nullptr, nullptr);
    CHECK(ggml_backend_sched_graph_compute_ext(sched.get(), &callback_view, &certificate) == GGML_STATUS_FAILED);
    CHECK(probe.calls == 14);
    certificate.abi_version++;
    CHECK(ggml_backend_sched_graph_compute_ext(sched.get(), graph, &certificate) == GGML_STATUS_FAILED);
    CHECK(probe.calls == 14);
    certificate.abi_version = GGML_GRAPH_EXECUTION_CERTIFICATE_VERSION;

    const auto synchronize_before = probe.synchronize_calls;
    probe.fail_backend = probe.records[13].backend_index;
    CHECK(ggml_backend_sched_graph_compute_async_ext(sched.get(), graph, &certificate) == GGML_STATUS_FAILED);
    CHECK(probe.calls == 16 && probe.records[14].backend_index != probe.fail_backend &&
        probe.records[15].backend_index == probe.fail_backend);
    for (uint32_t backend_index = 0; backend_index < probe.n_backends; ++backend_index) {
        CHECK(probe.synchronize_calls[backend_index] >= synchronize_before[backend_index] + 1);
    }
    probe.fail_backend = UINT32_MAX;

    certificate.domain = GGML_GRAPH_EXECUTION_DOMAIN_DRAFT;
    certificate.row_semantics = GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL;
    CHECK(ggml_backend_sched_graph_compute_ext(sched.get(), graph, &certificate) == GGML_STATUS_SUCCESS);
    CHECK(probe.calls == 18);
    for (uint32_t record = 16; record < 18; ++record) {
        CHECK(probe.records[record].certificate.domain == GGML_GRAPH_EXECUTION_DOMAIN_DRAFT &&
            probe.records[record].certificate.row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL &&
            probe.records[record].certificate.flags == GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED &&
            probe.records[record].certificate.source_graph_uid == graph->uid &&
            probe.records[record].certificate.split_graph_uid == probe.records[record].graph_uid);
    }

    first_backend->iface.graph_compute = probe.delegates[0];
    second_backend->iface.graph_compute = probe.delegates[1];
    first_backend->iface.synchronize = probe.synchronize_delegates[0];
    second_backend->iface.synchronize = probe.synchronize_delegates[1];
    scheduler_certificate_probe_current = nullptr;
}

void test_graph_execution_certificate_policy() {
    candidate_test_fixture fixture;
    const int64_t gate_up_ne[] = {256, 512, 4};
    const int64_t down_ne[] = {256, 256, 4};
    ggml_tensor * gate_up = fixture.cached_tensor(GGML_TYPE_Q4_0, 3, gate_up_ne);
    ggml_tensor * down = fixture.cached_tensor(GGML_TYPE_Q4_0, 3, down_ne);
    ggml_set_name(gate_up, "test_policy_gate_up_weight");
    ggml_set_name(down, "test_policy_down_weight");
    std::array<ggml_backend_moe_candidate_bank_v1, 2> banks = {{
        {gate_up, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, 0},
        {down, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
    }};
    const ggml_backend_moe_candidate_group_v1 group = {
        banks.data(), banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 0, 0,
    };
    const auto snapshot = candidate_snapshot(12, &group, 1);
    ggml_cuda_moe_grouped_context registry(&fixture.owner, 0);
    CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);

    struct graph_case {
        candidate_route route;
        ggml_tensor * gate_up = nullptr;
        ggml_tensor * down = nullptr;
        ggml_cgraph * graph = nullptr;
    };
    const auto make_graph = [&](int64_t top_k, int64_t n_rows) {
        graph_case result;
        result.route = candidate_top_k_route(fixture, 4, top_k, n_rows);
        result.gate_up = candidate_mmid(fixture, gate_up, result.route.ids);
        result.down = candidate_mmid(fixture, down, result.route.ids);
        result.graph = candidate_graph(fixture, {result.route.root, result.route.ids, result.gate_up, result.down});
        return result;
    };
    const auto compile = [&](graph_case & current, uint64_t graph_uid,
            ggml_cuda_moe_graph_plan & plan, ggml_cuda_moe_graph_execution & execution) {
        registry.compile_graph_plan(current.graph, graph_uid, &plan, &execution);
    };
    const auto check_execution_legacy = [&](graph_case & current, uint64_t graph_uid) {
        ggml_cuda_moe_graph_plan plan;
        ggml_cuda_moe_graph_execution execution;
        compile(current, graph_uid, plan, execution);
        CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_LEGACY && execution.size() == 1);
        CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_execution_reason(plan, 0));
    };

    for (uint32_t n_rows : {1u, 2u, 3u, 4u}) {
        auto current = make_graph(2, n_rows);
        candidate_stamp_execution(current.graph, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
            GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, n_rows, n_rows);
        const uint32_t row_stride = current.route.ids->nb[1] / sizeof(int32_t);
        CHECK(row_stride == 4 && row_stride > static_cast<uint32_t>(current.route.ids->ne[0]));
        int32_t * ids = static_cast<int32_t *>(current.route.ids->data);
        for (uint32_t row = 0; row < n_rows; ++row) {
            ids[row * row_stride] = row == 0 ? 3 : 1;
            ids[row * row_stride + 1] = row == 2 ? 3 : 1;
        }
        ggml_cuda_moe_graph_plan plan;
        ggml_cuda_moe_graph_execution execution;
        compile(current, 1100 + n_rows, plan, execution);
        ggml_cuda_moe_graph_binding binding;
        CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED &&
            execution.find(current.down, &binding));
        CHECK(binding.key.ids.ne[0] == 2 && binding.key.ids.ne[1] == n_rows &&
            binding.key.ids.nb[1] / sizeof(int32_t) == row_stride && binding.key.execution_semantic_key != 0);
    }

    ggml_set_name(gate_up, "test.policy.gate_up");
    ggml_set_name(down, "test.policy.down");
    auto compact_mmvq_without_mmq_name = make_graph(2, 2);
    candidate_stamp_execution(compact_mmvq_without_mmq_name.graph, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
        GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 2, 2);
    {
        ggml_cuda_moe_graph_plan plan;
        ggml_cuda_moe_graph_execution execution;
        compile(compact_mmvq_without_mmq_name, 1199, plan, execution);
        CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED && execution.size() == 1);
        CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_eligible_reason(plan, 0));
        for (uint32_t bank_index = 0; bank_index < 2; ++bank_index) {
            const auto capability = ggml_cuda_moe_grouped_context_test_access::graph_bank_capability(plan, 0, bank_index);
            CHECK(capability.consumer == GGML_CUDA_MMID_CONSUMER_MMVQ && capability.use_mmq == 0 &&
                capability.mapping == GGML_CUDA_MMID_MAPPING_DIRECT);
        }
    }
    ggml_set_name(gate_up, "test_policy_gate_up_weight");
    ggml_set_name(down, "test_policy_down_weight");

    auto invalid = make_graph(2, 1);
    const auto valid_certificate = invalid.graph->execution_certificate;
    invalid.graph->execution_certificate = {};
    check_execution_legacy(invalid, 1200);
    invalid.graph->execution_certificate = valid_certificate;
    invalid.graph->execution_certificate.abi_version++;
    check_execution_legacy(invalid, 1201);
    invalid.graph->execution_certificate = valid_certificate;
    invalid.graph->execution_certificate.reserved[0] = 1;
    check_execution_legacy(invalid, 1202);
    invalid.graph->execution_certificate = valid_certificate;
    invalid.graph->execution_certificate.owner_generation = 0;
    check_execution_legacy(invalid, 1203);
    invalid.graph->execution_certificate = valid_certificate;
    invalid.graph->execution_certificate.source_graph_uid = invalid.graph->uid;
    check_execution_legacy(invalid, 1204);
    invalid.graph->execution_certificate = valid_certificate;
    invalid.graph->execution_certificate.split_graph_uid++;
    check_execution_legacy(invalid, 1205);
    invalid.graph->execution_certificate = valid_certificate;
    invalid.graph->execution_certificate.n_rows = 2;
    invalid.graph->execution_certificate.n_sequences = 2;
    check_execution_legacy(invalid, 1206);

    auto parent_case = make_graph(2, 1);
    ggml_cgraph split = ggml_graph_view(parent_case.graph, 0, parent_case.graph->n_nodes);
    candidate_stamp_execution(&split, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
        GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 1, 1, parent_case.graph->uid);
    graph_case split_case = {parent_case.route, parent_case.gate_up, parent_case.down, &split};
    {
        ggml_cuda_moe_graph_plan plan;
        ggml_cuda_moe_graph_execution execution;
        compile(split_case, 1210, plan, execution);
        CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED && execution.find(parent_case.down, nullptr));
    }
    split.execution_certificate.split_graph_uid = parent_case.graph->uid;
    check_execution_legacy(split_case, 1211);

    auto isolated = make_graph(2, 2);
    for (uint32_t domain : {GGML_GRAPH_EXECUTION_DOMAIN_DRAFT, GGML_GRAPH_EXECUTION_DOMAIN_MTP}) {
        candidate_stamp_execution(isolated.graph, domain,
            GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 2, 2);
        check_execution_legacy(isolated, 1220 + domain);
    }
    for (uint32_t domain : {GGML_GRAPH_EXECUTION_DOMAIN_DRAFT, GGML_GRAPH_EXECUTION_DOMAIN_MTP}) {
        candidate_stamp_execution(isolated.graph, domain,
            GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 2, 2, 0,
            GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED);
        const auto required_coverage = candidate_certify_graph(registry, isolated.graph);
        std::shared_ptr<ggml_cuda_moe_graph_plan> plan;
        ggml_cuda_moe_graph_execution execution;
        CHECK(registry.prepare_graph_execution(
            isolated.graph, 1225 + domain, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &plan, &execution,
            required_coverage.epoch, required_coverage.nodes, required_coverage.mmid_count,
            required_coverage.mmid_fingerprint) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(plan != nullptr && ggml_cuda_moe_required_grouped_plan_ready(*plan, execution) &&
            execution.allows_graph_capture());
        const auto capability = ggml_cuda_moe_grouped_context_test_access::graph_bank_capability(*plan, 0, 0);
        CHECK(capability.row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT &&
            capability.device == 0 &&
            capability.strategy == GGML_CUDA_MOE_EXECUTION_STRATEGY_DEVICE_DIRECT &&
            capability.materialized_phase == GGML_CUDA_MMID_PHASE_DECODE);
    }
    const auto check_required_sequential_direct = [&](graph_case & current, uint32_t domain, uint64_t graph_uid,
                                                       uint32_t n_rows, uint32_t n_sequences) {
        candidate_stamp_execution(current.graph, domain,
            GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL, n_rows, n_sequences, 0,
            GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED);
        const auto required_coverage = candidate_certify_graph(registry, current.graph);
        std::shared_ptr<ggml_cuda_moe_graph_plan> plan;
        ggml_cuda_moe_graph_execution execution;
        CHECK(registry.prepare_graph_execution(
            current.graph, graph_uid, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &plan, &execution,
            required_coverage.epoch, required_coverage.nodes, required_coverage.mmid_count,
            required_coverage.mmid_fingerprint) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(plan != nullptr && ggml_cuda_moe_required_grouped_plan_ready(*plan, execution) &&
            execution.allows_graph_capture());
        const auto capability = ggml_cuda_moe_grouped_context_test_access::graph_bank_capability(*plan, 0, 0);
        ggml_cuda_mmid_capability_query native;
        native.source_type = static_cast<ggml_type>(capability.source_type);
        native.input_type = static_cast<ggml_type>(capability.input_type);
        native.output_type = static_cast<ggml_type>(capability.output_type);
        memcpy(native.source_ne, capability.source_ne, sizeof(native.source_ne));
        memcpy(native.source_nb, capability.source_nb, sizeof(native.source_nb));
        native.n_tokens = capability.n_tokens;
        native.n_experts = capability.n_experts;
        native.cc = capability.cc;
        native.warp_size = capability.warp_size;
        native.smpbo = capability.smpbo;
        native.phase = static_cast<ggml_cuda_mmid_phase>(capability.phase);
        native.mapping = static_cast<ggml_cuda_mmid_mapping>(capability.mapping);
        native.use_mmq = capability.use_mmq != 0;
        const auto native_capability = ggml_cuda_mmid_get_capability(native);
        const uint32_t expected_phase = current.route.ids->ne[1] == 1 ?
            GGML_CUDA_MMID_PHASE_DECODE : GGML_CUDA_MMID_PHASE_PREFILL;
        CHECK(capability.row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL &&
            capability.device == 0 &&
            capability.strategy == GGML_CUDA_MOE_EXECUTION_STRATEGY_DEVICE_DIRECT &&
            capability.phase == expected_phase && capability.materialized_phase == capability.phase &&
            capability.materialized_mapping == GGML_CUDA_MMID_MAPPING_DIRECT &&
            native_capability.reason == GGML_CUDA_MMID_CAPABILITY_OK &&
            native_capability.selection == capability.consumer);
    };
    check_required_sequential_direct(isolated, GGML_GRAPH_EXECUTION_DOMAIN_MTP, 1228, 2, 1);
    check_required_sequential_direct(isolated, GGML_GRAPH_EXECUTION_DOMAIN_DRAFT, 1229, 2, 1);
    auto subrow = make_graph(2, 1);
    check_required_sequential_direct(subrow, GGML_GRAPH_EXECUTION_DOMAIN_MTP, 1230, 2, 1);
    candidate_stamp_execution(isolated.graph, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
        GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SPECULATIVE, 2, 2);
    check_execution_legacy(isolated, 1230);
    candidate_stamp_execution(isolated.graph, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
        GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SPECULATIVE, 2, 1, 0,
        GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED);
    {
        const auto required_coverage = candidate_certify_graph(registry, isolated.graph);
        std::shared_ptr<ggml_cuda_moe_graph_plan> plan;
        ggml_cuda_moe_graph_execution execution;
        CHECK(registry.prepare_graph_execution(
            isolated.graph, 1231, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &plan, &execution,
            required_coverage.epoch, required_coverage.nodes, required_coverage.mmid_count,
            required_coverage.mmid_fingerprint) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(plan != nullptr && ggml_cuda_moe_required_grouped_plan_ready(*plan, execution) &&
            execution.allows_graph_capture());
        const auto capability = ggml_cuda_moe_grouped_context_test_access::graph_bank_capability(*plan, 0, 0);
        CHECK(capability.row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SPECULATIVE &&
            capability.strategy == GGML_CUDA_MOE_EXECUTION_STRATEGY_DEVICE_DIRECT &&
            capability.materialized_phase == GGML_CUDA_MMID_PHASE_PREFILL);
    }
    candidate_stamp_execution(isolated.graph, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
        GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL, 2, 2);
    {
        ggml_cuda_moe_graph_plan plan;
        ggml_cuda_moe_graph_execution execution;
        compile(isolated, 1231, plan, execution);
        CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_PREFILL_LEGACY && !execution.find(isolated.down, nullptr));
        CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_prefill_reason(plan, 0));
    }

    auto np4_like = make_graph(2, 6);
    check_required_sequential_direct(np4_like, GGML_GRAPH_EXECUTION_DOMAIN_MTP, 1238, 6, 4);
    check_required_sequential_direct(np4_like, GGML_GRAPH_EXECUTION_DOMAIN_DRAFT, 1239, 6, 4);

    auto too_many_routes = make_graph(4, 4);
    candidate_stamp_execution(too_many_routes.graph, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
        GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 4, 4);
    check_execution_legacy(too_many_routes, 1240);
    for (uint32_t domain : {GGML_GRAPH_EXECUTION_DOMAIN_DRAFT, GGML_GRAPH_EXECUTION_DOMAIN_MTP}) {
        for (uint32_t row_semantics : {
                GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT,
                GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL}) {
            const uint32_t n_sequences = row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT ? 4 : 1;
            candidate_stamp_execution(too_many_routes.graph, domain, row_semantics, 4, n_sequences, 0,
                GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED);
            const auto required_coverage = candidate_certify_graph(registry, too_many_routes.graph);
            std::shared_ptr<ggml_cuda_moe_graph_plan> plan;
            ggml_cuda_moe_graph_execution execution;
            CHECK(registry.prepare_graph_execution(
                too_many_routes.graph, 1241 + 4 * domain + row_semantics,
                GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &plan, &execution,
                required_coverage.epoch, required_coverage.nodes, required_coverage.mmid_count,
                required_coverage.mmid_fingerprint) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
            CHECK(plan != nullptr && ggml_cuda_moe_required_grouped_plan_ready(*plan, execution) &&
                !execution.allows_graph_capture());
            const auto capability = ggml_cuda_moe_grouped_context_test_access::graph_bank_capability(*plan, 0, 0);
            const uint32_t phase = row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT ?
                GGML_CUDA_MMID_PHASE_DECODE : GGML_CUDA_MMID_PHASE_PREFILL;
            CHECK(capability.row_semantics == row_semantics && capability.device == 0 &&
                capability.strategy == GGML_CUDA_MOE_EXECUTION_STRATEGY_HOST_STAGED &&
                capability.phase == GGML_CUDA_MMID_PHASE_PREFILL && capability.materialized_phase == phase &&
                capability.consumer != GGML_CUDA_MMID_CONSUMER_UNSUPPORTED &&
                (capability.materialized_mapping == GGML_CUDA_MMID_MAPPING_DIRECT ||
                    capability.materialized_mapping == GGML_CUDA_MMID_MAPPING_SOURCE_MAP));
        }
    }

    auto stable = make_graph(2, 3);
    candidate_stamp_execution(stable.graph, GGML_GRAPH_EXECUTION_DOMAIN_DRAFT,
        GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 3, 3, 0,
        GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED);
    const auto coverage = candidate_certify_graph(registry, stable.graph);
    std::shared_ptr<ggml_cuda_moe_graph_plan> stable_plan;
    ggml_cuda_moe_graph_execution prepared;
    CHECK(registry.prepare_graph_execution(
        stable.graph, 1250, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &stable_plan, &prepared,
        coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
        GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
    const auto * first_plan = stable_plan.get();
    const uint64_t first_semantic_key = ggml_cuda_moe_grouped_context_test_access::graph_execution_semantic_key(*stable_plan);
    CHECK(first_semantic_key != 0 && first_semantic_key == ggml_cuda_moe_execution_semantic_key(stable.graph));
    static_cast<int32_t *>(stable.route.ids->data)[0] = 2;
    CHECK(first_semantic_key == ggml_cuda_moe_execution_semantic_key(stable.graph));
    stable.graph->execution_certificate.source_graph_uid = ggml_graph_next_uid();
    stable.graph->uid = ggml_graph_next_uid();
    stable.graph->execution_certificate.split_graph_uid = stable.graph->uid;
    CHECK(first_semantic_key == ggml_cuda_moe_execution_semantic_key(stable.graph));
    CHECK(registry.prepare_graph_execution(
        stable.graph, 1251, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, &stable_plan, &prepared,
        coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
        GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
    CHECK(stable_plan.get() == first_plan);
    stable.graph->execution_certificate.owner_generation++;
    CHECK(!registry.bind_graph_plan(
        stable.graph, 1252, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, *stable_plan, &prepared,
        coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint));
    CHECK(registry.prepare_graph_execution(
        stable.graph, 1252, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, &stable_plan, &prepared,
        coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
        GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
    CHECK(stable_plan.get() != first_plan &&
        ggml_cuda_moe_grouped_context_test_access::graph_execution_semantic_key(*stable_plan) != first_semantic_key);

    ggml_cgraph callback = ggml_graph_view(parent_case.graph, 0, parent_case.graph->n_nodes);
    graph_case callback_case = {parent_case.route, parent_case.gate_up, parent_case.down, &callback};
    CHECK(callback.uid == 0 && ggml_cuda_moe_execution_semantic_key(&callback) == 0);
    check_execution_legacy(callback_case, 0);

    const int64_t parallel_ne[] = {256, 256, 8};
    ggml_tensor * parallel_gate = fixture.cached_tensor(GGML_TYPE_Q4_K, 3, parallel_ne);
    ggml_tensor * parallel_up = fixture.cached_tensor(GGML_TYPE_Q4_K, 3, parallel_ne);
    ggml_tensor * parallel_down = fixture.cached_tensor(GGML_TYPE_Q4_K, 3, parallel_ne);
    ggml_set_name(parallel_gate, "test_parallel_gate_weight");
    ggml_set_name(parallel_up, "test_parallel_up_weight");
    ggml_set_name(parallel_down, "test_parallel_down_weight");
    std::array<ggml_backend_moe_candidate_bank_v1, 3> parallel_banks = {{
        {parallel_gate, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT, 0},
        {parallel_up, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT, 0},
        {parallel_down, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
    }};
    const ggml_backend_moe_candidate_group_v1 parallel_group = {
        parallel_banks.data(), parallel_banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 0, 0,
    };
    const auto make_parallel_graph = [&](int64_t n_rows) {
        graph_case result;
        result.route = candidate_top_k_route(fixture, 8, 8, n_rows);
        result.gate_up = candidate_mmid(fixture, parallel_gate, result.route.ids);
        ggml_tensor * up = candidate_mmid(fixture, parallel_up, result.route.ids);
        result.down = candidate_mmid(fixture, parallel_down, result.route.ids);
        result.graph = candidate_graph(fixture, {result.route.root, result.route.ids, result.gate_up, up, result.down});
        candidate_stamp_execution(result.graph, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
            GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, n_rows, n_rows);
        return result;
    };

    const auto parallel_snapshot = candidate_snapshot(132, &parallel_group, 1);
    CHECK(registry.replace(&parallel_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    auto parallel = make_parallel_graph(16);
    {
        ggml_cuda_moe_graph_plan plan;
        ggml_cuda_moe_graph_execution execution;
        compile(parallel, 1260, plan, execution);
        CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED && execution.find(parallel.down, nullptr));
        for (uint32_t bank_index = 0; bank_index < parallel_banks.size(); ++bank_index) {
            const auto capability = ggml_cuda_moe_grouped_context_test_access::graph_bank_capability(plan, 0, bank_index);
            CHECK(capability.consumer == GGML_CUDA_MMID_CONSUMER_MMQ && capability.use_mmq == 1 &&
                capability.mapping == GGML_CUDA_MMID_MAPPING_SOURCE_MAP);
        }
    }

    const auto low_slot_snapshot = candidate_snapshot(48, &parallel_group, 1);
    CHECK(registry.replace(&low_slot_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    check_execution_legacy(parallel, 1261);

    CHECK(registry.replace(&parallel_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    auto over_slots = make_parallel_graph(17);
    check_execution_legacy(over_slots, 1262);

    const auto wide_snapshot = candidate_snapshot(300, &parallel_group, 1);
    CHECK(registry.replace(&wide_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    auto wide = make_parallel_graph(33);
    {
        ggml_cuda_moe_graph_plan plan;
        ggml_cuda_moe_graph_execution execution;
        compile(wide, 1263, plan, execution);
        CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED && execution.find(wide.down, nullptr));
    }
}

void test_candidate_generic_physical_truth() {
    candidate_test_fixture fixture;
    ggml_cuda_moe_grouped_context registry(&fixture.owner, 0);
    const int64_t weight_ne[] = {256, 256, 4};
    constexpr uint32_t cached = GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_CACHED_BUFFER;

    ggml_tensor * gate_q3 = fixture.cached_tensor(GGML_TYPE_Q3_K, 3, weight_ne);
    ggml_tensor * up_iq3 = fixture.cached_tensor(GGML_TYPE_IQ3_XXS, 3, weight_ne);
    ggml_tensor * down_iq3 = fixture.cached_tensor(GGML_TYPE_IQ3_S, 3, weight_ne);
    std::array<ggml_backend_moe_candidate_bank_v1, 3> mixed_banks = {{
        {gate_q3, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT, 0},
        {up_iq3, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT, 0},
        {down_iq3, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
    }};
    ggml_backend_moe_candidate_group_v1 mixed_group = {
        mixed_banks.data(), mixed_banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 0, 0,
    };
    const auto v1_snapshot = candidate_snapshot(12, &mixed_group, 1);
    CHECK(registry.replace(&v1_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    const auto v1_state = registry.state();

    ggml_cuda_moe_candidate_group_key key;
    ggml_cuda_moe_candidate_group_info group_info;
    CHECK(registry.find_down_group_key(down_iq3, &key) && registry.get_group(key, &group_info));
    CHECK(group_info.layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE);
    CHECK(group_info.domain == GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY);
    CHECK(group_info.semantic_group_index == 0 && group_info.flags == 0);

    std::array<ggml_backend_moe_candidate_group_v2, 2> groups_v2 = {{
        {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_UNGATED, GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_CHUNK, 0, 0},
        {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY, 0, 0},
    }};
    std::array<ggml_backend_moe_candidate_tensor_v2, 3> tensors_v2 = {{
        {gate_q3, 1, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE, cached, 0},
        {up_iq3, 1, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE, cached, 0},
        {down_iq3, 1, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE, cached, 0},
    }};
    const auto v2_snapshot = candidate_snapshot_v2(12, groups_v2.data(), groups_v2.size(), tensors_v2.data(), tensors_v2.size());
    CHECK(registry.replace(&v2_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    const auto v2_state = registry.state();
    CHECK(v2_state.n_groups == 1 && v2_state.n_weights == 3);
    CHECK(v2_state.logical_signature == v1_state.logical_signature);
    CHECK(registry.find_down_group_key(down_iq3, &key) && key.group_index == 0 && registry.get_group(key, &group_info));
    CHECK(group_info.domain == GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY);
    CHECK(group_info.semantic_group_index == 1 && group_info.flags == 0);

    for (const ggml_tensor * weight : {gate_q3, up_iq3, down_iq3}) {
        ggml_cuda_moe_candidate_bank_info info;
        const auto source = ggml_cuda_mmid_source_capability_for(weight->type);
        CHECK(registry.find_weight(weight, &info));
        CHECK(info.type == weight->type && info.source_flags == source.flags);
        CHECK(info.source_flags & GGML_CUDA_MMID_SOURCE_ADVERTISED);
        CHECK(info.encoding == GGML_CUDA_MOE_CANDIDATE_ENCODING_PLAIN);
        CHECK(info.movement == GGML_CUDA_MOE_CANDIDATE_MOVEMENT_SLOT_BOUND);
        CHECK(info.index_modes == (GGML_CUDA_MOE_CANDIDATE_INDEX_GROUP_SLOT_DIRECT |
            GGML_CUDA_MOE_CANDIDATE_INDEX_ORIGINAL_SOURCE_MAP));
        CHECK(info.byte_extent == ggml_nbytes(weight) && info.expert_stride == weight->nb[2]);
    }

    const candidate_route mixed_route = candidate_top_k_route(fixture, 4, 2);
    ggml_tensor * gate_q3_reader = candidate_mmid(fixture, gate_q3, mixed_route.ids);
    ggml_tensor * up_iq3_reader = candidate_mmid(fixture, up_iq3, mixed_route.ids);
    ggml_tensor * down_iq3_reader = candidate_mmid(fixture, down_iq3, mixed_route.ids);
    ggml_cgraph * mixed_graph = candidate_graph(fixture, {
        mixed_route.root, mixed_route.ids, gate_q3_reader, up_iq3_reader, down_iq3_reader,
    });
    ggml_cuda_moe_graph_plan plan;
    ggml_cuda_moe_graph_execution execution;
    registry.compile_graph_plan(mixed_graph, 1001, &plan, &execution);
    CHECK(execution.size() == 1 && execution.find(down_iq3_reader, nullptr));
    CHECK(ggml_cuda_moe_grouped_context_test_access::graph_outcome(plan) ==
        GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED);
    const std::array<ggml_tensor *, 3> mixed_weights = {gate_q3, up_iq3, down_iq3};
    for (uint32_t bank = 0; bank < mixed_weights.size(); ++bank) {
        const auto capability = ggml_cuda_moe_grouped_context_test_access::graph_bank_capability(plan, 0, bank);
        const auto source = ggml_cuda_mmid_source_capability_for(mixed_weights[bank]->type);
        CHECK(capability.tensor == mixed_weights[bank] && capability.source_data == mixed_weights[bank]->data);
        CHECK(capability.byte_extent == ggml_nbytes(mixed_weights[bank]) && capability.expert_stride == mixed_weights[bank]->nb[2]);
        CHECK(capability.n_tokens == 1 && capability.n_experts == 4);
        CHECK(capability.device == 0 && capability.cc > 0 && capability.warp_size > 0 && capability.smpbo > 0);
        CHECK(capability.role == mixed_banks[bank].role && capability.source_type == (uint32_t) mixed_weights[bank]->type);
        CHECK(capability.source_flags == source.flags && capability.input_type == GGML_TYPE_F32 && capability.output_type == GGML_TYPE_F32);
        CHECK(capability.phase == GGML_CUDA_MMID_PHASE_DECODE && capability.mapping == GGML_CUDA_MMID_MAPPING_DIRECT);
        CHECK(capability.consumer == GGML_CUDA_MMID_CONSUMER_MMVQ && capability.reason == GGML_CUDA_MMID_CAPABILITY_OK);
    }
    CHECK(!ggml_cuda_moe_grouped_context_test_access::has_device_resource(registry, key));

    const ggml_type saved_activation_type = gate_q3_reader->src[1]->type;
    gate_q3_reader->src[1]->type = GGML_TYPE_F16;
    registry.compile_graph_plan(mixed_graph, 1002, &plan, &execution);
    CHECK(!execution.find(down_iq3_reader, nullptr));
    CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_capability_reason(plan, 0));
    gate_q3_reader->src[1]->type = saved_activation_type;

    std::array<ggml_backend_moe_candidate_bank_v1, 3> reordered_banks = {{mixed_banks[1], mixed_banks[0], mixed_banks[2]}};
    const ggml_backend_moe_candidate_group_v1 reordered_group = {
        reordered_banks.data(), reordered_banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 0, 0,
    };
    const auto reordered_snapshot = candidate_snapshot(12, &reordered_group, 1);
    CHECK(registry.replace(&reordered_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    registry.compile_graph_plan(mixed_graph, 1003, &plan, &execution);
    CHECK(!execution.find(down_iq3_reader, nullptr));
    CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_descriptor_reason(plan, 0));
    CHECK(registry.find_down_group_key(down_iq3, &key));
    ggml_cuda_moe_grouped_acquisition resource;
    CHECK(!registry.acquire_group_resources(key, &resource));
    CHECK(!ggml_cuda_moe_grouped_context_test_access::has_device_resource(registry, key));

    CHECK(registry.replace(&v1_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);

    ggml_tensor * gate_q8k = fixture.cached_tensor(GGML_TYPE_Q8_K, 3, weight_ne);
    ggml_tensor * up_q8k = fixture.cached_tensor(GGML_TYPE_Q8_K, 3, weight_ne);
    ggml_tensor * down_q8k = fixture.cached_tensor(GGML_TYPE_Q8_K, 3, weight_ne);
    std::array<ggml_backend_moe_candidate_bank_v1, 3> q8k_banks = {{
        {gate_q8k, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT, 0},
        {up_q8k, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT, 0},
        {down_q8k, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
    }};
    ggml_backend_moe_candidate_group_v1 q8k_group = {
        q8k_banks.data(), q8k_banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 0, 0,
    };
    const auto q8k_snapshot = candidate_snapshot(12, &q8k_group, 1);
    CHECK(registry.replace(&q8k_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(registry.find_down_group_key(down_q8k, &key));
    for (const ggml_tensor * weight : {gate_q8k, up_q8k, down_q8k}) {
        ggml_cuda_moe_candidate_bank_info info;
        CHECK(registry.find_weight(weight, &info));
        CHECK(info.type == GGML_TYPE_Q8_K && info.source_flags == GGML_CUDA_MMID_SOURCE_ADVERTISED);
        CHECK(info.encoding == GGML_CUDA_MOE_CANDIDATE_ENCODING_PLAIN);
    }
    const candidate_route q8k_route = candidate_top_k_route(fixture, 4, 2);
    ggml_tensor * gate_q8k_reader = candidate_mmid(fixture, gate_q8k, q8k_route.ids);
    ggml_tensor * up_q8k_reader = candidate_mmid(fixture, up_q8k, q8k_route.ids);
    ggml_tensor * down_q8k_reader = candidate_mmid(fixture, down_q8k, q8k_route.ids);
    ggml_cgraph * q8k_graph = candidate_graph(fixture, {
        q8k_route.root, q8k_route.ids, gate_q8k_reader, up_q8k_reader, down_q8k_reader,
    });
    registry.compile_graph_plan(q8k_graph, 1004, &plan, &execution);
    CHECK(execution.size() == 1 && !execution.find(down_q8k_reader, nullptr));
    CHECK(ggml_cuda_moe_grouped_context_test_access::graph_outcome(plan) ==
        GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR);
    CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_capability_reason(plan, 0));
    for (uint32_t bank = 0; bank < q8k_banks.size(); ++bank) {
        const auto capability = ggml_cuda_moe_grouped_context_test_access::graph_bank_capability(plan, 0, bank);
        CHECK(capability.tensor == q8k_banks[bank].tensor && capability.source_type == GGML_TYPE_Q8_K);
        CHECK(capability.source_flags == GGML_CUDA_MMID_SOURCE_ADVERTISED);
        CHECK(capability.consumer == GGML_CUDA_MMID_CONSUMER_UNSUPPORTED);
        CHECK(capability.reason == GGML_CUDA_MMID_CAPABILITY_UNSUPPORTED_CONSUMER);
    }
    CHECK(!registry.acquire_group_resources(key, &resource));
    CHECK(!ggml_cuda_moe_grouped_context_test_access::has_device_resource(registry, key));
    CHECK(execution.resolve_streams(candidate_test_graph_stream, reinterpret_cast<void *>(uintptr_t{1})));
    CHECK(!registry.begin_graph_dispatch(&execution, true));
    CHECK(!ggml_cuda_moe_grouped_context_test_access::has_device_resource(registry, key));
    CHECK(registry.begin_graph_dispatch(&execution, false));
    auto q8k_legacy = registry.acquire_legacy_cache(gate_q8k);
    CHECK(q8k_legacy && q8k_legacy.acquisition().registered_source == 1);
    q8k_legacy = {};
    CHECK(registry.finish_graph_dispatch(&execution));

    ggml_tensor * gate_q4 = fixture.cached_tensor(GGML_TYPE_Q4_K, 3, weight_ne);
    ggml_tensor * up_q4 = fixture.cached_tensor(GGML_TYPE_Q4_K, 3, weight_ne);
    ggml_tensor * down_q4 = fixture.cached_tensor(GGML_TYPE_Q4_K, 3, weight_ne);
    std::array<ggml_backend_moe_candidate_bank_v1, 3> q4_banks = {{
        {gate_q4, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT, 0},
        {up_q4, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT, 0},
        {down_q4, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
    }};
    ggml_backend_moe_candidate_group_v1 q4_group = {
        q4_banks.data(), q4_banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 0, 0,
    };
    const auto q4_snapshot = candidate_snapshot(12, &q4_group, 1);
    CHECK(registry.replace(&q4_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    const candidate_route q4_route = candidate_top_k_route(fixture, 4, 2);
    ggml_tensor * gate_q4_reader = candidate_mmid(fixture, gate_q4, q4_route.ids);
    ggml_tensor * up_q4_reader = candidate_mmid(fixture, up_q4, q4_route.ids);
    ggml_tensor * down_q4_reader = candidate_mmid(fixture, down_q4, q4_route.ids);
    ggml_cgraph * q4_graph = candidate_graph(fixture, {
        q4_route.root, q4_route.ids, gate_q4_reader, up_q4_reader, down_q4_reader,
    });
    registry.compile_graph_plan(q4_graph, 1005, &plan, &execution);
    auto * dispatch = execution.find_group(down_q4_reader, nullptr);
    CHECK(dispatch != nullptr && execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED);
    fprintf(stderr, "test-moe-cache: generic physical candidate truth OK\n");
}

void test_candidate_producer() {
    candidate_test_fixture fixture;
    llama_model_params params = llama_model_default_params();
    params.moe_expert_cache_slots = 12;
    std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_LLAMA, params));
    CHECK(model != nullptr && model->moe_expert_cache_slots() == 12);
    model->layers.resize(4);

    const int64_t router_ne[] = {64, 4};
    const int64_t gate_ne[] = {64, 32, 4};
    const int64_t down_ne[] = {32, 64, 4};
    const int64_t fused_ne[] = {64, 64, 4};
    const int64_t scale_ne[] = {4};
    const int64_t gate_bias_ne[] = {32, 4};
    const int64_t fused_bias_ne[] = {64, 4};
    const int64_t down_bias_ne[] = {64, 4};
    const int64_t scalar_ne[] = {1};

    auto & separate = model->layers[0];
    separate.ffn_gate_inp = fixture.tensor(GGML_TYPE_F32, 2, router_ne);
    separate.ffn_gate_exps = fixture.cached_tensor(GGML_TYPE_Q4_0, 3, gate_ne);
    separate.ffn_up_exps = fixture.cached_tensor(GGML_TYPE_Q4_0, 3, gate_ne);
    separate.ffn_down_exps = fixture.cached_tensor(GGML_TYPE_Q4_0, 3, down_ne);
    separate.ffn_gate_exps_s = fixture.cached_tensor(GGML_TYPE_F32, 1, scale_ne);
    separate.ffn_up_exps_s = fixture.cached_tensor(GGML_TYPE_F32, 1, scale_ne);
    separate.ffn_down_exps_s = fixture.cached_tensor(GGML_TYPE_F32, 1, scale_ne);
    separate.ffn_gate_exps_b = fixture.cached_tensor(GGML_TYPE_F32, 2, gate_bias_ne);
    separate.ffn_up_exps_b = fixture.cached_tensor(GGML_TYPE_F32, 2, gate_bias_ne);
    separate.ffn_down_exps_b = fixture.cached_tensor(GGML_TYPE_F32, 2, down_bias_ne);
    separate.ffn_gate = fixture.tensor(GGML_TYPE_BF16, 2, gate_ne);
    separate.ffn_up_shexp = fixture.tensor(GGML_TYPE_BF16, 2, gate_ne);
    separate.ffn_gate_exps_in_s = fixture.cached_tensor(GGML_TYPE_F32, 1, scalar_ne);

    auto & fused = model->layers[1];
    fused.ffn_gate_inp = fixture.tensor(GGML_TYPE_F32, 2, router_ne);
    fused.ffn_gate_up_exps = fixture.cached_tensor(GGML_TYPE_BF16, 3, fused_ne);
    fused.ffn_down_exps = fixture.cached_tensor(GGML_TYPE_BF16, 3, down_ne);
    fused.ffn_gate_up_exps_b = fixture.cached_tensor(GGML_TYPE_F32, 2, fused_bias_ne);
    fused.ffn_down_exps_b = fixture.cached_tensor(GGML_TYPE_F32, 2, down_bias_ne);

    auto & nvfp4 = model->layers[2];
    nvfp4.ffn_gate_inp = fixture.tensor(GGML_TYPE_F32, 2, router_ne);
    nvfp4.ffn_gate_exps = fixture.cached_tensor(GGML_TYPE_NVFP4, 3, fused_ne);
    nvfp4.ffn_up_exps = fixture.cached_tensor(GGML_TYPE_NVFP4, 3, fused_ne);
    nvfp4.ffn_down_exps = fixture.cached_tensor(GGML_TYPE_NVFP4, 3, fused_ne);
    nvfp4.ffn_gate_exps_in_s = fixture.cached_tensor(GGML_TYPE_F32, 1, scalar_ne);
    nvfp4.ffn_down_shexp = fixture.tensor(GGML_TYPE_BF16, 2, down_ne);

    auto & excluded = model->layers[3];
    excluded.ffn_gate = fixture.tensor(GGML_TYPE_BF16, 2, gate_ne);
    excluded.ffn_up_shexp = fixture.tensor(GGML_TYPE_BF16, 2, gate_ne);
    excluded.ffn_down_shexp = fixture.tensor(GGML_TYPE_BF16, 2, down_ne);
    excluded.ffn_up_chexps = fixture.cached_tensor(GGML_TYPE_Q4_0, 3, gate_ne);
    excluded.ffn_down_chexps = fixture.cached_tensor(GGML_TYPE_Q4_0, 3, down_ne);

    ggml_tensor * opaque_cached = fixture.cached_tensor(GGML_TYPE_Q8_K, 3, gate_ne);
    model->tensors_by_name.push_back({"opaque.cached", opaque_cached});
    model->tensors_by_name.push_back({"opaque.alias", opaque_cached});

    ggml_set_name(separate.ffn_gate_exps, "blk.0.ffn_gate_exps.weight");
    ggml_set_name(separate.ffn_up_exps, "blk.0.ffn_up_exps.weight");
    ggml_set_name(separate.ffn_down_exps, "blk.0.ffn_down_exps.weight");

    llama_adapter_loras loras;
    model->build_moe_sources();
    CHECK(model->moe_sources().size() == 4);
    for (int32_t layer = 0; layer < 4; ++layer) {
        CHECK(model->moe_sources()[layer].layer == layer);
    }
    llama_moe_candidate_snapshot produced(*model, loras);
    const auto & snapshot = produced.get();
    CHECK(snapshot.magic == GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_MAGIC);
    CHECK(snapshot.abi_version == GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_VERSION);
    CHECK(snapshot.struct_size == sizeof(snapshot));
    CHECK(snapshot.n_slots == 12 && snapshot.n_groups == 4);
    CHECK(snapshot.flags == GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_FLAG_NONE);

    const auto & separate_group = snapshot.groups[0];
    CHECK(separate_group.layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE);
    CHECK(separate_group.domain == GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY);
    CHECK(candidate_tensor(snapshot, separate.ffn_gate_exps)->group_index == 0);
    CHECK(candidate_tensor(snapshot, separate.ffn_gate_exps)->role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT);
    CHECK(candidate_tensor(snapshot, separate.ffn_gate_exps)->status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE);
    CHECK(candidate_tensor(snapshot, separate.ffn_gate_exps)->flags & GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_CACHED_BUFFER);
    CHECK(candidate_tensor(snapshot, separate.ffn_down_exps_s)->status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_OUTPUT_SCALE);
    CHECK(candidate_tensor(snapshot, separate.ffn_down_exps_b)->status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_OUTPUT_BIAS);
    CHECK(candidate_tensor(snapshot, separate.ffn_gate_exps_in_s)->status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_INPUT_SCALE);
    CHECK(candidate_tensor(snapshot, separate.ffn_gate_exps_in_s)->role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_INPUT_SCALE);

    const auto & fused_group = snapshot.groups[1];
    CHECK(fused_group.layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP);
    CHECK(candidate_tensor(snapshot, fused.ffn_gate_up_exps)->role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT);
    CHECK(candidate_tensor(snapshot, fused.ffn_gate_up_exps_b)->role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_BIAS);

    const auto & chunk_group = snapshot.groups[3];
    CHECK(chunk_group.layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_UNGATED);
    CHECK(chunk_group.domain == GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_CHUNK);
    CHECK(candidate_tensor(snapshot, excluded.ffn_up_chexps)->group_index == 3);
    CHECK(candidate_tensor(snapshot, excluded.ffn_up_chexps)->status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE);
    CHECK(candidate_tensor(snapshot, separate.ffn_up_shexp)->status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_EXCLUDED_SHARED);
    CHECK(candidate_tensor(snapshot, separate.ffn_gate)->status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_EXCLUDED_DENSE);
    CHECK(candidate_tensor(snapshot, opaque_cached)->status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_UNCLASSIFIED);
    CHECK(candidate_tensor(snapshot, opaque_cached)->group_index == UINT32_MAX);

    ggml_cuda_moe_grouped_context registry(&fixture.owner);
    CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(registry.state().n_groups == 3 && registry.state().n_weights == 8);
    CHECK(!registry.find_weight(excluded.ffn_up_chexps, nullptr));
    CHECK(!registry.find_weight(separate.ffn_gate_exps_in_s, nullptr));
    CHECK(!registry.find_weight(opaque_cached, nullptr));

    std::array<ggml_backend_moe_candidate_bank_v1, 9> separate_v1 = {{
        {separate.ffn_gate_exps, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT, 0},
        {separate.ffn_up_exps, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT, 0},
        {separate.ffn_down_exps, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
        {separate.ffn_gate_exps_s, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_SCALE, 0},
        {separate.ffn_up_exps_s, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_SCALE, 0},
        {separate.ffn_down_exps_s, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_SCALE, 0},
        {separate.ffn_gate_exps_b, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_BIAS, 0},
        {separate.ffn_up_exps_b, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_BIAS, 0},
        {separate.ffn_down_exps_b, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_BIAS, 0},
    }};
    std::array<ggml_backend_moe_candidate_bank_v1, 4> fused_v1 = {{
        {fused.ffn_gate_up_exps, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, 0},
        {fused.ffn_down_exps, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
        {fused.ffn_gate_up_exps_b, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_BIAS, 0},
        {fused.ffn_down_exps_b, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_BIAS, 0},
    }};
    std::array<ggml_backend_moe_candidate_bank_v1, 3> nvfp4_v1 = {{
        {nvfp4.ffn_gate_exps, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT, 0},
        {nvfp4.ffn_up_exps, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT, 0},
        {nvfp4.ffn_down_exps, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
    }};
    std::array<ggml_backend_moe_candidate_group_v1, 3> groups_v1 = {{
        {separate_v1.data(), separate_v1.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 0, 0},
        {fused_v1.data(), fused_v1.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 0, 0},
        {nvfp4_v1.data(), nvfp4_v1.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 0, 0},
    }};
    ggml_cuda_moe_grouped_context v1_registry(&fixture.owner);
    const auto v1_snapshot = candidate_snapshot(12, groups_v1.data(), groups_v1.size());
    CHECK(v1_registry.replace(&v1_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    const auto v1_state = v1_registry.state();
    const auto v2_state = registry.state();
    CHECK(v1_state.n_groups == v2_state.n_groups && v1_state.n_weights == v2_state.n_weights);
    CHECK(v1_state.logical_signature == v2_state.logical_signature);
    CHECK(v1_state.slot_bound_bytes == v2_state.slot_bound_bytes);
    CHECK(v1_state.permanent_candidate_bytes == v2_state.permanent_candidate_bytes);

    llama_adapter_lora adapter(model.get());
    adapter.ab_map.emplace(separate.ffn_gate_exps->name, llama_adapter_lora_weight());
    loras.emplace(&adapter, 1.0f);
    model->build_moe_sources();
    llama_moe_candidate_snapshot lora_on(*model, loras);
    CHECK(lora_on.get().n_groups == 4);
    CHECK(lora_on.get().groups[0].flags & GGML_BACKEND_MOE_CANDIDATE_GROUP_V2_FLAG_ACTIVE_LORA);
    CHECK(candidate_tensor(lora_on.get(), separate.ffn_gate_exps)->flags & GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_ACTIVE_LORA);
    CHECK(registry.replace(&lora_on.get()) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(registry.state().n_groups == 2);
    CHECK(!registry.find_weight(separate.ffn_gate_exps, nullptr));
    loras.clear();
    model->build_moe_sources();
    llama_moe_candidate_snapshot lora_off(*model, loras);
    CHECK(lora_off.get().n_groups == 4);
    CHECK(registry.replace(&lora_off.get()) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(registry.find_weight(separate.ffn_gate_exps, nullptr));

    fused.ffn_up_exps_s = fixture.cached_tensor(GGML_TYPE_F32, 1, scale_ne);
    model->build_moe_sources();
    llama_moe_candidate_snapshot fused_scale(*model, loras);
    CHECK(fused_scale.get().n_groups == 4);
    const auto * fused_scale_record = candidate_tensor(fused_scale.get(), fused.ffn_up_exps_s);
    CHECK(fused_scale_record != nullptr);
    CHECK(fused_scale_record->role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_SCALE);
    CHECK(fused_scale_record->status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_OUTPUT_SCALE);
    CHECK(registry.replace(&fused_scale.get()) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(registry.state().n_groups == 2 && registry.state().n_weights == 6);
    CHECK(!registry.find_down_group_key(fused.ffn_down_exps, nullptr));
    CHECK(!registry.find_weight(fused.ffn_gate_up_exps, nullptr));
    fused.ffn_up_exps_s = nullptr;

    ggml_tensor * saved_shared = separate.ffn_up_shexp;
    separate.ffn_up_shexp = separate.ffn_gate_exps;
    model->build_moe_sources();
    llama_moe_candidate_snapshot typed_alias(*model, loras);
    CHECK(typed_alias.get().flags & GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_FLAG_INCOMPLETE);
    CHECK(typed_alias.get().groups[0].flags & GGML_BACKEND_MOE_CANDIDATE_GROUP_V2_FLAG_INCOMPLETE);
    CHECK(candidate_tensor(typed_alias.get(), separate.ffn_gate_exps)->status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE);
    CHECK(registry.replace(&typed_alias.get()) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(registry.state().n_groups == 0);
    separate.ffn_up_shexp = saved_shared;
    CHECK(registry.replace(&lora_off.get()) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);

    llama_model_tensor_buft_override overrides[] = {{".*", ggml_backend_cpu_buffer_type()}, {nullptr, nullptr}};
    params.tensor_buft_overrides = overrides;
    params.moe_expert_cache_slots = 48;
    std::unique_ptr<llama_model> overridden(llama_model_create(LLM_ARCH_LLAMA, params));
    overridden->layers = model->layers;
    CHECK(overridden->has_tensor_overrides());
    ggml_backend_buffer_t saved_buffer = separate.ffn_gate_exps->buffer;
    separate.ffn_gate_exps->buffer = fixture.buffer;
    overridden->build_moe_sources();
    llama_moe_candidate_snapshot affected(*overridden, loras);
    CHECK(affected.get().n_slots == 48 && affected.get().n_groups == 4);
    CHECK(affected.get().flags == GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_FLAG_NONE);
    CHECK(affected.get().groups[0].flags & GGML_BACKEND_MOE_CANDIDATE_GROUP_V2_FLAG_TENSOR_OVERRIDES);
    CHECK(candidate_tensor(affected.get(), separate.ffn_gate_exps)->flags &
        GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_TENSOR_OVERRIDES);
    CHECK((candidate_tensor(affected.get(), separate.ffn_gate_exps)->flags &
        GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_CACHED_BUFFER) == 0);
    CHECK(registry.replace(&affected.get()) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(registry.state().accepted == 1 && registry.state().n_groups == 2 && registry.state().n_slots == 48);

    separate.ffn_gate_exps->buffer = saved_buffer;
    overridden->build_moe_sources();
    llama_moe_candidate_snapshot shadowed(*overridden, loras);
    CHECK(shadowed.get().flags == GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_FLAG_NONE);
    CHECK((shadowed.get().groups[0].flags & GGML_BACKEND_MOE_CANDIDATE_GROUP_V2_FLAG_TENSOR_OVERRIDES) == 0);
    CHECK((candidate_tensor(shadowed.get(), separate.ffn_gate_exps)->flags &
        GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_TENSOR_OVERRIDES) == 0);
    CHECK(candidate_tensor(shadowed.get(), separate.ffn_gate_exps)->flags &
        GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_CACHED_BUFFER);
    CHECK(registry.replace(&shadowed.get()) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(registry.state().accepted == 1 && registry.state().n_groups == 3 && registry.state().n_slots == 48);

    llama_adapter_lora overridden_adapter(overridden.get());
    overridden_adapter.ab_map.emplace(separate.ffn_gate_exps->name, llama_adapter_lora_weight());
    loras.emplace(&overridden_adapter, 1.0f);
    overridden->build_moe_sources();
    llama_moe_candidate_snapshot shadowed_lora(*overridden, loras);
    CHECK(shadowed_lora.get().flags == GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_FLAG_NONE);
    CHECK((shadowed_lora.get().groups[0].flags & GGML_BACKEND_MOE_CANDIDATE_GROUP_V2_FLAG_TENSOR_OVERRIDES) == 0);
    CHECK(shadowed_lora.get().groups[0].flags & GGML_BACKEND_MOE_CANDIDATE_GROUP_V2_FLAG_ACTIVE_LORA);
    const auto * shadowed_lora_tensor = candidate_tensor(shadowed_lora.get(), separate.ffn_gate_exps);
    CHECK((shadowed_lora_tensor->flags & GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_TENSOR_OVERRIDES) == 0);
    CHECK(shadowed_lora_tensor->flags & GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_CACHED_BUFFER);
    CHECK(shadowed_lora_tensor->flags & GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_ACTIVE_LORA);
    CHECK(registry.replace(&shadowed_lora.get()) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(registry.state().accepted == 1 && registry.state().n_groups == 2 && registry.state().n_slots == 48);
    CHECK(!registry.find_weight(separate.ffn_gate_exps, nullptr));
    loras.clear();

    llama_model_params uncached_params = llama_model_default_params();
    std::unique_ptr<llama_model> uncached(llama_model_create(LLM_ARCH_LLAMA, uncached_params));
    CHECK(uncached != nullptr && uncached->moe_expert_cache_slots() == 0);
    uncached->build_moe_sources();
    llama_moe_candidate_snapshot uncached_snapshot(*uncached, loras);
    CHECK(uncached_snapshot.get().n_slots == 0 && uncached_snapshot.get().n_groups == 0);
    CHECK(registry.replace(&uncached_snapshot.get()) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(registry.state().accepted == 1 && registry.state().n_groups == 0 && registry.state().n_slots == 0);

    llama_model_params gemma_params = llama_model_default_params();
    gemma_params.moe_expert_cache_slots = 12;
    gemma_params.tensor_buft_overrides = overrides;
    std::unique_ptr<llama_model> gemma(llama_model_create(LLM_ARCH_GEMMA4, gemma_params));
    CHECK(gemma != nullptr && gemma->has_tensor_overrides());
    gemma->layers.resize(30);
    for (auto & layer : gemma->layers) {
        layer.ffn_gate_inp = fixture.tensor(GGML_TYPE_F32, 2, router_ne);
        layer.ffn_gate_up_exps = fixture.cached_tensor(GGML_TYPE_Q4_0, 3, fused_ne);
        layer.ffn_down_exps = fixture.cached_tensor(GGML_TYPE_Q4_0, 3, down_ne);
        layer.ffn_down_exps_s = fixture.cached_tensor(GGML_TYPE_F32, 1, scale_ne);
    }
    gemma->build_moe_sources();
    llama_moe_candidate_snapshot gemma_produced(*gemma, loras);
    const auto & gemma_snapshot = gemma_produced.get();
    CHECK(gemma_snapshot.n_slots == 12 && gemma_snapshot.n_groups == 30);
    CHECK(gemma_snapshot.flags == GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_FLAG_NONE);
    for (uint32_t group_index = 0; group_index < gemma_snapshot.n_groups; ++group_index) {
        const auto & group = gemma_snapshot.groups[group_index];
        CHECK(group.layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP);
        CHECK(group.domain == GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY);
        CHECK((group.flags & GGML_BACKEND_MOE_CANDIDATE_GROUP_V2_FLAG_TENSOR_OVERRIDES) == 0);
        const auto * gate_up = candidate_tensor(gemma_snapshot, gemma->layers[group_index].ffn_gate_up_exps);
        const auto * down = candidate_tensor(gemma_snapshot, gemma->layers[group_index].ffn_down_exps);
        CHECK(gate_up->group_index == group_index && down->group_index == group_index);
        CHECK((gate_up->flags & GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_TENSOR_OVERRIDES) == 0);
        CHECK((down->flags & GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_TENSOR_OVERRIDES) == 0);
        CHECK(gate_up->flags & GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_CACHED_BUFFER);
        CHECK(down->flags & GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_CACHED_BUFFER);
        CHECK(candidate_tensor(gemma_snapshot, gemma->layers[group_index].ffn_down_exps_s)->status ==
            GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_OUTPUT_SCALE);
    }
    CHECK(registry.replace(&gemma_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    const auto gemma_state = registry.state();
    CHECK(gemma_state.n_groups == 30 && gemma_state.n_weights == 60);
    CHECK(gemma_state.slot_bound_bytes == 30 * 12 * (gemma->layers[0].ffn_gate_up_exps->nb[2] + gemma->layers[0].ffn_down_exps->nb[2]));
    CHECK(gemma_state.permanent_candidate_bytes == 30 * ggml_nbytes(gemma->layers[0].ffn_down_exps_s));
    for (uint32_t group_index = 0; group_index < gemma_snapshot.n_groups; ++group_index) {
        ggml_cuda_moe_candidate_group_key key;
        ggml_cuda_moe_candidate_group_info group_info;
        ggml_cuda_moe_candidate_bank_info scale_info;
        CHECK(registry.find_down_group_key(gemma->layers[group_index].ffn_down_exps, &key));
        CHECK(key.group_index == group_index && registry.get_group(key, &group_info) && group_info.n_banks == 3);
        CHECK(registry.get_bank(key, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_SCALE, &scale_info));
        CHECK(scale_info.movement == GGML_CUDA_MOE_CANDIDATE_MOVEMENT_PERMANENT_CANDIDATE);
        CHECK(scale_info.index_modes == GGML_CUDA_MOE_CANDIDATE_INDEX_ORIGINAL_DIRECT);
    }
    fprintf(stderr, "test-moe-cache: Gemma fused registry 30x(2 slot + 1 original-direct) OK\n");
}

void test_candidate_registry(bool benchmark) {
    candidate_test_fixture fixture;
    ggml_cuda_moe_grouped_context registry(&fixture.owner);

    const int64_t gate_ne[] = {64, 32, 4};
    const int64_t down_ne[] = {32, 64, 4};
    const int64_t scale_ne[] = {4};
    const int64_t bias_ne[] = {64, 4};
    const int64_t gate_bias_ne[] = {32, 4};
    const int64_t ids_ne[] = {2, 1, 1};
    ggml_tensor * gate = fixture.tensor(GGML_TYPE_Q4_0, 3, gate_ne);
    ggml_tensor * up = fixture.tensor(GGML_TYPE_Q4_0, 3, gate_ne);
    ggml_tensor * down = fixture.tensor(GGML_TYPE_Q4_0, 3, down_ne);
    ggml_tensor * down_scale = fixture.tensor(GGML_TYPE_F32, 1, scale_ne);
    ggml_tensor * down_bias = fixture.tensor(GGML_TYPE_F32, 2, bias_ne);
    ggml_tensor * gate_bias = fixture.tensor(GGML_TYPE_F32, 2, gate_bias_ne);
    ggml_tensor * up_bias = fixture.tensor(GGML_TYPE_F32, 2, gate_bias_ne);
    ggml_tensor * ids = fixture.tensor(GGML_TYPE_I32, 3, ids_ne);
    ggml_tensor * other_ids = fixture.tensor(GGML_TYPE_I32, 3, ids_ne);

    std::array<ggml_backend_moe_candidate_bank_v1, 5> banks = {{
        {gate, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT, 0},
        {up, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT, 0},
        {down, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
        {down_scale, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_SCALE, 0},
        {down_bias, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_BIAS, 0},
    }};
    ggml_backend_moe_candidate_group_v1 group = {banks.data(), banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 0, 0};
    auto snapshot = candidate_snapshot(12, &group, 1);

    CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    auto state = registry.state();
    CHECK(state.accepted == 1 && state.generation == 1 && state.n_slots == 12);
    CHECK(state.n_groups == 1 && state.n_weights == 3 && state.rejection == GGML_CUDA_MOE_CANDIDATE_REJECT_NONE);
    CHECK(state.slot_bound_bytes == 12 * (gate->nb[2] + up->nb[2] + down->nb[2]));
    CHECK(state.permanent_candidate_bytes == ggml_nbytes(down_scale) + ggml_nbytes(down_bias));
    const uint64_t logical_signature = state.logical_signature;

    uint32_t group_index = UINT32_MAX;
    CHECK(registry.find_down_group(down, &group_index) && group_index == 0);
    CHECK(!registry.find_down_group(gate, nullptr));
    ggml_cuda_moe_candidate_group_key group_key;
    CHECK(registry.find_down_group_key(down, &group_key));
    CHECK(group_key.generation == 1 && group_key.group_index == 0);
    ggml_cuda_moe_candidate_group_info group_info;
    CHECK(registry.get_group(group_key, &group_info));
    CHECK(group_info.down == down && group_info.layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE && group_info.n_banks == banks.size() && group_info.n_slots == 12);
    ggml_cuda_moe_candidate_bank_info info;
    CHECK(registry.find_weight(gate, &info));
    CHECK(info.generation == 1 && info.group_index == 0 && info.tensor == gate && info.source_data == gate->data && info.type == GGML_TYPE_Q4_0);
    CHECK(info.encoding == GGML_CUDA_MOE_CANDIDATE_ENCODING_PLAIN);
    CHECK(info.movement == GGML_CUDA_MOE_CANDIDATE_MOVEMENT_SLOT_BOUND);
    CHECK(info.index_modes == (GGML_CUDA_MOE_CANDIDATE_INDEX_GROUP_SLOT_DIRECT | GGML_CUDA_MOE_CANDIDATE_INDEX_ORIGINAL_SOURCE_MAP));
    CHECK(!registry.find_weight(down_scale, nullptr));
    CHECK(registry.get_bank(group_key, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_BIAS, &info));
    CHECK(info.tensor == down_bias && info.source_data == down_bias->data && info.movement == GGML_CUDA_MOE_CANDIDATE_MOVEMENT_PERMANENT_CANDIDATE);
    CHECK(!registry.get_bank(group_key, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, nullptr));

    ggml_cuda_moe_candidate_probe_input probe = {};
    ggml_cuda_moe_candidate_probe_result probe_result;
    probe.n_banks = 1;
    probe.banks[0].weight = gate;
    probe.banks[0].ids = ids;
    CHECK(registry.probe(probe, &probe_result));
    CHECK(probe_result.key.generation == 1 && probe_result.key.group_index == 0 && probe_result.roles[0] == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT);
    probe.banks[0].weight = other_ids;
    CHECK(!registry.probe(probe, nullptr));
    probe.banks[0].weight = gate;
    probe.banks[0].expected_role = GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT;
    CHECK(!registry.probe(probe, nullptr));

    probe = {};
    probe.n_banks = 2;
    probe.exact_auxiliaries = 1;
    probe.banks[0] = {up, ids, nullptr, nullptr, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT};
    probe.banks[1] = {gate, ids, nullptr, nullptr, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT};
    CHECK(registry.probe(probe, &probe_result));
    probe.banks[1].ids = other_ids;
    CHECK(!registry.probe(probe, nullptr));
    probe.banks[1].ids = ids;
    probe.banks[1].weight = nullptr;
    CHECK(!registry.probe(probe, nullptr));
    probe.banks[1].weight = gate;
    ggml_tensor copied_gate = *gate;
    probe.banks[1].weight = &copied_gate;
    CHECK(!registry.probe(probe, nullptr));

    probe = {};
    probe.n_banks = 1;
    probe.exact_auxiliaries = 1;
    probe.banks[0] = {down, ids, down_scale, down_bias, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT};
    CHECK(registry.probe(probe, &probe_result));
    probe.banks[0].bias = nullptr;
    CHECK(!registry.probe(probe, nullptr));
    probe.banks[0].bias = down_bias;
    ggml_tensor copied_scale = *down_scale;
    probe.banks[0].scale = &copied_scale;
    CHECK(!registry.probe(probe, nullptr));
    probe.banks[0].scale = down_scale;
    probe.expected_generation = probe_result.key.generation;

    snapshot.n_slots = 48;
    CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    state = registry.state();
    CHECK(state.generation == 2 && state.n_slots == 48 && state.logical_signature == logical_signature);
    CHECK(state.slot_bound_bytes == 48 * (gate->nb[2] + up->nb[2] + down->nb[2]));
    CHECK(!registry.get_group(group_key, nullptr) && !registry.get_bank(group_key, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, nullptr));
    CHECK(registry.find_down_group_key(down, &group_key) && group_key.generation == 2);
    CHECK(registry.get_group(group_key, &group_info) && group_info.n_slots == 48);
    CHECK(!registry.probe(probe, nullptr));
    probe.expected_generation = 0;
    CHECK(registry.probe(probe, &probe_result) && probe_result.key.generation == 2);
    snapshot.n_slots = 12;
    CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    state = registry.state();
    CHECK(state.generation == 3 && state.n_slots == 12 && state.logical_signature == logical_signature);
    CHECK(!registry.get_group(group_key, nullptr));

    auto expect_rejected = [&](ggml_cuda_moe_candidate_rejection rejection) {
        const uint64_t generation = registry.state().generation;
        CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_REJECTED);
        const auto rejected = registry.state();
        CHECK(rejected.generation == generation + 1 && rejected.accepted == 0 && rejected.n_groups == 0);
        CHECK(rejected.rejection == rejection && !registry.find_weight(gate, nullptr));
    };

    group.flags = 1;
    expect_rejected(GGML_CUDA_MOE_CANDIDATE_REJECT_INVALID_FLAGS);
    group.flags = 0;
    group.reserved = 1;
    expect_rejected(GGML_CUDA_MOE_CANDIDATE_REJECT_INVALID_FLAGS);
    group.reserved = 0;
    banks[0].reserved = 1;
    expect_rejected(GGML_CUDA_MOE_CANDIDATE_REJECT_INVALID_FLAGS);
    banks[0].reserved = 0;
    snapshot.flags = 1;
    expect_rejected(GGML_CUDA_MOE_CANDIDATE_REJECT_INVALID_FLAGS);
    snapshot.flags = 0;
    snapshot.reserved[0] = 1;
    expect_rejected(GGML_CUDA_MOE_CANDIDATE_REJECT_INVALID_FLAGS);
    snapshot.reserved[0] = 0;
    CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);

    auto bad_banks = banks;
    bad_banks[1].role = GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT;
    group.banks = bad_banks.data();
    expect_rejected(GGML_CUDA_MOE_CANDIDATE_REJECT_DUPLICATE_ROLE);
    bad_banks = banks;
    bad_banks[1].tensor = gate;
    group.banks = bad_banks.data();
    expect_rejected(GGML_CUDA_MOE_CANDIDATE_REJECT_DUPLICATE_TENSOR);
    group.banks = banks.data();
    CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);

    void * down_data = down->data;
    down->data = static_cast<uint8_t *>(fixture.storage) + candidate_test_fixture::BUFFER_SIZE - 64;
    expect_rejected(GGML_CUDA_MOE_CANDIDATE_REJECT_INVALID_BOUNDS);
    down->data = down_data;

    const size_t down_stride = down->nb[2];
    down->nb[2] += 64;
    expect_rejected(GGML_CUDA_MOE_CANDIDATE_REJECT_INCOMPATIBLE_SHAPE);
    down->nb[2] = down_stride;

    const int64_t unsupported_ne[] = {256, 32, 4};
    ggml_tensor * unsupported = fixture.tensor(GGML_TYPE_I8, 3, unsupported_ne);
    bad_banks = banks;
    bad_banks[0].tensor = unsupported;
    group.banks = bad_banks.data();
    expect_rejected(GGML_CUDA_MOE_CANDIDATE_REJECT_UNSUPPORTED_TYPE);

    group.banks = banks.data();
    group.n_banks = 2;
    expect_rejected(GGML_CUDA_MOE_CANDIDATE_REJECT_INVALID_LAYOUT);
    group.n_banks = banks.size();

    ggml_tensor * block_scale = fixture.tensor(GGML_TYPE_F32, 1, scale_ne);
    std::array<ggml_backend_moe_candidate_bank_v1, 6> block_banks;
    std::copy(banks.begin(), banks.end(), block_banks.begin());
    block_banks[5] = {block_scale, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_BLOCK_SCALE, 0};
    group.banks = block_banks.data();
    group.n_banks = block_banks.size();
    expect_rejected(GGML_CUDA_MOE_CANDIDATE_REJECT_UNSUPPORTED_ROLE);
    group.banks = banks.data();
    group.n_banks = banks.size();

    snapshot.n_groups = GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS + 1;
    expect_rejected(GGML_CUDA_MOE_CANDIDATE_REJECT_INVALID_COUNT);
    snapshot.n_groups = 1;
    fixture.supports_buft = false;
    expect_rejected(GGML_CUDA_MOE_CANDIDATE_REJECT_INACCESSIBLE_SOURCE);
    fixture.supports_buft = true;

    snapshot.magic = 0;
    const uint64_t generation = registry.state().generation;
    CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_INVALID_ABI);
    state = registry.state();
    CHECK(state.generation == generation + 1 && state.accepted == 0 && state.rejection == GGML_CUDA_MOE_CANDIDATE_REJECT_INVALID_ABI);
    snapshot.magic = GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V1_MAGIC;

    const int64_t fused_ne[] = {64, 64, 4};
    ggml_tensor * gate_up_bf16 = fixture.tensor(GGML_TYPE_BF16, 3, fused_ne);
    ggml_tensor * down_bf16 = fixture.tensor(GGML_TYPE_BF16, 3, down_ne);
    std::array<ggml_backend_moe_candidate_bank_v1, 2> fused_banks = {{
        {gate_up_bf16, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, 0},
        {down_bf16, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
    }};
    ggml_backend_moe_candidate_group_v1 fused_group = {fused_banks.data(), fused_banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 0, 0};
    snapshot = candidate_snapshot(12, &fused_group, 1);
    CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(registry.find_weight(gate_up_bf16, &info) && info.type == GGML_TYPE_BF16);
    CHECK(info.index_modes == (GGML_CUDA_MOE_CANDIDATE_INDEX_GROUP_SLOT_DIRECT | GGML_CUDA_MOE_CANDIDATE_INDEX_ORIGINAL_SOURCE_MAP));

    const int64_t q4k_ne[] = {256, 256, 2};
    ggml_tensor * gate_q4k = fixture.tensor(GGML_TYPE_Q4_K, 3, q4k_ne);
    ggml_tensor * up_q4k = fixture.tensor(GGML_TYPE_Q4_K, 3, q4k_ne);
    ggml_tensor * down_q4k = fixture.tensor(GGML_TYPE_Q4_K, 3, q4k_ne);
    std::array<ggml_backend_moe_candidate_bank_v1, 3> q4k_banks = {{
        {gate_q4k, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT, 0},
        {up_q4k, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT, 0},
        {down_q4k, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
    }};
    ggml_backend_moe_candidate_group_v1 q4k_group = {q4k_banks.data(), q4k_banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 0, 0};
    snapshot = candidate_snapshot(12, &q4k_group, 1);
    CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    for (const ggml_tensor * weight : {gate_q4k, up_q4k, down_q4k}) {
        CHECK(registry.find_weight(weight, &info));
        CHECK(info.type == GGML_TYPE_Q4_K && info.encoding == GGML_CUDA_MOE_CANDIDATE_ENCODING_PLAIN);
        CHECK(info.movement == GGML_CUDA_MOE_CANDIDATE_MOVEMENT_SLOT_BOUND);
        CHECK(info.index_modes == (GGML_CUDA_MOE_CANDIDATE_INDEX_GROUP_SLOT_DIRECT | GGML_CUDA_MOE_CANDIDATE_INDEX_ORIGINAL_SOURCE_MAP));
        CHECK(info.byte_extent == ggml_nbytes(weight) && info.expert_stride == weight->nb[2]);
    }
    CHECK(registry.find_down_group(down_q4k, &group_index) && group_index == 0);

    const int64_t fused_q4k_ne[] = {256, 512, 2};
    ggml_tensor * gate_up_q4k = fixture.tensor(GGML_TYPE_Q4_K, 3, fused_q4k_ne);
    ggml_tensor * fused_down_q4k = fixture.tensor(GGML_TYPE_Q4_K, 3, q4k_ne);
    std::array<ggml_backend_moe_candidate_bank_v1, 2> fused_q4k_banks = {{
        {gate_up_q4k, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, 0},
        {fused_down_q4k, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
    }};
    ggml_backend_moe_candidate_group_v1 fused_q4k_group = {fused_q4k_banks.data(), fused_q4k_banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 0, 0};
    snapshot = candidate_snapshot(12, &fused_q4k_group, 1);
    CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    for (const ggml_tensor * weight : {gate_up_q4k, fused_down_q4k}) {
        CHECK(registry.find_weight(weight, &info));
        CHECK(info.type == GGML_TYPE_Q4_K && info.encoding == GGML_CUDA_MOE_CANDIDATE_ENCODING_PLAIN);
        CHECK(info.index_modes == (GGML_CUDA_MOE_CANDIDATE_INDEX_GROUP_SLOT_DIRECT | GGML_CUDA_MOE_CANDIDATE_INDEX_ORIGINAL_SOURCE_MAP));
    }
    CHECK(registry.find_down_group(fused_down_q4k, &group_index) && group_index == 0);

    const int64_t nvfp4_ne[] = {64, 64, 4};
    ggml_tensor * gate_nvfp4 = fixture.tensor(GGML_TYPE_NVFP4, 3, nvfp4_ne);
    ggml_tensor * up_nvfp4 = fixture.tensor(GGML_TYPE_NVFP4, 3, nvfp4_ne);
    ggml_tensor * down_nvfp4 = fixture.tensor(GGML_TYPE_NVFP4, 3, nvfp4_ne);
    std::array<ggml_backend_moe_candidate_bank_v1, 3> nvfp4_banks = {{
        {gate_nvfp4, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT, 0},
        {up_nvfp4, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT, 0},
        {down_nvfp4, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
    }};
    ggml_backend_moe_candidate_group_v1 nvfp4_group = {nvfp4_banks.data(), nvfp4_banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 0, 0};
    snapshot = candidate_snapshot(12, &nvfp4_group, 1);
    CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(registry.find_weight(gate_nvfp4, &info));
    CHECK(info.encoding == GGML_CUDA_MOE_CANDIDATE_ENCODING_NVFP4_COMPOUND);
    CHECK(info.index_modes == GGML_CUDA_MOE_CANDIDATE_INDEX_GROUP_SLOT_DIRECT);

    std::array<ggml_backend_moe_candidate_bank_v1, 5> pair_aux_banks = {{
        {gate, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT, 0},
        {up, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT, 0},
        {down, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
        {gate_bias, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_BIAS, 0},
        {up_bias, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_BIAS, 0},
    }};
    ggml_backend_moe_candidate_group_v1 pair_aux_group = {pair_aux_banks.data(), pair_aux_banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 0, 0};
    snapshot = candidate_snapshot(12, &pair_aux_group, 1);
    CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(!registry.acquire_legacy_cache(gate_bias));
    CHECK(!registry.acquire_legacy_cache(up_bias));
    probe = {};
    probe.n_banks = 2;
    probe.exact_auxiliaries = 1;
    probe.banks[0] = {up, ids, nullptr, up_bias, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT};
    probe.banks[1] = {gate, ids, nullptr, gate_bias, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT};
    CHECK(registry.probe(probe, nullptr));
    std::swap(probe.banks[0].bias, probe.banks[1].bias);
    CHECK(!registry.probe(probe, nullptr));

    std::array<ggml_backend_moe_candidate_group_v1, 2> groups = {pair_aux_group, nvfp4_group};
    snapshot = candidate_snapshot(12, groups.data(), groups.size());
    CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    probe = {};
    probe.n_banks = 2;
    probe.exact_auxiliaries = 1;
    probe.banks[0] = {up, ids, nullptr, nullptr, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT};
    probe.banks[1] = {gate_nvfp4, ids, nullptr, nullptr, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT};
    CHECK(!registry.probe(probe, nullptr));

    probe = {};
    probe.n_banks = 1;
    probe.banks[0].weight = gate_nvfp4;
    probe.banks[0].ids = ids;
    if (benchmark) {
        constexpr uint32_t n_probes = 200000;
        const auto benchmark_probe = [&](const char * label) {
            uint32_t n_matches = 0;
            const auto begin = std::chrono::steady_clock::now();
            for (uint32_t i = 0; i < n_probes; ++i) {
                n_matches += registry.probe(probe, nullptr) ? 1 : 0;
            }
            const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin).count();
            CHECK(n_matches == n_probes);
            fprintf(stderr, "test-moe-cache: registered %s shadow %.1f ns/probe\n", label, static_cast<double>(elapsed) / n_probes);
        };
        benchmark_probe("one-bank");

        probe = {};
        probe.n_banks = 2;
        probe.banks[0] = {up, ids, nullptr, nullptr, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT};
        probe.banks[1] = {gate, ids, nullptr, nullptr, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT};
        benchmark_probe("pair");
        probe.exact_auxiliaries = 1;
        probe.banks[0].bias = up_bias;
        probe.banks[1].bias = gate_bias;
        benchmark_probe("pair-auxiliary");
    }

    ggml_cuda_moe_grouped_context other_registry(&fixture.owner);
    snapshot.n_slots = 48;
    CHECK(other_registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(other_registry.state().generation == 1 && other_registry.state().n_slots == 48);
    CHECK(registry.state().n_slots == 12 && registry.state().generation > 1);

    fprintf(stderr, "test-moe-cache: registry OK\n");
}

void test_moe_route_publication_lifetime() {
    constexpr int N_EXPERTS = 64;
    constexpr int N_USED = 6;
    constexpr int N_ROUTES = 2 * GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS;
    ggml_backend_ptr backend(ggml_backend_cuda_init(0));
    ggml_context_ptr ctx(ggml_init({ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(32, false), nullptr, true}));
    CHECK(backend != nullptr && ctx != nullptr);
    const auto snapshot = candidate_snapshot(16, nullptr, 0);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v1(backend.get(), &snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    ggml_tensor * logits = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, N_EXPERTS, 1);
    ggml_tensor * probs = ggml_soft_max(ctx.get(), logits);
    ggml_tensor * ids = ggml_argsort_top_k(ctx.get(), probs, N_USED);
    ggml_tensor * weights = ggml_get_rows(ctx.get(), ggml_reshape_3d(ctx.get(), probs, 1, N_EXPERTS, 1), ids);
    ggml_tensor * storage = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, N_EXPERTS, N_ROUTES);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, weights);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    CHECK(buffer != nullptr);
    std::array<float, N_EXPERTS> values;
    float sum = 0.0f;
    for (int i = 0; i < N_EXPERTS; ++i) {
        values[i] = i * 0.01f;
        sum += std::exp(values[i]);
    }
    ggml_backend_tensor_set(logits, values.data(), 0, sizeof(values));
    for (int i = 0; i < N_ROUTES; ++i) {
        ids->view_src->data = static_cast<char *>(storage->data) + i * storage->nb[1];
        ids->data = ids->view_src->data;
        CHECK(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(backend.get());
        CHECK(ggml_cuda_moe_ids_cache_count_for_test(backend.get()) ==
            static_cast<size_t>(std::min(i + 1, static_cast<int>(GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS))));
        std::array<float, N_USED> actual;
        ggml_backend_tensor_get(weights, actual.data(), 0, sizeof(actual));
        for (int route = 0; route < N_USED; ++route) {
            CHECK(std::abs(actual[route] - std::exp(values[N_EXPERTS - route - 1]) / sum) < 1e-6f);
        }
    }
    ggml_backend_ptr other(ggml_backend_cuda_init(0));
    CHECK(other != nullptr && ggml_cuda_moe_ids_cache_count_for_test(other.get()) == 0);
    CHECK(ggml_backend_graph_compute(other.get(), graph) == GGML_STATUS_SUCCESS);
    CHECK(ggml_cuda_moe_ids_cache_count_for_test(other.get()) == 0);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v1(other.get(), &snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(ggml_backend_graph_compute(other.get(), graph) == GGML_STATUS_SUCCESS);
    CHECK(ggml_cuda_moe_ids_cache_count_for_test(other.get()) == 1);
    other.reset();
    CHECK(ggml_cuda_moe_ids_cache_count_for_test(backend.get()) == GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS);
    CHECK(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS);
    backend.reset();
    fprintf(stderr, "test-moe-cache: bounded route publication and backend ownership OK\n");
}

void test_moe_cache_proc_api() {
    ggml_backend_reg_t reg = ggml_backend_cuda_reg();
    CHECK(reg != nullptr);
    CHECK(ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_CACHE_BUFFER_TYPE_PROC_NAME) != nullptr);
    CHECK(ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_CACHE_BOUNDED_BUFFER_TYPE_PROC_NAME) != nullptr);
    CHECK(ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_CACHE_FREE_BUFFER_TYPE_PROC_NAME) != nullptr);
    CHECK(ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_CACHE_CONFIGURE_SOURCES_PROC_NAME) != nullptr);
    CHECK(ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_CACHE_IS_BUFFER_TYPE_PROC_NAME) != nullptr);
    CHECK(ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_CACHE_BUFFER_FROM_HOST_PTR_PROC_NAME) != nullptr);
    CHECK(ggml_backend_reg_get_proc_address(reg, "ggml_backend_moe_cache_set_slots") == nullptr);
    CHECK(ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_CACHE_SET_DEBUG_PROC_NAME) != nullptr);
    CHECK(ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_CACHE_LOG_AND_RESET_STATS_PROC_NAME) != nullptr);
    CHECK(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_set_decode_boundary_overlap") != nullptr);
    fprintf(stderr, "test-moe-cache: dynamic backend procedure API OK\n");
}
