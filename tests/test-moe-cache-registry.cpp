#include "test-moe-cache.h"
#include "ggml-cpu/ops.h"
#include "ggml-cpu/moe-fidelity.h"
#include "moe-fidelity-config.h"

#include "../src/llama-batch.h"
#include "../src/llama-context.h"
#include "../src/llama-model.h"
#include "../src/llama-vocab.h"

#include <condition_variable>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>

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
    const size_t budget = 64 * 1024 * 1024;
    params.moe_expert_cache_byte_budgets = &budget;
    params.n_moe_expert_cache_byte_budgets = 1;
    rejected = false;
    try {
        llama_model_free(llama_model_create(LLM_ARCH_DEEPSEEK4, params));
    } catch (const std::runtime_error & error) {
        rejected = std::string(error.what()).find("MoE expert caching does not support tensor split") != std::string::npos;
    }
    CHECK(rejected);
    params.moe_expert_cache_byte_budgets = nullptr;
    params.n_moe_expert_cache_byte_budgets = 0;
    llama_model * model = llama_model_create(LLM_ARCH_DEEPSEEK4, params);
    CHECK(model != nullptr);
    llama_model_free(model);

    params.split_mode = LLAMA_SPLIT_MODE_LAYER;
    params.moe_expert_cache_slots = 8;
    model = llama_model_create(LLM_ARCH_DEEPSEEK4, params);
    CHECK(model != nullptr);
    CHECK(model->moe_expert_cache_slots(nullptr) == 8);
    llama_model_free(model);

    llama_moe_cache_memory memory = {};
    memory.fixed_device_bytes = 100;
    memory.per_slot_device_bytes = 20;
    memory.max_slots = 8;
    CHECK(memory.device_bytes(3) == 160);
    // max_slots caps automatic byte-budget derivation; legacy slot mode still
    // accounts for the exact user-requested allocation.
    CHECK(memory.device_bytes(9) == 280);

    memory.fixed_device_bytes = std::numeric_limits<size_t>::max() - 5;
    memory.per_slot_device_bytes = 6;
    memory.max_slots = 1;
    rejected = false;
    try {
        (void) memory.device_bytes(1);
    } catch (const std::overflow_error &) {
        rejected = true;
    }
    CHECK(rejected);
    fprintf(stderr, "test-moe-cache: unsupported tensor/cache combination rejected before weights OK\n");
}

static ggml_backend_moe_source_lease_v1 source_lease() {
    ggml_backend_moe_source_lease_v1 lease = {};
    lease.struct_size = sizeof(lease);
    lease.abi_version = GGML_BACKEND_MOE_SOURCE_OWNER_V1_VERSION;
    return lease;
}

void test_moe_source_lifetime() {
    const llama_model_params params = llama_model_default_params();
    std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_DEEPSEEK4, params));
    ggml_backend_moe_source_owner_v1 owner = {};
    CHECK(model->moe_source_owner_v1(&owner));
    CHECK(owner.struct_size == sizeof(owner));
    CHECK(owner.abi_version == GGML_BACKEND_MOE_SOURCE_OWNER_V1_VERSION);
    CHECK(owner.generation != 0 && owner.flags == GGML_BACKEND_MOE_SOURCE_OWNER_FLAG_V1_NONE);
    CHECK(owner.retain != nullptr && owner.release != nullptr && owner.validate_span != nullptr);

    ggml_context_ptr source_context(ggml_init({ggml_tensor_overhead() * 2, nullptr, true}));
    CHECK(source_context != nullptr);
    ggml_tensor * source_tensor = ggml_new_tensor_3d(source_context.get(), GGML_TYPE_Q4_0, 256, 256, 4);
    ggml_backend_buffer_ptr source_buffer(
        ggml_backend_alloc_ctx_tensors_from_buft(source_context.get(), ggml_backend_cpu_buffer_type()));
    CHECK(source_buffer != nullptr);
    CHECK(model->record_moe_readable_source(source_tensor, source_tensor->data, ggml_nbytes(source_tensor)));
    ggml_backend_moe_source_span_v1 source_span = {};
    source_span.struct_size   = sizeof(source_span);
    source_span.abi_version   = GGML_BACKEND_MOE_SOURCE_OWNER_V1_VERSION;
    source_span.witness       = source_tensor;
    source_span.data          = source_tensor->data;
    source_span.bytes         = ggml_nbytes(source_tensor);
    source_span.expert_stride = source_tensor->nb[2];
    source_span.type          = source_tensor->type;
    std::copy(std::begin(source_tensor->ne), std::end(source_tensor->ne), std::begin(source_span.ne));
    std::copy(std::begin(source_tensor->nb), std::end(source_tensor->nb), std::begin(source_span.nb));
    CHECK(owner.validate_span(&owner, owner.generation, &source_span) == GGML_BACKEND_MOE_SOURCE_STATUS_V1_OK);
    ++source_span.bytes;
    CHECK(owner.validate_span(&owner, owner.generation, &source_span) == GGML_BACKEND_MOE_SOURCE_STATUS_V1_INVALID_ARGUMENT);
    --source_span.bytes;
    ++source_span.nb[1];
    CHECK(owner.validate_span(&owner, owner.generation, &source_span) == GGML_BACKEND_MOE_SOURCE_STATUS_V1_INVALID_ARGUMENT);
    --source_span.nb[1];
    source_span.data = static_cast<const uint8_t *>(source_span.data) + 1;
    CHECK(owner.validate_span(&owner, owner.generation, &source_span) == GGML_BACKEND_MOE_SOURCE_STATUS_V1_INVALID_ARGUMENT);
    source_span.data = source_tensor->data;
    source_span.witness = source_context.get();
    CHECK(owner.validate_span(&owner, owner.generation, &source_span) == GGML_BACKEND_MOE_SOURCE_STATUS_V1_INVALID_ARGUMENT);
    source_span.witness = source_tensor;
    CHECK(owner.validate_span(&owner, owner.generation + 1, &source_span) ==
          GGML_BACKEND_MOE_SOURCE_STATUS_V1_GENERATION_MISMATCH);

    auto lease = source_lease();
    CHECK(owner.retain(&owner, owner.generation, &lease) == GGML_BACKEND_MOE_SOURCE_STATUS_V1_OK);
    CHECK(owner.retain(&owner, owner.generation, &lease) == GGML_BACKEND_MOE_SOURCE_STATUS_V1_INVALID_ARGUMENT);
    lease.reserved[0] = 1;
    CHECK(owner.release(&lease) == GGML_BACKEND_MOE_SOURCE_STATUS_V1_INVALID_ARGUMENT);
    lease.reserved[0] = 0;
    auto stale = lease;
    CHECK(owner.release(&lease) == GGML_BACKEND_MOE_SOURCE_STATUS_V1_OK);
    CHECK(owner.release(&lease) == GGML_BACKEND_MOE_SOURCE_STATUS_V1_ALREADY_RELEASED);
    CHECK(owner.release(&stale) == GGML_BACKEND_MOE_SOURCE_STATUS_V1_ALREADY_RELEASED);

    lease = source_lease();
    CHECK(owner.retain(&owner, owner.generation + 1, &lease) == GGML_BACKEND_MOE_SOURCE_STATUS_V1_GENERATION_MISMATCH);
    CHECK(lease.owner == nullptr && lease.lease_id == 0);

    std::vector<ggml_backend_moe_source_lease_v1> capacity_leases(GGML_BACKEND_MOE_SOURCE_MAX_LEASES_V1);
    for (auto & capacity_lease : capacity_leases) {
        capacity_lease = source_lease();
        CHECK(owner.retain(&owner, owner.generation, &capacity_lease) == GGML_BACKEND_MOE_SOURCE_STATUS_V1_OK);
    }
    lease = source_lease();
    CHECK(owner.retain(&owner, owner.generation, &lease) == GGML_BACKEND_MOE_SOURCE_STATUS_V1_CAPACITY);
    for (auto & capacity_lease : capacity_leases) {
        CHECK(owner.release(&capacity_lease) == GGML_BACKEND_MOE_SOURCE_STATUS_V1_OK);
    }
    model->close_moe_source_owner();
    model->close_moe_source_owner();
    CHECK(owner.retain(&owner, owner.generation, &lease) == GGML_BACKEND_MOE_SOURCE_STATUS_V1_CLOSED);
    model.reset();

    llama_model * draining_model = llama_model_create(LLM_ARCH_DEEPSEEK4, params);
    CHECK(draining_model != nullptr);
    owner = {};
    CHECK(draining_model->moe_source_owner_v1(&owner));
    lease = source_lease();
    CHECK(owner.retain(&owner, owner.generation, &lease) == GGML_BACKEND_MOE_SOURCE_STATUS_V1_OK);
    draining_model->close_moe_source_owner();
    std::atomic<bool> destructor_started { false };
    std::atomic<bool> destructor_done { false };
    std::thread destroyer([&] {
        destructor_started.store(true, std::memory_order_release);
        llama_model_free(draining_model);
        destructor_done.store(true, std::memory_order_release);
    });
    while (!destructor_started.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    CHECK(!destructor_done.load(std::memory_order_acquire));
    stale = lease;
    ++stale.owner_generation;
    CHECK(owner.release(&stale) == GGML_BACKEND_MOE_SOURCE_STATUS_V1_GENERATION_MISMATCH);
    CHECK(stale.owner == lease.owner && stale.lease_id == lease.lease_id);
    CHECK(!destructor_done.load(std::memory_order_acquire));
    CHECK(owner.release(&lease) == GGML_BACKEND_MOE_SOURCE_STATUS_V1_OK);
    destroyer.join();
    CHECK(destructor_done.load(std::memory_order_acquire));

    std::unique_ptr<llama_model> target(llama_model_create(LLM_ARCH_DEEPSEEK4, params));
    std::unique_ptr<llama_model> draft(llama_model_create(LLM_ARCH_DEEPSEEK4, params));
    ggml_backend_moe_source_owner_v1 target_owner = {};
    CHECK(target->moe_source_owner_v1(&target_owner));
    lease = source_lease();
    CHECK(target_owner.retain(&target_owner, target_owner.generation, &lease) == GGML_BACKEND_MOE_SOURCE_STATUS_V1_OK);
    CHECK(target_owner.release(&lease) == GGML_BACKEND_MOE_SOURCE_STATUS_V1_OK);

    draft->record_shared_tensor(
        "borrowed.weight", 4, GGML_TYPE_F32,
        { 1, 1, 1, 1 }, { 4, 4, 4, 4 },
        "CPU", "host", "target", "model_shared", "CPU", true, true, "borrowed");
    ggml_backend_moe_source_owner_v1 draft_owner = {};
    CHECK(draft->moe_source_owner_v1(&draft_owner));
    CHECK(draft_owner.flags == GGML_BACKEND_MOE_SOURCE_OWNER_FLAG_V1_BORROWED);
    lease = source_lease();
    CHECK(draft_owner.retain(&draft_owner, draft_owner.generation, &lease) == GGML_BACKEND_MOE_SOURCE_STATUS_V1_BORROWED);
    CHECK(lease.owner == nullptr && lease.lease_id == 0);
    fprintf(stderr, "test-moe-cache: model-owned source lifetime admission and drain OK\n");
}

struct mtp_batch_fixture {
    std::vector<llama_token> token;
    std::vector<float> embd;
    std::vector<llama_pos> pos;
    std::vector<int32_t> n_seq_id;
    std::vector<llama_seq_id> seq_id_data;
    std::vector<llama_seq_id *> seq_id;
    std::vector<int8_t> output;

    void add(llama_seq_id sequence, llama_pos position) {
        token.push_back(0);
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

    llama_batch token_batch() {
        llama_batch result = batch();
        result.token = token.data();
        result.embd = nullptr;
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
    for (llama_context_type context_type : {LLAMA_CONTEXT_TYPE_DEFAULT, LLAMA_CONTEXT_TYPE_DRAFT, LLAMA_CONTEXT_TYPE_MTP}) {
        CHECK(!llama_speculative_grouped_intent_test_access::hybrid_required(nullptr, context_type));
        CHECK(!llama_speculative_grouped_intent_test_access::hybrid_required("off", context_type));
        CHECK(llama_speculative_grouped_intent_test_access::hybrid_required("required", context_type));
        bool rejected = false;
        try {
            (void) llama_speculative_grouped_intent_test_access::hybrid_required("invalid", context_type);
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        CHECK(rejected);
    }
    for (bool required : {false, true}) {
        CHECK(llama_speculative_grouped_intent_test_access::hybrid_execution_supported(required,
            GGML_GRAPH_EXECUTION_DOMAIN_MAIN, GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL));
        CHECK(llama_speculative_grouped_intent_test_access::hybrid_execution_supported(required,
            GGML_GRAPH_EXECUTION_DOMAIN_MAIN, GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT));
        CHECK(llama_speculative_grouped_intent_test_access::hybrid_execution_supported(required,
            GGML_GRAPH_EXECUTION_DOMAIN_MAIN, GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SPECULATIVE));
    }
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

    mtp_batch_fixture target_verification;
    for (llama_seq_id sequence = 0; sequence < 4; ++sequence) {
        target_verification.add_span(sequence, 40 + 10 * sequence, 4);
    }
    llama_vocab vocab;
    llama_vocab_test_access::add_dummy_token(vocab);
    llama_batch_allocr balloc(1);
    CHECK(balloc.init(target_verification.token_batch(), vocab, nullptr, 0, 4, false));
    const llama_ubatch equal = balloc.split_equal(512, true, 0);
    CHECK(llama_speculative_grouped_intent_test_access::matches_target_verification_ubatch(equal, 4));
    balloc.split_reset();
    const llama_ubatch unified = balloc.split_simple(512);
    CHECK(unified.n_seq_tokens == 1 && unified.n_seqs == unified.n_tokens && unified.n_seqs_unq == 4);
    CHECK(llama_speculative_grouped_intent_test_access::matches_target_verification_ubatch(unified, 4));
    balloc.split_reset();
    CHECK(!llama_speculative_grouped_intent_test_access::matches_target_verification_ubatch(
        balloc.split_simple(5), 4));
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
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_PREFILL_STAGED);
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
        CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_STAGED && execution.size() == 1);
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
    {
        ggml_cuda_moe_graph_plan plan;
        ggml_cuda_moe_graph_execution execution;
        compile(invalid, 1206, plan, execution);
        CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED && execution.size() == 1);
        CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_eligible_reason(plan, 0));
    }
    invalid.graph->execution_certificate.n_sequences = 3;
    check_execution_legacy(invalid, 1207);

    auto reduced_multirow = make_graph(2, 2);
    candidate_stamp_execution(reduced_multirow.graph, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
        GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 4, 4);
    check_execution_legacy(reduced_multirow, 1208);

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
        CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_PREFILL_STAGED && !execution.find(isolated.down, nullptr));
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

void test_candidate_routed_matrix() {
    constexpr uint32_t role = GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_ROUTED_WEIGHT;
    constexpr uint32_t cached = GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_CACHED_BUFFER;
    uint32_t cases = 0, eligible = 0, pending = 0;
    for (const int64_t rows : {int64_t(51), int64_t(64)}) {
        for (const auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q4_0,
                GGML_TYPE_Q5_K, GGML_TYPE_Q3_K, GGML_TYPE_IQ1_S, GGML_TYPE_MXFP4, GGML_TYPE_NVFP4}) {
            candidate_test_fixture fixture;
            std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> ids_storage(nullptr, ggml_backend_buffer_free);
            ggml_cuda_moe_grouped_context registry(&fixture.owner, 0);
            const int64_t ne[] = {256, rows, 7};
            auto * weight = fixture.cached_tensor(type, 3, ne);
            ggml_set_name(weight, "arbitrary.routed.storage");
            const ggml_backend_moe_candidate_bank_v1 bank = {weight, role, 0};
            const ggml_backend_moe_candidate_group_v1 group = {
                &bank, 1, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_ROUTED_MATRIX, 0, 0,
            };
            const auto snapshot = candidate_snapshot(2, &group, 1);
            CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
            auto state = registry.state();
            CHECK(state.n_groups == 1 && state.n_weights == 1 && state.slot_bound_bytes == 2 * weight->nb[2]);
            ggml_cuda_moe_candidate_bank_info info;
            CHECK(registry.find_weight(weight, &info) && info.role == role && info.expert_stride == weight->nb[2]);
            ggml_cuda_moe_candidate_group_key key = {info.generation, info.group_index};
            ggml_cuda_moe_candidate_group_info group_info;
            CHECK(registry.get_group(key, &group_info) && group_info.n_banks == 1 && group_info.n_resource_banks == 1);
            ggml_cuda_moe_grouped_acquisition resource, repeated;
            CHECK(registry.acquire_group_resources(key, &resource));
            CHECK(registry.acquire_group_resources(key, &repeated) && repeated.resource_generation == resource.resource_generation);
            ggml_cuda_moe_grouped_transaction transaction;
            CHECK(registry.begin_group_transaction(resource, &transaction));
            ggml_cuda_moe_grouped_bank_descriptor descriptor;
            CHECK(registry.get_group_resource_bank(transaction, 0, &descriptor));
            CHECK(descriptor.tensor == weight && descriptor.role == role && descriptor.source_data == weight->data);
            CHECK(descriptor.byte_extent == ggml_nbytes(weight) && descriptor.expert_stride == weight->nb[2]);
            CHECK(!registry.get_group_resource_bank(transaction, 1, nullptr));
            CHECK(registry.end_group_transaction(transaction));

            const ggml_backend_moe_candidate_group_v2 group_v2 = {
                GGML_BACKEND_MOE_CANDIDATE_LAYOUT_ROUTED_MATRIX, GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY, 0, 0,
            };
            const ggml_backend_moe_candidate_tensor_v2 tensor_v2 = {
                weight, 0, role, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE, cached, 0,
            };
            const auto snapshot_v2 = candidate_snapshot_v2(3, &group_v2, 1, &tensor_v2, 1);
            CHECK(registry.replace(&snapshot_v2) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
            CHECK(!registry.get_group(key, nullptr) && !registry.get_group_resources(resource, nullptr));
            CHECK(registry.find_weight(weight, &info));
            key = {info.generation, info.group_index};
            CHECK(registry.acquire_group_resources(key, &resource));
            CHECK(registry.state().slot_bound_bytes == 3 * weight->nb[2]);
            const auto route = candidate_top_k_route(fixture, ne[2], 2);
            ids_storage.reset(ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(0), ggml_nbytes(route.root)));
            CHECK(ids_storage);
            route.root->buffer = nullptr;
            route.root->data = nullptr;
            CHECK(ggml_backend_tensor_alloc(ids_storage.get(), route.root, ggml_backend_buffer_get_base(ids_storage.get())) == GGML_STATUS_SUCCESS);
            route.ids->buffer = nullptr;
            route.ids->data = nullptr;
            CHECK(ggml_backend_view_init(route.ids) == GGML_STATUS_SUCCESS);
            const int32_t selected[] = {1, 3};
            ggml_backend_tensor_set(route.root, selected, 0, sizeof(selected));
            auto * reader = candidate_mmid(fixture, weight, route.ids);
            auto * graph = candidate_graph(fixture, {route.root, route.ids, reader});
            const auto coverage = candidate_certify_graph(registry, graph);
            const uint64_t graph_uid = ggml_graph_next_uid();
            ggml_cuda_moe_graph_plan plan;
            ggml_cuda_moe_graph_execution execution;
            registry.compile_graph_plan(graph, graph_uid, &plan, &execution,
                coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint);
            CHECK(ggml_cuda_moe_grouped_context_test_access::graph_has_complete_mmid_inventory(plan));
            const auto capability = ggml_cuda_moe_grouped_context_test_access::graph_bank_capability(plan, 0, 0);
            if (capability.consumer == GGML_CUDA_MMID_CONSUMER_GENERIC) {
                CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_consumer_equivalence_reason(plan, 0));
                CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_STAGED);
                ++pending;
            } else if (weight->nb[2] % alignof(uint4) != 0) {
                CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_materialization_reason(plan, 0));
                CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_STAGED);
                ++pending;
            } else {
                CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_eligible_reason(plan, 0));
                const int64_t bias_ne[] = {rows, ne[2]};
                auto * bias = fixture.tensor(GGML_TYPE_F32, 2, bias_ne);
                auto * biased = ggml_add_id(fixture.ctx, reader, bias, route.ids);
                const int64_t residual_ne[] = {rows, route.ids->ne[0], route.ids->ne[1]};
                auto * residual = fixture.tensor(GGML_TYPE_F32, 3, residual_ne);
                auto * added = ggml_add(fixture.ctx, reader, residual);
                const int64_t scale_ne[] = {ne[2]};
                auto * scale = fixture.tensor(GGML_TYPE_F32, 1, scale_ne);
                auto * shaped = ggml_reshape_2d(fixture.ctx, scale, 1, ne[2]);
                auto * repeated_scale = ggml_repeat(fixture.ctx, shaped, bias);
                auto * routed_scale = ggml_get_rows(fixture.ctx, repeated_scale, route.ids);
                auto * scaled = ggml_mul(fixture.ctx, reader, routed_scale);
                auto * normalized = ggml_norm(fixture.ctx, reader, 1e-5f);
                graph = candidate_graph(fixture, {route.root, route.ids, reader, biased, added,
                    shaped, repeated_scale, routed_scale, scaled, normalized});
                const auto consumer_coverage = candidate_certify_graph(registry, graph);
                registry.compile_graph_plan(graph, graph_uid, &plan, &execution,
                    consumer_coverage.epoch, consumer_coverage.nodes, consumer_coverage.mmid_count, consumer_coverage.mmid_fingerprint);
                CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_eligible_reason(plan, 0));
                CHECK(ggml_cuda_moe_grouped_context_test_access::graph_has_complete_mmid_inventory(plan));
                auto * dispatch = execution.find_group(reader, nullptr);
                CHECK(dispatch && dispatch->key.n_banks == 1);
                CHECK(dispatch->first_reader == reader && dispatch->last_reader == reader);
                ggml_cuda_moe_graph_execution rebound;
                CHECK(registry.bind_graph_plan(graph, graph_uid, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, plan, &rebound,
                    consumer_coverage.epoch, consumer_coverage.nodes, consumer_coverage.mmid_count, consumer_coverage.mmid_fingerprint));
                dispatch = rebound.find_group(reader, nullptr);
                CHECK(dispatch && dispatch->last_reader == reader);
                const size_t stride = weight->nb[2];
                weight->nb[2] += 1;
                CHECK(!registry.bind_graph_plan(graph, graph_uid, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, plan, &rebound,
                    consumer_coverage.epoch, consumer_coverage.nodes, consumer_coverage.mmid_count, consumer_coverage.mmid_fingerprint));
                weight->nb[2] = stride;
                for (auto * consumer : {biased, added, scaled, normalized}) {
                    const auto saved_op = consumer->op;
                    consumer->op = GGML_OP_NONE;
                    CHECK(!registry.bind_graph_plan(graph, graph_uid, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, plan, &rebound,
                        consumer_coverage.epoch, consumer_coverage.nodes, consumer_coverage.mmid_count, consumer_coverage.mmid_fingerprint));
                    consumer->op = saved_op;
                    auto * saved_source = consumer->src[0];
                    consumer->src[0] = residual;
                    CHECK(!registry.bind_graph_plan(graph, graph_uid, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, plan, &rebound,
                        consumer_coverage.epoch, consumer_coverage.nodes, consumer_coverage.mmid_count, consumer_coverage.mmid_fingerprint));
                    consumer->src[0] = saved_source;
                }
                CHECK(registry.bind_graph_plan(graph, graph_uid, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, plan, &rebound,
                    consumer_coverage.epoch, consumer_coverage.nodes, consumer_coverage.mmid_count, consumer_coverage.mmid_fingerprint));
                cudaStream_t stream = nullptr;
                CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
                CHECK(execution.resolve_streams(candidate_test_graph_stream, stream));
                CHECK(registry.begin_graph_dispatch(&execution, GGML_CUDA_MOE_GRAPH_DISPATCH_DIRECT));
                ggml_cuda_moe_graph_binding binding;
                dispatch = execution.find_group(reader, &binding);
                CHECK(dispatch && registry.prepare_graph_group(dispatch, binding, reader, stream) == GGML_CUDA_MOE_GROUPED_DECODE_READY);
                CHECK(dispatch->bank_data[0] && dispatch->n_slots == 3 && dispatch->remapped_ids);
                CHECK(registry.finish_graph_group(dispatch, binding, reader, stream));
                CHECK(registry.finish_graph_dispatch(&execution));
                CUDA_OK(cudaStreamSynchronize(stream));
                CUDA_OK(cudaStreamDestroy(stream));
                ++eligible;
            }
            const size_t stride = weight->nb[2];
            weight->nb[2] += 1;
            auto invalid = snapshot;
            CHECK(registry.replace(&invalid) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_REJECTED);
            weight->nb[2] = stride;
            CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
            ++cases;
        }
    }
    candidate_test_fixture fixture;
    ggml_cuda_moe_grouped_context registry(&fixture.owner, 0);
    const int64_t ne[] = {96, 51, 1};
    auto * weight = fixture.cached_tensor(GGML_TYPE_F32, 3, ne);
    CHECK(ggml_n_dims(weight) == 2);
    const ggml_backend_moe_candidate_bank_v1 bank = {weight, role, 0};
    const ggml_backend_moe_candidate_group_v1 group = {
        &bank, 1, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_ROUTED_MATRIX, 0, 0,
    };
    const auto snapshot = candidate_snapshot(1, &group, 1);
    CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    ggml_cuda_moe_candidate_bank_info info;
    CHECK(registry.find_weight(weight, &info));
    ggml_cuda_moe_candidate_group_key key = {info.generation, info.group_index};
    ggml_cuda_moe_grouped_acquisition resource;
    CHECK(registry.acquire_group_resources(key, &resource));
    CHECK(cases == 18 && eligible > 0 && pending > 0);
    fprintf(stderr, "test-candidate-routed-matrix: %u source/resource cases, %u eligible graph bindings, %u pending generic consumers, one-expert source OK\n", cases, eligible, pending);
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
    CHECK(!registry.begin_graph_dispatch(&execution, false));

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
        CHECK(group_info.n_resource_banks == 2);
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

    {
        ggml_cuda_moe_grouped_context auxiliary_registry(&fixture.owner, 0);
        auto * gate_scale = fixture.tensor(GGML_TYPE_F32, 1, scale_ne);
        auto * up_scale = fixture.tensor(GGML_TYPE_F32, 1, scale_ne);
        const std::array<ggml_backend_moe_candidate_bank_v1, 9> auxiliary_banks = {{
            banks[0], banks[1], banks[2], banks[3], banks[4],
            {gate_scale, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_SCALE, 0},
            {up_scale, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_SCALE, 0},
            {gate_bias, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_BIAS, 0},
            {up_bias, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_BIAS, 0},
        }};
        const ggml_backend_moe_candidate_group_v1 auxiliary_group = {
            auxiliary_banks.data(), auxiliary_banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 0, 0,
        };
        const auto auxiliary_snapshot = candidate_snapshot(12, &auxiliary_group, 1);
        CHECK(auxiliary_registry.replace(&auxiliary_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        uint64_t permanent_bytes = 0;
        for (size_t i = 3; i < auxiliary_banks.size(); ++i) { permanent_bytes += ggml_nbytes(auxiliary_banks[i].tensor); }
        CHECK(auxiliary_registry.state().permanent_candidate_bytes == permanent_bytes);
        ggml_cuda_moe_candidate_group_key key;
        CHECK(auxiliary_registry.find_down_group_key(down, &key));
        ggml_cuda_moe_grouped_acquisition resource, repeated;
        CHECK(auxiliary_registry.acquire_group_resources(key, &resource));
        CHECK(auxiliary_registry.acquire_group_resources(key, &repeated) && repeated.resource_generation == resource.resource_generation);
        ggml_cuda_moe_grouped_resource_info resource_info;
        CHECK(auxiliary_registry.get_group_resources(resource, &resource_info) && resource_info.n_banks == 3);
        ggml_cuda_moe_grouped_transaction transaction;
        CHECK(auxiliary_registry.begin_group_transaction(resource, &transaction));
        for (uint32_t i = 0; i < 3; ++i) {
            ggml_cuda_moe_grouped_bank_descriptor descriptor;
            CHECK(auxiliary_registry.get_group_resource_bank(transaction, i, &descriptor));
            CHECK(descriptor.tensor == auxiliary_banks[i].tensor && descriptor.role == auxiliary_banks[i].role);
            CHECK(descriptor.source_data == descriptor.tensor->data && descriptor.byte_extent == ggml_nbytes(descriptor.tensor));
        }
        CHECK(auxiliary_registry.end_group_transaction(transaction));
        CHECK(auxiliary_registry.replace(&auxiliary_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        CHECK(!auxiliary_registry.get_group_resources(resource, nullptr) && !auxiliary_registry.get_group(key, nullptr));
        CHECK(auxiliary_registry.find_down_group_key(down, &key));
        CHECK(auxiliary_registry.acquire_group_resources(key, &repeated) && repeated.resource_generation != resource.resource_generation);
        fprintf(stderr, "test-moe-cache: six original auxiliary banks resource reuse and stale generation rejection OK\n");
    }

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
    CHECK(group_info.n_resource_banks == 3);
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
    constexpr int N_DIM = 32;
    constexpr int N_OUTPUT = 8;
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
    const int n_router_nodes = ggml_graph_n_nodes(graph);
    ggml_tensor * bank = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, N_DIM, N_OUTPUT, N_EXPERTS);
    ggml_set_name(bank, "blk.0.ffn_up_exps.weight");
    auto * cached_buft = ggml_backend_cuda_moe_cached_buffer_type();
    ggml_backend_buffer_ptr cached(ggml_backend_buft_alloc_buffer(cached_buft, ggml_backend_buft_get_alloc_size(cached_buft, bank)));
    CHECK(cached != nullptr);
    CHECK(ggml_backend_tensor_alloc(cached.get(), bank, ggml_backend_buffer_get_base(cached.get())) == GGML_STATUS_SUCCESS);
    ggml_tensor * input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, N_DIM, 1, 1);
    ggml_tensor * output = ggml_mul_mat_id(ctx.get(), bank, input, ids);
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    CHECK(buffer != nullptr);
    std::vector<float> bank_values(N_DIM * N_OUTPUT * N_EXPERTS);
    for (int expert = 0; expert < N_EXPERTS; ++expert) {
        std::fill_n(bank_values.data() + expert * N_DIM * N_OUTPUT, N_DIM * N_OUTPUT, (float) expert);
    }
    ggml_backend_tensor_set(bank, bank_values.data(), 0, ggml_nbytes(bank));
    std::array<float, N_DIM> input_values;
    input_values.fill(1.0f);
    ggml_backend_tensor_set(input, input_values.data(), 0, sizeof(input_values));
    std::array<float, N_EXPERTS> values;
    float sum = 0.0f;
    for (int i = 0; i < N_EXPERTS; ++i) {
        values[i] = i * 0.01f;
        sum += std::exp(values[i]);
    }
    ggml_backend_tensor_set(logits, values.data(), 0, sizeof(values));
    auto router_only = ggml_graph_view(graph, 0, n_router_nodes);
    for (int pass = 0; pass < 3; ++pass) {
        CHECK(ggml_backend_graph_compute(backend.get(), &router_only) == GGML_STATUS_SUCCESS);
        CHECK(ggml_cuda_moe_ids_cache_count_for_test(backend.get()) == 0);
    }
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
        const auto actual_output = active_grouped_tensor_values(output);
        for (int route = 0; route < N_USED; ++route) {
            for (int row = 0; row < N_OUTPUT; ++row) {
                CHECK(actual_output[route * N_OUTPUT + row] == (float) (N_DIM * (N_EXPERTS - route - 1)));
            }
        }
    }
    ggml_backend_ptr other(ggml_backend_cuda_init(0));
    CHECK(other != nullptr && ggml_cuda_moe_ids_cache_count_for_test(other.get()) == 0);
    CHECK(ggml_backend_graph_compute(other.get(), graph) == GGML_STATUS_SUCCESS);
    CHECK(ggml_cuda_moe_ids_cache_count_for_test(other.get()) == 0);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v1(other.get(), &snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(ggml_backend_graph_compute(other.get(), graph) == GGML_STATUS_SUCCESS);
    CHECK(ggml_cuda_moe_ids_cache_count_for_test(other.get()) == 1);
    // A rebuilt graph safely reuses the producer-owned slot instead of consuming the bounded table.
    graph->uid = ggml_graph_next_uid();
    CHECK(ggml_backend_graph_compute(other.get(), graph) == GGML_STATUS_SUCCESS);
    CHECK(ggml_cuda_moe_ids_cache_count_for_test(other.get()) == 1);
    CHECK(ggml_backend_graph_compute(other.get(), graph) == GGML_STATUS_SUCCESS);
    CHECK(ggml_cuda_moe_ids_cache_count_for_test(other.get()) == 1);
    std::array<int32_t, N_USED> imported_ids;
    std::iota(imported_ids.begin(), imported_ids.end(), 0);
    ggml_backend_tensor_set(ids, imported_ids.data(), 0, sizeof(imported_ids));
    auto consumer_only = ggml_graph_view(graph, ggml_graph_n_nodes(graph) - 1, ggml_graph_n_nodes(graph));
    consumer_only.uid = graph->uid;
    CHECK(ggml_backend_graph_compute(other.get(), &consumer_only) == GGML_STATUS_SUCCESS);
    const auto imported_output = active_grouped_tensor_values(output);
    for (int route = 0; route < N_USED; ++route) {
        for (int row = 0; row < N_OUTPUT; ++row) {
            CHECK(imported_output[route * N_OUTPUT + row] == (float) (N_DIM * route));
        }
    }
    other.reset();
    CHECK(ggml_cuda_moe_ids_cache_count_for_test(backend.get()) == GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS);
    CHECK(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS);
    backend.reset();
    fprintf(stderr, "test-moe-cache: bounded route publication and backend ownership OK\n");
}

struct cpu_region_query_fixture {
    ggml_context_ptr                                     context;
    ggml_backend_buffer_ptr                              buffer;
    std::vector<uint8_t>                                 oracle_work;
    ggml_cgraph *                                        graph          = nullptr;
    ggml_tensor *                                        activation     = nullptr;
    ggml_tensor *                                        ids            = nullptr;
    ggml_tensor *                                        hidden         = nullptr;
    ggml_tensor *                                        raw_down       = nullptr;
    ggml_tensor *                                        common_tail    = nullptr;
    std::vector<ggml_tensor *>                          weights;
    std::array<const ggml_tensor *, 2>                   dynamic_inputs = {};
    std::array<const ggml_tensor *, 2>                   live_outputs   = {};
    std::vector<ggml_backend_moe_cpu_region_source_v1>   sources;
    ggml_backend_moe_cpu_region_query_v1                 query          = {};
};

static std::unique_ptr<cpu_region_query_fixture> cpu_region_query_graph(
        ggml_type type, ggml_glu_op activation, bool merged = false, bool gate_bias = false,
        ggml_type down_type = GGML_TYPE_COUNT, int64_t n_ff = 256, int64_t n_used = 2, int64_t n_rows = 2,
        uint32_t n_threads = 2, ggml_type up_type = GGML_TYPE_COUNT, int64_t n_embd = 256, int64_t n_experts = 4) {
    const int64_t          N_EMBD     = n_embd;
    const int64_t          N_EXPERTS  = n_experts;
    constexpr uint64_t     GENERATION = 37;
    const ggml_init_params params     = {
        /* .mem_size = */ ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc = */ true,
    };
    auto result = std::make_unique<cpu_region_query_fixture>();
    if (down_type == GGML_TYPE_COUNT) {
        down_type = type;
    }
    result->context.reset(ggml_init(params));
    CHECK(result->context != nullptr);
    result->weights.push_back(ggml_new_tensor_3d(result->context.get(), type, N_EMBD, merged ? 2 * n_ff : n_ff, N_EXPERTS));
    if (!merged) {
        result->weights.push_back(ggml_new_tensor_3d(result->context.get(), up_type == GGML_TYPE_COUNT ? type : up_type, N_EMBD, n_ff, N_EXPERTS));
    }
    auto * down_weight = ggml_new_tensor_3d(result->context.get(), down_type, n_ff, N_EMBD, N_EXPERTS);
    result->weights.push_back(down_weight);
    result->activation  = ggml_new_tensor_3d(result->context.get(), GGML_TYPE_F32, N_EMBD, 1, n_rows);
    result->ids         = ggml_new_tensor_2d(result->context.get(), GGML_TYPE_I32, n_used, n_rows);
    ggml_tensor * gate  = ggml_mul_mat_id(result->context.get(), result->weights[0], result->activation, result->ids);
    ggml_tensor * up;
    if (merged) {
        auto * fused = gate;
        gate = ggml_view_3d(result->context.get(), fused, n_ff, n_used, n_rows, fused->nb[1], fused->nb[2], 0);
        up = ggml_view_3d(result->context.get(), fused, n_ff, n_used, n_rows, fused->nb[1], fused->nb[2], n_ff * sizeof(float));
    } else {
        up = ggml_mul_mat_id(result->context.get(), result->weights[1], result->activation, result->ids);
    }
    if (gate_bias) {
        auto * bias = ggml_new_tensor_2d(result->context.get(), GGML_TYPE_F32, n_ff, N_EXPERTS);
        result->weights.push_back(bias);
        gate = ggml_add_id(result->context.get(), gate, bias, result->ids);
    }
    result->hidden      = ggml_glu_split(result->context.get(), gate, up, activation);
    result->raw_down    = ggml_mul_mat_id(result->context.get(), down_weight, result->hidden, result->ids);
    ggml_tensor * bias  = ggml_new_tensor_2d(result->context.get(), GGML_TYPE_F32, N_EMBD, N_EXPERTS);
    result->common_tail = ggml_add_id(result->context.get(), result->raw_down, bias, result->ids);
    result->graph       = ggml_new_graph_custom(result->context.get(), 64, false);
    ggml_build_forward_expand(result->graph, result->raw_down);
    result->graph->uid = ggml_graph_next_uid();
    result->buffer.reset(
        ggml_backend_alloc_ctx_tensors_from_buft(result->context.get(), ggml_backend_cuda_moe_cached_buffer_type()));
    CHECK(result->buffer != nullptr);

    result->dynamic_inputs = { result->activation, result->ids };
    result->live_outputs   = { result->raw_down, result->hidden };
    result->sources.resize(result->weights.size());
    for (size_t i = 0; i < result->weights.size(); ++i) {
        result->sources[i] = {
            result->weights[i],        result->weights[i],
            result->weights[i]->data,  ggml_nbytes(result->weights[i]),
            result->weights[i]->nb[2], GENERATION,
        };
    }
    result->query.struct_size               = sizeof(result->query);
    result->query.graph                     = result->graph;
    result->query.graph_uid                 = result->graph->uid;
    result->query.graph_generation          = GENERATION;
    result->query.source_generation         = GENERATION;
    result->query.body_nodes                = ggml_graph_nodes(result->graph);
    result->query.n_body_nodes              = ggml_graph_n_nodes(result->graph);
    result->query.activation                = result->activation;
    result->query.ids                       = result->ids;
    result->query.dynamic_inputs            = result->dynamic_inputs.data();
    result->query.n_dynamic_inputs          = result->dynamic_inputs.size();
    result->query.live_outputs              = result->live_outputs.data();
    result->query.n_live_outputs            = type == GGML_TYPE_Q4_0 ? 1 : 2;
    result->query.sources                   = result->sources.data();
    result->query.n_sources                 = result->sources.size();
    result->query.bucket_rows               = n_rows;
    result->query.routes_per_row            = n_used;
    result->query.source_row_capacity       = std::max<int64_t>(16, n_rows + 5);
    result->query.scatter_capacity          = std::max<int64_t>(16, n_rows * n_used);
    result->query.n_threads                 = n_threads;
    result->query.n_lanes                   = 2;
    result->query.output_staging_limit      = 16 * 1024 * 1024;
    result->query.lane_execution_byte_limit = 64 * 1024 * 1024;
    return result;
}

void test_hybrid_row_sources() {
    auto wide = cpu_region_query_graph(GGML_TYPE_Q4_0, GGML_GLU_OP_SWIGLU, false, true);
    std::vector<std::unique_ptr<cpu_region_query_fixture>> buckets;
    std::vector<const ggml_backend_moe_cpu_region_query_v1 *> queries;
    for (uint32_t count = 1; count <= wide->query.routes_per_row; ++count) {
        auto bucket = cpu_region_query_graph(GGML_TYPE_Q4_0, GGML_GLU_OP_SWIGLU, false, true, GGML_TYPE_COUNT,
            wide->weights[0]->ne[1], count, 1);
        bucket->query.n_lanes = 1;
        bucket->query.source_row_capacity = wide->query.bucket_rows;
        bucket->query.scatter_capacity = wide->query.bucket_rows * wide->query.routes_per_row;
        if (!buckets.empty()) {
            for (size_t bank = 0; bank < bucket->sources.size(); ++bank) {
                auto * tensor = bucket->weights[bank];
                bucket->sources[bank] = buckets[0]->sources[bank];
                bucket->sources[bank].tensor = tensor;
            }
        }
        queries.push_back(&bucket->query);
        buckets.push_back(std::move(bucket));
    }
    ggml_backend_moe_hybrid_region_v1 region = {};
    region.struct_size = sizeof(region);
    region.source_graph_uid = wide->query.graph_uid;
    region.split_graph_uid = queries[0]->graph_uid;
    region.owner_generation = 1;
    region.allocator_generation = queries[0]->graph_generation;
    region.activation = wide->activation;
    region.ids = wide->ids;
    region.output = wide->raw_down;
    region.down = wide->raw_down->src[0];
    region.query = queries[0];
    region.cpu_queries = queries.data();
    region.n_cpu_queries = queries.size();
    region.last_node = region.query->n_body_nodes - 1;
    CHECK(ggml_backend_moe_hybrid_get_geometry_v1(region.activation, region.ids, region.output,
        wide->weights[0]->ne[2], &region.geometry) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    ggml_cuda_moe_hybrid_rows_query query;
    query.region = &region;
    query.certificate = {GGML_GRAPH_EXECUTION_CERTIFICATE_MAGIC, GGML_GRAPH_EXECUTION_CERTIFICATE_VERSION,
        sizeof(query.certificate), GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED,
        GGML_GRAPH_EXECUTION_DOMAIN_MAIN, GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SPECULATIVE,
        region.geometry.row_capacity, 1, 41, region.owner_generation, region.source_graph_uid, region.split_graph_uid, {}};
    query.plan_capacity = region.geometry.route_capacity;
    query.slot_capacity = 1;
    query.transfer_capacity = region.geometry.routes_per_row;
    query.workspace_generation = region.allocator_generation;
    query.resource_fingerprint = reinterpret_cast<uintptr_t>(queries[0]);
    query.device_alignment = 256;
    query.host_alignment = 64;
    std::vector<uint8_t> accessible(region.query->n_sources, 1);
    query.source_device_accessible = accessible.data();
    query.n_sources = accessible.size();
    ggml_cuda_moe_hybrid_rows_layout layout;
    std::vector<ggml_cuda_moe_hybrid_rows_source> sources;
    CHECK(ggml_cuda_moe_hybrid_rows_measure(query, layout, sources));
    size_t bias_index = SIZE_MAX;
    for (size_t i = 0; i < sources.size(); ++i) {
        const auto * tensor = region.query->sources[i].tensor;
        const bool bias = tensor->ne[2] == 1;
        CHECK(sources[i].role == (bias ? GGML_CUDA_MOE_HYBRID_SOURCE_ADD_ID_BIAS : GGML_CUDA_MOE_HYBRID_SOURCE_MMID_WEIGHT));
        CHECK(sources[i].expert_bytes == tensor->nb[bias ? 1 : 2]);
        CHECK(sources[i].bytes == sources[i].expert_bytes * query.transfer_capacity);
        if (bias) { bias_index = i; CHECK(tensor->nb[2] > tensor->nb[1]); }
    }
    CHECK(bias_index != SIZE_MAX && layout.staging_tile_bytes == 0);
    accessible[bias_index] = 0;
    CHECK(ggml_cuda_moe_hybrid_rows_measure(query, layout, sources));
    CHECK(layout.staging_tile_bytes == region.query->sources[bias_index].tensor->nb[1]);
    for (auto & bucket : buckets) { --bucket->sources[bias_index].bytes; }
    CHECK(!ggml_cuda_moe_hybrid_rows_measure(query, layout, sources));
    for (auto & bucket : buckets) { ++bucket->sources[bias_index].bytes; }
    for (auto & bucket : buckets) {
        for (uint32_t i = 0; i < bucket->query.n_body_nodes; ++i) {
            auto * node = const_cast<ggml_tensor *>(bucket->query.body_nodes[i]);
            if (node->op == GGML_OP_ADD_ID) { node->op = GGML_OP_ADD; }
        }
    }
    CHECK(!ggml_cuda_moe_hybrid_rows_measure(query, layout, sources));
    fprintf(stderr, "test-moe-cache: hybrid role-aware weight/bias slices mapped/mixed/short/unsupported OK\n");
}

struct cpu_region_oracle {
    std::vector<float> source_activation;
    std::vector<std::vector<uint8_t>> outputs;
};

void test_cpu_routed_service() {
    auto * reg = ggml_backend_cpu_reg();
    const bool fidelity = getenv("GGML_TEST_MOE_CPU_FIDELITY") != nullptr;
    const auto get_api = reinterpret_cast<ggml_backend_moe_cpu_region_service_v1_t>(
        ggml_backend_reg_get_proc_address(reg, fidelity ? GGML_BACKEND_MOE_CPU_FIDELITY_SERVICE_V1_PROC_NAME :
            GGML_BACKEND_MOE_CPU_REGION_SERVICE_V1_PROC_NAME));
    const auto execute_routed = reinterpret_cast<ggml_backend_moe_cpu_routed_execute_v1_t>(
        ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_CPU_ROUTED_EXECUTE_V1_PROC_NAME));
    CHECK(get_api && execute_routed);
    const auto * api = get_api();
    constexpr uint32_t rows = 3, routes = 4, experts = 7, outputs = 51, input_width = 256;
    constexpr uint32_t capacity = rows * routes, source_rows_count = 8, output_stride = outputs + 6;
    uint32_t checks = 0;
    for (const auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q4_0, GGML_TYPE_Q5_K,
                           GGML_TYPE_Q3_K, GGML_TYPE_IQ1_S, GGML_TYPE_MXFP4, GGML_TYPE_NVFP4}) {
        for (const uint32_t period : {1u, 2u, 4u}) {
            ggml_context_ptr ctx(ggml_init({ggml_tensor_overhead() * 8 + ggml_graph_overhead_custom(8, false), nullptr, true}));
            CHECK(ctx);
            auto * weight = ggml_new_tensor_3d(ctx.get(), type, input_width, outputs, experts);
            auto * input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, input_width, period, rows);
            auto * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, routes, rows);
            auto * output = ggml_mul_mat_id(ctx.get(), weight, input, ids);
            CHECK(ggml_prec_set_acc(output, GGML_PREC_F32));
            auto * graph = ggml_new_graph_custom(ctx.get(), 8, false);
            ggml_build_forward_expand(graph, output);
            graph->uid = ggml_graph_next_uid();
            ggml_backend_buffer_ptr storage(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type()));
            CHECK(storage);
            const auto weight_bytes = cached_fusion_test_data(weight, 127);
            memcpy(weight->data, weight_bytes.data(), weight_bytes.size());
            std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_QWEN3MOE, llama_model_default_params()));
            CHECK(model->record_moe_readable_source(weight, weight->data, ggml_nbytes(weight)));
            ggml_backend_moe_source_owner_v1 owner = {};
            CHECK(model->moe_source_owner_v1(&owner));
            const ggml_tensor * dynamic[] = {input, ids};
            const ggml_tensor * live[] = {output};
            ggml_backend_moe_cpu_region_source_v1 source = {weight, weight, weight->data, ggml_nbytes(weight), weight->nb[2], owner.generation};
            ggml_backend_moe_cpu_region_query_v1 query = {};
            query.struct_size = sizeof(query);
            query.flags = GGML_BACKEND_MOE_CPU_REGION_FLAG_V1_ROUTED_OPERATION;
            query.graph = graph; query.graph_uid = graph->uid;
            query.graph_generation = 17; query.source_generation = owner.generation;
            query.body_nodes = ggml_graph_nodes(graph); query.n_body_nodes = 1;
            query.activation = input; query.ids = ids; query.dynamic_inputs = dynamic; query.n_dynamic_inputs = 2;
            query.live_outputs = live; query.n_live_outputs = 1; query.sources = &source; query.n_sources = 1;
            query.bucket_rows = rows; query.routes_per_row = routes; query.source_row_capacity = source_rows_count;
            query.scatter_capacity = capacity; query.n_threads = 2; query.n_lanes = 1;
            ggml_backend_moe_cpu_service_config_v1 config = {};
            config.struct_size = sizeof(config); config.abi_version = 1; config.source_owner = &owner;
            config.n_threads = query.n_threads; config.n_lanes = query.n_lanes; config.max_regions = 1;
            config.flags = GGML_BACKEND_MOE_CPU_SERVICE_FLAG_V1_ALLOW_UNKNOWN_THREAD_STACK_BYTES |
                           GGML_BACKEND_MOE_CPU_SERVICE_FLAG_V1_ALLOW_UNPROVEN_RUNTIME_ALLOCATIONS;
            ggml_backend_moe_cpu_service_v1_t service = nullptr;
            CHECK(api->create(&config, &service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
            ggml_backend_moe_cpu_prepared_requirements_v1 prepared = {};
            prepared.struct_size = sizeof(prepared); prepared.abi_version = 1;
            ggml_backend_moe_cpu_prepared_region_v1_t region = 0;
            auto short_query = query;
            short_query.struct_size = sizeof(uint32_t);
            CHECK(api->prepare(service, &short_query, &prepared, &region) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
            CHECK(region == 0);
            CHECK(api->prepare(service, &query, &prepared, &region) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
            ggml_backend_moe_cpu_service_state_v1 state = {};
            state.struct_size = sizeof(state);
            CHECK(api->state(service, &state) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
            const auto payload = state.prepared_payload_bytes;
            std::array<int32_t, capacity> expert_ids = {};
            std::array<uint32_t, capacity> bound_rows = {}, scatter = {};
            std::array<uint8_t, capacity> ownership = {};
            for (uint32_t route = 0; route < capacity; ++route) {
                bound_rows[route] = 5 - 2 * (route / routes);
                scatter[route] = route * 5 % capacity;
            }
            std::vector<float> input_values(size_t(source_rows_count) * input_width * period);
            std::vector<uint8_t> actual(size_t(capacity) * output_stride * sizeof(float), 0xA7), expected(actual.size());
            ggml_backend_moe_cpu_region_binding_v1 binding = {sizeof(binding), rows, capacity,
                expert_ids.data(), bound_rows.data(), scatter.data()};
            ggml_backend_moe_cpu_dynamic_input_v1 inputs[] = {
                {input_values.data(), input_values.size() * sizeof(float), input->nb[2]}, {}};
            ggml_backend_moe_cpu_output_v1 destination = {actual.data(), actual.size(), output_stride * sizeof(float)};
            ggml_backend_moe_cpu_execute_v1 execution = {};
            execution.struct_size = sizeof(execution); execution.graph_uid = query.graph_uid;
            execution.graph_generation = query.graph_generation; execution.source_generation = query.source_generation;
            execution.binding = &binding; execution.dynamic_inputs = inputs; execution.n_dynamic_inputs = 2;
            execution.outputs = &destination; execution.n_outputs = 1;
            for (uint32_t frame = 0; frame < 2; ++frame) {
                for (size_t i = 0; i < input_values.size(); ++i) { input_values[i] = 0.013f * (int((i * 3 + frame * 7) % 23) - 11); }
                for (uint32_t row = 0; row < rows; ++row) {
                    memcpy(static_cast<uint8_t *>(input->data) + row * input->nb[2],
                        input_values.data() + bound_rows[row * routes] * input_width * period, input->nb[2]);
                }
                for (uint32_t route = 0; route < capacity; ++route) { expert_ids[route] = (route / routes + frame * 3) % experts; }
                memcpy(ids->data, expert_ids.data(), sizeof(expert_ids));
                auto plan = ggml_graph_plan(graph, query.n_threads, nullptr);
                std::vector<uint8_t> work(plan.work_size + GGML_MEM_ALIGN);
                void * aligned = work.data(); size_t remaining = work.size();
                CHECK(!plan.work_size || std::align(GGML_MEM_ALIGN, plan.work_size, aligned, remaining));
                plan.work_data = static_cast<uint8_t *>(aligned);
                CHECK(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS);
                const uint64_t allocations = ggml_allocation_count();
                for (uint32_t layout = 0; layout < 3; ++layout) {
                    destination.route_stride = (layout == 0 ? output_stride : outputs) * sizeof(float);
                    for (uint32_t route = 0; route < capacity; ++route) {
                        scatter[route] = layout == 1 ? route : route * 5 % capacity;
                    }
                    for (uint32_t pattern = 0; pattern < 5; ++pattern) {
                        uint32_t assigned = 0;
                        for (uint32_t route = 0; route < capacity; ++route) {
                            ownership[route] = pattern == 0 || (pattern == 2 && route % 2 == 0) || (pattern == 3 && route % 2 != 0) || (pattern == 4 && route % 4 < 2);
                            assigned += ownership[route];
                        }
                        std::fill(actual.begin(), actual.end(), 0xA7);
                        expected = actual;
                        for (uint32_t route = 0; route < capacity; ++route) {
                            if (ownership[route]) {
                                memcpy(expected.data() + scatter[route] * destination.route_stride,
                                    static_cast<const uint8_t *>(output->data) + route * outputs * sizeof(float), outputs * sizeof(float));
                            }
                        }
                        execution.epoch = 1 + (frame * 3 + layout) * 5 + pattern;
                        ggml_backend_moe_cpu_execute_result_v1 result = {};
                        result.struct_size = sizeof(result);
                        CHECK(execute_routed(service, region, &execution, ownership.data(), capacity, &result) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
                        CHECK(actual == expected && result.published_routes == assigned && result.published_outputs == 1);
                        CHECK(result.flags == GGML_BACKEND_MOE_CPU_EXECUTE_RESULT_FLAG_V1_PUBLISHED);
                        ++checks;
                    }
                }
                destination.route_stride = output_stride * sizeof(float);
                for (uint32_t route = 0; route < capacity; ++route) { scatter[route] = route * 5 % capacity; }
                CHECK(ggml_allocation_count() == allocations);
            }
            execution.epoch = 9;
            ggml_backend_moe_cpu_execute_result_v1 result = {};
            result.struct_size = sizeof(result);
            const auto unchanged = actual;
            CHECK(api->execute(service, region, &execution, &result) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_BINDING);
            CHECK(execute_routed(service, region, &execution, ownership.data(), capacity - 1, &result) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_BINDING);
            ownership[0] = 2;
            CHECK(execute_routed(service, region, &execution, ownership.data(), capacity, &result) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_BINDING);
            ownership[0] = 1;
            auto * alias = actual.data() + actual.size() - capacity;
            std::fill_n(alias, capacity, 0);
            const auto alias_unchanged = actual;
            CHECK(execute_routed(service, region, &execution, alias, capacity, &result) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
            CHECK(actual == alias_unchanged && result.flags == 0);
            actual = unchanged;
            CHECK(actual == unchanged && result.flags == 0);
            CHECK(api->cancel(service, execution.epoch) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
            CHECK(execute_routed(service, region, &execution, ownership.data(), capacity, &result) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CANCELED);
            CHECK(actual == unchanged && result.flags == 0);
            struct cancel_context {
                const ggml_backend_moe_cpu_region_service_api_v1 * api;
                ggml_backend_moe_cpu_service_v1_t service;
                uint64_t epoch;
                bool reached = false;
            } cancel = {api, service, 10};
            const auto cancel_before_commit = [](void * opaque, uint32_t phase) {
                auto & cancel = *static_cast<cancel_context *>(opaque);
                if (phase == GGML_BACKEND_MOE_CPU_TEST_PHASE_V1_BEFORE_COMMIT) {
                    cancel.reached = true;
                    CHECK(cancel.api->cancel(cancel.service, cancel.epoch) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
                }
            };
            CHECK(api->set_test_hook(service, cancel_before_commit, &cancel) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
            execution.epoch = cancel.epoch;
            ownership.fill(1);
            CHECK(execute_routed(service, region, &execution, ownership.data(), capacity, &result) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CANCELED);
            CHECK(cancel.reached && actual == unchanged && result.flags == 0);
            CHECK(api->set_test_hook(service, nullptr, nullptr) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
            execution.epoch = 11;
            expected = actual;
            for (uint32_t route = 0; route < capacity; ++route) {
                memcpy(expected.data() + scatter[route] * destination.route_stride,
                    static_cast<const uint8_t *>(output->data) + route * outputs * sizeof(float), outputs * sizeof(float));
            }
            CHECK(execute_routed(service, region, &execution, ownership.data(), capacity, &result) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
            CHECK(actual == expected && result.published_routes == capacity && result.flags == GGML_BACKEND_MOE_CPU_EXECUTE_RESULT_FLAG_V1_PUBLISHED);
            CHECK(api->state(service, &state) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
            CHECK(state.prepared_payload_bytes == payload && state.active_jobs == 0);
            CHECK(api->close(service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
            CHECK(api->destroy_region(service, &region) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
            CHECK(api->destroy(&service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        }
    }
    fprintf(stderr, "test-moe-cache: CPU routed service fidelity=%d %u exact original-layout replays, nine types, three input periods, ownership/guards/cancellation/ledger OK\n", fidelity, checks);
}

template<class Routes, class Rows>
static cpu_region_oracle cpu_region_reference(
        cpu_region_query_fixture & fixture,
        const Routes & expert_ids,
        const Rows & source_rows) {
    for (size_t i = 0; i < fixture.weights.size(); ++i) {
        const auto bytes = cached_fusion_test_data(fixture.weights[i], 11 + i);
        memcpy(fixture.weights[i]->data, bytes.data(), bytes.size());
    }
    cpu_region_oracle result;
    result.source_activation.resize(fixture.query.source_row_capacity * fixture.activation->ne[0]);
    for (uint32_t row = 0; row < fixture.query.source_row_capacity; ++row) {
        for (int64_t column = 0; column < fixture.activation->ne[0]; ++column) {
            result.source_activation[row * fixture.activation->ne[0] + column] =
                0.01f * (1 + (row * 7 + column * 3) % 19);
        }
    }
    for (uint32_t row = 0; row < fixture.query.bucket_rows; ++row) {
        memcpy(static_cast<uint8_t *>(fixture.activation->data) + row * fixture.activation->nb[2],
               result.source_activation.data() + source_rows[row * fixture.query.routes_per_row] * fixture.activation->ne[0],
               fixture.activation->nb[2]);
    }
    memcpy(fixture.ids->data, expert_ids.data(), expert_ids.size() * sizeof(expert_ids[0]));
    auto plan = ggml_graph_plan(fixture.graph, fixture.query.n_threads, nullptr);
    CHECK(plan.work_size <= SIZE_MAX - (GGML_MEM_ALIGN - 1));
    fixture.oracle_work.resize(plan.work_size ? plan.work_size + GGML_MEM_ALIGN - 1 : 0);
    if (plan.work_size) {
        void * work = fixture.oracle_work.data();
        size_t bytes = fixture.oracle_work.size();
        CHECK(std::align(GGML_MEM_ALIGN, plan.work_size, work, bytes) != nullptr);
        plan.work_data = static_cast<uint8_t *>(work);
    }
    CHECK(ggml_graph_compute(fixture.graph, &plan) == GGML_STATUS_SUCCESS);
    result.outputs.resize(fixture.query.n_live_outputs);
    for (uint32_t i = 0; i < fixture.query.n_live_outputs; ++i) {
        result.outputs[i].resize(ggml_nbytes(fixture.query.live_outputs[i]));
        memcpy(result.outputs[i].data(), fixture.query.live_outputs[i]->data, result.outputs[i].size());
    }
    return result;
}

template<class Routes, class Rows, class Scatter>
static void cpu_region_check_execute(
        const ggml_backend_moe_cpu_region_service_api_v1 * service_api,
        ggml_backend_moe_cpu_service_v1_t service,
        ggml_backend_moe_cpu_prepared_region_v1_t region,
        const cpu_region_query_fixture & fixture,
        const cpu_region_oracle & oracle,
        const Routes & expert_ids,
        const Rows & source_rows,
        const Scatter & scatter,
        uint64_t epoch, bool reject_wrong_count = false, uint32_t active_rows = 0,
        uint32_t execution_flags = 0, uint32_t output_padding = 0, uint32_t output_offset = 0) {
    ggml_backend_moe_cpu_region_binding_v1 binding = {
        sizeof(binding), active_rows == 0 ? fixture.query.bucket_rows : active_rows, (uint32_t) expert_ids.size(),
        expert_ids.data(), source_rows.data(), scatter.data(),
    };
    std::vector<ggml_backend_moe_cpu_dynamic_input_v1> inputs(fixture.query.n_dynamic_inputs);
    for (uint32_t i = 0; i < fixture.query.n_dynamic_inputs; ++i) {
        if (fixture.query.dynamic_inputs[i] == fixture.activation) {
            inputs[i].data = oracle.source_activation.data();
            inputs[i].bytes = oracle.source_activation.size() * sizeof(float);
            inputs[i].row_stride = fixture.activation->nb[2];
        }
    }
    std::vector<std::vector<uint8_t>> published(fixture.query.n_live_outputs);
    std::vector<ggml_backend_moe_cpu_output_v1> outputs(fixture.query.n_live_outputs);
    for (uint32_t i = 0; i < fixture.query.n_live_outputs; ++i) {
        const auto * tensor = fixture.query.live_outputs[i];
        const uint64_t stride = tensor->nb[1] + output_padding;
        published[i].assign(fixture.query.scatter_capacity * stride + output_offset, 0xa5);
        outputs[i] = { published[i].data() + output_offset, published[i].size() - output_offset, stride };
    }
    ggml_backend_moe_cpu_execute_v1 execution = {};
    execution.struct_size = sizeof(execution);
    execution.flags = execution_flags;
    execution.epoch = epoch;
    execution.graph_uid = fixture.query.graph_uid;
    execution.graph_generation = fixture.query.graph_generation;
    execution.source_generation = fixture.query.source_generation;
    execution.binding = &binding;
    execution.dynamic_inputs = inputs.data();
    execution.n_dynamic_inputs = inputs.size();
    execution.outputs = outputs.data();
    execution.n_outputs = outputs.size();
    ggml_backend_moe_cpu_execute_result_v1 result = {};
    result.struct_size = sizeof(result);
    if (reject_wrong_count) {
        --binding.n_routes;
        CHECK(service_api->execute(service, region, &execution, &result) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_BINDING);
        ++binding.n_routes;
        for (const auto & bytes : published) {
            CHECK(std::all_of(bytes.begin(), bytes.end(), [](uint8_t value) { return value == 0xa5; }));
        }
    }
    const int32_t execute_status = service_api->execute(service, region, &execution, &result);
    if (execute_status != GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK) {
        fprintf(stderr, "test-moe-cache: CPU region execute failed status=%d rows=%u routes=%u flags=%u\n",
            execute_status, binding.n_rows, binding.n_routes, fixture.query.flags);
    }
    CHECK(execute_status == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    CHECK(result.flags == GGML_BACKEND_MOE_CPU_EXECUTE_RESULT_FLAG_V1_PUBLISHED && result.epoch == epoch &&
          result.graph_uid == fixture.query.graph_uid && result.graph_generation == fixture.query.graph_generation &&
          result.published_routes == expert_ids.size() &&
          result.source_generation == fixture.query.source_generation &&
          result.published_outputs == fixture.query.n_live_outputs && result.lane_index < fixture.query.n_lanes);
    for (uint32_t output_index = 0; output_index < fixture.query.n_live_outputs; ++output_index) {
        const auto * tensor = fixture.query.live_outputs[output_index];
        std::vector<uint8_t> expected(published[output_index].size(), 0xa5);
        for (uint32_t route = 0; route < expert_ids.size(); ++route) {
            const uint32_t row = route / fixture.query.routes_per_row;
            const uint32_t selected = route % fixture.query.routes_per_row;
            memcpy(expected.data() + output_offset + scatter[route] * outputs[output_index].route_stride,
                   oracle.outputs[output_index].data() + row * tensor->nb[2] + selected * tensor->nb[1], tensor->nb[1]);
        }
        CHECK(published[output_index] == expected);
    }
}

struct cpu_region_test_barrier {
    std::mutex mutex;
    std::condition_variable changed;
    uint32_t phase;
    size_t entered = 0;
    bool released = false;

    explicit cpu_region_test_barrier(uint32_t phase) : phase(phase) {}

    static void hook(void * data, uint32_t phase) {
        auto & barrier = *static_cast<cpu_region_test_barrier *>(data);
        if (phase != barrier.phase) {
            return;
        }
        std::unique_lock<std::mutex> lock(barrier.mutex);
        ++barrier.entered;
        barrier.changed.notify_all();
        CHECK(barrier.changed.wait_for(lock, std::chrono::seconds(20), [&] { return barrier.released; }));
    }

    void wait() {
        std::unique_lock<std::mutex> lock(mutex);
        CHECK(changed.wait_for(lock, std::chrono::seconds(20), [&] { return entered != 0; }));
    }

    void release() {
        std::lock_guard<std::mutex> lock(mutex);
        released = true;
        changed.notify_all();
    }
};

static void test_cpu_region_mixed_banks(
        const ggml_backend_moe_cpu_region_query_api_v1 * api,
        const ggml_backend_moe_cpu_region_service_api_v1 * service_api) {
    struct signature {
        ggml_type gate_up;
        ggml_type down;
        int64_t n_ff;
    };
    for (const auto & types : {signature{GGML_TYPE_Q5_K, GGML_TYPE_Q5_K, 1792},
                              signature{GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, 1792},
                              signature{GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, 1792},
                              signature{GGML_TYPE_Q4_K, GGML_TYPE_Q5_0, 1408},
                              signature{GGML_TYPE_Q4_K, GGML_TYPE_Q8_0, 1408}}) {
        auto fixture = cpu_region_query_graph(types.gate_up, GGML_GLU_OP_SWIGLU, false, false, types.down, types.n_ff);
        ggml_backend_moe_cpu_region_requirements_v1 requirements = {};
        requirements.struct_size = sizeof(requirements);
        CHECK(api->query(&fixture->query, &requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        auto * down = fixture->weights.back();
        const auto saved = *down;
        --down->ne[0];
        CHECK(api->query(&fixture->query, &requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION);
        *down = saved;
        --down->nb[1];
        CHECK(api->query(&fixture->query, &requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION);
        *down = saved;
        down->nb[2] = SIZE_MAX;
        CHECK(api->query(&fixture->query, &requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION);
        *down = saved;
        --down->ne[2];
        down->nb[3] = down->nb[2] * down->ne[2];
        CHECK(api->query(&fixture->query, &requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION);
        *down = saved;
        --fixture->sources.back().expert_stride;
        CHECK(api->query(&fixture->query, &requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_MISSING_SOURCE);
        ++fixture->sources.back().expert_stride;
        --fixture->sources.back().bytes;
        CHECK(api->query(&fixture->query, &requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_MISSING_SOURCE);
        ++fixture->sources.back().bytes;
        auto limited = fixture->query;
        limited.output_staging_limit = 1;
        CHECK(api->query(&limited, &requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY);
        limited = fixture->query;
        limited.lane_execution_byte_limit = requirements.all_lane_execution_bytes - 1;
        CHECK(api->query(&limited, &requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY);
        CHECK(api->query(&fixture->query, &requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);

        const std::array<int32_t, 4> expert_ids = {1, 3, 0, 2};
        const std::array<uint32_t, 4> source_rows = {9, 9, 2, 2};
        const std::array<uint32_t, 4> scatter = {6, 7, 2, 3};
        const auto oracle = cpu_region_reference(*fixture, expert_ids, source_rows);
        std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_QWEN3MOE, llama_model_default_params()));
        for (auto * weight : fixture->weights) {
            CHECK(model->record_moe_readable_source(weight, weight->data, ggml_nbytes(weight)));
        }
        ggml_backend_moe_source_owner_v1 owner = {};
        CHECK(model->moe_source_owner_v1(&owner));
        fixture->query.source_generation = owner.generation;
        for (auto & source : fixture->sources) {
            source.generation = owner.generation;
        }
        ggml_backend_moe_cpu_service_config_v1 config = {};
        config.struct_size = sizeof(config);
        config.abi_version = 1;
        config.source_owner = &owner;
        config.n_threads = fixture->query.n_threads;
        config.n_lanes = fixture->query.n_lanes;
        config.max_regions = 1;
        config.flags = GGML_BACKEND_MOE_CPU_SERVICE_FLAG_V1_ALLOW_UNKNOWN_THREAD_STACK_BYTES |
                       GGML_BACKEND_MOE_CPU_SERVICE_FLAG_V1_ALLOW_UNPROVEN_RUNTIME_ALLOCATIONS;
        config.prepared_payload_limit = 16 * 1024 * 1024;
        ggml_backend_moe_cpu_service_v1_t service = nullptr;
        CHECK(service_api->create(&config, &service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        ggml_backend_moe_cpu_prepared_requirements_v1 prepared = {};
        prepared.struct_size = sizeof(prepared);
        prepared.abi_version = 1;
        ggml_backend_moe_cpu_prepared_region_v1_t region = 0;
        CHECK(service_api->prepare(service, &fixture->query, &prepared, &region) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        const uint64_t allocations = ggml_allocation_count();
        for (uint64_t epoch : {1, 2}) {
            cpu_region_check_execute(service_api, service, region, *fixture, oracle, expert_ids, source_rows, scatter, epoch);
        }
        CHECK(ggml_allocation_count() == allocations);
        CHECK(service_api->close(service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        CHECK(service_api->destroy_region(service, &region) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        CHECK(service_api->destroy(&service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        fprintf(stderr, "test-moe-cache: CPU mixed-bank oracle gate_up=%s down=%s embd=256 ff=%lld geometry/source/capacity rejection OK\n",
            ggml_type_name(types.gate_up), ggml_type_name(types.down), (long long) types.n_ff);
    }
}

static void test_cpu_region_iq3_persistent_parity(
        const ggml_backend_moe_cpu_region_service_api_v1 * service_api) {
    auto fixture = cpu_region_query_graph(
        GGML_TYPE_IQ3_XXS, GGML_GLU_OP_SWIGLU, false, false, GGML_TYPE_IQ3_XXS, 1792, 1, 1, 16);
    fixture->query.n_lanes = 1;
    const std::array<int32_t, 1> first_ids = {1};
    const std::array<uint32_t, 1> first_rows = {9};
    const std::array<uint32_t, 1> first_scatter = {6};
    const auto first_oracle = cpu_region_reference(*fixture, first_ids, first_rows);
    const std::array<int32_t, 1> second_ids = {3};
    const std::array<uint32_t, 1> second_rows = {2};
    const std::array<uint32_t, 1> second_scatter = {3};
    const auto second_oracle = cpu_region_reference(*fixture, second_ids, second_rows);

    std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_QWEN3MOE, llama_model_default_params()));
    for (auto * weight : fixture->weights) {
        CHECK(model->record_moe_readable_source(weight, weight->data, ggml_nbytes(weight)));
    }
    ggml_backend_moe_source_owner_v1 owner = {};
    CHECK(model->moe_source_owner_v1(&owner));
    fixture->query.source_generation = owner.generation;
    for (auto & source : fixture->sources) {
        source.generation = owner.generation;
    }

    ggml_backend_moe_cpu_service_config_v1 config = {};
    config.struct_size = sizeof(config);
    config.abi_version = 1;
    config.source_owner = &owner;
    config.n_threads = fixture->query.n_threads;
    config.n_lanes = 1;
    config.max_regions = 1;
    config.flags = GGML_BACKEND_MOE_CPU_SERVICE_FLAG_V1_ALLOW_UNKNOWN_THREAD_STACK_BYTES |
                   GGML_BACKEND_MOE_CPU_SERVICE_FLAG_V1_ALLOW_UNPROVEN_RUNTIME_ALLOCATIONS;
    config.prepared_payload_limit = 64 * 1024 * 1024;
    ggml_backend_moe_cpu_service_v1_t service = nullptr;
    CHECK(service_api->create(&config, &service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    ggml_backend_moe_cpu_prepared_requirements_v1 prepared = {};
    prepared.struct_size = sizeof(prepared);
    prepared.abi_version = 1;
    ggml_backend_moe_cpu_prepared_region_v1_t region = 0;
    CHECK(service_api->prepare(service, &fixture->query, &prepared, &region) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    const uint64_t allocations = ggml_allocation_count();
    for (uint64_t epoch = 1; epoch <= 16; ++epoch) {
        if ((epoch & 1) != 0) {
            cpu_region_check_execute(
                service_api, service, region, *fixture, first_oracle, first_ids, first_rows, first_scatter, epoch);
        } else {
            cpu_region_check_execute(
                service_api, service, region, *fixture, second_oracle, second_ids, second_rows, second_scatter, epoch);
        }
    }
    CHECK(ggml_allocation_count() == allocations);
    CHECK(service_api->close(service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    CHECK(service_api->destroy_region(service, &region) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    CHECK(service_api->destroy(&service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    fprintf(stderr, "test-moe-cache: CPU IQ3_XXS persistent pool parity threads=16 repeats=16 OK\n");
}

static void test_cpu_region_compact_routes(
        const ggml_backend_moe_cpu_region_query_api_v1 * query_api,
        const ggml_backend_moe_cpu_region_service_api_v1 * service_api, bool fidelity = false) {
    std::array<uint64_t, 5> shared_input_bytes = {};
    for (const uint32_t variant : {0u, 1u, 2u, 3u}) {
        const bool merged = variant == 1;
        const bool biased = variant == 2;
        for (const int64_t rows : {1, 2, 4}) {
            auto wide = cpu_region_query_graph(merged ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K,
                GGML_GLU_OP_SWIGLU, merged, biased, merged ? GGML_TYPE_Q6_K : GGML_TYPE_Q5_0,
                merged ? 1792 : 1408, 2, rows, 2, variant == 3 ? GGML_TYPE_Q5_0 : GGML_TYPE_COUNT);
            auto compact = cpu_region_query_graph(merged ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K,
                GGML_GLU_OP_SWIGLU, merged, biased, merged ? GGML_TYPE_Q6_K : GGML_TYPE_Q5_0,
                merged ? 1792 : 1408, 1, 2 * rows, 2, variant == 3 ? GGML_TYPE_Q5_0 : GGML_TYPE_COUNT);
            compact->query.flags = GGML_BACKEND_MOE_CPU_REGION_FLAG_V1_COMPACT_ROUTES;
            compact->query.n_lanes = 1;

            std::vector<int32_t> expert_ids(2 * rows);
            std::vector<uint32_t> source_rows(2 * rows);
            std::vector<uint32_t> scatter(2 * rows);
            for (int64_t route = 0; route < 2 * rows; ++route) {
                expert_ids[route] = int32_t((route * 3 + route / 2) % 4);
                source_rows[route] = uint32_t(3 + route / 2);
                scatter[route] = uint32_t(2 * rows - 1 - route);
            }
            const auto wide_oracle = cpu_region_reference(*wide, expert_ids, source_rows);
            const auto compact_oracle = cpu_region_reference(*compact, expert_ids, source_rows);
            CHECK(wide_oracle.outputs.size() == compact_oracle.outputs.size());
            for (uint32_t output = 0; output < wide->query.n_live_outputs; ++output) {
                CHECK(wide_oracle.outputs[output] == compact_oracle.outputs[output]);
            }

            ggml_backend_moe_cpu_region_binding_v1 binding = {
                sizeof(binding), uint32_t(2 * rows), uint32_t(2 * rows), expert_ids.data(),
                source_rows.data(), scatter.data(),
            };
            CHECK(query_api->validate_binding(&compact->query, &binding) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
            compact->query.flags = GGML_BACKEND_MOE_CPU_REGION_FLAG_V1_NONE;
            CHECK(query_api->validate_binding(&compact->query, &binding) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_BINDING);
            compact->query.flags = GGML_BACKEND_MOE_CPU_REGION_FLAG_V1_COMPACT_ROUTES;

            std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_QWEN3MOE, llama_model_default_params()));
            for (auto * weight : compact->weights) {
                CHECK(model->record_moe_readable_source(weight, weight->data, ggml_nbytes(weight)));
            }
            ggml_backend_moe_source_owner_v1 owner = {};
            CHECK(model->moe_source_owner_v1(&owner));
            compact->query.source_generation = owner.generation;
            for (auto & source : compact->sources) {
                source.generation = owner.generation;
            }
            ggml_backend_moe_cpu_service_config_v1 config = {};
            config.struct_size = sizeof(config);
            config.abi_version = 1;
            config.source_owner = &owner;
            config.n_threads = compact->query.n_threads;
            config.n_lanes = 1;
            config.max_regions = fidelity ? 2 : 1;
            config.flags = GGML_BACKEND_MOE_CPU_SERVICE_FLAG_V1_ALLOW_UNKNOWN_THREAD_STACK_BYTES |
                           GGML_BACKEND_MOE_CPU_SERVICE_FLAG_V1_ALLOW_UNPROVEN_RUNTIME_ALLOCATIONS;
            config.prepared_payload_limit = 64 * 1024 * 1024;
            ggml_backend_moe_cpu_service_v1_t service = nullptr;
            CHECK(service_api->create(&config, &service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
            ggml_backend_moe_cpu_prepared_requirements_v1 requirements = {};
            requirements.struct_size = sizeof(requirements);
            requirements.abi_version = 1;
            ggml_backend_moe_cpu_prepared_region_v1_t region = 0;
            CHECK(service_api->prepare(service, &compact->query, &requirements, &region) ==
                  GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
            if (fidelity) {
                ggml_backend_moe_cpu_region_requirements_v1 graph_requirements = {};
                graph_requirements.struct_size = sizeof(graph_requirements);
                CHECK(query_api->query(&compact->query, &graph_requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
                CHECK(requirements.execution.graph_nodes == 0 && requirements.execution.graph_work_bytes == 0 &&
                    requirements.execution.node_data_bytes == 0 && requirements.execution.dynamic_input_bytes == 0 &&
                    requirements.lane_context_bytes == 0 &&
                    requirements.tensor_count == compact->query.n_dynamic_inputs + compact->query.n_live_outputs &&
                    requirements.lane_metadata_bytes == requirements.tensor_count * sizeof(ggml_tensor));
                CHECK(requirements.execution.binding_metadata_bytes == graph_requirements.binding_metadata_bytes &&
                    requirements.execution.output_staging_bytes == graph_requirements.output_staging_bytes);
                const auto extra = requirements.execution.lane_execution_bytes -
                    requirements.execution.binding_metadata_bytes - requirements.execution.output_staging_bytes;
                if (variant == 0) { shared_input_bytes[rows] = extra; }
                if (variant == 3) {
                    const auto type = ggml_get_type_traits_cpu(compact->weights[1]->type)->vec_dot_type;
                    const uint64_t separate = compact->query.source_row_capacity * ggml_row_size(type, compact->activation->ne[0]);
                    CHECK(extra == shared_input_bytes[rows] + (separate + 63) / 64 * 64);
                }
                auto limited = compact->query;
                limited.lane_execution_byte_limit = requirements.execution.all_lane_execution_bytes;
                auto exact = requirements;
                ggml_backend_moe_cpu_prepared_region_v1_t exact_region = 0;
                CHECK(service_api->prepare(service, &limited, &exact, &exact_region) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
                CHECK(exact.execution.all_lane_execution_bytes == limited.lane_execution_byte_limit);
                CHECK(service_api->destroy_region(service, &exact_region) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
                --limited.lane_execution_byte_limit;
                CHECK(service_api->prepare(service, &limited, &exact, &exact_region) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY);
                CHECK(exact_region == 0);
                fprintf(stderr, "test-moe-cache: CPU fidelity compact R=%lld variant=%u graph_lane=%llu fidelity_lane=%llu views=%llu context=%llu prepared=%llu\n",
                    (long long) rows, variant, (unsigned long long) graph_requirements.lane_execution_bytes,
                    (unsigned long long) requirements.execution.lane_execution_bytes,
                    (unsigned long long) requirements.lane_metadata_bytes, (unsigned long long) requirements.lane_context_bytes,
                    (unsigned long long) requirements.prepared_payload_bytes);
            }
            const uint64_t allocations = ggml_allocation_count();
            cpu_region_check_execute(service_api, service, region, *compact, compact_oracle,
                expert_ids, source_rows, scatter, 100 + variant * 10 + rows, true);
            const uint32_t active = std::max<uint32_t>(1, expert_ids.size() / 2);
            std::vector<int32_t> active_experts(expert_ids.begin(), expert_ids.begin() + active);
            std::vector<uint32_t> active_sources(source_rows.begin(), source_rows.begin() + active);
            std::vector<uint32_t> active_scatter(scatter.begin(), scatter.begin() + active);
            cpu_region_check_execute(service_api, service, region, *compact, compact_oracle,
                active_experts, active_sources, active_scatter, 200 + variant * 10 + rows, false, active);
            const auto private_flag = GGML_BACKEND_MOE_CPU_EXECUTE_FLAG_V1_PRIVATE_OUTPUTS;
            cpu_region_check_execute(service_api, service, region, *compact, compact_oracle,
                expert_ids, source_rows, scatter, 300 + variant * 10 + rows, true, 0, private_flag, 16);
            cpu_region_check_execute(service_api, service, region, *compact, compact_oracle,
                active_experts, active_sources, active_scatter, 400 + variant * 10 + rows, false, active, private_flag, 3, 1);
            CHECK(ggml_allocation_count() == allocations);
            CHECK(service_api->close(service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
            CHECK(service_api->drain(service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
            CHECK(service_api->destroy_region(service, &region) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
            CHECK(service_api->destroy(&service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        }
    }
    fprintf(stderr, "test-moe-cache: CPU compact route batches R1/R2/R4 merged/bias/mixed-format shared/separate quantization parity OK\n");
}

static void test_cpu_region_exact_buckets(const ggml_backend_moe_cpu_region_service_api_v1 * api) {
    for (const uint32_t variant : {0u, 1u, 2u}) {
        const bool merged = variant == 1;
        const bool biased = variant == 2;
        std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_QWEN3MOE, llama_model_default_params()));
        std::array<std::unique_ptr<cpu_region_query_fixture>, 4> fixtures;
        std::array<cpu_region_oracle, 4> oracles;
        for (uint32_t count = 1; count <= fixtures.size(); ++count) {
            auto & fixture = fixtures[count - 1];
            fixture = cpu_region_query_graph(merged ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K,
                GGML_GLU_OP_SWIGLU, merged, biased, merged ? GGML_TYPE_Q6_K : GGML_TYPE_Q5_0,
                merged ? 1792 : 1408, count, 1);
            fixture->query.n_lanes = 1;
            const std::vector<int32_t> ids(count, 2);
            const std::vector<uint32_t> rows(count, 3);
            oracles[count - 1] = cpu_region_reference(*fixture, ids, rows);
            for (auto * weight : fixture->weights) {
                CHECK(model->record_moe_readable_source(weight, weight->data, ggml_nbytes(weight)));
            }
        }
        std::array<ggml_backend_moe_cpu_region_query_v1, 4> queries;
        std::array<const ggml_backend_moe_cpu_region_query_v1 *, 4> query_ptrs;
        std::array<std::vector<ggml_backend_moe_cpu_region_source_v1>, 4> saved_sources;
        for (uint32_t i = 0; i < fixtures.size(); ++i) {
            auto & fixture = *fixtures[i];
            saved_sources[i] = fixture.sources;
            for (size_t source = 0; source < fixture.sources.size(); ++source) {
                fixture.sources[source] = fixtures[0]->sources[source];
                fixture.sources[source].tensor = fixture.weights[source];
            }
            queries[i] = fixture.query;
            queries[i].n_live_outputs = 1;
            queries[i].source_row_capacity = 1;
            queries[i].scatter_capacity = fixtures.size();
            query_ptrs[i] = &queries[i];
        }
        ggml_backend_moe_hybrid_region_v1 descriptor = {};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.activation = fixtures.back()->activation;
        descriptor.ids = fixtures.back()->ids;
        descriptor.output = fixtures.back()->raw_down;
        descriptor.query = query_ptrs[0];
        descriptor.cpu_queries = query_ptrs.data();
        descriptor.n_cpu_queries = query_ptrs.size();
        CHECK(ggml_backend_moe_hybrid_get_geometry_v1(descriptor.activation, descriptor.ids, descriptor.output,
            fixtures.back()->weights[0]->ne[2], &descriptor.geometry) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        CHECK(ggml_backend_moe_hybrid_validate_buckets_v1(&descriptor) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        const int32_t weight_expert = 2;
        std::vector<ggml_backend_moe_hybrid_route_v1> route_records(descriptor.geometry.route_capacity);
        for (uint32_t rank = 0; rank < route_records.size(); ++rank) {
            route_records[rank] = {0, rank, 0, rank};
        }
        ggml_backend_moe_hybrid_binding_v1 routing = {};
        routing.struct_size = sizeof(routing);
        routing.active_rows = 1;
        routing.n_weights = 1;
        routing.n_routes = route_records.size();
        routing.epoch = 1;
        routing.source_generation = descriptor.query->source_generation;
        routing.weight_experts = &weight_expert;
        routing.routes = route_records.data();
        std::array<uint32_t, 3> selected = {2, 0, 3}, bound_rows = {}, bound_scatter = {};
        std::array<int32_t, 3> bound_experts = {};
        std::vector<uint8_t> marks(size_t(descriptor.geometry.expert_count) + descriptor.geometry.route_capacity);
        ggml_backend_moe_cpu_region_binding_v1 binding = {};
        CHECK(ggml_backend_moe_hybrid_bind_cpu_row_v1(&descriptor, &routing, 1, 0, selected.data(), selected.size(),
            selected.size(), bound_experts.data(), bound_rows.data(), bound_scatter.data(), &binding, marks.data(), marks.size()) ==
            GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        CHECK(binding.n_rows == 1 && binding.n_routes == selected.size() && bound_scatter == selected);
        CHECK(std::all_of(bound_experts.begin(), bound_experts.end(), [](int32_t expert) { return expert == 2; }));
        auto & altered = *fixtures[2];
        const auto rejected = [&] {
            CHECK(ggml_backend_moe_hybrid_validate_buckets_v1(&descriptor) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
            CHECK(memcmp(altered.raw_down->data, oracles[2].outputs[0].data(), oracles[2].outputs[0].size()) == 0);
        };
        altered.raw_down->op_params[0] ^= 1;
        rejected();
        altered.raw_down->op_params[0] ^= 1;
        const auto * input = altered.raw_down->src[1];
        altered.raw_down->src[1] = altered.activation;
        rejected();
        altered.raw_down->src[1] = const_cast<ggml_tensor *>(input);
        ++altered.sources[0].generation;
        rejected();
        --altered.sources[0].generation;
        altered.sources[0].witness = altered.weights[0];
        rejected();
        altered.sources[0].witness = fixtures[0]->sources[0].witness;
        const ggml_tensor * wrong_output = altered.hidden;
        queries[2].live_outputs = &wrong_output;
        rejected();
        queries[2].live_outputs = altered.live_outputs.data();
        auto * hidden_source = altered.hidden->src[0];
        const auto saved_stride = hidden_source->nb[1];
        ++hidden_source->nb[1];
        rejected();
        hidden_source->nb[1] = saved_stride;
        if (merged) {
            ++hidden_source->view_offs;
            rejected();
            --hidden_source->view_offs;
        }
        CHECK(ggml_backend_moe_hybrid_validate_buckets_v1(&descriptor) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        for (uint32_t i = 0; i < fixtures.size(); ++i) {
            fixtures[i]->sources = std::move(saved_sources[i]);
            fixtures[i]->query.sources = fixtures[i]->sources.data();
        }
        ggml_backend_moe_source_owner_v1 owner = {};
        CHECK(model->moe_source_owner_v1(&owner));
        for (auto & fixture : fixtures) {
            fixture->query.source_generation = owner.generation;
            for (auto & source : fixture->sources) {
                source.generation = owner.generation;
            }
        }
        ggml_backend_moe_cpu_service_config_v1 config = {};
        config.struct_size = sizeof(config);
        config.abi_version = 1;
        config.source_owner = &owner;
        config.n_threads = 2;
        config.n_lanes = 1;
        config.max_regions = fixtures.size();
        config.flags = GGML_BACKEND_MOE_CPU_SERVICE_FLAG_V1_ALLOW_UNKNOWN_THREAD_STACK_BYTES |
                       GGML_BACKEND_MOE_CPU_SERVICE_FLAG_V1_ALLOW_UNPROVEN_RUNTIME_ALLOCATIONS;
        config.prepared_payload_limit = 16 * 1024 * 1024;
        uint64_t payload_bytes = 0;
        for (const uint32_t budget : {0u, 1u, 2u}) {
            if (budget != 0) {
                config.prepared_payload_limit = payload_bytes - (budget == 2);
            }
            ggml_backend_moe_cpu_service_v1_t service = nullptr;
            CHECK(api->create(&config, &service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
            if (budget == 0) {
                ggml_backend_moe_cpu_service_state_v1 state = {};
                state.struct_size = sizeof(state);
                CHECK(api->state(service, &state) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
                payload_bytes = state.prepared_payload_bytes;
            }
            std::array<ggml_backend_moe_cpu_prepared_region_v1_t, 4> regions = {};
            bool rejected = false;
            for (uint32_t count = 1; count <= fixtures.size(); ++count) {
                auto & fixture = *fixtures[count - 1];
                ggml_backend_moe_cpu_prepared_requirements_v1 requirements = {};
                requirements.struct_size = sizeof(requirements);
                requirements.abi_version = 1;
                const int32_t status = api->prepare(service, &fixture.query, &requirements, &regions[count - 1]);
                if (budget == 2 && status == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY) {
                    CHECK(regions[count - 1] == 0);
                    rejected = true;
                    break;
                }
                CHECK(status == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
                CHECK(requirements.execution.route_capacity == count && requirements.prepared_payload_bytes != 0);
                if (budget != 2) {
                    if (budget == 0) { payload_bytes += requirements.prepared_payload_bytes; }
                    const std::vector<int32_t> ids(count, 2);
                    const std::vector<uint32_t> rows(count, 3);
                    std::vector<uint32_t> scatter(count);
                    std::iota(scatter.begin(), scatter.end(), 1);
                    const uint64_t allocations = ggml_allocation_count();
                    for (uint64_t pass : {0, 1}) {
                        cpu_region_check_execute(api, service, regions[count - 1], fixture, oracles[count - 1],
                            ids, rows, scatter, 2 * count + pass, true);
                    }
                    CHECK(ggml_allocation_count() == allocations);
                }
            }
            CHECK(rejected == (budget == 2));
            CHECK(api->close(service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
            CHECK(api->drain(service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
            for (auto & region : regions) {
                if (region != 0) {
                    CHECK(api->destroy_region(service, &region) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
                }
            }
            CHECK(api->destroy(&service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        }
        fprintf(stderr, "test-moe-cache: CPU exact count buckets=1..4 merged=%d biased=%d charged_bytes=%llu capacity/count/allocation gates OK\n",
            merged, biased, (unsigned long long) payload_bytes);
    }
}

static void test_cpu_source_pool_descriptors(
        const ggml_backend_moe_cpu_region_service_api_v1 * api,
        const ggml_backend_moe_cpu_region_service_api_v1 * graph_api) {
    struct signature {
        ggml_type gu, down, up;
        ggml_glu_op activation;
        bool fused, bias;
        int64_t embd, ff, rows, experts;
        uint32_t threads;
    };
    for (const auto & v : {
            signature{GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, GGML_GLU_OP_GEGLU, false, false, 32, 48, 9, 7, 1},
            signature{GGML_TYPE_Q4_K, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0, GGML_GLU_OP_SWIGLU, false, true, 512, 768, 17, 4, 1},
            signature{GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_COUNT, GGML_GLU_OP_GEGLU, true, true, 256, 256, 769, 97, 4}}) {
        auto fixture = cpu_region_query_graph(v.gu, v.activation, v.fused, v.bias, v.down,
            v.ff, 2, v.rows, v.threads, v.up, v.embd, v.experts);
        fixture->query.n_lanes = 1;
        std::vector<int32_t> ids(2 * v.rows);
        std::vector<uint32_t> rows(ids.size()), scatter(ids.size());
        for (uint32_t i = 0; i < ids.size(); ++i) {
            ids[i] = int32_t((3 * i + i / 2) % v.experts);
            rows[i] = 3 + i / 2;
            scatter[i] = ids.size() - 1 - i;
        }
        const auto oracle = cpu_region_reference(*fixture, ids, rows);
        std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_QWEN3MOE, llama_model_default_params()));
        for (auto * weight : fixture->weights) { CHECK(model->record_moe_readable_source(weight, weight->data, ggml_nbytes(weight))); }
        ggml_backend_moe_source_owner_v1 owner = {};
        CHECK(model->moe_source_owner_v1(&owner));
        fixture->query.source_generation = owner.generation;
        for (auto & source : fixture->sources) { source.generation = owner.generation; }
        ggml_backend_moe_cpu_service_config_v1 config = {};
        config.struct_size = sizeof(config);
        config.abi_version = 1;
        config.source_owner = &owner;
        config.n_threads = v.threads;
        config.n_lanes = 1;
        config.max_regions = 2;
        config.flags = GGML_BACKEND_MOE_CPU_SERVICE_FLAG_V1_ALLOW_UNKNOWN_THREAD_STACK_BYTES |
            GGML_BACKEND_MOE_CPU_SERVICE_FLAG_V1_ALLOW_UNPROVEN_RUNTIME_ALLOCATIONS;
        config.prepared_payload_limit = 128 * 1024 * 1024;
        ggml_backend_moe_cpu_service_v1_t service = nullptr;
        CHECK(api->create(&config, &service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        ggml_backend_moe_cpu_prepared_requirements_v1 prepared = {};
        prepared.struct_size = sizeof(prepared); prepared.abi_version = 1;
        ggml_backend_moe_cpu_prepared_region_v1_t region = 0;
        CHECK(api->prepare(service, &fixture->query, &prepared, &region) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        const uint64_t allocations = ggml_allocation_count();
        cpu_region_check_execute(api, service, region, *fixture, oracle, ids, rows, scatter, 1);
        cpu_region_check_execute(api, service, region, *fixture, oracle, ids, rows, scatter, 2);
        CHECK(ggml_allocation_count() == allocations);
        ggml_backend_moe_cpu_service_state_v1 before = {}, after = {};
        before.struct_size = sizeof(before); after.struct_size = sizeof(after);
        CHECK(api->state(service, &before) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        ggml_backend_moe_cpu_prepared_region_v1_t graph_region = 0;
        auto graph_prepared = prepared;
        CHECK(graph_api->prepare(service, &fixture->query, &graph_prepared, &graph_region) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        CHECK(api->state(service, &after) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        CHECK(after.prepared_payload_bytes > before.prepared_payload_bytes + graph_prepared.prepared_payload_bytes);
        const uint64_t mixed_allocations = ggml_allocation_count();
        cpu_region_check_execute(graph_api, service, graph_region, *fixture, oracle, ids, rows, scatter, 3);
        cpu_region_check_execute(api, service, region, *fixture, oracle, ids, rows, scatter, 4);
        cpu_region_check_execute(graph_api, service, graph_region, *fixture, oracle, ids, rows, scatter, 5);
        cpu_region_check_execute(api, service, region, *fixture, oracle, ids, rows, scatter, 6);
        CHECK(ggml_allocation_count() == mixed_allocations);
        CHECK(api->close(service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        CHECK(api->drain(service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        CHECK(graph_api->destroy_region(service, &graph_region) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        CHECK(api->destroy_region(service, &region) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        CHECK(api->destroy(&service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        fprintf(stderr, "test-moe-cache: source pool embd=%lld ff=%lld rows=%lld experts=%lld threads=%u mixed-lane exact parity/ledger/cleanup OK\n",
            (long long) v.embd, (long long) v.ff, (long long) v.rows, (long long) v.experts, v.threads);
    }
}

static void test_cpu_region_query_proc(bool fidelity = false) {
    ggml_backend_reg_t cpu_reg = ggml_backend_cpu_reg();
    CHECK(cpu_reg != nullptr);
    auto get_api = reinterpret_cast<ggml_backend_moe_cpu_region_query_v1_t>(
        ggml_backend_reg_get_proc_address(cpu_reg, GGML_BACKEND_MOE_CPU_REGION_QUERY_V1_PROC_NAME));
    CHECK(get_api != nullptr);
    const auto * api = get_api();
    CHECK(api != nullptr && api->abi_version == 1 && api->struct_size == sizeof(*api));
    auto get_service_api = reinterpret_cast<ggml_backend_moe_cpu_region_service_v1_t>(
        ggml_backend_reg_get_proc_address(cpu_reg, fidelity ? GGML_BACKEND_MOE_CPU_FIDELITY_SERVICE_V1_PROC_NAME :
            GGML_BACKEND_MOE_CPU_REGION_SERVICE_V1_PROC_NAME));
    CHECK(get_service_api != nullptr);
    const auto * service_api = get_service_api();
    CHECK(service_api != nullptr && service_api->abi_version == 1 && service_api->struct_size == sizeof(*service_api));
    test_cpu_region_mixed_banks(api, service_api);
    test_cpu_region_iq3_persistent_parity(service_api);
    test_cpu_region_compact_routes(api, service_api, fidelity);
    test_cpu_region_exact_buckets(service_api);
    if (fidelity && ggml_moe_fidelity_selection().source_pool && !ggml_moe_fidelity_selection().reference) {
        const auto get_graph_api = reinterpret_cast<ggml_backend_moe_cpu_region_service_v1_t>(
            ggml_backend_reg_get_proc_address(cpu_reg, GGML_BACKEND_MOE_CPU_REGION_SERVICE_V1_PROC_NAME));
        CHECK(get_graph_api);
        test_cpu_source_pool_descriptors(service_api, get_graph_api());
    }

    auto q4_0 = cpu_region_query_graph(GGML_TYPE_Q4_0, GGML_GLU_OP_SWIGLU);
    auto q4_k = cpu_region_query_graph(GGML_TYPE_Q4_K, GGML_GLU_OP_GEGLU);
    auto merged_bias = cpu_region_query_graph(GGML_TYPE_Q4_0, GGML_GLU_OP_GEGLU, true, true);
    CHECK(q4_0->common_tail->op == GGML_OP_ADD_ID && q4_0->common_tail->src[0] == q4_0->raw_down);
    CHECK(q4_k->common_tail->op == GGML_OP_ADD_ID && q4_k->common_tail->src[0] == q4_k->raw_down);
    CHECK(q4_0->query.graph_uid != 0 && q4_k->query.graph_uid != 0);
    for (uint32_t i = 0; i < q4_k->query.n_body_nodes; ++i) {
        CHECK(q4_k->query.body_nodes[i] == ggml_graph_node(q4_k->graph, i));
    }
    CHECK(ggml_get_glu_op(q4_0->hidden) == GGML_GLU_OP_SWIGLU);
    CHECK(ggml_get_glu_op(q4_k->hidden) == GGML_GLU_OP_GEGLU);

    ggml_backend_moe_cpu_region_requirements_v1 q4_0_requirements = {};
    q4_0_requirements.struct_size                                 = sizeof(q4_0_requirements);
    CHECK(api->query(&q4_0->query, &q4_0_requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    CHECK(q4_0_requirements.expert_count == 4 && q4_0_requirements.bucket_rows == 2 &&
          q4_0_requirements.routes_per_row == 2 && q4_0_requirements.route_capacity == 4 &&
          q4_0_requirements.immutable_sources == 3 && q4_0_requirements.n_live_outputs == 1 &&
          q4_0_requirements.graph_work_bytes > 0 && q4_0_requirements.node_data_bytes > 0 &&
          q4_0_requirements.dynamic_input_bytes > 0 && q4_0_requirements.output_staging_bytes > 0 &&
          q4_0_requirements.binding_metadata_bytes >= 4 * 2 * sizeof(uint32_t) &&
          q4_0_requirements.binding_metadata_bytes % CACHE_LINE_SIZE == 0 &&
          q4_0_requirements.all_lane_execution_bytes == q4_0_requirements.lane_execution_bytes * q4_0->query.n_lanes &&
          q4_0_requirements.live_outputs[0].tensor == q4_0->raw_down);

    ggml_backend_moe_cpu_region_requirements_v1 q4_k_requirements = {};
    q4_k_requirements.struct_size                                 = sizeof(q4_k_requirements);
    CHECK(api->query(&q4_k->query, &q4_k_requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    CHECK(q4_k_requirements.n_live_outputs == 2 && q4_k_requirements.live_outputs[0].tensor == q4_k->raw_down &&
          q4_k_requirements.live_outputs[1].tensor == q4_k->hidden &&
          q4_k_requirements.live_outputs[1].offset >= q4_k_requirements.live_outputs[0].bytes);

    std::array<int32_t, 4>                 expert_ids  = { 1, 3, 0, 2 };
    std::array<uint32_t, 4>                source_rows = { 9, 9, 2, 2 };
    std::array<uint32_t, 4>                scatter     = { 6, 7, 2, 3 };
    ggml_backend_moe_cpu_region_binding_v1 binding     = {
        sizeof(binding), 2, 4, expert_ids.data(), source_rows.data(), scatter.data(),
    };
    CHECK(api->validate_binding(&q4_k->query, &binding) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    expert_ids[1] = -1;
    CHECK(api->validate_binding(&q4_k->query, &binding) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_BINDING);
    expert_ids[1] = 3;
    scatter[3]    = scatter[0];
    CHECK(api->validate_binding(&q4_k->query, &binding) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_BINDING);
    scatter[3] = 3;

    const auto q4_0_oracle = cpu_region_reference(*q4_0, expert_ids, source_rows);
    const auto q4_k_oracle = cpu_region_reference(*q4_k, expert_ids, source_rows);
    const auto merged_bias_oracle = cpu_region_reference(*merged_bias, expert_ids, source_rows);
    ggml_backend_moe_cpu_region_requirements_v1 merged_requirements = {};
    merged_requirements.struct_size = sizeof(merged_requirements);
    CHECK(api->query(&merged_bias->query, &merged_requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    auto * merged_view = merged_bias->hidden->src[1];
    CHECK(merged_view->op == GGML_OP_VIEW && merged_bias->hidden->src[0]->op == GGML_OP_ADD_ID);
    const size_t view_offset = merged_view->view_offs;
    merged_view->view_offs = merged_view->view_src->nb[1];
    CHECK(api->query(&merged_bias->query, &merged_requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION);
    merged_view->view_offs = view_offset;
    const int32_t view_param = merged_view->op_params[0];
    merged_view->op_params[0] = 0;
    CHECK(api->query(&merged_bias->query, &merged_requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION);
    merged_view->op_params[0] = view_param;
    auto view_live_query = merged_bias->query;
    const ggml_tensor * view_output = merged_view;
    view_live_query.live_outputs = &view_output;
    view_live_query.n_live_outputs = 1;
    CHECK(api->query(&view_live_query, &merged_requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION);
    auto * expert_bias = merged_bias->hidden->src[0]->src[1];
    --expert_bias->ne[1];
    CHECK(api->query(&merged_bias->query, &merged_requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION);
    ++expert_bias->ne[1];

    auto rejected                      = q4_k->query;
    rejected.lane_execution_byte_limit = q4_k_requirements.all_lane_execution_bytes - 1;
    const uint64_t accepted_bytes      = q4_k_requirements.all_lane_execution_bytes;
    CHECK(api->query(&rejected, &q4_k_requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY);
    CHECK(q4_k_requirements.all_lane_execution_bytes == accepted_bytes);
    rejected                      = q4_k->query;
    rejected.output_staging_limit = 1;
    CHECK(api->query(&rejected, &q4_k_requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY);
    rejected       = q4_k->query;
    rejected.flags = GGML_BACKEND_MOE_CPU_REGION_FLAG_V1_HAS_LORA;
    CHECK(api->query(&rejected, &q4_k_requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION);
    rejected = q4_k->query;
    rejected.n_sources -= 1;
    CHECK(api->query(&rejected, &q4_k_requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_MISSING_SOURCE);
    rejected           = q4_k->query;
    auto wrong_sources = q4_k->sources;
    wrong_sources[0].generation += 1;
    rejected.sources = wrong_sources.data();
    CHECK(api->query(&rejected, &q4_k_requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_MISSING_SOURCE);
    rejected = q4_k->query;
    std::vector<const ggml_tensor *> wrong_order(rejected.body_nodes, rejected.body_nodes + rejected.n_body_nodes);
    std::swap(wrong_order[0], wrong_order[1]);
    rejected.body_nodes = wrong_order.data();
    CHECK(api->query(&rejected, &q4_k_requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);

    const int graph_size = q4_k->graph->size;
    q4_k->graph->size    = q4_k->graph->n_nodes - 1;
    CHECK(api->query(&q4_k->query, &q4_k_requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
    q4_k->graph->size          = graph_size;
    ggml_tensor ** graph_nodes = q4_k->graph->nodes;
    q4_k->graph->nodes         = nullptr;
    CHECK(api->query(&q4_k->query, &q4_k_requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
    q4_k->graph->nodes = graph_nodes;

    rejected           = q4_k->query;
    rejected.n_threads = GGML_MAX_N_THREADS + 1;
    CHECK(api->query(&rejected, &q4_k_requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
    const int64_t activation_ne1 = q4_k->activation->ne[1];
    q4_k->activation->ne[1] = 2;
    CHECK(api->query(&q4_k->query, &q4_k_requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
    q4_k->activation->ne[1] = activation_ne1;
    rejected                  = q4_k->query;
    const int64_t ids_ne0     = q4_k->ids->ne[0];
    q4_k->ids->ne[0]          = UINT32_MAX;
    rejected.routes_per_row   = UINT32_MAX;
    rejected.scatter_capacity = UINT32_MAX;
    CHECK(api->query(&rejected, &q4_k_requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY);
    q4_k->ids->ne[0] = ids_ne0;

    int hidden_index = -1;
    for (int i = 0; i < ggml_graph_n_nodes(q4_k->graph); ++i) {
        if (ggml_graph_node(q4_k->graph, i) == q4_k->hidden) {
            hidden_index = i;
        }
    }
    CHECK(hidden_index > 0);
    std::swap(ggml_graph_nodes(q4_k->graph)[0], ggml_graph_nodes(q4_k->graph)[hidden_index]);
    CHECK(api->query(&q4_k->query, &q4_k_requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
    std::swap(ggml_graph_nodes(q4_k->graph)[0], ggml_graph_nodes(q4_k->graph)[hidden_index]);

    ggml_tensor * graph_node_1       = ggml_graph_nodes(q4_k->graph)[1];
    ggml_graph_nodes(q4_k->graph)[1] = ggml_graph_nodes(q4_k->graph)[0];
    CHECK(api->query(&q4_k->query, &q4_k_requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
    ggml_graph_nodes(q4_k->graph)[1] = graph_node_1;

    ggml_tensor * glu_src0 = q4_k->hidden->src[0];
    q4_k->hidden->src[0]   = q4_k->hidden;
    CHECK(api->query(&q4_k->query, &q4_k_requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
    q4_k->hidden->src[0] = glu_src0;

    const size_t weight_row_stride = q4_k->weights[0]->nb[1];
    q4_k->weights[0]->nb[1]        = ggml_row_size(q4_k->weights[0]->type, q4_k->weights[0]->ne[0]) - 1;
    CHECK(api->query(&q4_k->query, &q4_k_requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION);
    q4_k->weights[0]->nb[1] = weight_row_stride;

    const ggml_type weight_type = q4_k->weights[0]->type;
    q4_k->weights[0]->type      = (ggml_type) 31;
    CHECK(api->query(&q4_k->query, &q4_k_requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION);
    q4_k->weights[0]->type = weight_type;
    void * weight_extra     = q4_k->weights[0]->extra;
    q4_k->weights[0]->extra = q4_k->weights[0];
    CHECK(api->query(&q4_k->query, &q4_k_requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION);
    q4_k->weights[0]->extra = weight_extra;

    const int32_t glu_op       = q4_k->hidden->op_params[0];
    q4_k->hidden->op_params[0] = GGML_GLU_OP_COUNT;
    CHECK(api->query(&q4_k->query, &q4_k_requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION);
    q4_k->hidden->op_params[0] = glu_op;
    q4_k->hidden->op_params[1] = 1;
    CHECK(api->query(&q4_k->query, &q4_k_requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION);
    q4_k->hidden->op_params[1] = 0;
    ggml_tensor * glu_src1     = q4_k->hidden->src[1];
    q4_k->hidden->src[1]       = nullptr;
    CHECK(api->query(&q4_k->query, &q4_k_requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION);
    q4_k->hidden->src[1] = glu_src1;

    const size_t glu_stride = q4_k->hidden->nb[2];
    q4_k->hidden->nb[2]     = q4_k->hidden->nb[1] - 1;
    CHECK(api->query(&q4_k->query, &q4_k_requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION);
    q4_k->hidden->nb[2]            = glu_stride;
    const size_t glu_source_stride = q4_k->hidden->src[0]->nb[2];
    q4_k->hidden->src[0]->nb[2]    = q4_k->hidden->src[0]->nb[1] - 1;
    CHECK(api->query(&q4_k->query, &q4_k_requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION);
    q4_k->hidden->src[0]->nb[2] = glu_source_stride;

    ggml_tensor * glu_src2 = q4_k->hidden->src[2];
    q4_k->hidden->src[2]   = q4_k->activation;
    CHECK(api->query(&q4_k->query, &q4_k_requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION);
    q4_k->hidden->src[2]    = glu_src2;
    ggml_tensor * down_src3 = q4_k->raw_down->src[3];
    q4_k->raw_down->src[3]  = q4_k->activation;
    CHECK(api->query(&q4_k->query, &q4_k_requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION);
    q4_k->raw_down->src[3] = down_src3;

    ggml_tensor * hidden_view = q4_k->hidden->view_src;
    q4_k->hidden->view_src    = q4_k->hidden->src[0];
    CHECK(api->query(&q4_k->query, &q4_k_requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION);
    q4_k->hidden->view_src = hidden_view;

    const ggml_op hidden_op = q4_k->hidden->op;
    for (ggml_op unsupported : { GGML_OP_ADD_ID, GGML_OP_MUL, GGML_OP_UNARY, GGML_OP_VIEW }) {
        q4_k->hidden->op = unsupported;
        CHECK(api->query(&q4_k->query, &q4_k_requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION);
    }
    q4_k->hidden->op = hidden_op;

    const int64_t activation_ne0 = q4_k->activation->ne[0];
    const size_t  activation_nb1 = q4_k->activation->nb[1];
    q4_k->activation->ne[0]      = INT64_C(1) << 62;
    q4_k->activation->nb[1]      = 0;
    CHECK(api->query(&q4_k->query, &q4_k_requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION);
    q4_k->activation->ne[0] = activation_ne0;
    q4_k->activation->nb[1] = activation_nb1;

    const int64_t  expert_count         = q4_k->weights[0]->ne[2];
    const size_t   weight_nb3           = q4_k->weights[0]->nb[3];
    const uint64_t source_bytes         = q4_k->sources[0].bytes;
    const int64_t  planner_expert_limit = (INT_MAX - CACHE_LINE_SIZE) / CACHE_LINE_SIZE;
    q4_k->weights[0]->ne[2]             = planner_expert_limit + 1;
    q4_k->weights[0]->nb[3]             = q4_k->weights[0]->nb[2] * (uint64_t) q4_k->weights[0]->ne[2];
    q4_k->sources[0].bytes              = UINT64_MAX;
    CHECK(api->query(&q4_k->query, &q4_k_requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION);
    q4_k->weights[0]->ne[2] = expert_count;
    q4_k->weights[0]->nb[3] = weight_nb3;
    q4_k->sources[0].bytes  = source_bytes;

    std::array<int64_t, 3>  expert_counts       = {};
    std::array<size_t, 3>   expert_nb3          = {};
    std::array<uint64_t, 3> expert_source_bytes = {};
    for (size_t i = 0; i < q4_k->weights.size(); ++i) {
        expert_counts[i]        = q4_k->weights[i]->ne[2];
        expert_nb3[i]           = q4_k->weights[i]->nb[3];
        expert_source_bytes[i]  = q4_k->sources[i].bytes;
        q4_k->weights[i]->ne[2] = planner_expert_limit;
        q4_k->weights[i]->nb[3] = q4_k->weights[i]->nb[2] * (uint64_t) q4_k->weights[i]->ne[2];
        q4_k->sources[i].bytes  = UINT64_MAX;
    }
    CHECK(api->query(&q4_k->query, &q4_k_requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY);
    for (size_t i = 0; i < q4_k->weights.size(); ++i) {
        q4_k->weights[i]->ne[2] = expert_counts[i];
        q4_k->weights[i]->nb[3] = expert_nb3[i];
        q4_k->sources[i].bytes  = expert_source_bytes[i];
    }

    CHECK(ggml_prec_set_acc(q4_k->raw_down, GGML_PREC_F32));
    CHECK(api->query(&q4_k->query, &q4_k_requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_PRECISION);
    CHECK(ggml_prec_set_acc(q4_k->raw_down, GGML_PREC_UNDEFINED));
    CHECK(api->query(&q4_k->query, &q4_k_requirements) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);

    auto readable_source = reinterpret_cast<ggml_backend_moe_cache_readable_source_t>(
        ggml_backend_reg_get_proc_address(ggml_backend_cuda_reg(), GGML_BACKEND_MOE_CACHE_READABLE_SOURCE_PROC_NAME));
    CHECK(readable_source != nullptr);
    const llama_model_params model_params = llama_model_default_params();
    std::unique_ptr<llama_model> source_model(llama_model_create(LLM_ARCH_DEEPSEEK4, model_params));
    for (auto * fixture : { q4_0.get(), q4_k.get(), merged_bias.get() }) {
        for (auto * weight : fixture->weights) {
            CHECK(readable_source(weight->buffer, weight->data, ggml_nbytes(weight)));
            CHECK(source_model->record_moe_readable_source(weight, weight->data, ggml_nbytes(weight)));
        }
    }
    ggml_backend_moe_source_owner_v1 source_owner = {};
    CHECK(source_model->moe_source_owner_v1(&source_owner));
    for (auto * fixture : { q4_0.get(), q4_k.get(), merged_bias.get() }) {
        fixture->query.source_generation = source_owner.generation;
        for (auto & source : fixture->sources) {
            source.generation = source_owner.generation;
        }
    }
    ggml_backend_moe_cpu_service_config_v1 config = {};
    config.struct_size            = sizeof(config);
    config.abi_version            = 1;
    config.source_owner           = &source_owner;
    config.n_threads              = q4_0->query.n_threads;
    config.n_lanes                = q4_0->query.n_lanes;
    config.max_regions            = 3;
    config.flags                  = GGML_BACKEND_MOE_CPU_SERVICE_FLAG_V1_ALLOW_UNKNOWN_THREAD_STACK_BYTES;
    config.prepared_payload_limit = 256 * 1024 * 1024;
    ggml_backend_moe_cpu_service_v1_t service = nullptr;
    const int32_t runtime_gate_status = service_api->create(&config, &service);
    if (runtime_gate_status == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK) {
        CHECK(service_api->close(service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        CHECK(service_api->destroy(&service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    } else {
        CHECK(runtime_gate_status == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT && service == nullptr);
    }
    config.flags |= GGML_BACKEND_MOE_CPU_SERVICE_FLAG_V1_ALLOW_UNPROVEN_RUNTIME_ALLOCATIONS;
    ggml_backend_moe_cpu_prepared_requirements_v1 measured_q4_0 = {};
    measured_q4_0.struct_size = sizeof(measured_q4_0);
    measured_q4_0.abi_version = 1;
    uint64_t measured_service_bytes = 0;
    const ggml_backend_moe_cpu_fidelity_requirements_api_v1 * requirements_api = nullptr;
    if (fidelity) {
        const auto get_requirements = reinterpret_cast<ggml_backend_moe_cpu_fidelity_requirements_v1_t>(
            ggml_backend_reg_get_proc_address(cpu_reg, GGML_BACKEND_MOE_CPU_FIDELITY_REQUIREMENTS_V1_PROC_NAME));
        CHECK(get_requirements);
        requirements_api = get_requirements();
        CHECK(requirements_api && requirements_api->struct_size == sizeof(*requirements_api) && requirements_api->abi_version == 1);
        auto sizing_owner = source_owner;
        sizing_owner.retain = [](const ggml_backend_moe_source_owner_v1 *, uint64_t, ggml_backend_moe_source_lease_v1 *) -> int32_t {
            CHECK(false); return GGML_BACKEND_MOE_SOURCE_STATUS_V1_INVALID_ARGUMENT;
        };
        sizing_owner.release = [](ggml_backend_moe_source_lease_v1 *) -> int32_t {
            CHECK(false); return GGML_BACKEND_MOE_SOURCE_STATUS_V1_INVALID_ARGUMENT;
        };
        sizing_owner.validate_span = [](const ggml_backend_moe_source_owner_v1 *, uint64_t, const ggml_backend_moe_source_span_v1 *) -> int32_t {
            CHECK(false); return GGML_BACKEND_MOE_SOURCE_STATUS_V1_INVALID_ARGUMENT;
        };
        auto sizing_config = config;
        sizing_config.source_owner = &sizing_owner;
        const uint64_t allocations = ggml_allocation_count();
        CHECK(requirements_api->service(&sizing_config, &measured_service_bytes) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        CHECK(requirements_api->region(&sizing_config, &q4_0->query, &measured_q4_0) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        CHECK(ggml_allocation_count() == allocations);
        CHECK(measured_service_bytes > 0 && measured_q4_0.prepared_payload_bytes > 0);
        auto short_config = sizing_config;
        short_config.prepared_payload_limit = measured_service_bytes + measured_q4_0.prepared_payload_bytes - 1;
        auto sentinel = measured_q4_0;
        sentinel.prepared_payload_bytes = 73;
        CHECK(requirements_api->region(&short_config, &q4_0->query, &sentinel) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY);
        CHECK(sentinel.prepared_payload_bytes == 73);
        ++short_config.prepared_payload_limit;
        CHECK(requirements_api->region(&short_config, &q4_0->query, &sentinel) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        auto invalid = q4_0->query;
        invalid.n_threads = config.n_threads + 1;
        sentinel.prepared_payload_bytes = 73;
        CHECK(requirements_api->region(&sizing_config, &invalid, &sentinel) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
        CHECK(sentinel.prepared_payload_bytes == 73);
        invalid = q4_0->query;
        invalid.struct_size = sizeof(invalid) - 1;
        CHECK(requirements_api->region(&sizing_config, &invalid, &sentinel) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
        CHECK(sentinel.prepared_payload_bytes == 73);
        invalid.struct_size = sizeof(invalid) + 8;
        CHECK(requirements_api->region(&sizing_config, &invalid, &sentinel) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        CHECK(sentinel.prepared_payload_bytes == measured_q4_0.prepared_payload_bytes);
        short_config.prepared_payload_limit = measured_service_bytes - 1;
        uint64_t service_sentinel = 73;
        CHECK(requirements_api->service(&short_config, &service_sentinel) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY);
        CHECK(service_sentinel == 73);
    }
    CHECK(service_api->create(&config, &service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    if (fidelity) {
        ggml_backend_moe_cpu_service_state_v1 initial = {};
        initial.struct_size = sizeof(initial);
        CHECK(service_api->state(service, &initial) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        CHECK(initial.prepared_payload_bytes == measured_service_bytes);
    }

    ggml_backend_moe_cpu_prepared_requirements_v1 prepared_q4_0 = {};
    prepared_q4_0.struct_size = sizeof(prepared_q4_0);
    prepared_q4_0.abi_version = 1;
    ggml_backend_moe_cpu_prepared_region_v1_t region_q4_0 = 0;
    CHECK(service_api->prepare(service, &q4_0->query, &prepared_q4_0, &region_q4_0) ==
          GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    CHECK(region_q4_0 != 0 && prepared_q4_0.tensor_count ==
          q4_0->query.n_dynamic_inputs + (fidelity ? q4_0->query.n_live_outputs : q4_0->query.n_sources + q4_0->query.n_body_nodes));
    if (fidelity) {
        CHECK(prepared_q4_0.prepared_payload_bytes == measured_q4_0.prepared_payload_bytes &&
            prepared_q4_0.lane_metadata_bytes == measured_q4_0.lane_metadata_bytes &&
            prepared_q4_0.lane_allocation_bytes == measured_q4_0.lane_allocation_bytes &&
            prepared_q4_0.control_bytes == measured_q4_0.control_bytes && prepared_q4_0.flags == measured_q4_0.flags &&
            prepared_q4_0.execution.lane_execution_bytes == measured_q4_0.execution.lane_execution_bytes);
        ggml_backend_moe_cpu_service_state_v1 allocated = {};
        allocated.struct_size = sizeof(allocated);
        CHECK(service_api->state(service, &allocated) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        CHECK(allocated.prepared_payload_bytes == measured_service_bytes + measured_q4_0.prepared_payload_bytes);
    }
    CHECK(prepared_q4_0.external_source_count == q4_0->query.n_sources &&
          prepared_q4_0.lane_alignment >= CACHE_LINE_SIZE &&
          prepared_q4_0.lane_allocation_bytes ==
              prepared_q4_0.execution.lane_execution_bytes + prepared_q4_0.lane_alignment - 1 &&
          prepared_q4_0.prepared_payload_bytes > prepared_q4_0.execution.all_lane_execution_bytes &&
          (prepared_q4_0.flags & GGML_BACKEND_MOE_CPU_PREPARED_FLAG_V1_THREAD_STACK_BYTES_UNKNOWN) != 0 &&
          prepared_q4_0.lane_context_bytes == (fidelity ? 0 : ggml_context_overhead()) &&
          prepared_q4_0.thread_stack_bytes == UINT64_MAX);
    const uint64_t execute_allocations = ggml_allocation_count();
    cpu_region_check_execute(service_api, service, region_q4_0, *q4_0, q4_0_oracle,
                             expert_ids, source_rows, scatter, 2);
    for (uint64_t epoch = 3; epoch < 7; ++epoch) {
        cpu_region_check_execute(service_api, service, region_q4_0, *q4_0, q4_0_oracle,
                                 expert_ids, source_rows, scatter, epoch);
    }
    CHECK(ggml_allocation_count() == execute_allocations);
    CHECK(service_api->drain(service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);

    ggml_backend_moe_cpu_prepared_requirements_v1 merged_prepared = {};
    merged_prepared.struct_size = sizeof(merged_prepared);
    merged_prepared.abi_version = 1;
    ggml_backend_moe_cpu_prepared_region_v1_t merged_region = 0;
    CHECK(service_api->prepare(service, &merged_bias->query, &merged_prepared, &merged_region) ==
          GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    const uint64_t merged_allocations = ggml_allocation_count();
    cpu_region_check_execute(service_api, service, merged_region, *merged_bias, merged_bias_oracle,
                             expert_ids, source_rows, scatter, 7);
    CHECK(ggml_allocation_count() == merged_allocations);
    CHECK(service_api->destroy_region(service, &merged_region) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);

    std::vector<uint8_t> canceled_output(q4_0->query.scatter_capacity * q4_0->raw_down->nb[1], 0x5a);
    std::vector<uint8_t> canceled_before = canceled_output;
    std::array<ggml_backend_moe_cpu_dynamic_input_v1, 2> canceled_inputs = {{
        { q4_0_oracle.source_activation.data(), q4_0_oracle.source_activation.size() * sizeof(float), q4_0->activation->nb[2] },
        {},
    }};
    ggml_backend_moe_cpu_region_binding_v1 canceled_binding = {
        sizeof(canceled_binding), 2, 4, expert_ids.data(), source_rows.data(), scatter.data(),
    };
    ggml_backend_moe_cpu_output_v1 canceled_destination = {
        canceled_output.data(), canceled_output.size(), q4_0->raw_down->nb[1],
    };
    ggml_backend_moe_cpu_execute_v1 canceled_execution = {};
    canceled_execution.struct_size = sizeof(canceled_execution);
    canceled_execution.epoch = 7;
    canceled_execution.graph_uid = q4_0->query.graph_uid;
    canceled_execution.graph_generation = q4_0->query.graph_generation;
    canceled_execution.source_generation = q4_0->query.source_generation;
    canceled_execution.binding = &canceled_binding;
    canceled_execution.dynamic_inputs = canceled_inputs.data();
    canceled_execution.n_dynamic_inputs = canceled_inputs.size();
    canceled_execution.outputs = &canceled_destination;
    canceled_execution.n_outputs = 1;
    ggml_backend_moe_cpu_execute_result_v1 canceled_result = {};
    canceled_result.struct_size = sizeof(canceled_result);
    canceled_result.epoch = 99;
    ++canceled_execution.graph_generation;
    CHECK(service_api->execute(service, region_q4_0, &canceled_execution, &canceled_result) ==
          GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
    CHECK(canceled_output == canceled_before && canceled_result.epoch == 99);
    canceled_execution.graph_generation = q4_0->query.graph_generation;
    ++canceled_execution.source_generation;
    CHECK(service_api->execute(service, region_q4_0, &canceled_execution, &canceled_result) ==
          GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
    CHECK(canceled_output == canceled_before && canceled_result.epoch == 99);
    canceled_execution.source_generation = q4_0->query.source_generation;
    const uint64_t output_bytes = canceled_destination.bytes;
    canceled_destination.bytes = 1;
    CHECK(service_api->execute(service, region_q4_0, &canceled_execution, &canceled_result) ==
          GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
    CHECK(canceled_output == canceled_before && canceled_result.epoch == 99);
    canceled_destination.bytes = output_bytes;
    void * invalid_outputs[] = { q4_0->weights[0]->data, &canceled_result, &canceled_destination, &canceled_binding };
    for (void * data : invalid_outputs) {
        canceled_destination.data = data;
        CHECK(service_api->execute(service, region_q4_0, &canceled_execution, &canceled_result) ==
              GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
        CHECK(canceled_output == canceled_before && canceled_result.epoch == 99);
    }
    canceled_destination.data = canceled_output.data();
    canceled_destination.bytes = UINT64_MAX;
    CHECK(service_api->execute(service, region_q4_0, &canceled_execution, &canceled_result) ==
          GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
    canceled_destination.bytes = output_bytes;
    const uint64_t input_bytes = canceled_inputs[0].bytes;
    canceled_inputs[0].bytes = UINT64_MAX;
    CHECK(service_api->execute(service, region_q4_0, &canceled_execution, &canceled_result) ==
          GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
    canceled_inputs[0].bytes = input_bytes;
    CHECK(canceled_output == canceled_before && canceled_result.epoch == 99);
    CHECK(service_api->cancel(service, 7) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    CHECK(service_api->execute(service, region_q4_0, &canceled_execution, &canceled_result) ==
          GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CANCELED);
    CHECK(canceled_output == canceled_before && canceled_result.epoch == 99);

    std::array<ggml_tensor, 3> detached_weights;
    for (size_t i = 0; i < detached_weights.size(); ++i) {
        detached_weights[i] = *q4_k->weights[i];
        detached_weights[i].buffer = nullptr;
        detached_weights[i].data   = nullptr;
        detached_weights[i].extra  = nullptr;
        q4_k->sources[i].tensor    = &detached_weights[i];
    }
    q4_k->hidden->src[0]->src[0] = &detached_weights[0];
    q4_k->hidden->src[1]->src[0] = &detached_weights[1];
    q4_k->raw_down->src[0]       = &detached_weights[2];
    ggml_backend_moe_cpu_prepared_requirements_v1 prepared_q4_k = {};
    prepared_q4_k.struct_size = sizeof(prepared_q4_k);
    prepared_q4_k.abi_version = 1;
    ggml_backend_moe_cpu_prepared_region_v1_t region_q4_k = 0;
    CHECK(service_api->prepare(service, &q4_k->query, &prepared_q4_k, &region_q4_k) ==
          GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    CHECK(region_q4_k != 0 && prepared_q4_k.execution.n_live_outputs == 2);
    cpu_region_check_execute(service_api, service, region_q4_k, *q4_k, q4_k_oracle,
                             expert_ids, source_rows, scatter, 8);

    ggml_backend_moe_cpu_prepared_requirements_v1 rejected_prepare = {};
    rejected_prepare.struct_size = sizeof(rejected_prepare);
    rejected_prepare.abi_version = 1;
    ggml_backend_moe_cpu_prepared_region_v1_t rejected_region = 0;
    auto missing_source = q4_k->sources;
    missing_source[0].data = static_cast<const uint8_t *>(missing_source[0].data) + 1;
    auto missing_query = q4_k->query;
    missing_query.sources = missing_source.data();
    CHECK(service_api->prepare(service, &missing_query, &rejected_prepare, &rejected_region) ==
          GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_MISSING_SOURCE);
    CHECK(rejected_region == 0);

    const auto retired_region = region_q4_k;
    CHECK(service_api->destroy_region(service, &region_q4_k) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    CHECK(service_api->prepare(service, &q4_k->query, &prepared_q4_k, &region_q4_k) ==
          GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    CHECK(region_q4_k != retired_region);
    auto stale_region = retired_region;
    CHECK(service_api->destroy_region(service, &stale_region) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
    CHECK(stale_region == 0);
    CHECK(service_api->execute(service, retired_region, &canceled_execution, &canceled_result) ==
          GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
    cpu_region_check_execute(service_api, service, region_q4_k, *q4_k, q4_k_oracle,
                             expert_ids, source_rows, scatter, 9);

    CHECK(service_api->state != nullptr && service_api->set_test_hook != nullptr);
    std::vector<uint32_t> cancel_phases = {GGML_BACKEND_MOE_CPU_TEST_PHASE_V1_BEFORE_COMMIT,
        GGML_BACKEND_MOE_CPU_TEST_PHASE_V1_AFTER_COMMIT};
    if (fidelity) {
        cancel_phases.push_back(GGML_MOE_CPU_FIDELITY_AFTER_GU);
        cancel_phases.push_back(GGML_MOE_CPU_FIDELITY_AFTER_QUANT);
    }
    for (uint32_t phase : cancel_phases) {
        cpu_region_test_barrier barrier(phase);
        CHECK(service_api->set_test_hook(service, cpu_region_test_barrier::hook, &barrier) ==
              GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        canceled_output = canceled_before;
        canceled_destination.data = canceled_output.data();
        canceled_execution.epoch = 20 + phase;
        canceled_result = {};
        canceled_result.struct_size = sizeof(canceled_result);
        int32_t status = GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
        std::thread worker([&, worker_region = region_q4_0, worker_service = service] {
            status = service_api->execute(worker_service, worker_region, &canceled_execution, &canceled_result);
        });
        barrier.wait();
        CHECK(service_api->cancel(service, canceled_execution.epoch) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        barrier.release();
        worker.join();
        if (phase != GGML_BACKEND_MOE_CPU_TEST_PHASE_V1_AFTER_COMMIT) {
            CHECK(status == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CANCELED);
            CHECK(canceled_output == canceled_before && canceled_result.flags == 0);
        } else {
            CHECK(status == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
            CHECK(canceled_result.flags == GGML_BACKEND_MOE_CPU_EXECUTE_RESULT_FLAG_V1_PUBLISHED);
            auto expected = canceled_before;
            for (size_t route = 0; route < scatter.size(); ++route) {
                memcpy(expected.data() + scatter[route] * q4_0->raw_down->nb[1],
                       q4_0_oracle.outputs[0].data() + route * q4_0->raw_down->nb[1], q4_0->raw_down->nb[1]);
            }
            CHECK(canceled_output == expected);
        }
        CHECK(service_api->set_test_hook(service, nullptr, nullptr) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        if (fidelity) {
            fprintf(stderr, "test-moe-cache: CPU fidelity cancellation phase=%u status=%d retained_output=%u OK\n",
                phase, status, unsigned(canceled_output == canceled_before));
        }
    }

    constexpr size_t N_LIFECYCLE_WORKERS = 2;
    if (fidelity) {
        for (const bool alias : {false, true}) {
            cpu_region_test_barrier barrier(GGML_BACKEND_MOE_CPU_TEST_PHASE_V1_BEFORE_COMMIT);
            CHECK(service_api->set_test_hook(service, cpu_region_test_barrier::hook, &barrier) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
            std::vector<uint8_t> bytes(std::max(canceled_before.size(), q4_0_oracle.source_activation.size() * sizeof(float)), 0x5c);
            if (alias) { memcpy(bytes.data(), q4_0_oracle.source_activation.data(), q4_0_oracle.source_activation.size() * sizeof(float)); }
            const auto before = bytes;
            auto inputs = canceled_inputs;
            if (alias) { inputs[0].data = bytes.data(); }
            auto output = canceled_destination;
            output.data = bytes.data(); output.bytes = bytes.size();
            auto execution = canceled_execution;
            execution.flags = GGML_BACKEND_MOE_CPU_EXECUTE_FLAG_V1_PRIVATE_OUTPUTS;
            execution.epoch = 60 + unsigned(alias);
            execution.dynamic_inputs = inputs.data(); execution.outputs = &output;
            ggml_backend_moe_cpu_execute_result_v1 result = {};
            result.struct_size = sizeof(result);
            int32_t status = GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
            std::thread worker([&] { status = service_api->execute(service, region_q4_0, &execution, &result); });
            barrier.wait();
            CHECK(service_api->cancel(service, execution.epoch) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
            barrier.release(); worker.join();
            CHECK(status == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CANCELED && result.flags == 0);
            auto expected = before;
            if (!alias) {
                for (size_t route = 0; route < scatter.size(); ++route) {
                    memcpy(expected.data() + scatter[route] * output.route_stride,
                           q4_0_oracle.outputs[0].data() + route * q4_0->raw_down->nb[1], q4_0->raw_down->nb[1]);
                }
            }
            CHECK(bytes == expected);
            CHECK(service_api->set_test_hook(service, nullptr, nullptr) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
            fprintf(stderr, "test-moe-cache: CPU private output cancel alias=%u unpublished OK\n", unsigned(alias));
        }
    }
    cpu_region_test_barrier lifecycle_barrier(GGML_BACKEND_MOE_CPU_TEST_PHASE_V1_BEFORE_COMMIT);
    CHECK(service_api->set_test_hook(service, cpu_region_test_barrier::hook, &lifecycle_barrier) ==
          GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    std::array<int32_t, N_LIFECYCLE_WORKERS> lifecycle_status = {};
    std::vector<std::thread> lifecycle_workers;
    lifecycle_workers.reserve(N_LIFECYCLE_WORKERS);
    for (size_t worker = 0; worker < N_LIFECYCLE_WORKERS; ++worker) {
        lifecycle_workers.emplace_back([&, worker, worker_region = region_q4_0, worker_service = service]() {
            auto worker_execution = canceled_execution;
            auto worker_output = canceled_before;
            auto worker_destination = canceled_destination;
            worker_destination.data = worker_output.data();
            worker_execution.outputs = &worker_destination;
            worker_execution.epoch = 100 + worker;
            ggml_backend_moe_cpu_execute_result_v1 worker_result = {};
            worker_result.struct_size = sizeof(worker_result);
            lifecycle_status[worker] = service_api->execute(worker_service, worker_region, &worker_execution, &worker_result);
            CHECK(worker_output == canceled_before && worker_result.flags == 0);
        });
    }
    lifecycle_barrier.wait();
    ggml_backend_moe_cpu_service_state_v1 service_state = {};
    service_state.struct_size = sizeof(service_state);
    const auto lifecycle_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    do {
        CHECK(service_api->state(service, &service_state) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        CHECK(std::chrono::steady_clock::now() < lifecycle_deadline);
        std::this_thread::yield();
    } while (service_state.active_jobs < 2);
    CHECK(service_state.active_regions == 2 && service_state.closed == 0);
    CHECK(service_api->destroy_region(service, &region_q4_0) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_ACTIVE_JOBS);
    CHECK(region_q4_0 != 0);
    CHECK(service_api->set_test_hook(service, nullptr, nullptr) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_ACTIVE_JOBS);
    CHECK(service_api->execute(service, region_q4_0, &canceled_execution, &canceled_result) ==
          GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY);
    CHECK(service_api->cancel(service, UINT64_MAX) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    std::atomic<bool> close_done = false;
    std::thread closer([&, worker_service = service] {
        CHECK(service_api->close(worker_service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        close_done.store(true, std::memory_order_release);
    });
    do {
        CHECK(service_api->state(service, &service_state) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        CHECK(std::chrono::steady_clock::now() < lifecycle_deadline);
        std::this_thread::yield();
    } while (service_state.closed == 0);
    CHECK(service_state.active_jobs == 2 && !close_done.load(std::memory_order_acquire));
    lifecycle_barrier.release();
    for (auto & worker : lifecycle_workers) {
        worker.join();
    }
    closer.join();
    for (const auto & status : lifecycle_status) {
        CHECK(status == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CANCELED);
    }
    CHECK(close_done.load(std::memory_order_acquire));
    CHECK(service_api->set_test_hook(service, nullptr, nullptr) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    service_state = {};
    service_state.struct_size = sizeof(service_state);
    CHECK(service_api->state(service, &service_state) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    CHECK(service_state.active_jobs == 0 && service_state.active_regions == 2 && service_state.closed == 1);
    CHECK(service_api->drain(service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    CHECK(service_api->close(service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    CHECK(service_api->prepare(service, &q4_0->query, &rejected_prepare, &rejected_region) ==
          GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CLOSED);
    CHECK(service_api->destroy(&service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_ACTIVE_REGIONS);
    CHECK(service != nullptr);
    CHECK(service_api->destroy_region(service, &region_q4_k) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    CHECK(service_api->destroy_region(service, &region_q4_0) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    CHECK(service_api->destroy(&service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    CHECK(service == nullptr);

    config.max_regions            = 1;
    config.prepared_payload_limit = prepared_q4_0.prepared_payload_bytes - 1;
    CHECK(service_api->create(&config, &service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    rejected_prepare.prepared_payload_bytes = 73;
    CHECK(service_api->prepare(service, &q4_0->query, &rejected_prepare, &rejected_region) ==
          GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY);
    CHECK(rejected_region == 0 && rejected_prepare.prepared_payload_bytes == 73);
    CHECK(service_api->destroy(&service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    fprintf(stderr, "test-moe-cache: private CPU prepared-region execute oracle executor=%s OK\n", fidelity ? "fidelity" : "graph");
}

void test_moe_cpu_region_proc_api() {
    test_cpu_region_query_proc();
    if (getenv("GGML_TEST_MOE_CPU_FIDELITY")) { test_cpu_region_query_proc(true); }
}

void test_moe_cache_proc_api() {
    test_moe_cpu_region_proc_api();
    ggml_backend_reg_t reg = ggml_backend_cuda_reg();
    CHECK(reg != nullptr);
    CHECK(ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_CACHE_BUFFER_TYPE_PROC_NAME) != nullptr);
    CHECK(ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_CACHE_BOUNDED_BUFFER_TYPE_PROC_NAME) != nullptr);
    auto staging_size = reinterpret_cast<ggml_backend_moe_staging_size_v1_t>(
        ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_STAGING_SIZE_V1_PROC_NAME));
    CHECK(staging_size != nullptr);
    auto device_size = reinterpret_cast<ggml_backend_moe_device_size_v1_t>(
        ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_DEVICE_SIZE_V1_PROC_NAME));
    CHECK(device_size != nullptr);
    auto device_size_v2 = reinterpret_cast<ggml_backend_moe_device_size_v2_t>(
        ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_DEVICE_SIZE_V2_PROC_NAME));
    CHECK(device_size_v2 != nullptr);
    CHECK(ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_CACHE_FREE_BUFFER_TYPE_PROC_NAME) != nullptr);
    CHECK(ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_CACHE_CONFIGURE_SOURCES_PROC_NAME) != nullptr);
    CHECK(ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_CACHE_IS_BUFFER_TYPE_PROC_NAME) != nullptr);
    CHECK(ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_CACHE_BUFFER_FROM_HOST_PTR_PROC_NAME) != nullptr);
    auto writable_load_data = reinterpret_cast<ggml_backend_moe_cache_writable_load_data_t>(
        ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_CACHE_WRITABLE_LOAD_DATA_PROC_NAME));
    CHECK(writable_load_data != nullptr);
    auto readable_source = reinterpret_cast<ggml_backend_moe_cache_readable_source_t>(
        ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_CACHE_READABLE_SOURCE_PROC_NAME));
    CHECK(readable_source != nullptr);
    auto cache_buft = reinterpret_cast<ggml_backend_moe_cache_buffer_type_t>(
        ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_CACHE_BUFFER_TYPE_PROC_NAME));
    ggml_backend_buffer_ptr pinned(ggml_backend_buft_alloc_buffer(cache_buft(), 64));
    CHECK(pinned != nullptr);
    auto * pinned_data = static_cast<uint8_t *>(ggml_backend_buffer_get_base(pinned.get()));
    CHECK(writable_load_data(pinned.get(), pinned_data + 8, 16) == pinned_data + 8);
    CHECK(writable_load_data(pinned.get(), pinned_data + 60, 8) == nullptr);
    CHECK(readable_source(pinned.get(), pinned_data + 8, 16));
    CHECK(!readable_source(pinned.get(), pinned_data + 60, 8));
    ggml_backend_buffer_ptr cpu(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 64));
    CHECK(cpu != nullptr);
    CHECK(writable_load_data(cpu.get(), ggml_backend_buffer_get_base(cpu.get()), 64) == nullptr);
    CHECK(!readable_source(cpu.get(), ggml_backend_buffer_get_base(cpu.get()), 64));

    ggml_cuda_moe_cache_fail_full_pinning_for_test(true);
    ggml_backend_buffer_ptr pageable(ggml_backend_buft_alloc_buffer(cache_buft(), 64));
    ggml_cuda_moe_cache_fail_full_pinning_for_test(false);
    CHECK(pageable != nullptr);
    auto * pageable_data = ggml_backend_buffer_get_base(pageable.get());
    CHECK(writable_load_data(pageable.get(), pageable_data, 64) == pageable_data);
    CHECK(readable_source(pageable.get(), pageable_data, 64));

    auto bounded_buft = reinterpret_cast<ggml_backend_moe_cache_bounded_buffer_type_t>(
        ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_CACHE_BOUNDED_BUFFER_TYPE_PROC_NAME));
    auto free_buft = reinterpret_cast<ggml_backend_moe_cache_free_buffer_type_t>(
        ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_CACHE_FREE_BUFFER_TYPE_PROC_NAME));
    auto from_host_ptr = reinterpret_cast<ggml_backend_moe_cache_buffer_from_host_ptr_t>(
        ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_CACHE_BUFFER_FROM_HOST_PTR_PROC_NAME));
    auto * bounded = bounded_buft(1024);
    CHECK(bounded != nullptr);
    {
        ggml_backend_buffer_ptr writable(ggml_backend_buft_alloc_buffer(bounded, 64));
        CHECK(writable != nullptr);
        auto * writable_data = ggml_backend_buffer_get_base(writable.get());
        CHECK(writable_load_data(writable.get(), writable_data, 64) == writable_data);
        CHECK(readable_source(writable.get(), writable_data, 64));
        alignas(TENSOR_ALIGNMENT) uint8_t mapped_data[64] = {};
        ggml_backend_buffer_ptr mapped(from_host_ptr(bounded, mapped_data, sizeof(mapped_data)));
        CHECK(mapped != nullptr);
        CHECK(writable_load_data(mapped.get(), mapped_data, sizeof(mapped_data)) == nullptr);
        CHECK(readable_source(mapped.get(), mapped_data, sizeof(mapped_data)));
    }
    free_buft(bounded);
    CHECK(ggml_backend_reg_get_proc_address(reg, "ggml_backend_moe_cache_set_slots") == nullptr);
    CHECK(ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_CACHE_SET_DEBUG_PROC_NAME) != nullptr);
    CHECK(ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_EARLY_ROUTER_SET_ENABLED_PROC_NAME) != nullptr);
    CHECK(ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_EARLY_ROUTER_SET_MAX_ROWS_PROC_NAME) != nullptr);
    CHECK(ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_CACHE_LOG_AND_RESET_STATS_PROC_NAME) != nullptr);
    CHECK(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_set_decode_boundary_overlap") != nullptr);

    ggml_backend_moe_staging_query_v1 query = {};
    query.struct_size = sizeof(query);
    query.n_slots = 8;
    query.n_experts = 16;
    query.top_k = 4;
    query.n_banks = 2;
    query.staged_bank_mask = 0x3;
    query.family_mask = GGML_BACKEND_MOE_STAGING_FAMILY_V1_GROUPED |
        GGML_BACKEND_MOE_STAGING_FAMILY_V1_LEGACY |
        GGML_BACKEND_MOE_STAGING_FAMILY_V1_HOST_STAGED;
    query.flags = GGML_BACKEND_MOE_STAGING_FLAG_V1_PREDICTION_CONTROL;
    query.bank_expert_strides[0] = 1000;
    query.bank_expert_strides[1] = 2000;
    ggml_backend_moe_staging_size_v1 sizing = {};
    sizing.struct_size = sizeof(sizing);
    CHECK(staging_size(&query, &sizing));
    CHECK(sizing.grouped_min_bytes > 0);
    CHECK(sizing.legacy_min_bytes > 0);
    CHECK(sizing.host_staged_min_bytes == sizing.legacy_min_bytes);
    CHECK(sizing.optional_growth_max_bytes > 0);
    CHECK(sizing.prepack_tile_bytes == 48000);

    query.family_mask = GGML_BACKEND_MOE_STAGING_FAMILY_V1_GROUPED;
    sizing = {};
    sizing.struct_size = sizeof(sizing);
    CHECK(staging_size(&query, &sizing));
    CHECK(sizing.grouped_min_bytes > 0 && sizing.legacy_min_bytes == 0 && sizing.host_staged_min_bytes == 0);
    query.bank_expert_strides[0] = UINT64_MAX;
    CHECK(!staging_size(&query, &sizing));

    ggml_backend_moe_device_size_query_v1 device_query = {};
    device_query.struct_size = sizeof(device_query);
    device_query.n_slots = 8;
    device_query.n_experts = 16;
    device_query.n_banks = 1;
    device_query.n_slot_auxiliaries = 1;
    device_query.flags = GGML_BACKEND_MOE_DEVICE_SIZE_FLAG_V1_DEBUG;
    device_query.slot_auxiliary_values = 2;
    device_query.original_shadow_bytes = 100;
    device_query.prefill_copy_bytes = 200;
    device_query.bank_expert_strides[0] = 1000;
    device_query.bank_ne0[0] = 32;
    device_query.bank_types[0] = GGML_TYPE_F32;
    ggml_backend_moe_device_size_v1 device_sizing = {};
    device_sizing.struct_size = sizeof(device_sizing);
    CHECK(device_size(&device_query, &device_sizing));
    CHECK(device_sizing.group_fixed_bytes >= 300);
    CHECK(device_sizing.group_per_slot_bytes > 1000);
    CHECK(device_sizing.context_fixed_bytes > 0);

    device_query = {};
    device_query.struct_size = sizeof(device_query);
    device_query.flags = GGML_BACKEND_MOE_DEVICE_SIZE_FLAG_V1_DEBUG;
    device_query.early_width = 4096;
    device_query.early_experts = 256;
    device_query.early_top_k = 8;
    device_query.early_hc_rank = 512;
    device_sizing = {};
    device_sizing.struct_size = sizeof(device_sizing);
    CHECK(device_size(&device_query, &device_sizing));
    CHECK(device_sizing.group_fixed_bytes == 0 && device_sizing.group_per_slot_bytes == 0);
    CHECK(device_sizing.context_fixed_bytes > 4096 * sizeof(float));
    ggml_backend_moe_device_size_query_v2 device_query_v2 = {};
    device_query_v2.struct_size                           = sizeof(device_query_v2);
    device_query_v2.flags                                 = GGML_BACKEND_MOE_DEVICE_SIZE_FLAG_V1_DEBUG;
    device_query_v2.early_width                           = device_query.early_width;
    device_query_v2.early_experts                         = device_query.early_experts;
    device_query_v2.early_top_k                           = device_query.early_top_k;
    device_query_v2.early_hc_rank                         = device_query.early_hc_rank;
    device_query_v2.early_route_capacity                  = 48;
    device_query_v2.early_row_capacity                    = 6;
    device_query_v2.early_groups                          = 12;
    device_query_v2.early_expert_bytes                    = 3000;
    ggml_backend_moe_device_size_v2 device_sizing_v2      = {};
    device_sizing_v2.struct_size                          = sizeof(device_sizing_v2);
    CHECK(device_size_v2(&device_query_v2, &device_sizing_v2));
    const uint64_t early_rows = device_query_v2.early_row_capacity;
    const uint64_t extra_workspace =
        uint64_t(device_query_v2.early_route_capacity - device_query_v2.early_top_k) * sizeof(int32_t) +
        uint64_t(device_query_v2.early_experts) * sizeof(int32_t) +
        (early_rows - 1) * (uint64_t(device_query_v2.early_width) * 3 * sizeof(float) +
                            uint64_t(device_query_v2.early_experts) * sizeof(float) +
                            uint64_t(device_query_v2.early_hc_rank) * sizeof(float));
    CHECK(device_sizing_v2.context_fixed_bytes ==
          device_sizing.context_fixed_bytes +
              2 * device_query_v2.early_route_capacity * device_query_v2.early_expert_bytes +
              2 * sizeof(uint32_t) + uint64_t(device_query_v2.early_experts) * sizeof(int32_t) +
              uint64_t(device_query_v2.early_width) * early_rows * sizeof(float) +
              extra_workspace);
    CHECK(device_sizing_v2.host_fixed_bytes > 0);
    device_query_v2.early_route_capacity = 4;
    device_query_v2.early_row_capacity = 1;
    CHECK(device_size_v2(&device_query_v2, &device_sizing_v2));
    device_query_v2.early_route_capacity = 65;
    device_query_v2.early_row_capacity = 6;
    CHECK(device_size_v2(&device_query_v2, &device_sizing_v2));
    device_query.early_top_k = 0;
    CHECK(!device_size(&device_query, &device_sizing));
    fprintf(stderr, "test-moe-cache: dynamic backend procedure API OK\n");
}
