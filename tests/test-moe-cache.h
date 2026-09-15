#pragma once

#include "ggml-cuda/moe-cache.cuh"
#include "ggml-cuda/mmid.cuh"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include "ggml-alloc.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
        std::exit(1); \
    } \
} while (0)

#define CUDA_OK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "FAIL %s:%d  cuda: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        std::exit(1); \
    } \
} while (0)

bool grouped_frequency_enabled();

void test_speculative_grouped_intent_splits();

int  test_grouped_multigpu();
void test_grouped_layer_placement();

void test_speculative_required_grouped_backend_capability(int device);

struct ggml_cuda_moe_grouped_context_test_access {
    static bool early_graph(ggml_cuda_moe_grouped_context & context) {
        return context.early_graph_for_test();
    }
    static uint64_t early_bytes(ggml_cuda_moe_grouped_context & context, uint64_t * calls = nullptr) {
        return context.early_bytes_for_test(calls);
    }
    static size_t early_programs(ggml_cuda_moe_grouped_context & context) {
        return context.early_program_count_for_test();
    }
    static bool early_select(ggml_cuda_moe_grouped_context & context) {
        return context.early_select_for_test();
    }
    static bool early_hc(ggml_cuda_moe_grouped_context & context) {
        return context.early_hc_for_test();
    }
    static bool prepack(ggml_cuda_moe_grouped_context & context) {
        return context.prepack_for_test();
    }
    static uint64_t host_copy_jobs(const ggml_cuda_moe_grouped_context & context, const ggml_cuda_moe_candidate_group_key & key) {
        return context.host_copy_jobs_for_test(key);
    }

    static bool admission_closed(const ggml_cuda_moe_grouped_context & context) {
        return context.admission_closed_for_test();
    }

    static bool attach_prepack(ggml_cuda_moe_grouped_context & context, const ggml_cuda_moe_candidate_group_key & key, cudaStream_t stream) {
        return context.attach_prepack_for_test(key, stream);
    }

    static bool set_clock_bound(
            ggml_cuda_moe_grouped_context & context,
            const ggml_cuda_moe_grouped_acquisition & acquisition,
            uint64_t clock_bound) {
        return context.set_clock_bound_for_test(acquisition, clock_bound);
    }

    static bool has_device_resource(
            const ggml_cuda_moe_grouped_context & context,
            const ggml_cuda_moe_candidate_group_key & key) {
        return context.has_device_resource_for_test(key);
    }

    static bool get_clock_bound(
            const ggml_cuda_moe_grouped_context & context,
            const ggml_cuda_moe_candidate_group_key & key,
            uint64_t * clock_bound) {
        return context.get_clock_bound_for_test(key, clock_bound);
    }

    static int32_t device_slot_for_expert(
            const ggml_cuda_moe_grouped_context & context,
            const ggml_cuda_moe_candidate_group_key & key,
            uint32_t expert) {
        int32_t slot = -2;
        CHECK(context.device_slot_for_expert_for_test(key, expert, &slot));
        return slot;
    }

    static void * device_bank_data(
            const ggml_cuda_moe_grouped_context & context,
            const ggml_cuda_moe_candidate_group_key & key,
            const ggml_tensor * tensor) {
        return context.device_bank_data_for_test(key, tensor);
    }

    static const float * device_auxiliary_data(
            const ggml_cuda_moe_grouped_context & context,
            const ggml_cuda_moe_candidate_group_key & key,
            const ggml_tensor * tensor) {
        return context.device_auxiliary_data_for_test(key, tensor);
    }

    static const float * device_prefill_auxiliary_data(
            const ggml_cuda_moe_grouped_context & context,
            const ggml_cuda_moe_candidate_group_key & key,
            const ggml_tensor * tensor,
            size_t * byte_extent = nullptr,
            uint64_t * resource_generation = nullptr) {
        return context.device_prefill_auxiliary_data_for_test(key, tensor, byte_extent, resource_generation);
    }

    static bool set_prefill_resident_budget(
            ggml_cuda_moe_grouped_context & context,
            size_t byte_budget) {
        return context.set_prefill_resident_budget_for_test(byte_budget);
    }

    static bool prefill_auxiliary_ordering(
            const ggml_cuda_moe_grouped_context & context,
            const ggml_cuda_moe_candidate_group_key & key,
            uint64_t * cross_stream_waits,
            uint64_t * pending_declines) {
        return context.prefill_auxiliary_ordering_for_test(key, cross_stream_waits, pending_declines);
    }

    static bool device_resource_complete(
            const ggml_cuda_moe_grouped_context & context,
            const ggml_cuda_moe_candidate_group_key & key) {
        return context.device_resource_complete_for_test(key);
    }

    static bool graph_clock_active(
            const ggml_cuda_moe_grouped_context & context,
            const ggml_cuda_moe_candidate_group_key & key) {
        return context.graph_clock_active_for_test(key);
    }

    static size_t legacy_backing_count(
            const ggml_cuda_moe_grouped_context & context,
            const ggml_cuda_moe_candidate_group_key & key) {
        return context.legacy_backing_count_for_test(key);
    }

    static void fail_borrowed_cache_init_after_probe(ggml_cuda_moe_grouped_context & context) {
        context.fail_borrowed_cache_init_after_probe_for_test();
    }

    static void poison_split_staging(ggml_cuda_moe_grouped_context & context, uint32_t calls) {
        context.poison_split_staging_for_test(calls);
    }

    static void fail_host_staged_evaluator(ggml_cuda_moe_grouped_context & context) {
        context.fail_host_staged_evaluator_for_test();
    }

    static uint32_t split_staging_poison_calls(const ggml_cuda_moe_grouped_context & context) {
        return context.split_staging_poison_calls_for_test();
    }

    static uint64_t legacy_op_count(const ggml_cuda_moe_grouped_context & context, bool is_decode) {
        return context.legacy_op_count_for_test(is_decode);
    }

    static ggml_cuda_moe_legacy_debug_telemetry legacy_debug_telemetry(
            const ggml_cuda_moe_grouped_context & context,
            bool is_decode) {
        return context.legacy_debug_telemetry_for_test(is_decode);
    }

    static ggml_cuda_moe_grouped_debug_telemetry take_grouped_debug_telemetry(
            ggml_cuda_moe_grouped_context & context) {
        return context.take_grouped_debug_telemetry_for_test();
    }

    static size_t prefill_add_id_witness_count(const ggml_cuda_moe_graph_plan & plan) {
        return plan.prefill_add_id_witnesses_.size();
    }

    static size_t graph_reader_witness_size() {
        return sizeof(ggml_cuda_moe_graph_plan::reader_witness);
    }

    static size_t graph_group_record_size() {
        return sizeof(ggml_cuda_moe_graph_plan::group_record);
    }

    static size_t graph_group_observation_size() {
        return sizeof(ggml_cuda_moe_graph_plan::group_observation);
    }

    static ggml_cuda_moe_graph_outcome graph_outcome(const ggml_cuda_moe_graph_plan & plan) {
        return plan.outcome_;
    }

    static bool graph_group_has_decode_discovery(const ggml_cuda_moe_graph_plan & plan, uint32_t group_index) {
        CHECK(group_index < plan.groups_.size());
        const auto & group = plan.groups_[group_index];
        if (group.ids.tensor != nullptr || group.ids_root.tensor != nullptr || group.ids_source.tensor != nullptr || group.n_readers != 0 ||
                group.witness_reusable != 0) {
            return true;
        }
        for (uint32_t index = 0; index < GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS; ++index) {
            if (group.bank_uses[index].tensor != nullptr) {
                return true;
            }
        }
        for (uint32_t index = 0; index < 4; ++index) {
            if (group.nodes[index] != nullptr || group.capabilities[index].tensor != nullptr) {
                return true;
            }
        }
        return false;
    }

    static bool graph_has_complete_mmid_inventory(const ggml_cuda_moe_graph_plan & plan) {
        return plan.inventory_complete_ && plan.mmid_inventory_.size() == plan.coverage_diagnostics_.cached_mmid;
    }

    static bool graph_group_has_capability_reason(const ggml_cuda_moe_graph_plan & plan, uint32_t group_index) {
        CHECK(group_index < plan.groups_.size());
        return plan.groups_[group_index].reason == ggml_cuda_moe_graph_plan::GROUP_REASON_CAPABILITY;
    }

    static bool graph_group_has_consumer_equivalence_reason(const ggml_cuda_moe_graph_plan & plan, uint32_t group_index) {
        CHECK(group_index < plan.groups_.size());
        return plan.groups_[group_index].reason == ggml_cuda_moe_graph_plan::GROUP_REASON_CONSUMER_EQUIVALENCE;
    }

    static bool graph_group_has_descriptor_reason(const ggml_cuda_moe_graph_plan & plan, uint32_t group_index) {
        CHECK(group_index < plan.groups_.size());
        return plan.groups_[group_index].reason == ggml_cuda_moe_graph_plan::GROUP_REASON_DESCRIPTOR;
    }

    static bool graph_group_has_geometry_reason(const ggml_cuda_moe_graph_plan & plan, uint32_t group_index) {
        CHECK(group_index < plan.groups_.size());
        return plan.groups_[group_index].reason == ggml_cuda_moe_graph_plan::GROUP_REASON_GEOMETRY;
    }

    static bool graph_group_has_auxiliary_reason(const ggml_cuda_moe_graph_plan & plan, uint32_t group_index) {
        CHECK(group_index < plan.groups_.size());
        return plan.groups_[group_index].reason == ggml_cuda_moe_graph_plan::GROUP_REASON_AUXILIARY;
    }

    static bool graph_group_has_eligible_reason(const ggml_cuda_moe_graph_plan & plan, uint32_t group_index) {
        CHECK(group_index < plan.groups_.size());
        return plan.groups_[group_index].reason == ggml_cuda_moe_graph_plan::GROUP_REASON_ELIGIBLE;
    }

    static bool graph_group_has_materialization_reason(const ggml_cuda_moe_graph_plan & plan, uint32_t group_index) {
        CHECK(group_index < plan.groups_.size());
        return plan.groups_[group_index].reason == ggml_cuda_moe_graph_plan::GROUP_REASON_MATERIALIZATION;
    }

    static bool graph_group_has_prefill_reason(const ggml_cuda_moe_graph_plan & plan, uint32_t group_index) {
        CHECK(group_index < plan.groups_.size());
        return plan.groups_[group_index].reason == ggml_cuda_moe_graph_plan::GROUP_REASON_PREFILL;
    }

    static bool graph_group_has_execution_reason(const ggml_cuda_moe_graph_plan & plan, uint32_t group_index) {
        CHECK(group_index < plan.groups_.size());
        return plan.groups_[group_index].reason == ggml_cuda_moe_graph_plan::GROUP_REASON_EXECUTION;
    }

    static uint64_t graph_execution_semantic_key(const ggml_cuda_moe_graph_plan & plan) {
        return plan.execution_semantic_key_;
    }

    static bool graph_group_has_missing_role_reason(const ggml_cuda_moe_graph_plan & plan, uint32_t group_index) {
        CHECK(group_index < plan.groups_.size());
        return plan.groups_[group_index].reason == ggml_cuda_moe_graph_plan::GROUP_REASON_MISSING_ROLE;
    }

    static uint32_t graph_group_auxiliary_node_count(const ggml_cuda_moe_graph_plan & plan, uint32_t group_index) {
        CHECK(group_index < plan.groups_.size());
        uint32_t result = 0;
        for (uint32_t reader_index = 0; reader_index < plan.groups_[group_index].n_readers; ++reader_index) {
            result += plan.groups_[group_index].readers[reader_index].n_auxiliary_nodes;
        }
        return result;
    }

    static bool graph_group_has_bank_use(
            const ggml_cuda_moe_graph_plan & plan,
            uint32_t group_index,
            uint32_t bank_index) {
        CHECK(group_index < plan.groups_.size() && bank_index < GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS);
        return plan.groups_[group_index].bank_uses[bank_index].tensor != nullptr;
    }

    static ggml_cuda_moe_graph_capability_witness graph_bank_capability(
            const ggml_cuda_moe_graph_plan & plan,
            uint32_t group_index,
            uint32_t bank_index) {
        CHECK(group_index < plan.groups_.size() && bank_index < plan.groups_[group_index].n_banks);
        return plan.groups_[group_index].capabilities[bank_index];
    }

};

cudaStream_t candidate_test_graph_stream(void * data, const ggml_tensor *);

struct host_barrier {
    std::atomic<bool> entered{false};
    std::atomic<bool> released{false};
};

void CUDART_CB wait_on_host_barrier(void * data);

bool candidate_test_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t);

struct candidate_test_fixture {
    static constexpr size_t BUFFER_SIZE = 4 * 1024 * 1024;

    bool supports_buft = true;
    ggml_backend_device owner = {};
    ggml_context * ctx = nullptr;
    void * storage = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_backend_buffer_t cached_buffer = nullptr;
    size_t next_offset = 0;

    candidate_test_fixture() {
        owner.context = &supports_buft;
        owner.iface.supports_buft = candidate_test_supports_buft;
        CUDA_OK(cudaMallocHost(&storage, BUFFER_SIZE));
        memset(storage, 0, BUFFER_SIZE);
        buffer = ggml_backend_cpu_buffer_from_ptr(storage, BUFFER_SIZE);
        CHECK(buffer != nullptr);
        ggml_init_params params = {};
        params.mem_size = 65536 * ggml_tensor_overhead();
        params.no_alloc = true;
        ctx = ggml_init(params);
        CHECK(ctx != nullptr);
    }

    ~candidate_test_fixture() {
        ggml_free(ctx);
        if (cached_buffer != nullptr) {
            ggml_backend_buffer_free(cached_buffer);
        }
        ggml_backend_buffer_free(buffer);
        CUDA_OK(cudaFreeHost(storage));
    }

    void materialize(ggml_tensor * tensor) {
        if (tensor->view_src != nullptr) {
            CHECK(tensor->view_src->buffer != nullptr && tensor->data != nullptr);
            tensor->buffer = tensor->view_src->buffer;
            return;
        }
        const size_t alignment = ggml_backend_buffer_get_alignment(buffer);
        next_offset = (next_offset + alignment - 1) / alignment * alignment;
        CHECK(next_offset + ggml_nbytes(tensor) <= BUFFER_SIZE);
        tensor->buffer = buffer;
        tensor->data = static_cast<uint8_t *>(storage) + next_offset;
        next_offset += ggml_nbytes(tensor);
    }

    ggml_tensor * tensor(enum ggml_type type, int n_dims, const int64_t * ne) {
        ggml_tensor * result = ggml_new_tensor(ctx, type, n_dims, ne);
        materialize(result);
        return result;
    }

    ggml_tensor * cached_tensor(enum ggml_type type, int n_dims, const int64_t * ne) {
        if (cached_buffer == nullptr) {
            cached_buffer = ggml_backend_cuda_moe_cached_buffer_from_host_ptr(ggml_backend_cuda_moe_cached_buffer_type(), storage, BUFFER_SIZE);
            CHECK(cached_buffer != nullptr);
        }
        ggml_tensor * result = tensor(type, n_dims, ne);
        result->buffer = cached_buffer;
        return result;
    }
};

struct candidate_route {
    ggml_tensor * source = nullptr;
    ggml_tensor * root = nullptr;
    ggml_tensor * ids = nullptr;
};

candidate_route candidate_top_k_route(
        candidate_test_fixture & fixture,
        int64_t n_experts,
        int64_t n_routes,
        int64_t n_tokens = 1,
        size_t view_offs = 0);

ggml_tensor * candidate_mmid(candidate_test_fixture & fixture, ggml_tensor * weight, ggml_tensor * ids);

void candidate_set_route_tokens(
        const candidate_route & route,
        std::initializer_list<ggml_tensor *> readers,
        int64_t n_tokens);

void candidate_rebuild_graph_uses(ggml_cgraph * graph);

void candidate_insert_graph_node(ggml_cgraph * graph, int32_t node_index, ggml_tensor * node);

int32_t candidate_graph_use_count(const ggml_cgraph * graph, const ggml_tensor * tensor);

void candidate_stamp_execution(
        ggml_cgraph * graph,
        uint32_t domain,
        uint32_t row_semantics,
        uint32_t n_rows,
        uint32_t n_sequences,
        uint64_t source_graph_uid = 0,
        uint32_t flags = GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_NONE);

void candidate_stamp_single_row_execution(ggml_cgraph * graph);

ggml_cgraph * candidate_graph(candidate_test_fixture & fixture, std::initializer_list<ggml_tensor *> nodes);

struct candidate_graph_coverage {
    const void * nodes = nullptr;
    uint64_t epoch = 0;
    uint64_t mmid_fingerprint = 0;
    uint32_t mmid_count = 0;
};

candidate_graph_coverage candidate_certify_graph(
        ggml_cuda_moe_grouped_context & registry,
        ggml_cgraph * graph);

ggml_backend_moe_candidate_snapshot_v1 candidate_snapshot(
        uint32_t n_slots,
        const ggml_backend_moe_candidate_group_v1 * groups,
        uint32_t n_groups);

ggml_backend_moe_candidate_snapshot_v2 candidate_snapshot_v2(
        uint32_t n_slots,
        const ggml_backend_moe_candidate_group_v2 * groups,
        uint32_t n_groups,
        const ggml_backend_moe_candidate_tensor_v2 * tensors,
        uint32_t n_tensors);

void test_candidate_graph_coverage_ledger();

void test_candidate_graph_inventory_reuse();

void test_mmid_capabilities();

void test_scheduler_execution_certificate();

void test_graph_execution_certificate_policy();

void test_candidate_generic_physical_truth();

void test_candidate_producer();

void test_candidate_registry(bool benchmark);

void test_legacy_owner_leases();

void test_grouped_context_resources();

void test_grouped_graph_preflight(bool benchmark);

void test_grouped_graph_mixed_phase();

std::vector<uint8_t> cached_fusion_test_data(const ggml_tensor * tensor, size_t salt);

#ifdef __linux__
ggml_backend_buffer_type_t file_mmap_cached_buffer_type();
#endif

struct active_grouped_dispatch_graph {
    ggml_context_ptr lookup_weights;
    ggml_backend_buffer_ptr lookup_buffer;
    ggml_tensor * lookup_table = nullptr;
    ggml_tensor * token_ids = nullptr;
    ggml_context_ptr weights;
    ggml_context_ptr nodes;
    ggml_context_ptr router_weights;
    ggml_backend_buffer_ptr weight_buffer;
    ggml_backend_buffer_ptr node_buffer;
    ggml_backend_buffer_ptr router_buffer;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * down = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * logits = nullptr;
    ggml_tensor * concurrent_root = nullptr;
    ggml_tensor * router_rows = nullptr;
    ggml_tensor * ids = nullptr;
    ggml_tensor * gate_output = nullptr;
    ggml_tensor * up_output = nullptr;
    ggml_tensor * down_output = nullptr;
    ggml_tensor * gate_scale = nullptr;
    ggml_tensor * up_scale = nullptr;
    ggml_tensor * down_scale = nullptr;
    ggml_tensor * gate_scale_reshape = nullptr;
    ggml_tensor * gate_scale_repeat = nullptr;
    ggml_tensor * gate_scale_rows = nullptr;
    ggml_tensor * up_scale_reshape = nullptr;
    ggml_tensor * up_scale_repeat = nullptr;
    ggml_tensor * up_scale_rows = nullptr;
    ggml_tensor * down_scale_reshape = nullptr;
    ggml_tensor * down_scale_repeat = nullptr;
    ggml_tensor * down_scale_rows = nullptr;
    ggml_tensor * output = nullptr;
    std::vector<ggml_tensor *> banks;
    std::vector<ggml_tensor *> readers;
    std::vector<uint32_t> roles;
    std::vector<ggml_tensor *> biases;
    std::vector<uint32_t> bias_roles;
    uint32_t n_experts = 0;
    uint32_t n_used = 0;
    uint32_t n_rows = 0;
};

active_grouped_dispatch_graph build_active_grouped_dispatch_graph_types(
        ggml_backend_t backend,
        ggml_backend_buffer_type_t weight_buft,
        const std::array<ggml_type, 3> & types,
        uint32_t layout,
        bool original_direct_down_scale = false,
        bool original_direct_nvfp4_scales = false,
        uint32_t n_rows = 1,
        uint32_t n_experts = 8,
        uint32_t n_used = 2,
        uint32_t n_dim = 256,
        const active_grouped_dispatch_graph * shared_banks = nullptr,
        bool concurrent_stream_fixture = false,
        bool original_direct_biases = false,
        uint32_t n_ff = 0,
        bool mapped_host_biases = false,
        bool lookup_route = false,
        int router_variant = -1);

active_grouped_dispatch_graph build_active_grouped_dispatch_graph(
        ggml_backend_t backend,
        ggml_backend_buffer_type_t weight_buft,
        ggml_type type,
        uint32_t layout,
        bool original_direct_down_scale = false,
        uint32_t n_rows = 1,
        uint32_t n_experts = 8,
        uint32_t n_used = 2,
        uint32_t n_dim = 256,
        const active_grouped_dispatch_graph * shared_banks = nullptr,
        bool concurrent_stream_fixture = false);

int32_t active_grouped_route(
        const active_grouped_dispatch_graph & graph,
        uint32_t route_variant,
        uint32_t row,
        uint32_t route);

void set_active_grouped_dispatch_routes(
        const std::vector<active_grouped_dispatch_graph *> & graphs,
        const std::vector<int32_t> & routes);

void set_active_grouped_dispatch_logits(
        const std::vector<active_grouped_dispatch_graph *> & graphs,
        uint32_t route_variant);

void initialize_active_grouped_dispatch_graphs(
        const std::vector<active_grouped_dispatch_graph *> & graphs,
        const std::vector<std::vector<uint8_t>> * bank_data = nullptr);

void initialize_active_grouped_dispatch_graph(active_grouped_dispatch_graph & graph, size_t salt);

void register_active_grouped_dispatch(
        ggml_backend_t backend,
        const active_grouped_dispatch_graph & graph,
        uint32_t layout,
        uint32_t n_slots,
        bool auxiliary_first = false);

int32_t replace_active_grouped_nvfp4_dispatch_v2(
        ggml_backend_t backend,
        const active_grouped_dispatch_graph & graph,
        uint32_t n_slots,
        int32_t omitted_scale = -1);

std::vector<float> active_grouped_intermediate_sentinel(active_grouped_dispatch_graph & graph);

void check_active_grouped_intermediates(
        const active_grouped_dispatch_graph & graph,
        const std::vector<float> & sentinel,
        bool skipped);

uint64_t active_grouped_legacy_op_count(ggml_backend_t backend, bool is_decode = true);

void check_active_grouped_debug_telemetry(
        ggml_backend_t backend,
        const active_grouped_dispatch_graph & graph,
        uint64_t expected_loaded_experts = 0,
        uint64_t expected_calls = 5);

void check_active_grouped_legacy_caches(
        ggml_backend_t backend,
        const active_grouped_dispatch_graph & graph,
        bool registered_source,
        bool expect_slot_activity = true);

void check_active_grouped_contract(
        ggml_backend_t backend,
        active_grouped_dispatch_graph & graph,
        uint32_t n_slots,
        bool auxiliary_first = false,
        bool expect_compact_mmvq = false);

void test_grouped_graph_replay_lifecycle(int device, size_t host_budget = 0, size_t expected_host_nodes = 0, uint32_t n_dim = 256);

std::vector<float> run_active_grouped_dispatch(
        ggml_backend_t backend,
        active_grouped_dispatch_graph & graph,
        uint64_t expected_clock,
        bool f3_skipped = false,
        bool down_skipped = false);

void check_active_grouped_routes(
        const active_grouped_dispatch_graph & graph,
        uint32_t n_rows,
        const std::vector<int32_t> & routes);

void check_active_grouped_routes(
        const active_grouped_dispatch_graph & graph,
        uint32_t n_rows,
        uint32_t route_variant);

void check_active_grouped_exact_output(
        const std::vector<float> & expected,
        const std::vector<float> & actual);

void test_early_grouped_graphs();

struct cached_mmid_path_test_graph {
    ggml_context_ptr weights;
    ggml_context_ptr nodes;
    ggml_backend_buffer_ptr weight_buffer;
    ggml_backend_buffer_ptr node_buffer;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * ids = nullptr;
    ggml_tensor * output = nullptr;
    std::vector<ggml_tensor *> leaves;
};

void test_active_grouped_lookup_routes(int device);

void test_active_grouped_legacy_phase_telemetry(int device);

void test_active_grouped_multirow_graph_modes(int device);

std::vector<float> active_grouped_tensor_sentinel(ggml_tensor * tensor);

std::vector<float> active_grouped_tensor_values(const ggml_tensor * tensor);

void test_active_grouped_nvfp4_scales();

void test_prefill_resident_biases();

void test_active_grouped_dispatch();
void test_moe_tensor_split_rejection();

void test_cached_mmid_fusion_decline();

cached_mmid_path_test_graph build_cached_mmid_path_test_graph(
        ggml_backend_t backend,
        ggml_backend_buffer_type_t weight_buft,
        ggml_type weight_type,
        int64_t n_out,
        int64_t n_used,
        int64_t n_tokens,
        int64_t n_experts = 8);

cached_mmid_path_test_graph build_cached_mmid_path_test_graph(
        ggml_backend_t backend,
        ggml_tensor * gate_up,
        ggml_tensor * down,
        int64_t n_used,
        int64_t n_tokens);

void initialize_cached_mmid_path_test_graphs(
        cached_mmid_path_test_graph & cuda_graph,
        cached_mmid_path_test_graph & reference_graph);

std::vector<float> run_cached_mmid_path_test(
        ggml_backend_t cached_backend,
        ggml_backend_t reference_backend,
        cached_mmid_path_test_graph & cuda_graph,
        cached_mmid_path_test_graph & reference_graph,
        const std::vector<int32_t> & ids);

void test_cached_mmid_prefill_and_overflow();

void test_cached_mmid_routed_separate_chain();

ggml_cuda_mmid_capability native_mmid_capability(
        int device,
        const ggml_tensor * weight,
        int64_t n_rows,
        ggml_cuda_mmid_mapping mapping);

void test_mmid_direct_source_view(int device);

void test_gemma_q4_cached_cuda_parity(int device);

void test_pageable_staging_pipeline(int device);

void test_owner_legacy_cache(int device);

void test_grouped_decode(int device);

void test_grouped_decode_benchmark(int device);

void test_moe_route_publication_lifetime();

void test_moe_cache_proc_api();

void test_strided_copy_graph_update(int device, bool enabled);

void test_pageable_cache_fallback();

void test_grouped_staging_benchmark();

void test_cache_slots_and_staging(int dev);
