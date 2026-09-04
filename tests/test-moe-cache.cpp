// test-moe-cache: standalone correctness test for the MoE expert cache.
//
// This test does NOT exercise the larger llama.cpp graph; it pokes the cache
// API directly to verify:
//   1. acquire() returns a valid slot in [0, n_slots).
//   2. After the workload, every (expert, slot) mapping in the cache points
//      to a slot whose contents bit-match that expert's source data.
//   3. hits + misses == total acquires.
//   4. No expert is ever resident in two slots simultaneously.
//   5. Hit rate for a Zipf-ish access pattern is non-trivial (>40%).
//   6. Backend-context owners distinguish identical tensor sources.
//   7. Cache-assisted staging orders coalesced D2D and H2D copies after prior compute.
//
// The workload is a synthetic stand-in for MoE routing: most tokens land on a
// hot subset of experts. Real Gemma 4 / Mixtral / Qwen3 routing has stronger
// locality than this, so a passing test here is a lower bound.

#include "../ggml/src/ggml-cuda/moe-cache.cuh"
#include "../ggml/src/ggml-cuda/mmid.cuh"
#include "../ggml/src/ggml-backend-impl.h"
#include "../ggml/src/ggml-impl.h"
#include "../src/llama-context.h"
#include "../src/llama-model.h"

#include "ggml-alloc.h"
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

#ifdef __linux__
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

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

struct ggml_cuda_moe_grouped_context_test_access {
    static bool admission_closed(const ggml_cuda_moe_grouped_context & context) {
        return context.admission_closed_for_test();
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

static cudaStream_t candidate_test_graph_stream(void * data, const ggml_tensor *) {
    return static_cast<cudaStream_t>(data);
}

struct scheduler_certificate_probe {
    struct record {
        ggml_graph_execution_certificate certificate = {};
        uint64_t graph_uid = 0;
        uint32_t backend_index = UINT32_MAX;
    };

    std::array<ggml_backend_t, 2> backends = {};
    std::array<enum ggml_status (*)(ggml_backend_t, ggml_cgraph *), 2> delegates = {};
    std::array<record, 16> records = {};
    uint32_t n_backends = 0;
    uint32_t calls = 0;
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
    return probe.delegates[backend_index](backend, graph);
}

static bool scheduler_certificate_eval_callback(ggml_tensor *, bool, void *) {
    return false;
}

struct candidate_test_graph_streams {
    const ggml_tensor * producer = nullptr;
    cudaStream_t producer_stream = nullptr;
    cudaStream_t reader_stream = nullptr;
};

static cudaStream_t candidate_test_graph_mixed_stream(void * data, const ggml_tensor * node) {
    auto * streams = static_cast<candidate_test_graph_streams *>(data);
    return node == streams->producer ? streams->producer_stream : streams->reader_stream;
}

static int sample_zipf(std::mt19937 & rng, int n, double s) {
    // Rejection-sample a Zipf(s) over {0..n-1}. Cheap; fine for a test.
    static thread_local std::vector<double> cdf;
    if ((int)cdf.size() != n) {
        cdf.assign(n, 0.0);
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            sum += 1.0 / std::pow((double)(i + 1), s);
            cdf[i] = sum;
        }
        for (auto & v : cdf) v /= sum;
    }
    std::uniform_real_distribution<double> u(0.0, 1.0);
    double r = u(rng);
    auto it = std::lower_bound(cdf.begin(), cdf.end(), r);
    return (int)(it - cdf.begin());
}

struct host_barrier {
    std::atomic<bool> entered{false};
    std::atomic<bool> released{false};
};

static void CUDART_CB wait_on_host_barrier(void * data) {
    auto * barrier = static_cast<host_barrier *>(data);
    barrier->entered.store(true, std::memory_order_release);
    while (!barrier->released.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
}

static bool candidate_test_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t) {
    return *static_cast<const bool *>(dev->context);
}

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
            cached_buffer = ggml_backend_cuda_moe_cached_buffer_from_host_ptr(storage, BUFFER_SIZE);
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

static candidate_route candidate_top_k_route(
        candidate_test_fixture & fixture,
        int64_t n_experts,
        int64_t n_routes,
        int64_t n_tokens = 1,
        size_t view_offs = 0) {
    candidate_route result;
    const int64_t source_ne[] = {n_experts, n_tokens};
    result.source = fixture.tensor(GGML_TYPE_F32, 2, source_ne);
    result.root = ggml_argsort(fixture.ctx, result.source, GGML_SORT_ORDER_DESC);
    fixture.materialize(result.root);
    result.root->flags |= GGML_TENSOR_FLAG_COMPUTE;
    result.ids = ggml_view_4d(fixture.ctx, result.root, n_routes, n_tokens, 1, 1,
        result.root->nb[1], result.root->nb[2], result.root->nb[3], view_offs);
    fixture.materialize(result.ids);
    result.ids->flags |= GGML_TENSOR_FLAG_COMPUTE;
    return result;
}

struct candidate_fused_top_k_route {
    candidate_route route;
    ggml_tensor * softmax = nullptr;
    ggml_tensor * reshaped = nullptr;
    ggml_tensor * weights = nullptr;
};

static candidate_fused_top_k_route candidate_fused_top_k(candidate_test_fixture & fixture, int64_t n_experts, int64_t n_routes) {
    candidate_fused_top_k_route result;
    const int64_t logits_ne[] = {n_experts, 1};
    ggml_tensor * logits = fixture.tensor(GGML_TYPE_F32, 2, logits_ne);
    result.softmax = ggml_soft_max(fixture.ctx, logits);
    fixture.materialize(result.softmax);
    result.softmax->flags |= GGML_TENSOR_FLAG_COMPUTE;
    result.reshaped = ggml_reshape_2d(fixture.ctx, result.softmax, n_experts, 1);
    fixture.materialize(result.reshaped);
    result.reshaped->flags |= GGML_TENSOR_FLAG_COMPUTE;
    result.route.source = result.softmax;
    result.route.root = ggml_argsort(fixture.ctx, result.softmax, GGML_SORT_ORDER_DESC);
    fixture.materialize(result.route.root);
    result.route.root->flags |= GGML_TENSOR_FLAG_COMPUTE;
    result.route.ids = ggml_view_4d(fixture.ctx, result.route.root, n_routes, 1, 1, 1,
        result.route.root->nb[1], result.route.root->nb[2], result.route.root->nb[3], 0);
    fixture.materialize(result.route.ids);
    result.route.ids->flags |= GGML_TENSOR_FLAG_COMPUTE;
    result.weights = ggml_get_rows(fixture.ctx, result.reshaped, result.route.ids);
    fixture.materialize(result.weights);
    result.weights->flags |= GGML_TENSOR_FLAG_COMPUTE;
    return result;
}

static ggml_tensor * candidate_mmid(candidate_test_fixture & fixture, ggml_tensor * weight, ggml_tensor * ids) {
    const int64_t activation_ne[] = {weight->ne[0], 1, ids->ne[1]};
    const int64_t output_ne[] = {weight->ne[1], ids->ne[0], ids->ne[1]};
    ggml_tensor * activation = fixture.tensor(GGML_TYPE_F32, 3, activation_ne);
    ggml_tensor * result = fixture.tensor(GGML_TYPE_F32, 3, output_ne);
    result->op = GGML_OP_MUL_MAT_ID;
    result->src[0] = weight;
    result->src[1] = activation;
    result->src[2] = ids;
    result->flags |= GGML_TENSOR_FLAG_COMPUTE;
    return result;
}

static void candidate_set_route_tokens(
        const candidate_route & route,
        std::initializer_list<ggml_tensor *> readers,
        int64_t n_tokens) {
    route.source->ne[1] = n_tokens;
    route.source->nb[2] = route.source->nb[1] * n_tokens;
    route.source->nb[3] = route.source->nb[2];
    route.root->ne[1] = n_tokens;
    route.root->nb[2] = route.root->nb[1] * n_tokens;
    route.root->nb[3] = route.root->nb[2];
    route.ids->ne[1] = n_tokens;
    memcpy(route.ids->nb, route.root->nb, sizeof(route.ids->nb));
    for (ggml_tensor * reader : readers) {
        reader->src[1]->ne[2] = n_tokens;
        reader->src[1]->nb[3] = reader->src[1]->nb[2] * n_tokens;
        reader->ne[2] = n_tokens;
        reader->nb[3] = reader->nb[2] * n_tokens;
    }
}

static void candidate_rebuild_graph_uses(ggml_cgraph * graph) {
    ggml_hash_set_reset(&graph->visited_hash_set);
    memset(graph->use_counts, 0, graph->visited_hash_set.size * sizeof(graph->use_counts[0]));
    for (int node_index = 0; node_index < graph->n_nodes; ++node_index) {
        ggml_tensor * node = graph->nodes[node_index];
        CHECK(ggml_hash_find_or_insert(&graph->visited_hash_set, node) != GGML_HASHSET_FULL);
        for (uint32_t src_index = 0; src_index < GGML_MAX_SRC; ++src_index) {
            if (node->src[src_index] == nullptr) {
                continue;
            }
            const size_t hash_pos = ggml_hash_find_or_insert(&graph->visited_hash_set, node->src[src_index]);
            CHECK(hash_pos != GGML_HASHSET_FULL);
            graph->use_counts[hash_pos]++;
        }
    }
}

static void candidate_insert_graph_node(ggml_cgraph * graph, int32_t node_index, ggml_tensor * node) {
    CHECK(graph != nullptr && node != nullptr && node_index >= 0 && node_index <= graph->n_nodes && graph->n_nodes < graph->size);
    memmove(graph->nodes + node_index + 1, graph->nodes + node_index,
        (graph->n_nodes - node_index) * sizeof(graph->nodes[0]));
    graph->nodes[node_index] = node;
    ++graph->n_nodes;
    candidate_rebuild_graph_uses(graph);
}

static int32_t candidate_graph_use_count(const ggml_cgraph * graph, const ggml_tensor * tensor) {
    const size_t hash_pos = ggml_hash_find(&graph->visited_hash_set, tensor);
    CHECK(hash_pos != GGML_HASHSET_FULL && ggml_bitset_get(graph->visited_hash_set.used, hash_pos));
    return graph->use_counts[hash_pos];
}

static void candidate_stamp_execution(
        ggml_cgraph * graph,
        uint32_t domain,
        uint32_t row_semantics,
        uint32_t n_rows,
        uint32_t n_sequences,
        uint64_t source_graph_uid = 0) {
    CHECK(graph != nullptr);
    if (source_graph_uid == 0) {
        source_graph_uid = ggml_graph_next_uid();
    }
    graph->uid = ggml_graph_next_uid();
    CHECK(source_graph_uid != 0 && graph->uid != 0 && source_graph_uid != graph->uid);
    graph->execution_certificate = {};
    graph->execution_certificate.magic = GGML_GRAPH_EXECUTION_CERTIFICATE_MAGIC;
    graph->execution_certificate.abi_version = GGML_GRAPH_EXECUTION_CERTIFICATE_VERSION;
    graph->execution_certificate.struct_size = sizeof(graph->execution_certificate);
    graph->execution_certificate.domain = domain;
    graph->execution_certificate.row_semantics = row_semantics;
    graph->execution_certificate.n_rows = n_rows;
    graph->execution_certificate.n_sequences = n_sequences;
    graph->execution_certificate.owner_namespace = 0x746573742d6d6f65ULL;
    graph->execution_certificate.owner_generation = 1;
    graph->execution_certificate.source_graph_uid = source_graph_uid;
    graph->execution_certificate.split_graph_uid = graph->uid;
}

static void candidate_stamp_single_row_execution(ggml_cgraph * graph) {
    for (int32_t node_index = 0; node_index < graph->n_nodes; ++node_index) {
        const ggml_tensor * node = graph->nodes[node_index];
        if (node != nullptr && node->op == GGML_OP_MUL_MAT_ID && node->src[2] != nullptr) {
            if (node->src[2]->ne[1] == 1 && node->src[2]->ne[2] == 1 && node->src[2]->ne[3] == 1) {
                candidate_stamp_execution(graph, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
                    GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 1, 1);
            }
            return;
        }
    }
}

static ggml_cgraph * candidate_graph(candidate_test_fixture & fixture, std::initializer_list<ggml_tensor *> nodes) {
    ggml_cgraph * graph = ggml_new_graph_custom(fixture.ctx, 32, false);
    CHECK(graph != nullptr);
    for (ggml_tensor * node : nodes) {
        ggml_graph_add_node(graph, node);
    }
    candidate_rebuild_graph_uses(graph);
    candidate_stamp_single_row_execution(graph);
    return graph;
}

static ggml_cgraph * candidate_padded_graph(
        candidate_test_fixture & fixture,
        ggml_tensor * padding,
        uint32_t n_padding,
        std::initializer_list<ggml_tensor *> nodes) {
    ggml_cgraph * graph = ggml_new_graph_custom(fixture.ctx, n_padding + nodes.size(), false);
    CHECK(graph != nullptr);
    for (uint32_t i = 0; i < n_padding; ++i) {
        ggml_tensor * node = ggml_dup(fixture.ctx, padding);
        fixture.materialize(node);
        node->flags |= GGML_TENSOR_FLAG_COMPUTE;
        ggml_graph_add_node(graph, node);
    }
    for (ggml_tensor * node : nodes) {
        ggml_graph_add_node(graph, node);
    }
    candidate_rebuild_graph_uses(graph);
    candidate_stamp_single_row_execution(graph);
    return graph;
}

struct candidate_graph_coverage {
    const void * nodes = nullptr;
    uint64_t epoch = 0;
    uint64_t mmid_fingerprint = 0;
    uint32_t mmid_count = 0;
};

static candidate_graph_coverage candidate_certify_graph(
        ggml_cuda_moe_grouped_context & registry,
        ggml_cgraph * graph) {
    candidate_graph_coverage result;
    result.nodes = graph->nodes;
    result.epoch = registry.certify_graph_coverage(graph, &result.mmid_count, &result.mmid_fingerprint);
    CHECK(result.epoch != 0 && result.mmid_fingerprint != 0);
    return result;
}

static void candidate_test_graph_views(
        candidate_test_fixture & fixture,
        ggml_cuda_moe_grouped_context & global_registry,
        const candidate_route & fused_route,
        const candidate_route & separate_route,
        ggml_tensor * fused_gate_up,
        ggml_tensor * fused_down,
        ggml_tensor * separate_gate,
        ggml_tensor * separate_up,
        ggml_tensor * separate_down) {
    ggml_cgraph * split_parent = candidate_graph(fixture, {
        fused_route.root, fused_route.ids, fused_gate_up, fused_down,
        separate_route.root, separate_route.ids, separate_gate, separate_up, separate_down,
    });
    ggml_cgraph split_view = ggml_graph_view(split_parent, 0, 4);
    candidate_stamp_execution(&split_view, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
        GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 1, 1, split_parent->uid);
    const auto split_coverage = candidate_certify_graph(global_registry, &split_view);
    std::shared_ptr<ggml_cuda_moe_graph_plan> split_plan;
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(global_registry.prepare_graph_execution(&split_view, 31, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &split_plan, prepared.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(prepared->size() == 1 && prepared->find(fused_down, nullptr));
    }
    const ggml_cuda_moe_graph_plan * stable_split_plan = split_plan.get();
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(global_registry.prepare_graph_execution(&split_view, 32, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &split_plan, prepared.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(split_plan.get() != stable_split_plan && prepared->find(fused_down, nullptr));
    }

    split_plan.reset();
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(global_registry.prepare_graph_execution(
            &split_view, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &split_plan, prepared.get(),
            split_coverage.epoch, split_coverage.nodes, split_coverage.mmid_count, split_coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(prepared->size() == 1 && prepared->find(fused_down, nullptr));
    }
    stable_split_plan = split_plan.get();
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(global_registry.prepare_graph_execution(
            &split_view, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &split_plan, prepared.get(),
            split_coverage.epoch, split_coverage.nodes, split_coverage.mmid_count, split_coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(split_plan.get() == stable_split_plan && prepared->find(fused_down, nullptr));
    }
    {
        ggml_cgraph exact_callback = ggml_graph_view(&split_view, 0, split_view.n_nodes);
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(!global_registry.bind_graph_plan(
            &exact_callback, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, *split_plan, prepared.get(),
            split_coverage.epoch, split_coverage.nodes, split_coverage.mmid_count, split_coverage.mmid_fingerprint));
        ggml_cuda_moe_graph_plan callback_plan;
        global_registry.compile_graph_plan(&exact_callback, 0, &callback_plan, prepared.get(),
            split_coverage.epoch, split_coverage.nodes, split_coverage.mmid_count, split_coverage.mmid_fingerprint);
        CHECK(prepared->outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_LEGACY);
    }
    {
        ggml_cgraph prefix_callback = ggml_graph_view(&split_view, 0, split_view.n_nodes - 1);
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(!global_registry.bind_graph_plan(
            &prefix_callback, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, *split_plan, prepared.get(),
            split_coverage.epoch, split_coverage.nodes, split_coverage.mmid_count, split_coverage.mmid_fingerprint));
        CHECK(!global_registry.bind_graph_plan(
            &split_view, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, *split_plan, prepared.get(),
            split_coverage.epoch + 1, split_coverage.nodes, split_coverage.mmid_count, split_coverage.mmid_fingerprint));
    }

    ggml_cgraph second_split_view = ggml_graph_view(split_parent, 4, split_parent->n_nodes);
    candidate_stamp_execution(&second_split_view, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
        GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 1, 1, split_parent->uid);
    const auto second_split_coverage = candidate_certify_graph(global_registry, &second_split_view);
    std::shared_ptr<ggml_cuda_moe_graph_plan> second_split_plan;
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(global_registry.prepare_graph_execution(
            &second_split_view, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &second_split_plan, prepared.get(),
            second_split_coverage.epoch, second_split_coverage.nodes,
            second_split_coverage.mmid_count, second_split_coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(prepared->size() == 1 && prepared->find(separate_down, nullptr));
    }
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(global_registry.prepare_graph_execution(
            &second_split_view, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &second_split_plan, prepared.get(),
            second_split_coverage.epoch, second_split_coverage.nodes,
            second_split_coverage.mmid_count, second_split_coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(prepared->find(separate_down, nullptr));
    }

    const std::shared_ptr<ggml_cuda_moe_graph_plan> decode_split_plan = split_plan;
    candidate_set_route_tokens(fused_route, {fused_gate_up, fused_down}, 4);
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(global_registry.prepare_graph_execution(
            &split_view, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &split_plan, prepared.get(),
            split_coverage.epoch, split_coverage.nodes, split_coverage.mmid_count, split_coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(split_plan.get() != decode_split_plan.get() &&
            prepared->outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_LEGACY);
        CHECK(!global_registry.bind_graph_plan(
            &split_view, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, *decode_split_plan, prepared.get(),
            split_coverage.epoch, split_coverage.nodes, split_coverage.mmid_count, split_coverage.mmid_fingerprint));
    }
    candidate_set_route_tokens(fused_route, {fused_gate_up, fused_down}, 1);
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(global_registry.prepare_graph_execution(
            &split_view, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &split_plan, prepared.get(),
            split_coverage.epoch, split_coverage.nodes, split_coverage.mmid_count, split_coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(prepared->find(fused_down, nullptr));
    }

    ggml_tensor * split_consumer = ggml_dup(fixture.ctx, fused_down);
    fixture.materialize(split_consumer);
    split_consumer->flags |= GGML_TENSOR_FLAG_COMPUTE;
    ggml_cgraph * split_consumer_parent = candidate_graph(fixture, {
        fused_route.root, fused_route.ids, fused_gate_up, fused_down, split_consumer,
        separate_route.root, separate_route.ids, separate_gate, separate_up, separate_down,
    });
    ggml_cgraph split_consumer_view = ggml_graph_view(split_consumer_parent, 0, 4);
    candidate_stamp_execution(&split_consumer_view, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
        GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 1, 1, split_consumer_parent->uid);
    const auto split_consumer_coverage = candidate_certify_graph(global_registry, &split_consumer_view);
    split_plan.reset();
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(global_registry.prepare_graph_execution(
            &split_consumer_view, 35, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &split_plan, prepared.get(),
            split_consumer_coverage.epoch, split_consumer_coverage.nodes,
            split_consumer_coverage.mmid_count, split_consumer_coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(prepared->size() == 1 && !prepared->find(fused_down, nullptr));
    }
    stable_split_plan = split_plan.get();
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(global_registry.prepare_graph_execution(
            &split_consumer_view, 36, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &split_plan, prepared.get(),
            split_consumer_coverage.epoch, split_consumer_coverage.nodes,
            split_consumer_coverage.mmid_count, split_consumer_coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(split_plan.get() == stable_split_plan && !prepared->find(fused_down, nullptr));
    }
    split_consumer->src[0] = nullptr;
    candidate_rebuild_graph_uses(split_consumer_parent);
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(global_registry.prepare_graph_execution(
            &split_consumer_view, 37, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &split_plan, prepared.get(),
            split_consumer_coverage.epoch, split_consumer_coverage.nodes,
            split_consumer_coverage.mmid_count, split_consumer_coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(split_plan.get() != stable_split_plan && prepared->find(fused_down, nullptr));
    }
}

static ggml_backend_moe_candidate_snapshot_v1 candidate_snapshot(
        uint32_t n_slots,
        const ggml_backend_moe_candidate_group_v1 * groups,
        uint32_t n_groups) {
    ggml_backend_moe_candidate_snapshot_v1 result = {};
    result.magic = GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V1_MAGIC;
    result.abi_version = GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V1_VERSION;
    result.struct_size = sizeof(result);
    result.n_slots = n_slots;
    result.groups = groups;
    result.n_groups = n_groups;
    return result;
}

static ggml_backend_moe_candidate_snapshot_v2 candidate_snapshot_v2(
        uint32_t n_slots,
        const ggml_backend_moe_candidate_group_v2 * groups,
        uint32_t n_groups,
        const ggml_backend_moe_candidate_tensor_v2 * tensors,
        uint32_t n_tensors) {
    ggml_backend_moe_candidate_snapshot_v2 result = {};
    result.magic = GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_MAGIC;
    result.abi_version = GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_VERSION;
    result.struct_size = sizeof(result);
    result.n_slots = n_slots;
    result.groups = groups;
    result.n_groups = n_groups;
    result.tensors = tensors;
    result.n_tensors = n_tensors;
    return result;
}

struct candidate_graph_holder {
    void reset() {
        plan.reset();
        nodes = nullptr;
        epoch = 0;
        mmid_fingerprint = 0;
        n_nodes = 0;
        mmid_count = 0;
    }

    std::shared_ptr<ggml_cuda_moe_graph_plan> plan;
    const void * nodes = nullptr;
    uint64_t epoch = 0;
    uint64_t mmid_fingerprint = 0;
    int32_t n_nodes = 0;
    uint32_t mmid_count = 0;
};

struct candidate_graph_holder_context {
    explicit candidate_graph_holder_context(ggml_cuda_moe_grouped_context & registry) : registry(registry) {}

    candidate_graph_holder & holder(const void * key) {
        auto & result = holders[key];
        if (result == nullptr) {
            result = std::make_unique<candidate_graph_holder>();
        }
        return *result;
    }

    void certify(ggml_cgraph * graph) {
        ggml_cuda_moe_graph_span span;
        CHECK(graph != nullptr && ggml_cuda_moe_graph_span_bounds(graph->nodes, graph->n_nodes, &span));
        const void * key = graph->nodes[0];
        for (auto & entry : holders) {
            if (entry.first == key || ggml_cuda_moe_graph_spans_overlap(
                    entry.second->nodes, entry.second->n_nodes, graph->nodes, graph->n_nodes)) {
                entry.second->reset();
            }
        }
        uint32_t mmid_count = 0;
        uint64_t mmid_fingerprint = 0;
        const uint64_t epoch = registry.certify_graph_coverage(graph, &mmid_count, &mmid_fingerprint);
        auto & current = holder(key);
        current.reset();
        CHECK(epoch != 0);
        current.nodes = graph->nodes;
        current.epoch = epoch;
        current.mmid_fingerprint = mmid_fingerprint;
        current.n_nodes = graph->n_nodes;
        current.mmid_count = mmid_count;
    }

    bool recover(ggml_cgraph * graph, candidate_graph_holder & holder) {
        holder.reset();
        uint64_t epoch = 0;
        uint32_t mmid_count = 0;
        uint64_t mmid_fingerprint = 0;
        if (!registry.recover_graph_coverage(graph, &epoch, &mmid_count, &mmid_fingerprint)) {
            return false;
        }
        holder.nodes = graph->nodes;
        holder.epoch = epoch;
        holder.mmid_fingerprint = mmid_fingerprint;
        holder.n_nodes = graph->n_nodes;
        holder.mmid_count = mmid_count;
        return true;
    }

    void evict(const void * key) {
        holders.erase(key);
    }

    ggml_cuda_moe_grouped_context & registry;
    std::unordered_map<const void *, std::unique_ptr<candidate_graph_holder>> holders;
};

static ggml_cuda_moe_graph_prepare_result candidate_prepare_graph_holder(
        candidate_graph_holder_context & holder_context,
        ggml_cgraph * graph,
        uint64_t graph_uid,
        ggml_cuda_moe_graph_property_hint property_hint,
        ggml_cuda_moe_graph_execution * execution) {
    auto & holder = holder_context.holder(graph->nodes[0]);
    if (holder.epoch == 0 || holder.nodes != graph->nodes || holder.n_nodes != graph->n_nodes) {
        holder_context.recover(graph, holder);
    }
    std::shared_ptr<ggml_cuda_moe_graph_plan> local_plan;
    auto * plan = &local_plan;
    uint64_t coverage_epoch = 0;
    uint64_t coverage_mmid_fingerprint = 0;
    const void * coverage_nodes = nullptr;
    uint32_t coverage_mmid_count = 0;
    if (holder.epoch != 0 && holder.nodes == graph->nodes && holder.n_nodes == graph->n_nodes) {
        plan = &holder.plan;
        coverage_epoch = holder.epoch;
        coverage_mmid_fingerprint = holder.mmid_fingerprint;
        coverage_nodes = holder.nodes;
        coverage_mmid_count = holder.mmid_count;
    }
    return holder_context.registry.prepare_graph_execution(
        graph, graph_uid, property_hint, plan, execution, coverage_epoch, coverage_nodes,
        coverage_mmid_count, coverage_mmid_fingerprint);
}

static void candidate_test_graph_holder_coverage(
        candidate_test_fixture & fixture,
        const ggml_backend_moe_candidate_snapshot_v1 & snapshot,
        const candidate_route & fused_route,
        const candidate_route & separate_route,
        ggml_tensor * fused_gate_up,
        ggml_tensor * fused_down,
        ggml_tensor * separate_gate,
        ggml_tensor * separate_up,
        ggml_tensor * separate_down) {
    ggml_cuda_moe_grouped_context registry(&fixture.owner, 0);
    CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    candidate_graph_holder_context holder_context(registry);

    ggml_cgraph * parent = candidate_graph(fixture, {
        fused_route.root, fused_route.ids, fused_gate_up, fused_down,
        separate_route.root, separate_route.ids, separate_gate, separate_up, separate_down,
    });
    ggml_cgraph first = ggml_graph_view(parent, 0, 4);
    ggml_cgraph suffix = ggml_graph_view(parent, 4, parent->n_nodes);
    candidate_stamp_execution(&first, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
        GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 1, 1, parent->uid);
    candidate_stamp_execution(&suffix, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
        GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 1, 1, parent->uid);

    holder_context.certify(&first);
    auto & first_holder = holder_context.holder(first.nodes[0]);
    {
        auto execution = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(candidate_prepare_graph_holder(
            holder_context, &first, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, execution.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(execution->find(fused_down, nullptr));
    }
    const uint64_t first_epoch = first_holder.epoch;
    const auto first_plan = first_holder.plan;

    holder_context.certify(&suffix);
    CHECK(first_holder.epoch == first_epoch && first_holder.plan == first_plan);
    {
        auto execution = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(candidate_prepare_graph_holder(
            holder_context, &suffix, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, execution.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(execution->find(separate_down, nullptr));
    }

    const void * suffix_key = suffix.nodes[0];
    const uint64_t suffix_epoch = holder_context.holder(suffix_key).epoch;
    holder_context.evict(suffix_key);
    CHECK(holder_context.holders.find(suffix_key) == holder_context.holders.end());
    {
        auto execution = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(candidate_prepare_graph_holder(
            holder_context, &suffix, 100, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, execution.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(holder_context.holder(suffix_key).epoch == suffix_epoch && execution->find(separate_down, nullptr));
    }
    const auto recovered_plan = holder_context.holder(suffix_key).plan;
    {
        auto execution = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(candidate_prepare_graph_holder(
            holder_context, &suffix, 101, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, execution.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(holder_context.holder(suffix_key).plan == recovered_plan);
    }

    const auto old_suffix_plan = holder_context.holder(suffix_key).plan;
    holder_context.certify(&suffix);
    auto & suffix_holder = holder_context.holder(suffix_key);
    CHECK(suffix_holder.epoch > suffix_epoch && suffix_holder.plan == nullptr);
    {
        auto execution = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(!registry.bind_graph_plan(
            &suffix, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, *old_suffix_plan, execution.get(),
            suffix_holder.epoch, suffix.nodes));
        CHECK(candidate_prepare_graph_holder(
            holder_context, &suffix, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, execution.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
    }

    const auto pre_replace_plan = suffix_holder.plan;
    CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    {
        auto execution = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(candidate_prepare_graph_holder(
            holder_context, &suffix, 102, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, execution.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(suffix_holder.plan != pre_replace_plan && execution->find(separate_down, nullptr));
    }

    candidate_set_route_tokens(separate_route, {separate_gate, separate_up, separate_down}, 4);
    {
        auto execution = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(candidate_prepare_graph_holder(
            holder_context, &suffix, 103, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, execution.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(execution->outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_LEGACY);
    }
    candidate_set_route_tokens(separate_route, {separate_gate, separate_up, separate_down}, 1);
    {
        auto execution = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(candidate_prepare_graph_holder(
            holder_context, &suffix, 104, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, execution.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(execution->find(separate_down, nullptr));
    }

    const candidate_route disjoint_route = candidate_top_k_route(fixture, 4, 2);
    ggml_tensor * disjoint_gate_up = candidate_mmid(fixture, fused_gate_up->src[0], disjoint_route.ids);
    ggml_tensor * disjoint_down = candidate_mmid(fixture, fused_down->src[0], disjoint_route.ids);
    ggml_cgraph * disjoint = candidate_graph(fixture, {
        disjoint_route.root, disjoint_route.ids, disjoint_gate_up, disjoint_down,
    });
    holder_context.certify(disjoint);
    {
        auto execution = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(candidate_prepare_graph_holder(
            holder_context, disjoint, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, execution.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
    }
    auto & disjoint_holder = holder_context.holder(disjoint->nodes[0]);
    const uint64_t disjoint_epoch = disjoint_holder.epoch;
    const auto disjoint_plan = disjoint_holder.plan;
    const auto stale_suffix_plan = suffix_holder.plan;

    holder_context.certify(parent);
    CHECK(suffix_holder.epoch == 0 && suffix_holder.plan == nullptr);
    CHECK(disjoint_holder.epoch == disjoint_epoch && disjoint_holder.plan == disjoint_plan);
    CHECK(stale_suffix_plan != nullptr);
    CHECK(!holder_context.recover(&suffix, suffix_holder));
    for (uint64_t uid = 105; uid < 107; ++uid) {
        auto execution = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(candidate_prepare_graph_holder(
            holder_context, &suffix, uid, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, execution.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(execution->find(separate_down, nullptr) && suffix_holder.plan == nullptr);
    }
}

static const ggml_backend_moe_candidate_bank_v1 * candidate_bank(
        const ggml_backend_moe_candidate_group_v1 & group,
        uint32_t role) {
    for (uint32_t i = 0; i < group.n_banks; ++i) {
        if (group.banks[i].role == role) {
            return &group.banks[i];
        }
    }
    return nullptr;
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

static void test_candidate_graph_coverage_ledger() {
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

static void test_candidate_graph_inventory_reuse() {
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

static void test_mmid_capabilities() {
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

static void test_scheduler_execution_certificate() {
    ggml_backend_ptr first_backend(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    ggml_backend_ptr second_backend(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    CHECK(first_backend != nullptr && second_backend != nullptr);

    scheduler_certificate_probe probe;
    probe.backends = {first_backend.get(), second_backend.get()};
    probe.delegates = {first_backend->iface.graph_compute, second_backend->iface.graph_compute};
    probe.n_backends = probe.backends.size();
    CHECK(probe.delegates[0] != nullptr && probe.delegates[1] != nullptr && scheduler_certificate_probe_current == nullptr);
    scheduler_certificate_probe_current = &probe;
    first_backend->iface.graph_compute = scheduler_certificate_graph_compute;
    second_backend->iface.graph_compute = scheduler_certificate_graph_compute;

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

    first_backend->iface.graph_compute = probe.delegates[0];
    second_backend->iface.graph_compute = probe.delegates[1];
    scheduler_certificate_probe_current = nullptr;
}

static void test_graph_execution_certificate_policy() {
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
    candidate_stamp_execution(isolated.graph, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
        GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SPECULATIVE, 2, 2);
    check_execution_legacy(isolated, 1230);
    candidate_stamp_execution(isolated.graph, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
        GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL, 2, 2);
    {
        ggml_cuda_moe_graph_plan plan;
        ggml_cuda_moe_graph_execution execution;
        compile(isolated, 1231, plan, execution);
        CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_PREFILL_LEGACY && !execution.find(isolated.down, nullptr));
        CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_prefill_reason(plan, 0));
    }

    auto too_many_routes = make_graph(4, 4);
    candidate_stamp_execution(too_many_routes.graph, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
        GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 4, 4);
    check_execution_legacy(too_many_routes, 1240);

    auto stable = make_graph(2, 3);
    candidate_stamp_execution(stable.graph, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
        GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 3, 3);
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

static void test_candidate_generic_physical_truth() {
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

static void test_candidate_producer() {
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
    llama_moe_candidate_snapshot lora_on(*model, loras);
    CHECK(lora_on.get().n_groups == 4);
    CHECK(lora_on.get().groups[0].flags & GGML_BACKEND_MOE_CANDIDATE_GROUP_V2_FLAG_ACTIVE_LORA);
    CHECK(candidate_tensor(lora_on.get(), separate.ffn_gate_exps)->flags & GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_ACTIVE_LORA);
    CHECK(registry.replace(&lora_on.get()) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(registry.state().n_groups == 2);
    CHECK(!registry.find_weight(separate.ffn_gate_exps, nullptr));
    loras.clear();
    llama_moe_candidate_snapshot lora_off(*model, loras);
    CHECK(lora_off.get().n_groups == 4);
    CHECK(registry.replace(&lora_off.get()) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(registry.find_weight(separate.ffn_gate_exps, nullptr));

    fused.ffn_up_exps_s = fixture.cached_tensor(GGML_TYPE_F32, 1, scale_ne);
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

static void test_candidate_registry(bool benchmark) {
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

static void test_legacy_owner_leases() {
    candidate_test_fixture fixture;
    const int64_t gate_ne[] = {64, 32, 4};
    const int64_t gate_up_ne[] = {64, 64, 4};
    const int64_t down_ne[] = {32, 64, 4};
    ggml_tensor * gate = fixture.tensor(GGML_TYPE_BF16, 3, gate_ne);
    ggml_tensor * up = fixture.tensor(GGML_TYPE_BF16, 3, gate_ne);
    ggml_tensor * gate_up = fixture.tensor(GGML_TYPE_BF16, 3, gate_up_ne);
    ggml_tensor * down = fixture.tensor(GGML_TYPE_BF16, 3, down_ne);
    ggml_tensor * unsupported = fixture.tensor(GGML_TYPE_BF16, 3, gate_up_ne);
    std::array<ggml_backend_moe_candidate_bank_v1, 2> banks = {{
        {gate_up, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, 0},
        {down, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
    }};
    ggml_backend_moe_candidate_group_v1 group = {banks.data(), banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 0, 0};
    auto snapshot = candidate_snapshot(12, &group, 1);

    auto first = std::make_unique<ggml_cuda_moe_grouped_context>(&fixture.owner);
    ggml_cuda_moe_grouped_context second(&fixture.owner);
    CHECK(first->replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(second.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);

    auto first_lease = first->acquire_legacy_cache(gate_up);
    auto second_lease = second.acquire_legacy_cache(gate_up);
    CHECK(first_lease && second_lease);
    CHECK(first_lease.get() == nullptr && second_lease.get() == nullptr);
    CHECK(first_lease.acquisition().owner != second_lease.acquisition().owner);
    CHECK(first_lease.acquisition().tensor == gate_up && first_lease.acquisition().candidate_generation == 1);
    CHECK(first_lease.acquisition().authority_epoch != 0 && first_lease.acquisition().group_index == 0);
    CHECK(first_lease.acquisition().role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT);
    CHECK(first_lease.acquisition().n_slots == 12 && first_lease.acquisition().registered_source == 1);
    CHECK(!second.acquire_legacy_cache(gate_up, &first_lease.acquisition()));

    auto moved_lease = std::move(first_lease);
    CHECK(!first_lease && moved_lease && moved_lease.get() == nullptr);
    const auto stale = moved_lease.acquisition();

    std::atomic<bool> replacement_started{false};
    std::atomic<bool> replacement_done{false};
    std::thread replacement_thread([&]() {
        replacement_started.store(true, std::memory_order_release);
        CHECK(first->replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        replacement_done.store(true, std::memory_order_release);
    });
    while (!replacement_started.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    for (;;) {
        auto rejected = first->acquire_legacy_cache(gate_up);
        if (!rejected) {
            break;
        }
        std::this_thread::yield();
    }
    CHECK(!replacement_done.load(std::memory_order_acquire));
    moved_lease = {};
    replacement_thread.join();
    CHECK(replacement_done.load(std::memory_order_acquire));
    CHECK(first->state().generation == 2 && second.state().generation == 1);
    CHECK(!first->acquire_legacy_cache(gate_up, &stale));

    auto current = first->acquire_legacy_cache(gate_up);
    CHECK(current && current.get() == nullptr);
    CHECK(current.acquisition().candidate_generation == 2 && current.acquisition().authority_epoch > stale.authority_epoch);
    auto wrong_generation = current.acquisition();
    wrong_generation.candidate_generation--;
    CHECK(!first->acquire_legacy_cache(gate_up, &wrong_generation));
    auto wrong_epoch = current.acquisition();
    wrong_epoch.authority_epoch--;
    CHECK(!first->acquire_legacy_cache(gate_up, &wrong_epoch));
    current = {};

    std::array<ggml_backend_moe_candidate_bank_v1, 3> separate_banks = {{
        {gate, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT, 0},
        {up, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT, 0},
        {down, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
    }};
    ggml_backend_moe_candidate_group_v1 separate_group = {
        separate_banks.data(), separate_banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 0, 0,
    };
    auto separate_snapshot = candidate_snapshot(12, &separate_group, 1);
    CHECK(first->replace(&separate_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);

    auto gate_lease = first->acquire_legacy_cache(gate);
    auto up_lease = first->acquire_legacy_cache(up);
    auto down_lease = first->acquire_legacy_cache(down);
    CHECK(gate_lease && up_lease && down_lease);
    CHECK(gate_lease.get() == nullptr && up_lease.get() == nullptr && down_lease.get() == nullptr);
    CHECK(gate_lease.acquisition().group_index == 0 && gate_lease.acquisition().role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT);
    CHECK(up_lease.acquisition().group_index == 0 && up_lease.acquisition().role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT);
    CHECK(down_lease.acquisition().group_index == 0 && down_lease.acquisition().role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT);
    CHECK(gate_lease.acquisition().registered_source == 1 && up_lease.acquisition().registered_source == 1 && down_lease.acquisition().registered_source == 1);

    const auto stale_data = gate_lease.acquisition();
    gate_lease = {};
    void * gate_data = gate->data;
    gate->data = static_cast<uint8_t *>(gate_data) + 1;
    CHECK(!first->acquire_legacy_cache(gate, &stale_data));
    CHECK(!first->acquire_legacy_cache(gate));
    gate->data = gate_data;
    gate_lease = first->acquire_legacy_cache(gate);
    CHECK(gate_lease && gate_lease.get() == nullptr);
    CHECK(gate_lease.acquisition().authority_epoch > stale_data.authority_epoch);
    gate_lease = {};

    const auto stale_stride = up_lease.acquisition();
    up_lease = {};
    const size_t up_stride = up->nb[1];
    up->nb[1]++;
    CHECK(!first->acquire_legacy_cache(up, &stale_stride));
    CHECK(!first->acquire_legacy_cache(up));
    up->nb[1] = up_stride;
    up_lease = first->acquire_legacy_cache(up);
    CHECK(up_lease && up_lease.get() == nullptr);
    CHECK(up_lease.acquisition().authority_epoch > stale_stride.authority_epoch);
    up_lease = {};
    down_lease = {};

    separate_group.flags = 1;
    separate_snapshot.n_slots = 7;
    CHECK(first->replace(&separate_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_REJECTED);
    separate_group.flags = 0;
    auto rejected_lease = first->acquire_legacy_cache(gate);
    CHECK(rejected_lease && rejected_lease.get() == nullptr);
    CHECK(rejected_lease.acquisition().n_slots == 7 && rejected_lease.acquisition().registered_source == 0);
    CHECK(rejected_lease.acquisition().group_index == UINT32_MAX);
    CHECK(rejected_lease.acquisition().role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID);
    rejected_lease = {};

    auto disabled_snapshot = candidate_snapshot(9, nullptr, 0);
    CHECK(first->replace(&disabled_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    auto disabled_lease = first->acquire_legacy_cache(gate);
    CHECK(disabled_lease && disabled_lease.get() == nullptr);
    CHECK(disabled_lease.acquisition().n_slots == 9 && disabled_lease.acquisition().registered_source == 0);
    CHECK(disabled_lease.acquisition().group_index == UINT32_MAX);
    CHECK(disabled_lease.acquisition().role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID);
    disabled_lease = {};

    auto unsupported_lease = first->acquire_legacy_cache(unsupported);
    CHECK(unsupported_lease && unsupported_lease.get() == nullptr);
    CHECK(unsupported_lease.acquisition().group_index == UINT32_MAX);
    CHECK(unsupported_lease.acquisition().role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID);
    CHECK(unsupported_lease.acquisition().registered_source == 0 && unsupported_lease.acquisition().n_slots == 9);
    unsupported_lease = {};
    second_lease = {};

    auto null_cache_owner = std::make_unique<ggml_cuda_moe_grouped_context>(&fixture.owner);
    auto null_cache_snapshot = candidate_snapshot(0, nullptr, 0);
    CHECK(null_cache_owner->replace(&null_cache_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    auto null_cache_operation = null_cache_owner->begin_legacy_operation();
    CHECK(null_cache_operation && !null_cache_owner->acquire_legacy_cache(down));
    auto replacement_snapshot = candidate_snapshot(9, nullptr, 0);
    std::atomic<bool> null_replacement_started{false};
    std::atomic<bool> null_replacement_done{false};
    std::thread null_replacement_thread([&]() {
        null_replacement_started.store(true, std::memory_order_release);
        CHECK(null_cache_owner->replace(&replacement_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        null_replacement_done.store(true, std::memory_order_release);
    });
    while (!null_replacement_started.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    for (;;) {
        auto rejected = null_cache_owner->begin_legacy_operation();
        if (!rejected) {
            break;
        }
        std::this_thread::yield();
    }
    CHECK(!null_replacement_done.load(std::memory_order_acquire));
    null_cache_operation = {};
    null_replacement_thread.join();
    CHECK(null_replacement_done.load(std::memory_order_acquire));

    CHECK(null_cache_owner->replace(&null_cache_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    null_cache_operation = null_cache_owner->begin_legacy_operation();
    CHECK(null_cache_operation && !null_cache_owner->acquire_legacy_cache(down));
    auto * null_cache_context = null_cache_owner.get();
    std::atomic<bool> null_shutdown_started{false};
    std::atomic<bool> null_shutdown_done{false};
    std::thread null_shutdown_thread([&]() {
        null_shutdown_started.store(true, std::memory_order_release);
        null_cache_context->shutdown();
        null_shutdown_done.store(true, std::memory_order_release);
    });
    while (!null_shutdown_started.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    for (;;) {
        auto rejected = null_cache_context->begin_legacy_operation();
        if (!rejected) {
            break;
        }
        std::this_thread::yield();
    }
    CHECK(!null_shutdown_done.load(std::memory_order_acquire));
    null_cache_operation = {};
    null_shutdown_thread.join();
    CHECK(null_shutdown_done.load(std::memory_order_acquire));
    null_cache_owner.reset();

    auto terminal = std::make_unique<ggml_cuda_moe_grouped_context>(&fixture.owner);
    CHECK(terminal->replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    auto terminal_lease = terminal->acquire_legacy_cache(down);
    CHECK(terminal_lease && terminal_lease.get() == nullptr);
    auto * terminal_context = terminal.get();
    std::atomic<bool> shutdown_started{false};
    std::atomic<bool> shutdown_done{false};
    std::thread shutdown_thread([&]() {
        shutdown_started.store(true, std::memory_order_release);
        terminal_context->shutdown();
        shutdown_done.store(true, std::memory_order_release);
        terminal.reset();
    });
    while (!shutdown_started.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    for (;;) {
        auto rejected = terminal_context->acquire_legacy_cache(down);
        if (!rejected) {
            break;
        }
        std::this_thread::yield();
    }
    CHECK(!shutdown_done.load(std::memory_order_acquire));
    terminal_lease = {};
    shutdown_thread.join();
    CHECK(shutdown_done.load(std::memory_order_acquire));

    fprintf(stderr, "test-moe-cache: legacy owner leases OK\n");
}

static void test_grouped_context_resources() {
    candidate_test_fixture fixture;
    const int64_t gate_up_ne[] = {64, 64, 4};
    const int64_t down_ne[] = {32, 64, 4};
    ggml_tensor * gate_up = fixture.tensor(GGML_TYPE_BF16, 3, gate_up_ne);
    ggml_tensor * down = fixture.tensor(GGML_TYPE_BF16, 3, down_ne);
    ggml_tensor * gate_up_peer = fixture.tensor(GGML_TYPE_BF16, 3, gate_up_ne);
    ggml_tensor * down_peer = fixture.tensor(GGML_TYPE_BF16, 3, down_ne);
    std::array<ggml_backend_moe_candidate_bank_v1, 2> banks = {{
        {gate_up, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, 0},
        {down, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
    }};
    std::array<ggml_backend_moe_candidate_bank_v1, 2> peer_banks = {{
        {gate_up_peer, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, 0},
        {down_peer, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
    }};
    ggml_backend_moe_candidate_group_v1 group = {banks.data(), banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 0, 0};
    ggml_backend_moe_candidate_group_v1 peer_group = {peer_banks.data(), peer_banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 0, 0};
    std::array<ggml_backend_moe_candidate_group_v1, 2> groups = {group, peer_group};
    auto snapshot = candidate_snapshot(12, groups.data(), groups.size());

    auto first = std::make_unique<ggml_cuda_moe_grouped_context>(&fixture.owner);
    ggml_cuda_moe_grouped_context second(&fixture.owner);
    CHECK(first->replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    snapshot.n_slots = 48;
    CHECK(second.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);

    ggml_cuda_moe_candidate_group_key first_key;
    ggml_cuda_moe_candidate_group_key peer_key;
    ggml_cuda_moe_candidate_group_key second_key;
    CHECK(first->find_down_group_key(down, &first_key));
    CHECK(first->find_down_group_key(down_peer, &peer_key));
    CHECK(second.find_down_group_key(down, &second_key));
    ggml_cuda_moe_grouped_acquisition first_acquisition;
    ggml_cuda_moe_grouped_acquisition peer_acquisition;
    ggml_cuda_moe_grouped_acquisition repeated_acquisition;
    ggml_cuda_moe_grouped_acquisition second_acquisition;
    CHECK(first->acquire_group_resources(first_key, &first_acquisition));
    CHECK(first->acquire_group_resources(first_key, &repeated_acquisition));
    CHECK(first->acquire_group_resources(peer_key, &peer_acquisition));
    CHECK(second.acquire_group_resources(second_key, &second_acquisition));
    CHECK(first_acquisition.resource_generation == 1);
    CHECK(repeated_acquisition.resource_generation == first_acquisition.resource_generation);
    CHECK(peer_acquisition.resource_generation == 2);
    CHECK(second_acquisition.resource_generation == 1);

    ggml_cuda_moe_grouped_resource_info resource_info;
    CHECK(first->get_group_resources(first_acquisition, &resource_info));
    CHECK(resource_info.acquisition.candidate.generation == 1 && resource_info.n_slots == 12);
    CHECK(resource_info.down == down && resource_info.layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP && resource_info.n_banks == 2);
    ggml_cuda_moe_grouped_transaction inactive_transaction;
    inactive_transaction.acquisition = first_acquisition;
    CHECK(!first->get_group_resource_bank(inactive_transaction, 0, nullptr));
    ggml_cuda_moe_grouped_transaction first_transaction;
    ggml_cuda_moe_grouped_transaction repeated_transaction;
    CHECK(first->begin_group_transaction(first_acquisition, &first_transaction));
    CHECK(!first->begin_group_transaction(first_acquisition, &repeated_transaction));
    CHECK(repeated_transaction.transaction_token == 0);
    CHECK(first->get_group_resources(first_acquisition, &resource_info) && resource_info.transaction_active == 1);
    bool found_gate_up = false;
    bool found_down = false;
    for (uint32_t i = 0; i < resource_info.n_banks; ++i) {
        ggml_cuda_moe_grouped_bank_descriptor descriptor;
        CHECK(first->get_group_resource_bank(first_transaction, i, &descriptor));
        CHECK(descriptor.buffer == descriptor.tensor->buffer && descriptor.buft == descriptor.tensor->buffer->buft);
        CHECK(descriptor.source_data == descriptor.tensor->data && descriptor.buffer_base == fixture.storage);
        CHECK(descriptor.buffer_size == candidate_test_fixture::BUFFER_SIZE && descriptor.byte_extent == ggml_nbytes(descriptor.tensor));
        CHECK(descriptor.data_offset == static_cast<uint64_t>(static_cast<const uint8_t *>(descriptor.source_data) - static_cast<const uint8_t *>(descriptor.buffer_base)));
        CHECK(descriptor.expert_stride == descriptor.tensor->nb[2] && descriptor.alignment == ggml_backend_buffer_get_alignment(descriptor.buffer));
        CHECK(descriptor.encoding == GGML_CUDA_MOE_CANDIDATE_ENCODING_PLAIN);
        CHECK(descriptor.movement == GGML_CUDA_MOE_CANDIDATE_MOVEMENT_SLOT_BOUND);
        CHECK(descriptor.index_modes == (GGML_CUDA_MOE_CANDIDATE_INDEX_GROUP_SLOT_DIRECT | GGML_CUDA_MOE_CANDIDATE_INDEX_ORIGINAL_SOURCE_MAP));
        for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
            CHECK(descriptor.ne[dim] == descriptor.tensor->ne[dim] && descriptor.nb[dim] == descriptor.tensor->nb[dim]);
        }
        if (descriptor.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT) {
            CHECK(descriptor.tensor == gate_up && descriptor.type == GGML_TYPE_BF16);
            found_gate_up = true;
        } else if (descriptor.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT) {
            CHECK(descriptor.tensor == down && descriptor.type == GGML_TYPE_BF16);
            found_down = true;
        } else {
            CHECK(false);
        }
    }
    CHECK(found_gate_up && found_down);

    auto wrong_transaction = first_transaction;
    ++wrong_transaction.acquisition.resource_generation;
    CHECK(!first->end_group_transaction(wrong_transaction));
    const uint64_t logical_signature = first->state().logical_signature;
    CHECK(first->end_group_transaction(first_transaction));
    CHECK(first->begin_group_transaction(first_acquisition, &repeated_transaction));
    CHECK(repeated_transaction.transaction_token > first_transaction.transaction_token);
    CHECK(!first->end_group_transaction(first_transaction));
    CHECK(!first->get_group_resource_bank(first_transaction, 0, nullptr));
    CHECK(first->get_group_resource_bank(repeated_transaction, 0, nullptr));
    CHECK(first->end_group_transaction(repeated_transaction));
    CHECK(!first->end_group_transaction(repeated_transaction));

    ggml_cuda_moe_grouped_transaction held_transaction;
    CHECK(first->begin_group_transaction(first_acquisition, &held_transaction));
    auto * first_context = first.get();
    std::atomic<bool> replacement_started{false};
    std::atomic<bool> replacement_done{false};
    std::atomic<int32_t> replacement_result{-1};
    std::thread replacement_thread([&]() {
        replacement_started.store(true, std::memory_order_release);
        replacement_result.store(first_context->replace(&snapshot), std::memory_order_release);
        replacement_done.store(true, std::memory_order_release);
    });
    while (!replacement_started.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    for (;;) {
        ggml_cuda_moe_grouped_transaction peer_transaction;
        if (!first->begin_group_transaction(peer_acquisition, &peer_transaction)) {
            break;
        }
        CHECK(first->end_group_transaction(peer_transaction));
        std::this_thread::yield();
    }
    CHECK(!replacement_done.load(std::memory_order_acquire));
    CHECK(first->state().generation == 1 && first->state().n_slots == 12);
    CHECK(first->get_group_resources(first_acquisition, &resource_info) && resource_info.transaction_active == 1);
    CHECK(!first->acquire_group_resources(peer_key, &repeated_acquisition));
    CHECK(repeated_acquisition.resource_generation == 0);
    CHECK(first->get_group_resource_bank(held_transaction, 0, nullptr));
    CHECK(first->end_group_transaction(held_transaction));
    replacement_thread.join();
    CHECK(replacement_done.load(std::memory_order_acquire));
    CHECK(replacement_result.load(std::memory_order_acquire) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);

    CHECK(first->state().generation == 2 && first->state().n_slots == 48);
    CHECK(first->state().logical_signature == logical_signature);
    CHECK(!first->get_group_resources(first_acquisition, nullptr));
    CHECK(!first->get_group_resources(peer_acquisition, nullptr));
    CHECK(!first->end_group_transaction(held_transaction));
    CHECK(!first->get_group_resource_bank(held_transaction, 0, nullptr));
    CHECK(second.get_group_resources(second_acquisition, &resource_info));
    CHECK(resource_info.n_slots == 48 && resource_info.acquisition.candidate.generation == 1);

    ggml_cuda_moe_candidate_group_key replacement_key;
    ggml_cuda_moe_grouped_acquisition replacement_acquisition;
    CHECK(first->find_down_group_key(down, &replacement_key));
    CHECK(first->acquire_group_resources(replacement_key, &replacement_acquisition));
    CHECK(replacement_acquisition.resource_generation == 3);
    auto disabled = candidate_snapshot(12, nullptr, 0);
    CHECK(first->replace(&disabled) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(first->state().generation == 3 && first->state().accepted == 1 && first->state().n_groups == 0);
    CHECK(!first->get_group_resources(replacement_acquisition, nullptr));
    CHECK(!first->acquire_group_resources(replacement_key, &repeated_acquisition));

    snapshot.n_slots = 12;
    CHECK(first->replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(first->state().generation == 4 && first->state().logical_signature == logical_signature);
    CHECK(first->find_down_group_key(down, &replacement_key));
    CHECK(first->acquire_group_resources(replacement_key, &replacement_acquisition));
    CHECK(replacement_acquisition.resource_generation == 4);
    ggml_cuda_moe_grouped_transaction shutdown_transaction;
    CHECK(first->begin_group_transaction(replacement_acquisition, &shutdown_transaction));
    std::atomic<bool> shutdown_started{false};
    std::atomic<bool> shutdown_done{false};
    std::thread shutdown_thread([&]() {
        shutdown_started.store(true, std::memory_order_release);
        first_context->shutdown();
        shutdown_done.store(true, std::memory_order_release);
    });
    while (!shutdown_started.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    while (first->get_group_resources(replacement_acquisition, nullptr)) {
        std::this_thread::yield();
    }
    CHECK(!shutdown_done.load(std::memory_order_acquire));
    CHECK(first->get_group_resource_bank(shutdown_transaction, 0, nullptr));
    CHECK(first->end_group_transaction(shutdown_transaction));
    shutdown_thread.join();
    CHECK(shutdown_done.load(std::memory_order_acquire));
    first.reset();
    CHECK(second.find_down_group_key(down, nullptr));
    CHECK(second.get_group_resources(second_acquisition, &resource_info));
    ggml_cuda_moe_grouped_transaction second_transaction;
    CHECK(second.begin_group_transaction(second_acquisition, &second_transaction));
    CHECK(second.end_group_transaction(second_transaction));

    fprintf(stderr, "test-moe-cache: grouped context resources OK\n");
}

static void test_grouped_graph_preflight(bool benchmark) {
    CHECK(sizeof(ggml_cuda_moe_graph_plan) <= 128 * 1024);
    CHECK(ggml_cuda_moe_grouped_context_test_access::graph_reader_witness_size() <= 640);
    CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_record_size() <= 4096);
    CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_observation_size() <= 4096);
    fprintf(stderr, "test-moe-cache: graph witness sizes plan=%zu reader=%zu group=%zu observation=%zu\n",
        sizeof(ggml_cuda_moe_graph_plan),
        ggml_cuda_moe_grouped_context_test_access::graph_reader_witness_size(),
        ggml_cuda_moe_grouped_context_test_access::graph_group_record_size(),
        ggml_cuda_moe_grouped_context_test_access::graph_group_observation_size());

    candidate_test_fixture fixture;
    const int64_t gate_ne[] = {256, 256, 4};
    const int64_t fused_ne[] = {256, 512, 4};
    const int64_t ids_ne[] = {2, 1};
    const int64_t fused_bias_ne[] = {512, 4};

    ggml_tensor * fused_gate_up = fixture.tensor(GGML_TYPE_Q4_0, 3, fused_ne);
    ggml_tensor * fused_down = fixture.tensor(GGML_TYPE_Q4_0, 3, gate_ne);
    ggml_tensor * separate_gate = fixture.tensor(GGML_TYPE_Q4_K, 3, gate_ne);
    ggml_tensor * separate_up = fixture.tensor(GGML_TYPE_Q4_K, 3, gate_ne);
    ggml_tensor * separate_down = fixture.tensor(GGML_TYPE_Q4_K, 3, gate_ne);
    ggml_tensor * fused_bias = fixture.tensor(GGML_TYPE_F32, 2, fused_bias_ne);
    ggml_tensor * external_ids = fixture.tensor(GGML_TYPE_I32, 2, ids_ne);
    const candidate_route fused_route = candidate_top_k_route(fixture, 4, 2);
    const candidate_route fused_route_other = candidate_top_k_route(fixture, 4, 2);
    const candidate_route separate_route = candidate_top_k_route(fixture, 4, 2);

    std::array<ggml_backend_moe_candidate_bank_v1, 2> fused_banks = {{
        {fused_gate_up, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, 0},
        {fused_down, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
    }};
    std::array<ggml_backend_moe_candidate_bank_v1, 3> separate_banks = {{
        {separate_gate, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT, 0},
        {separate_up, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT, 0},
        {separate_down, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
    }};
    std::array<ggml_backend_moe_candidate_group_v1, 2> groups = {{
        {fused_banks.data(), fused_banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 0, 0},
        {separate_banks.data(), separate_banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 0, 0},
    }};
    auto snapshot = candidate_snapshot(12, groups.data(), groups.size());
    ggml_cuda_moe_grouped_context registry(&fixture.owner, 0);
    CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);

    ggml_tensor * fused_gate_up_node = candidate_mmid(fixture, fused_gate_up, fused_route.ids);
    ggml_tensor * fused_down_node = candidate_mmid(fixture, fused_down, fused_route.ids);
    ggml_tensor * separate_up_node = candidate_mmid(fixture, separate_up, separate_route.ids);
    ggml_tensor * separate_gate_node = candidate_mmid(fixture, separate_gate, separate_route.ids);
    ggml_tensor * separate_down_node = candidate_mmid(fixture, separate_down, separate_route.ids);
    ggml_cgraph * complete_graph = candidate_graph(fixture, {
        fused_route.root, fused_route.ids, separate_route.root, separate_route.ids,
        fused_gate_up_node, fused_down_node, separate_up_node, separate_gate_node, separate_down_node,
    });

    candidate_test_graph_views(fixture, registry, fused_route, separate_route,
        fused_gate_up_node, fused_down_node, separate_gate_node, separate_up_node, separate_down_node);
    candidate_test_graph_holder_coverage(fixture, snapshot, fused_route, separate_route,
        fused_gate_up_node, fused_down_node, separate_gate_node, separate_up_node, separate_down_node);

    ggml_cuda_moe_graph_plan plan;
    ggml_cuda_moe_graph_execution execution;
    auto reused = std::make_unique<ggml_cuda_moe_graph_execution>();
    registry.compile_graph_plan(complete_graph, 41, &plan, &execution);
    CHECK(plan.size() == 2 && execution.size() == 2);
    CHECK(!execution.has_stream_grouped_candidate());
    CHECK(!execution.has_coherent_grouped_streams());
    CHECK(execution.resolve_streams(candidate_test_graph_stream, reinterpret_cast<void *>(uintptr_t{1})));
    CHECK(execution.has_stream_grouped_candidate());
    CHECK(execution.has_coherent_grouped_streams());
    CHECK(plan.registry_generation() == 1 && plan.graph_uid() == 41 && plan.graph_node_count() == 9);
    ggml_cuda_moe_graph_binding binding;
    CHECK(execution.find(fused_gate_up_node, &binding));
    CHECK(binding.key.candidate.generation == 1 && binding.key.candidate.group_index == 0);
    CHECK(binding.key.ids.tensor == fused_route.ids && binding.key.ids.data == fused_route.ids->data && binding.key.ids.buffer == fused_route.ids->buffer);
    CHECK(binding.key.layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP && binding.key.n_banks == 2);
    CHECK(binding.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT && binding.bank_index == 0 && binding.slot_index == 0);
    CHECK(execution.find(fused_down_node, &binding) && binding.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT &&
        binding.bank_index == 1 && binding.slot_index == 1);
    CHECK(execution.find(separate_up_node, &binding) && binding.key.candidate.group_index == 1 && binding.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT);
    CHECK(execution.find(separate_gate_node, &binding) && binding.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT);
    CHECK(execution.find(separate_down_node, &binding) && binding.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT && binding.key.n_banks == 3);
    CHECK(!execution.find(fused_bias, nullptr));

    const int64_t route_weight_ne[] = {1, 4, 1};
    ggml_tensor * route_weight_source = fixture.tensor(GGML_TYPE_F32, 3, route_weight_ne);
    ggml_tensor * route_weights = ggml_get_rows(fixture.ctx, route_weight_source, fused_route.ids);
    fixture.materialize(route_weights);
    route_weights->flags |= GGML_TENSOR_FLAG_COMPUTE;
    ggml_tensor * weighted_gate_up = candidate_mmid(fixture, fused_gate_up, fused_route.ids);
    ggml_tensor * weighted_down = candidate_mmid(fixture, fused_down, fused_route.ids);
    ggml_tensor * weighted_output = ggml_mul(fixture.ctx, weighted_down, route_weights);
    fixture.materialize(weighted_output);
    weighted_output->flags |= GGML_TENSOR_FLAG_COMPUTE;
    ggml_cgraph * weighted_graph = candidate_graph(fixture, {
        fused_route.root, fused_route.ids, route_weights, weighted_gate_up, weighted_down, weighted_output,
    });
    registry.compile_graph_plan(weighted_graph, 410, &plan, &execution);
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED && execution.size() == 1);
    CHECK(execution.find(weighted_gate_up, nullptr) && execution.find(weighted_down, nullptr));
    CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_eligible_reason(plan, 0));

    {
        ggml_cuda_moe_graph_execution stream_execution;
        registry.compile_graph_plan(complete_graph, 41, &plan, &stream_execution);
        candidate_test_graph_streams streams = {
            fused_route.root,
            reinterpret_cast<cudaStream_t>(uintptr_t{2}),
            reinterpret_cast<cudaStream_t>(uintptr_t{1}),
        };
        CHECK(stream_execution.resolve_streams(candidate_test_graph_mixed_stream, &streams));
        CHECK(stream_execution.has_stream_grouped_candidate());
        CHECK(!stream_execution.has_coherent_grouped_streams());
        CHECK(!registry.begin_graph_dispatch(&stream_execution, true));
        CHECK(stream_execution.find_authority(fused_gate_up_node) == nullptr);
        CHECK(stream_execution.find_authority(separate_gate_node) == nullptr);
        CHECK(registry.begin_graph_dispatch(&stream_execution, GGML_CUDA_MOE_GRAPH_DISPATCH_LEGACY));
        const auto * fused_authority = stream_execution.find_authority(fused_gate_up_node);
        const auto * separate_authority = stream_execution.find_authority(separate_gate_node);
        const auto * fused_dispatch = stream_execution.find_group(fused_gate_up_node, nullptr);
        CHECK(fused_authority != nullptr && fused_authority->authority() == GGML_CUDA_MOE_GROUP_AUTHORITY_LEGACY);
        CHECK(fused_dispatch != nullptr && fused_dispatch->state == GGML_CUDA_MOE_GRAPH_GROUP_WHOLE_LEGACY &&
            fused_dispatch->transaction.transaction_token == 0);
        CHECK(separate_authority != nullptr && separate_authority->authority() == GGML_CUDA_MOE_GROUP_AUTHORITY_LEGACY);
        CHECK(registry.finish_graph_dispatch(&stream_execution));
    }

    CHECK(registry.bind_graph_plan(complete_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, plan, reused.get()));
    CHECK(reused->size() == 2 && reused->find(separate_down_node, nullptr));
    CHECK(!registry.bind_graph_plan(complete_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, plan, reused.get()) && reused->size() == 0);
    CHECK(registry.bind_graph_plan(complete_graph, 42, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, plan, reused.get()) && reused->size() == 2);
    CHECK(!registry.bind_graph_plan(complete_graph, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, plan, reused.get()) && reused->size() == 0);
    ggml_cgraph * reordered_graph = candidate_graph(fixture, {
        fused_route.root, fused_route.ids, separate_route.root, separate_route.ids,
        fused_gate_up_node, fused_down_node, separate_gate_node, separate_up_node, separate_down_node,
    });
    CHECK(execution.find(fused_gate_up_node, &binding) && binding.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT);
    CHECK(execution.find(fused_down_node, &binding) && binding.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT);
    CHECK(!registry.bind_graph_plan(reordered_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, plan, reused.get()) && reused->size() == 0);
    CHECK(registry.bind_graph_plan(complete_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, plan, reused.get()));
    ggml_tensor * stale_gate_up_node = candidate_mmid(fixture, fused_gate_up, fused_route.ids);
    ggml_cgraph * stale_graph = candidate_graph(fixture, {
        fused_route.root, fused_route.ids, separate_route.root, separate_route.ids,
        stale_gate_up_node, fused_down_node, separate_up_node, separate_gate_node, separate_down_node,
    });
    CHECK(!registry.bind_graph_plan(stale_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, plan, reused.get()) && reused->size() == 0);
    CHECK(registry.bind_graph_plan(complete_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, plan, reused.get()));

    std::shared_ptr<ggml_cuda_moe_graph_plan> cached_plan;
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(registry.prepare_graph_execution(complete_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &cached_plan, prepared.get()) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(cached_plan != nullptr && prepared->size() == 2);
    }
    const ggml_cuda_moe_graph_plan * first_cached_plan = cached_plan.get();
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(registry.prepare_graph_execution(complete_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, &cached_plan, prepared.get()) == GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(cached_plan.get() == first_cached_plan && prepared->find(separate_down_node, nullptr));
    }
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(registry.prepare_graph_execution(complete_graph, 42, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &cached_plan, prepared.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(cached_plan.get() != first_cached_plan && prepared->find(fused_down_node, nullptr));
    }
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(registry.prepare_graph_execution(complete_graph, 42, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &cached_plan, prepared.get()) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(prepared->size() == 2);
    }
    {
        std::weak_ptr<ggml_cuda_moe_graph_plan> lifetime;
        {
            std::shared_ptr<ggml_cuda_moe_graph_plan> lifetime_plan;
            auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
            CHECK(registry.prepare_graph_execution(complete_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &lifetime_plan, prepared.get()) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
            lifetime = lifetime_plan;
            lifetime_plan.reset();
            CHECK(!lifetime.expired() && prepared->find(fused_down_node, nullptr));
        }
        CHECK(lifetime.expired());
    }
    {
        const ggml_cuda_moe_graph_plan * current_plan = cached_plan.get();
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(registry.prepare_graph_execution(reordered_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, &cached_plan, prepared.get()) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(cached_plan.get() != current_plan && prepared->find(fused_gate_up_node, nullptr));
    }
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(registry.prepare_graph_execution(complete_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, &cached_plan, prepared.get()) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(prepared->size() == 2);
    }

    const candidate_route transition_fused_route = candidate_top_k_route(fixture, 4, 2, 4);
    const candidate_route transition_separate_route = candidate_top_k_route(fixture, 4, 2, 4);
    ggml_tensor * transition_fused_gate_up = candidate_mmid(fixture, fused_gate_up, transition_fused_route.ids);
    ggml_tensor * transition_fused_down = candidate_mmid(fixture, fused_down, transition_fused_route.ids);
    ggml_tensor * transition_separate_gate = candidate_mmid(fixture, separate_gate, transition_separate_route.ids);
    ggml_tensor * transition_separate_up = candidate_mmid(fixture, separate_up, transition_separate_route.ids);
    ggml_tensor * transition_separate_down = candidate_mmid(fixture, separate_down, transition_separate_route.ids);
    ggml_cgraph * transition_graph = candidate_graph(fixture, {
        transition_fused_route.root, transition_fused_route.ids, transition_separate_route.root, transition_separate_route.ids,
        transition_fused_gate_up, transition_fused_down, transition_separate_gate, transition_separate_up, transition_separate_down,
    });
    const auto transition_coverage = candidate_certify_graph(registry, transition_graph);
    std::shared_ptr<ggml_cuda_moe_graph_plan> transition_plan;
    const auto prepare_transition = [&](uint64_t uid, ggml_cuda_moe_graph_property_hint hint, ggml_cuda_moe_graph_execution * prepared) {
        return registry.prepare_graph_execution(
            transition_graph, uid, hint, &transition_plan, prepared,
            transition_coverage.epoch, transition_coverage.nodes,
            transition_coverage.mmid_count, transition_coverage.mmid_fingerprint);
    };
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(prepare_transition(200, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, prepared.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(prepared->size() == 2 && !prepared->find(transition_fused_gate_up, nullptr) && !prepared->find(transition_separate_gate, nullptr));
    }
    const std::shared_ptr<ggml_cuda_moe_graph_plan> warmup_plan = transition_plan;
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(prepare_transition(2001, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, prepared.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(transition_plan.get() == warmup_plan.get() && !prepared->find(transition_fused_gate_up, nullptr));
    }
    candidate_set_route_tokens(transition_fused_route, {transition_fused_gate_up, transition_fused_down}, 1);
    candidate_set_route_tokens(transition_separate_route, {transition_separate_gate, transition_separate_up, transition_separate_down}, 1);
    candidate_stamp_execution(transition_graph, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
        GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 1, 1);
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(prepare_transition(201, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, prepared.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(transition_plan.get() != warmup_plan.get() && prepared->find(transition_fused_gate_up, nullptr) &&
            prepared->find(transition_separate_down, nullptr));
    }
    const ggml_cuda_moe_graph_plan * decode_plan = transition_plan.get();
    for (uint64_t uid = 202; uid < 234; ++uid) {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(prepare_transition(uid, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, prepared.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(transition_plan.get() == decode_plan && prepared->find(transition_fused_down, nullptr));
    }
    candidate_set_route_tokens(transition_fused_route, {transition_fused_gate_up, transition_fused_down}, 4);
    candidate_set_route_tokens(transition_separate_route, {transition_separate_gate, transition_separate_up, transition_separate_down}, 4);
    transition_graph->execution_certificate = {};
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(prepare_transition(234, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, prepared.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(!prepared->find(transition_fused_gate_up, nullptr) && !prepared->find(transition_separate_gate, nullptr));
    }
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(prepare_transition(2341, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, prepared.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(!prepared->find(transition_fused_gate_up, nullptr));
    }
    candidate_set_route_tokens(transition_fused_route, {transition_fused_gate_up, transition_fused_down}, 1);
    candidate_set_route_tokens(transition_separate_route, {transition_separate_gate, transition_separate_up, transition_separate_down}, 1);
    candidate_stamp_execution(transition_graph, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
        GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 1, 1);
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(prepare_transition(235, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, prepared.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(prepared->find(transition_fused_gate_up, nullptr) && prepared->find(transition_separate_down, nullptr));
    }

    ggml_tensor * transition_consumer = ggml_dup(fixture.ctx, transition_fused_gate_up);
    fixture.materialize(transition_consumer);
    transition_consumer->flags |= GGML_TENSOR_FLAG_COMPUTE;
    ggml_cgraph * mutation_graph = candidate_graph(fixture, {
        transition_fused_route.root, transition_fused_route.ids, transition_separate_route.root, transition_separate_route.ids,
        transition_fused_gate_up, transition_consumer, transition_fused_down,
        transition_separate_gate, transition_separate_up, transition_separate_down,
    });
    auto mutation_coverage = candidate_certify_graph(registry, mutation_graph);
    std::shared_ptr<ggml_cuda_moe_graph_plan> mutation_plan;
    const auto prepare_mutation = [&](uint64_t uid, ggml_cuda_moe_graph_property_hint hint, ggml_cuda_moe_graph_execution * prepared) {
        return registry.prepare_graph_execution(
            mutation_graph, uid, hint, &mutation_plan, prepared,
            mutation_coverage.epoch, mutation_coverage.nodes,
            mutation_coverage.mmid_count, mutation_coverage.mmid_fingerprint);
    };
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(prepare_mutation(240, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, prepared.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(prepared->find(transition_fused_gate_up, nullptr) && prepared->find(transition_separate_down, nullptr));
    }
    const ggml_op saved_consumer_op = transition_consumer->op;
    ggml_tensor * saved_consumer_src[3] = {transition_consumer->src[0], transition_consumer->src[1], transition_consumer->src[2]};
    transition_consumer->op = GGML_OP_ADD_ID;
    transition_consumer->src[0] = transition_fused_gate_up;
    transition_consumer->src[1] = fused_bias;
    transition_consumer->src[2] = transition_fused_route.ids;
    candidate_rebuild_graph_uses(mutation_graph);
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(prepare_mutation(241, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, prepared.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(!prepared->find(transition_fused_gate_up, nullptr) && !prepared->find(transition_separate_down, nullptr));
        CHECK(prepared->outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR);
    }
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(prepare_mutation(2411, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, prepared.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(!prepared->find(transition_fused_gate_up, nullptr));
    }
    transition_consumer->op = saved_consumer_op;
    memcpy(transition_consumer->src, saved_consumer_src, sizeof(saved_consumer_src));
    candidate_rebuild_graph_uses(mutation_graph);
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(prepare_mutation(242, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, prepared.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(prepared->find(transition_fused_gate_up, nullptr));
    }
    transition_consumer->op = GGML_OP_MUL_MAT_ID;
    transition_consumer->src[0] = fused_gate_up;
    transition_consumer->src[1] = transition_fused_gate_up->src[1];
    transition_consumer->src[2] = transition_fused_route.ids;
    candidate_rebuild_graph_uses(mutation_graph);
    mutation_coverage = candidate_certify_graph(registry, mutation_graph);
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(prepare_mutation(243, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, prepared.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(!prepared->find(transition_fused_gate_up, nullptr) && !prepared->find(transition_separate_down, nullptr));
        CHECK(prepared->outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR);
    }
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(prepare_mutation(2431, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, prepared.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(!prepared->find(transition_fused_gate_up, nullptr));
    }
    transition_consumer->op = saved_consumer_op;
    memcpy(transition_consumer->src, saved_consumer_src, sizeof(saved_consumer_src));
    candidate_rebuild_graph_uses(mutation_graph);
    mutation_coverage = candidate_certify_graph(registry, mutation_graph);
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(prepare_mutation(244, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, prepared.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(prepared->find(transition_fused_gate_up, nullptr));
    }
    transition_fused_route.ids->view_offs = sizeof(int32_t);
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(prepare_mutation(245, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, prepared.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(!prepared->find(transition_fused_gate_up, nullptr));
    }
    transition_fused_route.ids->view_offs = 0;
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(prepare_mutation(246, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, prepared.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(prepared->find(transition_fused_gate_up, nullptr));
    }
    {
        const ggml_cuda_moe_graph_plan * current_plan = cached_plan.get();
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(registry.prepare_graph_execution(stale_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, &cached_plan, prepared.get()) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(cached_plan.get() != current_plan && prepared->find(stale_gate_up_node, nullptr));
    }
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(registry.prepare_graph_execution(complete_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, &cached_plan, prepared.get()) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(prepared->size() == 2);
    }
    ggml_cuda_moe_grouped_context other_registry(&fixture.owner, 0);
    CHECK(other_registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(!other_registry.bind_graph_plan(complete_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, plan, reused.get()) && reused->size() == 0);
    {
        std::shared_ptr<ggml_cuda_moe_graph_plan> foreign_plan = cached_plan;
        const ggml_cuda_moe_graph_plan * original_owner_plan = foreign_plan.get();
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(other_registry.prepare_graph_execution(complete_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, &foreign_plan, prepared.get()) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(foreign_plan.get() != original_owner_plan && prepared->size() == 2);
    }

    fused_gate_up_node->src[0] = separate_gate;
    CHECK(!registry.bind_graph_plan(complete_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, plan, reused.get()) && reused->size() == 0);
    fused_gate_up_node->src[0] = fused_gate_up;
    CHECK(registry.bind_graph_plan(complete_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, plan, reused.get()));

    CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(!registry.bind_graph_plan(complete_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, plan, reused.get()) && reused->size() == 0);
    const ggml_cuda_moe_graph_plan * generation_one_plan = cached_plan.get();
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(registry.prepare_graph_execution(complete_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, &cached_plan, prepared.get()) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(cached_plan.get() != generation_one_plan && cached_plan->registry_generation() == 2 && prepared->size() == 2);
    }
    registry.compile_graph_plan(complete_graph, 43, &plan, &execution);
    CHECK(plan.registry_generation() == 2 && execution.size() == 2);

    ggml_cuda_moe_candidate_group_key held_key;
    ggml_cuda_moe_grouped_acquisition held_acquisition;
    ggml_cuda_moe_grouped_transaction held_transaction;
    CHECK(registry.find_down_group_key(fused_down, &held_key));
    CHECK(registry.acquire_group_resources(held_key, &held_acquisition));
    CHECK(registry.begin_group_transaction(held_acquisition, &held_transaction));
    std::atomic<bool> replacement_started{false};
    std::atomic<bool> replacement_done{false};
    std::thread replacement_thread([&]() {
        replacement_started.store(true, std::memory_order_release);
        CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        replacement_done.store(true, std::memory_order_release);
    });
    while (!replacement_started.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    do {
        std::this_thread::yield();
    } while (registry.bind_graph_plan(complete_graph, 43, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, plan, reused.get()));
    CHECK(reused->size() == 0 && !replacement_done.load(std::memory_order_acquire));
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(registry.prepare_graph_execution(complete_graph, 43, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, &cached_plan, prepared.get()) == GGML_CUDA_MOE_GRAPH_PREPARE_UNAVAILABLE);
        CHECK(cached_plan == nullptr && prepared->size() == 0);
    }
    registry.compile_graph_plan(complete_graph, 43, &plan, &execution);
    CHECK(plan.size() == 0 && execution.size() == 0 && plan.registry_generation() == 0);
    CHECK(registry.end_group_transaction(held_transaction));
    replacement_thread.join();
    CHECK(replacement_done.load(std::memory_order_acquire));
    CHECK(!registry.bind_graph_plan(complete_graph, 43, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, plan, reused.get()) && reused->size() == 0);

    registry.compile_graph_plan(complete_graph, 44, &plan, &execution);
    CHECK(execution.size() == 2);
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(registry.prepare_graph_execution(complete_graph, 44, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &cached_plan, prepared.get()) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(cached_plan != nullptr && prepared->size() == 2);
    }
    auto disabled = candidate_snapshot(12, nullptr, 0);
    CHECK(registry.replace(&disabled) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    registry.compile_graph_plan(complete_graph, 44, &plan, &execution);
    CHECK(plan.size() == 0 && execution.size() == 0 && plan.registry_generation() != 0);
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(registry.prepare_graph_execution(complete_graph, 44, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, &cached_plan, prepared.get()) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(cached_plan != nullptr && prepared->size() == 0);
        const ggml_cuda_moe_graph_plan * disabled_plan = cached_plan.get();
        CHECK(registry.prepare_graph_execution(complete_graph, 44, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, &cached_plan, prepared.get()) == GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(cached_plan.get() == disabled_plan && prepared->size() == 0);
    }
    CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);

    const candidate_fused_top_k_route evaluator_route = candidate_fused_top_k(fixture, 4, 2);
    ggml_tensor * evaluator_gate_up_node = candidate_mmid(fixture, fused_gate_up, evaluator_route.route.ids);
    ggml_tensor * evaluator_down_node = candidate_mmid(fixture, fused_down, evaluator_route.route.ids);
    ggml_cgraph * evaluator_graph = candidate_graph(fixture, {
        evaluator_route.softmax, evaluator_route.reshaped, evaluator_route.route.root, evaluator_route.route.ids,
        evaluator_route.weights, evaluator_gate_up_node, evaluator_down_node,
    });
    registry.compile_graph_plan(evaluator_graph, 45, &plan, &execution);
    CHECK(plan.size() == 1 && execution.size() == 1);

    ggml_tensor * external_gate_up_node = candidate_mmid(fixture, fused_gate_up, external_ids);
    ggml_tensor * external_down_node = candidate_mmid(fixture, fused_down, external_ids);
    ggml_cgraph * external_graph = candidate_graph(fixture, {external_gate_up_node, external_down_node});
    registry.compile_graph_plan(external_graph, 46, &plan, &execution);
    CHECK(plan.size() == 1 && execution.size() == 1 && !execution.find(external_gate_up_node, nullptr));

    ggml_tensor * copied_ids = fixture.tensor(GGML_TYPE_I32, 2, ids_ne);
    copied_ids->op = GGML_OP_CPY;
    copied_ids->src[0] = external_ids;
    copied_ids->flags |= GGML_TENSOR_FLAG_COMPUTE;
    ggml_tensor * copied_gate_up_node = candidate_mmid(fixture, fused_gate_up, copied_ids);
    ggml_tensor * copied_down_node = candidate_mmid(fixture, fused_down, copied_ids);
    ggml_cgraph * copied_graph = candidate_graph(fixture, {copied_ids, copied_gate_up_node, copied_down_node});
    registry.compile_graph_plan(copied_graph, 47, &plan, &execution);
    CHECK(plan.size() == 1 && execution.size() == 1 && !execution.find(copied_gate_up_node, nullptr));

    const candidate_route offset_route = candidate_top_k_route(fixture, 4, 2, 1, sizeof(int32_t));
    ggml_tensor * offset_gate_up_node = candidate_mmid(fixture, fused_gate_up, offset_route.ids);
    ggml_tensor * offset_down_node = candidate_mmid(fixture, fused_down, offset_route.ids);
    ggml_cgraph * offset_graph = candidate_graph(fixture, {offset_route.root, offset_route.ids, offset_gate_up_node, offset_down_node});
    registry.compile_graph_plan(offset_graph, 48, &plan, &execution);
    CHECK(plan.size() == 1 && execution.size() == 1 && !execution.find(offset_gate_up_node, nullptr));

    ggml_tensor * transformed_ids = ggml_reshape_2d(fixture.ctx, fused_route_other.ids, 2, 1);
    fixture.materialize(transformed_ids);
    transformed_ids->flags |= GGML_TENSOR_FLAG_COMPUTE;
    ggml_tensor * transformed_gate_up_node = candidate_mmid(fixture, fused_gate_up, transformed_ids);
    ggml_tensor * transformed_down_node = candidate_mmid(fixture, fused_down, transformed_ids);
    ggml_cgraph * transformed_graph = candidate_graph(fixture, {
        fused_route_other.root, fused_route_other.ids, transformed_ids, transformed_gate_up_node, transformed_down_node,
    });
    registry.compile_graph_plan(transformed_graph, 49, &plan, &execution);
    CHECK(plan.size() == 1 && execution.size() == 1 && !execution.find(transformed_gate_up_node, nullptr));

    const candidate_route wrong_axis_route = candidate_top_k_route(fixture, 5, 2);
    ggml_tensor * wrong_axis_gate_up_node = candidate_mmid(fixture, fused_gate_up, wrong_axis_route.ids);
    ggml_tensor * wrong_axis_down_node = candidate_mmid(fixture, fused_down, wrong_axis_route.ids);
    ggml_cgraph * wrong_axis_graph = candidate_graph(fixture, {
        wrong_axis_route.root, wrong_axis_route.ids, wrong_axis_gate_up_node, wrong_axis_down_node,
    });
    registry.compile_graph_plan(wrong_axis_graph, 50, &plan, &execution);
    CHECK(plan.size() == 1 && execution.size() == 1 && !execution.find(wrong_axis_gate_up_node, nullptr));

    ggml_tensor * order_gate_up_node = candidate_mmid(fixture, fused_gate_up, fused_route_other.ids);
    ggml_tensor * order_down_node = candidate_mmid(fixture, fused_down, fused_route_other.ids);
    ggml_cgraph * producer_order_graph = candidate_graph(fixture, {
        fused_route_other.ids, fused_route_other.root, order_gate_up_node, order_down_node,
    });
    registry.compile_graph_plan(producer_order_graph, 51, &plan, &execution);
    CHECK(plan.size() == 1 && execution.size() == 1 && !execution.find(order_gate_up_node, nullptr));

    const int64_t padding_ne[] = {1};
    ggml_tensor * padding_input = fixture.tensor(GGML_TYPE_F32, 1, padding_ne);
    ggml_tensor * padding_node = ggml_dup(fixture.ctx, padding_input);
    fixture.materialize(padding_node);
    padding_node->flags |= GGML_TENSOR_FLAG_COMPUTE;
    const candidate_route padded_route = candidate_top_k_route(fixture, 4, 2);
    ggml_tensor * padded_gate_up_node = candidate_mmid(fixture, fused_gate_up, padded_route.ids);
    ggml_tensor * padded_down_node = candidate_mmid(fixture, fused_down, padded_route.ids);
    constexpr uint32_t n_padding_nodes = 32768;
    ggml_cgraph * padded_graph = candidate_padded_graph(fixture, padding_node, n_padding_nodes, {
        padded_route.root, padded_route.ids, separate_route.root, separate_route.ids,
        padded_gate_up_node, padded_down_node, separate_up_node, separate_gate_node, separate_down_node,
    });
    const auto padded_coverage = candidate_certify_graph(registry, padded_graph);
    ggml_cuda_moe_graph_plan padded_plan;
    ggml_cuda_moe_graph_execution padded_execution;
    registry.compile_graph_plan(
        padded_graph, 100, &padded_plan, &padded_execution,
        padded_coverage.epoch, padded_coverage.nodes, padded_coverage.mmid_count, padded_coverage.mmid_fingerprint);
    CHECK(padded_plan.size() == 2 && padded_execution.size() == 2);
    CHECK(padded_plan.graph_node_count() == static_cast<int32_t>(n_padding_nodes + 9));
    CHECK(registry.bind_graph_plan(
        padded_graph, 101, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, padded_plan, &padded_execution,
        padded_coverage.epoch, padded_coverage.nodes, padded_coverage.mmid_count, padded_coverage.mmid_fingerprint) &&
        padded_execution.size() == 2);

    const auto complete_coverage = candidate_certify_graph(registry, complete_graph);
    registry.compile_graph_plan(
        complete_graph, 52, &plan, &execution,
        complete_coverage.epoch, complete_coverage.nodes,
        complete_coverage.mmid_count, complete_coverage.mmid_fingerprint);
    CHECK(plan.size() == 2 && execution.size() == 2);
    const auto bind_complete_unknown = [&]() {
        return registry.bind_graph_plan(
            complete_graph, 52, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, plan, reused.get(),
            complete_coverage.epoch, complete_coverage.nodes,
            complete_coverage.mmid_count, complete_coverage.mmid_fingerprint);
    };
    void * saved_ids_data = fused_route.ids->data;
    fused_route.ids->data = fused_route_other.ids->data;
    CHECK(!bind_complete_unknown() && reused->size() == 0);
    fused_route.ids->data = saved_ids_data;
    CHECK(bind_complete_unknown());
    ggml_backend_buffer_t saved_ids_buffer = fused_route.ids->buffer;
    fused_route.ids->buffer = nullptr;
    CHECK(!bind_complete_unknown() && reused->size() == 0);
    fused_route.ids->buffer = saved_ids_buffer;
    CHECK(bind_complete_unknown());
    const size_t saved_ids_stride = fused_route.ids->nb[0];
    fused_route.ids->nb[0] += sizeof(int32_t);
    CHECK(!bind_complete_unknown() && reused->size() == 0);
    fused_route.ids->nb[0] = saved_ids_stride;
    CHECK(bind_complete_unknown());
    const int64_t saved_activation_tokens = fused_gate_up_node->src[1]->ne[2];
    fused_gate_up_node->src[1]->ne[2] = 2;
    CHECK(!bind_complete_unknown() && reused->size() == 0);
    fused_gate_up_node->src[1]->ne[2] = saved_activation_tokens;
    CHECK(bind_complete_unknown());
    const int64_t saved_output_tokens = fused_gate_up_node->ne[2];
    fused_gate_up_node->ne[2] = 2;
    CHECK(!bind_complete_unknown() && reused->size() == 0);
    fused_gate_up_node->ne[2] = saved_output_tokens;
    CHECK(bind_complete_unknown());
    ggml_tensor * saved_weight = fused_gate_up_node->src[0];
    fused_gate_up_node->src[0] = separate_gate;
    CHECK(!bind_complete_unknown() && reused->size() == 0);
    fused_gate_up_node->src[0] = saved_weight;
    CHECK(bind_complete_unknown());
    ggml_tensor * saved_view_src = fused_route.ids->view_src;
    fused_route.ids->view_src = fused_route_other.root;
    CHECK(!bind_complete_unknown() && reused->size() == 0);
    fused_route.ids->view_src = saved_view_src;
    CHECK(bind_complete_unknown());
    ggml_tensor * saved_view_source = fused_route.ids->src[0];
    fused_route.ids->src[0] = fused_route_other.root;
    CHECK(!bind_complete_unknown() && reused->size() == 0);
    fused_route.ids->src[0] = saved_view_source;
    CHECK(bind_complete_unknown());
    fused_route.ids->view_offs = sizeof(int32_t);
    CHECK(!bind_complete_unknown() && reused->size() == 0);
    fused_route.ids->view_offs = 0;
    CHECK(bind_complete_unknown());
    const size_t saved_root_stride = fused_route.root->nb[1];
    fused_route.root->nb[1] += sizeof(int32_t);
    CHECK(!bind_complete_unknown() && reused->size() == 0);
    fused_route.root->nb[1] = saved_root_stride;
    CHECK(bind_complete_unknown());
    ggml_tensor * saved_root_source = fused_route.root->src[0];
    fused_route.root->src[0] = fused_route_other.source;
    CHECK(!bind_complete_unknown() && reused->size() == 0);
    fused_route.root->src[0] = saved_root_source;
    CHECK(bind_complete_unknown());
    const int32_t sort_asc = GGML_SORT_ORDER_ASC;
    memcpy(fused_route.root->op_params, &sort_asc, sizeof(sort_asc));
    CHECK(!bind_complete_unknown() && reused->size() == 0);
    const int32_t sort_desc = GGML_SORT_ORDER_DESC;
    memcpy(fused_route.root->op_params, &sort_desc, sizeof(sort_desc));
    CHECK(bind_complete_unknown());
    ggml_cgraph * producer_reordered_graph = candidate_graph(fixture, {
        fused_route.root, fused_route.ids, separate_route.ids, separate_route.root,
        fused_gate_up_node, fused_down_node, separate_up_node, separate_gate_node, separate_down_node,
    });
    CHECK(!registry.bind_graph_plan(producer_reordered_graph, 52, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, plan, reused.get()) && reused->size() == 0);
    CHECK(bind_complete_unknown());

    ggml_tensor * mixed_gate_up_node = candidate_mmid(fixture, fused_gate_up, fused_route.ids);
    ggml_tensor * mixed_down_node = candidate_mmid(fixture, fused_down, fused_route_other.ids);
    ggml_cgraph * mixed_graph = candidate_graph(fixture, {
        fused_route.root, fused_route.ids, fused_route_other.root, fused_route_other.ids, mixed_gate_up_node, mixed_down_node,
    });
    registry.compile_graph_plan(mixed_graph, 53, &plan, &execution);
    CHECK(plan.size() == 1 && execution.size() == 1 && !execution.find(mixed_gate_up_node, nullptr));

    ggml_tensor * missing_gate_up_node = candidate_mmid(fixture, fused_gate_up, fused_route.ids);
    ggml_cgraph * missing_graph = candidate_graph(fixture, {fused_route.root, fused_route.ids, missing_gate_up_node});
    registry.compile_graph_plan(missing_graph, 54, &plan, &execution);
    CHECK(plan.size() == 1 && execution.size() == 1 && !execution.find(missing_gate_up_node, nullptr));
    {
        std::shared_ptr<ggml_cuda_moe_graph_plan> missing_plan;
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(registry.prepare_graph_execution(missing_graph, 540, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &missing_plan, prepared.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        const std::shared_ptr<ggml_cuda_moe_graph_plan> first_missing_plan = missing_plan;
        CHECK(registry.prepare_graph_execution(missing_graph, 541, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &missing_plan, prepared.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(missing_plan.get() != first_missing_plan.get() && !prepared->find(missing_gate_up_node, nullptr));
    }
    ggml_cgraph * complete_missing_graph = candidate_graph(fixture, {
        fused_route.root, fused_route.ids, separate_route.root, separate_route.ids,
        missing_gate_up_node, separate_up_node, separate_gate_node, separate_down_node,
    });
    const auto complete_missing_coverage = candidate_certify_graph(registry, complete_missing_graph);
    {
        std::shared_ptr<ggml_cuda_moe_graph_plan> missing_plan;
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(registry.prepare_graph_execution(
            complete_missing_graph, 542, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &missing_plan, prepared.get(),
            complete_missing_coverage.epoch, complete_missing_coverage.nodes,
            complete_missing_coverage.mmid_count, complete_missing_coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        const ggml_cuda_moe_graph_plan * stable_missing_plan = missing_plan.get();
        CHECK(!prepared->find(missing_gate_up_node, nullptr) && !prepared->find(separate_down_node, nullptr));
        CHECK(prepared->outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR);
        CHECK(registry.prepare_graph_execution(
            complete_missing_graph, 543, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &missing_plan, prepared.get(),
            complete_missing_coverage.epoch, complete_missing_coverage.nodes,
            complete_missing_coverage.mmid_count, complete_missing_coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(missing_plan.get() == stable_missing_plan && !prepared->find(missing_gate_up_node, nullptr));
    }

    ggml_tensor * duplicate_gate_up_node = candidate_mmid(fixture, fused_gate_up, fused_route.ids);
    ggml_tensor * duplicate_gate_up_peer = candidate_mmid(fixture, fused_gate_up, fused_route.ids);
    ggml_tensor * duplicate_down_node = candidate_mmid(fixture, fused_down, fused_route.ids);
    ggml_cgraph * duplicate_graph = candidate_graph(fixture, {
        fused_route.root, fused_route.ids, duplicate_gate_up_node, duplicate_gate_up_peer, duplicate_down_node,
    });
    registry.compile_graph_plan(duplicate_graph, 55, &plan, &execution);
    CHECK(plan.size() == 1 && execution.size() == 1 && !execution.find(duplicate_gate_up_node, nullptr));

    for (int64_t n_sequences : {2, 4}) {
        const candidate_route multi_sequence_route = candidate_top_k_route(fixture, 4, 2, n_sequences);
        ggml_tensor * multi_sequence_gate_up = candidate_mmid(fixture, fused_gate_up, multi_sequence_route.ids);
        ggml_tensor * multi_sequence_down = candidate_mmid(fixture, fused_down, multi_sequence_route.ids);
        ggml_cgraph * multi_sequence_graph = candidate_graph(fixture, {
            multi_sequence_route.root, multi_sequence_route.ids, multi_sequence_gate_up, multi_sequence_down,
        });
        registry.compile_graph_plan(multi_sequence_graph, 54 + n_sequences, &plan, &execution);
        CHECK(multi_sequence_route.ids->ne[0] == 2 && multi_sequence_route.ids->ne[1] == n_sequences);
        CHECK(plan.size() == 1 && execution.size() == 1 && !execution.find(multi_sequence_gate_up, nullptr));
        CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_PREFILL_LEGACY);
        CHECK(!registry.begin_graph_dispatch(&execution, GGML_CUDA_MOE_GRAPH_DISPATCH_CAPTURE));
        CHECK(registry.begin_graph_dispatch(&execution, GGML_CUDA_MOE_GRAPH_DISPATCH_DIRECT));
        CHECK(execution.dispatch_mode() == GGML_CUDA_MOE_GRAPH_DISPATCH_DIRECT);
        CHECK(registry.finish_graph_dispatch(&execution));
    }

    ggml_tensor * wrong_source_node = candidate_mmid(fixture, separate_gate, fused_route.ids);
    ggml_tensor * correct_down_node = candidate_mmid(fixture, fused_down, fused_route.ids);
    ggml_cgraph * wrong_source_graph = candidate_graph(fixture, {
        fused_route.root, fused_route.ids, wrong_source_node, correct_down_node,
    });
    registry.compile_graph_plan(wrong_source_graph, 57, &plan, &execution);
    CHECK(plan.size() == 2 && execution.size() == 2 && !execution.find(wrong_source_node, nullptr));

    ggml_tensor * auxiliary_gate_up_node = candidate_mmid(fixture, fused_gate_up, fused_route.ids);
    ggml_tensor * auxiliary_down_node = candidate_mmid(fixture, fused_down, fused_route.ids);
    const int64_t auxiliary_ne[] = {auxiliary_gate_up_node->ne[0], auxiliary_gate_up_node->ne[1], auxiliary_gate_up_node->ne[2]};
    ggml_tensor * auxiliary_out = fixture.tensor(GGML_TYPE_F32, 3, auxiliary_ne);
    auxiliary_out->op = GGML_OP_ADD_ID;
    auxiliary_out->src[0] = auxiliary_gate_up_node;
    auxiliary_out->src[1] = fused_bias;
    auxiliary_out->src[2] = fused_route.ids;
    auxiliary_out->flags |= GGML_TENSOR_FLAG_COMPUTE;
    ggml_cgraph * auxiliary_graph = candidate_graph(fixture, {
        fused_route.root, fused_route.ids, auxiliary_gate_up_node, auxiliary_out, auxiliary_down_node,
    });
    registry.compile_graph_plan(auxiliary_graph, 58, &plan, &execution);
    CHECK(plan.size() == 1 && execution.size() == 1 && !execution.find(auxiliary_gate_up_node, nullptr));
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR);
    CHECK(!registry.begin_graph_dispatch(&execution, GGML_CUDA_MOE_GRAPH_DISPATCH_DIRECT));

    std::array<ggml_backend_moe_candidate_bank_v1, 3> auxiliary_banks = {{
        {fused_gate_up, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, 0},
        {fused_down, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
        {fused_bias, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_BIAS, 0},
    }};
    ggml_backend_moe_candidate_group_v1 auxiliary_group = {
        auxiliary_banks.data(), auxiliary_banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 0, 0,
    };
    auto auxiliary_snapshot = candidate_snapshot(12, &auxiliary_group, 1);
    CHECK(registry.replace(&auxiliary_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    ggml_cgraph * auxiliary_registered_graph = candidate_graph(fixture, {
        fused_route.root, fused_route.ids, auxiliary_gate_up_node, auxiliary_down_node,
    });
    registry.compile_graph_plan(auxiliary_registered_graph, 59, &plan, &execution);
    CHECK(plan.size() == 1 && execution.size() == 1 && !execution.find(auxiliary_gate_up_node, nullptr));
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR);
    CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_auxiliary_reason(plan, 0));

    registry.compile_graph_plan(auxiliary_graph, 60, &plan, &execution);
    CHECK(plan.size() == 1 && execution.size() == 1);
    CHECK(execution.find(auxiliary_gate_up_node, nullptr) && execution.find(auxiliary_down_node, nullptr));
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED);
    auxiliary_out->src[2] = fused_route_other.ids;
    candidate_rebuild_graph_uses(auxiliary_graph);
    registry.compile_graph_plan(auxiliary_graph, 61, &plan, &execution);
    CHECK(plan.size() == 1 && execution.size() == 1 && !execution.find(auxiliary_gate_up_node, nullptr));
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR);
    CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_auxiliary_reason(plan, 0));
    auxiliary_out->src[2] = fused_route.ids;
    candidate_rebuild_graph_uses(auxiliary_graph);
    registry.compile_graph_plan(auxiliary_graph, 62, &plan, &execution);
    CHECK(execution.find(auxiliary_gate_up_node, nullptr));

    const int64_t separate_bias_ne[] = {256, 4};
    const int64_t separate_scale_ne[] = {4};
    ggml_tensor * separate_gate_bias = fixture.tensor(GGML_TYPE_F32, 2, separate_bias_ne);
    ggml_tensor * separate_up_bias = fixture.tensor(GGML_TYPE_F32, 2, separate_bias_ne);
    ggml_tensor * separate_down_bias = fixture.tensor(GGML_TYPE_F32, 2, separate_bias_ne);
    ggml_tensor * separate_down_scale = fixture.tensor(GGML_TYPE_F32, 1, separate_scale_ne);
    std::array<ggml_backend_moe_candidate_bank_v1, 6> separate_bias_banks = {{
        {separate_gate, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT, 0},
        {separate_up, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT, 0},
        {separate_down, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
        {separate_gate_bias, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_BIAS, 0},
        {separate_up_bias, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_BIAS, 0},
        {separate_down_bias, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_BIAS, 0},
    }};
    const ggml_backend_moe_candidate_group_v1 separate_bias_group = {
        separate_bias_banks.data(), separate_bias_banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 0, 0,
    };
    const auto separate_bias_snapshot = candidate_snapshot(12, &separate_bias_group, 1);
    CHECK(registry.replace(&separate_bias_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);

    ggml_tensor * biased_gate_reader = candidate_mmid(fixture, separate_gate, separate_route.ids);
    ggml_tensor * biased_up_reader = candidate_mmid(fixture, separate_up, separate_route.ids);
    ggml_tensor * biased_down_reader = candidate_mmid(fixture, separate_down, separate_route.ids);
    ggml_tensor * biased_gate_out = ggml_add_id(fixture.ctx, biased_gate_reader, separate_gate_bias, separate_route.ids);
    ggml_tensor * biased_up_out = ggml_add_id(fixture.ctx, biased_up_reader, separate_up_bias, separate_route.ids);
    ggml_tensor * biased_down_out = ggml_add_id(fixture.ctx, biased_down_reader, separate_down_bias, separate_route.ids);
    ggml_tensor * biased_extra_consumer = ggml_dup(fixture.ctx, biased_gate_out);
    for (ggml_tensor * node : {biased_gate_out, biased_up_out, biased_down_out, biased_extra_consumer}) {
        fixture.materialize(node);
        node->flags |= GGML_TENSOR_FLAG_COMPUTE;
    }
    ggml_cgraph * separate_bias_graph = candidate_graph(fixture, {
        separate_route.root, separate_route.ids,
        biased_gate_reader, biased_gate_out, biased_extra_consumer,
        biased_up_reader, biased_up_out, biased_down_reader, biased_down_out,
    });
    const auto separate_bias_coverage = candidate_certify_graph(registry, separate_bias_graph);
    registry.compile_graph_plan(
        separate_bias_graph, 63, &plan, &execution,
        separate_bias_coverage.epoch, separate_bias_coverage.nodes,
        separate_bias_coverage.mmid_count, separate_bias_coverage.mmid_fingerprint);
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED);
    CHECK(execution.find(biased_gate_reader, nullptr) && execution.find(biased_up_reader, nullptr) &&
        execution.find(biased_down_reader, nullptr));
    ggml_cuda_moe_graph_execution separate_bias_reused;
    const auto bind_separate_bias = [&](uint64_t graph_uid) {
        return registry.bind_graph_plan(
            separate_bias_graph, graph_uid, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, plan, &separate_bias_reused,
            separate_bias_coverage.epoch, separate_bias_coverage.nodes,
            separate_bias_coverage.mmid_count, separate_bias_coverage.mmid_fingerprint);
    };
    CHECK(bind_separate_bias(64));

    biased_gate_out->src[1] = separate_up_bias;
    candidate_rebuild_graph_uses(separate_bias_graph);
    CHECK(!bind_separate_bias(65));
    biased_gate_out->src[1] = separate_gate_bias;
    candidate_rebuild_graph_uses(separate_bias_graph);
    CHECK(bind_separate_bias(66));

    biased_gate_out->src[2] = fused_route_other.ids;
    candidate_rebuild_graph_uses(separate_bias_graph);
    CHECK(!bind_separate_bias(67));
    biased_gate_out->src[2] = separate_route.ids;
    candidate_rebuild_graph_uses(separate_bias_graph);

    biased_gate_out->src[0] = biased_up_reader;
    candidate_rebuild_graph_uses(separate_bias_graph);
    CHECK(!bind_separate_bias(671));
    biased_gate_out->src[0] = biased_gate_reader;
    candidate_rebuild_graph_uses(separate_bias_graph);
    biased_gate_out->src[3] = separate_gate_bias;
    candidate_rebuild_graph_uses(separate_bias_graph);
    CHECK(!bind_separate_bias(672));
    biased_gate_out->src[3] = nullptr;
    candidate_rebuild_graph_uses(separate_bias_graph);
    const ggml_type saved_bias_output_type = biased_gate_out->type;
    biased_gate_out->type = GGML_TYPE_BF16;
    CHECK(!bind_separate_bias(673));
    biased_gate_out->type = saved_bias_output_type;
    const int64_t saved_bias_output_ne = biased_gate_out->ne[0];
    biased_gate_out->ne[0]--;
    CHECK(!bind_separate_bias(674));
    biased_gate_out->ne[0] = saved_bias_output_ne;
    const size_t saved_bias_output_nb = biased_gate_out->nb[1];
    biased_gate_out->nb[1] += sizeof(float);
    CHECK(!bind_separate_bias(675));
    biased_gate_out->nb[1] = saved_bias_output_nb;

    const ggml_type saved_bias_type = separate_gate_bias->type;
    separate_gate_bias->type = GGML_TYPE_BF16;
    CHECK(!bind_separate_bias(68));
    separate_gate_bias->type = saved_bias_type;
    const int64_t saved_bias_ne = separate_gate_bias->ne[0];
    separate_gate_bias->ne[0]--;
    CHECK(!bind_separate_bias(69));
    separate_gate_bias->ne[0] = saved_bias_ne;
    const size_t saved_bias_nb = separate_gate_bias->nb[1];
    separate_gate_bias->nb[1] += sizeof(float);
    CHECK(!bind_separate_bias(70));
    separate_gate_bias->nb[1] = saved_bias_nb;
    void * saved_bias_data = separate_gate_bias->data;
    separate_gate_bias->data = static_cast<char *>(separate_gate_bias->data) + sizeof(float);
    CHECK(!bind_separate_bias(71));
    separate_gate_bias->data = saved_bias_data;

    std::swap(separate_bias_graph->nodes[2], separate_bias_graph->nodes[3]);
    CHECK(!bind_separate_bias(72));
    std::swap(separate_bias_graph->nodes[2], separate_bias_graph->nodes[3]);
    biased_extra_consumer->src[0] = biased_gate_reader;
    candidate_rebuild_graph_uses(separate_bias_graph);
    CHECK(!bind_separate_bias(73));
    ggml_cuda_moe_graph_plan extra_consumer_plan;
    registry.compile_graph_plan(separate_bias_graph, 731, &extra_consumer_plan, &execution);
    CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_auxiliary_reason(extra_consumer_plan, 0));
    biased_extra_consumer->src[0] = biased_gate_out;
    candidate_rebuild_graph_uses(separate_bias_graph);
    CHECK(bind_separate_bias(74));

    separate_bias_banks[3].role = GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_BIAS;
    separate_bias_banks[4].role = GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_BIAS;
    const auto wrong_bias_role_snapshot = candidate_snapshot(12, &separate_bias_group, 1);
    CHECK(registry.replace(&wrong_bias_role_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    registry.compile_graph_plan(separate_bias_graph, 75, &plan, &execution);
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR && !execution.find(biased_gate_reader, nullptr));
    CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_auxiliary_reason(plan, 0));
    separate_bias_banks[3].role = GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_BIAS;
    separate_bias_banks[4].role = GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_BIAS;
    CHECK(registry.replace(&separate_bias_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);

    std::array<ggml_backend_moe_candidate_bank_v1, 7> scale_bias_banks;
    std::copy(separate_bias_banks.begin(), separate_bias_banks.end(), scale_bias_banks.begin());
    scale_bias_banks.back() = {separate_down_scale, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_SCALE, 0};
    const ggml_backend_moe_candidate_group_v1 scale_bias_group = {
        scale_bias_banks.data(), scale_bias_banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 0, 0,
    };
    const auto scale_bias_snapshot = candidate_snapshot(12, &scale_bias_group, 1);
    CHECK(registry.replace(&scale_bias_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    registry.compile_graph_plan(separate_bias_graph, 76, &plan, &execution);
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR && !execution.find(biased_gate_reader, nullptr));
    CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_descriptor_reason(plan, 0));

    CHECK(registry.replace(&separate_bias_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    const candidate_route biased_prefill_route = candidate_top_k_route(fixture, 4, 2, 2);
    ggml_tensor * biased_prefill_gate = candidate_mmid(fixture, separate_gate, biased_prefill_route.ids);
    ggml_tensor * biased_prefill_up = candidate_mmid(fixture, separate_up, biased_prefill_route.ids);
    ggml_tensor * biased_prefill_down = candidate_mmid(fixture, separate_down, biased_prefill_route.ids);
    ggml_tensor * biased_prefill_gate_out = ggml_add_id(
        fixture.ctx, biased_prefill_gate, separate_gate_bias, biased_prefill_route.ids);
    ggml_tensor * biased_prefill_up_out = ggml_add_id(
        fixture.ctx, biased_prefill_up, separate_up_bias, biased_prefill_route.ids);
    ggml_tensor * biased_prefill_down_out = ggml_add_id(
        fixture.ctx, biased_prefill_down, separate_down_bias, biased_prefill_route.ids);
    for (ggml_tensor * node : {biased_prefill_gate_out, biased_prefill_up_out, biased_prefill_down_out}) {
        fixture.materialize(node);
        node->flags |= GGML_TENSOR_FLAG_COMPUTE;
    }
    ggml_cgraph * biased_prefill_graph = candidate_graph(fixture, {
        biased_prefill_route.root, biased_prefill_route.ids,
        biased_prefill_gate, biased_prefill_gate_out,
        biased_prefill_up, biased_prefill_up_out,
        biased_prefill_down, biased_prefill_down_out,
    });
    registry.compile_graph_plan(biased_prefill_graph, 77, &plan, &execution);
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_PREFILL_LEGACY && execution.size() == 1);
    CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_prefill_reason(plan, 0));

    const auto check_biased_execution_unknown_reuse = [&](ggml_cgraph * graph, uint64_t graph_uid) {
        const auto coverage = candidate_certify_graph(registry, graph);
        std::shared_ptr<ggml_cuda_moe_graph_plan> reusable_plan;
        ggml_cuda_moe_graph_execution reusable_execution;
        CHECK(registry.prepare_graph_execution(
            graph, graph_uid, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &reusable_plan, &reusable_execution,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(reusable_execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_LEGACY && reusable_execution.size() == 1);
        CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_execution_reason(*reusable_plan, 0));
        const auto * compiled_plan = reusable_plan.get();
        CHECK(registry.prepare_graph_execution(
            graph, graph_uid + 1, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &reusable_plan, &reusable_execution,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(reusable_plan.get() == compiled_plan &&
            reusable_execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_LEGACY);
    };
    for (uint32_t domain : {GGML_GRAPH_EXECUTION_DOMAIN_DRAFT, GGML_GRAPH_EXECUTION_DOMAIN_MTP}) {
        candidate_stamp_execution(separate_bias_graph, domain,
            GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 1, 1);
        check_biased_execution_unknown_reuse(separate_bias_graph, 780 + domain * 2);
    }

    const candidate_route biased_overslot_route = candidate_top_k_route(fixture, 4, 4, 4);
    ggml_tensor * biased_overslot_gate = candidate_mmid(fixture, separate_gate, biased_overslot_route.ids);
    ggml_tensor * biased_overslot_up = candidate_mmid(fixture, separate_up, biased_overslot_route.ids);
    ggml_tensor * biased_overslot_down = candidate_mmid(fixture, separate_down, biased_overslot_route.ids);
    ggml_tensor * biased_overslot_gate_out = ggml_add_id(
        fixture.ctx, biased_overslot_gate, separate_gate_bias, biased_overslot_route.ids);
    ggml_tensor * biased_overslot_up_out = ggml_add_id(
        fixture.ctx, biased_overslot_up, separate_up_bias, biased_overslot_route.ids);
    ggml_tensor * biased_overslot_down_out = ggml_add_id(
        fixture.ctx, biased_overslot_down, separate_down_bias, biased_overslot_route.ids);
    for (ggml_tensor * node : {biased_overslot_gate_out, biased_overslot_up_out, biased_overslot_down_out}) {
        fixture.materialize(node);
        node->flags |= GGML_TENSOR_FLAG_COMPUTE;
    }
    ggml_cgraph * biased_overslot_graph = candidate_graph(fixture, {
        biased_overslot_route.root, biased_overslot_route.ids,
        biased_overslot_gate, biased_overslot_gate_out,
        biased_overslot_up, biased_overslot_up_out,
        biased_overslot_down, biased_overslot_down_out,
    });
    candidate_stamp_execution(biased_overslot_graph, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
        GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 4, 4);
    check_biased_execution_unknown_reuse(biased_overslot_graph, 790);

    {
        candidate_test_fixture oversized_fixture;
        constexpr int64_t n_experts = 1025;
        const int64_t oversized_gate_up_ne[] = {32, 64, n_experts};
        const int64_t oversized_down_ne[] = {32, 32, n_experts};
        const int64_t oversized_scale_ne[] = {n_experts};
        ggml_tensor * oversized_gate_up = oversized_fixture.tensor(GGML_TYPE_Q4_0, 3, oversized_gate_up_ne);
        ggml_tensor * oversized_down = oversized_fixture.tensor(GGML_TYPE_Q4_0, 3, oversized_down_ne);
        ggml_tensor * oversized_scale = oversized_fixture.tensor(GGML_TYPE_F32, 1, oversized_scale_ne);
        const candidate_route oversized_route = candidate_top_k_route(oversized_fixture, n_experts, 2);
        ggml_tensor * oversized_gate_up_node = candidate_mmid(oversized_fixture, oversized_gate_up, oversized_route.ids);
        ggml_tensor * oversized_down_node = candidate_mmid(oversized_fixture, oversized_down, oversized_route.ids);
        ggml_tensor * scale = ggml_reshape_3d(oversized_fixture.ctx, oversized_scale, 1, n_experts, 1);
        oversized_fixture.materialize(scale);
        scale->flags |= GGML_TENSOR_FLAG_COMPUTE;
        scale = ggml_repeat_4d(oversized_fixture.ctx, scale, 1, n_experts, 1, 1);
        oversized_fixture.materialize(scale);
        scale->flags |= GGML_TENSOR_FLAG_COMPUTE;
        scale = ggml_get_rows(oversized_fixture.ctx, scale, oversized_route.ids);
        oversized_fixture.materialize(scale);
        scale->flags |= GGML_TENSOR_FLAG_COMPUTE;
        ggml_tensor * oversized_output = ggml_mul(oversized_fixture.ctx, oversized_down_node, scale);
        oversized_fixture.materialize(oversized_output);
        oversized_output->flags |= GGML_TENSOR_FLAG_COMPUTE;
        ggml_cgraph * oversized_graph = candidate_graph(oversized_fixture, {
            oversized_route.root, oversized_route.ids, oversized_gate_up_node, oversized_down_node,
            scale->src[0]->src[0], scale->src[0], scale, oversized_output,
        });
        std::array<ggml_backend_moe_candidate_bank_v1, 3> oversized_banks = {{
            {oversized_gate_up, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, 0},
            {oversized_down, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
            {oversized_scale, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_SCALE, 0},
        }};
        const ggml_backend_moe_candidate_group_v1 oversized_group = {
            oversized_banks.data(), oversized_banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 0, 0,
        };
        const auto oversized_snapshot = candidate_snapshot(12, &oversized_group, 1);
        CHECK(registry.replace(&oversized_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        CHECK(registry.state().n_groups == 1 && registry.state().permanent_candidate_bytes == 4100);
        registry.compile_graph_plan(oversized_graph, 591, &plan, &execution);
        CHECK(plan.size() == 1 && execution.size() == 1 && !execution.find(oversized_gate_up_node, nullptr));
        fprintf(stderr, "test-moe-cache: oversized original-direct auxiliary decline OK\n");
    }

    CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    registry.compile_graph_plan(complete_graph, 60, &plan, &execution);
    CHECK(execution.resolve_streams(candidate_test_graph_stream, reinterpret_cast<void *>(uintptr_t{1})));
    CHECK(registry.begin_graph_dispatch(&execution, false));
    const auto * legacy_authority = execution.find_authority(fused_gate_up_node);
    CHECK(legacy_authority != nullptr && legacy_authority->authority() == GGML_CUDA_MOE_GROUP_AUTHORITY_LEGACY);
    auto legacy_before = registry.acquire_legacy_cache(fused_gate_up, nullptr, legacy_authority);
    CHECK(legacy_before);
    const auto legacy_before_state = legacy_before.acquisition();
    legacy_before = {};
    CHECK(registry.finish_graph_dispatch(&execution));

    CHECK(registry.begin_graph_dispatch(&execution, true));
    const auto * grouped_authority = execution.find_authority(fused_gate_up_node);
    CHECK(grouped_authority != nullptr && grouped_authority->authority() == GGML_CUDA_MOE_GROUP_AUTHORITY_GROUPED);
    CHECK(!registry.acquire_legacy_cache(fused_gate_up, nullptr, grouped_authority));
    ggml_cuda_moe_candidate_group_key authority_key;
    ggml_cuda_moe_grouped_acquisition authority_resource;
    CHECK(registry.find_down_group_key(fused_down, &authority_key));
    CHECK(registry.acquire_group_resources(authority_key, &authority_resource));
    CHECK(!registry.finish_graph_dispatch(&execution));

    registry.compile_graph_plan(missing_graph, 61, &plan, &execution);
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR);
    CHECK(!registry.begin_graph_dispatch(&execution, true));
    CHECK(registry.get_group_resources(authority_resource, nullptr));
    CHECK(registry.begin_graph_dispatch(&execution, false));
    CHECK(!registry.get_group_resources(authority_resource, nullptr));
    auto legacy_after = registry.acquire_legacy_cache(fused_gate_up);
    CHECK(legacy_after && legacy_after.acquisition().group_authority_epoch > legacy_before_state.group_authority_epoch);
    legacy_after = {};
    CHECK(registry.finish_graph_dispatch(&execution));

    auto finalization_registry = std::make_unique<ggml_cuda_moe_grouped_context>(&fixture.owner, 0);
    CHECK(finalization_registry->replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    ggml_cuda_moe_graph_plan finalization_plan;
    ggml_cuda_moe_graph_execution finalization_execution;
    finalization_registry->compile_graph_plan(complete_graph, 62, &finalization_plan, &finalization_execution);
    CHECK(finalization_execution.resolve_streams(candidate_test_graph_stream, reinterpret_cast<void *>(uintptr_t{1})));
    CHECK(finalization_registry->begin_graph_dispatch(&finalization_execution, true));
    ggml_cuda_moe_candidate_group_key failed_keys[2];
    ggml_cuda_moe_grouped_acquisition failed_resources[2];
    ggml_cuda_moe_grouped_transaction failed_transactions[2];
    CHECK(finalization_registry->find_down_group_key(fused_down, &failed_keys[0]));
    CHECK(finalization_registry->find_down_group_key(separate_down, &failed_keys[1]));
    for (uint32_t i = 0; i < 2; ++i) {
        CHECK(finalization_registry->acquire_group_resources(failed_keys[i], &failed_resources[i]));
        CHECK(finalization_registry->begin_group_transaction(failed_resources[i], &failed_transactions[i]));
    }
    auto * failed_fused = finalization_execution.find_group(fused_gate_up_node, nullptr);
    auto * failed_separate = finalization_execution.find_group(separate_gate_node, nullptr);
    CHECK(failed_fused != nullptr && failed_separate != nullptr && failed_fused != failed_separate);
    failed_fused->transaction = failed_transactions[0];
    failed_fused->state = GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_ACTIVE;
    failed_separate->transaction = failed_transactions[1];
    failed_separate->state = GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_ACTIVE;
    CHECK(!finalization_registry->finish_graph_dispatch(&finalization_execution));
    for (uint32_t i = 0; i < 2; ++i) {
        CHECK(!finalization_registry->get_group_resources(failed_resources[i], nullptr));
        CHECK(!finalization_registry->end_group_transaction(failed_transactions[i]));
    }
    CHECK(finalization_execution.find_authority(fused_gate_up_node) == nullptr);
    CHECK(finalization_execution.find_authority(separate_gate_node) == nullptr);
    CHECK(finalization_registry->replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    finalization_registry->shutdown();
    finalization_registry.reset();

    registry.compile_graph_plan(complete_graph, 62, &plan, &execution);
    CHECK(execution.resolve_streams(candidate_test_graph_stream, reinterpret_cast<void *>(uintptr_t{1})));
    CHECK(registry.begin_graph_dispatch(&execution, true));
    std::atomic<bool> authority_replacement_started{false};
    std::atomic<bool> authority_replacement_done{false};
    std::thread authority_replacement([&]() {
        authority_replacement_started.store(true, std::memory_order_release);
        CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        authority_replacement_done.store(true, std::memory_order_release);
    });
    while (!authority_replacement_started.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    do {
        std::this_thread::yield();
    } while (registry.bind_graph_plan(complete_graph, 62, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, plan, reused.get()));
    CHECK(!authority_replacement_done.load(std::memory_order_acquire));
    CHECK(!registry.finish_graph_dispatch(&execution));
    authority_replacement.join();
    CHECK(authority_replacement_done.load(std::memory_order_acquire));

    {
        ggml_cuda_moe_graph_plan scoped_plan;
        ggml_cuda_moe_graph_execution scoped_execution;
        registry.compile_graph_plan(complete_graph, 63, &scoped_plan, &scoped_execution);
        CHECK(scoped_execution.resolve_streams(candidate_test_graph_stream, reinterpret_cast<void *>(uintptr_t{1})));
        CHECK(registry.begin_graph_dispatch(&scoped_execution, true));
    }
    CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);

    auto terminal = std::make_unique<ggml_cuda_moe_grouped_context>(&fixture.owner, 0);
    CHECK(terminal->replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    ggml_cuda_moe_graph_plan terminal_plan;
    ggml_cuda_moe_graph_execution terminal_execution;
    ggml_cuda_moe_graph_execution terminal_probe;
    terminal->compile_graph_plan(complete_graph, 64, &terminal_plan, &terminal_execution);
    CHECK(terminal_execution.resolve_streams(candidate_test_graph_stream, reinterpret_cast<void *>(uintptr_t{1})));
    CHECK(terminal->begin_graph_dispatch(&terminal_execution, true));
    std::atomic<bool> authority_shutdown_started{false};
    std::atomic<bool> authority_shutdown_done{false};
    std::thread authority_shutdown([&]() {
        authority_shutdown_started.store(true, std::memory_order_release);
        terminal->shutdown();
        authority_shutdown_done.store(true, std::memory_order_release);
    });
    while (!authority_shutdown_started.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    do {
        std::this_thread::yield();
    } while (terminal->bind_graph_plan(complete_graph, 64, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, terminal_plan, &terminal_probe));
    CHECK(!authority_shutdown_done.load(std::memory_order_acquire));
    CHECK(!terminal->finish_graph_dispatch(&terminal_execution));
    authority_shutdown.join();
    CHECK(authority_shutdown_done.load(std::memory_order_acquire));
    terminal.reset();

    if (benchmark) {
        CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        std::shared_ptr<ggml_cuda_moe_graph_plan> benchmark_plan;
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(registry.prepare_graph_execution(
            complete_graph, 60, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &benchmark_plan, prepared.get(),
            complete_coverage.epoch, complete_coverage.nodes,
            complete_coverage.mmid_count, complete_coverage.mmid_fingerprint) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        const ggml_cuda_moe_graph_plan * stable_plan = benchmark_plan.get();
        constexpr uint32_t n_reuses = 200000;
        uint32_t reused_count = 0;
        const auto begin = std::chrono::steady_clock::now();
        for (uint32_t i = 0; i < n_reuses; ++i) {
            reused_count += registry.prepare_graph_execution(
                complete_graph, 61 + i, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &benchmark_plan, prepared.get(),
                complete_coverage.epoch, complete_coverage.nodes,
                complete_coverage.mmid_count, complete_coverage.mmid_fingerprint) ==
                GGML_CUDA_MOE_GRAPH_PREPARE_REUSED ? 1 : 0;
        }
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin).count();
        CHECK(reused_count == n_reuses && benchmark_plan.get() == stable_plan && prepared->size() == 2);
        fprintf(stderr, "test-moe-cache: grouped graph plan %.1f ns/reuse with fresh UIDs, recompiles=0/%u\n",
            static_cast<double>(elapsed) / n_reuses, n_reuses);

        std::shared_ptr<ggml_cuda_moe_graph_plan> padded_benchmark_plan;
        CHECK(registry.prepare_graph_execution(
            padded_graph, 100, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &padded_benchmark_plan, prepared.get(),
            padded_coverage.epoch, padded_coverage.nodes,
            padded_coverage.mmid_count, padded_coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        const ggml_cuda_moe_graph_plan * stable_padded_plan = padded_benchmark_plan.get();
        constexpr uint32_t n_padded_reuses = 100000;
        uint32_t padded_reused_count = 0;
        const auto padded_begin = std::chrono::steady_clock::now();
        for (uint32_t i = 0; i < n_padded_reuses; ++i) {
            padded_reused_count += registry.prepare_graph_execution(
                padded_graph, 101 + i, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &padded_benchmark_plan, prepared.get(),
                padded_coverage.epoch, padded_coverage.nodes,
                padded_coverage.mmid_count, padded_coverage.mmid_fingerprint) ==
                GGML_CUDA_MOE_GRAPH_PREPARE_REUSED ? 1 : 0;
        }
        const auto padded_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - padded_begin).count();
        const double padded_ns = static_cast<double>(padded_elapsed) / n_padded_reuses;
        CHECK(padded_reused_count == n_padded_reuses && padded_benchmark_plan.get() == stable_padded_plan && prepared->size() == 2);
        fprintf(stderr, "test-moe-cache: grouped padded graph plan %.1f ns/reuse with %u-node inventory scan, recompiles=0/%u\n",
            padded_ns, n_padding_nodes + 9, n_padded_reuses);
        CHECK(padded_ns < 100000.0);
    }

    fprintf(stderr, "test-moe-cache: grouped graph preflight OK\n");
}

static void test_grouped_graph_mixed_phase() {
    candidate_test_fixture fixture;
    const int64_t gate_up_ne[] = {256, 512, 4};
    const int64_t down_ne[] = {256, 256, 4};
    ggml_tensor * prefill_gate_up = fixture.cached_tensor(GGML_TYPE_Q4_0, 3, gate_up_ne);
    ggml_tensor * prefill_down = fixture.cached_tensor(GGML_TYPE_Q4_0, 3, down_ne);
    ggml_tensor * decode_gate_up = fixture.cached_tensor(GGML_TYPE_Q4_0, 3, gate_up_ne);
    ggml_tensor * decode_down = fixture.cached_tensor(GGML_TYPE_Q4_0, 3, down_ne);
    ggml_tensor * late_gate_up = fixture.cached_tensor(GGML_TYPE_Q4_0, 3, gate_up_ne);
    ggml_tensor * late_down = fixture.cached_tensor(GGML_TYPE_Q4_0, 3, down_ne);
    ggml_tensor * uncovered = fixture.cached_tensor(GGML_TYPE_Q4_0, 3, gate_up_ne);
    std::array<ggml_backend_moe_candidate_bank_v1, 2> prefill_banks = {{
        {prefill_gate_up, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, 0},
        {prefill_down, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
    }};
    std::array<ggml_backend_moe_candidate_bank_v1, 2> decode_banks = {{
        {decode_gate_up, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, 0},
        {decode_down, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
    }};
    std::array<ggml_backend_moe_candidate_bank_v1, 2> late_banks = {{
        {late_gate_up, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, 0},
        {late_down, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
    }};
    std::array<ggml_backend_moe_candidate_group_v1, 3> groups = {{
        {prefill_banks.data(), prefill_banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 0, 0},
        {decode_banks.data(), decode_banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 0, 0},
        {late_banks.data(), late_banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 0, 0},
    }};
    const auto snapshot = candidate_snapshot(12, groups.data(), groups.size());
    ggml_cuda_moe_grouped_context registry(&fixture.owner, 0);
    CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);

    const candidate_route prefill_route = candidate_top_k_route(fixture, 4, 2, 4);
    const candidate_route decode_route = candidate_top_k_route(fixture, 4, 2);
    const candidate_route late_route = candidate_top_k_route(fixture, 4, 2);
    ggml_tensor * prefill_gate_up_reader = candidate_mmid(fixture, prefill_gate_up, prefill_route.ids);
    ggml_tensor * prefill_down_reader = candidate_mmid(fixture, prefill_down, prefill_route.ids);
    ggml_tensor * decode_gate_up_reader = candidate_mmid(fixture, decode_gate_up, decode_route.ids);
    ggml_tensor * decode_down_reader = candidate_mmid(fixture, decode_down, decode_route.ids);
    ggml_tensor * late_gate_up_reader = candidate_mmid(fixture, late_gate_up, late_route.ids);
    ggml_tensor * late_down_reader = candidate_mmid(fixture, late_down, late_route.ids);
    late_gate_up_reader->op = GGML_OP_DUP;
    late_down_reader->op = GGML_OP_DUP;
    ggml_tensor * cross_phase_down_reader = candidate_mmid(fixture, prefill_down, decode_route.ids);
    ggml_tensor * uncovered_reader = candidate_mmid(fixture, uncovered, prefill_route.ids);
    ggml_cgraph * mixed_graph = candidate_graph(fixture, {
        prefill_route.root, prefill_route.ids, decode_route.root, decode_route.ids, late_route.root, late_route.ids,
        prefill_gate_up_reader, prefill_down_reader, decode_gate_up_reader, decode_down_reader,
        late_gate_up_reader, late_down_reader,
    });
    const auto mixed_coverage = candidate_certify_graph(registry, mixed_graph);
    CHECK(mixed_coverage.mmid_count == 4);
    auto stale_plan = std::make_shared<ggml_cuda_moe_graph_plan>();
    ggml_cuda_moe_graph_execution execution;
    registry.compile_graph_plan(
        mixed_graph, 701, stale_plan.get(), &execution, mixed_coverage.epoch, mixed_coverage.nodes,
        mixed_coverage.mmid_count, mixed_coverage.mmid_fingerprint);
    CHECK(stale_plan->size() == 2 && execution.size() == 2);
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_PREFILL_LEGACY && execution.requires_dispatch());
    CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_prefill_reason(*stale_plan, 0));
    CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_execution_reason(*stale_plan, 1));
    CHECK(execution.find(prefill_gate_up_reader, nullptr) && execution.find(prefill_down_reader, nullptr));
    CHECK(execution.find(decode_gate_up_reader, nullptr) && execution.find(decode_down_reader, nullptr));
    CHECK(registry.begin_graph_dispatch(&execution, true));
    const auto * prefill_authority = execution.find_authority(prefill_gate_up_reader);
    const auto * decode_authority = execution.find_authority(decode_gate_up_reader);
    CHECK(prefill_authority != nullptr && prefill_authority->authority() == GGML_CUDA_MOE_GROUP_AUTHORITY_LEGACY);
    CHECK(decode_authority != nullptr && decode_authority->authority() == GGML_CUDA_MOE_GROUP_AUTHORITY_LEGACY);
    CHECK(registry.finish_graph_dispatch(&execution));
    ggml_cuda_moe_graph_execution reused;
    CHECK(registry.bind_graph_plan(
        mixed_graph, 702, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, *stale_plan, &reused,
        mixed_coverage.epoch, mixed_coverage.nodes, mixed_coverage.mmid_count, mixed_coverage.mmid_fingerprint));
    CHECK(reused.find(prefill_down_reader, nullptr) && reused.find(decode_down_reader, nullptr));

    late_gate_up_reader->op = GGML_OP_MUL_MAT_ID;
    late_down_reader->op = GGML_OP_MUL_MAT_ID;
    candidate_rebuild_graph_uses(mixed_graph);
    CHECK(!registry.bind_graph_plan(
        mixed_graph, 702, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, *stale_plan, &reused,
        mixed_coverage.epoch, mixed_coverage.nodes, mixed_coverage.mmid_count, mixed_coverage.mmid_fingerprint));
    CHECK(reused.size() == 0 && !reused.find(prefill_down_reader, nullptr) && !reused.find(decode_down_reader, nullptr) &&
        !reused.find(late_gate_up_reader, nullptr) && !reused.find(late_down_reader, nullptr));
    const ggml_cuda_moe_graph_plan * stale_plan_ptr = stale_plan.get();
    CHECK(registry.prepare_graph_execution(
        mixed_graph, 7021, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &stale_plan, &reused,
        mixed_coverage.epoch, mixed_coverage.nodes, mixed_coverage.mmid_count, mixed_coverage.mmid_fingerprint) ==
        GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
    CHECK(stale_plan.get() != stale_plan_ptr && stale_plan->size() == 3 && reused.size() == 3 &&
        stale_plan->coverage_diagnostics().cached_mmid == 6);
    CHECK(reused.find(prefill_gate_up_reader, nullptr) && reused.find(prefill_down_reader, nullptr) &&
        reused.find(decode_gate_up_reader, nullptr) && reused.find(decode_down_reader, nullptr) &&
        reused.find(late_gate_up_reader, nullptr) && reused.find(late_down_reader, nullptr));

    ggml_cuda_moe_graph_plan plan;

    ggml_cgraph * incomplete_prefill_graph = candidate_graph(fixture, {
        prefill_route.root, prefill_route.ids, decode_route.root, decode_route.ids,
        prefill_gate_up_reader, decode_gate_up_reader, decode_down_reader,
    });
    const auto incomplete_prefill_coverage = candidate_certify_graph(registry, incomplete_prefill_graph);
    registry.compile_graph_plan(
        incomplete_prefill_graph, 703, &plan, &execution,
        incomplete_prefill_coverage.epoch, incomplete_prefill_coverage.nodes,
        incomplete_prefill_coverage.mmid_count, incomplete_prefill_coverage.mmid_fingerprint);
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR);
    CHECK(execution.requires_dispatch());
    CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_missing_role_reason(plan, 0));
    CHECK(execution.rejects_cached_mmid(prefill_gate_up_reader) && execution.rejects_cached_mmid(decode_down_reader));

    ggml_cgraph * cross_phase_group_graph = candidate_graph(fixture, {
        prefill_route.root, prefill_route.ids, decode_route.root, decode_route.ids,
        prefill_gate_up_reader, cross_phase_down_reader,
    });
    const auto cross_phase_group_coverage = candidate_certify_graph(registry, cross_phase_group_graph);
    registry.compile_graph_plan(
        cross_phase_group_graph, 704, &plan, &execution,
        cross_phase_group_coverage.epoch, cross_phase_group_coverage.nodes,
        cross_phase_group_coverage.mmid_count, cross_phase_group_coverage.mmid_fingerprint);
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR && execution.requires_dispatch());

    ggml_cgraph * uncovered_graph = candidate_graph(fixture, {
        prefill_route.root, prefill_route.ids, decode_route.root, decode_route.ids,
        prefill_gate_up_reader, prefill_down_reader, decode_gate_up_reader, decode_down_reader, uncovered_reader,
    });
    const auto uncovered_coverage = candidate_certify_graph(registry, uncovered_graph);
    registry.compile_graph_plan(
        uncovered_graph, 705, &plan, &execution, uncovered_coverage.epoch, uncovered_coverage.nodes,
        uncovered_coverage.mmid_count, uncovered_coverage.mmid_fingerprint);
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR && execution.rejects_cached_mmid(uncovered_reader));

    ggml_cgraph * pure_prefill_graph = candidate_graph(fixture, {
        prefill_route.root, prefill_route.ids, prefill_gate_up_reader, prefill_down_reader,
    });
    const auto pure_prefill_coverage = candidate_certify_graph(registry, pure_prefill_graph);
    registry.compile_graph_plan(
        pure_prefill_graph, 706, &plan, &execution, pure_prefill_coverage.epoch, pure_prefill_coverage.nodes,
        pure_prefill_coverage.mmid_count, pure_prefill_coverage.mmid_fingerprint);
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_PREFILL_LEGACY && execution.size() == 1);
    CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_prefill_reason(plan, 0));

    ggml_cgraph * pure_decode_graph = candidate_graph(fixture, {
        decode_route.root, decode_route.ids, decode_gate_up_reader, decode_down_reader,
    });
    const auto pure_decode_coverage = candidate_certify_graph(registry, pure_decode_graph);
    registry.compile_graph_plan(
        pure_decode_graph, 707, &plan, &execution, pure_decode_coverage.epoch, pure_decode_coverage.nodes,
        pure_decode_coverage.mmid_count, pure_decode_coverage.mmid_fingerprint);
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED && execution.size() == 1);
    CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_eligible_reason(plan, 0));
    CHECK(execution.resolve_streams(candidate_test_graph_stream, reinterpret_cast<void *>(uintptr_t{1})));
    CHECK(execution.has_stream_grouped_candidate());

    ggml_cgraph * incomplete_decode_graph = candidate_graph(fixture, {
        decode_route.root, decode_route.ids, decode_gate_up_reader,
    });
    const auto incomplete_decode_coverage = candidate_certify_graph(registry, incomplete_decode_graph);
    registry.compile_graph_plan(
        incomplete_decode_graph, 708, &plan, &execution,
        incomplete_decode_coverage.epoch, incomplete_decode_coverage.nodes,
        incomplete_decode_coverage.mmid_count, incomplete_decode_coverage.mmid_fingerprint);
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR && execution.rejects_cached_mmid(decode_gate_up_reader));

    fprintf(stderr, "test-moe-cache: mixed phase graph repair OK\n");
}

struct cached_fusion_test_graph {
    ggml_context_ptr weights;
    ggml_context_ptr auxiliaries;
    ggml_context_ptr nodes;
    ggml_backend_buffer_ptr weight_buffer;
    ggml_backend_buffer_ptr auxiliary_buffer;
    ggml_backend_buffer_ptr node_buffer;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * output = nullptr;
    std::vector<ggml_tensor *> leaves;
    std::vector<ggml_tensor *> cached_weights;
};

static cached_fusion_test_graph build_cached_fusion_test_graph(
        ggml_backend_t backend,
        ggml_backend_buffer_type_t weight_buft) {
    constexpr int64_t N_EXPERTS = 4;
    constexpr int64_t N_USED = 2;
    constexpr int64_t N_TOKENS = 1;
    constexpr int64_t N_IN = 256;
    constexpr int64_t N_OUT = 32;

    ggml_init_params weight_params = {
        /* .mem_size = */ ggml_tensor_overhead() * 64,
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };
    ggml_init_params node_params = {
        /* .mem_size = */ ggml_tensor_overhead() * 512 + ggml_graph_overhead_custom(256, false),
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };

    cached_fusion_test_graph result;
    result.weights.reset(ggml_init(weight_params));
    result.auxiliaries.reset(ggml_init(weight_params));
    result.nodes.reset(ggml_init(node_params));
    CHECK(result.weights != nullptr && result.auxiliaries != nullptr && result.nodes != nullptr);

    auto new_leaf = [&](ggml_context * ctx, ggml_type type, int n_dims, const int64_t * ne, const std::string & name) {
        ggml_tensor * tensor = ggml_new_tensor(ctx, type, n_dims, ne);
        ggml_set_name(tensor, name.c_str());
        result.leaves.push_back(tensor);
        return tensor;
    };
    auto new_weight = [&](ggml_type type, const std::string & name) {
        const int64_t ne[] = {N_IN, N_OUT, N_EXPERTS};
        ggml_tensor * tensor = new_leaf(result.weights.get(), type, 3, ne, name + ".weight");
        result.cached_weights.push_back(tensor);
        return tensor;
    };
    auto new_activation = [&](const std::string & name) {
        const int64_t ne[] = {N_IN, N_USED, N_TOKENS};
        return new_leaf(result.nodes.get(), GGML_TYPE_F32, 3, ne, name + ".input");
    };
    auto new_ids = [&](const std::string & name) {
        const int64_t ne[] = {N_USED, N_TOKENS};
        return new_leaf(result.nodes.get(), GGML_TYPE_I32, 2, ne, name + ".ids");
    };
    auto add_scale = [&](ggml_tensor * mmid, ggml_tensor * ids, const std::string & name) {
        const int64_t ne[] = {N_EXPERTS};
        ggml_tensor * scale = new_leaf(result.auxiliaries.get(), GGML_TYPE_F32, 1, ne, name + ".scale");
        ggml_tensor * selected = ggml_reshape_3d(result.nodes.get(), scale, 1, N_EXPERTS, 1);
        selected = ggml_repeat_4d(result.nodes.get(), selected, 1, N_EXPERTS, N_TOKENS, 1);
        selected = ggml_get_rows(result.nodes.get(), selected, ids);
        return ggml_mul(result.nodes.get(), mmid, selected);
    };
    auto add_bias = [&](ggml_tensor * mmid, ggml_tensor * ids, const std::string & name) {
        const int64_t ne[] = {N_OUT, N_EXPERTS};
        ggml_tensor * bias = new_leaf(result.nodes.get(), GGML_TYPE_F32, 2, ne, name + ".bias");
        return ggml_add_id(result.nodes.get(), mmid, bias, ids);
    };
    auto build_single = [&](ggml_type type, const std::string & name, bool with_scale, bool with_bias) {
        ggml_tensor * weight = new_weight(type, name);
        ggml_tensor * input = new_activation(name);
        ggml_tensor * ids = new_ids(name);
        ggml_tensor * output = ggml_mul_mat_id(result.nodes.get(), weight, input, ids);
        if (with_scale) {
            output = add_scale(output, ids, name);
        }
        if (with_bias) {
            output = add_bias(output, ids, name);
        }
        return output;
    };
    auto build_pair = [&](ggml_type type, const std::string & name, bool with_scale, bool with_bias) {
        ggml_tensor * gate_weight = new_weight(type, name + ".gate");
        ggml_tensor * up_weight = new_weight(type, name + ".up");
        ggml_tensor * input = new_activation(name);
        ggml_tensor * ids = new_ids(name);
        ggml_tensor * gate = ggml_mul_mat_id(result.nodes.get(), gate_weight, input, ids);
        ggml_tensor * up = ggml_mul_mat_id(result.nodes.get(), up_weight, input, ids);
        if (with_scale) {
            gate = add_scale(gate, ids, name + ".gate");
            up = add_scale(up, ids, name + ".up");
        }
        if (with_bias) {
            gate = add_bias(gate, ids, name + ".gate");
            up = add_bias(up, ids, name + ".up");
        }
        return ggml_glu_split(result.nodes.get(), gate, up, GGML_GLU_OP_SWIGLU);
    };

    std::vector<ggml_tensor *> outputs;
    outputs.push_back(build_single(GGML_TYPE_Q4_0, "test.ordinary.q4_0", false, false));
    outputs.push_back(build_single(GGML_TYPE_Q4_K, "test.ordinary.q4_k", false, false));
    outputs.push_back(build_pair(GGML_TYPE_NVFP4, "test.f1.no_bias", true, false));
    outputs.push_back(build_pair(GGML_TYPE_NVFP4, "test.f1.bias", true, true));
    outputs.push_back(build_pair(GGML_TYPE_Q4_0, "test.f2.q4_0", false, true));
    outputs.push_back(build_pair(GGML_TYPE_Q4_K, "test.f2.q4_k", false, true));
    outputs.push_back(build_pair(GGML_TYPE_Q4_0, "test.f3.q4_0", false, false));
    outputs.push_back(build_pair(GGML_TYPE_Q4_K, "test.f3.q4_k", false, false));
    outputs.push_back(build_single(GGML_TYPE_NVFP4, "test.f4.no_bias", true, false));
    outputs.push_back(build_single(GGML_TYPE_NVFP4, "test.f4.bias", true, true));
    outputs.push_back(build_single(GGML_TYPE_Q4_0, "test.f5.q4_0", false, true));
    outputs.push_back(build_single(GGML_TYPE_Q4_K, "test.f5.q4_k", false, true));
    outputs.push_back(build_single(GGML_TYPE_BF16, "test.f5.bf16", false, true));

    result.output = outputs[0];
    for (size_t i = 1; i < outputs.size(); ++i) {
        result.output = ggml_add(result.nodes.get(), result.output, outputs[i]);
    }
    ggml_set_name(result.output, "test.cached_fusion.output");

    result.graph = ggml_new_graph_custom(result.nodes.get(), 256, false);
    ggml_build_forward_expand(result.graph, result.output);
    result.weight_buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(result.weights.get(), weight_buft));
    result.auxiliary_buffer.reset(ggml_backend_alloc_ctx_tensors(result.auxiliaries.get(), backend));
    result.node_buffer.reset(ggml_backend_alloc_ctx_tensors(result.nodes.get(), backend));
    CHECK(result.weight_buffer != nullptr && result.auxiliary_buffer != nullptr && result.node_buffer != nullptr);
    ggml_backend_buffer_set_usage(result.weight_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_set_usage(result.auxiliary_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    return result;
}

static std::vector<uint8_t> cached_fusion_test_data(const ggml_tensor * tensor, size_t salt) {
    std::vector<uint8_t> bytes(ggml_nbytes(tensor));
    if (tensor->type == GGML_TYPE_I32) {
        std::vector<int32_t> values(ggml_nelements(tensor));
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = i % 2 == 0 ? 0 : 2;
        }
        memcpy(bytes.data(), values.data(), bytes.size());
        return bytes;
    }

    std::vector<float> values(ggml_nelements(tensor));
    const bool scale = strstr(tensor->name, ".scale") != nullptr;
    const bool bias = strstr(tensor->name, ".bias") != nullptr;
    for (size_t i = 0; i < values.size(); ++i) {
        const int phase = static_cast<int>((i + 3 * salt) % 17) - 8;
        values[i] = scale ? 0.75f + 0.025f * (phase + 8) : (bias ? 0.01f : 0.035f) * phase;
    }

    if (tensor->type == GGML_TYPE_F32) {
        memcpy(bytes.data(), values.data(), bytes.size());
    } else if (tensor->type == GGML_TYPE_BF16) {
        ggml_fp32_to_bf16_row_ref(values.data(), reinterpret_cast<ggml_bf16_t *>(bytes.data()), values.size());
    } else {
        CHECK(ggml_is_quantized(tensor->type));
        const int64_t nrows = ggml_nelements(tensor) / tensor->ne[0];
        CHECK(ggml_quantize_chunk(tensor->type, values.data(), bytes.data(), 0, nrows, tensor->ne[0], nullptr) == bytes.size());
    }
    return bytes;
}

static std::vector<uint8_t> active_grouped_q4k_expert_data(const ggml_tensor * tensor, uint32_t salt) {
    CHECK(tensor->type == GGML_TYPE_Q4_K && tensor->ne[0] > 0 && tensor->ne[1] > 0 && tensor->ne[2] > 0);
    CHECK(tensor->nb[2] == ggml_row_size(tensor->type, tensor->ne[0]) * tensor->ne[1] &&
        ggml_nbytes(tensor) == tensor->nb[2] * tensor->ne[2]);
    std::vector<uint8_t> bytes(ggml_nbytes(tensor));
    std::vector<float> values(tensor->ne[0] * tensor->ne[1]);
    std::vector<uint64_t> fingerprints(tensor->ne[2]);
    std::unordered_set<uint64_t> unique_fingerprints;
    for (uint32_t expert = 0; expert < tensor->ne[2]; ++expert) {
        uint32_t state = 0x9e3779b9u * (expert + 1) ^ 0x85ebca6bu * (salt + 1);
        for (uint32_t value = 0; value < values.size(); ++value) {
            state = 1664525u * state + 1013904223u;
            const int32_t sample = static_cast<int32_t>((state >> 8) & 0xffff) - 32768;
            values[value] = sample * (0.5f + 0.01f * ((expert + salt) % 23)) / 32768.0f;
        }
        uint8_t * slab = bytes.data() + expert * tensor->nb[2];
        CHECK(ggml_quantize_chunk(
            tensor->type, values.data(), slab, 0, tensor->ne[1], tensor->ne[0], nullptr) == tensor->nb[2]);
        uint64_t fingerprint = 1469598103934665603ULL;
        for (size_t byte = 0; byte < tensor->nb[2]; ++byte) {
            fingerprint = (fingerprint ^ slab[byte]) * 1099511628211ULL;
        }
        fingerprints[expert] = fingerprint;
        CHECK(unique_fingerprints.insert(fingerprint).second);
        if (expert >= 17) {
            CHECK(fingerprints[expert] != fingerprints[expert - 17]);
        }
    }
    CHECK(unique_fingerprints.size() == static_cast<size_t>(tensor->ne[2]));
    return bytes;
}

#ifdef __linux__
static void (*file_mmap_cached_unregister)(ggml_backend_buffer_t) = nullptr;

static void file_mmap_cached_buffer_free(ggml_backend_buffer_t buffer) {
    CHECK(file_mmap_cached_unregister != nullptr);
    file_mmap_cached_unregister(buffer);
    CHECK(munmap(buffer->context, buffer->size) == 0);
}

static const char * file_mmap_cached_buffer_type_name(ggml_backend_buffer_type_t) {
    return "CUDA_MoE_Cached_File_Mmap_Test";
}

static ggml_backend_buffer_t file_mmap_cached_buffer_type_alloc_buffer(
        ggml_backend_buffer_type_t,
        size_t size) {
    char path[] = "/tmp/test-moe-cache-mmap-XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0) {
        return nullptr;
    }
    const bool sized = ftruncate(fd, static_cast<off_t>(size)) == 0;
    void * data = sized ? mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0) : MAP_FAILED;
    (void) unlink(path);
    (void) close(fd);
    if (data == MAP_FAILED) {
        return nullptr;
    }
    ggml_backend_buffer_t buffer = ggml_backend_cuda_moe_cached_buffer_from_host_ptr(data, size);
    if (buffer == nullptr) {
        (void) munmap(data, size);
        return nullptr;
    }
    if (file_mmap_cached_unregister == nullptr) {
        file_mmap_cached_unregister = buffer->iface.free_buffer;
    } else {
        CHECK(file_mmap_cached_unregister == buffer->iface.free_buffer);
    }
    buffer->iface.free_buffer = file_mmap_cached_buffer_free;
    return buffer;
}

static size_t file_mmap_cached_buffer_type_get_alignment(ggml_backend_buffer_type_t) {
    auto * cached = ggml_backend_cuda_moe_cached_buffer_type();
    return cached->iface.get_alignment(cached);
}

static size_t file_mmap_cached_buffer_type_get_alloc_size(
        ggml_backend_buffer_type_t,
        const ggml_tensor * tensor) {
    auto * cached = ggml_backend_cuda_moe_cached_buffer_type();
    return cached->iface.get_alloc_size != nullptr ? cached->iface.get_alloc_size(cached, tensor) : ggml_nbytes(tensor);
}

static bool file_mmap_cached_buffer_type_is_host(ggml_backend_buffer_type_t) {
    return false;
}

static ggml_backend_buffer_type_t file_mmap_cached_buffer_type() {
    static ggml_backend_buffer_type buffer_type = {
        /* .iface   = */ {
            /* .get_name         = */ file_mmap_cached_buffer_type_name,
            /* .alloc_buffer     = */ file_mmap_cached_buffer_type_alloc_buffer,
            /* .get_alignment    = */ file_mmap_cached_buffer_type_get_alignment,
            /* .get_max_size     = */ nullptr,
            /* .get_alloc_size   = */ file_mmap_cached_buffer_type_get_alloc_size,
            /* .is_host          = */ file_mmap_cached_buffer_type_is_host,
        },
        /* .device  = */ nullptr,
        /* .context = */ nullptr,
    };
    buffer_type.device = ggml_backend_cuda_moe_cached_buffer_type()->device;
    return &buffer_type;
}
#endif

struct active_grouped_dispatch_graph {
    ggml_context_ptr weights;
    ggml_context_ptr nodes;
    ggml_backend_buffer_ptr weight_buffer;
    ggml_backend_buffer_ptr node_buffer;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * down = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * logits = nullptr;
    ggml_tensor * concurrent_root = nullptr;
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

static active_grouped_dispatch_graph build_active_grouped_dispatch_graph_types(
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
        bool mapped_host_biases = false) {
    CHECK(n_rows >= 1 && n_used >= 1 && n_used <= n_experts);
    CHECK(layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP ||
        layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE ||
        layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_UNGATED);
    const uint32_t ff_dim = n_ff != 0 ? n_ff : n_dim;
    CHECK(!concurrent_stream_fixture ||
        (layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE && n_rows == 1 && n_used == 1));
    CHECK(!original_direct_biases || (shared_banks == nullptr && !original_direct_down_scale && !original_direct_nvfp4_scales));
    CHECK(!mapped_host_biases || original_direct_biases);
    const ggml_init_params weight_params = {
        /* .mem_size = */ ggml_tensor_overhead() * 12,
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };
    const ggml_init_params node_params = {
        /* .mem_size = */ ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false),
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };

    active_grouped_dispatch_graph result;
    result.n_experts = n_experts;
    result.n_used = n_used;
    result.n_rows = n_rows;
    CHECK(shared_banks == nullptr ||
        (shared_banks->n_experts == n_experts && shared_banks->n_used == n_used));
    if (shared_banks == nullptr) {
        result.weights.reset(ggml_init(weight_params));
    }
    result.nodes.reset(ggml_init(node_params));
    CHECK((shared_banks != nullptr || result.weights != nullptr) && result.nodes != nullptr);

    size_t shared_bank_index = 0;
    auto add_bank = [&](ggml_type type, int64_t ne0, int64_t ne1, uint32_t role, const char * name) {
        ggml_tensor * tensor = nullptr;
        if (shared_banks != nullptr) {
            CHECK(shared_bank_index < shared_banks->banks.size() && shared_bank_index < shared_banks->roles.size());
            tensor = shared_banks->banks[shared_bank_index];
            CHECK(tensor != nullptr && tensor->type == type && tensor->ne[0] == ne0 && tensor->ne[1] == ne1 &&
                tensor->ne[2] == n_experts && shared_banks->roles[shared_bank_index] == role);
            ++shared_bank_index;
        } else {
            tensor = ggml_new_tensor_3d(result.weights.get(), type, ne0, ne1, n_experts);
            ggml_set_name(tensor, name);
        }
        result.banks.push_back(tensor);
        result.roles.push_back(role);
        return tensor;
    };
    ggml_tensor * gate_up = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * up = nullptr;
    if (layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP) {
        gate_up = add_bank(types[0], n_dim, 2 * ff_dim, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT,
            "test_active_gate_up_weight");
    } else if (layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE) {
        gate = add_bank(types[0], n_dim, ff_dim, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT,
            "test_active_gate_weight");
        up = add_bank(types[1], n_dim, ff_dim, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT,
            "test_active_up_weight");
    } else {
        up = add_bank(types[0], n_dim, ff_dim, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT,
            "test_active_up_weight");
    }
    const uint32_t down_type = layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE ? 2 : 1;
    result.down = add_bank(types[down_type], ff_dim,
        n_dim, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT,
        "test_active_down_weight");
    ggml_tensor * gate_bias = nullptr;
    ggml_tensor * up_bias = nullptr;
    ggml_tensor * gate_up_bias = nullptr;
    ggml_tensor * down_bias = nullptr;
    if (original_direct_biases) {
        const auto add_bias = [&](int64_t ne0, uint32_t role, const char * name) {
            ggml_tensor * tensor = ggml_new_tensor_2d(
                mapped_host_biases ? result.weights.get() : result.nodes.get(), GGML_TYPE_F32, ne0, n_experts);
            ggml_set_name(tensor, name);
            result.biases.push_back(tensor);
            result.bias_roles.push_back(role);
            return tensor;
        };
        if (layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP) {
            gate_up_bias = add_bias(2 * ff_dim, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_BIAS,
                "test_active_gate_up_bias");
        } else if (layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE) {
            gate_bias = add_bias(ff_dim, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_BIAS, "test_active_gate_bias");
            up_bias = add_bias(ff_dim, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_BIAS, "test_active_up_bias");
        } else {
            up_bias = add_bias(ff_dim, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_BIAS, "test_active_up_bias");
        }
        down_bias = add_bias(n_dim, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_BIAS, "test_active_down_bias");
    }
    if (shared_banks != nullptr) {
        result.gate_scale = shared_banks->gate_scale;
        result.up_scale = shared_banks->up_scale;
        result.down_scale = shared_banks->down_scale;
        CHECK((original_direct_nvfp4_scales && result.gate_scale != nullptr && result.up_scale != nullptr && result.down_scale != nullptr) ||
            (!original_direct_nvfp4_scales && result.gate_scale == nullptr && result.up_scale == nullptr &&
                (original_direct_down_scale == (result.down_scale != nullptr))));
    } else if (original_direct_nvfp4_scales) {
        CHECK(layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE &&
            types[0] == GGML_TYPE_NVFP4 && types[1] == GGML_TYPE_NVFP4 && types[2] == GGML_TYPE_NVFP4);
        result.gate_scale = ggml_new_tensor_1d(result.weights.get(), GGML_TYPE_F32, n_experts);
        result.up_scale = ggml_new_tensor_1d(result.weights.get(), GGML_TYPE_F32, n_experts);
        result.down_scale = ggml_new_tensor_1d(result.weights.get(), GGML_TYPE_F32, n_experts);
        ggml_set_name(result.gate_scale, "test.active.gate_scale");
        ggml_set_name(result.up_scale, "test.active.up_scale");
        ggml_set_name(result.down_scale, "test.active.down_scale");
    } else if (original_direct_down_scale) {
        result.down_scale = ggml_new_tensor_1d(result.weights.get(), GGML_TYPE_F32, n_experts);
        ggml_set_name(result.down_scale, "test.active.down_scale");
    }

    result.input = ggml_new_tensor_3d(result.nodes.get(), GGML_TYPE_F32, n_dim, 1, n_rows);
    result.logits = ggml_new_tensor_2d(result.nodes.get(), GGML_TYPE_F32, n_experts, n_rows);
    ggml_set_name(result.input, "test.active.input");
    ggml_set_name(result.logits, "test.active.logits");
    ggml_tensor * mmid_input = result.input;
    ggml_tensor * concurrent_branch = nullptr;
    if (concurrent_stream_fixture) {
        result.concurrent_root = ggml_dup(result.nodes.get(), result.input);
        ggml_set_name(result.concurrent_root, "test.active.attn_norm");
        concurrent_branch = ggml_sqr(result.nodes.get(), result.concurrent_root);
        mmid_input = result.concurrent_root;
    }
    result.ids = ggml_argsort_top_k(result.nodes.get(), result.logits, n_used);
    ggml_set_name(result.ids, "test.active.ids");
    ggml_tensor * hidden = nullptr;
    if (gate_up != nullptr) {
        hidden = ggml_mul_mat_id(result.nodes.get(), gate_up, mmid_input, result.ids);
        result.readers.push_back(hidden);
        if (gate_up_bias != nullptr) {
            hidden = ggml_add_id(result.nodes.get(), hidden, gate_up_bias, result.ids);
        }
        hidden = ggml_dup(result.nodes.get(), hidden);
        hidden = ggml_glu(result.nodes.get(), hidden, GGML_GLU_OP_SWIGLU, false);
    } else if (gate != nullptr) {
        result.gate_output = ggml_mul_mat_id(result.nodes.get(), gate, mmid_input, result.ids);
        result.up_output = ggml_mul_mat_id(result.nodes.get(), up, mmid_input, result.ids);
        result.readers.push_back(result.gate_output);
        result.readers.push_back(result.up_output);
        ggml_tensor * gate_input = result.gate_output;
        ggml_tensor * up_input = result.up_output;
        if (gate_bias != nullptr) {
            gate_input = ggml_add_id(result.nodes.get(), gate_input, gate_bias, result.ids);
            up_input = ggml_add_id(result.nodes.get(), up_input, up_bias, result.ids);
        }
        if (result.gate_scale != nullptr) {
            result.gate_scale_reshape = ggml_reshape_3d(result.nodes.get(), result.gate_scale, 1, n_experts, 1);
            result.gate_scale_repeat = ggml_repeat_4d(result.nodes.get(), result.gate_scale_reshape, 1, n_experts, n_rows, 1);
            result.gate_scale_rows = ggml_get_rows(result.nodes.get(), result.gate_scale_repeat, result.ids);
            gate_input = ggml_mul(result.nodes.get(), result.gate_output, result.gate_scale_rows);
            result.up_scale_reshape = ggml_reshape_3d(result.nodes.get(), result.up_scale, 1, n_experts, 1);
            result.up_scale_repeat = ggml_repeat_4d(result.nodes.get(), result.up_scale_reshape, 1, n_experts, n_rows, 1);
            result.up_scale_rows = ggml_get_rows(result.nodes.get(), result.up_scale_repeat, result.ids);
            up_input = ggml_mul(result.nodes.get(), result.up_output, result.up_scale_rows);
        }
        hidden = ggml_glu_split(result.nodes.get(), gate_input, up_input, GGML_GLU_OP_SWIGLU);
    } else {
        result.up_output = ggml_mul_mat_id(result.nodes.get(), up, mmid_input, result.ids);
        result.readers.push_back(result.up_output);
        ggml_tensor * up_input = result.up_output;
        if (up_bias != nullptr) {
            up_input = ggml_add_id(result.nodes.get(), up_input, up_bias, result.ids);
        }
        hidden = ggml_sqr(result.nodes.get(), up_input);
    }
    result.down_output = ggml_mul_mat_id(result.nodes.get(), result.down, hidden, result.ids);
    result.readers.push_back(result.down_output);
    result.output = result.down_output;
    if (down_bias != nullptr) {
        result.output = ggml_add_id(result.nodes.get(), result.output, down_bias, result.ids);
    }
    if (result.down_scale != nullptr) {
        result.down_scale_reshape = ggml_reshape_3d(result.nodes.get(), result.down_scale, 1, n_experts, 1);
        result.down_scale_repeat = ggml_repeat_4d(result.nodes.get(), result.down_scale_reshape, 1, n_experts, n_rows, 1);
        result.down_scale_rows = ggml_get_rows(result.nodes.get(), result.down_scale_repeat, result.ids);
        result.output = ggml_mul(result.nodes.get(), result.down_output, result.down_scale_rows);
    }
    ggml_set_name(result.output, "test.active.output");
    result.graph = ggml_new_graph_custom(result.nodes.get(), 64, false);
    ggml_build_forward_expand(result.graph, result.output);
    if (concurrent_stream_fixture) {
        ggml_build_forward_expand(result.graph, concurrent_branch);
        CHECK((concurrent_branch->flags & GGML_TENSOR_FLAG_COMPUTE) != 0);
        const auto move_node = [&](ggml_tensor * node, int32_t target) {
            int32_t source = -1;
            for (int32_t index = target; index < result.graph->n_nodes; ++index) {
                if (result.graph->nodes[index] == node) {
                    source = index;
                    break;
                }
            }
            CHECK(source >= target);
            memmove(result.graph->nodes + target + 1, result.graph->nodes + target,
                (source - target) * sizeof(result.graph->nodes[0]));
            result.graph->nodes[target] = node;
        };
        move_node(result.ids->view_src, 0);
        move_node(result.ids, 1);
        CHECK(result.graph->nodes[2] == result.concurrent_root);
        move_node(concurrent_branch, 3);
        candidate_rebuild_graph_uses(result.graph);
    }
    candidate_stamp_execution(result.graph, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
        GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, n_rows, n_rows);
    if (shared_banks == nullptr) {
        result.weight_buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(result.weights.get(), weight_buft));
    } else {
        CHECK(shared_bank_index == shared_banks->banks.size());
    }
    result.node_buffer.reset(ggml_backend_alloc_ctx_tensors(result.nodes.get(), backend));
    CHECK((shared_banks != nullptr || result.weight_buffer != nullptr) && result.node_buffer != nullptr);
    if (result.weight_buffer != nullptr) {
        ggml_backend_buffer_set_usage(result.weight_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    return result;
}

static active_grouped_dispatch_graph build_active_grouped_dispatch_graph(
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
        bool concurrent_stream_fixture = false) {
    return build_active_grouped_dispatch_graph_types(
        backend, weight_buft, {type, type, type}, layout, original_direct_down_scale, false,
        n_rows, n_experts, n_used, n_dim, shared_banks, concurrent_stream_fixture, false);
}

static constexpr int32_t active_grouped_route_variants[3][4][2] = {
    {
        {3, 5},
        {5, 2},
        {3, 1},
        {7, 5},
    },
    {
        {2, 6},
        {6, 4},
        {0, 2},
        {4, 1},
    },
    {
        {1, 7},
        {3, 1},
        {7, 4},
        {2, 3},
    },
};

static constexpr int32_t active_grouped_practical_route_variants[3][4][8] = {
    {
        {3, 5, 17, 29, 41, 53, 65, 77},
        {5, 2, 18, 30, 42, 54, 66, 78},
        {3, 1, 19, 31, 43, 55, 67, 79},
        {7, 5, 20, 32, 44, 56, 68, 80},
    },
    {
        {82, 6, 21, 33, 45, 57, 69, 81},
        {6, 4, 22, 34, 46, 58, 70, 83},
        {82, 2, 23, 35, 47, 59, 71, 84},
        {4, 1, 24, 36, 48, 60, 72, 85},
    },
    {
        {9, 87, 25, 37, 49, 61, 73, 86},
        {11, 9, 26, 38, 50, 62, 74, 88},
        {87, 13, 27, 39, 51, 63, 75, 89},
        {15, 11, 28, 40, 52, 64, 76, 90},
    },
};

static int32_t active_grouped_route(
        const active_grouped_dispatch_graph & graph,
        uint32_t route_variant,
        uint32_t row,
        uint32_t route) {
    CHECK(route_variant < 3 && row < graph.n_rows && route < graph.n_used);
    if (graph.n_experts == 8 && (graph.n_used == 1 || graph.n_used == 2)) {
        return active_grouped_route_variants[route_variant][row][route];
    }
    CHECK((graph.n_experts == 128 || graph.n_experts == 256) && graph.n_used == 8);
    if (graph.n_experts == 256) {
        return (64 * route_variant + 8 * row + route) % graph.n_experts;
    }
    if (row < 4) {
        return active_grouped_practical_route_variants[route_variant][row][route];
    }
    return (31 * route_variant + 7 * row + 13 * route) % graph.n_experts;
}

static void set_active_grouped_contract_routes(active_grouped_dispatch_graph & graph) {
    CHECK(graph.ids != nullptr && graph.ids->view_src != nullptr);
    ggml_tensor * sorted = graph.ids->view_src;
    CHECK(sorted->type == GGML_TYPE_I32 && sorted->ne[0] == graph.n_experts && sorted->ne[1] == graph.n_rows);
    std::vector<int32_t> values(ggml_nelements(sorted));
    for (uint32_t row = 0; row < graph.n_rows; ++row) {
        std::vector<bool> used(graph.n_experts, false);
        uint32_t offset = row * graph.n_experts;
        uint32_t index = 0;
        for (uint32_t route = 0; route < graph.n_used; ++route) {
            const int32_t expert = active_grouped_route(graph, 0, row, route);
            CHECK(expert >= 0 && static_cast<uint32_t>(expert) < graph.n_experts && !used[expert]);
            values[offset + index++] = expert;
            used[expert] = true;
        }
        for (uint32_t expert = 0; expert < graph.n_experts; ++expert) {
            if (!used[expert]) {
                values[offset + index++] = expert;
            }
        }
        CHECK(index == graph.n_experts);
    }
    ggml_backend_tensor_set(sorted, values.data(), 0, ggml_nbytes(sorted));
}

static std::vector<int32_t> active_grouped_routes(
        const active_grouped_dispatch_graph & graph,
        uint32_t route_variant) {
    CHECK(route_variant < 3);
    std::vector<int32_t> routes(graph.n_rows * graph.n_used);
    for (uint32_t row = 0; row < graph.n_rows; ++row) {
        for (uint32_t route = 0; route < graph.n_used; ++route) {
            routes[row * graph.n_used + route] = active_grouped_route(graph, route_variant, row, route);
        }
    }
    return routes;
}

static std::vector<float> active_grouped_logits(
        const active_grouped_dispatch_graph & graph,
        const std::vector<int32_t> & routes) {
    CHECK(graph.logits != nullptr && graph.logits->ne[0] == graph.n_experts &&
        graph.logits->ne[1] == graph.n_rows && graph.n_rows >= 1 && routes.size() == graph.n_rows * graph.n_used);
    std::vector<float> result(graph.logits->ne[0] * graph.logits->ne[1]);
    for (uint32_t row = 0; row < graph.logits->ne[1]; ++row) {
        std::vector<bool> used(graph.n_experts, false);
        for (uint32_t expert = 0; expert < graph.logits->ne[0]; ++expert) {
            result[row * graph.logits->ne[0] + expert] = -static_cast<float>(expert + 1);
        }
        for (uint32_t route = 0; route < graph.n_used; ++route) {
            const int32_t expert = routes[row * graph.n_used + route];
            CHECK(expert >= 0 && static_cast<uint32_t>(expert) < graph.n_experts && !used[expert]);
            used[expert] = true;
            result[row * graph.logits->ne[0] + expert] = static_cast<float>(graph.n_used - route + 8);
        }
    }
    return result;
}

static std::vector<float> active_grouped_logits(
        const active_grouped_dispatch_graph & graph,
        uint32_t route_variant = 0) {
    return active_grouped_logits(graph, active_grouped_routes(graph, route_variant));
}

static void set_active_grouped_dispatch_routes(
        const std::vector<active_grouped_dispatch_graph *> & graphs,
        const std::vector<int32_t> & routes) {
    CHECK(!graphs.empty());
    const auto logits = active_grouped_logits(*graphs[0], routes);
    for (auto * graph : graphs) {
        CHECK(graph != nullptr && graph->n_rows == graphs[0]->n_rows && graph->n_used == graphs[0]->n_used &&
            graph->n_experts == graphs[0]->n_experts && ggml_nbytes(graph->logits) == logits.size() * sizeof(float));
        ggml_backend_tensor_set(graph->logits, logits.data(), 0, logits.size() * sizeof(float));
    }
}

static void set_active_grouped_dispatch_logits(
        const std::vector<active_grouped_dispatch_graph *> & graphs,
        uint32_t route_variant) {
    CHECK(!graphs.empty());
    set_active_grouped_dispatch_routes(graphs, active_grouped_routes(*graphs[0], route_variant));
}

static void initialize_active_grouped_dispatch_graphs(
        const std::vector<active_grouped_dispatch_graph *> & graphs,
        const std::vector<std::vector<uint8_t>> * bank_data = nullptr) {
    CHECK(graphs.size() >= 2);
    const auto logits = active_grouped_logits(*graphs[0]);
    for (auto * graph : graphs) {
        CHECK(graph->banks.size() == graphs[0]->banks.size());
    }
    CHECK(bank_data == nullptr || bank_data->size() == graphs[0]->banks.size());
    for (size_t bank = 0; bank < graphs[0]->banks.size(); ++bank) {
        const auto generated = bank_data == nullptr ? cached_fusion_test_data(graphs[0]->banks[bank], bank + 101) : std::vector<uint8_t>();
        const auto & bytes = bank_data != nullptr ? (*bank_data)[bank] : generated;
        for (auto * graph : graphs) {
            CHECK(graph->banks[bank]->type == graphs[0]->banks[bank]->type);
            CHECK(graph->roles[bank] == graphs[0]->roles[bank]);
            CHECK(ggml_nbytes(graph->banks[bank]) == ggml_nbytes(graphs[0]->banks[bank]));
            for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
                CHECK(graph->banks[bank]->ne[dim] == graphs[0]->banks[bank]->ne[dim]);
                CHECK(graph->banks[bank]->nb[dim] == graphs[0]->banks[bank]->nb[dim]);
            }
            ggml_backend_tensor_set(graph->banks[bank], bytes.data(), 0, bytes.size());
        }
    }
    const std::array<ggml_tensor *, 3> first_scales = {
        graphs[0]->gate_scale, graphs[0]->up_scale, graphs[0]->down_scale,
    };
    for (uint32_t scale_index = 0; scale_index < first_scales.size(); ++scale_index) {
        if (first_scales[scale_index] == nullptr) {
            for (auto * graph : graphs) {
                const std::array<ggml_tensor *, 3> scales = {graph->gate_scale, graph->up_scale, graph->down_scale};
                CHECK(scales[scale_index] == nullptr);
            }
            continue;
        }
        const auto bytes = cached_fusion_test_data(first_scales[scale_index], 149 + scale_index);
        for (auto * graph : graphs) {
            const std::array<ggml_tensor *, 3> scales = {graph->gate_scale, graph->up_scale, graph->down_scale};
            CHECK(scales[scale_index] != nullptr && ggml_nbytes(scales[scale_index]) == bytes.size());
            ggml_backend_tensor_set(scales[scale_index], bytes.data(), 0, bytes.size());
        }
    }
    for (size_t bias_index = 0; bias_index < graphs[0]->biases.size(); ++bias_index) {
        const auto bytes = cached_fusion_test_data(graphs[0]->biases[bias_index], 157 + bias_index);
        for (auto * graph : graphs) {
            CHECK(graph->biases.size() == graphs[0]->biases.size() && graph->bias_roles.size() == graph->biases.size());
            CHECK(graph->bias_roles[bias_index] == graphs[0]->bias_roles[bias_index]);
            CHECK(ggml_nbytes(graph->biases[bias_index]) == bytes.size());
            ggml_backend_tensor_set(graph->biases[bias_index], bytes.data(), 0, bytes.size());
        }
    }
    const auto input = cached_fusion_test_data(graphs[0]->input, 131);
    for (auto * graph : graphs) {
        CHECK(graph->readers.size() == graph->banks.size());
        CHECK(graph->readers.back() == graph->down_output && graph->down_output->src[0] == graph->down);
        int32_t previous_reader = -1;
        for (ggml_tensor * reader : graph->readers) {
            int32_t reader_index = -1;
            for (int32_t node = 0; node < ggml_graph_n_nodes(graph->graph); ++node) {
                if (ggml_graph_node(graph->graph, node) == reader) {
                    reader_index = node;
                    break;
                }
            }
            CHECK(reader_index > previous_reader);
            previous_reader = reader_index;
        }
        CHECK(graph->input->type == graphs[0]->input->type && ggml_nbytes(graph->input) == ggml_nbytes(graphs[0]->input));
        CHECK(graph->logits->type == graphs[0]->logits->type && ggml_nbytes(graph->logits) == ggml_nbytes(graphs[0]->logits));
        CHECK(graph->ids->type == graphs[0]->ids->type && ggml_nbytes(graph->ids) == ggml_nbytes(graphs[0]->ids));
        CHECK(ggml_nbytes(graph->output) == ggml_nbytes(graphs[0]->output));
        ggml_backend_tensor_set(graph->input, input.data(), 0, input.size());
        ggml_backend_tensor_set(graph->logits, logits.data(), 0, logits.size() * sizeof(float));
    }
}

static void initialize_active_grouped_dispatch_graph(active_grouped_dispatch_graph & graph, size_t salt) {
    const auto logits = active_grouped_logits(graph);
    for (size_t bank = 0; bank < graph.banks.size(); ++bank) {
        const auto bytes = cached_fusion_test_data(graph.banks[bank], bank + salt);
        ggml_backend_tensor_set(graph.banks[bank], bytes.data(), 0, bytes.size());
    }
    const std::array<ggml_tensor *, 3> scales = {graph.gate_scale, graph.up_scale, graph.down_scale};
    for (uint32_t scale_index = 0; scale_index < scales.size(); ++scale_index) {
        if (scales[scale_index] != nullptr) {
            const auto bytes = cached_fusion_test_data(scales[scale_index], salt + 48 + scale_index);
            ggml_backend_tensor_set(scales[scale_index], bytes.data(), 0, bytes.size());
        }
    }
    for (size_t bias_index = 0; bias_index < graph.biases.size(); ++bias_index) {
        const auto bytes = cached_fusion_test_data(graph.biases[bias_index], salt + 56 + bias_index);
        ggml_backend_tensor_set(graph.biases[bias_index], bytes.data(), 0, bytes.size());
    }
    const auto input = cached_fusion_test_data(graph.input, salt + 30);
    ggml_backend_tensor_set(graph.input, input.data(), 0, input.size());
    ggml_backend_tensor_set(graph.logits, logits.data(), 0, logits.size() * sizeof(float));
}

static void register_active_grouped_dispatch(
        ggml_backend_t backend,
        const active_grouped_dispatch_graph & graph,
        uint32_t layout,
        uint32_t n_slots,
        bool auxiliary_first = false) {
    std::array<ggml_backend_moe_candidate_bank_v1, GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS> banks = {};
    uint32_t n_banks = 0;
    if (auxiliary_first && graph.down_scale != nullptr) {
        banks[n_banks++] = {graph.down_scale, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_SCALE, 0};
    }
    for (size_t i = 0; i < graph.banks.size(); ++i) {
        banks[n_banks++] = {graph.banks[i], graph.roles[i], 0};
    }
    for (size_t i = 0; i < graph.biases.size(); ++i) {
        banks[n_banks++] = {graph.biases[i], graph.bias_roles[i], 0};
    }
    if (!auxiliary_first && graph.down_scale != nullptr) {
        banks[n_banks++] = {graph.down_scale, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_SCALE, 0};
    }
    const ggml_backend_moe_candidate_group_v1 group = {
        banks.data(), n_banks, layout, 0, 0,
    };
    const auto snapshot = candidate_snapshot(n_slots, &group, 1);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v1(backend, &snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
}

static int32_t replace_active_grouped_nvfp4_dispatch_v2(
        ggml_backend_t backend,
        const active_grouped_dispatch_graph & graph,
        uint32_t n_slots,
        int32_t omitted_scale = -1) {
    CHECK(graph.banks.size() == 3 && graph.gate_scale != nullptr && graph.up_scale != nullptr && graph.down_scale != nullptr);
    const ggml_backend_moe_candidate_group_v2 group = {
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE,
        GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY,
        GGML_BACKEND_MOE_CANDIDATE_GROUP_V2_FLAG_NONE,
        0,
    };
    std::array<ggml_backend_moe_candidate_tensor_v2, 6> tensors = {};
    uint32_t n_tensors = 0;
    for (uint32_t i = 0; i < graph.banks.size(); ++i) {
        tensors[n_tensors++] = {
            graph.banks[i], 0, graph.roles[i], GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE,
            GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_CACHED_BUFFER, 0,
        };
    }
    const std::array<ggml_tensor *, 3> scales = {graph.gate_scale, graph.up_scale, graph.down_scale};
    const std::array<uint32_t, 3> scale_roles = {
        GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_SCALE,
        GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_SCALE,
        GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_SCALE,
    };
    for (uint32_t i = 0; i < scales.size(); ++i) {
        if (static_cast<int32_t>(i) == omitted_scale) {
            continue;
        }
        tensors[n_tensors++] = {
            scales[i], 0, scale_roles[i],
            GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_OUTPUT_SCALE,
            GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_CACHED_BUFFER, 0,
        };
    }
    const auto snapshot = candidate_snapshot_v2(n_slots, &group, 1, tensors.data(), n_tensors);
    return ggml_backend_cuda_moe_candidate_replace_v2(backend, &snapshot);
}

static std::vector<float> active_grouped_intermediate_sentinel(active_grouped_dispatch_graph & graph) {
    if (graph.gate_output == nullptr) {
        return {};
    }
    CHECK(graph.up_output != nullptr && ggml_nelements(graph.gate_output) == ggml_nelements(graph.up_output));
    std::vector<float> sentinel(ggml_nelements(graph.gate_output), -12345.25f);
    ggml_backend_tensor_set(graph.gate_output, sentinel.data(), 0, ggml_nbytes(graph.gate_output));
    ggml_backend_tensor_set(graph.up_output, sentinel.data(), 0, ggml_nbytes(graph.up_output));
    return sentinel;
}

static void check_active_grouped_intermediates(
        const active_grouped_dispatch_graph & graph,
        const std::vector<float> & sentinel,
        bool skipped) {
    if (graph.gate_output == nullptr) {
        CHECK(sentinel.empty());
        return;
    }
    std::vector<float> gate(ggml_nelements(graph.gate_output));
    std::vector<float> up(ggml_nelements(graph.up_output));
    ggml_backend_tensor_get(graph.gate_output, gate.data(), 0, ggml_nbytes(graph.gate_output));
    ggml_backend_tensor_get(graph.up_output, up.data(), 0, ggml_nbytes(graph.up_output));
    CHECK((gate == sentinel) == skipped);
    CHECK((up == sentinel) == skipped);
}

static uint64_t active_grouped_legacy_op_count(ggml_backend_t backend, bool is_decode = true) {
    auto * context = ggml_cuda_moe_grouped_context_for_test(backend);
    CHECK(context != nullptr);
    return ggml_cuda_moe_grouped_context_test_access::legacy_op_count(*context, is_decode);
}

static void check_active_grouped_debug_telemetry(
        ggml_backend_t backend,
        const active_grouped_dispatch_graph & graph,
        uint64_t expected_loaded_experts = 0,
        uint64_t expected_calls = 5) {
    auto * context = ggml_cuda_moe_grouped_context_for_test(backend);
    CHECK(context != nullptr);
    const auto telemetry = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
    CHECK(telemetry.registered == 1 && telemetry.covered == 1);
    CHECK(telemetry.plan_calls == expected_calls + 1 && telemetry.plan_compiles == 2 &&
        telemetry.plan_reuses == expected_calls - 1);
    CHECK(telemetry.calls == expected_calls && telemetry.ready == expected_calls && telemetry.completed == expected_calls);
    CHECK(telemetry.ready_min == expected_calls && telemetry.ready_max == expected_calls);
    CHECK(telemetry.completed_min == expected_calls && telemetry.completed_max == expected_calls);
    CHECK(telemetry.admitted_banks == expected_calls * graph.banks.size());
    CHECK(telemetry.fallback == 0 && telemetry.rollback == 0);
    CHECK(telemetry.prepare_error == 0 && telemetry.finish_error == 0);
    if (expected_loaded_experts == 0) {
        std::vector<bool> used(graph.n_experts, false);
        for (uint32_t row = 0; row < graph.n_rows; ++row) {
            for (uint32_t route = 0; route < graph.n_used; ++route) {
                const uint32_t expert = active_grouped_route(graph, 0, row, route);
                if (!used[expert]) {
                    used[expert] = true;
                    ++expected_loaded_experts;
                }
            }
        }
    }
    CHECK(telemetry.h2d_banks == expected_loaded_experts * graph.banks.size());
    uint64_t bytes_per_expert = 0;
    for (const ggml_tensor * bank : graph.banks) {
        bytes_per_expert += bank->nb[2];
    }
    CHECK(telemetry.h2d_bytes == expected_loaded_experts * bytes_per_expert);

    const auto reset = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
    CHECK(reset.registered == 1 && reset.covered == 0 && reset.plan_calls == 0);
    CHECK(reset.calls == 0 && reset.ready == 0 && reset.completed == 0);
    CHECK(reset.ready_min == 0 && reset.ready_max == 0);
    CHECK(reset.completed_min == 0 && reset.completed_max == 0);
    CHECK(reset.admitted_banks == 0 && reset.h2d_banks == 0 && reset.h2d_bytes == 0);
}

static void check_active_grouped_legacy_caches(
        ggml_backend_t backend,
        const active_grouped_dispatch_graph & graph,
        bool registered_source,
        bool expect_slot_activity = true) {
    auto * context = ggml_cuda_moe_grouped_context_for_test(backend);
    CHECK(context != nullptr);
    for (ggml_tensor * bank : graph.banks) {
        auto lease = context->acquire_legacy_cache(bank);
        CHECK(lease && lease.get() != nullptr);
        CHECK(lease.acquisition().registered_source == registered_source);
        uint64_t hits = 0;
        uint64_t misses = 0;
        uint64_t evictions = 0;
        ggml_cuda_moe_cache_stats(lease.get(), &hits, &misses, &evictions);
        CHECK((hits + misses > 0) == expect_slot_activity);
    }
}

static void check_active_grouped_contract(
        ggml_backend_t backend,
        active_grouped_dispatch_graph & graph,
        uint32_t n_slots,
        bool auxiliary_first = false,
        bool expect_compact_mmvq = false) {
    auto * context = ggml_cuda_moe_grouped_context_for_test(backend);
    CHECK(context != nullptr);
    set_active_grouped_contract_routes(graph);
    ggml_backend_synchronize(backend);

    std::shared_ptr<ggml_cuda_moe_graph_plan> plan;
    ggml_cuda_moe_graph_execution execution;
    const auto coverage = candidate_certify_graph(*context, graph.graph);
    CHECK(context->prepare_graph_execution(
        graph.graph, 801, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &plan, &execution,
        coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
        GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
    CHECK(execution.size() == 1);
    if (expect_compact_mmvq) {
        CHECK(graph.n_rows >= 2 && graph.n_rows <= 4);
        for (uint32_t bank_index = 0; bank_index < graph.banks.size(); ++bank_index) {
            const auto capability = ggml_cuda_moe_grouped_context_test_access::graph_bank_capability(*plan, 0, bank_index);
            CHECK(capability.tensor == graph.banks[bank_index] &&
                capability.consumer == GGML_CUDA_MMID_CONSUMER_MMVQ &&
                capability.reason == GGML_CUDA_MMID_CAPABILITY_OK &&
                capability.equivalence_reason == GGML_CUDA_MMID_CAPABILITY_OK &&
                capability.phase == GGML_CUDA_MMID_PHASE_PREFILL &&
                capability.mapping == GGML_CUDA_MMID_MAPPING_DIRECT &&
                capability.n_rows == graph.n_rows && capability.n_routes == graph.n_rows * graph.n_used &&
                capability.n_experts == graph.n_experts && capability.grouped_ne[2] == n_slots &&
                capability.n_slots == n_slots);
        }
    }
    CHECK(context->prepare_graph_execution(
        graph.graph, 801, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, &plan, &execution,
        coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
        GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
    CHECK(execution.size() == 1);
    if (graph.down_scale != nullptr) {
        ggml_cuda_moe_graph_execution unknown;
        CHECK(context->bind_graph_plan(
            graph.graph, 802, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, *plan, &unknown,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint));
        CHECK(unknown.find_group(graph.down_output, nullptr) != nullptr);

        const std::array<std::array<ggml_tensor *, 3>, 3> scale_chains = {{
            {graph.gate_scale_reshape, graph.gate_scale_repeat, graph.gate_scale_rows},
            {graph.up_scale_reshape, graph.up_scale_repeat, graph.up_scale_rows},
            {graph.down_scale_reshape, graph.down_scale_repeat, graph.down_scale_rows},
        }};
        for (const auto & scale_chain : scale_chains) {
            if (scale_chain[0] == nullptr) {
                continue;
            }
            ggml_tensor * saved_scale = scale_chain[0]->src[0];
            scale_chain[0]->src[0] = graph.input;
            CHECK(!context->bind_graph_plan(
                graph.graph, 803, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, *plan, &unknown,
                coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint));
            scale_chain[0]->src[0] = saved_scale;

            for (ggml_tensor * auxiliary : scale_chain) {
                int32_t node_index = -1;
                for (int32_t i = 0; i < ggml_graph_n_nodes(graph.graph); ++i) {
                    if (ggml_graph_node(graph.graph, i) == auxiliary) {
                        node_index = i;
                        break;
                    }
                }
                CHECK(node_index >= 0);
                ggml_tensor * saved_node = graph.graph->nodes[node_index];
                graph.graph->nodes[node_index] = graph.output;
                CHECK(!context->bind_graph_plan(
                    graph.graph, 804, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, *plan, &unknown,
                    coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint));
                graph.graph->nodes[node_index] = saved_node;
            }

            if (graph.gate_scale_reshape != nullptr) {
                for (ggml_tensor * auxiliary : scale_chain) {
                    const ggml_type saved_type = auxiliary->type;
                    auxiliary->type = GGML_TYPE_BF16;
                    CHECK(!context->bind_graph_plan(
                        graph.graph, 806, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, *plan, &unknown,
                        coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint));
                    auxiliary->type = saved_type;

                    const int64_t saved_ne = auxiliary->ne[3];
                    auxiliary->ne[3]++;
                    CHECK(!context->bind_graph_plan(
                        graph.graph, 807, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, *plan, &unknown,
                        coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint));
                    auxiliary->ne[3] = saved_ne;

                    const size_t saved_nb = auxiliary->nb[3];
                    auxiliary->nb[3]++;
                    CHECK(!context->bind_graph_plan(
                        graph.graph, 808, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, *plan, &unknown,
                        coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint));
                    auxiliary->nb[3] = saved_nb;
                }
            }
        }
        CHECK(context->bind_graph_plan(
            graph.graph, 809, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, *plan, &unknown,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint));
        CHECK(unknown.find_group(graph.down_output, nullptr) != nullptr);
    }
    cudaStream_t stream = nullptr;
    CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CHECK(execution.resolve_streams(candidate_test_graph_stream, stream));
    CHECK(context->begin_graph_dispatch(&execution, true));

    ggml_cuda_moe_graph_binding first_binding;
    auto * group = execution.find_group(graph.readers[0], &first_binding);
    CHECK(group != nullptr && group->state == GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_ARMED);
    CHECK(context->prepare_graph_group(group, first_binding, graph.readers[0], stream) == GGML_CUDA_MOE_GROUPED_DECODE_READY);
    CHECK(group->state == GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_ACTIVE && group->transaction.transaction_token != 0);
    CHECK(group->remapped_ids != nullptr && group->n_slots == n_slots);
    if (graph.gate_scale != nullptr) {
        const std::array<ggml_tensor *, 3> scales = {graph.gate_scale, graph.up_scale, graph.down_scale};
        const std::array<uint32_t, 3> roles = {
            GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_SCALE,
            GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_SCALE,
            GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_SCALE,
        };
        CHECK(group->n_auxiliary_shadows == scales.size());
        for (uint32_t scale = 0; scale < scales.size(); ++scale) {
            CHECK(group->auxiliary_tensors[scale] == scales[scale] && group->auxiliary_data[scale] != nullptr &&
                group->auxiliary_roles[scale] == roles[scale]);
            CHECK(group->auxiliary_data[scale] ==
                ggml_cuda_moe_grouped_context_test_access::device_auxiliary_data(*context, group->key.candidate, scales[scale]));
        }
    } else if (!graph.biases.empty()) {
        CHECK(group->n_auxiliary_shadows == graph.biases.size());
        for (uint32_t bias = 0; bias < graph.biases.size(); ++bias) {
            CHECK(group->auxiliary_tensors[bias] == graph.biases[bias] && group->auxiliary_data[bias] != nullptr &&
                group->auxiliary_roles[bias] == graph.bias_roles[bias]);
            CHECK(group->auxiliary_data[bias] ==
                ggml_cuda_moe_grouped_context_test_access::device_auxiliary_data(
                    *context, group->key.candidate, graph.biases[bias]));
        }
    } else {
        CHECK(group->n_auxiliary_shadows == 0);
    }
    const int32_t * remapped_ids = group->remapped_ids;

    for (size_t i = 0; i < graph.readers.size(); ++i) {
        ggml_cuda_moe_graph_binding binding;
        CHECK(execution.find_group(graph.readers[i], &binding) == group);
        CHECK(binding.bank_index == i + (auxiliary_first ? 1 : 0) && binding.slot_index == i && binding.role == graph.roles[i]);
        CHECK(binding.key.ids.tensor == graph.ids && group->remapped_ids == remapped_ids);
        CHECK(!context->acquire_legacy_cache(graph.banks[i], nullptr, &group->authority));
    }

    ggml_cuda_moe_grouped_resource_info info;
    CHECK(context->get_group_resources(group->transaction.acquisition, &info));
    CHECK(info.transaction_active && info.n_banks == graph.banks.size() && info.n_slots == n_slots);
    for (uint32_t slot_index = 0; slot_index < info.n_banks; ++slot_index) {
        ggml_cuda_moe_grouped_bank_descriptor descriptor;
        CHECK(context->get_group_resource_bank(group->transaction, slot_index, &descriptor));
        CHECK(descriptor.tensor == graph.banks[slot_index] && descriptor.role == graph.roles[slot_index]);
    }
    const auto acquisition = group->transaction.acquisition;
    ggml_cuda_moe_graph_binding last_binding;
    CHECK(execution.find_group(graph.readers.back(), &last_binding) == group);
    CHECK(context->finish_graph_group(group, last_binding, graph.readers.back(), stream));
    CHECK(context->finish_graph_dispatch(&execution));
    CHECK(context->get_group_resources(acquisition, &info) && !info.transaction_active);
    CUDA_OK(cudaStreamSynchronize(stream));
    CUDA_OK(cudaStreamDestroy(stream));

    ggml_cuda_moe_candidate_group_key key;
    uint64_t clock_bound = 0;
    CHECK(context->find_down_group_key(graph.down, &key));
    CHECK(ggml_cuda_moe_grouped_context_test_access::get_clock_bound(*context, key, &clock_bound));
    CHECK(clock_bound == graph.n_rows * graph.n_used);
}

static void test_grouped_graph_replay_lifecycle(int device) {
    const bool old_debug_mm = ggml_backend_cuda_moe_get_debug_mm();
    ggml_backend_cuda_moe_set_debug_mm(true);
    ggml_backend_ptr backend(ggml_backend_cuda_init(device));
    CHECK(backend != nullptr);
    auto graph = build_active_grouped_dispatch_graph(
        backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP);
    initialize_active_grouped_dispatch_graph(graph, 177);
    register_active_grouped_dispatch(
        backend.get(), graph, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 12);
    auto * context = ggml_cuda_moe_grouped_context_for_test(backend.get());
    CHECK(context != nullptr);
    const int32_t routes[] = {3, 5};
    ggml_backend_tensor_set(graph.ids, routes, 0, sizeof(routes));
    ggml_backend_synchronize(backend.get());

    cudaStream_t stream = nullptr;
    cudaStream_t previous_stream = nullptr;
    CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_OK(cudaStreamCreateWithFlags(&previous_stream, cudaStreamNonBlocking));
    std::shared_ptr<ggml_cuda_moe_graph_plan> plan;
    auto coverage = candidate_certify_graph(*context, graph.graph);
    const auto prepare_execution = [&](ggml_cuda_moe_graph_execution & execution, ggml_cuda_moe_graph_property_hint hint, cudaStream_t target) {
        CHECK(context->prepare_graph_execution(
            graph.graph, 817, hint, &plan, &execution,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) !=
            GGML_CUDA_MOE_GRAPH_PREPARE_UNAVAILABLE);
        CHECK(execution.resolve_streams(candidate_test_graph_stream, target));
    };
    const auto run_callbacks = [&](ggml_cuda_moe_graph_execution & execution, cudaStream_t target) {
        ggml_cuda_moe_graph_binding first_binding;
        auto * group = execution.find_group(graph.readers.front(), &first_binding);
        CHECK(group != nullptr);
        CHECK(context->prepare_graph_group(group, first_binding, graph.readers.front(), target) ==
            GGML_CUDA_MOE_GROUPED_DECODE_READY);
        const auto acquisition = group->transaction.acquisition;
        ggml_cuda_moe_graph_binding last_binding;
        CHECK(execution.find_group(graph.readers.back(), &last_binding) == group);
        CHECK(context->finish_graph_group(group, last_binding, graph.readers.back(), target));
        return acquisition;
    };

    ggml_cuda_moe_graph_execution direct;
    prepare_execution(direct, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, previous_stream);
    CHECK(context->begin_graph_dispatch(&direct, GGML_CUDA_MOE_GRAPH_DISPATCH_DIRECT));
    CHECK(direct.dispatch_mode() == GGML_CUDA_MOE_GRAPH_DISPATCH_DIRECT);
    run_callbacks(direct, previous_stream);
    CHECK(context->finish_graph_dispatch(&direct));

    ggml_cuda_moe_graph_execution capture;
    prepare_execution(capture, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, stream);
    uint64_t first_fingerprint = 0;
    std::vector<std::shared_ptr<void>> capture_leases;
    CHECK(context->graph_resource_fingerprint(capture, stream, &first_fingerprint, &capture_leases) && first_fingerprint != 0);
    CHECK(context->begin_graph_dispatch(&capture, GGML_CUDA_MOE_GRAPH_DISPATCH_CAPTURE));
    CHECK(context->activate_graph_resources(
        &capture, GGML_CUDA_MOE_GRAPH_DISPATCH_CAPTURE, first_fingerprint));
    cudaGraph_t captured_graph = nullptr;
    cudaGraphExec_t captured_instance = nullptr;
    CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
    const auto capture_acquisition = run_callbacks(capture, stream);
    CUDA_OK(cudaStreamEndCapture(stream, &captured_graph));
    size_t captured_node_count = 0;
    CUDA_OK(cudaGraphGetNodes(captured_graph, nullptr, &captured_node_count));
    std::vector<cudaGraphNode_t> captured_nodes(captured_node_count);
    CUDA_OK(cudaGraphGetNodes(captured_graph, captured_nodes.data(), &captured_node_count));
    for (cudaGraphNode_t node : captured_nodes) {
        cudaGraphNodeType type = cudaGraphNodeTypeCount;
        CUDA_OK(cudaGraphNodeGetType(node, &type));
        CHECK(type != cudaGraphNodeTypeWaitEvent && type != cudaGraphNodeTypeEventRecord);
    }
    CUDA_OK(cudaGraphInstantiate(&captured_instance, captured_graph, nullptr, nullptr, 0));
    CUDA_OK(cudaGraphLaunch(captured_instance, stream));
    auto * capture_group = capture.find_group(graph.readers.front(), nullptr);
    CHECK(capture_group != nullptr && capture_group->state == GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_ACTIVE &&
        capture_group->defer_completion);
    ggml_cuda_moe_grouped_resource_info capture_info;
    CHECK(context->get_group_resources(capture_acquisition, &capture_info) && capture_info.transaction_active);
    CHECK(context->finish_graph_dispatch(&capture));
    CHECK(context->get_group_resources(capture_acquisition, &capture_info) && !capture_info.transaction_active);
    CUDA_OK(cudaStreamSynchronize(stream));

    uint64_t captured_fingerprint = 0;
    CHECK(context->graph_resource_fingerprint(capture, stream, &captured_fingerprint));
    CHECK(captured_fingerprint == first_fingerprint);
    std::vector<std::weak_ptr<void>> capture_witnesses;
    capture_witnesses.reserve(capture_leases.size());
    for (const auto & lease : capture_leases) {
        capture_witnesses.emplace_back(lease);
    }
    capture_leases.clear();
    CHECK(capture_witnesses.size() == 1 && !capture_witnesses[0].expired());

    host_barrier direct_barrier;
    CUDA_OK(cudaLaunchHostFunc(previous_stream, wait_on_host_barrier, &direct_barrier));
    ggml_cuda_moe_graph_execution intervening_direct;
    prepare_execution(intervening_direct, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, previous_stream);
    CHECK(context->begin_graph_dispatch(&intervening_direct, GGML_CUDA_MOE_GRAPH_DISPATCH_DIRECT));
    run_callbacks(intervening_direct, previous_stream);
    CHECK(context->finish_graph_dispatch(&intervening_direct));
    while (!direct_barrier.entered.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    ggml_cuda_moe_graph_execution replay;
    prepare_execution(replay, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, stream);
    CHECK(context->begin_graph_dispatch(&replay, GGML_CUDA_MOE_GRAPH_DISPATCH_REPLAY));
    CHECK(context->activate_graph_resources(
        &replay, GGML_CUDA_MOE_GRAPH_DISPATCH_REPLAY, captured_fingerprint, &capture_witnesses));
    CUDA_OK(cudaGraphLaunch(captured_instance, stream));
    cudaEvent_t replay_done = nullptr;
    CUDA_OK(cudaEventCreateWithFlags(&replay_done, cudaEventDisableTiming));
    CUDA_OK(cudaEventRecord(replay_done, stream));
    auto * replay_group = replay.find_group(graph.readers.front(), nullptr);
    CHECK(replay_group != nullptr && replay_group->state == GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_REPLAY);
    ggml_cuda_moe_grouped_resource_info replay_info;
    CHECK(context->get_group_resources(replay_group->transaction.acquisition, &replay_info));
    CHECK(replay_info.transaction_active);
    bool replay_reached = false;
    for (int attempt = 0; attempt < 100; ++attempt) {
        const cudaError_t query = cudaEventQuery(replay_done);
        CHECK(query == cudaSuccess || query == cudaErrorNotReady);
        if (query == cudaSuccess) {
            replay_reached = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const bool replay_waited_for_direct = !replay_reached;
    direct_barrier.released.store(true, std::memory_order_release);

    std::atomic<bool> replacement_started{false};
    std::atomic<bool> replacement_done{false};
    std::thread replacement([&]() {
        replacement_started.store(true, std::memory_order_release);
        register_active_grouped_dispatch(
            backend.get(), graph, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 12);
        replacement_done.store(true, std::memory_order_release);
    });
    while (!replacement_started.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    while (!ggml_cuda_moe_grouped_context_test_access::admission_closed(*context)) {
        std::this_thread::yield();
    }
    CHECK(!replacement_done.load(std::memory_order_acquire));
    CHECK(context->finish_graph_dispatch(&replay));
    replacement.join();
    CHECK(replacement_done.load(std::memory_order_acquire));
    CHECK(capture_witnesses[0].expired());
    CUDA_OK(cudaStreamSynchronize(stream));
    CUDA_OK(cudaStreamSynchronize(previous_stream));
    CHECK(replay_waited_for_direct && cudaEventQuery(replay_done) == cudaSuccess);
    CUDA_OK(cudaEventDestroy(replay_done));
    CUDA_OK(cudaGraphExecDestroy(captured_instance));
    CUDA_OK(cudaGraphDestroy(captured_graph));

    const auto replay_telemetry = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
    CHECK(replay_telemetry.calls == 4 && replay_telemetry.ready == 4 && replay_telemetry.completed == 4);
    CHECK(replay_telemetry.admitted_banks == 4 * graph.banks.size());
    CHECK(replay_telemetry.prepare_error == 0 && replay_telemetry.finish_error == 0);

    coverage = candidate_certify_graph(*context, graph.graph);
    plan.reset();
    ggml_cuda_moe_graph_execution replacement_direct;
    prepare_execution(replacement_direct, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, stream);
    CHECK(context->begin_graph_dispatch(&replacement_direct, GGML_CUDA_MOE_GRAPH_DISPATCH_DIRECT));
    run_callbacks(replacement_direct, stream);
    CHECK(context->finish_graph_dispatch(&replacement_direct));
    CUDA_OK(cudaStreamSynchronize(stream));
    uint64_t replacement_fingerprint = 0;
    std::vector<std::shared_ptr<void>> replacement_leases;
    CHECK(context->graph_resource_fingerprint(
        replacement_direct, stream, &replacement_fingerprint, &replacement_leases));
    CHECK(replacement_fingerprint != captured_fingerprint);
    std::vector<std::weak_ptr<void>> replacement_witnesses;
    replacement_witnesses.reserve(replacement_leases.size());
    for (const auto & lease : replacement_leases) {
        replacement_witnesses.emplace_back(lease);
    }
    replacement_leases.clear();
    CHECK(replacement_witnesses.size() == 1 && !replacement_witnesses[0].expired());

    ggml_cuda_moe_graph_execution replacement_capture;
    prepare_execution(replacement_capture, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, stream);
    CHECK(context->begin_graph_dispatch(&replacement_capture, GGML_CUDA_MOE_GRAPH_DISPATCH_CAPTURE));
    CHECK(context->activate_graph_resources(
        &replacement_capture, GGML_CUDA_MOE_GRAPH_DISPATCH_CAPTURE, replacement_fingerprint));
    const auto replacement_acquisition = run_callbacks(replacement_capture, stream);
    CHECK(context->finish_graph_dispatch(&replacement_capture));
    CUDA_OK(cudaStreamSynchronize(stream));
    CHECK(ggml_cuda_moe_grouped_context_test_access::set_clock_bound(
        *context, replacement_acquisition, UINT64_MAX));

    ggml_cuda_moe_graph_execution overflow_replay;
    prepare_execution(overflow_replay, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, stream);
    CHECK(context->begin_graph_dispatch(&overflow_replay, GGML_CUDA_MOE_GRAPH_DISPATCH_REPLAY));
    CHECK(!context->activate_graph_resources(
        &overflow_replay, GGML_CUDA_MOE_GRAPH_DISPATCH_REPLAY, replacement_fingerprint, &replacement_witnesses));
    CHECK(context->finish_graph_dispatch(&overflow_replay));
    CHECK(context->begin_graph_dispatch(&overflow_replay, GGML_CUDA_MOE_GRAPH_DISPATCH_DIRECT));
    run_callbacks(overflow_replay, stream);
    CHECK(context->finish_graph_dispatch(&overflow_replay));
    CUDA_OK(cudaStreamSynchronize(stream));
    uint64_t refreshed_fingerprint = 0;
    CHECK(context->graph_resource_fingerprint(overflow_replay, stream, &refreshed_fingerprint));
    CHECK(refreshed_fingerprint != replacement_fingerprint);
    CHECK(replacement_witnesses[0].expired());

    ggml_cuda_moe_candidate_group_key key;
    uint64_t clock_bound = 0;
    CHECK(context->find_down_group_key(graph.down, &key));
    CHECK(ggml_cuda_moe_grouped_context_test_access::get_clock_bound(*context, key, &clock_bound));
    CHECK(clock_bound == 2);
    CUDA_OK(cudaStreamSynchronize(previous_stream));
    CUDA_OK(cudaStreamDestroy(previous_stream));
    CUDA_OK(cudaStreamDestroy(stream));
    ggml_backend_cuda_moe_set_debug_mm(old_debug_mm);
    fprintf(stderr, "test-moe-cache: grouped graph replay lifecycle OK\n");
}

static std::vector<float> run_active_grouped_dispatch(
        ggml_backend_t backend,
        active_grouped_dispatch_graph & graph,
        uint64_t expected_clock,
        bool f3_skipped = false,
        bool down_skipped = false) {
    const auto sentinel = active_grouped_intermediate_sentinel(graph);
    std::vector<float> down_sentinel;
    if (down_skipped) {
        CHECK(graph.output != graph.down_output);
        down_sentinel.resize(ggml_nelements(graph.down_output), -34567.75f);
        ggml_backend_tensor_set(graph.down_output, down_sentinel.data(), 0, ggml_nbytes(graph.down_output));
    }
    CHECK(ggml_backend_graph_compute(backend, graph.graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
    check_active_grouped_intermediates(graph, sentinel, f3_skipped);
    if (down_skipped) {
        std::vector<float> down(ggml_nelements(graph.down_output));
        ggml_backend_tensor_get(graph.down_output, down.data(), 0, ggml_nbytes(graph.down_output));
        CHECK(down == down_sentinel);
    }
    if (expected_clock != 0) {
        auto * context = ggml_cuda_moe_grouped_context_for_test(backend);
        ggml_cuda_moe_candidate_group_key key;
        CHECK(context != nullptr && context->find_down_group_key(graph.down, &key));
        CHECK(ggml_cuda_moe_grouped_context_test_access::has_device_resource(*context, key));
        uint64_t clock_bound = 0;
        CHECK(ggml_cuda_moe_grouped_context_test_access::get_clock_bound(*context, key, &clock_bound));
        CHECK(clock_bound == expected_clock);
        ggml_cuda_moe_grouped_acquisition acquisition;
        ggml_cuda_moe_grouped_resource_info info;
        CHECK(context->acquire_group_resources(key, &acquisition));
        CHECK(context->get_group_resources(acquisition, &info));
        CHECK(!info.transaction_active && info.down == graph.down && info.n_banks == graph.banks.size());
    }
    std::vector<float> output(ggml_nelements(graph.output));
    ggml_backend_tensor_get(graph.output, output.data(), 0, ggml_nbytes(graph.output));
    return output;
}

static void check_active_grouped_routes(
        const active_grouped_dispatch_graph & graph,
        uint32_t n_rows,
        const std::vector<int32_t> & routes) {
    CHECK(routes.size() == n_rows * graph.n_used);
    CHECK(graph.ids != nullptr && graph.ids->view_src != nullptr);
    CHECK(graph.ids->type == GGML_TYPE_I32 && graph.ids->ne[0] == graph.n_used && graph.ids->ne[1] == n_rows);
    CHECK(graph.ids->ne[2] == 1 && graph.ids->ne[3] == 1 && graph.ids->nb[0] == sizeof(int32_t));
    CHECK(graph.ids->nb[1] == graph.n_experts * sizeof(int32_t) && graph.ids->nb[1] > graph.ids->ne[0] * graph.ids->nb[0]);
    ggml_tensor * sorted = graph.ids->view_src;
    CHECK(sorted->type == GGML_TYPE_I32 && sorted->ne[0] == graph.n_experts && sorted->ne[1] == n_rows);
    std::vector<int32_t> actual(ggml_nelements(sorted));
    ggml_backend_tensor_get(sorted, actual.data(), 0, ggml_nbytes(sorted));
    const uint32_t row_stride = graph.ids->nb[1] / sizeof(int32_t);
    for (uint32_t row = 0; row < n_rows; ++row) {
        for (uint32_t route = 0; route < graph.n_used; ++route) {
            CHECK(actual[row * row_stride + route] == routes[row * graph.n_used + route]);
        }
    }
}

static void check_active_grouped_routes(
        const active_grouped_dispatch_graph & graph,
        uint32_t n_rows,
        uint32_t route_variant) {
    const auto routes = active_grouped_routes(graph, route_variant);
    check_active_grouped_routes(graph, n_rows, routes);
    bool has_repeated_expert = false;
    for (uint32_t row = 0; row < n_rows; ++row) {
        for (uint32_t route = 0; route < graph.n_used; ++route) {
            for (uint32_t previous_row = 0; previous_row < row; ++previous_row) {
                for (uint32_t previous_route = 0; previous_route < graph.n_used; ++previous_route) {
                    has_repeated_expert = has_repeated_expert || routes[row * graph.n_used + route] ==
                        routes[previous_row * graph.n_used + previous_route];
                }
            }
        }
    }
    CHECK(n_rows == 1 || has_repeated_expert || graph.n_experts == 256);
}

static void check_active_grouped_exact_output(
        const std::vector<float> & expected,
        const std::vector<float> & actual) {
    CHECK(expected.size() == actual.size());
    double squared_error = 0.0;
    double squared_expected = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        CHECK(std::isfinite(expected[i]) && std::isfinite(actual[i]));
        const double difference = expected[i] - actual[i];
        squared_error += difference * difference;
        squared_expected += static_cast<double>(expected[i]) * expected[i];
    }
    CHECK(squared_expected > 0.0 && squared_error == 0.0);
}

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

static cached_mmid_path_test_graph build_cached_mmid_path_test_graph(
        ggml_backend_t backend,
        ggml_tensor * gate_up,
        ggml_tensor * down,
        int64_t n_used,
        int64_t n_tokens);
static void initialize_cached_mmid_path_test_graphs(
        cached_mmid_path_test_graph & cuda_graph,
        cached_mmid_path_test_graph & reference_graph);
static std::vector<float> run_cached_mmid_path_test(
        ggml_backend_t cached_backend,
        ggml_backend_t reference_backend,
        cached_mmid_path_test_graph & cuda_graph,
        cached_mmid_path_test_graph & reference_graph,
        const std::vector<int32_t> & ids);
static std::vector<float> active_grouped_tensor_values(const ggml_tensor * tensor);
static ggml_cuda_mmid_capability native_mmid_capability(
        int device,
        const ggml_tensor * weight,
        int64_t n_rows,
        ggml_cuda_mmid_mapping mapping);

static void check_active_grouped_capabilities(
        ggml_cuda_moe_grouped_context & context,
        active_grouped_dispatch_graph & graph,
        uint32_t n_slots,
        ggml_cuda_mmid_consumer expected_consumer,
        ggml_cuda_mmid_mapping expected_mapping) {
    (void) candidate_certify_graph(context, graph.graph);
    ggml_cuda_moe_graph_plan plan;
    ggml_cuda_moe_graph_execution execution;
    context.compile_graph_plan(graph.graph, graph.graph->uid, &plan, &execution);
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED && execution.size() == 1);
    for (uint32_t bank = 0; bank < graph.banks.size(); ++bank) {
        const auto capability = ggml_cuda_moe_grouped_context_test_access::graph_bank_capability(plan, 0, bank);
        CHECK(capability.reason == GGML_CUDA_MMID_CAPABILITY_OK &&
            capability.equivalence_reason == GGML_CUDA_MMID_CAPABILITY_OK &&
            capability.consumer == expected_consumer && capability.use_mmq == 1 &&
            capability.mapping == expected_mapping && capability.tensor == graph.banks[bank] &&
            capability.phase == GGML_CUDA_MMID_PHASE_PREFILL &&
            capability.n_experts == graph.n_experts && capability.source_ne[2] == graph.n_experts &&
            capability.grouped_ne[2] == n_slots && capability.top_k == graph.n_used &&
            capability.n_rows == graph.n_rows && capability.n_routes == graph.n_used * graph.n_rows &&
            capability.row_stride == graph.n_experts && capability.n_slots == n_slots);
        for (uint32_t dim = 0; dim < GGML_MAX_DIMS; ++dim) {
            CHECK(capability.source_ne[dim] == graph.banks[bank]->ne[dim] &&
                capability.source_nb[dim] == graph.banks[bank]->nb[dim]);
        }
        CHECK(capability.grouped_ne[0] == capability.source_ne[0] &&
            capability.grouped_ne[1] == capability.source_ne[1] &&
            capability.grouped_nb[2] == capability.source_nb[2] &&
            capability.grouped_nb[3] == capability.grouped_nb[2] * n_slots);
    }
}

static void test_active_grouped_multirow_graph_modes_case(
        int device,
        uint32_t n_rows,
        uint32_t n_experts = 8,
        uint32_t n_used = 2,
        uint32_t n_slots = 12,
        bool test_large_transition = false,
        uint32_t layout = GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP,
        ggml_type type = GGML_TYPE_Q4_0,
        ggml_cuda_mmid_consumer expected_consumer = GGML_CUDA_MMID_CONSUMER_MMVQ,
        ggml_cuda_mmid_mapping expected_mapping = GGML_CUDA_MMID_MAPPING_DIRECT) {
    CHECK(n_rows >= 2 && n_slots > 0 && n_used <= n_slots / n_rows);
    const bool old_debug_mm = ggml_backend_cuda_moe_get_debug_mm();
    ggml_backend_cuda_moe_set_debug_mm(true);
    ggml_backend_ptr reference_backend(ggml_backend_cuda_init(device));
    ggml_backend_ptr candidate_backend(ggml_backend_cuda_init(device));
    CHECK(reference_backend != nullptr && candidate_backend != nullptr);
    auto reference = build_active_grouped_dispatch_graph(
        reference_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), type,
        layout, false, n_rows, n_experts, n_used, 256);
    auto candidate = build_active_grouped_dispatch_graph(
        candidate_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), type,
        layout, false, n_rows, n_experts, n_used, 256);
    initialize_active_grouped_dispatch_graphs({&reference, &candidate});
    const auto disabled = candidate_snapshot(n_slots, nullptr, 0);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v1(reference_backend.get(), &disabled) ==
        GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    register_active_grouped_dispatch(candidate_backend.get(), candidate, layout, n_slots);
    auto * context = ggml_cuda_moe_grouped_context_for_test(candidate_backend.get());
    CHECK(context != nullptr);
    check_active_grouped_capabilities(*context, candidate, n_slots, expected_consumer, expected_mapping);
    const bool test_transition = test_large_transition || (n_slots == 12 && n_rows == 2);
    cached_mmid_path_test_graph reference_prefill;
    cached_mmid_path_test_graph candidate_prefill;
    std::vector<int32_t> prefill_ids;
    if (test_transition) {
        const int64_t prefill_used = n_experts >= 16 ? 8 : n_used;
        reference_prefill = build_cached_mmid_path_test_graph(
            reference_backend.get(), reference.banks[0], reference.banks[1], prefill_used, 2);
        candidate_prefill = build_cached_mmid_path_test_graph(
            candidate_backend.get(), candidate.banks[0], candidate.banks[1], prefill_used, 2);
        initialize_cached_mmid_path_test_graphs(candidate_prefill, reference_prefill);
        candidate_insert_graph_node(candidate_prefill.graph, 0, candidate.graph->nodes[0]);
        CHECK(candidate_prefill.graph->nodes[0] == candidate.graph->nodes[0]);
        prefill_ids.reserve(2 * prefill_used);
        for (int64_t route = 0; route < prefill_used; ++route) {
            prefill_ids.push_back(static_cast<int32_t>(route));
        }
        for (int64_t route = 0; route < prefill_used; ++route) {
            prefill_ids.push_back(static_cast<int32_t>((route + 5) % n_experts));
        }
    }
    std::vector<float> direct_expected;
    std::vector<float> direct_actual;
    uintptr_t captured_graph = 0;
    uintptr_t captured_instance = 0;
    uint64_t captured_semantic_key = 0;
    uint64_t captured_resource_fingerprint = 0;
    const uint64_t n_routes = n_used * n_rows;
    const bool f3_skipped = layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE &&
        expected_consumer == GGML_CUDA_MMID_CONSUMER_MMVQ &&
        candidate.banks[0]->type == candidate.banks[1]->type &&
        ggml_are_same_shape(candidate.banks[0], candidate.banks[1]) &&
        ggml_are_same_stride(candidate.banks[0], candidate.banks[1]);
    bool capture_available = false;
    bool transition_executed = false;
    uint32_t executed_passes = 0;
    for (uint32_t pass = 0; pass < 4; ++pass) {
        if (pass != 0 && !capture_available) {
            break;
        }
        const uint32_t route_variant = pass >= 2 ? pass - 1 : 0;
        if (pass >= 2) {
            set_active_grouped_dispatch_logits({&reference, &candidate}, route_variant);
        }
        const auto expected = run_active_grouped_dispatch(reference_backend.get(), reference, 0, false);
        const uint64_t clock_pass = transition_executed ? pass - 1 : pass + 1;
        const auto actual = run_active_grouped_dispatch(candidate_backend.get(), candidate, n_routes * clock_pass, f3_skipped);
        ++executed_passes;
        check_active_grouped_exact_output(expected, actual);
        if (pass == 0) {
            direct_expected = expected;
            direct_actual = actual;
        } else if (pass == 1) {
            CHECK(expected == direct_expected && actual == direct_actual);
        }
        check_active_grouped_routes(reference, n_rows, route_variant);
        check_active_grouped_routes(candidate, n_rows, route_variant);

        ggml_cuda_graph_capture_state_for_test reference_graph;
        ggml_cuda_graph_capture_state_for_test candidate_graph;
        CHECK(ggml_cuda_graph_capture_state_query_for_test(reference_backend.get(), reference.graph, &reference_graph));
        CHECK(ggml_cuda_graph_capture_state_query_for_test(candidate_backend.get(), candidate.graph, &candidate_graph));
        CHECK(reference_graph.instance == 0 && reference_graph.graph == 0 &&
            !reference_graph.warmup_complete && reference_graph.moe_resource_fingerprint == 0);
        if (pass == 0) {
            capture_available = candidate_graph.capture_available;
            CHECK(candidate_graph.instance == 0 && candidate_graph.graph == 0 &&
                !candidate_graph.warmup_complete && candidate_graph.moe_resource_fingerprint == 0);
        } else if (pass == 1) {
            CHECK(candidate_graph.instance != 0 && candidate_graph.graph != 0 &&
                candidate_graph.warmup_complete && candidate_graph.moe_resource_fingerprint != 0);
            captured_graph = candidate_graph.graph;
            captured_instance = candidate_graph.instance;
            captured_semantic_key = candidate_graph.execution_semantic_key;
            captured_resource_fingerprint = candidate_graph.moe_resource_fingerprint;
            CHECK(captured_semantic_key != 0 && captured_resource_fingerprint != 0);
        } else {
            CHECK(candidate_graph.graph == captured_graph && candidate_graph.instance == captured_instance &&
                candidate_graph.warmup_complete && candidate_graph.moe_resource_fingerprint != 0 &&
                candidate_graph.execution_semantic_key == captured_semantic_key);
            if (transition_executed) {
                ggml_cuda_moe_candidate_group_key key;
                CHECK(context->find_down_group_key(candidate.down, &key));
                CHECK(ggml_cuda_moe_grouped_context_test_access::graph_clock_active(*context, key));
            }
        }
        if (test_transition && !transition_executed && ((capture_available && pass == 1) || (!capture_available && pass == 0))) {
            ggml_cuda_moe_candidate_group_key key;
            CHECK(context->find_down_group_key(candidate.down, &key));
            if (capture_available) {
                CHECK(ggml_cuda_moe_grouped_context_test_access::graph_clock_active(*context, key));
            }
            (void) candidate_certify_graph(*context, candidate_prefill.graph);
            (void) run_cached_mmid_path_test(
                candidate_backend.get(), reference_backend.get(), candidate_prefill, reference_prefill, prefill_ids);
            CHECK(ggml_cuda_moe_grouped_context_test_access::has_device_resource(*context, key));
            auto lease = context->acquire_legacy_cache(candidate.banks[0]);
            CHECK(lease && lease.get() != nullptr &&
                ggml_cuda_moe_cache_slot_ptr(lease.get(), 0) ==
                    ggml_cuda_moe_grouped_context_test_access::device_bank_data(*context, key, candidate.banks[0]));
            if (capture_available) {
                ggml_cuda_graph_capture_state_for_test retained_graph;
                CHECK(ggml_cuda_graph_capture_state_query_for_test(candidate_backend.get(), candidate.graph, &retained_graph));
                CHECK(retained_graph.graph == captured_graph && retained_graph.instance == captured_instance &&
                    retained_graph.warmup_complete && retained_graph.execution_semantic_key == captured_semantic_key &&
                    retained_graph.moe_resource_fingerprint == captured_resource_fingerprint);
            }
            (void) candidate_certify_graph(*context, candidate.graph);
            transition_executed = true;
        }
    }

    if (transition_executed && !capture_available) {
        set_active_grouped_dispatch_logits({&reference, &candidate}, 1);
        const auto expected = run_active_grouped_dispatch(reference_backend.get(), reference, 0, false);
        const auto actual = run_active_grouped_dispatch(candidate_backend.get(), candidate, n_routes, f3_skipped);
        ++executed_passes;
        check_active_grouped_exact_output(expected, actual);
    }

    const uint64_t transition_legacy_ops = transition_executed ? reference.banks.size() : 0;
    CHECK(active_grouped_legacy_op_count(reference_backend.get(), true) +
        active_grouped_legacy_op_count(reference_backend.get(), false) ==
            executed_passes * reference.banks.size() + transition_legacy_ops);
    CHECK(active_grouped_legacy_op_count(candidate_backend.get(), true) == 0 &&
        active_grouped_legacy_op_count(candidate_backend.get(), false) == transition_legacy_ops);
    const auto telemetry = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
    CHECK(telemetry.registered == 1 && telemetry.covered == 1);
    CHECK(executed_passes == (capture_available ? 4u : transition_executed ? 2u : 1u));
    const uint64_t transition_plan_compiles = transition_executed ? 2 : 1;
    CHECK(telemetry.plan_calls == executed_passes && telemetry.plan_compiles == transition_plan_compiles &&
        telemetry.plan_reuses == executed_passes - transition_plan_compiles);
    CHECK(telemetry.calls == executed_passes && telemetry.ready == executed_passes && telemetry.completed == executed_passes);
    CHECK(telemetry.ready_min == executed_passes && telemetry.ready_max == executed_passes);
    CHECK(telemetry.completed_min == executed_passes && telemetry.completed_max == executed_passes);
    CHECK(telemetry.admitted_banks == executed_passes * candidate.banks.size());
    CHECK(telemetry.fallback == 0 && telemetry.rollback == 0);
    CHECK(telemetry.prepare_error == 0 && telemetry.finish_error == 0);

    if (capture_available && test_transition) {
        auto expected = std::vector<float>();
        auto actual = std::vector<float>();
        ggml_cuda_graph_capture_state_for_test invalidated_graph;
        const auto recapture_main = [&]() {
            candidate_stamp_execution(candidate.graph, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
                GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, n_rows, n_rows);
            (void) candidate_certify_graph(*context, candidate.graph);
            for (uint32_t pass = 0; pass < 2; ++pass) {
                expected = run_active_grouped_dispatch(reference_backend.get(), reference, 0, false);
                actual = run_active_grouped_dispatch(candidate_backend.get(), candidate, 0, f3_skipped);
                check_active_grouped_exact_output(expected, actual);
            }
            ggml_cuda_graph_capture_state_for_test state;
            CHECK(ggml_cuda_graph_capture_state_query_for_test(candidate_backend.get(), candidate.graph, &state));
            CHECK(state.graph != 0 && state.instance != 0 && state.warmup_complete && state.moe_resource_fingerprint != 0);
        };

        const auto inventory_shape = [](const ggml_cgraph * graph) {
            std::pair<uint32_t, int32_t> result = {0, -1};
            for (int32_t node_index = 0; node_index < graph->n_nodes; ++node_index) {
                const ggml_tensor * node = graph->nodes[node_index];
                const ggml_tensor * source = node != nullptr && node->op == GGML_OP_MUL_MAT_ID ? node->src[0] : nullptr;
                if (source == nullptr || source->buffer == nullptr ||
                        !ggml_backend_buft_is_cuda_moe_cached(ggml_backend_buffer_get_type(source->buffer))) {
                    continue;
                }
                if (result.second < 0) {
                    result.second = node_index;
                }
                ++result.first;
            }
            return result;
        };
        CHECK(candidate.graph->n_nodes > 1 && candidate.graph->nodes[1]->op == GGML_OP_VIEW);
        candidate_insert_graph_node(candidate_prefill.graph, 1, candidate.graph->nodes[1]);
        const auto certified_inventory = inventory_shape(candidate_prefill.graph);
        const auto stale_coverage = candidate_certify_graph(*context, candidate_prefill.graph);
        CHECK(certified_inventory.first == stale_coverage.mmid_count && stale_coverage.mmid_count == candidate.banks.size());
        ggml_tensor * inventory_padding = candidate_prefill.graph->nodes[1];
        memmove(candidate_prefill.graph->nodes + 1, candidate_prefill.graph->nodes + 2,
            (candidate_prefill.graph->n_nodes - 2) * sizeof(candidate_prefill.graph->nodes[0]));
        candidate_prefill.graph->nodes[candidate_prefill.graph->n_nodes - 1] = inventory_padding;
        candidate_rebuild_graph_uses(candidate_prefill.graph);
        const auto stale_inventory = inventory_shape(candidate_prefill.graph);
        CHECK(candidate_prefill.graph->nodes[0] == candidate.graph->nodes[0] &&
            stale_inventory.first == certified_inventory.first && stale_inventory.second + 1 == certified_inventory.second);
        (void) run_cached_mmid_path_test(
            candidate_backend.get(), reference_backend.get(), candidate_prefill, reference_prefill, prefill_ids);
        CHECK(ggml_cuda_graph_capture_state_query_for_test(candidate_backend.get(), candidate.graph, &invalidated_graph));
        CHECK(invalidated_graph.graph == 0 && invalidated_graph.instance == 0 &&
            !invalidated_graph.warmup_complete && invalidated_graph.moe_resource_fingerprint == 0);

        recapture_main();
        candidate_stamp_execution(candidate.graph, GGML_GRAPH_EXECUTION_DOMAIN_DRAFT,
            GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, n_rows, n_rows);
        (void) candidate_certify_graph(*context, candidate.graph);
        expected = run_active_grouped_dispatch(reference_backend.get(), reference, 0, false);
        actual = run_active_grouped_dispatch(candidate_backend.get(), candidate, 0, f3_skipped);
        check_active_grouped_exact_output(expected, actual);
        CHECK(ggml_cuda_graph_capture_state_query_for_test(candidate_backend.get(), candidate.graph, &invalidated_graph));
        CHECK(invalidated_graph.graph == 0 && invalidated_graph.instance == 0 &&
            !invalidated_graph.warmup_complete && invalidated_graph.moe_resource_fingerprint == 0);

        recapture_main();
        register_active_grouped_dispatch(
            candidate_backend.get(), candidate, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, n_slots);
        (void) candidate_certify_graph(*context, candidate.graph);
        expected = run_active_grouped_dispatch(reference_backend.get(), reference, 0, false);
        actual = run_active_grouped_dispatch(candidate_backend.get(), candidate, 0, false);
        check_active_grouped_exact_output(expected, actual);
        CHECK(ggml_cuda_graph_capture_state_query_for_test(candidate_backend.get(), candidate.graph, &invalidated_graph));
        CHECK(invalidated_graph.graph == 0 && invalidated_graph.instance == 0 &&
            !invalidated_graph.warmup_complete && invalidated_graph.moe_resource_fingerprint == 0);
    }
    ggml_backend_cuda_moe_set_debug_mm(old_debug_mm);
    fprintf(stderr, capture_available ?
        "test-moe-cache: active grouped E%u K%u B%u direct/capture/replay dynamic routes OK\n" :
        "test-moe-cache: active grouped E%u K%u B%u direct OK (CUDA graphs unavailable)\n",
        n_experts, n_used, n_rows);
}

static bool test_active_grouped_q4k_eviction_refill_case(int device, uint32_t primary_rows) {
    constexpr uint32_t n_experts = 256;
    constexpr uint32_t n_used = 8;
    constexpr uint32_t n_slots = 132;
    constexpr uint32_t b5_rows = 5;
    CHECK(primary_rows == 8 || primary_rows == 16);

    const bool old_debug_mm = ggml_backend_cuda_moe_get_debug_mm();
    ggml_backend_cuda_moe_set_debug_mm(true);
    ggml_backend_ptr direct_backend(ggml_backend_cuda_init(device));
    ggml_backend_ptr legacy_backend(ggml_backend_cuda_init(device));
    ggml_backend_ptr grouped_backend(ggml_backend_cuda_init(device));
    CHECK(direct_backend != nullptr && legacy_backend != nullptr && grouped_backend != nullptr);

    auto direct_primary = build_active_grouped_dispatch_graph(
        direct_backend.get(), ggml_backend_cuda_buffer_type(device), GGML_TYPE_Q4_K,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, false, primary_rows, n_experts, n_used);
    auto direct_b5 = build_active_grouped_dispatch_graph(
        direct_backend.get(), ggml_backend_cuda_buffer_type(device), GGML_TYPE_Q4_K,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, false, b5_rows, n_experts, n_used, 256, &direct_primary);
    auto legacy_primary = build_active_grouped_dispatch_graph(
        legacy_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_K,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, false, primary_rows, n_experts, n_used);
    auto legacy_b5 = build_active_grouped_dispatch_graph(
        legacy_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_K,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, false, b5_rows, n_experts, n_used, 256, &legacy_primary);
    auto grouped_primary = build_active_grouped_dispatch_graph(
        grouped_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_K,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, false, primary_rows, n_experts, n_used);
    auto grouped_b5 = build_active_grouped_dispatch_graph(
        grouped_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_K,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, false, b5_rows, n_experts, n_used, 256, &grouped_primary);

    std::vector<std::vector<uint8_t>> bank_data;
    bank_data.reserve(direct_primary.banks.size());
    for (uint32_t bank = 0; bank < direct_primary.banks.size(); ++bank) {
        bank_data.push_back(active_grouped_q4k_expert_data(direct_primary.banks[bank], 211 + bank));
    }
    initialize_active_grouped_dispatch_graphs({&direct_primary, &legacy_primary, &grouped_primary}, &bank_data);
    initialize_active_grouped_dispatch_graphs({&direct_b5, &legacy_b5, &grouped_b5}, &bank_data);
    for (const auto * graph : {&direct_primary, &legacy_primary, &grouped_primary, &direct_b5, &legacy_b5, &grouped_b5}) {
        CHECK(graph->output == graph->down_output);
    }

    const auto disabled = candidate_snapshot(n_slots, nullptr, 0);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v1(legacy_backend.get(), &disabled) ==
        GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    register_active_grouped_dispatch(
        grouped_backend.get(), grouped_primary, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, n_slots);
    auto * context = ggml_cuda_moe_grouped_context_for_test(grouped_backend.get());
    CHECK(context != nullptr);
    const auto primary_native = native_mmid_capability(device, direct_primary.down, primary_rows,
        primary_rows == 8 ? GGML_CUDA_MMID_MAPPING_DIRECT : GGML_CUDA_MMID_MAPPING_SOURCE_MAP);
    const auto b5_native = native_mmid_capability(device, direct_b5.down, b5_rows, GGML_CUDA_MMID_MAPPING_DIRECT);
    const bool primary_supported = primary_native.reason == GGML_CUDA_MMID_CAPABILITY_OK &&
        primary_native.selection == (primary_rows == 8 ? GGML_CUDA_MMID_CONSUMER_MMVQ : GGML_CUDA_MMID_CONSUMER_MMQ);
    const bool b5_supported = b5_native.reason == GGML_CUDA_MMID_CAPABILITY_OK &&
        b5_native.selection == GGML_CUDA_MMID_CONSUMER_MMVQ;
    if (primary_rows == 8 && (!primary_supported || !b5_supported)) {
        ggml_backend_cuda_moe_set_debug_mm(old_debug_mm);
        fprintf(stderr, "test-moe-cache: active grouped Q4_K cache132 B%u/B5 skipped (native primary=%u B5=%u)\n",
            primary_rows, primary_supported, b5_supported);
        return false;
    }
    CHECK(primary_supported && b5_supported);
    check_active_grouped_capabilities(
        *context, grouped_primary, n_slots,
        primary_rows == 8 ? GGML_CUDA_MMID_CONSUMER_MMVQ : GGML_CUDA_MMID_CONSUMER_MMQ,
        primary_rows == 8 ? GGML_CUDA_MMID_MAPPING_DIRECT : GGML_CUDA_MMID_MAPPING_SOURCE_MAP);
    check_active_grouped_capabilities(
        *context, grouped_b5, n_slots, GGML_CUDA_MMID_CONSUMER_MMVQ, GGML_CUDA_MMID_MAPPING_DIRECT);

    struct host_cache_state {
        std::vector<int32_t> slot_expert;
        std::vector<int32_t> expert_slot;
        std::vector<uint64_t> last_used;
        uint64_t clock = 0;
    } host = {
        std::vector<int32_t>(n_slots, -1),
        std::vector<int32_t>(n_experts, -1),
        std::vector<uint64_t>(n_slots, 0),
        0,
    };
    const auto plan_routes = [&](const std::vector<int32_t> & routes) {
        CHECK(!routes.empty() && routes.size() <= n_slots && routes.size() % n_used == 0);
        std::vector<int32_t> unique;
        for (uint32_t route = 0; route < routes.size(); ++route) {
            CHECK(routes[route] >= 0 && static_cast<uint32_t>(routes[route]) < n_experts);
            auto found = std::find(unique.begin(), unique.end(), routes[route]);
            if (found == unique.end()) {
                unique.push_back(routes[route]);
            }
        }
        std::vector<int32_t> unique_slots(unique.size(), -1);
        std::vector<bool> reserved(n_slots, false);
        for (uint32_t unique_index = 0; unique_index < unique.size(); ++unique_index) {
            const int32_t slot = host.expert_slot[unique[unique_index]];
            unique_slots[unique_index] = slot;
            if (slot >= 0) {
                CHECK(static_cast<uint32_t>(slot) < n_slots && host.slot_expert[slot] == unique[unique_index]);
                reserved[slot] = true;
            }
        }
        uint32_t misses = 0;
        for (uint32_t unique_index = 0; unique_index < unique.size(); ++unique_index) {
            if (unique_slots[unique_index] >= 0) {
                continue;
            }
            uint64_t best_age = UINT64_MAX;
            uint32_t best_slot = UINT32_MAX;
            for (uint32_t slot = 0; slot < n_slots; ++slot) {
                if (reserved[slot]) {
                    continue;
                }
                const uint64_t age = host.slot_expert[slot] < 0 ? 0 : host.last_used[slot];
                if (age < best_age || (age == best_age && slot < best_slot)) {
                    best_age = age;
                    best_slot = slot;
                }
            }
            CHECK(best_slot < n_slots);
            unique_slots[unique_index] = best_slot;
            reserved[best_slot] = true;
            ++misses;
        }
        for (uint32_t unique_index = 0; unique_index < unique.size(); ++unique_index) {
            const int32_t expert = unique[unique_index];
            const int32_t slot = unique_slots[unique_index];
            if (host.expert_slot[expert] == slot) {
                continue;
            }
            const int32_t old_expert = host.slot_expert[slot];
            if (old_expert >= 0) {
                host.expert_slot[old_expert] = -1;
            }
            host.slot_expert[slot] = expert;
            host.expert_slot[expert] = slot;
        }
        for (uint32_t unique_index = 0; unique_index < unique.size(); ++unique_index) {
            host.last_used[unique_slots[unique_index]] = host.clock + unique_index + 1;
        }
        host.clock += routes.size();
        return misses;
    };

    struct active_values {
        std::vector<float> gate;
        std::vector<float> up;
        std::vector<float> output;
    };
    const auto run_graph = [&](ggml_backend_t backend, active_grouped_dispatch_graph & graph,
                               uint64_t expected_clock, bool f3_skipped) {
        active_values values;
        values.output = run_active_grouped_dispatch(backend, graph, expected_clock, f3_skipped);
        if (!f3_skipped) {
            values.gate = active_grouped_tensor_values(graph.gate_output);
            values.up = active_grouped_tensor_values(graph.up_output);
        }
        return values;
    };
    const auto check_values = [](const active_values & expected, const active_values & actual) {
        if (!expected.gate.empty() && !actual.gate.empty()) {
            check_active_grouped_exact_output(expected.gate, actual.gate);
            check_active_grouped_exact_output(expected.up, actual.up);
        }
        check_active_grouped_exact_output(expected.output, actual.output);
    };
    ggml_cuda_moe_candidate_group_key group_key;
    CHECK(context->find_down_group_key(grouped_primary.down, &group_key));
    struct capture_state {
        uint32_t passes = 0;
        uintptr_t graph = 0;
        uintptr_t instance = 0;
        uint64_t semantic_key = 0;
        uint64_t resource_fingerprint = 0;
    } primary_capture, b5_capture;
    bool capture_available = false;
    uint64_t total_misses = 0;
    uint32_t total_passes = 0;
    const auto run_step = [&](const std::array<active_grouped_dispatch_graph *, 3> & graphs,
            const std::vector<int32_t> & routes, uint32_t expected_misses, capture_state & captured) {
        const std::vector<int32_t> committed_slots = host.slot_expert;
        const uint32_t misses = plan_routes(routes);
        CHECK(misses == expected_misses);
        total_misses += misses;
        ++total_passes;
        set_active_grouped_dispatch_routes(
            std::vector<active_grouped_dispatch_graph *>(graphs.begin(), graphs.end()), routes);
        const auto direct_capability = native_mmid_capability(
            device, graphs[0]->banks[0], graphs[0]->n_rows, GGML_CUDA_MMID_MAPPING_DIRECT);
        const bool direct_f3_skipped = direct_capability.reason == GGML_CUDA_MMID_CAPABILITY_OK &&
            direct_capability.selection == GGML_CUDA_MMID_CONSUMER_MMVQ;
        const bool grouped_f3_skipped = graphs[2]->n_rows == primary_rows ?
            primary_native.selection == GGML_CUDA_MMID_CONSUMER_MMVQ :
            b5_native.selection == GGML_CUDA_MMID_CONSUMER_MMVQ;
        const auto direct = run_graph(direct_backend.get(), *graphs[0], 0, direct_f3_skipped);
        const auto legacy = run_graph(legacy_backend.get(), *graphs[1], 0, false);
        const auto grouped = run_graph(grouped_backend.get(), *graphs[2], host.clock, grouped_f3_skipped);
        check_values(direct, legacy);
        check_values(direct, grouped);
        for (auto * graph : graphs) {
            CHECK(graph->ids->data == static_cast<const char *>(graph->ids->view_src->data) + graph->ids->view_offs);
            check_active_grouped_routes(*graph, graph->n_rows, routes);
        }
        for (uint32_t expert = 0; expert < n_experts; ++expert) {
            int32_t expected_slot = -1;
            for (uint32_t slot = 0; slot < n_slots; ++slot) {
                if (committed_slots[slot] == static_cast<int32_t>(expert)) {
                    expected_slot = slot;
                    break;
                }
            }
            CHECK(ggml_cuda_moe_grouped_context_test_access::device_slot_for_expert(*context, group_key, expert) == expected_slot);
        }
        for (uint32_t bank = 0; bank < grouped_primary.banks.size(); ++bank) {
            const ggml_tensor * source = grouped_primary.banks[bank];
            const void * device_data = ggml_cuda_moe_grouped_context_test_access::device_bank_data(
                *context, group_key, source);
            CHECK(device_data != nullptr);
            std::vector<uint8_t> physical(n_slots * source->nb[2]);
            CUDA_OK(cudaMemcpy(physical.data(), device_data, physical.size(), cudaMemcpyDeviceToHost));
            for (uint32_t slot = 0; slot < n_slots; ++slot) {
                const int32_t expert = host.slot_expert[slot];
                if (expert < 0) {
                    continue;
                }
                CHECK(memcmp(physical.data() + slot * source->nb[2],
                    static_cast<const uint8_t *>(source->data) + expert * source->nb[2], source->nb[2]) == 0);
            }
        }

        ggml_cuda_graph_capture_state_for_test state;
        CHECK(ggml_cuda_graph_capture_state_query_for_test(grouped_backend.get(), graphs[2]->graph, &state));
        if (total_passes == 1) {
            capture_available = state.capture_available;
        } else {
            CHECK(state.capture_available == capture_available);
        }
        ++captured.passes;
        if (capture_available && captured.passes == 1) {
            CHECK(state.graph == 0 && state.instance == 0 && !state.warmup_complete && state.moe_resource_fingerprint == 0);
        } else if (capture_available && captured.passes == 2) {
            CHECK(state.graph != 0 && state.instance != 0 && state.warmup_complete &&
                state.execution_semantic_key != 0 && state.moe_resource_fingerprint != 0);
            captured.graph = state.graph;
            captured.instance = state.instance;
            captured.semantic_key = state.execution_semantic_key;
            captured.resource_fingerprint = state.moe_resource_fingerprint;
        } else if (capture_available) {
            CHECK(state.graph == captured.graph && state.instance == captured.instance && state.warmup_complete &&
                state.execution_semantic_key == captured.semantic_key &&
                state.moe_resource_fingerprint == captured.resource_fingerprint);
        }
    };

    std::vector<int32_t> wave_a(primary_rows * n_used);
    std::vector<int32_t> wave_b(primary_rows * n_used);
    std::vector<int32_t> wave_c(primary_rows * n_used);
    std::iota(wave_a.begin(), wave_a.end(), 0);
    std::iota(wave_b.begin(), wave_b.end(), 64);
    std::iota(wave_c.begin(), wave_c.end(), 128);
    std::vector<int32_t> b5_a(b5_rows * n_used);
    std::iota(b5_a.begin(), b5_a.end(), 0);
    std::vector<int32_t> mixed(primary_rows * n_used);
    const uint32_t mixed_half = mixed.size() / 2;
    const uint32_t mixed_miss_base = primary_rows == 8 ? 192 : 128;
    for (uint32_t route = 0; route < mixed_half; ++route) {
        mixed[route] = route;
        mixed[mixed_half + route] = mixed_miss_base + route;
    }

    const uint32_t primary_routes = primary_rows * n_used;
    run_step({&direct_primary, &legacy_primary, &grouped_primary}, wave_a, primary_routes, primary_capture);
    run_step({&direct_primary, &legacy_primary, &grouped_primary}, wave_a, 0, primary_capture);
    run_step({&direct_primary, &legacy_primary, &grouped_primary}, wave_a, 0, primary_capture);
    run_step({&direct_primary, &legacy_primary, &grouped_primary}, wave_b, 64, primary_capture);
    run_step({&direct_primary, &legacy_primary, &grouped_primary}, wave_c, 64, primary_capture);
    run_step({&direct_b5, &legacy_b5, &grouped_b5}, b5_a, 40, b5_capture);
    run_step({&direct_b5, &legacy_b5, &grouped_b5}, b5_a, 0, b5_capture);
    run_step({&direct_b5, &legacy_b5, &grouped_b5}, b5_a, 0, b5_capture);
    run_step({&direct_primary, &legacy_primary, &grouped_primary}, wave_a,
        primary_rows == 8 ? 24 : 88, primary_capture);
    run_step({&direct_primary, &legacy_primary, &grouped_primary}, mixed, mixed_half, primary_capture);
    run_step({&direct_primary, &legacy_primary, &grouped_primary}, mixed, 0, primary_capture);
    CHECK(total_passes == 11 && total_misses == (primary_rows == 8 ? 288 : 448));
    CHECK(host.clock == 8 * primary_routes + 3 * b5_rows * n_used);

    CHECK(ggml_cuda_moe_grouped_context_for_test(direct_backend.get()) == nullptr);
    CHECK(active_grouped_legacy_op_count(legacy_backend.get(), true) +
        active_grouped_legacy_op_count(legacy_backend.get(), false) == total_passes * legacy_primary.banks.size());
    CHECK(active_grouped_legacy_op_count(grouped_backend.get(), true) == 0 &&
        active_grouped_legacy_op_count(grouped_backend.get(), false) == 0);
    const auto telemetry = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
    CHECK(telemetry.registered == 1 && telemetry.covered == 1 && telemetry.plan_calls == total_passes &&
        telemetry.plan_compiles == 2 && telemetry.plan_reuses == total_passes - telemetry.plan_compiles &&
        telemetry.calls == total_passes && telemetry.ready == total_passes && telemetry.completed == total_passes &&
        telemetry.ready_min == total_passes && telemetry.ready_max == total_passes &&
        telemetry.completed_min == total_passes && telemetry.completed_max == total_passes &&
        telemetry.admitted_banks == total_passes * grouped_primary.banks.size());
    CHECK(telemetry.h2d_banks == total_misses * grouped_primary.banks.size());
    uint64_t bytes_per_expert = 0;
    for (const ggml_tensor * bank : grouped_primary.banks) {
        bytes_per_expert += bank->nb[2];
    }
    CHECK(telemetry.h2d_bytes == total_misses * bytes_per_expert && telemetry.fallback == 0 &&
        telemetry.rollback == 0 && telemetry.prepare_error == 0 && telemetry.finish_error == 0);
    ggml_backend_cuda_moe_set_debug_mm(old_debug_mm);
    fprintf(stderr, capture_available ?
        "test-moe-cache: active grouped Q4_K cache132 B%u/B5 eviction/refill capture exact OK\n" :
        "test-moe-cache: active grouped Q4_K cache132 B%u/B5 eviction/refill direct exact OK (CUDA graphs unavailable)\n",
        primary_rows);
    return true;
}

static void test_active_grouped_q4k_eviction_refill(int device) {
    const bool b8 = test_active_grouped_q4k_eviction_refill_case(device, 8);
    const bool b16 = test_active_grouped_q4k_eviction_refill_case(device, 16);
    fprintf(stderr, "test-moe-cache: active grouped Q4_K natural cases B8=%s B16=%s\n",
        b8 ? "executed" : "skipped", b16 ? "executed" : "skipped");
}

static void test_active_grouped_same_key_row_transitions(int device, uint32_t n_slots) {
    const bool old_debug_mm = ggml_backend_cuda_moe_get_debug_mm();
    ggml_backend_cuda_moe_set_debug_mm(true);
    ggml_backend_ptr reference_backend(ggml_backend_cuda_init(device));
    ggml_backend_ptr candidate_backend(ggml_backend_cuda_init(device));
    CHECK(reference_backend != nullptr && candidate_backend != nullptr);

    auto candidate_b4 = build_active_grouped_dispatch_graph(
        candidate_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, false, 4);
    auto candidate_b1 = build_active_grouped_dispatch_graph(
        candidate_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, false, 1, 8, 2, 256, &candidate_b4);
    auto candidate_b2 = build_active_grouped_dispatch_graph(
        candidate_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, false, 2, 8, 2, 256, &candidate_b4);
    auto candidate_b3 = build_active_grouped_dispatch_graph(
        candidate_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, false, 3, 8, 2, 256, &candidate_b4);
    auto reference_b1 = build_active_grouped_dispatch_graph(
        reference_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, false, 1);
    auto reference_b2 = build_active_grouped_dispatch_graph(
        reference_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, false, 2);
    auto reference_b3 = build_active_grouped_dispatch_graph(
        reference_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, false, 3);
    auto reference_b4 = build_active_grouped_dispatch_graph(
        reference_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, false, 4);

    initialize_active_grouped_dispatch_graphs({&reference_b1, &candidate_b1});
    initialize_active_grouped_dispatch_graphs({&reference_b2, &candidate_b2});
    initialize_active_grouped_dispatch_graphs({&reference_b3, &candidate_b3});
    initialize_active_grouped_dispatch_graphs({&reference_b4, &candidate_b4});
    const auto disabled = candidate_snapshot(n_slots, nullptr, 0);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v1(reference_backend.get(), &disabled) ==
        GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    register_active_grouped_dispatch(
        candidate_backend.get(), candidate_b4, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, n_slots);

    ggml_tensor * shared_first_node = candidate_b4.graph->nodes[0];
    CHECK(shared_first_node != nullptr && shared_first_node->op != GGML_OP_MUL_MAT_ID);
    candidate_insert_graph_node(candidate_b1.graph, 0, shared_first_node);
    candidate_insert_graph_node(candidate_b2.graph, 0, shared_first_node);
    candidate_insert_graph_node(candidate_b3.graph, 0, shared_first_node);
    CHECK(candidate_b1.graph->nodes[0] == shared_first_node && candidate_b2.graph->nodes[0] == shared_first_node &&
        candidate_b3.graph->nodes[0] == shared_first_node && candidate_b4.graph->nodes[0] == shared_first_node);

    auto * context = ggml_cuda_moe_grouped_context_for_test(candidate_backend.get());
    CHECK(context != nullptr);
    std::array<active_grouped_dispatch_graph *, 4> references = {
        &reference_b1, &reference_b2, &reference_b3, &reference_b4,
    };
    std::array<active_grouped_dispatch_graph *, 4> candidates = {
        &candidate_b1, &candidate_b2, &candidate_b3, &candidate_b4,
    };
    for (const auto * candidate : candidates) {
        CHECK(candidate->banks == candidate_b4.banks && candidate->down == candidate_b4.down);
    }
    std::array<uint64_t, 4> semantic_keys = {};
    for (uint32_t row_index = 0; row_index < candidates.size(); ++row_index) {
        semantic_keys[row_index] = ggml_cuda_moe_execution_semantic_key(candidates[row_index]->graph);
        CHECK(semantic_keys[row_index] != 0);
        for (uint32_t previous = 0; previous < row_index; ++previous) {
            CHECK(semantic_keys[previous] != semantic_keys[row_index]);
        }
    }

    uint32_t executed_passes = 0;
    const auto run_checked = [&](uint32_t n_rows, uint32_t route_variant) {
        CHECK(n_rows >= 1 && n_rows <= 4 && route_variant < 3);
        auto & reference = *references[n_rows - 1];
        auto & candidate = *candidates[n_rows - 1];
        set_active_grouped_dispatch_logits({&reference, &candidate}, route_variant);
        std::vector<float> sentinel(ggml_nelements(candidate.output), -4096.0f - n_rows);
        ggml_backend_tensor_set(candidate.output, sentinel.data(), 0, sentinel.size() * sizeof(float));
        const auto expected = run_active_grouped_dispatch(reference_backend.get(), reference, 0, false);
        const auto actual = run_active_grouped_dispatch(candidate_backend.get(), candidate, 0, false);
        ++executed_passes;
        check_active_grouped_exact_output(expected, actual);
        check_active_grouped_routes(reference, n_rows, route_variant);
        check_active_grouped_routes(candidate, n_rows, route_variant);
    };

    (void) candidate_certify_graph(*context, candidate_b4.graph);
    run_checked(4, 0);
    ggml_cuda_moe_candidate_group_key group_key;
    CHECK(context->find_down_group_key(candidate_b4.down, &group_key));
    ggml_cuda_moe_grouped_acquisition initial_resource;
    ggml_cuda_moe_grouped_resource_info initial_info;
    CHECK(context->acquire_group_resources(group_key, &initial_resource) &&
        context->get_group_resources(initial_resource, &initial_info));
    CHECK(initial_info.down == candidate_b4.down && initial_info.n_slots == n_slots &&
        initial_info.n_banks == candidate_b4.banks.size() && !initial_info.transaction_active);
    const uint64_t resource_generation = initial_resource.resource_generation;
    std::array<void *, 2> bank_data = {
        ggml_cuda_moe_grouped_context_test_access::device_bank_data(*context, group_key, candidate_b4.banks[0]),
        ggml_cuda_moe_grouped_context_test_access::device_bank_data(*context, group_key, candidate_b4.banks[1]),
    };
    CHECK(resource_generation != 0 && bank_data[0] != nullptr && bank_data[1] != nullptr);
    const auto check_resource_identity = [&]() {
        ggml_cuda_moe_grouped_acquisition resource;
        ggml_cuda_moe_grouped_resource_info info;
        CHECK(context->acquire_group_resources(group_key, &resource) && context->get_group_resources(resource, &info));
        CHECK(resource.resource_generation == resource_generation && info.down == candidate_b4.down &&
            info.n_slots == n_slots && info.n_banks == candidate_b4.banks.size() && !info.transaction_active);
        CHECK(ggml_cuda_moe_grouped_context_test_access::device_bank_data(
                *context, group_key, candidate_b4.banks[0]) == bank_data[0] &&
            ggml_cuda_moe_grouped_context_test_access::device_bank_data(
                *context, group_key, candidate_b4.banks[1]) == bank_data[1]);
    };
    ggml_cuda_graph_capture_state_for_test state;
    CHECK(ggml_cuda_graph_capture_state_query_for_test(candidate_backend.get(), candidate_b4.graph, &state));
    const bool capture_available = state.capture_available;
    CHECK(state.graph == 0 && state.instance == 0 && !state.warmup_complete && state.moe_resource_fingerprint == 0 &&
        (!capture_available || state.execution_semantic_key == semantic_keys[3]));

    if (!capture_available) {
        for (uint32_t n_rows : {3u, 2u, 1u, 4u}) {
            (void) candidate_certify_graph(*context, candidates[n_rows - 1]->graph);
            run_checked(n_rows, n_rows % 3);
            check_resource_identity();
        }
    } else {
        run_checked(4, 0);
        CHECK(ggml_cuda_graph_capture_state_query_for_test(candidate_backend.get(), candidate_b4.graph, &state));
        CHECK(state.graph != 0 && state.instance != 0 && state.warmup_complete &&
            state.execution_semantic_key == semantic_keys[3] && state.moe_resource_fingerprint != 0);
        uintptr_t captured_graph = state.graph;
        uintptr_t captured_instance = state.instance;
        run_checked(4, 1);
        CHECK(ggml_cuda_graph_capture_state_query_for_test(candidate_backend.get(), candidate_b4.graph, &state));
        CHECK(state.graph == captured_graph && state.instance == captured_instance && state.warmup_complete &&
            state.execution_semantic_key == semantic_keys[3] && state.moe_resource_fingerprint != 0);
        const uint64_t b4_resource_fingerprint = state.moe_resource_fingerprint;

        uint64_t previous_semantic_key = semantic_keys[3];
        const std::array<uint32_t, 4> transitions = {3, 2, 1, 4};
        for (uint32_t transition = 0; transition < transitions.size(); ++transition) {
            const uint32_t n_rows = transitions[transition];
            auto * graph = candidates[n_rows - 1]->graph;
            const uint64_t semantic_key = semantic_keys[n_rows - 1];
            CHECK(semantic_key != previous_semantic_key);
            (void) candidate_certify_graph(*context, graph);
            const uint32_t direct_variant = (transition + 2) % 3;
            run_checked(n_rows, direct_variant);
            check_resource_identity();
            CHECK(ggml_cuda_graph_capture_state_query_for_test(candidate_backend.get(), graph, &state));
            CHECK(state.graph == 0 && state.instance == 0 && !state.warmup_complete &&
                state.execution_semantic_key == semantic_key && state.moe_resource_fingerprint == 0);
            if (n_rows == 4) {
                run_checked(4, direct_variant);
                CHECK(ggml_cuda_graph_capture_state_query_for_test(candidate_backend.get(), graph, &state));
                CHECK(state.graph != 0 && state.instance != 0 && state.warmup_complete &&
                    state.execution_semantic_key == semantic_keys[3] &&
                    state.moe_resource_fingerprint == b4_resource_fingerprint);
                captured_graph = state.graph;
                captured_instance = state.instance;
                run_checked(4, (direct_variant + 1) % 3);
                CHECK(ggml_cuda_graph_capture_state_query_for_test(candidate_backend.get(), graph, &state));
                CHECK(state.graph == captured_graph && state.instance == captured_instance && state.warmup_complete &&
                    state.execution_semantic_key == semantic_keys[3] &&
                    state.moe_resource_fingerprint == b4_resource_fingerprint);
            }
            previous_semantic_key = semantic_key;
        }
        CHECK(previous_semantic_key == semantic_keys[3]);
    }

    CHECK(active_grouped_legacy_op_count(reference_backend.get(), true) +
        active_grouped_legacy_op_count(reference_backend.get(), false) == executed_passes * reference_b1.banks.size());
    CHECK(active_grouped_legacy_op_count(candidate_backend.get(), true) == 0 &&
        active_grouped_legacy_op_count(candidate_backend.get(), false) == 0);
    const auto telemetry = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
    CHECK(telemetry.registered == 1 && telemetry.covered == 1 && telemetry.plan_calls == executed_passes &&
        telemetry.plan_compiles == 5 && telemetry.plan_reuses == executed_passes - telemetry.plan_compiles &&
        telemetry.calls == executed_passes &&
        telemetry.ready == executed_passes && telemetry.completed == executed_passes &&
        telemetry.admitted_banks == executed_passes * candidate_b1.banks.size());
    CHECK(telemetry.ready_min == executed_passes && telemetry.ready_max == executed_passes &&
        telemetry.completed_min == executed_passes && telemetry.completed_max == executed_passes);
    CHECK(telemetry.fallback == 0 && telemetry.rollback == 0 && telemetry.prepare_error == 0 && telemetry.finish_error == 0);
    ggml_backend_cuda_moe_set_debug_mm(old_debug_mm);
    fprintf(stderr, capture_available ?
        "test-moe-cache: shared-bank cache%u B4/B3/B2/B1 capture transition OK\n" :
        "test-moe-cache: shared-bank cache%u B4/B3/B2/B1 direct transition OK (CUDA graphs unavailable)\n",
        n_slots);
}

static void test_active_grouped_cache12_route_limit_transition(int device) {
    const bool old_debug_mm = ggml_backend_cuda_moe_get_debug_mm();
    ggml_backend_cuda_moe_set_debug_mm(true);
    ggml_backend_ptr reference_backend(ggml_backend_cuda_init(device));
    ggml_backend_ptr candidate_backend(ggml_backend_cuda_init(device));
    CHECK(reference_backend != nullptr && candidate_backend != nullptr);

    auto candidate_b1 = build_active_grouped_dispatch_graph(
        candidate_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, false, 1, 128, 8);
    auto candidate_b2 = build_active_grouped_dispatch_graph(
        candidate_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, false, 2, 128, 8, 256, &candidate_b1);
    auto reference_b1 = build_active_grouped_dispatch_graph(
        reference_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, false, 1, 128, 8);
    auto reference_b2 = build_active_grouped_dispatch_graph(
        reference_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, false, 2, 128, 8);
    initialize_active_grouped_dispatch_graphs({&reference_b1, &candidate_b1});
    initialize_active_grouped_dispatch_graphs({&reference_b2, &candidate_b2});
    const auto disabled = candidate_snapshot(12, nullptr, 0);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v1(reference_backend.get(), &disabled) ==
        GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    register_active_grouped_dispatch(
        candidate_backend.get(), candidate_b1, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 12);
    CHECK(candidate_b2.banks == candidate_b1.banks && candidate_b2.down == candidate_b1.down);

    ggml_tensor * shared_first_node = candidate_b1.graph->nodes[0];
    CHECK(shared_first_node != nullptr && shared_first_node->op != GGML_OP_MUL_MAT_ID);
    candidate_insert_graph_node(candidate_b2.graph, 0, shared_first_node);
    CHECK(candidate_b2.graph->nodes[0] == candidate_b1.graph->nodes[0]);
    auto * context = ggml_cuda_moe_grouped_context_for_test(candidate_backend.get());
    CHECK(context != nullptr);

    uint32_t executed_passes = 0;
    const auto run_checked = [&](active_grouped_dispatch_graph & reference,
                                 active_grouped_dispatch_graph & candidate,
                                 uint32_t route_variant) {
        set_active_grouped_dispatch_logits({&reference, &candidate}, route_variant);
        std::vector<float> sentinel(ggml_nelements(candidate.output), -8192.0f - candidate.n_rows);
        ggml_backend_tensor_set(candidate.output, sentinel.data(), 0, sentinel.size() * sizeof(float));
        const auto expected = run_active_grouped_dispatch(reference_backend.get(), reference, 0, false);
        const auto actual = run_active_grouped_dispatch(candidate_backend.get(), candidate, 0, false);
        ++executed_passes;
        check_active_grouped_exact_output(expected, actual);
        check_active_grouped_routes(reference, reference.n_rows, route_variant);
        check_active_grouped_routes(candidate, candidate.n_rows, route_variant);
    };

    (void) candidate_certify_graph(*context, candidate_b1.graph);
    run_checked(reference_b1, candidate_b1, 0);
    ggml_cuda_moe_candidate_group_key group_key;
    ggml_cuda_moe_grouped_acquisition initial_resource;
    CHECK(context->find_down_group_key(candidate_b1.down, &group_key) &&
        context->acquire_group_resources(group_key, &initial_resource));
    const uint64_t resource_generation = initial_resource.resource_generation;
    std::array<void *, 2> bank_data = {
        ggml_cuda_moe_grouped_context_test_access::device_bank_data(*context, group_key, candidate_b1.banks[0]),
        ggml_cuda_moe_grouped_context_test_access::device_bank_data(*context, group_key, candidate_b1.banks[1]),
    };
    CHECK(resource_generation != 0 && bank_data[0] != nullptr && bank_data[1] != nullptr);
    ggml_cuda_graph_capture_state_for_test state;
    CHECK(ggml_cuda_graph_capture_state_query_for_test(candidate_backend.get(), candidate_b1.graph, &state));
    const bool capture_available = state.capture_available;
    uint64_t b1_resource_fingerprint = 0;
    if (capture_available) {
        run_checked(reference_b1, candidate_b1, 0);
        run_checked(reference_b1, candidate_b1, 1);
        CHECK(ggml_cuda_graph_capture_state_query_for_test(candidate_backend.get(), candidate_b1.graph, &state));
        CHECK(state.graph != 0 && state.instance != 0 && state.warmup_complete && state.moe_resource_fingerprint != 0);
        b1_resource_fingerprint = state.moe_resource_fingerprint;
    }

    (void) candidate_certify_graph(*context, candidate_b2.graph);
    run_checked(reference_b2, candidate_b2, 2);
    CHECK(active_grouped_legacy_op_count(candidate_backend.get(), true) +
        active_grouped_legacy_op_count(candidate_backend.get(), false) == candidate_b1.banks.size());
    if (capture_available) {
        CHECK(ggml_cuda_graph_capture_state_query_for_test(candidate_backend.get(), candidate_b2.graph, &state));
        CHECK(state.graph == 0 && state.instance == 0 && !state.warmup_complete && state.moe_resource_fingerprint == 0);
    }

    (void) candidate_certify_graph(*context, candidate_b1.graph);
    run_checked(reference_b1, candidate_b1, 2);
    if (capture_available) {
        run_checked(reference_b1, candidate_b1, 2);
        run_checked(reference_b1, candidate_b1, 0);
        CHECK(ggml_cuda_graph_capture_state_query_for_test(candidate_backend.get(), candidate_b1.graph, &state));
        CHECK(state.graph != 0 && state.instance != 0 && state.warmup_complete &&
            state.moe_resource_fingerprint == b1_resource_fingerprint);
    }

    ggml_cuda_moe_grouped_acquisition final_resource;
    CHECK(context->acquire_group_resources(group_key, &final_resource));
    CHECK(final_resource.resource_generation == resource_generation &&
        ggml_cuda_moe_grouped_context_test_access::device_bank_data(
            *context, group_key, candidate_b1.banks[0]) == bank_data[0] &&
        ggml_cuda_moe_grouped_context_test_access::device_bank_data(
            *context, group_key, candidate_b1.banks[1]) == bank_data[1]);
    CHECK(active_grouped_legacy_op_count(reference_backend.get(), true) +
        active_grouped_legacy_op_count(reference_backend.get(), false) == executed_passes * candidate_b1.banks.size());
    CHECK(active_grouped_legacy_op_count(candidate_backend.get(), true) +
        active_grouped_legacy_op_count(candidate_backend.get(), false) == candidate_b1.banks.size());
    const auto telemetry = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
    const uint32_t grouped_passes = executed_passes - 1;
    CHECK(telemetry.registered == 1 && telemetry.covered == 1);
    CHECK(telemetry.plan_calls == grouped_passes);
    CHECK(telemetry.plan_compiles == 2 && telemetry.plan_reuses == grouped_passes - telemetry.plan_compiles);
    CHECK(telemetry.calls == grouped_passes && telemetry.ready == grouped_passes && telemetry.completed == grouped_passes);
    CHECK(telemetry.admitted_banks == grouped_passes * candidate_b1.banks.size());
    CHECK(telemetry.fallback == 0 && telemetry.rollback == 0 && telemetry.prepare_error == 0 && telemetry.finish_error == 0);
    ggml_backend_cuda_moe_set_debug_mm(old_debug_mm);
    fprintf(stderr, capture_available ?
        "test-moe-cache: shared-bank cache12 K8 B1/B2-legacy/B1 capture transition OK\n" :
        "test-moe-cache: shared-bank cache12 K8 B1/B2-legacy/B1 direct transition OK (CUDA graphs unavailable)\n");
}

static void test_active_grouped_legacy_phase_telemetry(int device) {
    const bool old_debug_mm = ggml_backend_cuda_moe_get_debug_mm();
    ggml_backend_cuda_moe_set_debug_mm(true);
    ggml_backend_ptr reference_backend(ggml_backend_cuda_init(device));
    ggml_backend_ptr candidate_backend(ggml_backend_cuda_init(device));
    CHECK(reference_backend != nullptr && candidate_backend != nullptr);
    auto reference = build_active_grouped_dispatch_graph(
        reference_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, false, 2, 128, 8);
    auto candidate = build_active_grouped_dispatch_graph(
        candidate_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, false, 2, 128, 8);
    initialize_active_grouped_dispatch_graphs({&reference, &candidate});
    const auto disabled = candidate_snapshot(12, nullptr, 0);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v1(reference_backend.get(), &disabled) ==
        GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    register_active_grouped_dispatch(
        candidate_backend.get(), candidate, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 12);
    auto * context = ggml_cuda_moe_grouped_context_for_test(candidate_backend.get());
    CHECK(context != nullptr);

    struct phase_case {
        uint32_t domain;
        uint32_t row_semantics;
        uint32_t n_sequences;
        bool certified;
        bool is_decode;
    };
    const std::array<phase_case, 6> cases = {{
        {GGML_GRAPH_EXECUTION_DOMAIN_MAIN,  GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 2, true,  true},
        {GGML_GRAPH_EXECUTION_DOMAIN_DRAFT, GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 2, true,  true},
        {GGML_GRAPH_EXECUTION_DOMAIN_MTP,   GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 2, true,  true},
        {GGML_GRAPH_EXECUTION_DOMAIN_MAIN,  GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SPECULATIVE, 1, true,  true},
        {GGML_GRAPH_EXECUTION_DOMAIN_MAIN,  GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL,  1, true,  false},
        {GGML_GRAPH_EXECUTION_DOMAIN_INVALID, GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INVALID,    0, false, false},
    }};
    for (size_t index = 0; index < cases.size(); ++index) {
        const auto & current = cases[index];
        if (current.certified) {
            candidate_stamp_execution(
                candidate.graph, current.domain, current.row_semantics, candidate.n_rows, current.n_sequences);
        } else {
            candidate.graph->uid = ggml_graph_next_uid();
            candidate.graph->execution_certificate = {};
        }
        (void) candidate_certify_graph(*context, candidate.graph);
        set_active_grouped_dispatch_logits({&reference, &candidate}, index % 3);
        const uint64_t decode_before = active_grouped_legacy_op_count(candidate_backend.get(), true);
        const uint64_t prefill_before = active_grouped_legacy_op_count(candidate_backend.get(), false);
        const auto expected = run_active_grouped_dispatch(reference_backend.get(), reference, 0, false);
        const auto actual = run_active_grouped_dispatch(candidate_backend.get(), candidate, 0, false);
        CHECK(expected == actual);
        check_active_grouped_routes(reference, reference.n_rows, index % 3);
        check_active_grouped_routes(candidate, candidate.n_rows, index % 3);
        CHECK(active_grouped_legacy_op_count(candidate_backend.get(), true) ==
            decode_before + (current.is_decode ? candidate.banks.size() : 0));
        CHECK(active_grouped_legacy_op_count(candidate_backend.get(), false) ==
            prefill_before + (current.is_decode ? 0 : candidate.banks.size()));
    }
    ggml_backend_cuda_moe_set_debug_mm(old_debug_mm);
    fprintf(stderr, "test-moe-cache: certified legacy phase telemetry OK\n");
}

static void test_active_grouped_stream_coherence_fallback(int device) {
    const bool old_debug_mm = ggml_backend_cuda_moe_get_debug_mm();
    ggml_backend_cuda_moe_set_debug_mm(true);
    ggml_backend_ptr reference_backend(ggml_backend_cuda_init(device));
    ggml_backend_ptr candidate_backend(ggml_backend_cuda_init(device));
    CHECK(reference_backend != nullptr && candidate_backend != nullptr);
    auto reference = build_active_grouped_dispatch_graph(
        reference_backend.get(), ggml_backend_cuda_buffer_type(device), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, false, 1, 8, 1, 256, nullptr, true);
    auto candidate = build_active_grouped_dispatch_graph(
        candidate_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, false, 1, 8, 1, 256, nullptr, true);
    initialize_active_grouped_dispatch_graphs({&reference, &candidate});
    register_active_grouped_dispatch(
        candidate_backend.get(), candidate, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 12);
    auto * context = ggml_cuda_moe_grouped_context_for_test(candidate_backend.get());
    CHECK(context != nullptr && candidate.concurrent_root != nullptr && candidate.graph->nodes[0] == candidate.ids->view_src &&
        candidate.graph->nodes[1] == candidate.ids);
    (void) candidate_certify_graph(*context, candidate.graph);
    const auto run_graph = [](ggml_backend_t backend, active_grouped_dispatch_graph & graph) {
        CHECK(ggml_backend_graph_compute(backend, graph.graph) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(backend);
        std::vector<float> output(ggml_nelements(graph.output));
        ggml_backend_tensor_get(graph.output, output.data(), 0, ggml_nbytes(graph.output));
        return output;
    };

    ggml_cuda_graph_capture_state_for_test captured = {};
    ggml_cuda_graph_capture_state_for_test grouped_capture = {};
    for (uint32_t pass = 0; pass < 3; ++pass) {
        const uint32_t route_variant = pass == 2 ? 1 : 0;
        set_active_grouped_dispatch_logits({&reference, &candidate}, route_variant);
        const auto expected = run_graph(reference_backend.get(), reference);
        const auto actual = run_graph(candidate_backend.get(), candidate);
        check_active_grouped_exact_output(expected, actual);
        CHECK(ggml_cuda_graph_capture_state_query_for_test(candidate_backend.get(), candidate.graph, &captured));
        if (pass == 0 && !captured.capture_available) {
            ggml_backend_cuda_moe_set_debug_mm(old_debug_mm);
            fprintf(stderr, "test-moe-cache: grouped stream coherence direct OK (CUDA graphs unavailable)\n");
            return;
        }
        if (pass == 1) {
            CHECK(captured.graph != 0 && captured.instance != 0 && captured.warmup_complete &&
                captured.execution_semantic_key != 0 && captured.moe_resource_fingerprint != 0);
            grouped_capture = captured;
        } else if (pass == 2) {
            CHECK(captured.graph == grouped_capture.graph && captured.instance == grouped_capture.instance &&
                captured.execution_semantic_key == grouped_capture.execution_semantic_key &&
                captured.moe_resource_fingerprint == grouped_capture.moe_resource_fingerprint);
        }
    }
    CHECK(captured.graph != 0 && captured.instance != 0 && captured.warmup_complete &&
        captured.execution_semantic_key != 0 && captured.moe_resource_fingerprint != 0);
    const uintptr_t stale_graph = captured.graph;
    const uintptr_t stale_instance = captured.instance;
    const uint64_t stale_fingerprint = captured.moe_resource_fingerprint;
    const auto grouped_before = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
    CHECK(grouped_before.registered == 1 && grouped_before.covered == 1 && grouped_before.calls == 3 &&
        grouped_before.ready == 3 && grouped_before.completed == 3 &&
        grouped_before.admitted_banks == 3 * candidate.banks.size());
    const uint64_t legacy_before = active_grouped_legacy_op_count(candidate_backend.get());

    const char * previous_graph_opt = getenv("GGML_CUDA_GRAPH_OPT");
    const std::string previous_graph_opt_value = previous_graph_opt != nullptr ? previous_graph_opt : "";
#ifdef _WIN32
    CHECK(_putenv_s("GGML_CUDA_GRAPH_OPT", "1") == 0);
#else
    CHECK(setenv("GGML_CUDA_GRAPH_OPT", "1", 1) == 0);
#endif
    ggml_backend_ptr cpu_backend(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    CHECK(cpu_backend != nullptr);
    ggml_backend_t backends[] = {candidate_backend.get(), cpu_backend.get()};
    ggml_backend_buffer_type_t bufts[] = {ggml_backend_cuda_buffer_type(device), ggml_backend_cpu_buffer_type()};
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, bufts, 2, 128, false, false));
    CHECK(sched != nullptr && ggml_backend_sched_alloc_graph(sched.get(), candidate.graph));
    CHECK(ggml_backend_sched_get_n_splits(sched.get()) == 1);
    ggml_graph_execution_certificate certificate = candidate.graph->execution_certificate;
    certificate.source_graph_uid = 0;
    certificate.split_graph_uid = 0;
    candidate.graph->execution_certificate = {};

    set_active_grouped_dispatch_logits({&reference, &candidate}, 2);
    const auto fallback_expected = run_graph(reference_backend.get(), reference);
    CHECK(ggml_backend_sched_graph_compute_ext(sched.get(), candidate.graph, &certificate) == GGML_STATUS_SUCCESS);
    std::vector<float> fallback_actual(ggml_nelements(candidate.output));
    ggml_backend_tensor_get(candidate.output, fallback_actual.data(), 0, ggml_nbytes(candidate.output));
    check_active_grouped_exact_output(fallback_expected, fallback_actual);
    CHECK(active_grouped_legacy_op_count(candidate_backend.get()) == legacy_before + candidate.banks.size());
    const auto fallback_telemetry = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
    CHECK(fallback_telemetry.registered == 1 && fallback_telemetry.covered == 1 && fallback_telemetry.plan_calls == 1 &&
        fallback_telemetry.plan_compiles + fallback_telemetry.plan_reuses == 1 &&
        fallback_telemetry.calls == 0 && fallback_telemetry.ready == 0 && fallback_telemetry.completed == 0 &&
        fallback_telemetry.admitted_banks == 0 && fallback_telemetry.fallback == 0 && fallback_telemetry.rollback == 0 &&
        fallback_telemetry.prepare_error == 0 && fallback_telemetry.finish_error == 0);
    ggml_cuda_graph_capture_state_for_test invalidated = {};
    CHECK(ggml_cuda_graph_capture_state_query_for_test(candidate_backend.get(), candidate.graph, &invalidated));
    CHECK(stale_graph != 0 && stale_instance != 0 && stale_fingerprint != 0 &&
        invalidated.graph == 0 && invalidated.instance == 0 && invalidated.moe_resource_fingerprint == 0 &&
        !invalidated.warmup_complete);

    ggml_set_name(candidate.concurrent_root, "test.active.parallel_root");
    ggml_backend_sched_reset(sched.get());
    CHECK(ggml_backend_sched_alloc_graph(sched.get(), candidate.graph));
    CHECK(ggml_backend_sched_get_n_splits(sched.get()) == 1);
    ggml_cuda_graph_capture_state_for_test recaptured = {};
    for (uint32_t pass = 0; pass < 3; ++pass) {
        const uint32_t route_variant = pass;
        set_active_grouped_dispatch_logits({&reference, &candidate}, route_variant);
        const auto expected = run_graph(reference_backend.get(), reference);
        CHECK(ggml_backend_sched_graph_compute_ext(sched.get(), candidate.graph, &certificate) == GGML_STATUS_SUCCESS);
        std::vector<float> actual(ggml_nelements(candidate.output));
        ggml_backend_tensor_get(candidate.output, actual.data(), 0, ggml_nbytes(candidate.output));
        check_active_grouped_exact_output(expected, actual);
        ggml_cuda_graph_capture_state_for_test state = {};
        CHECK(ggml_cuda_graph_capture_state_query_for_test(candidate_backend.get(), candidate.graph, &state));
        if (pass == 0) {
            CHECK(state.graph == 0 && state.instance == 0 && !state.warmup_complete && state.moe_resource_fingerprint == 0);
        } else if (pass == 1) {
            CHECK(state.graph != 0 && state.instance != 0 && state.warmup_complete && state.moe_resource_fingerprint != 0);
            recaptured = state;
        } else {
            CHECK(state.graph == recaptured.graph && state.instance == recaptured.instance &&
                state.warmup_complete && state.moe_resource_fingerprint == recaptured.moe_resource_fingerprint);
        }
    }
#ifdef _WIN32
    CHECK(_putenv_s("GGML_CUDA_GRAPH_OPT", previous_graph_opt != nullptr ? previous_graph_opt_value.c_str() : "") == 0);
#else
    CHECK(previous_graph_opt != nullptr ?
        setenv("GGML_CUDA_GRAPH_OPT", previous_graph_opt_value.c_str(), 1) == 0 :
        unsetenv("GGML_CUDA_GRAPH_OPT") == 0);
#endif
    ggml_backend_cuda_moe_set_debug_mm(old_debug_mm);
    fprintf(stderr, "test-moe-cache: grouped stream coherence fallback/capture recovery OK\n");
}

static void test_active_grouped_multirow_graph_modes(int device) {
    test_active_grouped_multirow_graph_modes_case(device, 2);
    test_active_grouped_multirow_graph_modes_case(device, 3);
    test_active_grouped_multirow_graph_modes_case(device, 4);
    test_active_grouped_multirow_graph_modes_case(
        device, 4, 8, 2, 12, false, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE);
    test_active_grouped_multirow_graph_modes_case(
        device, 4, 8, 2, 12, false, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_UNGATED);
    test_active_grouped_multirow_graph_modes_case(device, 4, 128, 8, 48, true);
    test_active_grouped_q4k_eviction_refill(device);
    test_active_grouped_same_key_row_transitions(device, 48);
    test_active_grouped_same_key_row_transitions(device, 138);
    test_active_grouped_cache12_route_limit_transition(device);
    test_active_grouped_legacy_phase_telemetry(device);
    test_active_grouped_stream_coherence_fallback(device);
}

static void check_active_grouped_bias_shadows(
        ggml_backend_t backend,
        const active_grouped_dispatch_graph & graph,
        uint32_t expected_resident = 0) {
    auto * context = ggml_cuda_moe_grouped_context_for_test(backend);
    ggml_cuda_moe_candidate_group_key key;
    CHECK(context != nullptr && context->find_down_group_key(graph.down, &key));
    std::vector<const float *> shadows;
    std::vector<std::vector<float>> originals;
    for (ggml_tensor * bias : graph.biases) {
        shadows.push_back(ggml_cuda_moe_grouped_context_test_access::device_auxiliary_data(*context, key, bias));
        CHECK(shadows.back() != nullptr && bias->type == GGML_TYPE_F32 && ggml_n_dims(bias) == 2);
        originals.emplace_back(ggml_nelements(bias));
        ggml_backend_tensor_get(bias, originals.back().data(), 0, ggml_nbytes(bias));
    }
    std::vector<bool> resident_slots(context->state().n_slots, false);
    uint32_t n_resident = 0;
    for (uint32_t expert = 0; expert < graph.n_experts; ++expert) {
        const int32_t slot = ggml_cuda_moe_grouped_context_test_access::device_slot_for_expert(*context, key, expert);
        if (slot < 0) {
            continue;
        }
        CHECK(static_cast<uint32_t>(slot) < resident_slots.size() && !resident_slots[slot]);
        resident_slots[slot] = true;
        ++n_resident;
        for (uint32_t bias_index = 0; bias_index < graph.biases.size(); ++bias_index) {
            ggml_tensor * bias = graph.biases[bias_index];
            std::vector<float> actual(bias->ne[0]);
            CUDA_OK(cudaMemcpy(actual.data(), shadows[bias_index] + slot * bias->ne[0],
                actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK(memcmp(actual.data(), originals[bias_index].data() + expert * bias->ne[0],
                actual.size() * sizeof(float)) == 0);
        }
    }
    if (expected_resident != 0) {
        CHECK(n_resident == expected_resident);
    }
}

static void test_active_grouped_dispatch_types_case(
        const std::array<ggml_type, 3> & types,
        uint32_t layout,
        uint32_t n_slots,
        bool original_direct_down_scale = false,
        bool test_owner_concurrency = false,
        bool original_direct_biases = false,
        uint32_t n_rows = 1,
        uint32_t n_ff = 0,
        bool auxiliary_proof = false) {
    CHECK(!auxiliary_proof || (layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE && n_slots == 3 &&
        test_owner_concurrency && original_direct_biases && n_rows == 1 && n_ff != 0));
    const bool old_debug_mm = ggml_backend_cuda_moe_get_debug_mm();
    ggml_backend_cuda_moe_set_debug_mm(true);
    ggml_backend_ptr reference_backend(ggml_backend_cuda_init(0));
    ggml_backend_ptr first_backend(ggml_backend_cuda_init(0));
    ggml_backend_ptr second_backend(ggml_backend_cuda_init(0));
    CHECK(reference_backend != nullptr && first_backend != nullptr && second_backend != nullptr);
    auto reference = build_active_grouped_dispatch_graph_types(
        reference_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), types, layout, original_direct_down_scale,
        false, n_rows, 8, 2, 256, nullptr, false, original_direct_biases, n_ff, auxiliary_proof);
    auto first = build_active_grouped_dispatch_graph_types(
        first_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), types, layout, original_direct_down_scale,
        false, n_rows, 8, 2, 256, nullptr, false, original_direct_biases, n_ff, auxiliary_proof);
    auto second = build_active_grouped_dispatch_graph_types(
        second_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), types, layout, original_direct_down_scale,
        false, n_rows, 8, 2, 256, nullptr, false, original_direct_biases, n_ff, auxiliary_proof);
    initialize_active_grouped_dispatch_graphs({&reference, &first, &second});
    const auto disabled = candidate_snapshot(n_slots, nullptr, 0);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v1(reference_backend.get(), &disabled) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    register_active_grouped_dispatch(first_backend.get(), first, layout, n_slots, original_direct_down_scale);
    register_active_grouped_dispatch(second_backend.get(), second, layout, n_slots);
    if (original_direct_down_scale) {
        auto * first_context = ggml_cuda_moe_grouped_context_for_test(first_backend.get());
        CHECK(first_context != nullptr);
        const uint64_t reordered_signature = first_context->state().logical_signature;
        register_active_grouped_dispatch(first_backend.get(), first, layout, n_slots);
        CHECK(first_context->state().logical_signature == reordered_signature);
        register_active_grouped_dispatch(first_backend.get(), first, layout, n_slots, true);
        CHECK(first_context->state().logical_signature == reordered_signature);
        for (auto * backend : {first_backend.get(), second_backend.get()}) {
            auto * context = ggml_cuda_moe_grouped_context_for_test(backend);
            ggml_cuda_moe_candidate_group_key key;
            ggml_cuda_moe_candidate_group_info group_info;
            ggml_cuda_moe_candidate_bank_info scale_info;
            const auto & graph = backend == first_backend.get() ? first : second;
            CHECK(context != nullptr && context->find_down_group_key(graph.down, &key));
            CHECK(context->get_group(key, &group_info) && group_info.n_banks == 3);
            CHECK(context->get_bank(key, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_SCALE, &scale_info));
            CHECK(scale_info.tensor == graph.down_scale && scale_info.type == GGML_TYPE_F32);
            CHECK(scale_info.movement == GGML_CUDA_MOE_CANDIDATE_MOVEMENT_PERMANENT_CANDIDATE);
            CHECK(scale_info.index_modes == GGML_CUDA_MOE_CANDIDATE_INDEX_ORIGINAL_DIRECT);
        }
    }
    if (original_direct_biases) {
        for (auto * backend : {first_backend.get(), second_backend.get()}) {
            auto * context = ggml_cuda_moe_grouped_context_for_test(backend);
            const auto & graph = backend == first_backend.get() ? first : second;
            ggml_cuda_moe_candidate_group_key key;
            ggml_cuda_moe_candidate_group_info group_info;
            CHECK(context != nullptr && context->find_down_group_key(graph.down, &key));
            CHECK(context->get_group(key, &group_info) &&
                group_info.n_banks == graph.banks.size() + graph.biases.size());
            for (uint32_t i = 0; i < graph.biases.size(); ++i) {
                ggml_cuda_moe_candidate_bank_info bias_info;
                CHECK(context->get_bank(key, graph.bias_roles[i], &bias_info));
                CHECK(bias_info.tensor == graph.biases[i] && bias_info.type == GGML_TYPE_F32);
                CHECK(bias_info.movement == GGML_CUDA_MOE_CANDIDATE_MOVEMENT_PERMANENT_CANDIDATE);
                CHECK(bias_info.index_modes == GGML_CUDA_MOE_CANDIDATE_INDEX_ORIGINAL_DIRECT);
            }
        }
    }
    if (auxiliary_proof) {
        CHECK(first.biases.size() == 3 && second.biases.size() == first.biases.size());
        CHECK(first.biases[0]->ne[0] == n_ff && first.biases[1]->ne[0] == n_ff && first.biases[2]->ne[0] == 256);
        for (const auto * graph : {&reference, &first, &second}) {
            for (ggml_tensor * bias : graph->biases) {
                CHECK(ggml_backend_buft_is_cuda_moe_cached(ggml_backend_buffer_get_type(bias->buffer)));
                cudaPointerAttributes attributes = {};
                CUDA_OK(cudaPointerGetAttributes(&attributes, bias->data));
                CHECK(attributes.type == cudaMemoryTypeHost);
            }
        }
    }
    check_active_grouped_contract(first_backend.get(), first, n_slots, original_direct_down_scale);
    check_active_grouped_contract(second_backend.get(), second, n_slots);

    std::vector<float> expected;
    std::vector<float> reference_output;
    std::vector<float> first_output;
    std::vector<float> second_output;
    const auto gate_capability = native_mmid_capability(0, first.banks[0], n_rows, GGML_CUDA_MMID_MAPPING_DIRECT);
    const bool f3_skipped = layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE &&
        gate_capability.reason == GGML_CUDA_MMID_CAPABILITY_OK &&
        gate_capability.selection == GGML_CUDA_MMID_CONSUMER_MMVQ &&
        first.banks[0]->type == first.banks[1]->type && ggml_are_same_shape(first.banks[0], first.banks[1]) &&
        ggml_are_same_stride(first.banks[0], first.banks[1]);
    for (int pass = 0; pass < 4; ++pass) {
        if (auxiliary_proof) {
            set_active_grouped_dispatch_logits({&reference, &first, &second}, pass % 3);
        }
        expected = run_active_grouped_dispatch(reference_backend.get(), reference, 0, false);
        const uint64_t expected_clock = auxiliary_proof ? 4 * (pass + 1) : 2 * n_rows * (pass + 2);
        auto current_first = run_active_grouped_dispatch(
            first_backend.get(), first, expected_clock, f3_skipped, original_direct_biases && n_rows == 1);
        auto current_second = run_active_grouped_dispatch(
            second_backend.get(), second, expected_clock, f3_skipped, original_direct_biases && n_rows == 1);
        CHECK(current_first.size() == expected.size() && current_second.size() == expected.size());
        CHECK(memcmp(current_first.data(), expected.data(), expected.size() * sizeof(float)) == 0);
        CHECK(memcmp(current_second.data(), expected.data(), expected.size() * sizeof(float)) == 0);
        if (auxiliary_proof) {
            current_first = run_active_grouped_dispatch(
                first_backend.get(), first, expected_clock + 2, f3_skipped, true);
            current_second = run_active_grouped_dispatch(
                second_backend.get(), second, expected_clock + 2, f3_skipped, true);
            CHECK(memcmp(current_first.data(), expected.data(), expected.size() * sizeof(float)) == 0);
            CHECK(memcmp(current_second.data(), expected.data(), expected.size() * sizeof(float)) == 0);
        }
        if (pass == 0) {
            reference_output = expected;
            first_output = current_first;
            second_output = current_second;
        } else if (!auxiliary_proof || pass == 3) {
            CHECK(expected == reference_output);
            CHECK(current_first == first_output && current_second == second_output);
        }
        if (auxiliary_proof) {
            const uint32_t expected_resident = pass == 0 ? 2 : n_slots;
            check_active_grouped_bias_shadows(first_backend.get(), first, expected_resident);
            check_active_grouped_bias_shadows(second_backend.get(), second, expected_resident);
        }
    }
    if (original_direct_biases && !auxiliary_proof) {
        check_active_grouped_bias_shadows(first_backend.get(), first);
        check_active_grouped_bias_shadows(second_backend.get(), second);
    }
    expected = reference_output;
    CHECK(first_output.size() == expected.size() && second_output.size() == expected.size());
    CHECK(memcmp(first_output.data(), second_output.data(), first_output.size() * sizeof(float)) == 0);
    double squared_error = 0.0;
    double squared_expected = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        CHECK(std::isfinite(expected[i]) && std::isfinite(first_output[i]));
        const double difference = expected[i] - first_output[i];
        squared_error += difference * difference;
        squared_expected += static_cast<double>(expected[i]) * expected[i];
    }
    CHECK(squared_expected > 0.0 && squared_error == 0.0);
    const bool reference_decode = n_rows == 1;
    CHECK(active_grouped_legacy_op_count(reference_backend.get(), reference_decode) == 4 * reference.banks.size());
    CHECK(active_grouped_legacy_op_count(reference_backend.get(), !reference_decode) == 0);
    CHECK(active_grouped_legacy_op_count(first_backend.get(), true) == 0 &&
        active_grouped_legacy_op_count(first_backend.get(), false) == 0);
    CHECK(active_grouped_legacy_op_count(second_backend.get(), true) == 0 &&
        active_grouped_legacy_op_count(second_backend.get(), false) == 0);
    check_active_grouped_debug_telemetry(first_backend.get(), first, auxiliary_proof ? 8 : 0, auxiliary_proof ? 9 : 5);
    check_active_grouped_debug_telemetry(second_backend.get(), second, auxiliary_proof ? 8 : 0, auxiliary_proof ? 9 : 5);
    if (test_owner_concurrency) {
        std::array<std::vector<float>, 2> concurrent_outputs;
        std::thread first_decode([&]() {
            concurrent_outputs[0] = run_active_grouped_dispatch(
                first_backend.get(), first, auxiliary_proof ? 20 : 12 * n_rows,
                f3_skipped, original_direct_biases && n_rows == 1);
        });
        std::thread second_decode([&]() {
            concurrent_outputs[1] = run_active_grouped_dispatch(
                second_backend.get(), second, auxiliary_proof ? 20 : 12 * n_rows,
                f3_skipped, original_direct_biases && n_rows == 1);
        });
        first_decode.join();
        second_decode.join();
        CHECK(concurrent_outputs[0] == first_output && concurrent_outputs[1] == second_output);
        for (auto * backend : {first_backend.get(), second_backend.get()}) {
            auto * context = ggml_cuda_moe_grouped_context_for_test(backend);
            CHECK(context != nullptr);
            const auto telemetry = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
            CHECK(telemetry.registered == 1 && telemetry.covered == 1 && telemetry.plan_calls == 1);
            CHECK(telemetry.plan_compiles == 0 && telemetry.plan_reuses == 1);
            CHECK(telemetry.calls == 1 && telemetry.ready == 1 && telemetry.completed == 1);
            CHECK(telemetry.admitted_banks == first.banks.size());
            CHECK(telemetry.prepare_error == 0 && telemetry.finish_error == 0);
        }
    }
    if (auxiliary_proof) {
        auto * first_context = ggml_cuda_moe_grouped_context_for_test(first_backend.get());
        auto * second_context = ggml_cuda_moe_grouped_context_for_test(second_backend.get());
        ggml_cuda_moe_candidate_group_key first_key;
        ggml_cuda_moe_candidate_group_key second_key;
        CHECK(first_context != nullptr && second_context != nullptr);
        CHECK(first_context->find_down_group_key(first.down, &first_key));
        CHECK(second_context->find_down_group_key(second.down, &second_key));
        for (uint32_t bias = 0; bias < first.biases.size(); ++bias) {
            CHECK(first.biases[bias]->data != second.biases[bias]->data);
            const float * first_shadow = ggml_cuda_moe_grouped_context_test_access::device_auxiliary_data(
                *first_context, first_key, first.biases[bias]);
            const float * second_shadow = ggml_cuda_moe_grouped_context_test_access::device_auxiliary_data(
                *second_context, second_key, second.biases[bias]);
            CHECK(first_shadow != nullptr && second_shadow != nullptr && first_shadow != second_shadow);
        }

        const uint64_t legacy_before = active_grouped_legacy_op_count(first_backend.get());
        void * saved_bias_data = first.biases[0]->data;
        first.biases[0]->data = static_cast<char *>(saved_bias_data) + sizeof(float);
        const auto coverage = candidate_certify_graph(*first_context, first.graph);
        std::shared_ptr<ggml_cuda_moe_graph_plan> failed_plan;
        ggml_cuda_moe_graph_execution failed_execution;
        CHECK(first_context->prepare_graph_execution(
            first.graph, 971, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &failed_plan, &failed_execution,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(failed_execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR && failed_execution.requires_dispatch());
        CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_auxiliary_reason(*failed_plan, 0));
        CHECK(!first_context->begin_graph_dispatch(&failed_execution, GGML_CUDA_MOE_GRAPH_DISPATCH_DIRECT));
        CHECK(active_grouped_legacy_op_count(first_backend.get()) == legacy_before);
        first.biases[0]->data = saved_bias_data;
        const auto failure_telemetry =
            ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*first_context);
        CHECK(failure_telemetry.registered == 1 && failure_telemetry.covered == 0 &&
            failure_telemetry.plan_calls == 0 && failure_telemetry.calls == 0);
        CHECK(failure_telemetry.fallback == 0 && failure_telemetry.rollback == 0);
    }
    check_active_grouped_legacy_caches(reference_backend.get(), reference, false);
    for (ggml_tensor * bank : first.banks) {
        auto * context = ggml_cuda_moe_grouped_context_for_test(first_backend.get());
        CHECK(context != nullptr && !context->acquire_legacy_cache(bank));
    }
    for (ggml_tensor * bank : second.banks) {
        auto * context = ggml_cuda_moe_grouped_context_for_test(second_backend.get());
        CHECK(context != nullptr && !context->acquire_legacy_cache(bank));
    }

    if (!original_direct_down_scale && types[0] == GGML_TYPE_Q4_0 && types[1] == GGML_TYPE_Q4_0 &&
            layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP && n_slots == 12) {
        auto * context = ggml_cuda_moe_grouped_context_for_test(first_backend.get());
        CHECK(context != nullptr);
        std::array<ggml_backend_moe_candidate_bank_v1, 3> replacement_banks = {};
        for (size_t i = 0; i < first.banks.size(); ++i) {
            replacement_banks[i] = {first.banks[i], first.roles[i], 0};
        }
        const ggml_backend_moe_candidate_group_v1 replacement_group = {
            replacement_banks.data(), static_cast<uint32_t>(first.banks.size()), layout, 0, 0,
        };
        const auto replacement_snapshot = candidate_snapshot(n_slots, &replacement_group, 1);
        CHECK(context->replace(&replacement_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        const auto zero_activity_replacement =
            ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
        CHECK(zero_activity_replacement.registered == 1 && zero_activity_replacement.covered == 0);
        CHECK(zero_activity_replacement.plan_calls == 0 && zero_activity_replacement.calls == 0);
        CHECK(zero_activity_replacement.ready == 0 && zero_activity_replacement.completed == 0);

        cudaStream_t stream = nullptr;
        cudaStream_t wrong_stream = nullptr;
        CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_OK(cudaStreamCreateWithFlags(&wrong_stream, cudaStreamNonBlocking));
        ggml_cuda_moe_candidate_group_key failure_key;
        CHECK(context->find_down_group_key(first.down, &failure_key));
        for (ggml_tensor * bank : first.banks) {
            auto legacy = context->acquire_legacy_cache(bank, nullptr, nullptr, stream);
            CHECK(legacy && legacy.get() != nullptr);
        }
        CHECK(ggml_cuda_moe_grouped_context_test_access::legacy_backing_count(*context, failure_key) == first.banks.size());
        CHECK(ggml_cuda_moe_grouped_context_test_access::device_resource_complete(*context, failure_key));

        std::shared_ptr<ggml_cuda_moe_graph_plan> failure_plan;
        ggml_cuda_moe_graph_execution failure_execution;
        CHECK(context->prepare_graph_execution(first.graph, 701, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &failure_plan, &failure_execution) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(failure_execution.resolve_streams(candidate_test_graph_stream, stream));
        CHECK(context->begin_graph_dispatch(&failure_execution, true));
        ggml_cuda_moe_graph_binding binding;
        auto * failure_group = failure_execution.find_group(first.readers[0], &binding);
        CHECK(failure_group != nullptr);
        CHECK(context->prepare_graph_group(failure_group, binding, first.readers[0], stream) == GGML_CUDA_MOE_GROUPED_DECODE_READY);
        const auto failed_resource = failure_group->transaction.acquisition;
        failure_group->stream = wrong_stream;
        CHECK(!context->finish_graph_dispatch(&failure_execution));
        CHECK(!context->get_group_resources(failed_resource, nullptr));
        CHECK(ggml_cuda_moe_grouped_context_test_access::legacy_backing_count(*context, failure_key) == 0);
        const auto failure_telemetry = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
        CHECK(failure_telemetry.registered == 1 && failure_telemetry.covered == 1);
        CHECK(failure_telemetry.plan_calls == 1 && failure_telemetry.plan_compiles == 1);
        CHECK(failure_telemetry.calls == 1 && failure_telemetry.ready == 1 && failure_telemetry.completed == 0);
        CHECK(failure_telemetry.prepare_error == 0 && failure_telemetry.finish_error == 1);
        CUDA_OK(cudaStreamDestroy(wrong_stream));
        CUDA_OK(cudaStreamDestroy(stream));
        CHECK(run_active_grouped_dispatch(first_backend.get(), first, 2, false) == first_output);

        ggml_cuda_moe_candidate_group_key key;
        ggml_cuda_moe_grouped_acquisition resource;
        ggml_cuda_moe_grouped_transaction held;
        CHECK(context->find_down_group_key(first.down, &key));
        CHECK(context->acquire_group_resources(key, &resource));
        CHECK(context->begin_group_transaction(resource, &held));
        std::atomic<bool> replacement_started{false};
        std::atomic<int32_t> replacement_result{GGML_BACKEND_MOE_CANDIDATE_REPLACE_ERROR};
        std::thread replacement([&]() {
            replacement_started.store(true, std::memory_order_release);
            replacement_result.store(context->replace(&replacement_snapshot), std::memory_order_release);
        });
        while (!replacement_started.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        while (!ggml_cuda_moe_grouped_context_test_access::admission_closed(*context)) {
            std::this_thread::yield();
        }
        std::shared_ptr<ggml_cuda_moe_graph_plan> unavailable_plan;
        ggml_cuda_moe_graph_execution unavailable_execution;
        while (context->prepare_graph_execution(first.graph, 702, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &unavailable_plan, &unavailable_execution) !=
                GGML_CUDA_MOE_GRAPH_PREPARE_UNAVAILABLE) {
            std::this_thread::yield();
        }
        std::vector<float> zero_input(ggml_nelements(first.input));
        ggml_backend_tensor_set(first.input, zero_input.data(), 0, ggml_nbytes(first.input));
        CHECK(ggml_backend_graph_compute(first_backend.get(), first.graph) == GGML_STATUS_FAILED);
        std::vector<float> preserved_output(ggml_nelements(first.output));
        ggml_backend_tensor_get(first.output, preserved_output.data(), 0, ggml_nbytes(first.output));
        CHECK(preserved_output == first_output);
        const auto restored_input = cached_fusion_test_data(first.input, 131);
        ggml_backend_tensor_set(first.input, restored_input.data(), 0, restored_input.size());
        CHECK(context->end_group_transaction(held));
        replacement.join();
        CHECK(replacement_result.load(std::memory_order_acquire) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        CHECK(run_active_grouped_dispatch(first_backend.get(), first, 2, false) == first_output);
        const auto replacement_telemetry =
            ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
        CHECK(replacement_telemetry.registered == 2 && replacement_telemetry.covered == 2);
        CHECK(replacement_telemetry.plan_calls == 2 &&
            replacement_telemetry.plan_compiles + replacement_telemetry.plan_reuses == replacement_telemetry.plan_calls);
        CHECK(replacement_telemetry.calls == 2 && replacement_telemetry.ready == 2 && replacement_telemetry.completed == 2);
        CHECK(replacement_telemetry.ready_min == 1 && replacement_telemetry.ready_max == 1);
        CHECK(replacement_telemetry.completed_min == 1 && replacement_telemetry.completed_max == 1);
        CHECK(replacement_telemetry.h2d_banks == 4 * first.banks.size());

        CHECK(run_active_grouped_dispatch(first_backend.get(), first, 4, false) == first_output);
        const auto disabled_snapshot = candidate_snapshot(n_slots, nullptr, 0);
        CHECK(context->replace(&disabled_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        const auto disabled_telemetry = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
        CHECK(disabled_telemetry.registered == 1 && disabled_telemetry.covered == 1);
        CHECK(disabled_telemetry.plan_calls == 1 && disabled_telemetry.calls == 1);
        CHECK(disabled_telemetry.ready == 1 && disabled_telemetry.completed == 1);
        register_active_grouped_dispatch(first_backend.get(), first, layout, n_slots);
        CHECK(run_active_grouped_dispatch(first_backend.get(), first, 2, false) == first_output);
        const ggml_backend_moe_candidate_group_v1 rejected_group = {
            replacement_banks.data(), static_cast<uint32_t>(first.banks.size()), layout, 1, 0,
        };
        const auto rejected_snapshot = candidate_snapshot(n_slots, &rejected_group, 1);
        CHECK(context->replace(&rejected_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_REJECTED);
        const auto rejected_telemetry = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
        CHECK(rejected_telemetry.registered == 1 && rejected_telemetry.covered == 1);
        CHECK(rejected_telemetry.plan_calls == 1 && rejected_telemetry.calls == 1);
        CHECK(rejected_telemetry.ready == 1 && rejected_telemetry.completed == 1);
        CHECK(rejected_telemetry.h2d_banks == 2 * first.banks.size());
        register_active_grouped_dispatch(first_backend.get(), first, layout, n_slots);
        ggml_cuda_moe_grouped_context::log_and_reset_legacy_stats();
        const int32_t shutdown_routes[] = {3, 5};
        ggml_backend_tensor_set(first.ids, shutdown_routes, 0, sizeof(shutdown_routes));
        std::shared_ptr<ggml_cuda_moe_graph_plan> shutdown_plan;
        ggml_cuda_moe_graph_execution shutdown_execution;
        CHECK(context->prepare_graph_execution(first.graph, 703, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &shutdown_plan, &shutdown_execution) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        cudaStream_t shutdown_stream = nullptr;
        CUDA_OK(cudaStreamCreateWithFlags(&shutdown_stream, cudaStreamNonBlocking));
        CHECK(shutdown_execution.resolve_streams(candidate_test_graph_stream, shutdown_stream));
        CHECK(context->begin_graph_dispatch(&shutdown_execution, true));
        ggml_cuda_moe_graph_binding shutdown_binding;
        auto * shutdown_group = shutdown_execution.find_group(first.readers[0], &shutdown_binding);
        CHECK(shutdown_group != nullptr);
        CHECK(context->prepare_graph_group(shutdown_group, shutdown_binding, first.readers[0], shutdown_stream) ==
            GGML_CUDA_MOE_GROUPED_DECODE_READY);
        const auto shutdown_resource = shutdown_group->transaction.acquisition;
        host_barrier barrier;
        CUDA_OK(cudaLaunchHostFunc(shutdown_stream, wait_on_host_barrier, &barrier));
        ggml_cuda_moe_graph_binding shutdown_last_binding;
        CHECK(shutdown_execution.find_group(first.readers.back(), &shutdown_last_binding) == shutdown_group);
        CHECK(context->finish_graph_group(shutdown_group, shutdown_last_binding, first.readers.back(), shutdown_stream));
        CHECK(context->finish_graph_dispatch(&shutdown_execution));
        while (!barrier.entered.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        ggml_cuda_moe_grouped_resource_info shutdown_info;
        CHECK(context->get_group_resources(shutdown_resource, &shutdown_info) && !shutdown_info.transaction_active);
        CHECK(context->find_down_group_key(first.down, &key));
        std::atomic<bool> shutdown_started{false};
        std::atomic<bool> shutdown_done{false};
        std::thread shutdown([&]() {
            shutdown_started.store(true, std::memory_order_release);
            context->shutdown();
            shutdown_done.store(true, std::memory_order_release);
        });
        while (!shutdown_started.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        while (!ggml_cuda_moe_grouped_context_test_access::admission_closed(*context)) {
            std::this_thread::yield();
        }
        ggml_cuda_moe_grouped_acquisition shutdown_probe;
        CHECK(!context->acquire_group_resources(key, &shutdown_probe));
        CHECK(!shutdown_done.load(std::memory_order_acquire));
        std::atomic<bool> logger_started{false};
        std::atomic<bool> logger_done{false};
        ggml_cuda_moe_grouped_debug_telemetry shutdown_telemetry;
        std::thread logger([&]() {
            logger_started.store(true, std::memory_order_release);
            shutdown_telemetry = ggml_cuda_moe_grouped_context::log_and_reset_legacy_stats();
            logger_done.store(true, std::memory_order_release);
        });
        while (!logger_started.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        for (uint32_t attempt = 0; attempt < 100 && !logger_done.load(std::memory_order_acquire); ++attempt) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        CHECK(!logger_done.load(std::memory_order_acquire) && !shutdown_done.load(std::memory_order_acquire));
        barrier.released.store(true, std::memory_order_release);
        logger.join();
        shutdown.join();
        CHECK(logger_done.load(std::memory_order_acquire) && shutdown_done.load(std::memory_order_acquire));
        CUDA_OK(cudaStreamDestroy(shutdown_stream));
        CHECK(shutdown_telemetry.covered == 1 && shutdown_telemetry.plan_calls == 1);
        CHECK(shutdown_telemetry.plan_compiles + shutdown_telemetry.plan_reuses == shutdown_telemetry.plan_calls);
        CHECK(shutdown_telemetry.calls == 1 && shutdown_telemetry.ready == 1 && shutdown_telemetry.completed == 1);
        CHECK(shutdown_telemetry.admitted_banks == first.banks.size());
        CHECK(shutdown_telemetry.fallback == 0 && shutdown_telemetry.rollback == 0);
        CHECK(shutdown_telemetry.prepare_error == 0 && shutdown_telemetry.finish_error == 0);
        CHECK(shutdown_telemetry.h2d_banks == 2 * first.banks.size());
        uint64_t shutdown_h2d_bytes = 0;
        for (const ggml_tensor * bank : first.banks) {
            shutdown_h2d_bytes += 2 * bank->nb[2];
        }
        CHECK(shutdown_telemetry.h2d_bytes == shutdown_h2d_bytes);
        const auto shutdown_reset = ggml_cuda_moe_grouped_context::log_and_reset_legacy_stats();
        CHECK(shutdown_reset.covered == 0 && shutdown_reset.plan_calls == 0);
        CHECK(shutdown_reset.plan_compiles == 0 && shutdown_reset.plan_reuses == 0 && shutdown_reset.calls == 0);
        CHECK(shutdown_reset.ready == 0 && shutdown_reset.ready_min == 0 && shutdown_reset.ready_max == 0);
        CHECK(shutdown_reset.completed == 0 && shutdown_reset.completed_min == 0 && shutdown_reset.completed_max == 0);
        CHECK(shutdown_reset.admitted_banks == 0);
        CHECK(shutdown_reset.fallback == 0 && shutdown_reset.rollback == 0);
        CHECK(shutdown_reset.prepare_error == 0 && shutdown_reset.finish_error == 0);
        CHECK(shutdown_reset.h2d_banks == 0 && shutdown_reset.h2d_bytes == 0);
    }
    ggml_backend_cuda_moe_set_debug_mm(old_debug_mm);
}

static void test_active_grouped_dispatch_case(
        ggml_type type,
        uint32_t layout,
        uint32_t n_slots,
        bool original_direct_down_scale = false,
        bool test_owner_concurrency = false) {
    test_active_grouped_dispatch_types_case(
        {type, type, type}, layout, n_slots, original_direct_down_scale, test_owner_concurrency);
}

static void test_active_grouped_materialization_eligibility() {
#ifdef __linux__
    const bool old_debug_mm = ggml_backend_cuda_moe_get_debug_mm();
    ggml_backend_cuda_moe_set_debug_mm(true);
    ggml_backend_ptr reference_backend(ggml_backend_cuda_init(0));
    ggml_backend_ptr pinned_backend(ggml_backend_cuda_init(0));
    ggml_backend_ptr pageable_backend(ggml_backend_cuda_init(0));
    CHECK(reference_backend != nullptr && pinned_backend != nullptr && pageable_backend != nullptr);
    auto reference = build_active_grouped_dispatch_graph(
        reference_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, true);
    auto pinned = build_active_grouped_dispatch_graph(
        pinned_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, true);
    auto pageable = build_active_grouped_dispatch_graph(
        pageable_backend.get(), file_mmap_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, true);
    initialize_active_grouped_dispatch_graphs({&reference, &pinned, &pageable});

    cudaPointerAttributes attributes = {};
    const cudaError_t pointer_status = cudaPointerGetAttributes(
        &attributes, ggml_backend_buffer_get_base(pageable.weight_buffer.get()));
    CHECK(pointer_status != cudaSuccess || attributes.type == cudaMemoryTypeUnregistered);
    if (pointer_status != cudaSuccess) {
        (void) cudaGetLastError();
    }

    const auto disabled = candidate_snapshot(12, nullptr, 0);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v1(
        reference_backend.get(), &disabled) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    register_active_grouped_dispatch(
        pinned_backend.get(), pinned, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 12);
    register_active_grouped_dispatch(
        pageable_backend.get(), pageable, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 12);

    const auto check_plan = [](
            ggml_backend_t backend,
            const active_grouped_dispatch_graph & graph,
            ggml_cuda_moe_graph_outcome expected_outcome,
            bool materialization) {
        auto * context = ggml_cuda_moe_grouped_context_for_test(backend);
        CHECK(context != nullptr);
        const auto coverage = candidate_certify_graph(*context, graph.graph);
        std::shared_ptr<ggml_cuda_moe_graph_plan> plan;
        ggml_cuda_moe_graph_execution execution;
        CHECK(context->prepare_graph_execution(
            graph.graph, 901, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &plan, &execution,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(plan != nullptr && execution.outcome() == expected_outcome && execution.size() == 1);
        CHECK(execution.find_group(graph.down_output, nullptr) != nullptr);
        CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_materialization_reason(*plan, 0) == materialization);
        CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_eligible_reason(*plan, 0) != materialization);
        ggml_cuda_moe_graph_execution unknown;
        CHECK(context->bind_graph_plan(
            graph.graph, 902, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, *plan, &unknown,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint));
        CHECK(unknown.outcome() == expected_outcome && unknown.find_group(graph.down_output, nullptr) != nullptr);
    };
    check_plan(pinned_backend.get(), pinned, GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED, false);
    check_plan(pageable_backend.get(), pageable, GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_LEGACY, true);

    const auto expected = run_active_grouped_dispatch(reference_backend.get(), reference, 0, false);
    const auto pinned_output = run_active_grouped_dispatch(pinned_backend.get(), pinned, 2, true);
    const auto pageable_output = run_active_grouped_dispatch(pageable_backend.get(), pageable, 0, false);
    CHECK(pinned_output == expected && pageable_output == expected);
    CHECK(active_grouped_legacy_op_count(reference_backend.get()) == reference.banks.size());
    CHECK(active_grouped_legacy_op_count(pinned_backend.get()) == 0);
    CHECK(active_grouped_legacy_op_count(pageable_backend.get()) == pageable.banks.size());

    auto * pinned_context = ggml_cuda_moe_grouped_context_for_test(pinned_backend.get());
    auto * pageable_context = ggml_cuda_moe_grouped_context_for_test(pageable_backend.get());
    ggml_cuda_moe_candidate_group_key pinned_key;
    ggml_cuda_moe_candidate_group_key pageable_key;
    CHECK(pinned_context != nullptr && pageable_context != nullptr);
    CHECK(pinned_context->find_down_group_key(pinned.down, &pinned_key));
    CHECK(pageable_context->find_down_group_key(pageable.down, &pageable_key));
    CHECK(ggml_cuda_moe_grouped_context_test_access::has_device_resource(*pinned_context, pinned_key));
    CHECK(!ggml_cuda_moe_grouped_context_test_access::has_device_resource(*pageable_context, pageable_key));
    const auto pageable_telemetry =
        ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*pageable_context);
    CHECK(pageable_telemetry.registered == 1 && pageable_telemetry.covered == 0);
    CHECK(pageable_telemetry.calls == 0 && pageable_telemetry.ready == 0 && pageable_telemetry.completed == 0);
    CHECK(pageable_telemetry.prepare_error == 0 && pageable_telemetry.finish_error == 0);
    check_active_grouped_legacy_caches(pageable_backend.get(), pageable, true);
    for (ggml_tensor * bank : pinned.banks) {
        CHECK(!pinned_context->acquire_legacy_cache(bank));
    }
    ggml_backend_cuda_moe_set_debug_mm(old_debug_mm);
    fprintf(stderr, "test-moe-cache: grouped mapped-source eligibility OK\n");
#else
    fprintf(stderr, "test-moe-cache: grouped mapped-source eligibility SKIP\n");
#endif
}

static void test_active_grouped_dispatch_staged_legacy() {
    const bool old_debug_mm = ggml_backend_cuda_moe_get_debug_mm();
    ggml_backend_cuda_moe_set_debug_mm(true);
    ggml_backend_ptr reference_backend(ggml_backend_cuda_init(0));
    ggml_backend_ptr candidate_backend(ggml_backend_cuda_init(0));
    CHECK(reference_backend != nullptr && candidate_backend != nullptr);
    auto reference = build_active_grouped_dispatch_graph(
        reference_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE);
    auto candidate = build_active_grouped_dispatch_graph(
        candidate_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE);
    initialize_active_grouped_dispatch_graphs({&reference, &candidate});
    const auto disabled = candidate_snapshot(1, nullptr, 0);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v1(reference_backend.get(), &disabled) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    register_active_grouped_dispatch(candidate_backend.get(), candidate, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 1);

    const auto expected = run_active_grouped_dispatch(reference_backend.get(), reference, 0, false);
    CHECK(!expected.empty());
    const auto actual = run_active_grouped_dispatch(candidate_backend.get(), candidate, 0, false);
    check_active_grouped_exact_output(expected, actual);
    CHECK(active_grouped_legacy_op_count(reference_backend.get()) == reference.banks.size());
    CHECK(active_grouped_legacy_op_count(candidate_backend.get()) == candidate.banks.size());
    check_active_grouped_legacy_caches(reference_backend.get(), reference, false, false);
    check_active_grouped_legacy_caches(candidate_backend.get(), candidate, true, false);

    auto * context = ggml_cuda_moe_grouped_context_for_test(candidate_backend.get());
    ggml_cuda_moe_candidate_group_key key;
    CHECK(context != nullptr && context->find_down_group_key(candidate.down, &key));
    CHECK(ggml_cuda_moe_grouped_context_test_access::has_device_resource(*context, key));
    CHECK(ggml_cuda_moe_grouped_context_test_access::device_resource_complete(*context, key));
    CHECK(ggml_cuda_moe_grouped_context_test_access::legacy_backing_count(*context, key) == candidate.banks.size());
    const auto telemetry = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
    CHECK(telemetry.registered == 1 && telemetry.covered == 0 && telemetry.plan_calls == 0);
    CHECK(telemetry.calls == 0 && telemetry.ready == 0 && telemetry.completed == 0 && telemetry.admitted_banks == 0);
    CHECK(telemetry.fallback == 0 && telemetry.rollback == 0 && telemetry.prepare_error == 0 && telemetry.finish_error == 0);
    CHECK(telemetry.h2d_banks == 0 && telemetry.h2d_bytes == 0);
    ggml_backend_cuda_moe_set_debug_mm(old_debug_mm);
}

static void test_active_grouped_inactive_v2_case(uint32_t snapshot_flags, uint32_t group_flags, uint32_t coverage_reason) {
    ggml_backend_ptr backend(ggml_backend_cuda_init(0));
    CHECK(backend != nullptr);
    auto graph = build_active_grouped_dispatch_graph(
        backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE);
    initialize_active_grouped_dispatch_graph(graph, 201);

    const ggml_backend_moe_candidate_group_v2 group = {
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY, group_flags, 0,
    };
    std::array<ggml_backend_moe_candidate_tensor_v2, 3> tensors = {};
    for (uint32_t i = 0; i < tensors.size(); ++i) {
        tensors[i] = {
            graph.banks[i], 0, graph.roles[i], GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE,
            GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_CACHED_BUFFER, 0,
        };
    }
    auto snapshot = candidate_snapshot_v2(12, &group, 1, tensors.data(), tensors.size());
    snapshot.flags = snapshot_flags;
    CHECK(ggml_backend_cuda_moe_candidate_replace_v2(backend.get(), &snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    auto * context = ggml_cuda_moe_grouped_context_for_test(backend.get());
    CHECK(context != nullptr && context->state().accepted && context->state().n_slots == 12 && context->state().n_groups == 0);

    ggml_cuda_moe_graph_plan plan;
    ggml_cuda_moe_graph_execution execution;
    context->compile_graph_plan(graph.graph, 901, &plan, &execution);
    CHECK(execution.size() == 0 && execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR && execution.requires_dispatch());
    CHECK(plan.coverage_diagnostics().cached_mmid == graph.readers.size());
    CHECK(plan.coverage_diagnostics().counts[coverage_reason] == graph.readers.size());

    const auto sentinel = active_grouped_intermediate_sentinel(graph);
    CHECK(ggml_backend_graph_compute(backend.get(), graph.graph) == GGML_STATUS_FAILED);
    ggml_backend_synchronize(backend.get());
    check_active_grouped_intermediates(graph, sentinel, true);
    CHECK(active_grouped_legacy_op_count(backend.get()) == 0);
}

static void test_active_grouped_descriptive_v2() {
    ggml_backend_ptr backend(ggml_backend_cuda_init(0));
    CHECK(backend != nullptr);
    auto graph = build_active_grouped_dispatch_graph(
        backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE);
    initialize_active_grouped_dispatch_graph(graph, 251);

    const ggml_backend_moe_candidate_group_v2 group = {
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_UNGATED, GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_CHUNK,
        GGML_BACKEND_MOE_CANDIDATE_GROUP_V2_FLAG_NONE, 0,
    };
    const auto snapshot = candidate_snapshot_v2(12, &group, 1, nullptr, 0);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v2(backend.get(), &snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    auto * context = ggml_cuda_moe_grouped_context_for_test(backend.get());
    CHECK(context != nullptr && context->state().accepted && context->state().n_slots == 12 && context->state().n_groups == 0);

    ggml_cuda_moe_graph_plan plan;
    ggml_cuda_moe_graph_execution execution;
    context->compile_graph_plan(graph.graph, 9021, &plan, &execution);
    CHECK(execution.size() == 0 && execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR && execution.requires_dispatch());
    CHECK(plan.coverage_diagnostics().cached_mmid == graph.readers.size());
    CHECK(plan.coverage_diagnostics().counts[GGML_CUDA_MOE_GRAPH_COVERAGE_REVERSE_MAP_MISS] == graph.readers.size());

    std::vector<float> sentinel(ggml_nelements(graph.output), -12345.25f);
    ggml_backend_tensor_set(graph.output, sentinel.data(), 0, ggml_nbytes(graph.output));
    ggml_backend_synchronize(backend.get());
    CHECK(ggml_backend_graph_compute(backend.get(), graph.graph) == GGML_STATUS_FAILED);
    ggml_backend_synchronize(backend.get());
    std::vector<float> output(ggml_nelements(graph.output));
    ggml_backend_tensor_get(graph.output, output.data(), 0, ggml_nbytes(graph.output));
    CHECK(output == sentinel);
    CHECK(active_grouped_legacy_op_count(backend.get()) == 0);
}

static void test_active_grouped_flags_only_v2(uint32_t flags) {
    ggml_backend_ptr backend(ggml_backend_cuda_init(0));
    CHECK(backend != nullptr);
    auto graph = build_active_grouped_dispatch_graph(
        backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE);
    initialize_active_grouped_dispatch_graph(graph, 271 + flags);

    auto snapshot = candidate_snapshot_v2(12, nullptr, 0, nullptr, 0);
    snapshot.flags = flags;
    CHECK(ggml_backend_cuda_moe_candidate_replace_v2(backend.get(), &snapshot) ==
        GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    auto * context = ggml_cuda_moe_grouped_context_for_test(backend.get());
    CHECK(context != nullptr && context->state().accepted && context->state().n_groups == 0);

    ggml_cuda_moe_graph_plan plan;
    ggml_cuda_moe_graph_execution execution;
    context->compile_graph_plan(graph.graph, 9022 + flags, &plan, &execution);
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR && execution.requires_dispatch());
    CHECK(execution.rejects_cached_mmid(graph.readers[0]));
    CHECK(!execution.rejects_cached_mmid(graph.ids));

    const auto sentinel = active_grouped_intermediate_sentinel(graph);
    CHECK(ggml_backend_graph_compute(backend.get(), graph.graph) == GGML_STATUS_FAILED);
    ggml_backend_synchronize(backend.get());
    check_active_grouped_intermediates(graph, sentinel, true);
    CHECK(active_grouped_legacy_op_count(backend.get()) == 0);
}

static void test_active_grouped_explicit_empty_case(bool v2) {
    ggml_backend_ptr reference_backend(ggml_backend_cuda_init(0));
    ggml_backend_ptr candidate_backend(ggml_backend_cuda_init(0));
    CHECK(reference_backend != nullptr && candidate_backend != nullptr);
    auto reference = build_active_grouped_dispatch_graph(
        reference_backend.get(), ggml_backend_cuda_buffer_type(0), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE);
    auto candidate = build_active_grouped_dispatch_graph(
        candidate_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE);
    initialize_active_grouped_dispatch_graphs({&reference, &candidate});
    if (v2) {
        const auto disabled = candidate_snapshot_v2(12, nullptr, 0, nullptr, 0);
        CHECK(ggml_backend_cuda_moe_candidate_replace_v2(candidate_backend.get(), &disabled) ==
            GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    } else {
        const auto disabled = candidate_snapshot(12, nullptr, 0);
        CHECK(ggml_backend_cuda_moe_candidate_replace_v1(candidate_backend.get(), &disabled) ==
            GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    }

    auto * context = ggml_cuda_moe_grouped_context_for_test(candidate_backend.get());
    CHECK(context != nullptr);
    const auto coverage = candidate_certify_graph(*context, candidate.graph);
    std::shared_ptr<ggml_cuda_moe_graph_plan> plan;
    ggml_cuda_moe_graph_execution execution;
    CHECK(context->prepare_graph_execution(
        candidate.graph, 903, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &plan, &execution,
        coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
    CHECK(execution.size() == 0 && execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_PREFILL_LEGACY && !execution.requires_dispatch());
    const ggml_cuda_moe_graph_plan * stable_plan = plan.get();
    CHECK(context->prepare_graph_execution(
        candidate.graph, 904, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &plan, &execution,
        coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) == GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
    CHECK(plan.get() == stable_plan && execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_PREFILL_LEGACY);

    const auto expected = run_active_grouped_dispatch(reference_backend.get(), reference, 0, true);
    const auto first = run_active_grouped_dispatch(candidate_backend.get(), candidate, 0, false);
    const auto second = run_active_grouped_dispatch(candidate_backend.get(), candidate, 0, false);
    CHECK(first == expected && second == expected);
    CHECK(active_grouped_legacy_op_count(candidate_backend.get()) == 2 * candidate.readers.size());
}

static void test_active_grouped_empty_policy() {
    test_active_grouped_inactive_v2_case(
        GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_FLAG_NONE,
        GGML_BACKEND_MOE_CANDIDATE_GROUP_V2_FLAG_ACTIVE_LORA,
        GGML_CUDA_MOE_GRAPH_COVERAGE_ACTIVE_LORA);
    test_active_grouped_inactive_v2_case(
        GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_FLAG_TENSOR_OVERRIDES,
        GGML_BACKEND_MOE_CANDIDATE_GROUP_V2_FLAG_NONE,
        GGML_CUDA_MOE_GRAPH_COVERAGE_TENSOR_OVERRIDE);
    test_active_grouped_inactive_v2_case(
        GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_FLAG_INCOMPLETE,
        GGML_BACKEND_MOE_CANDIDATE_GROUP_V2_FLAG_NONE,
        GGML_CUDA_MOE_GRAPH_COVERAGE_INCOMPLETE);
    test_active_grouped_descriptive_v2();
    test_active_grouped_flags_only_v2(GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_FLAG_TENSOR_OVERRIDES);
    test_active_grouped_flags_only_v2(GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_FLAG_INCOMPLETE);

    {
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        CHECK(backend != nullptr);
        auto graph = build_active_grouped_dispatch_graph(
            backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
            GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE);
        initialize_active_grouped_dispatch_graph(graph, 301);
        std::array<ggml_backend_moe_candidate_bank_v1, 3> banks = {};
        for (uint32_t i = 0; i < banks.size(); ++i) {
            banks[i] = {graph.banks[i], graph.roles[i], 0};
        }
        const ggml_backend_moe_candidate_group_v1 group = {
            banks.data(), banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 1, 0,
        };
        const auto snapshot = candidate_snapshot(12, &group, 1);
        CHECK(ggml_backend_cuda_moe_candidate_replace_v1(backend.get(), &snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_REJECTED);
        auto * context = ggml_cuda_moe_grouped_context_for_test(backend.get());
        CHECK(context != nullptr && !context->state().accepted && context->state().n_slots == 12);
        ggml_cuda_moe_graph_plan plan;
        ggml_cuda_moe_graph_execution execution;
        context->compile_graph_plan(graph.graph, 902, &plan, &execution);
        CHECK(execution.size() == 0 && execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR && execution.requires_dispatch());
        std::vector<float> sentinel(ggml_nelements(graph.output), -12345.25f);
        ggml_backend_tensor_set(graph.output, sentinel.data(), 0, ggml_nbytes(graph.output));
        ggml_backend_synchronize(backend.get());
        CHECK(ggml_backend_graph_compute(backend.get(), graph.graph) == GGML_STATUS_FAILED);
        ggml_backend_synchronize(backend.get());
        std::vector<float> output(ggml_nelements(graph.output));
        ggml_backend_tensor_get(graph.output, output.data(), 0, ggml_nbytes(graph.output));
        CHECK(output == sentinel);
        CHECK(active_grouped_legacy_op_count(backend.get()) == 0);
    }
    const bool old_debug_mm = ggml_backend_cuda_moe_get_debug_mm();
    ggml_backend_cuda_moe_set_debug_mm(true);
    test_active_grouped_explicit_empty_case(false);
    test_active_grouped_explicit_empty_case(true);
    ggml_backend_cuda_moe_set_debug_mm(old_debug_mm);
    fprintf(stderr, "test-moe-cache: empty grouped policy production seam OK\n");
}

static void test_active_grouped_inventory_reuse_case(ggml_type second_type, bool second_supported) {
    ggml_backend_ptr backend(ggml_backend_cuda_init(0));
    CHECK(backend != nullptr);
    auto first = build_active_grouped_dispatch_graph(
        backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE);
    auto second = build_active_grouped_dispatch_graph(
        backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), second_type,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE);
    initialize_active_grouped_dispatch_graph(first, 501);
    if (second_supported) {
        initialize_active_grouped_dispatch_graph(second, 601);
    }

    std::array<std::array<ggml_backend_moe_candidate_bank_v1, 3>, 2> banks = {};
    std::array<ggml_backend_moe_candidate_group_v1, 2> groups = {};
    for (uint32_t group_index = 0; group_index < groups.size(); ++group_index) {
        const auto & source = group_index == 0 ? first : second;
        CHECK(source.banks.size() == banks[group_index].size());
        for (uint32_t bank_index = 0; bank_index < source.banks.size(); ++bank_index) {
            banks[group_index][bank_index] = {source.banks[bank_index], source.roles[bank_index], 0};
        }
        groups[group_index] = {
            banks[group_index].data(), static_cast<uint32_t>(banks[group_index].size()),
            GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 0, 0,
        };
    }
    const auto snapshot = candidate_snapshot(12, groups.data(), groups.size());
    CHECK(ggml_backend_cuda_moe_candidate_replace_v1(backend.get(), &snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    auto * context = ggml_cuda_moe_grouped_context_for_test(backend.get());
    CHECK(context != nullptr && context->state().n_groups == groups.size());

    const int32_t combined_nodes = first.graph->n_nodes + second.graph->n_nodes;
    const ggml_init_params graph_params = {
        /* .mem_size = */ ggml_tensor_overhead() * 2 + ggml_graph_overhead_custom(combined_nodes, false),
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };
    ggml_context_ptr graph_context(ggml_init(graph_params));
    CHECK(graph_context != nullptr);
    ggml_cgraph * graph = ggml_new_graph_custom(graph_context.get(), combined_nodes, false);
    CHECK(graph != nullptr);
    for (int32_t node_index = 0; node_index < first.graph->n_nodes; ++node_index) {
        ggml_graph_add_node(graph, first.graph->nodes[node_index]);
    }
    const int32_t tail_index = graph->n_nodes;
    for (int32_t node_index = 0; node_index < second.graph->n_nodes; ++node_index) {
        ggml_graph_add_node(graph, first.graph->nodes[0]);
    }
    candidate_rebuild_graph_uses(graph);
    candidate_stamp_execution(graph, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
        GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 1, 1);

    const auto coverage = candidate_certify_graph(*context, graph);
    std::shared_ptr<ggml_cuda_moe_graph_plan> plan;
    ggml_cuda_moe_graph_execution execution;
    CHECK(context->prepare_graph_execution(
        graph, graph->uid, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &plan, &execution,
        coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
        GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED && execution.size() == 1);
    CHECK(execution.find(first.down_output, nullptr) && !execution.find(second.down_output, nullptr));
    const std::shared_ptr<ggml_cuda_moe_graph_plan> stale_plan = plan;

    for (int32_t node_index = 0; node_index < second.graph->n_nodes; ++node_index) {
        graph->nodes[tail_index + node_index] = second.graph->nodes[node_index];
    }
    candidate_rebuild_graph_uses(graph);
    const auto updated_coverage = candidate_certify_graph(*context, graph);
    CHECK(!context->bind_graph_plan(
        graph, graph->uid, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, *stale_plan, &execution,
        updated_coverage.epoch, updated_coverage.nodes,
        updated_coverage.mmid_count, updated_coverage.mmid_fingerprint));
    CHECK(context->prepare_graph_execution(
        graph, graph->uid, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &plan, &execution,
        updated_coverage.epoch, updated_coverage.nodes,
        updated_coverage.mmid_count, updated_coverage.mmid_fingerprint) ==
        GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
    CHECK(plan != stale_plan);
    if (second_supported) {
        CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED && execution.size() == 2);
        CHECK(execution.find(first.down_output, nullptr) && execution.find(second.down_output, nullptr));
    } else {
        CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR && execution.size() == 2);
        CHECK(!execution.find(first.down_output, nullptr) && !execution.find(second.down_output, nullptr));
    }

    const auto first_sentinel = active_grouped_intermediate_sentinel(first);
    const auto second_sentinel = active_grouped_intermediate_sentinel(second);
    const ggml_status status = ggml_backend_graph_compute(backend.get(), graph);
    ggml_backend_synchronize(backend.get());
    CHECK(status == (second_supported ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED));
    check_active_grouped_intermediates(first, first_sentinel, true);
    check_active_grouped_intermediates(second, second_sentinel, true);
    CHECK(active_grouped_legacy_op_count(backend.get()) == 0);
    if (second_supported) {
        for (const auto * current : {&first, &second}) {
            ggml_cuda_moe_candidate_group_key key;
            CHECK(context->find_down_group_key(current->down, &key));
            CHECK(ggml_cuda_moe_grouped_context_test_access::has_device_resource(*context, key));
        }
    }
}

static void test_active_grouped_in_place_inventory_reuse_case(bool registered) {
    ggml_backend_ptr backend(ggml_backend_cuda_init(0));
    CHECK(backend != nullptr);
    auto first = build_active_grouped_dispatch_graph(
        backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE);
    auto added = build_active_grouped_dispatch_graph(
        backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), registered ? GGML_TYPE_Q8_K : GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE);
    initialize_active_grouped_dispatch_graph(first, 701);

    std::array<std::array<ggml_backend_moe_candidate_bank_v1, 3>, 2> banks = {};
    std::array<ggml_backend_moe_candidate_group_v1, 2> groups = {};
    for (uint32_t group_index = 0; group_index < groups.size(); ++group_index) {
        const auto & source = group_index == 0 ? first : added;
        for (uint32_t bank_index = 0; bank_index < banks[group_index].size(); ++bank_index) {
            banks[group_index][bank_index] = {source.banks[bank_index], source.roles[bank_index], 0};
        }
        groups[group_index] = {
            banks[group_index].data(), static_cast<uint32_t>(banks[group_index].size()),
            GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 0, 0,
        };
    }
    const auto snapshot = candidate_snapshot(12, groups.data(), registered ? 2 : 1);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v1(backend.get(), &snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    auto * context = ggml_cuda_moe_grouped_context_for_test(backend.get());
    CHECK(context != nullptr && context->state().n_groups == (registered ? 2u : 1u));

    std::array<ggml_op, 3> saved_ops = {};
    for (uint32_t reader_index = 0; reader_index < added.readers.size(); ++reader_index) {
        saved_ops[reader_index] = added.readers[reader_index]->op;
        added.readers[reader_index]->op = GGML_OP_NONE;
    }
    const int32_t graph_capacity = first.graph->n_nodes + added.graph->n_nodes;
    const ggml_init_params graph_params = {
        /* .mem_size = */ ggml_tensor_overhead() * 2 + ggml_graph_overhead_custom(graph_capacity, false),
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };
    ggml_context_ptr graph_context(ggml_init(graph_params));
    CHECK(graph_context != nullptr);
    ggml_cgraph * graph = ggml_new_graph_custom(graph_context.get(), graph_capacity, false);
    CHECK(graph != nullptr);
    for (int32_t node_index = 0; node_index < first.graph->n_nodes; ++node_index) {
        ggml_graph_add_node(graph, first.graph->nodes[node_index]);
    }
    ggml_graph_add_node(graph, added.ids->src[0]);
    ggml_graph_add_node(graph, added.ids);
    for (ggml_tensor * reader : added.readers) {
        ggml_graph_add_node(graph, reader);
    }
    candidate_rebuild_graph_uses(graph);
    candidate_stamp_execution(graph, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
        GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 1, 1);
    const uint64_t source_graph_uid = graph->execution_certificate.source_graph_uid;
    std::array<int32_t, 3> saved_use_counts = {};
    for (uint32_t bank_index = 0; bank_index < added.banks.size(); ++bank_index) {
        saved_use_counts[bank_index] = candidate_graph_use_count(graph, added.banks[bank_index]);
        CHECK(saved_use_counts[bank_index] == 1);
    }

    const auto coverage = candidate_certify_graph(*context, graph);
    std::shared_ptr<ggml_cuda_moe_graph_plan> plan;
    ggml_cuda_moe_graph_execution execution;
    CHECK(context->prepare_graph_execution(
        graph, graph->uid, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &plan, &execution,
        coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
        GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED && execution.size() == 1);
    CHECK(execution.find(first.down_output, nullptr) && !execution.find(added.down_output, nullptr));
    const std::shared_ptr<ggml_cuda_moe_graph_plan> stale_plan = plan;
    CHECK(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend.get());
    CHECK(active_grouped_legacy_op_count(backend.get()) == 0);

    for (uint32_t reader_index = 0; reader_index < added.readers.size(); ++reader_index) {
        added.readers[reader_index]->op = saved_ops[reader_index];
    }
    candidate_rebuild_graph_uses(graph);
    candidate_stamp_execution(graph, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
        GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 1, 1, source_graph_uid);
    for (uint32_t bank_index = 0; bank_index < added.banks.size(); ++bank_index) {
        CHECK(candidate_graph_use_count(graph, added.banks[bank_index]) == saved_use_counts[bank_index]);
    }
    CHECK(!context->bind_graph_plan(
        graph, graph->uid, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, *stale_plan, &execution,
        coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint));
    CHECK(execution.size() == 0 && !execution.find(first.down_output, nullptr) && !execution.find(added.down_output, nullptr));
    const auto updated_coverage = candidate_certify_graph(*context, graph);
    CHECK(!context->bind_graph_plan(
        graph, graph->uid, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, *stale_plan, &execution,
        updated_coverage.epoch, updated_coverage.nodes,
        updated_coverage.mmid_count, updated_coverage.mmid_fingerprint));

    const auto first_sentinel = active_grouped_intermediate_sentinel(first);
    const auto added_sentinel = active_grouped_intermediate_sentinel(added);
    ggml_backend_synchronize(backend.get());
    CHECK(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_FAILED);
    ggml_backend_synchronize(backend.get());
    check_active_grouped_intermediates(first, first_sentinel, true);
    check_active_grouped_intermediates(added, added_sentinel, true);
    CHECK(active_grouped_legacy_op_count(backend.get()) == 0);
    for (uint32_t reader_index = 0; reader_index < added.readers.size(); ++reader_index) {
        added.readers[reader_index]->op = GGML_OP_NONE;
    }
    candidate_rebuild_graph_uses(graph);
    candidate_stamp_execution(graph, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
        GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 1, 1, source_graph_uid);
    CHECK(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend.get());
    CHECK(active_grouped_legacy_op_count(backend.get()) == 0);
}

static void test_active_grouped_inventory_reuse() {
    test_active_grouped_inventory_reuse_case(GGML_TYPE_Q4_K, true);
    test_active_grouped_in_place_inventory_reuse_case(true);
    test_active_grouped_in_place_inventory_reuse_case(false);
    fprintf(stderr, "test-moe-cache: complete MMID inventory production seam OK\n");
}

static std::vector<float> active_grouped_tensor_sentinel(ggml_tensor * tensor) {
    std::vector<float> result(ggml_nelements(tensor), -23456.5f);
    ggml_backend_tensor_set(tensor, result.data(), 0, ggml_nbytes(tensor));
    return result;
}

static std::vector<float> active_grouped_tensor_values(const ggml_tensor * tensor) {
    std::vector<float> result(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, result.data(), 0, ggml_nbytes(tensor));
    return result;
}

static void check_active_grouped_scale_shadows(
        ggml_cuda_moe_grouped_context & context,
        const ggml_cuda_moe_candidate_group_key & key,
        const active_grouped_dispatch_graph & graph,
        const std::array<int32_t, 2> & experts,
        const std::array<const float *, 3> & expected_pointers) {
    const std::array<ggml_tensor *, 3> scales = {graph.gate_scale, graph.up_scale, graph.down_scale};
    for (uint32_t scale = 0; scale < scales.size(); ++scale) {
        const float * shadow = ggml_cuda_moe_grouped_context_test_access::device_auxiliary_data(context, key, scales[scale]);
        CHECK(shadow != nullptr && shadow == expected_pointers[scale]);
        const auto original = active_grouped_tensor_values(scales[scale]);
        for (int32_t expert : experts) {
            const int32_t slot = ggml_cuda_moe_grouped_context_test_access::device_slot_for_expert(context, key, expert);
            CHECK(slot >= 0);
            float actual = 0.0f;
            CUDA_OK(cudaMemcpy(&actual, shadow + slot, sizeof(actual), cudaMemcpyDeviceToHost));
            CHECK(memcmp(&actual, &original[expert], sizeof(actual)) == 0);
        }
    }
}

static void test_active_grouped_nvfp4_scales() {
    const bool old_debug_mm = ggml_backend_cuda_moe_get_debug_mm();
    ggml_backend_cuda_moe_set_debug_mm(true);
    {
        constexpr uint32_t n_slots = 4;
        ggml_backend_ptr reference_backend(ggml_backend_cuda_init(0));
        ggml_backend_ptr candidate_backend(ggml_backend_cuda_init(0));
        CHECK(reference_backend != nullptr && candidate_backend != nullptr);
        auto reference = build_active_grouped_dispatch_graph_types(
            reference_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(),
            {GGML_TYPE_NVFP4, GGML_TYPE_NVFP4, GGML_TYPE_NVFP4},
            GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, false, true, 1);
        auto candidate = build_active_grouped_dispatch_graph_types(
            candidate_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(),
            {GGML_TYPE_NVFP4, GGML_TYPE_NVFP4, GGML_TYPE_NVFP4},
            GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, false, true, 1);
        initialize_active_grouped_dispatch_graphs({&reference, &candidate});
        const auto disabled = candidate_snapshot(n_slots, nullptr, 0);
        CHECK(ggml_backend_cuda_moe_candidate_replace_v1(reference_backend.get(), &disabled) ==
            GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        CHECK(replace_active_grouped_nvfp4_dispatch_v2(candidate_backend.get(), candidate, n_slots) ==
            GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);

        auto * context = ggml_cuda_moe_grouped_context_for_test(candidate_backend.get());
        ggml_cuda_moe_candidate_group_key key;
        CHECK(context != nullptr && context->find_down_group_key(candidate.down, &key));
        check_active_grouped_contract(candidate_backend.get(), candidate, n_slots);
        const std::array<ggml_tensor *, 3> scales = {candidate.gate_scale, candidate.up_scale, candidate.down_scale};
        std::array<const float *, 3> scale_pointers = {};
        for (uint32_t scale = 0; scale < scales.size(); ++scale) {
            scale_pointers[scale] = ggml_cuda_moe_grouped_context_test_access::device_auxiliary_data(
                *context, key, scales[scale]);
            CHECK(scale_pointers[scale] != nullptr);
        }

        ggml_cuda_graph_capture_state_for_test captured = {};
        const std::array<int32_t, 2> first_experts = {
            active_grouped_route(candidate, 0, 0, 0), active_grouped_route(candidate, 0, 0, 1),
        };
        const std::array<int32_t, 2> middle_experts = {
            active_grouped_route(candidate, 1, 0, 0), active_grouped_route(candidate, 1, 0, 1),
        };
        const std::array<int32_t, 2> replacement_experts = {
            active_grouped_route(candidate, 2, 0, 0), active_grouped_route(candidate, 2, 0, 1),
        };
        for (uint32_t pass = 0; pass < 6; ++pass) {
            const uint32_t route_variant = pass >= 4 ? 2 : pass >= 2 ? 1 : 0;
            set_active_grouped_dispatch_logits({&reference, &candidate}, route_variant);
            const auto reference_down_sentinel = active_grouped_tensor_sentinel(reference.down_output);
            const auto candidate_down_sentinel = active_grouped_tensor_sentinel(candidate.down_output);
            const auto expected = run_active_grouped_dispatch(reference_backend.get(), reference, 0, false);
            const auto actual = run_active_grouped_dispatch(
                candidate_backend.get(), candidate, candidate.n_used * (pass + 2), true);
            check_active_grouped_exact_output(expected, actual);
            CHECK(active_grouped_tensor_values(reference.down_output) != reference_down_sentinel);
            CHECK(active_grouped_tensor_values(candidate.down_output) == candidate_down_sentinel);
            for (uint32_t scale = 0; scale < scales.size(); ++scale) {
                CHECK(ggml_cuda_moe_grouped_context_test_access::device_auxiliary_data(
                    *context, key, scales[scale]) == scale_pointers[scale]);
            }

            ggml_cuda_graph_capture_state_for_test state = {};
            CHECK(ggml_cuda_graph_capture_state_query_for_test(candidate_backend.get(), candidate.graph, &state));
            if (pass == 0 || !state.capture_available) {
                CHECK(state.graph == 0 && state.instance == 0 && !state.warmup_complete &&
                    state.moe_resource_fingerprint == 0);
            } else if (pass == 1) {
                CHECK(state.graph != 0 && state.instance != 0 && state.warmup_complete &&
                    state.execution_semantic_key != 0 && state.moe_resource_fingerprint != 0);
                captured = state;
            } else {
                CHECK(state.graph == captured.graph && state.instance == captured.instance &&
                    state.execution_semantic_key == captured.execution_semantic_key &&
                    state.moe_resource_fingerprint == captured.moe_resource_fingerprint);
            }
            if (pass == 0) {
                check_active_grouped_scale_shadows(*context, key, candidate, first_experts, scale_pointers);
            }
            if (pass == 3) {
                check_active_grouped_scale_shadows(*context, key, candidate, middle_experts, scale_pointers);
            }
            if (pass == 5) {
                check_active_grouped_scale_shadows(*context, key, candidate, replacement_experts, scale_pointers);
                for (int32_t expert : first_experts) {
                    CHECK(ggml_cuda_moe_grouped_context_test_access::device_slot_for_expert(*context, key, expert) == -1);
                }
            }
        }
        CHECK(active_grouped_legacy_op_count(candidate_backend.get(), true) == 0);
        CHECK(active_grouped_legacy_op_count(candidate_backend.get(), false) == 0);
        const auto telemetry = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
        CHECK(telemetry.registered == 1 && telemetry.covered == 1);
        CHECK(telemetry.plan_calls == 8 && telemetry.plan_compiles == 2 && telemetry.plan_reuses == 6);
        CHECK(telemetry.calls == 7 && telemetry.ready == 7 && telemetry.completed == 7);
        CHECK(telemetry.admitted_banks == 7 * candidate.banks.size());
        CHECK(telemetry.fallback == 0 && telemetry.rollback == 0 &&
            telemetry.prepare_error == 0 && telemetry.finish_error == 0);
        CHECK(telemetry.h2d_banks == 6 * candidate.banks.size());
        uint64_t bytes_per_expert = 0;
        for (const ggml_tensor * bank : candidate.banks) {
            bytes_per_expert += bank->nb[2];
        }
        CHECK(telemetry.h2d_bytes == 6 * bytes_per_expert);
    }

    constexpr uint32_t n_rows = 2;
    for (uint32_t n_slots : {12u, 48u}) {
        ggml_backend_ptr reference_backend(ggml_backend_cuda_init(0));
        ggml_backend_ptr candidate_backend(ggml_backend_cuda_init(0));
        CHECK(reference_backend != nullptr && candidate_backend != nullptr);
        auto reference = build_active_grouped_dispatch_graph_types(
            reference_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(),
            {GGML_TYPE_NVFP4, GGML_TYPE_NVFP4, GGML_TYPE_NVFP4},
            GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, false, true, n_rows);
        auto candidate = build_active_grouped_dispatch_graph_types(
            candidate_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(),
            {GGML_TYPE_NVFP4, GGML_TYPE_NVFP4, GGML_TYPE_NVFP4},
            GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, false, true, n_rows);
        initialize_active_grouped_dispatch_graphs({&reference, &candidate});
        const auto disabled = candidate_snapshot(n_slots, nullptr, 0);
        CHECK(ggml_backend_cuda_moe_candidate_replace_v1(reference_backend.get(), &disabled) ==
            GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        CHECK(replace_active_grouped_nvfp4_dispatch_v2(candidate_backend.get(), candidate, n_slots) ==
            GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);

        auto * context = ggml_cuda_moe_grouped_context_for_test(candidate_backend.get());
        ggml_cuda_moe_candidate_group_key key;
        ggml_cuda_moe_candidate_group_info group_info;
        CHECK(context != nullptr && context->find_down_group_key(candidate.down, &key));
        CHECK(context->get_group(key, &group_info) && group_info.n_banks == 6);
        const std::array<ggml_tensor *, 3> scales = {candidate.gate_scale, candidate.up_scale, candidate.down_scale};
        const std::array<uint32_t, 3> scale_roles = {
            GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_SCALE,
            GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_SCALE,
            GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_SCALE,
        };
        for (uint32_t i = 0; i < candidate.banks.size(); ++i) {
            ggml_cuda_moe_candidate_bank_info bank_info;
            CHECK(context->get_bank(key, candidate.roles[i], &bank_info));
            CHECK(bank_info.tensor == candidate.banks[i] && bank_info.type == GGML_TYPE_NVFP4);
            CHECK(bank_info.encoding == GGML_CUDA_MOE_CANDIDATE_ENCODING_NVFP4_COMPOUND);
            CHECK(bank_info.movement == GGML_CUDA_MOE_CANDIDATE_MOVEMENT_SLOT_BOUND);
            CHECK(bank_info.index_modes == GGML_CUDA_MOE_CANDIDATE_INDEX_GROUP_SLOT_DIRECT);
        }
        for (uint32_t i = 0; i < scales.size(); ++i) {
            ggml_cuda_moe_candidate_bank_info scale_info;
            CHECK(context->get_bank(key, scale_roles[i], &scale_info));
            CHECK(scale_info.tensor == scales[i] && scale_info.type == GGML_TYPE_F32);
            CHECK(scale_info.movement == GGML_CUDA_MOE_CANDIDATE_MOVEMENT_PERMANENT_CANDIDATE);
            CHECK(scale_info.index_modes == GGML_CUDA_MOE_CANDIDATE_INDEX_ORIGINAL_DIRECT);
        }

        check_active_grouped_contract(candidate_backend.get(), candidate, n_slots, false, true);
        std::vector<float> expected;
        std::vector<float> actual;
        ggml_cuda_graph_capture_state_for_test captured = {};
        for (int pass = 0; pass < 4; ++pass) {
            const auto candidate_down_sentinel = active_grouped_tensor_sentinel(candidate.down_output);
            const auto current_expected = run_active_grouped_dispatch(reference_backend.get(), reference, 0, false);
            const auto current_actual = run_active_grouped_dispatch(
                candidate_backend.get(), candidate, n_rows * candidate.n_used * (pass + 2), false);
            check_active_grouped_exact_output(current_expected, current_actual);
            CHECK(active_grouped_tensor_values(candidate.down_output) != candidate_down_sentinel);
            check_active_grouped_routes(reference, n_rows, 0);
            check_active_grouped_routes(candidate, n_rows, 0);
            if (pass == 0) {
                expected = current_expected;
                actual = current_actual;
            } else {
                CHECK(current_expected == expected && current_actual == actual);
            }
            ggml_cuda_graph_capture_state_for_test state = {};
            CHECK(ggml_cuda_graph_capture_state_query_for_test(candidate_backend.get(), candidate.graph, &state));
            if (pass == 0 || !state.capture_available) {
                CHECK(state.graph == 0 && state.instance == 0 && !state.warmup_complete &&
                    state.moe_resource_fingerprint == 0);
            } else if (pass == 1) {
                CHECK(state.graph != 0 && state.instance != 0 && state.warmup_complete &&
                    state.execution_semantic_key != 0 && state.moe_resource_fingerprint != 0);
                captured = state;
            } else {
                CHECK(state.graph == captured.graph && state.instance == captured.instance &&
                    state.execution_semantic_key == captured.execution_semantic_key &&
                    state.moe_resource_fingerprint == captured.moe_resource_fingerprint);
            }
        }
        CHECK(expected.size() == actual.size());
        CHECK(memcmp(expected.data(), actual.data(), expected.size() * sizeof(float)) == 0);
        CHECK(active_grouped_legacy_op_count(reference_backend.get(), true) == 0);
        CHECK(active_grouped_legacy_op_count(reference_backend.get(), false) == 4 * reference.banks.size());
        CHECK(active_grouped_legacy_op_count(candidate_backend.get(), true) == 0);
        CHECK(active_grouped_legacy_op_count(candidate_backend.get(), false) == 0);
        check_active_grouped_debug_telemetry(candidate_backend.get(), candidate);
        check_active_grouped_legacy_caches(reference_backend.get(), reference, false);
    }

    {
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        CHECK(backend != nullptr);
        auto graph = build_active_grouped_dispatch_graph_types(
            backend.get(), ggml_backend_cuda_moe_cached_buffer_type(),
            {GGML_TYPE_NVFP4, GGML_TYPE_NVFP4, GGML_TYPE_NVFP4},
            GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, false, true, n_rows);
        initialize_active_grouped_dispatch_graph(graph, 677);
        CHECK(replace_active_grouped_nvfp4_dispatch_v2(backend.get(), graph, 12) ==
            GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        auto * context = ggml_cuda_moe_grouped_context_for_test(backend.get());
        CHECK(context != nullptr);

        graph.gate_output->ne[2] = n_rows + 1;
        candidate_rebuild_graph_uses(graph.graph);
        auto coverage = candidate_certify_graph(*context, graph.graph);
        std::shared_ptr<ggml_cuda_moe_graph_plan> plan;
        ggml_cuda_moe_graph_execution execution;
        CHECK(context->prepare_graph_execution(
            graph.graph, 850, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &plan, &execution,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR);
        CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_geometry_reason(*plan, 0));
        CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_auxiliary_node_count(*plan, 0) == 0);
        for (uint32_t bank_index = 0; bank_index < 6; ++bank_index) {
            CHECK(!ggml_cuda_moe_grouped_context_test_access::graph_group_has_bank_use(*plan, 0, bank_index));
        }
        const ggml_cuda_moe_graph_plan * geometry_plan = plan.get();
        CHECK(context->prepare_graph_execution(
            graph.graph, 851, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &plan, &execution,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(plan.get() == geometry_plan && execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR);

        graph.gate_output->ne[2] = n_rows;
        candidate_rebuild_graph_uses(graph.graph);
        coverage = candidate_certify_graph(*context, graph.graph);
        CHECK(context->prepare_graph_execution(
            graph.graph, 852, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &plan, &execution,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED);
        CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_auxiliary_node_count(*plan, 0) == 9);

        graph.gate_scale_rows->nb[3]++;
        CHECK(context->prepare_graph_execution(
            graph.graph, 8521, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &plan, &execution,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR);
        graph.gate_scale_rows->nb[3]--;
        CHECK(context->prepare_graph_execution(
            graph.graph, 8522, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &plan, &execution,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED);
        const ggml_cuda_moe_graph_plan * restored_metadata_plan = plan.get();
        CHECK(context->prepare_graph_execution(
            graph.graph, 8523, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &plan, &execution,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(plan.get() == restored_metadata_plan && execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED);

        graph.gate_output->ne[2] = n_rows + 1;
        candidate_rebuild_graph_uses(graph.graph);
        coverage = candidate_certify_graph(*context, graph.graph);
        CHECK(context->prepare_graph_execution(
            graph.graph, 853, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &plan, &execution,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR);
        CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_geometry_reason(*plan, 0));
        CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_auxiliary_node_count(*plan, 0) == 0);
    }

    {
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        CHECK(backend != nullptr);
        auto graph = build_active_grouped_dispatch_graph_types(
            backend.get(), ggml_backend_cuda_moe_cached_buffer_type(),
            {GGML_TYPE_NVFP4, GGML_TYPE_NVFP4, GGML_TYPE_NVFP4},
            GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, false, true, n_rows);
        initialize_active_grouped_dispatch_graph(graph, 689);
        CHECK(replace_active_grouped_nvfp4_dispatch_v2(backend.get(), graph, 12) ==
            GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        auto * context = ggml_cuda_moe_grouped_context_for_test(backend.get());
        CHECK(context != nullptr);

        ggml_tensor * fourth_consumer = ggml_dup(graph.nodes.get(), graph.output);
        fourth_consumer->src[0] = graph.gate_output;
        ggml_graph_add_node(graph.graph, fourth_consumer);
        candidate_rebuild_graph_uses(graph.graph);
        CHECK(candidate_graph_use_count(graph.graph, graph.gate_output) == 2);
        const auto coverage = candidate_certify_graph(*context, graph.graph);
        std::shared_ptr<ggml_cuda_moe_graph_plan> plan;
        ggml_cuda_moe_graph_execution execution;
        CHECK(context->prepare_graph_execution(
            graph.graph, 870, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &plan, &execution,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR && execution.size() == 1 && execution.requires_dispatch());
        CHECK(execution.find_group(graph.gate_output, nullptr) == nullptr);
        const ggml_cuda_moe_graph_plan * fourth_consumer_plan = plan.get();
        CHECK(context->prepare_graph_execution(
            graph.graph, 871, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &plan, &execution,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(plan.get() == fourth_consumer_plan && execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR);
        CHECK(execution.find_group(graph.gate_output, nullptr) == nullptr);

        fourth_consumer->src[0] = graph.output;
        candidate_rebuild_graph_uses(graph.graph);
        CHECK(candidate_graph_use_count(graph.graph, graph.gate_output) == 1);
        CHECK(context->prepare_graph_execution(
            graph.graph, 872, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &plan, &execution,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED && execution.size() == 1);
        CHECK(execution.find_group(graph.gate_output, nullptr) != nullptr);
        const ggml_cuda_moe_graph_plan * exact_consumer_plan = plan.get();
        CHECK(context->prepare_graph_execution(
            graph.graph, 873, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &plan, &execution,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(plan.get() == exact_consumer_plan && execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED);

        fourth_consumer->src[0] = graph.gate_output;
        candidate_rebuild_graph_uses(graph.graph);
        CHECK(context->prepare_graph_execution(
            graph.graph, 874, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &plan, &execution,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR && execution.size() == 1 && execution.requires_dispatch());
        CHECK(execution.find_group(graph.gate_output, nullptr) == nullptr);
    }

    for (int32_t omitted_scale = 0; omitted_scale < 3; ++omitted_scale) {
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        CHECK(backend != nullptr);
        auto graph = build_active_grouped_dispatch_graph_types(
            backend.get(), ggml_backend_cuda_moe_cached_buffer_type(),
            {GGML_TYPE_NVFP4, GGML_TYPE_NVFP4, GGML_TYPE_NVFP4},
            GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, false, true, n_rows);
        initialize_active_grouped_dispatch_graph(graph, 701 + omitted_scale);
        CHECK(replace_active_grouped_nvfp4_dispatch_v2(backend.get(), graph, 12, omitted_scale) ==
            GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        auto * context = ggml_cuda_moe_grouped_context_for_test(backend.get());
        CHECK(context != nullptr);
        const auto coverage = candidate_certify_graph(*context, graph.graph);
        std::shared_ptr<ggml_cuda_moe_graph_plan> plan;
        ggml_cuda_moe_graph_execution execution;
        CHECK(context->prepare_graph_execution(
            graph.graph, 910 + omitted_scale, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &plan, &execution,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR && execution.requires_dispatch());
        CHECK(ggml_backend_graph_compute(backend.get(), graph.graph) == GGML_STATUS_FAILED);
        CHECK(active_grouped_legacy_op_count(backend.get()) == 0);
    }
    ggml_backend_cuda_moe_set_debug_mm(old_debug_mm);
    fprintf(stderr, "test-moe-cache: compound NVFP4 B2 original-ID output scales exact OK\n");
}

static void test_active_grouped_dispatch_generic() {
    for (ggml_type type : {GGML_TYPE_Q3_K, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S}) {
        for (uint32_t layout : {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP}) {
            for (uint32_t n_slots : {12u, 48u}) {
                test_active_grouped_dispatch_case(type, layout, n_slots);
            }
        }
    }
    for (uint32_t n_slots : {12u, 48u}) {
        test_active_grouped_dispatch_types_case(
            {GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_IQ3_S}, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, n_slots);
        test_active_grouped_dispatch_types_case(
            {GGML_TYPE_Q3_K, GGML_TYPE_BF16, GGML_TYPE_IQ3_S}, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, n_slots);
    }
    fprintf(stderr, "test-moe-cache: active grouped Q3 and mixed ordinary/F3 OK\n");
}

static void test_active_grouped_fused_down_scale_extent() {
    const bool old_debug_mm = ggml_backend_cuda_moe_get_debug_mm();
    ggml_backend_cuda_moe_set_debug_mm(true);
    ggml_backend_ptr backend(ggml_backend_cuda_init(0));
    CHECK(backend != nullptr);
    auto graph = build_active_grouped_dispatch_graph(
        backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, true);
    initialize_active_grouped_dispatch_graph(graph, 733);
    register_active_grouped_dispatch(
        backend.get(), graph, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 12);
    auto * context = ggml_cuda_moe_grouped_context_for_test(backend.get());
    CHECK(context != nullptr);

    const auto coverage = candidate_certify_graph(*context, graph.graph);
    std::shared_ptr<ggml_cuda_moe_graph_plan> plan;
    ggml_cuda_moe_graph_execution execution;
    CHECK(context->prepare_graph_execution(
        graph.graph, 880, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &plan, &execution,
        coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
        GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED);

    graph.down_scale_reshape->ne[1]++;
    graph.down_scale_repeat->ne[1]++;
    CHECK(context->prepare_graph_execution(
        graph.graph, 881, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &plan, &execution,
        coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
        GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR);

    graph.down_scale_reshape->ne[1]--;
    graph.down_scale_repeat->ne[1]--;
    CHECK(context->prepare_graph_execution(
        graph.graph, 882, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &plan, &execution,
        coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
        GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED);
    const ggml_cuda_moe_graph_plan * recovered_plan = plan.get();
    CHECK(context->prepare_graph_execution(
        graph.graph, 883, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &plan, &execution,
        coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
        GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
    CHECK(plan.get() == recovered_plan && execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED);
    ggml_backend_cuda_moe_set_debug_mm(old_debug_mm);
}

static void test_prefill_resident_biases() {
    const bool old_debug_mm = ggml_backend_cuda_moe_get_debug_mm();
    ggml_backend_cuda_moe_set_debug_mm(true);
    constexpr uint32_t n_slots = 12;
    constexpr uint32_t n_rows = 2;
    constexpr uint32_t n_experts = 128;
    constexpr uint32_t n_used = 8;

    ggml_backend_ptr reference_backend(ggml_backend_cuda_init(0));
    ggml_backend_ptr candidate_backend(ggml_backend_cuda_init(0));
    ggml_backend_ptr parallel_backend(ggml_backend_cuda_init(0));
    ggml_backend_ptr declined_backend(ggml_backend_cuda_init(0));
    CHECK(reference_backend != nullptr && candidate_backend != nullptr &&
        parallel_backend != nullptr && declined_backend != nullptr);
    auto make_graph = [](ggml_backend_t backend, uint32_t rows, uint32_t experts, uint32_t used) {
        return build_active_grouped_dispatch_graph_types(
            backend, ggml_backend_cuda_moe_cached_buffer_type(),
            {GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0},
            GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, false, false,
            rows, experts, used, 64, nullptr, false, true, 96, true);
    };
    auto reference = make_graph(reference_backend.get(), n_rows, n_experts, n_used);
    auto candidate = make_graph(candidate_backend.get(), n_rows, n_experts, n_used);
    auto parallel = make_graph(parallel_backend.get(), n_rows, n_experts, n_used);
    auto declined = make_graph(declined_backend.get(), n_rows, n_experts, n_used);
    for (auto * graph : {&reference, &candidate, &parallel, &declined}) {
        candidate_stamp_execution(graph->graph, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
            GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL, n_rows, 1);
    }
    initialize_active_grouped_dispatch_graphs({&reference, &candidate, &parallel, &declined});

    size_t bias_bytes = 0;
    for (const ggml_tensor * bias : candidate.biases) {
        CHECK(ggml_nbytes(bias) <= SIZE_MAX - bias_bytes);
        bias_bytes += ggml_nbytes(bias);
    }
    CHECK(bias_bytes > 1 && candidate.biases.size() == 3);
    const auto disabled = candidate_snapshot(n_slots, nullptr, 0);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v1(reference_backend.get(), &disabled) ==
        GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v1(candidate_backend.get(), &disabled) ==
        GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v1(parallel_backend.get(), &disabled) ==
        GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v1(declined_backend.get(), &disabled) ==
        GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    auto * candidate_context = ggml_cuda_moe_grouped_context_for_test(candidate_backend.get());
    auto * parallel_context = ggml_cuda_moe_grouped_context_for_test(parallel_backend.get());
    auto * declined_context = ggml_cuda_moe_grouped_context_for_test(declined_backend.get());
    CHECK(candidate_context != nullptr && parallel_context != nullptr && declined_context != nullptr);
    CHECK(ggml_cuda_moe_grouped_context_test_access::set_prefill_resident_budget(*candidate_context, bias_bytes));
    CHECK(ggml_cuda_moe_grouped_context_test_access::set_prefill_resident_budget(*parallel_context, bias_bytes));
    CHECK(ggml_cuda_moe_grouped_context_test_access::set_prefill_resident_budget(*declined_context, bias_bytes - 1));

    register_active_grouped_dispatch(
        candidate_backend.get(), candidate, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, n_slots);
    register_active_grouped_dispatch(
        parallel_backend.get(), parallel, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, n_slots);
    register_active_grouped_dispatch(
        declined_backend.get(), declined, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, n_slots);
    const auto candidate_coverage = candidate_certify_graph(*candidate_context, candidate.graph);
    (void) candidate_certify_graph(*parallel_context, parallel.graph);
    (void) candidate_certify_graph(*declined_context, declined.graph);
    std::shared_ptr<ggml_cuda_moe_graph_plan> prefill_plan;
    ggml_cuda_moe_graph_execution prefill_execution;
    CHECK(candidate_context->prepare_graph_execution(
        candidate.graph, 990, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &prefill_plan, &prefill_execution,
        candidate_coverage.epoch, candidate_coverage.nodes, candidate_coverage.mmid_count,
        candidate_coverage.mmid_fingerprint) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
    CHECK(prefill_plan->outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_PREFILL_LEGACY);
    CHECK(prefill_plan->has_certified_complete_mmid_inventory());
    CHECK(ggml_cuda_moe_grouped_context_test_access::prefill_add_id_witness_count(*prefill_plan) == candidate.biases.size());
    for (ggml_tensor * reader : candidate.readers) {
        CHECK(prefill_execution.find_group(reader, nullptr) != nullptr);
    }

    const auto expected = run_active_grouped_dispatch(reference_backend.get(), reference, 0, false);
    std::array<std::vector<float>, 2> parallel_outputs;
    std::thread first([&]() {
        parallel_outputs[0] = run_active_grouped_dispatch(candidate_backend.get(), candidate, 0, false);
    });
    std::thread second([&]() {
        parallel_outputs[1] = run_active_grouped_dispatch(parallel_backend.get(), parallel, 0, false);
    });
    first.join();
    second.join();
    check_active_grouped_exact_output(expected, parallel_outputs[0]);
    check_active_grouped_exact_output(expected, parallel_outputs[1]);
    check_active_grouped_exact_output(
        expected, run_active_grouped_dispatch(declined_backend.get(), declined, 0, false));
    check_active_grouped_routes(candidate, n_rows, 0);
    CHECK(active_grouped_route(candidate, 0, 0, n_used - 1) >= static_cast<int32_t>(n_slots));

    ggml_cuda_moe_candidate_group_key candidate_key;
    ggml_cuda_moe_candidate_group_key parallel_key;
    ggml_cuda_moe_candidate_group_key declined_key;
    CHECK(candidate_context->find_down_group_key(candidate.down, &candidate_key));
    CHECK(parallel_context->find_down_group_key(parallel.down, &parallel_key));
    CHECK(declined_context->find_down_group_key(declined.down, &declined_key));
    std::array<const float *, 3> candidate_residents = {};
    uint64_t first_resource_generation = 0;
    for (uint32_t i = 0; i < candidate.biases.size(); ++i) {
        size_t extent = 0;
        uint64_t generation = 0;
        candidate_residents[i] = ggml_cuda_moe_grouped_context_test_access::device_prefill_auxiliary_data(
            *candidate_context, candidate_key, candidate.biases[i], &extent, &generation);
        CHECK(candidate_residents[i] != nullptr && extent == ggml_nbytes(candidate.biases[i]) && generation != 0);
        CHECK(candidate_residents[i] != ggml_cuda_moe_grouped_context_test_access::device_auxiliary_data(
            *candidate_context, candidate_key, candidate.biases[i]));
        if (i == 0) {
            first_resource_generation = generation;
        } else {
            CHECK(generation == first_resource_generation);
        }
        std::vector<float> resident(ggml_nelements(candidate.biases[i]));
        CUDA_OK(cudaMemcpy(resident.data(), candidate_residents[i], extent, cudaMemcpyDeviceToHost));
        CHECK(resident == active_grouped_tensor_values(candidate.biases[i]));

        const float * parallel_resident = ggml_cuda_moe_grouped_context_test_access::device_prefill_auxiliary_data(
            *parallel_context, parallel_key, parallel.biases[i]);
        CHECK(parallel_resident != nullptr && parallel_resident != candidate_residents[i]);
        CHECK(ggml_cuda_moe_grouped_context_test_access::device_prefill_auxiliary_data(
            *declined_context, declined_key, declined.biases[i]) == nullptr);
    }

    std::array<const ggml_tensor *, 3> add_nodes = {};
    for (int32_t node_index = 0; node_index < ggml_graph_n_nodes(candidate.graph); ++node_index) {
        const ggml_tensor * node = ggml_graph_node(candidate.graph, node_index);
        if (node->op != GGML_OP_ADD_ID) {
            continue;
        }
        for (uint32_t bias = 0; bias < candidate.biases.size(); ++bias) {
            if (node->src[1] == candidate.biases[bias]) {
                CHECK(add_nodes[bias] == nullptr);
                add_nodes[bias] = node;
            }
        }
    }
    CHECK(std::all_of(add_nodes.begin(), add_nodes.end(), [](const ggml_tensor * node) { return node != nullptr; }));
    const uint32_t routed_expert = active_grouped_route(candidate, 0, 0, 0);
    const size_t first_row_bytes = candidate.biases[0]->ne[0] * sizeof(float);
    const size_t second_row_bytes = candidate.biases[1]->ne[0] * sizeof(float);
    std::vector<float> first_expected(candidate.biases[0]->ne[0]);
    std::vector<float> second_expected(candidate.biases[1]->ne[0]);
    const auto first_original = active_grouped_tensor_values(candidate.biases[0]);
    const auto second_original = active_grouped_tensor_values(candidate.biases[1]);
    memcpy(first_expected.data(), first_original.data() + routed_expert * candidate.biases[0]->ne[0], first_row_bytes);
    memcpy(second_expected.data(), second_original.data() + routed_expert * candidate.biases[1]->ne[0], second_row_bytes);
    CHECK(std::any_of(first_expected.begin(), first_expected.end(), [](float value) { return value != 0.0f; }));

    cudaStream_t first_stream = nullptr;
    cudaStream_t second_stream = nullptr;
    float * staged = nullptr;
    float * first_result = nullptr;
    float * second_result = nullptr;
    CUDA_OK(cudaStreamCreateWithFlags(&first_stream, cudaStreamNonBlocking));
    CUDA_OK(cudaStreamCreateWithFlags(&second_stream, cudaStreamNonBlocking));
    CHECK(first_stream != second_stream);
    CUDA_OK(cudaMalloc(&staged, first_row_bytes));
    CUDA_OK(cudaMalloc(&first_result, first_row_bytes));
    CUDA_OK(cudaMalloc(&second_result, second_row_bytes));
    uint64_t waits_before = 0;
    uint64_t declines_before = 0;
    CHECK(ggml_cuda_moe_grouped_context_test_access::prefill_auxiliary_ordering(
        *candidate_context, candidate_key, &waits_before, &declines_before));
    CHECK(candidate_context->begin_graph_dispatch(&prefill_execution, GGML_CUDA_MOE_GRAPH_DISPATCH_LEGACY));
    const float * first_source = nullptr;
    const float * second_source = nullptr;
    CHECK(candidate_context->prefill_add_id_source(
        prefill_execution, add_nodes[0], first_stream, &first_source));
    CHECK(first_source == candidate_residents[0]);
    CHECK(!candidate_context->prefill_add_id_source(
        prefill_execution, add_nodes[1], second_stream, &second_source));
    CHECK(second_source == nullptr);
    CUDA_OK(cudaMemsetAsync(staged, 0, first_row_bytes, first_stream));
    host_barrier first_barrier;
    CUDA_OK(cudaLaunchHostFunc(first_stream, wait_on_host_barrier, &first_barrier));
    CUDA_OK(cudaMemcpyAsync(staged, first_source + routed_expert * candidate.biases[0]->ne[0],
        first_row_bytes, cudaMemcpyDeviceToDevice, first_stream));
    CHECK(candidate_context->finish_prefill_add_id(
        prefill_execution, add_nodes[0], first_stream, first_source));
    while (!first_barrier.entered.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    CHECK(candidate_context->prefill_add_id_source(
        prefill_execution, add_nodes[1], second_stream, &second_source));
    CHECK(second_source == candidate_residents[1]);
    CUDA_OK(cudaMemcpyAsync(first_result, staged, first_row_bytes, cudaMemcpyDeviceToDevice, second_stream));
    CUDA_OK(cudaMemcpyAsync(second_result, second_source + routed_expert * candidate.biases[1]->ne[0],
        second_row_bytes, cudaMemcpyDeviceToDevice, second_stream));
    CHECK(candidate_context->finish_prefill_add_id(
        prefill_execution, add_nodes[1], second_stream, second_source));
    CHECK(candidate_context->finish_graph_dispatch(&prefill_execution));
    uint64_t waits_after = 0;
    uint64_t declines_after = 0;
    CHECK(ggml_cuda_moe_grouped_context_test_access::prefill_auxiliary_ordering(
        *candidate_context, candidate_key, &waits_after, &declines_after));
    CHECK(waits_after == waits_before + 2 && declines_after == declines_before + 1);
    first_barrier.released.store(true, std::memory_order_release);
    CUDA_OK(cudaStreamSynchronize(second_stream));
    CUDA_OK(cudaStreamSynchronize(first_stream));
    std::vector<float> first_actual(first_expected.size());
    std::vector<float> second_actual(second_expected.size());
    CUDA_OK(cudaMemcpy(first_actual.data(), first_result, first_row_bytes, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(second_actual.data(), second_result, second_row_bytes, cudaMemcpyDeviceToHost));
    CHECK(first_actual == first_expected && second_actual == second_expected);
    CUDA_OK(cudaFree(second_result));
    CUDA_OK(cudaFree(first_result));
    CUDA_OK(cudaFree(staged));
    CUDA_OK(cudaStreamDestroy(second_stream));
    CUDA_OK(cudaStreamDestroy(first_stream));
    check_active_grouped_exact_output(
        expected, run_active_grouped_dispatch(candidate_backend.get(), candidate, 0, false));

    for (uint32_t i = 0; i < candidate.biases.size(); ++i) {
        const auto changed = cached_fusion_test_data(candidate.biases[i], 900 + i);
        ggml_backend_tensor_set(candidate.biases[i], changed.data(), 0, changed.size());
    }
    check_active_grouped_exact_output(
        expected, run_active_grouped_dispatch(candidate_backend.get(), candidate, 0, false));
    check_active_grouped_exact_output(
        expected, run_active_grouped_dispatch(parallel_backend.get(), parallel, 0, false));

    for (uint32_t i = 0; i < declined.biases.size(); ++i) {
        const auto changed = cached_fusion_test_data(declined.biases[i], 950 + i);
        ggml_backend_tensor_set(declined.biases[i], changed.data(), 0, changed.size());
        ggml_backend_tensor_set(reference.biases[i], changed.data(), 0, changed.size());
    }
    const auto declined_expected = run_active_grouped_dispatch(reference_backend.get(), reference, 0, false);
    const auto declined_actual = run_active_grouped_dispatch(declined_backend.get(), declined, 0, false);
    check_active_grouped_exact_output(declined_expected, declined_actual);
    CHECK(declined_actual != expected);

    for (uint32_t i = 0; i < candidate.biases.size(); ++i) {
        const auto changed = cached_fusion_test_data(candidate.biases[i], 900 + i);
        ggml_backend_tensor_set(reference.biases[i], changed.data(), 0, changed.size());
    }
    register_active_grouped_dispatch(
        candidate_backend.get(), candidate, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, n_slots);
    (void) candidate_certify_graph(*candidate_context, candidate.graph);
    ggml_cuda_moe_candidate_group_key refreshed_key;
    CHECK(candidate_context->find_down_group_key(candidate.down, &refreshed_key));
    CHECK(refreshed_key.generation != candidate_key.generation);
    const auto refreshed_expected = run_active_grouped_dispatch(reference_backend.get(), reference, 0, false);
    const auto refreshed_actual = run_active_grouped_dispatch(candidate_backend.get(), candidate, 0, false);
    check_active_grouped_exact_output(refreshed_expected, refreshed_actual);
    CHECK(refreshed_actual != expected);
    size_t refreshed_extent = 0;
    uint64_t refreshed_generation = 0;
    CHECK(ggml_cuda_moe_grouped_context_test_access::device_prefill_auxiliary_data(
        *candidate_context, candidate_key, candidate.biases[0]) == nullptr);
    CHECK(ggml_cuda_moe_grouped_context_test_access::device_prefill_auxiliary_data(
        *candidate_context, refreshed_key, candidate.biases[0], &refreshed_extent, &refreshed_generation) != nullptr);
    CHECK(refreshed_extent == ggml_nbytes(candidate.biases[0]) && refreshed_generation != first_resource_generation);

    {
        ggml_backend_ptr decode_reference_backend(ggml_backend_cuda_init(0));
        ggml_backend_ptr decode_candidate_backend(ggml_backend_cuda_init(0));
        CHECK(decode_reference_backend != nullptr && decode_candidate_backend != nullptr);
        auto decode_reference = make_graph(decode_reference_backend.get(), 1, 8, 2);
        auto decode_candidate = make_graph(decode_candidate_backend.get(), 1, 8, 2);
        initialize_active_grouped_dispatch_graphs({&decode_reference, &decode_candidate});
        const auto decode_disabled = candidate_snapshot(3, nullptr, 0);
        CHECK(ggml_backend_cuda_moe_candidate_replace_v1(decode_reference_backend.get(), &decode_disabled) ==
            GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        register_active_grouped_dispatch(
            decode_candidate_backend.get(), decode_candidate, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 3);
        auto * decode_context = ggml_cuda_moe_grouped_context_for_test(decode_candidate_backend.get());
        CHECK(decode_context != nullptr);
        (void) candidate_certify_graph(*decode_context, decode_candidate.graph);
        const auto decode_expected = run_active_grouped_dispatch(decode_reference_backend.get(), decode_reference, 0, false);
        const auto decode_actual = run_active_grouped_dispatch(
            decode_candidate_backend.get(), decode_candidate, decode_candidate.n_used, true);
        check_active_grouped_exact_output(decode_expected, decode_actual);
        ggml_cuda_moe_candidate_group_key decode_key;
        CHECK(decode_context->find_down_group_key(decode_candidate.down, &decode_key));
        for (ggml_tensor * bias : decode_candidate.biases) {
            const float * resident = ggml_cuda_moe_grouped_context_test_access::device_prefill_auxiliary_data(
                *decode_context, decode_key, bias);
            const float * shadow = ggml_cuda_moe_grouped_context_test_access::device_auxiliary_data(
                *decode_context, decode_key, bias);
            CHECK(resident == nullptr && shadow != nullptr);
        }
    }

    CHECK(ggml_backend_graph_compute(candidate_backend.get(), candidate.graph) == GGML_STATUS_SUCCESS);
    candidate_context->shutdown();
    ggml_backend_synchronize(candidate_backend.get());
    CHECK(ggml_cuda_moe_grouped_context_test_access::device_prefill_auxiliary_data(
        *candidate_context, refreshed_key, candidate.biases[0]) == nullptr);
    check_active_grouped_exact_output(
        expected, run_active_grouped_dispatch(parallel_backend.get(), parallel, 0, false));
    ggml_backend_cuda_moe_set_debug_mm(old_debug_mm);
    fprintf(stderr, "test-moe-cache: bounded prefill resident F32 biases exact OK\n");
}

static void test_active_grouped_dispatch() {
    test_prefill_resident_biases();
    test_active_grouped_materialization_eligibility();
    test_active_grouped_dispatch_case(
        GGML_TYPE_Q4_0, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 12, false, true);
    test_active_grouped_dispatch_case(
        GGML_TYPE_Q4_0, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 48);
    test_active_grouped_dispatch_case(
        GGML_TYPE_Q4_K, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 12);
    test_active_grouped_dispatch_case(
        GGML_TYPE_Q4_K, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 48);
    test_active_grouped_dispatch_case(
        GGML_TYPE_Q4_0, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 12);
    test_active_grouped_dispatch_case(
        GGML_TYPE_Q4_0, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 48);
    test_active_grouped_dispatch_case(
        GGML_TYPE_Q4_K, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 12);
    test_active_grouped_dispatch_case(
        GGML_TYPE_Q4_K, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 48);
    test_active_grouped_dispatch_case(
        GGML_TYPE_Q4_0, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 12, true);
    test_active_grouped_dispatch_types_case(
        {GGML_TYPE_MXFP4, GGML_TYPE_MXFP4, GGML_TYPE_MXFP4},
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 24, false, false, true);
    test_active_grouped_dispatch_types_case(
        {GGML_TYPE_MXFP4, GGML_TYPE_MXFP4, GGML_TYPE_MXFP4},
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 24, false, false, true, 2);
    test_active_grouped_dispatch_types_case(
        {GGML_TYPE_MXFP4, GGML_TYPE_MXFP4, GGML_TYPE_MXFP4},
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 24, false, false, true);
    test_active_grouped_dispatch_types_case(
        {GGML_TYPE_MXFP4, GGML_TYPE_MXFP4, GGML_TYPE_MXFP4},
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 3, false, true, true, 1, 512, true);
    fprintf(stderr, "test-moe-cache: active grouped MXFP4 separate/fused output biases exact OK\n");
    test_active_grouped_fused_down_scale_extent();
    fprintf(stderr, "test-moe-cache: Gemma fused original-direct down scale exact OK\n");
    test_active_grouped_dispatch_staged_legacy();
    test_active_grouped_empty_policy();
    test_active_grouped_inventory_reuse();
    test_active_grouped_nvfp4_scales();
    fprintf(stderr, "test-moe-cache: active grouped Q4 ordinary/F3 OK\n");
    test_active_grouped_dispatch_generic();
}

static void test_cached_mmid_fusion_decline() {
    const bool old_debug_mm = ggml_backend_cuda_moe_get_debug_mm();
    ggml_backend_cuda_moe_set_debug_mm(true);
    ggml_backend_ptr cuda_backend(ggml_backend_cuda_init(0));
    ggml_backend_ptr cpu_backend(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    CHECK(cuda_backend != nullptr && cpu_backend != nullptr);

    const int old_slots = ggml_backend_cuda_moe_get_cache_slots();
    ggml_backend_cuda_moe_set_cache_slots(4);
    ggml_cuda_moe_cache_free_all();

    cached_fusion_test_graph cuda_graph = build_cached_fusion_test_graph(
        cuda_backend.get(), ggml_backend_cuda_moe_cached_buffer_type());
    cached_fusion_test_graph cpu_graph = build_cached_fusion_test_graph(
        cpu_backend.get(), ggml_backend_cpu_buffer_type());
    CHECK(cuda_graph.leaves.size() == cpu_graph.leaves.size());
    CHECK(cuda_graph.cached_weights.size() == 19);

    for (size_t i = 0; i < cuda_graph.leaves.size(); ++i) {
        CHECK(cuda_graph.leaves[i]->type == cpu_graph.leaves[i]->type);
        CHECK(ggml_nbytes(cuda_graph.leaves[i]) == ggml_nbytes(cpu_graph.leaves[i]));
        std::vector<uint8_t> data = cached_fusion_test_data(cuda_graph.leaves[i], i);
        ggml_backend_tensor_set(cuda_graph.leaves[i], data.data(), 0, data.size());
        ggml_backend_tensor_set(cpu_graph.leaves[i], data.data(), 0, data.size());
    }
    for (ggml_tensor * weight : cuda_graph.cached_weights) {
        CHECK(weight->buffer != nullptr && ggml_backend_buft_is_cuda_moe_cached(weight->buffer->buft));
    }
    const auto disabled = candidate_snapshot(4, nullptr, 0);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v1(cuda_backend.get(), &disabled) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);

    for (int pass = 0; pass < 2; ++pass) {
        CHECK(ggml_backend_graph_compute(cpu_backend.get(), cpu_graph.graph) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_graph_compute(cuda_backend.get(), cuda_graph.graph) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(cpu_backend.get());
        ggml_backend_synchronize(cuda_backend.get());
    }
    CHECK(active_grouped_legacy_op_count(cuda_backend.get()) == 2 * cuda_graph.cached_weights.size());

    std::vector<float> expected(ggml_nelements(cpu_graph.output));
    std::vector<float> actual(ggml_nelements(cuda_graph.output));
    ggml_backend_tensor_get(cpu_graph.output, expected.data(), 0, ggml_nbytes(cpu_graph.output));
    ggml_backend_tensor_get(cuda_graph.output, actual.data(), 0, ggml_nbytes(cuda_graph.output));
    double squared_error = 0.0;
    double squared_expected = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        CHECK(std::isfinite(actual[i]) && std::isfinite(expected[i]));
        const double difference = actual[i] - expected[i];
        squared_error += difference * difference;
        squared_expected += static_cast<double>(expected[i]) * expected[i];
    }
    CHECK(squared_expected > 0.0 && squared_error / squared_expected < 2e-2);

    ggml_cuda_moe_cache_free_all();
    ggml_backend_cuda_moe_set_cache_slots(old_slots);
    ggml_backend_cuda_moe_set_debug_mm(old_debug_mm);
    fprintf(stderr, "test-moe-cache: cached MMID F1-F5 decline OK\n");
}

static cached_mmid_path_test_graph build_cached_mmid_path_test_graph(
        ggml_backend_t backend,
        ggml_backend_buffer_type_t weight_buft,
        ggml_type weight_type,
        int64_t n_out,
        int64_t n_used,
        int64_t n_tokens) {
    constexpr int64_t N_EXPERTS = 8;
    constexpr int64_t N_IN = 256;
    const ggml_init_params weight_params = {
        /* .mem_size = */ ggml_tensor_overhead() * 8,
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };
    const ggml_init_params node_params = {
        /* .mem_size = */ ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(32, false),
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };

    cached_mmid_path_test_graph result;
    result.weights.reset(ggml_init(weight_params));
    result.nodes.reset(ggml_init(node_params));
    CHECK(result.weights != nullptr && result.nodes != nullptr);

    const int64_t weight_ne[] = {N_IN, n_out, N_EXPERTS};
    ggml_tensor * weight = ggml_new_tensor(result.weights.get(), weight_type, 3, weight_ne);
    ggml_set_name(weight, "test.paths.ffn_up_exps.weight");
    const int64_t input_ne[] = {N_IN, 1, n_tokens};
    ggml_tensor * input = ggml_new_tensor(result.nodes.get(), GGML_TYPE_F32, 3, input_ne);
    ggml_set_name(input, "test.paths.input");
    const int64_t ids_ne[] = {n_used, n_tokens};
    result.ids = ggml_new_tensor(result.nodes.get(), GGML_TYPE_I32, 2, ids_ne);
    ggml_set_name(result.ids, "test.paths.ids");
    result.output = ggml_mul_mat_id(result.nodes.get(), weight, input, result.ids);
    ggml_set_name(result.output, "test.paths.output");
    result.leaves = {weight, input, result.ids};

    result.graph = ggml_new_graph_custom(result.nodes.get(), 32, false);
    ggml_build_forward_expand(result.graph, result.output);
    result.weight_buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(result.weights.get(), weight_buft));
    result.node_buffer.reset(ggml_backend_alloc_ctx_tensors(result.nodes.get(), backend));
    CHECK(result.weight_buffer != nullptr && result.node_buffer != nullptr);
    ggml_backend_buffer_set_usage(result.weight_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    return result;
}

static cached_mmid_path_test_graph build_cached_mmid_path_test_graph(
        ggml_backend_t backend,
        ggml_tensor * gate_up,
        ggml_tensor * down,
        int64_t n_used,
        int64_t n_tokens) {
    CHECK(gate_up != nullptr && down != nullptr && gate_up->buffer != nullptr && down->buffer != nullptr &&
        ggml_n_dims(gate_up) == 3 && ggml_n_dims(down) == 3 && gate_up->ne[0] == down->ne[0] &&
        gate_up->ne[1] == 2 * down->ne[0] && gate_up->ne[2] == down->ne[2]);
    const ggml_init_params node_params = {
        /* .mem_size = */ ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(32, false),
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };

    cached_mmid_path_test_graph result;
    result.nodes.reset(ggml_init(node_params));
    CHECK(result.nodes != nullptr);

    const int64_t input_ne[] = {gate_up->ne[0], 1, n_tokens};
    ggml_tensor * input = ggml_new_tensor(result.nodes.get(), GGML_TYPE_F32, 3, input_ne);
    ggml_set_name(input, "test.paths.registered.input");
    const int64_t ids_ne[] = {n_used, n_tokens};
    result.ids = ggml_new_tensor(result.nodes.get(), GGML_TYPE_I32, 2, ids_ne);
    ggml_set_name(result.ids, "test.paths.registered.ids");
    ggml_tensor * hidden = ggml_mul_mat_id(result.nodes.get(), gate_up, input, result.ids);
    hidden = ggml_dup(result.nodes.get(), hidden);
    hidden = ggml_glu(result.nodes.get(), hidden, GGML_GLU_OP_SWIGLU, false);
    result.output = ggml_mul_mat_id(result.nodes.get(), down, hidden, result.ids);
    ggml_set_name(result.output, "test.paths.registered.output");
    result.leaves = {gate_up, down, input, result.ids};

    result.graph = ggml_new_graph_custom(result.nodes.get(), 32, false);
    ggml_build_forward_expand(result.graph, result.output);
    result.node_buffer.reset(ggml_backend_alloc_ctx_tensors(result.nodes.get(), backend));
    CHECK(result.node_buffer != nullptr);
    return result;
}

static void initialize_cached_mmid_path_test_graphs(
        cached_mmid_path_test_graph & cuda_graph,
        cached_mmid_path_test_graph & reference_graph) {
    CHECK(cuda_graph.leaves.size() == reference_graph.leaves.size());
    for (size_t i = 0; i < cuda_graph.leaves.size(); ++i) {
        CHECK(cuda_graph.leaves[i]->type == reference_graph.leaves[i]->type);
        CHECK(ggml_nbytes(cuda_graph.leaves[i]) == ggml_nbytes(reference_graph.leaves[i]));
        std::vector<uint8_t> data = cached_fusion_test_data(cuda_graph.leaves[i], i + 31);
        ggml_backend_tensor_set(cuda_graph.leaves[i], data.data(), 0, data.size());
        ggml_backend_tensor_set(reference_graph.leaves[i], data.data(), 0, data.size());
    }
}

static std::vector<float> run_cached_mmid_path_test(
        ggml_backend_t cached_backend,
        ggml_backend_t reference_backend,
        cached_mmid_path_test_graph & cuda_graph,
        cached_mmid_path_test_graph & reference_graph,
        const std::vector<int32_t> & ids) {
    CHECK(ids.size() == (size_t) ggml_nelements(cuda_graph.ids));
    ggml_backend_tensor_set(cuda_graph.ids, ids.data(), 0, ids.size() * sizeof(ids[0]));
    ggml_backend_tensor_set(reference_graph.ids, ids.data(), 0, ids.size() * sizeof(ids[0]));
    std::vector<uint8_t> output_zero(ggml_nbytes(cuda_graph.output), 0);
    ggml_backend_tensor_set(cuda_graph.output, output_zero.data(), 0, output_zero.size());
    ggml_backend_tensor_set(reference_graph.output, output_zero.data(), 0, output_zero.size());
    CHECK(ggml_backend_graph_compute(reference_backend, reference_graph.graph) == GGML_STATUS_SUCCESS);
    CHECK(ggml_backend_graph_compute(cached_backend, cuda_graph.graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(reference_backend);
    ggml_backend_synchronize(cached_backend);

    std::vector<float> expected(ggml_nelements(reference_graph.output));
    std::vector<float> actual(ggml_nelements(cuda_graph.output));
    ggml_backend_tensor_get(reference_graph.output, expected.data(), 0, ggml_nbytes(reference_graph.output));
    ggml_backend_tensor_get(cuda_graph.output, actual.data(), 0, ggml_nbytes(cuda_graph.output));
    double squared_error = 0.0;
    double squared_expected = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        CHECK(std::isfinite(actual[i]) && std::isfinite(expected[i]));
        const double difference = actual[i] - expected[i];
        squared_error += difference * difference;
        squared_expected += static_cast<double>(expected[i]) * expected[i];
    }
    const double relative_squared_error = squared_error / squared_expected;
    CHECK(squared_expected > 0.0 && relative_squared_error < 2e-5);
    return actual;
}

static void test_cached_mmid_prefill_and_overflow() {
    ggml_backend_ptr cuda_backend(ggml_backend_cuda_init(0));
    ggml_backend_ptr reference_backend(ggml_backend_cuda_init(0));
    CHECK(cuda_backend != nullptr && reference_backend != nullptr);
    const auto disabled = candidate_snapshot(4, nullptr, 0);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v1(cuda_backend.get(), &disabled) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);

    auto cuda_prefill = build_cached_mmid_path_test_graph(
        cuda_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0, 128, 3, 2);
    auto reference_prefill = build_cached_mmid_path_test_graph(
        reference_backend.get(), ggml_backend_cuda_buffer_type(0), GGML_TYPE_Q4_0, 128, 3, 2);
    initialize_cached_mmid_path_test_graphs(cuda_prefill, reference_prefill);
    const std::vector<int32_t> mapped_ids = {0, 1, 2, 1, 2, 3};
    const auto mapped_first = run_cached_mmid_path_test(
        cuda_backend.get(), reference_backend.get(), cuda_prefill, reference_prefill, mapped_ids);
    const auto mapped_second = run_cached_mmid_path_test(
        cuda_backend.get(), reference_backend.get(), cuda_prefill, reference_prefill, mapped_ids);
    CHECK(mapped_first == mapped_second);
    (void) run_cached_mmid_path_test(
        cuda_backend.get(), reference_backend.get(), cuda_prefill, reference_prefill, {0, 1, 2, 3, 4, 5});

    for (ggml_type type : {GGML_TYPE_MXFP4, GGML_TYPE_NVFP4}) {
        auto cuda_fp4 = build_cached_mmid_path_test_graph(
            cuda_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), type, 128, 3, 2);
        auto reference_fp4 = build_cached_mmid_path_test_graph(
            reference_backend.get(), ggml_backend_cuda_buffer_type(0), type, 128, 3, 2);
        initialize_cached_mmid_path_test_graphs(cuda_fp4, reference_fp4);
        const auto fp4_first = run_cached_mmid_path_test(
            cuda_backend.get(), reference_backend.get(), cuda_fp4, reference_fp4, mapped_ids);
        const auto fp4_second = run_cached_mmid_path_test(
            cuda_backend.get(), reference_backend.get(), cuda_fp4, reference_fp4, mapped_ids);
        CHECK(fp4_first == fp4_second);
    }

    for (ggml_type type : {GGML_TYPE_MXFP4, GGML_TYPE_NVFP4}) {
        auto cuda_fp4_overflow = build_cached_mmid_path_test_graph(
            cuda_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), type, 128, 4, 16);
        auto reference_fp4_overflow = build_cached_mmid_path_test_graph(
            reference_backend.get(), ggml_backend_cuda_buffer_type(0), type, 128, 4, 16);
        initialize_cached_mmid_path_test_graphs(cuda_fp4_overflow, reference_fp4_overflow);
        static constexpr int32_t expert_order[] = {5, 2, 7, 1, 6, 0, 4, 3};
        std::vector<int32_t> overflow_ids(4 * 16);
        for (int32_t token = 0; token < 16; ++token) {
            for (int32_t route = 0; route < 4; ++route) {
                overflow_ids[token * 4 + route] = expert_order[(token + route) % 8];
            }
        }
        const auto overflow_output = run_cached_mmid_path_test(
            cuda_backend.get(), reference_backend.get(), cuda_fp4_overflow, reference_fp4_overflow, overflow_ids);
        std::vector<float> overflow_expected(ggml_nelements(reference_fp4_overflow.output));
        ggml_backend_tensor_get(reference_fp4_overflow.output, overflow_expected.data(), 0,
            ggml_nbytes(reference_fp4_overflow.output));
        CHECK(overflow_output == overflow_expected);
    }

    auto cuda_decode = build_cached_mmid_path_test_graph(
        cuda_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0, 128, 6, 1);
    auto reference_decode = build_cached_mmid_path_test_graph(
        reference_backend.get(), ggml_backend_cuda_buffer_type(0), GGML_TYPE_Q4_0, 128, 6, 1);
    initialize_cached_mmid_path_test_graphs(cuda_decode, reference_decode);
    (void) run_cached_mmid_path_test(
        cuda_backend.get(), reference_backend.get(), cuda_decode, reference_decode, {0, 1, 2, 3, 4, 5});

    auto cuda_q4_k = build_cached_mmid_path_test_graph(
        cuda_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_K, 256, 3, 2);
    auto reference_q4_k = build_cached_mmid_path_test_graph(
        reference_backend.get(), ggml_backend_cuda_buffer_type(0), GGML_TYPE_Q4_K, 256, 3, 2);
    initialize_cached_mmid_path_test_graphs(cuda_q4_k, reference_q4_k);
    (void) run_cached_mmid_path_test(
        cuda_backend.get(), reference_backend.get(), cuda_q4_k, reference_q4_k, mapped_ids);

    auto cuda_q4_k_tiny = build_cached_mmid_path_test_graph(
        cuda_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_K, 64, 3, 2);
    auto reference_q4_k_tiny = build_cached_mmid_path_test_graph(
        reference_backend.get(), ggml_backend_cuda_buffer_type(0), GGML_TYPE_Q4_K, 64, 3, 2);
    initialize_cached_mmid_path_test_graphs(cuda_q4_k_tiny, reference_q4_k_tiny);
    (void) run_cached_mmid_path_test(
        cuda_backend.get(), reference_backend.get(), cuda_q4_k_tiny, reference_q4_k_tiny, mapped_ids);

    ggml_backend_ptr transition_backend(ggml_backend_cuda_init(0));
    CHECK(transition_backend != nullptr);
    auto grouped = build_active_grouped_dispatch_graph(
        transition_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP);
    auto grouped_reference = build_active_grouped_dispatch_graph(
        reference_backend.get(), ggml_backend_cuda_buffer_type(0), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP);
    const auto grouped_input = cached_fusion_test_data(grouped.input, 181);
    const float grouped_logits[] = {-1.0f, 3.0f, 0.5f, 9.0f, 2.0f, 8.0f, -2.0f, 1.0f};
    ggml_backend_tensor_set(grouped.input, grouped_input.data(), 0, grouped_input.size());
    ggml_backend_tensor_set(grouped.logits, grouped_logits, 0, sizeof(grouped_logits));
    register_active_grouped_dispatch(
        transition_backend.get(), grouped, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 4);

    auto registered_prefill = build_cached_mmid_path_test_graph(
        transition_backend.get(), grouped.banks[0], grouped.banks[1], 3, 2);
    auto registered_reference = build_cached_mmid_path_test_graph(
        reference_backend.get(), grouped_reference.banks[0], grouped_reference.banks[1], 3, 2);
    initialize_cached_mmid_path_test_graphs(registered_prefill, registered_reference);
    const auto grouped_first = run_active_grouped_dispatch(transition_backend.get(), grouped, 2);
    CHECK(!grouped_first.empty());

    auto * transition_context = ggml_cuda_moe_grouped_context_for_test(transition_backend.get());
    ggml_cuda_moe_candidate_group_key transition_key;
    ggml_cuda_moe_grouped_acquisition overflow_resource;
    CHECK(transition_context != nullptr && transition_context->find_down_group_key(grouped.down, &transition_key));
    CHECK(transition_context->acquire_group_resources(transition_key, &overflow_resource));
    CHECK(ggml_cuda_moe_grouped_context_test_access::has_device_resource(*transition_context, transition_key));
    CHECK(ggml_cuda_moe_grouped_context_test_access::device_resource_complete(*transition_context, transition_key));
    CHECK(grouped.graph->nodes[0] != registered_prefill.graph->nodes[0]);
    const uint64_t overflow_resource_generation = overflow_resource.resource_generation;

    cudaStream_t transition_stream = nullptr;
    CUDA_OK(cudaStreamCreateWithFlags(&transition_stream, cudaStreamNonBlocking));
    const auto grouped_coverage = candidate_certify_graph(*transition_context, grouped.graph);
    std::shared_ptr<ggml_cuda_moe_graph_plan> grouped_plan;
    ggml_cuda_moe_graph_execution grouped_execution;
    CHECK(transition_context->prepare_graph_execution(
        grouped.graph, 800, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, &grouped_plan, &grouped_execution,
        grouped_coverage.epoch, grouped_coverage.nodes,
        grouped_coverage.mmid_count, grouped_coverage.mmid_fingerprint) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
    CHECK(grouped_execution.resolve_streams(candidate_test_graph_stream, transition_stream));
    uint64_t overflow_fingerprint = 0;
    std::vector<std::shared_ptr<void>> overflow_leases;
    CHECK(transition_context->graph_resource_fingerprint(
        grouped_execution, transition_stream, &overflow_fingerprint, &overflow_leases));
    CHECK(overflow_fingerprint != 0 && overflow_leases.size() == 1);
    std::weak_ptr<void> overflow_witness = overflow_leases[0];
    overflow_leases.clear();
    CHECK(!overflow_witness.expired());
    CHECK(ggml_cuda_moe_grouped_context_test_access::set_clock_bound(
        *transition_context, overflow_resource, UINT64_MAX));
    CHECK(run_active_grouped_dispatch(transition_backend.get(), grouped, 2) == grouped_first);
    CHECK(overflow_witness.expired());
    ggml_cuda_moe_grouped_acquisition refreshed_resource;
    CHECK(transition_context->acquire_group_resources(transition_key, &refreshed_resource));
    CHECK(refreshed_resource.resource_generation != overflow_resource_generation);
    CHECK(run_active_grouped_dispatch(transition_backend.get(), grouped, 4) == grouped_first);

    ggml_cuda_moe_grouped_acquisition grouped_resource;
    CHECK(transition_context->acquire_group_resources(transition_key, &grouped_resource));
    const uint64_t grouped_resource_generation = grouped_resource.resource_generation;
    uint64_t grouped_fingerprint = 0;
    std::vector<std::shared_ptr<void>> grouped_leases;
    CHECK(transition_context->graph_resource_fingerprint(
        grouped_execution, transition_stream, &grouped_fingerprint, &grouped_leases));
    CHECK(grouped_fingerprint != 0 && grouped_fingerprint != overflow_fingerprint && grouped_leases.size() == 1);
    std::weak_ptr<void> grouped_witness = grouped_leases[0];
    grouped_leases.clear();
    CHECK(!grouped_witness.expired());

    std::shared_ptr<ggml_cuda_moe_graph_plan> transition_plan;
    ggml_cuda_moe_graph_execution transition_execution;
    CHECK(transition_context->prepare_graph_execution(
        registered_prefill.graph, 801, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &transition_plan, &transition_execution) ==
        GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
    CHECK(transition_execution.size() == 1);
    CHECK(transition_execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_PREFILL_LEGACY);
    CHECK(ggml_cuda_moe_grouped_context_test_access::graph_has_complete_mmid_inventory(*transition_plan));
    CHECK(!ggml_cuda_moe_grouped_context_test_access::graph_group_has_decode_discovery(*transition_plan, 0));
    const ggml_cuda_moe_graph_plan * uncovered_prefill_plan = transition_plan.get();
    CHECK(transition_context->prepare_graph_execution(
        registered_prefill.graph, 802, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &transition_plan, &transition_execution) ==
        GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
    CHECK(transition_plan.get() != uncovered_prefill_plan);
    const auto prefill_coverage = candidate_certify_graph(*transition_context, registered_prefill.graph);
    CHECK(transition_context->prepare_graph_execution(
        registered_prefill.graph, 803, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &transition_plan, &transition_execution,
        prefill_coverage.epoch, prefill_coverage.nodes,
        prefill_coverage.mmid_count, prefill_coverage.mmid_fingerprint) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
    const ggml_cuda_moe_graph_plan * covered_prefill_plan = transition_plan.get();
    CHECK(transition_context->prepare_graph_execution(
        registered_prefill.graph, 804, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &transition_plan, &transition_execution,
        prefill_coverage.epoch, prefill_coverage.nodes,
        prefill_coverage.mmid_count, prefill_coverage.mmid_fingerprint) == GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
    CHECK(transition_plan.get() == covered_prefill_plan);
    CHECK(ggml_cuda_moe_grouped_context_test_access::graph_has_complete_mmid_inventory(*transition_plan));
    CHECK(!ggml_cuda_moe_grouped_context_test_access::graph_group_has_decode_discovery(*transition_plan, 0));
    CHECK(transition_execution.resolve_streams(candidate_test_graph_stream, reinterpret_cast<void *>(uintptr_t{1})));
    CHECK(transition_context->begin_graph_dispatch(&transition_execution, true));
    CHECK(transition_context->get_group_resources(grouped_resource, nullptr));
    CHECK(ggml_cuda_moe_grouped_context_test_access::has_device_resource(*transition_context, transition_key));
    CHECK(!grouped_witness.expired());
    void * first_bank_data = ggml_cuda_moe_grouped_context_test_access::device_bank_data(
        *transition_context, transition_key, grouped.banks[0]);
    CHECK(first_bank_data != nullptr);
    const uint32_t payload_sentinel = 0x5a17c3e9;
    uint32_t payload_probe = 0;
    CUDA_OK(cudaMemcpy(first_bank_data, &payload_sentinel, sizeof(payload_sentinel), cudaMemcpyHostToDevice));
    ggml_cuda_moe_grouped_context_test_access::fail_borrowed_cache_init_after_probe(*transition_context);
    CHECK(!transition_context->acquire_legacy_cache(grouped.banks[0]));
    CUDA_OK(cudaMemcpy(&payload_probe, first_bank_data, sizeof(payload_probe), cudaMemcpyDeviceToHost));
    CHECK(payload_probe == payload_sentinel);
    CHECK(ggml_cuda_moe_grouped_context_test_access::legacy_backing_count(*transition_context, transition_key) == 0);
    auto first_legacy = transition_context->acquire_legacy_cache(grouped.banks[0]);
    CHECK(first_legacy && first_legacy.get() != nullptr && first_legacy.acquisition().registered_source == 1 &&
        first_legacy.acquisition().group_index == transition_key.group_index);
    CUDA_OK(cudaMemcpy(&payload_probe, first_bank_data, sizeof(payload_probe), cudaMemcpyDeviceToHost));
    CHECK(payload_probe == payload_sentinel);
    CHECK(ggml_cuda_moe_cache_slot_ptr(first_legacy.get(), 0) ==
        ggml_cuda_moe_grouped_context_test_access::device_bank_data(*transition_context, transition_key, grouped.banks[0]));
    CHECK(ggml_cuda_moe_grouped_context_test_access::legacy_backing_count(*transition_context, transition_key) == 1);
    const uint64_t legacy_epoch = first_legacy.acquisition().group_authority_epoch;
    CHECK(ggml_cuda_moe_grouped_context_test_access::has_device_resource(*transition_context, transition_key));
    first_legacy = {};
    CHECK(transition_context->finish_graph_dispatch(&transition_execution));

    const auto registered_mapped_first = run_cached_mmid_path_test(
        transition_backend.get(), reference_backend.get(), registered_prefill, registered_reference, mapped_ids);
    const auto registered_mapped_second = run_cached_mmid_path_test(
        transition_backend.get(), reference_backend.get(), registered_prefill, registered_reference, mapped_ids);
    CHECK(registered_mapped_first == registered_mapped_second);
    CHECK(ggml_cuda_moe_grouped_context_test_access::has_device_resource(*transition_context, transition_key));
    auto repeated_legacy = transition_context->acquire_legacy_cache(grouped.banks[0]);
    CHECK(repeated_legacy && repeated_legacy.acquisition().group_authority_epoch == legacy_epoch);
    repeated_legacy = {};
    (void) run_cached_mmid_path_test(
        transition_backend.get(), reference_backend.get(), registered_prefill, registered_reference, {0, 1, 2, 3, 4, 5});
    CHECK(ggml_cuda_moe_grouped_context_test_access::has_device_resource(*transition_context, transition_key));
    CHECK(run_active_grouped_dispatch(transition_backend.get(), grouped, 2) == grouped_first);
    ggml_cuda_moe_grouped_acquisition regrouped_resource;
    CHECK(transition_context->acquire_group_resources(transition_key, &regrouped_resource));
    CHECK(regrouped_resource.resource_generation == grouped_resource_generation && !grouped_witness.expired());
    uint64_t stable_fingerprint = 0;
    CHECK(transition_context->graph_resource_fingerprint(grouped_execution, transition_stream, &stable_fingerprint));
    CHECK(stable_fingerprint == grouped_fingerprint);
    CUDA_OK(cudaStreamDestroy(transition_stream));
    fprintf(stderr, "test-moe-cache: registered grouped prefill stable backing OK\n");
    fprintf(stderr, "test-moe-cache: cached mapped prefill and overflow OK\n");
}

struct routed_separate_chain_test_graph {
    ggml_context_ptr weights;
    ggml_context_ptr nodes;
    ggml_backend_buffer_ptr weight_buffer;
    ggml_backend_buffer_ptr node_buffer;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * logits = nullptr;
    ggml_tensor * ids = nullptr;
    ggml_tensor * route_weights = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * up = nullptr;
    ggml_tensor * hidden = nullptr;
    ggml_tensor * down = nullptr;
    ggml_tensor * weighted = nullptr;
    ggml_tensor * output = nullptr;
    std::array<ggml_tensor *, 3> banks = {};
};

static routed_separate_chain_test_graph build_routed_separate_chain_test_graph(
        ggml_backend_t backend,
        ggml_backend_buffer_type_t weight_buft,
        int64_t n_tokens) {
    constexpr int64_t N_EXPERTS = 64;
    constexpr int64_t N_USED = 6;
    constexpr int64_t N_EMBD = 2048;
    constexpr int64_t N_FF = 1408;
    const ggml_init_params weight_params = {
        /* .mem_size = */ ggml_tensor_overhead() * 8,
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };
    const ggml_init_params node_params = {
        /* .mem_size = */ ggml_tensor_overhead() * 128 + ggml_graph_overhead_custom(128, false),
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };

    routed_separate_chain_test_graph result;
    result.weights.reset(ggml_init(weight_params));
    result.nodes.reset(ggml_init(node_params));
    CHECK(result.weights != nullptr && result.nodes != nullptr);

    result.banks[0] = ggml_new_tensor_3d(result.weights.get(), GGML_TYPE_Q4_K, N_EMBD, N_FF, N_EXPERTS);
    result.banks[1] = ggml_new_tensor_3d(result.weights.get(), GGML_TYPE_Q4_K, N_EMBD, N_FF, N_EXPERTS);
    result.banks[2] = ggml_new_tensor_3d(result.weights.get(), GGML_TYPE_Q8_0, N_FF, N_EMBD, N_EXPERTS);
    ggml_set_name(result.banks[0], "blk.1.ffn_gate_exps.weight");
    ggml_set_name(result.banks[1], "blk.1.ffn_up_exps.weight");
    ggml_set_name(result.banks[2], "blk.1.ffn_down_exps.weight");

    result.input = ggml_new_tensor_3d(result.nodes.get(), GGML_TYPE_F32, N_EMBD, 1, n_tokens);
    result.logits = ggml_new_tensor_2d(result.nodes.get(), GGML_TYPE_F32, N_EXPERTS, n_tokens);
    ggml_set_name(result.input, "test.routed.input");
    ggml_set_name(result.logits, "test.routed.logits");
    ggml_tensor * probs = ggml_soft_max(result.nodes.get(), result.logits);
    result.ids = ggml_argsort_top_k(result.nodes.get(), probs, N_USED);
    ggml_tensor * probs_3d = ggml_reshape_3d(result.nodes.get(), probs, 1, N_EXPERTS, n_tokens);
    result.route_weights = ggml_get_rows(result.nodes.get(), probs_3d, result.ids);

    result.gate = ggml_mul_mat_id(result.nodes.get(), result.banks[0], result.input, result.ids);
    result.up = ggml_mul_mat_id(result.nodes.get(), result.banks[1], result.input, result.ids);
    result.hidden = ggml_swiglu_split(result.nodes.get(), result.gate, result.up);
    result.down = ggml_mul_mat_id(result.nodes.get(), result.banks[2], result.hidden, result.ids);
    result.weighted = ggml_mul(result.nodes.get(), result.down, result.route_weights);

    std::array<ggml_tensor *, N_USED> routes = {};
    for (int64_t route = 0; route < N_USED; ++route) {
        routes[route] = ggml_view_2d(
            result.nodes.get(), result.weighted, N_EMBD, n_tokens, result.weighted->nb[2], route * result.weighted->nb[1]);
    }
    result.output = routes[0];
    for (int64_t route = 1; route < N_USED; ++route) {
        result.output = ggml_add(result.nodes.get(), result.output, routes[route]);
    }
    ggml_set_name(result.output, "test.routed.output");

    result.graph = ggml_new_graph_custom(result.nodes.get(), 128, false);
    ggml_build_forward_expand(result.graph, result.route_weights);
    ggml_build_forward_expand(result.graph, result.output);
    candidate_stamp_execution(result.graph, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
        n_tokens == 1 ? GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT : GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL,
        n_tokens, 1);
    result.weight_buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(result.weights.get(), weight_buft));
    result.node_buffer.reset(ggml_backend_alloc_ctx_tensors(result.nodes.get(), backend));
    CHECK(result.weight_buffer != nullptr && result.node_buffer != nullptr);
    ggml_backend_buffer_set_usage(result.weight_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    return result;
}

static std::vector<float> routed_separate_chain_logits(
        int64_t n_tokens,
        int64_t expert_span,
        int64_t expert_offset) {
    constexpr int64_t N_EXPERTS = 64;
    constexpr int64_t N_USED = 6;
    CHECK(expert_span >= N_USED && expert_span <= N_EXPERTS);
    std::vector<float> result(N_EXPERTS * n_tokens);
    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t expert = 0; expert < N_EXPERTS; ++expert) {
            result[token * N_EXPERTS + expert] = -20.0f - 0.01f * expert;
        }
        for (int64_t route = 0; route < N_USED; ++route) {
            const int64_t expert = (expert_offset + (token * N_USED + route * 11) % expert_span) % N_EXPERTS;
            result[token * N_EXPERTS + expert] = 10.0f - route;
        }
    }
    return result;
}

static std::vector<uint8_t> routed_separate_chain_test_weight_data(
        const ggml_tensor * tensor,
        uint32_t salt) {
    CHECK(ggml_is_quantized(tensor->type));
    std::vector<float> values(ggml_nelements(tensor));
    uint32_t state = 0x9e3779b9u ^ salt;
    for (float & value : values) {
        state = state * 1664525u + 1013904223u;
        value = (static_cast<int32_t>((state >> 8) % 2001) - 1000) * 0.00035f;
    }
    std::vector<uint8_t> bytes(ggml_nbytes(tensor));
    const int64_t nrows = ggml_nelements(tensor) / tensor->ne[0];
    CHECK(ggml_quantize_chunk(
        tensor->type, values.data(), bytes.data(), 0, nrows, tensor->ne[0], nullptr) == bytes.size());
    return bytes;
}

static void initialize_routed_separate_chain_test_weights(
        routed_separate_chain_test_graph & candidate,
        routed_separate_chain_test_graph & reference) {
    for (size_t bank = 0; bank < candidate.banks.size(); ++bank) {
        CHECK(candidate.banks[bank]->type == reference.banks[bank]->type &&
            ggml_nbytes(candidate.banks[bank]) == ggml_nbytes(reference.banks[bank]));
        ggml_tensor expert = *candidate.banks[bank];
        expert.ne[2] = 1;
        expert.ne[3] = 1;
        expert.nb[3] = expert.nb[2];
        CHECK(ggml_nbytes(&expert) == candidate.banks[bank]->nb[2]);
        for (int64_t expert_id = 0; expert_id < candidate.banks[bank]->ne[2]; ++expert_id) {
            const auto bytes = routed_separate_chain_test_weight_data(
                &expert, 311 + 64 * bank + static_cast<uint32_t>(expert_id));
            const size_t offset = expert_id * candidate.banks[bank]->nb[2];
            ggml_backend_tensor_set(candidate.banks[bank], bytes.data(), offset, bytes.size());
            ggml_backend_tensor_set(reference.banks[bank], bytes.data(), offset, bytes.size());
        }
    }
}

static void set_routed_separate_chain_test_inputs(
        routed_separate_chain_test_graph & candidate,
        routed_separate_chain_test_graph & reference,
        size_t pass,
        int64_t expert_span,
        int64_t expert_offset) {
    CHECK(candidate.input->ne[2] == reference.input->ne[2]);
    const auto input = cached_fusion_test_data(candidate.input, 317 + pass);
    const auto logits = routed_separate_chain_logits(candidate.input->ne[2], expert_span, expert_offset);
    ggml_backend_tensor_set(candidate.input, input.data(), 0, input.size());
    ggml_backend_tensor_set(reference.input, input.data(), 0, input.size());
    ggml_backend_tensor_set(candidate.logits, logits.data(), 0, logits.size() * sizeof(float));
    ggml_backend_tensor_set(reference.logits, logits.data(), 0, logits.size() * sizeof(float));
}

static void register_routed_separate_chain_test_graph(
        ggml_backend_t backend,
        const routed_separate_chain_test_graph & graph,
        uint32_t n_slots) {
    const std::array<ggml_backend_moe_candidate_bank_v1, 3> banks = {{
        {graph.banks[0], GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT, 0},
        {graph.banks[1], GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT, 0},
        {graph.banks[2], GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
    }};
    const ggml_backend_moe_candidate_group_v1 group = {
        banks.data(), banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 0, 0,
    };
    const auto snapshot = candidate_snapshot(n_slots, &group, 1);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v1(backend, &snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
}

struct routed_separate_chain_test_result {
    std::vector<int32_t> ids;
    std::vector<float> route_weights;
    std::vector<float> gate;
    std::vector<float> up;
    std::vector<float> hidden;
    std::vector<float> down;
    std::vector<float> weighted;
    std::vector<float> output;
};

static routed_separate_chain_test_result run_routed_separate_chain_test_graph(
        ggml_backend_t backend,
        routed_separate_chain_test_graph & graph) {
    CHECK(ggml_backend_graph_compute(backend, graph.graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
    routed_separate_chain_test_result result;
    result.ids.resize(ggml_nelements(graph.ids));
    CHECK(graph.ids->view_src != nullptr && graph.ids->view_src->type == GGML_TYPE_I32 &&
        graph.ids->nb[0] == sizeof(int32_t) && graph.ids->nb[1] % sizeof(int32_t) == 0);
    std::vector<int32_t> sorted(ggml_nelements(graph.ids->view_src));
    ggml_backend_tensor_get(graph.ids->view_src, sorted.data(), 0, ggml_nbytes(graph.ids->view_src));
    const size_t row_stride = graph.ids->nb[1] / sizeof(int32_t);
    for (int64_t row = 0; row < graph.ids->ne[1]; ++row) {
        for (int64_t route = 0; route < graph.ids->ne[0]; ++route) {
            result.ids[row * graph.ids->ne[0] + route] = sorted[row * row_stride + route];
        }
    }
    result.route_weights.resize(ggml_nelements(graph.route_weights));
    result.gate.resize(ggml_nelements(graph.gate));
    result.up.resize(ggml_nelements(graph.up));
    result.hidden.resize(ggml_nelements(graph.hidden));
    result.down.resize(ggml_nelements(graph.down));
    result.weighted.resize(ggml_nelements(graph.weighted));
    result.output.resize(ggml_nelements(graph.output));
    ggml_backend_tensor_get(graph.route_weights, result.route_weights.data(), 0, ggml_nbytes(graph.route_weights));
    ggml_backend_tensor_get(graph.gate, result.gate.data(), 0, ggml_nbytes(graph.gate));
    ggml_backend_tensor_get(graph.up, result.up.data(), 0, ggml_nbytes(graph.up));
    ggml_backend_tensor_get(graph.hidden, result.hidden.data(), 0, ggml_nbytes(graph.hidden));
    ggml_backend_tensor_get(graph.down, result.down.data(), 0, ggml_nbytes(graph.down));
    ggml_backend_tensor_get(graph.weighted, result.weighted.data(), 0, ggml_nbytes(graph.weighted));
    ggml_backend_tensor_get(graph.output, result.output.data(), 0, ggml_nbytes(graph.output));
    return result;
}

static void check_routed_separate_chain_exact(
        const char * phase,
        const routed_separate_chain_test_result & expected,
        const routed_separate_chain_test_result & actual) {
    CHECK(actual.ids == expected.ids && actual.route_weights == expected.route_weights);
    const auto check_values = [phase, &actual](
            const char * name,
            const std::vector<float> & expected_values,
            const std::vector<float> & actual_values) {
        CHECK(actual_values.size() == expected_values.size());
        for (size_t i = 0; i < actual_values.size(); ++i) {
            if (!std::isfinite(actual_values[i]) || !std::isfinite(expected_values[i])) {
                fprintf(stderr, "test-moe-cache: routed %s %s first_nonfinite=%zu expected=%.9g actual=%.9g\n",
                    phase, name, i, expected_values[i], actual_values[i]);
                if (strcmp(name, "down") == 0) {
                    constexpr size_t N_EMBD = 2048;
                    constexpr size_t N_USED = 6;
                    const size_t row = i / N_EMBD;
                    const size_t token = row / N_USED;
                    const size_t route = row % N_USED;
                    fprintf(stderr, "test-moe-cache: routed %s down token=%zu route=%zu expert=%d element=%zu\n",
                        phase, token, route, actual.ids[token * N_USED + route], i % N_EMBD);
                }
                CHECK(false);
            }
            if (actual_values[i] != expected_values[i]) {
                fprintf(stderr, "test-moe-cache: routed %s %s first_difference=%zu expected=%.9g actual=%.9g\n",
                    phase, name, i, expected_values[i], actual_values[i]);
                CHECK(false);
            }
        }
    };
    check_values("gate", expected.gate, actual.gate);
    check_values("up", expected.up, actual.up);
    check_values("hidden", expected.hidden, actual.hidden);
    check_values("down", expected.down, actual.down);
    check_values("weighted", expected.weighted, actual.weighted);
    check_values("output", expected.output, actual.output);
}

static void test_cached_mmid_routed_separate_chain() {
    constexpr uint32_t N_SLOTS = 63;
    const bool old_debug_mm = ggml_backend_cuda_moe_get_debug_mm();
    ggml_backend_cuda_moe_set_debug_mm(true);

    {
        constexpr int64_t N_TOKENS = 512;
        ggml_backend_ptr reference_backend(ggml_backend_cuda_init(0));
        ggml_backend_ptr candidate_backend(ggml_backend_cuda_init(0));
        CHECK(reference_backend != nullptr && candidate_backend != nullptr);
        auto reference = build_routed_separate_chain_test_graph(
            reference_backend.get(), ggml_backend_cuda_buffer_type(0), N_TOKENS);
        auto candidate = build_routed_separate_chain_test_graph(
            candidate_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), N_TOKENS);
        initialize_routed_separate_chain_test_weights(candidate, reference);
        const auto disabled = candidate_snapshot(N_SLOTS, nullptr, 0);
        CHECK(ggml_backend_cuda_moe_candidate_replace_v1(reference_backend.get(), &disabled) ==
            GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        register_routed_separate_chain_test_graph(candidate_backend.get(), candidate, N_SLOTS);

        auto * context = ggml_cuda_moe_grouped_context_for_test(candidate_backend.get());
        CHECK(context != nullptr);
        const auto coverage = candidate_certify_graph(*context, candidate.graph);
        std::shared_ptr<ggml_cuda_moe_graph_plan> plan;
        ggml_cuda_moe_graph_execution execution;
        CHECK(context->prepare_graph_execution(
            candidate.graph, 991, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &plan, &execution,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_PREFILL_LEGACY);

        cudaStream_t cache_stream = nullptr;
        CUDA_OK(cudaStreamCreateWithFlags(&cache_stream, cudaStreamNonBlocking));
        std::array<ggml_cuda_moe_legacy_cache_lease, 3> caches;
        for (size_t bank = 0; bank < caches.size(); ++bank) {
            caches[bank] = context->acquire_legacy_cache(candidate.banks[bank], nullptr, nullptr, cache_stream);
            CHECK(caches[bank] && caches[bank].acquisition().registered_source == 1);
            CHECK(ggml_cuda_moe_cache_n_slots(caches[bank].get()) == static_cast<int>(N_SLOTS));
            CHECK(ggml_cuda_moe_cache_can_overlap_staging(caches[bank].get()));
            ggml_cuda_moe_cache_reset_stats(caches[bank].get());
        }
        CUDA_OK(cudaStreamSynchronize(cache_stream));
        CUDA_OK(cudaStreamDestroy(cache_stream));
        ggml_cuda_moe_candidate_group_key group_key;
        CHECK(context->find_down_group_key(candidate.banks[2], &group_key));
        CHECK(ggml_cuda_moe_grouped_context_test_access::legacy_backing_count(*context, group_key) == 3);

        struct phase_spec {
            const char * name;
            int64_t expert_span;
            int64_t expert_offset;
            std::array<std::array<uint64_t, 3>, 3> cache_stats;
            ggml_cuda_moe_legacy_debug_telemetry telemetry;
        };
        const std::array<phase_spec, 3> phases = {{
            {"overflow-1", 64, 0, {{{63, 63, 0}, {0, 63, 0}, {63, 63, 0}}}, {3, 3, 3, 3, 64, 2}},
            {"narrow", 7, 45, {{{83, 64, 1}, {20, 64, 1}, {83, 64, 1}}}, {6, 3, 3, 3, 64, 4}},
            {"overflow-2", 64, 0, {{{89, 121, 58}, {26, 121, 58}, {152, 121, 58}}}, {9, 6, 6, 6, 64, 6}},
        }};
        for (size_t pass = 0; pass < phases.size(); ++pass) {
            const auto & phase = phases[pass];
            set_routed_separate_chain_test_inputs(
                candidate, reference, pass, phase.expert_span, phase.expert_offset);
            const auto expected = run_routed_separate_chain_test_graph(reference_backend.get(), reference);
            const auto actual = run_routed_separate_chain_test_graph(candidate_backend.get(), candidate);
            CHECK(std::unordered_set<int32_t>(actual.ids.begin(), actual.ids.end()).size() ==
                static_cast<size_t>(phase.expert_span));
            fprintf(stderr, "test-moe-cache: routed %s cache", phase.name);
            for (size_t bank = 0; bank < caches.size(); ++bank) {
                const auto & cache = caches[bank];
                uint64_t hits = 0;
                uint64_t misses = 0;
                uint64_t evictions = 0;
                ggml_cuda_moe_cache_stats(cache.get(), &hits, &misses, &evictions);
                fprintf(stderr, " %llu/%llu/%llu",
                    (unsigned long long) hits, (unsigned long long) misses, (unsigned long long) evictions);
                CHECK(hits == phase.cache_stats[bank][0]);
                CHECK(misses == phase.cache_stats[bank][1]);
                CHECK(evictions == phase.cache_stats[bank][2]);
            }
            fprintf(stderr, "\n");
            const auto telemetry = ggml_cuda_moe_grouped_context_test_access::legacy_debug_telemetry(*context, false);
            fprintf(stderr, "test-moe-cache: routed %s legacy %llu/%llu/%llu/%llu/%llu/%llu\n",
                phase.name,
                (unsigned long long) telemetry.ops,
                (unsigned long long) telemetry.staged_ops,
                (unsigned long long) telemetry.split_staged_ops,
                (unsigned long long) telemetry.overflow_ops,
                (unsigned long long) telemetry.unique_experts_max,
                (unsigned long long) telemetry.ids_cache_hits);
            CHECK(telemetry.ops == phase.telemetry.ops);
            CHECK(telemetry.staged_ops == phase.telemetry.staged_ops);
            CHECK(telemetry.split_staged_ops == phase.telemetry.split_staged_ops);
            CHECK(telemetry.overflow_ops == phase.telemetry.overflow_ops);
            CHECK(telemetry.unique_experts_max == phase.telemetry.unique_experts_max);
            CHECK(telemetry.ids_cache_hits == phase.telemetry.ids_cache_hits);
            check_routed_separate_chain_exact(phase.name, expected, actual);
        }
        CHECK(active_grouped_legacy_op_count(candidate_backend.get(), true) == 0);
        fprintf(stderr, "test-moe-cache: routed separate prefill exact OK\n");
    }

    {
        constexpr uint32_t MULTIWAVE_N_SLOTS = 24;
        constexpr int64_t N_TOKENS = 512;
        ggml_backend_ptr reference_backend(ggml_backend_cuda_init(0));
        ggml_backend_ptr candidate_backend(ggml_backend_cuda_init(0));
        CHECK(reference_backend != nullptr && candidate_backend != nullptr);
        auto reference = build_routed_separate_chain_test_graph(
            reference_backend.get(), ggml_backend_cuda_buffer_type(0), N_TOKENS);
        auto candidate = build_routed_separate_chain_test_graph(
            candidate_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), N_TOKENS);
        initialize_routed_separate_chain_test_weights(candidate, reference);
        const auto disabled = candidate_snapshot(MULTIWAVE_N_SLOTS, nullptr, 0);
        CHECK(ggml_backend_cuda_moe_candidate_replace_v1(reference_backend.get(), &disabled) ==
            GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        register_routed_separate_chain_test_graph(candidate_backend.get(), candidate, MULTIWAVE_N_SLOTS);

        auto * context = ggml_cuda_moe_grouped_context_for_test(candidate_backend.get());
        CHECK(context != nullptr);
        const auto coverage = candidate_certify_graph(*context, candidate.graph);
        std::shared_ptr<ggml_cuda_moe_graph_plan> plan;
        ggml_cuda_moe_graph_execution execution;
        CHECK(context->prepare_graph_execution(
            candidate.graph, 993, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &plan, &execution,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_PREFILL_LEGACY);

        cudaStream_t cache_stream = nullptr;
        CUDA_OK(cudaStreamCreateWithFlags(&cache_stream, cudaStreamNonBlocking));
        std::array<ggml_cuda_moe_legacy_cache_lease, 3> caches;
        for (size_t bank = 0; bank < caches.size(); ++bank) {
            caches[bank] = context->acquire_legacy_cache(candidate.banks[bank], nullptr, nullptr, cache_stream);
            CHECK(caches[bank] && caches[bank].acquisition().registered_source == 1);
            CHECK(ggml_cuda_moe_cache_n_slots(caches[bank].get()) == static_cast<int>(MULTIWAVE_N_SLOTS));
            CHECK(ggml_cuda_moe_cache_can_overlap_staging(caches[bank].get()));
            ggml_cuda_moe_cache_reset_stats(caches[bank].get());
        }
        CUDA_OK(cudaStreamSynchronize(cache_stream));
        CUDA_OK(cudaStreamDestroy(cache_stream));
        ggml_cuda_moe_candidate_group_key group_key;
        CHECK(context->find_down_group_key(candidate.banks[2], &group_key));
        CHECK(ggml_cuda_moe_grouped_context_test_access::legacy_backing_count(*context, group_key) == 3);
        CHECK(ggml_cuda_moe_cache_trailing_padding_bytes_for_test(caches[0].get()) == 0);
        CHECK(ggml_cuda_moe_cache_trailing_padding_bytes_for_test(caches[1].get()) == 0);
        CHECK(ggml_cuda_moe_cache_trailing_padding_bytes_for_test(caches[2].get()) == 136);
        CHECK(ggml_cuda_moe_cache_trailing_padding_zero_for_test(caches[2].get()));

        set_routed_separate_chain_test_inputs(candidate, reference, 17, 64, 0);
        ggml_cuda_moe_grouped_context_test_access::poison_split_staging(*context, 3);
        const auto expected = run_routed_separate_chain_test_graph(reference_backend.get(), reference);
        const auto actual = run_routed_separate_chain_test_graph(candidate_backend.get(), candidate);
        CHECK(ggml_cuda_moe_grouped_context_test_access::split_staging_poison_calls(*context) == 0);
        CHECK(std::unordered_set<int32_t>(actual.ids.begin(), actual.ids.end()).size() == 64);
        const std::array<std::array<uint64_t, 3>, 3> expected_cache_stats = {{
            {24, 24, 0}, {0, 24, 0}, {24, 24, 0},
        }};
        for (size_t bank = 0; bank < caches.size(); ++bank) {
            uint64_t hits = 0;
            uint64_t misses = 0;
            uint64_t evictions = 0;
            ggml_cuda_moe_cache_stats(caches[bank].get(), &hits, &misses, &evictions);
            CHECK(hits == expected_cache_stats[bank][0]);
            CHECK(misses == expected_cache_stats[bank][1]);
            CHECK(evictions == expected_cache_stats[bank][2]);
        }
        const auto telemetry = ggml_cuda_moe_grouped_context_test_access::legacy_debug_telemetry(*context, false);
        CHECK(telemetry.ops == 3 && telemetry.staged_ops == 3 && telemetry.split_staged_ops == 3);
        CHECK(telemetry.overflow_ops == 3 && telemetry.unique_experts_max == 64 && telemetry.ids_cache_hits == 2);
        check_routed_separate_chain_exact("multiwave", expected, actual);
        CHECK(ggml_cuda_moe_cache_trailing_padding_zero_for_test(caches[2].get()));
        CHECK(active_grouped_legacy_op_count(candidate_backend.get(), true) == 0);
        fprintf(stderr, "test-moe-cache: routed separate cache24 two-wave exact OK\n");
    }

    {
        constexpr int64_t N_TOKENS = 1;
        ggml_backend_ptr reference_backend(ggml_backend_cuda_init(0));
        ggml_backend_ptr candidate_backend(ggml_backend_cuda_init(0));
        CHECK(reference_backend != nullptr && candidate_backend != nullptr);
        auto reference = build_routed_separate_chain_test_graph(
            reference_backend.get(), ggml_backend_cuda_buffer_type(0), N_TOKENS);
        auto candidate = build_routed_separate_chain_test_graph(
            candidate_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), N_TOKENS);
        initialize_routed_separate_chain_test_weights(candidate, reference);
        set_routed_separate_chain_test_inputs(candidate, reference, 0, 64, 0);
        const auto disabled = candidate_snapshot(N_SLOTS, nullptr, 0);
        CHECK(ggml_backend_cuda_moe_candidate_replace_v1(reference_backend.get(), &disabled) ==
            GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        register_routed_separate_chain_test_graph(candidate_backend.get(), candidate, N_SLOTS);

        auto * context = ggml_cuda_moe_grouped_context_for_test(candidate_backend.get());
        CHECK(context != nullptr);
        const auto coverage = candidate_certify_graph(*context, candidate.graph);
        std::shared_ptr<ggml_cuda_moe_graph_plan> plan;
        ggml_cuda_moe_graph_execution execution;
        CHECK(context->prepare_graph_execution(
            candidate.graph, 992, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &plan, &execution,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        fprintf(stderr, "test-moe-cache: routed decode outcome=%u size=%u eligible=%u execution=%u materialization=%u consumer=%u auxiliary=%u\n",
            execution.outcome(), execution.size(),
            ggml_cuda_moe_grouped_context_test_access::graph_group_has_eligible_reason(*plan, 0),
            ggml_cuda_moe_grouped_context_test_access::graph_group_has_execution_reason(*plan, 0),
            ggml_cuda_moe_grouped_context_test_access::graph_group_has_materialization_reason(*plan, 0),
            ggml_cuda_moe_grouped_context_test_access::graph_group_has_consumer_equivalence_reason(*plan, 0),
            ggml_cuda_moe_grouped_context_test_access::graph_group_has_auxiliary_reason(*plan, 0));
        CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED && execution.size() == 1);
        CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_eligible_reason(*plan, 0));

        const auto expected = run_routed_separate_chain_test_graph(reference_backend.get(), reference);
        const auto actual = run_routed_separate_chain_test_graph(candidate_backend.get(), candidate);
        check_routed_separate_chain_exact("decode", expected, actual);
        CHECK(active_grouped_legacy_op_count(candidate_backend.get()) == 0);
        const auto telemetry = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
        fprintf(stderr, "test-moe-cache: routed decode telemetry registered=%llu covered=%llu plan_calls=%llu calls=%llu ready=%llu completed=%llu\n",
            (unsigned long long) telemetry.registered, (unsigned long long) telemetry.covered,
            (unsigned long long) telemetry.plan_calls, (unsigned long long) telemetry.calls,
            (unsigned long long) telemetry.ready, (unsigned long long) telemetry.completed);
        CHECK(telemetry.registered == 1 && telemetry.covered == 1 && telemetry.plan_calls >= 1);
        CHECK(telemetry.calls == 1 && telemetry.ready == 1 && telemetry.completed == 1);
        CHECK(telemetry.fallback == 0 && telemetry.prepare_error == 0 && telemetry.finish_error == 0);
        fprintf(stderr, "test-moe-cache: routed separate decode exact OK\n");
    }

    ggml_backend_cuda_moe_set_debug_mm(old_debug_mm);
}

struct gemma_q4_parity_weights {
    ggml_context_ptr context;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * gate_up = nullptr;
    ggml_tensor * down = nullptr;
};

struct gemma_q4_parity_graphs {
    ggml_context_ptr context;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * gate_input = nullptr;
    ggml_tensor * down_input = nullptr;
    ggml_tensor * ids_storage = nullptr;
    std::array<ggml_tensor *, 3> outputs = {};
    ggml_cgraph * graph = nullptr;
};

struct gemma_q4_routes {
    std::vector<int32_t> padded;
    std::vector<int32_t> unique;
};

static gemma_q4_parity_weights build_gemma_q4_parity_weights(
        ggml_backend_buffer_type_t buft) {
    constexpr int64_t N_EXPERTS = 128;
    constexpr int64_t N_EMBD = 2816;
    constexpr int64_t N_FF = 704;
    const ggml_init_params params = {
        /* .mem_size = */ ggml_tensor_overhead() * 4,
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };

    gemma_q4_parity_weights result;
    result.context.reset(ggml_init(params));
    CHECK(result.context != nullptr);
    const int64_t gate_up_ne[] = {N_EMBD, 2 * N_FF, N_EXPERTS};
    const int64_t down_ne[] = {N_FF, N_EMBD, N_EXPERTS};
    result.gate_up = ggml_new_tensor(result.context.get(), GGML_TYPE_Q4_0, 3, gate_up_ne);
    result.down = ggml_new_tensor(result.context.get(), GGML_TYPE_Q4_0, 3, down_ne);
    ggml_set_name(result.gate_up, "blk.0.ffn_gate_up_exps.weight");
    ggml_set_name(result.down, "blk.0.ffn_down_exps.weight");
    result.buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(result.context.get(), buft));
    CHECK(result.buffer != nullptr);
    ggml_backend_buffer_set_usage(result.buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    CHECK(result.gate_up->nb[2] == 2230272 && result.down->nb[2] == 1115136);
    CHECK(ggml_nbytes(result.gate_up) == 285474816 && ggml_nbytes(result.down) == 142737408);
    return result;
}

static gemma_q4_parity_graphs build_gemma_q4_parity_graphs(
        ggml_backend_t backend,
        ggml_tensor * gate_up,
        ggml_tensor * down,
        int64_t n_rows) {
    constexpr int64_t N_EMBD = 2816;
    constexpr int64_t N_FF = 704;
    constexpr int64_t N_USED = 8;
    constexpr int64_t IDS_STRIDE = 12;
    const ggml_init_params params = {
        /* .mem_size = */ ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false),
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };

    gemma_q4_parity_graphs result;
    result.context.reset(ggml_init(params));
    CHECK(result.context != nullptr);
    result.gate_input = ggml_new_tensor_3d(result.context.get(), GGML_TYPE_F32, N_EMBD, 1, n_rows);
    result.down_input = ggml_new_tensor_3d(result.context.get(), GGML_TYPE_F32, N_FF, N_USED, n_rows);
    result.ids_storage = ggml_new_tensor_2d(result.context.get(), GGML_TYPE_I32, IDS_STRIDE, n_rows);
    ggml_tensor * ids = ggml_view_2d(
        result.context.get(), result.ids_storage, N_USED, n_rows, result.ids_storage->nb[1], 0);
    ggml_set_name(result.gate_input, "test.gemma_q4.gate_input");
    ggml_set_name(result.down_input, "test.gemma_q4.down_input");
    ggml_set_name(result.ids_storage, "test.gemma_q4.ids_storage");
    ggml_set_name(ids, "test.gemma_q4.ids");

    result.outputs[0] = ggml_mul_mat_id(result.context.get(), gate_up, result.gate_input, ids);
    result.outputs[1] = ggml_mul_mat_id(result.context.get(), down, result.down_input, ids);
    ggml_tensor * gate_up_composed = ggml_mul_mat_id(result.context.get(), gate_up, result.gate_input, ids);
    ggml_tensor * gate = ggml_view_3d(
        result.context.get(), gate_up_composed, N_FF, N_USED, n_rows,
        gate_up_composed->nb[1], gate_up_composed->nb[2], 0);
    ggml_tensor * up = ggml_view_3d(
        result.context.get(), gate_up_composed, N_FF, N_USED, n_rows,
        gate_up_composed->nb[1], gate_up_composed->nb[2], N_FF * gate_up_composed->nb[0]);
    ggml_tensor * hidden = ggml_geglu_split(result.context.get(), gate, up);
    result.outputs[2] = ggml_mul_mat_id(result.context.get(), down, hidden, ids);
    ggml_set_name(result.outputs[0], "test.gemma_q4.gate_output");
    ggml_set_name(result.outputs[1], "test.gemma_q4.down_output");
    ggml_set_name(result.outputs[2], "test.gemma_q4.composed_output");

    result.graph = ggml_new_graph_custom(result.context.get(), 64, false);
    for (ggml_tensor * output : result.outputs) {
        ggml_build_forward_expand(result.graph, output);
    }
    result.buffer.reset(ggml_backend_alloc_ctx_tensors(result.context.get(), backend));
    CHECK(result.buffer != nullptr);
    return result;
}

static std::vector<uint8_t> gemma_q4_expert_data(const ggml_tensor * tensor, uint32_t salt) {
    struct q4_0_block {
        ggml_fp16_t d;
        uint8_t qs[16];
    };
    static_assert(sizeof(q4_0_block) == 18);
    constexpr size_t QK4_0_TEST = 32;
    CHECK(tensor != nullptr && tensor->type == GGML_TYPE_Q4_0 && tensor->ne[0] % QK4_0_TEST == 0);
    std::vector<uint8_t> result(ggml_nbytes(tensor), 0);
    const size_t blocks_per_row = tensor->ne[0] / QK4_0_TEST;
    CHECK(tensor->nb[1] == blocks_per_row * sizeof(q4_0_block));
    for (int64_t expert = 0; expert < tensor->ne[2]; ++expert) {
        for (int64_t row = 0; row < tensor->ne[1]; ++row) {
            for (size_t block_index = 0; block_index < blocks_per_row; ++block_index) {
                uint32_t state = salt ^ (uint32_t) expert * 0x9e3779b9u ^ (uint32_t) row * 0x85ebca6bu ^
                    (uint32_t) block_index * 0xc2b2ae35u;
                q4_0_block block;
                block.d = ggml_fp32_to_fp16(0.0005f * (1 + (state % 29)));
                for (size_t byte = 0; byte < sizeof(block.qs); ++byte) {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    block.qs[byte] = (uint8_t) state;
                }
                const size_t offset = (size_t) expert * tensor->nb[2] + (size_t) row * tensor->nb[1] +
                    block_index * sizeof(q4_0_block);
                memcpy(result.data() + offset, &block, sizeof(block));
            }
        }
    }
    return result;
}

static std::vector<float> gemma_q4_input_data(const ggml_tensor * tensor, uint32_t salt) {
    CHECK(tensor != nullptr && tensor->type == GGML_TYPE_F32);
    std::vector<float> result(ggml_nelements(tensor));
    uint32_t state = salt;
    for (float & value : result) {
        state = state * 1664525u + 1013904223u;
        value = ((int32_t) (state % 2001) - 1000) * 0.0005f;
    }
    return result;
}

static gemma_q4_routes gemma_q4_route_data(int64_t n_rows, uint32_t n_unique) {
    constexpr uint32_t N_EXPERTS = 128;
    constexpr uint32_t N_USED = 8;
    constexpr uint32_t IDS_STRIDE = 12;
    CHECK(n_rows > 0 && n_unique >= N_USED && n_unique <= N_EXPERTS);
    std::vector<int32_t> experts = {127, 0};
    for (uint32_t index = 0; experts.size() < n_unique; ++index) {
        const int32_t expert = (int32_t) ((37 * index + 11) % N_EXPERTS);
        if (std::find(experts.begin(), experts.end(), expert) == experts.end()) {
            experts.push_back(expert);
        }
    }

    gemma_q4_routes result;
    result.padded.assign((size_t) IDS_STRIDE * n_rows, -7777777);
    std::array<bool, N_EXPERTS> seen = {};
    for (int64_t row = 0; row < n_rows; ++row) {
        for (uint32_t route = 0; route < N_USED; ++route) {
            const size_t position = (size_t) row * N_USED + route;
            const size_t expert_index = position < experts.size() ? position : (position * 5 + (size_t) row * 3) % experts.size();
            const int32_t expert = experts[expert_index];
            result.padded[(size_t) row * IDS_STRIDE + route] = expert;
            if (!seen[expert]) {
                seen[expert] = true;
                result.unique.push_back(expert);
            }
        }
    }
    CHECK(std::find(result.unique.begin(), result.unique.end(), 0) != result.unique.end());
    CHECK(std::find(result.unique.begin(), result.unique.end(), 127) != result.unique.end());
    CHECK(result.unique.size() == n_unique);
    return result;
}

static void check_gemma_q4_slab(
        const std::vector<uint8_t> & expected,
        size_t expert_stride,
        const void * actual_device,
        int32_t expert,
        const char * boundary) {
    CHECK((size_t) expert < expected.size() / expert_stride);
    std::vector<uint8_t> actual_bytes(expert_stride);
    CUDA_OK(cudaMemcpy(actual_bytes.data(), actual_device, actual_bytes.size(), cudaMemcpyDeviceToHost));
    const uint8_t * expected_bytes = expected.data() + (size_t) expert * expert_stride;
    if (memcmp(expected_bytes, actual_bytes.data(), expert_stride) != 0) {
        fprintf(stderr, "test-moe-cache: first Gemma Q4_0 byte mismatch at %s expert=%d\n", boundary, expert);
    }
    CHECK(memcmp(expected_bytes, actual_bytes.data(), expert_stride) == 0);
}

static void check_gemma_q4_materialization(
        int device,
        uint32_t n_slots,
        const gemma_q4_parity_weights & cached,
        const std::array<const std::vector<uint8_t> *, 2> & expected,
        const std::vector<int32_t> & unique) {
    CHECK((n_slots == 12 && unique.size() > n_slots) || (n_slots == 48 && unique.size() <= n_slots));
    cudaStream_t stream = nullptr;
    CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const std::array<ggml_tensor *, 2> banks = {cached.gate_up, cached.down};
    const std::array<size_t, 3> representative = {0, unique.size() / 2, unique.size() - 1};
    for (size_t bank_index = 0; bank_index < banks.size(); ++bank_index) {
        ggml_tensor * bank = banks[bank_index];
        const auto & bytes = *expected[bank_index];
        CHECK(bytes.size() == ggml_nbytes(bank));
        for (size_t index : representative) {
            const int32_t expert = unique[index];
            CHECK(memcmp(
                static_cast<const char *>(bank->data) + (size_t) expert * bank->nb[2],
                bytes.data() + (size_t) expert * bank->nb[2], bank->nb[2]) == 0);
        }

        std::unique_ptr<ggml_cuda_moe_cache, decltype(&ggml_cuda_moe_cache_free)> cache(
            ggml_cuda_moe_cache_init(device, bank->nb[2], n_slots, false, 0, 0),
            ggml_cuda_moe_cache_free);
        CHECK(cache != nullptr);
        const char * source = static_cast<const char *>(bank->data);
        cudaStream_t copy_stream = ggml_cuda_moe_cache_copy_stream(cache.get());
        std::vector<int> slots(unique.size(), -1);
        void * misses = nullptr;
        if (n_slots == 48) {
            size_t index = 0;
            for (int32_t expert : unique) {
                const int slot = ggml_cuda_moe_cache_acquire(
                    cache.get(), source + (size_t) expert * bank->nb[2], bank->nb[2],
                    copy_stream, false, false, false, true);
                CHECK(slot >= 0);
                slots[index++] = slot;
            }
            CUDA_OK(cudaStreamSynchronize(copy_stream));
        } else {
            std::vector<int> primed_slots;
            for (size_t index = 0; index < 4; ++index) {
                const int slot = ggml_cuda_moe_cache_acquire(
                    cache.get(), source + (size_t) unique[index] * bank->nb[2], bank->nb[2],
                    copy_stream, false, false, false, true);
                CHECK(slot >= 0);
                primed_slots.push_back(slot);
            }
            CUDA_OK(cudaStreamSynchronize(copy_stream));
            ggml_cuda_moe_cache_release_slots(cache.get(), primed_slots.data(), primed_slots.size());

            std::vector<const void *> sources;
            for (int32_t expert : unique) {
                sources.push_back(source + (size_t) expert * bank->nb[2]);
            }
            CUDA_OK(cudaMalloc(&misses, (unique.size() - n_slots) * bank->nb[2]));
            int n_resident = 0;
            int n_wait_classes = 0;
            CHECK(ggml_cuda_moe_cache_prepare_split_staging(
                cache.get(), sources.data(), sources.size(), bank->nb[2], 0, 2,
                slots.data(), nullptr, &n_resident, misses, nullptr, 0, &n_wait_classes, stream));
            CHECK(n_resident == (int) n_slots && n_wait_classes == 1);
            CHECK(ggml_cuda_moe_cache_finish_split_staging(cache.get(), stream));
            CUDA_OK(cudaStreamSynchronize(stream));
        }

        std::vector<std::pair<size_t, const void *>> resident;
        std::vector<std::pair<size_t, const void *>> overflow;
        size_t miss_index = 0;
        for (size_t index = 0; index < unique.size(); ++index) {
            if (slots[index] >= 0) {
                resident.push_back({index, ggml_cuda_moe_cache_slot_ptr(cache.get(), slots[index])});
            } else {
                overflow.push_back({index, static_cast<const char *>(misses) + miss_index++ * bank->nb[2]});
            }
        }
        CHECK(resident.size() == std::min<size_t>(n_slots, unique.size()));
        CHECK(resident.size() + overflow.size() == unique.size());
        for (const auto & materialized : {&resident, &overflow}) {
            if (materialized->empty()) {
                continue;
            }
            for (size_t position : std::array<size_t, 3>{0, materialized->size() / 2, materialized->size() - 1}) {
                const auto [index, data] = (*materialized)[position];
                check_gemma_q4_slab(bytes, bank->nb[2], data, unique[index],
                    slots[index] >= 0 ? "resident slot" : "split overflow staging");
            }
        }
        if (n_slots == 48) {
            ggml_cuda_moe_cache_release_slots(cache.get(), slots.data(), slots.size());
        } else {
            CHECK(ggml_cuda_moe_cache_release_split_slots(cache.get(), slots.data(), slots.size(), stream));
            CUDA_OK(cudaStreamSynchronize(stream));
            CUDA_OK(cudaFree(misses));
        }
    }
    CUDA_OK(cudaStreamDestroy(stream));
}

static ggml_cuda_mmid_capability native_mmid_capability(
        int device,
        const ggml_tensor * weight,
        int64_t n_rows,
        ggml_cuda_mmid_mapping mapping) {
    cudaDeviceProp properties;
    CUDA_OK(cudaGetDeviceProperties(&properties, device));
    ggml_cuda_mmid_capability_query query;
    query.source_type = weight->type;
    query.input_type = GGML_TYPE_F32;
    query.output_type = GGML_TYPE_F32;
    memcpy(query.source_ne, weight->ne, sizeof(query.source_ne));
    memcpy(query.source_nb, weight->nb, sizeof(query.source_nb));
    query.n_tokens = n_rows;
    query.n_experts = weight->ne[2];
    query.cc = properties.major * 100 + properties.minor * 10;
    query.warp_size = properties.warpSize;
    query.smpbo = std::max(properties.sharedMemPerBlock, properties.sharedMemPerBlockOptin);
    query.phase = n_rows == 1 ? GGML_CUDA_MMID_PHASE_DECODE : GGML_CUDA_MMID_PHASE_PREFILL;
    query.mapping = mapping;
    query.use_mmq = ggml_cuda_moe_use_mmq(weight, n_rows);
    return ggml_cuda_mmid_get_capability(query);
}

static std::array<std::vector<float>, 3> run_gemma_q4_graph(
        ggml_backend_t backend,
        gemma_q4_parity_graphs & graphs) {
    CHECK(ggml_backend_graph_compute(backend, graphs.graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
    std::array<std::vector<float>, 3> result;
    for (size_t index = 0; index < result.size(); ++index) {
        result[index].resize(ggml_nelements(graphs.outputs[index]));
        ggml_backend_tensor_get(graphs.outputs[index], result[index].data(), 0, ggml_nbytes(graphs.outputs[index]));
    }
    return result;
}

static void compare_gemma_q4_output(
        const char * boundary,
        const std::vector<float> & expected,
        const std::vector<float> & actual) {
    CHECK(expected.size() == actual.size());
    size_t first_mismatch = expected.size();
    for (size_t index = 0; index < expected.size(); ++index) {
        CHECK(std::isfinite(expected[index]) && std::isfinite(actual[index]));
        if (first_mismatch == expected.size() && expected[index] != actual[index]) {
            first_mismatch = index;
        }
    }
    if (first_mismatch != expected.size()) {
        fprintf(stderr, "test-moe-cache: first Gemma Q4_0 output mismatch at %s index=%zu expected=%.9g actual=%.9g\n",
            boundary, first_mismatch, expected[first_mismatch], actual[first_mismatch]);
    }
    CHECK(first_mismatch == expected.size());
}

static void run_gemma_q4_parity_case(
        ggml_backend_t direct_backend,
        const gemma_q4_parity_weights & direct_weights,
        ggml_backend_t cached_backend,
        const gemma_q4_parity_weights & cached_weights,
        const char * boundary,
        int64_t n_rows,
        uint32_t route_union) {
    const auto routes = gemma_q4_route_data(n_rows, route_union);
    auto direct_graphs = build_gemma_q4_parity_graphs(
        direct_backend, direct_weights.gate_up, direct_weights.down, n_rows);
    auto cached_graphs = build_gemma_q4_parity_graphs(
        cached_backend, cached_weights.gate_up, cached_weights.down, n_rows);
    const auto gate_input = gemma_q4_input_data(direct_graphs.gate_input, 0x1451u);
    const auto down_input = gemma_q4_input_data(direct_graphs.down_input, 0x8d37u);
    for (gemma_q4_parity_graphs * graph : {&direct_graphs, &cached_graphs}) {
        ggml_backend_tensor_set(graph->gate_input, gate_input.data(), 0, ggml_nbytes(graph->gate_input));
        ggml_backend_tensor_set(graph->down_input, down_input.data(), 0, ggml_nbytes(graph->down_input));
        CHECK(ggml_nelements(graph->ids_storage) == (int64_t) routes.padded.size());
        ggml_backend_tensor_set(graph->ids_storage, routes.padded.data(), 0, ggml_nbytes(graph->ids_storage));
    }
    const auto expected = run_gemma_q4_graph(direct_backend, direct_graphs);
    const auto actual = run_gemma_q4_graph(cached_backend, cached_graphs);
    const std::array<const char *, 3> outputs = {"gate_up", "down", "GEGLU/down"};
    for (size_t index = 0; index < outputs.size(); ++index) {
        const std::string name = std::string(boundary) + " " + outputs[index];
        compare_gemma_q4_output(name.c_str(), expected[index], actual[index]);
    }
}

static void test_gemma_q4_cached_cuda_parity(int device) {
    const int old_slots = ggml_backend_cuda_moe_get_cache_slots();
    ggml_backend_ptr direct_backend(ggml_backend_cuda_init(device));
    CHECK(direct_backend != nullptr);
    auto direct_weights = build_gemma_q4_parity_weights(ggml_backend_cuda_buffer_type(device));
    const auto gate_up_data = gemma_q4_expert_data(direct_weights.gate_up, 0x31a5u);
    const auto down_data = gemma_q4_expert_data(direct_weights.down, 0x7c29u);
    ggml_backend_tensor_set(direct_weights.gate_up, gate_up_data.data(), 0, gate_up_data.size());
    ggml_backend_tensor_set(direct_weights.down, down_data.data(), 0, down_data.size());
    ggml_backend_synchronize(direct_backend.get());

    const auto b16_direct = native_mmid_capability(
        device, direct_weights.gate_up, 16, GGML_CUDA_MMID_MAPPING_DIRECT);
    const auto b16_mapped = native_mmid_capability(
        device, direct_weights.gate_up, 16, GGML_CUDA_MMID_MAPPING_SOURCE_MAP);
    CHECK(b16_direct.reason == GGML_CUDA_MMID_CAPABILITY_OK &&
        b16_direct.selection == GGML_CUDA_MMID_CONSUMER_MMQ &&
        b16_mapped.reason == GGML_CUDA_MMID_CAPABILITY_OK &&
        b16_mapped.selection == GGML_CUDA_MMID_CONSUMER_MMQ);
    const auto b4_direct = native_mmid_capability(
        device, direct_weights.gate_up, 4, GGML_CUDA_MMID_MAPPING_DIRECT);
    const auto b4_mapped = native_mmid_capability(
        device, direct_weights.gate_up, 4, GGML_CUDA_MMID_MAPPING_SOURCE_MAP);
    CHECK(b4_direct.reason == GGML_CUDA_MMID_CAPABILITY_OK &&
        b4_direct.selection == GGML_CUDA_MMID_CONSUMER_MMVQ &&
        b4_mapped.reason == GGML_CUDA_MMID_CAPABILITY_OK &&
        b4_mapped.selection == GGML_CUDA_MMID_CONSUMER_MMQ);
    const auto b112_direct = native_mmid_capability(
        device, direct_weights.gate_up, 112, GGML_CUDA_MMID_MAPPING_DIRECT);
    const auto b112_mapped = native_mmid_capability(
        device, direct_weights.gate_up, 112, GGML_CUDA_MMID_MAPPING_SOURCE_MAP);
    CHECK(b112_direct.reason == GGML_CUDA_MMID_CAPABILITY_OK &&
        b112_direct.selection == GGML_CUDA_MMID_CONSUMER_MMQ &&
        b112_mapped.reason == GGML_CUDA_MMID_CAPABILITY_OK &&
        b112_mapped.selection == GGML_CUDA_MMID_CONSUMER_MMQ);

    for (uint32_t n_slots : {12u, 48u}) {
        ggml_backend_cuda_moe_set_cache_slots(n_slots);
        ggml_backend_ptr cached_backend(ggml_backend_cuda_init(device));
        CHECK(cached_backend != nullptr);
        const auto disabled = candidate_snapshot(n_slots, nullptr, 0);
        CHECK(ggml_backend_cuda_moe_candidate_replace_v1(cached_backend.get(), &disabled) ==
            GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        auto cached_weights = build_gemma_q4_parity_weights(ggml_backend_cuda_moe_cached_buffer_type());
        ggml_backend_tensor_set(cached_weights.gate_up, gate_up_data.data(), 0, gate_up_data.size());
        ggml_backend_tensor_set(cached_weights.down, down_data.data(), 0, down_data.size());

        const uint32_t b16_union = n_slots == 12 ? 24 : 40;
        const auto materialization_routes = gemma_q4_route_data(16, b16_union);
        check_gemma_q4_materialization(
            device, n_slots, cached_weights, {&gate_up_data, &down_data}, materialization_routes.unique);
        run_gemma_q4_parity_case(
            direct_backend.get(), direct_weights, cached_backend.get(), cached_weights,
            n_slots == 12 ? "cache12 B16" : "cache48 B16", 16, b16_union);
        run_gemma_q4_parity_case(
            direct_backend.get(), direct_weights, cached_backend.get(), cached_weights,
            n_slots == 12 ? "cache12 B4" : "cache48 B4", 4, 24);
        if (n_slots == 48) {
            run_gemma_q4_parity_case(
                direct_backend.get(), direct_weights, cached_backend.get(), cached_weights,
                "cache48 B112 sparse", 112, 96);
            run_gemma_q4_parity_case(
                direct_backend.get(), direct_weights, cached_backend.get(), cached_weights,
                "cache48 B136 sparse fixup", 136, 96);
        }
    }
    ggml_backend_cuda_moe_set_cache_slots(old_slots);
    fprintf(stderr, "test-moe-cache: opt-in Gemma Q4_0 cached CUDA parity diagnostic OK\n");
}

struct grouped_decode_fixture {
    static constexpr uint32_t N_EXPERTS = 8;
    static constexpr uint32_t N_SLOTS = 4;
    static constexpr size_t SOURCE_BYTES = 1024 * 1024;

    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t source_buffer = nullptr;
    ggml_backend_buffer_t ids_buffer = nullptr;
    ggml_context * ctx = nullptr;
    void * source_storage = nullptr;
    uint32_t n_experts = N_EXPERTS;
    size_t source_offset = 0;

    explicit grouped_decode_fixture(
            int device,
            bool pinned = true,
            size_t source_bytes = SOURCE_BYTES,
            uint32_t n_experts = N_EXPERTS) : n_experts(n_experts) {
        backend = ggml_backend_cuda_init(device);
        CHECK(backend != nullptr);
        if (pinned) {
            source_buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_moe_cached_buffer_type(), source_bytes);
        } else {
            CHECK(posix_memalign(&source_storage, 64, source_bytes) == 0);
            source_buffer = ggml_backend_cuda_moe_cached_buffer_from_host_ptr(source_storage, source_bytes);
        }
        ids_buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(device), 4096);
        CHECK(source_buffer != nullptr && ids_buffer != nullptr);
        ggml_init_params params = {};
        params.mem_size = 32 * ggml_tensor_overhead();
        params.no_alloc = true;
        ctx = ggml_init(params);
        CHECK(ctx != nullptr);
    }

    ~grouped_decode_fixture() {
        ggml_free(ctx);
        ggml_backend_buffer_free(ids_buffer);
        ggml_backend_buffer_free(source_buffer);
        free(source_storage);
        ggml_backend_free(backend);
    }

    ggml_tensor * weight(ggml_type type, int64_t ne0, int64_t ne1) {
        const int64_t ne[] = {ne0, ne1, n_experts};
        ggml_tensor * tensor = ggml_new_tensor(ctx, type, 3, ne);
        const size_t alignment = ggml_backend_buffer_get_alignment(source_buffer);
        source_offset = (source_offset + alignment - 1) / alignment * alignment;
        CHECK(source_offset + ggml_nbytes(tensor) <= ggml_backend_buffer_get_size(source_buffer));
        tensor->buffer = source_buffer;
        tensor->data = static_cast<char *>(ggml_backend_buffer_get_base(source_buffer)) + source_offset;
        source_offset += ggml_nbytes(tensor);
        return tensor;
    }

    ggml_tensor * scale() {
        ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_experts);
        const size_t alignment = ggml_backend_buffer_get_alignment(source_buffer);
        source_offset = (source_offset + alignment - 1) / alignment * alignment;
        CHECK(source_offset + ggml_nbytes(tensor) <= ggml_backend_buffer_get_size(source_buffer));
        tensor->buffer = source_buffer;
        tensor->data = static_cast<char *>(ggml_backend_buffer_get_base(source_buffer)) + source_offset;
        source_offset += ggml_nbytes(tensor);
        return tensor;
    }

    ggml_tensor * ids(int64_t n_routes = 4) {
        const int64_t ne[] = {n_routes, 1, 1, 1};
        ggml_tensor * tensor = ggml_new_tensor(ctx, GGML_TYPE_I32, 4, ne);
        tensor->buffer = ids_buffer;
        tensor->data = ggml_backend_buffer_get_base(ids_buffer);
        return tensor;
    }
};

static void test_owner_legacy_cache(int device) {
    grouped_decode_fixture fixture(device);
    ggml_tensor * gate_up = fixture.weight(GGML_TYPE_Q4_0, 32, 64);
    ggml_tensor * down = fixture.weight(GGML_TYPE_Q4_0, 32, 32);
    ggml_tensor * gate = fixture.weight(GGML_TYPE_Q4_K, 256, 256);
    ggml_tensor * up = fixture.weight(GGML_TYPE_Q4_K, 256, 256);
    ggml_tensor * separate_down = fixture.weight(GGML_TYPE_Q4_K, 256, 256);
    ggml_set_name(gate_up, "test.owner.ffn_gate_up_exps.weight");
    ggml_set_name(down, "test.owner.ffn_down_exps.weight");
    ggml_set_name(gate, "test.owner.ffn_gate_exps.weight");
    ggml_set_name(up, "test.owner.ffn_up_exps.weight");
    ggml_set_name(separate_down, "test.owner.separate.ffn_down_exps.weight");
    std::array<ggml_backend_moe_candidate_bank_v1, 2> fused_banks = {{
        {gate_up, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, 0},
        {down, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
    }};
    std::array<ggml_backend_moe_candidate_bank_v1, 3> separate_banks = {{
        {gate, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT, 0},
        {up, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT, 0},
        {separate_down, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
    }};
    std::array<ggml_backend_moe_candidate_group_v1, 2> groups = {{
        {fused_banks.data(), fused_banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 0, 0},
        {separate_banks.data(), separate_banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 0, 0},
    }};
    const auto snapshot = candidate_snapshot(grouped_decode_fixture::N_SLOTS, groups.data(), groups.size());

    ggml_cuda_moe_grouped_context first(ggml_backend_get_device(fixture.backend), device);
    ggml_cuda_moe_grouped_context second(ggml_backend_get_device(fixture.backend), device);
    CHECK(first.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(second.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);

    auto first_gate = first.acquire_legacy_cache(gate_up);
    auto second_gate = second.acquire_legacy_cache(gate_up);
    CHECK(first_gate && second_gate && first_gate.get() != nullptr && second_gate.get() != nullptr);
    CHECK(first_gate.get() != second_gate.get());
    CHECK(ggml_cuda_moe_cache_n_slots(first_gate.get()) == (int) grouped_decode_fixture::N_SLOTS);
    for (uint32_t expert = 0; expert < grouped_decode_fixture::N_EXPERTS; ++expert) {
        memset(static_cast<char *>(down->data) + expert * down->nb[2], 1 + expert, down->nb[2]);
    }

    auto first_down = first.acquire_legacy_cache(down);
    CHECK(first_down && first_down.get() != nullptr && first_down.acquisition().registered_source == 1);
    const bool overlap = ggml_cuda_moe_cache_can_overlap_staging(first_down.get());
    cudaStream_t prefetch_compute_stream = nullptr;
    void * prefetch_staging = nullptr;
    uint32_t * prefetch_stage_ready = nullptr;
    host_barrier prefetch_barrier;
    if (overlap) {
        CUDA_OK(cudaStreamCreateWithFlags(&prefetch_compute_stream, cudaStreamNonBlocking));
        CUDA_OK(cudaMalloc(&prefetch_staging, 2 * down->nb[2]));
        CUDA_OK(cudaMalloc(&prefetch_stage_ready, 3 * sizeof(uint32_t)));
        CUDA_OK(cudaLaunchHostFunc(ggml_cuda_moe_cache_copy_stream(first_down.get()), wait_on_host_barrier, &prefetch_barrier));
        while (!prefetch_barrier.entered.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
    }

    const int32_t experts[] = {1, 3};
    first.prefetch_legacy_siblings(first_gate, experts, 2, true, true);
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
    ggml_cuda_moe_cache_stats(first_down.get(), &hits, &misses, &evictions);
    CHECK(hits == 0 && misses == 2 && evictions == 0);

    if (overlap) {
        const int32_t split_experts[] = {1, 3, 0, 2, 4, 5};
        const void * split_sources[6];
        for (int i = 0; i < 6; ++i) {
            split_sources[i] = static_cast<const char *>(down->data) + split_experts[i] * down->nb[2];
        }
        int slot_ids[6] = {-1, -1, -1, -1, -1, -1};
        int32_t wait_classes[6] = {-1, -1, -1, -1, -1, -1};
        int n_resident = 0;
        int n_wait_classes = 0;
        const bool prepared = ggml_cuda_moe_cache_prepare_split_staging(
            first_down.get(), split_sources, 6, down->nb[2], 0, 1, slot_ids, wait_classes,
            &n_resident, prefetch_staging, prefetch_stage_ready, 3, &n_wait_classes, prefetch_compute_stream);
        prefetch_barrier.released.store(true, std::memory_order_release);
        if (prepared) {
            CHECK(ggml_cuda_moe_cache_finish_split_staging(first_down.get(), prefetch_compute_stream));
            CHECK(ggml_cuda_moe_cache_release_split_slots(first_down.get(), slot_ids, 6, prefetch_compute_stream));
        }
        CUDA_OK(cudaStreamSynchronize(prefetch_compute_stream));
        CUDA_OK(cudaStreamSynchronize(ggml_cuda_moe_cache_copy_stream(first_down.get())));
        CHECK(prepared);
        CHECK(n_resident == 4 && n_wait_classes == 3);
        CHECK(wait_classes[0] == 1 && wait_classes[1] == 1);
        CHECK(wait_classes[2] == 1 && wait_classes[3] == 1);
        CHECK(wait_classes[4] == 2 && wait_classes[5] == 2);

        std::vector<uint8_t> prefetch_readback(down->nb[2]);
        for (int i = 0; i < n_resident; ++i) {
            CUDA_OK(cudaMemcpy(prefetch_readback.data(), ggml_cuda_moe_cache_slot_ptr(first_down.get(), slot_ids[i]),
                down->nb[2], cudaMemcpyDeviceToHost));
            CHECK(std::all_of(prefetch_readback.begin(), prefetch_readback.end(),
                [&](uint8_t value) { return value == 1 + split_experts[i]; }));
        }
        std::vector<uint8_t> staging_readback(2 * down->nb[2]);
        CUDA_OK(cudaMemcpy(staging_readback.data(), prefetch_staging, staging_readback.size(), cudaMemcpyDeviceToHost));
        for (int i = n_resident; i < 6; ++i) {
            CHECK(std::all_of(
                staging_readback.begin() + (i - n_resident) * down->nb[2],
                staging_readback.begin() + (i - n_resident + 1) * down->nb[2],
                [&](uint8_t value) { return value == 1 + split_experts[i]; }));
        }
        CUDA_OK(cudaFree(prefetch_stage_ready));
        CUDA_OK(cudaFree(prefetch_staging));
        CUDA_OK(cudaStreamDestroy(prefetch_compute_stream));
    }

    auto second_up = second.acquire_legacy_cache(up);
    CHECK(second_up && second_up.get() != nullptr && second_up.acquisition().role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT);
    second.prefetch_legacy_siblings(second_up, experts, 2, true, true);
    auto second_separate_gate = second.acquire_legacy_cache(gate);
    auto second_separate_down = second.acquire_legacy_cache(separate_down);
    CHECK(second_separate_gate && second_separate_down);
    CHECK(second_separate_gate.acquisition().role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT);
    CHECK(second_separate_down.acquisition().role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT);
    ggml_cuda_moe_cache_stats(second_separate_gate.get(), &hits, &misses, &evictions);
    CHECK(hits == 0 && misses == 2 && evictions == 0);
    ggml_cuda_moe_cache_stats(second_separate_down.get(), &hits, &misses, &evictions);
    CHECK(hits == 0 && misses == 2 && evictions == 0);

    ggml_backend_cuda_moe_observe_expert_tensor(gate_up->data, gate_up->name, gate_up->nb[2], gate_up->ne[2]);
    ggml_backend_cuda_moe_preallocate_pools(device);
    ggml_cuda_moe_cache_free_all();
    CHECK(ggml_cuda_moe_cache_n_slots(first_gate.get()) == (int) grouped_decode_fixture::N_SLOTS);

    ggml_backend_cuda_moe_log_and_reset_stats();
    ggml_cuda_moe_cache_stats(first_down.get(), &hits, &misses, &evictions);
    CHECK(hits == 0 && misses == 0 && evictions == 0);
    first_down = {};
    first_gate = {};
    const auto disabled = candidate_snapshot(grouped_decode_fixture::N_SLOTS, nullptr, 0);
    CHECK(first.replace(&disabled) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(ggml_cuda_moe_cache_n_slots(second_gate.get()) == (int) grouped_decode_fixture::N_SLOTS);
    ggml_backend_cuda_moe_log_and_reset_stats();
    fprintf(stderr, "test-moe-cache: owner-local legacy cache OK\n");
}

static ggml_cuda_moe_complete_group_key grouped_decode_key(
        ggml_cuda_moe_grouped_context & registry,
        const ggml_tensor * down,
        const ggml_tensor * ids,
        uint32_t layout,
        uint32_t n_banks) {
    ggml_cuda_moe_complete_group_key key;
    CHECK(registry.find_down_group_key(down, &key.candidate));
    key.ids.tensor = ids;
    key.ids.data = ids->data;
    key.ids.buffer = ids->buffer;
    key.ids.type = ids->type;
    for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
        key.ids.ne[dim] = ids->ne[dim];
        key.ids.nb[dim] = ids->nb[dim];
    }
    key.layout = layout;
    key.n_banks = n_banks;
    return key;
}

struct grouped_clock_fixture {
    explicit grouped_clock_fixture(int device, uint32_t n_slots) : fixture(device) {
        weights[0] = fixture.weight(GGML_TYPE_Q4_0, 32, 64);
        weights[1] = fixture.weight(GGML_TYPE_Q4_0, 32, 32);
        banks[0].tensor = weights[0];
        banks[0].role = GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT;
        banks[1].tensor = weights[1];
        banks[1].role = GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT;
        for (uint32_t bank = 0; bank < banks.size(); ++bank) {
            memset(weights[bank]->data, 17 + bank, ggml_nbytes(weights[bank]));
        }
        group.banks = banks.data();
        group.n_banks = banks.size();
        group.layout = GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP;
        snapshot = candidate_snapshot(n_slots, &group, 1);
        registry = std::make_unique<ggml_cuda_moe_grouped_context>(ggml_backend_get_device(fixture.backend), device);
        CHECK(registry->replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        ids = fixture.ids();
        key = grouped_decode_key(*registry, weights[1], ids, group.layout, group.n_banks);
        CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }

    ~grouped_clock_fixture() {
        if (stream != nullptr) {
            CUDA_OK(cudaStreamSynchronize(stream));
            CUDA_OK(cudaStreamDestroy(stream));
        }
    }

    ggml_cuda_moe_grouped_decode_result prepare(
            const std::array<int32_t, 4> & routes,
            ggml_cuda_moe_grouped_decode_acquisition & decode,
            cudaStream_t target) {
        CUDA_OK(cudaMemcpyAsync(ids->data, routes.data(), sizeof(routes), cudaMemcpyHostToDevice, target));
        return registry->prepare_decode(key, target, &decode);
    }

    ggml_cuda_moe_grouped_decode_acquisition warm(const std::array<int32_t, 4> & routes) {
        ggml_cuda_moe_grouped_decode_acquisition decode;
        CHECK(prepare(routes, decode, stream) == GGML_CUDA_MOE_GROUPED_DECODE_READY);
        CHECK(registry->finish_decode(decode, stream));
        CUDA_OK(cudaStreamSynchronize(stream));
        return decode;
    }

    grouped_decode_fixture fixture;
    std::array<ggml_tensor *, 2> weights = {};
    std::array<ggml_backend_moe_candidate_bank_v1, 2> banks = {};
    ggml_backend_moe_candidate_group_v1 group = {};
    ggml_backend_moe_candidate_snapshot_v1 snapshot = {};
    std::unique_ptr<ggml_cuda_moe_grouped_context> registry;
    ggml_tensor * ids = nullptr;
    ggml_cuda_moe_complete_group_key key;
    cudaStream_t stream = nullptr;
};

static void test_grouped_decode_type(
        int device,
        ggml_type type,
        uint32_t layout,
        bool pinned = true,
        uint32_t n_slots = grouped_decode_fixture::N_SLOTS,
        bool auxiliary_scale = false) {
    grouped_decode_fixture fixture(device, pinned);
    const uint32_t n_banks = layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE ? 3 : 2;
    std::array<ggml_tensor *, 3> weights = {};
    std::array<ggml_backend_moe_candidate_bank_v1, 4> banks = {};
    const uint32_t separate_roles[] = {
        GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT,
        GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT,
        GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT,
    };
    const uint32_t fused_roles[] = {
        GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT,
        GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT,
    };
    for (uint32_t bank = 0; bank < n_banks; ++bank) {
        const int64_t ne0 = type == GGML_TYPE_Q4_K ? 256 : type == GGML_TYPE_Q4_0 ? 32 : 64;
        const int64_t ne1 = layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP && bank == 0 ? 2 * ne0 : ne0;
        weights[bank] = fixture.weight(type, ne0, ne1);
        banks[bank].tensor = weights[bank];
        banks[bank].role = layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE ? separate_roles[bank] : fused_roles[bank];
        CHECK(weights[bank]->nb[2] == ggml_row_size(type, ne0) * ne1);
        for (uint32_t expert = 0; expert < grouped_decode_fixture::N_EXPERTS; ++expert) {
            auto * expert_data = static_cast<uint8_t *>(weights[bank]->data) + expert * weights[bank]->nb[2];
            for (size_t byte = 0; byte < weights[bank]->nb[2]; ++byte) {
                expert_data[byte] = static_cast<uint8_t>(17 * (bank + 1) + 13 * expert + byte);
            }
        }
    }
    if (type == GGML_TYPE_NVFP4) {
        CHECK(ggml_blck_size(type) == 64 && ggml_type_size(type) == 36);
    }
    if (auxiliary_scale) {
        banks[n_banks] = {fixture.scale(), GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_SCALE, 0};
    }
    ggml_backend_moe_candidate_group_v1 group = {};
    group.banks = banks.data();
    group.n_banks = n_banks + auxiliary_scale;
    group.layout = layout;
    const auto snapshot = candidate_snapshot(n_slots, &group, 1);
    ggml_cuda_moe_grouped_context registry(ggml_backend_get_device(fixture.backend), device);
    CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);

    ggml_tensor * ids = fixture.ids();
    auto key = grouped_decode_key(registry, weights[n_banks - 1], ids, layout, n_banks);
    const bool check_deferred_commit = type == GGML_TYPE_Q4_K && layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE &&
        pinned && n_slots == grouped_decode_fixture::N_SLOTS && !auxiliary_scale;
    cudaStream_t stream = nullptr;
    cudaStream_t wrong_stream = nullptr;
    CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_OK(cudaStreamCreateWithFlags(&wrong_stream, cudaStreamNonBlocking));

    auto wrong_key = key;
    --wrong_key.n_banks;
    ggml_cuda_moe_grouped_decode_acquisition decode;
    CHECK(registry.prepare_decode(wrong_key, stream, &decode) == GGML_CUDA_MOE_GROUPED_DECODE_FALLBACK);

    const int32_t first_ids[] = {3, 1, 3, 6};
    CUDA_OK(cudaMemcpyAsync(ids->data, first_ids, sizeof(first_ids), cudaMemcpyHostToDevice, stream));
    if (!pinned) {
        CHECK(registry.prepare_decode(key, stream, &decode) == GGML_CUDA_MOE_GROUPED_DECODE_FALLBACK);
        CUDA_OK(cudaStreamSynchronize(stream));
        CUDA_OK(cudaStreamDestroy(wrong_stream));
        CUDA_OK(cudaStreamDestroy(stream));
        return;
    }
    CHECK(registry.prepare_decode(key, stream, &decode) == GGML_CUDA_MOE_GROUPED_DECODE_READY);
    CHECK(decode.n_banks == n_banks && decode.n_slots == n_slots && decode.layout == layout);
    CUDA_OK(cudaStreamSynchronize(stream));
    std::array<int32_t, 4> remapped = {};
    CUDA_OK(cudaMemcpy(remapped.data(), decode.remapped_ids, sizeof(remapped), cudaMemcpyDeviceToHost));
    CHECK((remapped == std::array<int32_t, 4>{0, 1, 0, 2}));
    if (check_deferred_commit) {
        CHECK(ggml_cuda_moe_grouped_context_test_access::device_slot_for_expert(registry, key.candidate, 1) == -1);
        CHECK(ggml_cuda_moe_grouped_context_test_access::device_slot_for_expert(registry, key.candidate, 3) == -1);
        CHECK(ggml_cuda_moe_grouped_context_test_access::device_slot_for_expert(registry, key.candidate, 6) == -1);
    }
    for (uint32_t bank = 0; bank < n_banks; ++bank) {
        CHECK(decode.banks[bank].tensor == weights[bank] && decode.banks[bank].role == banks[bank].role && decode.banks[bank].type == (uint32_t) type);
        std::vector<uint8_t> row(weights[bank]->nb[2]);
        for (uint32_t route : {0u, 1u, 3u}) {
            const int32_t expert = first_ids[route];
            const int32_t slot = remapped[route];
            CUDA_OK(cudaMemcpy(row.data(), static_cast<const char *>(decode.banks[bank].data) + slot * weights[bank]->nb[2], row.size(), cudaMemcpyDeviceToHost));
            CHECK(memcmp(row.data(), static_cast<const char *>(weights[bank]->data) + expert * weights[bank]->nb[2], row.size()) == 0);
        }
    }
    CHECK(!registry.finish_decode(decode, wrong_stream));
    CHECK(registry.finish_decode(decode, stream));

    const int32_t second_ids[] = {6, 2, 7, 6};
    CUDA_OK(cudaMemcpyAsync(ids->data, second_ids, sizeof(second_ids), cudaMemcpyHostToDevice, stream));
    CHECK(registry.prepare_decode(key, stream, &decode) == GGML_CUDA_MOE_GROUPED_DECODE_READY);
    CUDA_OK(cudaStreamSynchronize(stream));
    CUDA_OK(cudaMemcpy(remapped.data(), decode.remapped_ids, sizeof(remapped), cudaMemcpyDeviceToHost));
    const std::array<int32_t, 4> expected_second = n_slots == grouped_decode_fixture::N_SLOTS ?
        std::array<int32_t, 4>{2, 3, 0, 2} : std::array<int32_t, 4>{2, 3, 4, 2};
    CHECK(remapped == expected_second);
    if (check_deferred_commit) {
        CHECK(ggml_cuda_moe_grouped_context_test_access::device_slot_for_expert(registry, key.candidate, 1) == 1);
        CHECK(ggml_cuda_moe_grouped_context_test_access::device_slot_for_expert(registry, key.candidate, 3) == 0);
        CHECK(ggml_cuda_moe_grouped_context_test_access::device_slot_for_expert(registry, key.candidate, 6) == 2);
        CHECK(ggml_cuda_moe_grouped_context_test_access::device_slot_for_expert(registry, key.candidate, 2) == -1);
        CHECK(ggml_cuda_moe_grouped_context_test_access::device_slot_for_expert(registry, key.candidate, 7) == -1);
    }
    CHECK(registry.finish_decode(decode, stream));

    CHECK(registry.prepare_decode(key, stream, &decode) == GGML_CUDA_MOE_GROUPED_DECODE_READY);
    CUDA_OK(cudaStreamSynchronize(stream));
    if (check_deferred_commit) {
        CHECK(ggml_cuda_moe_grouped_context_test_access::device_slot_for_expert(registry, key.candidate, 1) == 1);
        CHECK(ggml_cuda_moe_grouped_context_test_access::device_slot_for_expert(registry, key.candidate, 3) == -1);
        CHECK(ggml_cuda_moe_grouped_context_test_access::device_slot_for_expert(registry, key.candidate, 6) == 2);
        CHECK(ggml_cuda_moe_grouped_context_test_access::device_slot_for_expert(registry, key.candidate, 2) == 3);
        CHECK(ggml_cuda_moe_grouped_context_test_access::device_slot_for_expert(registry, key.candidate, 7) == 0);
    }
    host_barrier barrier;
    CUDA_OK(cudaLaunchHostFunc(stream, wait_on_host_barrier, &barrier));
    while (!barrier.entered.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    std::atomic<bool> finish_done{false};
    std::atomic<bool> finish_result{false};
    std::thread finish_thread([&]() {
        finish_result.store(registry.finish_decode(decode, stream), std::memory_order_release);
        finish_done.store(true, std::memory_order_release);
    });
    for (int attempt = 0; attempt < 100 && !finish_done.load(std::memory_order_acquire); ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const bool finish_was_async = finish_done.load(std::memory_order_acquire);
    barrier.released.store(true, std::memory_order_release);
    finish_thread.join();
    CHECK(finish_was_async && finish_result.load(std::memory_order_acquire));
    CUDA_OK(cudaStreamSynchronize(stream));
    CUDA_OK(cudaStreamDestroy(wrong_stream));
    CUDA_OK(cudaStreamDestroy(stream));
}

static void test_grouped_decode_independent_rows(int device) {
    grouped_decode_fixture fixture(device);
    ggml_tensor * gate_up = fixture.weight(GGML_TYPE_Q4_0, 32, 64);
    ggml_tensor * down = fixture.weight(GGML_TYPE_Q4_0, 32, 32);
    memset(gate_up->data, 17, ggml_nbytes(gate_up));
    memset(down->data, 29, ggml_nbytes(down));
    std::array<ggml_backend_moe_candidate_bank_v1, 2> banks = {{
        {gate_up, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, 0},
        {down, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
    }};
    const ggml_backend_moe_candidate_group_v1 group = {
        banks.data(), banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 0, 0,
    };
    const auto snapshot = candidate_snapshot(12, &group, 1);
    ggml_cuda_moe_grouped_context registry(ggml_backend_get_device(fixture.backend), device);
    CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);

    ggml_tensor * ids = fixture.ids(2);
    ids->ne[1] = 3;
    ids->nb[1] = 4 * sizeof(int32_t);
    ids->nb[2] = ids->ne[1] * ids->nb[1];
    ids->nb[3] = ids->nb[2];
    auto key = grouped_decode_key(registry, down, ids, group.layout, group.n_banks);
    key.execution_semantic_key = 1;
    const std::array<int32_t, 12> padded_routes = {3, 1, -1, -1, 1, 2, -1, -1, 3, 2, -1, -1};
    cudaStream_t stream = nullptr;
    CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_OK(cudaMemcpyAsync(ids->data, padded_routes.data(), sizeof(padded_routes), cudaMemcpyHostToDevice, stream));
    ggml_cuda_moe_grouped_decode_acquisition decode;
    CHECK(registry.prepare_decode(key, stream, &decode) == GGML_CUDA_MOE_GROUPED_DECODE_READY);
    CUDA_OK(cudaStreamSynchronize(stream));
    std::array<int32_t, 6> remapped = {};
    CUDA_OK(cudaMemcpy(remapped.data(), decode.remapped_ids, sizeof(remapped), cudaMemcpyDeviceToHost));
    CHECK((remapped == std::array<int32_t, 6>{0, 1, 1, 2, 0, 2}));
    CHECK(registry.finish_decode(decode, stream));

    ids->ne[0] = 4;
    ids->ne[1] = 4;
    ids->nb[1] = 4 * sizeof(int32_t);
    ids->nb[2] = ids->ne[1] * ids->nb[1];
    ids->nb[3] = ids->nb[2];
    key = grouped_decode_key(registry, down, ids, group.layout, group.n_banks);
    key.execution_semantic_key = 1;
    CHECK(registry.prepare_decode(key, stream, &decode) == GGML_CUDA_MOE_GROUPED_DECODE_FALLBACK);
    CUDA_OK(cudaStreamDestroy(stream));

    {
        grouped_decode_fixture k8_fixture(device, true, grouped_decode_fixture::SOURCE_BYTES, 256);
        ggml_tensor * k8_gate_up = k8_fixture.weight(GGML_TYPE_Q4_0, 32, 64);
        ggml_tensor * k8_down = k8_fixture.weight(GGML_TYPE_Q4_0, 32, 32);
        memset(k8_gate_up->data, 37, ggml_nbytes(k8_gate_up));
        memset(k8_down->data, 43, ggml_nbytes(k8_down));
        std::array<ggml_backend_moe_candidate_bank_v1, 2> k8_banks = {{
            {k8_gate_up, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, 0},
            {k8_down, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
        }};
        const ggml_backend_moe_candidate_group_v1 k8_group = {
            k8_banks.data(), k8_banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 0, 0,
        };
        const auto k8_snapshot = candidate_snapshot(138, &k8_group, 1);
        ggml_cuda_moe_grouped_context k8_registry(ggml_backend_get_device(k8_fixture.backend), device);
        CHECK(k8_registry.replace(&k8_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        ggml_tensor * k8_ids = k8_fixture.ids(8);
        auto k8_key = grouped_decode_key(k8_registry, k8_down, k8_ids, k8_group.layout, k8_group.n_banks);
        const std::array<int32_t, 8> first_routes = {255, 0, 127, 42, 7, 201, 86, 13};
        auto changed_routes = first_routes;
        changed_routes.back() = 241;
        const std::array<int32_t, 8> first_remap = {0, 1, 2, 3, 4, 5, 6, 7};
        const std::array<int32_t, 8> changed_remap = {0, 1, 2, 3, 4, 5, 6, 8};
        cudaStream_t k8_stream = nullptr;
        CUDA_OK(cudaStreamCreateWithFlags(&k8_stream, cudaStreamNonBlocking));
        const auto run_k8 = [&](const std::array<int32_t, 8> & routes, const std::array<int32_t, 8> & expected) {
            CUDA_OK(cudaMemcpyAsync(k8_ids->data, routes.data(), sizeof(routes), cudaMemcpyHostToDevice, k8_stream));
            ggml_cuda_moe_grouped_decode_acquisition k8_decode;
            CHECK(k8_registry.prepare_decode(k8_key, k8_stream, &k8_decode) == GGML_CUDA_MOE_GROUPED_DECODE_READY);
            CUDA_OK(cudaStreamSynchronize(k8_stream));
            std::array<int32_t, 8> actual = {};
            CUDA_OK(cudaMemcpy(actual.data(), k8_decode.remapped_ids, sizeof(actual), cudaMemcpyDeviceToHost));
            CHECK(actual == expected);
            CHECK(k8_registry.finish_decode(k8_decode, k8_stream));
        };
        run_k8(first_routes, first_remap);
        run_k8(first_routes, first_remap);
        run_k8(changed_routes, changed_remap);
        CUDA_OK(cudaStreamSynchronize(k8_stream));
        CUDA_OK(cudaStreamDestroy(k8_stream));
    }

    {
        grouped_decode_fixture k32_fixture(device, true, grouped_decode_fixture::SOURCE_BYTES, 64);
        ggml_tensor * k32_gate_up = k32_fixture.weight(GGML_TYPE_Q4_0, 32, 64);
        ggml_tensor * k32_down = k32_fixture.weight(GGML_TYPE_Q4_0, 32, 32);
        memset(k32_gate_up->data, 41, ggml_nbytes(k32_gate_up));
        memset(k32_down->data, 53, ggml_nbytes(k32_down));
        std::array<ggml_backend_moe_candidate_bank_v1, 2> k32_banks = {{
            {k32_gate_up, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, 0},
            {k32_down, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
        }};
        const ggml_backend_moe_candidate_group_v1 k32_group = {
            k32_banks.data(), k32_banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 0, 0,
        };
        const auto k32_snapshot = candidate_snapshot(32, &k32_group, 1);
        ggml_cuda_moe_grouped_context k32_registry(ggml_backend_get_device(k32_fixture.backend), device);
        CHECK(k32_registry.replace(&k32_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        ggml_tensor * k32_ids = k32_fixture.ids(32);
        auto k32_key = grouped_decode_key(k32_registry, k32_down, k32_ids, k32_group.layout, k32_group.n_banks);
        const std::array<int32_t, 32> first_routes = {
            31, 7, 31, 2, 7, 18, 4, 18, 9, 2, 11, 4, 15, 9, 23, 11,
            27, 15, 5, 23, 29, 27, 1, 5, 30, 29, 3, 1, 6, 30, 3, 63,
        };
        auto changed_routes = first_routes;
        changed_routes.back() = 62;
        std::array<int32_t, 64> slots;
        slots.fill(-1);
        int32_t next_slot = 0;
        const auto expected_remap = [&](const std::array<int32_t, 32> & routes) {
            std::array<int32_t, 32> result = {};
            for (uint32_t route = 0; route < routes.size(); ++route) {
                if (slots[routes[route]] < 0) {
                    slots[routes[route]] = next_slot++;
                }
                result[route] = slots[routes[route]];
            }
            return result;
        };
        const auto first_remap = expected_remap(first_routes);
        const auto changed_remap = expected_remap(changed_routes);
        cudaStream_t k32_stream = nullptr;
        CUDA_OK(cudaStreamCreateWithFlags(&k32_stream, cudaStreamNonBlocking));
        const auto run_k32 = [&](const std::array<int32_t, 32> & routes, const std::array<int32_t, 32> & expected) {
            CUDA_OK(cudaMemcpyAsync(k32_ids->data, routes.data(), sizeof(routes), cudaMemcpyHostToDevice, k32_stream));
            ggml_cuda_moe_grouped_decode_acquisition k32_decode;
            CHECK(k32_registry.prepare_decode(k32_key, k32_stream, &k32_decode) == GGML_CUDA_MOE_GROUPED_DECODE_READY);
            CUDA_OK(cudaStreamSynchronize(k32_stream));
            std::array<int32_t, 32> actual = {};
            CUDA_OK(cudaMemcpy(actual.data(), k32_decode.remapped_ids, sizeof(actual), cudaMemcpyDeviceToHost));
            CHECK(actual == expected);
            CHECK(k32_registry.finish_decode(k32_decode, k32_stream));
        };
        run_k32(first_routes, first_remap);
        run_k32(first_routes, first_remap);
        run_k32(changed_routes, changed_remap);
        CUDA_OK(cudaStreamSynchronize(k32_stream));
        CUDA_OK(cudaStreamDestroy(k32_stream));
    }

    {
        constexpr uint32_t n_experts = 320;
        constexpr uint32_t n_slots = 300;
        constexpr uint32_t top_k = 8;
        constexpr uint32_t large_rows = 33;
        constexpr uint32_t large_routes = top_k * large_rows;
        constexpr uint32_t small_rows = 16;
        constexpr uint32_t small_routes = top_k * small_rows;
        grouped_decode_fixture generic_fixture(device, true, grouped_decode_fixture::SOURCE_BYTES, n_experts);
        ggml_tensor * generic_gate_up = generic_fixture.weight(GGML_TYPE_Q4_0, 32, 64);
        ggml_tensor * generic_down = generic_fixture.weight(GGML_TYPE_Q4_0, 32, 32);
        const std::array<ggml_tensor *, 2> generic_weights = {generic_gate_up, generic_down};
        for (uint32_t bank = 0; bank < generic_weights.size(); ++bank) {
            ggml_tensor * weight = generic_weights[bank];
            for (uint32_t expert = 0; expert < n_experts; ++expert) {
                auto * row = static_cast<uint8_t *>(weight->data) + expert * weight->nb[2];
                for (size_t byte = 0; byte < weight->nb[2]; ++byte) {
                    row[byte] = static_cast<uint8_t>(17 * (bank + 1) + 13 * expert + byte + 29 * (expert >> 8));
                }
                row[0] = expert & 0xff;
                row[1] = expert >> 8;
                row[2] = bank;
            }
        }
        std::array<ggml_backend_moe_candidate_bank_v1, 2> generic_banks = {{
            {generic_gate_up, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, 0},
            {generic_down, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
        }};
        const ggml_backend_moe_candidate_group_v1 generic_group = {
            generic_banks.data(), generic_banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 0, 0,
        };
        const auto generic_snapshot = candidate_snapshot(n_slots, &generic_group, 1);
        ggml_cuda_moe_grouped_context generic_registry(ggml_backend_get_device(generic_fixture.backend), device);
        CHECK(generic_registry.replace(&generic_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        ggml_tensor * generic_ids = generic_fixture.ids(large_routes);
        const auto set_geometry = [&](uint32_t n_rows) {
            generic_ids->ne[0] = top_k;
            generic_ids->ne[1] = n_rows;
            generic_ids->nb[1] = top_k * sizeof(int32_t);
            generic_ids->nb[2] = n_rows * generic_ids->nb[1];
            generic_ids->nb[3] = generic_ids->nb[2];
        };
        set_geometry(large_rows);
        auto generic_key = grouped_decode_key(
            generic_registry, generic_down, generic_ids, generic_group.layout, generic_group.n_banks);
        cudaStream_t generic_stream = nullptr;
        CUDA_OK(cudaStreamCreateWithFlags(&generic_stream, cudaStreamNonBlocking));

        std::vector<int32_t> large(large_routes);
        std::vector<int32_t> large_expected(large_routes);
        for (uint32_t route = 0; route < large_routes; ++route) {
            large[route] = (37 * route) % n_experts;
            large_expected[route] = route;
        }
        const auto run = [&](const ggml_cuda_moe_complete_group_key & key,
                const std::vector<int32_t> & routes, const std::vector<int32_t> & expected,
                uint32_t expected_slots, const std::vector<int32_t> * expected_slot_experts = nullptr) {
            CUDA_OK(cudaMemcpyAsync(generic_ids->data, routes.data(), routes.size() * sizeof(int32_t),
                cudaMemcpyHostToDevice, generic_stream));
            ggml_cuda_moe_grouped_decode_acquisition current;
            CHECK(generic_registry.prepare_decode(key, generic_stream, &current) == GGML_CUDA_MOE_GROUPED_DECODE_READY);
            CHECK(current.n_slots == expected_slots && current.n_banks == generic_weights.size());
            CUDA_OK(cudaStreamSynchronize(generic_stream));
            std::vector<int32_t> actual(routes.size());
            CUDA_OK(cudaMemcpy(actual.data(), current.remapped_ids,
                actual.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
            CHECK(actual == expected);
            if (expected_slot_experts != nullptr) {
                CHECK(expected_slot_experts->size() == expected_slots);
                for (uint32_t bank = 0; bank < generic_weights.size(); ++bank) {
                    const ggml_tensor * weight = generic_weights[bank];
                    std::vector<uint8_t> physical(expected_slots * weight->nb[2]);
                    CUDA_OK(cudaMemcpy(physical.data(), current.banks[bank].data,
                        physical.size(), cudaMemcpyDeviceToHost));
                    for (uint32_t slot = 0; slot < expected_slots; ++slot) {
                        const int32_t expert = (*expected_slot_experts)[slot];
                        CHECK(expert >= 0 && static_cast<uint32_t>(expert) < n_experts);
                        CHECK(memcmp(physical.data() + slot * weight->nb[2],
                            static_cast<const uint8_t *>(weight->data) + expert * weight->nb[2],
                            weight->nb[2]) == 0);
                    }
                }
            }
            CHECK(generic_registry.finish_decode(current, generic_stream));
        };
        run(generic_key, large, large_expected, n_slots);

        std::vector<bool> initially_resident(n_experts, false);
        for (int32_t expert : large) {
            CHECK(!initially_resident[expert]);
            initially_resident[expert] = true;
        }
        std::vector<int32_t> updated(large.begin(), large.begin() + 208);
        for (uint32_t expert = 0; expert < n_experts; ++expert) {
            if (!initially_resident[expert]) {
                updated.push_back(expert);
            }
        }
        CHECK(updated.size() == large_routes);
        std::vector<int32_t> updated_expected(large_routes);
        std::iota(updated_expected.begin(), updated_expected.begin() + 208, 0);
        std::vector<int32_t> expected_slot_experts(n_slots, -1);
        for (uint32_t route = 0; route < large_routes; ++route) {
            expected_slot_experts[route] = large[route];
        }
        for (uint32_t miss = 0; miss < n_experts - large_routes; ++miss) {
            updated_expected[208 + miss] = miss < n_slots - large_routes ?
                large_routes + miss : 208 + miss - (n_slots - large_routes);
            expected_slot_experts[updated_expected[208 + miss]] = updated[208 + miss];
        }
        CHECK(std::all_of(expected_slot_experts.begin(), expected_slot_experts.end(),
            [](int32_t expert) { return expert >= 0; }));
        run(generic_key, updated, updated_expected, n_slots, &expected_slot_experts);

        set_geometry(small_rows);
        generic_key = grouped_decode_key(
            generic_registry, generic_down, generic_ids, generic_group.layout, generic_group.n_banks);
        std::vector<int32_t> small(small_routes);
        std::vector<int32_t> small_expected(small_routes);
        for (uint32_t route = 0; route < small_routes; ++route) {
            small[route] = large[small_routes - 1 - route];
            small_expected[route] = small_routes - 1 - route;
        }
        run(generic_key, small, small_expected, n_slots);
        CUDA_OK(cudaStreamSynchronize(generic_stream));
        for (uint32_t route = 0; route < 208; ++route) {
            CHECK(ggml_cuda_moe_grouped_context_test_access::device_slot_for_expert(
                generic_registry, generic_key.candidate, large[route]) == static_cast<int32_t>(route));
        }
        for (uint32_t miss = 0; miss < n_experts - large_routes; ++miss) {
            CHECK(ggml_cuda_moe_grouped_context_test_access::device_slot_for_expert(
                generic_registry, generic_key.candidate, updated[208 + miss]) == updated_expected[208 + miss]);
        }
        for (uint32_t route = 208; route < 228; ++route) {
            CHECK(ggml_cuda_moe_grouped_context_test_access::device_slot_for_expert(
                generic_registry, generic_key.candidate, large[route]) == -1);
        }
        for (uint32_t route = 228; route < large_routes; ++route) {
            CHECK(ggml_cuda_moe_grouped_context_test_access::device_slot_for_expert(
                generic_registry, generic_key.candidate, large[route]) == static_cast<int32_t>(route));
        }

        const auto replacement_snapshot = candidate_snapshot(340, &generic_group, 1);
        CHECK(generic_registry.replace(&replacement_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        ggml_cuda_moe_grouped_decode_acquisition stale;
        CHECK(generic_registry.prepare_decode(generic_key, generic_stream, &stale) == GGML_CUDA_MOE_GROUPED_DECODE_FALLBACK);
        const auto replacement_key = grouped_decode_key(
            generic_registry, generic_down, generic_ids, generic_group.layout, generic_group.n_banks);
        std::vector<int32_t> replacement_expected(small_routes);
        std::iota(replacement_expected.begin(), replacement_expected.end(), 0);
        run(replacement_key, small, replacement_expected, 340);
        CUDA_OK(cudaStreamSynchronize(generic_stream));
        CUDA_OK(cudaStreamDestroy(generic_stream));
    }
}

static void test_grouped_clock_refresh_case(int device, uint32_t n_slots) {
    grouped_clock_fixture fixture(device, n_slots);
    const std::array<int32_t, 4> routes = {0, 1, 0, 2};
    const auto initial = fixture.warm(routes);
    CHECK(ggml_cuda_moe_grouped_context_test_access::set_clock_bound(
        *fixture.registry, initial.transaction.acquisition, UINT64_MAX - routes.size()));

    ggml_cuda_moe_grouped_decode_acquisition boundary;
    CHECK(fixture.prepare(routes, boundary, fixture.stream) == GGML_CUDA_MOE_GROUPED_DECODE_READY);
    CHECK(boundary.transaction.acquisition.resource_generation == initial.transaction.acquisition.resource_generation);
    CHECK(fixture.registry->finish_decode(boundary, fixture.stream));
    CUDA_OK(cudaStreamSynchronize(fixture.stream));

    ggml_cuda_moe_grouped_decode_acquisition refreshed;
    CHECK(fixture.prepare(routes, refreshed, fixture.stream) == GGML_CUDA_MOE_GROUPED_DECODE_READY);
    CHECK(!fixture.registry->get_group_resources(initial.transaction.acquisition, nullptr));
    ggml_cuda_moe_grouped_resource_info info;
    CHECK(fixture.registry->get_group_resources(refreshed.transaction.acquisition, &info) && info.n_slots == n_slots);
    CHECK(refreshed.transaction.acquisition.resource_generation > initial.transaction.acquisition.resource_generation);
    CHECK(fixture.registry->finish_decode(refreshed, fixture.stream));
    CUDA_OK(cudaStreamSynchronize(fixture.stream));
    std::array<int32_t, 4> remapped = {};
    CUDA_OK(cudaMemcpy(remapped.data(), refreshed.remapped_ids, sizeof(remapped), cudaMemcpyDeviceToHost));
    CHECK((remapped == std::array<int32_t, 4>{0, 1, 0, 2}));
}

static void test_grouped_clock_failed_rebuild(int device) {
    grouped_clock_fixture fixture(device, 12);
    const std::array<int32_t, 4> routes = {0, 1, 2, 3};
    const auto initial = fixture.warm(routes);
    CHECK(ggml_cuda_moe_grouped_context_test_access::set_clock_bound(
        *fixture.registry, initial.transaction.acquisition, UINT64_MAX));

    ggml_backend_buffer_t saved_buffer = fixture.weights[0]->buffer;
    fixture.weights[0]->buffer = nullptr;
    ggml_cuda_moe_grouped_decode_acquisition failed;
    CHECK(fixture.prepare(routes, failed, fixture.stream) == GGML_CUDA_MOE_GROUPED_DECODE_FALLBACK);
    fixture.weights[0]->buffer = saved_buffer;
    CHECK(!fixture.registry->get_group_resources(initial.transaction.acquisition, nullptr));

    const auto recovered = fixture.warm(routes);
    CHECK(recovered.transaction.acquisition.resource_generation > initial.transaction.acquisition.resource_generation);
}

static bool wait_for_grouped_detach(
        ggml_cuda_moe_grouped_context & registry,
        const ggml_cuda_moe_grouped_acquisition & acquisition) {
    for (uint32_t attempt = 0; attempt < 1000; ++attempt) {
        if (!registry.get_group_resources(acquisition, nullptr)) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

static void test_grouped_clock_replacement(int device) {
    grouped_clock_fixture fixture(device, 12);
    const std::array<int32_t, 4> routes = {0, 1, 2, 3};
    fixture.warm(routes);

    host_barrier barrier;
    CUDA_OK(cudaLaunchHostFunc(fixture.stream, wait_on_host_barrier, &barrier));
    ggml_cuda_moe_grouped_decode_acquisition pending;
    CHECK(fixture.prepare(routes, pending, fixture.stream) == GGML_CUDA_MOE_GROUPED_DECODE_READY);
    CHECK(fixture.registry->finish_decode(pending, fixture.stream));
    while (!barrier.entered.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    CHECK(ggml_cuda_moe_grouped_context_test_access::set_clock_bound(
        *fixture.registry, pending.transaction.acquisition, UINT64_MAX));

    cudaStream_t maintenance_stream = nullptr;
    CUDA_OK(cudaStreamCreateWithFlags(&maintenance_stream, cudaStreamNonBlocking));
    std::atomic<uint32_t> maintenance_result{GGML_CUDA_MOE_GROUPED_DECODE_READY};
    std::thread maintenance([&]() {
        ggml_cuda_moe_grouped_decode_acquisition decode;
        maintenance_result.store(fixture.registry->prepare_decode(fixture.key, maintenance_stream, &decode), std::memory_order_release);
    });
    const bool detached = wait_for_grouped_detach(*fixture.registry, pending.transaction.acquisition);

    const auto replacement = candidate_snapshot(48, &fixture.group, 1);
    std::atomic<bool> replacement_done{false};
    std::atomic<int32_t> replacement_result{GGML_BACKEND_MOE_CANDIDATE_REPLACE_ERROR};
    std::thread replace_thread([&]() {
        replacement_result.store(fixture.registry->replace(&replacement), std::memory_order_release);
        replacement_done.store(true, std::memory_order_release);
    });
    for (uint32_t attempt = 0; attempt < 100 && !replacement_done.load(std::memory_order_acquire); ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const bool replacement_waited = !replacement_done.load(std::memory_order_acquire);
    barrier.released.store(true, std::memory_order_release);
    maintenance.join();
    replace_thread.join();
    CUDA_OK(cudaStreamSynchronize(fixture.stream));
    CHECK(detached && replacement_waited && replacement_result.load(std::memory_order_acquire) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(maintenance_result.load(std::memory_order_acquire) == GGML_CUDA_MOE_GROUPED_DECODE_FALLBACK);
    CHECK(fixture.registry->state().generation == 2 && fixture.registry->state().n_slots == 48);

    fixture.key = grouped_decode_key(*fixture.registry, fixture.weights[1], fixture.ids, fixture.group.layout, fixture.group.n_banks);
    const auto replaced = fixture.warm(routes);
    CHECK(replaced.n_slots == 48 && replaced.transaction.acquisition.candidate.generation == 2);
    CUDA_OK(cudaStreamDestroy(maintenance_stream));
}

static void test_grouped_clock_teardown(int device) {
    grouped_clock_fixture fixture(device, 120);
    const std::array<int32_t, 4> routes = {0, 1, 2, 3};
    fixture.warm(routes);

    host_barrier barrier;
    CUDA_OK(cudaLaunchHostFunc(fixture.stream, wait_on_host_barrier, &barrier));
    ggml_cuda_moe_grouped_decode_acquisition pending;
    CHECK(fixture.prepare(routes, pending, fixture.stream) == GGML_CUDA_MOE_GROUPED_DECODE_READY);
    CHECK(fixture.registry->finish_decode(pending, fixture.stream));
    while (!barrier.entered.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    CHECK(ggml_cuda_moe_grouped_context_test_access::set_clock_bound(
        *fixture.registry, pending.transaction.acquisition, UINT64_MAX));

    cudaStream_t maintenance_stream = nullptr;
    CUDA_OK(cudaStreamCreateWithFlags(&maintenance_stream, cudaStreamNonBlocking));
    std::atomic<uint32_t> maintenance_result{GGML_CUDA_MOE_GROUPED_DECODE_READY};
    std::thread maintenance([&]() {
        ggml_cuda_moe_grouped_decode_acquisition decode;
        maintenance_result.store(fixture.registry->prepare_decode(fixture.key, maintenance_stream, &decode), std::memory_order_release);
    });
    const bool detached = wait_for_grouped_detach(*fixture.registry, pending.transaction.acquisition);

    std::array<std::atomic<bool>, 2> shutdown_done = {};
    std::thread shutdown_thread([&]() {
        fixture.registry->shutdown();
        shutdown_done[0].store(true, std::memory_order_release);
    });
    std::thread second_shutdown_thread([&]() {
        fixture.registry->shutdown();
        shutdown_done[1].store(true, std::memory_order_release);
    });
    for (uint32_t attempt = 0; attempt < 100 &&
            !shutdown_done[0].load(std::memory_order_acquire) && !shutdown_done[1].load(std::memory_order_acquire); ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const bool shutdown_waited = !shutdown_done[0].load(std::memory_order_acquire) && !shutdown_done[1].load(std::memory_order_acquire);
    barrier.released.store(true, std::memory_order_release);
    shutdown_thread.join();
    second_shutdown_thread.join();
    CHECK(detached && shutdown_waited);
    CHECK(shutdown_done[0].load(std::memory_order_acquire) && shutdown_done[1].load(std::memory_order_acquire));
    CHECK(fixture.registry->prepare_decode(fixture.key, maintenance_stream, &pending) == GGML_CUDA_MOE_GROUPED_DECODE_FALLBACK);
    fixture.registry.reset();
    maintenance.join();
    CUDA_OK(cudaStreamSynchronize(fixture.stream));
    CHECK(maintenance_result.load(std::memory_order_acquire) == GGML_CUDA_MOE_GROUPED_DECODE_FALLBACK);
    CUDA_OK(cudaStreamDestroy(maintenance_stream));
}

static void test_grouped_clock_maintenance(int device) {
    for (uint32_t n_slots : {12u, 48u, 120u}) {
        test_grouped_clock_refresh_case(device, n_slots);
    }
    test_grouped_clock_failed_rebuild(device);
    test_grouped_clock_replacement(device);
    test_grouped_clock_teardown(device);
    fprintf(stderr, "test-moe-cache: grouped clock maintenance OK\n");
}

static void test_grouped_decode(int device) {
    test_grouped_decode_independent_rows(device);
    test_grouped_decode_type(device, GGML_TYPE_Q4_0, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE);
    test_grouped_decode_type(device, GGML_TYPE_Q4_0, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP);
    test_grouped_decode_type(device, GGML_TYPE_Q4_K, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE);
    test_grouped_decode_type(device, GGML_TYPE_Q4_K, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP);
    test_grouped_decode_type(device, GGML_TYPE_Q4_0, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, true, 12);
    test_grouped_decode_type(device, GGML_TYPE_Q4_0, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, false);
    for (ggml_type type : {GGML_TYPE_BF16, GGML_TYPE_NVFP4}) {
        for (uint32_t layout : {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP}) {
            for (uint32_t n_slots : {12u, 48u}) {
                test_grouped_decode_type(device, type, layout, true, n_slots);
            }
        }
    }
    test_grouped_decode_type(device, GGML_TYPE_NVFP4, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, true, 12, true);
    test_grouped_clock_maintenance(device);
    fprintf(stderr, "test-moe-cache: grouped decode resources OK\n");
}

struct grouped_decode_bench_spec {
    const char * name;
    ggml_type type;
    uint32_t layout;
    uint32_t n_experts;
    uint32_t high_slots;
    uint32_t n_banks;
    int64_t ne0[3];
    int64_t ne1[3];
    uint32_t roles[3];
};

struct grouped_decode_sample {
    float total_us;
};

struct grouped_decode_timer {
    grouped_decode_timer() {
        CUDA_OK(cudaEventCreate(&begin));
        CUDA_OK(cudaEventCreate(&end));
    }

    ~grouped_decode_timer() {
        CUDA_OK(cudaEventDestroy(end));
        CUDA_OK(cudaEventDestroy(begin));
    }

    grouped_decode_sample sample() const {
        grouped_decode_sample result = {};
        CUDA_OK(cudaEventElapsedTime(&result.total_us, begin, end));
        result.total_us *= 1000.0f;
        return result;
    }

    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
};

static grouped_decode_sample grouped_decode_median(const std::vector<grouped_decode_sample> & samples) {
    std::vector<float> values;
    values.reserve(samples.size());
    for (const auto & sample : samples) {
        values.push_back(sample.total_us);
    }
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    return {values.size() % 2 == 0 ? (values[middle - 1] + values[middle]) / 2.0f : values[middle]};
}

static grouped_decode_sample grouped_decode_timed(
        ggml_cuda_moe_grouped_context & registry,
        const ggml_cuda_moe_complete_group_key & key,
        ggml_tensor * ids,
        cudaStream_t stream,
        const std::array<int32_t, 8> & routes,
        grouped_decode_timer & timer) {
    CUDA_OK(cudaMemcpyAsync(ids->data, routes.data(), sizeof(routes), cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaEventRecord(timer.begin, stream));
    ggml_cuda_moe_grouped_decode_acquisition decode;
    CHECK(registry.prepare_decode(key, stream, &decode) == GGML_CUDA_MOE_GROUPED_DECODE_READY);
    CHECK(registry.finish_decode(decode, stream));
    CUDA_OK(cudaEventRecord(timer.end, stream));
    CUDA_OK(cudaStreamSynchronize(stream));
    return timer.sample();
}

static void grouped_decode_submit(
        ggml_cuda_moe_grouped_context & registry,
        const ggml_cuda_moe_complete_group_key & key,
        ggml_tensor * ids,
        cudaStream_t stream,
        const std::array<int32_t, 8> & routes) {
    CUDA_OK(cudaMemcpyAsync(ids->data, routes.data(), sizeof(routes), cudaMemcpyHostToDevice, stream));
    ggml_cuda_moe_grouped_decode_acquisition decode;
    CHECK(registry.prepare_decode(key, stream, &decode) == GGML_CUDA_MOE_GROUPED_DECODE_READY);
    CHECK(registry.finish_decode(decode, stream));
}

static void grouped_decode_print_sample(const char * phase, const grouped_decode_sample & sample, size_t payload_bytes) {
    const double rate = payload_bytes == 0 ? 0.0 : payload_bytes / (sample.total_us * 1e-6) / (1024.0 * 1024.0 * 1024.0);
    fprintf(stderr, "  %-7s total=%8.3f us payload=%9zu B rate=%6.2f GiB/s\n", phase, sample.total_us, payload_bytes, rate);
}

static size_t grouped_decode_source_bytes(const grouped_decode_bench_spec & spec) {
    size_t result = 256;
    for (uint32_t bank = 0; bank < spec.n_banks; ++bank) {
        result += ggml_row_size(spec.type, spec.ne0[bank]) * spec.ne1[bank] * spec.n_experts + 256;
    }
    return result;
}

static void grouped_decode_benchmark_case(int device, const grouped_decode_bench_spec & spec, uint32_t n_slots) {
    grouped_decode_fixture fixture(device, true, grouped_decode_source_bytes(spec), spec.n_experts);
    std::array<ggml_tensor *, 3> weights = {};
    std::array<ggml_backend_moe_candidate_bank_v1, 3> banks = {};
    size_t expert_bytes = 0;
    for (uint32_t bank = 0; bank < spec.n_banks; ++bank) {
        weights[bank] = fixture.weight(spec.type, spec.ne0[bank], spec.ne1[bank]);
        memset(weights[bank]->data, 17 + bank, ggml_nbytes(weights[bank]));
        banks[bank].tensor = weights[bank];
        banks[bank].role = spec.roles[bank];
        expert_bytes += weights[bank]->nb[2];
    }
    ggml_backend_moe_candidate_group_v1 group = {};
    group.banks = banks.data();
    group.n_banks = spec.n_banks;
    group.layout = spec.layout;
    const auto snapshot = candidate_snapshot(n_slots, &group, 1);
    ggml_cuda_moe_grouped_context registry(ggml_backend_get_device(fixture.backend), device);
    CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    ggml_tensor * ids = fixture.ids(8);
    auto key = grouped_decode_key(registry, weights[spec.n_banks - 1], ids, spec.layout, spec.n_banks);
    cudaStream_t stream = nullptr;
    CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    grouped_decode_timer timer;

    std::array<int32_t, 8> routes = {};
    for (uint32_t route = 0; route < routes.size(); ++route) {
        routes[route] = route;
    }
    const auto startup = grouped_decode_timed(registry, key, ids, stream, routes, timer);
    for (uint32_t base = 0; base < spec.n_experts; base += routes.size()) {
        for (uint32_t route = 0; route < routes.size(); ++route) {
            routes[route] = std::min(base + route, spec.n_experts - 1);
        }
        grouped_decode_submit(registry, key, ids, stream, routes);
    }
    CUDA_OK(cudaStreamSynchronize(stream));
    CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    key = grouped_decode_key(registry, weights[spec.n_banks - 1], ids, spec.layout, spec.n_banks);
    for (uint32_t route = 0; route < routes.size(); ++route) {
        routes[route] = route;
    }
    const auto cold = grouped_decode_timed(registry, key, ids, stream, routes, timer);

    std::vector<grouped_decode_sample> hits;
    hits.reserve(100);
    for (uint32_t iteration = 0; iteration < 100; ++iteration) {
        hits.push_back(grouped_decode_timed(registry, key, ids, stream, routes, timer));
    }

    std::array<grouped_decode_sample, 4> warm = {};
    const uint32_t miss_counts[] = {1, 2, 4, 8};
    for (int miss_case = 3; miss_case >= 0; --miss_case) {
        std::vector<grouped_decode_sample> misses;
        misses.reserve(20);
        for (uint32_t iteration = 0; iteration < 20; ++iteration) {
            for (uint32_t base = 0; base < n_slots; base += routes.size()) {
                for (uint32_t route = 0; route < routes.size(); ++route) {
                    routes[route] = std::min(base + route, n_slots - 1);
                }
                grouped_decode_submit(registry, key, ids, stream, routes);
            }
            const uint32_t n_misses = miss_counts[miss_case];
            const uint32_t n_hits = routes.size() - n_misses;
            const uint32_t target_span = spec.n_experts - n_slots - n_misses + 1;
            const uint32_t target_base = n_slots + (17 * iteration) % target_span;
            for (uint32_t route = 0; route < routes.size(); ++route) {
                routes[route] = route < n_hits ? route : target_base + route - n_hits;
            }
            misses.push_back(grouped_decode_timed(registry, key, ids, stream, routes, timer));
        }
        warm[miss_case] = grouped_decode_median(misses);
    }

    const auto hit = grouped_decode_median(hits);
    fprintf(stderr, "grouped-bench: model=%s slots=%u experts=%u banks=%u rows=", spec.name, n_slots, spec.n_experts, spec.n_banks);
    for (uint32_t bank = 0; bank < spec.n_banks; ++bank) {
        fprintf(stderr, "%s%zu", bank == 0 ? "" : "/", weights[bank]->nb[1]);
    }
    fprintf(stderr, " B experts=");
    for (uint32_t bank = 0; bank < spec.n_banks; ++bank) {
        fprintf(stderr, "%s%zu", bank == 0 ? "" : "/", weights[bank]->nb[2]);
    }
    fprintf(stderr, " B\n");
    grouped_decode_print_sample("startup", startup, 8 * expert_bytes);
    grouped_decode_print_sample("cold", cold, 8 * expert_bytes);
    grouped_decode_print_sample("hit", hit, 0);
    for (uint32_t miss_case = 0; miss_case < 4; ++miss_case) {
        char label[8];
        snprintf(label, sizeof(label), "miss%u", miss_counts[miss_case]);
        grouped_decode_print_sample(label, warm[miss_case], miss_counts[miss_case] * expert_bytes);
    }
    CUDA_OK(cudaStreamDestroy(stream));
}

static void test_grouped_decode_benchmark(int device) {
    const grouped_decode_bench_spec gemma = {
        "gemma4-q4_0-fused", GGML_TYPE_Q4_0, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 128, 120, 2,
        {2816, 704, 0}, {1408, 2816, 0},
        {GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
    };
    const grouped_decode_bench_spec qwen = {
        "qwen3.6-q4_k-separate", GGML_TYPE_Q4_K, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 256, 188, 3,
        {2048, 2048, 512}, {512, 512, 2048},
        {GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT,
            GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT},
    };
    for (uint32_t n_slots : {12u, 48u, gemma.high_slots}) {
        grouped_decode_benchmark_case(device, gemma, n_slots);
    }
    for (uint32_t n_slots : {12u, 48u, qwen.high_slots}) {
        grouped_decode_benchmark_case(device, qwen, n_slots);
    }
}

int main(int argc, char ** argv) {
    const bool registry_only = argc == 2 && strcmp(argv[1], "--registry-only") == 0;
    const bool registry_bench = argc == 2 && strcmp(argv[1], "--registry-bench") == 0;
    const bool cached_fusion_only = argc == 2 && strcmp(argv[1], "--cached-fusion-only") == 0;
    const bool grouped_bench = argc == 2 && strcmp(argv[1], "--grouped-bench") == 0;
    const bool grouped_decode_only = argc == 2 && strcmp(argv[1], "--grouped-decode-only") == 0;
    const bool grouped_replay_only = argc == 2 && strcmp(argv[1], "--grouped-replay-only") == 0;
    const bool grouped_multirow_only = argc == 2 && strcmp(argv[1], "--grouped-multirow-only") == 0;
    const bool grouped_nvfp4_only = argc == 2 && strcmp(argv[1], "--grouped-nvfp4-only") == 0;
    const bool legacy_phase_telemetry_only = argc == 2 && strcmp(argv[1], "--legacy-phase-telemetry-only") == 0;
    const bool gemma_q4_parity_only = argc == 2 && strcmp(argv[1], "--gemma-q4-parity-only") == 0;
    const bool prefill_resident_only = argc == 2 && strcmp(argv[1], "--prefill-resident-only") == 0;
    if (prefill_resident_only) {
        test_prefill_resident_biases();
        return 0;
    }
    if (gemma_q4_parity_only) {
        int dev = 0;
        CUDA_OK(cudaGetDevice(&dev));
        test_gemma_q4_cached_cuda_parity(dev);
        return 0;
    }
    if (grouped_multirow_only) {
        int dev = 0;
        CUDA_OK(cudaGetDevice(&dev));
        test_active_grouped_multirow_graph_modes(dev);
        return 0;
    }
    if (grouped_decode_only) {
        int dev = 0;
        CUDA_OK(cudaGetDevice(&dev));
        test_grouped_decode(dev);
        return 0;
    }
    if (grouped_replay_only) {
        int dev = 0;
        CUDA_OK(cudaGetDevice(&dev));
        test_grouped_graph_replay_lifecycle(dev);
        return 0;
    }
    if (grouped_nvfp4_only) {
        test_active_grouped_nvfp4_scales();
        return 0;
    }
    if (legacy_phase_telemetry_only) {
        int dev = 0;
        CUDA_OK(cudaGetDevice(&dev));
        test_active_grouped_legacy_phase_telemetry(dev);
        return 0;
    }
    test_candidate_graph_coverage_ledger();
    test_candidate_graph_inventory_reuse();
    test_mmid_capabilities();
    test_scheduler_execution_certificate();
    test_graph_execution_certificate_policy();
    test_candidate_generic_physical_truth();
    test_candidate_producer();
    test_candidate_registry(registry_bench);
    test_legacy_owner_leases();
    test_grouped_context_resources();
    test_grouped_graph_preflight(registry_bench);
    test_grouped_graph_mixed_phase();
    if (registry_only || registry_bench) {
        return 0;
    }

    int dev = 0;
    CUDA_OK(cudaGetDevice(&dev));
    if (grouped_bench) {
        test_grouped_decode(dev);
        test_grouped_decode_benchmark(dev);
        return 0;
    }

    test_cached_mmid_fusion_decline();
    test_cached_mmid_prefill_and_overflow();
    test_owner_legacy_cache(dev);
    if (cached_fusion_only) {
        test_cached_mmid_routed_separate_chain();
        return 0;
    }
    test_grouped_decode(dev);
    test_grouped_graph_replay_lifecycle(dev);
    test_active_grouped_multirow_graph_modes(dev);
    test_active_grouped_dispatch();

    // Toy parameters. Small enough to run in a few ms on any CUDA device,
    // large enough that LRU has work to do.
    constexpr int    N_EXPERTS = 64;
    constexpr int    N_SLOTS   = 16;
    constexpr size_t SLOT_BYTES = 1024;       // 256 floats
    constexpr size_t SOURCE_PADDING = 64;
    constexpr int    N_FLOATS  = SLOT_BYTES / sizeof(float);
    constexpr int    N_ACCESS  = 4000;
    constexpr double ZIPF_S    = 1.1;          // mild skew
    constexpr unsigned SEED    = 0xC0FFEE;

    fprintf(stderr, "test-moe-cache: device=%d  experts=%d  slots=%d  slot=%zuB  ops=%d\n",
            dev, N_EXPERTS, N_SLOTS, SLOT_BYTES, N_ACCESS);

    // Pinned host source: expert i is filled with float value (float)i in every cell.
    float * host_experts = nullptr;
    CUDA_OK(cudaMallocHost(&host_experts, (size_t)N_EXPERTS * SLOT_BYTES));
    for (int e = 0; e < N_EXPERTS; ++e) {
        for (int j = 0; j < N_FLOATS; ++j) {
            host_experts[e * N_FLOATS + j] = (float)e;
        }
    }

    cudaStream_t copy_stream = nullptr;
    CUDA_OK(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));

    auto * cache = ggml_cuda_moe_cache_init(dev, SLOT_BYTES, N_SLOTS, false, 0, 0);
    CHECK(cache != nullptr);

    std::mt19937 rng(SEED);
    std::vector<int> trace; trace.reserve(N_ACCESS);

    for (int t = 0; t < N_ACCESS; ++t) {
        int eid = sample_zipf(rng, N_EXPERTS, ZIPF_S);
        const void * src = host_experts + (size_t)eid * N_FLOATS;

        int slot = ggml_cuda_moe_cache_acquire(cache, src, SLOT_BYTES, copy_stream, false, false, false, false);
        CHECK(slot >= 0 && slot < N_SLOTS);
        trace.push_back(eid);
    }

    // All copies must be done before we read slabs back.
    CUDA_OK(cudaStreamSynchronize(copy_stream));

    // --- Snapshot stats BEFORE the verification sweep so the workload-phase
    // numbers aren't contaminated by sweep acquires. ---
    uint64_t hits = 0, misses = 0, evictions = 0;
    ggml_cuda_moe_cache_stats(cache, &hits, &misses, &evictions);
    CHECK(hits + misses == (uint64_t)N_ACCESS);
    double hit_rate = (double)hits / (double)N_ACCESS;

    fprintf(stderr, "  workload hits=%llu  misses=%llu  hit-rate=%.2f%%  evictions=%llu\n",
            (unsigned long long)hits,
            (unsigned long long)misses,
            100.0 * hit_rate,
            (unsigned long long)evictions);

    // Zipf(1.1) over 64 experts with 16 slots should easily clear 40% hit rate.
    CHECK(hit_rate > 0.40);

    // --- Verification sweep: re-acquire every expert and confirm slot
    // contents are bit-exact. Sweep order = expert id ascending. The sweep
    // mutates LRU but that's fine; we just want to walk every expert once.
    // Re-acquire forces a fresh H2D copy on miss, which is correct because
    // we're testing both hit and miss paths return correct data. ---
    std::vector<float> readback(N_FLOATS);
    int verified = 0;
    for (int eid = 0; eid < N_EXPERTS; ++eid) {
        const float * src = host_experts + (size_t)eid * N_FLOATS;
        int slot = ggml_cuda_moe_cache_acquire(cache, src, SLOT_BYTES, copy_stream, false, false, false, false);
        CHECK(slot >= 0 && slot < N_SLOTS);
        CUDA_OK(cudaStreamSynchronize(copy_stream));

        void * d = ggml_cuda_moe_cache_slot_ptr(cache, slot);
        CHECK(d != nullptr);

        CUDA_OK(cudaMemcpy(readback.data(), d, SLOT_BYTES, cudaMemcpyDeviceToHost));
        for (int j = 0; j < N_FLOATS; ++j) {
            CHECK(readback[j] == (float)eid);
        }
        verified++;
    }
    CHECK(verified == N_EXPERTS);

    // Final stats sanity: hits + misses across both phases must match acquires.
    ggml_cuda_moe_cache_stats(cache, &hits, &misses, &evictions);
    CHECK(hits + misses == (uint64_t)(N_ACCESS + N_EXPERTS));

    ggml_cuda_moe_cache_free(cache);

    auto * batch_cache = ggml_cuda_moe_cache_init(dev, SLOT_BYTES, 4, false, 0, 0);
    CHECK(batch_cache != nullptr);
    int batch_slots[4];
    for (int eid = 0; eid < 4; ++eid) {
        batch_slots[eid] = ggml_cuda_moe_cache_acquire(
            batch_cache, host_experts + (size_t) eid * N_FLOATS,
            SLOT_BYTES, copy_stream, false, true, false, true);
        CHECK(batch_slots[eid] >= 0);
    }
    CHECK(ggml_cuda_moe_cache_acquire(
        batch_cache, host_experts + 4 * N_FLOATS,
        SLOT_BYTES, copy_stream, false, true, false, false) < 0);
    CHECK(!ggml_cuda_moe_cache_grow_pool(batch_cache, 2 * SLOT_BYTES));
    CUDA_OK(cudaStreamSynchronize(copy_stream));
    for (int eid = 0; eid < 4; ++eid) {
        void * d = ggml_cuda_moe_cache_slot_ptr(batch_cache, batch_slots[eid]);
        CUDA_OK(cudaMemcpy(readback.data(), d, SLOT_BYTES, cudaMemcpyDeviceToHost));
        for (int j = 0; j < N_FLOATS; ++j) {
            CHECK(readback[j] == (float) eid);
        }
    }
    ggml_cuda_moe_cache_release_slots(batch_cache, batch_slots, 4);
    CHECK(ggml_cuda_moe_cache_grow_pool(batch_cache, 2 * SLOT_BYTES));
    ggml_cuda_moe_cache_free(batch_cache);

    auto * staging_cache = ggml_cuda_moe_cache_init(dev, SLOT_BYTES, 2, false, 0, 0);
    CHECK(staging_cache != nullptr);
    cudaStream_t staging_copy_stream = ggml_cuda_moe_cache_copy_stream(staging_cache);
    CHECK(staging_copy_stream != nullptr);

    const void * resident_src_0 = host_experts;
    const void * resident_src_1 = host_experts + N_FLOATS;
    CHECK(ggml_cuda_moe_cache_acquire(
        staging_cache, resident_src_0, SLOT_BYTES, staging_copy_stream, false, false, false, false) == 0);
    CHECK(ggml_cuda_moe_cache_acquire(
        staging_cache, resident_src_1, SLOT_BYTES, staging_copy_stream, false, false, false, false) == 1);
    CUDA_OK(cudaStreamSynchronize(staging_copy_stream));

    cudaStream_t compute_stream = nullptr;
    CUDA_OK(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));
    void * staging_dst = nullptr;
    CUDA_OK(cudaMalloc(&staging_dst, 4 * SLOT_BYTES + SOURCE_PADDING));
    const void * staging_srcs[] = {
        resident_src_0,
        resident_src_1,
        host_experts + 2 * N_FLOATS,
        host_experts + 3 * N_FLOATS,
    };
    std::vector<float> staging_readback(4 * N_FLOATS);

    for (int repeat = 0; repeat < 2; ++repeat) {
        float nonresident_value_0 = 100.0f + repeat;
        float nonresident_value_1 = 200.0f + repeat;
        std::fill_n(host_experts + 2 * N_FLOATS, N_FLOATS, nonresident_value_0);
        std::fill_n(host_experts + 3 * N_FLOATS, N_FLOATS, nonresident_value_1);

        host_barrier barrier;
        CUDA_OK(cudaLaunchHostFunc(compute_stream, wait_on_host_barrier, &barrier));
        while (!barrier.entered.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }

        CHECK(ggml_cuda_moe_cache_copy_to_staging(
            staging_cache, staging_srcs, 4, SLOT_BYTES, staging_dst, compute_stream));
        cudaError_t staging_copy_status = cudaErrorNotReady;
        for (int attempt = 0; attempt < 100 && staging_copy_status == cudaErrorNotReady; ++attempt) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            staging_copy_status = cudaStreamQuery(staging_copy_stream);
        }
        CHECK(staging_copy_status == cudaErrorNotReady);

        barrier.released.store(true, std::memory_order_release);
        CUDA_OK(cudaStreamSynchronize(compute_stream));
        CUDA_OK(cudaStreamSynchronize(staging_copy_stream));
        CUDA_OK(cudaMemcpy(staging_readback.data(), staging_dst, 4 * SLOT_BYTES, cudaMemcpyDeviceToHost));
        for (int j = 0; j < N_FLOATS; ++j) {
            CHECK(staging_readback[j] == 0.0f);
            CHECK(staging_readback[N_FLOATS + j] == 1.0f);
            CHECK(staging_readback[2 * N_FLOATS + j] == nonresident_value_0);
            CHECK(staging_readback[3 * N_FLOATS + j] == nonresident_value_1);
        }
    }

    CHECK(!ggml_cuda_moe_cache_copy_to_staging(
        staging_cache, staging_srcs, 4, SLOT_BYTES + 1, staging_dst, compute_stream));
    CHECK(cudaStreamQuery(staging_copy_stream) == cudaSuccess);
    CHECK(cudaStreamQuery(compute_stream) == cudaSuccess);

    CUDA_OK(cudaFree(staging_dst));
    CUDA_OK(cudaStreamDestroy(compute_stream));
    ggml_cuda_moe_cache_free(staging_cache);

    auto * split_cache = ggml_cuda_moe_cache_init(dev, SLOT_BYTES, 2, false, 0, 0);
    CHECK(split_cache != nullptr);
    cudaStream_t split_copy_stream = ggml_cuda_moe_cache_copy_stream(split_cache);
    CHECK(split_copy_stream != nullptr);
    CUDA_OK(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));
    CUDA_OK(cudaMalloc(&staging_dst, 4 * SLOT_BYTES + SOURCE_PADDING));
    std::vector<float> split_readback(6 * N_FLOATS);

    constexpr size_t SMALL_STRIDE = SOURCE_PADDING / 2;
    auto * padding_decline_cache = ggml_cuda_moe_cache_init(dev, SMALL_STRIDE, 2, false, 0, 0);
    CHECK(padding_decline_cache != nullptr);
    const void * padding_decline_srcs[] = {
        (const char *) host_experts + 48 * SMALL_STRIDE,
        (const char *) host_experts + 49 * SMALL_STRIDE,
        (const char *) host_experts + 50 * SMALL_STRIDE,
        (const char *) host_experts + 51 * SMALL_STRIDE,
    };
    int padding_decline_slots[4] = {-7, -7, -7, -7};
    int padding_decline_resident = -7;
    int padding_decline_wait_classes = -7;
    CHECK(!ggml_cuda_moe_cache_prepare_split_staging(
        padding_decline_cache, padding_decline_srcs, 4, SMALL_STRIDE, SOURCE_PADDING, 1,
        padding_decline_slots, nullptr, &padding_decline_resident, staging_dst, nullptr, 0,
        &padding_decline_wait_classes, compute_stream));
    CHECK(std::all_of(std::begin(padding_decline_slots), std::end(padding_decline_slots), [](int slot) { return slot == -7; }));
    CHECK(padding_decline_resident == -7 && padding_decline_wait_classes == -7);
    uint64_t padding_decline_hits = 0;
    uint64_t padding_decline_misses = 0;
    uint64_t padding_decline_evictions = 0;
    ggml_cuda_moe_cache_stats(
        padding_decline_cache, &padding_decline_hits, &padding_decline_misses, &padding_decline_evictions);
    CHECK(padding_decline_hits == 0 && padding_decline_misses == 0 && padding_decline_evictions == 0);
    CHECK(cudaStreamQuery(ggml_cuda_moe_cache_copy_stream(padding_decline_cache)) == cudaSuccess);
    CHECK(cudaStreamQuery(compute_stream) == cudaSuccess);
    ggml_cuda_moe_cache_free(padding_decline_cache);
    fprintf(stderr, "test-moe-cache: split padding decline OK\n");

    for (int repeat = 0; repeat < 2; ++repeat) {
        const int first = 2 * repeat;
        const void * split_srcs[] = {
            host_experts + (size_t)(first + 0) * N_FLOATS,
            host_experts + (size_t)(first + 1) * N_FLOATS,
            host_experts + (size_t)(first + 2) * N_FLOATS,
            host_experts + (size_t)(first + 3) * N_FLOATS,
        };
        for (int e = first; e < first + 4; ++e) {
            std::fill_n(host_experts + (size_t)e * N_FLOATS, N_FLOATS, (float)e);
        }

        host_barrier barrier;
        CUDA_OK(cudaLaunchHostFunc(compute_stream, wait_on_host_barrier, &barrier));
        while (!barrier.entered.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        CUDA_OK(cudaMemsetAsync(staging_dst, 0xff, 4 * SLOT_BYTES + SOURCE_PADDING, compute_stream));

        int slot_ids[4] = {-1, -1, -1, -1};
        int n_resident = 0;
        int n_wait_classes = 0;
        CHECK(ggml_cuda_moe_cache_prepare_split_staging(
            split_cache, split_srcs, 4, SLOT_BYTES, SOURCE_PADDING, 1, slot_ids, nullptr,
            &n_resident, staging_dst, nullptr, 0, &n_wait_classes, compute_stream));
        CHECK(n_resident == 2);
        CHECK(n_wait_classes == 1);
        CHECK(slot_ids[0] >= 0 && slot_ids[1] >= 0);
        CHECK(slot_ids[0] != slot_ids[1]);
        CHECK(slot_ids[2] == -1 && slot_ids[3] == -1);
        CHECK(cudaStreamQuery(split_copy_stream) == cudaErrorNotReady);

        barrier.released.store(true, std::memory_order_release);
        CUDA_OK(cudaStreamSynchronize(compute_stream));
        CUDA_OK(cudaMemcpy(
            split_readback.data(), ggml_cuda_moe_cache_slot_ptr(split_cache, slot_ids[0]), SLOT_BYTES,
            cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(
            split_readback.data() + N_FLOATS, ggml_cuda_moe_cache_slot_ptr(split_cache, slot_ids[1]), SLOT_BYTES,
            cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(
            split_readback.data() + 2 * N_FLOATS, staging_dst, 2 * SLOT_BYTES,
            cudaMemcpyDeviceToHost));
        for (int e = 0; e < 4; ++e) {
            for (int j = 0; j < N_FLOATS; ++j) {
                CHECK(split_readback[e * N_FLOATS + j] == (float)(first + e));
            }
        }
        CHECK(ggml_cuda_moe_cache_release_split_slots(split_cache, slot_ids, 4, compute_stream));
        CUDA_OK(cudaStreamSynchronize(compute_stream));
    }

    if (ggml_cuda_moe_cache_can_overlap_staging(split_cache)) {
        constexpr int N_SPLIT_SRCS = 6;
        const void * split_srcs[N_SPLIT_SRCS];
        for (int e = 0; e < N_SPLIT_SRCS; ++e) {
            split_srcs[e] = host_experts + (size_t)(8 + e) * N_FLOATS;
        }

        host_barrier barrier;
        CUDA_OK(cudaLaunchHostFunc(compute_stream, wait_on_host_barrier, &barrier));
        while (!barrier.entered.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        CUDA_OK(cudaMemsetAsync(staging_dst, 0xff, 4 * SLOT_BYTES + SOURCE_PADDING, compute_stream));

        int slot_ids[N_SPLIT_SRCS] = {-1, -1, -1, -1, -1, -1};
        int32_t wait_classes[N_SPLIT_SRCS] = {-1, -1, -1, -1, -1, -1};
        uint32_t * stage_ready = nullptr;
        CUDA_OK(cudaMalloc(&stage_ready, 3 * sizeof(uint32_t)));
        int n_resident = 0;
        int n_wait_classes = 0;
        CHECK(ggml_cuda_moe_cache_prepare_split_staging(
            split_cache, split_srcs, N_SPLIT_SRCS, SLOT_BYTES, SOURCE_PADDING, 1, slot_ids, wait_classes,
            &n_resident, staging_dst, stage_ready, 3, &n_wait_classes, compute_stream));
        CHECK(n_resident == 2);
        CHECK(n_wait_classes == 4);
        CHECK(wait_classes[0] == 1 && wait_classes[1] == 1);
        CHECK(wait_classes[2] == 2 && wait_classes[3] == 2);
        CHECK(wait_classes[4] == 3 && wait_classes[5] == 3);
        CHECK(cudaStreamQuery(split_copy_stream) == cudaErrorNotReady);

        barrier.released.store(true, std::memory_order_release);
        CHECK(ggml_cuda_moe_cache_finish_split_staging(split_cache, compute_stream));
        CUDA_OK(cudaStreamSynchronize(compute_stream));
        CUDA_OK(cudaMemcpy(
            split_readback.data(), ggml_cuda_moe_cache_slot_ptr(split_cache, slot_ids[0]), SLOT_BYTES,
            cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(
            split_readback.data() + N_FLOATS, ggml_cuda_moe_cache_slot_ptr(split_cache, slot_ids[1]), SLOT_BYTES,
            cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(
            split_readback.data() + 2 * N_FLOATS, staging_dst, 4 * SLOT_BYTES,
            cudaMemcpyDeviceToHost));
        for (int e = 0; e < N_SPLIT_SRCS; ++e) {
            for (int j = 0; j < N_FLOATS; ++j) {
                CHECK(split_readback[e * N_FLOATS + j] == (float)(8 + e));
            }
        }
        std::array<uint8_t, SOURCE_PADDING> split_padding;
        CUDA_OK(cudaMemcpy(
            split_padding.data(), static_cast<const char *>(staging_dst) + 4 * SLOT_BYTES,
            split_padding.size(), cudaMemcpyDeviceToHost));
        CHECK(std::all_of(split_padding.begin(), split_padding.end(), [](uint8_t value) { return value == 0; }));
        uint32_t stage_ready_host[3] = {};
        CUDA_OK(cudaMemcpy(stage_ready_host, stage_ready, sizeof(stage_ready_host), cudaMemcpyDeviceToHost));
        CHECK(stage_ready_host[0] == 1 && stage_ready_host[1] == 1 && stage_ready_host[2] == 1);
        CHECK(ggml_cuda_moe_cache_release_split_slots(
            split_cache, slot_ids, N_SPLIT_SRCS, compute_stream));
        CUDA_OK(cudaStreamSynchronize(compute_stream));
        CUDA_OK(cudaFree(stage_ready));

        int invalid_slots[N_SPLIT_SRCS] = {-1, -1, -1, -1, -1, -1};
        int32_t invalid_wait_classes[N_SPLIT_SRCS] = {-1, -1, -1, -1, -1, -1};
        int invalid_n_resident = 0;
        int invalid_n_wait_classes = 0;
        CHECK(!ggml_cuda_moe_cache_prepare_split_staging(
            split_cache, split_srcs, N_SPLIT_SRCS, SLOT_BYTES, 0, 1, invalid_slots, invalid_wait_classes,
            &invalid_n_resident, staging_dst, (uint32_t *)staging_dst, 1,
            &invalid_n_wait_classes, compute_stream));
        CHECK(cudaStreamQuery(split_copy_stream) == cudaSuccess);
        CHECK(cudaStreamQuery(compute_stream) == cudaSuccess);
    }

    const void * non_overflow_srcs[] = {host_experts, host_experts + N_FLOATS};
    int non_overflow_slots[2] = {-1, -1};
    int non_overflow_resident = 0;
    int non_overflow_wait_classes = 0;
    CHECK(!ggml_cuda_moe_cache_prepare_split_staging(
        split_cache, non_overflow_srcs, 2, SLOT_BYTES, 0, 1, non_overflow_slots,
        nullptr, &non_overflow_resident, staging_dst, nullptr, 0,
        &non_overflow_wait_classes, compute_stream));
    CHECK(cudaStreamQuery(split_copy_stream) == cudaSuccess);
    CHECK(cudaStreamQuery(compute_stream) == cudaSuccess);

    CUDA_OK(cudaFree(staging_dst));
    CUDA_OK(cudaStreamDestroy(compute_stream));
    ggml_cuda_moe_cache_free(split_cache);

    ggml_backend_cuda_moe_observe_expert_tensor(host_experts, "duplicate.name", SLOT_BYTES, 1);
    ggml_backend_cuda_moe_preallocate_pools(dev);
    ggml_cuda_moe_cache_free_all();

    CUDA_OK(cudaStreamDestroy(copy_stream));
    CUDA_OK(cudaFreeHost(host_experts));

    fprintf(stderr, "test-moe-cache: OK\n");
    return 0;
}
