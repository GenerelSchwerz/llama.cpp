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
#include <random>
#include <string>
#include <thread>
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

struct ggml_cuda_moe_grouped_context_test_access {
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

    static uint64_t legacy_op_count(const ggml_cuda_moe_grouped_context & context, bool is_decode) {
        return context.legacy_op_count_for_test(is_decode);
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
};

static cudaStream_t candidate_test_graph_stream(void * data, const ggml_tensor *) {
    return static_cast<cudaStream_t>(data);
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
    size_t next_offset = 0;

    candidate_test_fixture() {
        owner.context = &supports_buft;
        owner.iface.supports_buft = candidate_test_supports_buft;
        CHECK(posix_memalign(&storage, 64, BUFFER_SIZE) == 0);
        memset(storage, 0, BUFFER_SIZE);
        buffer = ggml_backend_cpu_buffer_from_ptr(storage, BUFFER_SIZE);
        CHECK(buffer != nullptr);
        ggml_init_params params = {};
        params.mem_size = 8192 * ggml_tensor_overhead();
        params.no_alloc = true;
        ctx = ggml_init(params);
        CHECK(ctx != nullptr);
    }

    ~candidate_test_fixture() {
        ggml_free(ctx);
        ggml_backend_buffer_free(buffer);
        free(storage);
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

static ggml_cgraph * candidate_graph(candidate_test_fixture & fixture, std::initializer_list<ggml_tensor *> nodes) {
    ggml_cgraph * graph = ggml_new_graph_custom(fixture.ctx, 32, false);
    CHECK(graph != nullptr);
    for (ggml_tensor * node : nodes) {
        ggml_graph_add_node(graph, node);
    }
    candidate_rebuild_graph_uses(graph);
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
        ggml_graph_add_node(graph, padding);
    }
    for (ggml_tensor * node : nodes) {
        ggml_graph_add_node(graph, node);
    }
    candidate_rebuild_graph_uses(graph);
    return graph;
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
            &split_view, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &split_plan, prepared.get(), 1, split_view.nodes) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(prepared->size() == 1 && prepared->find(fused_down, nullptr));
    }
    stable_split_plan = split_plan.get();
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(global_registry.prepare_graph_execution(
            &split_view, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &split_plan, prepared.get(), 1, split_view.nodes) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(split_plan.get() == stable_split_plan && prepared->find(fused_down, nullptr));
    }
    {
        ggml_cgraph exact_callback = ggml_graph_view(&split_view, 0, split_view.n_nodes);
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(global_registry.bind_graph_plan(
            &exact_callback, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, *split_plan, prepared.get(), 1, split_view.nodes));
        CHECK(prepared->find(fused_down, nullptr));
    }
    {
        ggml_cgraph prefix_callback = ggml_graph_view(&split_view, 0, split_view.n_nodes - 1);
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(!global_registry.bind_graph_plan(
            &prefix_callback, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, *split_plan, prepared.get(), 1, split_view.nodes));
        CHECK(!global_registry.bind_graph_plan(
            &split_view, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, *split_plan, prepared.get(), 2, split_view.nodes));
    }

    ggml_cgraph second_split_view = ggml_graph_view(split_parent, 4, split_parent->n_nodes);
    std::shared_ptr<ggml_cuda_moe_graph_plan> second_split_plan;
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(global_registry.prepare_graph_execution(
            &second_split_view, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &second_split_plan, prepared.get(), 2, second_split_view.nodes) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(prepared->size() == 1 && prepared->find(separate_down, nullptr));
    }
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(global_registry.prepare_graph_execution(
            &second_split_view, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &second_split_plan, prepared.get(), 2, second_split_view.nodes) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(prepared->find(separate_down, nullptr));
    }

    const std::shared_ptr<ggml_cuda_moe_graph_plan> decode_split_plan = split_plan;
    candidate_set_route_tokens(fused_route, {fused_gate_up, fused_down}, 4);
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(global_registry.prepare_graph_execution(
            &split_view, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &split_plan, prepared.get(), 3, split_view.nodes) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(split_plan.get() != decode_split_plan.get() && !prepared->find(fused_down, nullptr));
        CHECK(!global_registry.bind_graph_plan(
            &split_view, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, *decode_split_plan, prepared.get(), 3, split_view.nodes));
    }
    candidate_set_route_tokens(fused_route, {fused_gate_up, fused_down}, 1);
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(global_registry.prepare_graph_execution(
            &split_view, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &split_plan, prepared.get(), 4, split_view.nodes) ==
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
    split_plan.reset();
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(global_registry.prepare_graph_execution(
            &split_consumer_view, 35, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &split_plan, prepared.get(), 5, split_consumer_view.nodes) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(prepared->size() == 1 && !prepared->find(fused_down, nullptr));
    }
    stable_split_plan = split_plan.get();
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(global_registry.prepare_graph_execution(
            &split_consumer_view, 36, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &split_plan, prepared.get(), 5, split_consumer_view.nodes) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(split_plan.get() == stable_split_plan && !prepared->find(fused_down, nullptr));
    }
    split_consumer->src[0] = nullptr;
    candidate_rebuild_graph_uses(split_consumer_parent);
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(global_registry.prepare_graph_execution(
            &split_consumer_view, 37, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &split_plan, prepared.get(), 5, split_consumer_view.nodes) ==
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
    separate.ffn_gate_exps = fixture.tensor(GGML_TYPE_Q4_0, 3, gate_ne);
    separate.ffn_up_exps = fixture.tensor(GGML_TYPE_Q4_0, 3, gate_ne);
    separate.ffn_down_exps = fixture.tensor(GGML_TYPE_Q4_0, 3, down_ne);
    separate.ffn_gate_exps_s = fixture.tensor(GGML_TYPE_F32, 1, scale_ne);
    separate.ffn_up_exps_s = fixture.tensor(GGML_TYPE_F32, 1, scale_ne);
    separate.ffn_down_exps_s = fixture.tensor(GGML_TYPE_F32, 1, scale_ne);
    separate.ffn_gate_exps_b = fixture.tensor(GGML_TYPE_F32, 2, gate_bias_ne);
    separate.ffn_up_exps_b = fixture.tensor(GGML_TYPE_F32, 2, gate_bias_ne);
    separate.ffn_down_exps_b = fixture.tensor(GGML_TYPE_F32, 2, down_bias_ne);
    separate.ffn_gate = fixture.tensor(GGML_TYPE_BF16, 2, gate_ne);
    separate.ffn_up_shexp = fixture.tensor(GGML_TYPE_BF16, 2, gate_ne);
    separate.ffn_gate_exps_in_s = fixture.tensor(GGML_TYPE_F32, 1, scalar_ne);

    auto & fused = model->layers[1];
    fused.ffn_gate_inp = fixture.tensor(GGML_TYPE_F32, 2, router_ne);
    fused.ffn_gate_up_exps = fixture.tensor(GGML_TYPE_BF16, 3, fused_ne);
    fused.ffn_down_exps = fixture.tensor(GGML_TYPE_BF16, 3, down_ne);
    fused.ffn_gate_up_exps_b = fixture.tensor(GGML_TYPE_F32, 2, fused_bias_ne);
    fused.ffn_down_exps_b = fixture.tensor(GGML_TYPE_F32, 2, down_bias_ne);

    auto & nvfp4 = model->layers[2];
    nvfp4.ffn_gate_inp = fixture.tensor(GGML_TYPE_F32, 2, router_ne);
    nvfp4.ffn_gate_exps = fixture.tensor(GGML_TYPE_NVFP4, 3, fused_ne);
    nvfp4.ffn_up_exps = fixture.tensor(GGML_TYPE_NVFP4, 3, fused_ne);
    nvfp4.ffn_down_exps = fixture.tensor(GGML_TYPE_NVFP4, 3, fused_ne);
    nvfp4.ffn_gate_exps_in_s = fixture.tensor(GGML_TYPE_F32, 1, scalar_ne);
    nvfp4.ffn_down_shexp = fixture.tensor(GGML_TYPE_BF16, 2, down_ne);

    auto & excluded = model->layers[3];
    excluded.ffn_gate = fixture.tensor(GGML_TYPE_BF16, 2, gate_ne);
    excluded.ffn_up_shexp = fixture.tensor(GGML_TYPE_BF16, 2, gate_ne);
    excluded.ffn_down_shexp = fixture.tensor(GGML_TYPE_BF16, 2, down_ne);

    ggml_set_name(separate.ffn_gate_exps, "blk.0.ffn_gate_exps.weight");
    ggml_set_name(separate.ffn_up_exps, "blk.0.ffn_up_exps.weight");
    ggml_set_name(separate.ffn_down_exps, "blk.0.ffn_down_exps.weight");

    llama_adapter_loras loras;
    llama_moe_candidate_snapshot produced(*model, loras);
    const auto & snapshot = produced.get();
    CHECK(snapshot.magic == GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V1_MAGIC);
    CHECK(snapshot.abi_version == GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V1_VERSION);
    CHECK(snapshot.struct_size == sizeof(snapshot));
    CHECK(snapshot.n_slots == 12 && snapshot.n_groups == 3);

    const auto & separate_group = snapshot.groups[0];
    CHECK(separate_group.layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE);
    CHECK(separate_group.n_banks == 9);
    CHECK(candidate_bank(separate_group, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT)->tensor == separate.ffn_gate_exps);
    CHECK(candidate_bank(separate_group, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT)->tensor == separate.ffn_up_exps);
    CHECK(candidate_bank(separate_group, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT)->tensor == separate.ffn_down_exps);
    CHECK(candidate_bank(separate_group, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_SCALE)->tensor == separate.ffn_gate_exps_s);
    CHECK(candidate_bank(separate_group, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_BIAS)->tensor == separate.ffn_down_exps_b);
    CHECK(candidate_bank(separate_group, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_BLOCK_SCALE) == nullptr);
    for (uint32_t i = 0; i < separate_group.n_banks; ++i) {
        CHECK(separate_group.banks[i].tensor != separate.ffn_gate);
        CHECK(separate_group.banks[i].tensor != separate.ffn_up_shexp);
        CHECK(separate_group.banks[i].tensor != separate.ffn_gate_exps_in_s);
    }

    const auto & fused_group = snapshot.groups[1];
    CHECK(fused_group.layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP);
    CHECK(candidate_bank(fused_group, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT)->tensor == fused.ffn_gate_up_exps);
    CHECK(candidate_bank(fused_group, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_BIAS)->tensor == fused.ffn_gate_up_exps_b);
    CHECK(candidate_bank(fused_group, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_SCALE) == nullptr);

    ggml_cuda_moe_grouped_context registry(&fixture.owner);
    CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(registry.state().n_groups == 3 && registry.state().n_weights == 8);

    llama_adapter_lora adapter(model.get());
    adapter.ab_map.emplace(separate.ffn_gate_exps->name, llama_adapter_lora_weight());
    loras.emplace(&adapter, 1.0f);
    llama_moe_candidate_snapshot lora_on(*model, loras);
    CHECK(lora_on.get().n_groups == 2);
    CHECK(registry.replace(&lora_on.get()) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(!registry.find_weight(separate.ffn_gate_exps, nullptr));
    loras.clear();
    llama_moe_candidate_snapshot lora_off(*model, loras);
    CHECK(lora_off.get().n_groups == 3);
    CHECK(registry.replace(&lora_off.get()) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(registry.find_weight(separate.ffn_gate_exps, nullptr));

    fused.ffn_up_exps_s = separate.ffn_up_exps_s;
    llama_moe_candidate_snapshot untyped_fused_scale(*model, loras);
    CHECK(untyped_fused_scale.get().n_groups == 2);
    fused.ffn_up_exps_s = nullptr;

    llama_model_tensor_buft_override overrides[] = {{".*", ggml_backend_cpu_buffer_type()}, {nullptr, nullptr}};
    params.tensor_buft_overrides = overrides;
    params.moe_expert_cache_slots = 48;
    std::unique_ptr<llama_model> overridden(llama_model_create(LLM_ARCH_LLAMA, params));
    overridden->layers = model->layers;
    llama_moe_candidate_snapshot disabled(*overridden, loras);
    CHECK(disabled.get().n_slots == 48 && disabled.get().n_groups == 0 && disabled.get().groups == nullptr);
    CHECK(registry.replace(&disabled.get()) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(registry.state().accepted == 1 && registry.state().n_groups == 0 && registry.state().n_slots == 48);
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

    const int64_t q5k_ne[] = {256, 32, 4};
    ggml_tensor * q5k = fixture.tensor(GGML_TYPE_Q5_K, 3, q5k_ne);
    bad_banks = banks;
    bad_banks[0].tensor = q5k;
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
    const candidate_route prefill_route = candidate_top_k_route(fixture, 4, 2, 4);

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
    ggml_cuda_moe_grouped_context registry(&fixture.owner);
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

    ggml_cuda_moe_graph_plan plan;
    ggml_cuda_moe_graph_execution execution;
    ggml_cuda_moe_graph_execution reused;
    registry.compile_graph_plan(complete_graph, 41, &plan, &execution);
    CHECK(plan.size() == 2 && execution.size() == 2);
    CHECK(!execution.has_stream_grouped_candidate());
    CHECK(execution.resolve_streams(candidate_test_graph_stream, reinterpret_cast<void *>(uintptr_t{1})));
    CHECK(execution.has_stream_grouped_candidate());
    CHECK(plan.registry_generation() == 1 && plan.graph_uid() == 41 && plan.graph_node_count() == 9);
    ggml_cuda_moe_graph_binding binding;
    CHECK(execution.find(fused_gate_up_node, &binding));
    CHECK(binding.key.candidate.generation == 1 && binding.key.candidate.group_index == 0);
    CHECK(binding.key.ids.tensor == fused_route.ids && binding.key.ids.data == fused_route.ids->data && binding.key.ids.buffer == fused_route.ids->buffer);
    CHECK(binding.key.layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP && binding.key.n_banks == 2);
    CHECK(binding.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT && binding.bank_index == 0);
    CHECK(execution.find(fused_down_node, &binding) && binding.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT && binding.bank_index == 1);
    CHECK(execution.find(separate_up_node, &binding) && binding.key.candidate.group_index == 1 && binding.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT);
    CHECK(execution.find(separate_gate_node, &binding) && binding.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT);
    CHECK(execution.find(separate_down_node, &binding) && binding.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT && binding.key.n_banks == 3);
    CHECK(!execution.find(fused_bias, nullptr));

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
        CHECK(registry.begin_graph_dispatch(&stream_execution, true));
        const auto * fused_authority = stream_execution.find_authority(fused_gate_up_node);
        const auto * separate_authority = stream_execution.find_authority(separate_gate_node);
        const auto * fused_dispatch = stream_execution.find_group(fused_gate_up_node, nullptr);
        CHECK(fused_authority != nullptr && fused_authority->authority() == GGML_CUDA_MOE_GROUP_AUTHORITY_LEGACY);
        CHECK(fused_dispatch != nullptr && fused_dispatch->state == GGML_CUDA_MOE_GRAPH_GROUP_WHOLE_LEGACY &&
            fused_dispatch->transaction.transaction_token == 0);
        CHECK(separate_authority != nullptr && separate_authority->authority() == GGML_CUDA_MOE_GROUP_AUTHORITY_GROUPED);
        CHECK(registry.finish_graph_dispatch(&stream_execution));
    }

    CHECK(registry.bind_graph_plan(complete_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, plan, &reused));
    CHECK(reused.size() == 2 && reused.find(separate_down_node, nullptr));
    CHECK(!registry.bind_graph_plan(complete_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, plan, &reused) && reused.size() == 0);
    CHECK(registry.bind_graph_plan(complete_graph, 42, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, plan, &reused) && reused.size() == 2);
    CHECK(!registry.bind_graph_plan(complete_graph, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, plan, &reused) && reused.size() == 0);
    ggml_cgraph * reordered_graph = candidate_graph(fixture, {
        fused_route.root, fused_route.ids, separate_route.root, separate_route.ids,
        fused_gate_up_node, fused_down_node, separate_gate_node, separate_up_node, separate_down_node,
    });
    CHECK(execution.find(fused_gate_up_node, &binding) && binding.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT);
    CHECK(execution.find(fused_down_node, &binding) && binding.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT);
    CHECK(!registry.bind_graph_plan(reordered_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, plan, &reused) && reused.size() == 0);
    CHECK(registry.bind_graph_plan(complete_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, plan, &reused));
    ggml_tensor * stale_gate_up_node = candidate_mmid(fixture, fused_gate_up, fused_route.ids);
    ggml_cgraph * stale_graph = candidate_graph(fixture, {
        fused_route.root, fused_route.ids, separate_route.root, separate_route.ids,
        stale_gate_up_node, fused_down_node, separate_up_node, separate_gate_node, separate_down_node,
    });
    CHECK(!registry.bind_graph_plan(stale_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, plan, &reused) && reused.size() == 0);
    CHECK(registry.bind_graph_plan(complete_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, plan, &reused));

    std::shared_ptr<ggml_cuda_moe_graph_plan> cached_plan;
    {
        ggml_cuda_moe_graph_execution prepared;
        CHECK(registry.prepare_graph_execution(complete_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &cached_plan, &prepared) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(cached_plan != nullptr && prepared.size() == 2);
    }
    const ggml_cuda_moe_graph_plan * first_cached_plan = cached_plan.get();
    {
        ggml_cuda_moe_graph_execution prepared;
        CHECK(registry.prepare_graph_execution(complete_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, &cached_plan, &prepared) == GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(cached_plan.get() == first_cached_plan && prepared.find(separate_down_node, nullptr));
    }
    {
        ggml_cuda_moe_graph_execution prepared;
        CHECK(registry.prepare_graph_execution(complete_graph, 42, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &cached_plan, &prepared) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(cached_plan.get() == first_cached_plan && prepared.find(fused_down_node, nullptr));
    }
    {
        ggml_cuda_moe_graph_execution prepared;
        CHECK(registry.prepare_graph_execution(complete_graph, 42, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &cached_plan, &prepared) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(cached_plan.get() != first_cached_plan && prepared.size() == 2);
    }
    {
        std::weak_ptr<ggml_cuda_moe_graph_plan> lifetime;
        {
            std::shared_ptr<ggml_cuda_moe_graph_plan> lifetime_plan;
            ggml_cuda_moe_graph_execution prepared;
            CHECK(registry.prepare_graph_execution(complete_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &lifetime_plan, &prepared) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
            lifetime = lifetime_plan;
            lifetime_plan.reset();
            CHECK(!lifetime.expired() && prepared.find(fused_down_node, nullptr));
        }
        CHECK(lifetime.expired());
    }
    {
        const ggml_cuda_moe_graph_plan * current_plan = cached_plan.get();
        ggml_cuda_moe_graph_execution prepared;
        CHECK(registry.prepare_graph_execution(reordered_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, &cached_plan, &prepared) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(cached_plan.get() != current_plan && prepared.find(fused_gate_up_node, nullptr));
    }
    {
        ggml_cuda_moe_graph_execution prepared;
        CHECK(registry.prepare_graph_execution(complete_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, &cached_plan, &prepared) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(prepared.size() == 2);
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
    std::shared_ptr<ggml_cuda_moe_graph_plan> transition_plan;
    {
        ggml_cuda_moe_graph_execution prepared;
        CHECK(registry.prepare_graph_execution(transition_graph, 200, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &transition_plan, &prepared) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(prepared.size() == 2 && !prepared.find(transition_fused_gate_up, nullptr) && !prepared.find(transition_separate_gate, nullptr));
    }
    const std::shared_ptr<ggml_cuda_moe_graph_plan> warmup_plan = transition_plan;
    {
        ggml_cuda_moe_graph_execution prepared;
        CHECK(registry.prepare_graph_execution(transition_graph, 2001, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &transition_plan, &prepared) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(transition_plan.get() == warmup_plan.get() && !prepared.find(transition_fused_gate_up, nullptr));
    }
    candidate_set_route_tokens(transition_fused_route, {transition_fused_gate_up, transition_fused_down}, 1);
    candidate_set_route_tokens(transition_separate_route, {transition_separate_gate, transition_separate_up, transition_separate_down}, 1);
    {
        ggml_cuda_moe_graph_execution prepared;
        CHECK(registry.prepare_graph_execution(transition_graph, 201, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &transition_plan, &prepared) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(transition_plan.get() != warmup_plan.get() && prepared.find(transition_fused_gate_up, nullptr) &&
            prepared.find(transition_separate_down, nullptr));
    }
    const ggml_cuda_moe_graph_plan * decode_plan = transition_plan.get();
    for (uint64_t uid = 202; uid < 234; ++uid) {
        ggml_cuda_moe_graph_execution prepared;
        CHECK(registry.prepare_graph_execution(transition_graph, uid, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &transition_plan, &prepared) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(transition_plan.get() == decode_plan && prepared.find(transition_fused_down, nullptr));
    }
    candidate_set_route_tokens(transition_fused_route, {transition_fused_gate_up, transition_fused_down}, 4);
    candidate_set_route_tokens(transition_separate_route, {transition_separate_gate, transition_separate_up, transition_separate_down}, 4);
    {
        ggml_cuda_moe_graph_execution prepared;
        CHECK(registry.prepare_graph_execution(transition_graph, 234, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &transition_plan, &prepared) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(!prepared.find(transition_fused_gate_up, nullptr) && !prepared.find(transition_separate_gate, nullptr));
    }
    {
        ggml_cuda_moe_graph_execution prepared;
        CHECK(registry.prepare_graph_execution(transition_graph, 2341, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &transition_plan, &prepared) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(!prepared.find(transition_fused_gate_up, nullptr));
    }
    candidate_set_route_tokens(transition_fused_route, {transition_fused_gate_up, transition_fused_down}, 1);
    candidate_set_route_tokens(transition_separate_route, {transition_separate_gate, transition_separate_up, transition_separate_down}, 1);
    {
        ggml_cuda_moe_graph_execution prepared;
        CHECK(registry.prepare_graph_execution(transition_graph, 235, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &transition_plan, &prepared) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(prepared.find(transition_fused_gate_up, nullptr) && prepared.find(transition_separate_down, nullptr));
    }

    ggml_tensor * transition_consumer = ggml_dup(fixture.ctx, transition_fused_gate_up);
    fixture.materialize(transition_consumer);
    transition_consumer->flags |= GGML_TENSOR_FLAG_COMPUTE;
    ggml_cgraph * mutation_graph = candidate_graph(fixture, {
        transition_fused_route.root, transition_fused_route.ids, transition_separate_route.root, transition_separate_route.ids,
        transition_fused_gate_up, transition_consumer, transition_fused_down,
        transition_separate_gate, transition_separate_up, transition_separate_down,
    });
    std::shared_ptr<ggml_cuda_moe_graph_plan> mutation_plan;
    {
        ggml_cuda_moe_graph_execution prepared;
        CHECK(registry.prepare_graph_execution(mutation_graph, 240, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &mutation_plan, &prepared) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(prepared.find(transition_fused_gate_up, nullptr) && prepared.find(transition_separate_down, nullptr));
    }
    const ggml_op saved_consumer_op = transition_consumer->op;
    ggml_tensor * saved_consumer_src[3] = {transition_consumer->src[0], transition_consumer->src[1], transition_consumer->src[2]};
    transition_consumer->op = GGML_OP_ADD_ID;
    transition_consumer->src[0] = transition_fused_gate_up;
    transition_consumer->src[1] = fused_bias;
    transition_consumer->src[2] = transition_fused_route.ids;
    candidate_rebuild_graph_uses(mutation_graph);
    {
        ggml_cuda_moe_graph_execution prepared;
        CHECK(registry.prepare_graph_execution(mutation_graph, 241, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &mutation_plan, &prepared) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(!prepared.find(transition_fused_gate_up, nullptr) && prepared.find(transition_separate_down, nullptr));
    }
    {
        ggml_cuda_moe_graph_execution prepared;
        CHECK(registry.prepare_graph_execution(mutation_graph, 2411, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &mutation_plan, &prepared) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(!prepared.find(transition_fused_gate_up, nullptr));
    }
    transition_consumer->op = saved_consumer_op;
    memcpy(transition_consumer->src, saved_consumer_src, sizeof(saved_consumer_src));
    candidate_rebuild_graph_uses(mutation_graph);
    {
        ggml_cuda_moe_graph_execution prepared;
        CHECK(registry.prepare_graph_execution(mutation_graph, 242, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &mutation_plan, &prepared) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(prepared.find(transition_fused_gate_up, nullptr));
    }
    transition_consumer->op = GGML_OP_MUL_MAT_ID;
    transition_consumer->src[0] = fused_gate_up;
    transition_consumer->src[1] = transition_fused_gate_up->src[1];
    transition_consumer->src[2] = transition_fused_route.ids;
    candidate_rebuild_graph_uses(mutation_graph);
    {
        ggml_cuda_moe_graph_execution prepared;
        CHECK(registry.prepare_graph_execution(mutation_graph, 243, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &mutation_plan, &prepared) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(!prepared.find(transition_fused_gate_up, nullptr) && prepared.find(transition_separate_down, nullptr));
    }
    {
        ggml_cuda_moe_graph_execution prepared;
        CHECK(registry.prepare_graph_execution(mutation_graph, 2431, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &mutation_plan, &prepared) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(!prepared.find(transition_fused_gate_up, nullptr));
    }
    transition_consumer->op = saved_consumer_op;
    memcpy(transition_consumer->src, saved_consumer_src, sizeof(saved_consumer_src));
    candidate_rebuild_graph_uses(mutation_graph);
    {
        ggml_cuda_moe_graph_execution prepared;
        CHECK(registry.prepare_graph_execution(mutation_graph, 244, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &mutation_plan, &prepared) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(prepared.find(transition_fused_gate_up, nullptr));
    }
    transition_fused_route.ids->view_offs = sizeof(int32_t);
    {
        ggml_cuda_moe_graph_execution prepared;
        CHECK(registry.prepare_graph_execution(mutation_graph, 245, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &mutation_plan, &prepared) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(!prepared.find(transition_fused_gate_up, nullptr));
    }
    transition_fused_route.ids->view_offs = 0;
    {
        ggml_cuda_moe_graph_execution prepared;
        CHECK(registry.prepare_graph_execution(mutation_graph, 246, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &mutation_plan, &prepared) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(prepared.find(transition_fused_gate_up, nullptr));
    }
    {
        const ggml_cuda_moe_graph_plan * current_plan = cached_plan.get();
        ggml_cuda_moe_graph_execution prepared;
        CHECK(registry.prepare_graph_execution(stale_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, &cached_plan, &prepared) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(cached_plan.get() != current_plan && prepared.find(stale_gate_up_node, nullptr));
    }
    {
        ggml_cuda_moe_graph_execution prepared;
        CHECK(registry.prepare_graph_execution(complete_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, &cached_plan, &prepared) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(prepared.size() == 2);
    }
    ggml_cuda_moe_grouped_context other_registry(&fixture.owner);
    CHECK(other_registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(!other_registry.bind_graph_plan(complete_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, plan, &reused) && reused.size() == 0);
    {
        std::shared_ptr<ggml_cuda_moe_graph_plan> foreign_plan = cached_plan;
        const ggml_cuda_moe_graph_plan * original_owner_plan = foreign_plan.get();
        ggml_cuda_moe_graph_execution prepared;
        CHECK(other_registry.prepare_graph_execution(complete_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, &foreign_plan, &prepared) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(foreign_plan.get() != original_owner_plan && prepared.size() == 2);
    }

    fused_gate_up_node->src[0] = separate_gate;
    CHECK(!registry.bind_graph_plan(complete_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, plan, &reused) && reused.size() == 0);
    fused_gate_up_node->src[0] = fused_gate_up;
    CHECK(registry.bind_graph_plan(complete_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, plan, &reused));

    CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(!registry.bind_graph_plan(complete_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, plan, &reused) && reused.size() == 0);
    const ggml_cuda_moe_graph_plan * generation_one_plan = cached_plan.get();
    {
        ggml_cuda_moe_graph_execution prepared;
        CHECK(registry.prepare_graph_execution(complete_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, &cached_plan, &prepared) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(cached_plan.get() != generation_one_plan && cached_plan->registry_generation() == 2 && prepared.size() == 2);
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
    } while (registry.bind_graph_plan(complete_graph, 43, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, plan, &reused));
    CHECK(reused.size() == 0 && !replacement_done.load(std::memory_order_acquire));
    {
        ggml_cuda_moe_graph_execution prepared;
        CHECK(registry.prepare_graph_execution(complete_graph, 43, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, &cached_plan, &prepared) == GGML_CUDA_MOE_GRAPH_PREPARE_UNAVAILABLE);
        CHECK(cached_plan == nullptr && prepared.size() == 0);
    }
    registry.compile_graph_plan(complete_graph, 43, &plan, &execution);
    CHECK(plan.size() == 0 && execution.size() == 0 && plan.registry_generation() == 0);
    CHECK(registry.end_group_transaction(held_transaction));
    replacement_thread.join();
    CHECK(replacement_done.load(std::memory_order_acquire));
    CHECK(!registry.bind_graph_plan(complete_graph, 43, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, plan, &reused) && reused.size() == 0);

    registry.compile_graph_plan(complete_graph, 44, &plan, &execution);
    CHECK(execution.size() == 2);
    {
        ggml_cuda_moe_graph_execution prepared;
        CHECK(registry.prepare_graph_execution(complete_graph, 44, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &cached_plan, &prepared) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(cached_plan != nullptr && prepared.size() == 2);
    }
    auto disabled = candidate_snapshot(12, nullptr, 0);
    CHECK(registry.replace(&disabled) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    registry.compile_graph_plan(complete_graph, 44, &plan, &execution);
    CHECK(plan.size() == 0 && execution.size() == 0 && plan.registry_generation() != 0);
    {
        ggml_cuda_moe_graph_execution prepared;
        CHECK(registry.prepare_graph_execution(complete_graph, 44, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, &cached_plan, &prepared) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(cached_plan != nullptr && prepared.size() == 0);
        const ggml_cuda_moe_graph_plan * disabled_plan = cached_plan.get();
        CHECK(registry.prepare_graph_execution(complete_graph, 44, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, &cached_plan, &prepared) == GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(cached_plan.get() == disabled_plan && prepared.size() == 0);
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
    ggml_cuda_moe_graph_plan padded_plan;
    ggml_cuda_moe_graph_execution padded_execution;
    registry.compile_graph_plan(padded_graph, 100, &padded_plan, &padded_execution);
    CHECK(padded_plan.size() == 2 && padded_execution.size() == 2);
    CHECK(padded_plan.graph_node_count() == static_cast<int32_t>(n_padding_nodes + 9));
    CHECK(registry.bind_graph_plan(padded_graph, 101, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, padded_plan, &padded_execution) && padded_execution.size() == 2);

    registry.compile_graph_plan(complete_graph, 52, &plan, &execution);
    CHECK(plan.size() == 2 && execution.size() == 2);
    void * saved_ids_data = fused_route.ids->data;
    fused_route.ids->data = fused_route_other.ids->data;
    CHECK(!registry.bind_graph_plan(complete_graph, 52, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, plan, &reused) && reused.size() == 0);
    fused_route.ids->data = saved_ids_data;
    CHECK(registry.bind_graph_plan(complete_graph, 52, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, plan, &reused));
    ggml_backend_buffer_t saved_ids_buffer = fused_route.ids->buffer;
    fused_route.ids->buffer = nullptr;
    CHECK(!registry.bind_graph_plan(complete_graph, 52, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, plan, &reused) && reused.size() == 0);
    fused_route.ids->buffer = saved_ids_buffer;
    CHECK(registry.bind_graph_plan(complete_graph, 52, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, plan, &reused));
    const size_t saved_ids_stride = fused_route.ids->nb[0];
    fused_route.ids->nb[0] += sizeof(int32_t);
    CHECK(!registry.bind_graph_plan(complete_graph, 52, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, plan, &reused) && reused.size() == 0);
    fused_route.ids->nb[0] = saved_ids_stride;
    CHECK(registry.bind_graph_plan(complete_graph, 52, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, plan, &reused));
    const int64_t saved_activation_tokens = fused_gate_up_node->src[1]->ne[2];
    fused_gate_up_node->src[1]->ne[2] = 2;
    CHECK(!registry.bind_graph_plan(complete_graph, 52, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, plan, &reused) && reused.size() == 0);
    fused_gate_up_node->src[1]->ne[2] = saved_activation_tokens;
    CHECK(registry.bind_graph_plan(complete_graph, 52, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, plan, &reused));
    const int64_t saved_output_tokens = fused_gate_up_node->ne[2];
    fused_gate_up_node->ne[2] = 2;
    CHECK(!registry.bind_graph_plan(complete_graph, 52, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, plan, &reused) && reused.size() == 0);
    fused_gate_up_node->ne[2] = saved_output_tokens;
    CHECK(registry.bind_graph_plan(complete_graph, 52, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, plan, &reused));
    ggml_tensor * saved_weight = fused_gate_up_node->src[0];
    fused_gate_up_node->src[0] = separate_gate;
    CHECK(!registry.bind_graph_plan(complete_graph, 52, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, plan, &reused) && reused.size() == 0);
    fused_gate_up_node->src[0] = saved_weight;
    CHECK(registry.bind_graph_plan(complete_graph, 52, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, plan, &reused));
    ggml_tensor * saved_view_src = fused_route.ids->view_src;
    fused_route.ids->view_src = fused_route_other.root;
    CHECK(!registry.bind_graph_plan(complete_graph, 52, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, plan, &reused) && reused.size() == 0);
    fused_route.ids->view_src = saved_view_src;
    CHECK(registry.bind_graph_plan(complete_graph, 52, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, plan, &reused));
    ggml_tensor * saved_view_source = fused_route.ids->src[0];
    fused_route.ids->src[0] = fused_route_other.root;
    CHECK(!registry.bind_graph_plan(complete_graph, 52, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, plan, &reused) && reused.size() == 0);
    fused_route.ids->src[0] = saved_view_source;
    CHECK(registry.bind_graph_plan(complete_graph, 52, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, plan, &reused));
    fused_route.ids->view_offs = sizeof(int32_t);
    CHECK(!registry.bind_graph_plan(complete_graph, 52, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, plan, &reused) && reused.size() == 0);
    fused_route.ids->view_offs = 0;
    CHECK(registry.bind_graph_plan(complete_graph, 52, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, plan, &reused));
    const size_t saved_root_stride = fused_route.root->nb[1];
    fused_route.root->nb[1] += sizeof(int32_t);
    CHECK(!registry.bind_graph_plan(complete_graph, 52, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, plan, &reused) && reused.size() == 0);
    fused_route.root->nb[1] = saved_root_stride;
    CHECK(registry.bind_graph_plan(complete_graph, 52, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, plan, &reused));
    ggml_tensor * saved_root_source = fused_route.root->src[0];
    fused_route.root->src[0] = fused_route_other.source;
    CHECK(!registry.bind_graph_plan(complete_graph, 52, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, plan, &reused) && reused.size() == 0);
    fused_route.root->src[0] = saved_root_source;
    CHECK(registry.bind_graph_plan(complete_graph, 52, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, plan, &reused));
    const int32_t sort_asc = GGML_SORT_ORDER_ASC;
    memcpy(fused_route.root->op_params, &sort_asc, sizeof(sort_asc));
    CHECK(!registry.bind_graph_plan(complete_graph, 52, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, plan, &reused) && reused.size() == 0);
    const int32_t sort_desc = GGML_SORT_ORDER_DESC;
    memcpy(fused_route.root->op_params, &sort_desc, sizeof(sort_desc));
    CHECK(registry.bind_graph_plan(complete_graph, 52, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, plan, &reused));
    ggml_cgraph * producer_reordered_graph = candidate_graph(fixture, {
        fused_route.root, fused_route.ids, separate_route.ids, separate_route.root,
        fused_gate_up_node, fused_down_node, separate_up_node, separate_gate_node, separate_down_node,
    });
    CHECK(!registry.bind_graph_plan(producer_reordered_graph, 52, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, plan, &reused) && reused.size() == 0);
    CHECK(registry.bind_graph_plan(complete_graph, 52, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, plan, &reused));

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
        ggml_cuda_moe_graph_execution prepared;
        CHECK(registry.prepare_graph_execution(missing_graph, 540, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &missing_plan, &prepared) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        const std::shared_ptr<ggml_cuda_moe_graph_plan> first_missing_plan = missing_plan;
        CHECK(registry.prepare_graph_execution(missing_graph, 541, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &missing_plan, &prepared) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(missing_plan.get() != first_missing_plan.get() && !prepared.find(missing_gate_up_node, nullptr));
    }
    ggml_cgraph * complete_missing_graph = candidate_graph(fixture, {
        fused_route.root, fused_route.ids, separate_route.root, separate_route.ids,
        missing_gate_up_node, separate_up_node, separate_gate_node, separate_down_node,
    });
    {
        std::shared_ptr<ggml_cuda_moe_graph_plan> missing_plan;
        ggml_cuda_moe_graph_execution prepared;
        CHECK(registry.prepare_graph_execution(complete_missing_graph, 542, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &missing_plan, &prepared) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        const ggml_cuda_moe_graph_plan * stable_missing_plan = missing_plan.get();
        CHECK(!prepared.find(missing_gate_up_node, nullptr) && prepared.find(separate_down_node, nullptr));
        CHECK(registry.prepare_graph_execution(complete_missing_graph, 543, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &missing_plan, &prepared) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(missing_plan.get() == stable_missing_plan && !prepared.find(missing_gate_up_node, nullptr));
    }

    ggml_tensor * duplicate_gate_up_node = candidate_mmid(fixture, fused_gate_up, fused_route.ids);
    ggml_tensor * duplicate_gate_up_peer = candidate_mmid(fixture, fused_gate_up, fused_route.ids);
    ggml_tensor * duplicate_down_node = candidate_mmid(fixture, fused_down, fused_route.ids);
    ggml_cgraph * duplicate_graph = candidate_graph(fixture, {
        fused_route.root, fused_route.ids, duplicate_gate_up_node, duplicate_gate_up_peer, duplicate_down_node,
    });
    registry.compile_graph_plan(duplicate_graph, 55, &plan, &execution);
    CHECK(plan.size() == 1 && execution.size() == 1 && !execution.find(duplicate_gate_up_node, nullptr));

    ggml_tensor * prefill_gate_up_node = candidate_mmid(fixture, fused_gate_up, prefill_route.ids);
    ggml_tensor * prefill_down_node = candidate_mmid(fixture, fused_down, prefill_route.ids);
    ggml_cgraph * prefill_graph = candidate_graph(fixture, {
        prefill_route.root, prefill_route.ids, prefill_gate_up_node, prefill_down_node,
    });
    registry.compile_graph_plan(prefill_graph, 56, &plan, &execution);
    CHECK(plan.size() == 1 && execution.size() == 1 && !execution.find(prefill_gate_up_node, nullptr));

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

    CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    registry.compile_graph_plan(complete_graph, 60, &plan, &execution);
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
    CHECK(registry.finish_graph_dispatch(&execution));

    registry.compile_graph_plan(missing_graph, 61, &plan, &execution);
    CHECK(registry.begin_graph_dispatch(&execution, true));
    CHECK(!registry.get_group_resources(authority_resource, nullptr));
    auto legacy_after = registry.acquire_legacy_cache(fused_gate_up);
    CHECK(legacy_after && legacy_after.acquisition().group_authority_epoch > legacy_before_state.group_authority_epoch);
    legacy_after = {};
    CHECK(registry.finish_graph_dispatch(&execution));

    auto finalization_registry = std::make_unique<ggml_cuda_moe_grouped_context>(&fixture.owner);
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
    } while (registry.bind_graph_plan(complete_graph, 62, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, plan, &reused));
    CHECK(!authority_replacement_done.load(std::memory_order_acquire));
    CHECK(registry.finish_graph_dispatch(&execution));
    authority_replacement.join();
    CHECK(authority_replacement_done.load(std::memory_order_acquire));

    {
        ggml_cuda_moe_graph_plan scoped_plan;
        ggml_cuda_moe_graph_execution scoped_execution;
        registry.compile_graph_plan(complete_graph, 63, &scoped_plan, &scoped_execution);
        CHECK(registry.begin_graph_dispatch(&scoped_execution, true));
    }
    CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);

    auto terminal = std::make_unique<ggml_cuda_moe_grouped_context>(&fixture.owner);
    CHECK(terminal->replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    ggml_cuda_moe_graph_plan terminal_plan;
    ggml_cuda_moe_graph_execution terminal_execution;
    ggml_cuda_moe_graph_execution terminal_probe;
    terminal->compile_graph_plan(complete_graph, 64, &terminal_plan, &terminal_execution);
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
    CHECK(terminal->finish_graph_dispatch(&terminal_execution));
    authority_shutdown.join();
    CHECK(authority_shutdown_done.load(std::memory_order_acquire));
    terminal.reset();

    if (benchmark) {
        CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        std::shared_ptr<ggml_cuda_moe_graph_plan> benchmark_plan;
        ggml_cuda_moe_graph_execution prepared;
        CHECK(registry.prepare_graph_execution(complete_graph, 60, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &benchmark_plan, &prepared) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        const ggml_cuda_moe_graph_plan * stable_plan = benchmark_plan.get();
        constexpr uint32_t n_reuses = 200000;
        uint32_t reused_count = 0;
        const auto begin = std::chrono::steady_clock::now();
        for (uint32_t i = 0; i < n_reuses; ++i) {
            reused_count += registry.prepare_graph_execution(complete_graph, 61 + i, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &benchmark_plan, &prepared) ==
                GGML_CUDA_MOE_GRAPH_PREPARE_REUSED ? 1 : 0;
        }
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin).count();
        CHECK(reused_count == n_reuses && benchmark_plan.get() == stable_plan && prepared.size() == 2);
        fprintf(stderr, "test-moe-cache: grouped graph plan %.1f ns/reuse with fresh UIDs, full scans=0/%u\n",
            static_cast<double>(elapsed) / n_reuses, n_reuses);

        std::shared_ptr<ggml_cuda_moe_graph_plan> padded_benchmark_plan;
        CHECK(registry.prepare_graph_execution(padded_graph, 100, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &padded_benchmark_plan, &prepared) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        const ggml_cuda_moe_graph_plan * stable_padded_plan = padded_benchmark_plan.get();
        constexpr uint32_t n_padded_reuses = 100000;
        uint32_t padded_reused_count = 0;
        const auto padded_begin = std::chrono::steady_clock::now();
        for (uint32_t i = 0; i < n_padded_reuses; ++i) {
            padded_reused_count += registry.prepare_graph_execution(padded_graph, 101 + i, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &padded_benchmark_plan, &prepared) ==
                GGML_CUDA_MOE_GRAPH_PREPARE_REUSED ? 1 : 0;
        }
        const auto padded_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - padded_begin).count();
        const double padded_ns = static_cast<double>(padded_elapsed) / n_padded_reuses;
        CHECK(padded_reused_count == n_padded_reuses && padded_benchmark_plan.get() == stable_padded_plan && prepared.size() == 2);
        CHECK(padded_ns < 10000.0);
        fprintf(stderr, "test-moe-cache: grouped padded graph plan %.1f ns/reuse with %u nodes, full scans=0/%u\n",
            padded_ns, n_padding_nodes + 9, n_padded_reuses);
    }

    fprintf(stderr, "test-moe-cache: grouped graph preflight OK\n");
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

struct active_grouped_dispatch_graph {
    ggml_context_ptr weights;
    ggml_context_ptr nodes;
    ggml_backend_buffer_ptr weight_buffer;
    ggml_backend_buffer_ptr node_buffer;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * down = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * logits = nullptr;
    ggml_tensor * ids = nullptr;
    ggml_tensor * gate_output = nullptr;
    ggml_tensor * up_output = nullptr;
    ggml_tensor * output = nullptr;
    std::vector<ggml_tensor *> banks;
    std::vector<ggml_tensor *> readers;
    std::vector<uint32_t> roles;
};

static active_grouped_dispatch_graph build_active_grouped_dispatch_graph(
        ggml_backend_t backend,
        ggml_backend_buffer_type_t weight_buft,
        ggml_type type,
        uint32_t layout) {
    constexpr int64_t N_EXPERTS = 8;
    constexpr int64_t N_USED = 2;
    constexpr int64_t N_DIM = 256;
    const ggml_init_params weight_params = {
        /* .mem_size = */ ggml_tensor_overhead() * 8,
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };
    const ggml_init_params node_params = {
        /* .mem_size = */ ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false),
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };

    active_grouped_dispatch_graph result;
    result.weights.reset(ggml_init(weight_params));
    result.nodes.reset(ggml_init(node_params));
    CHECK(result.weights != nullptr && result.nodes != nullptr);

    auto add_bank = [&](int64_t ne1, uint32_t role, const char * name) {
        ggml_tensor * tensor = ggml_new_tensor_3d(result.weights.get(), type, N_DIM, ne1, N_EXPERTS);
        ggml_set_name(tensor, name);
        result.banks.push_back(tensor);
        result.roles.push_back(role);
        return tensor;
    };
    ggml_tensor * gate_up = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * up = nullptr;
    if (layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP) {
        gate_up = add_bank(2 * N_DIM, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, "test.active.gate_up");
    } else {
        gate = add_bank(N_DIM, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT, "test.active.gate");
        up = add_bank(N_DIM, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT, "test.active.up");
    }
    result.down = add_bank(N_DIM, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, "test.active.down");

    result.input = ggml_new_tensor_2d(result.nodes.get(), GGML_TYPE_F32, N_DIM, 1);
    result.logits = ggml_new_tensor_2d(result.nodes.get(), GGML_TYPE_F32, N_EXPERTS, 1);
    ggml_set_name(result.input, "test.active.input");
    ggml_set_name(result.logits, "test.active.logits");
    result.ids = ggml_argsort_top_k(result.nodes.get(), result.logits, N_USED);
    ggml_set_name(result.ids, "test.active.ids");
    ggml_tensor * hidden = nullptr;
    if (gate_up != nullptr) {
        hidden = ggml_mul_mat_id(result.nodes.get(), gate_up, result.input, result.ids);
        result.readers.push_back(hidden);
        hidden = ggml_dup(result.nodes.get(), hidden);
        hidden = ggml_glu(result.nodes.get(), hidden, GGML_GLU_OP_SWIGLU, false);
    } else {
        result.gate_output = ggml_mul_mat_id(result.nodes.get(), gate, result.input, result.ids);
        result.up_output = ggml_mul_mat_id(result.nodes.get(), up, result.input, result.ids);
        result.readers.push_back(result.gate_output);
        result.readers.push_back(result.up_output);
        hidden = ggml_glu_split(result.nodes.get(), result.gate_output, result.up_output, GGML_GLU_OP_SWIGLU);
    }
    result.output = ggml_mul_mat_id(result.nodes.get(), result.down, hidden, result.ids);
    result.readers.push_back(result.output);
    ggml_set_name(result.output, "test.active.output");
    result.graph = ggml_new_graph_custom(result.nodes.get(), 64, false);
    ggml_build_forward_expand(result.graph, result.output);
    result.weight_buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(result.weights.get(), weight_buft));
    result.node_buffer.reset(ggml_backend_alloc_ctx_tensors(result.nodes.get(), backend));
    CHECK(result.weight_buffer != nullptr && result.node_buffer != nullptr);
    ggml_backend_buffer_set_usage(result.weight_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    return result;
}

static void initialize_active_grouped_dispatch_graphs(const std::vector<active_grouped_dispatch_graph *> & graphs) {
    CHECK(graphs.size() >= 2);
    const float logits[] = {-1.0f, 3.0f, 0.5f, 9.0f, 2.0f, 8.0f, -2.0f, 1.0f};
    for (auto * graph : graphs) {
        CHECK(graph->banks.size() == graphs[0]->banks.size());
    }
    for (size_t bank = 0; bank < graphs[0]->banks.size(); ++bank) {
        const auto bytes = cached_fusion_test_data(graphs[0]->banks[bank], bank + 101);
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
    const auto input = cached_fusion_test_data(graphs[0]->input, 131);
    for (auto * graph : graphs) {
        CHECK(graph->readers.size() == graph->banks.size());
        CHECK(graph->readers.back() == graph->output && graph->output->src[0] == graph->down);
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
        ggml_backend_tensor_set(graph->logits, logits, 0, sizeof(logits));
    }
}

static void register_active_grouped_dispatch(
        ggml_backend_t backend,
        const active_grouped_dispatch_graph & graph,
        uint32_t layout,
        uint32_t n_slots) {
    std::array<ggml_backend_moe_candidate_bank_v1, 3> banks = {};
    for (size_t i = 0; i < graph.banks.size(); ++i) {
        banks[i] = {graph.banks[i], graph.roles[i], 0};
    }
    const ggml_backend_moe_candidate_group_v1 group = {
        banks.data(), static_cast<uint32_t>(graph.banks.size()), layout, 0, 0,
    };
    const auto snapshot = candidate_snapshot(n_slots, &group, 1);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v1(backend, &snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
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

static uint64_t active_grouped_legacy_op_count(ggml_backend_t backend) {
    auto * context = ggml_cuda_moe_grouped_context_for_test(backend);
    CHECK(context != nullptr);
    return ggml_cuda_moe_grouped_context_test_access::legacy_op_count(*context, true);
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
        uint32_t n_slots) {
    auto * context = ggml_cuda_moe_grouped_context_for_test(backend);
    CHECK(context != nullptr);
    const int32_t routes[] = {3, 5};
    ggml_backend_tensor_set(graph.ids, routes, 0, sizeof(routes));

    std::shared_ptr<ggml_cuda_moe_graph_plan> plan;
    ggml_cuda_moe_graph_execution execution;
    CHECK(context->prepare_graph_execution(graph.graph, 801, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &plan, &execution) ==
        GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
    CHECK(execution.size() == 1);
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
    const int32_t * remapped_ids = group->remapped_ids;

    for (size_t i = 0; i < graph.readers.size(); ++i) {
        ggml_cuda_moe_graph_binding binding;
        CHECK(execution.find_group(graph.readers[i], &binding) == group);
        CHECK(binding.bank_index == i && binding.role == graph.roles[i]);
        CHECK(binding.key.ids.tensor == graph.ids && group->remapped_ids == remapped_ids);
        CHECK(!context->acquire_legacy_cache(graph.banks[i], nullptr, &group->authority));
    }

    ggml_cuda_moe_grouped_resource_info info;
    CHECK(context->get_group_resources(group->transaction.acquisition, &info));
    CHECK(info.transaction_active && info.n_banks == graph.banks.size() && info.n_slots == n_slots);
    const auto acquisition = group->transaction.acquisition;
    CHECK(context->finish_graph_dispatch(&execution));
    CHECK(context->get_group_resources(acquisition, &info) && !info.transaction_active);
    CUDA_OK(cudaStreamSynchronize(stream));
    CUDA_OK(cudaStreamDestroy(stream));

    ggml_cuda_moe_candidate_group_key key;
    uint64_t clock_bound = 0;
    CHECK(context->find_down_group_key(graph.down, &key));
    CHECK(ggml_cuda_moe_grouped_context_test_access::get_clock_bound(*context, key, &clock_bound));
    CHECK(clock_bound == 2);
}

static std::vector<float> run_active_grouped_dispatch(
        ggml_backend_t backend,
        active_grouped_dispatch_graph & graph,
        uint64_t expected_clock,
        bool f3_skipped = false) {
    const auto sentinel = active_grouped_intermediate_sentinel(graph);
    CHECK(ggml_backend_graph_compute(backend, graph.graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
    check_active_grouped_intermediates(graph, sentinel, f3_skipped);
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

static void test_active_grouped_dispatch_case(ggml_type type, uint32_t layout, uint32_t n_slots) {
    const bool old_debug_mm = ggml_backend_cuda_moe_get_debug_mm();
    ggml_backend_cuda_moe_set_debug_mm(true);
    ggml_backend_ptr reference_backend(ggml_backend_cuda_init(0));
    ggml_backend_ptr first_backend(ggml_backend_cuda_init(0));
    ggml_backend_ptr second_backend(ggml_backend_cuda_init(0));
    CHECK(reference_backend != nullptr && first_backend != nullptr && second_backend != nullptr);
    auto reference = build_active_grouped_dispatch_graph(reference_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), type, layout);
    auto first = build_active_grouped_dispatch_graph(first_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), type, layout);
    auto second = build_active_grouped_dispatch_graph(second_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), type, layout);
    initialize_active_grouped_dispatch_graphs({&reference, &first, &second});
    const auto disabled = candidate_snapshot(n_slots, nullptr, 0);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v1(reference_backend.get(), &disabled) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    register_active_grouped_dispatch(first_backend.get(), first, layout, n_slots);
    register_active_grouped_dispatch(second_backend.get(), second, layout, n_slots);
    check_active_grouped_contract(first_backend.get(), first, n_slots);
    check_active_grouped_contract(second_backend.get(), second, n_slots);

    std::vector<float> expected;
    std::vector<float> reference_output;
    std::vector<float> first_output;
    std::vector<float> second_output;
    const bool f3_skipped = layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE;
    for (int pass = 0; pass < 4; ++pass) {
        expected = run_active_grouped_dispatch(reference_backend.get(), reference, 0, false);
        const uint64_t expected_clock = 2 * (pass + 2);
        const auto current_first = run_active_grouped_dispatch(first_backend.get(), first, expected_clock, f3_skipped);
        const auto current_second = run_active_grouped_dispatch(second_backend.get(), second, expected_clock, f3_skipped);
        if (pass == 0) {
            reference_output = expected;
            first_output = current_first;
            second_output = current_second;
        } else {
            CHECK(expected == reference_output);
            CHECK(current_first == first_output && current_second == second_output);
        }
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
    CHECK(active_grouped_legacy_op_count(reference_backend.get()) == 4 * reference.banks.size());
    CHECK(active_grouped_legacy_op_count(first_backend.get()) == 0);
    CHECK(active_grouped_legacy_op_count(second_backend.get()) == 0);
    check_active_grouped_legacy_caches(reference_backend.get(), reference, false);
    for (ggml_tensor * bank : first.banks) {
        auto * context = ggml_cuda_moe_grouped_context_for_test(first_backend.get());
        CHECK(context != nullptr && !context->acquire_legacy_cache(bank));
    }
    for (ggml_tensor * bank : second.banks) {
        auto * context = ggml_cuda_moe_grouped_context_for_test(second_backend.get());
        CHECK(context != nullptr && !context->acquire_legacy_cache(bank));
    }

    if (type == GGML_TYPE_Q4_0 && layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP && n_slots == 12) {
        auto * context = ggml_cuda_moe_grouped_context_for_test(first_backend.get());
        CHECK(context != nullptr);

        std::shared_ptr<ggml_cuda_moe_graph_plan> failure_plan;
        ggml_cuda_moe_graph_execution failure_execution;
        CHECK(context->prepare_graph_execution(first.graph, 701, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &failure_plan, &failure_execution) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        cudaStream_t stream = nullptr;
        cudaStream_t wrong_stream = nullptr;
        CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_OK(cudaStreamCreateWithFlags(&wrong_stream, cudaStreamNonBlocking));
        CHECK(failure_execution.resolve_streams(candidate_test_graph_stream, stream));
        CHECK(context->begin_graph_dispatch(&failure_execution, true));
        ggml_cuda_moe_graph_binding binding;
        auto * failure_group = failure_execution.find_group(first.output, &binding);
        CHECK(failure_group != nullptr);
        CHECK(context->prepare_graph_group(failure_group, binding, first.output, stream) == GGML_CUDA_MOE_GROUPED_DECODE_READY);
        const auto failed_resource = failure_group->transaction.acquisition;
        failure_group->stream = wrong_stream;
        CHECK(!context->finish_graph_dispatch(&failure_execution));
        CHECK(!context->get_group_resources(failed_resource, nullptr));
        CUDA_OK(cudaStreamDestroy(wrong_stream));
        CUDA_OK(cudaStreamDestroy(stream));
        CHECK(run_active_grouped_dispatch(first_backend.get(), first, 2, false) == first_output);

        ggml_cuda_moe_candidate_group_key key;
        ggml_cuda_moe_grouped_acquisition resource;
        ggml_cuda_moe_grouped_transaction held;
        CHECK(context->find_down_group_key(first.down, &key));
        CHECK(context->acquire_group_resources(key, &resource));
        CHECK(context->begin_group_transaction(resource, &held));
        std::array<ggml_backend_moe_candidate_bank_v1, 3> replacement_banks = {};
        for (size_t i = 0; i < first.banks.size(); ++i) {
            replacement_banks[i] = {first.banks[i], first.roles[i], 0};
        }
        const ggml_backend_moe_candidate_group_v1 replacement_group = {
            replacement_banks.data(), static_cast<uint32_t>(first.banks.size()), layout, 0, 0,
        };
        const auto replacement_snapshot = candidate_snapshot(n_slots, &replacement_group, 1);
        std::atomic<bool> replacement_started{false};
        std::atomic<int32_t> replacement_result{GGML_BACKEND_MOE_CANDIDATE_REPLACE_ERROR};
        std::thread replacement([&]() {
            replacement_started.store(true, std::memory_order_release);
            replacement_result.store(context->replace(&replacement_snapshot), std::memory_order_release);
        });
        while (!replacement_started.load(std::memory_order_acquire)) {
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
    }
    ggml_backend_cuda_moe_set_debug_mm(old_debug_mm);
}

static void test_active_grouped_dispatch_decline() {
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
    const auto actual = run_active_grouped_dispatch(candidate_backend.get(), candidate, 0, false);
    CHECK(actual == expected);
    CHECK(active_grouped_legacy_op_count(reference_backend.get()) == reference.banks.size());
    CHECK(active_grouped_legacy_op_count(candidate_backend.get()) == candidate.banks.size());
    check_active_grouped_legacy_caches(reference_backend.get(), reference, false, false);
    check_active_grouped_legacy_caches(candidate_backend.get(), candidate, true, false);

    auto * context = ggml_cuda_moe_grouped_context_for_test(candidate_backend.get());
    ggml_cuda_moe_candidate_group_key key;
    CHECK(context != nullptr && context->find_down_group_key(candidate.down, &key));
    CHECK(!ggml_cuda_moe_grouped_context_test_access::has_device_resource(*context, key));
    ggml_backend_cuda_moe_set_debug_mm(old_debug_mm);
}

static void test_active_grouped_dispatch() {
    test_active_grouped_dispatch_case(
        GGML_TYPE_Q4_0, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 12);
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
    test_active_grouped_dispatch_decline();
    fprintf(stderr, "test-moe-cache: active grouped Q4 ordinary/F3 OK\n");
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
    CHECK(!run_active_grouped_dispatch(transition_backend.get(), grouped, 2).empty());

    auto * transition_context = ggml_cuda_moe_grouped_context_for_test(transition_backend.get());
    ggml_cuda_moe_candidate_group_key transition_key;
    ggml_cuda_moe_grouped_acquisition grouped_resource;
    CHECK(transition_context != nullptr && transition_context->find_down_group_key(grouped.down, &transition_key));
    CHECK(transition_context->acquire_group_resources(transition_key, &grouped_resource));
    CHECK(ggml_cuda_moe_grouped_context_test_access::has_device_resource(*transition_context, transition_key));

    std::shared_ptr<ggml_cuda_moe_graph_plan> transition_plan;
    ggml_cuda_moe_graph_execution transition_execution;
    CHECK(transition_context->prepare_graph_execution(
        registered_prefill.graph, 801, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &transition_plan, &transition_execution) ==
        GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
    CHECK(transition_execution.size() == 1);
    CHECK(transition_execution.resolve_streams(candidate_test_graph_stream, reinterpret_cast<void *>(uintptr_t{1})));
    CHECK(transition_context->begin_graph_dispatch(&transition_execution, true));
    CHECK(!transition_context->get_group_resources(grouped_resource, nullptr));
    CHECK(!ggml_cuda_moe_grouped_context_test_access::has_device_resource(*transition_context, transition_key));
    auto first_legacy = transition_context->acquire_legacy_cache(grouped.banks[0]);
    CHECK(first_legacy && first_legacy.get() != nullptr && first_legacy.acquisition().registered_source == 1 &&
        first_legacy.acquisition().group_index == transition_key.group_index);
    const uint64_t legacy_epoch = first_legacy.acquisition().group_authority_epoch;
    CHECK(!ggml_cuda_moe_grouped_context_test_access::has_device_resource(*transition_context, transition_key));
    first_legacy = {};
    CHECK(transition_context->finish_graph_dispatch(&transition_execution));

    const auto registered_mapped_first = run_cached_mmid_path_test(
        transition_backend.get(), reference_backend.get(), registered_prefill, registered_reference, mapped_ids);
    const auto registered_mapped_second = run_cached_mmid_path_test(
        transition_backend.get(), reference_backend.get(), registered_prefill, registered_reference, mapped_ids);
    CHECK(registered_mapped_first == registered_mapped_second);
    CHECK(!ggml_cuda_moe_grouped_context_test_access::has_device_resource(*transition_context, transition_key));
    auto repeated_legacy = transition_context->acquire_legacy_cache(grouped.banks[0]);
    CHECK(repeated_legacy && repeated_legacy.acquisition().group_authority_epoch == legacy_epoch);
    repeated_legacy = {};
    (void) run_cached_mmid_path_test(
        transition_backend.get(), reference_backend.get(), registered_prefill, registered_reference, {0, 1, 2, 3, 4, 5});
    CHECK(!ggml_cuda_moe_grouped_context_test_access::has_device_resource(*transition_context, transition_key));
    fprintf(stderr, "test-moe-cache: registered grouped prefill rollback OK\n");
    fprintf(stderr, "test-moe-cache: cached mapped prefill and overflow OK\n");
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
        ids_buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(device), 256);
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
    const int32_t experts[] = {1, 3};
    first.prefetch_legacy_siblings(first_gate, experts, 2, true, true);
    auto first_down = first.acquire_legacy_cache(down);
    CHECK(first_down && first_down.get() != nullptr && first_down.acquisition().registered_source == 1);
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
    ggml_cuda_moe_cache_stats(first_down.get(), &hits, &misses, &evictions);
    CHECK(hits == 0 && misses == 2 && evictions == 0);

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
    auto key = grouped_decode_key(registry, weights[n_banks - 1], ids, layout, group.n_banks);
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
    if (!pinned || auxiliary_scale) {
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
    CHECK(registry.finish_decode(decode, stream));

    CHECK(registry.prepare_decode(key, stream, &decode) == GGML_CUDA_MOE_GROUPED_DECODE_READY);
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
    test_grouped_decode_type(device, GGML_TYPE_Q4_0, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE);
    test_grouped_decode_type(device, GGML_TYPE_Q4_0, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP);
    test_grouped_decode_type(device, GGML_TYPE_Q4_K, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE);
    test_grouped_decode_type(device, GGML_TYPE_Q4_K, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP);
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
    test_candidate_producer();
    test_candidate_registry(registry_bench);
    test_legacy_owner_leases();
    test_grouped_context_resources();
    test_grouped_graph_preflight(registry_bench);
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
        return 0;
    }
    test_grouped_decode(dev);
    test_active_grouped_dispatch();

    // Toy parameters. Small enough to run in a few ms on any CUDA device,
    // large enough that LRU has work to do.
    constexpr int    N_EXPERTS = 64;
    constexpr int    N_SLOTS   = 16;
    constexpr size_t SLOT_BYTES = 1024;       // 256 floats
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
    CUDA_OK(cudaMalloc(&staging_dst, 4 * SLOT_BYTES));
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
    CUDA_OK(cudaMalloc(&staging_dst, 4 * SLOT_BYTES));
    std::vector<float> split_readback(6 * N_FLOATS);

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

        int slot_ids[4] = {-1, -1, -1, -1};
        int n_resident = 0;
        int n_wait_classes = 0;
        CHECK(ggml_cuda_moe_cache_prepare_split_staging(
            split_cache, split_srcs, 4, SLOT_BYTES, 1, slot_ids, nullptr,
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

        int slot_ids[N_SPLIT_SRCS] = {-1, -1, -1, -1, -1, -1};
        int32_t wait_classes[N_SPLIT_SRCS] = {-1, -1, -1, -1, -1, -1};
        uint32_t * stage_ready = nullptr;
        CUDA_OK(cudaMalloc(&stage_ready, 3 * sizeof(uint32_t)));
        int n_resident = 0;
        int n_wait_classes = 0;
        CHECK(ggml_cuda_moe_cache_prepare_split_staging(
            split_cache, split_srcs, N_SPLIT_SRCS, SLOT_BYTES, 1, slot_ids, wait_classes,
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
            split_cache, split_srcs, N_SPLIT_SRCS, SLOT_BYTES, 1, invalid_slots, invalid_wait_classes,
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
        split_cache, non_overflow_srcs, 2, SLOT_BYTES, 1, non_overflow_slots,
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
