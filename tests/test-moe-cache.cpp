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
//   6. Registry keys distinguish tensors that have the same name.
//   7. Cache-assisted staging orders coalesced D2D and H2D copies after prior compute.
//
// The workload is a synthetic stand-in for MoE routing: most tokens land on a
// hot subset of experts. Real Gemma 4 / Mixtral / Qwen3 routing has stronger
// locality than this, so a passing test here is a lower bound.

#include "../ggml/src/ggml-cuda/moe-cache.cuh"
#include "../ggml/src/ggml-backend-impl.h"
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
        params.mem_size = 512 * ggml_tensor_overhead();
        params.no_alloc = true;
        ctx = ggml_init(params);
        CHECK(ctx != nullptr);
    }

    ~candidate_test_fixture() {
        ggml_free(ctx);
        ggml_backend_buffer_free(buffer);
        free(storage);
    }

    ggml_tensor * tensor(enum ggml_type type, int n_dims, const int64_t * ne) {
        ggml_tensor * result = ggml_new_tensor(ctx, type, n_dims, ne);
        const size_t alignment = ggml_backend_buffer_get_alignment(buffer);
        next_offset = (next_offset + alignment - 1) / alignment * alignment;
        CHECK(next_offset + ggml_nbytes(result) <= BUFFER_SIZE);
        result->buffer = buffer;
        result->data = static_cast<uint8_t *>(storage) + next_offset;
        next_offset += ggml_nbytes(result);
        return result;
    }
};

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

static ggml_cgraph * candidate_graph(candidate_test_fixture & fixture, std::initializer_list<ggml_tensor *> nodes) {
    ggml_cgraph * graph = ggml_new_graph_custom(fixture.ctx, 32, false);
    CHECK(graph != nullptr);
    for (ggml_tensor * node : nodes) {
        ggml_graph_add_node(graph, node);
    }
    return graph;
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
    candidate_test_fixture fixture;
    const int64_t gate_ne[] = {256, 256, 4};
    const int64_t fused_ne[] = {256, 512, 4};
    const int64_t ids_ne[] = {2, 1};
    const int64_t prefill_ids_ne[] = {2, 4};
    const int64_t fused_bias_ne[] = {512, 4};

    ggml_tensor * fused_gate_up = fixture.tensor(GGML_TYPE_Q4_0, 3, fused_ne);
    ggml_tensor * fused_down = fixture.tensor(GGML_TYPE_Q4_0, 3, gate_ne);
    ggml_tensor * separate_gate = fixture.tensor(GGML_TYPE_Q4_K, 3, gate_ne);
    ggml_tensor * separate_up = fixture.tensor(GGML_TYPE_Q4_K, 3, gate_ne);
    ggml_tensor * separate_down = fixture.tensor(GGML_TYPE_Q4_K, 3, gate_ne);
    ggml_tensor * fused_bias = fixture.tensor(GGML_TYPE_F32, 2, fused_bias_ne);
    ggml_tensor * fused_ids = fixture.tensor(GGML_TYPE_I32, 2, ids_ne);
    ggml_tensor * fused_ids_other = fixture.tensor(GGML_TYPE_I32, 2, ids_ne);
    ggml_tensor * separate_ids = fixture.tensor(GGML_TYPE_I32, 2, ids_ne);
    ggml_tensor * prefill_ids = fixture.tensor(GGML_TYPE_I32, 2, prefill_ids_ne);

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

    ggml_tensor * fused_gate_up_node = candidate_mmid(fixture, fused_gate_up, fused_ids);
    ggml_tensor * fused_down_node = candidate_mmid(fixture, fused_down, fused_ids);
    ggml_tensor * separate_up_node = candidate_mmid(fixture, separate_up, separate_ids);
    ggml_tensor * separate_gate_node = candidate_mmid(fixture, separate_gate, separate_ids);
    ggml_tensor * separate_down_node = candidate_mmid(fixture, separate_down, separate_ids);
    ggml_cgraph * complete_graph = candidate_graph(fixture, {
        fused_gate_up_node, fused_down_node, separate_up_node, separate_gate_node, separate_down_node,
    });

    ggml_cuda_moe_graph_plan plan;
    ggml_cuda_moe_graph_execution execution;
    ggml_cuda_moe_graph_execution reused;
    registry.compile_graph_plan(complete_graph, 41, &plan, &execution);
    CHECK(plan.size() == 2 && execution.size() == 2);
    CHECK(plan.registry_generation() == 1 && plan.graph_uid() == 41 && plan.graph_node_count() == 5);
    ggml_cuda_moe_graph_binding binding;
    CHECK(execution.find(fused_gate_up_node, &binding));
    CHECK(binding.key.candidate.generation == 1 && binding.key.candidate.group_index == 0);
    CHECK(binding.key.ids.tensor == fused_ids && binding.key.ids.data == fused_ids->data && binding.key.ids.buffer == fused_ids->buffer);
    CHECK(binding.key.layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP && binding.key.n_banks == 2);
    CHECK(binding.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT && binding.bank_index == 0);
    CHECK(execution.find(fused_down_node, &binding) && binding.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT && binding.bank_index == 1);
    CHECK(execution.find(separate_up_node, &binding) && binding.key.candidate.group_index == 1 && binding.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT);
    CHECK(execution.find(separate_gate_node, &binding) && binding.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT);
    CHECK(execution.find(separate_down_node, &binding) && binding.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT && binding.key.n_banks == 3);
    CHECK(!execution.find(fused_bias, nullptr));

    CHECK(registry.bind_graph_plan(complete_graph, 41, true, plan, &reused));
    CHECK(reused.size() == 2 && reused.find(separate_down_node, nullptr));
    CHECK(!registry.bind_graph_plan(complete_graph, 41, false, plan, &reused) && reused.size() == 0);
    CHECK(registry.bind_graph_plan(complete_graph, 42, true, plan, &reused) && reused.size() == 2);
    CHECK(!registry.bind_graph_plan(complete_graph, 0, true, plan, &reused) && reused.size() == 0);
    ggml_cgraph * reordered_graph = candidate_graph(fixture, {
        fused_gate_up_node, fused_down_node, separate_gate_node, separate_up_node, separate_down_node,
    });
    CHECK(execution.find(fused_gate_up_node, &binding) && binding.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT);
    CHECK(execution.find(fused_down_node, &binding) && binding.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT);
    CHECK(!registry.bind_graph_plan(reordered_graph, 41, true, plan, &reused) && reused.size() == 0);
    CHECK(registry.bind_graph_plan(complete_graph, 41, true, plan, &reused));
    ggml_tensor * stale_gate_up_node = candidate_mmid(fixture, fused_gate_up, fused_ids);
    ggml_cgraph * stale_graph = candidate_graph(fixture, {
        stale_gate_up_node, fused_down_node, separate_up_node, separate_gate_node, separate_down_node,
    });
    CHECK(!registry.bind_graph_plan(stale_graph, 41, true, plan, &reused) && reused.size() == 0);
    CHECK(registry.bind_graph_plan(complete_graph, 41, true, plan, &reused));

    std::shared_ptr<ggml_cuda_moe_graph_plan> cached_plan;
    {
        ggml_cuda_moe_graph_execution prepared;
        CHECK(registry.prepare_graph_execution(complete_graph, 41, false, &cached_plan, &prepared) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(cached_plan != nullptr && prepared.size() == 2);
    }
    const ggml_cuda_moe_graph_plan * first_cached_plan = cached_plan.get();
    {
        ggml_cuda_moe_graph_execution prepared;
        CHECK(registry.prepare_graph_execution(complete_graph, 41, true, &cached_plan, &prepared) == GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(cached_plan.get() == first_cached_plan && prepared.find(separate_down_node, nullptr));
    }
    {
        ggml_cuda_moe_graph_execution prepared;
        CHECK(registry.prepare_graph_execution(complete_graph, 42, false, &cached_plan, &prepared) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(cached_plan.get() != first_cached_plan && prepared.size() == 2);
    }
    {
        std::weak_ptr<ggml_cuda_moe_graph_plan> lifetime;
        {
            std::shared_ptr<ggml_cuda_moe_graph_plan> lifetime_plan;
            ggml_cuda_moe_graph_execution prepared;
            CHECK(registry.prepare_graph_execution(complete_graph, 41, false, &lifetime_plan, &prepared) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
            lifetime = lifetime_plan;
            lifetime_plan.reset();
            CHECK(!lifetime.expired() && prepared.find(fused_down_node, nullptr));
        }
        CHECK(lifetime.expired());
    }
    {
        const ggml_cuda_moe_graph_plan * current_plan = cached_plan.get();
        ggml_cuda_moe_graph_execution prepared;
        CHECK(registry.prepare_graph_execution(reordered_graph, 41, true, &cached_plan, &prepared) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(cached_plan.get() != current_plan && prepared.find(fused_gate_up_node, nullptr));
    }
    {
        ggml_cuda_moe_graph_execution prepared;
        CHECK(registry.prepare_graph_execution(complete_graph, 41, true, &cached_plan, &prepared) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(prepared.size() == 2);
    }
    {
        const ggml_cuda_moe_graph_plan * current_plan = cached_plan.get();
        ggml_cuda_moe_graph_execution prepared;
        CHECK(registry.prepare_graph_execution(stale_graph, 41, true, &cached_plan, &prepared) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(cached_plan.get() != current_plan && prepared.find(stale_gate_up_node, nullptr));
    }
    {
        ggml_cuda_moe_graph_execution prepared;
        CHECK(registry.prepare_graph_execution(complete_graph, 41, true, &cached_plan, &prepared) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(prepared.size() == 2);
    }
    ggml_cuda_moe_grouped_context other_registry(&fixture.owner);
    CHECK(other_registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(!other_registry.bind_graph_plan(complete_graph, 41, true, plan, &reused) && reused.size() == 0);
    {
        std::shared_ptr<ggml_cuda_moe_graph_plan> foreign_plan = cached_plan;
        const ggml_cuda_moe_graph_plan * original_owner_plan = foreign_plan.get();
        ggml_cuda_moe_graph_execution prepared;
        CHECK(other_registry.prepare_graph_execution(complete_graph, 41, true, &foreign_plan, &prepared) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(foreign_plan.get() != original_owner_plan && prepared.size() == 2);
    }

    fused_gate_up_node->src[0] = separate_gate;
    CHECK(!registry.bind_graph_plan(complete_graph, 41, true, plan, &reused) && reused.size() == 0);
    fused_gate_up_node->src[0] = fused_gate_up;
    CHECK(registry.bind_graph_plan(complete_graph, 41, true, plan, &reused));

    CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(!registry.bind_graph_plan(complete_graph, 41, true, plan, &reused) && reused.size() == 0);
    const ggml_cuda_moe_graph_plan * generation_one_plan = cached_plan.get();
    {
        ggml_cuda_moe_graph_execution prepared;
        CHECK(registry.prepare_graph_execution(complete_graph, 41, true, &cached_plan, &prepared) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
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
    } while (registry.bind_graph_plan(complete_graph, 43, true, plan, &reused));
    CHECK(reused.size() == 0 && !replacement_done.load(std::memory_order_acquire));
    {
        ggml_cuda_moe_graph_execution prepared;
        CHECK(registry.prepare_graph_execution(complete_graph, 43, true, &cached_plan, &prepared) == GGML_CUDA_MOE_GRAPH_PREPARE_UNAVAILABLE);
        CHECK(cached_plan == nullptr && prepared.size() == 0);
    }
    registry.compile_graph_plan(complete_graph, 43, &plan, &execution);
    CHECK(plan.size() == 0 && execution.size() == 0 && plan.registry_generation() == 0);
    CHECK(registry.end_group_transaction(held_transaction));
    replacement_thread.join();
    CHECK(replacement_done.load(std::memory_order_acquire));
    CHECK(!registry.bind_graph_plan(complete_graph, 43, true, plan, &reused) && reused.size() == 0);

    registry.compile_graph_plan(complete_graph, 44, &plan, &execution);
    CHECK(execution.size() == 2);
    {
        ggml_cuda_moe_graph_execution prepared;
        CHECK(registry.prepare_graph_execution(complete_graph, 44, false, &cached_plan, &prepared) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(cached_plan != nullptr && prepared.size() == 2);
    }
    auto disabled = candidate_snapshot(12, nullptr, 0);
    CHECK(registry.replace(&disabled) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    registry.compile_graph_plan(complete_graph, 44, &plan, &execution);
    CHECK(plan.size() == 0 && execution.size() == 0 && plan.registry_generation() != 0);
    {
        ggml_cuda_moe_graph_execution prepared;
        CHECK(registry.prepare_graph_execution(complete_graph, 44, true, &cached_plan, &prepared) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(cached_plan != nullptr && prepared.size() == 0);
        const ggml_cuda_moe_graph_plan * disabled_plan = cached_plan.get();
        CHECK(registry.prepare_graph_execution(complete_graph, 44, true, &cached_plan, &prepared) == GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(cached_plan.get() == disabled_plan && prepared.size() == 0);
    }
    CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);

    ggml_tensor * mixed_gate_up_node = candidate_mmid(fixture, fused_gate_up, fused_ids);
    ggml_tensor * mixed_down_node = candidate_mmid(fixture, fused_down, fused_ids_other);
    ggml_cgraph * mixed_graph = candidate_graph(fixture, {mixed_gate_up_node, mixed_down_node});
    registry.compile_graph_plan(mixed_graph, 45, &plan, &execution);
    CHECK(plan.size() == 0 && execution.size() == 0);

    ggml_tensor * missing_gate_up_node = candidate_mmid(fixture, fused_gate_up, fused_ids);
    ggml_cgraph * missing_graph = candidate_graph(fixture, {missing_gate_up_node});
    registry.compile_graph_plan(missing_graph, 46, &plan, &execution);
    CHECK(plan.size() == 0 && execution.size() == 0);

    ggml_tensor * duplicate_gate_up_node = candidate_mmid(fixture, fused_gate_up, fused_ids);
    ggml_tensor * duplicate_gate_up_peer = candidate_mmid(fixture, fused_gate_up, fused_ids);
    ggml_tensor * duplicate_down_node = candidate_mmid(fixture, fused_down, fused_ids);
    ggml_cgraph * duplicate_graph = candidate_graph(fixture, {duplicate_gate_up_node, duplicate_gate_up_peer, duplicate_down_node});
    registry.compile_graph_plan(duplicate_graph, 47, &plan, &execution);
    CHECK(plan.size() == 0 && execution.size() == 0);

    ggml_tensor * prefill_gate_up_node = candidate_mmid(fixture, fused_gate_up, prefill_ids);
    ggml_tensor * prefill_down_node = candidate_mmid(fixture, fused_down, prefill_ids);
    ggml_cgraph * prefill_graph = candidate_graph(fixture, {prefill_gate_up_node, prefill_down_node});
    registry.compile_graph_plan(prefill_graph, 48, &plan, &execution);
    CHECK(plan.size() == 0 && execution.size() == 0);

    ggml_tensor * wrong_source_node = candidate_mmid(fixture, separate_gate, fused_ids);
    ggml_tensor * correct_down_node = candidate_mmid(fixture, fused_down, fused_ids);
    ggml_cgraph * wrong_source_graph = candidate_graph(fixture, {wrong_source_node, correct_down_node});
    registry.compile_graph_plan(wrong_source_graph, 49, &plan, &execution);
    CHECK(plan.size() == 0 && execution.size() == 0);

    ggml_tensor * auxiliary_gate_up_node = candidate_mmid(fixture, fused_gate_up, fused_ids);
    ggml_tensor * auxiliary_down_node = candidate_mmid(fixture, fused_down, fused_ids);
    const int64_t auxiliary_ne[] = {auxiliary_gate_up_node->ne[0], auxiliary_gate_up_node->ne[1], auxiliary_gate_up_node->ne[2]};
    ggml_tensor * auxiliary_out = fixture.tensor(GGML_TYPE_F32, 3, auxiliary_ne);
    auxiliary_out->op = GGML_OP_ADD_ID;
    auxiliary_out->src[0] = auxiliary_gate_up_node;
    auxiliary_out->src[1] = fused_bias;
    auxiliary_out->src[2] = fused_ids;
    auxiliary_out->flags |= GGML_TENSOR_FLAG_COMPUTE;
    ggml_cgraph * auxiliary_graph = candidate_graph(fixture, {auxiliary_gate_up_node, auxiliary_out, auxiliary_down_node});
    registry.compile_graph_plan(auxiliary_graph, 50, &plan, &execution);
    CHECK(plan.size() == 0 && execution.size() == 0);

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
    ggml_cgraph * auxiliary_registered_graph = candidate_graph(fixture, {auxiliary_gate_up_node, auxiliary_down_node});
    registry.compile_graph_plan(auxiliary_registered_graph, 51, &plan, &execution);
    CHECK(plan.size() == 0 && execution.size() == 0);

    if (benchmark) {
        CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        std::shared_ptr<ggml_cuda_moe_graph_plan> benchmark_plan;
        ggml_cuda_moe_graph_execution prepared;
        CHECK(registry.prepare_graph_execution(complete_graph, 52, false, &benchmark_plan, &prepared) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        const ggml_cuda_moe_graph_plan * stable_plan = benchmark_plan.get();
        constexpr uint32_t n_reuses = 200000;
        uint32_t reused_count = 0;
        const auto begin = std::chrono::steady_clock::now();
        for (uint32_t i = 0; i < n_reuses; ++i) {
            reused_count += registry.prepare_graph_execution(complete_graph, 53 + i, true, &benchmark_plan, &prepared) ==
                GGML_CUDA_MOE_GRAPH_PREPARE_REUSED ? 1 : 0;
        }
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin).count();
        CHECK(reused_count == n_reuses && benchmark_plan.get() == stable_plan && prepared.size() == 2);
        fprintf(stderr, "test-moe-cache: grouped graph plan %.1f ns/reuse with fresh UIDs, full scans=0/%u\n",
            static_cast<double>(elapsed) / n_reuses, n_reuses);
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

static void test_cached_mmid_fusion_decline() {
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
    bool all_sources_dispatched = true;
    for (ggml_tensor * weight : cuda_graph.cached_weights) {
        CHECK(weight->buffer != nullptr && ggml_backend_buft_is_cuda_moe_cached(weight->buffer->buft));
    }

    CHECK(ggml_backend_graph_compute(cpu_backend.get(), cpu_graph.graph) == GGML_STATUS_SUCCESS);
    CHECK(ggml_backend_graph_compute(cuda_backend.get(), cuda_graph.graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu_backend.get());
    ggml_backend_synchronize(cuda_backend.get());

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

    for (ggml_tensor * weight : cuda_graph.cached_weights) {
        ggml_cuda_moe_cache * cache = ggml_cuda_moe_cache_get_or_create_for_tensor(
            0, weight->data, weight->nb[2], 4, weight->ne[2], weight->name);
        CHECK(cache != nullptr);
        uint64_t hits = 0;
        uint64_t misses = 0;
        uint64_t evictions = 0;
        ggml_cuda_moe_cache_stats(cache, &hits, &misses, &evictions);
        if (hits + misses != 2) {
            fprintf(stderr, "FAIL cached fusion source %s: hits=%llu misses=%llu\n", weight->name,
                static_cast<unsigned long long>(hits), static_cast<unsigned long long>(misses));
            all_sources_dispatched = false;
        }
    }
    CHECK(all_sources_dispatched);

    ggml_cuda_moe_cache_free_all();
    ggml_backend_cuda_moe_set_cache_slots(old_slots);
    fprintf(stderr, "test-moe-cache: cached MMID F1-F5 decline OK\n");
}

int main(int argc, char ** argv) {
    const bool registry_only = argc == 2 && strcmp(argv[1], "--registry-only") == 0;
    const bool registry_bench = argc == 2 && strcmp(argv[1], "--registry-bench") == 0;
    const bool cached_fusion_only = argc == 2 && strcmp(argv[1], "--cached-fusion-only") == 0;
    test_candidate_producer();
    test_candidate_registry(registry_bench);
    test_grouped_context_resources();
    test_grouped_graph_preflight(registry_bench);
    if (registry_only || registry_bench) {
        return 0;
    }

    test_cached_mmid_fusion_decline();
    if (cached_fusion_only) {
        return 0;
    }

    // Toy parameters. Small enough to run in a few ms on any CUDA device,
    // large enough that LRU has work to do.
    constexpr int    N_EXPERTS = 64;
    constexpr int    N_SLOTS   = 16;
    constexpr size_t SLOT_BYTES = 1024;       // 256 floats
    constexpr int    N_FLOATS  = SLOT_BYTES / sizeof(float);
    constexpr int    N_ACCESS  = 4000;
    constexpr double ZIPF_S    = 1.1;          // mild skew
    constexpr unsigned SEED    = 0xC0FFEE;

    int dev = 0;
    CUDA_OK(cudaGetDevice(&dev));
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

    auto * registry_cache_a = ggml_cuda_moe_cache_get_or_create_for_tensor(
        dev, host_experts, SLOT_BYTES, 1, 1, "duplicate.name");
    auto * registry_cache_b = ggml_cuda_moe_cache_get_or_create_for_tensor(
        dev, host_experts + N_FLOATS, SLOT_BYTES, 1, 1, "duplicate.name");
    CHECK(registry_cache_a != nullptr);
    CHECK(registry_cache_b != nullptr);
    CHECK(registry_cache_a != registry_cache_b);
    ggml_cuda_moe_cache_free_all();

    CUDA_OK(cudaStreamDestroy(copy_stream));
    CUDA_OK(cudaFreeHost(host_experts));

    fprintf(stderr, "test-moe-cache: OK\n");
    return 0;
}
