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
    static constexpr size_t BUFFER_SIZE = 1024 * 1024;

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
        params.mem_size = 64 * ggml_tensor_overhead();
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

    ggml_cuda_moe_candidate_registry registry(&fixture.owner);
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
    ggml_cuda_moe_candidate_registry registry(&fixture.owner);

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

    ggml_cuda_moe_candidate_registry other_registry(&fixture.owner);
    snapshot.n_slots = 48;
    CHECK(other_registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(other_registry.state().generation == 1 && other_registry.state().n_slots == 48);
    CHECK(registry.state().n_slots == 12 && registry.state().generation > 1);

    fprintf(stderr, "test-moe-cache: registry OK\n");
}

int main(int argc, char ** argv) {
    const bool registry_only = argc == 2 && strcmp(argv[1], "--registry-only") == 0;
    const bool registry_bench = argc == 2 && strcmp(argv[1], "--registry-bench") == 0;
    test_candidate_producer();
    test_candidate_registry(registry_bench);
    if (registry_only || registry_bench) {
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
