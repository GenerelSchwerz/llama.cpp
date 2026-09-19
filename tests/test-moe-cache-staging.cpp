#include "test-moe-cache.h"

#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#endif

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

struct grouped_decode_fixture {
    static constexpr uint32_t N_EXPERTS = 8;
    static constexpr uint32_t N_SLOTS = 4;
    static constexpr size_t SOURCE_BYTES = 1024 * 1024;

    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t source_buffer = nullptr;
    ggml_backend_buffer_t ids_buffer = nullptr;
    ggml_context * ctx = nullptr;
    void * source_storage = nullptr;
    size_t source_storage_size = 0;
    uint32_t n_experts = N_EXPERTS;
    size_t source_offset = 0;

    explicit grouped_decode_fixture(
            int device,
            bool pinned = true,
            size_t source_bytes = SOURCE_BYTES,
            uint32_t n_experts = N_EXPERTS,
            size_t host_budget = 0) : n_experts(n_experts) {
        backend = ggml_backend_cuda_init(device);
        CHECK(backend != nullptr);
        auto buft = ggml_backend_cuda_moe_cached_bounded_buffer_type(host_budget);
        CHECK(buft != nullptr);
        if (pinned) {
            source_buffer = ggml_backend_buft_alloc_buffer(buft, source_bytes);
        } else {
            source_storage = ggml_aligned_malloc(source_bytes);
            source_storage_size = source_bytes;
            CHECK(source_storage != nullptr);
            source_buffer = ggml_backend_cuda_moe_cached_buffer_from_host_ptr(buft, source_storage, source_bytes);
        }
        ggml_backend_cuda_moe_cached_free_buffer_type(buft);
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
        if (source_storage != nullptr) {
            ggml_aligned_free(source_storage, source_storage_size);
        }
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

void test_pageable_staging_pipeline(int device) {
    const bool old_debug_mm = ggml_backend_cuda_moe_get_debug_mm();
    ggml_backend_cuda_moe_set_debug_mm(true);
    constexpr size_t HOST_BUDGET = 262144;
    constexpr int N_SLOTS = 6;
    grouped_decode_fixture fixture(device, false, grouped_decode_fixture::SOURCE_BYTES,
        grouped_decode_fixture::N_EXPERTS, HOST_BUDGET);
    ggml_tensor * gate_up = fixture.weight(GGML_TYPE_F32, 64, 128);
    ggml_tensor * down = fixture.weight(GGML_TYPE_F32, 64, 64);
    CHECK(gate_up->nb[2] > down->nb[2]);
    for (uint32_t expert = 0; expert < fixture.n_experts; ++expert) {
        memset(static_cast<char *>(gate_up->data) + expert * gate_up->nb[2], 17 + expert, gate_up->nb[2]);
        memset(static_cast<char *>(down->data) + expert * down->nb[2], 41 + expert, down->nb[2]);
    }
    const ggml_backend_moe_candidate_group_v2 group = {
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY, 0, 0,
    };
    const ggml_backend_moe_candidate_tensor_v2 tensors[] = {
        {gate_up, 0, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE, 0, 0},
        {down, 0, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE, 0, 0},
    };
    const auto snapshot = candidate_snapshot_v2(N_SLOTS, &group, 1, tensors, 2);
    auto buft = ggml_backend_buffer_get_type(fixture.source_buffer);
    CHECK(ggml_backend_cuda_moe_cached_configure_sources(buft, &snapshot));

    auto make_cache = [&](ggml_cuda_moe_staging_failure_for_test failure = GGML_CUDA_MOE_STAGING_FAIL_NONE) {
        auto * cache = ggml_cuda_moe_cache_init(device, gate_up->nb[2], N_SLOTS);
        CHECK(cache != nullptr);
        ggml_cuda_moe_cache_fail_staging_for_test(cache, failure);
        CHECK(ggml_cuda_moe_cache_set_source_for_test(cache, gate_up));
        return cache;
    };
    const auto source = [&](uint32_t expert) {
        return static_cast<const char *>(gate_up->data) + expert * gate_up->nb[2];
    };
    const auto check_slot = [&](ggml_cuda_moe_cache * cache, int slot, uint32_t expert) {
        std::vector<uint8_t> actual(gate_up->nb[2]);
        CUDA_OK(cudaMemcpy(actual.data(), ggml_cuda_moe_cache_slot_ptr(cache, slot), actual.size(), cudaMemcpyDeviceToHost));
        CHECK(memcmp(actual.data(), source(expert), actual.size()) == 0);
    };

    auto * cache = make_cache();
    const auto initial = ggml_cuda_moe_cache_staging_state_for_test(cache, false);
    CHECK(initial.tiles == 2 && initial.host_optional_bytes == gate_up->nb[2]);
    int slots[4];
    for (uint32_t expert = 0; expert < 4; ++expert) {
        slots[expert] = ggml_cuda_moe_cache_acquire(
            cache, source(expert), gate_up->nb[2], ggml_cuda_moe_cache_copy_stream(cache), false, false, true);
        CHECK(slots[expert] >= 0);
    }
    auto state = ggml_cuda_moe_cache_staging_state_for_test(cache, false);
    CHECK(state.pipeline_tiles == 4 && state.tile_wait_calls >= 2 && state.pending == 2);
    CHECK(state.pre_sync_calls == 0 && state.post_sync_calls == 0 && state.upload_errors == 0);
    CUDA_OK(cudaStreamSynchronize(ggml_cuda_moe_cache_copy_stream(cache)));
    for (uint32_t expert = 0; expert < 4; ++expert) {
        check_slot(cache, slots[expert], expert);
    }
    ggml_cuda_moe_cache_release_slots(cache, slots, 4);

    const int decode_slot = ggml_cuda_moe_cache_acquire(
        cache, source(4), gate_up->nb[2], ggml_cuda_moe_cache_copy_stream(cache), true, false, false);
    CHECK(decode_slot >= 0);
    const auto decode_state = ggml_cuda_moe_cache_staging_state_for_test(cache, true);
    CHECK(decode_state.pipeline_tiles == 0 && decode_state.pending == 0);
    CHECK(decode_state.pre_sync_calls == 1 && decode_state.post_sync_calls == 1 && decode_state.upload_errors == 0);
    check_slot(cache, decode_slot, 4);
    ggml_cuda_moe_cache_free(cache);

    for (auto failure : {GGML_CUDA_MOE_STAGING_FAIL_GROWTH, GGML_CUDA_MOE_STAGING_FAIL_EVENT_CREATE}) {
        cache = make_cache(failure);
        state = ggml_cuda_moe_cache_staging_state_for_test(cache, false);
        CHECK(state.tiles == 1 && state.host_optional_bytes == 0 && state.host_staging_bytes <= state.host_limit);
        for (uint32_t expert = 0; expert < 2; ++expert) {
            const int slot = ggml_cuda_moe_cache_acquire(
                cache, source(expert), gate_up->nb[2], ggml_cuda_moe_cache_copy_stream(cache), false, false, false);
            CHECK(slot >= 0);
            check_slot(cache, slot, expert);
        }
        ggml_cuda_moe_cache_free(cache);
    }

    for (auto failure : {GGML_CUDA_MOE_STAGING_FAIL_WAIT, GGML_CUDA_MOE_STAGING_FAIL_SUBMIT, GGML_CUDA_MOE_STAGING_FAIL_RECORD}) {
        cache = make_cache();
        const size_t staging_bytes = ggml_cuda_moe_cache_staging_state_for_test(cache, false).host_staging_bytes;
        for (uint32_t expert = 0; expert < 2; ++expert) {
            CHECK(ggml_cuda_moe_cache_acquire(
                cache, source(expert), gate_up->nb[2], ggml_cuda_moe_cache_copy_stream(cache), false, false, false) >= 0);
        }
        ggml_cuda_moe_cache_fail_staging_for_test(cache, failure);
        CHECK(ggml_cuda_moe_cache_acquire(
            cache, source(2), gate_up->nb[2], ggml_cuda_moe_cache_copy_stream(cache), false, false, false) < 0);
        state = ggml_cuda_moe_cache_staging_state_for_test(cache, false);
        CHECK(state.failed && state.pending == 0 && state.upload_errors == 1);
        ggml_cuda_moe_cache_fail_staging_for_test(cache, GGML_CUDA_MOE_STAGING_FAIL_NONE);
        CHECK(ggml_cuda_moe_cache_acquire(
            cache, source(0), gate_up->nb[2], ggml_cuda_moe_cache_copy_stream(cache), false, false, false) < 0);
        CHECK(ggml_cuda_moe_cache_acquire(
            cache, source(3), gate_up->nb[2], ggml_cuda_moe_cache_copy_stream(cache), false, false, false) < 0);
        ggml_cuda_moe_cache_free(cache);
        cache = make_cache();
        CHECK(ggml_cuda_moe_cache_staging_state_for_test(cache, false).host_staging_bytes == staging_bytes);
        ggml_cuda_moe_cache_free(cache);
    }

    cache = ggml_cuda_moe_cache_init(device, gate_up->nb[2], 2);
    CHECK(cache != nullptr && ggml_cuda_moe_cache_set_source_for_test(cache, gate_up));
    for (uint32_t expert = 0; expert < 2; ++expert) {
        CHECK(ggml_cuda_moe_cache_acquire(
            cache, source(expert), gate_up->nb[2], ggml_cuda_moe_cache_copy_stream(cache), false, false, false) >= 0);
    }
    cudaStream_t compute_stream = nullptr;
    void * miss_dst = nullptr;
    CUDA_OK(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));
    CUDA_OK(cudaMalloc(&miss_dst, 6 * gate_up->nb[2]));
    const void * split_sources[8];
    for (uint32_t expert = 0; expert < 8; ++expert) {
        split_sources[expert] = source(expert);
    }
    int split_slots[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    int n_resident = 0;
    int n_wait_classes = 0;
    ggml_cuda_moe_cache_fail_staging_for_test(cache, GGML_CUDA_MOE_STAGING_FAIL_SUBMIT);
    CHECK(!ggml_cuda_moe_cache_prepare_split_staging(
        cache, split_sources, 8, gate_up->nb[2], 0, 1, split_slots, nullptr,
        &n_resident, miss_dst, nullptr, 0, &n_wait_classes, compute_stream, false));
    CHECK(ggml_cuda_moe_cache_staging_state_for_test(cache, false).failed);
    CUDA_OK(cudaFree(miss_dst));
    CUDA_OK(cudaStreamDestroy(compute_stream));
    ggml_cuda_moe_cache_free(cache);
    ggml_backend_cuda_moe_set_debug_mm(old_debug_mm);
    fprintf(stderr, "test-moe-cache: pageable two-tile staging pipeline OK\n");
}

void test_owner_legacy_cache(int device) {
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
    first.prefetch_legacy_siblings(first_gate, experts, 2, true);
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
            &n_resident, prefetch_staging, prefetch_stage_ready, 3, &n_wait_classes, prefetch_compute_stream, false);
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
    second.prefetch_legacy_siblings(second_up, experts, 2, true);
    auto second_separate_gate = second.acquire_legacy_cache(gate);
    auto second_separate_down = second.acquire_legacy_cache(separate_down);
    CHECK(second_separate_gate && second_separate_down);
    CHECK(second_separate_gate.acquisition().role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT);
    CHECK(second_separate_down.acquisition().role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT);
    ggml_cuda_moe_cache_stats(second_separate_gate.get(), &hits, &misses, &evictions);
    CHECK(hits == 0 && misses == 2 && evictions == 0);
    ggml_cuda_moe_cache_stats(second_separate_down.get(), &hits, &misses, &evictions);
    CHECK(hits == 0 && misses == 2 && evictions == 0);

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
        bool auxiliary_scale = false,
        size_t host_budget = 0,
        bool mixed = false) {
    grouped_decode_fixture fixture(device, pinned, grouped_decode_fixture::SOURCE_BYTES, grouped_decode_fixture::N_EXPERTS, host_budget);
    ggml_tensor * registration_prefix = nullptr;
#ifdef __linux__
    if (pinned && host_budget == 65536) {
        const long page = sysconf(_SC_PAGESIZE);
        CHECK(page > 128);
        const uintptr_t base = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(fixture.source_buffer));
        fixture.source_offset = 2 * static_cast<size_t>(page) - base % page - 128;
        registration_prefix = fixture.scale();
        ggml_set_name(registration_prefix, "test.registration_prefix");
        fixture.source_offset += 64 - ggml_nbytes(registration_prefix);
    }
#endif
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
    const uint32_t ungated_roles[] = {
        GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT,
        GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT,
    };
    for (uint32_t bank = 0; bank < n_banks; ++bank) {
        const int64_t ne0 = type == GGML_TYPE_Q4_K ? 256 : type == GGML_TYPE_Q4_0 ? 32 : 64;
        const int64_t ne1 = layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP && bank == 0 ? 2 * ne0 : ne0;
        weights[bank] = fixture.weight(type, ne0, ne1);
        banks[bank].tensor = weights[bank];
        banks[bank].role = layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE ? separate_roles[bank] :
            layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_UNGATED ? ungated_roles[bank] : fused_roles[bank];
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
    if (registration_prefix != nullptr) {
        // The prefix pins the first 64 bytes of expert 0 in a separate registration.
        CHECK(static_cast<const char *>(weights[0]->data) - static_cast<const char *>(registration_prefix->data) == 64);
        CHECK(weights[0]->nb[2] > 64);
    }
    if (auxiliary_scale) {
        banks[n_banks] = {fixture.scale(), GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_SCALE, 0};
    }
    ggml_backend_moe_candidate_group_v1 group = {};
    group.banks = banks.data();
    group.n_banks = n_banks + auxiliary_scale;
    group.layout = layout;
    std::array<ggml_backend_moe_candidate_group_v1, 2> groups = {group, group};
    std::array<ggml_backend_moe_candidate_bank_v1, 3> extra_banks = {};
    if (mixed) {
        CHECK(host_budget != 0 && !auxiliary_scale);
        fixture.source_offset += 4096;
        for (uint32_t bank = 0; bank < n_banks; ++bank) {
            auto * extra = fixture.weight(type, weights[bank]->ne[0], weights[bank]->ne[1]);
            extra_banks[bank] = {extra, banks[bank].role, 0};
            for (uint32_t expert = 0; expert < fixture.n_experts; ++expert) {
                memset(static_cast<char *>(extra->data) + expert * extra->nb[2], 64 + 17 * bank + expert, extra->nb[2]);
            }
        }
        groups[1].banks = extra_banks.data();
    }
    const auto snapshot = candidate_snapshot(n_slots, groups.data(), mixed ? 2 : 1);
    if (host_budget != 0) {
        std::vector<ggml_backend_moe_candidate_group_v2> source_groups;
        std::vector<ggml_backend_moe_candidate_tensor_v2> sources;
        if (registration_prefix != nullptr) {
            sources.push_back({registration_prefix, UINT32_MAX, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_SCALE,
                GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_UNCLASSIFIED, 0, 0});
        }
        for (uint32_t group_index = 0; group_index < snapshot.n_groups; ++group_index) {
            source_groups.push_back({layout, GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY, 0, 0});
            for (uint32_t i = 0; i < groups[group_index].n_banks; ++i) {
                const auto & bank = groups[group_index].banks[i];
                sources.push_back({bank.tensor, group_index, bank.role,
                    i < n_banks ? GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE : GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_OUTPUT_SCALE, 0, 0});
            }
        }
        const auto source_snapshot = candidate_snapshot_v2(
            n_slots, source_groups.data(), source_groups.size(), sources.data(), sources.size());
        CHECK(ggml_backend_cuda_moe_cached_configure_sources(ggml_backend_buffer_get_type(fixture.source_buffer), &source_snapshot));
        int read_only_supported = 0;
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11010
        if (!pinned) {
            CUDA_OK(cudaDeviceGetAttribute(&read_only_supported, cudaDevAttrHostRegisterReadOnlySupported, device));
        }
#endif
        for (uint32_t group_index = 0; group_index < snapshot.n_groups; ++group_index) {
            const bool budget_admits = mixed ? group_index == 0 : host_budget >= 65536;
            const bool direct = budget_admits && (pinned || read_only_supported);
            for (uint32_t bank = 0; bank < n_banks; ++bank) {
                cudaPointerAttributes attributes = {};
                CUDA_OK(cudaPointerGetAttributes(&attributes, groups[group_index].banks[bank].tensor->data));
                CHECK(attributes.type == (direct ? cudaMemoryTypeHost : cudaMemoryTypeUnregistered));
            }
        }
    }
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
    const auto check_payload = [&](const int32_t * experts) {
        std::array<int32_t, 4> slots = {};
        CUDA_OK(cudaMemcpy(slots.data(), decode.remapped_ids, sizeof(slots), cudaMemcpyDeviceToHost));
        for (uint32_t bank = 0; bank < decode.n_banks; ++bank) {
            const auto * tensor = decode.banks[bank].tensor;
            std::vector<uint8_t> bytes(tensor->nb[2]);
            for (uint32_t route = 0; route < slots.size(); ++route) {
                CHECK(slots[route] >= 0 && static_cast<uint32_t>(slots[route]) < decode.n_slots);
                CUDA_OK(cudaMemcpy(bytes.data(), static_cast<const char *>(decode.banks[bank].data) + slots[route] * tensor->nb[2],
                    bytes.size(), cudaMemcpyDeviceToHost));
                CHECK(memcmp(bytes.data(), static_cast<const char *>(tensor->data) + experts[route] * tensor->nb[2], bytes.size()) == 0);
            }
        }
    };
    CHECK(registry.prepare_decode(wrong_key, stream, &decode) == GGML_CUDA_MOE_GROUPED_DECODE_FALLBACK);

    const int32_t first_ids[] = {3, 1, 3, 6};
    CUDA_OK(cudaMemcpyAsync(ids->data, first_ids, sizeof(first_ids), cudaMemcpyHostToDevice, stream));
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
    }
    check_payload(first_ids);
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
    check_payload(second_ids);
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
    check_payload(second_ids);
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
    const int32_t full_miss_ids[] = {0, 3, 4, 5};
    if (host_budget != 0) {
        for (int32_t expert : full_miss_ids) {
            CHECK(ggml_cuda_moe_grouped_context_test_access::device_slot_for_expert(registry, key.candidate, expert) == -1);
        }
        CUDA_OK(cudaMemcpyAsync(ids->data, full_miss_ids, sizeof(full_miss_ids), cudaMemcpyHostToDevice, stream));
        for (int attempt = 0; attempt < 2; ++attempt) {
            CHECK(registry.prepare_decode(key, stream, &decode) == GGML_CUDA_MOE_GROUPED_DECODE_READY);
            CUDA_OK(cudaStreamSynchronize(stream));
            check_payload(full_miss_ids);
            CHECK(registry.finish_decode(decode, stream));
        }
    }
    if (mixed) {
        const auto extra_key = grouped_decode_key(registry, extra_banks[n_banks - 1].tensor, ids, layout, n_banks);
        for (const int32_t * routes : {first_ids, second_ids, second_ids, full_miss_ids, full_miss_ids}) {
            CUDA_OK(cudaMemcpyAsync(ids->data, routes, sizeof(first_ids), cudaMemcpyHostToDevice, stream));
            CHECK(registry.prepare_decode(extra_key, stream, &decode) == GGML_CUDA_MOE_GROUPED_DECODE_READY);
            CUDA_OK(cudaStreamSynchronize(stream));
            check_payload(routes);
            CHECK(registry.finish_decode(decode, stream));
        }
        CUDA_OK(cudaStreamSynchronize(stream));
    }
    if (host_budget != 0) {
        ggml_cuda_moe_grouped_context legacy(ggml_backend_get_device(fixture.backend), device);
        const auto disabled = candidate_snapshot(n_slots, nullptr, 0);
        CHECK(legacy.replace(&disabled) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        for (uint32_t group_index = 0; group_index < snapshot.n_groups; ++group_index) {
            for (uint32_t bank = 0; bank < n_banks; ++bank) {
                const auto * tensor = groups[group_index].banks[bank].tensor;
                auto lease = legacy.acquire_legacy_cache(tensor);
                CHECK(lease && lease.get() != nullptr);
                std::vector<uint8_t> bytes(tensor->nb[2]);
                const std::array<int32_t, 8> split_routes = {3, 1, 6, 2, 7, 0, 4, 5};
                std::array<const void *, 8> split_sources = {};
                for (size_t i = 0; i < split_sources.size(); ++i) {
                    split_sources[i] = static_cast<const char *>(tensor->data) + split_routes[i] * tensor->nb[2];
                }
                constexpr size_t padding = 16;
                void * staging = nullptr;
                CUDA_OK(cudaMalloc(&staging, split_sources.size() * tensor->nb[2] + padding));
                for (bool overlap : {false, true}) {
                    if (overlap && !ggml_cuda_moe_cache_can_overlap_staging(lease.get())) {
                        continue;
                    }
                    uint32_t * ready = nullptr;
                    if (overlap) {
                        CUDA_OK(cudaMalloc(&ready, 3 * sizeof(uint32_t)));
                    }
                    CUDA_OK(cudaMemsetAsync(staging, 0xff, split_sources.size() * tensor->nb[2] + padding, stream));
                    std::array<int, 8> slots = {};
                    std::array<int32_t, 8> waits = {};
                    int resident = 0;
                    int wait_classes = 0;
                    CHECK(ggml_cuda_moe_cache_prepare_split_staging(lease.get(), split_sources.data(), split_sources.size(),
                        tensor->nb[2], padding, 1, slots.data(), overlap ? waits.data() : nullptr, &resident,
                        staging, ready, overlap ? 3 : 0, &wait_classes, stream, false));
                    CHECK(resident == 4 && wait_classes == (overlap ? 3 : 1));
                    CHECK(ggml_cuda_moe_cache_finish_split_staging(lease.get(), stream));
                    CUDA_OK(cudaStreamSynchronize(stream));
                    size_t miss = 0;
                    for (size_t i = 0; i < split_sources.size(); ++i) {
                        const void * source = slots[i] >= 0 ? ggml_cuda_moe_cache_slot_ptr(lease.get(), slots[i]) :
                            static_cast<char *>(staging) + miss++ * tensor->nb[2];
                        CUDA_OK(cudaMemcpy(bytes.data(), source, bytes.size(), cudaMemcpyDeviceToHost));
                        CHECK(memcmp(bytes.data(), split_sources[i], bytes.size()) == 0);
                        if (overlap) {
                            CHECK(waits[i] == (slots[i] >= 0 ? 1 : 2));
                        }
                    }
                    CHECK(miss == 4);
                    std::array<uint8_t, padding> tail = {};
                    CUDA_OK(cudaMemcpy(tail.data(), static_cast<char *>(staging) + miss * tensor->nb[2], tail.size(), cudaMemcpyDeviceToHost));
                    CHECK(std::all_of(tail.begin(), tail.end(), [](uint8_t value) { return value == 0; }));
                    if (overlap) {
                        std::array<uint32_t, 3> flags = {};
                        CUDA_OK(cudaMemcpy(flags.data(), ready, sizeof(flags), cudaMemcpyDeviceToHost));
                        CHECK((flags == std::array<uint32_t, 3>{1, 1, 0}));
                        CUDA_OK(cudaFree(ready));
                    }
                    CHECK(ggml_cuda_moe_cache_release_split_slots(lease.get(), slots.data(), slots.size(), stream));
                    CUDA_OK(cudaStreamSynchronize(stream));
                }
                CUDA_OK(cudaFree(staging));
                for (int32_t expert : {3, 1, 6, 2, 7, 0}) {
                    const auto * source = static_cast<const char *>(tensor->data) + expert * tensor->nb[2];
                    const int slot = ggml_cuda_moe_cache_acquire(
                        lease.get(), source, tensor->nb[2], ggml_cuda_moe_cache_copy_stream(lease.get()), true, false, false);
                    CHECK(slot >= 0);
                    CUDA_OK(cudaStreamSynchronize(ggml_cuda_moe_cache_copy_stream(lease.get())));
                    CUDA_OK(cudaMemcpy(bytes.data(), ggml_cuda_moe_cache_slot_ptr(lease.get(), slot), bytes.size(), cudaMemcpyDeviceToHost));
                    CHECK(memcmp(bytes.data(), source, bytes.size()) == 0);
                }
            }
        }
        fprintf(stderr, "test-moe-cache: bounded sources allocated=%d budget=%zu mixed=%d registration_boundary=%d grouped/ordinary/split exact OK\n",
            pinned, host_budget, mixed, registration_prefix != nullptr);
    }
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

void test_grouped_decode(int device) {
    test_grouped_decode_independent_rows(device);
    test_grouped_decode_type(device, GGML_TYPE_Q4_0, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE);
    test_grouped_decode_type(device, GGML_TYPE_Q4_0, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP);
    test_grouped_decode_type(device, GGML_TYPE_Q4_K, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE);
    test_grouped_decode_type(device, GGML_TYPE_Q4_K, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP);
    test_grouped_decode_type(device, GGML_TYPE_Q4_0, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, true, 12);
    test_grouped_decode_type(device, GGML_TYPE_Q4_0, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, false);
    for (bool allocated : {false, true}) {
        for (size_t budget : {size_t{16384}, size_t{20480}, size_t{65536}}) {
            test_grouped_decode_type(device, GGML_TYPE_Q4_0, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, allocated, 4, false, budget);
        }
        test_grouped_decode_type(device, GGML_TYPE_Q4_0, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, allocated, 4, false, 53248, true);
    }
    for (ggml_type type : {GGML_TYPE_BF16, GGML_TYPE_NVFP4}) {
        for (uint32_t layout : {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP}) {
            for (uint32_t n_slots : {12u, 48u}) {
                test_grouped_decode_type(device, type, layout, true, n_slots);
            }
        }
    }
    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_NVFP4}) {
        for (uint32_t layout : {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_UNGATED}) {
            test_grouped_decode_type(device, type, layout, false, 12);
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

static void grouped_decode_benchmark_case(int device, const grouped_decode_bench_spec & spec, uint32_t n_slots, size_t host_budget = 0) {
    grouped_decode_fixture fixture(device, true, grouped_decode_source_bytes(spec), spec.n_experts, host_budget);
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
    if (host_budget != 0) {
        const ggml_backend_moe_candidate_group_v2 source_group = {spec.layout, GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY, 0, 0};
        std::array<ggml_backend_moe_candidate_tensor_v2, 3> sources = {};
        for (uint32_t bank = 0; bank < spec.n_banks; ++bank) {
            sources[bank] = {weights[bank], 0, spec.roles[bank], GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE, 0, 0};
        }
        const auto source_snapshot = candidate_snapshot_v2(n_slots, &source_group, 1, sources.data(), spec.n_banks);
        CHECK(ggml_backend_cuda_moe_cached_configure_sources(ggml_backend_buffer_get_type(fixture.source_buffer), &source_snapshot));
    }
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

void test_grouped_decode_benchmark(int device) {
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

void test_pageable_cache_fallback() {
    CHECK(getenv("GGML_CUDA_NO_PINNED") != nullptr);
    const auto buft = ggml_backend_cuda_moe_cached_buffer_type();
    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer(buft, 1024));
    CHECK(buffer != nullptr && ggml_backend_buffer_get_type(buffer.get()) == buft);
    CHECK(!ggml_backend_buffer_is_host(buffer.get()));
    CHECK(ggml_backend_dev_supports_buft(ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), 0), buft));
    ggml_backend_buffer_clear(buffer.get(), 0x5a);
    auto * data = static_cast<const uint8_t *>(ggml_backend_buffer_get_base(buffer.get()));
    CHECK(std::all_of(data, data + 1024, [](uint8_t value) { return value == 0x5a; }));
    auto * cache = ggml_cuda_moe_cache_init(0, 256, 2);
    CHECK(cache != nullptr);
    cudaStream_t stream = nullptr;
    CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const int slot = ggml_cuda_moe_cache_acquire(cache, data, 256, stream, false, false, false);
    CHECK(slot >= 0 && ggml_cuda_moe_cache_acquire(cache, data, 256, stream, false, false, false) == slot);
    CUDA_OK(cudaStreamSynchronize(stream));
    std::array<uint8_t, 256> actual;
    CUDA_OK(cudaMemcpy(actual.data(), ggml_cuda_moe_cache_slot_ptr(cache, slot), actual.size(), cudaMemcpyDeviceToHost));
    CHECK(std::all_of(actual.begin(), actual.end(), [](uint8_t value) { return value == 0x5a; }));
    uint64_t hits = 0, misses = 0, evictions = 0;
    ggml_cuda_moe_cache_stats(cache, &hits, &misses, &evictions);
    CHECK(hits == 1 && misses == 1 && evictions == 0);
    CUDA_OK(cudaStreamDestroy(stream));
    ggml_cuda_moe_cache_free(cache);
    fprintf(stderr, "test-moe-cache: pageable allocation retains CUDA cache identity and exact cache fill OK\n");
}

void test_grouped_staging_benchmark() {
    int device = 0;
    CUDA_OK(cudaGetDevice(&device));
    const grouped_decode_bench_spec spec = {
        "staged-q4_k-separate", GGML_TYPE_Q4_K, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 512, 80, 3,
        {2048, 2048, 768}, {768, 768, 2048},
        {GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT,
            GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT},
    };
    grouped_decode_benchmark_case(device, spec, 80, 64 * 1024 * 1024);
}

void test_cache_slots_and_staging(int dev) {
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

    auto * cache = ggml_cuda_moe_cache_init(dev, SLOT_BYTES, N_SLOTS);
    CHECK(cache != nullptr);

    std::mt19937 rng(SEED);
    std::vector<int> trace; trace.reserve(N_ACCESS);

    for (int t = 0; t < N_ACCESS; ++t) {
        int eid = sample_zipf(rng, N_EXPERTS, ZIPF_S);
        const void * src = host_experts + (size_t)eid * N_FLOATS;

        int slot = ggml_cuda_moe_cache_acquire(cache, src, SLOT_BYTES, copy_stream, false, false, false);
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
        int slot = ggml_cuda_moe_cache_acquire(cache, src, SLOT_BYTES, copy_stream, false, false, false);
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

    auto * batch_cache = ggml_cuda_moe_cache_init(dev, SLOT_BYTES, 4);
    CHECK(batch_cache != nullptr);
    int batch_slots[4];
    for (int eid = 0; eid < 4; ++eid) {
        batch_slots[eid] = ggml_cuda_moe_cache_acquire(
            batch_cache, host_experts + (size_t) eid * N_FLOATS,
            SLOT_BYTES, copy_stream, true, false, true);
        CHECK(batch_slots[eid] >= 0);
    }
    CHECK(ggml_cuda_moe_cache_acquire(
        batch_cache, host_experts + 4 * N_FLOATS,
        SLOT_BYTES, copy_stream, true, false, false) < 0);
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

    auto * staging_cache = ggml_cuda_moe_cache_init(dev, SLOT_BYTES, 2);
    CHECK(staging_cache != nullptr);
    cudaStream_t staging_copy_stream = ggml_cuda_moe_cache_copy_stream(staging_cache);
    CHECK(staging_copy_stream != nullptr);

    const void * resident_src_0 = host_experts;
    const void * resident_src_1 = host_experts + N_FLOATS;
    CHECK(ggml_cuda_moe_cache_acquire(
        staging_cache, resident_src_0, SLOT_BYTES, staging_copy_stream, false, false, false) == 0);
    CHECK(ggml_cuda_moe_cache_acquire(
        staging_cache, resident_src_1, SLOT_BYTES, staging_copy_stream, false, false, false) == 1);
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
            staging_cache, staging_srcs, 4, SLOT_BYTES, staging_dst, compute_stream, false));
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
        staging_cache, staging_srcs, 4, SLOT_BYTES + 1, staging_dst, compute_stream, false));
    CHECK(cudaStreamQuery(staging_copy_stream) == cudaSuccess);
    CHECK(cudaStreamQuery(compute_stream) == cudaSuccess);

    CUDA_OK(cudaFree(staging_dst));
    CUDA_OK(cudaStreamDestroy(compute_stream));
    ggml_cuda_moe_cache_free(staging_cache);

    auto * split_cache = ggml_cuda_moe_cache_init(dev, SLOT_BYTES, 2);
    CHECK(split_cache != nullptr);
    cudaStream_t split_copy_stream = ggml_cuda_moe_cache_copy_stream(split_cache);
    CHECK(split_copy_stream != nullptr);
    CUDA_OK(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));
    CUDA_OK(cudaMalloc(&staging_dst, 4 * SLOT_BYTES + SOURCE_PADDING));
    std::vector<float> split_readback(6 * N_FLOATS);

    constexpr size_t SMALL_STRIDE = SOURCE_PADDING / 2;
    auto * padding_decline_cache = ggml_cuda_moe_cache_init(dev, SMALL_STRIDE, 2);
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
        &padding_decline_wait_classes, compute_stream, false));
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
            &n_resident, staging_dst, nullptr, 0, &n_wait_classes, compute_stream, false));
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
            &n_resident, staging_dst, stage_ready, 3, &n_wait_classes, compute_stream, false));
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
            &invalid_n_wait_classes, compute_stream, false));
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
        &non_overflow_wait_classes, compute_stream, false));
    CHECK(cudaStreamQuery(split_copy_stream) == cudaSuccess);
    CHECK(cudaStreamQuery(compute_stream) == cudaSuccess);

    CUDA_OK(cudaFree(staging_dst));
    CUDA_OK(cudaStreamDestroy(compute_stream));
    ggml_cuda_moe_cache_free(split_cache);


    CUDA_OK(cudaStreamDestroy(copy_stream));
    CUDA_OK(cudaFreeHost(host_experts));

    fprintf(stderr, "test-moe-cache: OK\n");
}
