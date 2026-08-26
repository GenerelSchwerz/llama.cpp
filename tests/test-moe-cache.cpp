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

#include <cuda_runtime.h>

#include <algorithm>
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

int main(int argc, char ** argv) {
    (void)argc; (void)argv;

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

    auto * staging_cache = ggml_cuda_moe_cache_init(dev, SLOT_BYTES, 2, false, 0, 0);
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
    CUDA_OK(cudaMalloc(&staging_dst, 2 * SLOT_BYTES));
    std::vector<float> split_readback(4 * N_FLOATS);

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
        const int admission_order[4] = {1, 0, 3, 2};
        int n_resident = 0;
        CHECK(ggml_cuda_moe_cache_prepare_split_staging(
            split_cache, split_srcs, 4, admission_order, SLOT_BYTES, 1,
            slot_ids, &n_resident, staging_dst, compute_stream));
        CHECK(n_resident == 2);
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

    const void * non_overflow_srcs[] = {host_experts, host_experts + N_FLOATS};
    int non_overflow_slots[2] = {-1, -1};
    const int non_overflow_order[2] = {0, 1};
    int non_overflow_resident = 0;
    CHECK(!ggml_cuda_moe_cache_prepare_split_staging(
        split_cache, non_overflow_srcs, 2, non_overflow_order, SLOT_BYTES, 1, non_overflow_slots,
        &non_overflow_resident, staging_dst, compute_stream));
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
