// MoE expert cache — STUB. See ../../DESIGN.md.
//
// Implementation order (week 1 plan):
//   1. Slot pool allocation in cudaMalloc'd device memory.
//   2. LRU bookkeeping (host-side arrays, atomic access counter).
//   3. acquire() with cudaMemcpyAsync on a dedicated copy stream.
//   4. Stats counters.
//   5. test-moe-cache.cpp: synthetic workload, verify hit/miss correctness.

#include "moe-cache.cuh"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

struct ggml_cuda_moe_cache {
    int      device;
    size_t   slot_size_bytes;
    int      n_slots;
    int      n_experts_total;

    void *   slot_pool_d;            // device alloc, n_slots * slot_size_bytes

    std::vector<int>      slot_to_expert;  // [n_slots],     -1 if empty
    std::vector<uint64_t> last_used;       // [n_slots]
    std::vector<int>      expert_to_slot;  // [n_experts_total], -1 if absent

    uint64_t access_counter;
    std::mutex mu;

    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> misses{0};
    std::atomic<uint64_t> evictions{0};
};

extern "C"
struct ggml_cuda_moe_cache * ggml_cuda_moe_cache_init(
    int    device,
    size_t slot_size_bytes,
    int    n_slots,
    int    n_experts_total) {
    // TODO(week-1): cudaSetDevice, cudaMalloc slot pool, init bookkeeping vectors.
    (void)device; (void)slot_size_bytes; (void)n_slots; (void)n_experts_total;
    return nullptr;
}

extern "C"
void ggml_cuda_moe_cache_free(struct ggml_cuda_moe_cache * cache) {
    // TODO(week-1): cudaFree slot pool, delete struct.
    (void)cache;
}

extern "C"
int ggml_cuda_moe_cache_acquire(
    struct ggml_cuda_moe_cache * cache,
    int          expert_id,
    const void * host_src,
    cudaStream_t copy_stream) {
    // TODO(week-1):
    //   lock cache->mu
    //   slot = cache->expert_to_slot[expert_id]
    //   if slot >= 0:
    //     hit++; last_used[slot] = ++access_counter; return slot
    //   else:
    //     miss++
    //     pick LRU slot (min last_used)
    //     if slot_to_expert[slot] >= 0: expert_to_slot[that] = -1; evictions++
    //     slot_to_expert[slot] = expert_id
    //     expert_to_slot[expert_id] = slot
    //     last_used[slot] = ++access_counter
    //     cudaMemcpyAsync(slot_pool_d + slot * slot_size, host_src,
    //                     slot_size, cudaMemcpyHostToDevice, copy_stream)
    //     return slot
    (void)cache; (void)expert_id; (void)host_src; (void)copy_stream;
    return -1;
}

extern "C"
void * ggml_cuda_moe_cache_slot_ptr(
    struct ggml_cuda_moe_cache * cache,
    int slot_id) {
    // TODO(week-1): return (char*)slot_pool_d + slot_id * slot_size_bytes
    (void)cache; (void)slot_id;
    return nullptr;
}

extern "C"
void ggml_cuda_moe_cache_stats(
    const struct ggml_cuda_moe_cache * cache,
    uint64_t * out_hits,
    uint64_t * out_misses,
    uint64_t * out_evictions) {
    if (!cache) {
        if (out_hits)      *out_hits      = 0;
        if (out_misses)    *out_misses    = 0;
        if (out_evictions) *out_evictions = 0;
        return;
    }
    if (out_hits)      *out_hits      = cache->hits.load();
    if (out_misses)    *out_misses    = cache->misses.load();
    if (out_evictions) *out_evictions = cache->evictions.load();
}

extern "C"
void ggml_cuda_moe_cache_reset_stats(struct ggml_cuda_moe_cache * cache) {
    if (!cache) return;
    cache->hits.store(0);
    cache->misses.store(0);
    cache->evictions.store(0);
}
