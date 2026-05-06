// MoE expert cache — week-1 implementation.
// See ../../../../DESIGN.md (project root) for the full design.
//
// Layout in this file:
//   - struct ggml_cuda_moe_cache  : private state
//   - init / free                 : slot-pool alloc, bookkeeping vectors
//   - acquire                     : O(1) hit, O(n_slots) LRU pick on miss
//   - slot_ptr / stats            : trivial accessors
//
// Design choices for v1 (kept deliberately simple):
//   * Single std::mutex around the metadata. Single-GPU only. Contention is
//     bounded by token rate, so this isn't a bottleneck in practice.
//   * Linear scan for LRU eviction. n_slots is typically 16-256, so an O(n)
//     scan per miss is in the hundreds of ns.
//   * acquire() hands back a slot_id; the caller is responsible for
//     synchronizing the compute stream against the copy stream before
//     touching the slab. The cache does not own the compute stream.

#include "moe-cache.cuh"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <vector>

struct ggml_cuda_moe_cache {
    int      device;
    size_t   slot_size_bytes;
    int      n_slots;
    int      n_experts_total;

    void *   slot_pool_d;            // device alloc, n_slots * slot_size_bytes

    std::vector<int>      slot_to_expert;  // [n_slots],          -1 if empty
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

    if (slot_size_bytes == 0 || n_slots <= 0 || n_experts_total <= 0) {
        return nullptr;
    }

    int prev_device = 0;
    cudaError_t err = cudaGetDevice(&prev_device);
    if (err != cudaSuccess) {
        fprintf(stderr, "moe-cache: cudaGetDevice failed: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        fprintf(stderr, "moe-cache: cudaSetDevice(%d) failed: %s\n", device, cudaGetErrorString(err));
        return nullptr;
    }

    auto * c = new ggml_cuda_moe_cache;
    c->device          = device;
    c->slot_size_bytes = slot_size_bytes;
    c->n_slots         = n_slots;
    c->n_experts_total = n_experts_total;
    c->slot_pool_d     = nullptr;
    c->access_counter  = 0;

    err = cudaMalloc(&c->slot_pool_d, (size_t)n_slots * slot_size_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "moe-cache: cudaMalloc(%zu bytes) failed: %s\n",
                (size_t)n_slots * slot_size_bytes, cudaGetErrorString(err));
        delete c;
        cudaSetDevice(prev_device);
        return nullptr;
    }

    c->slot_to_expert.assign(n_slots, -1);
    c->last_used     .assign(n_slots, 0);
    c->expert_to_slot.assign(n_experts_total, -1);

    cudaSetDevice(prev_device);
    return c;
}

extern "C"
void ggml_cuda_moe_cache_free(struct ggml_cuda_moe_cache * cache) {
    if (!cache) return;

    int prev_device = 0;
    cudaGetDevice(&prev_device);
    cudaSetDevice(cache->device);

    if (cache->slot_pool_d) {
        cudaFree(cache->slot_pool_d);
    }
    cudaSetDevice(prev_device);

    delete cache;
}

extern "C"
int ggml_cuda_moe_cache_acquire(
    struct ggml_cuda_moe_cache * cache,
    int          expert_id,
    const void * host_src,
    cudaStream_t copy_stream) {

    if (!cache || expert_id < 0 || expert_id >= cache->n_experts_total) {
        return -1;
    }

    std::lock_guard<std::mutex> lk(cache->mu);

    // Hit path.
    int slot = cache->expert_to_slot[expert_id];
    if (slot >= 0) {
        cache->last_used[slot] = ++cache->access_counter;
        cache->hits.fetch_add(1, std::memory_order_relaxed);
        return slot;
    }

    // Miss: pick the LRU slot. Empty slots (last_used==0, slot_to_expert==-1)
    // win automatically since 0 < any served counter.
    int      lru_slot = 0;
    uint64_t lru_t    = std::numeric_limits<uint64_t>::max();
    for (int i = 0; i < cache->n_slots; ++i) {
        if (cache->last_used[i] < lru_t) {
            lru_t    = cache->last_used[i];
            lru_slot = i;
        }
    }

    int evicted = cache->slot_to_expert[lru_slot];
    if (evicted >= 0) {
        cache->expert_to_slot[evicted] = -1;
        cache->evictions.fetch_add(1, std::memory_order_relaxed);
    }

    cache->slot_to_expert[lru_slot]  = expert_id;
    cache->expert_to_slot[expert_id] = lru_slot;
    cache->last_used[lru_slot]       = ++cache->access_counter;
    cache->misses.fetch_add(1, std::memory_order_relaxed);

    if (host_src) {
        void * dst = (char *)cache->slot_pool_d + (size_t)lru_slot * cache->slot_size_bytes;
        cudaError_t err = cudaMemcpyAsync(
            dst, host_src, cache->slot_size_bytes,
            cudaMemcpyHostToDevice, copy_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "moe-cache: cudaMemcpyAsync failed: %s\n",
                    cudaGetErrorString(err));
            // Roll back the mapping so the next acquire retries cleanly.
            cache->slot_to_expert[lru_slot]  = -1;
            cache->expert_to_slot[expert_id] = -1;
            return -1;
        }
    }

    return lru_slot;
}

extern "C"
void * ggml_cuda_moe_cache_slot_ptr(
    struct ggml_cuda_moe_cache * cache,
    int slot_id) {
    if (!cache || slot_id < 0 || slot_id >= cache->n_slots) {
        return nullptr;
    }
    return (char *)cache->slot_pool_d + (size_t)slot_id * cache->slot_size_bytes;
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
    if (out_hits)      *out_hits      = cache->hits.load(std::memory_order_relaxed);
    if (out_misses)    *out_misses    = cache->misses.load(std::memory_order_relaxed);
    if (out_evictions) *out_evictions = cache->evictions.load(std::memory_order_relaxed);
}

extern "C"
void ggml_cuda_moe_cache_reset_stats(struct ggml_cuda_moe_cache * cache) {
    if (!cache) return;
    cache->hits.store(0, std::memory_order_relaxed);
    cache->misses.store(0, std::memory_order_relaxed);
    cache->evictions.store(0, std::memory_order_relaxed);
}
