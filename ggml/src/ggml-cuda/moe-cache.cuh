#pragma once

// Pure host-side API: only depends on the CUDA runtime headers for
// cudaStream_t. Do NOT include common.cuh here -- it is full of device-side
// code (__device__, __shfl_*, threadIdx, ...) that only nvcc can compile,
// which would prevent .cpp consumers (e.g. tests) from including this header.
#include <cuda_runtime.h>
#include <stdint.h>

// MoE expert cache: keeps a fixed-size GPU slot pool of expert weight slabs
// while cold experts live in CPU pinned memory. On routing miss the slab is
// async-copied H2D and an LRU slot is evicted to make room.
//
// API design:
//   - The cache is keyed by the CPU pointer to an expert's weight slab. That
//     pointer is the unique identifier of the slab (different layers/matrix
//     types have different pointers because they live at different offsets in
//     the cached buffer). This avoids any "global expert id" namespacing.
//   - Slots are uniformly sized within a cache pool, but slot_size grows on
//     demand if a later op has experts larger than the current slot size. On
//     grow, the existing pool is freed, a new pool is allocated with the
//     larger slot size, and all bookkeeping is reset (existing cached entries
//     are evicted). This converges quickly: after the first few ops the cache
//     stabilizes at slot_size = max expert stride across the model.
//   - acquire() takes the actual byte count to copy on miss, which can be
//     less than slot_size_bytes. Smaller experts simply leave padding inside
//     their slot.
//
// See ../../DESIGN.md for the full design.

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_cuda_moe_cache;

// Create a cache for one device.
//   slot_size_bytes : size of one expert weight slab (uniform across slots)
//   n_slots         : how many slabs the GPU pool can hold simultaneously
struct ggml_cuda_moe_cache * ggml_cuda_moe_cache_init(
    int    device,
    size_t slot_size_bytes,
    int    n_slots);

void ggml_cuda_moe_cache_free(struct ggml_cuda_moe_cache * cache);

// Returns the slot index that holds `host_src`'s contents after this call.
// On hit, just bumps LRU; on miss, picks the LRU slot, evicts it, and issues
// an async H2D copy of `byte_count` bytes from `host_src` on `copy_stream`.
// The caller must synchronize the compute stream against `copy_stream` before
// reading the slab.
//
// `byte_count` MUST be <= ggml_cuda_moe_cache_slot_size_bytes(cache); the
// caller is expected to ensure this via grow_pool() if the current slot_size
// is too small. Returns -1 on bad args or a CUDA failure.
int ggml_cuda_moe_cache_acquire(
    struct ggml_cuda_moe_cache * cache,
    const void * host_src,
    size_t       byte_count,
    cudaStream_t copy_stream);

// Ensure the cache's slot size is at least `min_slot_size_bytes`. If the
// current slot size already covers it, no-op. Otherwise the existing pool is
// freed, a new pool is allocated with slot_size = min_slot_size_bytes, and
// all bookkeeping is reset (every previously-cached slab becomes a miss next
// access). Returns true on success, false if the new allocation fails (in
// which case the cache is left untouched and acquire() will keep working with
// the previous slot size).
bool ggml_cuda_moe_cache_grow_pool(
    struct ggml_cuda_moe_cache * cache,
    size_t min_slot_size_bytes);

// Returns device pointer to slot `slot_id`'s slab.
void * ggml_cuda_moe_cache_slot_ptr(
    struct ggml_cuda_moe_cache * cache,
    int slot_id);

// Inspectors.
size_t ggml_cuda_moe_cache_slot_size_bytes(const struct ggml_cuda_moe_cache * cache);
int    ggml_cuda_moe_cache_n_slots(const struct ggml_cuda_moe_cache * cache);

// Telemetry.
void ggml_cuda_moe_cache_stats(
    const struct ggml_cuda_moe_cache * cache,
    uint64_t * out_hits,
    uint64_t * out_misses,
    uint64_t * out_evictions);

void ggml_cuda_moe_cache_reset_stats(struct ggml_cuda_moe_cache * cache);

// Per-device singleton. Returns the existing cache for this device, or creates
// it on first call with the requested (slot_size_bytes, n_slots). On subsequent
// calls the existing cache is returned regardless of the requested args; the
// caller is expected to check ggml_cuda_moe_cache_slot_size_bytes() and fall
// back to per-op staging when sizes don't match.
struct ggml_cuda_moe_cache * ggml_cuda_moe_cache_get_or_create(
    int    device,
    size_t slot_size_bytes,
    int    n_slots);

// Global teardown: free all per-device caches. Safe to call repeatedly.
void ggml_cuda_moe_cache_free_all(void);

#ifdef __cplusplus
}
#endif
