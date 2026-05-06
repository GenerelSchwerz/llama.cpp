#pragma once

#include "common.cuh"

// MoE expert cache: keeps a fixed-size GPU slot pool of expert weight slabs
// while cold experts live in CPU pinned memory. On routing miss the slab is
// async-copied H2D and an LRU slot is evicted to make room.
//
// See ../../DESIGN.md for the full design.

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_cuda_moe_cache;

// Create a cache for one (device, matrix_type) pair.
//   slot_size_bytes : size of one expert weight slab (in the model's quant)
//   n_slots         : how many slabs the GPU pool can hold simultaneously
//   n_experts_total : universe of expert ids the cache must address
struct ggml_cuda_moe_cache * ggml_cuda_moe_cache_init(
    int      device,
    size_t   slot_size_bytes,
    int      n_slots,
    int      n_experts_total);

void ggml_cuda_moe_cache_free(struct ggml_cuda_moe_cache * cache);

// Returns the slot index holding `expert_id` after this call. Issues an async
// H2D copy on `copy_stream` if the expert wasn't resident; the caller must
// synchronize the compute stream against `copy_stream` before reading the
// slab. `host_src` is the pinned-CPU pointer to the slab.
int ggml_cuda_moe_cache_acquire(
    struct ggml_cuda_moe_cache * cache,
    int          expert_id,
    const void * host_src,
    cudaStream_t copy_stream);

// Returns device pointer to slot `slot_id`'s slab.
void * ggml_cuda_moe_cache_slot_ptr(
    struct ggml_cuda_moe_cache * cache,
    int slot_id);

// Telemetry
void ggml_cuda_moe_cache_stats(
    const struct ggml_cuda_moe_cache * cache,
    uint64_t * out_hits,
    uint64_t * out_misses,
    uint64_t * out_evictions);

void ggml_cuda_moe_cache_reset_stats(struct ggml_cuda_moe_cache * cache);

#ifdef __cplusplus
}
#endif
