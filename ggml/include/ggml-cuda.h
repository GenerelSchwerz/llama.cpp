#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(int main_device, const float * tensor_split);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

// MoE expert cache: tensors placed in this buffer type live in CPU pinned
// memory. A GPU-side LRU cache pages slabs in/out on access; the cache slot
// count is configured via llama_model_params::moe_expert_cache_slots.
// See ggml/src/ggml-cuda/moe-cache.cu and DESIGN.md.
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_moe_cached_buffer_type(void);
GGML_BACKEND_API bool ggml_backend_buft_is_cuda_moe_cached(ggml_backend_buffer_type_t buft);
GGML_BACKEND_API ggml_backend_buffer_t ggml_backend_cuda_moe_cached_buffer_from_host_ptr(void * ptr, size_t size);
GGML_BACKEND_API void ggml_cuda_moe_cache_free_all(void);

// Configure / inspect the GPU LRU cache slot count. Set once at model load
// time when --moe-expert-cache-size N is parsed; read by the dispatch hook
// when it lazy-creates the per-device cache. Default 0 = no slot pool, fall
// back to per-op staging.
GGML_BACKEND_API void ggml_backend_cuda_moe_set_cache_slots(int n_slots);
GGML_BACKEND_API int  ggml_backend_cuda_moe_get_cache_slots(void);

// Called by the model loader for every expert tensor going into the cached
// buffer type, recording its (data ptr, name, per-expert byte stride). After
// model load, ggml_backend_cuda_moe_preallocate_pools() walks the recorded
// list and creates one cache per tensor with the exact slot size it needs.
// Reset between models.
GGML_BACKEND_API void ggml_backend_cuda_moe_observe_expert_tensor(
    const void * tensor_data,
    const char * tensor_name,
    size_t       per_expert_bytes,
    int64_t      n_experts);
GGML_BACKEND_API void ggml_backend_cuda_moe_reset_expert_size_observation(void);

// Eagerly create one slot pool per observed tensor on the given device.
// Each pool's slot_size is the exact expert_stride for that tensor; no
// padding. Logs each pool as a 'load_tensors:'-style line so they group
// with the other model buffer allocations. Idempotent.
GGML_BACKEND_API void ggml_backend_cuda_moe_preallocate_pools(int device);
GGML_BACKEND_API void ggml_backend_cuda_moe_prefill_pools(int device);

// Issue async H2D prefetches for the given expert ids into the cache for
// `tensor_name`. Used by the dispatch hook to warm up sibling matrix
// caches (e.g. when up's dispatch fires, prefetch gate and down's same
// expert ids since the router picks the same set across all 3 matrices).
// No-op if tensor_name has no observed data or no cache yet. Failures in
// individual acquires are silently ignored; the regular dispatch path
// will re-issue if needed.
GGML_BACKEND_API void ggml_backend_cuda_moe_prefetch_experts(
    int           device,
    const char *  tensor_name,
    const int32_t * eids,
    int           n_eids,
    bool          use_l2,
    bool          is_decode);

// Print cache hit/miss/eviction counters per device and reset them. Call at
// end of a request (or on model unload) to surface telemetry. No-op if the
// cache is uninitialized.
GGML_BACKEND_API void ggml_backend_cuda_moe_log_and_reset_stats(void);

// Eagerly allocate the per-device cache slot pool using the slot count
// previously set by ggml_backend_cuda_moe_set_cache_slots() and the max
// expert size observed during model load. Logs an "load_tensors:"-style line
// reporting the pool size, so it groups with the other model buffer lines.
// No-op if cache slots is 0 or no expert tensors were observed. Idempotent.
GGML_BACKEND_API void ggml_backend_cuda_moe_preallocate_pool(int device);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

#ifdef  __cplusplus
}
#endif
