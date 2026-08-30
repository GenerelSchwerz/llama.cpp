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

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

// MoE expert cache: tensors placed in this buffer type live in CPU pinned
// memory. A GPU-side LRU cache pages slabs in/out on access; the cache slot
// count is configured via llama_model_params::moe_expert_cache_slots.
// See ggml/src/ggml-cuda/moe-cache.cu and DESIGN.md.
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_moe_cached_buffer_type(void);
GGML_BACKEND_API bool ggml_backend_buft_is_cuda_moe_cached(ggml_backend_buffer_type_t buft);
GGML_BACKEND_API ggml_backend_buffer_t ggml_backend_cuda_moe_cached_buffer_from_host_ptr(void * ptr, size_t size);
// Compatibility shims. Cache resources are owned by CUDA backend contexts;
// these process-wide entry points do not allocate or free resources.
GGML_BACKEND_API void ggml_cuda_moe_cache_free_all(void);

// Compatibility value for older callers. Model-owned cache resources use backend candidate snapshots and do not read this process-wide value.
GGML_BACKEND_API void ggml_backend_cuda_moe_set_cache_slots(int n_slots);
GGML_BACKEND_API int  ggml_backend_cuda_moe_get_cache_slots(void);
GGML_BACKEND_API void ggml_backend_cuda_moe_set_l2_pinned_cache_size(size_t bytes);
GGML_BACKEND_API size_t ggml_backend_cuda_moe_get_l2_pinned_cache_size(void);
GGML_BACKEND_API void ggml_backend_cuda_moe_set_debug_mm(bool enabled);
GGML_BACKEND_API bool ggml_backend_cuda_moe_get_debug_mm(void);

// Resource-free compatibility shims for older callers.
GGML_BACKEND_API void ggml_backend_cuda_moe_observe_expert_tensor(
    const void * tensor_data,
    const char * tensor_name,
    size_t       per_expert_bytes,
    int64_t      n_experts);
GGML_BACKEND_API void ggml_backend_cuda_moe_reset_expert_size_observation(void);

GGML_BACKEND_API void ggml_backend_cuda_moe_preallocate_pools(int device);

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
