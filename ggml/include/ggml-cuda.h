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

// Configure / inspect the GPU LRU cache slot count. Set once at model load
// time when --moe-expert-cache-size N is parsed; read by the dispatch hook
// when it lazy-creates the per-device cache. Default 0 = no slot pool, fall
// back to per-op staging.
GGML_BACKEND_API void ggml_backend_cuda_moe_set_cache_slots(int n_slots);
GGML_BACKEND_API int  ggml_backend_cuda_moe_get_cache_slots(void);

// Called by the model loader for every expert tensor going into the cached
// buffer type, with the per-expert byte stride (tensor->nb[2] for a 3D
// experts tensor). The cache uses max(observed) as its slot size on first
// creation, avoiding the grow-on-demand realloc that would otherwise happen
// when the first cached op was a small-quant matrix and a later op was
// larger. Reset to 0 between models.
GGML_BACKEND_API void   ggml_backend_cuda_moe_observe_expert_size(size_t per_expert_bytes);
GGML_BACKEND_API size_t ggml_backend_cuda_moe_get_max_expert_size(void);
GGML_BACKEND_API void   ggml_backend_cuda_moe_reset_expert_size_observation(void);

// Print cache hit/miss/eviction counters per device and reset them. Call at
// end of a request (or on model unload) to surface telemetry. No-op if the
// cache is uninitialized.
GGML_BACKEND_API void ggml_backend_cuda_moe_log_and_reset_stats(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

#ifdef  __cplusplus
}
#endif
