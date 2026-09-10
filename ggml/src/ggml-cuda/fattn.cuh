#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst);

// number of FlashAttention nodes that read a quantized K/V cache in place
// instead of casting it to F16 first; reached through get_proc_address
int64_t ggml_backend_cuda_fattn_native_count(void);
