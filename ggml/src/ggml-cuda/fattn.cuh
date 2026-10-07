#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>

struct ggml_backend_cuda_context;

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst);

struct ggml_cuda_fattn_resources {
    size_t pool_bytes = 0;
    uint64_t identity = 0;
};

bool ggml_cuda_flash_attn_ext_prepare_resources(ggml_backend_cuda_context & ctx, const ggml_tensor * dst,
    ggml_cuda_fattn_resources & resources);
