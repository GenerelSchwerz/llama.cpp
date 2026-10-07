#pragma once

#include "ggml.h"
#include <cstddef>
#include <cstdint>

struct ggml_backend_cuda_context;
struct ggml_backend_buffer;

#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__MUSACC__)
#include "common.cuh"

#define CUDA_SOFT_MAX_BLOCK_SIZE 1024

void ggml_cuda_op_soft_max(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_soft_max_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
#endif

struct ggml_cuda_source_softmax_resources {
    size_t pool_bytes = 0;
    uint64_t identity = 0;
};

// Query requires the selected device and bound optional mask/sink data.
// Caller owns backing, effects and graph instantiate/upload before publication.
bool ggml_cuda_softmax_prepare_resources(int device, const ggml_tensor * dst, ggml_cuda_source_softmax_resources & resources);

struct ggml_cuda_source_softmax_test_result {
    size_t eager_peak = 0;
    size_t capture_peak = 0;
    uint32_t replays = 0;
};
bool ggml_cuda_source_softmax_capture_for_test(int device, ggml_tensor * dst,
    ggml_backend_buffer * scratch, size_t capacity, ggml_cuda_source_softmax_test_result & result);
