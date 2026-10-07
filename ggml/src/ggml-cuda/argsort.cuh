#pragma once

#include "ggml.h"
#include <cstddef>
#include <cstdint>

struct ggml_backend_cuda_context;
struct ggml_backend_buffer;

#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__MUSACC__)
#include "common.cuh"

#define CUDA_ARGSORT_BLOCK_SIZE 1024

void ggml_cuda_op_argsort(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#ifdef GGML_CUDA_USE_CUB
int argsort_f32_i32_cuda_cub_chunk_nrows(const size_t nb01, const int64_t nrows);
void argsort_f32_i32_cuda_cub(ggml_cuda_pool & pool,
                              const float *    x,
                              int *            dst,
                              const int        ncols,
                              const int        nrows,
                              ggml_sort_order  order,
                              cudaStream_t     stream);
#endif  // GGML_CUDA_USE_CUB
void argsort_f32_i32_cuda_bitonic(const float *   x,
                                  int *           dst,
                                  const int       ncols,
                                  const int       nrows,
                                  ggml_sort_order order,
                                  cudaStream_t    stream);

#endif

struct ggml_cuda_source_sort_resources {
    size_t pool_bytes = 0;
    uint64_t identity = 0;
};

// Queries require the selected device; backing lifetime belongs to the caller.
bool ggml_cuda_argsort_prepare_resources(int device, const ggml_tensor * dst, ggml_cuda_source_sort_resources & resources);
bool ggml_cuda_top_k_prepare_resources(int device, const ggml_tensor * dst, ggml_cuda_source_sort_resources & resources);

namespace ggml_cuda_source_sort_detail {
bool metadata(int device, const ggml_tensor * dst, ggml_op op, ggml_cuda_source_sort_resources & resources, int & ncols, int & nrows);
inline void mix(uint64_t & identity, uint64_t value) {
    identity = (identity ^ value) * 1099511628211ULL;
}
inline bool add(size_t & total, size_t bytes) {
    if (bytes > SIZE_MAX - 255) { return false; }
    bytes = (bytes + 255) & ~size_t(255);
    if (bytes > SIZE_MAX - total) { return false; }
    total += bytes;
    return true;
}
}

struct ggml_cuda_source_sort_test_result {
    size_t eager_peak = 0;
    size_t capture_peak = 0;
    uint32_t replays = 0;
};
bool ggml_cuda_source_sort_capture_for_test(int device, ggml_tensor * dst,
    ggml_backend_buffer * scratch, size_t capacity, ggml_cuda_source_sort_test_result & result);
