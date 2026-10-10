#pragma once

#include "moe-source-state.cuh"

static constexpr uint32_t ggml_cuda_moe_source_import_threads = 256;

#if defined(GGML_CUDA_MOE_DEVICE_TAIL)
#include <cuda_runtime.h>

struct ggml_cuda_moe_source_tail_dispatch {
    cudaGraphExec_t next = nullptr;
    cudaGraphExec_t finalize = nullptr;
};

cudaError_t ggml_cuda_moe_source_tail_start(ggml_cuda_moe_source_runtime * runtime,
    ggml_cuda_moe_source_control * control, const ggml_cuda_moe_source_tail_dispatch * dispatch, cudaStream_t stream);
cudaError_t ggml_cuda_moe_source_tail_import(float * output, const float * input, size_t values,
    ggml_cuda_moe_source_runtime * runtime, const int32_t * resident, const int32_t * selected,
    ggml_cuda_moe_source_control * next_control, const ggml_cuda_moe_source_tail_dispatch * dispatch, cudaStream_t stream,
    size_t width = 0, uint32_t routes_per_row = 0, size_t output_column = 0, size_t output_row = 0, const uint32_t * gpu_mask = nullptr);
cudaError_t ggml_cuda_moe_source_join_import(float * output, const float * input, size_t values, size_t width, const uint32_t * gpu_mask,
    ggml_cuda_moe_source_runtime * runtime, const int32_t * resident, const int32_t * selected,
    ggml_cuda_moe_source_control * next_control, cudaGraphConditionalHandle next_handle, cudaStream_t stream);
#endif
