#pragma once
#include "common.cuh"
struct ggml_moe_reference_gpu_layout {
    int gu_type, d_type, n_embd, n_ff, n_out;
    size_t gate_row, up_row, d_row;
};
struct ggml_moe_reference_gpu_group {
    const uint64_t * gate, * up, * down;
    const int32_t * starts, * count, * tokens, * destinations;
};
bool ggml_moe_reference_gpu_supported(const ggml_moe_reference_gpu_layout & layout);
size_t ggml_moe_reference_gpu_scratch(size_t entries, size_t hidden);
bool ggml_moe_reference_gpu_quantize(const float * input, void * output, size_t values, cudaStream_t stream);
bool ggml_moe_reference_gpu_execute(const ggml_moe_reference_gpu_layout & layout,
    const ggml_moe_reference_gpu_group & group, unsigned groups, unsigned entries,
    const void * input, void * scratch, float * output, cudaStream_t stream);
