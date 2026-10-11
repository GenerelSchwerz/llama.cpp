#pragma once

#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst);


struct ggml_cuda_hc_pre_emit_data {
    void * f16 = nullptr;
    void * bf16 = nullptr;
};

void ggml_cuda_op_dsv4_hc_pre_emit(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_hc_pre_emit_data & images);

void ggml_cuda_op_dsv4_hc_pre_q8(ggml_backend_cuda_context & ctx, ggml_tensor * dst, void * image, int64_t cols, int64_t padded, const ggml_cuda_hc_pre_emit_data & images);

struct ggml_cuda_hc_pre_q8_emit_data {
    void * data;
    int64_t cols;
    int64_t padded;
};

struct ggml_cuda_hc_pre_mmq_emit_data {
    void * data;
    int64_t padded;
    int layout;
    void * images[3] = {};
};

void ggml_cuda_op_dsv4_hc_pre_convert(ggml_backend_cuda_context & ctx, ggml_type type, const void * weights, ggml_tensor * mm, ggml_tensor * dst, const ggml_cuda_hc_pre_emit_data * images = nullptr, const ggml_cuda_hc_pre_q8_emit_data * q8 = nullptr, const ggml_cuda_hc_pre_mmq_emit_data * mmq = nullptr);

void ggml_cuda_op_dsv4_hc_pre_mmq(ggml_backend_cuda_context & ctx, ggml_tensor * dst, void * image, int64_t padded, int layout, const ggml_cuda_hc_pre_emit_data & images);

void ggml_cuda_op_dsv4_hc_pre_mmq_multi(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_hc_pre_mmq_emit_data & mmq, const ggml_cuda_hc_pre_emit_data & images);
