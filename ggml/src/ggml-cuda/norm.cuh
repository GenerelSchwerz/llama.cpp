#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);

void ggml_cuda_op_rms_norm_scale_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_emit(ggml_backend_cuda_context & ctx, ggml_tensor * norm,
        ggml_tensor * mul, ggml_tensor * add, ggml_tensor * scale, void * f16, void * bf16);

bool ggml_cuda_should_fuse_hc_post_norm(const ggml_tensor * post, const ggml_tensor * norm, const ggml_tensor * mul);

void ggml_cuda_op_hc_post_norm(ggml_backend_cuda_context & ctx, ggml_tensor * post, ggml_tensor * norm, ggml_tensor * mul);

void ggml_cuda_op_hc_post_norm_emit(ggml_backend_cuda_context & ctx, ggml_tensor * post, ggml_tensor * norm, ggml_tensor * mul, void * f16, void * bf16);

void ggml_cuda_op_rms_norm_q8(ggml_backend_cuda_context & ctx, ggml_tensor * norm,
        ggml_tensor * mul, ggml_tensor * add, ggml_tensor * scale, void * image, int64_t cols, int64_t padded);

void ggml_cuda_op_rms_norm_emit_q8(ggml_backend_cuda_context & ctx, ggml_tensor * norm,
        ggml_tensor * mul, ggml_tensor * add, ggml_tensor * scale, void * f16, void * bf16, void * image, int64_t cols, int64_t padded);

void ggml_cuda_op_hc_post_norm_emit_q8(ggml_backend_cuda_context & ctx, ggml_tensor * post, ggml_tensor * norm,
        ggml_tensor * mul, void * f16, void * bf16, void * image, int64_t cols, int64_t padded);

void ggml_cuda_op_rms_norm_mmq(ggml_backend_cuda_context & ctx, ggml_tensor * norm, ggml_tensor * mul, ggml_tensor * add, ggml_tensor * scale, void * image, int64_t cols, int64_t padded, int64_t rows, int layout);

bool ggml_cuda_should_fuse_hc_post_norm_scale(const ggml_tensor * post, const ggml_tensor * norm, const ggml_tensor * scale);

void ggml_cuda_op_hc_post_norm_scale(ggml_backend_cuda_context & ctx, ggml_tensor * post, ggml_tensor * norm,
        ggml_tensor * scale, void * f16, void * bf16);

void ggml_cuda_op_hc_post_norm_emit_mmq(ggml_backend_cuda_context & ctx, ggml_tensor * post, ggml_tensor * norm, ggml_tensor * mul, ggml_tensor * scale, void * f16, void * bf16, void * image, int64_t cols, int64_t padded, int64_t rows, int layout);

void ggml_cuda_op_hc_injection(ggml_backend_cuda_context & ctx, ggml_tensor * first, ggml_tensor * unary, ggml_tensor * last, ggml_tensor * post, ggml_tensor * norm, ggml_tensor * mul);
