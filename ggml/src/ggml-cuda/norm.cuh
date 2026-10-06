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

bool ggml_cuda_should_fuse_hc_post_norm(const ggml_tensor * post, const ggml_tensor * norm, const ggml_tensor * mul);

void ggml_cuda_op_hc_post_norm(ggml_backend_cuda_context & ctx, ggml_tensor * post, ggml_tensor * norm, ggml_tensor * mul);

void ggml_cuda_op_hc_injection(ggml_backend_cuda_context & ctx, ggml_tensor * first, ggml_tensor * unary, ggml_tensor * last, ggml_tensor * post, ggml_tensor * norm, ggml_tensor * mul);
