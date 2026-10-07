#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

void ggml_cuda_op_ssm_conv_qk(ggml_backend_cuda_context & ctx, ggml_tensor * conv, ggml_tensor * silu,
        ggml_tensor * q_norm, ggml_tensor * q_scale, ggml_tensor * k_norm, ggml_tensor * k_scale);
