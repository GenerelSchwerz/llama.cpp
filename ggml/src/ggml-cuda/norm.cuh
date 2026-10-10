#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_pre_mul(ggml_backend_cuda_context & ctx, ggml_tensor * product,
        ggml_tensor * norm, ggml_tensor * weighted, ggml_tensor * repeat);

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

void ggml_cuda_op_rms_norm_mmq(ggml_backend_cuda_context & ctx, ggml_tensor * norm, ggml_tensor * mul, ggml_tensor * add, ggml_tensor * scale, void * image, int64_t cols, int64_t padded, int64_t rows, int layout, bool banked = false);

bool ggml_cuda_should_fuse_hc_post_norm_scale(const ggml_tensor * post, const ggml_tensor * norm, const ggml_tensor * scale);

void ggml_cuda_op_hc_post_norm_scale(ggml_backend_cuda_context & ctx, ggml_tensor * post, ggml_tensor * norm,
        ggml_tensor * scale, void * f16, void * bf16);

void ggml_cuda_op_hc_post_norm_emit_mmq(ggml_backend_cuda_context & ctx, ggml_tensor * post, ggml_tensor * norm, ggml_tensor * mul, ggml_tensor * scale, void * f16, void * bf16, void * image, int64_t cols, int64_t padded, int64_t rows, int layout, bool banked = false);

void ggml_cuda_op_hc_injection(ggml_backend_cuda_context & ctx, ggml_tensor * first, ggml_tensor * unary, ggml_tensor * last, ggml_tensor * post, ggml_tensor * norm, ggml_tensor * mul);

struct ggml_cuda_hc_affine_emit_data {
    void * f16 = nullptr;
    void * bf16 = nullptr;
    void * q8 = nullptr;
    void * mmq = nullptr;
    int64_t cols = 1;
    int64_t padded = 0;
    int64_t rows = 0;
    int layout = 0;
    bool banked = false;
};

void ggml_cuda_op_hc_affine_injection(ggml_backend_cuda_context & ctx, ggml_tensor * first, ggml_tensor * added, ggml_tensor * unary, ggml_tensor * last, ggml_tensor * post, ggml_tensor * norm, ggml_tensor * mul, const ggml_cuda_hc_affine_emit_data * emit = nullptr, ggml_tensor * scale = nullptr);

struct ggml_cuda_rms_gate_images {
    void * f16 = nullptr;
    void * bf16 = nullptr;
    void * q8 = nullptr;
    int64_t cols = 1;
    int64_t padded = 0;
};

void ggml_cuda_op_rms_norm_gated(ggml_backend_cuda_context & ctx, ggml_tensor * norm, ggml_tensor * mul,
        ggml_tensor * gate, ggml_tensor * dst, ggml_unary_op gate_op, const ggml_cuda_rms_gate_images * images = nullptr);
