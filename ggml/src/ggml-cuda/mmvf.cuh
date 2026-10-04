#include "common.cuh"
#include "unary.cuh"

#define MMVF_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVF kernels.

void ggml_cuda_mul_mat_vec_f(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

void ggml_cuda_op_mul_mat_vec_f(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

bool ggml_cuda_should_use_mmvf(enum ggml_type type, int cc, const int64_t * src0_ne, const size_t * src0_nb, int64_t ne11);

void ggml_cuda_mul_mat_vec_f_postop(ggml_backend_cuda_context & ctx, ggml_tensor * mm, const ggml_tensor * pre, const ggml_tensor * unary, const ggml_tensor * post);

struct ggml_cuda_mmvf_postop_store {
    float * pre;
    float * unary;
    float * post;
    float scale0;
    float bias0;
    float scale1;
    float bias1;
    ggml_unary_op op;

    __device__ __forceinline__ void operator()(float * dst, const float * base, int index, float value) const {
        dst[index] = value;
        const int64_t offset = dst - base + index;
        if (pre) {
            value = scale0 * value + bias0;
            if (pre == base) { dst[index] = value; } else { pre[offset] = value; }
        }
        value = op == GGML_UNARY_OP_SILU ? ggml_cuda_op_silu_single(value) : 1.0f / (1.0f + expf(-value));
        if (unary == base) { dst[index] = value; } else { unary[offset] = value; }
        if (post) {
            value = scale1 * value + bias1;
            if (post == base) { dst[index] = value; } else { post[offset] = value; }
        }
    }
};
