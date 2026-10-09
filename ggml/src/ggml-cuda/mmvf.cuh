#include "common.cuh"

struct ggml_cuda_mmid_execution;

#define MMVF_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVF kernels.

void ggml_cuda_mul_mat_vec_f(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    const ggml_cuda_mm_fusion_args_host * fusion = nullptr, bool gdn_bf16_round_input = false, const ggml_cuda_mmid_execution * execution = nullptr);

void ggml_cuda_op_mul_mat_vec_f(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

bool ggml_cuda_should_use_mmvf(enum ggml_type type, int cc, int warp_size, const int64_t * src0_ne, const size_t * src0_nb, int64_t ne11);
inline bool ggml_cuda_should_use_mmvf(enum ggml_type type, int cc, const int64_t * src0_ne, const size_t * src0_nb, int64_t ne11) {
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    return ggml_cuda_should_use_mmvf(type, cc, warp_size, src0_ne, src0_nb, ne11);
}

void ggml_cuda_mul_mat_vec_f_postop(ggml_backend_cuda_context & ctx, ggml_tensor * mm, const ggml_tensor * pre, const ggml_tensor * unary, const ggml_tensor * post);

void ggml_cuda_mul_mat_vec_f_hc_pre(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, const ggml_tensor * dst);

bool ggml_cuda_should_fuse_hc_up(const ggml_tensor * weight, const ggml_tensor * norm, int device);
