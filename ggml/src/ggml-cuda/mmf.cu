#include "ggml.h"
#include "mmf.cuh"
#include "mmid.cuh"
#include "mmvf.cuh"
#include "unary.cuh"

static __forceinline__ int mmf_get_rows_per_block(const int cc) {
    if (GGML_CUDA_CC_IS_CDNA(cc)) {
        return MMF_ROWS_PER_BLOCK_CDNA;
    } else {
        return MMF_ROWS_PER_BLOCK;
    }
}

template <bool two_banks>
static void ggml_cuda_mul_mat_f_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst, const ggml_tensor * src0_second = nullptr, const ggml_tensor * dst_second = nullptr) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(!ids ||  ids->type == GGML_TYPE_I32);
    GGML_ASSERT(         dst->type == GGML_TYPE_F32);


    GGML_TENSOR_BINARY_OP_LOCALS;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(ne13 == ne3);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));
    GGML_ASSERT(        nb0        == ts_dst);

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = src1->nb[1] / ts_src1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s12 = src1->nb[2] / ts_src1;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s13 = src1->nb[3] / ts_src1;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const int64_t ids_s0 = ids ? ids->nb[0] / ggml_type_size(ids->type) : 0;
    const int64_t ids_s1 = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;

    mmf_ids_data ids_info{};
    mmf_ids_data * ids_info_ptr = nullptr;
    ggml_cuda_pool_alloc<int32_t> ids_src_compact_dev;
    ggml_cuda_pool_alloc<int32_t> ids_dst_compact_dev;
    ggml_cuda_pool_alloc<int32_t> expert_bounds_dev;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_dst      = ids ? ne1 : ne2;

    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1 : s2;

    int64_t stride_channel_y         = ids ? s11  : s12;
    int64_t nchannels_y              = ids ? ne11 : ne12;

    //mul_mat_id: handle broadcast
    if (ids && nchannels_y == 1) {
        stride_channel_y = 0;
        nchannels_y      = ids->ne[0];
    }

    if (ids && ncols_dst > 16) {
        const int64_t n_expert_used = ids->ne[0];
        const int64_t n_experts     = ne02;
        const int64_t n_tokens      = ne12;
        const int64_t ne_get_rows   = n_tokens * n_expert_used;

        ids_src_compact_dev.alloc(ctx.pool(), ne_get_rows);
        ids_dst_compact_dev.alloc(ctx.pool(), ne_get_rows);
        expert_bounds_dev.alloc(ctx.pool(), n_experts + 1);

        const int si1  = static_cast<int>(ids_s1);
        const int sis1 = static_cast<int>(src1->nb[2] / src1->nb[1]);

        GGML_ASSERT(sis1 > 0);

        ggml_cuda_launch_mm_ids_helper(ids_d, ids_src_compact_dev.get(), ids_dst_compact_dev.get(), expert_bounds_dev.get(),
            static_cast<int>(n_experts), static_cast<int>(n_tokens), static_cast<int>(n_expert_used), static_cast<int>(ne11), si1, sis1, /*write_inverse =*/ false, ctx.stream());
        CUDA_CHECK(cudaGetLastError());

        ids_info.ids_src_compact   = ids_src_compact_dev.get();
        ids_info.ids_dst_compact   = ids_dst_compact_dev.get();
        ids_info.expert_bounds_dev = expert_bounds_dev.get();
        ids_info.n_experts         = static_cast<int>(n_experts);
        ids_info.sis1              = sis1;
        ids_info_ptr = &ids_info;
    }

    const int device    = ggml_cuda_get_device();
    const int cc        = ggml_cuda_info().devices[device].cc;
    const int rows_per_block = mmf_get_rows_per_block(cc);
    const int64_t nrows_x = two_banks ? 2*ne01 : ne01;

    switch (src0->type) {
        case GGML_TYPE_F32: {
            const float * src0_d = (const float *) src0->data;
            constexpr int vals_per_T = 1;
            mul_mat_f_switch_rows_per_block<float, two_banks>(
                rows_per_block, src0_d, src1_d, ids_d, dst_d, ne00/vals_per_T, nrows_x, ncols_dst, s01/vals_per_T, stride_col_y/vals_per_T, stride_col_dst,
                ids_s0, ids_s1, ne02, nchannels_y, nchannels_dst, s02/vals_per_T, stride_channel_y, stride_channel_dst,
                ne03, ne3, s03/vals_per_T, s13, s3, ctx.stream(), ids_info_ptr, (decltype(src0_d)) (two_banks ? src0_second->data : nullptr), two_banks ? ne01 : 0, two_banks ? (float *) dst_second->data : nullptr);
        } break;
        case GGML_TYPE_F16: {
            const half2 * src0_d = (const half2 *) src0->data;
            constexpr int vals_per_T = 2;
            mul_mat_f_switch_rows_per_block<half2, two_banks>(
                rows_per_block, src0_d, src1_d, ids_d, dst_d, ne00/vals_per_T, nrows_x, ncols_dst, s01/vals_per_T, stride_col_y/vals_per_T, stride_col_dst,
                ids_s0, ids_s1, ne02, nchannels_y, nchannels_dst, s02/vals_per_T, stride_channel_y, stride_channel_dst,
                ne03, ne3, s03/vals_per_T, s13, s3, ctx.stream(), ids_info_ptr, (decltype(src0_d)) (two_banks ? src0_second->data : nullptr), two_banks ? ne01 : 0, two_banks ? (float *) dst_second->data : nullptr);
        } break;
        case GGML_TYPE_BF16: {
            const nv_bfloat162 * src0_d = (const nv_bfloat162 *) src0->data;
            constexpr int vals_per_T = 2;
            mul_mat_f_switch_rows_per_block<nv_bfloat162, two_banks>(
                rows_per_block, src0_d, src1_d, ids_d, dst_d, ne00/vals_per_T, nrows_x, ncols_dst, s01/vals_per_T, stride_col_y/vals_per_T, stride_col_dst,
                ids_s0, ids_s1, ne02, nchannels_y, nchannels_dst, s02/vals_per_T, stride_channel_y, stride_channel_dst,
                ne03, ne3, s03/vals_per_T, s13, s3, ctx.stream(), ids_info_ptr, (decltype(src0_d)) (two_banks ? src0_second->data : nullptr), two_banks ? ne01 : 0, two_banks ? (float *) dst_second->data : nullptr);
        } break;
        default:
            GGML_ABORT("unsupported type: %s", ggml_type_name(src0->type));
    }
}

void ggml_cuda_mul_mat_f(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    ggml_cuda_mul_mat_f_impl<false>(ctx, src0, src1, ids, dst);
}

bool ggml_cuda_should_use_mmf(enum ggml_type type, int cc, int warp_size, const int64_t * src0_ne,
        const size_t * src0_nb, const int src1_ncols, bool mul_mat_id) {
    if (ggml_is_quantized(type)) {
        return false;
    }

    const size_t ts = ggml_type_size(type);
    if (src0_ne[0] % (warp_size * (4/ts)) != 0) {
        return false;
    }

    if (src0_nb[0] != ts) {
        return false;
    }

    // Pointers not aligned to the size of half2/nv_bfloat162/float2 would result in a crash:
    for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
        if (src0_nb[i] % (2*ts) != 0) {
            return false;
        }
    }
    if (src0_ne[1] % mmf_get_rows_per_block(cc) != 0) {
        return false;
    }

    if (GGML_CUDA_CC_IS_CDNA3(cc) && type == GGML_TYPE_BF16) {
        return false;
    }

    if (mul_mat_id) {
        if (src0_ne[1] <= 1024 && src1_ncols > 512) {
            return false;
        } else if(src0_ne[1] > 1024 && src1_ncols > 128) {
            return false;
        }
    } else {
        if (GGML_CUDA_CC_IS_RDNA3_0(cc) && src1_ncols > 8) {
            return false;
        } else if (GGML_CUDA_CC_IS_CDNA2(cc) && (type == GGML_TYPE_F16 || type == GGML_TYPE_BF16)) {
            //TODO: truse CDNA2 as CDNA1, tune the perf when CDNA2 is available.
            return false;
        } else if (GGML_CUDA_CC_IS_CDNA1(cc) && (type == GGML_TYPE_F16 || type == GGML_TYPE_BF16)) {
            return false;
        } else if (src1_ncols > 16) {
            return false;
        }
    }

    switch (type) {
        case GGML_TYPE_F32:
            return ampere_mma_available(cc) || amd_mfma_available(cc);
        case GGML_TYPE_F16:
            return volta_mma_available(cc) || turing_mma_available(cc) || amd_wmma_available(cc) || amd_mfma_available(cc);
        case GGML_TYPE_BF16:
            return ampere_mma_available(cc) || amd_wmma_available(cc) || amd_mfma_available(cc);
        default:
            return false;
    }
}

bool ggml_cuda_should_pair_mmf_id(const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * glu, const int device) {
    if (!up || !gate || !glu || up->op != GGML_OP_MUL_MAT_ID || gate->op != GGML_OP_MUL_MAT_ID || glu->op != GGML_OP_GLU ||
            !up->src[0] || !gate->src[0] || !up->src[1] || !up->src[2] ||
            up->src[1] != gate->src[1] || up->src[2] != gate->src[2] ||
            glu->src[0] != gate || glu->src[1] != up || ggml_get_op_params_i32(glu, 1) != 0 ||
            memcmp(up->op_params, gate->op_params, sizeof(up->op_params)) != 0 || ggml_get_op_params_i32(up, 3) == GGML_PREC_F32) {
        return false;
    }
    switch (ggml_get_glu_op(glu)) {
        case GGML_GLU_OP_SWIGLU:
        case GGML_GLU_OP_GEGLU:
        case GGML_GLU_OP_REGLU:
            break;
        case GGML_GLU_OP_SWIGLU_CLAMP:
            if (!std::isfinite(ggml_get_op_params_f32(glu, 3)) || ggml_get_op_params_f32(glu, 3) <= 0.0f) {
                return false;
            }
            break;
        default:
            return false;
    }
    const ggml_tensor * weight = up->src[0];
    const ggml_tensor * input  = up->src[1];
    const ggml_tensor * ids    = up->src[2];
    if ((weight->type != GGML_TYPE_F32 && weight->type != GGML_TYPE_F16 && weight->type != GGML_TYPE_BF16) ||
            weight->type != gate->src[0]->type || input->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 ||
            up->type != GGML_TYPE_F32 || gate->type != GGML_TYPE_F32 || glu->type != GGML_TYPE_F32 ||
            !ggml_are_same_shape(weight, gate->src[0]) || !ggml_are_same_stride(weight, gate->src[0])) {
        return false;
    }
    const ggml_tensor * tensors[] = { weight, gate->src[0], input, ids, up, gate, glu };
    for (const ggml_tensor * tensor : tensors) {
        const size_t ts = ggml_type_size(tensor->type);
        size_t span = ts;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            if (tensor->ne[d] <= 0 || tensor->ne[d] > INT_MAX || tensor->nb[d] == 0 || tensor->nb[d] % ts != 0 || tensor->nb[d]/ts > INT_MAX ||
                    (size_t) (tensor->ne[d] - 1) > (SIZE_MAX - span)/tensor->nb[d]) {
                return false;
            }
            span += (tensor->ne[d] - 1)*tensor->nb[d];
        }
        if (span/ts > INT_MAX) {
            return false;
        }
    }
    const size_t ts = ggml_type_size(weight->type);
    const int vals_per_T = weight->type == GGML_TYPE_F32 ? 1 : 2;
    const int cc = ggml_cuda_info().devices[device].cc;
    if ((GGML_CUDA_CC_IS_AMD(cc) && input->ne[2] <= MMVF_MAX_BATCH_SIZE) ||
            !ggml_cuda_should_use_mmf(weight->type, cc, WARP_SIZE, weight->ne, weight->nb, input->ne[2], true) ||
            !ggml_cuda_should_use_mmf(weight->type, cc, ggml_cuda_info().devices[device].warp_size, weight->ne, weight->nb, input->ne[2], true) ||
            weight->nb[1] % (2*ts*vals_per_T) != 0 || input->nb[1] % (2*sizeof(float)) != 0 ||
            input->nb[2] % (2*sizeof(float)*vals_per_T) != 0 || input->nb[2] % input->nb[1] != 0 ||
            weight->ne[0] != input->ne[0] || weight->ne[3] != 1 || input->ne[3] != 1 ||
            ids->ne[2] != 1 || ids->ne[3] != 1 || ids->ne[1] != input->ne[2] || ids->ne[0] > weight->ne[2] ||
            (input->ne[1] != 1 && input->ne[1] != ids->ne[0]) ||
            input->nb[0] != sizeof(float) || ids->nb[0] != sizeof(int32_t) ||
            !ggml_is_contiguous(up) || !ggml_is_contiguous(gate) || !ggml_is_contiguous(glu) ||
            !ggml_are_same_shape(up, gate) || !ggml_are_same_shape(up, glu) ||
            up->ne[0] != weight->ne[1] || up->ne[1] != ids->ne[0] || up->ne[2] != ids->ne[1] || up->ne[3] != 1 ||
            weight->ne[2] > 65535 || weight->ne[1] > INT_MAX/2 ||
            ids->ne[0] > INT_MAX/(2*weight->ne[1]) || input->ne[2] > INT_MAX/(2*weight->ne[1]*ids->ne[0])) {
        return false;
    }
    if (input->ne[2] > 16 && (ids->ne[0] >= (1 << 10) || input->ne[2] >= (1 << 22) ||
            (size_t) input->ne[2] > ggml_cuda_info().devices[device].smpbo/sizeof(int32_t))) {
        return false;
    }
    const size_t elements = 2*weight->ne[1]*ids->ne[0]*input->ne[2];
    return elements <= SIZE_MAX/sizeof(float);
}

void ggml_cuda_mul_mat_id_f_pair(ggml_backend_cuda_context & ctx, const ggml_tensor * up, const ggml_tensor * gate, ggml_tensor * dst) {
    ggml_cuda_mul_mat_f_impl<true>(ctx, up->src[0], up->src[1], up->src[2], up, gate->src[0], gate);
    CUDA_CHECK(cudaGetLastError());
    switch (ggml_get_glu_op(dst)) {
        case GGML_GLU_OP_SWIGLU:       ggml_cuda_op_swiglu(ctx, dst); break;
        case GGML_GLU_OP_GEGLU:        ggml_cuda_op_geglu(ctx, dst); break;
        case GGML_GLU_OP_REGLU:        ggml_cuda_op_reglu(ctx, dst); break;
        case GGML_GLU_OP_SWIGLU_CLAMP: ggml_cuda_op_swiglu_clamp(ctx, dst); break;
        default:                      GGML_ABORT("unsupported paired MMF GLU");
    }
    CUDA_CHECK(cudaGetLastError());
}
