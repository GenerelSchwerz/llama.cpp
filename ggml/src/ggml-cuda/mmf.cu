#include "ggml.h"
#include "mmf.cuh"
#include "mmid.cuh"
#include "unary.cuh"
#include "mmvf.cuh"

static __forceinline__ int mmf_get_rows_per_block(const int cc) {
    if (GGML_CUDA_CC_IS_CDNA(cc)) {
        return MMF_ROWS_PER_BLOCK_CDNA;
    } else {
        return MMF_ROWS_PER_BLOCK;
    }
}


static __global__ void mul_mat_f_reduce_warps(const float * partial, float * dst, const int64_t count) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    float sum = 0.0f;
#pragma unroll
    for (int warp = 0; warp < 8; ++warp) {
        sum += partial[int64_t(warp)*count + i];
    }
    dst[i] = sum;
}

template <typename T, int cols, bool prefetch>
static void mul_mat_f_split_warps(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, float * partial) {
    const int vals = sizeof(T) == 4 && !std::is_same_v<T, float> ? 2 : 1;
    const int64_t count = ggml_nelements(dst);
    const dim3 blocks(src0->ne[1]/MMF_ROWS_PER_BLOCK, dst->ne[2], dst->ne[3]*8);
    const dim3 threads(32, 1, 1);
    const int shared = std::max(16*36*4, GGML_PAD(cols, 8)*(32 + 4)*4) + (prefetch ? 32*36*4 : 0);
    // Each block keeps one original warp's K sequence; the second launch keeps its sum order.
    mul_mat_f<T, MMF_ROWS_PER_BLOCK, cols, 1, false, 8, prefetch><<<blocks, threads, shared, ctx.stream()>>>(
        (const T *) src0->data, (const float *) src1->data, nullptr, partial,
        src0->ne[0]/vals, cols, dst->ne[2], src0->nb[1]/sizeof(T), src1->nb[1]/sizeof(float)/vals, dst->nb[1]/sizeof(float),
        0, 0, dst->ne[2]/src0->ne[2], src0->nb[2]/sizeof(T), src1->nb[2]/sizeof(float), dst->nb[2]/sizeof(float),
        dst->ne[3]/src0->ne[3], src0->nb[3]/sizeof(T), src1->nb[3]/sizeof(float), dst->nb[3]/sizeof(float));
    mul_mat_f_reduce_warps<<<(count + 255)/256, 256, 0, ctx.stream()>>>(partial, (float *) dst->data, count);
}

template <typename T>
static void mul_mat_f_split_warps_switch(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, float * partial) {
    switch (dst->ne[1]) {
#define MMF_SPLIT_CASE(cols) case cols: \
        if (ggml_cuda_is_aligned(src0, 16)) { mul_mat_f_split_warps<T, cols, true>(ctx, src0, src1, dst, partial); } \
        else { mul_mat_f_split_warps<T, cols, false>(ctx, src0, src1, dst, partial); } break
        MMF_SPLIT_CASE(1);  MMF_SPLIT_CASE(2);  MMF_SPLIT_CASE(3);  MMF_SPLIT_CASE(4);
        MMF_SPLIT_CASE(5);  MMF_SPLIT_CASE(6);  MMF_SPLIT_CASE(7);  MMF_SPLIT_CASE(8);
        MMF_SPLIT_CASE(9);  MMF_SPLIT_CASE(10); MMF_SPLIT_CASE(11); MMF_SPLIT_CASE(12);
        MMF_SPLIT_CASE(13); MMF_SPLIT_CASE(14); MMF_SPLIT_CASE(15); MMF_SPLIT_CASE(16);
#undef MMF_SPLIT_CASE
        default: GGML_ABORT("unsupported column count");
    }
}

void ggml_cuda_mul_mat_f(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
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
    const int warp_size = ggml_cuda_info().devices[device].warp_size;

    const int64_t blocks = ne01/rows_per_block * ne2 * ne3;
    const int64_t packed_cols = ne00 / (src0->type == GGML_TYPE_F32 ? 1 : 2);
    const int64_t iterations = (packed_cols + 16*warp_size - 1)/(16*warp_size);
    const bool original_eight_warps = iterations < (packed_cols + 14*warp_size - 1)/(14*warp_size);
    if (!ids && GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_AMPERE && warp_size == 32 &&
            ggml_is_contiguous(dst) && packed_cols >= 2048 && original_eight_warps &&
            blocks < ggml_cuda_info().devices[device].nsm/2 && ne3 <= 65535/8) {
        ggml_cuda_pool_alloc<float> partial(ctx.pool(), ggml_nelements(dst)*8);
        switch (src0->type) {
            case GGML_TYPE_F32:  mul_mat_f_split_warps_switch<float>(ctx, src0, src1, dst, partial.get()); break;
            case GGML_TYPE_F16:  mul_mat_f_split_warps_switch<half2>(ctx, src0, src1, dst, partial.get()); break;
            case GGML_TYPE_BF16: mul_mat_f_split_warps_switch<nv_bfloat162>(ctx, src0, src1, dst, partial.get()); break;
            default: GGML_ABORT("unsupported type");
        }
        return;
    }

    switch (src0->type) {
        case GGML_TYPE_F32: {
            const float * src0_d = (const float *) src0->data;
            constexpr int vals_per_T = 1;
            mul_mat_f_switch_rows_per_block<float>(
                rows_per_block, src0_d, src1_d, ids_d, dst_d, ne00/vals_per_T, ne01, ncols_dst, s01/vals_per_T, stride_col_y/vals_per_T, stride_col_dst,
                ids_s0, ids_s1, ne02, nchannels_y, nchannels_dst, s02/vals_per_T, stride_channel_y, stride_channel_dst,
                ne03, ne3, s03/vals_per_T, s13, s3, ctx.stream(), ids_info_ptr);
        } break;
        case GGML_TYPE_F16: {
            const half2 * src0_d = (const half2 *) src0->data;
            constexpr int vals_per_T = 2;
            mul_mat_f_switch_rows_per_block<half2>(
                rows_per_block, src0_d, src1_d, ids_d, dst_d, ne00/vals_per_T, ne01, ncols_dst, s01/vals_per_T, stride_col_y/vals_per_T, stride_col_dst,
                ids_s0, ids_s1, ne02, nchannels_y, nchannels_dst, s02/vals_per_T, stride_channel_y, stride_channel_dst,
                ne03, ne3, s03/vals_per_T, s13, s3, ctx.stream(), ids_info_ptr);
        } break;
        case GGML_TYPE_BF16: {
            const nv_bfloat162 * src0_d = (const nv_bfloat162 *) src0->data;
            constexpr int vals_per_T = 2;
            mul_mat_f_switch_rows_per_block<nv_bfloat162>(
                rows_per_block, src0_d, src1_d, ids_d, dst_d, ne00/vals_per_T, ne01, ncols_dst, s01/vals_per_T, stride_col_y/vals_per_T, stride_col_dst,
                ids_s0, ids_s1, ne02, nchannels_y, nchannels_dst, s02/vals_per_T, stride_channel_y, stride_channel_dst,
                ne03, ne3, s03/vals_per_T, s13, s3, ctx.stream(), ids_info_ptr);
        } break;
        default:
            GGML_ABORT("unsupported type: %s", ggml_type_name(src0->type));
    }
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

struct ggml_cuda_mmf_pair_operand {
    const void * weight;
    float * output;
    int stride_row;
    int stride_col_output;
    int stride_channel_weight;
    int stride_sample_weight;
    int stride_channel_output;
    int stride_sample_output;
};

template <typename T, int Columns, int Warps, int Splits = 1, bool Prefetch = false>
__launch_bounds__(32*Warps, 1)
static __global__ void mul_mat_f_pair_banks(
        ggml_cuda_mmf_pair_operand first, ggml_cuda_mmf_pair_operand second, const float * input,
        int first_tiles, int k, int channels, int stride_col_input, int stride_channel_input, int stride_sample_input,
        int channel_ratio, int sample_ratio) {
    const bool use_first = blockIdx.x < first_tiles;
    const auto operand = use_first ? first : second;
    const int row = (int(blockIdx.x) - (use_first ? 0 : first_tiles))*MMF_ROWS_PER_BLOCK;
    mul_mat_f_impl<T, MMF_ROWS_PER_BLOCK, Columns, Warps, false, Splits, Prefetch>(
        (const T *) operand.weight, input, nullptr, operand.output, k, Columns, channels,
        operand.stride_row, stride_col_input, operand.stride_col_output, 0, 0,
        channel_ratio, operand.stride_channel_weight, stride_channel_input, operand.stride_channel_output,
        sample_ratio, operand.stride_sample_weight, stride_sample_input, operand.stride_sample_output, row);
}

template <typename T, int Columns, int Warps, int Splits = 1, bool Prefetch = false>
__launch_bounds__(32*Warps, 1)
static __global__ void mul_mat_f_pair_postop(
        ggml_cuda_mmf_pair_operand first, ggml_cuda_mmf_pair_operand second, const float * input,
        int first_tiles, int k, int channels, int stride_col_input, int stride_channel_input, int stride_sample_input,
        int channel_ratio, int sample_ratio, ggml_cuda_mmvf_postop_store first_write, ggml_cuda_mmvf_postop_store second_write) {
    const bool use_first = blockIdx.x < first_tiles;
    const auto operand = use_first ? first : second;
    const int row = (int(blockIdx.x) - (use_first ? 0 : first_tiles))*MMF_ROWS_PER_BLOCK;
    mul_mat_f_impl<T, MMF_ROWS_PER_BLOCK, Columns, Warps, false, Splits, Prefetch>(
        (const T *) operand.weight, input, nullptr, operand.output, k, Columns, channels,
        operand.stride_row, stride_col_input, operand.stride_col_output, 0, 0,
        channel_ratio, operand.stride_channel_weight, stride_channel_input, operand.stride_channel_output,
        sample_ratio, operand.stride_sample_weight, stride_sample_input, operand.stride_sample_output, row, use_first ? first_write : second_write);
}

static __global__ void mul_mat_f_pair_reduce(
        const float * partial, float * first, float * second, int64_t first_count, int64_t second_count) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= first_count + second_count) { return; }
    const bool use_first = i < first_count;
    const int64_t count = use_first ? first_count : second_count;
    const int64_t index = use_first ? i : i - first_count;
    partial += use_first ? 0 : 8*first_count;
    float sum = 0.0f;
#pragma unroll
    for (int warp = 0; warp < 8; ++warp) { sum += partial[int64_t(warp)*count + index]; }
    (use_first ? first : second)[index] = sum;
}

static __global__ void mul_mat_f_pair_reduce_postop(
        const float * partial, float * first, float * second, int64_t first_count, int64_t second_count, ggml_cuda_mmvf_postop_store first_write, ggml_cuda_mmvf_postop_store second_write) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= first_count + second_count) { return; }
    const bool use_first = i < first_count;
    const int64_t count = use_first ? first_count : second_count;
    const int64_t index = use_first ? i : i - first_count;
    partial += use_first ? 0 : 8*first_count;
    float sum = 0.0f;
#pragma unroll
    for (int warp = 0; warp < 8; ++warp) { sum += partial[int64_t(warp)*count + index]; }
    float * output = use_first ? first : second;
    (use_first ? first_write : second_write)(output, output, index, sum);
}

template <typename T, int Columns, int Warps, int Splits = 1, bool Prefetch = false, bool Postop = false>
static void ggml_cuda_mmf_pair_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * first, const ggml_tensor * second, float * partial, const ggml_cuda_scaled_unary_args & first_ops = {}, const ggml_cuda_scaled_unary_args & second_ops = {}) {
    const auto operand = [](const ggml_tensor * mm, float * dst) {
        const ggml_tensor * weight = mm->src[0];
        return ggml_cuda_mmf_pair_operand{weight->data, dst, int(weight->nb[1]/sizeof(T)), int(mm->nb[1]/sizeof(float)),
            int(weight->nb[2]/sizeof(T)), int(weight->nb[3]/sizeof(T)), int(mm->nb[2]/sizeof(float)), int(mm->nb[3]/sizeof(float))};
    };
    const ggml_tensor * input = first->src[1];
    float * first_dst = partial ? partial : (float *) first->data;
    float * second_dst = partial ? partial + 8*ggml_nelements(first) : (float *) second->data;
    const dim3 blocks((first->ne[0] + second->ne[0])/MMF_ROWS_PER_BLOCK, first->ne[2], first->ne[3]*Splits);
    const dim3 threads(32, Warps, 1);
    const int shared = std::max(Warps*16*36*4, GGML_PAD(Columns, 8)*(Warps*32 + 4)*4) + (Prefetch ? 32*36*4 : 0);
    if constexpr (Postop) {
    mul_mat_f_pair_postop<T, Columns, Warps, Splits, Prefetch><<<blocks, threads, shared, ctx.stream()>>>(
        operand(first, first_dst), operand(second, second_dst), (const float *) input->data,
        first->ne[0]/MMF_ROWS_PER_BLOCK, input->ne[0]/2, first->ne[2], input->nb[1]/(2*sizeof(float)),
        input->nb[2]/sizeof(float), input->nb[3]/sizeof(float), first->ne[2]/first->src[0]->ne[2], first->ne[3]/first->src[0]->ne[3], ggml_cuda_mmvf_postop_args(first_ops), ggml_cuda_mmvf_postop_args(second_ops));
    } else {
    mul_mat_f_pair_banks<T, Columns, Warps, Splits, Prefetch><<<blocks, threads, shared, ctx.stream()>>>(
        operand(first, first_dst), operand(second, second_dst), (const float *) input->data,
        first->ne[0]/MMF_ROWS_PER_BLOCK, input->ne[0]/2, first->ne[2], input->nb[1]/(2*sizeof(float)),
        input->nb[2]/sizeof(float), input->nb[3]/sizeof(float), first->ne[2]/first->src[0]->ne[2], first->ne[3]/first->src[0]->ne[3]);
    }
}

template <typename T, int Columns>
static void ggml_cuda_mmf_pair_warps(ggml_backend_cuda_context & ctx, const ggml_tensor * first, const ggml_tensor * second, int warps, bool split, const ggml_cuda_scaled_unary_args & first_ops, const ggml_cuda_scaled_unary_args & second_ops) {
    if (split) {
        const int64_t count = ggml_nelements(first) + ggml_nelements(second);
        ggml_cuda_pool_alloc<float> partial(ctx.pool(), 8*count);
        if (ggml_cuda_is_aligned(first->src[0], 16) && ggml_cuda_is_aligned(second->src[0], 16)) {
            ggml_cuda_mmf_pair_launch<T, Columns, 1, 8, true>(ctx, first, second, partial.get());
        } else {
            ggml_cuda_mmf_pair_launch<T, Columns, 1, 8, false>(ctx, first, second, partial.get());
        }
        if (first_ops.unary || second_ops.unary) {
            mul_mat_f_pair_reduce_postop<<<(count + 255)/256, 256, 0, ctx.stream()>>>(partial.get(), (float *) first->data, (float *) second->data, ggml_nelements(first), ggml_nelements(second), ggml_cuda_mmvf_postop_args(first_ops), ggml_cuda_mmvf_postop_args(second_ops));
        } else {
        mul_mat_f_pair_reduce<<<(count + 255)/256, 256, 0, ctx.stream()>>>(partial.get(), (float *) first->data, (float *) second->data, ggml_nelements(first), ggml_nelements(second));
        }
        return;
    }
    switch (warps) {
#define MMF_PAIR_WARPS(w) case w: \
        if (first_ops.unary || second_ops.unary) { ggml_cuda_mmf_pair_launch<T, Columns, w, 1, false, true>(ctx, first, second, nullptr, first_ops, second_ops); } \
        else { ggml_cuda_mmf_pair_launch<T, Columns, w>(ctx, first, second, nullptr); } break
        MMF_PAIR_WARPS(1); MMF_PAIR_WARPS(2); MMF_PAIR_WARPS(3); MMF_PAIR_WARPS(4);
        MMF_PAIR_WARPS(5); MMF_PAIR_WARPS(6); MMF_PAIR_WARPS(7); MMF_PAIR_WARPS(8);
#undef MMF_PAIR_WARPS
        default: GGML_ABORT("unsupported warp count");
    }
}

template <typename T>
static void ggml_cuda_mmf_pair_columns(ggml_backend_cuda_context & ctx, const ggml_tensor * first, const ggml_tensor * second, int warps, bool split, const ggml_cuda_scaled_unary_args & first_ops, const ggml_cuda_scaled_unary_args & second_ops) {
    switch (first->ne[1]) {
#define MMF_PAIR_COLS(c) case c: ggml_cuda_mmf_pair_warps<T, c>(ctx, first, second, warps, split, first_ops, second_ops); break
        MMF_PAIR_COLS(2); MMF_PAIR_COLS(3); MMF_PAIR_COLS(4); MMF_PAIR_COLS(5);
        MMF_PAIR_COLS(6); MMF_PAIR_COLS(7); MMF_PAIR_COLS(8); MMF_PAIR_COLS(9);
        MMF_PAIR_COLS(10); MMF_PAIR_COLS(11); MMF_PAIR_COLS(12); MMF_PAIR_COLS(13);
        MMF_PAIR_COLS(14); MMF_PAIR_COLS(15); MMF_PAIR_COLS(16);
#undef MMF_PAIR_COLS
        default: GGML_ABORT("unsupported column count");
    }
}

bool ggml_cuda_mul_mat_f_pair(ggml_backend_cuda_context & ctx, const ggml_tensor * first, const ggml_tensor * second, const ggml_cuda_scaled_unary_args & first_ops, const ggml_cuda_scaled_unary_args & second_ops) {
    const auto & info = ggml_cuda_info().devices[ctx.device];
    if (!GGML_CUDA_CC_IS_NVIDIA(info.cc) || info.cc < GGML_CUDA_CC_AMPERE || info.warp_size != 32 ||
            first->ne[1] < 2 || first->ne[1] > 16 || first->src[0]->type != second->src[0]->type ||
            (first->src[0]->type != GGML_TYPE_F16 && first->src[0]->type != GGML_TYPE_BF16)) { return false; }
    const int64_t k = first->src[0]->ne[0]/2;
    int warps = 1;
    int64_t iterations = (k + 63)/64;
    for (int w = 2; w <= 8; ++w) {
        const int64_t next = (k + 64*w - 1)/(64*w);
        if (next < iterations) { warps = w; iterations = next; }
    }
    const bool original_eight = (k + 511)/512 < (k + 447)/448;
    const auto original_split = [&](const ggml_tensor * mm) {
        const int64_t blocks = mm->ne[0]/MMF_ROWS_PER_BLOCK * mm->ne[2] * mm->ne[3];
        return k >= 2048 && original_eight && blocks < info.nsm/2 && mm->ne[3] <= 65535/8;
    };
    const bool split = original_split(first) || original_split(second);
    if (first->src[0]->type == GGML_TYPE_F16) { ggml_cuda_mmf_pair_columns<half2>(ctx, first, second, warps, split, first_ops, second_ops); }
    else { ggml_cuda_mmf_pair_columns<nv_bfloat162>(ctx, first, second, warps, split, first_ops, second_ops); }
    return true;
}

struct ggml_cuda_mmf_contract_store {
    ggml_cuda_mmvf_postop_store postop;
    bool half_output;

    __device__ __forceinline__ void operator()(float * dst, const float * base, int index, float value) const {
        if (half_output) { value = __half2float(__float2half_rn(value)); }
        postop(dst, base, index, value);
    }
};

template <typename T, int Columns, int Warps, int Splits = 1, bool Prefetch = false, bool Image = false>
__launch_bounds__(32*Warps, 1)
static __global__ void mul_mat_f_pair_contract(
        ggml_cuda_mmf_pair_operand first, ggml_cuda_mmf_pair_operand second, const float * input,
        int first_tiles, int first_rows, int second_rows, int k, int channels, int stride_col_input, int stride_channel_input, int stride_sample_input,
        int channel_ratio, int sample_ratio, ggml_cuda_mmvf_pair_policy first_policy, ggml_cuda_mmvf_pair_policy second_policy,
        ggml_cuda_mmvf_postop_store first_write, ggml_cuda_mmvf_postop_store second_write, const void * prepared_input) {
#if defined(TURING_MMA_AVAILABLE) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    const bool use_first = blockIdx.x < first_tiles;
    const auto operand = use_first ? first : second;
    const auto policy = use_first ? first_policy : second_policy;
    const int row = (int(blockIdx.x) - (use_first ? 0 : first_tiles))*MMF_ROWS_PER_BLOCK;
    using Write = std::conditional_t<Splits == 1, ggml_cuda_mmf_contract_store, ggml_cuda_mmf_store>;
    Write write{};
    if constexpr (Splits == 1) { write = {use_first ? first_write : second_write, policy.half_output}; }
    const auto run = [&](auto half_acc) {
        mul_mat_f_impl<T, MMF_ROWS_PER_BLOCK, Columns, Warps, false, Splits, Prefetch, Write, true, decltype(half_acc)::value, Image>(
            (const T *) operand.weight, input, nullptr, operand.output, k, Columns, channels,
            operand.stride_row, stride_col_input, operand.stride_col_output, 0, 0,
            channel_ratio, operand.stride_channel_weight, stride_channel_input, operand.stride_channel_output,
            sample_ratio, operand.stride_sample_weight, stride_sample_input, operand.stride_sample_output, row,
            write, use_first ? first_rows : second_rows, prepared_input);
    };
    if constexpr (std::is_same_v<T, half2>) {
        if (policy.half_acc) { run(std::true_type{}); return; }
    }
    run(std::false_type{});
#else
    NO_DEVICE_CODE;
#endif
}

static __global__ void mul_mat_f_pair_contract_reduce(
        const float * partial, float * first, float * second, int64_t first_count, int64_t second_count,
        ggml_cuda_mmvf_postop_store first_write, ggml_cuda_mmvf_postop_store second_write, bool first_half_output, bool second_half_output) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= first_count + second_count) { return; }
    const bool use_first = i < first_count;
    const int64_t count = use_first ? first_count : second_count;
    const int64_t index = use_first ? i : i - first_count;
    partial += use_first ? 0 : 8*first_count;
    float sum = 0.0f;
#pragma unroll
    for (int warp = 0; warp < 8; ++warp) { sum += partial[int64_t(warp)*count + index]; }
    float * output = use_first ? first : second;
    const ggml_cuda_mmf_contract_store write{use_first ? first_write : second_write, use_first ? first_half_output : second_half_output};
    write(output, output, index, sum);
}

template <typename T, int Columns, int Warps, int Splits = 1, bool Prefetch = false, bool Image = false>
static void ggml_cuda_mmf_contract_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * first, const ggml_tensor * second,
        const ggml_cuda_mmvf_pair_policy & first_policy, const ggml_cuda_mmvf_pair_policy & second_policy,
        const ggml_cuda_scaled_unary_args & first_ops, const ggml_cuda_scaled_unary_args & second_ops, const void * prepared_input, float * partial) {
    const auto operand = [](const ggml_tensor * mm, float * dst) {
        const ggml_tensor * weight = mm->src[0];
        return ggml_cuda_mmf_pair_operand{weight->data, dst, int(weight->nb[1]/sizeof(T)), int(mm->nb[1]/sizeof(float)),
            int(weight->nb[2]/sizeof(T)), int(weight->nb[3]/sizeof(T)), int(mm->nb[2]/sizeof(float)), int(mm->nb[3]/sizeof(float))};
    };
    const ggml_tensor * input = first->src[1];
    float * first_dst = partial ? partial : (float *) first->data;
    float * second_dst = partial ? partial + Splits*ggml_nelements(first) : (float *) second->data;
    const int first_tiles = (first->ne[0] + MMF_ROWS_PER_BLOCK - 1)/MMF_ROWS_PER_BLOCK;
    const int second_tiles = (second->ne[0] + MMF_ROWS_PER_BLOCK - 1)/MMF_ROWS_PER_BLOCK;
    const dim3 blocks(first_tiles + second_tiles, first->ne[2], first->ne[3]*Splits);
    const dim3 threads(32, Warps, 1);
    const int shared = std::max(Warps*16*36*4, GGML_PAD(Columns, 8)*(Warps*32 + 4)*4) + (Prefetch ? 32*36*4 : 0);
    mul_mat_f_pair_contract<T, Columns, Warps, Splits, Prefetch, Image><<<blocks, threads, shared, ctx.stream()>>>(
        operand(first, first_dst), operand(second, second_dst), (const float *) input->data,
        first_tiles, first->ne[0], second->ne[0], input->ne[0]/2, first->ne[2], input->nb[1]/(2*sizeof(float)),
        input->nb[2]/sizeof(float), input->nb[3]/sizeof(float), first->ne[2]/first->src[0]->ne[2], first->ne[3]/first->src[0]->ne[3],
        first_policy, second_policy, ggml_cuda_mmvf_postop_args(first_ops), ggml_cuda_mmvf_postop_args(second_ops), prepared_input);
}

template <typename T, int Columns, bool Image>
static void ggml_cuda_mmf_contract_warps(ggml_backend_cuda_context & ctx, const ggml_tensor * first, const ggml_tensor * second,
        const ggml_cuda_mmvf_pair_policy & first_policy, const ggml_cuda_mmvf_pair_policy & second_policy,
        const ggml_cuda_scaled_unary_args & first_ops, const ggml_cuda_scaled_unary_args & second_ops, const void * prepared_input, int warps, bool split) {
    if (split) {
        const int64_t count = ggml_nelements(first) + ggml_nelements(second);
        ggml_cuda_pool_alloc<float> partial(ctx.pool(), 8*count);
        if (ggml_cuda_is_aligned(first->src[0], 16) && ggml_cuda_is_aligned(second->src[0], 16)) {
            ggml_cuda_mmf_contract_launch<T, Columns, 1, 8, true, Image>(ctx, first, second, first_policy, second_policy, first_ops, second_ops, prepared_input, partial.get());
        } else {
            ggml_cuda_mmf_contract_launch<T, Columns, 1, 8, false, Image>(ctx, first, second, first_policy, second_policy, first_ops, second_ops, prepared_input, partial.get());
        }
        mul_mat_f_pair_contract_reduce<<<(count + 255)/256, 256, 0, ctx.stream()>>>(partial.get(), (float *) first->data, (float *) second->data, ggml_nelements(first), ggml_nelements(second),
            ggml_cuda_mmvf_postop_args(first_ops), ggml_cuda_mmvf_postop_args(second_ops), first_policy.half_output, second_policy.half_output);
        return;
    }
    switch (warps) {
#define MMF_CONTRACT_WARPS(w) case w: ggml_cuda_mmf_contract_launch<T, Columns, w, 1, false, Image>(ctx, first, second, first_policy, second_policy, first_ops, second_ops, prepared_input, nullptr); break
        MMF_CONTRACT_WARPS(1); MMF_CONTRACT_WARPS(2); MMF_CONTRACT_WARPS(3); MMF_CONTRACT_WARPS(4);
        MMF_CONTRACT_WARPS(5); MMF_CONTRACT_WARPS(6); MMF_CONTRACT_WARPS(7); MMF_CONTRACT_WARPS(8);
#undef MMF_CONTRACT_WARPS
        default: GGML_ABORT("unsupported warp count");
    }
}

template <typename T, bool Image>
static void ggml_cuda_mmf_contract_columns(ggml_backend_cuda_context & ctx, const ggml_tensor * first, const ggml_tensor * second,
        const ggml_cuda_mmvf_pair_policy & first_policy, const ggml_cuda_mmvf_pair_policy & second_policy,
        const ggml_cuda_scaled_unary_args & first_ops, const ggml_cuda_scaled_unary_args & second_ops, const void * prepared_input, int warps, bool split) {
    switch (first->ne[1]) {
#define MMF_CONTRACT_COLS(c) case c: ggml_cuda_mmf_contract_warps<T, c, Image>(ctx, first, second, first_policy, second_policy, first_ops, second_ops, prepared_input, warps, split); break
        MMF_CONTRACT_COLS(2); MMF_CONTRACT_COLS(3); MMF_CONTRACT_COLS(4); MMF_CONTRACT_COLS(5);
        MMF_CONTRACT_COLS(6); MMF_CONTRACT_COLS(7); MMF_CONTRACT_COLS(8);
#undef MMF_CONTRACT_COLS
        default: GGML_ABORT("unsupported column count");
    }
}

bool ggml_cuda_mul_mat_f_pair_contract(ggml_backend_cuda_context & ctx, const ggml_tensor * first, const ggml_tensor * second,
        const ggml_cuda_mmvf_pair_policy & first_policy, const ggml_cuda_mmvf_pair_policy & second_policy,
        const ggml_cuda_scaled_unary_args & first_ops, const ggml_cuda_scaled_unary_args & second_ops, const void * prepared_input, bool both_mmf) {
    const auto & info = ggml_cuda_info().devices[ctx.device];
    const int64_t tiles = ((first->ne[0] + 31)/32 + (second->ne[0] + 31)/32)*first->ne[2]*first->ne[3];
    if (first->ne[0] < MMF_ROWS_PER_BLOCK || second->ne[0] < MMF_ROWS_PER_BLOCK ||
            (first->ne[1] == 2 && tiles < info.nsm/2)) { return false; }
    if (both_mmf) { return ggml_cuda_mul_mat_f_pair(ctx, first, second, first_ops, second_ops); }
    if (!GGML_CUDA_CC_IS_NVIDIA(info.cc) || info.cc < GGML_CUDA_CC_AMPERE || info.warp_size != 32 ||
            first->ne[1] < 2 || first->ne[1] > 8 || first->src[0]->type != second->src[0]->type ||
            (first->src[0]->type != GGML_TYPE_F16 && first->src[0]->type != GGML_TYPE_BF16) ||
            !first_policy.round_input || !second_policy.round_input || first->src[0]->ne[0] % 64) { return false; }
    const int64_t k = first->src[0]->ne[0]/2;
    int warps = 1;
    int64_t iterations = (k + 63)/64;
    for (int w = 2; w <= 8; ++w) {
        const int64_t next = (k + 64*w - 1)/(64*w);
        if (next < iterations) { warps = w; iterations = next; }
    }
    const int64_t blocks = ((first->ne[0] + 31)/32 + (second->ne[0] + 31)/32)*first->ne[2]*first->ne[3];
    const bool split = k >= 2048 && (k + 511)/512 < (k + 447)/448 && blocks < info.nsm && first->ne[3] <= 65535/8;
    if (first->src[0]->type == GGML_TYPE_F16) {
        if (prepared_input) { ggml_cuda_mmf_contract_columns<half2, true>(ctx, first, second, first_policy, second_policy, first_ops, second_ops, prepared_input, warps, split); }
        else { ggml_cuda_mmf_contract_columns<half2, false>(ctx, first, second, first_policy, second_policy, first_ops, second_ops, nullptr, warps, split); }
    } else {
        if (prepared_input) { ggml_cuda_mmf_contract_columns<nv_bfloat162, true>(ctx, first, second, first_policy, second_policy, first_ops, second_ops, prepared_input, warps, split); }
        else { ggml_cuda_mmf_contract_columns<nv_bfloat162, false>(ctx, first, second, first_policy, second_policy, first_ops, second_ops, nullptr, warps, split); }
    }
    return true;
}
