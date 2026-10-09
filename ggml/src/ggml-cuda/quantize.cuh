#pragma once

#include "common.cuh"
#include "mmq.cuh"

#include <cstdint>

#define CUDA_QUANTIZE_BLOCK_SIZE     256
#define CUDA_QUANTIZE_BLOCK_SIZE_MMQ 128

static_assert(MATRIX_ROW_PADDING %    CUDA_QUANTIZE_BLOCK_SIZE      == 0, "Risk of out-of-bounds access.");
static_assert(MATRIX_ROW_PADDING % (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ) == 0, "Risk of out-of-bounds access.");

template <mmq_q8_1_ds_layout Layout>
struct ggml_cuda_norm_mmq_store {
    static constexpr int width = 4;
    block_q8_1_mmq * image;
    int64_t cols;
    int64_t padded;
    int64_t rows;

    __device__ __forceinline__ void emit(int64_t index, float4 xi) const {
        constexpr int vals_per_scale = Layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
        const int64_t column = index % cols;
        const int64_t row = index / cols;
        store(column, row, xi);
        if (column >= cols - vals_per_scale) {
            for (int64_t tail = cols; tail < padded; tail += vals_per_scale) {
                store(tail + column % vals_per_scale, row, make_float4(0.0f, 0.0f, 0.0f, 0.0f));
            }
        }
    }

    __device__ __forceinline__ void store(int64_t column, int64_t row, float4 xi) const {
        constexpr int vals_per_scale = Layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
        constexpr int vals_per_sum = Layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;
        const int64_t ib = (column/QK8_1_MMQ)*rows + row;
        const int iqs = column % QK8_1_MMQ;
        float amax = fabsf(xi.x);
        amax = fmaxf(amax, fabsf(xi.y));
        amax = fmaxf(amax, fabsf(xi.z));
        amax = fmaxf(amax, fabsf(xi.w));
        const unsigned mask = __activemask();
#pragma unroll
        for (int offset = vals_per_scale/8; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(mask, amax, offset, WARP_SIZE));
        }
        float sum;
        if constexpr (Layout != MMQ_Q8_1_DS_LAYOUT_D4) {
            sum = xi.x + xi.y + xi.z + xi.w;
#pragma unroll
            for (int offset = vals_per_sum/8; offset > 0; offset >>= 1) {
                sum += __shfl_xor_sync(mask, sum, offset, WARP_SIZE);
            }
        }
        const float d_inv = 127.0f/amax;
        const float d = 1.0f/d_inv;
        char4 q;
        q.x = roundf(xi.x*d_inv);
        q.y = roundf(xi.y*d_inv);
        q.z = roundf(xi.z*d_inv);
        q.w = roundf(xi.w*d_inv);
        ((char4 *) image[ib].qs)[iqs/4] = q;
        if constexpr (Layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
            if (iqs % 16 == 0 && iqs < 96) {
                image[ib].d2s6[2 + iqs/16] = sum;
                if (iqs % 64 == 0) { image[ib].d2s6[iqs/64] = d; }
            }
        } else if (iqs % 32 == 0) {
            if constexpr (Layout == MMQ_Q8_1_DS_LAYOUT_DS4) { image[ib].ds4[iqs/32] = make_half2(d, sum); }
            else { image[ib].d4[iqs/32] = d; }
        }
    }

    __device__ __forceinline__ void operator()(float * dst, const float * base, int col, float4 value) const {
        dst[col] = value.x; dst[col + 1] = value.y;
        dst[col + 2] = value.z; dst[col + 3] = value.w;
        emit(dst - base + col, value);
    }
};

struct ggml_cuda_norm_mxfp4_store {
    static constexpr int width = 4;
    block_fp4_mmq * image;
    int64_t cols;
    int64_t padded;
    int64_t rows;

    __device__ __forceinline__ void store(int64_t column, int64_t row, float4 value) const {
        const int lane = (column % 32) / 4;
        float amax = fabsf(value.x);
        amax = fmaxf(amax, fabsf(value.y));
        amax = fmaxf(amax, fabsf(value.z));
        amax = fmaxf(amax, fabsf(value.w));
        const unsigned mask = __activemask();
#pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(mask, amax, offset, 8));
        }
        uint8_t e = 0;
        if (amax > 0.0f) {
            e = static_cast<uint8_t>(min(max(__float2int_rn(log2f(amax)) - 2 + 127, 0), 254));
        }
        const float inv_s = amax == 0.0f ? 0.0f : __frcp_rn(ggml_cuda_e8m0_to_fp32(e));
        const int sender = lane / 2;
#if CUDART_VERSION >= 12080
        value.x *= inv_s; value.y *= inv_s; value.z *= inv_s; value.w *= inv_s;
#else
        value.x = ggml_cuda_float_to_fp4_e2m1(value.x, inv_s);
        value.y = ggml_cuda_float_to_fp4_e2m1(value.y, inv_s);
        value.z = ggml_cuda_float_to_fp4_e2m1(value.z, inv_s);
        value.w = ggml_cuda_float_to_fp4_e2m1(value.w, inv_s);
#endif
        const float lo0 = __shfl_sync(mask, value.x, sender, 8);
        const float lo1 = __shfl_sync(mask, value.y, sender, 8);
        const float lo2 = __shfl_sync(mask, value.z, sender, 8);
        const float lo3 = __shfl_sync(mask, value.w, sender, 8);
        const float hi0 = __shfl_sync(mask, value.x, sender + 4, 8);
        const float hi1 = __shfl_sync(mask, value.y, sender + 4, 8);
        const float hi2 = __shfl_sync(mask, value.z, sender + 4, 8);
        const float hi3 = __shfl_sync(mask, value.w, sender + 4, 8);
        const float v0 = lane % 2 ? lo2 : lo0;
        const float v1 = lane % 2 ? hi2 : hi0;
        const float v2 = lane % 2 ? lo3 : lo1;
        const float v3 = lane % 2 ? hi3 : hi1;
        char2 packed;
#if CUDART_VERSION >= 12080
        const __nv_fp4x4_e2m1 fp4(make_float4(v0, v1, v2, v3));
        packed = *reinterpret_cast<const char2 *>(&fp4);
#else
        packed = make_char2((uint8_t(v1) << 4) | uint8_t(v0), (uint8_t(v3) << 4) | uint8_t(v2));
#endif
        const int sub = (column % QK_FP4_MMQ) / 32;
        block_fp4_mmq & block = image[(column / QK_FP4_MMQ) * rows + row];
        reinterpret_cast<char2 *>(block.qs)[sub * 8 + lane] = packed;
        if (lane == 0) {
            uint8_t * header = reinterpret_cast<uint8_t *>(block.d4) + (sub / 2) * sizeof(uint32_t);
            header[sub % 2] = e;
            if (sub % 2 == 0) { header[2] = 0; header[3] = 0; }
        }
    }

    __device__ __forceinline__ void zero(int64_t column, int64_t row) const {
        const int lane = (column % 32) / 4;
        const int sub = (column % QK_FP4_MMQ) / 32;
        block_fp4_mmq & block = image[(column / QK_FP4_MMQ) * rows + row];
        reinterpret_cast<char2 *>(block.qs)[sub * 8 + lane] = make_char2(0, 0);
        if (lane == 0) {
            uint8_t * header = reinterpret_cast<uint8_t *>(block.d4) + (sub / 2) * sizeof(uint32_t);
            header[sub % 2] = 0;
            if (sub % 2 == 0) { header[2] = 0; header[3] = 0; }
        }
    }

    __device__ __forceinline__ void operator()(float * dst, const float * base, int col, float4 value) const {
        dst[col] = value.x; dst[col + 1] = value.y; dst[col + 2] = value.z; dst[col + 3] = value.w;
        const int64_t index = dst - base + col;
        const int64_t row = index / cols;
        const int64_t column = index % cols;
        store(column, row, value);
        if (column >= cols - 32) {
            for (int64_t tail = cols; tail < padded; tail += 32) {
                zero(tail + column % 32, row);
            }
        }
    }
};

typedef void (*quantize_cuda_t)(
        const float * x, const int32_t * ids, void * vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

void quantize_row_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

void quantize_mmq_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

void quantize_mmq_fp4_cuda(const float *   x,
                             const int32_t * ids,
                             void *          vy,
                             float *         scale,
                             ggml_type       type_src0,
                             bool            use_aligned_float8,
                             int64_t         ne00,
                             int64_t         s01,
                             int64_t         s02,
                             int64_t         s03,
                             int64_t         ne0,
                             int64_t         ne1,
                             int64_t         ne2,
                             int64_t         ne3,
                             cudaStream_t    stream);

// quantize each token once and scatter the block to its compact rows (via the inverse map)
void quantize_scatter_mmq_fp4_cuda(const float *   x,
                                   const int32_t * ids_src1_inv,
                                   void *          vy,
                                   float *         scale,
                                   ggml_type       type_src0,
                                   bool            use_aligned_float8,
                                   int64_t         ne00,
                                   int64_t         stride_token,
                                   int64_t         ne0,
                                   int64_t         n_tokens,
                                   int64_t         nrows_dst,
                                   int             n_expert_used,
                                   cudaStream_t    stream);

void quantize_scatter_mmq_q8_1_cuda(const float *   x,
                                    const int32_t * ids_src1_inv,
                                    void *          vy,
                                    ggml_type       type_src0,
                                    int64_t         ne00,
                                    int64_t         stride_token,
                                    int64_t         ne0,
                                    int64_t         n_tokens,
                                    int64_t         nrows_dst,
                                    int             n_expert_used,
                                    cudaStream_t    stream);
