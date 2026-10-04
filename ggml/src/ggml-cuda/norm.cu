#include "norm.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "convert.cuh"
#include <cstdint>

template <int block_size>
static __global__ void norm_f32(
        const float * x, float * dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    float2 mean_var = make_float2(0.0f, 0.0f);

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        mean_var.x += xi;
        mean_var.y += xi * xi;
    }

    // sum up partial sums
    extern __shared__ float2 s_sum2[];
    mean_var = block_reduce<block_reduce_method::SUM, block_size>(mean_var, s_sum2);

    const float mean = mean_var.x / ncols;
    const float var = mean_var.y / ncols - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = (x[col] - mean) * inv_std;
    }
}

template <int block_size>
static __global__ void group_norm_f32(const float * x, float * dst, const int group_size, const int ne_elements, const float eps) {
    // blockIdx.x: num_groups idx
    // threadIdx.x: block_size idx
    const int start =     blockIdx.x*group_size + threadIdx.x;
    const int end   = min(blockIdx.x*group_size + group_size,  ne_elements);

    float tmp = 0.0f; // partial sum for thread in warp

    ggml_cuda_pdl_sync();
    for (int j = start; j < end; j += block_size) {
        tmp += x[j];
    }

    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / group_size;
    tmp = 0.0f;

    for (int j = start; j < end; j += block_size) {
        const float xi = x[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum + 32);

    const float variance = tmp / group_size;
    const float scale = rsqrtf(variance + eps);
    for (int j = start; j < end; j += block_size) {
        dst[j] *= scale;
    }
}

struct ggml_cuda_norm_store {
    static constexpr int width = 1;
    __device__ __forceinline__ void operator()(float * dst, const float * base, int col, float value) const {
        dst[col] = value;
        GGML_UNUSED(base);
    }
};

struct ggml_cuda_norm_emit_store {
    static constexpr int width = 1;
    half * f16;
    nv_bfloat16 * bf16;
    __device__ __forceinline__ void operator()(float * dst, const float * base, int col, float value) const {
        dst[col] = value;
        const int64_t index = dst - base + col;
        if (f16) { f16[index] = ggml_cuda_cast<half>(value); }
        if (bf16) { bf16[index] = ggml_cuda_cast<nv_bfloat16>(value); }
    }
};

struct ggml_cuda_norm_q8_store {
    static constexpr int width = 1;
    block_q8_1 * image;
    int64_t cols;
    int64_t padded;

    __device__ __forceinline__ void operator()(float * dst, const float * base, int col, float value) const {
        dst[col] = value;
        const int64_t index = dst - base + col;
        const int64_t row = index / cols;
        const int64_t column = index % cols;
        const int lane = column % QK8_1;
        const int64_t block = (row*padded + column)/QK8_1;
        const float amax = warp_reduce_max<QK8_1>(fabsf(value));
        const float sum = warp_reduce_sum<QK8_1>(value);
        const float d = amax/127.0f;
        image[block].qs[lane] = amax == 0.0f ? 0 : roundf(value/d);
        if (lane == 0) { image[block].ds = make_half2(d, sum); }
        if (column/QK8_1 == cols/QK8_1 - 1) {
            for (int64_t tail = cols/QK8_1; tail < padded/QK8_1; ++tail) {
                block_q8_1 & zero = image[row*(padded/QK8_1) + tail];
                zero.qs[lane] = 0;
                if (lane == 0) { zero.ds = make_half2(0.0f, 0.0f); }
            }
        }
    }
};

struct ggml_cuda_norm_emit_q8_store {
    static constexpr int width = 1;
    block_q8_1 * image;
    uint3 cols;
    int64_t padded;
    half * f16;
    nv_bfloat16 * bf16;

    __device__ __forceinline__ void operator()(float * dst, const float * base, int col, float value) const {
        dst[col] = value;
        const int64_t index = dst - base + col;
        const uint2 rc = fast_div_modulo(uint32_t(index), cols);
        const int64_t row = rc.x;
        const int64_t column = rc.y;
        const int lane = column % QK8_1;
        const int64_t block = (row*padded + column)/QK8_1;
        const float amax = warp_reduce_max<QK8_1>(fabsf(value));
        const float sum = warp_reduce_sum<QK8_1>(value);
        const float d = amax/127.0f;
        image[block].qs[lane] = amax == 0.0f ? 0 : roundf(value/d);
        if (lane == 0) { image[block].ds = make_half2(d, sum); }
        if (column/QK8_1 == cols.z/QK8_1 - 1) {
            for (int64_t tail = cols.z/QK8_1; tail < padded/QK8_1; ++tail) {
                block_q8_1 & zero = image[row*(padded/QK8_1) + tail];
                zero.qs[lane] = 0;
                if (lane == 0) { zero.ds = make_half2(0.0f, 0.0f); }
            }
        }
        if (f16) { f16[index] = ggml_cuda_cast<half>(value); }
        if (bf16) { bf16[index] = ggml_cuda_cast<nv_bfloat16>(value); }
    }
};

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

template <int block_size, bool do_multiply, bool do_add, bool do_scale, typename Write>
static __device__ __forceinline__ void rms_norm_f32_impl(const float * x,
                                    float *       dst,
                                    const int     ncols,
                                    const int64_t stride_row,
                                    const int64_t stride_channel,
                                    const int64_t stride_sample,
                                    const float   eps,
                                    const Write   write,
                                    const float * mul                  = nullptr,
                                    const int64_t mul_stride_row       = 0,
                                    const int64_t mul_stride_channel   = 0,
                                    const int64_t mul_stride_sample    = 0,
                                    const uint3   mul_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   mul_nsamples_packed  = make_uint3(0, 0, 0),
                                    const float * add                  = nullptr,
                                    const int64_t add_stride_row       = 0,
                                    const int64_t add_stride_channel   = 0,
                                    const int64_t add_stride_sample    = 0,
                                    const uint3   add_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   add_nsamples_packed  = make_uint3(0, 0, 0),
                                    const float   scale_out            = 1.0f) {
    ggml_cuda_pdl_lc();
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");
    static_assert(!do_scale || !do_multiply, "fusing scale is not supported with multiplying");

    const float * dst_base = dst;
    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    if constexpr (do_multiply) {
        const uint32_t mul_row     = fastmodulo(row, mul_nrows_packed);
        const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
        const uint32_t mul_sample  = fastmodulo(sample, mul_nsamples_packed);
        mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;
    }

    if constexpr (do_add) {
        const int add_row     = fastmodulo(row, add_nrows_packed);
        const int add_channel = fastmodulo(channel, add_nchannels_packed);
        const int add_sample  = fastmodulo(sample, add_nsamples_packed);
        add += add_sample * add_stride_sample + add_channel * add_stride_channel + add_row * add_stride_row;
    }

    float tmp = 0.0f; // partial sum for thread in warp

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    // sum up partial sums
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    if constexpr (Write::width == 4) {
        for (int col = 4*tid; col < ncols; col += 4*block_size) {
            float values[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int column = col + j;
                if constexpr (do_multiply && do_add) {
                    const int mul_col = fastmodulo(column, mul_ncols_packed);
                    const int add_col = fastmodulo(column, add_ncols_packed);
                    values[j] = scale * x[column] * mul[mul_col] + add[add_col];
                } else if constexpr (do_multiply) {
                    const int mul_col = fastmodulo(column, mul_ncols_packed);
                    values[j] = scale * x[column] * mul[mul_col];
                } else if constexpr (do_scale) {
                    values[j] = scale_out * (scale * x[column]);
                } else { values[j] = scale * x[column]; }
            }
            write(dst, dst_base, col, make_float4(values[0], values[1], values[2], values[3]));
        }
    } else {
    for (int col = tid; col < ncols; col += block_size) {
        if constexpr (do_multiply && do_add) {
            const int mul_col = fastmodulo(col, mul_ncols_packed);
            const int add_col = fastmodulo(col, add_ncols_packed);
            write(dst, dst_base, col, scale * x[col] * mul[mul_col] + add[add_col]);
        } else if constexpr (do_multiply) {
            const int mul_col = fastmodulo(col, mul_ncols_packed);
            write(dst, dst_base, col, scale * x[col] * mul[mul_col]);
        } else if constexpr (do_scale) {
            write(dst, dst_base, col, scale_out * (scale * x[col]));
        } else {
            write(dst, dst_base, col, scale * x[col]);
        }
    }
    }
}

template <int block_size, bool do_multiply = false, bool do_add = false, bool do_scale = false>
static __global__ void rms_norm_f32(const float * x,
                                    float *       dst,
                                    const int     ncols,
                                    const int64_t stride_row,
                                    const int64_t stride_channel,
                                    const int64_t stride_sample,
                                    const float   eps,
                                    const float * mul                  = nullptr,
                                    const int64_t mul_stride_row       = 0,
                                    const int64_t mul_stride_channel   = 0,
                                    const int64_t mul_stride_sample    = 0,
                                    const uint3   mul_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   mul_nsamples_packed  = make_uint3(0, 0, 0),
                                    const float * add                  = nullptr,
                                    const int64_t add_stride_row       = 0,
                                    const int64_t add_stride_channel   = 0,
                                    const int64_t add_stride_sample    = 0,
                                    const uint3   add_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   add_nsamples_packed  = make_uint3(0, 0, 0),
                                    const float   scale_out            = 1.0f) {
    rms_norm_f32_impl<block_size, do_multiply, do_add, do_scale>(
        x, dst, ncols, stride_row, stride_channel, stride_sample, eps, ggml_cuda_norm_store{}, mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, add, add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed, add_nchannels_packed, add_nsamples_packed, scale_out);
}

template <int block_size, bool do_multiply = false, bool do_add = false, bool do_scale = false>
static __global__ void rms_norm_emit_f32(const float * x,
                                    float *       dst,
                                    const int     ncols,
                                    const int64_t stride_row,
                                    const int64_t stride_channel,
                                    const int64_t stride_sample,
                                    const float   eps,
                                    half *        f16,
                                    nv_bfloat16 * bf16,
                                    const float * mul                  = nullptr,
                                    const int64_t mul_stride_row       = 0,
                                    const int64_t mul_stride_channel   = 0,
                                    const int64_t mul_stride_sample    = 0,
                                    const uint3   mul_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   mul_nsamples_packed  = make_uint3(0, 0, 0),
                                    const float * add                  = nullptr,
                                    const int64_t add_stride_row       = 0,
                                    const int64_t add_stride_channel   = 0,
                                    const int64_t add_stride_sample    = 0,
                                    const uint3   add_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   add_nsamples_packed  = make_uint3(0, 0, 0),
                                    const float   scale_out            = 1.0f) {
    rms_norm_f32_impl<block_size, do_multiply, do_add, do_scale>(
        x, dst, ncols, stride_row, stride_channel, stride_sample, eps, ggml_cuda_norm_emit_store{f16, bf16}, mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, add, add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed, add_nchannels_packed, add_nsamples_packed, scale_out);
}

template <int block_size, bool do_multiply = false, bool do_add = false, bool do_scale = false>
static __global__ void rms_norm_q8_f32(const float * x,
                                    float *       dst,
                                    const int     ncols,
                                    const int64_t stride_row,
                                    const int64_t stride_channel,
                                    const int64_t stride_sample,
                                    const float   eps,
                                    void *        image,
                                    int64_t       cols,
                                    int64_t       padded,
                                    const float * mul                  = nullptr,
                                    const int64_t mul_stride_row       = 0,
                                    const int64_t mul_stride_channel   = 0,
                                    const int64_t mul_stride_sample    = 0,
                                    const uint3   mul_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   mul_nsamples_packed  = make_uint3(0, 0, 0),
                                    const float * add                  = nullptr,
                                    const int64_t add_stride_row       = 0,
                                    const int64_t add_stride_channel   = 0,
                                    const int64_t add_stride_sample    = 0,
                                    const uint3   add_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   add_nsamples_packed  = make_uint3(0, 0, 0),
                                    const float   scale_out            = 1.0f) {
    rms_norm_f32_impl<block_size, do_multiply, do_add, do_scale>(
        x, dst, ncols, stride_row, stride_channel, stride_sample, eps, ggml_cuda_norm_q8_store{(block_q8_1 *) image, cols, padded}, mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, add, add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed, add_nchannels_packed, add_nsamples_packed, scale_out);
}

template <int block_size, bool do_multiply = false, bool do_add = false, bool do_scale = false>
static __global__ void rms_norm_emit_q8_f32(const float * x,
                                    float *       dst,
                                    const int     ncols,
                                    const int64_t stride_row,
                                    const int64_t stride_channel,
                                    const int64_t stride_sample,
                                    const float   eps,
                                    half *        f16,
                                    nv_bfloat16 * bf16,
                                    void *        image,
                                    uint3         cols,
                                    int64_t       padded,
                                    const float * mul                  = nullptr,
                                    const int64_t mul_stride_row       = 0,
                                    const int64_t mul_stride_channel   = 0,
                                    const int64_t mul_stride_sample    = 0,
                                    const uint3   mul_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   mul_nsamples_packed  = make_uint3(0, 0, 0),
                                    const float * add                  = nullptr,
                                    const int64_t add_stride_row       = 0,
                                    const int64_t add_stride_channel   = 0,
                                    const int64_t add_stride_sample    = 0,
                                    const uint3   add_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   add_nsamples_packed  = make_uint3(0, 0, 0),
                                    const float   scale_out            = 1.0f) {
    rms_norm_f32_impl<block_size, do_multiply, do_add, do_scale>(
        x, dst, ncols, stride_row, stride_channel, stride_sample, eps, ggml_cuda_norm_emit_q8_store{(block_q8_1 *) image, cols, padded, f16, bf16}, mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, add, add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed, add_nchannels_packed, add_nsamples_packed, scale_out);
}

template <mmq_q8_1_ds_layout Layout, int block_size, bool do_multiply = false, bool do_add = false, bool do_scale = false>
static __global__ void rms_norm_mmq_f32(const float * x,
                                    float *       dst,
                                    const int     ncols,
                                    const int64_t stride_row,
                                    const int64_t stride_channel,
                                    const int64_t stride_sample,
                                    const float   eps,
                                    void *        image,
                                    int64_t       cols,
                                    int64_t       padded,
                                    int64_t       rows,
                                    const float * mul                  = nullptr,
                                    const int64_t mul_stride_row       = 0,
                                    const int64_t mul_stride_channel   = 0,
                                    const int64_t mul_stride_sample    = 0,
                                    const uint3   mul_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   mul_nsamples_packed  = make_uint3(0, 0, 0),
                                    const float * add                  = nullptr,
                                    const int64_t add_stride_row       = 0,
                                    const int64_t add_stride_channel   = 0,
                                    const int64_t add_stride_sample    = 0,
                                    const uint3   add_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   add_nsamples_packed  = make_uint3(0, 0, 0),
                                    const float   scale_out            = 1.0f) {
    rms_norm_f32_impl<block_size, do_multiply, do_add, do_scale>(
        x, dst, ncols, stride_row, stride_channel, stride_sample, eps, ggml_cuda_norm_mmq_store<Layout>{(block_q8_1_mmq *) image, cols, padded, rows}, mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, add, add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed, add_nchannels_packed, add_nsamples_packed, scale_out);
}

template <int block_size, bool do_multiply = false, bool do_add = false, bool do_scale = false>
static __global__ void rms_norm_mxfp4_f32(const float * x,
                                    float *       dst,
                                    const int     ncols,
                                    const int64_t stride_row,
                                    const int64_t stride_channel,
                                    const int64_t stride_sample,
                                    const float   eps,
                                    void *        image,
                                    int64_t       cols,
                                    int64_t       padded,
                                    int64_t       rows,
                                    const float * mul                  = nullptr,
                                    const int64_t mul_stride_row       = 0,
                                    const int64_t mul_stride_channel   = 0,
                                    const int64_t mul_stride_sample    = 0,
                                    const uint3   mul_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   mul_nsamples_packed  = make_uint3(0, 0, 0),
                                    const float * add                  = nullptr,
                                    const int64_t add_stride_row       = 0,
                                    const int64_t add_stride_channel   = 0,
                                    const int64_t add_stride_sample    = 0,
                                    const uint3   add_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   add_nsamples_packed  = make_uint3(0, 0, 0),
                                    const float   scale_out            = 1.0f) {
    rms_norm_f32_impl<block_size, do_multiply, do_add, do_scale>(
        x, dst, ncols, stride_row, stride_channel, stride_sample, eps, ggml_cuda_norm_mxfp4_store{(block_fp4_mmq *) image, cols, padded, rows}, mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, add, add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed, add_nchannels_packed, add_nsamples_packed, scale_out);
}

template <int block_size>
static __global__ void rms_norm_back_f32(
        const float * grad, const float * xf, float * dst, const int ncols, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    grad += int64_t(row)*ncols;
    xf   += int64_t(row)*ncols;
    dst  += int64_t(row)*ncols;

    float sum_xx = 0.0f; // sum for squares of x, equivalent to forward pass
    float sum_xg = 0.0f; // sum for x * gradient, needed because RMS norm mixes inputs

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xfi = xf[col];
        sum_xx += xfi * xfi;
        sum_xg += xfi * grad[col];
    }

    // sum up partial sums
    sum_xx = warp_reduce_sum(sum_xx);
    sum_xg = warp_reduce_sum(sum_xg);
    if constexpr (block_size > WARP_SIZE) {
        static_assert(block_size == 1024, "unexpected block_size");
        __shared__ float s_sum_xx[32];
        __shared__ float s_sum_xg[32];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum_xx[warp_id] = sum_xx;
            s_sum_xg[warp_id] = sum_xg;
        }
        __syncthreads();

        sum_xx = s_sum_xx[lane_id];
        sum_xx = warp_reduce_sum(sum_xx);

        sum_xg = s_sum_xg[lane_id];
        sum_xg = warp_reduce_sum(sum_xg);
    }

    const float mean_eps = sum_xx / ncols + eps;
    const float sum_eps  = sum_xx + ncols*eps;

    const float scale_grad = rsqrtf(mean_eps);
    const float scale_x    = -scale_grad * sum_xg/sum_eps;

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale_grad*grad[col] + scale_x*xf[col];
    }
}

// template <int block_size>
// static __global__ void l2_norm_f32(const float * x, float * dst, const int ncols, const float eps) {
//     const int row = blockIdx.x*blockDim.y + threadIdx.y;
//     const int tid = threadIdx.x;

//     float tmp = 0.0f; // partial sum for thread in warp

//     for (int col = tid; col < ncols; col += block_size) {
//         const float xi = x[row*ncols + col];
//         tmp += xi * xi;
//     }

//     // sum up partial sums
//     tmp = warp_reduce_sum(tmp);
//     if (block_size > WARP_SIZE) {
//         __shared__ float s_sum[32];
//         int warp_id = threadIdx.x / WARP_SIZE;
//         int lane_id = threadIdx.x % WARP_SIZE;
//         if (lane_id == 0) {
//             s_sum[warp_id] = tmp;
//         }
//         __syncthreads();
//         tmp = s_sum[lane_id];
//         tmp = warp_reduce_sum(tmp);
//     }

//     // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
//     const float scale = rsqrtf(fmaxf(tmp, eps * eps));

//     for (int col = tid; col < ncols; col += block_size) {
//         dst[row*ncols + col] = scale * x[row*ncols + col];
//     }
// }

template <int block_size>
static __global__ void l2_norm_f32(
        const float * x, float * dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    float tmp = 0.0f; // partial sum for thread in warp

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    // sum up partial sums
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
    ggml_cuda_pdl_lc();

    // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
    const float scale = rsqrtf(fmaxf(tmp, eps * eps));

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale * x[col];
    }
}

static void norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        norm_f32<WARP_SIZE><<<blocks_num, block_dims, 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        norm_f32<1024><<<blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float2): 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
}

static void group_norm_f32_cuda(
        const float * x, float * dst, const int num_groups, const float eps, const int group_size, const int ne_elements, cudaStream_t stream) {
    if (group_size < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        group_norm_f32<WARP_SIZE><<<num_groups, block_dims, 0, stream>>>(x, dst, group_size, ne_elements, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        group_norm_f32<1024><<<num_groups, block_dims, block_dims.x > WARP_SIZE ? 2 * 32 * sizeof(float): 0, stream>>>(x, dst, group_size, ne_elements, eps);
    }
}

template <bool do_scale = false>
static void rms_norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream,
        const float scale_out = 1.0f) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(256, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = {blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
        ggml_cuda_kernel_launch(rms_norm_f32<256, false, false, do_scale>, launch_params,
            x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
        // underlying cudaLaunchKernelEx does not support default params
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), scale_out);
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
        ggml_cuda_kernel_launch(rms_norm_f32<1024, false, false, do_scale>, launch_params, x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
        // underlying cudaLaunchKernelEx does not support default params
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), scale_out);
    }
}

static void rms_norm_mul_f32_cuda(const float *  x,
                                  const float *  mul,
                                  const float *  add,
                                  float *        dst,
                                  const int      ncols,
                                  const int      nrows,
                                  const int      nchannels,
                                  const int      nsamples,
                                  const int64_t  stride_row,
                                  const int64_t  stride_channel,
                                  const int64_t  stride_sample,
                                  const int64_t  mul_stride_row,
                                  const int64_t  mul_stride_channel,
                                  const int64_t  mul_stride_sample,
                                  const uint32_t mul_ncols,
                                  const uint32_t mul_nrows,
                                  const uint32_t mul_nchannels,
                                  const uint32_t mul_nsamples,
                                  const int64_t  add_stride_row,
                                  const int64_t  add_stride_channel,
                                  const int64_t  add_stride_sample,
                                  const uint32_t add_ncols,
                                  const uint32_t add_nrows,
                                  const uint32_t add_nchannels,
                                  const uint32_t add_nsamples,
                                  const float    eps,
                                  cudaStream_t   stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (mul == nullptr) {
        rms_norm_f32_cuda(x, dst, ncols, nrows, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps, stream);
        return;
    }
    if (add == nullptr) {
        const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
        const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
        const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
        const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);
        if (ncols < 1024) {
            const dim3 block_dims(256, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            ggml_cuda_kernel_launch(rms_norm_f32<256, true>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                // underlying cudaLaunchKernelEx does not support default params
            nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), 1.0f);
        } else {
            const dim3 block_dims(1024, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            ggml_cuda_kernel_launch(rms_norm_f32<1024, true>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                // underlying cudaLaunchKernelEx does not support default params
            nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), 1.0f);
        }
    } else {
        const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
        const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
        const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
        const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);

        const uint3 add_ncols_packed     = init_fastdiv_values(add_ncols);
        const uint3 add_nrows_packed     = init_fastdiv_values(add_nrows);
        const uint3 add_nchannels_packed = init_fastdiv_values(add_nchannels);
        const uint3 add_nsamples_packed  = init_fastdiv_values(add_nsamples);
        if (ncols < 1024) {
            const dim3 block_dims(256, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims,block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            ggml_cuda_kernel_launch(rms_norm_f32<256, true, true>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, add,
                add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed,
                add_nchannels_packed, add_nsamples_packed, 1.0f);
        } else {
            const dim3 block_dims(1024, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            ggml_cuda_kernel_launch(rms_norm_f32<1024, true, true>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, add,
                add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed,
                add_nchannels_packed, add_nsamples_packed, 1.0f);
        }
    }
}

static void rms_norm_back_f32_cuda(const float * grad, const float * xf, float * dst, const int ncols, const int nrows, const float eps, cudaStream_t stream) {
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        rms_norm_back_f32<WARP_SIZE><<<nrows, block_dims, 0, stream>>>(grad, xf, dst, ncols, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        rms_norm_back_f32<1024><<<nrows, block_dims, 0, stream>>>(grad, xf, dst, ncols, eps);
    }
}

static void l2_norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, 0, stream};
        ggml_cuda_kernel_launch(l2_norm_f32<WARP_SIZE>, launch_params, x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
        ggml_cuda_kernel_launch(l2_norm_f32<1024>, launch_params, x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
}

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    int num_groups = dst->op_params[0];

    float eps;
    memcpy(&eps, dst->op_params + 1, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    int group_size = src0->ne[0] * src0->ne[1] * ((src0->ne[2] + num_groups - 1) / num_groups);
    group_norm_f32_cuda(src0_d, dst_d, num_groups * src0->ne[3], eps, group_size, ggml_nelements(src0), stream);
}

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    rms_norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

void ggml_cuda_op_rms_norm_scale_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale_tensor) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    float *             dst_d  = (float *) scale_tensor->data;
    cudaStream_t        stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(scale_tensor->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    float scale;
    memcpy(&scale, (const float *) scale_tensor->op_params + 0, sizeof(float));

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    rms_norm_f32_cuda<true>(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream, scale);
}

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor) {
    const ggml_tensor * rms_norm_src = (ggml_tensor *) dst->src[0];
    float eps = 0.0f;

    memcpy(&eps, dst->op_params, sizeof(float));

    const float * src0_d = (const float *) rms_norm_src->data;
    const float * mul_d = nullptr;
    const ggml_tensor * mul_src = nullptr;

    if (mul_tensor->src[0] == dst) {
        mul_d = (float *) mul_tensor->src[1]->data;
        mul_src = mul_tensor->src[1];
    } else if(mul_tensor->src[1] == dst) {
        mul_d = (float *) mul_tensor->src[0]->data;
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    float * dst_d = (float *) mul_tensor->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const int mul_ncols     = mul_src->ne[0];
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    rms_norm_mul_f32_cuda(src0_d, mul_d, nullptr, dst_d,
                          ne00, ne01, ne02, ne03,
                          /*s00*/ s01, s02, s03,
                          /*mul_s00*/ mul_s01, mul_s02, mul_s03,
                          mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                          /*add_s00*/ 0, 0, 0,
                          0, 0, 0, 0,
                          eps, stream);
}

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor) {
    const ggml_tensor * rms_norm_src = (ggml_tensor *) dst->src[0];
    float               eps          = 0.0f;

    memcpy(&eps, dst->op_params, sizeof(float));

    const float *       src0_d  = (const float *) rms_norm_src->data;
    const float *       mul_d   = nullptr;
    const ggml_tensor * mul_src = nullptr;

    if (mul_tensor->src[0] == dst) {
        mul_d   = (float *) mul_tensor->src[1]->data;
        mul_src = mul_tensor->src[1];
    } else if (mul_tensor->src[1] == dst) {
        mul_d   = (float *) mul_tensor->src[0]->data;
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    const float *       add_d   = nullptr;
    const ggml_tensor * add_src = nullptr;

    if (add_tensor->src[0] == mul_tensor) {
        add_d   = (float *) add_tensor->src[1]->data;
        add_src = add_tensor->src[1];
    } else if (add_tensor->src[1] == mul_tensor) {
        add_d   = (float *) add_tensor->src[0]->data;
        add_src = add_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    float *      dst_d  = (float *) add_tensor->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(add_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const int mul_ncols     = mul_src->ne[0];
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    const size_t ts_add = ggml_type_size(add_src->type);
    GGML_ASSERT(add_src->nb[0] == ts_add);
    const int64_t add_s01 = add_src->nb[1] / ts_add;
    const int64_t add_s02 = add_src->nb[2] / ts_add;
    const int64_t add_s03 = add_src->nb[3] / ts_add;

    const int add_ncols     = add_src->ne[0];
    const int add_nrows     = add_src->ne[1];
    const int add_nchannels = add_src->ne[2];
    const int add_nsamples  = add_src->ne[3];

    rms_norm_mul_f32_cuda(src0_d, mul_d,add_d,dst_d,
                          ne00,ne01, ne02, ne03,
                          /*s00*/ s01, s02, s03,
                          /*mul_s00*/ mul_s01, mul_s02, mul_s03,
                          mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                          /*add_s00*/ add_s01, add_s02, add_s03,
                          add_ncols, add_nrows, add_nchannels, add_nsamples,
                          eps, stream);
}

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * grad  = dst->src[0]; // gradients
    const ggml_tensor * src0f = dst->src[1]; // src0 from forward pass

    const float * grad_d  = (const float *) grad->data;
    const float * src0f_d = (const float *) src0f->data;
    float       * dst_d   = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(grad));

    GGML_ASSERT( grad->type == GGML_TYPE_F32);
    GGML_ASSERT(src0f->type == GGML_TYPE_F32);
    GGML_ASSERT(  dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0f->ne[0];
    const int64_t nrows = ggml_nrows(src0f);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    rms_norm_back_f32_cuda(grad_d, src0f_d, dst_d, ne00, nrows, eps, stream);
}

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    l2_norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

template <int Block, bool Multiply, bool Add, bool Scale>
static void ggml_cuda_rms_norm_emit_launch(ggml_backend_cuda_context & ctx, ggml_tensor * norm,
        ggml_tensor * mul, ggml_tensor * add, ggml_tensor * scale, void * f16, void * bf16) {
    const ggml_tensor * x = norm->src[0];
    const ggml_tensor * weight = mul ? (mul->src[0] == norm ? mul->src[1] : mul->src[0]) : nullptr;
    const ggml_tensor * bias = add ? (add->src[0] == mul ? add->src[1] : add->src[0]) : nullptr;
    ggml_tensor * dst = scale ? scale : add ? add : mul ? mul : norm;
    const auto pointer = [](const ggml_tensor * t) { return t ? (const float *) t->data : nullptr; };
    const auto stride = [](const ggml_tensor * t, int d) { return t ? int64_t(t->nb[d]/sizeof(float)) : int64_t(0); };
    const auto divisor = [](const ggml_tensor * t, int d) { return t ? init_fastdiv_values(t->ne[d]) : make_uint3(0, 0, 0); };
    const ggml_cuda_kernel_launch_params params = {
        dim3(x->ne[1], x->ne[2], x->ne[3]), dim3(Block, 1, 1), 32*sizeof(float), ctx.stream()};
    ggml_cuda_kernel_launch(rms_norm_emit_f32<Block, Multiply, Add, Scale>, params,
        (const float *) x->data, (float *) dst->data, int(x->ne[0]), stride(x, 1), stride(x, 2), stride(x, 3),
        ggml_get_op_params_f32(norm, 0), (half *) f16, (nv_bfloat16 *) bf16,
        pointer(weight), stride(weight, 1), stride(weight, 2), stride(weight, 3),
        divisor(weight, 0), divisor(weight, 1), divisor(weight, 2), divisor(weight, 3),
        pointer(bias), stride(bias, 1), stride(bias, 2), stride(bias, 3),
        divisor(bias, 0), divisor(bias, 1), divisor(bias, 2), divisor(bias, 3),
        scale ? ggml_get_op_params_f32(scale, 0) : 1.0f);
}

template <bool Multiply, bool Add, bool Scale>
static void ggml_cuda_rms_norm_emit(ggml_backend_cuda_context & ctx, ggml_tensor * norm,
        ggml_tensor * mul, ggml_tensor * add, ggml_tensor * scale, void * f16, void * bf16) {
    if (norm->src[0]->ne[0] < 1024) {
        ggml_cuda_rms_norm_emit_launch<256, Multiply, Add, Scale>(ctx, norm, mul, add, scale, f16, bf16);
    } else {
        ggml_cuda_rms_norm_emit_launch<1024, Multiply, Add, Scale>(ctx, norm, mul, add, scale, f16, bf16);
    }
}

void ggml_cuda_op_rms_norm_emit(ggml_backend_cuda_context & ctx, ggml_tensor * norm,
        ggml_tensor * mul, ggml_tensor * add, ggml_tensor * scale, void * f16, void * bf16) {
    GGML_ASSERT(ggml_get_op_params_f32(norm, 0) >= 0.0f);
    if (add) { ggml_cuda_rms_norm_emit<true, true, false>(ctx, norm, mul, add, scale, f16, bf16); }
    else if (mul) { ggml_cuda_rms_norm_emit<true, false, false>(ctx, norm, mul, add, scale, f16, bf16); }
    else if (scale) { ggml_cuda_rms_norm_emit<false, false, true>(ctx, norm, mul, add, scale, f16, bf16); }
    else { ggml_cuda_rms_norm_emit<false, false, false>(ctx, norm, mul, add, scale, f16, bf16); }
}

template <mmq_q8_1_ds_layout Layout, bool Scale>
struct ggml_cuda_hc_norm_emit_mmq_pair_store {
    static constexpr int width = 2;
    block_q8_1_mmq * image;
    uint3 cols;
    int64_t padded;
    int64_t rows;
    half * f16;
    nv_bfloat16 * bf16;
    float scale;

    __device__ __forceinline__ void store(int64_t column, int64_t row, float2 xi) const {
        constexpr int vals_per_scale = Layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
        constexpr int vals_per_sum = Layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;
        const unsigned mask = __activemask();
        float amax = fmaxf(fabsf(xi.x), fabsf(xi.y));
#pragma unroll
        for (int offset = vals_per_scale/4; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(mask, amax, offset, WARP_SIZE));
        }
        float sum;
        if constexpr (Layout != MMQ_Q8_1_DS_LAYOUT_D4) {
            const int pair = threadIdx.x % WARP_SIZE & ~1;
            const float x = __shfl_sync(mask, xi.x, pair, WARP_SIZE);
            const float y = __shfl_sync(mask, xi.y, pair, WARP_SIZE);
            const float z = __shfl_sync(mask, xi.x, pair + 1, WARP_SIZE);
            const float w = __shfl_sync(mask, xi.y, pair + 1, WARP_SIZE);
            sum = x + y + z + w;
#pragma unroll
            for (int offset = vals_per_sum/4; offset >= 2; offset >>= 1) {
                sum += __shfl_xor_sync(mask, sum, offset, WARP_SIZE);
            }
        }
        const float d_inv = 127.0f/amax;
        const float d = 1.0f/d_inv;
        const int64_t ib = (column/QK8_1_MMQ)*rows + row;
        const int iqs = column % QK8_1_MMQ;
        char2 q;
        q.x = roundf(xi.x*d_inv);
        q.y = roundf(xi.y*d_inv);
        ((char2 *) image[ib].qs)[iqs/2] = q;
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

    __device__ __forceinline__ void operator()(float * dst, const float * base, int col, float2 xi) const {
        if constexpr (Scale) {
            xi.x = scale * xi.x;
            xi.y = scale * xi.y;
        }
        dst[col] = xi.x;
        dst[col + 1] = xi.y;
        const int64_t index = dst - base + col;
        const uint2 rc = fast_div_modulo(uint32_t(index), cols);
        store(rc.y, rc.x, xi);
        constexpr int vals_per_scale = Layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
        if (rc.y >= cols.z - vals_per_scale) {
            for (int64_t tail = cols.z; tail < padded; tail += vals_per_scale) {
                store(tail + rc.y % vals_per_scale, rc.x, make_float2(0.0f, 0.0f));
            }
        }
        if (f16) {
            f16[index] = ggml_cuda_cast<half>(xi.x);
            f16[index + 1] = ggml_cuda_cast<half>(xi.y);
        }
        if (bf16) {
            bf16[index] = ggml_cuda_cast<nv_bfloat16>(xi.x);
            bf16[index + 1] = ggml_cuda_cast<nv_bfloat16>(xi.y);
        }
    }
};

struct hc_post_norm_data {
    const float * x;
    const float * residual;
    const float * post;
    const float * comb;
    const float * mul;
    float * post_dst;
    float * dst;
    int64_t n_embd;
    int64_t hc;
    int ncols;
    int64_t sx[2];
    int64_t sr[3];
    int64_t sp[2];
    int64_t sc[3];
    int64_t sm[4];
    uint3 mul_ne[4];
    uint3 embd_ne;
    float eps;
};

template <int block_size, bool has_comb, bool do_multiply, int layout>
static __global__ __launch_bounds__(block_size) void hc_post_norm_f32(hc_post_norm_data a) {
    ggml_cuda_pdl_lc();
    const int row = blockIdx.x;
    const int channel = blockIdx.y;
    const int sample = blockIdx.z;
    const int64_t offset = ((int64_t(sample)*gridDim.y + channel)*gridDim.x + row)*a.ncols;
    if constexpr (do_multiply) {
        a.mul += fastmodulo(sample, a.mul_ne[3])*a.sm[3] + fastmodulo(channel, a.mul_ne[2])*a.sm[2] + fastmodulo(row, a.mul_ne[1])*a.sm[1];
    }
    constexpr bool grouped = layout == 1;
    constexpr bool flat = layout == 2;
    const int64_t stream = grouped ? (offset / a.n_embd) % a.hc : 0;
    const int64_t token = grouped || flat ? offset / (a.n_embd*a.hc) : 0;
    ggml_cuda_pdl_sync();
    float tmp = 0.0f;
    for (int col = threadIdx.x; col < a.ncols; col += block_size) {
        const int64_t ir = offset + col;
        const int64_t i0 = grouped ? col : flat ? fastmodulo(col, a.embd_ne) : ir % a.n_embd;
        const int64_t ih = grouped ? stream : flat ? fastdiv(col, a.embd_ne) : (ir / a.n_embd) % a.hc;
        const int64_t it = grouped || flat ? token : ir / (a.n_embd*a.hc);
        float value = a.x[i0*a.sx[0] + it*a.sx[1]] * a.post[ih*a.sp[0] + it*a.sp[1]];
        if constexpr (has_comb) {
            for (int64_t isrc = 0; isrc < a.hc; ++isrc) {
                value += a.residual[i0*a.sr[0] + isrc*a.sr[1] + it*a.sr[2]] * a.comb[ih*a.sc[0] + isrc*a.sc[1] + it*a.sc[2]];
            }
        } else {
            value += a.residual[i0*a.sr[0] + ih*a.sr[1] + it*a.sr[2]];
        }
        a.post_dst[ir] = value;
        tmp += value * value;
    }
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
    const float mean = tmp / a.ncols;
    const float scale = rsqrtf(mean + a.eps);
    for (int col = threadIdx.x; col < a.ncols; col += block_size) {
        if constexpr (do_multiply) {
            a.dst[offset + col] = scale * a.post_dst[offset + col] * a.mul[fastmodulo(col, a.mul_ne[0])];
        } else {
            a.dst[offset + col] = scale * a.post_dst[offset + col];
        }
    }
}

template <int block_size, bool has_comb, bool do_multiply, int layout, typename Write>
static __device__ __forceinline__ void hc_post_norm_f32_impl(hc_post_norm_data a, const Write write) {
    ggml_cuda_pdl_lc();
    const int row = blockIdx.x;
    const int channel = blockIdx.y;
    const int sample = blockIdx.z;
    const int64_t offset = ((int64_t(sample)*gridDim.y + channel)*gridDim.x + row)*a.ncols;
    if constexpr (do_multiply) {
        a.mul += fastmodulo(sample, a.mul_ne[3])*a.sm[3] + fastmodulo(channel, a.mul_ne[2])*a.sm[2] + fastmodulo(row, a.mul_ne[1])*a.sm[1];
    }
    constexpr bool grouped = layout == 1;
    constexpr bool flat = layout == 2;
    const int64_t stream = grouped ? (offset / a.n_embd) % a.hc : 0;
    const int64_t token = grouped || flat ? offset / (a.n_embd*a.hc) : 0;
    ggml_cuda_pdl_sync();
    float tmp = 0.0f;
    for (int col = threadIdx.x; col < a.ncols; col += block_size) {
        const int64_t ir = offset + col;
        const int64_t i0 = grouped ? col : flat ? fastmodulo(col, a.embd_ne) : ir % a.n_embd;
        const int64_t ih = grouped ? stream : flat ? fastdiv(col, a.embd_ne) : (ir / a.n_embd) % a.hc;
        const int64_t it = grouped || flat ? token : ir / (a.n_embd*a.hc);
        float value = a.x[i0*a.sx[0] + it*a.sx[1]] * a.post[ih*a.sp[0] + it*a.sp[1]];
        if constexpr (has_comb) {
            for (int64_t isrc = 0; isrc < a.hc; ++isrc) {
                value += a.residual[i0*a.sr[0] + isrc*a.sr[1] + it*a.sr[2]] * a.comb[ih*a.sc[0] + isrc*a.sc[1] + it*a.sc[2]];
            }
        } else {
            value += a.residual[i0*a.sr[0] + ih*a.sr[1] + it*a.sr[2]];
        }
        a.post_dst[ir] = value;
        tmp += value * value;
    }
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
    const float mean = tmp / a.ncols;
    const float scale = rsqrtf(mean + a.eps);
    if constexpr (Write::width == 2) {
        for (int64_t col = 2*threadIdx.x; col < a.ncols; col += 2*block_size) {
            float x, y;
            if constexpr (do_multiply) {
                x = scale * a.post_dst[offset + col] * a.mul[fastmodulo(col, a.mul_ne[0])];
                y = scale * a.post_dst[offset + col + 1] * a.mul[fastmodulo(col + 1, a.mul_ne[0])];
            } else {
                x = scale * a.post_dst[offset + col];
                y = scale * a.post_dst[offset + col + 1];
            }
            write(a.dst + offset, a.dst, int(col), make_float2(x, y));
        }
    } else {
    for (int col = threadIdx.x; col < a.ncols; col += block_size) {
        if constexpr (do_multiply) {
            write(a.dst + offset, a.dst, col, scale * a.post_dst[offset + col] * a.mul[fastmodulo(col, a.mul_ne[0])]);
        } else {
            write(a.dst + offset, a.dst, col, scale * a.post_dst[offset + col]);
        }
    }
    }
}

template <int block_size, bool has_comb, bool do_multiply, int layout>
static __global__ __launch_bounds__(block_size) void hc_post_norm_emit_f32(hc_post_norm_data a, half * f16, nv_bfloat16 * bf16) {
    hc_post_norm_f32_impl<block_size, has_comb, do_multiply, layout>(a, ggml_cuda_norm_emit_store{f16, bf16});
}

template <int block_size, bool has_comb, bool do_multiply, int layout>
static __global__ __launch_bounds__(block_size) void hc_post_norm_emit_q8_f32(hc_post_norm_data a, half * f16, nv_bfloat16 * bf16, void * image, uint3 cols, int64_t padded) {
    hc_post_norm_f32_impl<block_size, has_comb, do_multiply, layout>(a, ggml_cuda_norm_emit_q8_store{(block_q8_1 *) image, cols, padded, f16, bf16});
}

template <mmq_q8_1_ds_layout Layout, int Block, bool Comb, bool Multiply, bool Scale, int NormLayout>
static __global__ __launch_bounds__(Block) void hc_post_norm_emit_mmq_f32(hc_post_norm_data a,
        half * f16, nv_bfloat16 * bf16, void * image, uint3 cols, int64_t padded, int64_t rows, float scale) {
    hc_post_norm_f32_impl<Block, Comb, Multiply, NormLayout>(a,
        ggml_cuda_hc_norm_emit_mmq_pair_store<Layout, Scale>{(block_q8_1_mmq *) image, cols, padded, rows, f16, bf16, scale});
}

template <mmq_q8_1_ds_layout Layout, int Block, bool Comb, bool Multiply, bool Scale>
static auto hc_post_norm_emit_mmq_kernel(int layout) {
    if (layout == 1) { return hc_post_norm_emit_mmq_f32<Layout, Block, Comb, Multiply, Scale, 1>; }
    if (layout == 2) { return hc_post_norm_emit_mmq_f32<Layout, Block, Comb, Multiply, Scale, 2>; }
    return hc_post_norm_emit_mmq_f32<Layout, Block, Comb, Multiply, Scale, 0>;
}

struct ggml_cuda_norm_scale_store {
    static constexpr int width = 1;
    float scale;
    half * f16;
    nv_bfloat16 * bf16;

    __device__ __forceinline__ void operator()(float * dst, const float * base, int col, float value) const {
        value = scale * value;
        ggml_cuda_norm_emit_store{f16, bf16}(dst, base, col, value);
    }
};

template <int Block, bool Comb, int Layout>
static __global__ __launch_bounds__(Block) void hc_post_norm_scale_f32(hc_post_norm_data a, float scale,
        half * f16, nv_bfloat16 * bf16) {
    hc_post_norm_f32_impl<Block, Comb, false, Layout>(a, ggml_cuda_norm_scale_store{scale, f16, bf16});
}

template <int Block, bool Comb>
static auto hc_post_norm_scale_kernel(int layout) {
    if (layout == 1) { return hc_post_norm_scale_f32<Block, Comb, 1>; }
    if (layout == 2) { return hc_post_norm_scale_f32<Block, Comb, 2>; }
    return hc_post_norm_scale_f32<Block, Comb, 0>;
}

template <int block_size, bool has_comb, bool do_multiply>
static auto hc_post_norm_kernel(int layout) {
    if (layout == 1) {
        return hc_post_norm_f32<block_size, has_comb, do_multiply, 1>;
    }
    if (layout == 2) {
        return hc_post_norm_f32<block_size, has_comb, do_multiply, 2>;
    }
    return hc_post_norm_f32<block_size, has_comb, do_multiply, 0>;
}

template <int block_size, bool has_comb, bool do_multiply>
static auto hc_post_norm_emit_kernel(int layout) {
    if (layout == 1) {
        return hc_post_norm_emit_f32<block_size, has_comb, do_multiply, 1>;
    }
    if (layout == 2) {
        return hc_post_norm_emit_f32<block_size, has_comb, do_multiply, 2>;
    }
    return hc_post_norm_emit_f32<block_size, has_comb, do_multiply, 0>;
}

template <int block_size, bool has_comb, bool do_multiply>
static auto hc_post_norm_emit_q8_kernel(int layout) {
    if (layout == 1) {
        return hc_post_norm_emit_q8_f32<block_size, has_comb, do_multiply, 1>;
    }
    if (layout == 2) {
        return hc_post_norm_emit_q8_f32<block_size, has_comb, do_multiply, 2>;
    }
    return hc_post_norm_emit_q8_f32<block_size, has_comb, do_multiply, 0>;
}

bool ggml_cuda_should_fuse_hc_post_norm(const ggml_tensor * post, const ggml_tensor * norm, const ggml_tensor * mul) {
    const ggml_tensor * dst = mul ? mul : norm;
    if (!post->src[0] || !post->src[1] || !post->src[2] || post->op != GGML_OP_DSV4_HC_POST || norm->op != GGML_OP_RMS_NORM ||
            post->type != GGML_TYPE_F32 || norm->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(post) || !ggml_is_contiguous(norm) || !ggml_is_contiguous(dst) ||
            norm->ne[0] > INT_MAX - 1024 || norm->ne[1] > INT_MAX || norm->ne[2] > 65535 || norm->ne[3] > 65535) {
        return false;
    }
    const ggml_tensor * inputs[] = { post->src[0], post->src[1], post->src[2], post->src[3], post, norm, dst, mul ? mul->src[mul->src[0] == norm ? 1 : 0] : nullptr };
    for (const ggml_tensor * tensor : inputs) {
        if (!tensor) {
            continue;
        }
        if (tensor->type != GGML_TYPE_F32) {
            return false;
        }
        size_t span = sizeof(float);
        size_t elements = 1;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            if (tensor->ne[d] <= 0 || tensor->ne[d] > INT_MAX || tensor->nb[d] % sizeof(float) != 0 ||
                    elements > INT64_MAX/sizeof(float)/(size_t) tensor->ne[d] ||
                    (tensor->nb[d] != 0 && (size_t) (tensor->ne[d] - 1) > (SIZE_MAX - span)/tensor->nb[d])) {
                return false;
            }
            elements *= tensor->ne[d];
            span += (tensor->ne[d] - 1)*tensor->nb[d];
        }
        if (span > INT64_MAX) {
            return false;
        }
    }
    const float eps = ggml_get_op_params_f32(norm, 0);
    return ggml_nelements(norm) <= INT_MAX && ggml_nelements(post) == ggml_nelements(norm) && ggml_are_same_shape(norm, dst) && std::isfinite(eps) && eps >= 0.0f;
}

bool ggml_cuda_should_fuse_hc_post_norm_scale(const ggml_tensor * post, const ggml_tensor * norm, const ggml_tensor * scale) {
    return scale && ggml_cuda_should_fuse_hc_post_norm(post, norm, nullptr) && scale->op == GGML_OP_SCALE &&
        scale->type == GGML_TYPE_F32 && ggml_is_contiguous(scale) && ggml_are_same_shape(norm, scale) &&
        (norm->flags & GGML_TENSOR_FLAG_COMPUTE) && (scale->flags & GGML_TENSOR_FLAG_COMPUTE);
}

template <bool emit, bool postop = false>
static void ggml_cuda_op_hc_post_norm_impl(ggml_backend_cuda_context & ctx, ggml_tensor * post, ggml_tensor * norm, ggml_tensor * mul, void * f16, void * bf16,
        ggml_tensor * scale = nullptr) {
    const ggml_tensor * x = post->src[0];
    const ggml_tensor * residual = post->src[1];
    const ggml_tensor * weights = post->src[2];
    const ggml_tensor * comb = post->src[3];
    const ggml_tensor * gamma = mul ? mul->src[mul->src[0] == norm ? 1 : 0] : nullptr;
    hc_post_norm_data a{};
    a.x = (const float *) x->data;
    a.residual = (const float *) residual->data;
    a.post = (const float *) weights->data;
    a.comb = comb ? (const float *) comb->data : nullptr;
    a.mul = gamma ? (const float *) gamma->data : nullptr;
    a.post_dst = (float *) post->data;
    a.dst = (float *) (postop ? scale : (mul ? mul : norm))->data;
    a.n_embd = x->ne[0];
    a.hc = residual->ne[1];
    a.ncols = norm->ne[0];
    a.embd_ne = init_fastdiv_values(a.n_embd);
    a.eps = ggml_get_op_params_f32(norm, 0);
    for (int d = 0; d < 3; ++d) {
        if (d < 2) {
            a.sx[d] = x->nb[d]/sizeof(float);
            a.sp[d] = weights->nb[d]/sizeof(float);
        }
        a.sr[d] = residual->nb[d]/sizeof(float);
        a.sc[d] = comb ? comb->nb[d]/sizeof(float) : 0;
    }
    for (int d = 0; d < 4; ++d) {
        a.sm[d] = gamma ? gamma->nb[d]/sizeof(float) : 0;
        a.mul_ne[d] = init_fastdiv_values(gamma ? gamma->ne[d] : 1);
    }
    const int block_size = a.ncols < 1024 ? 256 : 1024;
    const dim3 grid(norm->ne[1], norm->ne[2], norm->ne[3]);
    const ggml_cuda_kernel_launch_params launch(grid, dim3(block_size), 32*sizeof(float), ctx.stream());
    const int layout = a.ncols == a.n_embd ? 1 : a.ncols == a.n_embd*a.hc ? 2 : 0;
    if constexpr (postop) {
        auto kernel = block_size == 256
            ? (comb ? hc_post_norm_scale_kernel<256, true>(layout) : hc_post_norm_scale_kernel<256, false>(layout))
            : (comb ? hc_post_norm_scale_kernel<1024, true>(layout) : hc_post_norm_scale_kernel<1024, false>(layout));
        ggml_cuda_kernel_launch(kernel, launch, a, ggml_get_op_params_f32(scale, 0), (half *) f16, (nv_bfloat16 *) bf16);
    } else if constexpr (emit) {
        auto kernel = block_size == 256
            ? (comb ? (mul ? hc_post_norm_emit_kernel<256, true, true>(layout) : hc_post_norm_emit_kernel<256, true, false>(layout)) : (mul ? hc_post_norm_emit_kernel<256, false, true>(layout) : hc_post_norm_emit_kernel<256, false, false>(layout)))
            : (comb ? (mul ? hc_post_norm_emit_kernel<1024, true, true>(layout) : hc_post_norm_emit_kernel<1024, true, false>(layout)) : (mul ? hc_post_norm_emit_kernel<1024, false, true>(layout) : hc_post_norm_emit_kernel<1024, false, false>(layout)));
        ggml_cuda_kernel_launch(kernel, launch, a, (half *) f16, (nv_bfloat16 *) bf16);
    } else {
        auto kernel = block_size == 256
            ? (comb ? (mul ? hc_post_norm_kernel<256, true, true>(layout) : hc_post_norm_kernel<256, true, false>(layout)) : (mul ? hc_post_norm_kernel<256, false, true>(layout) : hc_post_norm_kernel<256, false, false>(layout)))
            : (comb ? (mul ? hc_post_norm_kernel<1024, true, true>(layout) : hc_post_norm_kernel<1024, true, false>(layout)) : (mul ? hc_post_norm_kernel<1024, false, true>(layout) : hc_post_norm_kernel<1024, false, false>(layout)));
        ggml_cuda_kernel_launch(kernel, launch, a);
    }
}

void ggml_cuda_op_hc_post_norm(ggml_backend_cuda_context & ctx, ggml_tensor * post, ggml_tensor * norm, ggml_tensor * mul) {
    ggml_cuda_op_hc_post_norm_impl<false>(ctx, post, norm, mul, nullptr, nullptr);
}

void ggml_cuda_op_hc_post_norm_emit(ggml_backend_cuda_context & ctx, ggml_tensor * post, ggml_tensor * norm, ggml_tensor * mul, void * f16, void * bf16) {
    ggml_cuda_op_hc_post_norm_impl<true>(ctx, post, norm, mul, f16, bf16);
}

void ggml_cuda_op_hc_post_norm_scale(ggml_backend_cuda_context & ctx, ggml_tensor * post, ggml_tensor * norm,
        ggml_tensor * scale, void * f16, void * bf16) {
    GGML_ASSERT(ggml_cuda_should_fuse_hc_post_norm_scale(post, norm, scale));
    ggml_cuda_op_hc_post_norm_impl<true, true>(ctx, post, norm, nullptr, f16, bf16, scale);
}

template <int Block, bool Multiply, bool Add, bool Scale>
static void ggml_cuda_rms_norm_q8_launch(ggml_backend_cuda_context & ctx, ggml_tensor * norm,
        ggml_tensor * mul, ggml_tensor * add, ggml_tensor * scale, void * image, int64_t cols, int64_t padded) {
    const ggml_tensor * x = norm->src[0];
    const ggml_tensor * weight = mul ? (mul->src[0] == norm ? mul->src[1] : mul->src[0]) : nullptr;
    const ggml_tensor * bias = add ? (add->src[0] == mul ? add->src[1] : add->src[0]) : nullptr;
    ggml_tensor * dst = scale ? scale : add ? add : mul ? mul : norm;
    const auto pointer = [](const ggml_tensor * t) { return t ? (const float *) t->data : nullptr; };
    const auto stride = [](const ggml_tensor * t, int d) { return t ? int64_t(t->nb[d]/sizeof(float)) : int64_t(0); };
    const auto divisor = [](const ggml_tensor * t, int d) { return t ? init_fastdiv_values(t->ne[d]) : make_uint3(0, 0, 0); };
    const ggml_cuda_kernel_launch_params params = {
        dim3(x->ne[1], x->ne[2], x->ne[3]), dim3(Block, 1, 1), 32*sizeof(float), ctx.stream()};
    ggml_cuda_kernel_launch(rms_norm_q8_f32<Block, Multiply, Add, Scale>, params,
        (const float *) x->data, (float *) dst->data, int(x->ne[0]), stride(x, 1), stride(x, 2), stride(x, 3),
        ggml_get_op_params_f32(norm, 0), image, cols, padded,
        pointer(weight), stride(weight, 1), stride(weight, 2), stride(weight, 3),
        divisor(weight, 0), divisor(weight, 1), divisor(weight, 2), divisor(weight, 3),
        pointer(bias), stride(bias, 1), stride(bias, 2), stride(bias, 3),
        divisor(bias, 0), divisor(bias, 1), divisor(bias, 2), divisor(bias, 3),
        scale ? ggml_get_op_params_f32(scale, 0) : 1.0f);
}

template <bool Multiply, bool Add, bool Scale>
static void ggml_cuda_rms_norm_q8(ggml_backend_cuda_context & ctx, ggml_tensor * norm,
        ggml_tensor * mul, ggml_tensor * add, ggml_tensor * scale, void * image, int64_t cols, int64_t padded) {
    if (norm->src[0]->ne[0] < 1024) {
        ggml_cuda_rms_norm_q8_launch<256, Multiply, Add, Scale>(ctx, norm, mul, add, scale, image, cols, padded);
    } else {
        ggml_cuda_rms_norm_q8_launch<1024, Multiply, Add, Scale>(ctx, norm, mul, add, scale, image, cols, padded);
    }
}

void ggml_cuda_op_rms_norm_q8(ggml_backend_cuda_context & ctx, ggml_tensor * norm,
        ggml_tensor * mul, ggml_tensor * add, ggml_tensor * scale, void * image, int64_t cols, int64_t padded) {
    GGML_ASSERT(ggml_get_op_params_f32(norm, 0) >= 0.0f);
    if (add) { ggml_cuda_rms_norm_q8<true, true, false>(ctx, norm, mul, add, scale, image, cols, padded); }
    else if (mul) { ggml_cuda_rms_norm_q8<true, false, false>(ctx, norm, mul, add, scale, image, cols, padded); }
    else if (scale) { ggml_cuda_rms_norm_q8<false, false, true>(ctx, norm, mul, add, scale, image, cols, padded); }
    else { ggml_cuda_rms_norm_q8<false, false, false>(ctx, norm, mul, add, scale, image, cols, padded); }
}

template <int Block, bool Multiply, bool Add, bool Scale>
static void ggml_cuda_rms_norm_emit_q8_launch(ggml_backend_cuda_context & ctx, ggml_tensor * norm,
        ggml_tensor * mul, ggml_tensor * add, ggml_tensor * scale, void * f16, void * bf16, void * image, int64_t cols, int64_t padded) {
    const ggml_tensor * x = norm->src[0];
    const ggml_tensor * weight = mul ? (mul->src[0] == norm ? mul->src[1] : mul->src[0]) : nullptr;
    const ggml_tensor * bias = add ? (add->src[0] == mul ? add->src[1] : add->src[0]) : nullptr;
    ggml_tensor * dst = scale ? scale : add ? add : mul ? mul : norm;
    const auto pointer = [](const ggml_tensor * t) { return t ? (const float *) t->data : nullptr; };
    const auto stride = [](const ggml_tensor * t, int d) { return t ? int64_t(t->nb[d]/sizeof(float)) : int64_t(0); };
    const auto divisor = [](const ggml_tensor * t, int d) { return t ? init_fastdiv_values(t->ne[d]) : make_uint3(0, 0, 0); };
    const ggml_cuda_kernel_launch_params params = {
        dim3(x->ne[1], x->ne[2], x->ne[3]), dim3(Block, 1, 1), 32*sizeof(float), ctx.stream()};
    ggml_cuda_kernel_launch(rms_norm_emit_q8_f32<Block, Multiply, Add, Scale>, params,
        (const float *) x->data, (float *) dst->data, int(x->ne[0]), stride(x, 1), stride(x, 2), stride(x, 3),
        ggml_get_op_params_f32(norm, 0), (half *) f16, (nv_bfloat16 *) bf16, image, init_fastdiv_values(cols), padded,
        pointer(weight), stride(weight, 1), stride(weight, 2), stride(weight, 3),
        divisor(weight, 0), divisor(weight, 1), divisor(weight, 2), divisor(weight, 3),
        pointer(bias), stride(bias, 1), stride(bias, 2), stride(bias, 3),
        divisor(bias, 0), divisor(bias, 1), divisor(bias, 2), divisor(bias, 3),
        scale ? ggml_get_op_params_f32(scale, 0) : 1.0f);
}

template <bool Multiply, bool Add, bool Scale>
static void ggml_cuda_rms_norm_emit_q8(ggml_backend_cuda_context & ctx, ggml_tensor * norm,
        ggml_tensor * mul, ggml_tensor * add, ggml_tensor * scale, void * f16, void * bf16, void * image, int64_t cols, int64_t padded) {
    if (norm->src[0]->ne[0] < 1024) {
        ggml_cuda_rms_norm_emit_q8_launch<256, Multiply, Add, Scale>(ctx, norm, mul, add, scale, f16, bf16, image, cols, padded);
    } else {
        ggml_cuda_rms_norm_emit_q8_launch<1024, Multiply, Add, Scale>(ctx, norm, mul, add, scale, f16, bf16, image, cols, padded);
    }
}

void ggml_cuda_op_rms_norm_emit_q8(ggml_backend_cuda_context & ctx, ggml_tensor * norm,
        ggml_tensor * mul, ggml_tensor * add, ggml_tensor * scale, void * f16, void * bf16, void * image, int64_t cols, int64_t padded) {
    GGML_ASSERT(ggml_get_op_params_f32(norm, 0) >= 0.0f);
    if (add) { ggml_cuda_rms_norm_emit_q8<true, true, false>(ctx, norm, mul, add, scale, f16, bf16, image, cols, padded); }
    else if (mul) { ggml_cuda_rms_norm_emit_q8<true, false, false>(ctx, norm, mul, add, scale, f16, bf16, image, cols, padded); }
    else if (scale) { ggml_cuda_rms_norm_emit_q8<false, false, true>(ctx, norm, mul, add, scale, f16, bf16, image, cols, padded); }
    else { ggml_cuda_rms_norm_emit_q8<false, false, false>(ctx, norm, mul, add, scale, f16, bf16, image, cols, padded); }
}

void ggml_cuda_op_hc_post_norm_emit_q8(ggml_backend_cuda_context & ctx, ggml_tensor * post, ggml_tensor * norm, ggml_tensor * mul, void * f16, void * bf16, void * image, int64_t cols, int64_t padded) {
    const ggml_tensor * x = post->src[0];
    const ggml_tensor * residual = post->src[1];
    const ggml_tensor * weights = post->src[2];
    const ggml_tensor * comb = post->src[3];
    const ggml_tensor * gamma = mul ? mul->src[mul->src[0] == norm ? 1 : 0] : nullptr;
    hc_post_norm_data a{};
    a.x = (const float *) x->data;
    a.residual = (const float *) residual->data;
    a.post = (const float *) weights->data;
    a.comb = comb ? (const float *) comb->data : nullptr;
    a.mul = gamma ? (const float *) gamma->data : nullptr;
    a.post_dst = (float *) post->data;
    a.dst = (float *) (mul ? mul : norm)->data;
    a.n_embd = x->ne[0];
    a.hc = residual->ne[1];
    a.ncols = norm->ne[0];
    a.embd_ne = init_fastdiv_values(a.n_embd);
    a.eps = ggml_get_op_params_f32(norm, 0);
    for (int d = 0; d < 3; ++d) {
        if (d < 2) {
            a.sx[d] = x->nb[d]/sizeof(float);
            a.sp[d] = weights->nb[d]/sizeof(float);
        }
        a.sr[d] = residual->nb[d]/sizeof(float);
        a.sc[d] = comb ? comb->nb[d]/sizeof(float) : 0;
    }
    for (int d = 0; d < 4; ++d) {
        a.sm[d] = gamma ? gamma->nb[d]/sizeof(float) : 0;
        a.mul_ne[d] = init_fastdiv_values(gamma ? gamma->ne[d] : 1);
    }
    const int block_size = a.ncols < 1024 ? 256 : 1024;
    const dim3 grid(norm->ne[1], norm->ne[2], norm->ne[3]);
    const ggml_cuda_kernel_launch_params launch(grid, dim3(block_size), 32*sizeof(float), ctx.stream());
    const int layout = a.ncols == a.n_embd ? 1 : a.ncols == a.n_embd*a.hc ? 2 : 0;
        auto kernel = block_size == 256
            ? (comb ? (mul ? hc_post_norm_emit_q8_kernel<256, true, true>(layout) : hc_post_norm_emit_q8_kernel<256, true, false>(layout)) : (mul ? hc_post_norm_emit_q8_kernel<256, false, true>(layout) : hc_post_norm_emit_q8_kernel<256, false, false>(layout)))
            : (comb ? (mul ? hc_post_norm_emit_q8_kernel<1024, true, true>(layout) : hc_post_norm_emit_q8_kernel<1024, true, false>(layout)) : (mul ? hc_post_norm_emit_q8_kernel<1024, false, true>(layout) : hc_post_norm_emit_q8_kernel<1024, false, false>(layout)));
        ggml_cuda_kernel_launch(kernel, launch, a, (half *) f16, (nv_bfloat16 *) bf16, image, init_fastdiv_values(cols), padded);
}

template <mmq_q8_1_ds_layout Layout, int Block, bool Multiply, bool Add, bool Scale>
static void ggml_cuda_rms_norm_mmq_launch(ggml_backend_cuda_context & ctx, ggml_tensor * norm,
        ggml_tensor * mul, ggml_tensor * add, ggml_tensor * scale, void * image, int64_t cols, int64_t padded, int64_t rows) {
    const ggml_tensor * x = norm->src[0];
    const ggml_tensor * weight = mul ? (mul->src[0] == norm ? mul->src[1] : mul->src[0]) : nullptr;
    const ggml_tensor * bias = add ? (add->src[0] == mul ? add->src[1] : add->src[0]) : nullptr;
    ggml_tensor * dst = scale ? scale : add ? add : mul ? mul : norm;
    const auto pointer = [](const ggml_tensor * t) { return t ? (const float *) t->data : nullptr; };
    const auto stride = [](const ggml_tensor * t, int d) { return t ? int64_t(t->nb[d]/sizeof(float)) : int64_t(0); };
    const auto divisor = [](const ggml_tensor * t, int d) { return t ? init_fastdiv_values(t->ne[d]) : make_uint3(0, 0, 0); };
    const ggml_cuda_kernel_launch_params params = {
        dim3(x->ne[1], x->ne[2], x->ne[3]), dim3(Block, 1, 1), 32*sizeof(float), ctx.stream()};
    ggml_cuda_kernel_launch(rms_norm_mmq_f32<Layout, Block, Multiply, Add, Scale>, params,
        (const float *) x->data, (float *) dst->data, int(x->ne[0]), stride(x, 1), stride(x, 2), stride(x, 3),
        ggml_get_op_params_f32(norm, 0), image, cols, padded, rows,
        pointer(weight), stride(weight, 1), stride(weight, 2), stride(weight, 3),
        divisor(weight, 0), divisor(weight, 1), divisor(weight, 2), divisor(weight, 3),
        pointer(bias), stride(bias, 1), stride(bias, 2), stride(bias, 3),
        divisor(bias, 0), divisor(bias, 1), divisor(bias, 2), divisor(bias, 3),
        scale ? ggml_get_op_params_f32(scale, 0) : 1.0f);
}

template <mmq_q8_1_ds_layout Layout, bool Multiply, bool Add, bool Scale>
static void ggml_cuda_rms_norm_mmq(ggml_backend_cuda_context & ctx, ggml_tensor * norm,
        ggml_tensor * mul, ggml_tensor * add, ggml_tensor * scale, void * image, int64_t cols, int64_t padded, int64_t rows) {
    if (norm->src[0]->ne[0] < 1024) {
        ggml_cuda_rms_norm_mmq_launch<Layout, 256, Multiply, Add, Scale>(ctx, norm, mul, add, scale, image, cols, padded, rows);
    } else {
        ggml_cuda_rms_norm_mmq_launch<Layout, 1024, Multiply, Add, Scale>(ctx, norm, mul, add, scale, image, cols, padded, rows);
    }
}

template <mmq_q8_1_ds_layout Layout>
static void ggml_cuda_rms_norm_mmq_dispatch(ggml_backend_cuda_context & ctx, ggml_tensor * norm,
        ggml_tensor * mul, ggml_tensor * add, ggml_tensor * scale, void * image, int64_t cols, int64_t padded, int64_t rows) {
    GGML_ASSERT(ggml_get_op_params_f32(norm, 0) >= 0.0f);
    if (add) { ggml_cuda_rms_norm_mmq<Layout, true, true, false>(ctx, norm, mul, add, scale, image, cols, padded, rows); }
    else if (mul) { ggml_cuda_rms_norm_mmq<Layout, true, false, false>(ctx, norm, mul, add, scale, image, cols, padded, rows); }
    else if (scale) { ggml_cuda_rms_norm_mmq<Layout, false, false, true>(ctx, norm, mul, add, scale, image, cols, padded, rows); }
    else { ggml_cuda_rms_norm_mmq<Layout, false, false, false>(ctx, norm, mul, add, scale, image, cols, padded, rows); }
}

template <int Block, bool Multiply, bool Add, bool Scale>
static void ggml_cuda_rms_norm_mxfp4_launch(ggml_backend_cuda_context & ctx, ggml_tensor * norm,
        ggml_tensor * mul, ggml_tensor * add, ggml_tensor * scale, void * image, int64_t cols, int64_t padded, int64_t rows) {
    const ggml_tensor * x = norm->src[0];
    const ggml_tensor * weight = mul ? (mul->src[0] == norm ? mul->src[1] : mul->src[0]) : nullptr;
    const ggml_tensor * bias = add ? (add->src[0] == mul ? add->src[1] : add->src[0]) : nullptr;
    ggml_tensor * dst = scale ? scale : add ? add : mul ? mul : norm;
    const auto pointer = [](const ggml_tensor * t) { return t ? (const float *) t->data : nullptr; };
    const auto stride = [](const ggml_tensor * t, int d) { return t ? int64_t(t->nb[d]/sizeof(float)) : int64_t(0); };
    const auto divisor = [](const ggml_tensor * t, int d) { return t ? init_fastdiv_values(t->ne[d]) : make_uint3(0, 0, 0); };
    const ggml_cuda_kernel_launch_params params = {
        dim3(x->ne[1], x->ne[2], x->ne[3]), dim3(Block, 1, 1), 32*sizeof(float), ctx.stream()};
    ggml_cuda_kernel_launch(rms_norm_mxfp4_f32< Block, Multiply, Add, Scale>, params,
        (const float *) x->data, (float *) dst->data, int(x->ne[0]), stride(x, 1), stride(x, 2), stride(x, 3),
        ggml_get_op_params_f32(norm, 0), image, cols, padded, rows,
        pointer(weight), stride(weight, 1), stride(weight, 2), stride(weight, 3),
        divisor(weight, 0), divisor(weight, 1), divisor(weight, 2), divisor(weight, 3),
        pointer(bias), stride(bias, 1), stride(bias, 2), stride(bias, 3),
        divisor(bias, 0), divisor(bias, 1), divisor(bias, 2), divisor(bias, 3),
        scale ? ggml_get_op_params_f32(scale, 0) : 1.0f);
}

template <bool Multiply, bool Add, bool Scale>
static void ggml_cuda_rms_norm_mxfp4(ggml_backend_cuda_context & ctx, ggml_tensor * norm,
        ggml_tensor * mul, ggml_tensor * add, ggml_tensor * scale, void * image, int64_t cols, int64_t padded, int64_t rows) {
    if (norm->src[0]->ne[0] < 1024) {
        ggml_cuda_rms_norm_mxfp4_launch< 256, Multiply, Add, Scale>(ctx, norm, mul, add, scale, image, cols, padded, rows);
    } else {
        ggml_cuda_rms_norm_mxfp4_launch< 1024, Multiply, Add, Scale>(ctx, norm, mul, add, scale, image, cols, padded, rows);
    }
}

static void ggml_cuda_rms_norm_mxfp4_dispatch(ggml_backend_cuda_context & ctx, ggml_tensor * norm,
        ggml_tensor * mul, ggml_tensor * add, ggml_tensor * scale, void * image, int64_t cols, int64_t padded, int64_t rows) {
    GGML_ASSERT(ggml_get_op_params_f32(norm, 0) >= 0.0f);
    if (add) { ggml_cuda_rms_norm_mxfp4< true, true, false>(ctx, norm, mul, add, scale, image, cols, padded, rows); }
    else if (mul) { ggml_cuda_rms_norm_mxfp4< true, false, false>(ctx, norm, mul, add, scale, image, cols, padded, rows); }
    else if (scale) { ggml_cuda_rms_norm_mxfp4< false, false, true>(ctx, norm, mul, add, scale, image, cols, padded, rows); }
    else { ggml_cuda_rms_norm_mxfp4< false, false, false>(ctx, norm, mul, add, scale, image, cols, padded, rows); }
}

void ggml_cuda_op_rms_norm_mmq(ggml_backend_cuda_context & ctx, ggml_tensor * norm,
        ggml_tensor * mul, ggml_tensor * add, ggml_tensor * scale, void * image, int64_t cols, int64_t padded, int64_t rows, int layout) {
    switch (layout) {
        case MMQ_Q8_1_DS_LAYOUT_D4: ggml_cuda_rms_norm_mmq_dispatch<MMQ_Q8_1_DS_LAYOUT_D4>(ctx, norm, mul, add, scale, image, cols, padded, rows); break;
        case MMQ_Q8_1_DS_LAYOUT_DS4: ggml_cuda_rms_norm_mmq_dispatch<MMQ_Q8_1_DS_LAYOUT_DS4>(ctx, norm, mul, add, scale, image, cols, padded, rows); break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6: ggml_cuda_rms_norm_mmq_dispatch<MMQ_Q8_1_DS_LAYOUT_D2S6>(ctx, norm, mul, add, scale, image, cols, padded, rows); break;
        case 3: ggml_cuda_rms_norm_mxfp4_dispatch(ctx, norm, mul, add, scale, image, cols, padded, rows); break;
        default: GGML_ABORT("unsupported MMQ image layout");
    }
}

template <mmq_q8_1_ds_layout Layout>
static void ggml_cuda_hc_post_norm_emit_mmq_dispatch(ggml_backend_cuda_context & ctx, ggml_tensor * post, ggml_tensor * norm, ggml_tensor * mul, ggml_tensor * scale, void * f16, void * bf16, void * image, int64_t cols, int64_t padded, int64_t rows) {
    const ggml_tensor * x = post->src[0];
    const ggml_tensor * residual = post->src[1];
    const ggml_tensor * weights = post->src[2];
    const ggml_tensor * comb = post->src[3];
    const ggml_tensor * gamma = mul ? mul->src[mul->src[0] == norm ? 1 : 0] : nullptr;
    GGML_ASSERT(!mul || !scale);
    hc_post_norm_data a{};
    a.x = (const float *) x->data;
    a.residual = (const float *) residual->data;
    a.post = (const float *) weights->data;
    a.comb = comb ? (const float *) comb->data : nullptr;
    a.mul = gamma ? (const float *) gamma->data : nullptr;
    a.post_dst = (float *) post->data;
    a.dst = (float *) (scale ? scale : mul ? mul : norm)->data;
    a.n_embd = x->ne[0];
    a.hc = residual->ne[1];
    a.ncols = norm->ne[0];
    a.embd_ne = init_fastdiv_values(a.n_embd);
    a.eps = ggml_get_op_params_f32(norm, 0);
    for (int d = 0; d < 3; ++d) {
        if (d < 2) {
            a.sx[d] = x->nb[d]/sizeof(float);
            a.sp[d] = weights->nb[d]/sizeof(float);
        }
        a.sr[d] = residual->nb[d]/sizeof(float);
        a.sc[d] = comb ? comb->nb[d]/sizeof(float) : 0;
    }
    for (int d = 0; d < 4; ++d) {
        a.sm[d] = gamma ? gamma->nb[d]/sizeof(float) : 0;
        a.mul_ne[d] = init_fastdiv_values(gamma ? gamma->ne[d] : 1);
    }
    const int block_size = a.ncols < 1024 ? 256 : 1024;
    const dim3 grid(norm->ne[1], norm->ne[2], norm->ne[3]);
    const ggml_cuda_kernel_launch_params launch(grid, dim3(block_size), 32*sizeof(float), ctx.stream());
    const int layout = a.ncols == a.n_embd ? 1 : a.ncols == a.n_embd*a.hc ? 2 : 0;
    if (scale) {
        auto kernel = block_size == 256
            ? (comb ? hc_post_norm_emit_mmq_kernel<Layout, 256, true, false, true>(layout) : hc_post_norm_emit_mmq_kernel<Layout, 256, false, false, true>(layout))
            : (comb ? hc_post_norm_emit_mmq_kernel<Layout, 1024, true, false, true>(layout) : hc_post_norm_emit_mmq_kernel<Layout, 1024, false, false, true>(layout));
        ggml_cuda_kernel_launch(kernel, launch, a, (half *) f16, (nv_bfloat16 *) bf16, image, init_fastdiv_values(cols), padded, rows, ggml_get_op_params_f32(scale, 0));
    } else {
        auto kernel = block_size == 256
            ? (comb ? (mul ? hc_post_norm_emit_mmq_kernel<Layout, 256, true, true, false>(layout) : hc_post_norm_emit_mmq_kernel<Layout, 256, true, false, false>(layout)) : (mul ? hc_post_norm_emit_mmq_kernel<Layout, 256, false, true, false>(layout) : hc_post_norm_emit_mmq_kernel<Layout, 256, false, false, false>(layout)))
            : (comb ? (mul ? hc_post_norm_emit_mmq_kernel<Layout, 1024, true, true, false>(layout) : hc_post_norm_emit_mmq_kernel<Layout, 1024, true, false, false>(layout)) : (mul ? hc_post_norm_emit_mmq_kernel<Layout, 1024, false, true, false>(layout) : hc_post_norm_emit_mmq_kernel<Layout, 1024, false, false, false>(layout)));
        ggml_cuda_kernel_launch(kernel, launch, a, (half *) f16, (nv_bfloat16 *) bf16, image, init_fastdiv_values(cols), padded, rows, 1.0f);
    }
}

void ggml_cuda_op_hc_post_norm_emit_mmq(ggml_backend_cuda_context & ctx, ggml_tensor * post, ggml_tensor * norm, ggml_tensor * mul, ggml_tensor * scale, void * f16, void * bf16, void * image, int64_t cols, int64_t padded, int64_t rows, int layout) {
    switch (layout) {
        case MMQ_Q8_1_DS_LAYOUT_D4: ggml_cuda_hc_post_norm_emit_mmq_dispatch<MMQ_Q8_1_DS_LAYOUT_D4>(ctx, post, norm, mul, scale, f16, bf16, image, cols, padded, rows); break;
        case MMQ_Q8_1_DS_LAYOUT_DS4: ggml_cuda_hc_post_norm_emit_mmq_dispatch<MMQ_Q8_1_DS_LAYOUT_DS4>(ctx, post, norm, mul, scale, f16, bf16, image, cols, padded, rows); break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6: ggml_cuda_hc_post_norm_emit_mmq_dispatch<MMQ_Q8_1_DS_LAYOUT_D2S6>(ctx, post, norm, mul, scale, f16, bf16, image, cols, padded, rows); break;
        default: GGML_ABORT("unsupported MMQ image layout");
    }
}
