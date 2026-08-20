#pragma once

#include "fattn-common.cuh"
#include "convert.cuh"
#include "dequantize.cuh"

// Native quantized K/V tile loading for the MMA FlashAttention kernel.
//
// The reference this path must reproduce is the F16-casting route it replaces,
// so each type's dequant() below has to match that type's cast kernel in
// convert.cu bit for bit, not merely to within a few ulp. Which helper achieves
// that differs per type and cannot be assumed:
//
//   Q8_0 casts via dequantize_block_q8_0_f16, which multiplies in half2. The
//   vector path's dequantize_V_q8_0 performs the same half2 multiply, so
//   delegating to it is exact and keeps one implementation for both paths.
//
//   Q4_0 has no dedicated cast kernel. It goes through the generic
//   dequantize_block_q4_0<half>, which computes in float and rounds once at the
//   end (d = __half2float(x->d), dm = -8*d, y = cast<half>(d*q + dm)). The
//   vector path's dequantize_V_q4_0 instead biases the quant by -8 as an
//   integer and multiplies in half2. Both are correct and they disagree in the
//   last bit, so this type needs its own dequant rather than the shared helper.
//   dequantize_V_q4_0 is also restricted to ne of 2 or 4 and cannot serve the
//   16-element run the tile loader issues.

template <ggml_type type>
struct fattn_quant_type_traits;

template <>
struct fattn_quant_type_traits<GGML_TYPE_Q8_0> {
    static constexpr int qk = QK8_0;

    template <int nelem>
    static __device__ __forceinline__ void dequant(
            const char * const __restrict__ row, half2 * const __restrict__ vals, const int e0) {
        dequantize_V_q8_0<half, nelem>(row, vals, e0);
    }
};

template <>
struct fattn_quant_type_traits<GGML_TYPE_Q4_0> {
    static constexpr int qk = QK4_0;

    // Element j of a block is the low nibble of qs[j] for j < 16 and the high
    // nibble of qs[j-16] beyond, so a 16-element run always covers the same 16
    // bytes and only selects a nibble half. The 16-byte load is preserved.
    template <int nelem>
    static __device__ __forceinline__ void dequant(
            const char * const __restrict__ row, half2 * const __restrict__ vals, const int e0) {
        static_assert(nelem == QK4_0/2, "q4_0 run must be exactly half a block");

        const block_q4_0 * const blk = ((const block_q4_0 *) row) + e0/QK4_0;

        __align__(16) uint8_t qs[QK4_0/2];
        ggml_cuda_memcpy_1<QK4_0/2, 2>(qs, blk->qs);

        const float d  = __half2float(blk->d);
        const float dm = -8.0f*d;
        const bool  hi = (e0 % QK4_0) != 0;

#pragma unroll
        for (int l = 0; l < nelem/2; ++l) {
            const int q0 = hi ? (qs[2*l + 0] >> 4) : (qs[2*l + 0] & 0x0F);
            const int q1 = hi ? (qs[2*l + 1] >> 4) : (qs[2*l + 1] & 0x0F);
            vals[l] = ggml_cuda_cast<half2>(make_float2(d*q0 + dm, d*q1 + dm));
        }
    }
};


// Most types have no dedicated F16 cast kernel and are cast through the generic
// dequantize_block with a dequantize_q* functor. Calling that same functor here
// makes the tile bit-identical to that route by construction rather than by
// reimplementation, which is what the per-type reconstructions cannot promise.
//
// The functor yields element iqs in v.x and element iqs + qk/2 in v.y, so a
// 16-element run is one half of that pair across iqs 0..15.
template <int qk_, void (*dequant_fn)(const void *, const int64_t, const int, float2 &)>
struct fattn_quant_functor_traits {
    static constexpr int qk = qk_;

    template <int nelem>
    static __device__ __forceinline__ void dequant(
            const char * const __restrict__ row, half2 * const __restrict__ vals, const int e0) {
        static_assert(nelem == qk_/2, "run must be exactly half a block");

        const int64_t ib = e0 / qk_;
        const bool    hi = (e0 % qk_) != 0;

#pragma unroll
        for (int l = 0; l < nelem/2; ++l) {
            float2 a, b;
            dequant_fn(row, ib, 2*l + 0, a);
            dequant_fn(row, ib, 2*l + 1, b);
            vals[l] = ggml_cuda_cast<half2>(make_float2(hi ? a.y : a.x, hi ? b.y : b.x));
        }
    }
};

template <> struct fattn_quant_type_traits<GGML_TYPE_Q4_1>  : fattn_quant_functor_traits<QK4_1,  dequantize_q4_1>  {};
template <> struct fattn_quant_type_traits<GGML_TYPE_Q5_1>  : fattn_quant_functor_traits<QK5_1,  dequantize_q5_1>  {};
template <> struct fattn_quant_type_traits<GGML_TYPE_Q6_1>  : fattn_quant_functor_traits<QK6_1,  dequantize_q6_1>  {};
template <> struct fattn_quant_type_traits<GGML_TYPE_Q3_0>  : fattn_quant_functor_traits<QK3_0,  dequantize_q3_0>  {};
template <> struct fattn_quant_type_traits<GGML_TYPE_Q3_1>  : fattn_quant_functor_traits<QK3_1,  dequantize_q3_1>  {};
template <> struct fattn_quant_type_traits<GGML_TYPE_Q2_0S> : fattn_quant_functor_traits<QK2_0S, dequantize_q2_0s> {};
template <> struct fattn_quant_type_traits<GGML_TYPE_Q2_1>  : fattn_quant_functor_traits<QK2_1,  dequantize_q2_1>  {};


// q5_0 and q6_0 would work through fattn_quant_functor_traits, but the functor
// computes both halves of its pair and a 16-element run needs only one, so the
// generic form does twice the arithmetic it uses. That is free for cheap
// unpacks and is not free here: both types extract a separate high-bit plane
// per element, and measured at depth 32,768 the generic q5_0 loader ran 4.2%
// slower than the F16-casting path it replaces, against +4.5% for the
// hand-written q4_0 one. These reproduce the same dequantize_q5_0 and
// dequantize_q6_0 arithmetic for the selected half only.
template <>
struct fattn_quant_type_traits<GGML_TYPE_Q5_0> {
    static constexpr int qk = QK5_0;

    template <int nelem>
    static __device__ __forceinline__ void dequant(
            const char * const __restrict__ row, half2 * const __restrict__ vals, const int e0) {
        static_assert(nelem == QK5_0/2, "q5_0 run must be exactly half a block");

        const block_q5_0 * const blk = ((const block_q5_0 *) row) + e0/QK5_0;

        const float d = blk->d;
        uint32_t qh;
        ggml_cuda_memcpy_1<sizeof(qh), 2>(&qh, blk->qh);

        __align__(16) uint8_t qs[QK5_0/2];
        ggml_cuda_memcpy_1<QK5_0/2, 2>(qs, blk->qs);

        const bool hi = (e0 % QK5_0) != 0;

#pragma unroll
        for (int l = 0; l < nelem/2; ++l) {
            const int i0 = 2*l + 0;
            const int i1 = 2*l + 1;
            const int a = hi ? ((qs[i0] >> 4) | ((qh >> (i0 + 12)) & 0x10))
                             : ((qs[i0] & 0x0F) | (((qh >> i0) << 4) & 0x10));
            const int b = hi ? ((qs[i1] >> 4) | ((qh >> (i1 + 12)) & 0x10))
                             : ((qs[i1] & 0x0F) | (((qh >> i1) << 4) & 0x10));
            vals[l] = ggml_cuda_cast<half2>(make_float2((a - 16.0f)*d, (b - 16.0f)*d));
        }
    }
};

template <>
struct fattn_quant_type_traits<GGML_TYPE_Q6_0> {
    static constexpr int qk = QK6_0;

    template <int nelem>
    static __device__ __forceinline__ void dequant(
            const char * const __restrict__ row, half2 * const __restrict__ vals, const int e0) {
        static_assert(nelem == QK6_0/2, "q6_0 run must be exactly half a block");

        const block_q6_0 * const blk = ((const block_q6_0 *) row) + e0/QK6_0;

        const float d = blk->d;

        __align__(16) uint8_t qs[QK6_0/2];
        ggml_cuda_memcpy_1<QK6_0/2, 2>(qs, blk->qs);

        // qh is read as one 64-bit word and shifted rather than indexed as a byte
        // array: with the loop unrolled every shift is a compile-time constant,
        // which keeps the plane in registers. Indexing the array cost 3.5% of
        // prefill against the F16 path; shifting it does not.
        uint64_t qh;
        ggml_cuda_memcpy_1<sizeof(qh), 2>(&qh, blk->qh);

        const bool hi = (e0 % QK6_0) != 0;

#pragma unroll
        for (int l = 0; l < nelem/2; ++l) {
            const int i0 = 2*l + 0;
            const int i1 = 2*l + 1;
            const int h0 = (qh >> (8*(i0 % (QK6_0/4)) + 4*(i0 / (QK6_0/4)))) & 0x0F;
            const int h1 = (qh >> (8*(i1 % (QK6_0/4)) + 4*(i1 / (QK6_0/4)))) & 0x0F;
            const int a = hi ? ((qs[i0] >> 4) | ((h0 & 0x0C) << 2))
                             : ((qs[i0] & 0x0F) | ((h0 & 0x03) << 4));
            const int b = hi ? ((qs[i1] >> 4) | ((h1 & 0x0C) << 2))
                             : ((qs[i1] & 0x0F) | ((h1 & 0x03) << 4));
            vals[l] = ggml_cuda_cast<half2>(make_float2((a - 32.0f)*d, (b - 32.0f)*d));
        }
    }
};

// Compiled type tiers. The default set keeps build time in the same bracket as
// before; GGML_CUDA_FA_ALL_QUANTS adds the rest, mirroring how the vector
// FlashAttention pair matrix is tiered.
#ifdef GGML_CUDA_FA_ALL_QUANTS
#define FATTN_MMA_QUANT_TYPES_EXTRA(F) \
    F(GGML_TYPE_Q4_1) F(GGML_TYPE_Q5_1) F(GGML_TYPE_Q6_1) \
    F(GGML_TYPE_Q3_0) F(GGML_TYPE_Q3_1) F(GGML_TYPE_Q2_0S) F(GGML_TYPE_Q2_1)
#else
#define FATTN_MMA_QUANT_TYPES_EXTRA(F)
#endif

#define FATTN_MMA_QUANT_TYPES(F) \
    F(GGML_TYPE_Q8_0) F(GGML_TYPE_Q4_0) F(GGML_TYPE_Q5_0) F(GGML_TYPE_Q6_0) \
    FATTN_MMA_QUANT_TYPES_EXTRA(F)

// Cache types with a compiled native tile loader. Both the host-side route
// decision and the device-side kernel selection ask this same question.
static constexpr __host__ __device__ bool ggml_cuda_fattn_mma_quant_type(ggml_type type) {
#define FATTN_MMA_QUANT_TYPE_MATCH(t) || type == (t)
    return false FATTN_MMA_QUANT_TYPES(FATTN_MMA_QUANT_TYPE_MATCH);
#undef FATTN_MMA_QUANT_TYPE_MATCH
}

// Load quantized rows directly into the half2 shared-memory tile consumed by
// the existing MMA kernel.
template<ggml_type type, int D2, int stride_tile, int nthreads, int nbatch_fa, bool oob_check>
static __device__ __forceinline__ void flash_attn_ext_quant_load_tile(
        const char * const __restrict__ KV, half2 * const __restrict__ tile_KV,
        const int k0_h2, const int row0, const int stride_row, const int i_sup) {
#ifdef FP16_AVAILABLE
    using traits = fattn_quant_type_traits<type>;

    constexpr int warp_size        = ggml_cuda_get_physical_warp_size();
    constexpr int nelem_per_thread = 16;
    constexpr int nh2_per_thread   = nelem_per_thread/2;

    static_assert(traits::qk % nelem_per_thread == 0, "per-thread run straddles quant blocks");
    static_assert((2*D2) % nelem_per_thread == 0, "tile width is not divisible by the per-thread run");

    constexpr int threads_per_row = (2*D2) / nelem_per_thread;
    static_assert(nthreads % threads_per_row == 0, "bad nthreads");
    constexpr int rows_per_iter = nthreads / threads_per_row;
    static_assert(nbatch_fa % rows_per_iter == 0, "bad nbatch_fa");
    static_assert((stride_tile * sizeof(half2)) % 16 == 0, "tile row stride breaks 16-byte stores");

    const int tid   = threadIdx.y*warp_size + threadIdx.x;
    const int i_tid = tid / threads_per_row;
    const int k_tid = tid % threads_per_row;
    const int e0    = 2*k0_h2 + k_tid*nelem_per_thread;

#pragma unroll
    for (int i0 = 0; i0 < nbatch_fa; i0 += rows_per_iter) {
        const int i = i0 + i_tid;
        half2 * const dst = tile_KV + i*stride_tile + k_tid*nh2_per_thread;
        __align__(16) half2 vals[nh2_per_thread];

        if (!oob_check || i < i_sup) {
            const char * const row = KV + int64_t(row0 + i)*stride_row;
            traits::template dequant<nelem_per_thread>(row, vals, e0);
        } else {
#pragma unroll
            for (int l = 0; l < nh2_per_thread; ++l) {
                vals[l] = make_half2(0.0f, 0.0f);
            }
        }

#pragma unroll
        for (int l0 = 0; l0 < nh2_per_thread; l0 += 16/sizeof(half2)) {
            ggml_cuda_memcpy_1<16>(dst + l0, vals + l0);
        }
    }
#else
    GGML_UNUSED_VARS(KV, tile_KV, k0_h2, row0, stride_row, i_sup);
    NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
}
