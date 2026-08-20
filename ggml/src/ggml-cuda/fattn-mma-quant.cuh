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
template <> struct fattn_quant_type_traits<GGML_TYPE_Q3_0>  : fattn_quant_functor_traits<QK3_0,  dequantize_q3_0>  {};
template <> struct fattn_quant_type_traits<GGML_TYPE_Q3_1>  : fattn_quant_functor_traits<QK3_1,  dequantize_q3_1>  {};
template <> struct fattn_quant_type_traits<GGML_TYPE_Q2_0S> : fattn_quant_functor_traits<QK2_0S, dequantize_q2_0s> {};
template <> struct fattn_quant_type_traits<GGML_TYPE_Q2_1>  : fattn_quant_functor_traits<QK2_1,  dequantize_q2_1>  {};


// The generic functor computes both 16-element halves of a block even though a
// loader lane consumes only one. That is particularly expensive for Q5/Q6,
// where every element also extracts a separate high-bit plane. Work in four
// packed 32-bit groups instead: unpack four codes with one set of word
// operations, then preserve the reference cast path's scalar float arithmetic
// before the final float2-to-half2 conversion.
template <typename block_t, bool has_min>
struct fattn_quant_q5_traits {
    static_assert(QK5_0 == QK5_1, "Q5 block widths must match");
    static constexpr int qk = QK5_0;

    template <int nelem>
    static __device__ __forceinline__ void dequant(
            const char * const __restrict__ row, half2 * const __restrict__ vals, const int e0) {
        static_assert(nelem == qk/2, "q5 run must be exactly half a block");

        const block_t * const blk = ((const block_t *) row) + e0/qk;

        uint32_t qh;
        ggml_cuda_memcpy_1<sizeof(qh), 2>(&qh, blk->qh);

        __align__(16) uint32_t qs[qk/(2*sizeof(uint32_t))];
        ggml_cuda_memcpy_1<qk/2, 2>(qs, blk->qs);

        const bool hi = (e0 % qk) != 0;

        float d = 0.0f;
        float2 dm = make_float2(0.0f, 0.0f);
        if constexpr (has_min) {
            dm = __half22float2(blk->dm);
        } else {
            d = blk->d;
        }

#pragma unroll
        for (int g = 0; g < nelem/4; ++g) {
            uint32_t q = (qs[g] >> (4*hi)) & 0x0F0F0F0Fu;

            // Scatter four consecutive plane bits into bit 4 of four bytes.
            // Multiplication is exact here: after the mask, cross-products do
            // not overlap any selected destination bit.
            const uint32_t h4 = (qh >> (16*hi + 4*g)) & 0x0Fu;
            q |= (h4 * 0x02040810u) & 0x10101010u;

            if constexpr (has_min) {
                const uint8_t * const q8 = (const uint8_t *) &q;
                vals[2*g + 0] = ggml_cuda_cast<half2>(make_float2(q8[0]*dm.x + dm.y, q8[1]*dm.x + dm.y));
                vals[2*g + 1] = ggml_cuda_cast<half2>(make_float2(q8[2]*dm.x + dm.y, q8[3]*dm.x + dm.y));
            } else {
                q = __vsubss4(q, 0x10101010u);
                const int8_t * const q8 = (const int8_t *) &q;
                vals[2*g + 0] = ggml_cuda_cast<half2>(make_float2(q8[0]*d, q8[1]*d));
                vals[2*g + 1] = ggml_cuda_cast<half2>(make_float2(q8[2]*d, q8[3]*d));
            }
        }
    }
};

template <> struct fattn_quant_type_traits<GGML_TYPE_Q5_0> : fattn_quant_q5_traits<block_q5_0, false> {};
template <> struct fattn_quant_type_traits<GGML_TYPE_Q5_1> : fattn_quant_q5_traits<block_q5_1, true>  {};

template <typename block_t, bool has_min>
struct fattn_quant_q6_traits {
    static_assert(QK6_0 == QK6_1, "Q6 block widths must match");
    static constexpr int qk = QK6_0;

    template <int nelem>
    static __device__ __forceinline__ void dequant(
            const char * const __restrict__ row, half2 * const __restrict__ vals, const int e0) {
        static_assert(nelem == qk/2, "q6 run must be exactly half a block");

        const block_t * const blk = ((const block_t *) row) + e0/qk;

        __align__(16) uint32_t qs[qk/(2*sizeof(uint32_t))];
        ggml_cuda_memcpy_1<qk/2, 2>(qs, blk->qs);

        uint64_t qh;
        ggml_cuda_memcpy_1<sizeof(qh), 2>(&qh, blk->qh);

        const bool hi = (e0 % qk) != 0;

        float d = 0.0f;
        float2 dm = make_float2(0.0f, 0.0f);
        if constexpr (has_min) {
            dm = __half22float2(blk->dm);
        } else {
            d = blk->d;
        }

#pragma unroll
        for (int g = 0; g < nelem/4; ++g) {
            uint32_t q = (qs[g] >> (4*hi)) & 0x0F0F0F0Fu;

            // qh byte j stores one two-bit plane for elements j, j+8,
            // j+16, and j+24. Select the four bytes and their one relevant
            // plane as a word, then place it above the low nibble.
            const uint32_t hbytes = uint32_t(qh >> (32*(g & 1)));
            const int hshift = 4*(g/2) + 2*hi;
            q |= ((hbytes >> hshift) & 0x03030303u) << 4;

            if constexpr (has_min) {
                const uint8_t * const q8 = (const uint8_t *) &q;
                vals[2*g + 0] = ggml_cuda_cast<half2>(make_float2(q8[0]*dm.x + dm.y, q8[1]*dm.x + dm.y));
                vals[2*g + 1] = ggml_cuda_cast<half2>(make_float2(q8[2]*dm.x + dm.y, q8[3]*dm.x + dm.y));
            } else {
                q = __vsubss4(q, 0x20202020u);
                const int8_t * const q8 = (const int8_t *) &q;
                vals[2*g + 0] = ggml_cuda_cast<half2>(make_float2(q8[0]*d, q8[1]*d));
                vals[2*g + 1] = ggml_cuda_cast<half2>(make_float2(q8[2]*d, q8[3]*d));
            }
        }
    }
};

template <> struct fattn_quant_type_traits<GGML_TYPE_Q6_0> : fattn_quant_q6_traits<block_q6_0, false> {};
template <> struct fattn_quant_type_traits<GGML_TYPE_Q6_1> : fattn_quant_q6_traits<block_q6_1, true>  {};

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
