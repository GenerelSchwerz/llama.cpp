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


// The remaining types have no dedicated F16 cast kernel either; they are cast
// through the generic dequantize_block with a dequantize_q* functor from
// dequantize.cuh. That functor is the bit-identity reference the loaders below
// reproduce, but calling it per element is wasteful here: it recomputes both
// 16-element halves of a block even though a loader lane consumes only one, and
// it re-reads the block's bytes once per element. Each loader instead unpacks
// four codes at a time with 32-bit word operations, then performs the functor's
// own scalar float arithmetic and final float2-to-half2 conversion, so the tile
// is unchanged.

// Shared tail of every packed loader: four codes already unpacked into the
// bytes of one word become two half2 values. Types with a min use the stored
// pair directly; the others bias by half their code range first, which cannot
// saturate because the biased codes stay well inside int8.
template <bool has_min, uint32_t bias4>
static __device__ __forceinline__ void fattn_quant_store4(
        half2 * const __restrict__ vals, uint32_t q, const float d, const float2 dm) {
    if constexpr (has_min) {
        const uint8_t * const q8 = (const uint8_t *) &q;
        vals[0] = ggml_cuda_cast<half2>(make_float2(q8[0]*dm.x + dm.y, q8[1]*dm.x + dm.y));
        vals[1] = ggml_cuda_cast<half2>(make_float2(q8[2]*dm.x + dm.y, q8[3]*dm.x + dm.y));
    } else {
        q = __vsubss4(q, bias4);
        const int8_t * const q8 = (const int8_t *) &q;
        vals[0] = ggml_cuda_cast<half2>(make_float2(q8[0]*d, q8[1]*d));
        vals[1] = ggml_cuda_cast<half2>(make_float2(q8[2]*d, q8[3]*d));
    }
}

// Nibble-coded types: qs byte j holds element j in its low nibble and element
// j + qk/2 in its high one, so a 16-element run selects one nibble half of every
// qs byte. Q5 adds a one-bit-per-element plane in qh above that nibble; Q4_1 has
// no plane. Q4_0 is nibble-coded too but is not routed here because its cast
// path rounds a different expression.
template <typename block_t, int qk_, bool has_min, bool has_plane>
struct fattn_quant_nibble_traits {
    static constexpr int qk = qk_;

    template <int nelem>
    static __device__ __forceinline__ void dequant(
            const char * const __restrict__ row, half2 * const __restrict__ vals, const int e0) {
        static_assert(nelem == qk/2, "nibble run must be exactly half a block");

        const block_t * const blk = ((const block_t *) row) + e0/qk;

        __align__(16) uint32_t qs[qk/(2*sizeof(uint32_t))];
        ggml_cuda_memcpy_1<qk/2, 2>(qs, blk->qs);

        uint32_t qh = 0;
        if constexpr (has_plane) {
            ggml_cuda_memcpy_1<sizeof(qh), 2>(&qh, blk->qh);
        }

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

            if constexpr (has_plane) {
                // Scatter four consecutive plane bits into bit 4 of four bytes.
                // Multiplication is exact here: the shifted copies of the
                // multiplier occupy disjoint bits, so no carry can reach a
                // selected destination bit.
                const uint32_t h4 = (qh >> (16*hi + 4*g)) & 0x0Fu;
                q |= (h4 * 0x02040810u) & 0x10101010u;
            }

            fattn_quant_store4<has_min, has_plane ? 0x10101010u : 0x08080808u>(vals + 2*g, q, d, dm);
        }
    }
};

template <> struct fattn_quant_type_traits<GGML_TYPE_Q4_1> : fattn_quant_nibble_traits<block_q4_1, QK4_1, true,  false> {};
template <> struct fattn_quant_type_traits<GGML_TYPE_Q5_0> : fattn_quant_nibble_traits<block_q5_0, QK5_0, false, true>  {};
template <> struct fattn_quant_type_traits<GGML_TYPE_Q5_1> : fattn_quant_nibble_traits<block_q5_1, QK5_1, true,  true>  {};

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

            fattn_quant_store4<has_min, 0x20202020u>(vals + 2*g, q, d, dm);
        }
    }
};

template <> struct fattn_quant_type_traits<GGML_TYPE_Q6_0> : fattn_quant_q6_traits<block_q6_0, false> {};
template <> struct fattn_quant_type_traits<GGML_TYPE_Q6_1> : fattn_quant_q6_traits<block_q6_1, true>  {};

// Two-bit-coded types: qs byte j holds the two-bit fields of elements j, j+8,
// j+16 and j+24, which is how Q6 stores its high plane. Q3 adds a
// one-bit-per-element plane in qh above those two bits, at the Q5 plane's shape
// but a different destination bit; Q2 has no plane.
template <typename block_t, int qk_, bool has_min, bool has_plane>
struct fattn_quant_2bit_traits {
    static constexpr int qk = qk_;

    template <int nelem>
    static __device__ __forceinline__ void dequant(
            const char * const __restrict__ row, half2 * const __restrict__ vals, const int e0) {
        static_assert(nelem == qk/2, "two-bit run must be exactly half a block");

        const block_t * const blk = ((const block_t *) row) + e0/qk;

        uint32_t qh = 0;
        if constexpr (has_plane) {
            ggml_cuda_memcpy_1<sizeof(qh), 2>(&qh, blk->qh);
        }

        __align__(8) uint32_t qs[qk/(4*sizeof(uint32_t))];
        ggml_cuda_memcpy_1<qk/4, 2>(qs, blk->qs);

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
            // Four consecutive elements live in one four-byte half of qs, one
            // two-bit field each, at a shift set by which quarter of the block
            // this group is.
            uint32_t q = (qs[g & 1] >> (4*hi + 2*(g/2))) & 0x03030303u;

            if constexpr (has_plane) {
                // Scatter four consecutive plane bits into bit 2 of four bytes.
                // Multiplication is exact here: the shifted copies of the
                // multiplier occupy disjoint bits, so no carry can reach a
                // selected destination bit.
                const uint32_t h4 = (qh >> (16*hi + 4*g)) & 0x0Fu;
                q |= (h4 * 0x00810204u) & 0x04040404u;
            }

            fattn_quant_store4<has_min, has_plane ? 0x04040404u : 0x02020202u>(vals + 2*g, q, d, dm);
        }
    }
};

template <> struct fattn_quant_type_traits<GGML_TYPE_Q2_0S> : fattn_quant_2bit_traits<block_q2_0s, QK2_0S, false, false> {};
template <> struct fattn_quant_type_traits<GGML_TYPE_Q2_1>  : fattn_quant_2bit_traits<block_q2_1,  QK2_1,  true,  false> {};
template <> struct fattn_quant_type_traits<GGML_TYPE_Q3_0>  : fattn_quant_2bit_traits<block_q3_0,  QK3_0,  false, true>  {};
template <> struct fattn_quant_type_traits<GGML_TYPE_Q3_1>  : fattn_quant_2bit_traits<block_q3_1,  QK3_1,  true,  true>  {};

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
