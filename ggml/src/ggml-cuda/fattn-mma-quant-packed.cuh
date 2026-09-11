#pragma once

#include "fattn-mma-quant.cuh"

// The zero-point types subtract their bias in float rather than in the packed
// word. This matches the F16 cast path's (code - bias)*d expression exactly.
template <bool has_min, int bias>
static __device__ __forceinline__ void fattn_quant_store4(
        half2 * const __restrict__ vals, const uint32_t q, const float d, const float2 dm) {
    const uint8_t * const q8 = (const uint8_t *) &q;
    if constexpr (has_min) {
        vals[0] = ggml_cuda_cast<half2>(make_float2(q8[0]*dm.x + dm.y, q8[1]*dm.x + dm.y));
        vals[1] = ggml_cuda_cast<half2>(make_float2(q8[2]*dm.x + dm.y, q8[3]*dm.x + dm.y));
    } else {
        constexpr float b = bias;
        vals[0] = ggml_cuda_cast<half2>(make_float2((q8[0] - b)*d, (q8[1] - b)*d));
        vals[1] = ggml_cuda_cast<half2>(make_float2((q8[2] - b)*d, (q8[3] - b)*d));
    }
}

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
                const uint32_t h4 = (qh >> (16*hi + 4*g)) & 0x0Fu;
                q |= (h4 * 0x02040810u) & 0x10101010u;
            }

            fattn_quant_store4<has_min, has_plane ? 16 : 8>(vals + 2*g, q, d, dm);
        }
    }
};
