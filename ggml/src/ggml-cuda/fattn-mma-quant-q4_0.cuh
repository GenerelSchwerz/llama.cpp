#pragma once

#include "fattn-mma-quant.cuh"

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
