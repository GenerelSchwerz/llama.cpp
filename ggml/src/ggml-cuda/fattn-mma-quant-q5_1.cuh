#pragma once

#include "fattn-mma-quant-packed.cuh"

static_assert(QK5_1 == 32, "unexpected Q5_1 block size");

template <>
struct fattn_quant_type_traits<GGML_TYPE_Q5_1> {
    static constexpr int qk = QK5_1;

    template <int nelem>
    static __device__ __forceinline__ void dequant(
            const char * const __restrict__ row, half2 * const __restrict__ vals, const int e0) {
        static_assert(nelem == QK5_1/2, "unsupported Q5_1 load width");

        const block_q5_1 * const block = ((const block_q5_1 *) row) + e0/QK5_1;

        __align__(16) uint32_t qs[QK5_1/(2*sizeof(uint32_t))];
        ggml_cuda_memcpy_1<QK5_1/2, 8>(qs, block->qs);

        uint32_t qh;
        ggml_cuda_memcpy_1<sizeof(qh)>(&qh, block->qh);

        const float2 dm = __half22float2(block->dm);
        const int h0 = e0 % QK5_1;

#pragma unroll
        for (int i = 0; i < nelem/4; ++i) {
            const int shift = 4*(h0 >= QK5_1/2);
            uint32_t q = (qs[i] >> shift) & 0x0F0F0F0F;
            const uint32_t h = (qh >> (h0 + 4*i)) & 0x0F;
            q |= (h*0x02040810) & 0x10101010;
            fattn_quant_store4<true, 16>(vals + 2*i, q, 0.0f, dm);
        }
    }
};
