#pragma once

#include "fattn-mma-quant.cuh"

static_assert(QK4_0 == 32, "unexpected Q4_0 block size");

#if defined(__CUDA_ARCH__)
static __device__ __forceinline__ half2 fattn_mma_q4_centered_half2(const uint32_t q, const uint32_t selector) {
    const uint32_t bits = __byte_perm(q, 0x64646464, selector);
    const __half2_raw raw  = { (uint16_t) bits, (uint16_t) (bits >> 16) };
    const __half2_raw bias = { 0x6408, 0x6408 };
    return __hsub2(raw, bias);
}
#endif

template <int nthreads, int ncols1, int ncols2>
struct fattn_quant_load_width<GGML_TYPE_Q4_0, nthreads, ncols1, ncols2> {
    static constexpr int value = nthreads == 256 ? 16 : 8;
};

template <>
struct fattn_quant_incremental_rows<GGML_TYPE_Q4_0> {
    static constexpr bool value = true;
};

template <>
struct fattn_quant_type_traits<GGML_TYPE_Q4_0> {
    static constexpr int qk = QK4_0;

    template <int nelem>
    static __device__ __forceinline__ void dequant(
            const char * const __restrict__ row, half2 * const __restrict__ vals, const int e0) {
        static_assert(nelem == QK4_0/4 || nelem == QK4_0/2, "unsupported Q4_0 load width");

        const block_q4_0 * const block = ((const block_q4_0 *) row) + e0/QK4_0;
        const int byte0 = nelem < QK4_0/2 ? e0 % (QK4_0/2) : 0;
        __align__(16) uint32_t qs[QK4_0/(2*sizeof(uint32_t))] = {};
        ggml_cuda_memcpy_1<nelem, 2>(qs, block->qs + byte0);

        const half2 d = __half2half2(block->d);
        const int shift = 4*((e0 % QK4_0) >= QK4_0/2);

#if defined(__CUDA_ARCH__)
#pragma unroll
        for (int i = 0; i < nelem/4; ++i) {
            const uint32_t q = (qs[i] >> shift) & 0x0F0F0F0F;
            vals[2*i + 0] = __hfma2(fattn_mma_q4_centered_half2(q, 0x4140), d, make_half2(0.0f, 0.0f));
            vals[2*i + 1] = __hfma2(fattn_mma_q4_centered_half2(q, 0x4342), d, make_half2(0.0f, 0.0f));
        }
#else
#pragma unroll
        for (int i = 0; i < nelem/2; ++i) {
            const uint32_t q = qs[i/2] >> (8*(i % 2) + shift);
            vals[i] = __hfma2(make_half2(int(q & 0x0F) - 8, int((q >> 8) & 0x0F) - 8), d, make_half2(0.0f, 0.0f));
        }
#endif
    }
};
