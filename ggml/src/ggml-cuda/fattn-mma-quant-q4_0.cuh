#pragma once

#include "fattn-mma-quant.cuh"

static_assert(QK4_0 == 32, "unexpected Q4_0 block size");

#if defined(__CUDA_ARCH__)
static __device__ __forceinline__ half2 fattn_mma_q4_centered_half2(const uint32_t q, const uint32_t selector) {
    // The selected bytes become half(1024 + byte). Reinterpret the word as is, two 16-bit halves cost two extra instructions.
    const uint32_t bits = __byte_perm(q, 0x64646464, selector);
    half2 raw;
    memcpy(&raw, &bits, sizeof(raw));
    const __half2_raw bias = { 0x6408, 0x6408 };
    return __hsub2(raw, bias);
}
#endif

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

    // One staged block at p in shared memory, p % 4 == mis.
    template <int mis>
    static __device__ __forceinline__ void dequant_block(const char * const __restrict__ p, half2 * const __restrict__ vals) {
#if defined(__CUDA_ARCH__)
        uint32_t w[5];
        fattn_quant_lds_words<5, mis>(p, w);
        const half2 d = fattn_quant_scale_half2<mis>(w[0]);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t qs = fattn_quant_word_after_half<mis>(w, j);
            const uint32_t lo =  qs       & 0x0F0F0F0F;
            const uint32_t hi = (qs >> 4) & 0x0F0F0F0F;
            vals[    2*j + 0] = __hfma2(fattn_mma_q4_centered_half2(lo, 0x4140), d, make_half2(0.0f, 0.0f));
            vals[    2*j + 1] = __hfma2(fattn_mma_q4_centered_half2(lo, 0x4342), d, make_half2(0.0f, 0.0f));
            vals[8 + 2*j + 0] = __hfma2(fattn_mma_q4_centered_half2(hi, 0x4140), d, make_half2(0.0f, 0.0f));
            vals[8 + 2*j + 1] = __hfma2(fattn_mma_q4_centered_half2(hi, 0x4342), d, make_half2(0.0f, 0.0f));
        }
#else
        GGML_UNUSED_VARS(p, vals);
        NO_DEVICE_CODE;
#endif // defined(__CUDA_ARCH__)
    }
};
