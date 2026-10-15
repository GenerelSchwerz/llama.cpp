#pragma once

#include "fattn-mma-quant.cuh"

static_assert(QK5_0 == 32, "unexpected Q5_0 block size");

#if defined(__CUDA_ARCH__)
static __device__ __forceinline__ half2 fattn_mma_q5_centered_half2(const uint32_t q, const uint32_t selector) {
    // The selected bytes become half(1024 + byte). Reinterpret the word as is, two 16-bit halves cost two extra instructions.
    const uint32_t bits = __byte_perm(q, 0x64646464, selector);
    half2 raw;
    memcpy(&raw, &bits, sizeof(raw));
    const __half2_raw bias = { 0x6410, 0x6410 };
    return __hsub2(raw, bias);
}
#endif

template <>
struct fattn_quant_incremental_rows<GGML_TYPE_Q5_0> {
    static constexpr bool value = true;
};

template <>
struct fattn_quant_type_traits<GGML_TYPE_Q5_0> {
    static constexpr int qk = QK5_0;

    template <int nelem>
    static __device__ __forceinline__ void dequant(
            const char * const __restrict__ row, half2 * const __restrict__ vals, const int e0) {
        static_assert(nelem == QK5_0/2 || nelem == QK5_0, "unsupported Q5_0 load width");

        const block_q5_0 * const block = ((const block_q5_0 *) row) + e0/QK5_0;
        const char * const src = (const char *) block->qs;
        const int offset = (uintptr_t) src & 3;
        const char * const src_aligned = src - offset;

        __align__(16) uint32_t raw[QK5_0/(2*sizeof(uint32_t))];
        ggml_cuda_memcpy_1<QK5_0/2, 4>(raw, src_aligned);

        uint16_t tail;
        ggml_cuda_memcpy_1<sizeof(tail), 2>(&tail, src + QK5_0/2 - sizeof(tail));

        uint32_t qs[QK5_0/(2*sizeof(uint32_t))];
#pragma unroll
        for (int i = 0; i < QK5_0/(2*sizeof(uint32_t)); ++i) {
            const uint32_t next = i + 1 < QK5_0/(2*sizeof(uint32_t)) ? raw[i + 1] : tail;
            qs[i] = __funnelshift_r(raw[i], next, 8*offset);
        }

        uint32_t qh;
        ggml_cuda_memcpy_1<sizeof(qh), 2>(&qh, block->qh);

        const half2 d = __half2half2(block->d);
        const int h0 = e0 % QK5_0;

#pragma unroll
        for (int i = 0; i < nelem/4; ++i) {
            const int q_index = nelem == QK5_0 ? i % (QK5_0/(2*sizeof(uint32_t))) : i;
            const int shift = nelem == QK5_0 ? 4*(i / (QK5_0/(2*sizeof(uint32_t)))) : 4*(h0 >= QK5_0/2);
            uint32_t q = (qs[q_index] >> shift) & 0x0F0F0F0F;
            const uint32_t h = (qh >> (h0 + 4*i)) & 0x0F;
            q |= (h*0x02040810) & 0x10101010;
#if defined(__CUDA_ARCH__)
            vals[2*i + 0] = __hmul2(fattn_mma_q5_centered_half2(q, 0x4140), d);
            vals[2*i + 1] = __hmul2(fattn_mma_q5_centered_half2(q, 0x4342), d);
#else
            vals[2*i + 0] = d * make_half2(int(q & 0x1F) - 16, int((q >> 8) & 0x1F) - 16);
            vals[2*i + 1] = d * make_half2(int((q >> 16) & 0x1F) - 16, int((q >> 24) & 0x1F) - 16);
#endif
        }
    }

    // One staged block at p in shared memory, p % 4 == mis.
    template <int mis>
    static __device__ __forceinline__ void dequant_block(const char * const __restrict__ p, half2 * const __restrict__ vals) {
#if defined(__CUDA_ARCH__)
        uint32_t w[6];
        fattn_quant_lds_words<6, mis>(p, w);
        const half2    d  = fattn_quant_scale_half2<mis>(w[0]);
        const uint32_t qh = fattn_quant_word_after_half<mis>(w, 0);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t qs = fattn_quant_word_after_half<mis>(w, j + 1);
            const uint32_t lo = ( qs       & 0x0F0F0F0F) | ((((qh >> (     4*j)) & 0x0F)*0x02040810) & 0x10101010);
            const uint32_t hi = ((qs >> 4) & 0x0F0F0F0F) | ((((qh >> (16 + 4*j)) & 0x0F)*0x02040810) & 0x10101010);
            vals[    2*j + 0] = __hmul2(fattn_mma_q5_centered_half2(lo, 0x4140), d);
            vals[    2*j + 1] = __hmul2(fattn_mma_q5_centered_half2(lo, 0x4342), d);
            vals[8 + 2*j + 0] = __hmul2(fattn_mma_q5_centered_half2(hi, 0x4140), d);
            vals[8 + 2*j + 1] = __hmul2(fattn_mma_q5_centered_half2(hi, 0x4342), d);
        }
#else
        GGML_UNUSED_VARS(p, vals);
        NO_DEVICE_CODE;
#endif // defined(__CUDA_ARCH__)
    }
};
