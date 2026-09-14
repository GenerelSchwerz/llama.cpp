#pragma once

#include "fattn-mma-quant.cuh"

static_assert(QK8_0 == 32, "unexpected Q8_0 block size");

#if defined(__CUDA_ARCH__)
static __device__ __forceinline__ half2 fattn_mma_q8_biased_to_half2(const uint32_t q, const uint32_t selector) {
    const uint32_t bits = __byte_perm(q, 0x64646464, selector);
    const __half2_raw raw  = { (uint16_t) bits, (uint16_t) (bits >> 16) };
    const __half2_raw bias = { 0x6480, 0x6480 };
    return __hsub2(raw, bias);
}
#endif

template <>
struct fattn_quant_incremental_rows<GGML_TYPE_Q8_0> {
    static constexpr bool value = true;
};

template <>
struct fattn_quant_type_traits<GGML_TYPE_Q8_0> {
    static constexpr int qk = QK8_0;

    template <int nelem>
    static __device__ __forceinline__ void dequant(
            const char * const __restrict__ row, half2 * const __restrict__ vals, const int e0) {
        static_assert(nelem == 16, "unsupported Q8_0 load width");

        const block_q8_0 * const block = ((const block_q8_0 *) row) + e0/QK8_0;
        const char * const src = (const char *) block->qs + e0 % QK8_0;
        const int offset = (uintptr_t) src & 3;
        const char * const src_aligned = src - offset;

        __align__(16) uint32_t raw[4];
        ggml_cuda_memcpy_1<16, 4>(raw, src_aligned);

        uint16_t tail;
        ggml_cuda_memcpy_1<2>(&tail, src + 14);

        uint32_t qs[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t next = i < 3 ? raw[i + 1] : tail;
            qs[i] = __funnelshift_r(raw[i], next, 8*offset);
        }

        const half2 d = __half2half2(block->d);
#if defined(__CUDA_ARCH__)
#pragma unroll
        for (int i = 0; i < nelem/4; ++i) {
            const uint32_t q = qs[i] ^ 0x80808080;
            vals[2*i + 0] = d * fattn_mma_q8_biased_to_half2(q, 0x4140);
            vals[2*i + 1] = d * fattn_mma_q8_biased_to_half2(q, 0x4342);
        }
#else
#pragma unroll
        for (int i = 0; i < nelem/2; ++i) {
            const uint32_t q = qs[i/2] >> (16*(i % 2));
            vals[i] = d * make_half2((int8_t) q, (int8_t) (q >> 8));
        }
#endif
    }
};
