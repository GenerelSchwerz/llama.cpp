#pragma once

#include "fattn-mma-quant.cuh"

template <>
struct fattn_quant_type_traits<GGML_TYPE_Q8_0> {
    static constexpr int qk = QK8_0;

    template <int nelem>
    static __device__ __forceinline__ void dequant(
            const char * const __restrict__ row, half2 * const __restrict__ vals, const int e0) {
        dequantize_V_q8_0<half, nelem>(row, vals, e0);
    }
};
