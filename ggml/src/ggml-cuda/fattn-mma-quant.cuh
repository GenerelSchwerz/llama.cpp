#pragma once

#include "fattn-common.cuh"
#include "convert.cuh"
#include "dequantize.cuh"
#include "fattn-mma-quant-types.h"

// Native quantized K/V tile loading for the MMA FlashAttention kernel.
//
// The reference this path must reproduce is the F16-casting route it replaces,
// so each type's dequant() in the type-specific headers has to match that
// type's cast kernel in convert.cu bit for bit, not merely to within a few ulp.
// Which helper achieves that differs per type and cannot be assumed:
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

template <ggml_type type, int nthreads, int ncols1, int ncols2>
struct fattn_quant_load_width {
    static constexpr int value = 16;
};

template <ggml_type type>
struct fattn_quant_incremental_rows {
    static constexpr bool value = false;
};

// Cache types with a compiled native tile loader. Both the host-side route
// decision and the device-side kernel selection ask this same question. Which
// type sits in which tier is declared once, in fattn-mma-quant-types.h.
static constexpr __host__ __device__ bool ggml_cuda_fattn_mma_quant_type(ggml_type type) {
#define FATTN_MMA_QUANT_TYPE_MATCH(t) || type == (t)
    return false FATTN_MMA_QUANT_TYPES(FATTN_MMA_QUANT_TYPE_MATCH);
#undef FATTN_MMA_QUANT_TYPE_MATCH
}

// Sentinel V type meaning "the V cache type is not a template argument; it
// arrives as the runtime kernel parameter type_V_rt".
//
// This is what keeps the pair matrix linear instead of quadratic. With V
// compile-time, n native types cost n^2 instantiations of the entire attention
// body. With V runtime, they cost n: one body per K type, carrying a switch over
// the n V loaders at the tile-load site. The switch runs once per K/V tile, not
// per element, so it is not on the per-element path the packed loaders optimize.
static constexpr ggml_type GGML_CUDA_FATTN_QUANT_V_RUNTIME = (ggml_type) (GGML_TYPE_COUNT + 1);

// True when the V loader must be chosen at runtime rather than instantiated.
static constexpr __host__ __device__ bool ggml_cuda_fattn_mma_quant_v_runtime(ggml_type type_V) {
    return type_V == GGML_CUDA_FATTN_QUANT_V_RUNTIME;
}

// Ordered K/V pair policy.
//
// K and V are independent template parameters of the kernel and reach the tile
// loader through separate calls, so any ordered pair of native types is
// expressible, and the route compiles all of them that the F16-casting path
// would accept. Mixed pairs are only reachable in a GGML_CUDA_FA_ALL_QUANTS
// build, because ggml_cuda_get_best_fattn_kernel() declines K->type != V->type
// otherwise; that is also the build in which the runtime-V kernels exist.
//
// Instantiating per pair would cost n^2 explicit cases. Sharing one runtime-V
// kernel per K type across the mixed pairs makes coverage linear: five types
// cost ten kernels per tile shape, not twenty-five.
static constexpr __host__ __device__ bool ggml_cuda_fattn_mma_quant_pair(ggml_type type_K, ggml_type type_V) {
    if (!ggml_cuda_fattn_mma_quant_type(type_K) || !ggml_cuda_fattn_mma_quant_type(type_V)) {
        return false;
    }
#ifndef GGML_CUDA_FA_ALL_QUANTS
    if (type_K != type_V) {
        return false;
    }
#endif // GGML_CUDA_FA_ALL_QUANTS
    return true;
}

// Load quantized rows directly into the half2 shared-memory tile consumed by
// the existing MMA kernel.
template<ggml_type type, int D2, int stride_tile, int nthreads, int nbatch_fa, int ncols1, int ncols2, bool oob_check>
static __device__ __forceinline__ void flash_attn_ext_quant_load_tile(
        const char * const __restrict__ KV, half2 * const __restrict__ tile_KV,
        const int k0_h2, const int row0, const int stride_row, const int i_sup) {
#ifdef FP16_AVAILABLE
    using traits = fattn_quant_type_traits<type>;

    constexpr int warp_size        = ggml_cuda_get_physical_warp_size();
    constexpr int nelem_per_thread = fattn_quant_load_width<type, nthreads, ncols1, ncols2>::value;
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
    const int64_t stride_iter = int64_t(rows_per_iter)*stride_row;
    const char * row = KV + int64_t(row0 + i_tid)*stride_row;

#pragma unroll
    for (int i0 = 0; i0 < nbatch_fa; i0 += rows_per_iter) {
        const int i = i0 + i_tid;
        half2 * const dst = tile_KV + i*stride_tile + k_tid*nh2_per_thread;
        __align__(16) half2 vals[nh2_per_thread];

        if (!oob_check || i < i_sup) {
            if constexpr (fattn_quant_incremental_rows<type>::value) {
                traits::template dequant<nelem_per_thread>(row, vals, e0);
            } else {
                const char * const row_i = KV + int64_t(row0 + i)*stride_row;
                traits::template dequant<nelem_per_thread>(row_i, vals, e0);
            }
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
        if constexpr (fattn_quant_incremental_rows<type>::value) {
            row += stride_iter;
        }
    }
#else
    GGML_UNUSED_VARS(KV, tile_KV, k0_h2, row0, stride_row, i_sup);
    NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
}
