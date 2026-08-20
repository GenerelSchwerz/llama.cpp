#pragma once

#include "fattn-common.cuh"

static constexpr __host__ __device__ bool ggml_cuda_fattn_mma_q8_type(ggml_type type) {
    return type == GGML_TYPE_Q8_0;
}

// Load Q8_0 rows directly into the half2 shared-memory tile consumed by the
// existing MMA kernel. Dequantization deliberately uses the same helper as the
// vector FlashAttention path so the native and materialized paths perform the
// same half2 arithmetic.
template<int D2, int stride_tile, int nthreads, int nbatch_fa, bool oob_check>
static __device__ __forceinline__ void flash_attn_ext_q8_0_load_tile(
        const char * const __restrict__ KV, half2 * const __restrict__ tile_KV,
        const int k0_h2, const int row0, const int stride_row, const int i_sup) {
#ifdef FP16_AVAILABLE
    constexpr int warp_size        = ggml_cuda_get_physical_warp_size();
    constexpr int nelem_per_thread = 16;
    constexpr int nh2_per_thread   = nelem_per_thread/2;

    static_assert(QK8_0 % nelem_per_thread == 0, "per-thread run straddles Q8_0 blocks");
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
            dequantize_V_q8_0<half, nelem_per_thread>(row, vals, e0);
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
