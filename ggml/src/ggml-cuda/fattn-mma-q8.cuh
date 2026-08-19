#pragma once

#include "common.cuh"

// Native Q8_0 K/V tile loading for the MMA FlashAttention kernel.
//
// The MMA kernel consumes K and V exclusively through a shared-memory tile of
// half2 values laid out as tile[i*stride_tile + k], where i is the KV row
// (token) relative to the start of the current KV batch and k indexes half2
// pairs along the head dimension, starting at half2 index k0 of the row.
// flash_attn_ext_f16_load_tile fills that tile from a pre-cast F16 copy of the
// cache; this loader fills the byte-identical tile straight from the Q8_0
// cache, so nothing downstream of the load (ldmatrix, mma, softmax, combine)
// changes and no F16 copy has to be materialized.
//
// block_q8_0 (ggml-common.h) is { ggml_half d; int8_t qs[QK8_0]; }, 34 bytes,
// with element i of a block dequantizing to d*qs[i]. The F16-cast path
// (dequantize_block_q8_0_f16, convert.cu) computes that product as
// __hmul2(make_half2(qs.x, qs.y), __half2half2(d)); this loader performs the
// same half2 multiply, so the two paths produce bit-identical tiles rather
// than merely close ones.

static constexpr __host__ __device__ bool ggml_cuda_fattn_q8_template_type(ggml_type type) {
    return type == GGML_TYPE_Q8_0;
}

// Fill tile_KV[i*stride_tile + k], i < nbatch_fa, k < D2, from the Q8_0 rows
// starting at row0, reading half2 pair k0_h2 + k of each row.
//
// KV         is the base pointer of the K/V head, as a raw byte pointer:
//            Q8_0 rows are 34 bytes per 32 elements and therefore only
//            2-byte aligned, so no wider pointer type is well defined here.
// stride_row is the distance between consecutive KV rows in bytes (nb11/nb21),
//            not in half2 units as for the F16 loader.
// i_sup      bounds the valid rows when oob_check is set; rows at or past it
//            are zero-filled, matching the F16 loader's out-of-bounds contract.
template<int D2, int stride_tile, int nthreads, int nbatch_fa, bool oob_check>
static __device__ __forceinline__ void flash_attn_ext_q8_0_load_tile(
        const char * const __restrict__ KV, half2 * const __restrict__ tile_KV,
        const int k0_h2, const int row0, const int stride_row, const int i_sup) {
#ifdef FP16_AVAILABLE
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    // Each thread dequantizes one contiguous run of elements along the head
    // dimension. A run of 16 never straddles a Q8_0 block boundary as long as
    // it starts at a multiple of 16, which the static asserts below enforce for
    // the run's offset within the row (k_tid*16) and for the tile's own start
    // (2*k0_h2, checked at the call site via the batch geometry).
    constexpr int nelem_per_thread = 16;
    constexpr int nh2_per_thread   = nelem_per_thread/2;
    static_assert(QK8_0 % nelem_per_thread == 0, "per-thread run straddles Q8_0 blocks");
    static_assert((2*D2) % nelem_per_thread == 0, "head dimension batch not divisible by the per-thread run");

    constexpr int threads_per_row = (2*D2) / nelem_per_thread;
    static_assert(nthreads % threads_per_row == 0, "bad nthreads");
    constexpr int rows_per_iter = nthreads / threads_per_row;
    static_assert(nbatch_fa % rows_per_iter == 0, "bad nbatch_fa");

    // Shared-memory stores below are 16 bytes wide, so the tile row stride and
    // the per-thread column offset both have to keep that alignment.
    static_assert((stride_tile   * sizeof(half2)) % 16 == 0, "tile row stride breaks 16 byte stores");
    static_assert((nh2_per_thread * sizeof(half2)) % 16 == 0, "per-thread column offset breaks 16 byte stores");

    const int tid   = threadIdx.y*warp_size + threadIdx.x;
    const int i_tid = tid / threads_per_row; // KV row handled by this thread
    const int k_tid = tid % threads_per_row; // run along the head dimension

    // Element index within the row is the same for every row this thread walks,
    // so the block index and in-block offset are loop invariants.
    const int e0  = 2*k0_h2 + k_tid*nelem_per_thread;
    const int ib  = e0 / QK8_0;
    const int iqs = e0 % QK8_0;

#pragma unroll
    for (int i0 = 0; i0 < nbatch_fa; i0 += rows_per_iter) {
        const int i = i0 + i_tid;

        half2 * const dst = tile_KV + i*stride_tile + k_tid*nh2_per_thread;

        // Both staging arrays are copied through wider types below, so they are
        // aligned explicitly rather than relying on their element alignment.
        __align__(16) half2 vals[nh2_per_thread];

        if (!oob_check || i < i_sup) {
            const block_q8_0 * const blk =
                ((const block_q8_0 *) (KV + int64_t(row0 + i)*stride_row)) + ib;

            // Q8_0 rows are 2-byte aligned only, hence the explicit alignment.
            __align__(16) int8_t qs[nelem_per_thread];
            ggml_cuda_memcpy_1<nelem_per_thread, 2>(qs, blk->qs + iqs);

            const half2 d = __half2half2(blk->d);

#pragma unroll
            for (int l = 0; l < nh2_per_thread; ++l) {
                vals[l] = d * make_half2(qs[2*l + 0], qs[2*l + 1]);
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
    }
#else
    GGML_UNUSED_VARS(KV, tile_KV, k0_h2, row0, stride_row, i_sup);
    NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
}
