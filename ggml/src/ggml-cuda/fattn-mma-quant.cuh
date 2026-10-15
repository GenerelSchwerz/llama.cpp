#pragma once

#include "fattn-common.cuh"
#include "cp-async.cuh"
#include "mma.cuh"
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

// Only symmetric K/V pairs take the native route. A mixed pair would need a
// kernel that switches on V's type at run time, so the route declines it and
// nothing compiles one.
static constexpr __host__ __device__ bool ggml_cuda_fattn_mma_quant_pair(ggml_type type_K, ggml_type type_V) {
    return type_K == type_V && ggml_cuda_fattn_mma_quant_type(type_K);
}

// GeForce cards run MMAs with float accumulators at half the rate of half
// accumulators, so there the route sums KQ in half over short runs of D and in
// float across them. sm_80 runs both at the same rate. Defining
// GGML_FATTN_QUANT_KQ_FLOAT keeps float accumulation everywhere, for comparison.
#if defined(__CUDA_ARCH__) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && !defined(GGML_FATTN_QUANT_KQ_FLOAT) && \
    (__CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 890 || __CUDA_ARCH__ == 1200)
#define GGML_FATTN_QUANT_KQ_HALF
#endif

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
            const int col = k_tid*nh2_per_thread + l0;
            // Same layout as the F16 loader.
            half2 * const dst = ggml_cuda_mma::swizzle<stride_tile>(tile_KV, i*stride_tile + col, i);
            ggml_cuda_memcpy_1<16>(dst, vals + l0);
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

// Staged loading (cp.async available): raw quantized rows are copied to shared memory while the previous tile computes,
// then converted to the half2 tile. A row of 256 elements is a whole number of 16-byte chunks for every type here,
// so rows are copied as is. The host takes the route only for 16-byte aligned K/V and strides.

static constexpr __host__ __device__ int ggml_cuda_fattn_quant_block_bytes(const ggml_type type) {
    return type == GGML_TYPE_Q4_0 ? sizeof(block_q4_0) :
           type == GGML_TYPE_Q4_1 ? sizeof(block_q4_1) :
           type == GGML_TYPE_Q5_0 ? sizeof(block_q5_0) :
           type == GGML_TYPE_Q5_1 ? sizeof(block_q5_1) :
           type == GGML_TYPE_Q8_0 ? sizeof(block_q8_0) : 0;
}

// Bytes between two staged rows. Blocks of 2 mod 4 bytes are converted in even and odd passes, which is free of bank
// conflicts with packed rows. 4-byte aligned blocks need 16 bytes of padding for the same.
static constexpr __host__ __device__ int ggml_cuda_fattn_quant_stage_pitch(const ggml_type type, const int D) {
    return D/32 * ggml_cuda_fattn_quant_block_bytes(type) + (ggml_cuda_fattn_quant_block_bytes(type) % 4 == 0 ? 16 : 0);
}

static constexpr __host__ __device__ int ggml_cuda_fattn_quant_stage_bytes(const ggml_type type, const int D, const int nrows) {
    return nrows * ggml_cuda_fattn_quant_stage_pitch(type, D);
}

// Read nwords 4-byte words of shared memory starting mis bytes before p.
template <int nwords, int mis>
static __device__ __forceinline__ void fattn_quant_lds_words(const char * const __restrict__ p, uint32_t * const __restrict__ w) {
    static_assert(mis == 0 || mis == 2, "bad misalignment");
    const uint32_t * const src = (const uint32_t *) (p - mis);
#pragma unroll
    for (int j = 0; j < nwords; ++j) {
        w[j] = src[j];
    }
}

// The 4 bytes that start 2 bytes into word j of a block read with fattn_quant_lds_words.
template <int mis>
static __device__ __forceinline__ uint32_t fattn_quant_word_after_half(const uint32_t * const __restrict__ w, const int j) {
    return mis == 0 ? __funnelshift_r(w[j], w[j + 1], 16) : w[j + 1];
}

// The two bytes chosen by selector as half2, exactly.
static __device__ __forceinline__ half2 fattn_quant_byte_half2(const uint32_t q, const uint32_t selector) {
    const uint32_t bits = __byte_perm(q, 0x64646464, selector);
    half2 x;
    memcpy(&x, &bits, sizeof(x));
    const __half2_raw bias = { 0x6400, 0x6400 };
    return __hsub2(x, bias);
}

// The half scale at the start of a block, broadcast to both halves.
template <int mis>
static __device__ __forceinline__ half2 fattn_quant_scale_half2(const uint32_t w0) {
    const uint32_t d2 = __byte_perm(w0, 0, mis == 0 ? 0x1010 : 0x3232);
    half2 d;
    memcpy(&d, &d2, sizeof(d));
    return d;
}

static __device__ __forceinline__ void fattn_quant_cp_async_commit() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.commit_group;");
#endif // CP_ASYNC_AVAILABLE
}

// Wait until at most n of the committed cp.async groups are still in flight.
template <int n>
static __device__ __forceinline__ void fattn_quant_cp_async_wait_group() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.wait_group %0;" : : "n"(n));
#endif // CP_ASYNC_AVAILABLE
}

// Start copying nbatch_fa raw rows of D elements, row0 onward, into raw.
// Consecutive threads take consecutive 16-byte chunks so each row is read as whole sectors.
template<ggml_type type, int D, int nthreads, int nbatch_fa>
static __device__ __forceinline__ void flash_attn_ext_quant_stage_tile(
        const char * const __restrict__ KV, char * const __restrict__ raw, const int row0, const int stride_row) {
#ifdef CP_ASYNC_AVAILABLE
    constexpr int warp_size      = ggml_cuda_get_physical_warp_size();
    constexpr int row_bytes      = D/32 * ggml_cuda_fattn_quant_block_bytes(type);
    constexpr int pitch          = ggml_cuda_fattn_quant_stage_pitch(type, D);
    constexpr int chunks_per_row = row_bytes/16;
    constexpr int nchunks        = nbatch_fa*chunks_per_row;
    static_assert(row_bytes % 16 == 0, "a quantized row is not a whole number of 16-byte chunks");
    static_assert(pitch % 16 == 0, "bad stage pitch");

    const unsigned int raw_s = ggml_cuda_cvta_generic_to_shared(raw);
    const char * const src0 = KV + int64_t(row0)*stride_row;
    const int tid = threadIdx.y*warp_size + threadIdx.x;

#pragma unroll
    for (int c0 = 0; c0 + nthreads <= nchunks; c0 += nthreads) {
        const int c = c0 + tid;
        const int i = c / chunks_per_row;
        const int k = c - i*chunks_per_row;
        cp_async_cg_16<0>(raw_s + i*pitch + 16*k, src0 + (i*stride_row + 16*k));
    }

    // Copy the remaining chunks in 4-byte pieces so every thread takes part.
    // A 16-byte copy with most lanes predicated off costs extra shared memory wavefronts and L2 sectors.
    constexpr int nrem  = nchunks % nthreads;
    constexpr int c_rem = nchunks - nrem;
    static_assert(nrem % warp_size == 0, "bad remainder");
    if constexpr ((16*nrem) % (4*nthreads) != 0) {
        if (int(threadIdx.y) < nrem/warp_size) {
            const int c = c_rem + tid;
            const int i = c / chunks_per_row;
            const int k = c - i*chunks_per_row;
            cp_async_cg_16<0>(raw_s + i*pitch + 16*k, src0 + (i*stride_row + 16*k));
        }
    } else {
#pragma unroll
        for (int o0 = 0; o0 < 16*nrem; o0 += 4*nthreads) {
            const int o = o0 + 4*tid;
            const int c = c_rem + o/16;
            const int i = c / chunks_per_row;
            const int k = c - i*chunks_per_row;
            const int b = 16*k + o % 16;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" : : "r"(raw_s + i*pitch + b), "l"(src0 + (i*stride_row + b)));
        }
    }
#else
    GGML_UNUSED_VARS(KV, raw, row0, stride_row);
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

// Convert pass k of a staged tile. The 8 lanes of a quarter warp take 8 consecutive rows of the same block:
// through the tile swizzle their 16-byte stores hit 8 distinct bank groups, and their reads of the staged rows do not conflict.
template<ggml_type type, int D, int stride_tile, int nthreads, int nbatch_fa, int k>
static __device__ __forceinline__ void flash_attn_ext_quant_dequant_pass(
        const char * const __restrict__ raw, half2 * const __restrict__ tile, const int i_first, const int i_step) {
    using traits = fattn_quant_type_traits<type>;

    constexpr int nblocks   = D/32;
    constexpr int bb        = ggml_cuda_fattn_quant_block_bytes(type);
    constexpr int pitch     = ggml_cuda_fattn_quant_stage_pitch(type, D);
    // Blocks of 2 mod 4 bytes alternate between aligned and misaligned, pass k takes one parity only.
    constexpr bool parity_split = bb % 4 != 0;
    constexpr int  mis          = parity_split ? (k*bb) % 4 : 0;
    static_assert(!parity_split || ((nblocks/4)*bb) % 4 == 0, "bad block split");

    const int q = threadIdx.x / 8;
    const int b = parity_split ? (nblocks/4)*q + k : 4*k + q;

#pragma unroll
    for (int i0 = 0; i0 < nbatch_fa; i0 += i_step) {
        const int i = i0 + i_first;

        __align__(16) half2 vals[16];
        traits::template dequant_block<mis>(raw + i*pitch + b*bb, vals);

#pragma unroll
        for (int c = 0; c < 4; ++c) {
            ggml_cuda_memcpy_1<16>(ggml_cuda_mma::swizzle<stride_tile>(tile, i*stride_tile + 16*b + 4*c, i), vals + 4*c);
        }
    }
}

// Convert a staged tile of nbatch_fa rows of D elements into the half2 tile.
template<ggml_type type, int D, int stride_tile, int nthreads, int nbatch_fa>
static __device__ __forceinline__ void flash_attn_ext_quant_dequant_tile(
        const char * const __restrict__ raw, half2 * const __restrict__ tile) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nblocks   = D/32;
    constexpr int nwarps    = nthreads/warp_size;
    static_assert(warp_size == 32, "the conversion lane layout assumes 32 lanes");
    static_assert(nblocks == 8 || nblocks == 16, "unsupported head size");
    static_assert(stride_tile == D/2, "the staged tile must be swizzled, not padded");

    if constexpr (8*nwarps <= nbatch_fa) {
        // Every warp takes 8 rows of each pass.
        static_assert(nbatch_fa % (8*nwarps) == 0, "bad nbatch_fa");
        const int i_first = 8*threadIdx.y + threadIdx.x % 8;
        flash_attn_ext_quant_dequant_pass<type, D, stride_tile, nthreads, nbatch_fa, 0>(raw, tile, i_first, 8*nwarps);
        flash_attn_ext_quant_dequant_pass<type, D, stride_tile, nthreads, nbatch_fa, 1>(raw, tile, i_first, 8*nwarps);
        if constexpr (nblocks == 16) {
            flash_attn_ext_quant_dequant_pass<type, D, stride_tile, nthreads, nbatch_fa, 2>(raw, tile, i_first, 8*nwarps);
            flash_attn_ext_quant_dequant_pass<type, D, stride_tile, nthreads, nbatch_fa, 3>(raw, tile, i_first, 8*nwarps);
        }
    } else {
        // More warps than rows: each warp takes 8 rows of one pass.
        static_assert(nwarps == (nbatch_fa/8)*(nblocks/4), "bad warp count");
        const int i_first = 8*(threadIdx.y % (nbatch_fa/8)) + threadIdx.x % 8;
        const int k       = threadIdx.y / (nbatch_fa/8);
        if (k == 0) {
            flash_attn_ext_quant_dequant_pass<type, D, stride_tile, nthreads, nbatch_fa, 0>(raw, tile, i_first, nbatch_fa);
        } else if (nblocks == 8 || k == 1) {
            flash_attn_ext_quant_dequant_pass<type, D, stride_tile, nthreads, nbatch_fa, 1>(raw, tile, i_first, nbatch_fa);
        } else if constexpr (nblocks == 16) {
            if (k == 2) {
                flash_attn_ext_quant_dequant_pass<type, D, stride_tile, nthreads, nbatch_fa, 2>(raw, tile, i_first, nbatch_fa);
            } else {
                flash_attn_ext_quant_dequant_pass<type, D, stride_tile, nthreads, nbatch_fa, 3>(raw, tile, i_first, nbatch_fa);
            }
        }
    }
}
