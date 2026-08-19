#pragma once

#include "fattn-mma-f16.cuh"

// Host-side launcher for the quantized-native Q8_0/Q8_0 MMA FlashAttention
// kernel. It is the F16 launcher with three differences:
//
// - the kernel is instantiated with type_K = type_V = GGML_TYPE_Q8_0, so the
//   shared-memory K/V tiles are filled by flash_attn_ext_q8_0_load_tile;
// - launch_fattn is told not to materialize F16 copies of K and V, which is the
//   transient allocation this path exists to avoid (Experiment 011);
// - the shared-memory budget is sized for a single pipeline stage, matching the
//   kernel's own nstages == 0 for quantized-native K/V.
//
// Scope is DKQ == DV, head dimension 128 or 256, on Ampere/Ada. The plan
// (docs/qn0-native-mma-kernel-plan.md) scoped the first pass to 128; 256 is
// required because it is the head dimension of the Qwen3.8 27B target model
// this work exists to serve, and it needs no loader change: the Ampere table
// gives nbatch_K2 = nbatch_V2 = 128 there, so a tile chunk is eight whole Q8_0
// blocks instead of four and every alignment invariant still holds.
// ggml_cuda_get_best_fattn_kernel enforces the scope; the static asserts here
// keep an out-of-scope instantiation from compiling silently.

template <int DKQ, int DV, int ncols1, int ncols2>
void ggml_cuda_flash_attn_ext_mma_q8_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    static_assert(DKQ == DV, "quantized-native MMA requires matching K/V head dimensions");
    static_assert(DKQ == 64 || DKQ == 128 || DKQ == 256,
            "quantized-native MMA is scoped to head dimension 64, 128 or 256");

    const ggml_tensor * KQV = dst;
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    constexpr int ncols = ncols1 * ncols2;

    const int  nthreads       = ggml_cuda_fattn_mma_get_nthreads      (DKQ, DV, ncols, cc);
    const int  nbatch_fa      = ggml_cuda_fattn_mma_get_nbatch_fa     (DKQ, DV, ncols, cc);
    const int  nbatch_K2      = ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols, cc);
    const int  nbatch_V2      = ggml_cuda_fattn_mma_get_nbatch_V2     (DKQ, DV, ncols, cc);
    const int  nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols, cc);
    const bool Q_in_reg       = ggml_cuda_fattn_mma_get_Q_in_reg      (DKQ, DV, ncols, cc);

    const int cols_per_warp = std::min(ncols, get_cols_per_warp(cc));
    const int warp_size_host = ggml_cuda_info().devices[ctx.device].warp_size;
    const int nwarps         = nthreads / warp_size_host;

    // MLA-style K/V aliasing only occurs at DKQ == 576, which is out of scope.
    constexpr bool V_is_K_view = false;

    // The kernel forces nstages to 0 for quantized-native K/V, so only the
    // single-stage K/V tile is ever live.
    const size_t nbytes_shared_KV      = nbatch_fa            * std::max(nbatch_K2 + 4, nbatch_V2 + 4) * sizeof(half2);
    const size_t nbytes_shared_Q       = ncols                * (DKQ/2 + 4)                            * sizeof(half2);
    const size_t nbytes_shared_mask    = ncols1               * (nbatch_fa/2 + 4)                      * sizeof(half2);
    const size_t nbytes_shared_combine = nwarps*cols_per_warp * (nbatch_combine + 4)                   * sizeof(half2);

    const size_t nbytes_shared_total = std::max(nbytes_shared_combine, Q_in_reg ?
        std::max(nbytes_shared_Q,  nbytes_shared_KV + nbytes_shared_mask) :
                 nbytes_shared_Q + nbytes_shared_KV + nbytes_shared_mask);

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

#if defined(GGML_USE_HIP)
    using fattn_kernel_ptr_t = const void*;
#else
    using fattn_kernel_ptr_t = fattn_kernel_t;
#endif // defined(GGML_USE_HIP)
    fattn_kernel_t fattn_kernel;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view,
            GGML_TYPE_Q8_0, GGML_TYPE_Q8_0>;

#if !defined(GGML_USE_MUSA)
        static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
        if (!shared_memory_limit_raised[id]) {
            CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
            shared_memory_limit_raised[id] = true;
        }
#endif // !defined(GGML_USE_MUSA)
    } else {
        constexpr bool use_logit_softcap = true;
        fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view,
            GGML_TYPE_Q8_0, GGML_TYPE_Q8_0>;

#if !defined(GGML_USE_MUSA)
        static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
        if (!shared_memory_limit_raised[id]) {
            CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
            shared_memory_limit_raised[id] = true;
        }
#endif // !defined(GGML_USE_MUSA)
    }

    constexpr bool need_f16_K = false;
    constexpr bool need_f16_V = false;
    launch_fattn<DV, ncols1, ncols2>
        (ctx, dst, fattn_kernel, nwarps, nbytes_shared_total, nbatch_fa, need_f16_K, need_f16_V, true, warp_size_host);
}

#define DECL_FATTN_MMA_Q8_CASE(DKQ, DV, ncols1, ncols2)                           \
    template void ggml_cuda_flash_attn_ext_mma_q8_case                            \
    <DKQ, DV, ncols1, ncols2>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

// Only the (ncols1, ncols2) pairs reachable from
// ggml_cuda_flash_attn_ext_mma_q8_switch_ncols2/1 are instantiated, for each
// supported head dimension. The F16 path's DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2 expansion is
// deliberately not reused: most of what it generates is unreachable for
// Q8_0/Q8_0 and would only cost compile time, in the same spirit as the fork's
// existing 103-pair vs. 169-pair vector-kernel tradeoff.
extern DECL_FATTN_MMA_Q8_CASE( 64,  64,  8,  1);
extern DECL_FATTN_MMA_Q8_CASE( 64,  64, 16,  1);
extern DECL_FATTN_MMA_Q8_CASE( 64,  64, 32,  1);
extern DECL_FATTN_MMA_Q8_CASE( 64,  64, 64,  1);
extern DECL_FATTN_MMA_Q8_CASE( 64,  64,  4,  2);
extern DECL_FATTN_MMA_Q8_CASE( 64,  64,  8,  2);
extern DECL_FATTN_MMA_Q8_CASE( 64,  64, 16,  2);
extern DECL_FATTN_MMA_Q8_CASE( 64,  64, 32,  2);
extern DECL_FATTN_MMA_Q8_CASE( 64,  64,  2,  4);
extern DECL_FATTN_MMA_Q8_CASE( 64,  64,  4,  4);
extern DECL_FATTN_MMA_Q8_CASE( 64,  64,  8,  4);
extern DECL_FATTN_MMA_Q8_CASE( 64,  64, 16,  4);
extern DECL_FATTN_MMA_Q8_CASE( 64,  64,  1,  8);
extern DECL_FATTN_MMA_Q8_CASE( 64,  64,  2,  8);
extern DECL_FATTN_MMA_Q8_CASE( 64,  64,  4,  8);
extern DECL_FATTN_MMA_Q8_CASE( 64,  64,  8,  8);

extern DECL_FATTN_MMA_Q8_CASE(128, 128,  8,  1);
extern DECL_FATTN_MMA_Q8_CASE(128, 128, 16,  1);
extern DECL_FATTN_MMA_Q8_CASE(128, 128, 32,  1);
extern DECL_FATTN_MMA_Q8_CASE(128, 128, 64,  1);

extern DECL_FATTN_MMA_Q8_CASE(128, 128,  4,  2);
extern DECL_FATTN_MMA_Q8_CASE(128, 128,  8,  2);
extern DECL_FATTN_MMA_Q8_CASE(128, 128, 16,  2);
extern DECL_FATTN_MMA_Q8_CASE(128, 128, 32,  2);

extern DECL_FATTN_MMA_Q8_CASE(128, 128,  2,  4);
extern DECL_FATTN_MMA_Q8_CASE(128, 128,  4,  4);
extern DECL_FATTN_MMA_Q8_CASE(128, 128,  8,  4);
extern DECL_FATTN_MMA_Q8_CASE(128, 128, 16,  4);

extern DECL_FATTN_MMA_Q8_CASE(128, 128,  1,  8);
extern DECL_FATTN_MMA_Q8_CASE(128, 128,  2,  8);
extern DECL_FATTN_MMA_Q8_CASE(128, 128,  4,  8);
extern DECL_FATTN_MMA_Q8_CASE(128, 128,  8,  8);

extern DECL_FATTN_MMA_Q8_CASE(256, 256,  8,  1);
extern DECL_FATTN_MMA_Q8_CASE(256, 256, 16,  1);
extern DECL_FATTN_MMA_Q8_CASE(256, 256, 32,  1);
extern DECL_FATTN_MMA_Q8_CASE(256, 256, 64,  1);

extern DECL_FATTN_MMA_Q8_CASE(256, 256,  4,  2);
extern DECL_FATTN_MMA_Q8_CASE(256, 256,  8,  2);
extern DECL_FATTN_MMA_Q8_CASE(256, 256, 16,  2);
extern DECL_FATTN_MMA_Q8_CASE(256, 256, 32,  2);

extern DECL_FATTN_MMA_Q8_CASE(256, 256,  2,  4);
extern DECL_FATTN_MMA_Q8_CASE(256, 256,  4,  4);
extern DECL_FATTN_MMA_Q8_CASE(256, 256,  8,  4);
extern DECL_FATTN_MMA_Q8_CASE(256, 256, 16,  4);

extern DECL_FATTN_MMA_Q8_CASE(256, 256,  1,  8);
extern DECL_FATTN_MMA_Q8_CASE(256, 256,  2,  8);
extern DECL_FATTN_MMA_Q8_CASE(256, 256,  4,  8);
extern DECL_FATTN_MMA_Q8_CASE(256, 256,  8,  8);
