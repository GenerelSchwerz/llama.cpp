#pragma once

#include "fattn-mma-f16.cuh"
#include "fattn-mma-quant-types.h"

// The exact set of quantized-native MMA kernels this build compiles.
//
// It mirrors ggml_cuda_fattn_native_supported() in fattn.cu one to one: every
// row below is a geometry that function can select, and nothing else exists.
// The two must agree, and a disagreement is a link error rather than a silent
// fallback, because these are the only instantiations.
//
// The declarations live here instead of in fattn-mma-f16.cuh because only
// fattn.cu needs them, which keeps them out of every generated instance file.

#ifdef FATTN_MMA_QUANT_AVAILABLE

// Every row has two shapes: the wide one for Ampere and newer, and the Turing
// one that switch_ncols1() caps at 32 columns. A build serves whichever card it
// runs on, so both are declared.

// D=64, D=128 and D=256 have the same rows: 8 heads per tile, and 2 heads per tile, see ggml_cuda_fattn_native_supported().
// Each also has narrower shapes for decode: 1x8 and 2x8 for one or two query tokens, 4x2 for up to 4.
#define DECL_FATTN_MMA_QUANT_CASE_D(type_K, stem, tier, D) \
    FATTN_MMA_QUANT_TIER_##tier(stem, extern DECL_FATTN_MMA_QUANT_CASE(type_K, D, D,  8, 8);) \
    FATTN_MMA_QUANT_TIER_##tier(stem, extern DECL_FATTN_MMA_QUANT_CASE(type_K, D, D,  4, 8);) \
    FATTN_MMA_QUANT_TIER_##tier(stem, extern DECL_FATTN_MMA_QUANT_CASE(type_K, D, D,  2, 8);) \
    FATTN_MMA_QUANT_TIER_##tier(stem, extern DECL_FATTN_MMA_QUANT_CASE(type_K, D, D,  1, 8);) \
    FATTN_MMA_QUANT_TIER_##tier(stem, extern DECL_FATTN_MMA_QUANT_CASE(type_K, D, D, 32, 2);) \
    FATTN_MMA_QUANT_TIER_##tier(stem, extern DECL_FATTN_MMA_QUANT_CASE(type_K, D, D, 16, 2);) \
    FATTN_MMA_QUANT_TIER_##tier(stem, extern DECL_FATTN_MMA_QUANT_CASE(type_K, D, D,  4, 2);)

// D=128 and D=256 also have 128-column shapes for prompt processing on Ampere and newer.
#define DECL_FATTN_MMA_QUANT_CASE_D128(type_K, stem, tier, D) \
    FATTN_MMA_QUANT_TIER_##tier(stem, extern DECL_FATTN_MMA_QUANT_CASE(type_K, D, D, 64, 2);) \
    FATTN_MMA_QUANT_TIER_##tier(stem, extern DECL_FATTN_MMA_QUANT_CASE(type_K, D, D, 16, 8);)

FATTN_MMA_QUANT_TYPE_LIST(DECL_FATTN_MMA_QUANT_CASE_D, 64)
FATTN_MMA_QUANT_TYPE_LIST(DECL_FATTN_MMA_QUANT_CASE_D, 128)
FATTN_MMA_QUANT_TYPE_LIST(DECL_FATTN_MMA_QUANT_CASE_D, 256)
FATTN_MMA_QUANT_TYPE_LIST(DECL_FATTN_MMA_QUANT_CASE_D128, 128)
FATTN_MMA_QUANT_TYPE_LIST(DECL_FATTN_MMA_QUANT_CASE_D128, 256)

// D=512 has 8 heads per tile only, for a GQA ratio above 4.
#define DECL_FATTN_MMA_QUANT_CASE_D512(type_K, stem, tier, args) \
    FATTN_MMA_QUANT_TIER_##tier(stem, extern DECL_FATTN_MMA_QUANT_CASE(type_K, 512, 512, 8, 8);) \
    FATTN_MMA_QUANT_TIER_##tier(stem, extern DECL_FATTN_MMA_QUANT_CASE(type_K, 512, 512, 4, 8);) \
    FATTN_MMA_QUANT_TIER_##tier(stem, extern DECL_FATTN_MMA_QUANT_CASE(type_K, 512, 512, 2, 8);) \
    FATTN_MMA_QUANT_TIER_##tier(stem, extern DECL_FATTN_MMA_QUANT_CASE(type_K, 512, 512, 1, 8);)

FATTN_MMA_QUANT_TYPE_LIST(DECL_FATTN_MMA_QUANT_CASE_D512, ())

#endif // FATTN_MMA_QUANT_AVAILABLE
