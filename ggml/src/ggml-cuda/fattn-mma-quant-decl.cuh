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

// D=256, the two rows that cover the extra tier: 8 heads per tile, and 2 heads per tile for an even GQA ratio.
// Each also has narrower shapes for decode: 1x8 and 2x8 for one or two query tokens, 4x2 for up to 4.
#define DECL_FATTN_MMA_QUANT_CASE_D256(type_K, stem, tier, args) \
    FATTN_MMA_QUANT_TIER_##tier(stem, extern DECL_FATTN_MMA_QUANT_CASE(type_K, 256, 256,  8, 8);) \
    FATTN_MMA_QUANT_TIER_##tier(stem, extern DECL_FATTN_MMA_QUANT_CASE(type_K, 256, 256,  4, 8);) \
    FATTN_MMA_QUANT_TIER_##tier(stem, extern DECL_FATTN_MMA_QUANT_CASE(type_K, 256, 256,  2, 8);) \
    FATTN_MMA_QUANT_TIER_##tier(stem, extern DECL_FATTN_MMA_QUANT_CASE(type_K, 256, 256,  1, 8);) \
    FATTN_MMA_QUANT_TIER_##tier(stem, extern DECL_FATTN_MMA_QUANT_CASE(type_K, 256, 256, 32, 2);) \
    FATTN_MMA_QUANT_TIER_##tier(stem, extern DECL_FATTN_MMA_QUANT_CASE(type_K, 256, 256, 16, 2);) \
    FATTN_MMA_QUANT_TIER_##tier(stem, extern DECL_FATTN_MMA_QUANT_CASE(type_K, 256, 256,  4, 2);)

FATTN_MMA_QUANT_TYPE_LIST(DECL_FATTN_MMA_QUANT_CASE_D256, ())

// D=512 is Q4_0/Q8_0 only, so it needs no tier gate.
extern DECL_FATTN_MMA_QUANT_CASE(GGML_TYPE_Q4_0, 512, 512,  8, 8);
extern DECL_FATTN_MMA_QUANT_CASE(GGML_TYPE_Q8_0, 512, 512,  8, 8);
extern DECL_FATTN_MMA_QUANT_CASE(GGML_TYPE_Q4_0, 512, 512,  4, 8);
extern DECL_FATTN_MMA_QUANT_CASE(GGML_TYPE_Q8_0, 512, 512,  4, 8);

#endif // FATTN_MMA_QUANT_AVAILABLE
