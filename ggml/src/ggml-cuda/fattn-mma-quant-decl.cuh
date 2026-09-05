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

// D=256 with a GQA ratio above 4, the only row that covers the extra tier.
#define DECL_FATTN_MMA_QUANT_CASE_D256_GQA_WIDE(type_K, stem, tier, args) \
    FATTN_MMA_QUANT_TIER_##tier(extern DECL_FATTN_MMA_QUANT_CASE(type_K, 256, 256, 8, 8);) \
    FATTN_MMA_QUANT_TIER_##tier(extern DECL_FATTN_MMA_QUANT_CASE(type_K, 256, 256, 4, 8);)

FATTN_MMA_QUANT_TYPE_LIST(DECL_FATTN_MMA_QUANT_CASE_D256_GQA_WIDE, ())

// D=256 with a GQA ratio of 2, and D=512. Both rows are Q4_0/Q8_0 only, so they
// need no tier gate.
extern DECL_FATTN_MMA_QUANT_CASE(GGML_TYPE_Q4_0, 256, 256, 32, 2);
extern DECL_FATTN_MMA_QUANT_CASE(GGML_TYPE_Q8_0, 256, 256, 32, 2);
extern DECL_FATTN_MMA_QUANT_CASE(GGML_TYPE_Q4_0, 256, 256, 16, 2);
extern DECL_FATTN_MMA_QUANT_CASE(GGML_TYPE_Q8_0, 256, 256, 16, 2);
extern DECL_FATTN_MMA_QUANT_CASE(GGML_TYPE_Q4_0, 512, 512,  8, 8);
extern DECL_FATTN_MMA_QUANT_CASE(GGML_TYPE_Q8_0, 512, 512,  8, 8);
extern DECL_FATTN_MMA_QUANT_CASE(GGML_TYPE_Q4_0, 512, 512,  4, 8);
extern DECL_FATTN_MMA_QUANT_CASE(GGML_TYPE_Q8_0, 512, 512,  4, 8);

#endif // FATTN_MMA_QUANT_AVAILABLE
