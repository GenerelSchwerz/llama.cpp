#pragma once

#include "fattn-mma-f16.cuh"
#include "fattn-mma-quant-types.h"

// Explicit instantiation declarations for the quantized-native MMA kernels.
//
// These live in their own header rather than in fattn-mma-f16.cuh because only
// fattn.cu needs them, which keeps them out of every generated instance file.
//
// Up to two declarations per K type, matching the generator: the symmetric pair
// keeps V as a template argument, and every mixed pair shares one runtime-V
// kernel per K type. So coverage of every ordered pair is linear in the number
// of native types rather than quadratic.
//
// The type inventory itself comes from fattn-mma-quant-types.h, which is also
// what the route predicate and the generator read, so these declarations cannot
// fall out of step with the definitions they name.

#define DECL_FATTN_MMA_QUANT_CASE_ENTRY(type_K, stem, tier, args)                        \
    FATTN_MMA_QUANT_TIER_##tier(                                                         \
        extern FATTN_MMA_QUANT_WITH(DECL_FATTN_MMA_QUANT_CASE_SYMMETRIC, type_K, args);  \
        FATTN_MMA_QUANT_MIXED_PAIRS(                                                     \
            extern FATTN_MMA_QUANT_WITH(DECL_FATTN_MMA_QUANT_CASE_RUNTIME_V, type_K, args);))

#define DECL_FATTN_MMA_QUANT_CASE_TYPES(DKQ, DV, ncols1, ncols2) \
    FATTN_MMA_QUANT_TYPE_LIST(DECL_FATTN_MMA_QUANT_CASE_ENTRY, (DKQ, DV, ncols1, ncols2))

// The entry macro terminates each declaration itself, so these expansions carry
// no trailing semicolon of their own.
#define DECL_FATTN_MMA_QUANT_CASE_ALL_NCOLS2(D, ncols)          \
    DECL_FATTN_MMA_QUANT_CASE_TYPES(D, D, (ncols)/1, 1)         \
    DECL_FATTN_MMA_QUANT_CASE_TYPES(D, D, (ncols)/2, 2)         \
    DECL_FATTN_MMA_QUANT_CASE_TYPES(D, D, (ncols)/4, 4)         \
    DECL_FATTN_MMA_QUANT_CASE_TYPES(D, D, (ncols)/8, 8)         \

DECL_FATTN_MMA_QUANT_CASE_ALL_NCOLS2( 64,  8)
DECL_FATTN_MMA_QUANT_CASE_ALL_NCOLS2( 64, 16)
DECL_FATTN_MMA_QUANT_CASE_ALL_NCOLS2( 64, 32)
DECL_FATTN_MMA_QUANT_CASE_ALL_NCOLS2( 64, 64)
DECL_FATTN_MMA_QUANT_CASE_ALL_NCOLS2(128,  8)
DECL_FATTN_MMA_QUANT_CASE_ALL_NCOLS2(128, 16)
DECL_FATTN_MMA_QUANT_CASE_ALL_NCOLS2(128, 32)
DECL_FATTN_MMA_QUANT_CASE_ALL_NCOLS2(128, 64)
DECL_FATTN_MMA_QUANT_CASE_ALL_NCOLS2(256,  8)
DECL_FATTN_MMA_QUANT_CASE_ALL_NCOLS2(256, 16)
DECL_FATTN_MMA_QUANT_CASE_ALL_NCOLS2(256, 32)
DECL_FATTN_MMA_QUANT_CASE_ALL_NCOLS2(256, 64)
extern DECL_FATTN_MMA_QUANT_CASE_SYMMETRIC(GGML_TYPE_Q8_0, 512, 512, 8, 8);
