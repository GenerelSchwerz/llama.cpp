#pragma once

#include "fattn-mma-f16.cuh"

// Explicit instantiation declarations for the quantized-native MMA kernels.
//
// These live in their own header rather than in fattn-mma-f16.cuh because the
// pair matrix makes them numerous: the full ordered cross product of the
// compiled native types over three head sizes and sixteen column shapes. Only
// fattn.cu needs them, so parsing that cross product is confined to one
// translation unit instead of every generated instance file.
//
// Every ordered pair of the compiled types is declared, because every ordered
// pair is defined: whatever the F16-casting path accepts for K and V, the
// native path accepts too. The list is a plain cross product of the type list
// against itself, so it cannot disagree with the generator about which pairs
// exist; if it ever did, the result would be a missing-symbol link error naming
// the exact pair rather than a silent gap.

#ifdef GGML_CUDA_FATTN_Q8_NATIVE

// Kept in step with FATTN_MMA_QUANT_TYPES in fattn-mma-quant.cuh, which drives
// the route predicate, and with FATTN_MMA_QUANT_TYPES in generate_cu_files.py,
// which emits the definitions. All three must name the same types.
#ifdef GGML_CUDA_FA_ALL_QUANTS
#define DECL_FATTN_MMA_QUANT_CASE_V_EXTRA(type_K, DKQ, DV, ncols1, ncols2)                    \
    extern DECL_FATTN_MMA_QUANT_CASE(type_K, GGML_TYPE_Q4_1,  DKQ, DV, ncols1, ncols2);       \
    extern DECL_FATTN_MMA_QUANT_CASE(type_K, GGML_TYPE_Q5_1,  DKQ, DV, ncols1, ncols2);       \
    extern DECL_FATTN_MMA_QUANT_CASE(type_K, GGML_TYPE_Q6_1,  DKQ, DV, ncols1, ncols2);       \
    extern DECL_FATTN_MMA_QUANT_CASE(type_K, GGML_TYPE_Q3_0,  DKQ, DV, ncols1, ncols2);       \
    extern DECL_FATTN_MMA_QUANT_CASE(type_K, GGML_TYPE_Q3_1,  DKQ, DV, ncols1, ncols2);       \
    extern DECL_FATTN_MMA_QUANT_CASE(type_K, GGML_TYPE_Q2_0S, DKQ, DV, ncols1, ncols2);       \
    extern DECL_FATTN_MMA_QUANT_CASE(type_K, GGML_TYPE_Q2_1,  DKQ, DV, ncols1, ncols2);
#else
#define DECL_FATTN_MMA_QUANT_CASE_V_EXTRA(type_K, DKQ, DV, ncols1, ncols2)
#endif // GGML_CUDA_FA_ALL_QUANTS

// One K row: this K against every compiled V type.
#define DECL_FATTN_MMA_QUANT_CASE_V(type_K, DKQ, DV, ncols1, ncols2)                          \
    extern DECL_FATTN_MMA_QUANT_CASE(type_K, GGML_TYPE_Q8_0,  DKQ, DV, ncols1, ncols2);       \
    extern DECL_FATTN_MMA_QUANT_CASE(type_K, GGML_TYPE_Q4_0,  DKQ, DV, ncols1, ncols2);       \
    extern DECL_FATTN_MMA_QUANT_CASE(type_K, GGML_TYPE_Q5_0,  DKQ, DV, ncols1, ncols2);       \
    extern DECL_FATTN_MMA_QUANT_CASE(type_K, GGML_TYPE_Q6_0,  DKQ, DV, ncols1, ncols2);       \
    DECL_FATTN_MMA_QUANT_CASE_V_EXTRA(type_K, DKQ, DV, ncols1, ncols2)

#ifdef GGML_CUDA_FA_ALL_QUANTS
#define DECL_FATTN_MMA_QUANT_CASE_TYPES_EXTRA(DKQ, DV, ncols1, ncols2)                        \
    DECL_FATTN_MMA_QUANT_CASE_V(GGML_TYPE_Q4_1,  DKQ, DV, ncols1, ncols2)                     \
    DECL_FATTN_MMA_QUANT_CASE_V(GGML_TYPE_Q5_1,  DKQ, DV, ncols1, ncols2)                     \
    DECL_FATTN_MMA_QUANT_CASE_V(GGML_TYPE_Q6_1,  DKQ, DV, ncols1, ncols2)                     \
    DECL_FATTN_MMA_QUANT_CASE_V(GGML_TYPE_Q3_0,  DKQ, DV, ncols1, ncols2)                     \
    DECL_FATTN_MMA_QUANT_CASE_V(GGML_TYPE_Q3_1,  DKQ, DV, ncols1, ncols2)                     \
    DECL_FATTN_MMA_QUANT_CASE_V(GGML_TYPE_Q2_0S, DKQ, DV, ncols1, ncols2)                     \
    DECL_FATTN_MMA_QUANT_CASE_V(GGML_TYPE_Q2_1,  DKQ, DV, ncols1, ncols2)
#else
#define DECL_FATTN_MMA_QUANT_CASE_TYPES_EXTRA(DKQ, DV, ncols1, ncols2)
#endif // GGML_CUDA_FA_ALL_QUANTS

#define DECL_FATTN_MMA_QUANT_CASE_TYPES(DKQ, DV, ncols1, ncols2)                              \
    DECL_FATTN_MMA_QUANT_CASE_V(GGML_TYPE_Q8_0,  DKQ, DV, ncols1, ncols2)                     \
    DECL_FATTN_MMA_QUANT_CASE_V(GGML_TYPE_Q4_0,  DKQ, DV, ncols1, ncols2)                     \
    DECL_FATTN_MMA_QUANT_CASE_V(GGML_TYPE_Q5_0,  DKQ, DV, ncols1, ncols2)                     \
    DECL_FATTN_MMA_QUANT_CASE_V(GGML_TYPE_Q6_0,  DKQ, DV, ncols1, ncols2)                     \
    DECL_FATTN_MMA_QUANT_CASE_TYPES_EXTRA(DKQ, DV, ncols1, ncols2)

#define DECL_FATTN_MMA_QUANT_CASE_ALL_NCOLS2(D, ncols)          \
    DECL_FATTN_MMA_QUANT_CASE_TYPES(D, D, (ncols)/1, 1);        \
    DECL_FATTN_MMA_QUANT_CASE_TYPES(D, D, (ncols)/2, 2);        \
    DECL_FATTN_MMA_QUANT_CASE_TYPES(D, D, (ncols)/4, 4);        \
    DECL_FATTN_MMA_QUANT_CASE_TYPES(D, D, (ncols)/8, 8);        \

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

#endif // GGML_CUDA_FATTN_Q8_NATIVE
