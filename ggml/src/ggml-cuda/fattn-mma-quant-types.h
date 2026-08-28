#pragma once

// enum, generated file stem, build tier
// Keep one entry per line. CMake and generate_cu_files.py parse this list.

#define FATTN_MMA_QUANT_TYPE_LIST(ENTRY, ARGS)  \
    ENTRY(GGML_TYPE_Q8_0, q8_0, DEFAULT, ARGS)  \
    ENTRY(GGML_TYPE_Q5_1, q5_1, EXTRA,   ARGS)  \
    ENTRY(GGML_TYPE_Q5_0, q5_0, EXTRA,   ARGS)  \
    ENTRY(GGML_TYPE_Q4_1, q4_1, EXTRA,   ARGS)  \
    ENTRY(GGML_TYPE_Q4_0, q4_0, DEFAULT, ARGS)

// Select the types compiled by this build.
#define FATTN_MMA_QUANT_TIER_DEFAULT(...) __VA_ARGS__
#ifdef GGML_CUDA_FA_ALL_QUANTS
#define FATTN_MMA_QUANT_TIER_EXTRA(...) __VA_ARGS__
#else
#define FATTN_MMA_QUANT_TIER_EXTRA(...)
#endif // GGML_CUDA_FA_ALL_QUANTS

// Mixed K/V types require GGML_CUDA_FA_ALL_QUANTS.
#ifdef GGML_CUDA_FA_ALL_QUANTS
#define FATTN_MMA_QUANT_MIXED_PAIRS(...) __VA_ARGS__
#else
#define FATTN_MMA_QUANT_MIXED_PAIRS(...)
#endif // GGML_CUDA_FA_ALL_QUANTS

// Expand the types compiled by this build.
#define FATTN_MMA_QUANT_TYPES_ENTRY(type, stem, tier, F) FATTN_MMA_QUANT_TIER_##tier(F(type))
#define FATTN_MMA_QUANT_TYPES(F) FATTN_MMA_QUANT_TYPE_LIST(FATTN_MMA_QUANT_TYPES_ENTRY, F)

// Expand a parenthesized argument pack from FATTN_MMA_QUANT_TYPE_LIST.
#define FATTN_MMA_QUANT_UNWRAP(...) __VA_ARGS__
#define FATTN_MMA_QUANT_WITH_(M, type, ...) M(type, __VA_ARGS__)
#define FATTN_MMA_QUANT_WITH(M, type, args) FATTN_MMA_QUANT_WITH_(M, type, FATTN_MMA_QUANT_UNWRAP args)
