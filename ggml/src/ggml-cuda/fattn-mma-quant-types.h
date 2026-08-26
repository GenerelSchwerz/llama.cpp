#pragma once

// Single source of truth for the quantized-native MMA FlashAttention cache-type
// inventory.
//
// One entry per type: the ggml enum, the stem the generated translation unit
// carries in its file name, and the build tier.
//
//   DEFAULT  compiled by every CUDA FlashAttention build
//   EXTRA    compiled only when GGML_CUDA_FA_ALL_QUANTS is set
//
// Everything that has to agree about which types exist reads this list:
//
//   fattn-mma-quant.cuh                     route predicate, tier gate
//   fattn-mma-quant-decl.cuh                extern instantiation declarations
//   fattn.cu                                host-side K dispatch
//   fattn-mma-f16.cuh                       device-side runtime V selection
//   template-instances/generate_cu_files.py generated instance inventory
//   ggml/src/ggml-cuda/CMakeLists.txt       extra-tier source filter
//   tests/test-backend-ops.cpp              ordered-pair coverage
//
// The generator and CMake are not C++ and parse this file textually, so keep
// every entry on its own line in exactly the form below, and keep the stem
// spelled the way generate_cu_files.py's get_short_name() spells the enum.
//
// Adding a type is this one line plus its fattn_quant_type_traits
// specialization. Nothing else holds a second copy of the inventory, so a type
// can no longer be routable, generated, declared or tested in only some of
// those places. What the manifest does not decide is whether a type belongs
// here at all: the evidence a new type owes is the scope policy in
// docs/quantized-native-flash-attention.md.

#define FATTN_MMA_QUANT_TYPE_LIST(ENTRY, ARGS)   \
    ENTRY(GGML_TYPE_Q8_0,  q8_0,  DEFAULT, ARGS) \
    ENTRY(GGML_TYPE_Q6_1,  q6_1,  EXTRA,   ARGS) \
    ENTRY(GGML_TYPE_Q6_0,  q6_0,  DEFAULT, ARGS) \
    ENTRY(GGML_TYPE_Q5_1,  q5_1,  EXTRA,   ARGS) \
    ENTRY(GGML_TYPE_Q5_0,  q5_0,  DEFAULT, ARGS) \
    ENTRY(GGML_TYPE_Q4_1,  q4_1,  EXTRA,   ARGS) \
    ENTRY(GGML_TYPE_Q4_0,  q4_0,  DEFAULT, ARGS) \
    ENTRY(GGML_TYPE_Q3_1,  q3_1,  EXTRA,   ARGS) \
    ENTRY(GGML_TYPE_Q3_0,  q3_0,  EXTRA,   ARGS) \
    ENTRY(GGML_TYPE_Q2_1,  q2_1,  EXTRA,   ARGS) \
    ENTRY(GGML_TYPE_Q2_0S, q2_0s, EXTRA,   ARGS)

// Tier gates. FATTN_MMA_QUANT_TIER_<tier>(x) keeps x when this build compiles
// that tier and drops it otherwise. A visitor that wants only the compiled
// types wraps its expansion in FATTN_MMA_QUANT_TIER_##tier(...); one that wants
// the complete inventory, such as the backend-ops coverage that must also
// exercise the materializing fallback, simply ignores the tier argument.
#define FATTN_MMA_QUANT_TIER_DEFAULT(...) __VA_ARGS__
#ifdef GGML_CUDA_FA_ALL_QUANTS
#define FATTN_MMA_QUANT_TIER_EXTRA(...) __VA_ARGS__
#else
#define FATTN_MMA_QUANT_TIER_EXTRA(...)
#endif // GGML_CUDA_FA_ALL_QUANTS

// The compiled types as a single-argument X-macro list, which is the shape the
// selection switches want.
#define FATTN_MMA_QUANT_TYPES_ENTRY(type, stem, tier, F) FATTN_MMA_QUANT_TIER_##tier(F(type))
#define FATTN_MMA_QUANT_TYPES(F) FATTN_MMA_QUANT_TYPE_LIST(FATTN_MMA_QUANT_TYPES_ENTRY, F)

// Splices a parenthesized argument pack from FATTN_MMA_QUANT_TYPE_LIST's ARGS
// back into a call, for visitors that need the surrounding macro's parameters:
// FATTN_MMA_QUANT_WITH(M, t, (a, b)) is M(t, a, b). Passing them as one blob is
// what lets the manifest stay a plain list instead of one list per arity.
#define FATTN_MMA_QUANT_UNWRAP(...) __VA_ARGS__
#define FATTN_MMA_QUANT_WITH_(M, type, ...) M(type, __VA_ARGS__)
#define FATTN_MMA_QUANT_WITH(M, type, args) FATTN_MMA_QUANT_WITH_(M, type, FATTN_MMA_QUANT_UNWRAP args)
