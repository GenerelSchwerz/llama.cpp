#pragma once

#include "fattn-mma-quant-packed.cuh"

template <>
struct fattn_quant_type_traits<GGML_TYPE_Q4_1> : fattn_quant_nibble_traits<block_q4_1, QK4_1, true, false> {};
