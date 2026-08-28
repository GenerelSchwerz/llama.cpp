#pragma once

#include "fattn-mma-quant-packed.cuh"

template <>
struct fattn_quant_type_traits<GGML_TYPE_Q5_1> : fattn_quant_nibble_traits<block_q5_1, QK5_1, true, true> {};
