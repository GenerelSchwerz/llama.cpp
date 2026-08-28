#pragma once

#include "fattn-mma-quant-packed.cuh"

template <>
struct fattn_quant_type_traits<GGML_TYPE_Q5_0> : fattn_quant_nibble_traits<block_q5_0, QK5_0, false, true> {};
