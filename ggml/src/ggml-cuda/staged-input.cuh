#pragma once

#include "common.cuh"
#include "ggml-staged-input.h"

bool ggml_cuda_staged_input_supports(const ggml_tensor * tensor);
bool ggml_cuda_staged_input_compute(ggml_backend_cuda_context & ctx, ggml_tensor * tensor);
const ggml_staged_input_api * ggml_cuda_staged_input_api();
