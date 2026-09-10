#pragma once

#include "ggml-backend.h"

// Optional backend extension. The producer must publish every armed input, including on failure.
struct ggml_staged_input_api {
    void * (*create)(ggml_backend_t backend, size_t bytes);
    void (*destroy)(void * input);
    void * (*data)(void * input);
    void (*publish)(void * input);
    ggml_tensor * (*build)(void * input, ggml_context * ctx, ggml_tensor * dependency, int64_t elements);
};

typedef const ggml_staged_input_api * (*ggml_staged_input_get_api_t)();
#define GGML_STAGED_INPUT_PROC "ggml_backend_staged_input_v1"
