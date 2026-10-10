#pragma once

#include "ggml-backend.h"

// Optional backend extension. The producer must publish every armed input, including on failure.
struct ggml_staged_input_api {
    // The caller bounds capacity and keeps backing alive until all consumers retire.
    void * (*create)(ggml_backend_t backend, size_t bytes);
    void (*destroy)(void * input);
    void * (*data)(void * input);
    void (*publish)(void * input);
    // A consumer can use a logical prefix of the allocation.
    ggml_tensor * (*build)(void * input, ggml_context * ctx, ggml_tensor * dependency, int64_t elements);
};

typedef const ggml_staged_input_api * (*ggml_staged_input_get_api_t)();
#define GGML_STAGED_INPUT_PROC "ggml_backend_staged_input_v1"

// Register once before building consumers. Context stays live through their completion.
typedef bool (*ggml_staged_input_set_submit_t)(void * input, void (*submit)(void *), void * context);
#define GGML_STAGED_INPUT_SET_SUBMIT_PROC "ggml_backend_staged_input_set_submit_v1"

// True only after the previous consumer has copied the host payload.
typedef bool (*ggml_staged_input_consumed_t)(void * input);
#define GGML_STAGED_INPUT_CONSUMED_PROC "ggml_backend_staged_input_consumed_v1"
