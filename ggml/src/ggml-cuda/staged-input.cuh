#pragma once

#include "ggml.h"
#include <cstddef>
#include <cstdint>
#include "ggml-staged-input.h"

#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__MUSACC__)
#include "common.cuh"
#endif

struct ggml_backend_cuda_context;

bool ggml_cuda_staged_input_supports(const ggml_tensor * tensor);
bool ggml_cuda_staged_input_compute(ggml_backend_cuda_context & ctx, ggml_tensor * tensor);
const ggml_staged_input_api * ggml_cuda_staged_input_api();
bool ggml_cuda_staged_input_set_submit(void * input, void (*submit)(void *), void * context);
bool ggml_cuda_staged_input_pending_for_test(void * input);
bool ggml_cuda_staged_input_consumed(void * input);

struct ggml_cuda_source_staged_input_view {
    const void * host = nullptr;
    const void * device_alias = nullptr;
    const void * host_flag = nullptr;
    void * device_flag = nullptr;
    size_t bytes = 0;
    int device = -1;
    uint64_t identity = 0;
    void (*submit)(void *) = nullptr;
    void * context = nullptr;
};

// Borrowed view: caller retains genuine staged userdata/context and owns acquire/copy/consume.
// Query requires selected device and accepts pending flag0 without reading or consuming it.
bool ggml_cuda_staged_input_prepare_source_view(int device, const ggml_tensor * node, ggml_cuda_source_staged_input_view & view);
