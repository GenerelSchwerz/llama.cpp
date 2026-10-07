#pragma once

#include "ggml.h"
#include <cstddef>
#include <cstdint>

struct ggml_backend_cuda_context;

enum ggml_cuda_moe_source_norm_status {
    GGML_CUDA_MOE_SOURCE_NORM_INVALID = -1,
    GGML_CUDA_MOE_SOURCE_NORM_UNSUPPORTED = 0,
    GGML_CUDA_MOE_SOURCE_NORM_READY = 1,
};

// The owner retains dependencies, allocation lifetime and truthful backing extents.
struct ggml_cuda_moe_source_tensor {
    ggml_type type = GGML_TYPE_F32;
    int64_t ne[4] = {};
    size_t nb[4] = {};
    void * data = nullptr;
    void * buffer_base = nullptr;
    size_t buffer_bytes = 0;
    uintptr_t buffer_identity = 0;
    int device = -1;
};

class ggml_cuda_moe_source_binding {
public:
    ggml_cuda_moe_source_binding() = default;
private:
    ggml_cuda_moe_source_tensor tensors[7] = {};
    uint64_t rows = 0;
    float epsilon = 0;
    int count = 0;
    int device = -1;
    friend ggml_cuda_moe_source_norm_status ggml_cuda_moe_source_rms_binding_prepare(
        const ggml_cuda_moe_source_tensor *, float, ggml_cuda_moe_source_binding &);
    friend ggml_cuda_moe_source_norm_status ggml_cuda_moe_source_gdn_binding_prepare(
        const ggml_cuda_moe_source_tensor *, ggml_cuda_moe_source_binding &);
    friend bool ggml_cuda_moe_source_binding_matches(const ggml_cuda_moe_source_binding &,
        const ggml_cuda_moe_source_tensor *, int);
    friend bool ggml_cuda_moe_source_binding_emit(const ggml_cuda_moe_source_binding &, int, void *);
};

// RMS: input/gamma/output. GDN: input/alpha weights/beta weights/dt/A/gate/beta.
ggml_cuda_moe_source_norm_status ggml_cuda_moe_source_rms_binding_prepare(
    const ggml_cuda_moe_source_tensor * tensors, float epsilon, ggml_cuda_moe_source_binding & binding);
ggml_cuda_moe_source_norm_status ggml_cuda_moe_source_gdn_binding_prepare(
    const ggml_cuda_moe_source_tensor * tensors, ggml_cuda_moe_source_binding & binding);
bool ggml_cuda_moe_source_binding_matches(const ggml_cuda_moe_source_binding & binding,
    const ggml_cuda_moe_source_tensor * tensors, int count);
// The owner supplies the matching current device and CUDA stream; emission does not allocate or synchronize.
bool ggml_cuda_moe_source_binding_emit(const ggml_cuda_moe_source_binding & binding, int device, void * stream);

struct ggml_cuda_moe_source_norm {
    const ggml_cgraph * graph = nullptr;
    uint64_t graph_uid = 0;
    uint32_t index = 0;
    ggml_tensor * norm = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * gamma = nullptr;
    ggml_tensor * output = nullptr;
    int64_t ne[4] = {};
    int64_t gamma_ne[4] = {};
    size_t input_nb[4] = {};
    size_t gamma_nb[4] = {};
    size_t output_nb[4] = {};
    uint32_t rows = 0;
    float epsilon = 0;
    ggml_cuda_moe_source_binding operation;
    bool bound = false;
};

ggml_cuda_moe_source_norm_status ggml_cuda_moe_source_norm_prepare(
    const ggml_cgraph * graph, uint32_t index, ggml_tensor * norm, ggml_tensor * input,
    ggml_tensor * gamma, ggml_tensor * output, ggml_cuda_moe_source_norm & descriptor);
ggml_cuda_moe_source_norm_status ggml_cuda_moe_source_norm_bind(
    ggml_cuda_moe_source_norm & descriptor, int device);
bool ggml_cuda_moe_source_norm_emit(ggml_backend_cuda_context & context,
    const ggml_cuda_moe_source_norm & descriptor);

struct ggml_cuda_moe_source_gdn_ab {
    const ggml_cgraph * graph = nullptr;
    uint64_t graph_uid = 0;
    uint32_t index = 0;
    ggml_tensor * nodes[9] = {};
    ggml_tensor * reads[5] = {};
    const ggml_tensor * originals[14] = {};
    ggml_tensor metadata[14] = {};
    void * bound_data[7] = {};
    ggml_backend_buffer * bound_buffers[7] = {};
    uint64_t rows = 0;
    int device = -1;
    ggml_cuda_moe_source_binding operation;
    bool bound = false;
};

ggml_cuda_moe_source_norm_status ggml_cuda_moe_source_gdn_ab_prepare(
    const ggml_cgraph * graph, uint32_t index, ggml_tensor * const * nodes,
    ggml_cuda_moe_source_gdn_ab & descriptor);
ggml_cuda_moe_source_norm_status ggml_cuda_moe_source_gdn_ab_bind(
    ggml_cuda_moe_source_gdn_ab & descriptor, int device);
bool ggml_cuda_moe_source_gdn_ab_emit(ggml_backend_cuda_context & context,
    const ggml_cuda_moe_source_gdn_ab & descriptor);
