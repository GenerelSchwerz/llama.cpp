#pragma once

#include "../ggml-backend-moe.h"

#include <atomic>

struct ggml_cuda_moe_source_overlap_probe {
    uint32_t layer;
    const std::atomic<uint32_t> * completed;
};

struct ggml_cuda_moe_source_cpu_probe {
    uint32_t layer;
    const ggml_tensor * operation;
    const ggml_tensor * activation;
    const ggml_tensor * weight;
    const void * weight_data;
    const void * input_data;
    size_t input_bytes;
    const int32_t * ids;
    const uint8_t * ownership;
    const float * output;
    size_t output_bytes;
    uint32_t n_routes;
};

struct ggml_cuda_moe_source_gpu_probe {
    uint32_t layer;
    uint32_t n_layers;
    const ggml_tensor * operation;
    const ggml_tensor * activation;
    const ggml_tensor * weight;
    const void * weight_data;
    const void * input_data;
    size_t input_bytes;
    const uint32_t * route_indices;
    const int32_t * expert_ids;
    const int32_t * classes;
    uint32_t n_distinct;
    uint32_t n_routes;
    const ggml_tensor * source_witness;
    uint32_t source_domain;
};

const ggml_backend_moe_source_core_api_v1 * ggml_cuda_moe_source_core_api();
