#pragma once

#include "moe-fidelity.h"

struct ggml_moe_source_pool;

uint64_t ggml_moe_source_pool_bytes(uint32_t n_threads);
ggml_moe_source_pool * ggml_moe_source_pool_create(uint32_t n_threads);
void ggml_moe_source_pool_suspend(ggml_moe_source_pool * pool);
void ggml_moe_source_pool_stop(ggml_moe_source_pool * pool);
void ggml_moe_source_pool_free(ggml_moe_source_pool * pool);
ggml_status ggml_moe_source_pool_run(ggml_moe_source_pool * pool, ggml_moe_cpu_fidelity * body,
    const ggml_backend_moe_cpu_region_binding_v1 * binding, const ggml_backend_moe_cpu_dynamic_input_v1 * input,
    ggml_tensor * const * outputs, ggml_abort_callback abort, void * abort_data,
    ggml_backend_moe_cpu_test_hook_v1_t hook, void * hook_data,
    const ggml_backend_moe_cpu_output_v1 * private_outputs = nullptr, const uint32_t * destinations = nullptr);
