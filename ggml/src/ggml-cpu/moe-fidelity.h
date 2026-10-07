#pragma once

#include "ggml-backend-moe.h"
#include "ggml-cpu.h"

struct ggml_moe_cpu_fidelity;

enum ggml_moe_cpu_fidelity_test_phase {
    GGML_MOE_CPU_FIDELITY_AFTER_GU = 3,
    GGML_MOE_CPU_FIDELITY_AFTER_QUANT = 4,
};

// The caller validates the region query and retains its source lease.
int32_t ggml_moe_cpu_fidelity_measure(const ggml_backend_moe_cpu_region_query_v1 * query, uint64_t * bytes);
ggml_moe_cpu_fidelity * ggml_moe_cpu_fidelity_prepare(const ggml_backend_moe_cpu_region_query_v1 * query, void * storage, uint64_t bytes);
void ggml_moe_cpu_fidelity_destroy(ggml_moe_cpu_fidelity * body);
ggml_status ggml_moe_cpu_fidelity_run(ggml_moe_cpu_fidelity * body, ggml_threadpool * pool,
    const ggml_backend_moe_cpu_region_binding_v1 * binding, const ggml_backend_moe_cpu_dynamic_input_v1 * input,
    ggml_tensor * const * outputs, ggml_abort_callback abort, void * abort_data,
    ggml_backend_moe_cpu_test_hook_v1_t hook, void * hook_data,
    const ggml_backend_moe_cpu_output_v1 * private_outputs = nullptr, const uint32_t * destinations = nullptr);

// Shared arithmetic adapters; the experimental pool does not own bindings or scratch.
ggml_status ggml_moe_cpu_fidelity_begin(ggml_moe_cpu_fidelity * body,
    const ggml_backend_moe_cpu_region_binding_v1 * binding, const ggml_backend_moe_cpu_dynamic_input_v1 * input,
    ggml_tensor * const * outputs, ggml_abort_callback abort, void * abort_data,
    ggml_backend_moe_cpu_test_hook_v1_t hook, void * hook_data,
    const ggml_backend_moe_cpu_output_v1 * private_outputs = nullptr, const uint32_t * destinations = nullptr);
void ggml_moe_cpu_fidelity_task(ggml_moe_cpu_fidelity * body, bool down, uint32_t task, uint32_t tasks);
void ggml_moe_cpu_fidelity_after_gu(ggml_moe_cpu_fidelity * body);
void ggml_moe_cpu_fidelity_quantize_hidden(ggml_moe_cpu_fidelity * body);
ggml_status ggml_moe_cpu_fidelity_result(ggml_moe_cpu_fidelity * body);
