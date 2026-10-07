#pragma once

#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>

#define GGML_CUDA_MOE_FIDELITY_FIXTURE_V1_PROC_NAME "ggml_cuda_moe_fidelity_fixture_v1"

enum ggml_cuda_moe_fidelity_arm_v1 {
    GGML_CUDA_MOE_FIDELITY_DISABLED = 0,
    GGML_CUDA_MOE_FIDELITY_CALLBACK = 1,
    GGML_CUDA_MOE_FIDELITY_POLL = 2,
    GGML_CUDA_MOE_FIDELITY_SEGMENTED = 3,
    GGML_CUDA_MOE_FIDELITY_MEMOP = 4,
};

struct ggml_cuda_moe_fidelity_query_v1 {
    uint32_t struct_size;
    uint32_t boundaries;
    const ggml_cgraph * graph;
    const ggml_tensor * output;
};

// This certificate describes a synthetic transform fixture, not an executable model body.
struct ggml_cuda_moe_fidelity_certificate_v1 {
    uint64_t identity;
    uint64_t graph_uid;
    uint32_t row_capacity;
    uint32_t routes_per_row;
    uint32_t expert_count;
    uint32_t input_width;
    uint32_t output_width;
    uint32_t boundaries;
};

struct ggml_cuda_moe_fidelity_storage_v1 {
    uint64_t pinned_bytes;
    uint64_t device_bytes;
    uint64_t scratch_bytes;
};

struct ggml_cuda_moe_fidelity_options_v1 {
    uint32_t struct_size;
    uint32_t arm;
    uint32_t replays;
    uint32_t warmup;
    uint32_t cpu_delay_us;
    uint32_t gpu_delay_us;
    uint32_t validate_failures;
    uint32_t ready_memop;
    ggml_cuda_moe_fidelity_storage_v1 limits;
};

struct ggml_cuda_moe_fidelity_sample_v1 {
    uint64_t whole_ns;
    uint64_t launch_ns;
    uint64_t observation_ns;
    uint64_t continuation_ns;
    uint64_t cpu_service_ns;
    uint64_t ready_to_cpu_ns;
    uint64_t completion_publish_ns;
    uint64_t gpu_wait_ns;
    uint64_t gpu_work_ns;
    uint64_t gpu_commit_ns;
};

struct ggml_cuda_moe_fidelity_report_v1 {
    uint32_t struct_size;
    uint32_t samples;
    ggml_cuda_moe_fidelity_certificate_v1 certificate;
    ggml_cuda_moe_fidelity_storage_v1 storage;
    uint64_t captures;
    uint64_t graph_launches;
    uint64_t callbacks;
    uint64_t wait_nodes;
    uint64_t epochs;
    uint64_t services;
    uint64_t zero_cpu;
    uint64_t accepted;
    uint64_t rejected;
    uint64_t resident_routes;
    uint64_t transfer_routes;
    uint64_t cpu_routes;
    uint64_t coordinator_cpu_ns;
    uint64_t supervisor_cpu_ns;
    uint64_t caller_cpu_ns;
    uint64_t ready_observations;
    uint64_t finite_drains;
    char reason[128];
};

struct ggml_cuda_moe_fidelity_fixture_api_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    bool (*measure)(const ggml_cuda_moe_fidelity_query_v1 *, ggml_cuda_moe_fidelity_certificate_v1 *, ggml_cuda_moe_fidelity_storage_v1 *);
    // 1: passed, 0: transport unavailable, -1: failed, -2: storage limit, -3: invalid input.
    int32_t (*run)(ggml_backend_t, const ggml_cuda_moe_fidelity_query_v1 *, const ggml_cuda_moe_fidelity_options_v1 *, ggml_cuda_moe_fidelity_sample_v1 *, ggml_cuda_moe_fidelity_report_v1 *);
};

using ggml_cuda_moe_fidelity_fixture_get_v1_t = const ggml_cuda_moe_fidelity_fixture_api_v1 * (*)();
const ggml_cuda_moe_fidelity_fixture_api_v1 * ggml_cuda_moe_fidelity_fixture_api();
