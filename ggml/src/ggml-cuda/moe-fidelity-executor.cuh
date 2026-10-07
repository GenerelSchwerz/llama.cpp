#pragma once

#include "moe-fidelity.cuh"
#include "ggml-backend-moe.h"

#include <memory>

#define GGML_CUDA_MOE_FIDELITY_WINDOW_V1_PROC_NAME "ggml_cuda_moe_fidelity_window_v1"

struct ggml_backend_cuda_context;

struct ggml_cuda_moe_fidelity_window_query_v1 {
    uint32_t struct_size;
    uint32_t n_regions;
    ggml_cgraph * graph;
    ggml_tensor * public_output;
    uint32_t n_public_outputs;
    ggml_tensor * const * public_outputs;
    void * graph_owner;
    int32_t (*retain_graph)(void *);
    int32_t (*release_graph)(void *);
    uint64_t arena_generation;
    const ggml_backend_moe_hybrid_region_v1 * const * regions;
    const ggml_backend_moe_source_owner_v1 * source_owner;
    // Caller pins the CPU module through measure/prepare. A prepared window takes its own pin.
    const ggml_backend_moe_cpu_region_service_api_v1 * cpu_api;
    const ggml_backend_moe_cpu_fidelity_requirements_api_v1 * cpu_requirements_api;
    ggml_graph_execution_certificate certificate;
    uint32_t n_threads;
    uint32_t gpu_miss_quota;
    uint32_t admission_quota;
    uint32_t no_host_alias;
    uint64_t staging_tile_bytes;
};

class ggml_cuda_moe_fidelity_graph_allocator {
public:
    ggml_cuda_moe_fidelity_graph_allocator();
    ~ggml_cuda_moe_fidelity_graph_allocator();

    bool measure(ggml_backend_buffer_type_t buft, const ggml_cuda_moe_fidelity_window_query_v1 & query);
    bool allocate();
    void reset();
    const ggml_tensor * find(const ggml_tensor * original) const;
    size_t buffer_bytes() const;
    size_t metadata_bytes() const;

private:
    struct impl;
    std::unique_ptr<impl> state;
};

struct ggml_cuda_moe_fidelity_window_storage_v1 {
    uint64_t device_bytes;
    uint64_t pinned_bytes;
    uint64_t cpu_bytes;
    uint64_t metadata_bytes;
    uint64_t device_runtime_bytes;
    uint64_t device_rows_bytes;
    uint64_t device_resident_body_bytes;
    uint64_t device_transfer_body_bytes;
    uint64_t device_clone_bytes;
    uint64_t device_workspace_bytes;
    uint64_t device_cublas_bytes;
    uint64_t device_staged_input_bytes;
    uint64_t pinned_runtime_bytes;
    uint64_t pinned_rows_bytes;
};

struct ggml_cuda_moe_fidelity_window_state_v1 {
    uint64_t identity;
    uint64_t captures;
    uint64_t launches;
    uint64_t epochs;
    uint64_t accepted;
    uint64_t rejected;
    uint64_t cpu_calls;
    uint64_t cpu_max_group_entries;
    uint64_t reference_before_b_guards;
    uint64_t cpu_routes;
    uint64_t resident_routes;
    uint64_t transfer_routes;
    uint64_t transfer_bytes;
    uint64_t executed_selected_bytes;
    uint64_t planning_callbacks;
    uint64_t transfer_callbacks;
    uint64_t producers_expected;
    uint64_t producers_completed;
    uint64_t admission_reserved;
    uint64_t admission_aborted;
    uint64_t admission_committed;
    uint64_t fallbacks;
    uint64_t retries;
    uint64_t finite_drains;
    uint64_t address_changes;
    uint64_t planned_clone_bytes;
    uint64_t protocol_probes;
    uint64_t source_banks;
    uint64_t mapped_source_banks;
    uint32_t segments;
    uint32_t mapped_alias;
    uint32_t terminal;
    uint32_t published;
    uint32_t active_jobs;
    ggml_cuda_moe_fidelity_window_storage_v1 storage;
};

enum ggml_cuda_moe_fidelity_window_fault_v1 {
    GGML_CUDA_MOE_FIDELITY_WINDOW_FAULT_NONE = 0,
    GGML_CUDA_MOE_FIDELITY_WINDOW_FAULT_CPU,
    GGML_CUDA_MOE_FIDELITY_WINDOW_FAULT_GPU,
    GGML_CUDA_MOE_FIDELITY_WINDOW_FAULT_TRANSFER,
    GGML_CUDA_MOE_FIDELITY_WINDOW_FAULT_STALE_PACKET,
    GGML_CUDA_MOE_FIDELITY_WINDOW_FAULT_CANCEL_BEFORE_COMMIT,
    GGML_CUDA_MOE_FIDELITY_WINDOW_FAULT_CANCEL_AFTER_COMMIT,
    GGML_CUDA_MOE_FIDELITY_WINDOW_FAULT_TEARDOWN,
    GGML_CUDA_MOE_FIDELITY_WINDOW_FAULT_REFERENCE_GROUP_COUNT,
    GGML_CUDA_MOE_FIDELITY_WINDOW_FAULT_REFERENCE_GROUP_START,
    GGML_CUDA_MOE_FIDELITY_WINDOW_FAULT_REFERENCE_GROUP_TOKEN,
    GGML_CUDA_MOE_FIDELITY_WINDOW_FAULT_REFERENCE_GROUP_DESTINATION,
    GGML_CUDA_MOE_FIDELITY_WINDOW_FAULT_REFERENCE_GROUP_POINTER,
    GGML_CUDA_MOE_FIDELITY_WINDOW_FAULT_REFERENCE_BEFORE_A,
    GGML_CUDA_MOE_FIDELITY_WINDOW_FAULT_REFERENCE_BEFORE_B,
};

struct ggml_cuda_moe_fidelity_window_replay_v1 {
    uint64_t epoch;
    uint64_t identity;
    uint32_t active_rows;
    uint32_t fault;
};

struct ggml_cuda_moe_fidelity_window_test_snapshot_v1 {
    uint64_t epoch;
    uint64_t input_hash;
    uint64_t control_hash;
    uint64_t output_hash;
    uint64_t imported_bytes;
    uintptr_t host_address;
    uintptr_t device_address;
    uint32_t claimed;
    uint32_t cpu_terminal;
    uint32_t wake;
    uint32_t stopped;
    uint32_t streams_complete;
    uint32_t join_complete;
    uint32_t join_cpu_terminal;
};

struct ggml_cuda_moe_fidelity_window_api_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t (*measure)(ggml_backend_t, const ggml_cuda_moe_fidelity_window_query_v1 *, ggml_cuda_moe_fidelity_window_state_v1 *);
    int32_t (*prepare)(ggml_backend_t, const ggml_cuda_moe_fidelity_window_query_v1 *,
        const ggml_cuda_moe_fidelity_window_storage_v1 *, uint32_t arm, void **);
    int32_t (*replay)(void *, const ggml_cuda_moe_fidelity_window_replay_v1 *, ggml_cuda_moe_fidelity_sample_v1 *);
    bool (*state)(void *, ggml_cuda_moe_fidelity_window_state_v1 *);
    int32_t (*close)(void *);
    int32_t (*drain)(void *);
    int32_t (*destroy)(void **);
    // Install only while quiescent. Snapshot requires a paused AFTER_COMMIT hook.
    int32_t (*set_cpu_test_hook)(void *, ggml_backend_moe_cpu_test_hook_v1_t, void *);
    bool (*test_snapshot)(void *, ggml_cuda_moe_fidelity_window_test_snapshot_v1 *);
};

using ggml_cuda_moe_fidelity_window_get_v1_t = const ggml_cuda_moe_fidelity_window_api_v1 * (*)();
const ggml_cuda_moe_fidelity_window_api_v1 * ggml_cuda_moe_fidelity_window_api();
const ggml_backend_moe_hybrid_api_v1 * ggml_cuda_moe_fidelity_hybrid_api();

#if defined(USE_CUDA_GRAPH) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 12030
cudaError_t ggml_cuda_moe_fidelity_add_if(cudaGraph_t graph, cudaGraphConditionalHandle handle,
    const std::vector<cudaGraphNode_t> & dependencies, cudaGraphNode_t & node, cudaGraph_t & body);
cudaError_t ggml_cuda_moe_fidelity_capture_to_graph(cudaGraph_t graph, cudaStream_t stream,
    std::vector<cudaGraphNode_t> & dependencies, cudaError_t (*emit)(void *), void * opaque);
bool ggml_cuda_moe_fidelity_body_supported(cudaGraph_t graph);
#endif

// The prepared graph retains the source metadata until capture and drain finish.
bool ggml_cuda_moe_fidelity_emit_node(ggml_backend_cuda_context & context, const ggml_cgraph * graph,
    uint64_t graph_uid, uint32_t index, ggml_tensor * private_node);
