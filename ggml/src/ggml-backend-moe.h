#pragma once

#include "ggml-backend.h"

// Private fork protocol shared by llama and backend implementations.
// Resolve these procedures through the backend registry; this header is not installed.
#ifdef __cplusplus
extern "C" {
#endif

#define GGML_BACKEND_MOE_CACHE_BUFFER_TYPE_PROC_NAME "ggml_backend_moe_cache_buffer_type"
#define GGML_BACKEND_MOE_CACHE_BOUNDED_BUFFER_TYPE_PROC_NAME "ggml_backend_moe_cache_bounded_buffer_type"
#define GGML_BACKEND_MOE_STAGING_SIZE_V1_PROC_NAME "ggml_backend_moe_staging_size_v1"
#define GGML_BACKEND_MOE_DEVICE_SIZE_V1_PROC_NAME "ggml_backend_moe_device_size_v1"
#define GGML_BACKEND_MOE_DEVICE_SIZE_V2_PROC_NAME             "ggml_backend_moe_device_size_v2"
#define GGML_BACKEND_MOE_CACHE_FREE_BUFFER_TYPE_PROC_NAME "ggml_backend_moe_cache_free_buffer_type"
#define GGML_BACKEND_MOE_CACHE_CONFIGURE_SOURCES_PROC_NAME "ggml_backend_moe_cache_configure_sources"
#define GGML_BACKEND_MOE_CACHE_IS_BUFFER_TYPE_PROC_NAME "ggml_backend_moe_cache_is_buffer_type"
#define GGML_BACKEND_MOE_CACHE_BUFFER_FROM_HOST_PTR_PROC_NAME "ggml_backend_moe_cache_buffer_from_host_ptr"
#define GGML_BACKEND_MOE_CACHE_WRITABLE_LOAD_DATA_PROC_NAME "ggml_backend_moe_cache_writable_load_data"
#define GGML_BACKEND_MOE_CACHE_READABLE_SOURCE_PROC_NAME "ggml_backend_moe_cache_readable_source"
#define GGML_BACKEND_MOE_CACHE_SET_DEBUG_PROC_NAME "ggml_backend_moe_cache_set_debug"
#define GGML_BACKEND_MOE_EARLY_ROUTER_SET_ENABLED_PROC_NAME "ggml_backend_moe_early_router_set_enabled"
#define GGML_BACKEND_MOE_EARLY_ROUTER_SET_MAX_ROWS_PROC_NAME "ggml_backend_moe_early_router_set_max_rows"
#define GGML_BACKEND_MOE_CACHE_LOG_AND_RESET_STATS_PROC_NAME "ggml_backend_moe_cache_log_and_reset_stats"

typedef ggml_backend_buffer_type_t (*ggml_backend_moe_cache_buffer_type_t)(void);
typedef ggml_backend_buffer_type_t (*ggml_backend_moe_cache_bounded_buffer_type_t)(size_t bytes);
typedef void (*ggml_backend_moe_cache_free_buffer_type_t)(ggml_backend_buffer_type_t buft);
typedef bool (*ggml_backend_moe_cache_is_buffer_type_t)(ggml_backend_buffer_type_t buft);
typedef ggml_backend_buffer_t (*ggml_backend_moe_cache_buffer_from_host_ptr_t)(ggml_backend_buffer_type_t buft, void * ptr, size_t size);
typedef void * (*ggml_backend_moe_cache_writable_load_data_t)(ggml_backend_buffer_t buffer, void * data, size_t size);
typedef bool (*ggml_backend_moe_cache_readable_source_t)(ggml_backend_buffer_t buffer, const void * data, size_t size);
// Experimental logging remains process-wide, including retired backend counters.
typedef void (*ggml_backend_moe_cache_set_debug_t)(bool enabled);
typedef void (*ggml_backend_moe_early_router_set_enabled_t)(bool enabled);
typedef void (*ggml_backend_moe_early_router_set_max_rows_t)(ggml_backend_t backend, uint32_t max_rows);
typedef void (*ggml_backend_moe_cache_log_and_reset_stats_t)(void);

#define GGML_BACKEND_MOE_CANDIDATE_REPLACE_V1_PROC_NAME "ggml_backend_moe_candidate_replace_v1"
#define GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V1_MAGIC 0x4d4f4531u
#define GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V1_VERSION 1u
#define GGML_BACKEND_MOE_CANDIDATE_REPLACE_V2_PROC_NAME "ggml_backend_moe_candidate_replace_v2"
#define GGML_BACKEND_MOE_CANDIDATE_REPLACE_CAPACITIES_V1_PROC_NAME "ggml_backend_moe_candidate_replace_capacities_v1"
#define GGML_BACKEND_REQUIRED_GROUPED_EXECUTION_SUPPORTED_PROC_NAME "ggml_backend_required_grouped_execution_supported"
#define GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_MAGIC 0x4d4f4532u
#define GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_VERSION 2u

enum ggml_backend_moe_candidate_snapshot_flag {
    GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_FLAG_NONE = 0,
};

enum ggml_backend_moe_candidate_group_flag {
    GGML_BACKEND_MOE_CANDIDATE_GROUP_FLAG_NONE = 0,
};

enum ggml_backend_moe_candidate_layout {
    GGML_BACKEND_MOE_CANDIDATE_LAYOUT_INVALID       = 0,
    GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE      = 1,
    GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP = 2,
    GGML_BACKEND_MOE_CANDIDATE_LAYOUT_UNGATED       = 3,
    GGML_BACKEND_MOE_CANDIDATE_LAYOUT_ROUTED_MATRIX = 4,
};

enum ggml_backend_moe_candidate_bank_role {
    GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID             = 0,
    GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT         = 1,
    GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT           = 2,
    GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT      = 3,
    GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT         = 4,
    GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_SCALE          = 5,
    GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_SCALE            = 6,
    GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_SCALE       = 7,
    GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_SCALE          = 8,
    GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_BLOCK_SCALE    = 9,
    GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_BLOCK_SCALE      = 10,
    GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_BLOCK_SCALE = 11,
    GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_BLOCK_SCALE    = 12,
    GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_BIAS           = 13,
    GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_BIAS             = 14,
    GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_BIAS        = 15,
    GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_BIAS           = 16,
    GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_INPUT_SCALE    = 17,
    GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_INPUT_SCALE      = 18,
    GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_INPUT_SCALE    = 19,
    GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_ROUTED_WEIGHT       = 20,
    GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_COUNT,
};

enum ggml_backend_moe_candidate_replace_result {
    GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED         = 0,
    GGML_BACKEND_MOE_CANDIDATE_REPLACE_REJECTED         = 1,
    GGML_BACKEND_MOE_CANDIDATE_REPLACE_INVALID_ARGUMENT = 2,
    GGML_BACKEND_MOE_CANDIDATE_REPLACE_INVALID_ABI      = 3,
    GGML_BACKEND_MOE_CANDIDATE_REPLACE_ERROR            = 4,
};

enum {
    GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS = 512,
    GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS  = 16,
    GGML_BACKEND_MOE_CANDIDATE_MAX_TENSORS_V2 = 16384,
};

enum ggml_backend_moe_staging_family_v1 {
    GGML_BACKEND_MOE_STAGING_FAMILY_V1_GROUPED     = 1u << 0,
    GGML_BACKEND_MOE_STAGING_FAMILY_V1_LEGACY      = 1u << 1,
    GGML_BACKEND_MOE_STAGING_FAMILY_V1_HOST_STAGED = 1u << 2,
};

enum ggml_backend_moe_staging_flag_v1 {
    GGML_BACKEND_MOE_STAGING_FLAG_V1_PREDICTION_CONTROL = 1u << 0,
};

struct ggml_backend_moe_staging_query_v1 {
    uint32_t struct_size;
    uint32_t n_slots;
    uint32_t n_experts;
    uint32_t top_k;
    uint32_t n_banks;
    uint32_t staged_bank_mask;
    uint32_t family_mask;
    uint32_t flags;
    uint64_t bank_expert_strides[GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS];
};

struct ggml_backend_moe_staging_size_v1 {
    uint32_t struct_size;
    uint32_t reserved32;
    uint64_t grouped_min_bytes;
    uint64_t legacy_min_bytes;
    uint64_t host_staged_min_bytes;
    uint64_t optional_growth_max_bytes;
    uint64_t prepack_tile_bytes;
};

typedef bool (*ggml_backend_moe_staging_size_v1_t)(
    const struct ggml_backend_moe_staging_query_v1 * query,
    struct ggml_backend_moe_staging_size_v1 * result);

enum ggml_backend_moe_device_size_flag_v1 {
    GGML_BACKEND_MOE_DEVICE_SIZE_FLAG_V1_DEBUG = 1u << 0,
};

struct ggml_backend_moe_device_size_query_v1 {
    uint32_t struct_size;
    uint32_t n_slots;
    uint32_t n_experts;
    uint32_t n_banks;
    uint32_t n_slot_auxiliaries;
    uint32_t flags;
    uint32_t early_width;
    uint32_t early_experts;
    uint32_t early_top_k;
    uint32_t early_hc_rank;
    uint64_t slot_auxiliary_values;
    uint64_t original_shadow_bytes;
    uint64_t prefill_copy_bytes;
    uint64_t bank_expert_strides[GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS];
    int64_t bank_ne0[GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS];
    int32_t bank_types[GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS];
};

struct ggml_backend_moe_device_size_v1 {
    uint32_t struct_size;
    uint32_t reserved32;
    uint64_t group_fixed_bytes;
    uint64_t group_per_slot_bytes;
    uint64_t context_fixed_bytes;
};

typedef bool (*ggml_backend_moe_device_size_v1_t)(
    const struct ggml_backend_moe_device_size_query_v1 * query,
    struct ggml_backend_moe_device_size_v1 * result);

struct ggml_backend_moe_device_size_query_v2 {
    uint32_t struct_size;
    uint32_t n_slots;
    uint32_t n_experts;
    uint32_t n_banks;
    uint32_t n_slot_auxiliaries;
    uint32_t flags;
    uint32_t early_width;
    uint32_t early_experts;
    uint32_t early_top_k;
    uint32_t early_hc_rank;
    uint64_t slot_auxiliary_values;
    uint64_t original_shadow_bytes;
    uint64_t prefill_copy_bytes;
    uint64_t bank_expert_strides[GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS];
    int64_t  bank_ne0[GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS];
    int32_t  bank_types[GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS];
    // Native graph-cloned router scratch is not available during model fit accounting.
    uint32_t early_route_capacity;
    uint32_t early_row_capacity;
    uint32_t early_groups;
    uint64_t early_expert_bytes;
};

struct ggml_backend_moe_device_size_v2 {
    uint32_t struct_size;
    uint32_t reserved32;
    uint64_t group_fixed_bytes;
    uint64_t group_per_slot_bytes;
    uint64_t context_fixed_bytes;
    uint64_t host_fixed_bytes;
};

typedef bool (*ggml_backend_moe_device_size_v2_t)(const struct ggml_backend_moe_device_size_query_v2 * query,
                                                  struct ggml_backend_moe_device_size_v2 *             result);

struct ggml_backend_moe_candidate_bank_v1 {
    const struct ggml_tensor * tensor;
    uint32_t role;
    uint32_t reserved;
};

struct ggml_backend_moe_candidate_group_v1 {
    const struct ggml_backend_moe_candidate_bank_v1 * banks;
    uint32_t n_banks;
    uint32_t layout;
    uint32_t flags;
    uint32_t reserved;
};

struct ggml_backend_moe_candidate_snapshot_v1 {
    uint32_t magic;
    uint32_t abi_version;
    uint32_t struct_size;
    // Implementations reject unknown flags and nonzero reserved fields.
    uint32_t flags;
    // Physical configuration; not part of a logical group signature.
    uint32_t n_slots;
    uint32_t n_groups;
    const struct ggml_backend_moe_candidate_group_v1 * groups;
    uint64_t reserved[2];
};

// A well-formed call replaces the complete backend-local snapshot.
// Arrays are borrowed for the call. Tensor pointers must outlive the accepted snapshot.
typedef int32_t (*ggml_backend_moe_candidate_replace_v1_t)(ggml_backend_t backend, const struct ggml_backend_moe_candidate_snapshot_v1 * snapshot);

enum ggml_backend_moe_candidate_domain_v2 {
    GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_INVALID  = 0,
    GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY = 1,
    GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_CHUNK    = 2,
};

enum ggml_backend_moe_candidate_status_v2 {
    GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_INVALID       = 0,
    GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE   = 1,
    GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_OUTPUT_SCALE  = 2,
    GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_OUTPUT_BIAS   = 3,
    GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_INPUT_SCALE   = 4,
    GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_UNCLASSIFIED  = 5,
    GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_EXCLUDED_SHARED = 6,
    GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_EXCLUDED_DENSE  = 7,
};

enum ggml_backend_moe_candidate_snapshot_flag_v2 {
    GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_FLAG_NONE             = 0,
    GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_FLAG_TENSOR_OVERRIDES = 1u << 0,
    GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_FLAG_INCOMPLETE       = 1u << 1,
};

enum ggml_backend_moe_candidate_group_flag_v2 {
    GGML_BACKEND_MOE_CANDIDATE_GROUP_V2_FLAG_NONE             = 0,
    GGML_BACKEND_MOE_CANDIDATE_GROUP_V2_FLAG_ACTIVE_LORA      = 1u << 0,
    GGML_BACKEND_MOE_CANDIDATE_GROUP_V2_FLAG_TENSOR_OVERRIDES = 1u << 1,
    GGML_BACKEND_MOE_CANDIDATE_GROUP_V2_FLAG_INCOMPLETE       = 1u << 2,
};

enum ggml_backend_moe_candidate_tensor_flag_v2 {
    GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_NONE             = 0,
    GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_CACHED_BUFFER    = 1u << 0,
    GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_ACTIVE_LORA      = 1u << 1,
    GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_TENSOR_OVERRIDES = 1u << 2,
};

struct ggml_backend_moe_candidate_group_v2 {
    uint32_t layout;
    uint32_t domain;
    uint32_t flags;
    uint32_t reserved;
};

struct ggml_backend_moe_candidate_tensor_v2 {
    const struct ggml_tensor * tensor;
    uint32_t group_index;
    uint32_t role;
    uint32_t status;
    uint32_t flags;
    uint32_t reserved;
};

struct ggml_backend_moe_candidate_snapshot_v2 {
    uint32_t magic;
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t flags;
    uint32_t n_slots;
    uint32_t n_groups;
    const struct ggml_backend_moe_candidate_group_v2 * groups;
    uint32_t n_tensors;
    uint32_t reserved32;
    const struct ggml_backend_moe_candidate_tensor_v2 * tensors;
    uint64_t reserved[2];
};

typedef int32_t (*ggml_backend_moe_candidate_replace_v2_t)(ggml_backend_t backend, const struct ggml_backend_moe_candidate_snapshot_v2 * snapshot);
typedef int32_t (*ggml_backend_moe_candidate_replace_capacities_v1_t)(ggml_backend_t backend,
        const struct ggml_backend_moe_candidate_snapshot_v2 * snapshot, const uint32_t * capacities, uint32_t n_capacities);
typedef bool (*ggml_backend_moe_cache_configure_sources_t)(ggml_backend_buffer_type_t buft, const struct ggml_backend_moe_candidate_snapshot_v2 * snapshot);
typedef bool (*ggml_backend_required_grouped_execution_supported_t)(ggml_backend_t backend);

#define GGML_BACKEND_MOE_SOURCE_OWNER_V1_VERSION 1u

enum {
    GGML_BACKEND_MOE_SOURCE_MAX_LEASES_V1 = 4096,
};

enum ggml_backend_moe_source_status_v1 {
    GGML_BACKEND_MOE_SOURCE_STATUS_V1_OK = 0,
    GGML_BACKEND_MOE_SOURCE_STATUS_V1_INVALID_ARGUMENT,
    GGML_BACKEND_MOE_SOURCE_STATUS_V1_INVALID_ABI,
    GGML_BACKEND_MOE_SOURCE_STATUS_V1_CLOSED,
    GGML_BACKEND_MOE_SOURCE_STATUS_V1_GENERATION_MISMATCH,
    GGML_BACKEND_MOE_SOURCE_STATUS_V1_BORROWED,
    GGML_BACKEND_MOE_SOURCE_STATUS_V1_CAPACITY,
    GGML_BACKEND_MOE_SOURCE_STATUS_V1_ALREADY_RELEASED,
};

enum ggml_backend_moe_source_owner_flag_v1 {
    GGML_BACKEND_MOE_SOURCE_OWNER_FLAG_V1_NONE     = 0,
    GGML_BACKEND_MOE_SOURCE_OWNER_FLAG_V1_BORROWED = 1u << 0,
};

struct ggml_backend_moe_source_lease_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    void *   owner;
    uint64_t owner_generation;
    uint64_t lease_id;
    uint64_t reserved[2];
};

struct ggml_backend_moe_source_span_v1 {
    uint32_t     struct_size;
    uint32_t     abi_version;
    const void * witness;
    const void * data;
    uint64_t     bytes;
    uint64_t     expert_stride;
    int32_t      type;
    uint32_t     reserved32;
    int64_t      ne[GGML_MAX_DIMS];
    uint64_t     nb[GGML_MAX_DIMS];
};

// This descriptor is borrowed for an immediate retain call. The caller must keep the model alive until retain returns.
// Copying a lease does not retain it. Only one copy may release ownership, before model teardown completes.
// A failed release must not release ownership or modify the lease, so the caller can retry.
// Do not call retain or release through a stale descriptor or lease after model teardown.
struct ggml_backend_moe_source_owner_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    void *   owner;
    uint64_t generation;
    uint32_t flags;
    uint32_t reserved32;
    int32_t (*retain)(const struct ggml_backend_moe_source_owner_v1 * owner,
                      uint64_t expected_generation,
                      struct ggml_backend_moe_source_lease_v1 * lease);
    int32_t (*release)(struct ggml_backend_moe_source_lease_v1 * lease);
    int32_t (*validate_span)(const struct ggml_backend_moe_source_owner_v1 * owner,
                             uint64_t expected_generation,
                             const struct ggml_backend_moe_source_span_v1 * span);
    uint64_t reserved[2];
};

#define GGML_BACKEND_SCHED_REGION_FINALIZE_V1_VERSION 1u

enum {
    GGML_BACKEND_SCHED_REGION_MAX_DYNAMIC_INPUTS_V1 = 32,
};

enum ggml_backend_sched_region_status_v1 {
    GGML_BACKEND_SCHED_REGION_STATUS_V1_OK = 0,
    GGML_BACKEND_SCHED_REGION_STATUS_V1_INVALID_ARGUMENT,
    GGML_BACKEND_SCHED_REGION_STATUS_V1_NOT_FINALIZED,
    GGML_BACKEND_SCHED_REGION_STATUS_V1_CALLBACK,
    GGML_BACKEND_SCHED_REGION_STATUS_V1_INCOMPLETE_CUT,
    GGML_BACKEND_SCHED_REGION_STATUS_V1_CROSS_SPLIT,
    GGML_BACKEND_SCHED_REGION_STATUS_V1_CAPACITY,
};

struct ggml_backend_sched_region_live_output_v1 {
    const struct ggml_tensor * const * consumers;
    const struct ggml_tensor *         tensor;
    uint32_t                           n_consumers;
    uint32_t                           reserved;
};

struct ggml_backend_sched_region_query_v1 {
    uint32_t                                                struct_size;
    uint32_t                                                abi_version;
    const struct ggml_cgraph *                              graph;
    const struct ggml_tensor * const *                      body_nodes;
    uint32_t                                                n_body_nodes;
    uint32_t                                                reserved32;
    const struct ggml_tensor * const *                      dynamic_inputs;
    uint32_t                                                n_dynamic_inputs;
    uint32_t                                                n_live_outputs;
    const struct ggml_backend_sched_region_live_output_v1 * live_outputs;
    const struct ggml_tensor *                              tail_resume; // null only at the end of the graph
};

struct ggml_backend_sched_region_handoff_v1 {
    uint32_t                   struct_size;
    uint32_t                   abi_version;
    uint64_t                   source_graph_uid;
    uint64_t                   split_graph_uid;
    uint32_t                   split_index;
    uint32_t                   first_node_index;
    uint32_t                   last_node_index;
    uint32_t                   tail_node_index; // graph node count at the end of the graph
    uint32_t                   n_dynamic_inputs;
    uint32_t                   reserved32;
    const struct ggml_tensor * dynamic_inputs[GGML_BACKEND_SCHED_REGION_MAX_DYNAMIC_INPUTS_V1];
};

static inline bool ggml_backend_sched_region_consumes_v1(const struct ggml_tensor * consumer, const struct ggml_tensor * tensor) {
    if (!consumer || !tensor) { return false; }
    if (consumer->view_src == tensor) { return true; }
    for (int src = 0; src < GGML_MAX_SRC; ++src) {
        const struct ggml_tensor * input = consumer->src[src];
        if (input && (input == tensor || input->view_src == tensor)) { return true; }
    }
    return false;
}

// The returned tensor pointers are borrowed only for the call that clones the finalized metadata.
GGML_API int32_t ggml_backend_sched_region_finalize_v1(
    ggml_backend_sched_t                                      sched,
    const struct ggml_backend_sched_region_query_v1 *         region,
    struct ggml_backend_sched_region_handoff_v1 *             handoff);
GGML_API bool ggml_backend_sched_region_snapshot_graph_v1(
    struct ggml_cgraph *                                  graph,
    const struct ggml_backend_sched_region_handoff_v1 *   handoff,
    struct ggml_tensor * const *                          nodes,
    uint32_t                                              n_nodes,
    struct ggml_tensor * const *                          leafs,
    uint32_t                                              n_leafs);
GGML_API bool ggml_backend_moe_graph_assign_uid_v1(struct ggml_cgraph * graph);
GGML_API uint64_t ggml_backend_moe_graph_uid_v1(const struct ggml_cgraph * graph);

#define GGML_BACKEND_MOE_CPU_REGION_QUERY_V1_PROC_NAME "ggml_backend_moe_cpu_region_query_v1"
#define GGML_BACKEND_MOE_CPU_REGION_SERVICE_V1_PROC_NAME "ggml_backend_moe_cpu_region_service_v1"
#define GGML_BACKEND_MOE_CPU_FIDELITY_SERVICE_V1_PROC_NAME "ggml_backend_moe_cpu_fidelity_service_v1"
#define GGML_BACKEND_MOE_CPU_FIDELITY_REQUIREMENTS_V1_PROC_NAME "ggml_backend_moe_cpu_fidelity_requirements_v1"

enum {
    GGML_BACKEND_MOE_CPU_REGION_MAX_LIVE_OUTPUTS_V1 = 8,
};

enum ggml_backend_moe_cpu_region_status_v1 {
    GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK = 0,
    GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT,
    GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION,
    GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_PRECISION,
    GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_MISSING_SOURCE,
    GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_BINDING,
    GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY,
    GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CLOSED,
    GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_ACTIVE_REGIONS,
    GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CANCELED,
    GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_COMPUTE_FAILED,
    GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_ACTIVE_JOBS,
};

enum ggml_backend_moe_cpu_region_flag_v1 {
    GGML_BACKEND_MOE_CPU_REGION_FLAG_V1_NONE           = 0,
    GGML_BACKEND_MOE_CPU_REGION_FLAG_V1_HAS_LORA       = 1u << 0,
    GGML_BACKEND_MOE_CPU_REGION_FLAG_V1_COMPACT_ROUTES = 1u << 1,
    GGML_BACKEND_MOE_CPU_REGION_FLAG_V1_ROUTED_OPERATION = 1u << 2,
};

struct ggml_backend_moe_cpu_region_source_v1 {
    const struct ggml_tensor * tensor;
    const void *               witness;  // opaque cache source/materialization witness
    const void *               data;
    uint64_t                   bytes;
    uint64_t                   expert_stride;
    uint64_t                   generation;
};

// All pointers stay valid and immutable through query or prepare. The graph is the finalized body-only graph in body_nodes order.
// The scheduler owns cut validation and common-tail resumption; this procedure retains no graph or source state.
struct ggml_backend_moe_cpu_region_query_v1 {
    uint32_t                                             struct_size;
    uint32_t                                             flags;
    const struct ggml_cgraph *                           graph;
    uint64_t                                             graph_uid;
    uint64_t                                             graph_generation;
    uint64_t                                             source_generation;
    const struct ggml_tensor * const *                   body_nodes;
    uint32_t                                             n_body_nodes;
    const struct ggml_tensor *                           activation;
    const struct ggml_tensor *                           ids;
    const struct ggml_tensor * const *                   dynamic_inputs;
    uint32_t                                             n_dynamic_inputs;
    const struct ggml_tensor * const *                   live_outputs;
    uint32_t                                             n_live_outputs;
    const struct ggml_backend_moe_cpu_region_source_v1 * sources;
    uint32_t                                             n_sources;
    uint32_t                                             bucket_rows;
    uint32_t                                             routes_per_row;
    uint32_t                                             source_row_capacity;
    uint32_t                                             scatter_capacity;
    uint32_t                                             n_threads;
    uint32_t                                             n_lanes;
    uint64_t                                             output_staging_limit;      // zero derives the exact requirement
    uint64_t                                             lane_execution_byte_limit; // zero derives the exact requirement
};

struct ggml_backend_moe_cpu_region_live_output_v1 {
    const struct ggml_tensor * tensor;
    uint64_t                   offset;
    uint64_t                   bytes;
    int64_t                    ne[GGML_MAX_DIMS];
    uint64_t                   nb[GGML_MAX_DIMS];
};

struct ggml_backend_moe_cpu_region_requirements_v1 {
    uint32_t struct_size;
    uint32_t n_threads;
    uint32_t n_lanes;
    uint32_t graph_nodes;
    uint32_t bucket_rows;
    uint32_t routes_per_row;
    uint32_t route_capacity;
    uint32_t expert_count;
    uint32_t immutable_sources;
    uint32_t n_live_outputs;
    uint64_t graph_work_bytes;
    uint64_t node_data_bytes;         // private non-view node values, including computed live outputs
    uint64_t dynamic_input_bytes;
    uint64_t binding_metadata_bytes;  // original source rows and scatter destinations
    uint64_t output_staging_bytes;    // separate publish-on-success destination capacity
    uint64_t lane_execution_bytes;
    uint64_t all_lane_execution_bytes;
    uint64_t borrowed_source_bytes;
    struct ggml_backend_moe_cpu_region_live_output_v1 live_outputs[GGML_BACKEND_MOE_CPU_REGION_MAX_LIVE_OUTPUTS_V1];
};

struct ggml_backend_moe_cpu_region_binding_v1 {
    uint32_t         struct_size;
    uint32_t         n_rows;
    uint32_t         n_routes;
    const int32_t *  expert_ids;
    const uint32_t * source_rows;
    const uint32_t * scatter_destinations;
};

struct ggml_backend_moe_cpu_region_query_api_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t (*query)(const struct ggml_backend_moe_cpu_region_query_v1 *  query,
                     struct ggml_backend_moe_cpu_region_requirements_v1 * requirements);
    int32_t (*validate_binding)(const struct ggml_backend_moe_cpu_region_query_v1 *   query,
                                const struct ggml_backend_moe_cpu_region_binding_v1 * binding);
};

typedef const struct ggml_backend_moe_cpu_region_query_api_v1 * (*ggml_backend_moe_cpu_region_query_v1_t)(void);

typedef void * ggml_backend_moe_cpu_service_v1_t;
// Zero is invalid. IDs are not reused within a loaded CPU module.
typedef uint64_t ggml_backend_moe_cpu_prepared_region_v1_t;

enum ggml_backend_moe_cpu_test_phase_v1 {
    GGML_BACKEND_MOE_CPU_TEST_PHASE_V1_ADMITTED,
    GGML_BACKEND_MOE_CPU_TEST_PHASE_V1_BEFORE_COMMIT,
    GGML_BACKEND_MOE_CPU_TEST_PHASE_V1_AFTER_COMMIT,
};

typedef void (*ggml_backend_moe_cpu_test_hook_v1_t)(void * data, uint32_t phase);

enum ggml_backend_moe_cpu_service_flag_v1 {
    GGML_BACKEND_MOE_CPU_SERVICE_FLAG_V1_ALLOW_UNKNOWN_THREAD_STACK_BYTES = 1u << 0,
    GGML_BACKEND_MOE_CPU_SERVICE_FLAG_V1_ALLOW_UNPROVEN_RUNTIME_ALLOCATIONS = 1u << 1,
};

enum ggml_backend_moe_cpu_prepared_flag_v1 {
    GGML_BACKEND_MOE_CPU_PREPARED_FLAG_V1_THREAD_STACK_BYTES_UNKNOWN = 1u << 0,
    GGML_BACKEND_MOE_CPU_PREPARED_FLAG_V1_RUNTIME_ALLOCATIONS_UNPROVEN = 1u << 1,
};

struct ggml_backend_moe_cpu_service_config_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    const struct ggml_backend_moe_source_owner_v1 * source_owner;
    uint32_t n_threads;
    uint32_t n_lanes;  // also bounds admitted jobs, including jobs waiting for the shared threadpool
    uint32_t max_regions;
    uint32_t flags;
    uint64_t prepared_payload_limit; // zero admits exact prepared regions up to max_regions
    uint64_t reserved[2];
};

struct ggml_backend_moe_cpu_prepared_requirements_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    struct ggml_backend_moe_cpu_region_requirements_v1 execution;
    uint32_t tensor_count;
    uint32_t external_source_count;
    uint32_t flags;
    uint32_t reserved32;
    uint64_t lane_metadata_bytes;
    uint64_t lane_context_bytes;
    uint64_t lane_alignment;
    uint64_t lane_allocation_bytes;
    uint64_t control_bytes;
    uint64_t prepared_payload_bytes;
    uint64_t thread_stack_bytes;
};

enum ggml_backend_moe_cpu_execute_flag_v1 {
    GGML_BACKEND_MOE_CPU_EXECUTE_FLAG_V1_NONE = 0,
    // The caller discards private output bytes on failure and reads them only after success.
    GGML_BACKEND_MOE_CPU_EXECUTE_FLAG_V1_PRIVATE_OUTPUTS = 1u << 0,
};

enum ggml_backend_moe_cpu_execute_result_flag_v1 {
    GGML_BACKEND_MOE_CPU_EXECUTE_RESULT_FLAG_V1_PUBLISHED = 1u << 0,
};

struct ggml_backend_moe_cpu_dynamic_input_v1 {
    const void * data;
    uint64_t     bytes;
    uint64_t     row_stride;
};

struct ggml_backend_moe_cpu_output_v1 {
    void *   data;
    uint64_t bytes;
    uint64_t route_stride;
};

// Input and output arrays are borrowed for one synchronous call. The ids input is generated from binding.expert_ids.
// Outputs must not overlap other outputs, immutable sources, or the borrowed descriptors and binding arrays.
struct ggml_backend_moe_cpu_execute_v1 {
    uint32_t                                                struct_size;
    uint32_t                                                flags;
    uint64_t                                                epoch;
    uint64_t                                                graph_uid;
    uint64_t                                                graph_generation;
    uint64_t                                                source_generation;
    const struct ggml_backend_moe_cpu_region_binding_v1 *   binding;
    const struct ggml_backend_moe_cpu_dynamic_input_v1 *    dynamic_inputs;
    uint32_t                                                n_dynamic_inputs;
    uint32_t                                                n_outputs;
    const struct ggml_backend_moe_cpu_output_v1 *           outputs;
    uint64_t                                                reserved[2];
};

struct ggml_backend_moe_cpu_execute_result_v1 {
    uint32_t struct_size;
    uint32_t flags;
    uint64_t epoch;
    uint64_t graph_uid;
    uint64_t graph_generation;
    uint64_t source_generation;
    uint32_t lane_index;
    uint32_t published_routes;
    uint32_t published_outputs;
    uint32_t reserved32;
    uint64_t compute_ns; // CPU worker service only; excludes lane acquisition, binding and scatter.
    uint64_t reserved[1];
};

struct ggml_backend_moe_cpu_service_state_v1 {
    uint32_t struct_size;
    uint32_t active_jobs;
    uint32_t active_regions;
    uint32_t closed;
    uint64_t prepared_payload_bytes;
    uint64_t prepared_payload_peak;
    uint64_t prepared_payload_limit;
};

// The final cancellation check is the output commit boundary. Cancel after that check does not roll back publication.
// After cancel or close, drain before output storage is reused. Destroy requires caller quiescence: no thread may enter
// any service procedure after destroy starts. Close, join callers, drain, destroy every region, then destroy the service
// before unloading its module.
struct ggml_backend_moe_cpu_region_service_api_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    // A failed create can return a closed service if lease cleanup fails. Destroy that handle to retry release.
    int32_t (*create)(const struct ggml_backend_moe_cpu_service_config_v1 * config,
                      ggml_backend_moe_cpu_service_v1_t * service);
    int32_t (*prepare)(ggml_backend_moe_cpu_service_v1_t service,
                       const struct ggml_backend_moe_cpu_region_query_v1 * query,
                       struct ggml_backend_moe_cpu_prepared_requirements_v1 * requirements,
                       ggml_backend_moe_cpu_prepared_region_v1_t * region);
    int32_t (*execute)(ggml_backend_moe_cpu_service_v1_t service,
                       ggml_backend_moe_cpu_prepared_region_v1_t region,
                       const struct ggml_backend_moe_cpu_execute_v1 * execution,
                       struct ggml_backend_moe_cpu_execute_result_v1 * result);
    int32_t (*state)(ggml_backend_moe_cpu_service_v1_t service,
                     struct ggml_backend_moe_cpu_service_state_v1 * state);
    // Set only with no active jobs. Hooks must not execute, close, or drain this service.
    int32_t (*set_test_hook)(ggml_backend_moe_cpu_service_v1_t service,
                             ggml_backend_moe_cpu_test_hook_v1_t hook, void * data);
    int32_t (*cancel)(ggml_backend_moe_cpu_service_v1_t service, uint64_t through_epoch);
    int32_t (*drain)(ggml_backend_moe_cpu_service_v1_t service);
    int32_t (*close)(ggml_backend_moe_cpu_service_v1_t service);
    int32_t (*destroy_region)(ggml_backend_moe_cpu_service_v1_t service,
                              ggml_backend_moe_cpu_prepared_region_v1_t * region);
    int32_t (*destroy)(ggml_backend_moe_cpu_service_v1_t * service);
    int32_t (*execute_routed)(ggml_backend_moe_cpu_service_v1_t service,
                              ggml_backend_moe_cpu_prepared_region_v1_t region,
                              const struct ggml_backend_moe_cpu_execute_v1 * execution,
                              const uint8_t * ownership, uint32_t n_ownership,
                              struct ggml_backend_moe_cpu_execute_result_v1 * result);
};

typedef const struct ggml_backend_moe_cpu_region_service_api_v1 * (*ggml_backend_moe_cpu_region_service_v1_t)(void);

#define GGML_BACKEND_MOE_CPU_ROUTED_EXECUTE_V1_PROC_NAME "ggml_backend_moe_cpu_routed_execute_v1"
// Ownership is one byte per original logical route: zero skips, one computes. It is borrowed until return.
// The prepared region is one ordinary MUL_MAT_ID; publication writes only owned routes after the cancellation check.
typedef int32_t (*ggml_backend_moe_cpu_routed_execute_v1_t)(
    ggml_backend_moe_cpu_service_v1_t service, ggml_backend_moe_cpu_prepared_region_v1_t region,
    const struct ggml_backend_moe_cpu_execute_v1 * execution, const uint8_t * ownership, uint32_t n_ownership,
    struct ggml_backend_moe_cpu_execute_result_v1 * result);

struct ggml_backend_moe_cpu_execute_control_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t flags;
    uint32_t reserved32;
    ggml_abort_callback abort;
    void * abort_data;
    ggml_backend_moe_cpu_test_hook_v1_t hook;
    void * hook_data;
    uint64_t reserved[2];
};

#define GGML_BACKEND_MOE_CPU_CONTROLLED_EXECUTE_V1_PROC_NAME "ggml_backend_moe_cpu_controlled_execute_v1"
// Synchronous call: callbacks/data are borrowed until return and must not throw or reenter the service.
// Abort can run on worker threads. Scoped calls ignore the legacy epoch watermark; close still cancels all jobs.
// Flags/reserved must be zero. Ownership is null/zero for a combined region; routed regions use the original mask.
typedef int32_t (*ggml_backend_moe_cpu_controlled_execute_v1_t)(
    ggml_backend_moe_cpu_service_v1_t service, ggml_backend_moe_cpu_prepared_region_v1_t region,
    const struct ggml_backend_moe_cpu_execute_v1 * execution, const uint8_t * ownership, uint32_t n_ownership,
    const struct ggml_backend_moe_cpu_execute_control_v1 * control, struct ggml_backend_moe_cpu_execute_result_v1 * result);

// Reads metadata only; does not retain sources, create workers or allocate prepared payload.
// Service returns base bytes. Add each region's prepared_payload_bytes once.
struct ggml_backend_moe_cpu_fidelity_requirements_api_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t (*service)(const struct ggml_backend_moe_cpu_service_config_v1 * config, uint64_t * bytes);
    int32_t (*region)(const struct ggml_backend_moe_cpu_service_config_v1 * config,
                     const struct ggml_backend_moe_cpu_region_query_v1 * query,
                     struct ggml_backend_moe_cpu_prepared_requirements_v1 * requirements);
};

typedef const struct ggml_backend_moe_cpu_fidelity_requirements_api_v1 * (*ggml_backend_moe_cpu_fidelity_requirements_v1_t)(void);

// Pins refuse explicit unload. Release only after every object and call into the module has ended.
GGML_API bool ggml_backend_moe_module_retain_v1(ggml_backend_reg_t reg);
GGML_API bool ggml_backend_moe_module_release_v1(ggml_backend_reg_t reg);
GGML_API ggml_backend_reg_t ggml_backend_moe_cpu_module_acquire_v1(void);

#define GGML_BACKEND_MOE_HYBRID_V1_PROC_NAME "ggml_backend_moe_hybrid_v1"
#define GGML_BACKEND_MOE_HYBRID_FIDELITY_V1_PROC_NAME "ggml_backend_moe_hybrid_fidelity_v1"
#define GGML_BACKEND_MOE_HYBRID_WINDOW_PROBE_V1_PROC_NAME "ggml_backend_moe_hybrid_window_probe_v1"
#define GGML_BACKEND_MOE_HYBRID_STAGED_PENDING_V1_PROC_NAME "ggml_backend_moe_hybrid_staged_pending_v1"
typedef bool (*ggml_backend_moe_hybrid_staged_pending_v1_t)(void * input);

// Returns 1 on success, 0 for unavailable conditional graphs, or -1 on probe failure.
typedef int32_t (*ggml_backend_moe_hybrid_window_probe_v1_t)(ggml_backend_t backend, uint32_t replays);

struct ggml_backend_moe_static_profile_v1 {
    const struct ggml_tensor * down;
    const int32_t * experts;
    uint32_t n_experts;
};

#define GGML_BACKEND_MOE_PROFILE_INITIALIZE_V1_PROC_NAME "ggml_backend_moe_profile_initialize_v1"
// Seed source placement and learning at a quiescent owner boundary, after learned-state restore.
#define GGML_BACKEND_MOE_PLACEMENT_SOURCE_V1 (1u << 2)
// Flags are zero or PLACEMENT_SOURCE_V1; initialization preserves live frequency metadata.
typedef bool (*ggml_backend_moe_profile_initialize_v1_t)(ggml_backend_t backend,
    const struct ggml_backend_moe_static_profile_v1 * profiles, uint32_t n_profiles, uint32_t flags, uint64_t * copied_bytes);

struct ggml_backend_moe_source_statistics_v1 {
    const struct ggml_tensor * tensor;
    const uint64_t * counts;
    uint64_t observations;
    uint32_t n_experts;
    uint32_t domain;
};

#define GGML_BACKEND_MOE_STATISTICS_INITIALIZE_V1_PROC_NAME "ggml_backend_moe_statistics_initialize_v1"
// Full source statistics are projected at the canonical owner boundary. Flags match profile initialization.
typedef bool (*ggml_backend_moe_statistics_initialize_v1_t)(ggml_backend_t backend,
    const struct ggml_backend_moe_source_statistics_v1 * statistics, uint32_t n_statistics, uint32_t flags, uint64_t * copied_bytes);

#define GGML_BACKEND_MOE_STATISTICS_INITIALIZE_V2_PROC_NAME "ggml_backend_moe_statistics_initialize_v2"
typedef bool (*ggml_backend_moe_statistics_initialize_v2_t)(ggml_backend_t backend,
    const struct ggml_backend_moe_source_statistics_v1 * statistics, const double * const * scores,
    uint32_t n_statistics, uint32_t flags, uint64_t * copied_bytes);

struct ggml_backend_moe_source_identity_v1 {
    const struct ggml_tensor * tensor;
    uint32_t domain;
};

struct ggml_backend_moe_source_learning_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    struct ggml_backend_moe_source_statistics_v1 source;
    const double * heat;
    const float * usage;
    const int32_t * prior;
    uint64_t windows;
};

// Views are borrowed until the callback returns. The callback must not reenter the owner.
typedef bool (*ggml_backend_moe_learning_snapshot_callback_v1_t)(
    const struct ggml_backend_moe_source_learning_v1 * records, uint32_t count, void * data);
#define GGML_BACKEND_MOE_LEARNING_SNAPSHOT_V1_PROC_NAME "ggml_backend_moe_learning_snapshot_v1"
// Quiescent snapshot; known sources without learning are omitted. Flags must be zero.
typedef bool (*ggml_backend_moe_learning_snapshot_v1_t)(ggml_backend_t backend,
    const struct ggml_backend_moe_source_identity_v1 * sources, uint32_t count, uint32_t flags, uint64_t deadline_ns,
    ggml_backend_moe_learning_snapshot_callback_v1_t callback, void * data);

#define GGML_BACKEND_MOE_LEARNING_RESTORE_V1_PROC_NAME "ggml_backend_moe_learning_restore_v1"
// Install complete bank groups before their first device preparation. Flags must be zero.
typedef bool (*ggml_backend_moe_learning_restore_v1_t)(ggml_backend_t backend,
    const struct ggml_backend_moe_source_learning_v1 * records, uint32_t count, uint32_t flags, uint64_t deadline_ns);

struct ggml_backend_moe_hybrid_config_v1 {
    uint32_t struct_size;
    uint32_t n_threads;
    uint32_t max_regions;
    uint32_t max_prepared_regions;
    uint32_t cpu_flags;
    uint32_t gpu_miss_quota;
    uint32_t admission_quota;
    uint32_t demand_admission;
    uint32_t resident_batch;
    uint32_t executor;
    ggml_backend_t backend;
    const struct ggml_backend_moe_source_owner_v1 * source_owner;
    ggml_backend_reg_t (*cpu_module_acquire)(void);
    bool (*module_retain)(ggml_backend_reg_t reg);
    bool (*module_release)(ggml_backend_reg_t reg);
    uint64_t cpu_bytes;    // zero derives the exact prepared payload
    uint64_t device_bytes; // zero derives the largest prepared region and workspace
    uint64_t pinned_bytes; // zero derives input, output, and source staging
    // Arrays are borrowed for configure/create only. Tensor identities share the model source lifetime.
    // Legacy rank profiles and full source statistics are mutually exclusive.
    const struct ggml_backend_moe_static_profile_v1 * profiles;
    uint32_t n_profiles;
    const struct ggml_backend_moe_source_statistics_v1 * statistics;
    uint32_t n_statistics;
    uint32_t profile_adaptation; // zero disables, one selects synchronous occurrence adaptation, two selects asynchronous occurrence adaptation
    const double * const * statistics_scores; // optional derived scores; same source order as statistics
    uint32_t max_source_programs; // zero keeps per-session limits; otherwise zero byte limits derive a largest-program envelope
    void * resource_context;
    const void * resource_program;
    // Reserve managed private storage before allocation; zero bytes release the session reservation after disposal.
    bool (*reserve_resources)(void * context, const void * program, const void * session, ggml_backend_t backend, const void * shared_owner,
                              uint64_t device_bytes, uint64_t pinned_bytes, uint64_t shared_device_bytes);
};

enum ggml_backend_moe_hybrid_executor_v1 {
    GGML_BACKEND_MOE_HYBRID_EXECUTOR_V1_EAGER = 0,
    GGML_BACKEND_MOE_HYBRID_EXECUTOR_V1_FIDELITY = 1,
};

struct ggml_backend_moe_hybrid_geometry_v1 {
    uint32_t row_capacity;
    uint32_t routes_per_row;
    uint32_t route_capacity;
    uint32_t expert_count;
    uint32_t weight_capacity;
};

struct ggml_backend_moe_hybrid_route_v1 {
    uint32_t source_row;
    uint32_t source_route;
    uint32_t weight_index;
    uint32_t scatter_destination;
};

struct ggml_backend_moe_hybrid_binding_v1 {
    uint32_t struct_size;
    uint32_t active_rows;
    uint32_t n_weights;
    uint32_t n_routes;
    uint64_t epoch;
    uint64_t source_graph_uid;
    uint64_t split_graph_uid;
    uint64_t owner_generation;
    uint64_t allocator_generation;
    uint64_t source_generation;
    const int32_t * weight_experts;
    const struct ggml_backend_moe_hybrid_route_v1 * routes;
};

struct ggml_backend_moe_hybrid_region_v1 {
    uint32_t struct_size;
    uint32_t split_index;
    uint32_t first_node;
    uint32_t last_node;
    uint64_t source_graph_uid;
    uint64_t split_graph_uid;
    uint64_t owner_generation;
    uint64_t allocator_generation;
    const struct ggml_tensor * activation;
    const struct ggml_tensor * ids;
    struct ggml_tensor * output;
    const struct ggml_tensor * down;
    const struct ggml_backend_moe_cpu_region_query_v1 * query;
    const struct ggml_backend_moe_cpu_region_query_v1 * const * cpu_queries;
    uint32_t n_cpu_queries;
    const struct ggml_backend_moe_cpu_region_query_v1 * const * cpu_batch_queries;
    uint32_t n_cpu_batch_queries;
    struct ggml_backend_moe_hybrid_geometry_v1 geometry;
    struct ggml_graph_execution_certificate certificate;
    // Original body semantics before CPU bucket reshaping. Borrowed only through preparation.
    const struct ggml_backend_moe_cpu_region_query_v1 * body_query;
};

struct ggml_backend_moe_hybrid_state_v1 {
    uint32_t struct_size;
    uint32_t ticket_state;
    uint64_t resident_routes;
    uint64_t transfer_routes;
    uint64_t cpu_routes;
    uint64_t h2d_bytes;
    uint64_t device_bytes;
    uint64_t pinned_bytes;
    uint64_t work_peak;
    uint64_t distinct_experts;
    uint64_t resident_experts;
    uint64_t transfer_experts;
    uint64_t cpu_experts;
    uint64_t cpu_jobs;
    uint64_t cpu_upload_bytes;
    uint64_t cpu_us;
    uint64_t gpu_enqueue_us;
    uint64_t join_us;
    uint64_t errors;
    uint64_t capacity_errors;
    uint64_t cancellations;
    uint64_t last_epoch;
    uint32_t cpu_active_jobs;
    uint32_t dispatch_active;
    uint32_t quiescing;
    uint64_t submit_to_start_us;
    uint64_t gpu_branch_us;
    uint64_t last_submit_us;
    uint64_t last_cpu_start_us;
    uint64_t last_cpu_done_us;
    uint64_t last_gpu_enqueue_done_us;
    uint64_t last_gpu_done_observed_us;
    uint64_t last_join_start_us;
    uint64_t last_join_done_us;
    uint64_t admission_reserved;
    uint64_t admission_committed;
    uint64_t admission_replacements;
    uint64_t admission_no_slot;
    uint64_t admission_bytes;
    uint64_t admission_aborted;
    uint64_t resident_batches;
    uint64_t resident_body_submissions;
    uint64_t gpu_body_submissions;
    uint64_t readback_us;
    uint64_t publication_us;
    uint64_t admission_fence_us;
    uint64_t prepared_device_bytes;
    uint64_t prepared_cpu_bytes;
    uint64_t packet_regions;
    uint64_t producer_events;
    uint64_t producer_fences;
    uint64_t producer_drain_events;
    uint64_t window_launches;
    uint64_t window_captures;
    uint64_t window_waits;
    uint64_t window_fallbacks;
    uint64_t window_fused_nodes;
    uint64_t window_combined_regions;
    uint64_t window_direct_regions;
    uint64_t window_compact_select_regions;
    uint64_t window_fused_expert_bodies;
    uint64_t cpu_execute_calls;
    uint64_t cpu_batch_rows;
};

// Source and route views are borrowed until the observer returns.
struct ggml_backend_moe_source_access_v1 {
    uint32_t layer;
    uint32_t n_layers;
    const struct ggml_tensor * source;
    const struct ggml_tensor * source_witness;
    uint32_t source_domain;
    const struct ggml_tensor * ids;
    const uint32_t * route_indices;
    const int32_t * expert_ids;
    const int32_t * classes;
    uint32_t n_distinct;
    uint32_t n_routes;
    const uint8_t * skipped_routes;
    const struct ggml_tensor * output;
};

enum ggml_backend_moe_hybrid_test_phase_v1 {
    GGML_BACKEND_MOE_HYBRID_TEST_CPU_QUEUED = 1,
    GGML_BACKEND_MOE_HYBRID_TEST_CPU_ADMITTED,
    GGML_BACKEND_MOE_HYBRID_TEST_CPU_BEFORE_COMMIT,
    GGML_BACKEND_MOE_HYBRID_TEST_GPU_ENQUEUED,
    GGML_BACKEND_MOE_HYBRID_TEST_BEFORE_PUBLISH,
    GGML_BACKEND_MOE_HYBRID_TEST_BEFORE_ROUTES,
    GGML_BACKEND_MOE_HYBRID_TEST_ADMISSION_BANK_COPIED,
    GGML_BACKEND_MOE_HYBRID_TEST_TRANSFER_ENQUEUED,
    GGML_BACKEND_MOE_HYBRID_TEST_WINDOW_SUBMIT,
    GGML_BACKEND_MOE_HYBRID_TEST_WINDOW_BODY,
    GGML_BACKEND_MOE_HYBRID_TEST_CPU_JOINED,
    GGML_BACKEND_MOE_HYBRID_TEST_CPU_AFTER_COMMIT,
    GGML_BACKEND_MOE_HYBRID_TEST_SOURCE_COMPUTE_ENTERED,
    GGML_BACKEND_MOE_HYBRID_TEST_SOURCE_COMPUTE_RETURNING,
    GGML_BACKEND_MOE_HYBRID_TEST_SOURCE_GRAPH_LAUNCHED,
    GGML_BACKEND_MOE_HYBRID_TEST_SOURCE_OVERLAP_PROBE,
    GGML_BACKEND_MOE_HYBRID_TEST_SOURCE_ADAPT_BEFORE_COMPLETE,
    GGML_BACKEND_MOE_HYBRID_TEST_SOURCE_ROUTED_CPU_PROBE,
    GGML_BACKEND_MOE_HYBRID_TEST_SOURCE_ROUTED_PUBLICATION_PROBE,
    GGML_BACKEND_MOE_HYBRID_TEST_SOURCE_ROUTED_GPU_PUBLICATION_PROBE,
    GGML_BACKEND_MOE_HYBRID_TEST_SOURCE_ACCESS,
};

// Hooks run on the compute caller, coordinator or adaptation worker. Device hooks borrow their producer stream; CPU hooks receive null.
// Return false to fail that producer.
// SOURCE_OVERLAP_PROBE, SOURCE_ROUTED_*_PROBE and SOURCE_ACCESS borrow observation views for this call, not streams.
typedef bool (*ggml_backend_moe_hybrid_test_hook_v1_t)(void * data, uint32_t phase, uint64_t epoch, void * stream);

// Prepare borrows descriptors only for the call. Compute is synchronous through result publication.
struct ggml_backend_moe_hybrid_api_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t (*create)(const struct ggml_backend_moe_hybrid_config_v1 * config, void ** session);
    int32_t (*prepare)(void * session, const struct ggml_backend_moe_hybrid_region_v1 * region,
                       const struct ggml_backend_moe_cpu_region_service_api_v1 * cpu_api,
                       ggml_backend_moe_cpu_service_v1_t cpu_service,
                       const ggml_backend_moe_cpu_prepared_region_v1_t * cpu_regions, void ** prepared);
    enum ggml_status (*compute)(void * session, struct ggml_cgraph * graph, void * const * regions, uint32_t count);
    void (*destroy_region)(void * session, void * prepared);
    void (*destroy)(void * session);
    bool (*state)(void * session, struct ggml_backend_moe_hybrid_state_v1 * state);
    void (*quiesce)(void * session);
    bool (*set_test_hook)(void * session, ggml_backend_moe_hybrid_test_hook_v1_t hook, void * data);
};

typedef const struct ggml_backend_moe_hybrid_api_v1 * (*ggml_backend_moe_hybrid_v1_t)(void);

#define GGML_BACKEND_MOE_SOURCE_CORE_V1_PROC_NAME "ggml_backend_moe_source_core_v1"

enum ggml_backend_moe_source_core_status_v1 {
    GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK = 0,
    GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_BUSY = 1,
    GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_TIMEOUT = 2,
    GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_INVALID = 3,
    GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_FAILED = 4,
};

// Source-only classification in hybrid_state_v1.ticket_state.
#define GGML_BACKEND_MOE_SOURCE_CORE_TICKET_V1_REJECTED_BEFORE_EFFECTS 1u

struct ggml_backend_moe_source_core_api_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t (*create)(const struct ggml_backend_moe_hybrid_config_v1 * config, void ** session);
    int32_t (*prepare)(void * session, const struct ggml_backend_moe_hybrid_region_v1 * region,
                       const struct ggml_backend_moe_cpu_region_service_api_v1 * cpu_api,
                       ggml_backend_moe_cpu_service_v1_t cpu_service,
                       const ggml_backend_moe_cpu_prepared_region_v1_t * cpu_regions, void ** prepared);
    enum ggml_status (*compute)(void * session, struct ggml_cgraph * graph, void * const * regions, uint32_t count);
    int32_t (*close)(void * session);
    int32_t (*drain)(void * session);
    int32_t (*release_region)(void * session, void ** prepared);
    int32_t (*release)(void ** session);
    bool (*state)(void * session, struct ggml_backend_moe_hybrid_state_v1 * state);
    bool (*set_test_hook)(void * session, ggml_backend_moe_hybrid_test_hook_v1_t hook, void * data);
    enum ggml_status (*preflight)(void * session, const struct ggml_cgraph * graph,
                                 const struct ggml_graph_execution_certificate * certificate, void * const * regions, uint32_t count);
};

typedef const struct ggml_backend_moe_source_core_api_v1 * (*ggml_backend_moe_source_core_v1_t)(void);

GGML_API bool ggml_backend_sched_moe_source_selected_v1(ggml_backend_sched_t sched);
// Serialized variants have separate graph plans and share actual backing and the CPU service.
GGML_API int32_t ggml_backend_sched_moe_source_clone_v1(ggml_backend_sched_t sched, ggml_backend_sched_t * output);
// Close/drain may overlap admitted compute. Reset/free/fallback require no new calls or concurrent scheduler use.
// Retire source bindings before backing replacement without resetting graph or allocator plans.
// The epoch changes even on failure; require fresh preparation before grouped compute.
GGML_API int32_t ggml_backend_sched_moe_source_retire_v1(ggml_backend_sched_t sched);
GGML_API uint64_t ggml_backend_sched_moe_source_retirement_epoch_v1(ggml_backend_sched_t sched);
GGML_API int32_t ggml_backend_sched_moe_source_reset_v1(ggml_backend_sched_t sched);
GGML_API int32_t ggml_backend_sched_moe_source_reset_graph_v1(ggml_backend_sched_t sched);
GGML_API int32_t ggml_backend_sched_moe_source_drain_v1(ggml_backend_sched_t sched);
GGML_API int32_t ggml_backend_sched_moe_source_close_v1(ggml_backend_sched_t sched);
GGML_API int32_t ggml_backend_sched_moe_source_fallback_v1(ggml_backend_sched_t sched);
GGML_API int32_t ggml_backend_sched_moe_source_free_v1(ggml_backend_sched_t * sched);

GGML_API int32_t ggml_backend_sched_moe_hybrid_configure_v1(
    ggml_backend_sched_t sched, const struct ggml_backend_moe_hybrid_config_v1 * config);
GGML_API int32_t ggml_backend_moe_hybrid_validate_buckets_v1(
    const struct ggml_backend_moe_hybrid_region_v1 * region);

GGML_API int32_t ggml_backend_moe_hybrid_get_geometry_v1(
    const struct ggml_tensor * activation, const struct ggml_tensor * ids, const struct ggml_tensor * output,
    uint32_t expert_count, struct ggml_backend_moe_hybrid_geometry_v1 * geometry);

GGML_API int32_t ggml_backend_moe_hybrid_bind_cpu_row_v1(
    const struct ggml_backend_moe_hybrid_region_v1 * region, const struct ggml_backend_moe_hybrid_binding_v1 * routes,
    uint64_t epoch, uint32_t source_row, const uint32_t * selected_routes, uint32_t count, uint32_t capacity,
    int32_t * expert_ids, uint32_t * source_rows, uint32_t * scatter,
    struct ggml_backend_moe_cpu_region_binding_v1 * binding, uint8_t * marks, size_t marks_bytes);
// The scheduler converts source graph node indices to split-local indices before backend preparation.
GGML_API int32_t ggml_backend_sched_moe_hybrid_prepare_v1(
    ggml_backend_sched_t sched, const struct ggml_backend_moe_hybrid_region_v1 * region);
GGML_API bool ggml_backend_sched_moe_hybrid_state_v1(
    ggml_backend_sched_t sched, struct ggml_backend_moe_hybrid_state_v1 * state);
// Stop admission and join the coordinator and current device dispatch. Reset/free still require caller quiescence.
GGML_API void ggml_backend_sched_moe_hybrid_quiesce_v1(ggml_backend_sched_t sched);
GGML_API bool ggml_backend_sched_moe_hybrid_set_test_hook_v1(
    ggml_backend_sched_t sched, ggml_backend_moe_hybrid_test_hook_v1_t hook, void * data);

#ifdef __cplusplus
}
#endif
