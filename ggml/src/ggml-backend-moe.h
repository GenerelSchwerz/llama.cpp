#pragma once

#include "ggml-backend.h"

// Private fork protocol shared by llama and backend implementations.
// Resolve these procedures through the backend registry; this header is not installed.
#ifdef __cplusplus
extern "C" {
#endif

#define GGML_BACKEND_MOE_CACHE_BUFFER_TYPE_PROC_NAME "ggml_backend_moe_cache_buffer_type"
#define GGML_BACKEND_MOE_CACHE_BOUNDED_BUFFER_TYPE_PROC_NAME "ggml_backend_moe_cache_bounded_buffer_type"
#define GGML_BACKEND_MOE_CACHE_FREE_BUFFER_TYPE_PROC_NAME "ggml_backend_moe_cache_free_buffer_type"
#define GGML_BACKEND_MOE_CACHE_CONFIGURE_SOURCES_PROC_NAME "ggml_backend_moe_cache_configure_sources"
#define GGML_BACKEND_MOE_CACHE_IS_BUFFER_TYPE_PROC_NAME "ggml_backend_moe_cache_is_buffer_type"
#define GGML_BACKEND_MOE_CACHE_BUFFER_FROM_HOST_PTR_PROC_NAME "ggml_backend_moe_cache_buffer_from_host_ptr"
#define GGML_BACKEND_MOE_CACHE_SET_DEBUG_PROC_NAME "ggml_backend_moe_cache_set_debug"
#define GGML_BACKEND_MOE_CACHE_LOG_AND_RESET_STATS_PROC_NAME "ggml_backend_moe_cache_log_and_reset_stats"

typedef ggml_backend_buffer_type_t (*ggml_backend_moe_cache_buffer_type_t)(void);
typedef ggml_backend_buffer_type_t (*ggml_backend_moe_cache_bounded_buffer_type_t)(size_t bytes);
typedef void (*ggml_backend_moe_cache_free_buffer_type_t)(ggml_backend_buffer_type_t buft);
typedef bool (*ggml_backend_moe_cache_is_buffer_type_t)(ggml_backend_buffer_type_t buft);
typedef ggml_backend_buffer_t (*ggml_backend_moe_cache_buffer_from_host_ptr_t)(ggml_backend_buffer_type_t buft, void * ptr, size_t size);
// Experimental logging remains process-wide, including retired backend counters.
typedef void (*ggml_backend_moe_cache_set_debug_t)(bool enabled);
typedef void (*ggml_backend_moe_cache_log_and_reset_stats_t)(void);

#define GGML_BACKEND_MOE_CANDIDATE_REPLACE_V1_PROC_NAME "ggml_backend_moe_candidate_replace_v1"
#define GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V1_MAGIC 0x4d4f4531u
#define GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V1_VERSION 1u
#define GGML_BACKEND_MOE_CANDIDATE_REPLACE_V2_PROC_NAME "ggml_backend_moe_candidate_replace_v2"
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
typedef bool (*ggml_backend_moe_cache_configure_sources_t)(ggml_backend_buffer_type_t buft, const struct ggml_backend_moe_candidate_snapshot_v2 * snapshot);
typedef bool (*ggml_backend_required_grouped_execution_supported_t)(ggml_backend_t backend);

#ifdef __cplusplus
}
#endif
