#pragma once

// Keep this header usable by host C++ consumers.
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus

#include "ggml-backend.h"

#include <array>
#include <memory>
#include <vector>

struct ggml_cuda_moe_cache;
class ggml_cuda_moe_grouped_context;
struct ggml_cuda_moe_grouped_context_test_access;

struct ggml_cuda_moe_graph_span {
    uintptr_t begin;
    uintptr_t end;
};

static inline bool ggml_cuda_moe_graph_span_bounds(
        const void * nodes,
        int32_t n_nodes,
        ggml_cuda_moe_graph_span * span) {
    if (nodes == nullptr || n_nodes <= 0 || span == nullptr) {
        return false;
    }
    const uintptr_t begin = reinterpret_cast<uintptr_t>(nodes);
    const uintptr_t count = static_cast<uintptr_t>(n_nodes);
    if (count > UINTPTR_MAX / sizeof(ggml_tensor *) || begin > UINTPTR_MAX - count * sizeof(ggml_tensor *)) {
        return false;
    }
    span->begin = begin;
    span->end = begin + count * sizeof(ggml_tensor *);
    return true;
}

static inline bool ggml_cuda_moe_graph_spans_overlap(
        const void * first_nodes,
        int32_t first_n_nodes,
        const void * second_nodes,
        int32_t second_n_nodes) {
    ggml_cuda_moe_graph_span first;
    ggml_cuda_moe_graph_span second;
    return ggml_cuda_moe_graph_span_bounds(first_nodes, first_n_nodes, &first) &&
        ggml_cuda_moe_graph_span_bounds(second_nodes, second_n_nodes, &second) &&
        first.begin < second.end && second.begin < first.end;
}

enum ggml_cuda_moe_candidate_rejection : uint32_t {
    GGML_CUDA_MOE_CANDIDATE_REJECT_NONE = 0,
    GGML_CUDA_MOE_CANDIDATE_REJECT_INVALID_ABI,
    GGML_CUDA_MOE_CANDIDATE_REJECT_INVALID_FLAGS,
    GGML_CUDA_MOE_CANDIDATE_REJECT_INVALID_COUNT,
    GGML_CUDA_MOE_CANDIDATE_REJECT_INVALID_LAYOUT,
    GGML_CUDA_MOE_CANDIDATE_REJECT_INVALID_ROLE,
    GGML_CUDA_MOE_CANDIDATE_REJECT_DUPLICATE_ROLE,
    GGML_CUDA_MOE_CANDIDATE_REJECT_DUPLICATE_TENSOR,
    GGML_CUDA_MOE_CANDIDATE_REJECT_INVALID_TENSOR,
    GGML_CUDA_MOE_CANDIDATE_REJECT_UNSUPPORTED_TYPE,
    GGML_CUDA_MOE_CANDIDATE_REJECT_UNSUPPORTED_ROLE,
    GGML_CUDA_MOE_CANDIDATE_REJECT_INACCESSIBLE_SOURCE,
    GGML_CUDA_MOE_CANDIDATE_REJECT_INVALID_BOUNDS,
    GGML_CUDA_MOE_CANDIDATE_REJECT_INCOMPATIBLE_SHAPE,
    GGML_CUDA_MOE_CANDIDATE_REJECT_OVERFLOW,
    GGML_CUDA_MOE_CANDIDATE_REJECT_ALLOCATION,
    GGML_CUDA_MOE_CANDIDATE_REJECT_GENERATION_EXHAUSTED,
};

enum ggml_cuda_moe_candidate_encoding : uint32_t {
    GGML_CUDA_MOE_CANDIDATE_ENCODING_PLAIN = 0,
    GGML_CUDA_MOE_CANDIDATE_ENCODING_NVFP4_COMPOUND,
};

enum ggml_cuda_moe_candidate_movement : uint32_t {
    GGML_CUDA_MOE_CANDIDATE_MOVEMENT_SLOT_BOUND = 0,
    GGML_CUDA_MOE_CANDIDATE_MOVEMENT_PERMANENT_CANDIDATE,
};

enum ggml_cuda_moe_candidate_index_mode : uint32_t {
    GGML_CUDA_MOE_CANDIDATE_INDEX_ORIGINAL_DIRECT     = 1u << 0,
    GGML_CUDA_MOE_CANDIDATE_INDEX_GROUP_SLOT_DIRECT   = 1u << 1,
    GGML_CUDA_MOE_CANDIDATE_INDEX_ORIGINAL_SOURCE_MAP = 1u << 2,
};

struct ggml_cuda_moe_candidate_registry_state {
    uint64_t generation = 0;
    uint64_t logical_signature = 0;
    uint64_t slot_bound_bytes = 0;
    uint64_t permanent_candidate_bytes = 0;
    uint32_t n_slots = 0;
    uint32_t n_groups = 0;
    uint32_t n_weights = 0;
    uint32_t accepted = 0;
    ggml_cuda_moe_candidate_rejection rejection = GGML_CUDA_MOE_CANDIDATE_REJECT_NONE;
};

struct ggml_cuda_moe_candidate_bank_info {
    uint64_t generation = 0;
    uint64_t byte_extent = 0;
    uint64_t expert_stride = 0;
    const ggml_tensor * tensor = nullptr;
    const void * source_data = nullptr;
    uint32_t group_index = 0;
    uint32_t role = GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID;
    uint32_t type = GGML_TYPE_COUNT;
    uint32_t source_flags = 0;
    uint32_t encoding = GGML_CUDA_MOE_CANDIDATE_ENCODING_PLAIN;
    uint32_t movement = GGML_CUDA_MOE_CANDIDATE_MOVEMENT_SLOT_BOUND;
    uint32_t index_modes = 0;
};

struct ggml_cuda_moe_candidate_group_key {
    uint64_t generation = 0;
    uint32_t group_index = 0;
};

struct ggml_cuda_moe_candidate_group_info {
    ggml_cuda_moe_candidate_group_key key;
    const ggml_tensor * down = nullptr;
    uint32_t layout = GGML_BACKEND_MOE_CANDIDATE_LAYOUT_INVALID;
    uint32_t domain = GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_INVALID;
    uint32_t semantic_group_index = UINT32_MAX;
    uint32_t flags = 0;
    uint32_t n_banks = 0;
    uint32_t n_slots = 0;
};

struct ggml_cuda_moe_candidate_probe_bank {
    const ggml_tensor * weight = nullptr;
    const ggml_tensor * ids = nullptr;
    const ggml_tensor * scale = nullptr;
    const ggml_tensor * bias = nullptr;
    uint32_t expected_role = GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID;
};

struct ggml_cuda_moe_candidate_probe_input {
    ggml_cuda_moe_candidate_probe_bank banks[2];
    uint64_t expected_generation = 0;
    uint32_t n_banks = 0;
    uint32_t exact_auxiliaries = 0;
};

struct ggml_cuda_moe_candidate_probe_result {
    ggml_cuda_moe_candidate_group_key key;
    uint32_t roles[2] = {};
};

struct ggml_cuda_moe_grouped_acquisition {
    ggml_cuda_moe_candidate_group_key candidate;
    uint64_t resource_generation = 0;
};

struct ggml_cuda_moe_grouped_transaction {
    ggml_cuda_moe_grouped_acquisition acquisition;
    uint64_t transaction_token = 0;
};

struct ggml_cuda_moe_legacy_acquisition {
    const void * owner = nullptr;
    const ggml_tensor * tensor = nullptr;
    uint64_t candidate_generation = 0;
    uint64_t authority_epoch = 0;
    uint64_t group_authority_epoch = 0;
    uint32_t group_index = UINT32_MAX;
    uint32_t role = GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID;
    uint32_t n_slots = 0;
    uint32_t registered_source = 0;
};

class ggml_cuda_moe_legacy_operation_lease {
public:
    ggml_cuda_moe_legacy_operation_lease() noexcept;
    ~ggml_cuda_moe_legacy_operation_lease();

    ggml_cuda_moe_legacy_operation_lease(ggml_cuda_moe_legacy_operation_lease && other) noexcept;
    ggml_cuda_moe_legacy_operation_lease & operator=(ggml_cuda_moe_legacy_operation_lease && other) noexcept;

    ggml_cuda_moe_legacy_operation_lease(const ggml_cuda_moe_legacy_operation_lease &) = delete;
    ggml_cuda_moe_legacy_operation_lease & operator=(const ggml_cuda_moe_legacy_operation_lease &) = delete;

    explicit operator bool() const noexcept;

private:
    friend class ggml_cuda_moe_grouped_context;

    ggml_cuda_moe_grouped_context * owner_ = nullptr;
};

class ggml_cuda_moe_legacy_cache_lease {
public:
    ggml_cuda_moe_legacy_cache_lease() noexcept;
    ~ggml_cuda_moe_legacy_cache_lease();

    ggml_cuda_moe_legacy_cache_lease(ggml_cuda_moe_legacy_cache_lease && other) noexcept;
    ggml_cuda_moe_legacy_cache_lease & operator=(ggml_cuda_moe_legacy_cache_lease && other) noexcept;

    ggml_cuda_moe_legacy_cache_lease(const ggml_cuda_moe_legacy_cache_lease &) = delete;
    ggml_cuda_moe_legacy_cache_lease & operator=(const ggml_cuda_moe_legacy_cache_lease &) = delete;

    explicit operator bool() const noexcept;
    const ggml_cuda_moe_legacy_acquisition & acquisition() const noexcept;
    ggml_cuda_moe_cache * get() const noexcept;

private:
    friend class ggml_cuda_moe_grouped_context;

    ggml_cuda_moe_grouped_context * owner_ = nullptr;
    void * record_ = nullptr;
    ggml_cuda_moe_cache * cache_ = nullptr;
    ggml_cuda_moe_legacy_acquisition acquisition_;
};

struct ggml_cuda_moe_ids_signature {
    const ggml_tensor * tensor = nullptr;
    const void * data = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    int64_t ne[GGML_MAX_DIMS] = {};
    size_t nb[GGML_MAX_DIMS] = {};
    uint32_t type = GGML_TYPE_COUNT;
};

struct ggml_cuda_moe_complete_group_key {
    ggml_cuda_moe_candidate_group_key candidate;
    ggml_cuda_moe_ids_signature ids;
    uint32_t layout = GGML_BACKEND_MOE_CANDIDATE_LAYOUT_INVALID;
    uint32_t n_banks = 0;
};

enum ggml_cuda_moe_grouped_decode_result : uint32_t {
    GGML_CUDA_MOE_GROUPED_DECODE_FALLBACK = 0,
    GGML_CUDA_MOE_GROUPED_DECODE_READY,
    GGML_CUDA_MOE_GROUPED_DECODE_ERROR,
};

struct ggml_cuda_moe_grouped_decode_bank {
    const ggml_tensor * tensor = nullptr;
    const void * data = nullptr;
    uint32_t bank_index = 0;
    uint32_t role = GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID;
    uint32_t type = GGML_TYPE_COUNT;
};

struct ggml_cuda_moe_grouped_decode_acquisition {
    ggml_cuda_moe_grouped_transaction transaction;
    const int32_t * remapped_ids = nullptr;
    ggml_cuda_moe_grouped_decode_bank banks[GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS] = {};
    uint32_t layout = GGML_BACKEND_MOE_CANDIDATE_LAYOUT_INVALID;
    uint32_t n_banks = 0;
    uint32_t n_slots = 0;
};

enum ggml_cuda_moe_group_authority : uint32_t {
    GGML_CUDA_MOE_GROUP_AUTHORITY_LEGACY = 0,
    GGML_CUDA_MOE_GROUP_AUTHORITY_GROUPED,
};

enum ggml_cuda_moe_graph_group_class : uint32_t {
    GGML_CUDA_MOE_GRAPH_GROUP_LEGACY_ONLY = 0,
    GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_ELIGIBLE,
};

class ggml_cuda_moe_group_call_lease {
public:
    ggml_cuda_moe_group_call_lease() noexcept;
    ~ggml_cuda_moe_group_call_lease();

    ggml_cuda_moe_group_call_lease(ggml_cuda_moe_group_call_lease && other) noexcept;
    ggml_cuda_moe_group_call_lease & operator=(ggml_cuda_moe_group_call_lease && other) noexcept;

    ggml_cuda_moe_group_call_lease(const ggml_cuda_moe_group_call_lease &) = delete;
    ggml_cuda_moe_group_call_lease & operator=(const ggml_cuda_moe_group_call_lease &) = delete;

    explicit operator bool() const noexcept;
    ggml_cuda_moe_group_authority authority() const noexcept;

private:
    friend class ggml_cuda_moe_grouped_context;

    ggml_cuda_moe_grouped_context * owner_ = nullptr;
    uint64_t candidate_generation_ = 0;
    uint64_t authority_epoch_ = 0;
    uint32_t group_index_ = UINT32_MAX;
    ggml_cuda_moe_group_authority authority_ = GGML_CUDA_MOE_GROUP_AUTHORITY_LEGACY;
};

enum ggml_cuda_moe_graph_group_state : uint32_t {
    GGML_CUDA_MOE_GRAPH_GROUP_WHOLE_LEGACY = 0,
    GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_ARMED,
    GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_ACTIVE,
    GGML_CUDA_MOE_GRAPH_GROUP_FINISHED,
};

struct ggml_cuda_moe_graph_capability_witness {
    const ggml_tensor * tensor = nullptr;
    const void * source_data = nullptr;
    uint64_t byte_extent = 0;
    uint64_t expert_stride = 0;
    int64_t n_tokens = 0;
    int64_t n_experts = 0;
    size_t smpbo = 0;
    int32_t device = -1;
    int32_t cc = 0;
    int32_t warp_size = 0;
    uint32_t role = GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID;
    uint32_t source_type = GGML_TYPE_COUNT;
    uint32_t source_flags = 0;
    uint32_t input_type = GGML_TYPE_COUNT;
    uint32_t output_type = GGML_TYPE_COUNT;
    uint32_t phase = 0;
    uint32_t mapping = 0;
    uint32_t consumer = 0;
    uint32_t reason = 0;
};

struct ggml_cuda_moe_graph_group_dispatch {
    ggml_cuda_moe_complete_group_key key;
    ggml_cuda_moe_group_call_lease authority;
    ggml_cuda_moe_grouped_transaction transaction;
    const ggml_cuda_moe_graph_capability_witness * capabilities = nullptr;
    const int32_t * remapped_ids = nullptr;
    const void * bank_data[GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS] = {};
    cudaStream_t stream = nullptr;
    uint32_t classification = GGML_CUDA_MOE_GRAPH_GROUP_LEGACY_ONLY;
    uint32_t state = GGML_CUDA_MOE_GRAPH_GROUP_WHOLE_LEGACY;
    uint32_t n_slots = 0;
};

using ggml_cuda_moe_graph_stream_resolver = cudaStream_t (*)(void * data, const ggml_tensor * node);

struct ggml_cuda_moe_graph_binding {
    ggml_cuda_moe_complete_group_key key;
    uint32_t role = GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID;
    uint32_t bank_index = 0;
    uint32_t slot_index = UINT32_MAX;
};

enum ggml_cuda_moe_graph_prepare_result : uint32_t {
    GGML_CUDA_MOE_GRAPH_PREPARE_UNAVAILABLE = 0,
    GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED,
    GGML_CUDA_MOE_GRAPH_PREPARE_REUSED,
};

struct ggml_cuda_moe_grouped_debug_telemetry {
    uint64_t registered = 0;
    uint64_t covered = 0;
    uint64_t plan_calls = 0;
    uint64_t plan_compiles = 0;
    uint64_t plan_reuses = 0;
    uint64_t calls = 0;
    uint64_t ready = 0;
    uint64_t ready_min = 0;
    uint64_t ready_max = 0;
    uint64_t completed = 0;
    uint64_t completed_min = 0;
    uint64_t completed_max = 0;
    uint64_t admitted_banks = 0;
    uint64_t fallback = 0;
    uint64_t rollback = 0;
    uint64_t prepare_error = 0;
    uint64_t finish_error = 0;
    uint64_t h2d_banks = 0;
    uint64_t h2d_bytes = 0;
};

enum ggml_cuda_moe_graph_property_hint : uint32_t {
    GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN = 0,
    GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED,
    GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED,
};

enum ggml_cuda_moe_graph_coverage_reason : uint32_t {
    GGML_CUDA_MOE_GRAPH_COVERAGE_REGISTERED = 0,
    GGML_CUDA_MOE_GRAPH_COVERAGE_REVERSE_MAP_MISS,
    GGML_CUDA_MOE_GRAPH_COVERAGE_SOURCE_CHANGED,
    GGML_CUDA_MOE_GRAPH_COVERAGE_INVALID_REVERSE_MAP,
    GGML_CUDA_MOE_GRAPH_COVERAGE_DORMANT_LAYOUT,
    GGML_CUDA_MOE_GRAPH_COVERAGE_ACTIVE_LORA,
    GGML_CUDA_MOE_GRAPH_COVERAGE_TENSOR_OVERRIDE,
    GGML_CUDA_MOE_GRAPH_COVERAGE_NON_ROUTED_BASE,
    GGML_CUDA_MOE_GRAPH_COVERAGE_EXCLUDED,
    GGML_CUDA_MOE_GRAPH_COVERAGE_UNCLASSIFIED,
    GGML_CUDA_MOE_GRAPH_COVERAGE_UNSUPPORTED_DESCRIPTOR,
    GGML_CUDA_MOE_GRAPH_COVERAGE_INCOMPLETE,
    GGML_CUDA_MOE_GRAPH_COVERAGE_REASON_COUNT,
};

struct ggml_cuda_moe_graph_coverage_diagnostics {
    uint32_t cached_mmid = 0;
    uint32_t manifest_version = 0;
    uint32_t counts[GGML_CUDA_MOE_GRAPH_COVERAGE_REASON_COUNT] = {};
    const ggml_tensor * first_source[GGML_CUDA_MOE_GRAPH_COVERAGE_REASON_COUNT] = {};
    uint32_t first_node_index[GGML_CUDA_MOE_GRAPH_COVERAGE_REASON_COUNT] = {};
    uint32_t first_group_index[GGML_CUDA_MOE_GRAPH_COVERAGE_REASON_COUNT] = {};
    uint32_t first_bank_index[GGML_CUDA_MOE_GRAPH_COVERAGE_REASON_COUNT] = {};
    uint32_t first_role[GGML_CUDA_MOE_GRAPH_COVERAGE_REASON_COUNT] = {};
    uint32_t first_status[GGML_CUDA_MOE_GRAPH_COVERAGE_REASON_COUNT] = {};
    uint32_t first_layout[GGML_CUDA_MOE_GRAPH_COVERAGE_REASON_COUNT] = {};
    uint32_t first_domain[GGML_CUDA_MOE_GRAPH_COVERAGE_REASON_COUNT] = {};
    uint32_t first_flags[GGML_CUDA_MOE_GRAPH_COVERAGE_REASON_COUNT] = {};
    uint32_t first_group_flags[GGML_CUDA_MOE_GRAPH_COVERAGE_REASON_COUNT] = {};
    uint32_t first_rejection[GGML_CUDA_MOE_GRAPH_COVERAGE_REASON_COUNT] = {};
};

class ggml_cuda_moe_graph_plan {
public:
    ggml_cuda_moe_graph_plan();

    uint32_t size() const;
    uint64_t registry_generation() const;
    uint64_t graph_uid() const;
    int32_t graph_node_count() const;
    const ggml_cuda_moe_graph_coverage_diagnostics & coverage_diagnostics() const;

private:
    static constexpr uint32_t MAX_NODE_BINDINGS = GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS * 3;
    static constexpr uint32_t NODE_TABLE_SIZE = 4096;
    static constexpr uint32_t MAX_GROUP_READERS = 4;
    static constexpr uint32_t MAX_READER_CONSUMERS = 4;

    enum group_reason : uint32_t {
        GROUP_REASON_ELIGIBLE = 0,
        GROUP_REASON_DESCRIPTOR,
        GROUP_REASON_SOURCE,
        GROUP_REASON_GEOMETRY,
        GROUP_REASON_CAPABILITY,
        GROUP_REASON_ROUTE,
        GROUP_REASON_DUPLICATE_ROLE,
        GROUP_REASON_MIXED_IDS,
        GROUP_REASON_AUXILIARY,
        GROUP_REASON_EXTERNAL_CONSUMER,
        GROUP_REASON_MISSING_ROLE,
        GROUP_REASON_UNPROVEN,
    };

    struct consumer_witness {
        const ggml_tensor * node;
        const ggml_tensor * src[3];
        uint32_t node_index;
        uint32_t src_index;
        uint32_t op;
    };

    struct reader_witness {
        ggml_cuda_moe_ids_signature output;
        ggml_cuda_moe_ids_signature activation;
        ggml_cuda_moe_ids_signature ids;
        consumer_witness consumers[MAX_READER_CONSUMERS];
        const ggml_tensor * auxiliary_nodes[3];
        int32_t use_count;
        uint32_t node_index;
        uint32_t role;
        uint32_t bank_index;
        uint32_t n_consumers;
        uint32_t auxiliary_node_indices[3];
        uint32_t auxiliary_bank_index;
        uint32_t auxiliary_consumer_index;
        uint32_t n_auxiliary_nodes;
    };

    struct use_witness {
        const ggml_tensor * tensor;
        int32_t use_count;
        uint32_t present;
    };

    struct group_observation {
        ggml_cuda_moe_ids_signature ids;
        ggml_cuda_moe_ids_signature route_root;
        ggml_cuda_moe_ids_signature route_source;
        reader_witness readers[MAX_GROUP_READERS];
        uint32_t bank_readers[GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS];
        const ggml_tensor * nodes[4];
        uint32_t node_indices[4];
        uint32_t bank_indices[4];
        ggml_cuda_moe_graph_capability_witness capabilities[4];
        uint32_t route_root_node_index;
        uint32_t route_ids_node_index;
        uint32_t required_roles;
        uint32_t seen_roles;
        uint32_t n_banks;
        uint32_t n_readers;
        const ggml_tensor * authority_node;
        uint32_t authority_node_index;
        bool observed;
        bool descriptor_supported;
        bool source_invalid;
        bool geometry_invalid;
        bool capability_invalid;
        bool route_invalid;
        bool duplicate_role;
        bool mixed_ids;
        bool auxiliary;
        bool external_consumer;
        bool unproven;
        bool has_ids;
    };

    struct group_record {
        ggml_cuda_moe_candidate_group_key candidate;
        ggml_cuda_moe_ids_signature ids;
        ggml_cuda_moe_ids_signature ids_root;
        ggml_cuda_moe_ids_signature ids_source;
        uint32_t layout;
        uint32_t n_banks;
        uint32_t ids_root_node_index;
        uint32_t ids_node_index;
        uint32_t classification;
        const ggml_tensor * authority_node;
        uint32_t authority_node_index;
        const ggml_tensor * nodes[4];
        uint32_t node_indices[4];
        uint32_t bank_indices[4];
        ggml_cuda_moe_graph_capability_witness capabilities[4];
        reader_witness readers[MAX_GROUP_READERS];
        use_witness bank_uses[GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS];
        uint32_t reason;
        uint32_t n_readers;
        uint32_t witness_reusable;
    };

    struct node_entry {
        const ggml_tensor * node;
        uint32_t group_record;
        uint32_t role;
        uint32_t bank_index;
        uint32_t slot_index;
    };

    friend class ggml_cuda_moe_grouped_context;
    friend class ggml_cuda_moe_graph_execution;
    friend struct ggml_cuda_moe_grouped_context_test_access;

    void reset();
    bool insert(const ggml_tensor * node, uint32_t group_record, uint32_t role, uint32_t bank_index, uint32_t slot_index);
    const node_entry * find(const ggml_tensor * node) const;
    static_assert(sizeof(reader_witness) <= 640, "reader witness is too large");
    static_assert(sizeof(group_record) <= 4096, "group record is too large");
    static_assert(sizeof(group_observation) <= 4096, "group observation is too large");

    std::vector<group_record> groups_;
    std::array<node_entry, NODE_TABLE_SIZE> nodes_;
    const void * owner_;
    const void * graph_key_;
    const void * coverage_nodes_;
    uint64_t registry_generation_;
    uint64_t graph_uid_;
    uint64_t coverage_epoch_;
    int32_t graph_node_count_;
    uint32_t n_groups_;
    uint32_t n_nodes_;
    ggml_cuda_moe_graph_coverage_diagnostics coverage_diagnostics_;
    bool initialized_;
    bool unknown_reusable_;
};

static_assert(sizeof(ggml_cuda_moe_graph_plan) <= 128 * 1024, "graph plan is too large");

class ggml_cuda_moe_graph_execution {
public:
    ggml_cuda_moe_graph_execution();
    ~ggml_cuda_moe_graph_execution();

    ggml_cuda_moe_graph_execution(const ggml_cuda_moe_graph_execution &) = delete;
    ggml_cuda_moe_graph_execution & operator=(const ggml_cuda_moe_graph_execution &) = delete;

    bool find(const ggml_tensor * node, ggml_cuda_moe_graph_binding * binding) const;
    ggml_cuda_moe_graph_group_dispatch * find_group(const ggml_tensor * node, ggml_cuda_moe_graph_binding * binding);
    const ggml_cuda_moe_group_call_lease * find_authority(const ggml_tensor * node) const;
    bool resolve_streams(ggml_cuda_moe_graph_stream_resolver resolver, void * data);
    bool has_stream_grouped_candidate() const;
    uint32_t size() const;

private:
    friend class ggml_cuda_moe_grouped_context;

    void reset();
    void retain(const std::shared_ptr<const ggml_cuda_moe_graph_plan> & plan);

    const ggml_cuda_moe_graph_plan * plan_;
    std::shared_ptr<const ggml_cuda_moe_graph_plan> plan_lease_;
    std::array<ggml_cuda_moe_graph_group_dispatch, GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS> groups_;
    ggml_cuda_moe_grouped_context * owner_;
    uint32_t n_groups_;
    bool dispatch_active_;
};

struct ggml_cuda_moe_grouped_resource_info {
    ggml_cuda_moe_grouped_acquisition acquisition;
    const ggml_tensor * down = nullptr;
    uint32_t layout = GGML_BACKEND_MOE_CANDIDATE_LAYOUT_INVALID;
    uint32_t n_slots = 0;
    uint32_t n_banks = 0;
    uint32_t transaction_active = 0;
};

struct ggml_cuda_moe_grouped_bank_descriptor {
    const ggml_tensor * tensor = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_backend_buffer_type_t buft = nullptr;
    const void * source_data = nullptr;
    const void * buffer_base = nullptr;
    uint64_t buffer_size = 0;
    uint64_t data_offset = 0;
    uint64_t byte_extent = 0;
    uint64_t expert_stride = 0;
    uint64_t alignment = 0;
    int64_t ne[GGML_MAX_DIMS] = {};
    size_t nb[GGML_MAX_DIMS] = {};
    uint32_t role = GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID;
    uint32_t type = GGML_TYPE_COUNT;
    uint32_t encoding = GGML_CUDA_MOE_CANDIDATE_ENCODING_PLAIN;
    uint32_t movement = GGML_CUDA_MOE_CANDIDATE_MOVEMENT_SLOT_BOUND;
    uint32_t index_modes = 0;
};

struct ggml_cuda_moe_grouped_context_test_access;

class ggml_cuda_moe_grouped_context {
public:
    explicit ggml_cuda_moe_grouped_context(ggml_backend_dev_t owner, int device = -1);
    ~ggml_cuda_moe_grouped_context();

    ggml_cuda_moe_grouped_context(const ggml_cuda_moe_grouped_context &) = delete;
    ggml_cuda_moe_grouped_context & operator=(const ggml_cuda_moe_grouped_context &) = delete;

    int32_t replace(const ggml_backend_moe_candidate_snapshot_v1 * snapshot);
    int32_t replace(const ggml_backend_moe_candidate_snapshot_v2 * snapshot);
    ggml_cuda_moe_candidate_registry_state state() const;
    bool find_down_group(const ggml_tensor * tensor, uint32_t * group_index) const;
    bool find_down_group_key(const ggml_tensor * tensor, ggml_cuda_moe_candidate_group_key * key) const;
    bool find_weight(const ggml_tensor * tensor, ggml_cuda_moe_candidate_bank_info * info) const;
    bool get_group(const ggml_cuda_moe_candidate_group_key & key, ggml_cuda_moe_candidate_group_info * info) const;
    bool get_bank(const ggml_cuda_moe_candidate_group_key & key, uint32_t role, ggml_cuda_moe_candidate_bank_info * info) const;
    bool probe(const ggml_cuda_moe_candidate_probe_input & input, ggml_cuda_moe_candidate_probe_result * result) const;
    bool acquire_group_resources(const ggml_cuda_moe_candidate_group_key & key, ggml_cuda_moe_grouped_acquisition * acquisition);
    bool begin_group_transaction(const ggml_cuda_moe_grouped_acquisition & acquisition, ggml_cuda_moe_grouped_transaction * transaction);
    bool end_group_transaction(const ggml_cuda_moe_grouped_transaction & transaction);
    bool get_group_resources(const ggml_cuda_moe_grouped_acquisition & acquisition, ggml_cuda_moe_grouped_resource_info * info) const;
    bool get_group_resource_bank(const ggml_cuda_moe_grouped_transaction & transaction, uint32_t bank_index, ggml_cuda_moe_grouped_bank_descriptor * descriptor) const;
    ggml_cuda_moe_legacy_operation_lease begin_legacy_operation();
    ggml_cuda_moe_legacy_cache_lease acquire_legacy_cache(
            const ggml_tensor * tensor,
            const ggml_cuda_moe_legacy_acquisition * expected = nullptr,
            const ggml_cuda_moe_group_call_lease * authority = nullptr);
    void prefetch_legacy_siblings(
            const ggml_cuda_moe_legacy_cache_lease & source,
            const int32_t * expert_ids,
            int n_expert_ids,
            bool use_l2,
            bool is_decode);
    void record_legacy_op(
            bool is_decode,
            bool staged,
            bool overflow,
            uint64_t unique_experts,
            uint64_t ids_bytes,
            uint64_t ids_d2h_time_us,
            uint64_t ids_d2h_sync_count,
            uint64_t acquire_time_us,
            uint64_t remap_time_us,
            uint64_t copy_wait_event_count,
            uint64_t copy_wait_event_time_us,
            uint64_t total_time_us,
            bool ids_cache_hit);
    static ggml_cuda_moe_grouped_debug_telemetry log_and_reset_legacy_stats();
    uint64_t certify_graph_coverage(const ggml_cgraph * cgraph);
    bool recover_graph_coverage(const ggml_cgraph * cgraph, uint64_t * coverage_epoch) const;
    void compile_graph_plan(
            const ggml_cgraph * cgraph,
            uint64_t graph_uid,
            ggml_cuda_moe_graph_plan * plan,
            ggml_cuda_moe_graph_execution * execution,
            uint64_t coverage_epoch = 0,
            const void * coverage_nodes = nullptr) const;
    bool bind_graph_plan(
            const ggml_cgraph * cgraph,
            uint64_t graph_uid,
            ggml_cuda_moe_graph_property_hint property_hint,
            const ggml_cuda_moe_graph_plan & plan,
            ggml_cuda_moe_graph_execution * execution,
            uint64_t coverage_epoch = 0,
            const void * coverage_nodes = nullptr) const;
    ggml_cuda_moe_graph_prepare_result prepare_graph_execution(
            const ggml_cgraph * cgraph,
            uint64_t graph_uid,
            ggml_cuda_moe_graph_property_hint property_hint,
            std::shared_ptr<ggml_cuda_moe_graph_plan> * plan,
            ggml_cuda_moe_graph_execution * execution,
            uint64_t coverage_epoch = 0,
            const void * coverage_nodes = nullptr) const;
    bool begin_graph_dispatch(ggml_cuda_moe_graph_execution * execution, bool grouped_enabled);
    ggml_cuda_moe_grouped_decode_result prepare_graph_group(
            ggml_cuda_moe_graph_group_dispatch * group,
            const ggml_cuda_moe_graph_binding & binding,
            const ggml_tensor * node,
            cudaStream_t stream);
    bool finish_graph_dispatch(ggml_cuda_moe_graph_execution * execution);
    ggml_cuda_moe_grouped_decode_result prepare_decode(
            const ggml_cuda_moe_complete_group_key & key,
            cudaStream_t compute_stream,
            ggml_cuda_moe_grouped_decode_acquisition * acquisition,
            const ggml_cuda_moe_group_call_lease * authority = nullptr);
    bool finish_decode(const ggml_cuda_moe_grouped_decode_acquisition & acquisition, cudaStream_t compute_stream);
    void shutdown();

private:
    friend struct ggml_cuda_moe_grouped_context_test_access;
    friend class ggml_cuda_moe_group_call_lease;
    friend class ggml_cuda_moe_legacy_operation_lease;
    friend class ggml_cuda_moe_legacy_cache_lease;

    bool set_clock_bound_for_test(const ggml_cuda_moe_grouped_acquisition & acquisition, uint64_t clock_bound);
    bool admission_closed_for_test() const;
    bool has_device_resource_for_test(const ggml_cuda_moe_candidate_group_key & key) const;
    bool get_clock_bound_for_test(const ggml_cuda_moe_candidate_group_key & key, uint64_t * clock_bound) const;
    uint64_t legacy_op_count_for_test(bool is_decode) const;
    ggml_cuda_moe_grouped_debug_telemetry take_grouped_debug_telemetry_for_test();
    bool graph_group_witness_matches(const ggml_cgraph * cgraph, const ggml_cuda_moe_graph_plan::group_record & record) const;
    bool rollback_group_to_legacy(ggml_cuda_moe_graph_group_dispatch & group);
    void end_group_call(ggml_cuda_moe_group_call_lease & lease) noexcept;
    void end_legacy_operation(ggml_cuda_moe_legacy_operation_lease & lease) noexcept;
    void release_legacy_cache(ggml_cuda_moe_legacy_cache_lease & lease) noexcept;

    struct impl;
    std::unique_ptr<impl> impl_;
};

ggml_cuda_moe_grouped_context * ggml_cuda_moe_grouped_context_for_test(ggml_backend_t backend);

#endif

// MoE expert cache: keeps a fixed-size GPU slot pool of expert weight slabs
// while cold experts live in CPU pinned memory. On routing miss the slab is
// async-copied H2D and an LRU slot is evicted to make room.
//
// API design:
//   - The cache is keyed by the CPU pointer to an expert's weight slab. That
//     pointer is the unique identifier of the slab (different layers/matrix
//     types have different pointers because they live at different offsets in
//     the cached buffer). This avoids any "global expert id" namespacing.
//   - Slots are uniformly sized within a cache pool, but slot_size grows on
//     demand if a later op has experts larger than the current slot size. On
//     grow, the existing pool is freed, a new pool is allocated with the
//     larger slot size, and all bookkeeping is reset (existing cached entries
//     are evicted). This converges quickly: after the first few ops the cache
//     stabilizes at slot_size = max expert stride across the model.
//   - acquire() takes the actual byte count to copy on miss, which can be
//     less than slot_size_bytes. Smaller experts simply leave padding inside
//     their slot.
//
// See ../../DESIGN.md for the full design.

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_cuda_moe_cache;

int32_t ggml_backend_cuda_moe_candidate_replace_v1(
    ggml_backend_t backend,
    const struct ggml_backend_moe_candidate_snapshot_v1 * snapshot);

int32_t ggml_backend_cuda_moe_candidate_replace_v2(
    ggml_backend_t backend,
    const struct ggml_backend_moe_candidate_snapshot_v2 * snapshot);

// Create a cache for one device.
//   slot_size_bytes : size of one expert weight slab (uniform across slots)
//   n_slots         : how many slabs the GPU pool can hold simultaneously
struct ggml_cuda_moe_cache * ggml_cuda_moe_cache_init(
    int    device,
    size_t slot_size_bytes,
    int    n_slots,
    bool   source_is_mmap,
    size_t l2_budget_bytes,
    int    l2_target_slots);

void ggml_cuda_moe_cache_free(struct ggml_cuda_moe_cache * cache);

// Returns the slot index that holds `host_src`'s contents after this call.
// On hit, just bumps LRU; on miss, picks the LRU slot, evicts it, and issues
// an async H2D copy of `byte_count` bytes from `host_src` on `copy_stream`.
// The caller must synchronize the compute stream against `copy_stream` before
// reading the slab.
//
// `byte_count` MUST be <= ggml_cuda_moe_cache_slot_size_bytes(cache); the
// caller is expected to ensure this via grow_pool() if the current slot_size
// is too small. A pinned slot cannot be evicted until release_slots().
// Returns -1 on bad args or a CUDA failure.
int ggml_cuda_moe_cache_acquire(
    struct ggml_cuda_moe_cache * cache,
    const void * host_src,
    size_t       byte_count,
    cudaStream_t copy_stream,
    bool         use_l2,
    bool         is_decode,
    bool         is_prefetch,
    bool         pin);

void ggml_cuda_moe_cache_release_slots(
    struct ggml_cuda_moe_cache * cache,
    const int * slot_ids,
    int n_slot_ids);

// Copy host slabs into consecutive staging slots, using resident cache slots when available.
// Returns false without enqueuing work if the arguments are invalid.
bool ggml_cuda_moe_cache_copy_to_staging(
    struct ggml_cuda_moe_cache * cache,
    const void * const * host_srcs,
    int                  n_host_srcs,
    size_t               byte_count,
    void *               dst,
    cudaStream_t         compute_stream);

bool ggml_cuda_moe_cache_prepare_split_staging(
    struct ggml_cuda_moe_cache * cache,
    const void * const * host_srcs,
    int                  n_host_srcs,
    size_t               byte_count,
    int                  min_resident,
    int *                slot_ids,
    int32_t *            source_wait_class,
    int *                out_n_resident,
    void *               miss_dst,
    uint32_t *           stage_ready,
    int                  stage_ready_capacity,
    int *                out_n_wait_classes,
    cudaStream_t         compute_stream);

bool ggml_cuda_moe_cache_can_overlap_staging(
    const struct ggml_cuda_moe_cache * cache);

bool ggml_cuda_moe_cache_finish_split_staging(
    struct ggml_cuda_moe_cache * cache,
    cudaStream_t         compute_stream);

bool ggml_cuda_moe_cache_release_split_slots(
    struct ggml_cuda_moe_cache * cache,
    const int *          slot_ids,
    int                  n_slot_ids,
    cudaStream_t         compute_stream);

void ggml_cuda_moe_record_op_stats(
    bool     is_decode,
    bool     staged,
    bool     overflow,
    uint64_t unique_experts,
    uint64_t ids_bytes,
    uint64_t ids_d2h_time_us,
    uint64_t ids_d2h_sync_count,
    uint64_t acquire_time_us,
    uint64_t remap_time_us,
    uint64_t copy_wait_event_count,
    uint64_t copy_wait_event_time_us,
    uint64_t total_time_us,
    bool     ids_cache_hit);

// Ensure the cache's slot size is at least `min_slot_size_bytes`. If the
// current slot size already covers it, no-op. Otherwise the existing pool is
// freed, a new pool is allocated with slot_size = min_slot_size_bytes, and
// all bookkeeping is reset (every previously-cached slab becomes a miss next
// access). Returns true on success, false if the new allocation fails (in
// which case the cache is left untouched and acquire() will keep working with
// the previous slot size).
bool ggml_cuda_moe_cache_grow_pool(
    struct ggml_cuda_moe_cache * cache,
    size_t min_slot_size_bytes);

// Returns device pointer to slot `slot_id`'s slab.
void * ggml_cuda_moe_cache_slot_ptr(
    struct ggml_cuda_moe_cache * cache,
    int slot_id);

// Inspectors.
size_t       ggml_cuda_moe_cache_slot_size_bytes(const struct ggml_cuda_moe_cache * cache);
int          ggml_cuda_moe_cache_n_slots(const struct ggml_cuda_moe_cache * cache);
cudaStream_t ggml_cuda_moe_cache_copy_stream(const struct ggml_cuda_moe_cache * cache);

bool ggml_cuda_moe_cache_mark_used(
    struct ggml_cuda_moe_cache * cache,
    cudaStream_t compute_stream);

// Telemetry.
void ggml_cuda_moe_cache_stats(
    const struct ggml_cuda_moe_cache * cache,
    uint64_t * out_hits,
    uint64_t * out_misses,
    uint64_t * out_evictions);

void ggml_cuda_moe_cache_reset_stats(struct ggml_cuda_moe_cache * cache);

// Deprecated: keys solely by slot_size_bytes. Kept for compat; returns nullptr.
struct ggml_cuda_moe_cache * ggml_cuda_moe_cache_get_or_create(
    int    device,
    size_t slot_size_bytes,
    int    n_slots);

// Global teardown: free all per-device caches. Safe to call repeatedly.
void ggml_cuda_moe_cache_free_all(void);

#ifdef __cplusplus
}
#endif
