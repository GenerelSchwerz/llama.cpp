#pragma once

// Keep this header usable by host C++ consumers.
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus

#include "ggml-backend.h"

#include <array>
#include <memory>

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

struct ggml_cuda_moe_graph_binding {
    ggml_cuda_moe_complete_group_key key;
    uint32_t role = GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID;
    uint32_t bank_index = 0;
};

enum ggml_cuda_moe_graph_prepare_result : uint32_t {
    GGML_CUDA_MOE_GRAPH_PREPARE_UNAVAILABLE = 0,
    GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED,
    GGML_CUDA_MOE_GRAPH_PREPARE_REUSED,
};

class ggml_cuda_moe_graph_plan {
public:
    ggml_cuda_moe_graph_plan();

    uint32_t size() const;
    uint64_t registry_generation() const;
    uint64_t graph_uid() const;
    int32_t graph_node_count() const;

private:
    static constexpr uint32_t MAX_NODE_BINDINGS = GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS * 3;
    static constexpr uint32_t NODE_TABLE_SIZE = 4096;

    struct group_record {
        ggml_cuda_moe_candidate_group_key candidate;
        uint32_t layout;
        uint32_t n_banks;
        const ggml_tensor * nodes[4];
        uint32_t node_indices[4];
        uint32_t bank_indices[4];
    };

    struct node_entry {
        const ggml_tensor * node;
        uint32_t group_record;
        uint32_t role;
        uint32_t bank_index;
    };

    friend class ggml_cuda_moe_grouped_context;
    friend class ggml_cuda_moe_graph_execution;

    void reset();
    bool insert(const ggml_tensor * node, uint32_t group_record, uint32_t role, uint32_t bank_index);
    const node_entry * find(const ggml_tensor * node) const;

    std::array<group_record, GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS> groups_;
    std::array<node_entry, NODE_TABLE_SIZE> nodes_;
    const void * owner_;
    const void * graph_key_;
    uint64_t registry_generation_;
    uint64_t graph_uid_;
    int32_t graph_node_count_;
    uint32_t n_groups_;
    uint32_t n_nodes_;
    bool initialized_;
};

class ggml_cuda_moe_graph_execution {
public:
    ggml_cuda_moe_graph_execution();

    bool find(const ggml_tensor * node, ggml_cuda_moe_graph_binding * binding) const;
    uint32_t size() const;

private:
    friend class ggml_cuda_moe_grouped_context;

    void reset();
    void retain(const std::shared_ptr<const ggml_cuda_moe_graph_plan> & plan);

    const ggml_cuda_moe_graph_plan * plan_;
    std::shared_ptr<const ggml_cuda_moe_graph_plan> plan_lease_;
    std::array<ggml_cuda_moe_complete_group_key, GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS> groups_;
    uint32_t n_groups_;
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

class ggml_cuda_moe_grouped_context {
public:
    explicit ggml_cuda_moe_grouped_context(ggml_backend_dev_t owner);
    ~ggml_cuda_moe_grouped_context();

    ggml_cuda_moe_grouped_context(const ggml_cuda_moe_grouped_context &) = delete;
    ggml_cuda_moe_grouped_context & operator=(const ggml_cuda_moe_grouped_context &) = delete;

    int32_t replace(const ggml_backend_moe_candidate_snapshot_v1 * snapshot);
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
    void compile_graph_plan(
            const ggml_cgraph * cgraph,
            uint64_t graph_uid,
            ggml_cuda_moe_graph_plan * plan,
            ggml_cuda_moe_graph_execution * execution) const;
    bool bind_graph_plan(
            const ggml_cgraph * cgraph,
            uint64_t graph_uid,
            bool node_properties_unchanged,
            const ggml_cuda_moe_graph_plan & plan,
            ggml_cuda_moe_graph_execution * execution) const;
    ggml_cuda_moe_graph_prepare_result prepare_graph_execution(
            const ggml_cgraph * cgraph,
            uint64_t graph_uid,
            bool node_properties_unchanged,
            std::shared_ptr<ggml_cuda_moe_graph_plan> * plan,
            ggml_cuda_moe_graph_execution * execution) const;
    void shutdown();

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

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

// Per-tensor cache: returns the existing cache for this (device, tensor_data)
// pair, or creates a fresh one on first call. Each MoE expert tensor has its
// own pool of N slots dedicated to its experts only -- no cross-tensor or
// cross-layer slot contention. tensor_name_for_log is used for the
// "load_tensors:" log line on first creation; pass src0->name.
struct ggml_cuda_moe_cache * ggml_cuda_moe_cache_get_or_create_for_tensor(
    int          device,
    const void * tensor_data,
    size_t       slot_size_bytes,
    int          n_slots,
    int64_t      n_experts,
    const char * tensor_name_for_log);

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
