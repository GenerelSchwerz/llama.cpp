#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>

enum ggml_cuda_mmid_source_flag : uint32_t {
    GGML_CUDA_MMID_SOURCE_ADVERTISED = 1u << 0,
    GGML_CUDA_MMID_SOURCE_MMVQ       = 1u << 1,
    GGML_CUDA_MMID_SOURCE_MMQ        = 1u << 2,
    GGML_CUDA_MMID_SOURCE_MAPPED_MMQ = 1u << 3,
    GGML_CUDA_MMID_SOURCE_SCALAR     = 1u << 4,
    GGML_CUDA_MMID_SOURCE_GENERIC    = 1u << 5,
};

enum ggml_cuda_mmid_phase : uint32_t {
    GGML_CUDA_MMID_PHASE_DECODE = 0,
    GGML_CUDA_MMID_PHASE_PREFILL,
};

enum ggml_cuda_mmid_mapping : uint32_t {
    GGML_CUDA_MMID_MAPPING_DIRECT = 0,
    GGML_CUDA_MMID_MAPPING_SOURCE_MAP,
};

enum ggml_cuda_mmid_consumer : uint32_t {
    GGML_CUDA_MMID_CONSUMER_UNSUPPORTED = 0,
    GGML_CUDA_MMID_CONSUMER_MMVQ,
    GGML_CUDA_MMID_CONSUMER_MMVF,
    GGML_CUDA_MMID_CONSUMER_MMQ,
    GGML_CUDA_MMID_CONSUMER_MMF,
    GGML_CUDA_MMID_CONSUMER_GENERIC,
};

enum ggml_cuda_mmid_capability_reason : uint32_t {
    GGML_CUDA_MMID_CAPABILITY_OK = 0,
    GGML_CUDA_MMID_CAPABILITY_UNADVERTISED_SOURCE,
    GGML_CUDA_MMID_CAPABILITY_UNSUPPORTED_CONSUMER,
    GGML_CUDA_MMID_CAPABILITY_UNSUPPORTED_MAPPING,
    GGML_CUDA_MMID_CAPABILITY_INVALID_IO,
    GGML_CUDA_MMID_CAPABILITY_INVALID_GEOMETRY,
    GGML_CUDA_MMID_CAPABILITY_INVALID_PHASE,
    GGML_CUDA_MMID_CAPABILITY_INVALID_MAPPING,
    GGML_CUDA_MMID_CAPABILITY_INVALID_DEVICE,
};

struct ggml_cuda_mmid_source_capability {
    ggml_type type = GGML_TYPE_COUNT;
    uint32_t flags = 0;
};

struct ggml_cuda_mmid_capability_query {
    ggml_type source_type = GGML_TYPE_COUNT;
    ggml_type input_type = GGML_TYPE_COUNT;
    ggml_type output_type = GGML_TYPE_COUNT;
    int64_t source_ne[GGML_MAX_DIMS] = {};
    size_t source_nb[GGML_MAX_DIMS] = {};
    int64_t n_tokens = 0;
    int64_t n_experts = 0;
    int cc = 0;
    int warp_size = 0;
    size_t smpbo = 0;
    ggml_cuda_mmid_phase phase = GGML_CUDA_MMID_PHASE_DECODE;
    ggml_cuda_mmid_mapping mapping = GGML_CUDA_MMID_MAPPING_DIRECT;
    ggml_cuda_mmid_consumer preferred_consumer = GGML_CUDA_MMID_CONSUMER_UNSUPPORTED;
    bool use_mmq = false;
    bool independent_rows = false;
};

struct ggml_cuda_mmid_capability {
    ggml_cuda_mmid_source_capability source;
    ggml_cuda_mmid_consumer selection = GGML_CUDA_MMID_CONSUMER_UNSUPPORTED;
    ggml_cuda_mmid_capability_reason reason = GGML_CUDA_MMID_CAPABILITY_UNADVERTISED_SOURCE;
};

struct ggml_cuda_mmid_direct_source_view {
    ggml_type source_type = GGML_TYPE_COUNT;
    int64_t logical_n_experts = 0;
    int64_t physical_n_experts = 0;
    int64_t physical_id_upper_bound = 0;
    size_t expert_stride = 0;
};

// Images stay borrowed and immutable until the launch completes. Source pointers use logical expert IDs.
struct ggml_cuda_mmid_execution {
    const uint8_t * route_owners = nullptr;
    const void * const * expert_sources = nullptr;
    const uint32_t * active_rows = nullptr;
    uint32_t * status = nullptr;
    uint32_t row_capacity = 0;
    uint32_t routes_per_row = 0;
    uint32_t expert_count = 0;
    uint8_t owner = 0;
};

#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__MUSACC__)
static __device__ __forceinline__ bool ggml_cuda_mmid_route_owned(const ggml_cuda_mmid_execution & execution,
        uint32_t row, uint32_t column) {
    const uint32_t rows = execution.active_rows ? *execution.active_rows : execution.row_capacity;
    if (rows > execution.row_capacity || column >= execution.routes_per_row) {
        atomicOr(execution.status, 1u);
        return false;
    }
    return row < rows && (!execution.route_owners ||
        execution.route_owners[size_t(row) * execution.routes_per_row + column] == execution.owner);
}

static __device__ __forceinline__ const void * ggml_cuda_mmid_expert_source(const ggml_cuda_mmid_execution & execution,
        uint32_t expert, const void * direct) {
    if (expert >= execution.expert_count) {
        atomicOr(execution.status, 2u);
        return nullptr;
    }
    if (!execution.expert_sources) { return direct; }
    const void * source = execution.expert_sources[expert];
    if (!source) { atomicOr(execution.status, 4u); }
    return source;
}
#endif

struct ggml_backend_cuda_context;

struct ggml_cuda_mmid_resources {
    ggml_cuda_mmid_consumer consumer = GGML_CUDA_MMID_CONSUMER_UNSUPPORTED;
    size_t pool_bytes = 0;
};

bool ggml_cuda_mmid_pool_reserve(size_t & bytes, size_t count, size_t element_bytes);
bool ggml_cuda_mmid_requirements(int device, const ggml_tensor * dst, ggml_cuda_mmid_resources & resources);
bool ggml_cuda_mmid_execution_compute(ggml_backend_cuda_context & context, ggml_tensor * dst, const ggml_cuda_mmid_execution & execution);

bool ggml_cuda_mmid_execution_valid(const ggml_tensor * dst, const ggml_cuda_mmid_execution & execution);
bool ggml_cuda_mmid_shape_valid(const ggml_tensor * dst);

struct ggml_cuda_mmid_prefill_prepared;
ggml_cuda_mmid_prefill_prepared * ggml_cuda_mmid_prefill_prepare(ggml_backend_cuda_context & context,
    ggml_tensor * dst, const char * ids_host, size_t ids_bytes, size_t ids_row_stride);
bool ggml_cuda_mmid_prefill_launch_range(ggml_backend_cuda_context & context,
    const ggml_cuda_mmid_prefill_prepared * prepared, const void * resident, const void * staging,
    const int32_t * source_map, uint32_t n_slots, uint32_t n_staged, int32_t expert_begin, int32_t expert_count);
bool ggml_cuda_mmid_prefill_finish(ggml_backend_cuda_context & context, const ggml_cuda_mmid_prefill_prepared * prepared);
void ggml_cuda_mmid_prefill_free(ggml_cuda_mmid_prefill_prepared * prepared);
bool ggml_cuda_mmid_execution_compute(ggml_backend_t backend, ggml_tensor * dst, const ggml_cuda_mmid_execution & execution);
bool ggml_cuda_mmid_pool_compute_for_test(ggml_backend_t backend, ggml_tensor * dst,
    const ggml_cuda_mmid_execution & execution, void * workspace, size_t bytes, size_t * peak);
bool ggml_cuda_mmid_vector_compute_for_test(ggml_backend_t backend, ggml_tensor * dst);
cudaStream_t ggml_cuda_mmid_execution_stream_for_test(ggml_backend_t backend);
bool ggml_cuda_mmq_mmid_pair_compute_for_test(ggml_backend_t backend, const ggml_tensor * up,
    const ggml_tensor * gate, ggml_tensor * glu, uint32_t resident_slots, uint32_t staging_slots, uint32_t * waves);

ggml_cuda_mmid_source_capability ggml_cuda_mmid_source_capability_for(ggml_type type);
ggml_cuda_mmid_capability ggml_cuda_mmid_get_capability(const ggml_cuda_mmid_capability_query & query);
bool ggml_cuda_mmid_can_use_compact_mmvq(const ggml_cuda_mmid_capability_query & query, int64_t n_compact_experts);
bool ggml_cuda_moe_use_mmq(const ggml_tensor * src0, int64_t n_tokens);
bool ggml_cuda_mmid_direct_source_view_valid(
        const ggml_tensor * source,
        const ggml_cuda_mmid_direct_source_view & view,
        ggml_cuda_mmid_consumer consumer);
bool ggml_cuda_mmid_direct_source_view_compute_for_test(
        ggml_backend_t backend,
        ggml_tensor * dst,
        const ggml_cuda_mmid_direct_source_view * view,
        ggml_cuda_mmid_consumer consumer);
bool ggml_cuda_mmid_bounded_compute_for_test(
        ggml_backend_t backend, ggml_tensor * dst, const uint32_t * active_channels, uint32_t * status);

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, bool write_inverse, cudaStream_t stream,
        const ggml_cuda_mmid_execution * execution = nullptr);
