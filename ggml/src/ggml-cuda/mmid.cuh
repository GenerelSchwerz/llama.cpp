#pragma once

#include "ggml.h"

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

ggml_cuda_mmid_source_capability ggml_cuda_mmid_source_capability_for(ggml_type type);
ggml_cuda_mmid_capability ggml_cuda_mmid_get_capability(const ggml_cuda_mmid_capability_query & query);
bool ggml_cuda_moe_use_mmq(const ggml_tensor * src0, int64_t n_tokens);

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, bool write_inverse, cudaStream_t stream);
