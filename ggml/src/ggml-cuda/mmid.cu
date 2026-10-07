#include "common.cuh"
#include "mmid.cuh"
#include "mmf.cuh"
#include "mmvf.cuh"
#include "mmq.cuh"
#include "mmvq.cuh"

#include <climits>

static bool ggml_cuda_mmid_shape_valid(const ggml_tensor * dst) {
    if (!dst || dst->op != GGML_OP_MUL_MAT_ID || !dst->src[0] || !dst->src[1] || !dst->src[2]) { return false; }
    const auto * weight = dst->src[0];
    const auto * input = dst->src[1];
    const auto * ids = dst->src[2];
    return weight->type >= 0 && weight->type < GGML_TYPE_COUNT &&
        input->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        weight->ne[0] > 0 && weight->ne[0] <= INT_MAX && weight->ne[1] > 0 && weight->ne[1] <= INT_MAX &&
        weight->ne[2] > 0 && weight->ne[2] <= INT_MAX && weight->ne[3] == 1 &&
        input->ne[0] == weight->ne[0] && input->ne[1] > 0 && input->ne[1] <= INT_MAX && input->ne[2] > 0 && input->ne[2] <= INT_MAX && input->ne[3] == 1 &&
        ids->type == GGML_TYPE_I32 && ids->ne[0] > 0 && ids->ne[0] <= INT_MAX && ids->ne[1] == input->ne[2] && ids->ne[2] == 1 && ids->ne[3] == 1 &&
        ids->ne[0] % input->ne[1] == 0 && ids->nb[0] == sizeof(int32_t) && ids->nb[1] % sizeof(int32_t) == 0 &&
        dst->ne[0] == weight->ne[1] && dst->ne[1] == ids->ne[0] && dst->ne[2] == ids->ne[1] && dst->ne[3] == 1 &&
        uint64_t(ids->ne[0]) * uint64_t(ids->ne[1]) <= INT_MAX;
}

bool ggml_cuda_mmid_execution_valid(const ggml_tensor * dst, const ggml_cuda_mmid_execution & execution) {
    return ggml_cuda_mmid_shape_valid(dst) && execution.status &&
        execution.row_capacity == uint64_t(dst->src[2]->ne[1]) && execution.routes_per_row == uint64_t(dst->src[2]->ne[0]) &&
        execution.expert_count == uint64_t(dst->src[0]->ne[2]);
}

bool ggml_cuda_mmid_pool_reserve(size_t & bytes, size_t count, size_t element_bytes) {
    if (!element_bytes || count > (SIZE_MAX - 255) / element_bytes) { return false; }
    const size_t aligned = (count * element_bytes + 255) & ~size_t(255);
    if (aligned > SIZE_MAX - bytes) { return false; }
    bytes += aligned;
    return true;
}

bool ggml_cuda_mmid_requirements(int device, const ggml_tensor * dst, ggml_cuda_mmid_resources & resources) {
    if (device < 0 || device >= ggml_cuda_info().device_count || !ggml_cuda_mmid_shape_valid(dst)) { return false; }
    auto * registry = ggml_backend_cuda_reg();
    if (size_t(device) >= ggml_backend_reg_dev_count(registry) ||
            !ggml_backend_dev_supports_op(ggml_backend_reg_dev_get(registry, device), dst)) { return false; }
    const auto * weight = dst->src[0];
    const auto * input = dst->src[1];
    const auto * ids = dst->src[2];
    const auto layout = [](const ggml_tensor * tensor) {
        const size_t block = ggml_blck_size(tensor->type);
        const size_t element = ggml_type_size(tensor->type);
        if (!block || !element || tensor->ne[0] % block || tensor->nb[0] != element) { return false; }
        size_t span = size_t(tensor->ne[0]) / block;
        if (span > SIZE_MAX / element) { return false; }
        span *= element;
        for (int i = 1; i < GGML_MAX_DIMS; ++i) {
            if (tensor->nb[i] % element || tensor->nb[i] < span ||
                    (i < 3 && tensor->nb[i] / element > INT_MAX) || size_t(tensor->ne[i]) > SIZE_MAX / tensor->nb[i]) { return false; }
            span = size_t(tensor->ne[i]) * tensor->nb[i];
        }
        return true;
    };
    if (!layout(weight) || !layout(input) || !layout(ids) || !layout(dst) ||
            input->nb[2] % input->nb[1] || dst->nb[2] % dst->nb[1]) { return false; }
    const auto & info = ggml_cuda_info().devices[device];
    ggml_cuda_mmid_capability_query query;
    query.source_type = weight->type;
    query.input_type = input->type;
    query.output_type = dst->type;
    std::copy(std::begin(weight->ne), std::end(weight->ne), query.source_ne);
    std::copy(std::begin(weight->nb), std::end(weight->nb), query.source_nb);
    query.n_tokens = input->ne[2];
    query.n_experts = weight->ne[2];
    query.phase = input->ne[2] == 1 ? GGML_CUDA_MMID_PHASE_DECODE : GGML_CUDA_MMID_PHASE_PREFILL;
    query.mapping = GGML_CUDA_MMID_MAPPING_DIRECT;
    query.cc = info.cc;
    query.warp_size = info.warp_size;
    query.smpbo = info.smpbo;
    query.use_mmq = true;
    const auto capability = ggml_cuda_mmid_get_capability(query);
    if (capability.reason != GGML_CUDA_MMID_CAPABILITY_OK) { return false; }
    ggml_cuda_mmid_resources measured;
    measured.consumer = capability.selection;
    if (measured.consumer == GGML_CUDA_MMID_CONSUMER_GENERIC) {
        if (!(capability.source.flags & GGML_CUDA_MMID_SOURCE_SCALAR) || weight->ne[0] % 2 ||
                weight->nb[1] % (2*ggml_type_size(weight->type)) || input->nb[2] % (2*sizeof(float))) { return false; }
        measured.consumer = GGML_CUDA_MMID_CONSUMER_MMVF;
    }
    const size_t routes = size_t(ids->ne[0]) * size_t(ids->ne[1]);
    if (measured.consumer == GGML_CUDA_MMID_CONSUMER_MMVQ) {
        size_t input_bytes;
        if (!ggml_cuda_mmvq_input_bytes(input, input_bytes) || !ggml_cuda_mmid_pool_reserve(measured.pool_bytes, input_bytes, 1)) { return false; }
    } else if (measured.consumer == GGML_CUDA_MMID_CONSUMER_MMF) {
        if (!ggml_cuda_mmid_pool_reserve(measured.pool_bytes, routes, sizeof(int32_t)) ||
                !ggml_cuda_mmid_pool_reserve(measured.pool_bytes, routes, sizeof(int32_t)) ||
                !ggml_cuda_mmid_pool_reserve(measured.pool_bytes, size_t(weight->ne[2]) + 1, sizeof(int32_t))) { return false; }
    } else if (measured.consumer == GGML_CUDA_MMID_CONSUMER_MMQ) {
        ggml_cuda_mmq_routed_resources packed;
        if (!ggml_cuda_mmq_routed_requirements(device, dst, packed) ||
                !ggml_cuda_mmid_pool_reserve(measured.pool_bytes, packed.rows, sizeof(int32_t)) ||
                !ggml_cuda_mmid_pool_reserve(measured.pool_bytes, packed.padded_rows, sizeof(int32_t)) ||
                !ggml_cuda_mmid_pool_reserve(measured.pool_bytes, size_t(weight->ne[2]) + 1, sizeof(int32_t)) ||
                !ggml_cuda_mmid_pool_reserve(measured.pool_bytes, packed.quantized_bytes, 1) ||
                (packed.scale_count && !ggml_cuda_mmid_pool_reserve(measured.pool_bytes, packed.scale_count, sizeof(float))) ||
                (packed.fixup_elements && !ggml_cuda_mmid_pool_reserve(measured.pool_bytes, packed.fixup_elements, sizeof(float)))) { return false; }
    } else if (measured.consumer != GGML_CUDA_MMID_CONSUMER_MMVF) {
        return false;
    }
    resources = measured;
    return true;
}

ggml_cuda_mmid_source_capability ggml_cuda_mmid_source_capability_for(ggml_type type) {
    constexpr uint32_t scalar = GGML_CUDA_MMID_SOURCE_ADVERTISED | GGML_CUDA_MMID_SOURCE_SCALAR | GGML_CUDA_MMID_SOURCE_GENERIC;
    constexpr uint32_t quant = GGML_CUDA_MMID_SOURCE_ADVERTISED | GGML_CUDA_MMID_SOURCE_MMVQ | GGML_CUDA_MMID_SOURCE_MMQ |
        GGML_CUDA_MMID_SOURCE_MAPPED_MMQ | GGML_CUDA_MMID_SOURCE_GENERIC;
    constexpr uint32_t fp4 = GGML_CUDA_MMID_SOURCE_ADVERTISED | GGML_CUDA_MMID_SOURCE_MMVQ | GGML_CUDA_MMID_SOURCE_MMQ |
        GGML_CUDA_MMID_SOURCE_GENERIC;
    constexpr uint32_t mmvq = GGML_CUDA_MMID_SOURCE_ADVERTISED | GGML_CUDA_MMID_SOURCE_MMVQ | GGML_CUDA_MMID_SOURCE_GENERIC;

    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            return {type, scalar};
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return {type, quant};
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
            return {type, fp4};
        case GGML_TYPE_IQ1_M:
            return {type, mmvq};
        case GGML_TYPE_Q8_K:
            return {type, GGML_CUDA_MMID_SOURCE_ADVERTISED};
        default:
            return {type, 0};
    }
}

ggml_cuda_mmid_capability ggml_cuda_mmid_get_capability(const ggml_cuda_mmid_capability_query & query) {
    ggml_cuda_mmid_capability result;
    result.source = ggml_cuda_mmid_source_capability_for(query.source_type);
    if ((result.source.flags & GGML_CUDA_MMID_SOURCE_ADVERTISED) == 0) {
        result.reason = GGML_CUDA_MMID_CAPABILITY_UNADVERTISED_SOURCE;
        return result;
    }
    if ((result.source.flags & GGML_CUDA_MMID_SOURCE_GENERIC) == 0) {
        result.reason = GGML_CUDA_MMID_CAPABILITY_UNSUPPORTED_CONSUMER;
        return result;
    }
    if (query.input_type != GGML_TYPE_F32 || query.output_type != GGML_TYPE_F32) {
        result.reason = GGML_CUDA_MMID_CAPABILITY_INVALID_IO;
        return result;
    }
    if (query.phase != GGML_CUDA_MMID_PHASE_DECODE && query.phase != GGML_CUDA_MMID_PHASE_PREFILL) {
        result.reason = GGML_CUDA_MMID_CAPABILITY_INVALID_PHASE;
        return result;
    }
    if ((query.phase == GGML_CUDA_MMID_PHASE_DECODE && query.n_tokens != 1 && !query.independent_rows) ||
            (query.phase == GGML_CUDA_MMID_PHASE_PREFILL && query.n_tokens <= 1)) {
        result.reason = GGML_CUDA_MMID_CAPABILITY_INVALID_PHASE;
        return result;
    }
    if (query.mapping != GGML_CUDA_MMID_MAPPING_DIRECT && query.mapping != GGML_CUDA_MMID_MAPPING_SOURCE_MAP) {
        result.reason = GGML_CUDA_MMID_CAPABILITY_INVALID_MAPPING;
        return result;
    }
    const int64_t block_size = ggml_blck_size(query.source_type);
    if (query.n_tokens <= 0 || query.n_experts <= 0 || query.source_ne[0] <= 0 || query.source_ne[1] <= 0 ||
            query.source_ne[2] != query.n_experts || block_size <= 0 || query.source_ne[0] % block_size != 0 ||
            query.source_nb[0] != ggml_type_size(query.source_type)) {
        result.reason = GGML_CUDA_MMID_CAPABILITY_INVALID_GEOMETRY;
        return result;
    }
    if (query.cc <= 0 || query.warp_size <= 0 || query.smpbo == 0) {
        result.reason = GGML_CUDA_MMID_CAPABILITY_INVALID_DEVICE;
        return result;
    }

    const bool mapped = query.mapping == GGML_CUDA_MMID_MAPPING_SOURCE_MAP;
    const auto supported = [&](ggml_cuda_mmid_consumer consumer) {
        switch (consumer) {
            case GGML_CUDA_MMID_CONSUMER_MMVQ:
                return !mapped && (result.source.flags & GGML_CUDA_MMID_SOURCE_MMVQ) != 0 &&
                    query.n_tokens <= MMVQ_MAX_BATCH_SIZE &&
                    query.n_tokens <= get_mmvq_mmid_max_batch(query.source_type, query.cc);
            case GGML_CUDA_MMID_CONSUMER_MMVF:
                return !mapped && (result.source.flags & GGML_CUDA_MMID_SOURCE_SCALAR) != 0 &&
                    query.n_tokens <= MMVF_MAX_BATCH_SIZE && GGML_CUDA_CC_IS_AMD(query.cc);
            case GGML_CUDA_MMID_CONSUMER_MMQ:
                return query.use_mmq && (result.source.flags & GGML_CUDA_MMID_SOURCE_MMQ) != 0 &&
                    (!mapped || (result.source.flags & GGML_CUDA_MMID_SOURCE_MAPPED_MMQ) != 0) &&
                    ggml_cuda_should_use_mmq(query.source_type, query.cc, query.n_tokens, query.n_experts, query.smpbo);
            case GGML_CUDA_MMID_CONSUMER_MMF:
                return !mapped && (result.source.flags & GGML_CUDA_MMID_SOURCE_SCALAR) != 0 &&
                    ggml_cuda_should_use_mmf(query.source_type, query.cc, query.warp_size, query.source_ne, query.source_nb,
                        query.n_tokens, true);
            case GGML_CUDA_MMID_CONSUMER_GENERIC:
                return (result.source.flags & GGML_CUDA_MMID_SOURCE_GENERIC) != 0;
            case GGML_CUDA_MMID_CONSUMER_UNSUPPORTED:
                return false;
        }
        return false;
    };

    if (query.preferred_consumer != GGML_CUDA_MMID_CONSUMER_UNSUPPORTED) {
        if (!supported(query.preferred_consumer)) {
            result.reason = GGML_CUDA_MMID_CAPABILITY_UNSUPPORTED_CONSUMER;
            return result;
        }
        result.selection = query.preferred_consumer;
    } else if (supported(GGML_CUDA_MMID_CONSUMER_MMVQ)) {
        result.selection = GGML_CUDA_MMID_CONSUMER_MMVQ;
    } else if (supported(GGML_CUDA_MMID_CONSUMER_MMVF)) {
        result.selection = GGML_CUDA_MMID_CONSUMER_MMVF;
    } else if (supported(GGML_CUDA_MMID_CONSUMER_MMQ)) {
        result.selection = GGML_CUDA_MMID_CONSUMER_MMQ;
    } else if (supported(GGML_CUDA_MMID_CONSUMER_MMF)) {
        result.selection = GGML_CUDA_MMID_CONSUMER_MMF;
    } else {
        result.selection = GGML_CUDA_MMID_CONSUMER_GENERIC;
    }
    result.reason = GGML_CUDA_MMID_CAPABILITY_OK;
    return result;
}

bool ggml_cuda_mmid_can_use_compact_mmvq(
        const ggml_cuda_mmid_capability_query & query,
        int64_t n_compact_experts) {
    if (query.mapping != GGML_CUDA_MMID_MAPPING_DIRECT || n_compact_experts <= 0 ||
            query.source_nb[2] > SIZE_MAX / (size_t) n_compact_experts) {
        return false;
    }
    const auto direct = ggml_cuda_mmid_get_capability(query);
    if (direct.reason != GGML_CUDA_MMID_CAPABILITY_OK || direct.selection != GGML_CUDA_MMID_CONSUMER_MMVQ) {
        return false;
    }

    ggml_cuda_mmid_capability_query compact = query;
    compact.n_experts = n_compact_experts;
    compact.source_ne[2] = n_compact_experts;
    compact.source_nb[3] = compact.source_nb[2] * (size_t) n_compact_experts;
    compact.preferred_consumer = GGML_CUDA_MMID_CONSUMER_MMVQ;
    const auto capability = ggml_cuda_mmid_get_capability(compact);
    return capability.reason == GGML_CUDA_MMID_CAPABILITY_OK && capability.selection == GGML_CUDA_MMID_CONSUMER_MMVQ;
}

bool ggml_cuda_mmid_direct_source_view_valid(
        const ggml_tensor * source,
        const ggml_cuda_mmid_direct_source_view & view,
        ggml_cuda_mmid_consumer consumer) {
    if (source == nullptr || view.source_type != source->type || view.logical_n_experts <= 0 ||
            view.logical_n_experts > INT_MAX || view.physical_n_experts <= 0 || view.physical_n_experts > INT_MAX ||
            view.physical_id_upper_bound <= 0 || view.physical_id_upper_bound > view.physical_n_experts ||
            view.logical_n_experts > view.physical_n_experts || view.physical_n_experts != source->ne[2] ||
            source->ne[3] != 1 || view.expert_stride == 0 || view.expert_stride != source->nb[2] ||
            view.expert_stride > SIZE_MAX / (size_t) view.physical_n_experts ||
            source->nb[3] != view.expert_stride * (size_t) view.physical_n_experts) {
        return false;
    }

    const auto capability = ggml_cuda_mmid_source_capability_for(source->type);
    switch (consumer) {
        case GGML_CUDA_MMID_CONSUMER_MMVQ:
            return (capability.flags & GGML_CUDA_MMID_SOURCE_MMVQ) != 0;
        case GGML_CUDA_MMID_CONSUMER_MMVF:
        case GGML_CUDA_MMID_CONSUMER_MMF:
            return (capability.flags & GGML_CUDA_MMID_SOURCE_SCALAR) != 0;
        case GGML_CUDA_MMID_CONSUMER_MMQ:
            return (capability.flags & GGML_CUDA_MMID_SOURCE_MMQ) != 0;
        case GGML_CUDA_MMID_CONSUMER_UNSUPPORTED:
        case GGML_CUDA_MMID_CONSUMER_GENERIC:
            return false;
    }
    return false;
}

// the generic path passes 0, which needs no padding since it never groups lanes by token
template <int n> struct mm_ids_pow2 { static constexpr int value = 2*mm_ids_pow2<(n + 1)/2>::value; };
template <>      struct mm_ids_pow2<1> { static constexpr int value = 1; };
template <>      struct mm_ids_pow2<0> { static constexpr int value = 1; };

template <bool routed>
static __device__ __forceinline__ int mm_ids_route_expert(const int32_t * ids, uint32_t row, uint32_t column,
        int stride, const ggml_cuda_mmid_execution & execution) {
    if constexpr (routed) {
        if (!ggml_cuda_mmid_route_owned(execution, row, column)) { return INT_MAX; }
    }
    const int expert = ids[size_t(row)*stride + column];
    if constexpr (routed) {
        if (uint32_t(expert) >= execution.expert_count) {
            atomicOr(execution.status, 2u);
            return INT_MAX;
        }
        if (execution.expert_sources && !execution.expert_sources[expert]) {
            atomicOr(execution.status, 4u);
            return INT_MAX;
        }
    }
    return expert;
}

template <int n_expert_used_template, bool routed>
static __device__ __forceinline__ int mm_ids_write_routes(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, const int expert, const int nex_prev,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse,
        const ggml_cuda_mmid_execution execution) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int slots_per_token = n_expert_used_template == 0 ? n_expert_used : mm_ids_pow2<n_expert_used_template>::value;
    const int64_t slots = int64_t(n_tokens)*slots_per_token;
    int it_compact = 0;
    for (int64_t slot0 = 0; slot0 < slots; slot0 += warp_size) {
        const int64_t slot = slot0 + threadIdx.x;
        const int64_t token = slot/slots_per_token;
        const int column = slot%slots_per_token;
        const int match = slot < slots && column < n_expert_used &&
            mm_ids_route_expert<routed>(ids, token, column, si1, execution) == expert;
        const int prefix = warp_prefix_inclusive_sum<int, warp_size>(match);
        if (match) {
            const int sorted = nex_prev + it_compact + prefix - 1;
            const int route = token*n_expert_used + column;
            ids_dst[sorted] = route;
            if (write_inverse) {
                ids_src1[route] = sorted;
            } else {
                ids_src1[sorted] = token*sis1 + column%nchannels_y;
            }
        }
        it_compact += __shfl_sync(0xFFFFFFFF, prefix, warp_size - 1, warp_size);
    }
    return it_compact;
}

template <bool routed>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper_unbuffered(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse,
        const ggml_cuda_mmid_execution execution) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int expert = blockIdx.x;
    const int64_t routes = int64_t(n_tokens)*n_expert_used;
    int nex_prev = 0;
    for (int64_t route = threadIdx.x; route < routes; route += warp_size) {
        nex_prev += mm_ids_route_expert<routed>(ids, route/n_expert_used, route%n_expert_used, si1, execution) < expert;
    }
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);
    const int compact = mm_ids_write_routes<0, routed>(ids, ids_src1, ids_dst, expert, nex_prev,
        n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, execution);
    if (threadIdx.x == 0) {
        expert_bounds[expert] = nex_prev;
        if (expert == static_cast<int>(gridDim.x) - 1) {
            expert_bounds[gridDim.x] = nex_prev + compact;
        }
    }
}

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The upper and lower bounds for the ith expert in the compact src1 tensor are stored in expert_bounds[i:i+1].
template <int n_expert_used_template, bool routed>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse,
        const ggml_cuda_mmid_execution execution) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;

    const int grouped_used = n_expert_used <= warp_size ? n_expert_used : warp_size;
    const int token_shift = 32 - __clz(grouped_used - 1);
    const int neu_padded = 1 << token_shift;

    extern __shared__ char data_mm_ids_helper[];
    int32_t * store = (int32_t *) data_mm_ids_helper;

    int nex_prev   = 0; // Number of columns for experts with a lower index.
    int it_compact = 0; // Running index for the compact slice of this expert.
    int repeated = 0;

    if (n_expert_used > warp_size) {
        // Multiple route chunks per token:
        for (int it = 0; it < n_tokens; ++it) {
            int iex_used = -1; // The index at which the expert is used, if any.
            int matches = 0;
            for (int iex = threadIdx.x; iex < n_expert_used; iex += warp_size) {
                const int expert_used = mm_ids_route_expert<routed>(ids, it, iex, si1, execution);
                nex_prev += expert_used < expert;
                if (expert_used == expert) {
                    iex_used = iex;
                    ++matches;
                }
            }

#if defined(GGML_USE_HIP)
            const uint64_t mask = __ballot(iex_used != -1);
#else
            const uint32_t mask = __ballot_sync(0xFFFFFFFF, iex_used != -1);
#endif
            const bool duplicate = (mask & (mask - 1)) != 0;
            repeated |= duplicate || matches > 1;
            if (iex_used != -1 && !duplicate) {
                store[it_compact] = it*n_expert_used + iex_used;
            }
            it_compact += mask != 0;
        }
    } else {
        const int tokens_per_warp = warp_size >> token_shift;
        const int token_lane = threadIdx.x >> token_shift;
        const int iex = threadIdx.x & (neu_padded - 1);
        for (int it0 = 0; it0 < n_tokens; it0 += tokens_per_warp) {
            const int it = it0 + token_lane;
            const int expert_used = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
                mm_ids_route_expert<routed>(ids, it, iex, si1, execution) : INT_MAX;
            const int iex_used = expert_used == expert ? iex : -1;
            nex_prev += expert_used < expert;

#if defined(GGML_USE_HIP)
            const uint64_t matches = __ballot(iex_used != -1);
            const uint64_t token_mask = UINT64_MAX >> (64 - neu_padded);
#else
            const uint32_t matches = __ballot_sync(0xFFFFFFFF, iex_used != -1);
            const uint32_t token_mask = UINT32_MAX >> (32 - neu_padded);
#endif
            const auto token_matches = (matches >> (token_lane << token_shift)) & token_mask;
            const bool duplicate = (token_matches & (token_matches - 1)) != 0;
            repeated |= duplicate;
            const int it_compact_add_self = token_matches != 0;
            int it_compact_add_lower = 0;
#pragma unroll
            for (int offset = neu_padded; offset < warp_size; offset += neu_padded) {
                const int tmp = __shfl_up_sync(0xFFFFFFFF, it_compact_add_self, offset, warp_size);
                if (threadIdx.x >= static_cast<unsigned int>(offset)) {
                    it_compact_add_lower += tmp;
                }
            }
            if (iex_used != -1 && !duplicate) {
                store[it_compact + it_compact_add_lower] = it*n_expert_used + iex_used;
            }
            it_compact += __shfl_sync(0xFFFFFFFF, it_compact_add_lower + it_compact_add_self, warp_size - 1, warp_size);
        }
    }
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);
    repeated = warp_reduce_any<warp_size>(repeated);
    ggml_cuda_syncwarp();

    if (!repeated) {
        for (int itc = threadIdx.x; itc < it_compact; itc += warp_size) {
            const int route = store[itc];
            ids_dst[nex_prev + itc] = route;
            if (write_inverse) {
                ids_src1[route] = nex_prev + itc;
            } else {
                ids_src1[nex_prev + itc] = (route/n_expert_used)*sis1 + (route%n_expert_used)%nchannels_y;
            }
        }
    } else {
        it_compact = mm_ids_write_routes<n_expert_used_template, routed>(ids, ids_src1, ids_dst, expert, nex_prev,
            n_tokens, n_expert_used_var, nchannels_y, si1, sis1, write_inverse, execution);
    }

    if (threadIdx.x != 0) {
        return;
    }

    expert_bounds[expert] = nex_prev;

    if (expert < static_cast<int>(gridDim.x) - 1) {
        return;
    }

    expert_bounds[gridDim.x] = nex_prev + it_compact;
}

template <int n_expert_used_template, bool routed>
static void launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream,
        const ggml_cuda_mmid_execution execution) {
    GGML_ASSERT(n_experts > 0 && n_tokens > 0 && n_expert_used_var > 0);
    GGML_ASSERT(nchannels_y > 0 && n_expert_used_var % nchannels_y == 0);
    GGML_ASSERT(si1 >= 0 && sis1 >= 0);
    GGML_ASSERT(n_tokens <= INT_MAX/n_expert_used_var);
    GGML_ASSERT(sis1 == 0 || n_tokens - 1 <= (INT_MAX - (nchannels_y - 1))/sis1);

    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
    CUDA_SET_SHARED_MEMORY_LIMIT((mm_ids_helper<n_expert_used_template, routed>), smpbo);

    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);
    if (size_t(n_tokens) > smpbo/sizeof(int32_t)) {
        mm_ids_helper_unbuffered<routed><<<num_blocks, block_size, 0, stream>>>
            (ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used_var, nchannels_y, si1, sis1, write_inverse, execution);
        return;
    }
    const size_t nbytes_shared = size_t(n_tokens)*sizeof(int32_t);
    mm_ids_helper<n_expert_used_template, routed><<<num_blocks, block_size, nbytes_shared, stream>>>
        (ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used_var, nchannels_y, si1, sis1, write_inverse, execution);
}

template <bool routed>
static void dispatch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream,
        const ggml_cuda_mmid_execution execution) {
    switch (n_expert_used) {
        case  2:
            launch_mm_ids_helper< 2, routed>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream, execution);
            break;
        case  4:
            launch_mm_ids_helper< 4, routed>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream, execution);
            break;
        case  6:
            launch_mm_ids_helper< 6, routed>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream, execution);
            break;
        case  8:
            launch_mm_ids_helper< 8, routed>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream, execution);
            break;
        case 10:
            launch_mm_ids_helper<10, routed>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream, execution);
            break;
        case 16:
            launch_mm_ids_helper<16, routed>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream, execution);
            break;
        case 32:
            launch_mm_ids_helper<32, routed>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream, execution);
            break;
        default:
            launch_mm_ids_helper< 0, routed>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream, execution);
            break;
    }
}

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, bool write_inverse, cudaStream_t stream,
        const ggml_cuda_mmid_execution * execution) {
    if (execution) {
        GGML_ASSERT(execution->status && execution->row_capacity == uint32_t(n_tokens) &&
            execution->routes_per_row == uint32_t(n_expert_used) && execution->expert_count == uint32_t(n_experts));
        GGML_ASSERT(n_tokens > 0 && n_expert_used > 0 && n_tokens <= INT_MAX/n_expert_used);
        const size_t bytes = size_t(n_tokens)*n_expert_used*sizeof(int32_t);
        CUDA_CHECK(cudaMemsetAsync(ids_src1, write_inverse ? 0xff : 0, bytes, stream));
        CUDA_CHECK(cudaMemsetAsync(ids_dst, 0, bytes, stream));
        dispatch_mm_ids_helper<true>(ids, ids_src1, ids_dst, expert_bounds,
            n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream, *execution);
    } else {
        dispatch_mm_ids_helper<false>(ids, ids_src1, ids_dst, expert_bounds,
            n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream, {});
    }
}
