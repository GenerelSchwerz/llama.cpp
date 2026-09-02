#include "common.cuh"
#include "mmid.cuh"
#include "mmf.cuh"
#include "mmvf.cuh"
#include "mmq.cuh"
#include "mmvq.cuh"

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

// To reduce shared memory use, store "it" and "iex_used" with 22/10 bits each.
struct mm_ids_helper_store {
    uint32_t data;

    __device__ mm_ids_helper_store(const uint32_t it, const uint32_t iex_used) {
        data = (it & 0x003FFFFF) | (iex_used << 22);
    }

    __device__ uint32_t it() const {
        return data & 0x003FFFFF;
    }

    __device__ uint32_t iex_used() const {
        return data >> 22;
    }
};
static_assert(sizeof(mm_ids_helper_store) == 4, "unexpected size for mm_ids_helper_store");

// the generic path passes 0, which needs no padding since it never groups lanes by token
template <int n> struct mm_ids_pow2 { static constexpr int value = 2*mm_ids_pow2<(n + 1)/2>::value; };
template <>      struct mm_ids_pow2<1> { static constexpr int value = 1; };
template <>      struct mm_ids_pow2<0> { static constexpr int value = 1; };

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The upper and lower bounds for the ith expert in the compact src1 tensor are stored in expert_bounds[i:i+1].
template <int n_expert_used_template>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;

    // token slots per warp lane group, padded to a power of 2 so a warp divides evenly
    constexpr int neu_padded = mm_ids_pow2<n_expert_used_template>::value;

    extern __shared__ char data_mm_ids_helper[];
    mm_ids_helper_store * store = (mm_ids_helper_store *) data_mm_ids_helper;

    int nex_prev   = 0; // Number of columns for experts with a lower index.
    int it_compact = 0; // Running index for the compact slice of this expert.

    if constexpr (n_expert_used_template == 0) {
        // Generic implementation:
        for (int it = 0; it < n_tokens; ++it) {
            int iex_used = -1; // The index at which the expert is used, if any.
            for (int iex = threadIdx.x; iex < n_expert_used; iex += warp_size) {
                const int expert_used = ids[it*si1 + iex];
                nex_prev += expert_used < expert;
                if (expert_used == expert) {
                    iex_used = iex;
                }
            }

            if (iex_used != -1) {
                store[it_compact] = mm_ids_helper_store(it, iex_used);
            }

            if (warp_reduce_any<warp_size>(iex_used != -1)) {
                it_compact++;
            }
        }
    } else {
        // Implementation optimized for specific numbers of experts used:
        // a warp holds a whole number of token slots, so the slot count is padded to a power of 2
        static_assert(neu_padded <= warp_size && warp_size % neu_padded == 0, "bad n_expert_used");
        for (int it0 = 0; it0 < n_tokens; it0 += warp_size/neu_padded) {
            const int it = it0 + threadIdx.x / neu_padded;

            const int iex = threadIdx.x % neu_padded; // The index at which the expert is used, if any.
            const int expert_used = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
                ids[it*si1 + iex] : INT_MAX;
            const int iex_used = expert_used == expert ? iex : -1;
            nex_prev += expert_used < expert;

            // Whether the threads at this token position have used the expert:
            const int it_compact_add_self = warp_reduce_any<neu_padded>(iex_used != -1);

            // Do a scan over threads at lower token positions in warp to get the correct index for writing data:
            int it_compact_add_lower = 0;
#pragma unroll
            for (int offset = neu_padded; offset < warp_size; offset += neu_padded) {
                const int tmp = __shfl_up_sync(0xFFFFFFFF, it_compact_add_self, offset, warp_size);
                if (threadIdx.x >= static_cast<unsigned int>(offset)) {
                    it_compact_add_lower += tmp;
                }
            }

            if (iex_used != -1) {
                store[it_compact + it_compact_add_lower] = mm_ids_helper_store(it, iex_used);
            }

            // The thread with the highest index in the warp always has the sum over the whole warp, use it to increment all threads:
            it_compact += __shfl_sync(0xFFFFFFFF, it_compact_add_lower + it_compact_add_self, warp_size - 1, warp_size);
        }
    }
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);

    for (int itc = threadIdx.x; itc < it_compact; itc += warp_size) {
        const mm_ids_helper_store store_it = store[itc];
        const int it       = store_it.it();
        const int iex_used = store_it.iex_used();
        ids_dst[nex_prev + itc] = it*n_expert_used + iex_used;
        // ids_src1 holds the forward map, or the inverse map (token slot -> compact row) for quant dedup
        if (write_inverse) {
            ids_src1[it*n_expert_used + iex_used] = nex_prev + itc;
        } else {
            ids_src1[nex_prev + itc] = it*sis1 + iex_used % nchannels_y;
        }
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

template <int n_expert_used_template>
static void launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    GGML_ASSERT(n_tokens          < (1 << 22) && "too few bits in mm_ids_helper_store");
    GGML_ASSERT(n_expert_used_var < (1 << 10) && "too few bits in mm_ids_helper_store");

    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
    CUDA_SET_SHARED_MEMORY_LIMIT(mm_ids_helper<n_expert_used_template>, smpbo);

    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);
    const size_t nbytes_shared = n_tokens*sizeof(mm_ids_helper_store);
    GGML_ASSERT(nbytes_shared <= smpbo);
    mm_ids_helper<n_expert_used_template><<<num_blocks, block_size, nbytes_shared, stream>>>
        (ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used_var, nchannels_y, si1, sis1, write_inverse);
}

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    switch (n_expert_used) {
        case  2:
            launch_mm_ids_helper< 2>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  4:
            launch_mm_ids_helper< 4>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  6:
            launch_mm_ids_helper< 6>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  8:
            launch_mm_ids_helper< 8>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 10:
            launch_mm_ids_helper<10>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 16:
            launch_mm_ids_helper<16>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 32:
            launch_mm_ids_helper<32>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        default:
            launch_mm_ids_helper< 0>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
    }
}
