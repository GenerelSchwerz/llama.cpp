#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "mmid.cuh"
#include "mmvq.cuh"
#include "unary.cuh"

#include <cstdint>
#include <new>

template <bool use_x_map>
static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream, const ggml_prec prec_src1) {
    switch (args.type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_q_case<GGML_TYPE_Q1_0, use_x_map>(ctx, args, stream);
            break;
        case GGML_TYPE_Q2_0:
            mul_mat_q_case<GGML_TYPE_Q2_0, use_x_map>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_q_case<GGML_TYPE_Q4_0, use_x_map>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_q_case<GGML_TYPE_Q4_1, use_x_map>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_q_case<GGML_TYPE_Q5_0, use_x_map>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_q_case<GGML_TYPE_Q5_1, use_x_map>(ctx, args, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_q_case<GGML_TYPE_Q8_0, use_x_map>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_Q2_K:
            mul_mat_q_case<GGML_TYPE_Q2_K, use_x_map>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_q_case<GGML_TYPE_Q3_K, use_x_map>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K, use_x_map>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_q_case<GGML_TYPE_Q5_K, use_x_map>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K, use_x_map>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_IQ1_S:
            mul_mat_q_case<GGML_TYPE_IQ1_S, use_x_map>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_q_case<GGML_TYPE_IQ2_XXS, use_x_map>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_q_case<GGML_TYPE_IQ2_XS, use_x_map>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_q_case<GGML_TYPE_IQ2_S, use_x_map>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_q_case<GGML_TYPE_IQ3_XXS, use_x_map>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_q_case<GGML_TYPE_IQ3_S, use_x_map>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_q_case<GGML_TYPE_IQ4_XS, use_x_map>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_q_case<GGML_TYPE_IQ4_NL, use_x_map>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_MXFP4:
            // src1 at Q4 uses the native FP4 instructions, which are Blackwell-only
            if (prec_src1 == GGML_PREC_Q4) {
                mul_mat_q_case<GGML_TYPE_MXFP4, use_x_map, GGML_PREC_Q4>(ctx, args, stream);
                break;
            }
            mul_mat_q_case<GGML_TYPE_MXFP4, use_x_map>(ctx, args, stream);
            break;
        case GGML_TYPE_NVFP4:
            if (prec_src1 == GGML_PREC_Q4) {
                mul_mat_q_case<GGML_TYPE_NVFP4, use_x_map, GGML_PREC_Q4>(ctx, args, stream);
                break;
            }
            mul_mat_q_case<GGML_TYPE_NVFP4, use_x_map>(ctx, args, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

// overrides the src1 precision requested by the graph, "auto" keeps the requested one
static ggml_prec ggml_cuda_mmq_get_prec_env() {
    const char * env_c = getenv("GGML_CUDA_MMQ_PREC");
    if (env_c == nullptr) {
        return GGML_PREC_UNDEFINED;
    }
    std::string env_cpp = env_c;
    for (char & c : env_cpp) {
        c = std::tolower(c);
    }
    if (env_cpp == "q4") {
        return GGML_PREC_Q4;
    }
    if (env_cpp == "q8") {
        return GGML_PREC_Q8;
    }
    if (env_cpp != "auto") {
        GGML_LOG_WARN("%s: Unknown value for GGML_CUDA_MMQ_PREC: '%s'. Available: 'q4', 'q8', 'auto'.\n", __func__, env_cpp.c_str());
    }
    return GGML_PREC_UNDEFINED;
}

// src1 is quantized to Q8_1 unless the FP4 types can use 4-bit activations, in which case they
// default to the native W4A4 instructions on Blackwell.
static bool ggml_cuda_mmq_get_prec_src1_checked(const ggml_tensor * src0, const ggml_tensor * dst, int cc, ggml_prec & result) {
    static const ggml_prec prec_env = ggml_cuda_mmq_get_prec_env();

    ggml_prec prec = prec_env;
    if (prec == GGML_PREC_UNDEFINED) {
        prec = (ggml_prec) ggml_get_op_params_i32(dst, 3);
    }

    // Q4 only for the FP4 types on Blackwell
    if (prec != GGML_PREC_UNDEFINED && prec != GGML_PREC_Q8 && prec != GGML_PREC_Q4) { return false; }
    const bool can_use_q4 = (src0->type == GGML_TYPE_NVFP4 || src0->type == GGML_TYPE_MXFP4) && blackwell_mma_available(cc);
    if (prec == GGML_PREC_Q8 || !can_use_q4) {
        result = GGML_PREC_Q8;
    } else {
        result = GGML_PREC_Q4;
    }
    return true;
}

ggml_prec ggml_cuda_mmq_get_prec_src1(const ggml_tensor * src0, const ggml_tensor * dst, int cc) {
    ggml_prec precision;
    GGML_ASSERT(ggml_cuda_mmq_get_prec_src1_checked(src0, dst, cc, precision));
    return precision;
}

static void ggml_cuda_mmq_clear_padding(const ggml_tensor * src0, cudaStream_t stream) {
    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }
}

size_t ggml_cuda_mmq_input_size(const ggml_tensor * node, const int cc) {
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    const bool use_fp4 = ggml_cuda_mmq_get_prec_src1(src0, node, cc) == GGML_PREC_Q4;
    const size_t block_size = use_fp4 ? sizeof(block_fp4_mmq) : sizeof(block_q8_1_mmq);
    const size_t block_values = use_fp4 ? QK_FP4_MMQ : QK8_1_MMQ;
    const int J_max = ggml_cuda_mmq_get_J_max(src0->type, src0->ne[1] % 128 != 0, cc);
    return src1->ne[3]*src1->ne[2]*src1->ne[1]*GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING)*block_size/block_values +
        J_max*sizeof(block_q8_1_mmq);
}

static void ggml_cuda_quantize_mmq_input_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
        const ggml_tensor * src1, ggml_prec prec, size_t size, ggml_cuda_mmq_input & input, const ggml_tensor * logical_src1 = nullptr) {
    input.quantized.alloc(size);
    const bool use_fp4 = prec == GGML_PREC_Q4;
    if (src0->type == GGML_TYPE_NVFP4 && use_fp4) {
        const int cc = ggml_cuda_info().devices[ctx.device].cc;
        const ggml_tensor * scale_src1 = logical_src1 ? logical_src1 : src1;
        const int guard = ggml_cuda_mmq_get_J_max(src0->type, src0->ne[1] % 128 != 0, cc);
        input.scale.alloc(scale_src1->ne[3]*scale_src1->ne[2]*scale_src1->ne[1] + guard);
    }
    const int64_t padded = GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING);
    const int64_t s11 = src1->nb[1]/sizeof(float);
    const int64_t s12 = src1->nb[2]/sizeof(float);
    const int64_t s13 = src1->nb[3]/sizeof(float);
    if (use_fp4) {
        const bool aligned = ggml_cuda_is_aligned(src1, 32);
        if (src0->type == GGML_TYPE_NVFP4 && logical_src1 &&
                (src1->ne[2] != logical_src1->ne[2] || src1->ne[3] != logical_src1->ne[3])) {
            GGML_ASSERT(src1->ne[0] == logical_src1->ne[0] && src1->ne[1] == logical_src1->ne[1]);
            quantize_mmq_nvfp4_compact_cuda((const float *) src1->data, input.quantized.get(), input.scale.ptr,
                aligned, src1->ne[0], s11, s12, s13, padded, src1->ne[1], src1->ne[2], src1->ne[3], logical_src1->ne[2], logical_src1->ne[3], ctx.stream());
        } else {
            quantize_mmq_fp4_cuda((const float *) src1->data, nullptr, input.quantized.get(), input.scale.ptr, src0->type,
                aligned, src1->ne[0], s11, s12, s13, padded, src1->ne[1], src1->ne[2], src1->ne[3], ctx.stream());
        }
    } else {
        quantize_mmq_q8_1_cuda((const float *) src1->data, nullptr, input.quantized.get(), src0->type,
            src1->ne[0], s11, s12, s13, padded, src1->ne[1], src1->ne[2], src1->ne[3], ctx.stream());
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_quantize_mmq_input(ggml_backend_cuda_context & ctx, const ggml_tensor * node,
        size_t size, ggml_cuda_mmq_input & input, const ggml_tensor * logical_src1) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    ggml_cuda_quantize_mmq_input_impl(ctx, node->src[0], node->src[1],
        ggml_cuda_mmq_get_prec_src1(node->src[0], node, cc), size, input, logical_src1);
}

size_t ggml_cuda_mmq_id_input_size(const ggml_tensor * node, const int cc) {
    const ggml_tensor * weight = node->src[0];
    const ggml_tensor * input = node->src[1];
    const bool use_fp4 = ggml_cuda_mmq_get_prec_src1(weight, node, cc) == GGML_PREC_Q4;
    const size_t block_size = use_fp4 ? sizeof(block_fp4_mmq) : sizeof(block_q8_1_mmq);
    const size_t block_values = use_fp4 ? QK_FP4_MMQ : QK8_1_MMQ;
    const int J_max = ggml_cuda_mmq_get_J_max(weight->type, weight->ne[1] % 128 != 0, cc);
    return input->ne[2]*node->src[2]->ne[0]*GGML_PAD(input->ne[0], MATRIX_ROW_PADDING)*block_size/block_values +
        J_max*sizeof(block_q8_1_mmq);
}

void ggml_cuda_prepare_mmq_id_routes(ggml_backend_cuda_context & ctx, const ggml_tensor * node,
        ggml_cuda_mmq_id_input & input, size_t guard) {
    const ggml_tensor * src1 = node->src[1];
    const ggml_tensor * ids = node->src[2];
    const int64_t ne11 = src1->ne[1], ne12 = src1->ne[2], ne02 = node->src[0]->ne[2];
    const int64_t n_expert_used = ids->ne[0], ne_get_rows = ne12*n_expert_used;
    GGML_ASSERT(node->ne[1] == n_expert_used);
    GGML_ASSERT(ids->nb[0] == ggml_element_size(ids));
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const size_t tile_guard = ggml_cuda_mmq_get_J_max(node->src[0]->type, node->src[0]->ne[1] % 128 != 0, cc);
    guard = std::max(guard, tile_guard);
    GGML_ASSERT(ne_get_rows > 0 && uint64_t(ne_get_rows) <= SIZE_MAX - guard);
    input.guard = guard;
    input.ids_src1.alloc(ne_get_rows);
    input.ids_dst.alloc(ne_get_rows + guard);
    input.expert_bounds.alloc(ne02 + 1);
    const int si1 = ids->nb[1]/ggml_element_size(ids);
    const int sis1 = src1->nb[2]/src1->nb[1];
    const bool dedup_bcast = ne11 == 1 && n_expert_used > 1;
    ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, input.ids_src1.ptr, input.ids_dst.ptr, input.expert_bounds.ptr,
        ne02, ne12, n_expert_used, ne11, si1, sis1, /*write_inverse =*/ dedup_bcast, ctx.stream());
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_prepare_mmq_id_input(ggml_backend_cuda_context & ctx, const ggml_tensor * node,
        size_t size, ggml_cuda_mmq_id_input & input, const ggml_cuda_mmq_id_input * routes) {
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    const ggml_tensor * ids = node->src[2];
    const int64_t ne10 = src1->ne[0], ne11 = src1->ne[1], ne12 = src1->ne[2];
    const size_t ts_src1 = sizeof(float);
    const float * src1_d = (const float *) src1->data;
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const bool use_native_fp4 = ggml_cuda_mmq_get_prec_src1(src0, node, cc) == GGML_PREC_Q4;
    cudaStream_t stream = ctx.stream();

    const int64_t n_expert_used = ids->ne[0];
    GGML_ASSERT(node->ne[1] == n_expert_used);

    GGML_ASSERT(!routes || !routes->routes);
    input.routes = routes;
    if (!routes) { ggml_cuda_prepare_mmq_id_routes(ctx, node, input); }
    const auto & mapped = routes ? *routes : input;
    const bool dedup_bcast = ne11 == 1 && n_expert_used > 1;

    input.quantized.alloc(size);
    if (src0->type == GGML_TYPE_NVFP4 && use_native_fp4) {
        input.scale.alloc(ne12*n_expert_used + mapped.guard);
    }

    const int64_t ne11_flat = ne12*n_expert_used;
    const int64_t ne12_flat = 1;
    const int64_t ne13_flat = 1;

    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;

        if (use_native_fp4) {
            static constexpr size_t align_float8 = 32;
            const bool use_aligned_float8 = ggml_cuda_is_aligned(src1, align_float8);
            if (dedup_bcast) {
                quantize_scatter_mmq_fp4_cuda(src1_d, mapped.ids_src1.ptr, input.quantized.ptr, input.scale.ptr, src0->type, use_aligned_float8, ne10,
                                        /*stride_token=*/s12, ne10_padded, ne12, ne11_flat, n_expert_used, stream);
            } else {
                quantize_mmq_fp4_cuda(src1_d, mapped.ids_src1.ptr, input.quantized.ptr, input.scale.ptr, src0->type, use_aligned_float8, ne10, s11, s12, s13,
                                        ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
            }
        } else if (dedup_bcast) {
            quantize_scatter_mmq_q8_1_cuda(src1_d, mapped.ids_src1.ptr, input.quantized.ptr, src0->type, ne10,
                                    /*stride_token=*/s12, ne10_padded, ne12, ne11_flat, n_expert_used, stream);
        } else {
            quantize_mmq_q8_1_cuda(src1_d, mapped.ids_src1.ptr, input.quantized.ptr, src0->type, ne10, s11, s12, s13,
                                   ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        }
        CUDA_CHECK(cudaGetLastError());
    }
}

static void ggml_cuda_mul_mat_q_prepared(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mmq_input * input, const ggml_cuda_mmq_id_input * routed_input, const int64_t * quantized_ne = nullptr) {
    GGML_ASSERT(!ids || !input);
    GGML_ASSERT(!routed_input || ids);
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    const char  * src0_d = (const char  *) src0->data;
    float       *  dst_d = (float       *)  dst->data;

    ggml_cuda_mmq_clear_padding(src0, stream);

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const bool fallback = ne01 % 128 != 0;

    const ggml_prec prec_src1 = ggml_cuda_mmq_get_prec_src1(src0, dst, cc);

    const bool use_native_fp4 = prec_src1 == GGML_PREC_Q4;
    const size_t y_block_size       = use_native_fp4 ? sizeof(block_fp4_mmq) : sizeof(block_q8_1_mmq);
    const size_t y_values_per_block = use_native_fp4 ? QK_FP4_MMQ            : QK8_1_MMQ;

    GGML_ASSERT(!quantized_ne || (!ids && input && quantized_ne[0] == src1->ne[0]));
    if (!ids) {
        ggml_cuda_mmq_input local_input(ctx.pool());
        ggml_tensor packed = *src1;
        if (quantized_ne) {
            GGML_ASSERT(quantized_ne[0] == ne10 && quantized_ne[1] == ne11);
            GGML_ASSERT(!use_native_fp4 || src0->type != GGML_TYPE_NVFP4 || input->scale.ptr);
            for (int d = 2; d < GGML_MAX_DIMS; ++d) {
                GGML_ASSERT(quantized_ne[d] == 1 || quantized_ne[d] == src1->ne[d]);
                packed.ne[d] = quantized_ne[d];
            }
        }
        if (!input) {
            for (int d = 2; d < GGML_MAX_DIMS; ++d) {
                if (packed.nb[d] == 0) { packed.ne[d] = 1; }
            }
            const int J_max = ggml_cuda_mmq_get_J_max(src0->type, fallback, cc);
            const ggml_tensor * allocation_src1 = use_native_fp4 && src0->type == GGML_TYPE_NVFP4 ? src1 : &packed;
            const size_t size = allocation_src1->ne[3]*allocation_src1->ne[2]*ne11*ne10_padded*y_block_size/y_values_per_block +
                J_max*sizeof(block_q8_1_mmq);
            ggml_cuda_quantize_mmq_input_impl(ctx, src0, &packed, prec_src1, size, local_input, src1);
            input = &local_input;
        }

        // Stride depends on quantization format
        const int64_t s12 = use_native_fp4 ?
                                ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int)) :
                                ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
        const int64_t stride_channel_y = packed.ne[2] == ne12 ? s12 : 0;
        const int64_t stride_sample_y  = packed.ne[3] == ne13 ? packed.ne[2]*s12 : 0;

        const mmq_args args = {
            src0_d, src0->type, (const int *) input->quantized.ptr, nullptr, nullptr, dst_d,
            src0->type == GGML_TYPE_NVFP4 && use_native_fp4 ? input->scale.ptr : nullptr,
            ne00, ne01, ne1, s01, ne11, s1,
            ne02, ne12, s02, stride_channel_y, s2,
            ne03, ne13, s03, stride_sample_y, s3,
            ne1, ne1};
        ggml_cuda_mul_mat_q_switch_type<false>(ctx, args, stream, prec_src1);
        return;
    }

    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;
    ggml_cuda_mmq_id_input local_input(ctx.pool());
    if (!routed_input) {
        ggml_cuda_prepare_mmq_id_input(ctx, dst, ggml_cuda_mmq_id_input_size(dst, cc), local_input);
        routed_input = &local_input;
    }

    static_assert(QK_FP4_MMQ == 8 * QK_MXFP4, "QK_FP4_MMQ needs to be 8 * QK_MXFP4");
    const int64_t s12 = use_native_fp4 ? ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int)) :
                                         ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13 = ne12*s12;

    // Each expert only sees ne12*n_expert_used/ne02 tokens on average.
    // On RDNA3 and RDNA4 it is faster to pick the tile size against this value instead of ne12.
    int64_t ncols_opt = ne12;
    if (GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
        ncols_opt = (ne12*n_expert_used + ne02 - 1) / ne02;
    }

    // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA grid.
    const auto & mapped = routed_input->routes ? *routed_input->routes : *routed_input;
    const mmq_args args = {
        src0_d, src0->type, (const int *) routed_input->quantized.ptr, mapped.ids_dst.ptr, mapped.expert_bounds.ptr, dst_d,
        routed_input->scale.ptr,
        ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
        ne02, ne02, s02, s12, s2,
        ne03, ne13, s03, s13, s3,
        ne12, ncols_opt};

    ggml_cuda_mul_mat_q_switch_type<false>(ctx, args, stream, prec_src1);
}

static constexpr size_t MMQ_ID_PAIR_GUARD = 128;

bool ggml_cuda_should_fuse_mmq_id(const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * glu, const int cc, const size_t smpbo) {
    const ggml_tensor * weight = up->src[0];
    const ggml_tensor * input  = up->src[1];
    const ggml_tensor * ids    = up->src[2];
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || !turing_mma_available(cc) || !ggml_is_quantized(weight->type) ||
            up->type != GGML_TYPE_F32 || gate->type != GGML_TYPE_F32 || glu->type != GGML_TYPE_F32 ||
            input->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 ||
            gate->src[0]->type != weight->type || !ggml_are_same_layout(weight, gate->src[0]) ||
            gate->src[1] != input || gate->src[2] != ids ||
            glu->src[0] != gate || glu->src[1] != up || ggml_get_op_params_i32(glu, 1) != 0) {
        return false;
    }
    switch (ggml_get_glu_op(glu)) {
        case GGML_GLU_OP_SWIGLU:
        case GGML_GLU_OP_GEGLU:
        case GGML_GLU_OP_REGLU:
            break;
        case GGML_GLU_OP_SWIGLU_CLAMP: {
            const float limit = ggml_get_op_params_f32(glu, 3);
            if (!std::isfinite(limit) || limit <= 0.0f) {
                return false;
            }
            break;
        }
        default:
            return false;
    }
    const ggml_tensor * tensors[] = { weight, gate->src[0], input, ids, up, gate, glu };
    for (const ggml_tensor * tensor : tensors) {
        const size_t ts = ggml_type_size(tensor->type);
        uint64_t extent = 0;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            if (tensor->ne[d] <= 0 || tensor->ne[d] > INT_MAX || tensor->nb[d] == 0 || tensor->nb[d] % ts != 0 || tensor->nb[d]/ts > INT_MAX) {
                return false;
            }
            const uint64_t stride = tensor->nb[d]/ts;
            const uint64_t count = d == 0 ? (tensor->ne[0]/ggml_blck_size(tensor->type)) : tensor->ne[d];
            if (count == 0 || (count - 1) > (INT_MAX - extent)/std::max(stride, uint64_t(1))) {
                return false;
            }
            extent += (count - 1)*stride;
        }
    }
    const ggml_prec prec = ggml_cuda_mmq_get_prec_src1(weight, up, cc);
    if ((weight->type == GGML_TYPE_MXFP4 || weight->type == GGML_TYPE_NVFP4) && prec != GGML_PREC_Q8) {
        return false;
    }
    const int64_t tokens = input->ne[2];
    const int64_t used   = ids->ne[0];
    const int64_t experts = weight->ne[2];
    const size_t ts = ggml_type_size(weight->type);
    if (weight->ne[0] % MMQ_ITER_K != 0 || weight->ne[3] != 1 || input->ne[3] != 1 ||
            ids->ne[2] != 1 || ids->ne[3] != 1 || ids->ne[1] != tokens ||
            input->ne[0] != weight->ne[0] || (input->ne[1] != 1 && input->ne[1] != used) ||
            used > experts || used >= (1 << 10) || tokens >= (1 << 22) || experts > 65535 ||
            tokens > INT_MAX/used || weight->ne[1] > INT_MAX/(2*tokens*used) ||
            weight->ne[1] > SIZE_MAX/sizeof(float)/2/(tokens*used) ||
            weight->nb[0] != ts || weight->nb[1]/ts < weight->ne[0]/ggml_blck_size(weight->type) ||
            weight->nb[1]/ts > INT_MAX/weight->ne[1] || weight->nb[2]/ts < (weight->nb[1]/ts)*weight->ne[1] ||
            input->nb[0] != sizeof(float) || !ggml_cuda_is_aligned(input, 16) ||
            input->nb[2] % input->nb[1] != 0 || ids->nb[0] != sizeof(int32_t) ||
            !ggml_is_contiguous(up) || !ggml_is_contiguous(gate) || !ggml_is_contiguous(glu) ||
            !ggml_are_same_shape(up, gate) || !ggml_are_same_shape(up, glu) ||
            up->ne[0] != weight->ne[1] || up->ne[1] != used || up->ne[2] != tokens || up->ne[3] != 1) {
        return false;
    }
    const int64_t rows = tokens*used;
    const int64_t padded_k = GGML_PAD(input->ne[0], MATRIX_ROW_PADDING);
    const size_t input_row_size = padded_k*sizeof(block_q8_1_mmq)/QK8_1_MMQ;
    if (uint64_t(rows) > (SIZE_MAX - MMQ_ID_PAIR_GUARD*sizeof(block_q8_1_mmq))/input_row_size ||
            uint64_t(rows) > SIZE_MAX/sizeof(int32_t) - MMQ_ID_PAIR_GUARD) {
        return false;
    }
    if (tokens*sizeof(int32_t) > smpbo || rows > INT_MAX/(padded_k/QK8_1_MMQ*sizeof(block_q8_1_mmq)/sizeof(int)) ||
            tokens <= get_mmvq_mmid_max_batch(weight->type, cc) ||
            !ggml_cuda_should_use_mmq(weight->type, cc, tokens, experts) ||
            ggml_cuda_mmq_get_prec_src1(weight, up, cc) != ggml_cuda_mmq_get_prec_src1(gate->src[0], gate, cc)) {
        return false;
    }
    const bool fallback = (2*weight->ne[1]) % 128 != 0;
    const int J = mmq_get_J(weight->type, fallback, prec, cc, smpbo, tokens);
    if (J == 0) {
        return false;
    }
    const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(weight->type, J, fallback, cc, prec);
    if (weight->ne[1] % config.I != 0) {
        return false;
    }
    const int64_t tiles_i = 2*weight->ne[1]/config.I;
    const int64_t tiles_j = (tokens + J - 1)/J;
    const int64_t blocks_k = weight->ne[0]/ggml_blck_size(weight->type);
    const int64_t max_tiles = std::min(int64_t(INT_MAX/100), ((int64_t(1) << 30) - 1)/blocks_k);
    return tiles_i <= max_tiles/tiles_j/experts;
}

static void ggml_cuda_mmq_pair_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * up,
        const ggml_tensor * gate, ggml_tensor * dst, float * scratch, int64_t bank_rows) {
    ggml_tensor up_view = *up;
    ggml_tensor gate_view = *gate;
    up_view.buffer = gate_view.buffer = nullptr;
    up_view.view_src = gate_view.view_src = nullptr;
    up_view.view_offs = gate_view.view_offs = 0;
    up_view.data = scratch;
    gate_view.data = scratch + bank_rows;
    up_view.nb[1] = gate_view.nb[1] = 2*bank_rows*sizeof(float);
    for (int d = 2; d < GGML_MAX_DIMS; ++d) {
        up_view.nb[d] = gate_view.nb[d] = up_view.nb[d - 1]*up_view.ne[d - 1];
    }
    ggml_tensor glu_view = *dst;
    glu_view.src[0] = &gate_view;
    glu_view.src[1] = &up_view;
    switch (ggml_get_glu_op(dst)) {
        case GGML_GLU_OP_SWIGLU:       ggml_cuda_op_swiglu(ctx, &glu_view); break;
        case GGML_GLU_OP_GEGLU:        ggml_cuda_op_geglu(ctx, &glu_view); break;
        case GGML_GLU_OP_REGLU:        ggml_cuda_op_reglu(ctx, &glu_view); break;
        case GGML_GLU_OP_SWIGLU_CLAMP: ggml_cuda_op_swiglu_clamp(ctx, &glu_view); break;
        default:                      GGML_ABORT("unsupported paired MMQ GLU");
    }
}

void ggml_cuda_mul_mat_id_q_pair(ggml_backend_cuda_context & ctx, const ggml_tensor * up, const ggml_tensor * gate, ggml_tensor * dst) {
    const ggml_tensor * src0 = up->src[0];
    const ggml_tensor * src1 = up->src[1];
    const ggml_tensor * ids  = up->src[2];
    const int64_t rows = src1->ne[2]*ids->ne[0];
    const int64_t padded_k = GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING);
    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), rows + MMQ_ID_PAIR_GUARD);
    ggml_cuda_pool_alloc<int32_t> bounds(ctx.pool(), src0->ne[2] + 1);
    const bool dedup = src1->ne[1] == 1 && ids->ne[0] > 1;
    ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.ptr, ids_dst.ptr, bounds.ptr,
        src0->ne[2], src1->ne[2], ids->ne[0], src1->ne[1], ids->nb[1]/sizeof(int32_t), src1->nb[2]/src1->nb[1], dedup, stream);
    CUDA_CHECK(cudaGetLastError());
    // Tile loads include unused columns past the last expert.
    const size_t input_size = rows*padded_k*sizeof(block_q8_1_mmq)/QK8_1_MMQ + MMQ_ID_PAIR_GUARD*sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> input(ctx.pool(), input_size);
    if (dedup) {
        quantize_scatter_mmq_q8_1_cuda((const float *) src1->data, ids_src1.ptr, input.ptr, src0->type,
            src1->ne[0], src1->nb[2]/sizeof(float), padded_k, src1->ne[2], rows, ids->ne[0], stream);
    } else {
        quantize_mmq_q8_1_cuda((const float *) src1->data, ids_src1.ptr, input.ptr, src0->type, src1->ne[0],
            src1->nb[1]/sizeof(float), src1->nb[2]/sizeof(float), src1->nb[3]/sizeof(float), padded_k, rows, 1, 1, stream);
    }
    CUDA_CHECK(cudaGetLastError());
    const size_t ts = ggml_type_size(src0->type);
    const int64_t bank_rows = src0->ne[1];
    ggml_cuda_pool_alloc<float> scratch(ctx.pool(), 2*bank_rows*rows);
    mmq_args args = {
        (const char *) src0->data, src0->type, (const int *) input.ptr, ids_dst.ptr, bounds.ptr, scratch.ptr, nullptr,
        src0->ne[0], 2*bank_rows, rows, src0->nb[1]/ts, rows, 2*bank_rows,
        src0->ne[2], src0->ne[2], src0->nb[2]/ts, 0, 0,
        1, 1, 0, 0, 0, src1->ne[2], src1->ne[2] };
    args.x_second = (const char *) gate->src[0]->data;
    args.nrows_bank = bank_rows;
    ggml_cuda_mul_mat_q_switch_type<false>(ctx, args, stream, GGML_PREC_Q8);
    CUDA_CHECK(cudaGetLastError());
    ggml_cuda_mmq_pair_glu(ctx, up, gate, dst, scratch.ptr, bank_rows);
    CUDA_CHECK(cudaGetLastError());
}

static void ggml_cuda_mul_mat_q_impl(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const void * src0_secondary,
        const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const int32_t * source_map, int32_t source_split,
        const int32_t * source_wait_class, const uint32_t * stage_ready,
        const ggml_cuda_mmid_execution * execution = nullptr) {
    GGML_ASSERT(!execution || (ids && !source_map && !src0_secondary));
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));
    GGML_ASSERT((source_map == nullptr) == (src0_secondary == nullptr));
    GGML_ASSERT(source_map == nullptr || (ids != nullptr && source_split > 0));

    const char  * src0_d = (const char  *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (!execution && ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const bool fallback = ne01 % 128 != 0;
    const int mmq_max_tile_y = ggml_cuda_mmq_get_J_max(src0->type, fallback, cc);
    GGML_ASSERT(mmq_max_tile_y > 0);

    const ggml_prec prec_src1 = ggml_cuda_mmq_get_prec_src1(src0, dst, cc);

    const bool use_native_fp4 = prec_src1 == GGML_PREC_Q4;
    const size_t y_block_size       = use_native_fp4 ? sizeof(block_fp4_mmq) : sizeof(block_q8_1_mmq);
    const size_t y_values_per_block = use_native_fp4 ? QK_FP4_MMQ            : QK8_1_MMQ;

    if (!ids) {
        const size_t nbytes_src1_q8_1 = ne13*ne12 * ne11*ne10_padded * y_block_size/y_values_per_block +
            mmq_max_tile_y * sizeof(block_q8_1_mmq);
        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);
        ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());
        if (src0->type == GGML_TYPE_NVFP4 && use_native_fp4) {
            src1_scale.alloc(ne13*ne12*ne11 + mmq_max_tile_y);
        }

        {
            const int64_t s11 = src1->nb[1] / ts_src1;
            const int64_t s12 = src1->nb[2] / ts_src1;
            const int64_t s13 = src1->nb[3] / ts_src1;
            if (use_native_fp4) {
                static constexpr size_t align_float8 = 32;
                const bool use_aligned_float8 = ggml_cuda_is_aligned(src1, align_float8);
                static_assert(sizeof(block_fp4_mmq) == 4 * sizeof(block_q8_1));
                quantize_mmq_fp4_cuda(src1_d, nullptr, src1_q8_1.get(), src1_scale.ptr, src0->type, use_aligned_float8, ne10, s11, s12, s13, ne10_padded,
                                        ne11, ne12, ne13, stream);

            } else {
                quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,
                                       ne11, ne12, ne13, stream);
            }
            CUDA_CHECK(cudaGetLastError());
        }

        // Stride depends on quantization format
        const int64_t s12 = use_native_fp4 ?
                                ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int)) :
                                ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
        const int64_t s13 = ne12*s12;

        const mmq_args args = {
            src0_d, src0->type, (const int *) src1_q8_1.ptr, nullptr, nullptr, dst_d,
            src0->type == GGML_TYPE_NVFP4 && use_native_fp4 ? src1_scale.ptr : nullptr,
            ne00, ne01, ne1, s01, ne11, s1,
            ne02, ne12, s02, s12, s2,
            ne03, ne13, s03, s13, s3,
            ne1, ne1};
        ggml_cuda_mul_mat_q_switch_type<false>(ctx, args, stream, prec_src1);
        return;
    }

    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const int64_t n_expert_used = ids->ne[0];
    ggml_cuda_mmq_routed_resources resources;
    GGML_ASSERT(ggml_cuda_mmq_routed_requirements(ctx.device, dst, resources, execution != nullptr));
    const int64_t ne_get_rows = resources.rows;
    GGML_ASSERT(ne1 == n_expert_used);
    GGML_ASSERT(ne_get_rows > 0 && (uint64_t) ne_get_rows <= SIZE_MAX - (size_t) mmq_max_tile_y);
    const size_t ne_get_rows_padded = resources.padded_rows;

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_get_rows_padded);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), ne02 + 1);
    if (execution) { CUDA_CHECK(cudaMemsetAsync(ids_dst.get(), 0, ne_get_rows_padded*sizeof(int32_t), stream)); }

    // gate/up activations are broadcast across experts (ne11 == 1): quantize each token once and
    // scatter to its slots. ids_src1 then holds the inverse map (token slot -> compact row).
    const bool dedup_bcast = ne11 == 1 && n_expert_used > 1;

    {
        GGML_ASSERT(ids->nb[0] == ggml_element_size(ids));
        const int si1  = ids->nb[1] / ggml_element_size(ids);
        const int sis1 = nb12 / nb11;

        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
            ne02, ne12, n_expert_used, ne11, si1, sis1, /*write_inverse =*/ dedup_bcast, stream, execution);
        CUDA_CHECK(cudaGetLastError());
    }

    const size_t nbytes_src1_q8_1 = resources.quantized_bytes;
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);
    ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());
    if (resources.scale_count) { src1_scale.alloc(resources.scale_count); }

    const int64_t ne11_flat = ne12*n_expert_used;
    const int64_t ne12_flat = 1;
    const int64_t ne13_flat = 1;

    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;

        if (use_native_fp4) {
            static constexpr size_t align_float8 = 32;
            const bool use_aligned_float8 = ggml_cuda_is_aligned(src1, align_float8);
            if (dedup_bcast) {
                quantize_scatter_mmq_fp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src1_scale.ptr, src0->type, use_aligned_float8, ne10,
                                        /*stride_token=*/s12, ne10_padded, ne12, ne11_flat, n_expert_used, stream, execution != nullptr);
            } else {
                quantize_mmq_fp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src1_scale.ptr, src0->type, use_aligned_float8, ne10, s11, s12, s13,
                                        ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
            }
        } else if (dedup_bcast) {
            quantize_scatter_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10,
                                    /*stride_token=*/s12, ne10_padded, ne12, ne11_flat, n_expert_used, stream, execution != nullptr);
        } else {
            quantize_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
                                   ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    static_assert(QK_FP4_MMQ == 8 * QK_MXFP4, "QK_FP4_MMQ needs to be 8 * QK_MXFP4");
    const int64_t s12 = use_native_fp4 ? ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int)) :
                                         ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13 = ne12*s12;

    // Each expert only sees ne12*n_expert_used/ne02 tokens on average.
    // On RDNA3 and RDNA4 it is faster to pick the tile size against this value instead of ne12.
    int64_t ncols_opt = ne12;
    if (GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
        ncols_opt = (ne12*n_expert_used + ne02 - 1) / ne02;
    }

    // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA grid.
    mmq_args args = {
        src0_d, src0->type, (const int *) src1_q8_1.get(), ids_dst.get(), expert_bounds.get(), dst_d,
        src1_scale.ptr,
        ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
        ne02, ne02, s02, s12, s2,
        ne03, ne13, s03, s13, s3,
        ne12, ncols_opt};

    args.x_secondary = (const char *) src0_secondary;
    args.x_channel_map = source_map;
    args.x_channel_split = source_split;
    args.x_wait_class = source_wait_class;
    args.x_stage_ready = stage_ready;
    args.x_expert_sources = execution ? execution->expert_sources : nullptr;
    if (execution) { args.ncols_max = ne_get_rows; }
    if (source_map || args.x_expert_sources) {
        ggml_cuda_mul_mat_q_switch_type<true>(ctx, args, stream, prec_src1);
    } else {
        ggml_cuda_mul_mat_q_switch_type<false>(ctx, args, stream, prec_src1);
    }
}

void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mmq_input * input, const ggml_cuda_mmq_id_input * routed_input,
        const ggml_cuda_mmid_execution * execution, const int64_t * quantized_ne) {
    GGML_ASSERT(!execution || (ids && !input && !routed_input));
    GGML_ASSERT(!ids || !input);
    GGML_ASSERT(!routed_input || ids);
    if (input || routed_input) {
        ggml_cuda_mul_mat_q_prepared(ctx, src0, src1, ids, dst, input, routed_input, quantized_ne);
        return;
    }
    ggml_cuda_mul_mat_q_impl(ctx, src0, nullptr, src1, ids, dst, nullptr, 0, nullptr, nullptr, execution);
}

void ggml_cuda_mul_mat_q_mapped(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const void * src0_secondary,
        const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const int32_t * source_map, int32_t source_split,
        const int32_t * source_wait_class, const uint32_t * stage_ready) {
    GGML_ASSERT(src0_secondary != nullptr && source_map != nullptr && source_split > 0);
    GGML_ASSERT((source_wait_class == nullptr) == (stage_ready == nullptr));
    GGML_ASSERT(src0->type != GGML_TYPE_MXFP4 && src0->type != GGML_TYPE_NVFP4);
    ggml_cuda_mul_mat_q_impl(
        ctx, src0, src0_secondary, src1, ids, dst,
        source_map, source_split, source_wait_class, stage_ready);
}

struct ggml_cuda_mmq_mmid_prepared {
    explicit ggml_cuda_mmq_mmid_prepared(ggml_cuda_pool & pool) :
        ids_src1(pool), ids_dst(pool), expert_bounds(pool), src1_q8_1(pool), src1_scale(pool), pair_scratch(pool) {
    }

    ggml_cuda_pool_alloc<int32_t> ids_src1;
    ggml_cuda_pool_alloc<int32_t> ids_dst;
    ggml_cuda_pool_alloc<int32_t> expert_bounds;
    ggml_cuda_pool_alloc<char> src1_q8_1;
    ggml_cuda_pool_alloc<float> src1_scale;
    ggml_cuda_pool_alloc<float> pair_scratch;
    mmq_args args = {};
    int64_t n_experts = 0;
    ggml_tensor pair_up = {}, pair_gate = {}, pair_glu = {};
    bool paired = false;
};

static ggml_cuda_mmq_mmid_prepared * ggml_cuda_mmq_mmid_prepare_impl(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        ggml_tensor * dst,
        size_t minimum_guard) {
    if (src0 == nullptr || src1 == nullptr || ids == nullptr || dst == nullptr ||
            src1->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_F32 ||
            src0->type == GGML_TYPE_MXFP4 || src0->type == GGML_TYPE_NVFP4) {
        return nullptr;
    }

    GGML_TENSOR_BINARY_OP_LOCALS;
    if (ne13 != 1 || nb12 % nb11 != 0 || nb2 % nb1 != 0 || ids->nb[0] != ggml_element_size(ids)) {
        return nullptr;
    }

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const bool fallback = ne01 % 128 != 0;
    const int mmq_max_tile_y = ggml_cuda_mmq_get_J_max(src0->type, fallback, cc);
    const size_t input_guard = std::max(size_t(std::max(mmq_max_tile_y, 0)), minimum_guard);
    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;
    if (mmq_max_tile_y <= 0 || ne1 != n_expert_used || ne_get_rows <= 0 ||
            static_cast<uint64_t>(ne_get_rows) > SIZE_MAX - input_guard) {
        return nullptr;
    }

    auto * prepared = new (std::nothrow) ggml_cuda_mmq_mmid_prepared(ctx.pool());
    if (prepared == nullptr) {
        return nullptr;
    }
    const size_t ne_get_rows_padded = static_cast<size_t>(ne_get_rows) + input_guard;
    prepared->ids_src1.alloc(ne_get_rows);
    prepared->ids_dst.alloc(ne_get_rows_padded);
    prepared->expert_bounds.alloc(ne02 + 1);

    const int si1 = ids->nb[1] / ggml_element_size(ids);
    const int sis1 = nb12 / nb11;
    const bool dedup_bcast = ne11 == 1 && n_expert_used > 1;
    ggml_cuda_launch_mm_ids_helper(
        static_cast<const int32_t *>(ids->data), prepared->ids_src1.get(), prepared->ids_dst.get(),
        prepared->expert_bounds.get(), ne02, ne12, n_expert_used, ne11, si1, sis1, dedup_bcast, ctx.stream());
    CUDA_CHECK(cudaGetLastError());

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const size_t nbytes_src1_q8_1 = ne12*n_expert_used*ne10_padded * sizeof(block_q8_1_mmq)/QK8_1_MMQ +
        input_guard * sizeof(block_q8_1_mmq);
    prepared->src1_q8_1.alloc(nbytes_src1_q8_1);

    const float * src1_d = static_cast<const float *>(src1->data);
    const int64_t s11 = src1->nb[1] / ggml_type_size(src1->type);
    const int64_t s12_src = src1->nb[2] / ggml_type_size(src1->type);
    const int64_t s13_src = src1->nb[3] / ggml_type_size(src1->type);
    const int64_t ne11_flat = ne12 * n_expert_used;
    if (dedup_bcast) {
        quantize_scatter_mmq_q8_1_cuda(
            src1_d, prepared->ids_src1.get(), prepared->src1_q8_1.get(), src0->type,
            ne10, s12_src, ne10_padded, ne12, ne11_flat, n_expert_used, ctx.stream());
    } else {
        quantize_mmq_q8_1_cuda(
            src1_d, prepared->ids_src1.get(), prepared->src1_q8_1.get(), src0->type,
            ne10, s11, s12_src, s13_src, ne10_padded, ne11_flat, 1, 1, ctx.stream());
    }
    CUDA_CHECK(cudaGetLastError());

    const int64_t ts_src0 = ggml_type_size(src0->type);
    const int64_t ts_dst = ggml_type_size(dst->type);
    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1 = dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2 = dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3 = dst->nb[3] / ts_dst;
    const int64_t s12 = ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13 = ne12 * s12;
    int64_t ncols_opt = ne12;
    if (GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
        ncols_opt = (ne12*n_expert_used + ne02 - 1) / ne02;
    }

    prepared->args = {
        static_cast<const char *>(src0->data), src0->type,
        reinterpret_cast<const int *>(prepared->src1_q8_1.get()),
        prepared->ids_dst.get(), prepared->expert_bounds.get(), static_cast<float *>(dst->data), nullptr,
        ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
        ne02, ne02, s02, s12, s2,
        ne03, ne13, s03, s13, s3,
        ne12, ncols_opt,
    };
    prepared->n_experts = ne02;
    return prepared;
}

ggml_cuda_mmq_mmid_prepared * ggml_cuda_mmq_mmid_prepare(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * ids, ggml_tensor * dst) {
    return ggml_cuda_mmq_mmid_prepare_impl(ctx, src0, src1, ids, dst, 0);
}

ggml_cuda_mmq_mmid_prepared * ggml_cuda_mmq_mmid_prepare_pair(
        ggml_backend_cuda_context & ctx, const ggml_tensor * up, const ggml_tensor * gate, ggml_tensor * glu) {
    if (!up || !gate || !glu || up->op != GGML_OP_MUL_MAT_ID || gate->op != GGML_OP_MUL_MAT_ID ||
            glu->op != GGML_OP_GLU || !up->src[0] || !up->src[1] || !up->src[2] ||
            !gate->src[0] || !gate->src[1] || !gate->src[2]) { return nullptr; }
    const auto & device = ggml_cuda_info().devices[ctx.device];
    if (!ggml_cuda_should_fuse_mmq_id(up, gate, glu, device.cc, device.smpbo)) { return nullptr; }
    auto * prepared = ggml_cuda_mmq_mmid_prepare_impl(ctx, up->src[0], up->src[1], up->src[2],
        const_cast<ggml_tensor *>(up), MMQ_ID_PAIR_GUARD);
    if (!prepared) { return nullptr; }
    const int64_t bank_rows = up->src[0]->ne[1];
    const int64_t routed_rows = prepared->args.ncols_dst;
    prepared->pair_scratch.alloc(size_t(2 * bank_rows * routed_rows));
    prepared->args.nrows_x = 2 * bank_rows;
    prepared->args.nrows_dst = 2 * bank_rows;
    prepared->args.dst = prepared->pair_scratch.get();
    prepared->args.nrows_bank = bank_rows;
    prepared->pair_up = *up;
    prepared->pair_gate = *gate;
    prepared->pair_glu = *glu;
    prepared->paired = true;
    return prepared;
}

bool ggml_cuda_mmq_mmid_launch_range(
        ggml_backend_cuda_context & ctx,
        const ggml_cuda_mmq_mmid_prepared * prepared,
        const void * resident_data,
        const void * staging_data,
        const int32_t * source_map,
        int32_t source_split,
        int32_t expert_begin,
        int32_t expert_count,
        int64_t max_rows) {
    if (prepared == nullptr || prepared->paired || resident_data == nullptr || staging_data == nullptr || source_map == nullptr ||
            source_split <= 0 || expert_begin < 0 || expert_count <= 0 ||
            static_cast<int64_t>(expert_begin) + expert_count > prepared->n_experts ||
            max_rows <= 0 || max_rows > prepared->args.ncols_dst) {
        return false;
    }

    mmq_args args = prepared->args;
    args.x = static_cast<const char *>(resident_data);
    args.expert_bounds += expert_begin;
    args.nchannels_x = expert_count;
    args.nchannels_y = expert_count;
    // Routed experts need only the rows present in this wave.
    args.ncols_max = max_rows;
    args.ncols_opt = max_rows;
    args.x_secondary = static_cast<const char *>(staging_data);
    args.x_channel_map = source_map + expert_begin;
    args.x_channel_split = source_split;
    ggml_cuda_mul_mat_q_switch_type<true>(ctx, args, ctx.stream(), GGML_PREC_Q8);
    return cudaGetLastError() == cudaSuccess;
}

bool ggml_cuda_mmq_mmid_launch_pair_range(
        ggml_backend_cuda_context & ctx, const ggml_cuda_mmq_mmid_prepared * prepared,
        const void * resident_up, const void * staging_up, const void * resident_gate, const void * staging_gate,
        const int32_t * source_map, int32_t source_split, int32_t expert_begin, int32_t expert_count, int64_t max_rows) {
    if (!prepared || !prepared->paired || !resident_up || !staging_up || !resident_gate || !staging_gate || !source_map ||
            source_split <= 0 || expert_begin < 0 || expert_count <= 0 ||
            int64_t(expert_begin) + expert_count > prepared->n_experts || max_rows <= 0 || max_rows > prepared->args.ncols_dst) {
        return false;
    }
    mmq_args args = prepared->args;
    args.x = static_cast<const char *>(resident_up);
    args.x_secondary = static_cast<const char *>(staging_up);
    args.x_second = static_cast<const char *>(resident_gate);
    args.x_second_secondary = static_cast<const char *>(staging_gate);
    args.x_channel_map = source_map + expert_begin;
    args.x_channel_split = source_split;
    args.expert_bounds += expert_begin;
    args.nchannels_x = args.nchannels_y = expert_count;
    args.ncols_max = args.ncols_opt = max_rows;
    ggml_cuda_mul_mat_q_switch_type<true>(ctx, args, ctx.stream(), GGML_PREC_Q8);
    return cudaGetLastError() == cudaSuccess;
}

bool ggml_cuda_mmq_mmid_finish_pair(ggml_backend_cuda_context & ctx, const ggml_cuda_mmq_mmid_prepared * prepared) {
    if (!prepared || !prepared->paired) { return false; }
    ggml_tensor glu = prepared->pair_glu;
    ggml_cuda_mmq_pair_glu(ctx, &prepared->pair_up, &prepared->pair_gate, &glu,
        prepared->args.dst, prepared->args.nrows_bank);
    return cudaGetLastError() == cudaSuccess;
}

void ggml_cuda_mmq_mmid_free(ggml_cuda_mmq_mmid_prepared * prepared) {
    delete prepared;
}

static bool ggml_cuda_mmq_type_supported(enum ggml_type type) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    return false;
#endif // GGML_CUDA_FORCE_CUBLAS

    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
// -------------------------------------------------
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
// -------------------------------------------------
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
// -------------------------------------------------
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
            return true;
        default:
            return false;
    }
}

static bool ggml_cuda_should_use_mmq_impl(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts, size_t smpbo) {
    // MMQ tiles require at least 48 KiB per-block shared memory; fall back to BLAS otherwise.
    if (smpbo < 48 * 1024) {
        return false;
    }

    if (turing_mma_available(cc)) {
        return true;
    }

    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        // for MoE, mmq is faster even without native dp4a
        // TODO: check if cards older than pascal might benefit from this as well
        return cc >= GGML_CUDA_CC_PASCAL && n_experts > 0;
    }

#ifdef GGML_CUDA_FORCE_MMQ
    return true;
#endif //GGML_CUDA_FORCE_MMQ

    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

    if (amd_mfma_available(cc)) {
        // As of ROCM 7.0 rocblas/tensile performs very poorly on CDNA3 and hipblaslt (via ROCBLAS_USE_HIPBLASLT)
        // performs better but is currently suffering from a crash on this architecture.
        // TODO: Revisit when hipblaslt is fixed on CDNA3
        if (GGML_CUDA_CC_IS_CDNA3(cc)) {
            return true;
        }
        if (n_experts > 64 || ne11 <= 128) {
            return true;
        }
        if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) {
            return true;
        }
        if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
            return true;
        }
        return false;
    }

    if (amd_wmma_available(cc)) {
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            // High expert counts are almost always better on MMQ due to
            //     the synchronization overhead in the cuBLAS/hipBLAS path:
            // https://github.com/ggml-org/llama.cpp/pull/18202
            if (n_experts >= 64) {
                return true;
            }

            // For some quantization types MMQ can have lower peak TOPS than hipBLAS
            //     so it's only faster for sufficiently small batch sizes:
            switch (type) {
                case GGML_TYPE_Q2_K:
                    return ne11 <= 128;
                case GGML_TYPE_Q6_K:
                    return ne11 <= (GGML_CUDA_CC_IS_RDNA3_0(cc) ? 128 : 256);
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                    return GGML_CUDA_CC_IS_RDNA3_5(cc) || ne11 <= 128;
                default:
                    return true;
            }
        }

        // For RDNA4 MMQ is consistently faster than dequantization + hipBLAS:
        // https://github.com/ggml-org/llama.cpp/pull/18537#issuecomment-3706422301
        return true;
    }

    // gfx900 (Vega 10), gfx909, and gfx90c lack native dp4a, losing to dequant + hipBLAS
    // for dense matrices; keep MMQ only for MoE, where the
    // hipBLAS path is much slower.
    if (cc == GGML_CUDA_CC_VEGA || GGML_CUDA_CC_IS_GCN_APU(cc)) {
        return n_experts > 0;
    }

    // MUSA: the MMQ kernels compute wrong values on PH1 (MTT S5000).
    if (cc == GGML_CUDA_CC_PH1) {
        return false;
    }

    return (!GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
    if (!ggml_cuda_mmq_type_supported(type)) {
        return false;
    }
    const int id = ggml_cuda_get_device();
    return ggml_cuda_should_use_mmq_impl(type, cc, ne11, n_experts, ggml_cuda_info().devices[id].smpbo);
}

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts, size_t smpbo) {
    if (!ggml_cuda_mmq_type_supported(type)) {
        return false;
    }
    return ggml_cuda_should_use_mmq_impl(type, cc, ne11, n_experts, smpbo);
}

bool ggml_cuda_mmq_routed_requirements(int device, const ggml_tensor * dst, ggml_cuda_mmq_routed_resources & resources, bool bound) {
    if (device < 0 || device >= ggml_cuda_info().device_count || !dst || dst->op != GGML_OP_MUL_MAT_ID ||
            !dst->src[0] || !dst->src[1] || !dst->src[2]) { return false; }
    const auto * weight = dst->src[0];
    const auto * input = dst->src[1];
    const auto * ids = dst->src[2];
    if (input->ne[0] <= 0 || input->ne[0] > INT_MAX - MATRIX_ROW_PADDING ||
            ids->ne[0] <= 0 || ids->ne[0] > INT_MAX || ids->ne[1] <= 0 || ids->ne[1] > INT_MAX / ids->ne[0] ||
            !(ggml_cuda_mmid_source_capability_for(weight->type).flags & GGML_CUDA_MMID_SOURCE_MMQ)) { return false; }
    const int cc = ggml_cuda_info().devices[device].cc;
    const int tile_y = ggml_cuda_mmq_get_J_max(weight->type, weight->ne[1] % 128 != 0, cc);
    if (tile_y <= 0) { return false; }
    ggml_prec precision;
    if (!ggml_cuda_mmq_get_prec_src1_checked(weight, dst, cc, precision)) { return false; }
    const size_t block_bytes = precision == GGML_PREC_Q4 ? sizeof(block_fp4_mmq) : sizeof(block_q8_1_mmq);
    const size_t block_values = precision == GGML_PREC_Q4 ? QK_FP4_MMQ : QK8_1_MMQ;
    const size_t padded = size_t(GGML_PAD(input->ne[0], MATRIX_ROW_PADDING));
    ggml_cuda_mmq_routed_resources measured;
    measured.rows = size_t(ids->ne[0]) * size_t(ids->ne[1]);
    if (measured.rows > size_t(INT_MAX - tile_y) || padded / block_values > SIZE_MAX / measured.rows) { return false; }
    measured.padded_rows = measured.rows + tile_y;
    const size_t blocks = padded / block_values * measured.rows;
    const size_t tail = size_t(tile_y) * sizeof(block_q8_1_mmq);
    if (blocks > (SIZE_MAX - tail) / block_bytes) { return false; }
    measured.quantized_bytes = blocks * block_bytes + tail;
    if (weight->type == GGML_TYPE_NVFP4 && precision == GGML_PREC_Q4) { measured.scale_count = measured.padded_rows; }
    int64_t columns_opt = input->ne[2];
    if (GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
        columns_opt = (measured.rows + weight->ne[2] - 1) / weight->ne[2];
    }
    const bool fallback = weight->ne[1] % 128 != 0;
    const int j = mmq_get_J(weight->type, fallback, precision, cc, ggml_cuda_info().devices[device].smpbo, columns_opt);
    if (j <= 0) { return false; }
    const auto config = ggml_cuda_mmq_get_config(weight->type, j, fallback, cc, precision);
    mmq_stream_k_resources scratch;
    if (!mmq_stream_k_requirements(config, cc, ggml_cuda_info().devices[device].nsm,
            weight->ne[1], bound ? measured.rows : size_t(input->ne[2]), weight->ne[2], 1,
            weight->ne[0] / ggml_blck_size(weight->type), scratch)) { return false; }
    measured.fixup_elements = scratch.fixup_elements;
    if (bound) {
        // Smaller expert waves can need fixup when the full-row launch does not.
        for (int tile = 8; tile <= 128; tile += 8) {
            const auto wave = ggml_cuda_mmq_get_config(weight->type, tile, fallback, cc, precision);
            if (wave.type == GGML_TYPE_COUNT || !wave.stream_k || mmq_get_nbytes_shared(wave, cc) > ggml_cuda_info().devices[device].smpbo) { continue; }
            const size_t nsm = size_t(ggml_cuda_info().devices[device].nsm);
            if (wave.I <= 0 || wave.J <= 0 || !nsm || nsm > SIZE_MAX / size_t(wave.I) ||
                    nsm * size_t(wave.I) > SIZE_MAX / size_t(wave.J)) { return false; }
            measured.fixup_elements = std::max(measured.fixup_elements, nsm * size_t(wave.I) * size_t(wave.J));
        }
    }
    resources = measured;
    return true;
}
