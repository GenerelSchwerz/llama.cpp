#include "common.cuh"
#include "ssm-conv.cuh"
#include "unary.cuh"

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float * src0_ptr, const float * src1_ptr,
                                    const float * bias_ptr,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * dst_ptr, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float * GGML_CUDA_RESTRICT bias = bias_ptr;
    float       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

    ggml_cuda_pdl_sync();
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const float * __restrict__ bias,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const float * bias, const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_conv_f32<apply_silu, threads, kNC>, launch_params, src0, src1, bias, src0_nb0, src0_nb1,
                                                                        src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        } else {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                src0, src1, bias, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 9, 15 right now.");
    }
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node, ggml_tensor * silu_dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // conv_x
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
    const bool fuse_bias = bias_add_node != nullptr;
    const bool fuse_silu = silu_dst != nullptr;

    // bias always comes with silu.
    GGML_ASSERT(!fuse_bias || fuse_silu);

    // The bias (when fused) is the non-conv operand of the ADD node.
    const struct ggml_tensor * bias = fuse_bias ? (bias_add_node->src[0] == dst ? bias_add_node->src[1] : bias_add_node->src[0]) : nullptr;

    // When fusing, write to silu_dst (the node downstream references).
    const struct ggml_tensor * out = fuse_silu ? silu_dst : dst;

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = out->ne[1];                 // tokens per sequence
    const int64_t n_s = out->ne[2];                 // number of sequences in the batch

    GGML_ASSERT(out->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * bias_d = fuse_bias ? (const float *) bias->data : nullptr;
    float *       dst_d  = (float *) out->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);
    if (fuse_bias) {
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(ggml_nelements(bias) == nr);
    }

    if (fuse_silu) {
        ssm_conv_f32_cuda<true>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    } else {
        ssm_conv_f32_cuda<false>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    }
}

template <typename T>
static __global__ void dilated_conv_f32(const float * x, const T * weights, const float * residual0, const float * residual1,
                                      float * dst, uint32_t count, uint3 div_channels, uint3 div_tokens, int taps,
                                      int64_t stride_channel, int64_t stride_sequence, int64_t stride_weight,
                                      int64_t first_row, int64_t dilation, bool apply_silu, int other_first) {
    ggml_cuda_pdl_lc();
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    const uint2 channel = fast_div_modulo(i, div_channels);
    const uint2 token = fast_div_modulo(channel.x, div_tokens);
    const float * row = x + channel.y * stride_channel + token.x * stride_sequence + first_row + token.y;
    const T * column = weights + channel.y * stride_weight;
    ggml_cuda_pdl_sync();
    float sum = __fmul_rn(row[0], float(column[0]));
    for (int tap = 1; tap < taps; ++tap) {
        sum = __fadd_rn(sum, __fmul_rn(row[tap * dilation], float(column[tap])));
    }
    if (apply_silu) {
        sum = ggml_cuda_op_silu_single(sum);
    }
    if (residual0) {
        sum = other_first & 1 ? __fadd_rn(residual0[i], sum) : __fadd_rn(sum, residual0[i]);
    }
    if (residual1) {
        sum = other_first & 2 ? __fadd_rn(residual1[i], sum) : __fadd_rn(sum, residual1[i]);
    }
    dst[i] = sum;
}

void ggml_cuda_op_dilated_conv(ggml_backend_cuda_context & ctx, const ggml_cuda_dilated_conv_args & args) {
    const ggml_tensor * source = args.source;
    const ggml_tensor * weights = args.weights;
    const float * x = (const float *) source->data;
    const float * r0 = args.residual[0] ? (const float *) args.residual[0]->data : nullptr;
    const float * r1 = args.residual[1] ? (const float *) args.residual[1]->data : nullptr;
    float * dst = (float *) args.dst->data;
    const uint32_t count = ggml_nelements(args.dst);
    const uint3 channels = init_fastdiv_values(source->ne[1]);
    const uint3 tokens = init_fastdiv_values(count / (source->ne[1] * source->ne[2]));
    const int blocks = (count + 255) / 256;
    const int64_t stride_weight = weights->nb[1] / ggml_element_size(weights);
    const int64_t stride_channel = source->nb[1] / sizeof(float);
    const int64_t stride_sequence = source->nb[2] / sizeof(float);
    const cudaStream_t stream = ctx.stream();
    if (weights->type == GGML_TYPE_F32) {
        dilated_conv_f32<<<blocks, 256, 0, stream>>>(x, (const float *) weights->data, r0, r1, dst, count, channels, tokens, weights->ne[0], stride_channel, stride_sequence, stride_weight, args.first_row, args.dilation, args.apply_silu, args.other_first);
    } else if (weights->type == GGML_TYPE_F16) {
        dilated_conv_f32<<<blocks, 256, 0, stream>>>(x, (const half *) weights->data, r0, r1, dst, count, channels, tokens, weights->ne[0], stride_channel, stride_sequence, stride_weight, args.first_row, args.dilation, args.apply_silu, args.other_first);
    } else {
        GGML_ASSERT(weights->type == GGML_TYPE_BF16);
        dilated_conv_f32<<<blocks, 256, 0, stream>>>(x, (const nv_bfloat16 *) weights->data, r0, r1, dst, count, channels, tokens, weights->ne[0], stride_channel, stride_sequence, stride_weight, args.first_row, args.dilation, args.apply_silu, args.other_first);
    }
}

struct dilated_conv_history_params {
    const float * old;
    const float * current;
    const void * weights;
    const float * residual0;
    const float * residual1;
    float * dst;
    float * state;
    uint32_t count, state_count;
    int taps, history, tokens, channels, sequences, slots;
    int64_t old_stride[3], current_stride[3];
    int64_t weight_stride, state_sequence_stride, slot_stride, first_row, dilation;
    uint3 div_channels, div_tokens, div_history, div_sequences;
    bool apply_silu;
    int other_first;
};

static __device__ __forceinline__ float dilated_conv_history_read(const dilated_conv_history_params & p, int64_t row, int channel, int sequence) {
    if (row < p.history) {
        return p.old[row * p.old_stride[0] + channel * p.old_stride[1] + sequence * p.old_stride[2]];
    }
    return p.current[(row - p.history) * p.current_stride[0] + channel * p.current_stride[1] + sequence * p.current_stride[2]];
}

template <typename T>
static __global__ void dilated_conv_history_f32(dilated_conv_history_params p) {
    ggml_cuda_pdl_lc();
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.count && i >= p.state_count) {
        return;
    }
    ggml_cuda_pdl_sync();
    if (i < p.count) {
        const uint2 channel = fast_div_modulo(i, p.div_channels);
        const uint2 token = fast_div_modulo(channel.x, p.div_tokens);
        const T * column = (const T *) p.weights + channel.y * p.weight_stride;
        const int64_t row = p.first_row + token.y;
        float sum = __fmul_rn(dilated_conv_history_read(p, row, channel.y, token.x), float(column[0]));
        for (int tap = 1; tap < p.taps; ++tap) {
            sum = __fadd_rn(sum, __fmul_rn(dilated_conv_history_read(p, row + tap * p.dilation, channel.y, token.x), float(column[tap])));
        }
        if (p.apply_silu) {
            sum = ggml_cuda_op_silu_single(sum);
        }
        if (p.residual0) {
            sum = p.other_first & 1 ? __fadd_rn(p.residual0[i], sum) : __fadd_rn(sum, p.residual0[i]);
        }
        if (p.residual1) {
            sum = p.other_first & 2 ? __fadd_rn(p.residual1[i], sum) : __fadd_rn(sum, p.residual1[i]);
        }
        p.dst[i] = sum;
    }
    if (i < p.state_count) {
        const uint2 row = fast_div_modulo(i, p.div_history);
        const uint2 channel = fast_div_modulo(row.x, p.div_channels);
        const uint2 sequence = fast_div_modulo(channel.x, p.div_sequences);
        const int advance = sequence.x < p.tokens ? p.tokens - sequence.x : 0;
        p.state[sequence.x * p.slot_stride + sequence.y * p.state_sequence_stride + channel.y * p.history + row.y] =
            dilated_conv_history_read(p, advance + row.y, channel.y, sequence.y);
    }
}

void ggml_cuda_op_dilated_conv_history(ggml_backend_cuda_context & ctx, const ggml_cuda_dilated_conv_history_args & args) {
    const auto & conv = args.conv;
    dilated_conv_history_params p = {};
    p.old = (const float *) args.history->data;
    p.current = (const float *) args.current->data;
    p.weights = conv.weights->data;
    p.residual0 = conv.residual[0] ? (const float *) conv.residual[0]->data : nullptr;
    p.residual1 = conv.residual[1] ? (const float *) conv.residual[1]->data : nullptr;
    p.dst = (float *) conv.dst->data;
    p.state = args.state ? (float *) args.state->data : nullptr;
    p.count = ggml_nelements(conv.dst);
    p.state_count = ggml_nelements(args.history) * args.slots;
    p.taps = conv.weights->ne[0];
    p.history = args.history->ne[0];
    p.tokens = args.current->ne[0];
    p.channels = args.history->ne[1];
    p.sequences = args.history->ne[2];
    p.slots = args.slots;
    for (int d = 0; d < 3; ++d) {
        p.old_stride[d] = args.history->nb[d] / sizeof(float);
        p.current_stride[d] = args.current->nb[d] / sizeof(float);
    }
    p.weight_stride = conv.weights->nb[1] / ggml_element_size(conv.weights);
    p.state_sequence_stride = args.state ? args.state->nb[1] / sizeof(float) : 0;
    p.slot_stride = args.slot_stride / sizeof(float);
    p.first_row = conv.first_row;
    p.dilation = conv.dilation;
    p.div_channels = init_fastdiv_values(p.channels);
    p.div_tokens = init_fastdiv_values(p.tokens);
    p.div_history = init_fastdiv_values(p.history);
    p.div_sequences = init_fastdiv_values(p.sequences);
    p.apply_silu = conv.apply_silu;
    p.other_first = conv.other_first;
    const int blocks = (std::max(p.count, p.state_count) + 255) / 256;
    if (conv.weights->type == GGML_TYPE_F32) {
        dilated_conv_history_f32<float><<<blocks, 256, 0, ctx.stream()>>>(p);
    } else if (conv.weights->type == GGML_TYPE_F16) {
        dilated_conv_history_f32<half><<<blocks, 256, 0, ctx.stream()>>>(p);
    } else {
        GGML_ASSERT(conv.weights->type == GGML_TYPE_BF16);
        dilated_conv_history_f32<nv_bfloat16><<<blocks, 256, 0, ctx.stream()>>>(p);
    }
}

struct dilated_conv_output_params {
    float * data[5];
    int value[5];
    int count;
};

static __device__ __forceinline__ void dilated_conv_output_store(const dilated_conv_output_params & outputs, int value, uint32_t i, float x) {
#pragma unroll
    for (int k = 0; k < 5; ++k) {
        if (k < outputs.count && outputs.value[k] == value) {
            outputs.data[k][i] = x;
        }
    }
}

template <typename T, bool history>
static __global__ void dilated_conv_outputs_f32(dilated_conv_history_params p, dilated_conv_output_params outputs) {
    ggml_cuda_pdl_lc();
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.count && i >= p.state_count) {
        return;
    }
    ggml_cuda_pdl_sync();
    if (i < p.count) {
        const uint2 channel = fast_div_modulo(i, p.div_channels);
        const uint2 token = fast_div_modulo(channel.x, p.div_tokens);
        const T * column = (const T *) p.weights + channel.y * p.weight_stride;
        const int64_t row = p.first_row + token.y;
        const auto read = [&](int64_t r) {
            if constexpr (history) {
                return dilated_conv_history_read(p, r, channel.y, token.x);
            } else {
                return p.old[r + channel.y * p.old_stride[1] + token.x * p.old_stride[2]];
            }
        };
        float sum = __fmul_rn(read(row), float(column[0]));
        for (int tap = 1; tap < p.taps; ++tap) {
            sum = __fadd_rn(sum, __fmul_rn(read(row + tap * p.dilation), float(column[tap])));
        }
        dilated_conv_output_store(outputs, 0, i, sum);
        if (p.apply_silu) {
            sum = ggml_cuda_op_silu_single(sum);
            dilated_conv_output_store(outputs, 1, i, sum);
        }
        if (p.residual0) {
            sum = p.other_first & 1 ? __fadd_rn(p.residual0[i], sum) : __fadd_rn(sum, p.residual0[i]);
            dilated_conv_output_store(outputs, 2, i, sum);
        }
        if (p.residual1) {
            sum = p.other_first & 2 ? __fadd_rn(p.residual1[i], sum) : __fadd_rn(sum, p.residual1[i]);
        }
        p.dst[i] = sum;
    }
    if constexpr (history) {
        if (i < p.state_count) {
            const uint2 row = fast_div_modulo(i, p.div_history);
            const uint2 channel = fast_div_modulo(row.x, p.div_channels);
            const uint2 sequence = fast_div_modulo(channel.x, p.div_sequences);
            const int advance = sequence.x < p.tokens ? p.tokens - sequence.x : 0;
            p.state[sequence.x * p.slot_stride + sequence.y * p.state_sequence_stride + channel.y * p.history + row.y] =
                dilated_conv_history_read(p, advance + row.y, channel.y, sequence.y);
        }
    }
}

template <typename T>
static void dilated_conv_outputs_launch(cudaStream_t stream, const dilated_conv_history_params & p, const dilated_conv_output_params & outputs, bool history) {
    const int blocks = (std::max(p.count, p.state_count) + 255) / 256;
    if (history) {
        dilated_conv_outputs_f32<T, true><<<blocks, 256, 0, stream>>>(p, outputs);
    } else {
        dilated_conv_outputs_f32<T, false><<<blocks, 256, 0, stream>>>(p, outputs);
    }
}

void ggml_cuda_op_dilated_conv_outputs(ggml_backend_cuda_context & ctx, const ggml_cuda_dilated_conv_args & conv,
                                      const ggml_cuda_dilated_conv_outputs & outputs, const ggml_cuda_dilated_conv_history_args * history) {
    dilated_conv_history_params p = {};
    p.old = (const float *) (history ? history->history : conv.source)->data;
    p.current = history ? (const float *) history->current->data : nullptr;
    p.weights = conv.weights->data;
    p.residual0 = conv.residual[0] ? (const float *) conv.residual[0]->data : nullptr;
    p.residual1 = conv.residual[1] ? (const float *) conv.residual[1]->data : nullptr;
    p.dst = (float *) conv.dst->data;
    p.count = ggml_nelements(conv.dst);
    p.channels = conv.source->ne[1];
    p.sequences = conv.source->ne[2];
    p.tokens = p.count / (p.channels * p.sequences);
    p.taps = conv.weights->ne[0];
    p.history = history ? history->history->ne[0] : 1;
    p.slots = history ? history->slots : 0;
    p.state_count = history ? ggml_nelements(history->history) * p.slots : 0;
    p.state = history && history->state ? (float *) history->state->data : nullptr;
    for (int d = 0; d < 3; ++d) {
        p.old_stride[d] = (history ? history->history : conv.source)->nb[d] / sizeof(float);
        p.current_stride[d] = history ? history->current->nb[d] / sizeof(float) : 0;
    }
    p.weight_stride = conv.weights->nb[1] / ggml_element_size(conv.weights);
    p.state_sequence_stride = history && history->state ? history->state->nb[1] / sizeof(float) : 0;
    p.slot_stride = history ? history->slot_stride / sizeof(float) : 0;
    p.first_row = conv.first_row;
    p.dilation = conv.dilation;
    p.div_channels = init_fastdiv_values(p.channels);
    p.div_tokens = init_fastdiv_values(p.tokens);
    p.div_history = init_fastdiv_values(p.history);
    p.div_sequences = init_fastdiv_values(p.sequences);
    p.apply_silu = conv.apply_silu;
    p.other_first = conv.other_first;
    dilated_conv_output_params out = {};
    const int final_value = conv.residual[1] ? 3 : conv.residual[0] ? 2 : conv.apply_silu ? 1 : 0;
    for (int k = 0; k < outputs.count; ++k) {
        bool duplicate = outputs.nodes[k]->data == p.dst && outputs.value[k] == final_value;
        for (int j = 0; j < out.count; ++j) {
            duplicate |= out.data[j] == outputs.nodes[k]->data && out.value[j] == outputs.value[k];
        }
        if (!duplicate) {
            out.data[out.count] = (float *) outputs.nodes[k]->data;
            out.value[out.count++] = outputs.value[k];
        }
    }
    if (!out.count) {
        if (history) {
            ggml_cuda_op_dilated_conv_history(ctx, *history);
        } else {
            ggml_cuda_op_dilated_conv(ctx, conv);
        }
        return;
    }
    if (conv.weights->type == GGML_TYPE_F32) {
        dilated_conv_outputs_launch<float>(ctx.stream(), p, out, history != nullptr);
    } else if (conv.weights->type == GGML_TYPE_F16) {
        dilated_conv_outputs_launch<half>(ctx.stream(), p, out, history != nullptr);
    } else {
        GGML_ASSERT(conv.weights->type == GGML_TYPE_BF16);
        dilated_conv_outputs_launch<nv_bfloat16>(ctx.stream(), p, out, history != nullptr);
    }
}
