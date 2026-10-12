#include "reduce_rows.cuh"
#include "sumrows.cuh"

void sum_rows_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    const int  id  = ggml_cuda_get_device();
    const int  nsm = ggml_cuda_info().devices[id].nsm;
    const dim3 block_nums(nrows, 1, 1);
    if ((nrows / nsm) < 2) {
        const dim3 block_dims(512, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        ggml_cuda_kernel_launch(reduce_rows_f32</*norm=*/false>, launch_params, x, dst, ncols);
    } else {
        const dim3 block_dims(ncols < 1024 ? 32 : 128, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        ggml_cuda_kernel_launch(reduce_rows_f32</*norm=*/false>, launch_params, x, dst, ncols);
    }
}

void ggml_cuda_op_sum_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous_rows(src0));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    if (ggml_is_contiguous(src0)) {
        sum_rows_f32_cuda(src0_d, dst_d, ncols, nrows, stream);
        return;
    }

    const dim3 block_nums(nrows, 1, 1);

    const int id  = ggml_cuda_get_device();
    const int nsm = ggml_cuda_info().devices[id].nsm;
    dim3 block_dims;
    if ((nrows / nsm) < 2) {
        // Increase num threads to 512 for small nrows to better hide the latency
        block_dims = dim3(512, 1, 1);
    } else {
        // Enough active SMs to hide latency, use smaller blocks to allow better scheduling
        block_dims = dim3(ncols < 1024 ? 32 : 128, 1, 1);
    }
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
    const char * src0_d_bytes = (const char *) src0->data;
    ggml_cuda_kernel_launch(reduce_rows_f32_strided</*norm=*/false>, launch_params, src0_d_bytes, dst_d, ncols,
            src0->ne[1], src0->ne[2], src0->nb[1], src0->nb[2], src0->nb[3]);
}

struct mul_sum_rows_args {
    const char * x;
    const char * y;
    float * dst;
    uint3 ne1, ne2, yne[4];
    int64_t xnb[3], ynb[3];
    int ncols;
};

template <bool multiply>
static __device__ __forceinline__ void mul_sum_rows_readers(const mul_sum_rows_args & args, const float * & x, const float * & y) {
    const uint32_t row = blockIdx.x;
    const uint2 row12 = fast_div_modulo(row, args.ne1);
    const uint2 row23 = fast_div_modulo(row12.x, args.ne2);
    const uint32_t i1 = row12.y;
    const uint32_t i2 = row23.y;
    const uint32_t i3 = row23.x;
    x = (const float *) (args.x + i1 * args.xnb[0] + i2 * args.xnb[1] + i3 * args.xnb[2]);
    if constexpr (multiply) {
        y = (const float *) (args.y + fastmodulo(i1, args.yne[1]) * args.ynb[0] +
                fastmodulo(i2, args.yne[2]) * args.ynb[1] + fastmodulo(i3, args.yne[3]) * args.ynb[2]);
    } else {
        y = nullptr;
    }
}

static mul_sum_rows_args mul_sum_rows_get_args(const ggml_tensor * x, const ggml_tensor * y, ggml_tensor * dst) {
    mul_sum_rows_args args{};
    args.x = (const char *) x->data;
    args.y = y ? (const char *) y->data : nullptr;
    args.dst = (float *) dst->data;
    args.ne1 = init_fastdiv_values(x->ne[1]);
    args.ne2 = init_fastdiv_values(x->ne[2]);
    args.ncols = x->ne[0];
    for (int i = 0; i < 3; ++i) {
        args.xnb[i] = x->nb[i + 1];
        args.ynb[i] = y ? y->nb[i + 1] : 0;
    }
    for (int i = 0; i < 4; ++i) {
        args.yne[i] = init_fastdiv_values(y ? y->ne[i] : 1);
    }
    return args;
}

struct row_gate_args {
    mul_sum_rows_args reduction;
    float * post[7];
    int kind[7];
    float scale, bias, low, high;
};

template <bool multiply, bool broadcast>
static __global__ void row_gate_f32(row_gate_args args) {
    ggml_cuda_pdl_lc();
    const uint32_t row = blockIdx.x;
    const float * x, * y;
    mul_sum_rows_readers<multiply>(args.reduction, x, y);
    const float sum = reduce_row_f32<multiply, broadcast>(x, args.reduction.ncols, y, args.reduction.yne[0]);
    if (threadIdx.x != 0) {
        return;
    }
    args.reduction.dst[row] = sum;
    float values[7];
    values[0] = __fmaf_rn(args.scale, sum, args.bias);
    values[1] = values[0] > 0.0f ? 1.0f : values[0] < 0.0f ? -1.0f : 0.0f;
    values[2] = fabsf(values[0]);
    values[3] = fminf(fmaxf(values[2], args.low), args.high);
    values[4] = sqrtf(values[3]);
    values[5] = __fmul_rn(values[1], values[4]);
    values[6] = 1.0f / (1.0f + expf(-values[5]));
    for (int j = 0; j < 7; ++j) {
        args.post[j][row] = values[args.kind[j]];
    }
}

void ggml_cuda_op_row_gate(ggml_backend_cuda_context & ctx, const ggml_tensor * mul, ggml_tensor * sum, ggml_tensor * const * post, const int * kind) {
    const ggml_tensor * x = mul ? mul->src[0] : sum->src[0];
    const ggml_tensor * y = mul ? mul->src[1] : nullptr;
    row_gate_args args{};
    args.reduction = mul_sum_rows_get_args(x, y, sum);
    memcpy(&args.scale, post[0]->op_params, sizeof(float));
    memcpy(&args.bias, (const char *) post[0]->op_params + sizeof(float), sizeof(float));
    for (int j = 0; j < 7; ++j) {
        args.post[j] = (float *) post[j]->data;
        args.kind[j] = kind[j];
        if (kind[j] == 3) {
            memcpy(&args.low, post[j]->op_params, sizeof(float));
            memcpy(&args.high, (const char *) post[j]->op_params + sizeof(float), sizeof(float));
        }
    }
    const int nrows = ggml_nrows(sum);
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const dim3 blocks(nrows, 1, 1);
    const dim3 threads(nrows / nsm < 2 ? 512 : args.reduction.ncols < 1024 ? 32 : 128, 1, 1);
    const ggml_cuda_kernel_launch_params params(blocks, threads, 0, ctx.stream());
    if (!mul) {
        ggml_cuda_kernel_launch(row_gate_f32<false, false>, params, args);
    } else if (y->ne[0] == x->ne[0]) {
        ggml_cuda_kernel_launch(row_gate_f32<true, false>, params, args);
    } else {
        ggml_cuda_kernel_launch(row_gate_f32<true, true>, params, args);
    }
}
