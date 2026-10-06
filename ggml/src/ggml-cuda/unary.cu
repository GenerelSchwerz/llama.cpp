#include "unary.cuh"
#include "convert.cuh"

static __device__ __forceinline__ float op_abs(float x) {
    return fabsf(x);
}

static __device__ __forceinline__ float op_sgn(float x) {
    return (x > 0.f ? 1.f : ((x < 0.f ? -1.f : 0.f)));
}

static __device__ __forceinline__ float op_neg(float x) {
    return -x;
}

static __device__ __forceinline__ float op_step(float x) {
    return x > 0.0f;
}

static __device__ __forceinline__ float op_gelu(float x) {
    return ggml_cuda_op_gelu_single(x);
}

static __device__ __forceinline__ float op_gelu_erf(float x) {
    const float SQRT_2_INV = 0.70710678118654752440084436210484f;

    return 0.5f*x*(1.0f + erff(x*SQRT_2_INV));
}

static __device__ __forceinline__ float op_gelu_quick(float x) {
    const float GELU_QUICK_COEF = -1.702f;

    return x * (1.0f / (1.0f + expf(GELU_QUICK_COEF * x)));
}

static __device__ __forceinline__ float op_silu(float x) {
    return ggml_cuda_op_silu_single(x);
}

static __device__ __forceinline__ float op_tanh(float x) {
    return tanhf(x);
}

static __device__ __forceinline__ float op_relu(float x) {
    return fmaxf(x, 0);
}

static __device__ __forceinline__ float op_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static __device__ __forceinline__ float op_hardsigmoid(float x) {
    return fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

static __device__ __forceinline__ float op_hardswish(float x) {
    return x * fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

static __device__ __forceinline__ float op_exp(float x) {
    return expf(x);
}

static __device__ __forceinline__ float op_sqr(float x) {
    return x * x;
}

static __device__ __forceinline__ float op_relu_sqr(float x) {
    const float r = fmaxf(x, 0.0f);
    return r * r;
}

static __device__ __forceinline__ float op_sqrt(float x) {
    return sqrtf(x);
}

static __device__ __forceinline__ float op_sin(float x) {
    return sinf(x);
}

static __device__ __forceinline__ float op_cos(float x) {
    return cosf(x);
}

static __device__ __forceinline__ float op_log(float x) {
    return logf(x);
}

static __device__ __forceinline__ float op_expm1(float x) {
    return expm1f(x);
}

static __device__ __forceinline__ float op_softplus(float x) {
    return (x > 20.0f) ? x : logf(1.0f + expf(x));
}

static __device__ __forceinline__ float op_elu(float x) {
    return (x > 0.f) ? x : expm1f(x);
}

static __device__ __forceinline__ float op_floor(float x) {
    return floorf(x);
}

static __device__ __forceinline__ float op_ceil(float x) {
    return ceilf(x);
}

static __device__ __forceinline__ float op_round(float x) {
    return round(x);
}

static __device__ __forceinline__ float op_trunc(float x) {
    return trunc(x);
}

template <float (*op)(float), typename T>
static __global__ void unary_op_kernel(const T * x, T * dst, const int k) {
    ggml_cuda_pdl_lc();
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    ggml_cuda_pdl_sync();
    dst[i] = ggml_cuda_cast<T>(op(ggml_cuda_cast<float>(x[i])));
}

template <float (*op)(float), typename T>
static void unary_cuda(const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_NEG_BLOCK_SIZE - 1) / CUDA_NEG_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((dim3)num_blocks, CUDA_NEG_BLOCK_SIZE, 0, stream);
    ggml_cuda_kernel_launch(unary_op_kernel<op, T>, launch_params, x, dst, k);
}

template <float (*op)(float)>
void ggml_cuda_op_unary(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        unary_cuda<op>((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else if (src0->type == GGML_TYPE_BF16) {
        unary_cuda<op>((const nv_bfloat16 *)src0_d, (nv_bfloat16 *)dst_d, ggml_nelements(src0), stream);
    } else {
        unary_cuda<op>((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}

void ggml_cuda_op_abs(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_abs>(ctx, dst);
}

void ggml_cuda_op_sgn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_sgn>(ctx, dst);
}

void ggml_cuda_op_neg(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_neg>(ctx, dst);
}

void ggml_cuda_op_step(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_step>(ctx, dst);
}

void ggml_cuda_op_gelu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_gelu>(ctx, dst);
}

void ggml_cuda_op_gelu_erf(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_gelu_erf>(ctx, dst);
}

void ggml_cuda_op_gelu_quick(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_gelu_quick>(ctx, dst);
}

void ggml_cuda_op_silu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_silu>(ctx, dst);
}

void ggml_cuda_op_tanh(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_tanh>(ctx, dst);
}

void ggml_cuda_op_relu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_relu>(ctx, dst);
}

void ggml_cuda_op_sigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_sigmoid>(ctx, dst);
}

struct affine_unary_tensor {
    const void * data;
    ggml_type type;
    uint32_t ne[GGML_MAX_DIMS];
    size_t nb[GGML_MAX_DIMS];
};

struct affine_unary_args {
    affine_unary_tensor input[3];
    void * output[4];
    ggml_type output_type[4];
    uint32_t ne[GGML_MAX_DIMS];
    int count;
    ggml_unary_op op;
    float scale;
    float bias;
};

static __device__ __forceinline__ float affine_unary_read(const affine_unary_tensor & tensor, const uint32_t * pos) {
    size_t offset = 0;
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        offset += size_t(pos[d] % tensor.ne[d])*tensor.nb[d];
    }
    const char * ptr = (const char *) tensor.data + offset;
    if (tensor.type == GGML_TYPE_F16) {
        return ggml_cuda_cast<float>(*(const half *) ptr);
    }
    if (tensor.type == GGML_TYPE_BF16) {
        return ggml_cuda_cast<float>(*(const nv_bfloat16 *) ptr);
    }
    return *(const float *) ptr;
}

static __device__ __forceinline__ float affine_unary_store(void * dst, ggml_type type, int i, float value) {
    if (type == GGML_TYPE_F16) {
        const half rounded = ggml_cuda_cast<half>(value);
        ((half *) dst)[i] = rounded;
        return ggml_cuda_cast<float>(rounded);
    }
    if (type == GGML_TYPE_BF16) {
        const nv_bfloat16 rounded = ggml_cuda_cast<nv_bfloat16>(value);
        ((nv_bfloat16 *) dst)[i] = rounded;
        return ggml_cuda_cast<float>(rounded);
    }
    ((float *) dst)[i] = value;
    return value;
}

struct affine_unary_load {
    __device__ __forceinline__ float operator()(const affine_unary_tensor & tensor, const uint32_t * pos, int) const {
        return affine_unary_read(tensor, pos);
    }
};

struct affine_unary_convert_load {
    const void * data;
    ggml_type type;
    float * raw;
    __device__ __forceinline__ float operator()(const affine_unary_tensor &, const uint32_t *, int i) const {
        const float value = type == GGML_TYPE_F16 ? ggml_cuda_cast<float>(((const half *) data)[i]) : ggml_cuda_cast<float>(((const nv_bfloat16 *) data)[i]);
        raw[i] = value;
        return value;
    }
};

static __device__ __forceinline__ void affine_unary_apply(affine_unary_args args, float value, const uint32_t * pos, int i) {
    value = __fmul_rn(value, affine_unary_read(args.input[1], pos));
    value = affine_unary_store(args.output[0], args.output_type[0], i, value);
    value = __fadd_rn(value, affine_unary_read(args.input[2], pos));
    value = affine_unary_store(args.output[1], args.output_type[1], i, value);
    value = args.op == GGML_UNARY_OP_SIGMOID ? op_sigmoid(value) : op_silu(value);
    value = affine_unary_store(args.output[2], args.output_type[2], i, value);
    if (args.output[3]) {
        affine_unary_store(args.output[3], args.output_type[3], i, args.scale*value + args.bias);
    }
}

template <typename Load>
static __device__ __forceinline__ void affine_unary_impl(affine_unary_args args, Load load) {
    ggml_cuda_pdl_lc();
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= args.count) {
        return;
    }
    uint32_t remaining = i;
    uint32_t pos[GGML_MAX_DIMS];
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        pos[d] = remaining % args.ne[d];
        remaining /= args.ne[d];
    }
    ggml_cuda_pdl_sync();
    float value = __fmul_rn(load(args.input[0], pos, i), affine_unary_read(args.input[1], pos));
    value = affine_unary_store(args.output[0], args.output_type[0], i, value);
    value = __fadd_rn(value, affine_unary_read(args.input[2], pos));
    value = affine_unary_store(args.output[1], args.output_type[1], i, value);
    value = args.op == GGML_UNARY_OP_SIGMOID ? op_sigmoid(value) : op_silu(value);
    value = affine_unary_store(args.output[2], args.output_type[2], i, value);
    if (args.output[3]) {
        affine_unary_store(args.output[3], args.output_type[3], i, args.scale*value + args.bias);
    }
}

static __global__ void affine_unary_kernel(affine_unary_args args) {
    affine_unary_impl(args, affine_unary_load{});
}

static __global__ void affine_unary_convert_kernel(affine_unary_args args, const void * src, ggml_type type, float * raw) {
    affine_unary_impl(args, affine_unary_convert_load{src, type, raw});
}

static __global__ void affine_unary_window_convert_kernel(affine_unary_args args, const void * src, ggml_type type, float * raw, int count, int columns, int offset) {
    ggml_cuda_pdl_lc();
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= count) { return; }
    ggml_cuda_pdl_sync();
    const affine_unary_convert_load load{src, type, raw};
    const float value = load(args.input[0], nullptr, i);
    const int column = i % columns;
    if (column < offset || uint32_t(column - offset) >= args.ne[0]) { return; }
    const int j = (i/columns)*args.ne[0] + column - offset;
    uint32_t remaining = j;
    uint32_t pos[GGML_MAX_DIMS];
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        pos[d] = remaining % args.ne[d];
        remaining /= args.ne[d];
    }
    affine_unary_apply(args, value, pos, j);
}

static __global__ void affine_unary_tail_kernel(affine_unary_args args, void * tail, ggml_type tail_type, float scale, float bias,
        const void * src, ggml_type src_type, float * raw, int count, int columns, int offset) {
    ggml_cuda_pdl_lc();
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= count) { return; }
    ggml_cuda_pdl_sync();
    float value = 0.0f;
    int j = i;
    if (raw) {
        const affine_unary_convert_load load{src, src_type, raw};
        value = load(args.input[0], nullptr, i);
        const int column = i % columns;
        if (column < offset || uint32_t(column - offset) >= args.ne[0]) { return; }
        j = (i/columns)*args.ne[0] + column - offset;
    }
    uint32_t remaining = j;
    uint32_t pos[GGML_MAX_DIMS];
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        pos[d] = remaining % args.ne[d];
        remaining /= args.ne[d];
    }
    if (!raw) { value = affine_unary_read(args.input[0], pos); }
    value = __fmul_rn(value, affine_unary_read(args.input[1], pos));
    value = affine_unary_store(args.output[0], args.output_type[0], j, value);
    value = __fadd_rn(value, affine_unary_read(args.input[2], pos));
    value = affine_unary_store(args.output[1], args.output_type[1], j, value);
    value = args.op == GGML_UNARY_OP_SIGMOID ? op_sigmoid(value) : op_silu(value);
    value = affine_unary_store(args.output[2], args.output_type[2], j, value);
    value = affine_unary_store(args.output[3], args.output_type[3], j, args.scale*value + args.bias);
    affine_unary_store(tail, tail_type, j, scale*value + bias);
}

static affine_unary_args affine_unary_get_args(ggml_tensor * mul, ggml_tensor * add, ggml_tensor * unary, ggml_tensor * post) {
    affine_unary_args args{};
    const ggml_tensor * input[] = { mul->src[0], mul->src[1], add->src[1] };
    const ggml_tensor * output[] = { mul, add, unary, post };
    for (int j = 0; j < 3; ++j) {
        args.input[j].data = input[j]->data;
        args.input[j].type = input[j]->type;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            args.input[j].ne[d] = input[j]->ne[d];
            args.input[j].nb[d] = input[j]->nb[d];
        }
    }
    for (int j = 0; j < 4; ++j) {
        if (output[j]) {
            args.output[j] = output[j]->data;
            args.output_type[j] = output[j]->type;
        }
        args.ne[j] = mul->ne[j];
    }
    args.count = ggml_nelements(mul);
    args.op = ggml_get_unary_op(unary);
    if (post) {
        memcpy(&args.scale, post->op_params, sizeof(float));
        memcpy(&args.bias, (const char *) post->op_params + sizeof(float), sizeof(float));
    }
    return args;
}

struct mul_add_args {
    affine_unary_args affine;
    uint3 ne[GGML_MAX_DIMS];
    uint3 repeat[3][GGML_MAX_DIMS];
    uint32_t repeat_mask[3];
};

static __device__ __forceinline__ float mul_add_read(const affine_unary_tensor & tensor, const uint3 * repeat, const uint32_t * pos, uint32_t repeat_mask) {
    size_t offset = 0;
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        const uint32_t index = repeat_mask & (1u << d) ? fastmodulo(pos[d], repeat[d]) : pos[d];
        offset += size_t(index)*tensor.nb[d];
    }
    const char * ptr = (const char *) tensor.data + offset;
    if (tensor.type == GGML_TYPE_F16) { return ggml_cuda_cast<float>(*(const half *) ptr); }
    if (tensor.type == GGML_TYPE_BF16) { return ggml_cuda_cast<float>(*(const nv_bfloat16 *) ptr); }
    return *(const float *) ptr;
}

static __global__ void mul_add_kernel(mul_add_args binary, bool product_first) {
    const affine_unary_args & args = binary.affine;
    ggml_cuda_pdl_lc();
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= args.count) { return; }
    uint32_t remaining = i;
    uint32_t pos[GGML_MAX_DIMS];
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        const uint2 part = fast_div_modulo(remaining, binary.ne[d]);
        remaining = part.x;
        pos[d] = part.y;
    }
    ggml_cuda_pdl_sync();
    float value = __fmul_rn(mul_add_read(args.input[0], binary.repeat[0], pos, 0), mul_add_read(args.input[1], binary.repeat[1], pos, binary.repeat_mask[1]));
    value = affine_unary_store(args.output[0], args.output_type[0], i, value);
    const float base = mul_add_read(args.input[2], binary.repeat[2], pos, binary.repeat_mask[2]);
    value = product_first ? __fadd_rn(value, base) : __fadd_rn(base, value);
    affine_unary_store(args.output[1], args.output_type[1], i, value);
}

static __global__ void repeat_mul_add_kernel(mul_add_args binary, int repeated_input, bool product_first) {
    const affine_unary_args & args = binary.affine;
    ggml_cuda_pdl_lc();
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= args.count) { return; }
    uint32_t remaining = i;
    uint32_t pos[GGML_MAX_DIMS];
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        const uint2 part = fast_div_modulo(remaining, binary.ne[d]);
        remaining = part.x;
        pos[d] = part.y;
    }
    ggml_cuda_pdl_sync();
    float lhs = mul_add_read(args.input[0], binary.repeat[0], pos, binary.repeat_mask[0]);
    float rhs = mul_add_read(args.input[1], binary.repeat[1], pos, binary.repeat_mask[1]);
    if (repeated_input == 0) { lhs = affine_unary_store(args.output[2], args.output_type[2], i, lhs); }
    else { rhs = affine_unary_store(args.output[2], args.output_type[2], i, rhs); }
    float value = __fmul_rn(lhs, rhs);
    value = affine_unary_store(args.output[0], args.output_type[0], i, value);
    const float base = mul_add_read(args.input[2], binary.repeat[2], pos, binary.repeat_mask[2]);
    value = product_first ? __fadd_rn(value, base) : __fadd_rn(base, value);
    affine_unary_store(args.output[1], args.output_type[1], i, value);
}

static mul_add_args mul_add_get_args(ggml_tensor * mul, ggml_tensor * add, ggml_tensor * repeat = nullptr) {
    mul_add_args binary{};
    affine_unary_args & args = binary.affine;
    const bool product_first = add->src[0] == mul;
    const ggml_tensor * reads[] = {mul->src[0], mul->src[1], add->src[product_first ? 1 : 0]};
    if (repeat) { reads[mul->src[0] == repeat ? 0 : 1] = repeat->src[0]; }
    for (int j = 0; j < 3; ++j) {
        args.input[j].data = reads[j]->data;
        args.input[j].type = reads[j]->type;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            args.input[j].ne[d] = reads[j]->ne[d];
            args.input[j].nb[d] = reads[j]->ne[d] == 1 ? 0 : reads[j]->nb[d];
            binary.repeat[j][d] = init_fastdiv_values(reads[j]->ne[d]);
            if (reads[j]->ne[d] != 1 && reads[j]->ne[d] != mul->ne[d]) { binary.repeat_mask[j] |= 1u << d; }
        }
    }
    args.output[0] = mul->data;
    args.output[1] = add->data;
    args.output_type[0] = mul->type;
    args.output_type[1] = add->type;
    for (int d = 0; d < GGML_MAX_DIMS; ++d) { binary.ne[d] = init_fastdiv_values(mul->ne[d]); }
    args.count = ggml_nelements(mul);
    if (repeat) {
        args.output[2] = repeat->data;
        args.output_type[2] = repeat->type;
    }
    return binary;
}

void ggml_cuda_op_mul_add(ggml_backend_cuda_context & ctx, ggml_tensor * mul, ggml_tensor * add) {
    const auto binary = mul_add_get_args(mul, add);
    const int blocks = (binary.affine.count + CUDA_NEG_BLOCK_SIZE - 1)/CUDA_NEG_BLOCK_SIZE;
    const auto launch = ggml_cuda_kernel_launch_params(blocks, CUDA_NEG_BLOCK_SIZE, 0, ctx.stream());
    ggml_cuda_kernel_launch(mul_add_kernel, launch, binary, add->src[0] == mul);
}

void ggml_cuda_op_repeat_mul_add(ggml_backend_cuda_context & ctx, ggml_tensor * repeat, ggml_tensor * mul, ggml_tensor * add) {
    const auto binary = mul_add_get_args(mul, add, repeat);
    const int blocks = (binary.affine.count + CUDA_NEG_BLOCK_SIZE - 1)/CUDA_NEG_BLOCK_SIZE;
    const auto launch = ggml_cuda_kernel_launch_params(blocks, CUDA_NEG_BLOCK_SIZE, 0, ctx.stream());
    ggml_cuda_kernel_launch(repeat_mul_add_kernel, launch, binary, mul->src[0] == repeat ? 0 : 1, add->src[0] == mul);
}

static void affine_unary_tail_cuda(ggml_backend_cuda_context & ctx, affine_unary_args args, ggml_tensor * tail,
        const void * src = nullptr, ggml_type src_type = GGML_TYPE_F32, ggml_tensor * mm = nullptr, int offset = 0) {
    GGML_ASSERT(args.output[3] && tail && (tail->type == GGML_TYPE_F32 || tail->type == GGML_TYPE_BF16));
    float scale, bias;
    memcpy(&scale, tail->op_params, sizeof(float));
    memcpy(&bias, (const char *) tail->op_params + sizeof(float), sizeof(float));
    const int count = mm ? ggml_nelements(mm) : args.count;
    const int blocks = (count + CUDA_NEG_BLOCK_SIZE - 1)/CUDA_NEG_BLOCK_SIZE;
    const auto launch = ggml_cuda_kernel_launch_params(blocks, CUDA_NEG_BLOCK_SIZE, 0, ctx.stream());
    ggml_cuda_kernel_launch(affine_unary_tail_kernel, launch, args, tail->data, tail->type, scale, bias,
            src, src_type, mm ? (float *) mm->data : nullptr, count, mm ? (int) mm->ne[0] : 0, offset);
}

void ggml_cuda_op_affine_unary(ggml_backend_cuda_context & ctx, ggml_tensor * mul, ggml_tensor * add, ggml_tensor * unary, ggml_tensor * post, ggml_tensor * tail) {
    const auto args = affine_unary_get_args(mul, add, unary, post);
    if (tail) {
        affine_unary_tail_cuda(ctx, args, tail);
        return;
    }
    const int blocks = (args.count + CUDA_NEG_BLOCK_SIZE - 1)/CUDA_NEG_BLOCK_SIZE;
    const auto launch = ggml_cuda_kernel_launch_params(blocks, CUDA_NEG_BLOCK_SIZE, 0, ctx.stream());
    ggml_cuda_kernel_launch(affine_unary_kernel, launch, args);
}

void ggml_cuda_op_affine_unary_convert(ggml_backend_cuda_context & ctx, ggml_type type, const void * src, ggml_tensor * mm, const ggml_cuda_affine_unary_ops & ops) {
    GGML_ASSERT(type == GGML_TYPE_F16 || type == GGML_TYPE_BF16);
    GGML_ASSERT(mm->type == GGML_TYPE_F32 && ggml_is_contiguous(mm));
    const auto args = affine_unary_get_args(ops.mul, ops.add, ops.unary, ops.post);
    if (ops.tail) {
        const ggml_tensor * input = ops.mul->src[0];
        GGML_ASSERT(input == mm || (input->op == GGML_OP_VIEW && input->view_src == mm && input->type == GGML_TYPE_F32 &&
                input->nb[0] == sizeof(float) && input->view_offs % sizeof(float) == 0 && input->ne[0] > 0 && input->ne[0] <= mm->ne[0] &&
                input->view_offs/sizeof(float) <= uint64_t(mm->ne[0] - input->ne[0])));
        for (int d = 1; d < GGML_MAX_DIMS; ++d) {
            GGML_ASSERT(input->ne[d] == mm->ne[d] && input->nb[d] == mm->nb[d]);
        }
        GGML_ASSERT(ggml_nelements(mm) > 0 && ggml_nelements(mm) <= INT_MAX - CUDA_NEG_BLOCK_SIZE && args.count == ggml_nelements(input));
        affine_unary_tail_cuda(ctx, args, ops.tail, src, type, mm, input == mm ? 0 : (int) (input->view_offs/sizeof(float)));
        return;
    }
    if (ops.mul->src[0] != mm) {
        const ggml_tensor * input = ops.mul->src[0];
        GGML_ASSERT(input->op == GGML_OP_VIEW && input->view_src == mm && input->type == GGML_TYPE_F32 &&
                input->nb[0] == sizeof(float) && input->view_offs % sizeof(float) == 0 && input->ne[0] <= mm->ne[0] &&
                input->view_offs/sizeof(float) <= uint64_t(mm->ne[0] - input->ne[0]));
        for (int d = 1; d < GGML_MAX_DIMS; ++d) {
            GGML_ASSERT(input->ne[d] == mm->ne[d] && input->nb[d] == mm->nb[d]);
        }
        GGML_ASSERT(ggml_nelements(mm) > 0 && ggml_nelements(mm) <= INT_MAX - CUDA_NEG_BLOCK_SIZE && args.count == ggml_nelements(input));
        const int count = ggml_nelements(mm);
        const int blocks = (count + CUDA_NEG_BLOCK_SIZE - 1)/CUDA_NEG_BLOCK_SIZE;
        const auto launch = ggml_cuda_kernel_launch_params(blocks, CUDA_NEG_BLOCK_SIZE, 0, ctx.stream());
        ggml_cuda_kernel_launch(affine_unary_window_convert_kernel, launch, args, src, type, (float *) mm->data, count, (int) mm->ne[0], (int) (input->view_offs/sizeof(float)));
        return;
    }
    GGML_ASSERT(args.count == ggml_nelements(mm));
    const int blocks = (args.count + CUDA_NEG_BLOCK_SIZE - 1)/CUDA_NEG_BLOCK_SIZE;
    const auto launch = ggml_cuda_kernel_launch_params(blocks, CUDA_NEG_BLOCK_SIZE, 0, ctx.stream());
    ggml_cuda_kernel_launch(affine_unary_convert_kernel, launch, args, src, type, (float *) mm->data);
}

void ggml_cuda_op_hardsigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_hardsigmoid>(ctx, dst);
}

void ggml_cuda_op_hardswish(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_hardswish>(ctx, dst);
}

void ggml_cuda_op_exp(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_exp>(ctx, dst);
}

void ggml_cuda_op_sqr(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_sqr>(ctx, dst);
}

void ggml_cuda_op_sqrt(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_sqrt>(ctx, dst);
}

void ggml_cuda_op_sin(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_sin>(ctx, dst);
}

void ggml_cuda_op_cos(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_cos>(ctx, dst);
}

void ggml_cuda_op_log(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_log>(ctx, dst);
}

void ggml_cuda_op_elu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_elu>(ctx, dst);
}

void ggml_cuda_op_floor(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_floor>(ctx, dst);
}

void ggml_cuda_op_ceil(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_ceil>(ctx, dst);
}

void ggml_cuda_op_round(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_round>(ctx, dst);
}

void ggml_cuda_op_trunc(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_trunc>(ctx, dst);
}

void ggml_cuda_op_expm1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_expm1>(ctx, dst);
}

void ggml_cuda_op_softplus(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_softplus>(ctx, dst);
}
/* gated ops */

template <float (*op)(float), typename T>
static __global__ void unary_gated_op_kernel(const T * x, const T * g, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1) {
    ggml_cuda_pdl_lc();
    const int64_t i = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    // perform base op and multiply with gate (either offset in same tensor or a separate one)
    const int64_t j0 = (i / n) * o0 + (i % n);
    const int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    ggml_cuda_pdl_sync();
    dst[i] = ggml_cuda_cast<T>(op(ggml_cuda_cast<float>(x[j0])) * ggml_cuda_cast<float>(g[j1]));
}

template <float (*op)(float), typename T>
static void unary_gated_cuda(const T * x, const T * g, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1, cudaStream_t stream) {
    const int64_t num_blocks = (k + CUDA_GLU_BLOCK_SIZE - 1) / CUDA_GLU_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((dim3)num_blocks, CUDA_GLU_BLOCK_SIZE, 0, stream);
    ggml_cuda_kernel_launch(unary_gated_op_kernel<op, T>, launch_params, x, g, dst, k, n, o0, o1);
}

template <float (*op)(float)>
void ggml_cuda_op_unary_gated(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    void * src0_d = src0->data;
    void * src1_d = src1 ? src1->data : src0->data;
    const int64_t src0_o = src0->nb[1];
    const int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    void * dst_d = dst->data;
    const int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_size(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    const int32_t swapped = ((const int32_t *) dst->op_params)[1];

    if (src0->type == GGML_TYPE_F16) {
        half * src0_p = (half *) src0_d;
        half * src1_p = (half *) src1_d;

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        unary_gated_cuda<op>(src0_p, src1_p, (half *)dst_d, ggml_nelements(dst), nc, src0_o / sizeof(half), src1_o / sizeof(half), stream);
    } else if (src0->type == GGML_TYPE_BF16) {
        nv_bfloat16 * src0_p = (nv_bfloat16 *) src0_d;
        nv_bfloat16 * src1_p = (nv_bfloat16 *) src1_d;

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        unary_gated_cuda<op>(src0_p, src1_p, (nv_bfloat16 *)dst_d, ggml_nelements(dst), nc, src0_o / sizeof(nv_bfloat16), src1_o / sizeof(nv_bfloat16), stream);
    } else {
        float * src0_p = (float *) src0_d;
        float * src1_p = (float *) src1_d;

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        unary_gated_cuda<op>(src0_p, src1_p, (float *)dst_d, ggml_nelements(dst), nc, src0_o / sizeof(float), src1_o / sizeof(float), stream);
    }
}

void ggml_cuda_op_reglu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary_gated<op_relu>(ctx, dst);
}

void ggml_cuda_op_geglu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary_gated<op_gelu>(ctx, dst);
}

void ggml_cuda_op_swiglu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary_gated<op_silu>(ctx, dst);
}

void ggml_cuda_op_geglu_erf(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary_gated<op_gelu_erf>(ctx, dst);
}

void ggml_cuda_op_geglu_quick(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary_gated<op_gelu_quick>(ctx, dst);
}

// swiglu_oai

template <typename T>
static __global__ void swiglu_oai_kernel(const T * x, const T * g, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1, float alpha, float limit) {
    const int64_t i = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    // perform base op and multiply with gate (either offset in same tensor or a separate one)
    const int64_t j0 = (i / n) * o0 + (i % n);
    const int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    float xi = x[j0];
    float gi = g[j1];

    dst[i] = ggml_cuda_op_swiglu_oai_single(xi, gi, alpha, limit);
}

template <typename T>
static void swiglu_oai_cuda(const T * x, const T * g, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1, const float alpha, const float limit, cudaStream_t stream) {
    const int64_t num_blocks = (k + CUDA_GLU_BLOCK_SIZE - 1) / CUDA_GLU_BLOCK_SIZE;
    swiglu_oai_kernel<<<num_blocks, CUDA_GLU_BLOCK_SIZE, 0, stream>>>(x, g, dst, k, n, o0, o1, alpha, limit);
}

void ggml_cuda_op_swiglu_oai(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    void * src0_d = src0->data;
    void * src1_d = src1 ? src1->data : src0->data;
    const int64_t src0_o = src0->nb[1];
    const int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    void * dst_d = dst->data;
    const int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_size(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    //const int32_t swapped = ((const int32_t *) dst->op_params)[1];
    const int32_t swapped = ggml_get_op_params_i32(dst, 1);
    const float alpha = ggml_get_op_params_f32(dst, 2);
    const float limit = ggml_get_op_params_f32(dst, 3);

    float * src0_p = (float *) src0_d;
    float * src1_p = (float *) src1_d;

    if (!src1) {
        src0_p += swapped ? nc : 0;
        src1_p += swapped ? 0 : nc;
    }

    swiglu_oai_cuda(src0_p, src1_p, (float *)dst_d, ggml_nelements(dst), nc, src0_o / sizeof(float), src1_o / sizeof(float), alpha, limit, stream);
}

// swiglu_clamp

template <typename T>
static __global__ void swiglu_clamp_kernel(const T * gate, const T * up, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1, float limit) {
    const int64_t i = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    const int64_t j0 = (i / n) * o0 + (i % n);
    const int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    dst[i] = (T) ggml_cuda_op_swiglu_clamp_single((float) gate[j0], (float) up[j1], limit);
}

template <typename T>
static void swiglu_clamp_cuda(const T * gate, const T * up, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1, const float limit, cudaStream_t stream) {
    const int64_t num_blocks = (k + CUDA_GLU_BLOCK_SIZE - 1) / CUDA_GLU_BLOCK_SIZE;
    swiglu_clamp_kernel<<<num_blocks, CUDA_GLU_BLOCK_SIZE, 0, stream>>>(gate, up, dst, k, n, o0, o1, limit);
}

void ggml_cuda_op_swiglu_clamp(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    void * src0_d = src0->data;
    void * src1_d = src1 ? src1->data : src0->data;
    const int64_t src0_o = src0->nb[1];
    const int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    void * dst_d = dst->data;
    const int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_size(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    const int32_t swapped = ggml_get_op_params_i32(dst, 1);
    const float limit = ggml_get_op_params_f32(dst, 3);

    if (src0->type == GGML_TYPE_F16) {
        half * src0_p = (half *) src0_d;
        half * src1_p = (half *) src1_d;

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        swiglu_clamp_cuda(src0_p, src1_p, (half *) dst_d, ggml_nelements(dst), nc, src0_o / sizeof(half), src1_o / sizeof(half), limit, stream);
    } else {
        float * src0_p = (float *) src0_d;
        float * src1_p = (float *) src1_d;

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        swiglu_clamp_cuda(src0_p, src1_p, (float *) dst_d, ggml_nelements(dst), nc, src0_o / sizeof(float), src1_o / sizeof(float), limit, stream);
    }
}

/* CUDA kernel + launcher for xIELU */

template <typename T>
static __global__ void xielu_kernel(const T * x, T * dst, const int k, float alpha_n, float alpha_p, float beta, float eps) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    const float xi = ggml_cuda_cast<float>(x[i]);

    const float gate_pos = (xi > 0.0f);
    const float y_pos = alpha_p * xi * xi + beta * xi;
    const float min_v_eps = fminf(xi, eps);
    const float y_neg = (expm1f(min_v_eps) - xi) * alpha_n + beta * xi;
    const float out = gate_pos * y_pos + (1.0f - gate_pos) * y_neg;

    dst[i] = ggml_cuda_cast<T>(out);
}

template <typename T>
static void xielu_cuda(const T * x, T * dst, const int k, float alpha_n, float alpha_p, float beta, float eps, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_XIELU_BLOCK_SIZE) / CUDA_XIELU_BLOCK_SIZE;
    xielu_kernel<<<num_blocks, CUDA_XIELU_BLOCK_SIZE, 0, stream>>>(x, dst, k, alpha_n, alpha_p, beta, eps);
}

void ggml_cuda_op_xielu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    const float alpha_n = ggml_get_op_params_f32(dst, 1);
    const float alpha_p = ggml_get_op_params_f32(dst, 2);
    const float beta    = ggml_get_op_params_f32(dst, 3);
    const float eps     = ggml_get_op_params_f32(dst, 4);

    if (src0->type == GGML_TYPE_F16) {
        xielu_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), alpha_n, alpha_p, beta, eps, stream);
    } else {
        xielu_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), alpha_n, alpha_p, beta, eps, stream);
    }
}



/* silu_back */

static __device__ __forceinline__ float op_silu_back(float grad, float x) {
    const float s = 1.0f / (1.0f + expf(-x));
    return grad * s * (1.0f + x * (1.0f - s));
}

template <class T>
static __global__ void silu_back_kernel(const T * grad, const T * xf, T * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = (T)op_silu_back((float)grad[i], (float)xf[i]);
}

template <class T>
static void silu_back_cuda(const T * grad, const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SILU_BACK_BLOCK_SIZE - 1) / CUDA_SILU_BLOCK_SIZE;
    silu_back_kernel<<<num_blocks, CUDA_SILU_BACK_BLOCK_SIZE, 0, stream>>>(grad, x, dst, k);
}

void ggml_cuda_op_silu_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // input from forward pass
    const ggml_tensor * src1 = dst->src[1]; // grads of forward pass output

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        silu_back_cuda((const half *)src0_d, (const half *)src1_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        silu_back_cuda((const float*)src0_d, (const float*)src1_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}

/* leaky relu */

static __device__ __forceinline__ float op_leaky_relu(float x, const float negative_slope) {
    return fmaxf(x, 0) + fminf(x, 0.0f) * negative_slope;
}

template <class T>
static __global__ void leaky_relu_kernel(const T * x, T * dst, const int k, const float negative_slope) {
    const int i  = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = (T)op_leaky_relu((float)x[i], negative_slope);
}

template <class T>
static void leaky_relu_cuda(const T * x, T * dst, const int k, const float negative_slope, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_RELU_BLOCK_SIZE - 1) / CUDA_RELU_BLOCK_SIZE;
    leaky_relu_kernel<<<num_blocks, CUDA_RELU_BLOCK_SIZE, 0, stream>>>(x, dst, k, negative_slope);
}

void ggml_cuda_op_leaky_relu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    float negative_slope;
    memcpy(&negative_slope, dst->op_params, sizeof(float));

    if (src0->type == GGML_TYPE_F16) {
        leaky_relu_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), negative_slope, stream);
    } else {
        leaky_relu_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), negative_slope, stream);
    }
}

/* fused unary + mul */

template <float (*op)(float)>
static void ggml_cuda_op_unary_mul_impl(ggml_backend_cuda_context & ctx, ggml_tensor * unary_node, ggml_tensor * mul_node) {
    // unary_node: UNARY op applied to unary_node->src[0]
    // mul_node:   MUL(a, b) where one of a/b is unary_node
    // Output goes to mul_node->data

    const ggml_tensor * unary_src = unary_node->src[0];  // input to the unary op
    const ggml_tensor * other_src = (mul_node->src[0] == unary_node) ? mul_node->src[1] : mul_node->src[0];

    GGML_ASSERT(ggml_is_contiguous_1(unary_src));
    GGML_ASSERT(unary_src->nb[0] == ggml_element_size(unary_src));
    GGML_ASSERT(ggml_is_contiguous_1(other_src));
    GGML_ASSERT(other_src->nb[0] == ggml_element_size(other_src));
    GGML_ASSERT(ggml_are_same_shape(unary_src, other_src));

    GGML_ASSERT(unary_src->type == GGML_TYPE_F32 || unary_src->type == GGML_TYPE_F16 || unary_src->type == GGML_TYPE_BF16);
    GGML_ASSERT(unary_src->type == other_src->type);
    GGML_ASSERT(unary_src->type == mul_node->type);

    cudaStream_t stream = ctx.stream();

    const int64_t k  = ggml_nelements(mul_node);
    const int64_t nc = unary_src->ne[0];
    const int64_t unary_stride = unary_src->nb[1];
    const int64_t other_stride = other_src->nb[1];

    if (unary_src->type == GGML_TYPE_F16) {
        unary_gated_cuda<op>((const half *) unary_src->data, (const half *) other_src->data,
                             (half *) mul_node->data, k, nc,
                             unary_stride / sizeof(half), other_stride / sizeof(half), stream);
    } else if (unary_src->type == GGML_TYPE_BF16) {
        unary_gated_cuda<op>((const nv_bfloat16 *) unary_src->data, (const nv_bfloat16 *) other_src->data,
                             (nv_bfloat16 *) mul_node->data, k, nc,
                             unary_stride / sizeof(nv_bfloat16), other_stride / sizeof(nv_bfloat16), stream);
    } else {
        unary_gated_cuda<op>((const float *) unary_src->data, (const float *) other_src->data,
                             (float *) mul_node->data, k, nc,
                             unary_stride / sizeof(float), other_stride / sizeof(float), stream);
    }
}

void ggml_cuda_op_unary_mul(ggml_backend_cuda_context & ctx, ggml_tensor * unary_node, ggml_tensor * mul_node) {
    switch (ggml_get_unary_op(unary_node)) {
        case GGML_UNARY_OP_SILU:
            ggml_cuda_op_unary_mul_impl<op_silu>(ctx, unary_node, mul_node);
            break;
        case GGML_UNARY_OP_SIGMOID:
            ggml_cuda_op_unary_mul_impl<op_sigmoid>(ctx, unary_node, mul_node);
            break;
        case GGML_UNARY_OP_SOFTPLUS:
            ggml_cuda_op_unary_mul_impl<op_softplus>(ctx, unary_node, mul_node);
            break;
        default:
            GGML_ABORT("Unsupported unary op for fused unary+mul");
    }
}

/* fused relu + sqr */

void ggml_cuda_op_relu_sqr(ggml_backend_cuda_context & ctx, ggml_tensor * relu_node, ggml_tensor * sqr_node) {
    const ggml_tensor * src = relu_node->src[0];
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);
    GGML_ASSERT(src->type == sqr_node->type);

    const int k = ggml_nelements(src);
    if (src->type == GGML_TYPE_F16) {
        unary_cuda<op_relu_sqr>((const half *)src->data, (half *)sqr_node->data, k, stream);
    } else {
        unary_cuda<op_relu_sqr>((const float *)src->data, (float *)sqr_node->data, k, stream);
    }
}

/* fused scale and activation */

template <typename T>
struct ggml_cuda_scaled_unary_load {
    const T * data;
    __device__ __forceinline__ T operator()(int i) const { return data[i]; }
};

template <typename T>
struct ggml_cuda_scaled_unary_convert_load {
    const T * data;
    float * raw;
    __device__ __forceinline__ float operator()(int i) const {
        const float value = ggml_cuda_cast<float>(data[i]);
        raw[i] = value;
        return value;
    }
};

template <float (*op)(float), typename T, typename Load>
static __device__ __forceinline__ void scaled_unary_impl(Load x, T * pre_dst, T * unary_dst, T * dst, int k, float scale0, float bias0, float scale1, float bias1, bool before, bool after) {
    ggml_cuda_pdl_lc();
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    ggml_cuda_pdl_sync();
    float value = ggml_cuda_cast<float>(x(i));
    if (before) {
        const T pre = ggml_cuda_cast<T>(scale0 * value + bias0);
        pre_dst[i] = pre;
        value = ggml_cuda_cast<float>(pre);
    }
    const T unary = ggml_cuda_cast<T>(op(value));
    unary_dst[i] = unary;
    value = ggml_cuda_cast<float>(unary);
    if (after) {
        value = scale1 * value + bias1;
    }
    if (after) {
        dst[i] = ggml_cuda_cast<T>(value);
    }
}

template <float (*op)(float), typename T, bool before, bool after>
static __global__ void scaled_unary_kernel(const T * x, T * pre_dst, T * unary_dst, T * dst, int k, float scale0, float bias0, float scale1, float bias1) {
    scaled_unary_impl<op, T>(ggml_cuda_scaled_unary_load<T>{x}, pre_dst, unary_dst, dst, k, scale0, bias0, scale1, bias1, before, after);
}

template <float (*op)(float), typename T>
static __global__ void scaled_unary_convert_kernel(const void * x, float * raw, float * pre_dst, float * unary_dst, float * dst, int k, float scale0, float bias0, float scale1, float bias1, bool before, bool after) {
    scaled_unary_impl<op, float>(ggml_cuda_scaled_unary_convert_load<T>{(const T *) x, raw}, pre_dst, unary_dst, dst, k, scale0, bias0, scale1, bias1, before, after);
}

template <float (*op)(float), typename T>
static void scaled_unary_cuda(const ggml_tensor * first, const ggml_tensor * unary, const ggml_tensor * last, bool before, bool after, cudaStream_t stream) {
    const int k = ggml_nelements(last);
    const float scale0 = before ? ggml_get_op_params_f32(first, 0) : 1.0f;
    const float bias0  = before ? ggml_get_op_params_f32(first, 1) : 0.0f;
    const float scale1 = after  ? ggml_get_op_params_f32(last, 0)  : 1.0f;
    const float bias1  = after  ? ggml_get_op_params_f32(last, 1)  : 0.0f;
    const ggml_cuda_kernel_launch_params launch((k + CUDA_NEG_BLOCK_SIZE - 1)/CUDA_NEG_BLOCK_SIZE, CUDA_NEG_BLOCK_SIZE, 0, stream);
    const T * x = (const T *) first->src[0]->data;
    T * pre_dst = before ? (T *) first->data : nullptr;
    T * unary_dst = (T *) unary->data;
    T * dst = (T *) last->data;
    if (before && after) {
        ggml_cuda_kernel_launch(scaled_unary_kernel<op, T, true, true>, launch, x, pre_dst, unary_dst, dst, k, scale0, bias0, scale1, bias1);
    } else if (before) {
        ggml_cuda_kernel_launch(scaled_unary_kernel<op, T, true, false>, launch, x, pre_dst, unary_dst, dst, k, scale0, bias0, scale1, bias1);
    } else {
        ggml_cuda_kernel_launch(scaled_unary_kernel<op, T, false, true>, launch, x, pre_dst, unary_dst, dst, k, scale0, bias0, scale1, bias1);
    }
}

template <float (*op)(float)>
static void ggml_cuda_op_scaled_unary_impl(ggml_backend_cuda_context & ctx, ggml_tensor * first, ggml_tensor * unary, ggml_tensor * last) {
    const bool before = first->op == GGML_OP_SCALE;
    const bool after = last->op == GGML_OP_SCALE;
    if (first->type == GGML_TYPE_BF16) {
        scaled_unary_cuda<op, nv_bfloat16>(first, unary, last, before, after, ctx.stream());
    } else {
        scaled_unary_cuda<op, float>(first, unary, last, before, after, ctx.stream());
    }
}

void ggml_cuda_op_scaled_unary(ggml_backend_cuda_context & ctx, ggml_tensor * first, ggml_tensor * unary, ggml_tensor * last) {
    switch (ggml_get_unary_op(unary)) {
        case GGML_UNARY_OP_SILU:
            ggml_cuda_op_scaled_unary_impl<op_silu>(ctx, first, unary, last);
            break;
        case GGML_UNARY_OP_SIGMOID:
            ggml_cuda_op_scaled_unary_impl<op_sigmoid>(ctx, first, unary, last);
            break;
        default:
            GGML_ABORT("Unsupported scaled unary op");
    }
}

template <float (*op)(float), typename T>
static void scaled_unary_convert_cuda(ggml_backend_cuda_context & ctx, const void * src, ggml_tensor * mm, const ggml_cuda_scaled_unary_args & args) {
    const bool before = args.first->op == GGML_OP_SCALE;
    const bool after = args.last->op == GGML_OP_SCALE;
    const int k = ggml_nelements(mm);
    const ggml_cuda_kernel_launch_params launch((k + CUDA_NEG_BLOCK_SIZE - 1)/CUDA_NEG_BLOCK_SIZE, CUDA_NEG_BLOCK_SIZE, 0, ctx.stream());
    ggml_cuda_kernel_launch(scaled_unary_convert_kernel<op, T>, launch, src, (float *) mm->data,
            before ? (float *) args.first->data : nullptr, (float *) args.unary->data, (float *) args.last->data, k,
            before ? ggml_get_op_params_f32(args.first, 0) : 1.0f, before ? ggml_get_op_params_f32(args.first, 1) : 0.0f,
            after ? ggml_get_op_params_f32(args.last, 0) : 1.0f, after ? ggml_get_op_params_f32(args.last, 1) : 0.0f, before, after);
}

void ggml_cuda_op_scaled_unary_convert(ggml_backend_cuda_context & ctx, ggml_type type, const void * src, ggml_tensor * mm, const ggml_cuda_scaled_unary_args & args) {
    GGML_ASSERT(type == GGML_TYPE_F16 || type == GGML_TYPE_BF16);
    GGML_ASSERT(mm->type == GGML_TYPE_F32 && args.first->type == GGML_TYPE_F32 && args.unary->type == GGML_TYPE_F32 && args.last->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(mm) && ggml_are_same_shape(mm, args.last));
    const auto launch = [&](auto tag) {
        using T = decltype(tag);
        switch (ggml_get_unary_op(args.unary)) {
            case GGML_UNARY_OP_SILU: scaled_unary_convert_cuda<op_silu, T>(ctx, src, mm, args); break;
            case GGML_UNARY_OP_SIGMOID: scaled_unary_convert_cuda<op_sigmoid, T>(ctx, src, mm, args); break;
            default: GGML_ABORT("Unsupported scaled unary op");
        }
    };
    if (type == GGML_TYPE_F16) { launch(half{}); } else { launch(nv_bfloat16{}); }
}
