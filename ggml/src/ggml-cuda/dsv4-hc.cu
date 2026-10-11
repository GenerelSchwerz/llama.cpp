#include "common.cuh"
#include "dsv4-hc.cuh"
#include "convert.cuh"
#include "quantize.cuh"


static constexpr int DSV4_HC = 4;


static __device__ void dsv4_hc_comb_norm_cols(float * comb, float eps) {
    for (int idst = 0; idst < DSV4_HC; ++idst) {
        float sum = eps;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __device__ void dsv4_hc_comb_norm_rows(float * comb, float eps) {
    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float sum = eps;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __global__ void dsv4_hc_comb_f32(
        const float * mixes,
        const float * scale,
        const float * base,
        float * dst,
        int64_t n_tokens,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        float eps,
        int32_t n_iter) {
    constexpr int comb_offset = 2*DSV4_HC;

    ggml_cuda_pdl_lc();
    const int64_t it = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (it >= n_tokens) {
        return;
    }

    ggml_cuda_pdl_sync();

    const float scale_comb = scale[2*ss0];
    float comb[DSV4_HC*DSV4_HC];

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float max = -INFINITY;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = mixes[(comb_offset + idx)*sm0 + it*sm1] * scale_comb + base[(comb_offset + idx)*sb0];
            comb[idx] = v;
            max = fmaxf(max, v);
        }

        float sum = 0.0f;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = expf(comb[idx] - max);
            comb[idx] = v;
            sum += v;
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            comb[idx] = comb[idx] * inv_sum + eps;
        }
    }

    dsv4_hc_comb_norm_cols(comb, eps);
    for (int32_t i = 1; i < n_iter; ++i) {
        dsv4_hc_comb_norm_rows(comb, eps);
        dsv4_hc_comb_norm_cols(comb, eps);
    }

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            dst[idst*sd0 + isrc*sd1 + it*sd2] = comb[idx];
        }
    }
}

struct ggml_cuda_hc_gate_f32 {
    const float * data;
    __device__ __forceinline__ float operator()(int64_t i) const { return data[i]; }
};

template <typename T>
struct ggml_cuda_hc_gate_convert {
    const T * data;
    float * raw;
    __device__ __forceinline__ float operator()(int64_t i) const {
        const float value = ggml_cuda_cast<float>(data[i]);
        raw[i] = value;
        return value;
    }
};

struct ggml_cuda_hc_pre_store {
    __device__ __forceinline__ void operator()(float * dst, int64_t index, int64_t, int64_t, float value) const { dst[index] = value; }
};

struct ggml_cuda_hc_pre_image_store {
    ggml_cuda_hc_pre_emit_data images;
    int64_t cols;
    __device__ __forceinline__ void operator()(float * dst, int64_t index, int64_t col, int64_t row, float value) const {
        dst[index] = value;
        const int64_t flat = row*cols + col;
        if (images.f16) { ((half *) images.f16)[flat] = ggml_cuda_cast<half>(value); }
        if (images.bf16) { ((nv_bfloat16 *) images.bf16)[flat] = ggml_cuda_cast<nv_bfloat16>(value); }
    }
};

template <bool Typed>
struct ggml_cuda_hc_pre_q8_store {
    block_q8_1 * image;
    int64_t cols;
    int64_t padded;
    ggml_cuda_hc_pre_emit_data images;
    __device__ __forceinline__ void operator()(float * dst, int64_t index, int64_t, int64_t, float value) const {
        dst[index] = value;
        if constexpr (Typed) {
            if (images.f16) { ((half *) images.f16)[index] = ggml_cuda_cast<half>(value); }
            if (images.bf16) { ((nv_bfloat16 *) images.bf16)[index] = ggml_cuda_cast<nv_bfloat16>(value); }
        }
        const int64_t row = index/cols, column = index%cols;
        const int lane = column%QK8_1;
        const int64_t block = (row*padded + column)/QK8_1;
        ggml_cuda_store_q8_1(image[block], lane, value);
        if (column/QK8_1 == cols/QK8_1 - 1) {
            for (int64_t tail = cols/QK8_1; tail < padded/QK8_1; ++tail) {
                block_q8_1 & zero = image[row*(padded/QK8_1) + tail];
                zero.qs[lane] = 0;
                if (lane == 0) { zero.ds = make_half2(0.0f, 0.0f); }
            }
        }
    }
};

template <bool gated, typename Gate>
static __device__ __forceinline__ float dsv4_hc_pre_sum(
        const float * x, const Gate weights, int64_t i0, int64_t it, int64_t hc,
        int64_t sx0, int64_t sx1, int64_t sx2, int64_t sw0, int64_t sw1, int64_t sw2) {
    float sum = 0.0f;
    for (int64_t ih = 0; ih < hc; ++ih) {
        const float xv = x[i0*sx0 + ih*sx1 + it*sx2];
        float wv;
        if constexpr (gated) {
            wv = 1.0f / (1.0f + expf(-weights(i0*sw0 + ih*sw1 + it*sw2)));
        } else {
            wv = weights(ih*sw0 + it*sw1);
        }
        sum += xv * wv;
    }

    return sum;
}

template <bool gated, typename Gate, typename Store = ggml_cuda_hc_pre_store>
static __device__ __forceinline__ void dsv4_hc_pre_impl(
        const float * x,
        const Gate weights,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sx2,
        int64_t sw0,
        int64_t sw1,
        int64_t sw2,
        int64_t sd0,
        int64_t sd1,
        float   scale, Store store = {}) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0 = ir % n_embd;
    const int64_t it = ir / n_embd;

    const float sum = dsv4_hc_pre_sum<gated>(x, weights, i0, it, hc, sx0, sx1, sx2, sw0, sw1, sw2);

    store(dst, i0*sd0 + it*sd1, i0, it, scale * sum);
}

template <bool gated>
static __global__ void dsv4_hc_pre_f32(
        const float * x,
        const float * weights,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sx2,
        int64_t sw0,
        int64_t sw1,
        int64_t sw2,
        int64_t sd0,
        int64_t sd1,
        float   scale) {
    dsv4_hc_pre_impl<gated>(
            x, ggml_cuda_hc_gate_f32{weights}, dst, n_embd, hc, n_tokens,
            sx0, sx1, sx2, sw0, sw1, sw2, sd0, sd1, scale);
}

template <bool gated>
static __global__ void dsv4_hc_pre_images(
        const float * x, const float * weights, float * dst,
        int64_t n_embd, int64_t hc, int64_t n_tokens,
        int64_t sx0, int64_t sx1, int64_t sx2,
        int64_t sw0, int64_t sw1, int64_t sw2,
        int64_t sd0, int64_t sd1, float scale, ggml_cuda_hc_pre_emit_data images) {
    dsv4_hc_pre_impl<gated>(x, ggml_cuda_hc_gate_f32{weights}, dst, n_embd, hc, n_tokens,
        sx0, sx1, sx2, sw0, sw1, sw2, sd0, sd1, scale, ggml_cuda_hc_pre_image_store{images, n_embd});
}

template <bool gated, bool Typed>
static __global__ void dsv4_hc_pre_q8(
        const float * x,
        const float * weights,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sx2,
        int64_t sw0,
        int64_t sw1,
        int64_t sw2,
        int64_t sd0,
        int64_t sd1,
        float   scale, ggml_cuda_hc_pre_q8_store<Typed> write) {
    dsv4_hc_pre_impl<gated>(
            x, ggml_cuda_hc_gate_f32{weights}, dst, n_embd, hc, n_tokens,
            sx0, sx1, sx2, sw0, sw1, sw2, sd0, sd1, scale, write);
}

template <typename T>
static __global__ void dsv4_hc_pre_convert(
        const float * x, const void * weights, float * raw, float * dst,
        int64_t n_embd, int64_t hc, int64_t n_tokens,
        int64_t sx0, int64_t sx1, int64_t sx2,
        int64_t sw0, int64_t sw1, int64_t sw2,
        int64_t sd0, int64_t sd1, float scale) {
    dsv4_hc_pre_impl<true>(
            x, ggml_cuda_hc_gate_convert<T>{(const T *) weights, raw}, dst, n_embd, hc, n_tokens,
            sx0, sx1, sx2, sw0, sw1, sw2, sd0, sd1, scale);
}

template <typename T>
static __global__ void dsv4_hc_pre_convert_images(
        const float * x, const void * weights, float * raw, float * dst,
        int64_t n_embd, int64_t hc, int64_t n_tokens,
        int64_t sx0, int64_t sx1, int64_t sx2,
        int64_t sw0, int64_t sw1, int64_t sw2,
        int64_t sd0, int64_t sd1, float scale, ggml_cuda_hc_pre_emit_data images) {
    dsv4_hc_pre_impl<true>(
            x, ggml_cuda_hc_gate_convert<T>{(const T *) weights, raw}, dst, n_embd, hc, n_tokens,
            sx0, sx1, sx2, sw0, sw1, sw2, sd0, sd1, scale, ggml_cuda_hc_pre_image_store{images, n_embd});
}

template <typename T, bool Typed>
static __global__ void dsv4_hc_pre_convert_q8(
        const float * x, const void * weights, float * raw, float * dst,
        int64_t n_embd, int64_t hc, int64_t n_tokens,
        int64_t sx0, int64_t sx1, int64_t sx2,
        int64_t sw0, int64_t sw1, int64_t sw2,
        int64_t sd0, int64_t sd1, float scale, ggml_cuda_hc_pre_q8_store<Typed> write) {
    dsv4_hc_pre_impl<true>(
            x, ggml_cuda_hc_gate_convert<T>{(const T *) weights, raw}, dst, n_embd, hc, n_tokens,
            sx0, sx1, sx2, sw0, sw1, sw2, sd0, sd1, scale, write);
}

template <bool has_comb>
static __global__ void dsv4_hc_post_f32(
        const float * x,
        const float * residual,
        const float * post,
        const float * comb,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sr0,
        int64_t sr1,
        int64_t sr2,
        int64_t sp0,
        int64_t sp1,
        int64_t sc0,
        int64_t sc1,
        int64_t sc2,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * hc * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0   = ir % n_embd;
    const int64_t idst = (ir / n_embd) % hc;
    const int64_t it   = ir / (n_embd * hc);

    float sum = x[i0*sx0 + it*sx1] * post[idst*sp0 + it*sp1];
    if constexpr (has_comb) {
        for (int64_t isrc = 0; isrc < hc; ++isrc) {
            sum += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
        }
    } else {
        sum += residual[i0*sr0 + idst*sr1 + it*sr2];
    }

    dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
}

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc_mix_dim = (2 + DSV4_HC)*DSV4_HC;

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[0] == DSV4_HC);
    GGML_ASSERT(dst->ne[1] == DSV4_HC);
    GGML_ASSERT(dst->ne[2] == mixes->ne[1]);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(base->ne[0] == hc_mix_dim);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, base,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    const int64_t n_tokens = mixes->ne[1];
    const float eps = ggml_get_op_params_f32(dst, 0);
    const int32_t n_iter = ggml_get_op_params_i32(dst, 1);

    const int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_tokens + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_comb_f32, launch_params,
            (const float *) mixes->data, (const float *) scale->data, (const float *) base->data, (float *) dst->data,
            n_tokens,
            nbm0 / sizeof(float), nbm1 / sizeof(float),
            nbs0 / sizeof(float),
            nbb0 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
            eps, n_iter);
}

void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    const int64_t n_embd   = x->ne[0];
    const int64_t hc       = x->ne[1];
    const int64_t n_tokens = x->ne[2];

    const float scale = ggml_get_op_params_f32(dst, 0);
    const bool  gated = ggml_get_op_params_i32(dst, 1) != 0;

    const int block_size = 256;
    const int64_t nr = n_embd * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    auto kernel = gated ? dsv4_hc_pre_f32<true> : dsv4_hc_pre_f32<false>;
    ggml_cuda_kernel_launch(kernel, launch_params,
            (const float *) x->data, (const float *) weights->data, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float), nbx2 / sizeof(float),
            nbw0 / sizeof(float), nbw1 / sizeof(float), nbw2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float),
            scale);
}

void ggml_cuda_op_dsv4_hc_pre_emit(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_hc_pre_emit_data & images) {
    const ggml_tensor * x = dst->src[0], * weights = dst->src[1];
    GGML_ASSERT(x->type == GGML_TYPE_F32 && weights->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(dst) && (images.f16 || images.bf16));
    const int64_t nr = x->ne[0]*x->ne[2];
    const ggml_cuda_kernel_launch_params params(dim3((nr + 255)/256), dim3(256), 0, ctx.stream());
    auto kernel = ggml_get_op_params_i32(dst, 1) ? dsv4_hc_pre_images<true> : dsv4_hc_pre_images<false>;
    ggml_cuda_kernel_launch(kernel, params, (const float *) x->data, (const float *) weights->data, (float *) dst->data,
        x->ne[0], x->ne[1], x->ne[2], x->nb[0]/sizeof(float), x->nb[1]/sizeof(float), x->nb[2]/sizeof(float),
        weights->nb[0]/sizeof(float), weights->nb[1]/sizeof(float), weights->nb[2]/sizeof(float),
        dst->nb[0]/sizeof(float), dst->nb[1]/sizeof(float), ggml_get_op_params_f32(dst, 0), images);
}

template <bool Typed>
static void ggml_cuda_hc_pre_q8_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst, void * image, int64_t cols, int64_t padded, const ggml_cuda_hc_pre_emit_data & images) {
    const ggml_tensor * x = dst->src[0], * weight = dst->src[1];
    GGML_ASSERT(image && ggml_is_contiguous(dst) && dst->ne[0]%QK8_1 == 0 && cols%QK8_1 == 0);
    const int64_t count = ggml_nelements(dst);
    const ggml_cuda_kernel_launch_params params = {dim3((count + 255)/256, 1, 1), dim3(256, 1, 1), 0, ctx.stream()};
    auto kernel = ggml_get_op_params_i32(dst, 1) ? dsv4_hc_pre_q8<true, Typed> : dsv4_hc_pre_q8<false, Typed>;
    ggml_cuda_kernel_launch(kernel, params,
        (const float *) x->data, (const float *) weight->data, (float *) dst->data, x->ne[0], x->ne[1], x->ne[2],
        x->nb[0]/sizeof(float), x->nb[1]/sizeof(float), x->nb[2]/sizeof(float),
        weight->nb[0]/sizeof(float), weight->nb[1]/sizeof(float), weight->nb[2]/sizeof(float),
        dst->nb[0]/sizeof(float), dst->nb[1]/sizeof(float), ggml_get_op_params_f32(dst, 0),
        ggml_cuda_hc_pre_q8_store<Typed>{(block_q8_1 *) image, cols, padded, images});
}

void ggml_cuda_op_dsv4_hc_pre_q8(ggml_backend_cuda_context & ctx, ggml_tensor * dst, void * image, int64_t cols, int64_t padded, const ggml_cuda_hc_pre_emit_data & images) {
    if (images.f16 || images.bf16) { ggml_cuda_hc_pre_q8_launch<true>(ctx, dst, image, cols, padded, images); }
    else { ggml_cuda_hc_pre_q8_launch<false>(ctx, dst, image, cols, padded, images); }
}

static void ggml_cuda_hc_pre_convert_mmq(ggml_backend_cuda_context & ctx, ggml_type type, const void * weights,
        ggml_tensor * mm, ggml_tensor * dst, const ggml_cuda_hc_pre_emit_data * images, const ggml_cuda_hc_pre_mmq_emit_data & mmq);

void ggml_cuda_op_dsv4_hc_pre_convert(ggml_backend_cuda_context & ctx, ggml_type type, const void * weights, ggml_tensor * mm, ggml_tensor * dst, const ggml_cuda_hc_pre_emit_data * images, const ggml_cuda_hc_pre_q8_emit_data * q8, const ggml_cuda_hc_pre_mmq_emit_data * mmq) {
    const ggml_tensor * x = dst->src[0];
    const ggml_tensor * view = dst->src[1];
    GGML_ASSERT(type == GGML_TYPE_F16 || type == GGML_TYPE_BF16);
    GGML_ASSERT(x->type == GGML_TYPE_F32 && mm->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_get_op_params_i32(dst, 1) && ggml_is_contiguous(mm) && ggml_is_contiguous(view));

    if (mmq) {
        ggml_cuda_hc_pre_convert_mmq(ctx, type, weights, mm, dst, images, *mmq);
        return;
    }

    const int64_t nr = x->ne[0]*x->ne[2];
    const dim3 block_dims(256, 1, 1);
    const dim3 grid_dims((nr + block_dims.x - 1)/block_dims.x, 1, 1);
    const ggml_cuda_kernel_launch_params params(grid_dims, block_dims, 0, ctx.stream());
    const auto launch = [&](auto tag) {
        using T = decltype(tag);
        if (q8) {
            const auto emit = [&](auto typed) {
                constexpr bool Typed = decltype(typed)::value;
                ggml_cuda_kernel_launch(dsv4_hc_pre_convert_q8<T, Typed>, params,
                    (const float *) x->data, weights, (float *) mm->data, (float *) dst->data,
                    x->ne[0], x->ne[1], x->ne[2],
                    x->nb[0]/sizeof(float), x->nb[1]/sizeof(float), x->nb[2]/sizeof(float),
                    view->nb[0]/sizeof(float), view->nb[1]/sizeof(float), view->nb[2]/sizeof(float),
                    dst->nb[0]/sizeof(float), dst->nb[1]/sizeof(float), ggml_get_op_params_f32(dst, 0),
                    ggml_cuda_hc_pre_q8_store<Typed>{(block_q8_1 *) q8->data, q8->cols, q8->padded, images ? *images : ggml_cuda_hc_pre_emit_data{}});
            };
            if (images) { emit(std::true_type{}); } else { emit(std::false_type{}); }
        } else if (images) {
            ggml_cuda_kernel_launch(dsv4_hc_pre_convert_images<T>, params,
                    (const float *) x->data, weights, (float *) mm->data, (float *) dst->data,
                    x->ne[0], x->ne[1], x->ne[2],
                    x->nb[0]/sizeof(float), x->nb[1]/sizeof(float), x->nb[2]/sizeof(float),
                    view->nb[0]/sizeof(float), view->nb[1]/sizeof(float), view->nb[2]/sizeof(float),
                    dst->nb[0]/sizeof(float), dst->nb[1]/sizeof(float), ggml_get_op_params_f32(dst, 0), *images);
        } else {
            ggml_cuda_kernel_launch(dsv4_hc_pre_convert<T>, params,
                    (const float *) x->data, weights, (float *) mm->data, (float *) dst->data,
                    x->ne[0], x->ne[1], x->ne[2],
                    x->nb[0]/sizeof(float), x->nb[1]/sizeof(float), x->nb[2]/sizeof(float),
                    view->nb[0]/sizeof(float), view->nb[1]/sizeof(float), view->nb[2]/sizeof(float),
                    dst->nb[0]/sizeof(float), dst->nb[1]/sizeof(float), ggml_get_op_params_f32(dst, 0));
        }
    };
    if (type == GGML_TYPE_F16) {
        launch(half{});
    } else {
        launch(nv_bfloat16{});
    }
}

void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * residual = dst->src[1];
    const ggml_tensor * post     = dst->src[2];
    const ggml_tensor * comb     = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(comb == nullptr || comb->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,        nb);
    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb);
    GGML_TENSOR_LOCALS(size_t, nbp, post,     nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);

    const size_t nbc0 = comb ? comb->nb[0] : 0;
    const size_t nbc1 = comb ? comb->nb[1] : 0;
    const size_t nbc2 = comb ? comb->nb[2] : 0;

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t hc       = residual->ne[1];

    const int block_size = 256;
    const int64_t nr = n_embd * hc * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    auto kernel = comb ? dsv4_hc_post_f32<true> : dsv4_hc_post_f32<false>;
    ggml_cuda_kernel_launch(kernel, launch_params,
            (const float *) x->data, (const float *) residual->data,
            (const float *) post->data, comb ? (const float *) comb->data : nullptr, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float),
            nbr0 / sizeof(float), nbr1 / sizeof(float), nbr2 / sizeof(float),
            nbp0 / sizeof(float), nbp1 / sizeof(float),
            nbc0 / sizeof(float), nbc1 / sizeof(float), nbc2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float));
}

// Keep the single-image epilogue in place to preserve its scheduling.
template <mmq_q8_1_ds_layout Layout>
static __device__ __forceinline__ void dsv4_hc_pre_mmq_store(float value, int64_t column, int64_t row,
        block_q8_1_mmq * image, int64_t padded, int64_t rows) {
    const int lane = threadIdx.x%WARP_SIZE;
    float amax = warp_reduce_max<WARP_SIZE>(fabsf(value));
    if constexpr (Layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        __shared__ float maxima[CUDA_QUANTIZE_BLOCK_SIZE/WARP_SIZE];
        const int warp = threadIdx.x/WARP_SIZE;
        if (lane == 0) { maxima[warp] = amax; }
        __syncthreads();
        amax = fmaxf(maxima[warp & ~1], maxima[warp | 1]);
    }
    float sum;
    if constexpr (Layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        const int first = lane & ~3;
        const float x0 = __shfl_sync(0xFFFFFFFF, value, first, WARP_SIZE);
        const float x1 = __shfl_sync(0xFFFFFFFF, value, first + 1, WARP_SIZE);
        const float x2 = __shfl_sync(0xFFFFFFFF, value, first + 2, WARP_SIZE);
        const float x3 = __shfl_sync(0xFFFFFFFF, value, first + 3, WARP_SIZE);
        sum = x0 + x1 + x2 + x3;
        constexpr int vals_per_sum = Layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;
#pragma unroll
        for (int offset = vals_per_sum/2; offset >= 4; offset >>= 1) { sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE); }
    }
    const float d_inv = 127.0f/amax;
    const float d = 1.0f/d_inv;
    const int64_t ib = column/QK8_1_MMQ*rows + row;
    const int iqs = column%QK8_1_MMQ;
    image[ib].qs[iqs] = roundf(value*d_inv);
    if constexpr (Layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (iqs%16 == 0 && iqs < 96) {
            image[ib].d2s6[2 + iqs/16] = sum;
            if (iqs%64 == 0) { image[ib].d2s6[iqs/64] = d; }
        }
    } else if (iqs%32 == 0) {
        if constexpr (Layout == MMQ_Q8_1_DS_LAYOUT_DS4) { image[ib].ds4[iqs/32] = make_half2(d, sum); }
        else { image[ib].d4[iqs/32] = d; }
    }
    if (blockIdx.x + 1 == gridDim.x) {
        for (int64_t tail = int64_t(gridDim.x)*blockDim.x + threadIdx.x; tail < padded; tail += blockDim.x) {
            auto & zero = image[tail/QK8_1_MMQ*rows + row];
            const int index = tail%QK8_1_MMQ;
            zero.qs[index] = 0;
            if constexpr (Layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
                if (index%16 == 0 && index < 96) {
                    zero.d2s6[2 + index/16] = 0.0f;
                    if (index%64 == 0) { zero.d2s6[index/64] = 0.0f; }
                }
            } else if (index%32 == 0) {
                if constexpr (Layout == MMQ_Q8_1_DS_LAYOUT_DS4) { zero.ds4[index/32] = make_half2(0.0f, 0.0f); }
                else { zero.d4[index/32] = 0.0f; }
            }
        }
    }
}

template <mmq_q8_1_ds_layout Layout, bool Gated, typename Gate>
static __device__ __forceinline__ void dsv4_hc_pre_mmq_impl(
        const float * x, const Gate weights, float * dst, int64_t cols, int64_t hc,
        int64_t sx0, int64_t sx1, int64_t sx2, int64_t sw0, int64_t sw1, int64_t sw2,
        int64_t sd0, int64_t sd1, float scale, block_q8_1_mmq * image, int64_t padded, int64_t rows,
        ggml_cuda_hc_pre_emit_data images) {
    ggml_cuda_pdl_lc();
    const int64_t column = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t row = blockIdx.y;
    ggml_cuda_pdl_sync();
    float value = 0.0f;
    if (column < cols) {
        value = scale*dsv4_hc_pre_sum<Gated>(x, weights, column, row, hc, sx0, sx1, sx2, sw0, sw1, sw2);
        dst[column*sd0 + row*sd1] = value;
        if (images.f16) { ((half *) images.f16)[row*cols + column] = ggml_cuda_cast<half>(value); }
        if (images.bf16) { ((nv_bfloat16 *) images.bf16)[row*cols + column] = ggml_cuda_cast<nv_bfloat16>(value); }
    }
    const int lane = threadIdx.x%WARP_SIZE;
    float amax = warp_reduce_max<WARP_SIZE>(fabsf(value));
    if constexpr (Layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        __shared__ float maxima[CUDA_QUANTIZE_BLOCK_SIZE/WARP_SIZE];
        const int warp = threadIdx.x/WARP_SIZE;
        if (lane == 0) { maxima[warp] = amax; }
        __syncthreads();
        amax = fmaxf(maxima[warp & ~1], maxima[warp | 1]);
    }
    float sum;
    if constexpr (Layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        const int first = lane & ~3;
        const float x0 = __shfl_sync(0xFFFFFFFF, value, first, WARP_SIZE);
        const float x1 = __shfl_sync(0xFFFFFFFF, value, first + 1, WARP_SIZE);
        const float x2 = __shfl_sync(0xFFFFFFFF, value, first + 2, WARP_SIZE);
        const float x3 = __shfl_sync(0xFFFFFFFF, value, first + 3, WARP_SIZE);
        sum = x0 + x1 + x2 + x3;
        constexpr int vals_per_sum = Layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;
#pragma unroll
        for (int offset = vals_per_sum/2; offset >= 4; offset >>= 1) { sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE); }
    }
    const float d_inv = 127.0f/amax;
    const float d = 1.0f/d_inv;
    const int64_t ib = column/QK8_1_MMQ*rows + row;
    const int iqs = column%QK8_1_MMQ;
    image[ib].qs[iqs] = roundf(value*d_inv);
    if constexpr (Layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (iqs%16 == 0 && iqs < 96) {
            image[ib].d2s6[2 + iqs/16] = sum;
            if (iqs%64 == 0) { image[ib].d2s6[iqs/64] = d; }
        }
    } else if (iqs%32 == 0) {
        if constexpr (Layout == MMQ_Q8_1_DS_LAYOUT_DS4) { image[ib].ds4[iqs/32] = make_half2(d, sum); }
        else { image[ib].d4[iqs/32] = d; }
    }
    if (blockIdx.x + 1 == gridDim.x) {
        for (int64_t tail = int64_t(gridDim.x)*blockDim.x + threadIdx.x; tail < padded; tail += blockDim.x) {
            auto & zero = image[tail/QK8_1_MMQ*rows + row];
            const int index = tail%QK8_1_MMQ;
            zero.qs[index] = 0;
            if constexpr (Layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
                if (index%16 == 0 && index < 96) {
                    zero.d2s6[2 + index/16] = 0.0f;
                    if (index%64 == 0) { zero.d2s6[index/64] = 0.0f; }
                }
            } else if (index%32 == 0) {
                if constexpr (Layout == MMQ_Q8_1_DS_LAYOUT_DS4) { zero.ds4[index/32] = make_half2(0.0f, 0.0f); }
                else { zero.d4[index/32] = 0.0f; }
            }
        }
    }
}

template <mmq_q8_1_ds_layout Layout, bool Gated>
static __global__ void dsv4_hc_pre_mmq(
        const float * x, const float * weights, float * dst, int64_t cols, int64_t hc,
        int64_t sx0, int64_t sx1, int64_t sx2, int64_t sw0, int64_t sw1, int64_t sw2,
        int64_t sd0, int64_t sd1, float scale, block_q8_1_mmq * image, int64_t padded, int64_t rows,
        ggml_cuda_hc_pre_emit_data images) {
    dsv4_hc_pre_mmq_impl<Layout, Gated>(x, ggml_cuda_hc_gate_f32{weights}, dst, cols, hc,
        sx0, sx1, sx2, sw0, sw1, sw2, sd0, sd1, scale, image, padded, rows, images);
}

template <mmq_q8_1_ds_layout Layout, typename T>
static __global__ void dsv4_hc_pre_convert_mmq(
        const float * x, const void * weights, float * raw, float * dst, int64_t cols, int64_t hc,
        int64_t sx0, int64_t sx1, int64_t sx2, int64_t sw0, int64_t sw1, int64_t sw2,
        int64_t sd0, int64_t sd1, float scale, block_q8_1_mmq * image, int64_t padded, int64_t rows,
        ggml_cuda_hc_pre_emit_data images) {
    dsv4_hc_pre_mmq_impl<Layout, true>(x, ggml_cuda_hc_gate_convert<T>{(const T *) weights, raw}, dst, cols, hc,
        sx0, sx1, sx2, sw0, sw1, sw2, sd0, sd1, scale, image, padded, rows, images);
}

template <mmq_q8_1_ds_layout Layout>
static void ggml_cuda_hc_pre_mmq_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst, void * image, int64_t padded, const ggml_cuda_hc_pre_emit_data & images) {
    const ggml_tensor * x = dst->src[0], * weights = dst->src[1];
    const int block = CUDA_QUANTIZE_BLOCK_SIZE;
    const dim3 grid((dst->ne[0] + block - 1)/block, dst->ne[1]);
    auto kernel = ggml_get_op_params_i32(dst, 1) != 0 ? dsv4_hc_pre_mmq<Layout, true> : dsv4_hc_pre_mmq<Layout, false>;
    const ggml_cuda_kernel_launch_params launch(grid, dim3(block), 0, ctx.stream());
    ggml_cuda_kernel_launch(kernel, launch, (const float *) x->data, (const float *) weights->data, (float *) dst->data,
        dst->ne[0], x->ne[1], int64_t(x->nb[0]/4), int64_t(x->nb[1]/4), int64_t(x->nb[2]/4),
        int64_t(weights->nb[0]/4), int64_t(weights->nb[1]/4), int64_t(weights->nb[2]/4),
        int64_t(dst->nb[0]/4), int64_t(dst->nb[1]/4), ggml_get_op_params_f32(dst, 0), (block_q8_1_mmq *) image, padded, dst->ne[1], images);
}

void ggml_cuda_op_dsv4_hc_pre_mmq(ggml_backend_cuda_context & ctx, ggml_tensor * dst, void * image, int64_t padded, int layout, const ggml_cuda_hc_pre_emit_data & images) {
    switch (layout) {
        case MMQ_Q8_1_DS_LAYOUT_D4: ggml_cuda_hc_pre_mmq_launch<MMQ_Q8_1_DS_LAYOUT_D4>(ctx, dst, image, padded, images); break;
        case MMQ_Q8_1_DS_LAYOUT_DS4: ggml_cuda_hc_pre_mmq_launch<MMQ_Q8_1_DS_LAYOUT_DS4>(ctx, dst, image, padded, images); break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6: ggml_cuda_hc_pre_mmq_launch<MMQ_Q8_1_DS_LAYOUT_D2S6>(ctx, dst, image, padded, images); break;
        default: GGML_ABORT("invalid HC MMQ layout");
    }
}

template <bool Gated, typename Gate>
static __global__ void dsv4_hc_pre_mmq_multi(const float * x, Gate weights, float * dst, int64_t cols, int64_t hc,
        int64_t sx0, int64_t sx1, int64_t sx2, int64_t sw0, int64_t sw1, int64_t sw2,
        int64_t sd0, int64_t sd1, float scale, ggml_cuda_hc_pre_mmq_emit_data mmq, int64_t rows, ggml_cuda_hc_pre_emit_data images) {
    ggml_cuda_pdl_lc();
    const int64_t column = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t row = blockIdx.y;
    ggml_cuda_pdl_sync();
    float value = 0.0f;
    if (column < cols) {
        value = scale*dsv4_hc_pre_sum<Gated>(x, weights, column, row, hc, sx0, sx1, sx2, sw0, sw1, sw2);
        dst[column*sd0 + row*sd1] = value;
        if (images.f16) { ((half *) images.f16)[row*cols + column] = ggml_cuda_cast<half>(value); }
        if (images.bf16) { ((nv_bfloat16 *) images.bf16)[row*cols + column] = ggml_cuda_cast<nv_bfloat16>(value); }
    }
    auto * d4 = (block_q8_1_mmq *) mmq.images[0];
    auto * ds4 = (block_q8_1_mmq *) mmq.images[1];
    auto * d2s6 = (block_q8_1_mmq *) mmq.images[2];
    const int64_t padded = mmq.padded;
    if (d4) { dsv4_hc_pre_mmq_store<MMQ_Q8_1_DS_LAYOUT_D4>(value, column, row, d4, padded, rows); }
    if (ds4) { dsv4_hc_pre_mmq_store<MMQ_Q8_1_DS_LAYOUT_DS4>(value, column, row, ds4, padded, rows); }
    if (d2s6) { dsv4_hc_pre_mmq_store<MMQ_Q8_1_DS_LAYOUT_D2S6>(value, column, row, d2s6, padded, rows); }
}


template <bool Gated, typename Gate>
static void ggml_cuda_hc_pre_mmq_multi_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        Gate weights, const ggml_cuda_hc_pre_mmq_emit_data & mmq, const ggml_cuda_hc_pre_emit_data & images) {
    const ggml_tensor * x = dst->src[0], * view = dst->src[1];
    const int block = CUDA_QUANTIZE_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params params(dim3((dst->ne[0] + block - 1)/block, dst->ne[1]), dim3(block), 0, ctx.stream());
    ggml_cuda_kernel_launch(dsv4_hc_pre_mmq_multi<Gated, Gate>, params, (const float *) x->data, weights,
        (float *) dst->data, dst->ne[0], x->ne[1], int64_t(x->nb[0]/4), int64_t(x->nb[1]/4), int64_t(x->nb[2]/4),
        int64_t(view->nb[0]/4), int64_t(view->nb[1]/4), int64_t(view->nb[2]/4),
        int64_t(dst->nb[0]/4), int64_t(dst->nb[1]/4), ggml_get_op_params_f32(dst, 0), mmq, dst->ne[1], images);
}

void ggml_cuda_op_dsv4_hc_pre_mmq_multi(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_cuda_hc_pre_mmq_emit_data & mmq, const ggml_cuda_hc_pre_emit_data & images) {
    if (!mmq.images[0] && !mmq.images[1] && !mmq.images[2]) {
        ggml_cuda_op_dsv4_hc_pre_mmq(ctx, dst, mmq.data, mmq.padded, mmq.layout, images);
        return;
    }
    const ggml_cuda_hc_gate_f32 weights = {(const float *) dst->src[1]->data};
    if (ggml_get_op_params_i32(dst, 1)) { ggml_cuda_hc_pre_mmq_multi_launch<true>(ctx, dst, weights, mmq, images); }
    else { ggml_cuda_hc_pre_mmq_multi_launch<false>(ctx, dst, weights, mmq, images); }
}

template <mmq_q8_1_ds_layout Layout>
static void ggml_cuda_hc_pre_convert_mmq_launch(ggml_backend_cuda_context & ctx, ggml_type type, const void * weights,
        ggml_tensor * mm, ggml_tensor * dst, const ggml_cuda_hc_pre_emit_data * images, const ggml_cuda_hc_pre_mmq_emit_data & mmq) {
    const ggml_tensor * x = dst->src[0], * view = dst->src[1];
    const int block = CUDA_QUANTIZE_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params params(dim3((dst->ne[0] + block - 1)/block, dst->ne[1]), dim3(block), 0, ctx.stream());
    const auto launch = [&](auto tag) {
        using T = decltype(tag);
        ggml_cuda_kernel_launch(dsv4_hc_pre_convert_mmq<Layout, T>, params,
            (const float *) x->data, weights, (float *) mm->data, (float *) dst->data,
            dst->ne[0], x->ne[1], int64_t(x->nb[0]/4), int64_t(x->nb[1]/4), int64_t(x->nb[2]/4),
            int64_t(view->nb[0]/4), int64_t(view->nb[1]/4), int64_t(view->nb[2]/4),
            int64_t(dst->nb[0]/4), int64_t(dst->nb[1]/4), ggml_get_op_params_f32(dst, 0),
            (block_q8_1_mmq *) mmq.data, mmq.padded, dst->ne[1], images ? *images : ggml_cuda_hc_pre_emit_data{});
    };
    if (type == GGML_TYPE_F16) { launch(half{}); }
    else { GGML_ASSERT(type == GGML_TYPE_BF16); launch(nv_bfloat16{}); }
}

static void ggml_cuda_hc_pre_convert_mmq(ggml_backend_cuda_context & ctx, ggml_type type, const void * weights,
        ggml_tensor * mm, ggml_tensor * dst, const ggml_cuda_hc_pre_emit_data * images, const ggml_cuda_hc_pre_mmq_emit_data & mmq) {
    if (mmq.images[0] || mmq.images[1] || mmq.images[2]) {
        const auto launch = [&](auto tag) {
            using T = decltype(tag);
            ggml_cuda_hc_pre_mmq_multi_launch<true>(ctx, dst, ggml_cuda_hc_gate_convert<T>{(const T *) weights, (float *) mm->data}, mmq,
                images ? *images : ggml_cuda_hc_pre_emit_data{});
        };
        if (type == GGML_TYPE_F16) { launch(half{}); }
        else { GGML_ASSERT(type == GGML_TYPE_BF16); launch(nv_bfloat16{}); }
        return;
    }
    switch (mmq.layout) {
        case MMQ_Q8_1_DS_LAYOUT_D4: ggml_cuda_hc_pre_convert_mmq_launch<MMQ_Q8_1_DS_LAYOUT_D4>(ctx, type, weights, mm, dst, images, mmq); break;
        case MMQ_Q8_1_DS_LAYOUT_DS4: ggml_cuda_hc_pre_convert_mmq_launch<MMQ_Q8_1_DS_LAYOUT_DS4>(ctx, type, weights, mm, dst, images, mmq); break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6: ggml_cuda_hc_pre_convert_mmq_launch<MMQ_Q8_1_DS_LAYOUT_D2S6>(ctx, type, weights, mm, dst, images, mmq); break;
        default: GGML_ABORT("invalid HC MMQ layout");
    }
}
