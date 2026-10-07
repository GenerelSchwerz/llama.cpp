// Adapted from MIT-licensed expert primitives; see moe-reference-LICENSE.
// See ../ggml-cpu/moe-reference-LICENSE. Compile without fast math.
#include "moe-reference.cuh"
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "vecdotq.cuh"
#include <climits>
namespace {
template<int TY> struct Fmt;
template<> struct Fmt<16> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_xxs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<17> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_xs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<18> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq3_xxs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<20> { static constexpr int qk = 32, ipb = 2, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq4_nl_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<21> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq3_s_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<23> { static constexpr int qk = 256, ipb = 8, step = 4;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq4_xs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<22> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_s_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<29> { static constexpr int qk = 256, ipb = 8, step = 1;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq1_m_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<42> { static constexpr int qk = 64, ipb = 2, step = 1;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q2_0_q8_1(v, y, kbx, iqs); } };

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// One row against one q8_1 activation, the whole warp: call k = (block, part) is lane-strided.
template<int TY>
__device__ __forceinline__ float row_dot(const uint8_t* row, const block_q8_1* x, int nb, int lane) {
    using F = Fmt<TY>;
    float s = 0.0f;
    for (int k = lane; k < nb * F::ipb; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        s += F::dot(row, x + kbx * (F::qk / 32), kbx, iqs);
    }
    return warp_sum(s);
}

constexpr int GU_ROWS = 8;     // rows per block (one warp each)

template<int TG>
__global__ void __launch_bounds__(256) native_gu_kernel(const uint64_t* __restrict__ grp_gate,
                                                        const uint64_t* __restrict__ grp_up,
                                                        const int32_t* __restrict__ grp_start,
                                                        const int32_t* __restrict__ n_groups,
                                                        const int32_t* __restrict__ ent_tok,
                                                        const block_q8_1* __restrict__ xq, ggml_moe_reference_gpu_layout L,
                                                        float* __restrict__ gate, float* __restrict__ up) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * GU_ROWS + warp;             // 0 .. 2*n_ff
    if (row >= 2 * L.n_ff) return;
    const bool is_up = row >= L.n_ff;
    const int r = is_up ? row - (int) L.n_ff : row;
    const uint8_t* base = (const uint8_t*) (is_up ? grp_up[g] : grp_gate[g]);
    const uint8_t* wr = base + (size_t) r * (is_up ? L.up_row : L.gate_row);
    const int nb = (int) (L.n_embd / Fmt<TG>::qk), xb = (int) (L.n_embd / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    for (int e = e0; e < e1; ++e) {
        const float s = row_dot<TG>(wr, xq + (size_t) ent_tok[e] * xb, nb, lane);
        if (lane == 0) (is_up ? up : gate)[(size_t) e * L.n_ff + r] = s;
    }
}

__global__ void swiglu_entries_kernel(const float* __restrict__ gate, const float* __restrict__ up, float* __restrict__ h,
                                      long long n) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gate[i];
    h[i] = (g / (1.0f + __expf(-g))) * up[i];
}

template<int TD>
__global__ void __launch_bounds__(256) native_down_kernel(const uint64_t* __restrict__ grp_down,
                                                          const int32_t* __restrict__ grp_start,
                                                          const int32_t* __restrict__ n_groups,
                                                          const int32_t* __restrict__ ent_dst,
                                                          const block_q8_1* __restrict__ hq, ggml_moe_reference_gpu_layout L,
                                                          float* __restrict__ out) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = blockIdx.x * 8 + warp;
    if (r >= L.n_out) return;
    const uint8_t* base = (const uint8_t*) grp_down[g];
    const uint8_t* wr = base + (size_t) r * L.d_row;
    const int nb = (int) (L.n_ff / Fmt<TD>::qk), hb = (int) (L.n_ff / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    for (int e = e0; e < e1; ++e) {
        const float s = row_dot<TD>(wr, hq + (size_t) e * hb, nb, lane);
        if (lane == 0) out[(size_t) ent_dst[e] * L.n_out + r] = s;
    }
}

// ---------------------------------------------------------------- q8_1 (quantize.cu)
__global__ void quantize_q8_1_kernel(const float* __restrict__ x, block_q8_1* __restrict__ y, long long n) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float xi = x[i];
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    const float d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
    const long long ib = i / 32, iqs = i % 32;
    y[ib].qs[iqs] = q;
    if (iqs == 0) y[ib].ds = make_half2(d, sum);
}

}

bool ggml_moe_reference_gpu_supported(const ggml_moe_reference_gpu_layout & L) {
    const bool gu = L.gu_type == 16 || L.gu_type == 17 || L.gu_type == 18 || L.gu_type == 21 || L.gu_type == 22;
    return gu && (L.d_type == 20 || L.d_type == 42) && L.n_embd > 0 && L.n_embd % 256 == 0 &&
        L.n_ff > 0 && L.n_ff % 64 == 0 && L.n_out > 0 && L.n_ff <= INT_MAX / 2;
}
size_t ggml_moe_reference_gpu_scratch(size_t entries, size_t hidden) {
    if (!hidden || entries > SIZE_MAX / hidden || entries * hidden > (SIZE_MAX - 255) / sizeof(float)) { return SIZE_MAX; }
    const size_t f = (entries * hidden * sizeof(float) + 255) & ~size_t(255);
    if (entries * hidden / 32 > (SIZE_MAX - 255) / sizeof(block_q8_1)) { return SIZE_MAX; }
    const size_t q = (entries * hidden / 32 * sizeof(block_q8_1) + 255) & ~size_t(255);
    return f > (SIZE_MAX - q) / 3 ? SIZE_MAX : 3 * f + q;
}
bool ggml_moe_reference_gpu_quantize(const float * input, void * output, size_t values, cudaStream_t stream) {
    if (!values || values % 32 || values > size_t(LLONG_MAX) || values > SIZE_MAX - 255 || (values + 255) / 256 > INT_MAX) { return false; }
    quantize_q8_1_kernel<<<unsigned((values + 255) / 256), 256, 0, stream>>>(input, static_cast<block_q8_1 *>(output), values);
    return cudaGetLastError() == cudaSuccess;
}
bool ggml_moe_reference_gpu_execute(const ggml_moe_reference_gpu_layout & L,
        const ggml_moe_reference_gpu_group & group, unsigned cap_groups, unsigned cap_entries,
        const void * input, void * scratch, float * output, cudaStream_t stream) {
    if (!ggml_moe_reference_gpu_supported(L) || cap_groups == 0 || cap_groups > 65535 || cap_entries == 0) { return false; }
    const size_t required = ggml_moe_reference_gpu_scratch(cap_entries, L.n_ff);
    if (required == SIZE_MAX) { return false; }
    const size_t values = size_t(cap_entries) * size_t(L.n_ff);
    const size_t blocks = (values + 255) / 256;
    if (values > size_t(LLONG_MAX) || blocks > INT_MAX) { return false; }
    const size_t f = values * sizeof(float), fa = (f + 255) & ~size_t(255);
    auto * gate = static_cast<float *>(scratch);
    auto * up = reinterpret_cast<float *>(static_cast<uint8_t *>(scratch) + fa);
    auto * hidden = reinterpret_cast<float *>(static_cast<uint8_t *>(scratch) + 2 * fa);
    auto * hq = reinterpret_cast<block_q8_1 *>(static_cast<uint8_t *>(scratch) + 3 * fa);
    const auto * X = static_cast<const block_q8_1 *>(input);
    const dim3 gu(unsigned((size_t(L.n_ff) * 2 + 7) / 8), cap_groups);
#define GU(T) native_gu_kernel<T><<<gu, 256, 0, stream>>>(group.gate, group.up, group.starts, group.count, group.tokens, X, L, gate, up)
    switch (L.gu_type) {
        case 16: GU(16); break;
        case 17: GU(17); break;
        case 18: GU(18); break;
        case 21: GU(21); break;
        case 22: GU(22); break;
        default: return false;
    }
#undef GU
    swiglu_entries_kernel<<<unsigned(blocks), 256, 0, stream>>>(gate, up, hidden, values);
    if (!ggml_moe_reference_gpu_quantize(hidden, hq, values, stream)) { return false; }
    const dim3 down(unsigned((size_t(L.n_out) + 7) / 8), cap_groups);
#define DOWN(T) native_down_kernel<T><<<down, 256, 0, stream>>>(group.down, group.starts, group.count, group.destinations, hq, L, output)
    switch (L.d_type) {
        case 20: DOWN(20); break;
        case 42: DOWN(42); break;
        default: return false;
    }
#undef DOWN
    return cudaGetLastError() == cudaSuccess;
}

#else
bool ggml_moe_reference_gpu_supported(const ggml_moe_reference_gpu_layout &) { return false; }
size_t ggml_moe_reference_gpu_scratch(size_t, size_t) { return SIZE_MAX; }
bool ggml_moe_reference_gpu_quantize(const float *, void *, size_t, cudaStream_t) { return false; }
bool ggml_moe_reference_gpu_execute(const ggml_moe_reference_gpu_layout &, const ggml_moe_reference_gpu_group &,
        unsigned, unsigned, const void *, void *, float *, cudaStream_t) { return false; }
#endif
