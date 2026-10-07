// Weighted RMS reduction adapted from MIT-licensed reference kernels,
// themselves adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "moe-source-ordinary.cuh"
#include "common.cuh"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <climits>
#include <cmath>
#include <cstring>
#include <initializer_list>

namespace {

static bool same_tensor(const ggml_tensor * original, const ggml_tensor * clone) {
    return original && clone && original != clone && original->op == clone->op &&
        original->type == clone->type && original->flags == clone->flags &&
        original->view_offs == clone->view_offs &&
        !memcmp(original->ne, clone->ne, sizeof(original->ne)) &&
        !memcmp(original->nb, clone->nb, sizeof(original->nb)) &&
        !memcmp(original->op_params, clone->op_params, sizeof(original->op_params));
}

static bool provenance(const ggml_cuda_moe_source_norm & d) {
    if (!d.graph || d.graph->uid != d.graph_uid || !d.graph->nodes ||
            d.index + uint64_t(1) >= uint64_t(d.graph->n_nodes)) { return false; }
    const auto * norm = d.graph->nodes[d.index];
    const auto * output = d.graph->nodes[d.index + 1];
    if (!norm || !output || !norm->src[0] || !output->src[0] || !output->src[1]) { return false; }
    const bool first = output->src[0] == norm;
    return (first || output->src[1] == norm) && same_tensor(norm, d.norm) &&
        same_tensor(norm->src[0], d.input) && same_tensor(output->src[first ? 1 : 0], d.gamma) &&
        same_tensor(output, d.output) && d.norm->src[0] == d.input &&
        d.output->src[first ? 0 : 1] == d.norm && d.output->src[first ? 1 : 0] == d.gamma;
}

static bool span(const ggml_tensor * tensor, size_t & bytes) {
    bytes = sizeof(float);
    for (int i = 0; i < 4; ++i) {
        if (tensor->ne[i] <= 0 || tensor->nb[i] % sizeof(float)) { return false; }
        const auto n = uint64_t(tensor->ne[i] - 1);
        if (n && tensor->nb[i] > (SIZE_MAX - bytes) / n) { return false; }
        bytes += size_t(n) * tensor->nb[i];
    }
    return true;
}

static ggml_cuda_moe_source_tensor tensor_binding(const ggml_tensor * t, int device) {
    ggml_cuda_moe_source_tensor result;
    result.type = t->type;
    memcpy(result.ne, t->ne, sizeof(result.ne)); memcpy(result.nb, t->nb, sizeof(result.nb));
    result.data = t->data; result.device = device;
    if (t->buffer && t->buffer->buft == ggml_backend_cuda_buffer_type(device)) {
        result.buffer_base = ggml_backend_buffer_get_base(t->buffer);
        result.buffer_bytes = ggml_backend_buffer_get_size(t->buffer);
        result.buffer_identity = reinterpret_cast<uintptr_t>(t->buffer);
    }
    return result;
}

static void norm_bindings(const ggml_cuda_moe_source_norm & d, int device, ggml_cuda_moe_source_tensor * tensors) {
    tensors[0] = tensor_binding(d.input, device); tensors[1] = tensor_binding(d.gamma, device);
    tensors[2] = tensor_binding(d.output, device);
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
__device__ __forceinline__ float norm_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset, 32);
    }
    return value;
}

struct norm_launch {
    const float * input;
    const float * gamma;
    float * output;
    int64_t ne[4], gamma_ne[4];
    size_t input_nb[4], gamma_nb[4], output_nb[4];
    float epsilon;
};

template<int BlockSize>
__global__ void weighted_rms_norm(norm_launch d) {
    uint64_t rest = blockIdx.x;
    size_t input_offset = 0, gamma_offset = 0, output_offset = 0;
    for (int dim = 1; dim < 4; ++dim) {
        const auto row = rest % uint64_t(d.ne[dim]);
        rest /= uint64_t(d.ne[dim]);
        input_offset += row * d.input_nb[dim];
        gamma_offset += (row % uint64_t(d.gamma_ne[dim])) * d.gamma_nb[dim];
        output_offset += row * d.output_nb[dim];
    }
    const auto * input = reinterpret_cast<const float *>(reinterpret_cast<const char *>(d.input) + input_offset);
    const auto * gamma = reinterpret_cast<const float *>(reinterpret_cast<const char *>(d.gamma) + gamma_offset);
    auto * output = reinterpret_cast<float *>(reinterpret_cast<char *>(d.output) + output_offset);
    const int tid = threadIdx.x;
    float partial = 0.0f;
    for (int64_t col = tid; col < d.ne[0]; col += BlockSize) {
        const float value = input[col];
        partial += value * value;
    }
    __shared__ float sums[32];
    partial = norm_warp_sum(partial);
    const int lane = tid % 32;
    if (lane == 0) { sums[tid / 32] = partial; }
    __syncthreads();
    partial = lane < BlockSize / 32 ? sums[lane] : 0.0f;
    partial = norm_warp_sum(partial);
    const float mean = partial / d.ne[0];
    const float scale = rsqrtf(mean + d.epsilon);
    for (int64_t col = tid; col < d.ne[0]; col += BlockSize) {
        output[col] = scale * input[col] * gamma[col % d.gamma_ne[0]];
    }
}
#endif

} // namespace

ggml_cuda_moe_source_norm_status ggml_cuda_moe_source_norm_prepare(
        const ggml_cgraph * graph, uint32_t index, ggml_tensor * norm, ggml_tensor * input,
        ggml_tensor * gamma, ggml_tensor * output, ggml_cuda_moe_source_norm & d) {
    d = {};
    d.graph = graph; d.graph_uid = graph ? graph->uid : 0; d.index = index;
    d.norm = norm; d.input = input; d.gamma = gamma; d.output = output;
    if (!provenance(d)) { return GGML_CUDA_MOE_SOURCE_NORM_INVALID; }
    const auto * original = graph->nodes[index];
    if (norm->op != GGML_OP_RMS_NORM || output->op != GGML_OP_MUL ||
            norm->view_src || output->view_src || (norm->flags & GGML_TENSOR_FLAG_OUTPUT) ||
            !(norm->flags & GGML_TENSOR_FLAG_COMPUTE) || !(output->flags & GGML_TENSOR_FLAG_COMPUTE)) {
        return GGML_CUDA_MOE_SOURCE_NORM_UNSUPPORTED;
    }
    size_t uses = 0;
    for (int i = 0; i < graph->n_nodes; ++i) {
        const auto * node = graph->nodes[i];
        if (!node) { return GGML_CUDA_MOE_SOURCE_NORM_INVALID; }
        if (node->view_src == original) { return GGML_CUDA_MOE_SOURCE_NORM_UNSUPPORTED; }
        for (const auto * source : node->src) { uses += source == original; }
    }
    for (int i = 0; i < graph->n_leafs; ++i) {
        if (!graph->leafs || !graph->leafs[i]) { return GGML_CUDA_MOE_SOURCE_NORM_INVALID; }
        if (graph->leafs[i]->view_src == original) { return GGML_CUDA_MOE_SOURCE_NORM_UNSUPPORTED; }
    }
    if (uses != 1) { return GGML_CUDA_MOE_SOURCE_NORM_UNSUPPORTED; }
    // Owned graph copies omit the caller's tensor-identity hash table.
    if (graph->use_counts) {
        if (!graph->visited_hash_set.size || !graph->visited_hash_set.used || !graph->visited_hash_set.keys) {
            return GGML_CUDA_MOE_SOURCE_NORM_INVALID;
        }
        const auto hash = ggml_hash_find(&graph->visited_hash_set, original);
        if (hash == GGML_HASHSET_FULL || !ggml_bitset_get(graph->visited_hash_set.used, hash)) {
            return GGML_CUDA_MOE_SOURCE_NORM_INVALID;
        }
        if (graph->use_counts[hash] != 1) { return GGML_CUDA_MOE_SOURCE_NORM_UNSUPPORTED; }
    }
    size_t rows = 1;
    for (const auto * tensor : {input, gamma, norm, output}) {
        size_t bytes;
        if (tensor->type != GGML_TYPE_F32 || tensor->nb[0] != sizeof(float) || !span(tensor, bytes)) {
            return GGML_CUDA_MOE_SOURCE_NORM_UNSUPPORTED;
        }
    }
    if (!ggml_are_same_shape(input, norm) || !ggml_are_same_shape(norm, output) ||
            !ggml_is_contiguous(output) || input->ne[0] > INT_MAX) { return GGML_CUDA_MOE_SOURCE_NORM_UNSUPPORTED; }
    for (int i = 0; i < 4; ++i) {
        if (input->ne[i] % gamma->ne[i] || (i && uint64_t(input->ne[i]) > INT_MAX / rows)) {
            return GGML_CUDA_MOE_SOURCE_NORM_UNSUPPORTED;
        }
        if (i) { rows *= size_t(input->ne[i]); }
    }
    memcpy(&d.epsilon, norm->op_params, sizeof(float));
    if (!std::isfinite(d.epsilon) || d.epsilon < 0.0f) { return GGML_CUDA_MOE_SOURCE_NORM_UNSUPPORTED; }
    memcpy(d.ne, input->ne, sizeof(d.ne)); memcpy(d.gamma_ne, gamma->ne, sizeof(d.gamma_ne));
    memcpy(d.input_nb, input->nb, sizeof(d.input_nb)); memcpy(d.gamma_nb, gamma->nb, sizeof(d.gamma_nb));
    memcpy(d.output_nb, output->nb, sizeof(d.output_nb)); d.rows = uint32_t(rows);
    return GGML_CUDA_MOE_SOURCE_NORM_READY;
}

ggml_cuda_moe_source_norm_status ggml_cuda_moe_source_norm_bind(ggml_cuda_moe_source_norm & d, int device) {
    d.bound = false;
    if (!provenance(d) || memcmp(d.ne, d.input->ne, sizeof(d.ne)) ||
            memcmp(d.gamma_ne, d.gamma->ne, sizeof(d.gamma_ne)) ||
            memcmp(d.input_nb, d.input->nb, sizeof(d.input_nb)) ||
            memcmp(d.gamma_nb, d.gamma->nb, sizeof(d.gamma_nb)) ||
            memcmp(d.output_nb, d.output->nb, sizeof(d.output_nb)) ||
            memcmp(&d.epsilon, d.norm->op_params, sizeof(d.epsilon))) { return GGML_CUDA_MOE_SOURCE_NORM_INVALID; }
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(device);
    return GGML_CUDA_MOE_SOURCE_NORM_UNSUPPORTED;
#else
    if (d.output->data == d.graph->nodes[d.index + 1]->data) { return GGML_CUDA_MOE_SOURCE_NORM_INVALID; }
    ggml_cuda_moe_source_tensor tensors[3];
    norm_bindings(d, device, tensors);
    const auto status = ggml_cuda_moe_source_rms_binding_prepare(tensors, d.epsilon, d.operation);
    d.bound = status == GGML_CUDA_MOE_SOURCE_NORM_READY;
    return status;
#endif
}

bool ggml_cuda_moe_source_norm_emit(ggml_backend_cuda_context & context, const ggml_cuda_moe_source_norm & d) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(context); GGML_UNUSED(d);
    return false;
#else
    if (!d.bound || !provenance(d) || memcmp(d.ne, d.input->ne, sizeof(d.ne)) ||
            memcmp(d.gamma_ne, d.gamma->ne, sizeof(d.gamma_ne)) ||
            memcmp(d.input_nb, d.input->nb, sizeof(d.input_nb)) ||
            memcmp(d.gamma_nb, d.gamma->nb, sizeof(d.gamma_nb)) ||
            memcmp(d.output_nb, d.output->nb, sizeof(d.output_nb)) ||
            memcmp(&d.epsilon, d.norm->op_params, sizeof(d.epsilon))) { return false; }
    ggml_cuda_moe_source_tensor tensors[3];
    norm_bindings(d, context.device, tensors);
    return ggml_cuda_moe_source_binding_matches(d.operation, tensors, 3) &&
        ggml_cuda_moe_source_binding_emit(d.operation, context.device, context.stream());
#endif
}

namespace {

static bool gdn_ab_provenance(const ggml_cuda_moe_source_gdn_ab & d) {
    if (!d.graph || d.graph->n_nodes < 0 || d.graph->uid != d.graph_uid || !d.graph->nodes ||
            uint64_t(d.index) + 9 > uint64_t(d.graph->n_nodes)) { return false; }
    for (int i = 0; i < 14; ++i) {
        const auto * clone = i < 9 ? d.nodes[i] : d.reads[i - 9];
        if ((i < 9 && d.graph->nodes[d.index + i] != d.originals[i]) || !same_tensor(d.originals[i], clone)) { return false; }
        if (i >= 9) { continue; }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const auto * original = d.originals[i]->src[s];
            int j = 0;
            while (j < 14 && d.originals[j] != original) { ++j; }
            const auto * expected = original && j < 14 ? (j < 9 ? d.nodes[j] : d.reads[j - 9]) : nullptr;
            if ((original && j == 14) || clone->src[s] != expected) { return false; }
        }
        if (d.originals[i]->view_src) {
            int j = 0;
            while (j < 9 && d.originals[j] != d.originals[i]->view_src) { ++j; }
            if (j == 9 || clone->view_src != d.nodes[j]) { return false; }
        } else if (clone->view_src) { return false; }
    }
    return true;
}

static bool gdn_ab_metadata(const ggml_cuda_moe_source_gdn_ab & d) {
    if (!gdn_ab_provenance(d)) { return false; }
    for (int i = 0; i < 14; ++i) {
        const auto * clone = i < 9 ? d.nodes[i] : d.reads[i - 9];
        if (!same_tensor(&d.metadata[i], clone) || d.metadata[i].view_src != clone->view_src ||
                memcmp(d.metadata[i].src, clone->src, sizeof(clone->src))) { return false; }
    }
    return true;
}

static bool gdn_ab_span(const ggml_tensor * tensor, size_t & bytes) {
    const size_t element = tensor->type == GGML_TYPE_BF16 ? sizeof(uint16_t) : sizeof(float);
    bytes = element;
    for (int i = 0; i < 4; ++i) {
        if (tensor->ne[i] <= 0 || tensor->nb[i] % element) { return false; }
        const auto n = uint64_t(tensor->ne[i] - 1);
        if (n && tensor->nb[i] > (SIZE_MAX - bytes) / n) { return false; }
        bytes += size_t(n) * tensor->nb[i];
    }
    return true;
}

static bool gdn_ab_retained(int i) { return i == 4 || i == 5 || i == 8; }

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
struct gdn_ab_launch {
    const float * input;
    const uint16_t * wa;
    const uint16_t * wb;
    const float * dt;
    const float * a;
    float * gate;
    float * beta;
    int64_t input_ne[4], output_ne[4];
    size_t input_nb[4], gate_nb[4], beta_nb[4];
    size_t wa_row, wb_row;
    uint64_t rows, heads, head_blocks;
};

// Reference dot/FMA/XOR arithmetic and projection epilogues.
__global__ void __launch_bounds__(256) gdn_ab_kernel(gdn_ab_launch d) {
    const uint64_t row = (uint64_t(blockIdx.x) % d.head_blocks) * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= 2 * d.heads) { return; }
    const bool is_beta = row >= d.heads;
    const uint64_t head = is_beta ? row - d.heads : row;
    const uint64_t begin = (uint64_t(blockIdx.x) / d.head_blocks) * 8;
    const int count = int(d.rows - begin < 8 ? d.rows - begin : 8);
    const auto * weights = reinterpret_cast<const uint4 *>(reinterpret_cast<const char *>(is_beta ? d.wb : d.wa) + head * (is_beta ? d.wb_row : d.wa_row));
    float acc[8] = {};
    for (int64_t j = lane; j < d.input_ne[0] / 8; j += 32) {
        const uint4 wv = __ldg(weights + j);
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t >= count) { break; }
            uint64_t r = begin + t;
            const uint64_t i1 = r % d.input_ne[1]; r /= d.input_ne[1];
            const uint64_t i2 = r % d.input_ne[2]; r /= d.input_ne[2];
            const auto * x = reinterpret_cast<const float *>(reinterpret_cast<const char *>(d.input) + i1 * d.input_nb[1] + i2 * d.input_nb[2] + r * d.input_nb[3]) + j * 8;
            const float4 xa = *reinterpret_cast<const float4 *>(x);
            const float4 xb = *reinterpret_cast<const float4 *>(x + 4);
            float a = acc[t];
            a = fmaf(__uint_as_float(wv.x << 16), xa.x, a); a = fmaf(__uint_as_float(wv.x & 0xffff0000u), xa.y, a);
            a = fmaf(__uint_as_float(wv.y << 16), xa.z, a); a = fmaf(__uint_as_float(wv.y & 0xffff0000u), xa.w, a);
            a = fmaf(__uint_as_float(wv.z << 16), xb.x, a); a = fmaf(__uint_as_float(wv.z & 0xffff0000u), xb.y, a);
            a = fmaf(__uint_as_float(wv.w << 16), xb.z, a); a = fmaf(__uint_as_float(wv.w & 0xffff0000u), xb.w, a);
            acc[t] = a;
        }
    }
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        if (t >= count) { break; }
        float a = norm_warp_sum(acc[t]);
        if (lane != 0) { continue; }
        const uint64_t r = begin + t;
        const auto * nb = is_beta ? d.beta_nb : d.gate_nb;
        const size_t offset = head * nb[1] + (r % d.output_ne[2]) * nb[2] + (r / d.output_ne[2]) * nb[3];
        auto * output = reinterpret_cast<float *>(reinterpret_cast<char *>(is_beta ? d.beta : d.gate) + offset);
        if (is_beta) {
            *output = 1.0f / (1.0f + __expf(-a));
        } else {
            const float v = a + d.dt[head];
            const float sp = v > 20.0f ? v : log1pf(__expf(v));
            *output = sp * d.a[head];
        }
    }
}
#endif

} // namespace

ggml_cuda_moe_source_norm_status ggml_cuda_moe_source_gdn_ab_prepare(
        const ggml_cgraph * graph, uint32_t index, ggml_tensor * const * nodes, ggml_cuda_moe_source_gdn_ab & d) {
    d = {}; d.graph = graph; d.graph_uid = graph ? graph->uid : 0; d.index = index;
    if (!graph || graph->n_nodes < 0 || graph->n_leafs < 0 || !graph->nodes || !nodes || uint64_t(index) + 9 > uint64_t(graph->n_nodes)) { return GGML_CUDA_MOE_SOURCE_NORM_INVALID; }
    const ggml_op ops[] = {GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_MUL,
        GGML_OP_RESHAPE, GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_UNARY};
    for (int i = 0; i < 9; ++i) {
        d.nodes[i] = nodes[i]; d.originals[i] = graph->nodes[index + i];
        if (!same_tensor(d.originals[i], d.nodes[i])) { return GGML_CUDA_MOE_SOURCE_NORM_INVALID; }
        if (d.nodes[i]->op != ops[i]) { return GGML_CUDA_MOE_SOURCE_NORM_UNSUPPORTED; }
    }
    auto ** n = d.nodes;
    if (n[1]->src[0] != n[0] || n[3]->src[0] != n[2] || n[5]->src[0] != n[4] || n[7]->src[0] != n[6] || n[8]->src[0] != n[7] ||
            ggml_get_unary_op(n[3]) != GGML_UNARY_OP_SOFTPLUS || ggml_get_unary_op(n[8]) != GGML_UNARY_OP_SIGMOID ||
            (n[2]->src[0] != n[1] && n[2]->src[1] != n[1]) || (n[4]->src[0] != n[3] && n[4]->src[1] != n[3]) ||
            !n[0]->src[0] || !n[0]->src[1] || !n[6]->src[0] || n[6]->src[1] != n[0]->src[1]) { return GGML_CUDA_MOE_SOURCE_NORM_UNSUPPORTED; }
    const int dt_slot = n[2]->src[0] == n[1] ? 1 : 0, a_slot = n[4]->src[0] == n[3] ? 1 : 0;
    d.reads[0] = n[0]->src[1]; d.reads[1] = n[0]->src[0]; d.reads[2] = n[6]->src[0];
    d.reads[3] = n[2]->src[dt_slot]; d.reads[4] = n[4]->src[a_slot];
    d.originals[9] = d.originals[0]->src[1]; d.originals[10] = d.originals[0]->src[0]; d.originals[11] = d.originals[6]->src[0];
    d.originals[12] = d.originals[2]->src[dt_slot]; d.originals[13] = d.originals[4]->src[a_slot];
    if (!gdn_ab_provenance(d)) { return GGML_CUDA_MOE_SOURCE_NORM_INVALID; }
    const auto * input = d.reads[0];
    const int64_t f = input->ne[0], h = n[5]->ne[1];
    uint64_t elements[14];
    for (int i = 0; i < 14; ++i) {
        const auto * t = i < 9 ? n[i] : d.reads[i - 9];
        elements[i] = 1;
        for (const int64_t dim : t->ne) {
            if (dim <= 0 || uint64_t(dim) > uint64_t(INT64_MAX) / elements[i]) { return GGML_CUDA_MOE_SOURCE_NORM_UNSUPPORTED; }
            elements[i] *= uint64_t(dim);
        }
    }
    uint64_t rows = 1;
    for (int i = 1; i < 4; ++i) {
        if (input->ne[i] <= 0 || uint64_t(input->ne[i]) > UINT64_MAX / rows) { return GGML_CUDA_MOE_SOURCE_NORM_UNSUPPORTED; }
        rows *= uint64_t(input->ne[i]);
    }
    if (f <= 0 || f % 8 || h <= 0 || n[5]->ne[0] != 1 || !ggml_are_same_shape(n[5], n[8]) ||
            n[5]->ne[2] <= 0 || n[5]->ne[3] <= 0 || uint64_t(n[5]->ne[2]) > UINT64_MAX / uint64_t(n[5]->ne[3]) ||
            uint64_t(n[5]->ne[2]) * uint64_t(n[5]->ne[3]) != rows || !ggml_is_contiguous(n[0]) || !ggml_is_contiguous(n[6]) ||
            !ggml_is_contiguous(n[4]) || !ggml_is_contiguous(n[5]) || !ggml_are_same_shape(n[0], n[6]) ||
            n[0]->ne[0] != h || rows > uint64_t(INT64_MAX) / uint64_t(h) || elements[0] != uint64_t(h) * rows ||
            !ggml_are_same_shape(n[1], n[4]) || !ggml_are_same_shape(n[1], n[2]) ||
            !ggml_are_same_shape(n[2], n[3]) || !ggml_are_same_shape(n[7], n[8]) || elements[0] != elements[1] ||
            elements[6] != elements[7] || elements[4] != elements[5]) { return GGML_CUDA_MOE_SOURCE_NORM_UNSUPPORTED; }
    for (int i = 0; i < 14; ++i) {
        const auto * t = i < 9 ? n[i] : d.reads[i - 9];
        size_t bytes;
        if (t->type != (i == 10 || i == 11 ? GGML_TYPE_BF16 : GGML_TYPE_F32) || !gdn_ab_span(t, bytes)) { return GGML_CUDA_MOE_SOURCE_NORM_UNSUPPORTED; }
        if (i < 9 && !gdn_ab_retained(i) && (t->flags & GGML_TENSOR_FLAG_OUTPUT)) { return GGML_CUDA_MOE_SOURCE_NORM_UNSUPPORTED; }
        if (i < 9 && t->op != GGML_OP_RESHAPE && !(t->flags & GGML_TENSOR_FLAG_COMPUTE)) { return GGML_CUDA_MOE_SOURCE_NORM_UNSUPPORTED; }
        d.metadata[i] = *t;
    }
    for (int i = 1; i < 3; ++i) {
        const auto * w = d.reads[i];
        if (w->ne[0] != f || w->ne[1] != h || w->ne[2] != 1 || w->ne[3] != 1 || w->nb[0] != sizeof(uint16_t) || w->nb[1] % 16) { return GGML_CUDA_MOE_SOURCE_NORM_UNSUPPORTED; }
    }
    for (int i = 3; i < 5; ++i) {
        const auto * c = d.reads[i];
        if (c->ne[0] != h || c->ne[1] != 1 || c->ne[2] != 1 || c->ne[3] != 1 || c->nb[0] != sizeof(float)) { return GGML_CUDA_MOE_SOURCE_NORM_UNSUPPORTED; }
    }
    for (const auto * out : {n[5], n[8]}) {
        if (out->nb[1] < sizeof(float) || uint64_t(h) > out->nb[2] / out->nb[1] ||
                uint64_t(out->ne[2]) > out->nb[3] / out->nb[2]) { return GGML_CUDA_MOE_SOURCE_NORM_UNSUPPORTED; }
    }
    if (input->nb[0] != sizeof(float)) { return GGML_CUDA_MOE_SOURCE_NORM_UNSUPPORTED; }
    for (int i = 1; i < 4; ++i) {
        if (input->nb[i] % 16) { return GGML_CUDA_MOE_SOURCE_NORM_UNSUPPORTED; }
    }
    for (uint64_t i = 0; i < uint64_t(graph->n_nodes) + uint64_t(graph->n_leafs); ++i) {
        const bool leaf = i >= uint64_t(graph->n_nodes);
        const auto * consumer = leaf ? (graph->leafs ? graph->leafs[i - graph->n_nodes] : nullptr) : graph->nodes[i];
        if (!consumer) { return GGML_CUDA_MOE_SOURCE_NORM_INVALID; }
        if (!leaf && i >= index && i < uint64_t(index) + 9) { continue; }
        for (int k = 0; k < 9; ++k) {
            if (gdn_ab_retained(k)) { continue; }
            if (consumer->view_src == d.originals[k]) { return GGML_CUDA_MOE_SOURCE_NORM_UNSUPPORTED; }
            for (const auto * source : consumer->src) {
                if (source == d.originals[k]) { return GGML_CUDA_MOE_SOURCE_NORM_UNSUPPORTED; }
            }
        }
    }
    d.rows = rows;
    return GGML_CUDA_MOE_SOURCE_NORM_READY;
}

ggml_cuda_moe_source_norm_status ggml_cuda_moe_source_gdn_ab_bind(ggml_cuda_moe_source_gdn_ab & d, int device) {
    d.bound = false;
    if (!gdn_ab_metadata(d)) { return GGML_CUDA_MOE_SOURCE_NORM_INVALID; }
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(device);
    return GGML_CUDA_MOE_SOURCE_NORM_UNSUPPORTED;
#else
    if (d.nodes[5]->data != d.nodes[4]->data || d.nodes[5]->data == d.originals[5]->data ||
            d.nodes[8]->data == d.originals[8]->data) { return GGML_CUDA_MOE_SOURCE_NORM_INVALID; }
    ggml_tensor * tensors[7] = {d.reads[0], d.reads[1], d.reads[2], d.reads[3], d.reads[4], d.nodes[5], d.nodes[8]};
    ggml_cuda_moe_source_tensor bindings[7];
    for (int i = 0; i < 7; ++i) { bindings[i] = tensor_binding(tensors[i], device); }
    const auto status = ggml_cuda_moe_source_gdn_binding_prepare(bindings, d.operation);
    if (status != GGML_CUDA_MOE_SOURCE_NORM_READY) { return status; }
    for (int i = 0; i < 7; ++i) { d.bound_data[i] = tensors[i]->data; d.bound_buffers[i] = tensors[i]->buffer; }
    d.device = device; d.bound = true;
    return status;
#endif
}

bool ggml_cuda_moe_source_gdn_ab_emit(ggml_backend_cuda_context & context, const ggml_cuda_moe_source_gdn_ab & d) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(context); GGML_UNUSED(d);
    return false;
#else
    if (!d.bound || context.device != d.device || !gdn_ab_metadata(d) || d.nodes[5]->data != d.nodes[4]->data) { return false; }
    ggml_tensor * tensors[7] = {d.reads[0], d.reads[1], d.reads[2], d.reads[3], d.reads[4], d.nodes[5], d.nodes[8]};
    for (int i = 0; i < 7; ++i) {
        if (tensors[i]->data != d.bound_data[i] || tensors[i]->buffer != d.bound_buffers[i]) { return false; }
    }
    ggml_cuda_moe_source_tensor bindings[7];
    for (int i = 0; i < 7; ++i) { bindings[i] = tensor_binding(tensors[i], context.device); }
    return ggml_cuda_moe_source_binding_matches(d.operation, bindings, 7) &&
        ggml_cuda_moe_source_binding_emit(d.operation, context.device, context.stream());
#endif
}

namespace {

static bool binding_span(const ggml_cuda_moe_source_tensor & t, size_t & bytes) {
    const size_t element = t.type == GGML_TYPE_BF16 ? sizeof(uint16_t) : sizeof(float);
    bytes = element;
    for (int i = 0; i < 4; ++i) {
        if (t.ne[i] <= 0 || t.nb[i] % element) { return false; }
        const auto n = uint64_t(t.ne[i] - 1);
        if (n && t.nb[i] > (SIZE_MAX - bytes) / n) { return false; }
        bytes += size_t(n) * t.nb[i];
    }
    return true;
}

static bool binding_overlap(const ggml_cuda_moe_source_tensor & a, size_t an,
        const ggml_cuda_moe_source_tensor & b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a.data), bp = reinterpret_cast<uintptr_t>(b.data);
    return ap < bp + bn && bp < ap + an;
}

static bool binding_same(const ggml_cuda_moe_source_tensor & a, const ggml_cuda_moe_source_tensor & b) {
    return a.type == b.type && a.data == b.data && a.buffer_base == b.buffer_base &&
        a.buffer_bytes == b.buffer_bytes && a.buffer_identity == b.buffer_identity && a.device == b.device &&
        !memcmp(a.ne, b.ne, sizeof(a.ne)) && !memcmp(a.nb, b.nb, sizeof(a.nb));
}

static ggml_cuda_moe_source_norm_status binding_validate(const ggml_cuda_moe_source_tensor * t, int count,
        size_t * bytes, uint64_t & rows) {
    if (!t || t[0].device < 0) { return GGML_CUDA_MOE_SOURCE_NORM_INVALID; }
    rows = 1;
    for (int i = 0; i < count; ++i) {
        if (t[i].type != GGML_TYPE_F32 && t[i].type != GGML_TYPE_BF16) { return GGML_CUDA_MOE_SOURCE_NORM_UNSUPPORTED; }
        if (t[i].device != t[0].device || !binding_span(t[i], bytes[i])) { return GGML_CUDA_MOE_SOURCE_NORM_INVALID; }
        const auto base = reinterpret_cast<uintptr_t>(t[i].buffer_base), p = reinterpret_cast<uintptr_t>(t[i].data);
        if (!p || p % alignof(float) || bytes[i] > UINTPTR_MAX - p) { return GGML_CUDA_MOE_SOURCE_NORM_INVALID; }
        if (!base || !t[i].buffer_identity) { return GGML_CUDA_MOE_SOURCE_NORM_UNSUPPORTED; }
        if (t[i].buffer_bytes > UINTPTR_MAX - base || p < base || p - base > t[i].buffer_bytes ||
                bytes[i] > t[i].buffer_bytes - (p - base)) { return GGML_CUDA_MOE_SOURCE_NORM_INVALID; }
    }
    for (int i = 1; i < 4; ++i) {
        if (uint64_t(t[0].ne[i]) > UINT64_MAX / rows) { return GGML_CUDA_MOE_SOURCE_NORM_INVALID; }
        rows *= uint64_t(t[0].ne[i]);
    }
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    return GGML_CUDA_MOE_SOURCE_NORM_UNSUPPORTED;
#else
    for (int i = 0; i < count; ++i) {
        cudaPointerAttributes attr;
        if (cudaPointerGetAttributes(&attr, t[i].data) != cudaSuccess) { return GGML_CUDA_MOE_SOURCE_NORM_INVALID; }
        if (attr.type != cudaMemoryTypeDevice || attr.device != t[0].device) { return GGML_CUDA_MOE_SOURCE_NORM_INVALID; }
    }
    return GGML_CUDA_MOE_SOURCE_NORM_READY;
#endif
}

} // namespace

ggml_cuda_moe_source_norm_status ggml_cuda_moe_source_rms_binding_prepare(
        const ggml_cuda_moe_source_tensor * t, float epsilon, ggml_cuda_moe_source_binding & d) {
    d = {};
    size_t bytes[3]; uint64_t rows;
    const auto status = binding_validate(t, 3, bytes, rows);
    if (status != GGML_CUDA_MOE_SOURCE_NORM_READY) { return status; }
    if (!std::isfinite(epsilon) || epsilon < 0 || t[0].ne[0] > INT_MAX || rows > INT_MAX) { return GGML_CUDA_MOE_SOURCE_NORM_UNSUPPORTED; }
    size_t contiguous = sizeof(float);
    for (int i = 0; i < 4; ++i) {
        if (t[0].type != GGML_TYPE_F32 || t[1].type != GGML_TYPE_F32 || t[2].type != GGML_TYPE_F32 ||
                t[0].nb[0] != sizeof(float) || t[1].nb[0] != sizeof(float) ||
                t[0].ne[i] != t[2].ne[i] || t[0].ne[i] % t[1].ne[i] || t[2].nb[i] != contiguous) { return GGML_CUDA_MOE_SOURCE_NORM_UNSUPPORTED; }
        if (uint64_t(t[2].ne[i]) > SIZE_MAX / contiguous) { return GGML_CUDA_MOE_SOURCE_NORM_INVALID; }
        contiguous *= size_t(t[2].ne[i]);
    }
    if (binding_overlap(t[2], bytes[2], t[1], bytes[1]) ||
            (binding_overlap(t[2], bytes[2], t[0], bytes[0]) &&
             (t[2].data != t[0].data || memcmp(t[2].nb, t[0].nb, sizeof(t[0].nb))))) { return GGML_CUDA_MOE_SOURCE_NORM_UNSUPPORTED; }
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, t[0].device) != cudaSuccess) { return GGML_CUDA_MOE_SOURCE_NORM_INVALID; }
    const int threads = t[0].ne[0] < 1024 ? 256 : 1024;
    if (prop.warpSize != 32 || prop.maxThreadsPerBlock < threads || rows > uint64_t(prop.maxGridSize[0])) { return GGML_CUDA_MOE_SOURCE_NORM_UNSUPPORTED; }
#endif
    for (int i = 0; i < 3; ++i) { d.tensors[i] = t[i]; }
    d.rows = rows; d.epsilon = epsilon; d.device = t[0].device; d.count = 3;
    return GGML_CUDA_MOE_SOURCE_NORM_READY;
}

ggml_cuda_moe_source_norm_status ggml_cuda_moe_source_gdn_binding_prepare(
        const ggml_cuda_moe_source_tensor * t, ggml_cuda_moe_source_binding & d) {
    d = {};
    size_t bytes[7]; uint64_t rows;
    const auto status = binding_validate(t, 7, bytes, rows);
    if (status != GGML_CUDA_MOE_SOURCE_NORM_READY) { return status; }
    const int64_t f = t[0].ne[0], h = t[5].ne[1];
    if (f % 8 || t[0].type != GGML_TYPE_F32 || t[0].nb[0] != sizeof(float) || t[5].ne[0] != 1 ||
            memcmp(t[5].ne, t[6].ne, sizeof(t[5].ne)) ||
            uint64_t(t[5].ne[2]) > UINT64_MAX / uint64_t(t[5].ne[3]) ||
            uint64_t(t[5].ne[2]) * uint64_t(t[5].ne[3]) != rows) { return GGML_CUDA_MOE_SOURCE_NORM_UNSUPPORTED; }
    for (int i = 1; i < 3; ++i) {
        if (t[i].type != GGML_TYPE_BF16 || t[i].ne[0] != f || t[i].ne[1] != h ||
                t[i].ne[2] != 1 || t[i].ne[3] != 1 || t[i].nb[0] != sizeof(uint16_t) || t[i].nb[1] % 16) { return GGML_CUDA_MOE_SOURCE_NORM_UNSUPPORTED; }
    }
    for (int i = 3; i < 5; ++i) {
        if (t[i].type != GGML_TYPE_F32 || t[i].ne[0] != h || t[i].ne[1] != 1 || t[i].ne[2] != 1 ||
                t[i].ne[3] != 1 || t[i].nb[0] != sizeof(float)) { return GGML_CUDA_MOE_SOURCE_NORM_UNSUPPORTED; }
    }
    for (int i = 1; i < 4; ++i) {
        if (t[0].nb[i] % 16) { return GGML_CUDA_MOE_SOURCE_NORM_UNSUPPORTED; }
    }
    for (int i = 0; i < 3; ++i) {
        if (reinterpret_cast<uintptr_t>(t[i].data) % 16) { return GGML_CUDA_MOE_SOURCE_NORM_UNSUPPORTED; }
    }
    for (int i = 5; i < 7; ++i) {
        if (t[i].type != GGML_TYPE_F32 || t[i].nb[1] < sizeof(float) || !t[i].nb[2] ||
                uint64_t(h) > t[i].nb[2] / t[i].nb[1] ||
                uint64_t(t[i].ne[2]) > t[i].nb[3] / t[i].nb[2]) { return GGML_CUDA_MOE_SOURCE_NORM_UNSUPPORTED; }
        for (int j = 0; j < i; ++j) {
            if (binding_overlap(t[i], bytes[i], t[j], bytes[j])) { return GGML_CUDA_MOE_SOURCE_NORM_UNSUPPORTED; }
        }
    }
    if (uint64_t(h) > (UINT64_MAX - 7) / 2 || rows > UINT64_MAX - 7) { return GGML_CUDA_MOE_SOURCE_NORM_UNSUPPORTED; }
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    const uint64_t blocks = (2 * uint64_t(h) + 7) / 8, tiles = (rows + 7) / 8;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, t[0].device) != cudaSuccess) { return GGML_CUDA_MOE_SOURCE_NORM_INVALID; }
    if (prop.warpSize != 32 || prop.maxThreadsPerBlock < 256 ||
            tiles > uint64_t(prop.maxGridSize[0]) / blocks) { return GGML_CUDA_MOE_SOURCE_NORM_UNSUPPORTED; }
#endif
    for (int i = 0; i < 7; ++i) { d.tensors[i] = t[i]; }
    d.rows = rows; d.device = t[0].device; d.count = 7;
    return GGML_CUDA_MOE_SOURCE_NORM_READY;
}

bool ggml_cuda_moe_source_binding_matches(const ggml_cuda_moe_source_binding & d,
        const ggml_cuda_moe_source_tensor * tensors, int count) {
    if (!tensors || !d.count || d.count != count) { return false; }
    for (int i = 0; i < count; ++i) {
        if (!binding_same(d.tensors[i], tensors[i])) { return false; }
    }
    return true;
}

bool ggml_cuda_moe_source_binding_emit(const ggml_cuda_moe_source_binding & d, int device, void * stream) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(d); GGML_UNUSED(device); GGML_UNUSED(stream);
    return false;
#else
    int current_device = -1;
    if (!d.count || device != d.device || cudaGetDevice(&current_device) != cudaSuccess ||
            current_device != d.device) { return false; }
    const auto * t = d.tensors;
    if (d.count == 3) {
        norm_launch launch = {static_cast<const float *>(t[0].data), static_cast<const float *>(t[1].data),
            static_cast<float *>(t[2].data), {}, {}, {}, {}, {}, d.epsilon};
        memcpy(launch.ne, t[0].ne, sizeof(launch.ne)); memcpy(launch.gamma_ne, t[1].ne, sizeof(launch.gamma_ne));
        memcpy(launch.input_nb, t[0].nb, sizeof(launch.input_nb)); memcpy(launch.gamma_nb, t[1].nb, sizeof(launch.gamma_nb));
        memcpy(launch.output_nb, t[2].nb, sizeof(launch.output_nb));
        if (t[0].ne[0] < 1024) {
            weighted_rms_norm<256><<<uint32_t(d.rows), 256, 0, static_cast<cudaStream_t>(stream)>>>(launch);
        } else {
            weighted_rms_norm<1024><<<uint32_t(d.rows), 1024, 0, static_cast<cudaStream_t>(stream)>>>(launch);
        }
    } else if (d.count == 7) {
        gdn_ab_launch launch = {static_cast<const float *>(t[0].data), static_cast<const uint16_t *>(t[1].data),
            static_cast<const uint16_t *>(t[2].data), static_cast<const float *>(t[3].data), static_cast<const float *>(t[4].data),
            static_cast<float *>(t[5].data), static_cast<float *>(t[6].data), {}, {}, {}, {}, {},
            t[1].nb[1], t[2].nb[1], d.rows, uint64_t(t[5].ne[1]), (uint64_t(t[5].ne[1]) * 2 + 7) / 8};
        memcpy(launch.input_ne, t[0].ne, sizeof(launch.input_ne)); memcpy(launch.output_ne, t[5].ne, sizeof(launch.output_ne));
        memcpy(launch.input_nb, t[0].nb, sizeof(launch.input_nb)); memcpy(launch.gate_nb, t[5].nb, sizeof(launch.gate_nb));
        memcpy(launch.beta_nb, t[6].nb, sizeof(launch.beta_nb));
        const uint32_t blocks = uint32_t(launch.head_blocks * ((d.rows + 7) / 8));
        gdn_ab_kernel<<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(launch);
    } else { return false; }
    return cudaGetLastError() == cudaSuccess;
#endif
}
