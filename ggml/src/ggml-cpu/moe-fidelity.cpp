#include "moe-fidelity.h"
#include "ggml-cpu-impl.h"
#include "vec.h"
#include "moe-reference.h"
#include "../moe-fidelity-config.h"
#include <cmath>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstring>
#include <new>

namespace {

struct matrix {
    const ggml_tensor * node = nullptr;
    const ggml_tensor * view = nullptr;
    const ggml_tensor * add = nullptr;
    const uint8_t * weights = nullptr;
    const uint8_t * bias = nullptr;
    const ggml_type_traits_cpu * traits = nullptr;
    ggml_from_float_t quantize = nullptr;
    ggml_type dot_type = GGML_TYPE_COUNT;
    size_t row_stride = 0, expert_stride = 0, bias_stride = 0, quantized_row = 0;
    uint32_t width = 0, rows = 0, row_offset = 0;
};

struct plan {
    matrix gate, up, down;
    uint32_t experts = 0, routes = 0, source_rows = 0, threads = 0;
    ggml_glu_op activation = GGML_GLU_OP_SWIGLU;
    bool hidden_output = false, gated = true;
    bool reference = false, avx512 = false, q2_down = false;
    size_t actq, actq_codes, actq_scale, actq_sum, actq_hx;
    size_t ids, rows, groups, starts, order, map, marks, gate_input, up_input, gate_values, up_values, down_input, bytes;
};

static const ggml_backend_moe_cpu_region_source_v1 * source(const ggml_backend_moe_cpu_region_query_v1 & q, const ggml_tensor * tensor) {
    for (uint32_t i = 0; i < q.n_sources; ++i) { if (q.sources[i].tensor == tensor) { return &q.sources[i]; } }
    return nullptr;
}

static bool dimensions(const ggml_tensor * tensor) {
    if (!tensor || tensor->ne[3] != 1) { return false; }
    for (int i = 0; i < 3; ++i) { if (tensor->ne[i] <= 0 || uint64_t(tensor->ne[i]) > INT_MAX) { return false; } }
    return true;
}

static bool compile_matrix(const ggml_backend_moe_cpu_region_query_v1 & q, const ggml_tensor * value, matrix & m) {
    if (!dimensions(value) || value->type != GGML_TYPE_F32) { return false; }
    m.rows = uint32_t(value->ne[0]);
    if (value->op == GGML_OP_ADD_ID) {
        m.add = value;
        const auto * b = source(q, value->src[1]);
        if (!b || !dimensions(b->tensor) || b->tensor->type != GGML_TYPE_F32 || b->tensor->ne[0] != m.rows ||
                b->tensor->ne[2] != 1 || b->tensor->nb[0] != sizeof(float) ||
                uint64_t(m.rows) * sizeof(float) > SIZE_MAX || b->tensor->nb[1] != size_t(m.rows) * sizeof(float) ||
                uint64_t(b->tensor->nb[1]) * uint64_t(b->tensor->ne[1]) > SIZE_MAX ||
                b->tensor->nb[2] != b->tensor->nb[1] * size_t(b->tensor->ne[1]) ||
                b->tensor->nb[3] != b->tensor->nb[2] || !b->data || b->bytes < b->tensor->nb[2]) { return false; }
        m.bias = static_cast<const uint8_t *>(b->data);
        m.bias_stride = b->tensor->nb[1];
        value = value->src[0];
    }
    if (!dimensions(value) || value->type != GGML_TYPE_F32) { return false; }
    if (value->op == GGML_OP_VIEW) {
        m.view = value;
        if (!dimensions(value->view_src) || value->view_src->type != GGML_TYPE_F32 ||
                value->view_offs % sizeof(float) || value->view_offs / sizeof(float) > UINT32_MAX ||
                value->view_offs > value->view_src->nb[1] ||
                uint64_t(m.rows) * sizeof(float) > value->view_src->nb[1] - value->view_offs ||
                value->nb[0] != sizeof(float)) { return false; }
        for (int i = 1; i < GGML_MAX_DIMS; ++i) {
            if (value->ne[i] != value->view_src->ne[i] || value->nb[i] != value->view_src->nb[i]) { return false; }
        }
        m.row_offset = uint32_t(value->view_offs / sizeof(float));
        value = value->view_src;
    }
    if (!dimensions(value) || value->op != GGML_OP_MUL_MAT_ID || !dimensions(value->src[1]) ||
            value->src[1]->type != GGML_TYPE_F32) { return false; }
    m.node = value;
    const auto * bank = source(q, value->src[0]);
    if (!bank || !dimensions(bank->tensor) || !bank->data || bank->tensor->type < 0 || bank->tensor->type >= GGML_TYPE_COUNT ||
            uint64_t(m.row_offset) + m.rows > uint64_t(bank->tensor->ne[1]) ||
            value->ne[0] != bank->tensor->ne[1] || value->src[1]->ne[0] != bank->tensor->ne[0] ||
            bank->expert_stride != bank->tensor->nb[2] ||
            (m.add && m.add->src[1]->ne[1] != bank->tensor->ne[2])) { return false; }
    m.weights = static_cast<const uint8_t *>(bank->data);
    m.row_stride = bank->tensor->nb[1];
    m.expert_stride = bank->expert_stride;
    m.width = uint32_t(bank->tensor->ne[0]);
    m.traits = ggml_get_type_traits_cpu(bank->tensor->type);
    if (!m.traits || !m.traits->vec_dot || m.traits->vec_dot_type < 0 || m.traits->vec_dot_type >= GGML_TYPE_COUNT) { return false; }
    m.dot_type = m.traits->vec_dot_type;
    const auto * dot = ggml_get_type_traits_cpu(m.dot_type);
    m.quantize = dot ? dot->from_float : nullptr;
    const auto block = ggml_blck_size(m.dot_type);
    if (block <= 0 || m.width % block || (!m.quantize && m.dot_type != GGML_TYPE_F32) ||
            size_t(m.width / block) > SIZE_MAX / ggml_type_size(m.dot_type)) { return false; }
    m.quantized_row = ggml_row_size(m.dot_type, m.width);
    return m.quantized_row != 0;
}

static bool append(size_t & cursor, size_t count, size_t size, size_t & offset) {
    if (cursor > SIZE_MAX - 63 || (count && size > SIZE_MAX / count)) { return false; }
    offset = (cursor + 63) & ~size_t(63);
    if (count * size > SIZE_MAX - offset) { return false; }
    cursor = offset + count * size;
    return true;
}

static bool compile(const ggml_backend_moe_cpu_region_query_v1 & q, plan & p) {
    if (q.n_live_outputs < 1 || q.n_live_outputs > 2 || !q.live_outputs || q.n_dynamic_inputs != 2 ||
            !dimensions(q.activation) || q.activation->type != GGML_TYPE_F32 ||
            q.activation->ne[1] != 1 || q.n_threads == 0 || q.n_threads > GGML_MAX_N_THREADS ||
            q.routes_per_row == 0 || q.bucket_rows == 0 || q.source_row_capacity == 0 || q.bucket_rows > INT_MAX / q.routes_per_row ||
            !compile_matrix(q, q.live_outputs[0], p.down) || p.down.view) { return false; }
    const auto * hidden = p.down.node->src[1];
    const ggml_tensor * unary = nullptr;
    if (hidden && hidden->op == GGML_OP_SQR && hidden->src[0] &&
            hidden->src[0]->op == GGML_OP_UNARY && ggml_get_unary_op(hidden->src[0]) == GGML_UNARY_OP_RELU) {
        unary = hidden->src[0];
        p.gated = false;
        if (!compile_matrix(q, unary->src[0], p.gate)) { return false; }
        p.up = p.gate;
    } else {
        if (!hidden || hidden->op != GGML_OP_GLU || !hidden->src[0] || !hidden->src[1] ||
                !compile_matrix(q, hidden->src[0], p.gate) || !compile_matrix(q, hidden->src[1], p.up)) { return false; }
        p.activation = ggml_get_glu_op(hidden);
        if (p.activation != GGML_GLU_OP_SWIGLU && p.activation != GGML_GLU_OP_GEGLU) { return false; }
    }
    if (p.gate.node->src[1] != q.activation || p.up.node->src[1] != q.activation ||
            p.gate.width != p.up.width || p.gate.width != uint64_t(q.activation->ne[0]) ||
            p.gate.rows != p.up.rows || p.down.width != p.gate.rows ||
            p.gate.node->src[0]->ne[2] != p.up.node->src[0]->ne[2] ||
            p.gate.node->src[0]->ne[2] != p.down.node->src[0]->ne[2] ||
            (q.n_live_outputs == 2 && q.live_outputs[1] != hidden)) { return false; }
    for (uint32_t i = 0; i < q.n_body_nodes; ++i) {
        const auto * n = q.body_nodes[i];
        if (n != hidden && n != unary && n != p.gate.node && n != p.gate.view && n != p.gate.add &&
                n != p.up.node && n != p.up.view && n != p.up.add && n != p.down.node && n != p.down.add) { return false; }
    }
    p.experts = uint32_t(p.gate.node->src[0]->ne[2]);
    p.routes = q.bucket_rows * q.routes_per_row;
    p.source_rows = q.source_row_capacity;
    p.threads = q.n_threads;
    p.hidden_output = q.n_live_outputs == 2;
    const auto & config = ggml_moe_fidelity_selection();
    if (!config.valid) { return false; }
    p.reference = false;
#if defined(GGML_MOE_REFERENCE_X86)
    if (config.reference && p.gated) {
        __builtin_cpu_init();
        p.avx512 = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
            __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq") &&
            __builtin_cpu_supports("avx512vnni") && __builtin_cpu_supports("avx512vbmi");
        const auto gu_type = p.gate.node->src[0]->type, down_type = p.down.node->src[0]->type;
        p.reference = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c") &&
            ggml_moe_reference::iq256_supported(gu_type) && p.up.node->src[0]->type == gu_type &&
            (down_type == GGML_TYPE_IQ4_NL || down_type == GGML_TYPE_Q2_0) &&
            p.activation == GGML_GLU_OP_SWIGLU && !p.gate.bias && !p.up.bias && !p.down.bias &&
            p.gate.width % 256 == 0 && p.down.width % 64 == 0;
        p.q2_down = p.reference && down_type == GGML_TYPE_Q2_0;
    }
#endif
    return true;
}

static void quantize(const matrix & m, const float * input, void * output) {
    if (m.dot_type == GGML_TYPE_F32) { memcpy(output, input, m.quantized_row); }
    else { m.quantize(input, output, m.width); }
}

static float dot(const matrix & m, uint32_t expert, uint32_t row, const void * activation) {
    float result;
    m.traits->vec_dot(m.width, &result, 0, m.weights + size_t(expert) * m.expert_stride +
        size_t(row + m.row_offset) * m.row_stride, 0, activation, 0, 1);
    if (m.bias) { result += reinterpret_cast<const float *>(m.bias + size_t(expert) * m.bias_stride)[row]; }
    return result;
}

}

struct ggml_moe_cpu_fidelity {
    plan p;
    uint32_t n_routes = 0, n_groups = 0;
    int gu_type = 0;
    bool phase_down = false;
    std::atomic<uint32_t> gate_task{0}, down_task{0};
    std::atomic<bool> failed{false};
    ggml_abort_callback abort = nullptr;
    void * abort_data = nullptr;
    ggml_backend_moe_cpu_test_hook_v1_t hook = nullptr;
    void * hook_data = nullptr;
    float * output = nullptr, * hidden = nullptr;
    const uint32_t * output_destinations = nullptr;
    uint64_t output_stride = 0, hidden_stride = 0;

    float * output_row(uint32_t route) const {
        const uint32_t row = output_destinations ? output_destinations[route] : route;
        return reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(output) + size_t(row) * output_stride);
    }

    float * hidden_row(uint32_t route) const {
        const uint32_t row = output_destinations ? output_destinations[route] : route;
        return reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(hidden) + size_t(row) * hidden_stride);
    }

    template<typename T> T * at(size_t offset) { return reinterpret_cast<T *>(reinterpret_cast<uint8_t *>(this) + offset); }
    bool canceled() const { return failed.load(std::memory_order_relaxed) || (abort && abort(abort_data)); }

    void test_phase(uint32_t phase) noexcept {
        try { if (hook) { hook(hook_data, phase); } }
        catch (...) { failed.store(true, std::memory_order_relaxed); }
    }

    void native_task(bool down, uint32_t task, uint32_t tasks) {
#if defined(GGML_MOE_REFERENCE_X86)
        auto & s = *this;
        const auto & p = s.p;
        const auto & m = down ? p.down : p.gate;
        const auto * starts = s.at<uint32_t>(p.starts);
        const auto * order = s.at<uint32_t>(p.order);
        const auto * groups = s.at<int32_t>(p.groups);
        const auto * rows = s.at<uint32_t>(p.rows);
        const uint64_t total = uint64_t(s.n_groups) * m.rows;
        const uint64_t end = total * (task + 1) / tasks;
        for (uint64_t i = total * task / tasks; i < end;) {
            const auto group = uint32_t(i / m.rows), r0 = uint32_t(i % m.rows);
            const auto r1 = uint32_t(std::min<uint64_t>(m.rows, r0 + end - i));
            const auto expert = groups[group];
            for (uint32_t first = starts[group]; first < starts[group + 1]; first += 8) {
                const int nt = int(std::min(8u, starts[group + 1] - first));
                const void * acts[8];
                const ggml_moe_reference::ActQ * aq[8];
                float * outputs[8];
                for (int t = 0; t < nt; ++t) {
                    const auto route = order[first + t];
                    acts[t] = s.at<uint8_t>(down ? p.down_input : p.gate_input) +
                        size_t(down ? route : rows[route]) * m.quantized_row;
                    outputs[t] = down ? s.output_row(route) : s.at<float>(p.gate_values) + size_t(route) * p.gate.rows;
                    aq[t] = down && p.q2_down ? s.at<ggml_moe_reference::ActQ>(p.actq) + route : nullptr;
                }
                const auto * weights = m.weights + size_t(expert) * m.expert_stride + size_t(m.row_offset) * m.row_stride;
                if (down) {
                    if (p.q2_down) {
                        if (p.avx512) { ggml_moe_reference::q2_0_gguf_rows_multi(weights, m.row_stride, m.width / 64, aq, nt, outputs, r0, r1); }
                        else { ggml_moe_reference::q2_0_gguf_rows_multi_avx2(weights, m.row_stride, m.width / 64, aq, nt, outputs, r0, r1); }
                    } else if (starts[group + 1] - starts[group] >= 2) {
                        ggml_moe_reference::iq4nl256_down_rows(weights, m.row_stride, m.width, acts, nt, outputs, r0, r1);
                    } else {
                        for (uint32_t r = r0; r < r1; ++r) { outputs[0][r] = dot(p.down, expert, r, acts[0]); }
                    }
                } else if (starts[group + 1] - starts[group] >= 2) {
                    const auto * up = p.up.weights + size_t(expert) * p.up.expert_stride + size_t(p.up.row_offset) * p.up.row_stride;
                    if (p.avx512) { ggml_moe_reference::iq512_gu_rows(s.gu_type, weights, up, m.row_stride, p.up.row_stride, m.width, acts, nt, outputs, r0, r1); }
                    else { ggml_moe_reference::iq256_gu_rows(s.gu_type, weights, up, m.row_stride, p.up.row_stride, m.width, acts, nt, outputs, r0, r1); }
                } else {
                    const auto route = order[first];
                    for (uint32_t r = r0; r < r1; ++r) {
                        const float g = dot(p.gate, expert, r, acts[0]);
                        const float u = dot(p.up, expert, r, acts[0]);
                        s.at<float>(p.gate_values)[size_t(route) * m.rows + r] = (g / (1.f + std::exp(-g))) * u;
                    }
                }
            }
            i += r1 - r0;
        }

#else
        (void) down; (void) task; (void) tasks;
#endif
    }

    static void native_phase(void * data, int, int nth, ggml_threadpool *) {
        auto & s = *static_cast<ggml_moe_cpu_fidelity *>(data);
        auto & next = s.phase_down ? s.down_task : s.gate_task;
        const uint32_t tasks = uint32_t(nth) * 3;
        for (;;) {
            const auto task = next.fetch_add(1, std::memory_order_relaxed);
            if (task >= tasks) { break; }
            if (s.canceled()) { continue; }
            s.native_task(s.phase_down, task, tasks);
        }
    }

    void gate_work(uint32_t task, uint32_t tasks) {
        auto & s = *this;
        const auto & p = s.p;
        const auto * starts = s.at<uint32_t>(p.starts);
        const auto * order = s.at<uint32_t>(p.order);
        const auto * groups = s.at<int32_t>(p.groups);
        const auto * rows = s.at<uint32_t>(p.rows);
        const uint64_t gate_rows = uint64_t(s.n_groups) * p.gate.rows;
        const uint64_t end = gate_rows * (task + 1) / tasks;
        for (uint64_t i = gate_rows * task / tasks; i < end; ++i) {
            const auto group = uint32_t(i / p.gate.rows), channel = uint32_t(i % p.gate.rows);
            for (uint32_t entry = starts[group]; entry < starts[group + 1]; ++entry) {
                const auto route = order[entry], row = rows[route];
                s.at<float>(p.gate_values)[size_t(route) * p.gate.rows + channel] = dot(p.gate, groups[group], channel,
                    s.at<uint8_t>(p.gate_input) + size_t(row) * p.gate.quantized_row);
                if (p.gated) {
                    s.at<float>(p.up_values)[size_t(route) * p.up.rows + channel] = dot(p.up, groups[group], channel,
                        s.at<uint8_t>(p.up_input) + size_t(row) * p.up.quantized_row);
                }
            }
        }
    }

    void down_work(uint32_t task, uint32_t tasks) {
        auto & s = *this;
        const auto & p = s.p;
        const auto * starts = s.at<uint32_t>(p.starts);
        const auto * order = s.at<uint32_t>(p.order);
        const auto * groups = s.at<int32_t>(p.groups);
        const uint64_t down_rows = uint64_t(s.n_groups) * p.down.rows;
        const uint64_t end = down_rows * (task + 1) / tasks;
        for (uint64_t i = down_rows * task / tasks; i < end; ++i) {
            const auto group = uint32_t(i / p.down.rows), channel = uint32_t(i % p.down.rows);
            for (uint32_t entry = starts[group]; entry < starts[group + 1]; ++entry) {
                const auto route = order[entry];
                s.output_row(route)[channel] = dot(p.down, groups[group], channel,
                    s.at<uint8_t>(p.down_input) + size_t(route) * p.down.quantized_row);
            }
        }
    }

    static void run(void * data, int ith, int nth, ggml_threadpool * pool) {
        auto & s = *static_cast<ggml_moe_cpu_fidelity *>(data);
        const auto & p = s.p;
        const uint32_t tasks = uint32_t(nth) * 3;
        for (;;) {
            const auto task = s.gate_task.fetch_add(1, std::memory_order_relaxed);
            if (task >= tasks) { break; }
            if (s.canceled()) { continue; }
            s.gate_work(task, tasks);
        }
        ggml_barrier(pool);
        if (ith == 0) { s.test_phase(GGML_MOE_CPU_FIDELITY_AFTER_GU); }
        for (uint32_t route = uint32_t(ith); route < s.n_routes; route += uint32_t(nth)) {
            if (s.canceled()) { continue; }
            auto * gate = s.at<float>(p.gate_values) + size_t(route) * p.gate.rows;
            const auto * up = s.at<float>(p.up_values) + size_t(route) * p.up.rows;
            if (!p.gated) {
                ggml_vec_relu_f32(p.gate.rows, gate, gate);
                ggml_vec_sqr_f32(p.gate.rows, gate, gate);
            } else if (p.activation == GGML_GLU_OP_SWIGLU) { ggml_vec_swiglu_f32(p.gate.rows, gate, gate, up); }
            else { ggml_vec_geglu_f32(p.gate.rows, gate, gate, up); }
            if (s.hidden) { memcpy(s.hidden_row(route), gate, size_t(p.gate.rows) * sizeof(float)); }
            quantize(p.down, gate, s.at<uint8_t>(p.down_input) + size_t(route) * p.down.quantized_row);
        }
        ggml_barrier(pool);
        if (ith == 0) { s.test_phase(GGML_MOE_CPU_FIDELITY_AFTER_QUANT); }
        for (;;) {
            const auto task = s.down_task.fetch_add(1, std::memory_order_relaxed);
            if (task >= tasks) { break; }
            if (s.canceled()) { continue; }
            s.down_work(task, tasks);
        }
    }
};

static int32_t measure(const ggml_backend_moe_cpu_region_query_v1 & q, plan & p) {
    if (!compile(q, p)) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION; }
    if (size_t(p.gate.rows) > SIZE_MAX / sizeof(float) || size_t(p.up.rows) > SIZE_MAX / sizeof(float)) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY; }
    p.bytes = sizeof(ggml_moe_cpu_fidelity);
    if (!append(p.bytes, p.routes, sizeof(int32_t), p.ids) ||
            !append(p.bytes, p.routes, sizeof(uint32_t), p.rows) ||
            !append(p.bytes, p.routes, sizeof(int32_t), p.groups) ||
            !append(p.bytes, size_t(p.routes) + 1, sizeof(uint32_t), p.starts) ||
            !append(p.bytes, p.routes, sizeof(uint32_t), p.order) ||
            !append(p.bytes, p.experts, sizeof(int32_t), p.map) ||
            !append(p.bytes, p.source_rows, sizeof(uint8_t), p.marks) ||
            !append(p.bytes, p.source_rows, p.gate.quantized_row, p.gate_input)) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY; }
    if (p.gate.dot_type == p.up.dot_type && p.gate.quantized_row == p.up.quantized_row) { p.up_input = p.gate_input; }
    else if (!append(p.bytes, p.source_rows, p.up.quantized_row, p.up_input)) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY; }
    if (!append(p.bytes, p.routes, size_t(p.gate.rows) * sizeof(float), p.gate_values) ||
            !append(p.bytes, p.routes, size_t(p.up.rows) * sizeof(float), p.up_values) ||
            !append(p.bytes, p.routes, p.down.quantized_row, p.down_input)) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY; }
    if (p.reference && p.q2_down &&
            (!append(p.bytes, p.routes, sizeof(ggml_moe_reference::ActQ), p.actq) ||
             !append(p.bytes, p.routes, p.down.width, p.actq_codes) ||
             !append(p.bytes, p.routes, size_t(p.down.width / 32) * sizeof(float), p.actq_scale) ||
             !append(p.bytes, p.routes, size_t(p.down.width / 32) * sizeof(int32_t), p.actq_sum) ||
             !append(p.bytes, p.routes, size_t(p.down.width / 32) * sizeof(float), p.actq_hx))) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY; }
    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
}

int32_t ggml_moe_cpu_fidelity_measure(const ggml_backend_moe_cpu_region_query_v1 * query, uint64_t * bytes) {
    plan p;
    if (!query || !bytes) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT; }
    const int32_t status = measure(*query, p);
    if (status != GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK) { return status; }
    *bytes = p.bytes;
    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
}

ggml_moe_cpu_fidelity * ggml_moe_cpu_fidelity_prepare(const ggml_backend_moe_cpu_region_query_v1 * query, void * storage, uint64_t bytes) {
    plan p;
    if (!query || !storage || reinterpret_cast<uintptr_t>(storage) % 64 || measure(*query, p) != GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK || bytes < p.bytes) { return nullptr; }
    auto * body = new (storage) ggml_moe_cpu_fidelity;
    body->p = p;
    body->gu_type = p.gate.node->src[0]->type;
    if (p.reference && p.q2_down) {
        for (uint32_t route = 0; route < p.routes; ++route) {
            body->at<ggml_moe_reference::ActQ>(p.actq)[route] = {
                body->at<int8_t>(p.actq_codes) + size_t(route) * p.down.width,
                body->at<float>(p.actq_scale) + size_t(route) * (p.down.width / 32),
                body->at<int32_t>(p.actq_sum) + size_t(route) * (p.down.width / 32),
                body->at<float>(p.actq_hx) + size_t(route) * (p.down.width / 32), 0};
        }
    }
    for (auto * matrix : {&body->p.gate, &body->p.up, &body->p.down}) {
        matrix->node = matrix->view = matrix->add = nullptr;
    }
    return body;
}

void ggml_moe_cpu_fidelity_destroy(ggml_moe_cpu_fidelity * body) {
    if (body) { body->~ggml_moe_cpu_fidelity(); }
}

ggml_status ggml_moe_cpu_fidelity_run(ggml_moe_cpu_fidelity * body, ggml_threadpool * pool,
        const ggml_backend_moe_cpu_region_binding_v1 * binding, const ggml_backend_moe_cpu_dynamic_input_v1 * input,
        ggml_tensor * const * outputs, ggml_abort_callback abort, void * abort_data,
        ggml_backend_moe_cpu_test_hook_v1_t hook, void * hook_data,
        const ggml_backend_moe_cpu_output_v1 * private_outputs, const uint32_t * destinations) {
    auto & s = *body;
    const auto & p = s.p;
    s.n_routes = binding->n_routes;
    s.n_groups = 0;
    s.abort = abort;
    s.abort_data = abort_data;
    s.hook = hook;
    s.hook_data = hook_data;
    s.failed.store(false, std::memory_order_relaxed);
    s.output_destinations = private_outputs ? destinations : nullptr;
    s.output = private_outputs ? static_cast<float *>(private_outputs[0].data) : static_cast<float *>(outputs[0]->data);
    s.output_stride = private_outputs ? private_outputs[0].route_stride : uint64_t(p.down.rows) * sizeof(float);
    s.hidden = p.hidden_output ? (private_outputs ? static_cast<float *>(private_outputs[1].data) :
        static_cast<float *>(outputs[1]->data)) : nullptr;
    s.hidden_stride = p.hidden_output && private_outputs ? private_outputs[1].route_stride : uint64_t(p.gate.rows) * sizeof(float);
    s.gate_task.store(0, std::memory_order_relaxed);
    s.down_task.store(0, std::memory_order_relaxed);
    auto * ids = s.at<int32_t>(p.ids);
    auto * rows = s.at<uint32_t>(p.rows);
    auto * groups = s.at<int32_t>(p.groups);
    auto * starts = s.at<uint32_t>(p.starts);
    auto * order = s.at<uint32_t>(p.order);
    auto * map = s.at<int32_t>(p.map);
    auto * marks = s.at<uint8_t>(p.marks);
    memcpy(ids, binding->expert_ids, size_t(s.n_routes) * sizeof(*ids));
    memcpy(rows, binding->source_rows, size_t(s.n_routes) * sizeof(*rows));
    std::fill_n(map, p.experts, -1);
    memset(marks, 0, p.source_rows);
    for (uint32_t route = 0; route < s.n_routes; ++route) {
        if (s.canceled()) { return GGML_STATUS_ABORTED; }
        if (map[ids[route]] < 0) { map[ids[route]] = int32_t(s.n_groups); groups[s.n_groups++] = ids[route]; }
        const auto row = rows[route];
        if (!marks[row]) {
            const auto * activation = reinterpret_cast<const float *>(static_cast<const uint8_t *>(input->data) + size_t(row) * input->row_stride);
            quantize(p.gate, activation, s.at<uint8_t>(p.gate_input) + size_t(row) * p.gate.quantized_row);
            if (p.up_input != p.gate_input) {
                quantize(p.up, activation, s.at<uint8_t>(p.up_input) + size_t(row) * p.up.quantized_row);
            }
            marks[row] = 1;
        }
    }
    uint32_t count = 0;
    for (uint32_t group = 0; group < s.n_groups; ++group) {
        starts[group] = count;
        for (uint32_t route = 0; route < s.n_routes; ++route) { if (ids[route] == groups[group]) { order[count++] = route; } }
    }
    starts[s.n_groups] = count;
    ggml_status status;
    if (p.reference) {
        s.phase_down = false;
        status = ggml_threadpool_run_task(pool, p.threads, ggml_moe_cpu_fidelity::native_phase, &s);
        s.test_phase(GGML_MOE_CPU_FIDELITY_AFTER_GU);
        if (status == GGML_STATUS_SUCCESS && !s.canceled()) {
            for (uint32_t route = 0; route < s.n_routes && !s.canceled(); ++route) {
                const auto * hidden = s.at<float>(p.gate_values) + size_t(route) * p.gate.rows;
                if (s.hidden) { memcpy(s.hidden_row(route), hidden, size_t(p.gate.rows) * sizeof(float)); }
#if defined(GGML_MOE_REFERENCE_X86)
                if (p.q2_down) {
                    auto & aq = s.at<ggml_moe_reference::ActQ>(p.actq)[route];
                    if (p.avx512) { ggml_moe_reference::act_quant_q8_1(hidden, p.down.width, aq); }
                    else { ggml_moe_reference::act_quant_q8_1_avx2(hidden, p.down.width, aq); }
                } else
#endif
                { quantize(p.down, hidden, s.at<uint8_t>(p.down_input) + size_t(route) * p.down.quantized_row); }
            }
            s.test_phase(GGML_MOE_CPU_FIDELITY_AFTER_QUANT);
            s.phase_down = true;
            status = ggml_threadpool_run_task(pool, p.threads, ggml_moe_cpu_fidelity::native_phase, &s);
        }
    } else { status = ggml_threadpool_run_task(pool, p.threads, ggml_moe_cpu_fidelity::run, &s); }
    return s.failed.load(std::memory_order_relaxed) ? GGML_STATUS_FAILED : s.canceled() ? GGML_STATUS_ABORTED : status;
}

ggml_status ggml_moe_cpu_fidelity_begin(ggml_moe_cpu_fidelity * body,
        const ggml_backend_moe_cpu_region_binding_v1 * binding, const ggml_backend_moe_cpu_dynamic_input_v1 * input,
        ggml_tensor * const * outputs, ggml_abort_callback abort, void * abort_data,
        ggml_backend_moe_cpu_test_hook_v1_t hook, void * hook_data,
        const ggml_backend_moe_cpu_output_v1 * private_outputs, const uint32_t * destinations) {
    auto & s = *body;
    const auto & p = s.p;
    s.n_routes = binding->n_routes;
    s.n_groups = 0;
    s.abort = abort;
    s.abort_data = abort_data;
    s.hook = hook;
    s.hook_data = hook_data;
    s.failed.store(false, std::memory_order_relaxed);
    s.output_destinations = private_outputs ? destinations : nullptr;
    s.output = private_outputs ? static_cast<float *>(private_outputs[0].data) : static_cast<float *>(outputs[0]->data);
    s.output_stride = private_outputs ? private_outputs[0].route_stride : uint64_t(p.down.rows) * sizeof(float);
    s.hidden = p.hidden_output ? (private_outputs ? static_cast<float *>(private_outputs[1].data) :
        static_cast<float *>(outputs[1]->data)) : nullptr;
    s.hidden_stride = p.hidden_output && private_outputs ? private_outputs[1].route_stride : uint64_t(p.gate.rows) * sizeof(float);
    s.gate_task.store(0, std::memory_order_relaxed);
    s.down_task.store(0, std::memory_order_relaxed);
    auto * ids = s.at<int32_t>(p.ids);
    auto * rows = s.at<uint32_t>(p.rows);
    auto * groups = s.at<int32_t>(p.groups);
    auto * starts = s.at<uint32_t>(p.starts);
    auto * order = s.at<uint32_t>(p.order);
    auto * map = s.at<int32_t>(p.map);
    auto * marks = s.at<uint8_t>(p.marks);
    memcpy(ids, binding->expert_ids, size_t(s.n_routes) * sizeof(*ids));
    memcpy(rows, binding->source_rows, size_t(s.n_routes) * sizeof(*rows));
    std::fill_n(map, p.experts, -1);
    memset(marks, 0, p.source_rows);
    for (uint32_t route = 0; route < s.n_routes; ++route) {
        if (s.canceled()) { return GGML_STATUS_ABORTED; }
        if (map[ids[route]] < 0) { map[ids[route]] = int32_t(s.n_groups); groups[s.n_groups++] = ids[route]; }
        const auto row = rows[route];
        if (!marks[row]) {
            const auto * activation = reinterpret_cast<const float *>(static_cast<const uint8_t *>(input->data) + size_t(row) * input->row_stride);
            quantize(p.gate, activation, s.at<uint8_t>(p.gate_input) + size_t(row) * p.gate.quantized_row);
            if (p.up_input != p.gate_input) {
                quantize(p.up, activation, s.at<uint8_t>(p.up_input) + size_t(row) * p.up.quantized_row);
            }
            marks[row] = 1;
        }
    }
    uint32_t count = 0;
    for (uint32_t group = 0; group < s.n_groups; ++group) {
        starts[group] = count;
        for (uint32_t route = 0; route < s.n_routes; ++route) { if (ids[route] == groups[group]) { order[count++] = route; } }
    }
    starts[s.n_groups] = count;
    return GGML_STATUS_SUCCESS;
}

void ggml_moe_cpu_fidelity_task(ggml_moe_cpu_fidelity * body, bool down, uint32_t task, uint32_t tasks) {
    if (body->canceled()) { return; }
    if (body->p.reference) { body->native_task(down, task, tasks); }
    else if (down) { body->down_work(task, tasks); }
    else { body->gate_work(task, tasks); }
}

void ggml_moe_cpu_fidelity_after_gu(ggml_moe_cpu_fidelity * body) {
    body->test_phase(GGML_MOE_CPU_FIDELITY_AFTER_GU);
}

ggml_status ggml_moe_cpu_fidelity_result(ggml_moe_cpu_fidelity * body) {
    return body->failed.load(std::memory_order_relaxed) ? GGML_STATUS_FAILED :
        body->canceled() ? GGML_STATUS_ABORTED : GGML_STATUS_SUCCESS;
}

void ggml_moe_cpu_fidelity_quantize_hidden(ggml_moe_cpu_fidelity * body) {
    auto & s = *body;
    const auto & p = s.p;
    for (uint32_t route = 0; route < s.n_routes; ++route) {
        if (s.canceled()) { continue; }
        auto * gate = s.at<float>(p.gate_values) + size_t(route) * p.gate.rows;
        if (!p.reference) {
            const auto * up = s.at<float>(p.up_values) + size_t(route) * p.up.rows;
            if (!p.gated) {
                ggml_vec_relu_f32(p.gate.rows, gate, gate);
                ggml_vec_sqr_f32(p.gate.rows, gate, gate);
            } else if (p.activation == GGML_GLU_OP_SWIGLU) { ggml_vec_swiglu_f32(p.gate.rows, gate, gate, up); }
            else { ggml_vec_geglu_f32(p.gate.rows, gate, gate, up); }
        }
        if (s.hidden) { memcpy(s.hidden_row(route), gate, size_t(p.gate.rows) * sizeof(float)); }
#if defined(GGML_MOE_REFERENCE_X86)
        if (p.reference && p.q2_down) {
            auto & aq = s.at<ggml_moe_reference::ActQ>(p.actq)[route];
            if (p.avx512) { ggml_moe_reference::act_quant_q8_1(gate, p.down.width, aq); }
            else { ggml_moe_reference::act_quant_q8_1_avx2(gate, p.down.width, aq); }
        } else
#endif
        { quantize(p.down, gate, s.at<uint8_t>(p.down_input) + size_t(route) * p.down.quantized_row); }
    }
    s.test_phase(GGML_MOE_CPU_FIDELITY_AFTER_QUANT);
}
