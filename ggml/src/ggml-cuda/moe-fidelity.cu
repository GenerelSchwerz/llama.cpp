#include "moe-fidelity.cuh"
#include "common.cuh"
#include "ggml-backend-moe.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <new>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

#if defined(_WIN32)
#include <windows.h>
#else
#include <time.h>
#endif

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 12000
#include <cudaTypedefs.h>
#define GGML_MOE_FIDELITY_CUDA
#endif

namespace {

enum fault_kind { fault_none, fault_packet_epoch, fault_resident, fault_transfer, fault_cpu, fault_cpu_epoch, fault_cancel_before, fault_cancel_after, fault_teardown };

struct parameters {
    uint64_t epoch, identity;
    uint32_t rows, mode, fault, fault_boundary;
};

struct alignas(64) host_window {
    parameters params;
    std::atomic<uint32_t> cancel{0};
};

struct alignas(64) host_region {
    std::atomic<uint32_t> ready{0};
    std::atomic<uint32_t> done{0};
    uint64_t epoch = 0, identity = 0, cpu_epoch = 0;
    uint32_t rows = 0, cpu_count = 0, cpu_status = 0, accepted = 0;
    uint64_t ready_ns = 0, cpu_start_ns = 0, cpu_end_ns = 0, release_ns = 0;
};

struct producer { uint64_t epoch; uint32_t count, status; };
struct device_region {
    producer gpu[2];
    uint64_t work_start, work_end, wait_start, wait_end, commit;
    uint32_t accepted;
};
struct device_window { uint32_t failed, accepted; };

struct fixture;
struct callback_data { fixture * owner; uint32_t boundary; };
struct graph {
    cudaGraph_t definition = nullptr;
    cudaGraphExec_t instance = nullptr;
};

struct layout {
    size_t window, controls, inputs, ids, classes, weights, cpu;
    size_t state, records, gpu, combined, output;
    size_t routes, input_values, route_values, output_values;
    size_t pinned_bytes, device_bytes, scratch_bytes;
    size_t readback, record_copy, plan_events, transfer_events, callbacks, graphs;
};

static bool multiply(size_t a, size_t b, size_t & result) {
    if (a != 0 && b > SIZE_MAX / a) { return false; }
    result = a * b;
    return true;
}

static bool append(size_t & bytes, size_t count, size_t size, size_t & offset) {
    size_t added;
    if (bytes > SIZE_MAX - 255 || !multiply(count, size, added)) { return false; }
    offset = (bytes + 255) & ~size_t(255);
    if (added > SIZE_MAX - offset) { return false; }
    bytes = offset + added;
    return true;
}

static bool measure(const ggml_cuda_moe_fidelity_query_v1 * query, ggml_cuda_moe_fidelity_certificate_v1 * certificate,
        ggml_cuda_moe_fidelity_storage_v1 * storage, layout * measured = nullptr) {
    if (!query || !certificate || !storage || query->struct_size != sizeof(*query) || !query->graph ||
            !query->output || !query->boundaries || query->boundaries > 4096 || query->graph->uid == 0) { return false; }
    const auto * out = query->output;
    const auto * bank = out->src[0];
    const auto * input = out->src[1];
    const auto * ids = out->src[2];
    bool found = false;
    for (int i = 0; i < query->graph->n_nodes; ++i) { found |= query->graph->nodes[i] == out; }
    if (!found || out->op != GGML_OP_MUL_MAT_ID || !bank || !input || !ids || input->type != GGML_TYPE_F32 ||
            ids->type != GGML_TYPE_I32 || out->type != GGML_TYPE_F32 || input->ne[1] != 1 || input->ne[3] != 1 ||
            ids->ne[2] != 1 || ids->ne[3] != 1 || bank->ne[3] != 1 || out->ne[3] != 1 ||
            !ggml_is_contiguous(input) || !ggml_is_contiguous(ids) || !ggml_is_contiguous(bank) || !ggml_is_contiguous(out) ||
            input->ne[0] != bank->ne[0] || input->ne[2] != ids->ne[1] || out->ne[0] != bank->ne[1] ||
            out->ne[1] != ids->ne[0] || out->ne[2] != ids->ne[1]) { return false; }
    for (const int64_t n : {input->ne[0], input->ne[2], ids->ne[0], bank->ne[1], bank->ne[2]}) {
        if (n <= 0 || uint64_t(n) > UINT32_MAX) { return false; }
    }
    ggml_cuda_moe_fidelity_certificate_v1 cert = {1469598103934665603ull, query->graph->uid,
        uint32_t(input->ne[2]), uint32_t(ids->ne[0]), uint32_t(bank->ne[2]), uint32_t(input->ne[0]),
        uint32_t(bank->ne[1]), query->boundaries};
    const auto hash = [&](const void * data, size_t bytes) {
        const auto * values = static_cast<const uint8_t *>(data);
        for (size_t i = 0; i < bytes; ++i) { cert.identity = (cert.identity ^ values[i]) * 1099511628211ull; }
    };
    hash(&cert.graph_uid, sizeof(cert) - offsetof(ggml_cuda_moe_fidelity_certificate_v1, graph_uid));
    for (const auto * tensor : {input, ids, bank, out}) {
        hash(&tensor->type, sizeof(tensor->type));
        hash(tensor->ne, sizeof(tensor->ne));
        hash(tensor->nb, sizeof(tensor->nb));
    }
    layout l = {};
    size_t all_inputs, all_routes, all_values, all_outputs;
    if (!multiply(cert.row_capacity, cert.routes_per_row, l.routes) || l.routes > UINT32_MAX ||
            !multiply(cert.row_capacity, cert.input_width, l.input_values) ||
            !multiply(l.routes, cert.output_width, l.route_values) ||
            !multiply(cert.row_capacity, cert.output_width, l.output_values) ||
            !multiply(l.input_values, cert.boundaries, all_inputs) || !multiply(l.routes, cert.boundaries, all_routes) ||
            !multiply(l.route_values, cert.boundaries, all_values) || !multiply(l.output_values, cert.boundaries, all_outputs) ||
            !append(l.pinned_bytes, 1, sizeof(host_window), l.window) ||
            !append(l.pinned_bytes, cert.boundaries, sizeof(host_region), l.controls) ||
            !append(l.pinned_bytes, all_inputs, sizeof(float), l.inputs) ||
            !append(l.pinned_bytes, all_routes, sizeof(int32_t), l.ids) ||
            !append(l.pinned_bytes, all_routes, sizeof(uint32_t), l.classes) ||
            !append(l.pinned_bytes, all_routes, sizeof(float), l.weights) ||
            !append(l.pinned_bytes, all_values, sizeof(float), l.cpu) ||
            !append(l.device_bytes, 1, sizeof(device_window), l.state) ||
            !append(l.device_bytes, cert.boundaries, sizeof(device_region), l.records) ||
            !append(l.device_bytes, all_values, 2 * sizeof(float), l.gpu) ||
            !append(l.device_bytes, all_outputs, sizeof(float), l.combined) ||
            !append(l.device_bytes, all_outputs, sizeof(float), l.output)) { return false; }
    if (!append(l.scratch_bytes, all_outputs, sizeof(float), l.readback) ||
            !append(l.scratch_bytes, cert.boundaries, sizeof(device_region), l.record_copy) ||
            !append(l.scratch_bytes, cert.boundaries, sizeof(cudaEvent_t), l.plan_events) ||
            !append(l.scratch_bytes, cert.boundaries, sizeof(cudaEvent_t), l.transfer_events) ||
            !append(l.scratch_bytes, cert.boundaries, sizeof(callback_data), l.callbacks) ||
            !append(l.scratch_bytes, cert.boundaries + 1, sizeof(graph), l.graphs)) { return false; }
    *certificate = cert;
    *storage = {l.pinned_bytes, l.device_bytes, l.scratch_bytes};
    if (measured) { *measured = l; }
    return true;
}

static bool measure_api(const ggml_cuda_moe_fidelity_query_v1 * query, ggml_cuda_moe_fidelity_certificate_v1 * cert,
        ggml_cuda_moe_fidelity_storage_v1 * storage) {
    return measure(query, cert, storage);
}

#ifdef GGML_MOE_FIDELITY_CUDA

using clock_type = std::chrono::steady_clock;
static uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(clock_type::now().time_since_epoch()).count();
}

static uint64_t thread_ns() {
#ifdef _WIN32
    FILETIME created, exited, kernel, user;
    if (!GetThreadTimes(GetCurrentThread(), &created, &exited, &kernel, &user)) { return 0; }
    return ((uint64_t(kernel.dwHighDateTime) << 32) + kernel.dwLowDateTime +
        (uint64_t(user.dwHighDateTime) << 32) + user.dwLowDateTime) * 100;
#else
    timespec time = {};
    return clock_gettime(CLOCK_THREAD_CPUTIME_ID, &time) == 0 ? uint64_t(time.tv_sec) * 1000000000ull + time.tv_nsec : 0;
#endif
}

static void pause_host() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    _mm_pause();
#else
    std::this_thread::yield();
#endif
}

static __device__ uint64_t gpu_ns() {
    uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

static __device__ void pause_gpu() {
#if __CUDA_ARCH__ >= 700
    __nanosleep(100);
#endif
}

static __host__ __device__ float transform(float input, int32_t expert, uint32_t element) {
    return input + float(expert + 1) * 0.25f + float(element % 7) * 0.125f;
}

struct view {
    host_window * window;
    host_region * controls;
    float * inputs;
    int32_t * ids;
    uint32_t * classes;
    float * weights;
    float * cpu;
    device_window * state;
    device_region * records;
    float * gpu;
    float * combined;
    float * output;
    ggml_cuda_moe_fidelity_certificate_v1 cert;
    size_t input_values, routes, route_values, output_values;
};

static __global__ void initialize(view v) {
    if (threadIdx.x == 0) { *v.state = {}; }
    const size_t count = size_t(v.cert.boundaries) * v.output_values;
    for (size_t i = threadIdx.x; i < count; i += blockDim.x) { v.output[i] = -12345.0f; }
}

static __global__ void publish_ready(view v, uint32_t boundary, bool ring) {
    const auto p = v.window->params;
    auto & control = v.controls[boundary];
    if (threadIdx.x == 0) {
        control.epoch = p.epoch - (p.fault == fault_packet_epoch && boundary == p.fault_boundary);
        control.identity = p.identity;
        control.rows = p.rows;
    }
    for (size_t i = threadIdx.x; i < size_t(p.rows) * v.cert.input_width; i += blockDim.x) {
        v.inputs[boundary * v.input_values + i] = float((p.epoch + boundary + i) % 31) * 0.03125f;
    }
    for (size_t i = threadIdx.x; i < size_t(p.rows) * v.cert.routes_per_row; i += blockDim.x) {
        const size_t index = boundary * v.routes + i;
        v.ids[index] = int32_t((p.epoch + boundary + i / 2) % v.cert.expert_count);
        v.classes[index] = p.mode == 0 ? uint32_t(i % 2) : p.mode == 1 ? 2 : uint32_t(i % 3);
        v.weights[index] = float(1 + i % v.cert.routes_per_row % 3) * 0.25f;
    }
    __threadfence_system();
    __syncthreads();
    if (ring && threadIdx.x == 0) {
        __threadfence_system();
        *reinterpret_cast<volatile uint32_t *>(&control.ready) = 1;
    }
}

static __global__ void gpu_producer(view v, uint32_t boundary, uint32_t kind, uint32_t delay_us) {
    const auto p = v.window->params;
    auto & record = v.records[boundary];
    if (threadIdx.x == 0 && kind == 0) {
        record.work_start = gpu_ns();
        while (gpu_ns() - record.work_start < uint64_t(delay_us) * 1000) { pause_gpu(); }
    }
    __syncthreads();
    const size_t count = size_t(p.rows) * v.cert.routes_per_row;
    for (size_t i = threadIdx.x; i < count * v.cert.output_width; i += blockDim.x) {
        const size_t route = i / v.cert.output_width;
        const uint32_t element = i % v.cert.output_width;
        if (v.classes[boundary * v.routes + route] != kind) { continue; }
        const size_t input = boundary * v.input_values + route / v.cert.routes_per_row * v.cert.input_width + element % v.cert.input_width;
        v.gpu[(size_t(kind) * v.cert.boundaries + boundary) * v.route_values + i] =
            transform(v.inputs[input], v.ids[boundary * v.routes + route], element);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        uint32_t routes = 0;
        for (size_t i = 0; i < count; ++i) { routes += v.classes[boundary * v.routes + i] == kind; }
        record.gpu[kind] = {p.epoch, routes, uint32_t(boundary == p.fault_boundary &&
            p.fault == (kind == 0 ? fault_resident : fault_transfer))};
        if (kind == 0) { record.work_end = gpu_ns(); }
    }
}

static __global__ void stamp_wait(view v, uint32_t boundary, bool after) {
    (after ? v.records[boundary].wait_end : v.records[boundary].wait_start) = gpu_ns();
}

static __global__ void poll_done(view v, uint32_t boundary) {
    const auto * done = reinterpret_cast<const volatile uint32_t *>(&v.controls[boundary].done);
    const auto * cancel = reinterpret_cast<const volatile uint32_t *>(&v.window->cancel);
    const auto start = gpu_ns();
    while (*done != 1 && !*cancel) {
        if (gpu_ns() - start > 2000000000ull) { v.state->failed = 1; return; }
        pause_gpu();
    }
    __threadfence_system();
}

static __global__ void combine(view v, uint32_t boundary) {
    const auto p = v.window->params;
    const auto & control = v.controls[boundary];
    auto & record = v.records[boundary];
    if (threadIdx.x == 0) {
        uint32_t counts[3] = {};
        for (size_t i = 0; i < size_t(p.rows) * v.cert.routes_per_row; ++i) {
            const auto kind = v.classes[boundary * v.routes + i];
            if (kind < 3) { ++counts[kind]; }
        }
        const bool accepted = !v.state->failed && control.accepted && control.cpu_count == counts[2] &&
            control.cpu_epoch == p.epoch && record.gpu[0].epoch == p.epoch && record.gpu[1].epoch == p.epoch &&
            record.gpu[0].count == counts[0] && record.gpu[1].count == counts[1] &&
            !record.gpu[0].status && !record.gpu[1].status;
        record.accepted = accepted;
        v.state->failed |= !accepted;
    }
    __syncthreads();
    if (record.accepted) {
        for (size_t i = threadIdx.x; i < size_t(p.rows) * v.cert.output_width; i += blockDim.x) {
            const size_t row = i / v.cert.output_width;
            const uint32_t element = i % v.cert.output_width;
            float value = 0;
            for (uint32_t rank = 0; rank < v.cert.routes_per_row; ++rank) {
                const size_t route = row * v.cert.routes_per_row + rank;
                const auto kind = v.classes[boundary * v.routes + route];
                const float * source = kind == 2 ? v.cpu + boundary * v.route_values :
                    v.gpu + (size_t(kind) * v.cert.boundaries + boundary) * v.route_values;
                value += v.weights[boundary * v.routes + route] * source[route * v.cert.output_width + element];
            }
            v.combined[boundary * v.output_values + i] = value;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) { record.commit = gpu_ns(); }
}

static __global__ void commit_window(view v) {
    const auto * cancel = reinterpret_cast<const volatile uint32_t *>(&v.window->cancel);
    v.state->accepted = !v.state->failed && !*cancel;
    if (v.window->params.fault == fault_cancel_after) {
        __threadfence_system();
        *reinterpret_cast<volatile uint32_t *>(&v.window->cancel) = 1;
    }
}

static __global__ void publish_window(view v) {
    if (!v.state->accepted) { return; }
    for (size_t i = threadIdx.x; i < size_t(v.cert.boundaries) * v.output_values; i += blockDim.x) {
        if (i % v.output_values < size_t(v.window->params.rows) * v.cert.output_width) { v.output[i] = v.combined[i]; }
    }
}

struct fixture {
    ggml_cuda_moe_fidelity_certificate_v1 cert;
    ggml_cuda_moe_fidelity_options_v1 options;
    ggml_cuda_moe_fidelity_report_v1 & report;
    layout sizes;
    uint8_t * host = nullptr, * alias = nullptr, * device = nullptr;
    view h = {}, d = {};
    cudaStream_t stream = nullptr, io = nullptr;
    cudaEvent_t * plan_events = nullptr, * transfer_events = nullptr;
    graph * graphs = nullptr;
    callback_data * callbacks = nullptr;
    std::unique_ptr<uint8_t[]> scratch;
    size_t graph_count = 0;
    std::thread worker, supervisor;
    std::mutex mutex;
    std::once_flag stop_once;
    std::condition_variable condition;
    bool stopping = false, armed = false, worker_idle = true;
    bool supervisor_stopping = false, watching = false, supervisor_active = false;
    clock_type::time_point window_deadline;
    std::atomic<bool> closing{false}, teardown_entered{false};
    std::atomic<bool> timed_out{false};
    bool capture_unavailable = false;
    uint64_t epoch = 0;
    PFN_cuStreamWaitValue32_v11070 memop_wait = nullptr;
    PFN_cuStreamWriteValue32_v11070 memop_write = nullptr;

    fixture(const ggml_cuda_moe_fidelity_certificate_v1 & c, const ggml_cuda_moe_fidelity_options_v1 & o,
            ggml_cuda_moe_fidelity_report_v1 & r, const layout & l) : cert(c), options(o), report(r), sizes(l) {}

    void reason(const char * message) { snprintf(report.reason, sizeof(report.reason), "%s", message); }

    view make_view(uint8_t * mapped) {
        return {reinterpret_cast<host_window *>(mapped + sizes.window), reinterpret_cast<host_region *>(mapped + sizes.controls),
            reinterpret_cast<float *>(mapped + sizes.inputs), reinterpret_cast<int32_t *>(mapped + sizes.ids),
            reinterpret_cast<uint32_t *>(mapped + sizes.classes), reinterpret_cast<float *>(mapped + sizes.weights),
            reinterpret_cast<float *>(mapped + sizes.cpu), reinterpret_cast<device_window *>(device + sizes.state),
            reinterpret_cast<device_region *>(device + sizes.records), reinterpret_cast<float *>(device + sizes.gpu),
            reinterpret_cast<float *>(device + sizes.combined), reinterpret_cast<float *>(device + sizes.output),
            cert, sizes.input_values, sizes.routes, sizes.route_values, sizes.output_values};
    }

    bool resolve_memops() {
#ifdef _WIN32
        return false;
#else
        const auto resolve = [](const char * name, void ** function) {
            cudaDriverEntryPointQueryResult query;
#if CUDART_VERSION >= 12050
            const auto error = cudaGetDriverEntryPointByVersion(name, function, 11070, cudaEnableDefault, &query);
#else
            const auto error = cudaGetDriverEntryPoint(name, function, cudaEnableDefault, &query);
#endif
            return error == cudaSuccess && query == cudaDriverEntryPointSuccess && *function != nullptr;
        };
        return resolve("cuStreamWaitValue32", reinterpret_cast<void **>(&memop_wait)) &&
            resolve("cuStreamWriteValue32", reinterpret_cast<void **>(&memop_write));
#endif
    }

    bool allocate() {
        if (!std::atomic<uint32_t>::is_always_lock_free ||
                cudaHostAlloc(reinterpret_cast<void **>(&host), sizes.pinned_bytes, cudaHostAllocMapped) != cudaSuccess ||
                cudaHostGetDevicePointer(reinterpret_cast<void **>(&alias), host, 0) != cudaSuccess) {
            reason("mapped control alias unavailable"); return false;
        }
        if (cudaMalloc(reinterpret_cast<void **>(&device), sizes.device_bytes) != cudaSuccess ||
                cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess ||
                cudaStreamCreateWithFlags(&io, cudaStreamNonBlocking) != cudaSuccess) {
            reason("fixture allocation failed"); return false;
        }
        memset(host, 0, sizes.pinned_bytes);
        h = make_view(host);
        d = make_view(alias);
        new (h.window) host_window;
        for (uint32_t i = 0; i < cert.boundaries; ++i) { new (h.controls + i) host_region; }
        scratch.reset(new uint8_t[sizes.scratch_bytes]());
        plan_events = reinterpret_cast<cudaEvent_t *>(scratch.get() + sizes.plan_events);
        transfer_events = reinterpret_cast<cudaEvent_t *>(scratch.get() + sizes.transfer_events);
        callbacks = reinterpret_cast<callback_data *>(scratch.get() + sizes.callbacks);
        graphs = reinterpret_cast<graph *>(scratch.get() + sizes.graphs);
        graph_count = options.arm == GGML_CUDA_MOE_FIDELITY_SEGMENTED ? cert.boundaries + 1 : 1;
        for (size_t i = 0; i < graph_count; ++i) { new (graphs + i) graph; }
        for (uint32_t i = 0; i < cert.boundaries; ++i) {
            new (plan_events + i) cudaEvent_t(nullptr);
            new (transfer_events + i) cudaEvent_t(nullptr);
            new (callbacks + i) callback_data{this, i};
        }
        for (uint32_t i = 0; i < cert.boundaries; ++i) {
            if (cudaEventCreateWithFlags(&plan_events[i], cudaEventDisableTiming) != cudaSuccess ||
                    cudaEventCreateWithFlags(&transfer_events[i], cudaEventDisableTiming) != cudaSuccess) { return false; }
        }
        return true;
    }

    void serve() {
        const auto cpu_started = thread_ns();
        std::unique_lock<std::mutex> lock(mutex);
        for (;;) {
            condition.wait(lock, [&] { return stopping || armed; });
            if (stopping) { break; }
            worker_idle = false;
            lock.unlock();
            for (uint32_t b = 0; b < cert.boundaries && !closing.load(); ++b) {
                auto & control = h.controls[b];
                const auto deadline = now_ns() + 2000000000ull;
                while (control.ready.load(std::memory_order_acquire) != 1 && !closing.load()) {
                    if (now_ns() > deadline) { h.window->cancel.store(1, std::memory_order_release); break; }
                    pause_host();
                }
                if (closing.load()) { break; }
                control.ready_ns = now_ns();
                ++report.ready_observations;
                control.cpu_start_ns = now_ns();
                const auto p = h.window->params;
                const bool target = b == p.fault_boundary;
                control.cpu_status = control.epoch != p.epoch || control.identity != cert.identity || control.rows != p.rows;
                control.cpu_count = 0;
                control.cpu_epoch = p.epoch - (target && p.fault == fault_cpu_epoch);
                if (target && p.fault == fault_teardown) {
                    {
                        std::lock_guard<std::mutex> notify_lock(mutex);
                        teardown_entered.store(true, std::memory_order_release);
                        condition.notify_all();
                    }
                    while (!closing.load()) { pause_host(); }
                    break;
                }
                const auto until = control.cpu_start_ns + uint64_t(options.cpu_delay_us) * 1000;
                while (now_ns() < until && !closing.load()) { pause_host(); }
                for (size_t route = 0; route < size_t(p.rows) * cert.routes_per_row; ++route) {
                    const size_t index = b * sizes.routes + route;
                    if (h.classes[index] != 2) { continue; }
                    ++control.cpu_count;
                    for (uint32_t i = 0; i < cert.output_width; ++i) {
                        const size_t input = b * sizes.input_values + route / cert.routes_per_row * cert.input_width + i % cert.input_width;
                        h.cpu[b * sizes.route_values + route * cert.output_width + i] = transform(h.inputs[input], h.ids[index], i);
                    }
                }
                control.cpu_status |= (target && p.fault == fault_cpu) || closing.load() || h.window->cancel.load();
                control.cpu_end_ns = now_ns();
                ++report.services;
                report.zero_cpu += control.cpu_count == 0;
                report.cpu_routes += control.cpu_count;
                {
                    std::lock_guard<std::mutex> completion_lock(mutex);
                    control.done.store(1, std::memory_order_release);
                    control.release_ns = now_ns();
                    condition.notify_all();
                }
            }
            lock.lock();
            armed = false;
            worker_idle = true;
            condition.notify_all();
        }
        report.coordinator_cpu_ns = thread_ns() - cpu_started;
    }

    static void CUDART_CB join_callback(void * opaque) {
        const auto & data = *static_cast<callback_data *>(opaque);
        auto & self = *data.owner;
        std::unique_lock<std::mutex> lock(self.mutex);
        if (!self.condition.wait_for(lock, std::chrono::seconds(2), [&] {
                return self.h.controls[data.boundary].done.load(std::memory_order_acquire) == 1 || self.closing.load();
            })) { self.h.window->cancel.store(1, std::memory_order_release); }
    }

    static void CUDART_CB late_accept(void * opaque) {
        const auto & data = *static_cast<callback_data *>(opaque);
        auto & self = *data.owner;
        auto & control = self.h.controls[data.boundary];
        const auto p = self.h.window->params;
        if (data.boundary == p.fault_boundary && p.fault == fault_cancel_before) {
            self.h.window->cancel.store(1, std::memory_order_release);
        }
        control.accepted = control.done.load(std::memory_order_acquire) == 1 && control.cpu_status == 0 &&
            control.epoch == p.epoch && control.cpu_epoch == p.epoch && control.identity == p.identity &&
            !self.h.window->cancel.load() && !self.closing.load();
    }

    bool prefix(uint32_t b) {
        publish_ready<<<1, 256, 0, stream>>>(d, b, !options.ready_memop);
        if (options.ready_memop) {
            const auto result = memop_write(stream, reinterpret_cast<CUdeviceptr>(&d.controls[b].ready), 1, CU_STREAM_WRITE_VALUE_DEFAULT);
            if (result != CUDA_SUCCESS) {
                capture_unavailable = result == CUDA_ERROR_NOT_SUPPORTED || result == CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
                snprintf(report.reason, sizeof(report.reason), "ready memop capture CUresult=%d", int(result));
                return false;
            }
        }
        if (cudaEventRecord(plan_events[b], stream) != cudaSuccess || cudaStreamWaitEvent(io, plan_events[b], 0) != cudaSuccess) { return false; }
        gpu_producer<<<1, 256, 0, io>>>(d, b, 1, 0);
        if (cudaEventRecord(transfer_events[b], io) != cudaSuccess) { return false; }
        gpu_producer<<<1, 256, 0, stream>>>(d, b, 0, options.gpu_delay_us);
        if (cudaStreamWaitEvent(stream, transfer_events[b], 0) != cudaSuccess) { return false; }
        stamp_wait<<<1, 1, 0, stream>>>(d, b, false);
        return cudaGetLastError() == cudaSuccess;
    }

    bool tail(uint32_t b, bool wait) {
        if (wait) {
            if (options.arm == GGML_CUDA_MOE_FIDELITY_CALLBACK) {
                if (cudaLaunchHostFunc(stream, join_callback, &callbacks[b]) != cudaSuccess) { return false; }
            } else if (options.arm == GGML_CUDA_MOE_FIDELITY_POLL) {
                poll_done<<<1, 1, 0, stream>>>(d, b);
            } else {
                const auto result = memop_wait(stream, reinterpret_cast<CUdeviceptr>(&d.controls[b].done), 1, CU_STREAM_WAIT_VALUE_EQ);
                if (result != CUDA_SUCCESS) {
                    capture_unavailable = result == CUDA_ERROR_NOT_SUPPORTED || result == CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
                    snprintf(report.reason, sizeof(report.reason), "completion memop capture CUresult=%d", int(result));
                    return false;
                }
            }
        }
        stamp_wait<<<1, 1, 0, stream>>>(d, b, true);
        if (cudaLaunchHostFunc(stream, late_accept, &callbacks[b]) != cudaSuccess) { return false; }
        combine<<<1, 256, 0, stream>>>(d, b);
        return cudaGetLastError() == cudaSuccess;
    }

    bool capture() {
        const bool segmented = options.arm == GGML_CUDA_MOE_FIDELITY_SEGMENTED;
        for (size_t i = 0; i < graph_count; ++i) {
            if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { return false; }
            bool ok = true;
            if (i == 0) { initialize<<<1, 256, 0, stream>>>(d); }
            if (segmented) {
                if (i != 0) { ok = tail(uint32_t(i - 1), false); }
                if (ok && i < cert.boundaries) { ok = prefix(uint32_t(i)); }
            } else {
                for (uint32_t b = 0; ok && b < cert.boundaries; ++b) { ok = prefix(b) && tail(b, true); }
            }
            if (i + 1 == graph_count) {
                commit_window<<<1, 1, 0, stream>>>(d);
                publish_window<<<1, 256, 0, stream>>>(d);
            }
            const auto ended = cudaStreamEndCapture(stream, &graphs[i].definition);
            if (!ok || ended != cudaSuccess || !graphs[i].definition ||
                    cudaGraphInstantiate(&graphs[i].instance, graphs[i].definition, nullptr, nullptr, 0) != cudaSuccess ||
                    cudaGraphUpload(graphs[i].instance, stream) != cudaSuccess) { return false; }
            ++report.captures;
        }
        return drain(false);
    }

    void stop_worker() {
        std::call_once(stop_once, [&] {
            closing.store(true);
            if (h.window) { h.window->cancel.store(1, std::memory_order_release); }
            {
                std::lock_guard<std::mutex> lock(mutex);
                stopping = true;
                condition.notify_all();
            }
            if (worker.joinable()) { worker.join(); }
            std::lock_guard<std::mutex> lock(mutex);
            if (h.controls) {
                for (uint32_t b = 0; b < cert.boundaries; ++b) {
                    if (h.controls[b].done.load(std::memory_order_acquire) == 1) { continue; }
                    h.controls[b].cpu_status = 1;
                    h.controls[b].done.store(1, std::memory_order_release);
                }
            }
            condition.notify_all();
        });
    }

    void supervise() {
        const auto started = thread_ns();
        std::unique_lock<std::mutex> lock(mutex);
        for (;;) {
            condition.wait(lock, [&] { return supervisor_stopping || watching; });
            if (supervisor_stopping) { break; }
            supervisor_active = true;
            const bool notified = condition.wait_until(lock, window_deadline, [&] {
                return supervisor_stopping || !watching || teardown_entered.load(std::memory_order_acquire);
            });
            if (!notified || teardown_entered.load(std::memory_order_acquire)) {
                timed_out.store(!notified);
                lock.unlock();
                stop_worker();
                lock.lock();
                watching = false;
            }
            supervisor_active = false;
            condition.notify_all();
        }
        report.supervisor_cpu_ns = thread_ns() - started;
    }

    bool drain(bool cancel) {
        if (cancel) { stop_worker(); }
        const auto deadline = now_ns() + 3000000000ull;
        for (;;) {
            const auto main_status = stream ? cudaStreamQuery(stream) : cudaSuccess;
            const auto io_status = io ? cudaStreamQuery(io) : cudaSuccess;
            if (main_status == cudaSuccess && io_status == cudaSuccess) { ++report.finite_drains; return true; }
            if ((main_status != cudaSuccess && main_status != cudaErrorNotReady) ||
                    (io_status != cudaSuccess && io_status != cudaErrorNotReady)) { return false; }
            if (now_ns() > deadline) { GGML_ABORT("moe-fidelity: CUDA did not drain after terminal failure"); }
            std::this_thread::yield();
        }
    }

    bool observe_until_done(uint32_t boundary, uint64_t deadline) {
        uint64_t queried = 0;
        while (h.controls[boundary].done.load(std::memory_order_acquire) != 1) {
            const auto now = now_ns();
            if (now > deadline) { reason("host completion deadline"); return false; }
            if (now - queried > 2000000) {
                const auto status = cudaStreamQuery(stream);
                if (status != cudaSuccess && status != cudaErrorNotReady) { return false; }
                queried = now;
            }
            pause_host();
        }
        return true;
    }

    bool replay(uint32_t fault, ggml_cuda_moe_fidelity_sample_v1 & sample) {
        sample = {};
        ++epoch;
        h.window->params = {epoch, cert.identity, epoch % 2 ? cert.row_capacity : 1u, uint32_t(epoch % 3), fault,
            cert.boundaries / 2};
        h.window->cancel.store(0);
        for (uint32_t b = 0; b < cert.boundaries; ++b) {
            h.controls[b].~host_region();
            new (h.controls + b) host_region;
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            armed = true;
            worker_idle = false;
            window_deadline = clock_type::now() + std::chrono::seconds(2);
            watching = true;
            condition.notify_all();
        }
        const auto started = now_ns();
        const auto deadline = started + 2000000000ull;
        for (size_t i = 0; i < graph_count; ++i) {
            if (i) {
                const auto before = now_ns();
                if (!observe_until_done(uint32_t(i - 1), deadline)) { return false; }
                sample.continuation_ns += now_ns() - before;
            }
            const auto before = now_ns();
            const auto launched = cudaGraphLaunch(graphs[i].instance, stream);
            sample.launch_ns += now_ns() - before;
            ++report.graph_launches;
            if (launched != cudaSuccess) { reason("graph launch failed"); return false; }
        }
        const auto observing = now_ns();
        for (;;) {
            const auto status = cudaStreamQuery(stream);
            if (status == cudaSuccess) { break; }
            if (status != cudaErrorNotReady || now_ns() > deadline) {
                uint32_t pending = 0;
                while (pending < cert.boundaries && h.controls[pending].done.load() == 1) { ++pending; }
                snprintf(report.reason, sizeof(report.reason), "window deadline epoch=%llu fault=%u pending=%u cuda=%d",
                    (unsigned long long) epoch, fault, pending, int(status));
                return false;
            }
            std::this_thread::yield();
        }
        sample.observation_ns = now_ns() - observing;
        sample.whole_ns = now_ns() - started;
        {
            std::unique_lock<std::mutex> lock(mutex);
            watching = false;
            condition.notify_all();
            if (!condition.wait_for(lock, std::chrono::seconds(2), [&] { return (worker_idle || stopping) && !supervisor_active; })) { return false; }
        }
        if (timed_out.load()) { reason("independent host supervisor deadline"); return false; }
        if (!drain(false)) { return false; }
        auto * outputs = reinterpret_cast<float *>(scratch.get() + sizes.readback);
        const size_t output_bytes = size_t(cert.boundaries) * sizes.output_values * sizeof(float);
        auto * records = reinterpret_cast<device_region *>(scratch.get() + sizes.record_copy);
        device_window state = {};
        if (cudaMemcpy(outputs, h.output, output_bytes, cudaMemcpyDeviceToHost) != cudaSuccess ||
                cudaMemcpy(records, h.records, cert.boundaries * sizeof(device_region), cudaMemcpyDeviceToHost) != cudaSuccess ||
                cudaMemcpy(&state, h.state, sizeof(state), cudaMemcpyDeviceToHost) != cudaSuccess) { return false; }
        const bool accepted = fault == fault_none || fault == fault_cancel_after;
        if (state.accepted != uint32_t(accepted)) { reason("window acceptance mismatch"); return false; }
        const auto p = h.window->params;
        for (uint32_t b = 0; b < cert.boundaries; ++b) {
            const auto & control = h.controls[b];
            sample.cpu_service_ns += control.cpu_end_ns >= control.cpu_start_ns ? control.cpu_end_ns - control.cpu_start_ns : 0;
            sample.ready_to_cpu_ns += control.cpu_start_ns >= control.ready_ns ? control.cpu_start_ns - control.ready_ns : 0;
            sample.completion_publish_ns += control.release_ns >= control.cpu_end_ns ? control.release_ns - control.cpu_end_ns : 0;
            sample.gpu_wait_ns += records[b].wait_end - records[b].wait_start;
            sample.gpu_work_ns += records[b].work_end - records[b].work_start;
            sample.gpu_commit_ns += records[b].commit - records[b].wait_end;
            report.resident_routes += records[b].gpu[0].count;
            report.transfer_routes += records[b].gpu[1].count;
            for (size_t i = 0; i < sizes.output_values; ++i) {
                float expected = -12345.0f;
                if (accepted && i < size_t(p.rows) * cert.output_width) {
                    expected = 0;
                    const size_t row = i / cert.output_width;
                    const uint32_t element = i % cert.output_width;
                    for (uint32_t rank = 0; rank < cert.routes_per_row; ++rank) {
                        const size_t route = row * cert.routes_per_row + rank;
                        const float input = float((p.epoch + b + row * cert.input_width + element % cert.input_width) % 31) * 0.03125f;
                        const auto id = int32_t((p.epoch + b + route / 2) % cert.expert_count);
                        expected += float(1 + rank % 3) * 0.25f * transform(input, id, element);
                    }
                }
                if (outputs[b * sizes.output_values + i] != expected) { reason("private output or scatter mismatch"); return false; }
            }
        }
        ++report.epochs;
        report.accepted += accepted;
        report.rejected += !accepted;
        report.callbacks += cert.boundaries * (options.arm == GGML_CUDA_MOE_FIDELITY_CALLBACK ? 2 : 1);
        report.wait_nodes += options.arm == GGML_CUDA_MOE_FIDELITY_SEGMENTED ? 0 : cert.boundaries;
        return true;
    }

    ~fixture() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            supervisor_stopping = true;
            condition.notify_all();
        }
        if (supervisor.joinable()) { supervisor.join(); }
        stop_worker();
        if (!drain(false)) { GGML_ABORT("moe-fidelity: failed CUDA drain"); }
        for (size_t i = 0; i < graph_count; ++i) {
            if (graphs[i].instance) { CUDA_CHECK(cudaGraphExecDestroy(graphs[i].instance)); }
            if (graphs[i].definition) { CUDA_CHECK(cudaGraphDestroy(graphs[i].definition)); }
        }
        for (uint32_t i = 0; plan_events && i < cert.boundaries; ++i) { if (plan_events[i]) { CUDA_CHECK(cudaEventDestroy(plan_events[i])); } }
        for (uint32_t i = 0; transfer_events && i < cert.boundaries; ++i) { if (transfer_events[i]) { CUDA_CHECK(cudaEventDestroy(transfer_events[i])); } }
        if (stream) { CUDA_CHECK(cudaStreamDestroy(stream)); }
        if (io) { CUDA_CHECK(cudaStreamDestroy(io)); }
        if (device) { CUDA_CHECK(cudaFree(device)); }
        if (host) {
            if (h.window) {
                h.window->~host_window();
                for (uint32_t b = 0; b < cert.boundaries; ++b) { h.controls[b].~host_region(); }
            }
            CUDA_CHECK(cudaFreeHost(host));
        }
    }
};
#endif

static int32_t run(ggml_backend_t backend, const ggml_cuda_moe_fidelity_query_v1 * query,
        const ggml_cuda_moe_fidelity_options_v1 * options, ggml_cuda_moe_fidelity_sample_v1 * samples,
        ggml_cuda_moe_fidelity_report_v1 * report) {
    if (!report || report->struct_size != sizeof(*report) || !options || options->struct_size != sizeof(*options) ||
            !samples || options->arm < GGML_CUDA_MOE_FIDELITY_CALLBACK || options->arm > GGML_CUDA_MOE_FIDELITY_MEMOP ||
            !options->replays || options->replays > 100000 || options->warmup > 10000 ||
            options->cpu_delay_us > 10000 || options->gpu_delay_us > 10000 || options->ready_memop > 1 || options->validate_failures > 1) { return -3; }
    *report = {};
    report->struct_size = sizeof(*report);
    layout sizes;
    if (!measure(query, &report->certificate, &report->storage, &sizes)) { return -3; }
    if (report->storage.pinned_bytes > options->limits.pinned_bytes || report->storage.device_bytes > options->limits.device_bytes ||
            report->storage.scratch_bytes > options->limits.scratch_bytes) { return -2; }
#ifndef GGML_MOE_FIDELITY_CUDA
    GGML_UNUSED(backend);
    snprintf(report->reason, sizeof(report->reason), "native CUDA 12 or newer required");
    return 0;
#else
    if (!backend || !ggml_backend_is_cuda(backend)) { return -3; }
    auto & ctx = *static_cast<ggml_backend_cuda_context *>(backend->context);
    ggml_cuda_set_device(ctx.device);
    const auto caller_started = thread_ns();
    int32_t result = -1;
    try {
        fixture state(report->certificate, *options, *report, sizes);
        if ((options->arm == GGML_CUDA_MOE_FIDELITY_MEMOP || options->ready_memop) && !state.resolve_memops()) {
            state.reason("mapped stream memops unavailable; callback arm remains separate");
            return 0;
        }
        if (!state.allocate()) { return 0; }
        if (!state.capture()) {
            if (!report->reason[0]) { state.reason("capture or upload failed"); }
            return state.capture_unavailable ? 0 : -1;
        }
        state.worker = std::thread([&state] { state.serve(); });
        state.supervisor = std::thread([&state] { state.supervise(); });
        ggml_cuda_moe_fidelity_sample_v1 ignored;
        bool ok = true;
        for (uint32_t i = 0; ok && i < options->warmup; ++i) { ok = state.replay(fault_none, ignored); }
        for (uint32_t i = 0; ok && i < options->replays; ++i) {
            ok = state.replay(fault_none, samples[i]);
            report->samples += ok;
        }
        if (options->validate_failures) {
            for (uint32_t fault = fault_packet_epoch; ok && fault <= fault_teardown; ++fault) {
                ok = state.replay(fault, ignored);
                if (ok && fault != fault_teardown) { ok = state.replay(fault_none, ignored); }
            }
        }
        if (!ok) { state.drain(true); }
        result = ok ? 1 : -1;
        if (ok) { state.reason("passed; synthetic transform only"); }
    } catch (const std::exception & error) {
        snprintf(report->reason, sizeof(report->reason), "fixture exception: %.90s", error.what());
    }
    report->caller_cpu_ns = thread_ns() - caller_started;
    return result;
#endif
}
}

const ggml_cuda_moe_fidelity_fixture_api_v1 * ggml_cuda_moe_fidelity_fixture_api() {
    static const ggml_cuda_moe_fidelity_fixture_api_v1 api = {sizeof(api), 1, measure_api, run};
    return &api;
}
