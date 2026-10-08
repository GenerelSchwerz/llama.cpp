#include "moe-source-core.cuh"
#include "moe-source-state.cuh"
#include "moe-source-tail.cuh"
#include "common.cuh"
#include "moe-cache.cuh"
#include "moe-reference.cuh"
#include "moe-source-ordinary.cuh"
#include "moe-weighted-reduction.cuh"
#include "moe-caller-schedule.h"
#include "fattn.cuh"
#include "argsort.cuh"
#include "mmid.cuh"
#include "top-k.cuh"
#include "softmax.cuh"
#include "staged-input.cuh"
#include "../ggml-moe-source-program.h"
#include "../moe-fidelity-config.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include <cuda/atomic>
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define GGML_MOE_CPU_NVTX
#endif
#endif

namespace {

static uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct source_cpu_range {
    bool enabled;
    uint64_t epoch;
    uint32_t layer;
    source_cpu_range(bool enabled, const char * name, uint64_t epoch, uint32_t layer) : enabled(enabled), epoch(epoch), layer(layer) {
#ifdef GGML_MOE_CPU_NVTX
        if (enabled) { auto event = attributes(name); nvtxRangePushEx(&event); }
#else
        (void) name;
#endif
    }
#ifdef GGML_MOE_CPU_NVTX
    nvtxEventAttributes_t attributes(const char * name) const {
        nvtxEventAttributes_t event = {};
        event.version = NVTX_VERSION; event.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        event.category = layer; event.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
        event.payload.ullValue = epoch; event.messageType = NVTX_MESSAGE_TYPE_ASCII; event.message.ascii = name;
        return event;
    }
#endif
    void mark(const char * name) const {
#ifdef GGML_MOE_CPU_NVTX
        if (enabled) { auto event = attributes(name); nvtxMarkEx(&event); }
#else
        (void) name;
#endif
    }
    void finish() {
#ifdef GGML_MOE_CPU_NVTX
        if (enabled) { nvtxRangePop(); }
#endif
        enabled = false;
    }
    ~source_cpu_range() { finish(); }
};

struct source_host_timer {
    std::atomic<uint64_t> & total;
    uint64_t started;
    explicit source_host_timer(std::atomic<uint64_t> & total, bool enabled = true) : total(total), started(enabled ? now_ns() : 0) {}
    void finish() {
        if (started) { total.fetch_add(now_ns() - started, std::memory_order_relaxed); started = 0; }
    }
    ~source_host_timer() { finish(); }
};

static bool source_certificate_valid(const ggml_graph_execution_certificate & certificate, uint64_t split_uid) {
    if (certificate.magic != GGML_GRAPH_EXECUTION_CERTIFICATE_MAGIC ||
            certificate.abi_version != GGML_GRAPH_EXECUTION_CERTIFICATE_VERSION || certificate.struct_size != sizeof(certificate) ||
            (certificate.flags & ~GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED) != 0 ||
            certificate.domain < GGML_GRAPH_EXECUTION_DOMAIN_MAIN || certificate.domain > GGML_GRAPH_EXECUTION_DOMAIN_MTP ||
            certificate.row_semantics < GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT ||
            certificate.row_semantics > GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SPECULATIVE ||
            !certificate.n_rows || !certificate.n_sequences || !certificate.owner_namespace || !certificate.owner_generation ||
            !certificate.source_graph_uid || !split_uid || certificate.split_graph_uid != split_uid) { return false; }
    for (const auto reserved : certificate.reserved) { if (reserved) { return false; } }
    return true;
}

static bool product(size_t a, size_t b, size_t & result) {
    if (a && b > SIZE_MAX / a) { return false; }
    result = a * b;
    return true;
}

static bool reserve(size_t & total, size_t bytes, size_t & offset) {
    if (total > SIZE_MAX - 255) { return false; }
    offset = (total + 255) & ~size_t(255);
    if (bytes > SIZE_MAX - offset) { return false; }
    total = offset + bytes;
    return true;
}

using source_flag = ggml_cuda_moe_source_flag;
using source_control = ggml_cuda_moe_source_control;
using source_runtime = ggml_cuda_moe_source_runtime;

struct source_copy_job { const uint8_t * source; uint8_t * destination; uint64_t bytes; };
struct source_copy_plan { uint32_t kernel = 0, jobs = 0; uint64_t bytes = 0; };

struct source_group_view {
    uint64_t * gate = nullptr, * up = nullptr, * down = nullptr;
    int32_t * starts = nullptr, * tokens = nullptr, * destinations = nullptr, * count = nullptr;
};

static size_t group_bytes(uint32_t capacity) {
    return (size_t(capacity) * 3 * sizeof(uint64_t) + (size_t(capacity) * 3 + 2) * sizeof(int32_t) + 7) & ~size_t(7);
}

static source_copy_plan * copy_plan(uint8_t * data, uint32_t capacity) {
    return reinterpret_cast<source_copy_plan *>(data + 2 * group_bytes(capacity));
}

static source_copy_job * copy_jobs(source_copy_plan * plan) { return reinterpret_cast<source_copy_job *>(plan + 1); }

static source_group_view group_view(void * data, uint32_t capacity) {
    source_group_view v;
    v.gate = static_cast<uint64_t *>(data);
    v.up = v.gate + capacity;
    v.down = v.up + capacity;
    v.starts = reinterpret_cast<int32_t *>(v.down + capacity);
    v.tokens = v.starts + capacity + 1;
    v.destinations = v.tokens + capacity;
    v.count = v.destinations + capacity;
    return v;
}

#if defined(USE_CUDA_GRAPH) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 12030
#define GGML_MOE_SOURCE_GRAPH

static __device__ uint32_t acquire(const source_flag * flag) {
    auto * value = reinterpret_cast<uint32_t *>(const_cast<std::atomic<uint32_t> *>(&flag->value));
    return cuda::atomic_ref<uint32_t, cuda::thread_scope_system>(*value).load(cuda::memory_order_acquire);
}

static __device__ uint32_t observe(const source_flag * flag) {
    auto * value = reinterpret_cast<uint32_t *>(const_cast<std::atomic<uint32_t> *>(&flag->value));
    return cuda::atomic_ref<uint32_t, cuda::thread_scope_system>(*value).load(cuda::memory_order_relaxed);
}

static __device__ void release(source_flag * flag) {
    auto * value = reinterpret_cast<uint32_t *>(&flag->value);
    cuda::atomic_ref<uint32_t, cuda::thread_scope_system>(*value).store(1, cuda::memory_order_release);
}

static __global__ void decide_phase(source_runtime * runtime, const source_control * control, cudaGraphConditionalHandle handle, bool initialize) {
    const bool stopped = control && acquire(&control->stop);
    if (initialize) {
        *runtime = {};
        runtime->epoch = control->epoch;
    }
    if (stopped) { runtime->failed = 1; }
    cudaGraphSetConditional(handle, runtime->failed == 0);
}

static __global__ void publish_ready(source_control * control, const float * input, size_t values,
        const int32_t * ids, size_t ids_stride, uint32_t routes_per_row, size_t routes,
        float * host_input, int32_t * host_ids, size_t host_ids_stride) {
    for (size_t i = threadIdx.x; i < values; i += blockDim.x) { host_input[i] = input[i]; }
    for (size_t i = threadIdx.x; i < routes; i += blockDim.x) {
        const auto * row = reinterpret_cast<const int32_t *>(reinterpret_cast<const uint8_t *>(ids) + (i / routes_per_row) * ids_stride);
        auto * destination = reinterpret_cast<int32_t *>(reinterpret_cast<uint8_t *>(host_ids) + (i / routes_per_row) * host_ids_stride);
        destination[i % routes_per_row] = row[i % routes_per_row];
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) { release(&control->ready); }
}

static __global__ void publish_overlap(source_flag * flag, const source_runtime * runtime) {
    if (!runtime->failed) { release(flag); }
}

static __global__ void copy_input(uint8_t * destination, const uint8_t * input, size_t bytes,
        uint32_t * flag, const source_flag * stop, source_runtime * runtime) {
    if (threadIdx.x == 0) {
        auto ready = cuda::atomic_ref<uint32_t, cuda::thread_scope_system>(*flag);
        while (!ready.load(cuda::memory_order_acquire) && !acquire(stop)) {
#if __CUDA_ARCH__ >= 700
            __nanosleep(100);
#endif
        }
        if (acquire(stop)) { runtime->failed = 1; }
    }
    __syncthreads();
    if (runtime->failed) { return; }
    for (size_t i = threadIdx.x; i < bytes; i += blockDim.x) {
        destination[i] = input[i];
    }
    __syncthreads();
    // One block finishes every input read before the host can reuse the buffer.
    if (threadIdx.x == 0) { cuda::atomic_ref<uint32_t, cuda::thread_scope_system>(*flag).store(0, cuda::memory_order_release); }
}

static __global__ void wait_phase(source_control * control, source_runtime * runtime, uint32_t flag) {
    auto * target = flag == 0 ? &control->plan : flag == 1 ? &control->copied : &control->cpu;
    // Consume producer data only after the outer acquire succeeds.
    while (!acquire(target)) {
        while (!observe(target)) {
            if (observe(&control->stop) && acquire(&control->stop)) { runtime->failed = 1; return; }
#if __CUDA_ARCH__ >= 700
            __nanosleep(100);
#endif
        }
    }
    if (acquire(&control->stop)) { runtime->failed = 1; return; }
    if ((flag == 0 && control->epoch != runtime->epoch) || (flag == 2 && control->status)) { runtime->failed = 1; }
}

static __global__ void import_plan(uint8_t * destination, const uint8_t * source, size_t bytes,
        const source_control * control, const source_runtime * runtime) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < bytes; i += size_t(gridDim.x) * blockDim.x) {
        destination[i] = runtime->failed || control->epoch != runtime->epoch ? 0 : source[i];
    }
}

static __global__ void import_cpu(float * output, const float * input, size_t values,
        source_runtime * runtime, const int32_t * resident, const int32_t * selected,
        source_control * next_control, cudaGraphConditionalHandle next_handle) {
    if (runtime->failed) { return; }
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < values; i += size_t(gridDim.x) * blockDim.x) {
        output[i] += input[i];
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        ++runtime->joined;
        runtime->resident += *resident > 0;
        runtime->selected += *selected > 0;
        if (next_handle) { cudaGraphSetConditional(next_handle, !next_control || !acquire(&next_control->stop)); }
    }
}

static __global__ void import_routed_cpu(float * output, const float * input, size_t values,
        uint32_t width, uint32_t routes_per_row, size_t output_column, size_t output_row,
        source_runtime * runtime, const int32_t * resident, const int32_t * selected,
        source_control * next_control, cudaGraphConditionalHandle next_handle) {
    if (runtime->failed) { return; }
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < values; i += size_t(gridDim.x) * blockDim.x) {
        const size_t route = i / width, column = i % width;
        output[(route / routes_per_row) * output_row + (route % routes_per_row) * output_column + column] += input[route * output_column + column];
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        ++runtime->joined;
        runtime->resident += *resident > 0;
        runtime->selected += *selected > 0;
        if (next_handle) { cudaGraphSetConditional(next_handle, !next_control || !acquire(&next_control->stop)); }
    }
}

static __global__ void guard_count(int32_t * count, const source_runtime * runtime) { if (runtime->failed) { *count = 0; } }
static __global__ void copy_selected(const source_copy_plan * plan, source_runtime * runtime, int32_t * count) {
    if (runtime->failed) {
        if (blockIdx.x == 0 && threadIdx.x == 0) { *count = 0; }
        return;
    }
    if (!plan->kernel || !plan->jobs) { return; }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        ++runtime->kernel_layers;
        runtime->copy_jobs += plan->jobs;
        runtime->copy_bytes += plan->bytes;
    }
    const auto * jobs = reinterpret_cast<const source_copy_job *>(plan + 1);
    const uint32_t groups = min(plan->jobs, gridDim.x);
    const uint32_t group = blockIdx.x % groups;
    const size_t first = size_t(blockIdx.x / groups) * blockDim.x + threadIdx.x;
    const size_t stride = size_t((gridDim.x - 1 - group) / groups + 1) * blockDim.x;
    for (uint32_t j = group; j < plan->jobs; j += groups) {
        const auto job = jobs[j];
        if (!(reinterpret_cast<uintptr_t>(job.source) % 16) && !(reinterpret_cast<uintptr_t>(job.destination) % 16)) {
            const auto * input = reinterpret_cast<const uint4 *>(job.source);
            auto * output = reinterpret_cast<uint4 *>(job.destination);
            for (size_t v = first; v < job.bytes / 16; v += stride) { output[v] = input[v]; }
            for (size_t b = job.bytes / 16 * 16 + first; b < job.bytes; b += stride) { job.destination[b] = job.source[b]; }
        } else {
            for (size_t b = first; b < job.bytes; b += stride) { job.destination[b] = job.source[b]; }
        }
    }
}

template<class F>
static bool capture(cudaGraph_t graph, cudaStream_t stream, std::vector<cudaGraphNode_t> & dependencies, F emit) {
    if (cudaStreamBeginCaptureToGraph(stream, graph, dependencies.data(), nullptr, dependencies.size(),
            cudaStreamCaptureModeRelaxed) != cudaSuccess) { return false; }
    cudaError_t status = cudaSuccess;
    std::optional<ggml_cuda_cublas_request> request;
    try {
        if (!emit()) { status = cudaErrorInvalidValue; }
        if (status == cudaSuccess) {
            cudaStreamCaptureStatus active;
            const cudaGraphNode_t * frontier = nullptr;
            size_t count = 0;
#if CUDART_VERSION >= 13000
            status = cudaStreamGetCaptureInfo(stream, &active, nullptr, nullptr, &frontier, nullptr, &count);
#else
            status = cudaStreamGetCaptureInfo(stream, &active, nullptr, nullptr, &frontier, &count);
#endif
            if (status == cudaSuccess) {
                dependencies.clear();
                if (count) { dependencies.assign(frontier, frontier + count); }
            }
        }
    } catch (const ggml_cuda_cublas_request & resource) { request = resource; status = cudaErrorInvalidValue; } catch (...) { status = cudaErrorInvalidValue; }
    cudaGraph_t result = nullptr;
    const auto ended = cudaStreamEndCapture(stream, &result);
    if (request && ended == cudaSuccess && result == graph) { throw *request; }
    return status == cudaSuccess && ended == cudaSuccess && result == graph;
}

template<class F>
static bool phase(cudaGraph_t graph, cudaStream_t stream, std::vector<cudaGraphNode_t> & dependencies,
        source_runtime * runtime, source_control * control, F emit, cudaGraphConditionalHandle handle = 0, bool initialize = false) {
    if (initialize && !control) { return false; }
    if (!handle && (cudaGraphConditionalHandleCreate(&handle, graph, 0, cudaGraphCondAssignDefault) != cudaSuccess ||
            !capture(graph, stream, dependencies, [&] {
                decide_phase<<<1, 1, 0, stream>>>(runtime, control, handle, initialize);
                return cudaGetLastError() == cudaSuccess;
            }))) { return false; }
    cudaGraphNodeParams params = {};
    params.type = cudaGraphNodeTypeConditional;
    params.conditional.handle = handle;
    params.conditional.type = cudaGraphCondTypeIf;
    params.conditional.size = 1;
    cudaGraphNode_t node = nullptr;
#if CUDART_VERSION >= 13000
    const auto status = cudaGraphAddNode(&node, graph, dependencies.data(), nullptr, dependencies.size(), &params);
#else
    const auto status = cudaGraphAddNode(&node, graph, dependencies.data(), dependencies.size(), &params);
#endif
    if (status != cudaSuccess) { return false; }
    std::vector<cudaGraphNode_t> body_dependencies;
    if (!capture(params.conditional.phGraph_out[0], stream, body_dependencies, emit)) { return false; }
    dependencies.assign(1, node);
    return true;
}
#endif

struct prepared_operation {
    ggml_moe_source_operation operation;
    uint32_t consumed = 1;
    uint32_t source_kind = 0;
    ggml_cuda_moe_source_binding source_binding;
    ggml_cuda_moe_weighted_reduction_match weighted;
    ggml_cuda_moe_source_tensor source_views[7];
    const ggml_tensor * source_tensors[7] = {};
    ggml_cuda_moe_fidelity_mul_mat_resources matmul;
    bool matmul_estimated = false;
    ggml_cuda_fattn_resources attention;
    ggml_cuda_source_sort_resources sorter = {};
    ggml_cuda_source_softmax_resources softmax = {};
    ggml_cuda_source_staged_input_view staged;
};

struct core_region {
    void * owner = nullptr;
    ggml_moe_source_expert expert;
    const ggml_backend_moe_cpu_region_service_api_v1 * cpu_api = nullptr;
    ggml_backend_moe_cpu_service_v1_t cpu_service = nullptr;
};

struct core_layer {
    const ggml_moe_source_layer * descriptor = nullptr;
    ggml_cuda_moe_graph_group_dispatch * owner_group = nullptr;
    uint32_t probe_domain = 0;
    std::vector<ggml_cuda_moe_grouped_bank_descriptor> banks;
    std::vector<uint32_t> bank_indices;
    std::vector<uint32_t> transport_indices;
    std::vector<size_t> selected_offsets;
    std::vector<int32_t> distinct, slots, classes, selected;
    std::vector<uint32_t> weights, counts;
    std::vector<int32_t> cpu_ids;
    std::vector<uint32_t> cpu_rows, cpu_destinations;
    std::vector<prepared_operation> prelude, overlap;
    size_t overlap_probe_offset = SIZE_MAX;
    size_t control_offset = 0, input_offset = 0, ids_offset = 0, cpu_offset = 0;
    size_t map_offset = 0, owners_offset = 0, host_plan_offset = 0, device_plan_offset = 0;
    size_t plan_bytes = 0, selected_bytes = 0;
    size_t gpu_mask_offset = SIZE_MAX;
    size_t input_bytes = 0, output_bytes = 0, ids_stride = 0;
    size_t source_image_offset = 0, ownership_offset = 0;
    struct body_operation {
        ggml_tensor * tensor = nullptr;
        size_t bank = SIZE_MAX;
    };
    std::vector<body_operation> body;
    ggml_tensor routed_output = {};
    ggml_cuda_mmid_resources routed_resources;
    std::vector<uint8_t> cpu_ownership;
    uint32_t cpu_count = 0, resident_count = 0, selected_count = 0;
    uint32_t branch_expected = 0, branch_completed = 0;
    bool copy_submitted = false, kernel_copy = false;
    cudaEvent_t ready = nullptr, copied = nullptr, cpu_copied = nullptr;
    cudaEvent_t plan_ready = nullptr;
    uint64_t epoch = 0;
    uint64_t residency_token = 0;
};

struct core_source_statistics {
    uint32_t domain = 0;
    uint64_t observations = 0;
    std::vector<uint64_t> counts;
    std::vector<double> scores;
};

struct core_profile_group {
    uint32_t representative = 0;
    const std::vector<int32_t> * seed = nullptr;
    const ggml_tensor * source = nullptr;
    std::vector<uint32_t> members, bindings, observations;
    std::vector<int32_t> routes;
    size_t payload_bytes = 0;
    uint32_t domain = 0;
    std::vector<ggml_moe_profile_bank_statistics> statistics;
    std::vector<int32_t> derived_seed;
};

struct core_session {
    ggml_backend_moe_hybrid_config_v1 config = {};
    ggml_backend_moe_source_owner_v1 source_owner = {};
    ggml_backend_cuda_context * parent = nullptr;
    const ggml_backend_moe_cpu_region_service_api_v1 * cpu_api = nullptr;
    ggml_backend_moe_cpu_service_v1_t cpu_service = nullptr;
    std::unique_ptr<ggml_backend_cuda_context> compute;
    std::unique_ptr<ggml_moe_source_program> program;
    ggml_context * fusion_context = nullptr;
    ggml_cgraph * fusion_graph = nullptr;
    bool fusion_enabled = true;
    uint64_t fused_nodes = 0, image_groups = 0, emitted_images = 0;
    std::vector<core_layer> layers;
    std::unordered_map<const ggml_tensor *, std::vector<int32_t>> profiles;
    std::unordered_map<const ggml_tensor *, std::vector<core_source_statistics>> statistics;
    std::vector<core_profile_group> profile_groups;
    std::vector<ggml_cuda_moe_source_profile_update> profile_updates;
    std::vector<prepared_operation> epilogue;
    std::vector<core_region *> compute_regions;
    std::shared_ptr<ggml_cuda_moe_graph_plan> owner_plan;
    ggml_cuda_moe_graph_execution owner_execution;
    std::vector<std::shared_ptr<void>> resource_leases;
    std::vector<std::shared_ptr<void>> leaf_resource_leases;
    uint64_t resource_identity = 0;
    ggml_cuda_moe_source_transport * transport = nullptr;
    ggml_backend_buffer_t arena = nullptr;
    ggml_backend_buffer_t pool_arena = nullptr;
    size_t pool_arena_bytes = 0;
    uint64_t resource_recaptures = 0;
    std::vector<ggml_backend_buffer_ptr> library_workspaces;
    size_t library_bytes = 0;
    uint64_t library_preparations = 0;
    uint8_t * host = nullptr, * alias = nullptr, * cancel_alias = nullptr;
    size_t device_bytes = 0, pinned_bytes = 0;
    size_t runtime_offset = 0, host_runtime_offset = 0;
    size_t scratch_offset = 0, quant_offset = 0, selected_offset = 0, cpu_device_offset = 0;
    size_t pool_offset = 0, pool_bytes = 0, image_bytes = 0, cublas_offset = 0, cublas_bytes = 0;
    cudaStream_t stream = nullptr, io = nullptr;
    std::vector<cudaGraph_t> graphs;
    std::vector<cudaGraphExec_t> executables;
    size_t publication_graph = SIZE_MAX;
    uint64_t publication_captures = 0, publication_launches = 0;
    std::vector<std::pair<const ggml_tensor *, void *>> publication_targets;
    bool publication_packet = true;
    std::mutex mutex;
    std::timed_mutex terminal_mutex;
    std::condition_variable condition;
    std::thread supervisor;
    std::thread::id caller_id;
    bool supervisor_needs_wake = false;
    bool caller_active = false, closed = false, stopping = false, dispatch = false, resources_live = false;
    bool prepared = false, segmented = false, noalias = false, effect_failure = false;
    bool overlap_enabled = false, overlap_probe = false;
    bool weighted_enabled = false;
    bool kernel_copy_enabled = true;
    bool device_tail = true;
    size_t tail_offset = 0, host_tail_offset = 0;
    bool replay_diagnostic = false;
    bool metadata_diagnostic = false;
    bool retain_routed_outputs = false;
    bool cpu_profile = false;
    uint64_t weighted_matches = 0, weighted_complete = 0, weighted_closed = 0, weighted_ready = 0, weighted_aliases = 0;
    bool unsupported = false, normal_fallback = false, rejected_before_effects = false;
    ggml_cuda_moe_source_cpu_probe cpu_probe(uint32_t i) const {
        const auto & layer = layers[i];
        const auto & expert = *layer.descriptor->expert;
        const auto & source = expert.banks[0].source;
        return {i, &layer.routed_output, layer.descriptor->activation, source.tensor, source.data,
            host + layer.input_offset, layer.input_bytes, layer.cpu_ids.data(), layer.cpu_ownership.data(),
            reinterpret_cast<const float *>(host + layer.cpu_offset), layer.output_bytes, expert.region.geometry.route_capacity};
    }
    ggml_cuda_moe_source_gpu_probe gpu_probe(uint32_t i) const {
        const auto & layer = layers[i];
        const auto & expert = *layer.descriptor->expert;
        const auto & source = expert.banks[0].source;
        return {i, uint32_t(layers.size()), &layer.routed_output, layer.descriptor->activation, source.tensor, source.data,
            host + layer.input_offset, layer.input_bytes, layer.weights.data(), layer.distinct.data(), layer.classes.data(),
            uint32_t(layer.distinct.size()), expert.region.geometry.route_capacity,
            static_cast<const ggml_tensor *>(source.witness), layer.probe_domain};
    }
    bool admit_caller() {
        std::lock_guard<std::mutex> lock(mutex);
        rejected_before_effects = false;
        if (closed || effect_failure || caller_active || external_drains) {
            fprintf(stderr, "moe-source-core-fault: stage=caller_admission closed=%d poisoned=%d caller_active=%d external_drains=%u\n",
                int(closed), int(effect_failure), int(caller_active), external_drains);
            return false;
        }
        caller_active = true;
        caller_id = std::this_thread::get_id();
        canceled.store(false, std::memory_order_release);
        return true;
    }
    bool metadata_valid(const ggml_cgraph * graph, const ggml_graph_execution_certificate & certificate,
                        void * const * regions, uint32_t count, bool check_program = true) const;
    ggml_status reject_before_effects() {
        std::lock_guard<std::mutex> lock(mutex);
        rejected_before_effects = !resources_live && !dispatch && !closed && !effect_failure;
        return GGML_STATUS_FAILED;
    }
    void leave_caller() {
        snapshot_transport();
        std::lock_guard<std::mutex> lock(mutex);
        caller_active = false; caller_id = {}; deadline = 0;
        if (external_drains || supervisor_needs_wake || canceled.load()) { condition.notify_all(); }
    }
    uint32_t external_drains = 0, regions = 0;
    uint64_t deadline = 0, epoch = 0, cpu_epoch = 0;
    std::atomic<bool> canceled{false};
    std::atomic<bool> controls_ready{false};
    ggml_backend_moe_hybrid_state_v1 counters = {};
    ggml_backend_moe_hybrid_test_hook_v1_t hook = nullptr;
    void * hook_data = nullptr;
    uint64_t captures = 0, cpu_publications = 0, copy_completions = 0, gpu_completions = 0, publications = 0;
    uint64_t scatter_completions = 0, failures = 0, drains = 0, drain_failures = 0, map_bytes = 0;
    uint64_t gpu_expected = 0, copy_expected = 0, program_instance = 0;
    uint64_t kernel_jobs_expected = 0, kernel_jobs_completed = 0, kernel_bytes_expected = 0, kernel_bytes_completed = 0;
    std::atomic<uint64_t> prepare_wall_ns{0}, replay_wall_ns{0}, replay_acquire_map_wall_ns{0};
    std::atomic<uint64_t> internal_drain_wall_ns{0}, public_copy_wait_wall_ns{0}, external_drain_wall_ns{0};
    mutable std::atomic<uint64_t> summary_wall_ns{0}, metadata_validation_wall_ns{0};
    std::atomic<uint64_t> preflight_wall_ns{0}, binding_validation_wall_ns{0};
    std::atomic<uint64_t> compact_plan_wall_ns{0}, selected_copy_submit_wall_ns{0};
    ggml_cuda_moe_source_transport_stats transport_stats;
    void snapshot_transport() {
        ggml_cuda_moe_source_transport_stats current;
        if (transport && owner().source_transport_stats(transport, &current)) {
            transport_stats = current;
        }
    }

    uint8_t * data() const { return arena ? static_cast<uint8_t *>(ggml_backend_buffer_get_base(arena)) : nullptr; }
    source_runtime * runtime() const { return reinterpret_cast<source_runtime *>(data() + runtime_offset); }
    source_runtime * result() const { return reinterpret_cast<source_runtime *>(host + host_runtime_offset); }
    source_control * control(uint32_t i, bool mapped = false) const {
        return reinterpret_cast<source_control *>((mapped ? alias : host) + layers[i].control_offset);
    }
    ggml_cuda_moe_grouped_context & owner() const { return *parent->moe_grouped_context; }

    bool wait_stream(cudaStream_t target, uint64_t expiry) const {
        if (!target) { return true; }
        for (;;) {
            const auto status = cudaStreamQuery(target);
            if (status == cudaSuccess) { return true; }
            if (status != cudaErrorNotReady || now_ns() >= expiry) { return false; }
            std::this_thread::yield();
        }
    }

    bool stop() {
        canceled.store(true, std::memory_order_release);
        if (controls_ready.load(std::memory_order_acquire)) { for (uint32_t i = 0; i < layers.size(); ++i) { control(i)->stop.store(1); } }
        const bool ok = !cpu_api || !cpu_service || !cpu_epoch || cpu_api->cancel(cpu_service, cpu_epoch) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
        condition.notify_all();
        return ok;
    }

    void supervise() {
        std::unique_lock<std::mutex> lock(mutex);
        while (!stopping) {
            if (!caller_active || !deadline) {
                supervisor_needs_wake = true;
                condition.wait(lock);
                supervisor_needs_wake = false;
                continue;
            }
            const auto expiry = std::chrono::steady_clock::time_point(std::chrono::nanoseconds(deadline));
            // An earlier alarm also covers a later replay deadline.
            if (!condition.wait_until(lock, expiry, [&] { return stopping || canceled.load(); }) &&
                    caller_active && deadline && now_ns() >= deadline) { stop(); }
            if (canceled.load() && caller_active) {
                supervisor_needs_wake = true;
                condition.wait(lock, [&] { return stopping || !caller_active || !canceled.load(); });
                supervisor_needs_wake = false;
            }
        }
    }

    bool prepare_operations(const ggml_cgraph * graph, const std::vector<ggml_moe_source_operation> & input,
        std::vector<prepared_operation> & output);
    bool validate_operations(const std::vector<prepared_operation> & operations);
    bool emit_operations(const std::vector<prepared_operation> & operations);
    bool acquire_owner(const ggml_cgraph * graph, bool first);
    bool prepare_profiles();
    bool apply_profiles(bool first, bool maintenance = false);
    bool finish_owner();
    bool allocate(const ggml_cgraph * graph, const std::vector<core_region *> & regions);
    bool capture_program();
    bool prepare_capture_resources();
    bool execute_experts(uint32_t i, uint32_t kind);
    bool build_plan(uint32_t i);
    bool serve(uint32_t i);
    bool launch(uint32_t segment);
    bool launch_graph(size_t index) {
        if (index >= executables.size() || cudaGraphLaunch(executables[index], stream) != cudaSuccess) { return false; }
        if (index == publication_graph) { ++publication_launches; }
        else { ++counters.window_launches; }
        return index == publication_graph || !hook || hook(hook_data, GGML_BACKEND_MOE_HYBRID_TEST_SOURCE_GRAPH_LAUNCHED, epoch, stream);
    }
    bool ready(uint32_t i) const { return control(i)->ready.load() != 0; }
    ggml_moe_caller_progress progress() const;
    int32_t drain(bool internal = false);
    int32_t dispose(bool retire = false);
    ggml_status replay(ggml_cgraph * graph, void * const * regions, uint32_t count);
    void summary(const char * kind = "replay", int32_t drain_status = 0) const;
};

static ggml_cuda_moe_source_tensor tensor_view(const ggml_tensor * tensor, int device) {
    ggml_cuda_moe_source_tensor result;
    if (!tensor) { return result; }
    result.type = tensor->type;
    std::copy(std::begin(tensor->ne), std::end(tensor->ne), result.ne);
    std::copy(std::begin(tensor->nb), std::end(tensor->nb), result.nb);
    result.data = tensor->data;
    result.device = device;
    if (tensor->buffer) {
        result.buffer_base = ggml_backend_buffer_get_base(tensor->buffer);
        result.buffer_bytes = ggml_backend_buffer_get_size(tensor->buffer);
        result.buffer_identity = reinterpret_cast<uintptr_t>(tensor->buffer);
    }
    return result;
}

static bool tensor_storage(const ggml_tensor * tensor, int device, bool & unavailable) {
    if (!tensor || ggml_is_empty(tensor)) { return tensor != nullptr; }
    if (!tensor->buffer || !tensor->data) { return false; }
    const uintptr_t base = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(tensor->buffer));
    const uintptr_t pointer = reinterpret_cast<uintptr_t>(tensor->data);
    const size_t capacity = ggml_backend_buffer_get_size(tensor->buffer), bytes = ggml_nbytes(tensor);
    if (!base || pointer < base || pointer - base > capacity || bytes > capacity - (pointer - base)) { return false; }
    cudaPointerAttributes attributes = {};
    if (cudaPointerGetAttributes(&attributes, tensor->data) != cudaSuccess) { (void) cudaGetLastError(); unavailable = true; return false; }
    const bool supported = attributes.type == cudaMemoryTypeManaged ||
        (attributes.type == cudaMemoryTypeDevice && attributes.device == device) ||
        (attributes.type == cudaMemoryTypeHost && attributes.devicePointer == tensor->data);
    unavailable |= !supported;
    return supported;
}

static bool weighted_binding_matches(const prepared_operation & operation, int device) {
    const uint32_t count = operation.weighted.expert_scale ? 4 : 3;
    for (uint32_t i = 0; i < count; ++i) {
        const auto current = tensor_view(operation.source_tensors[i], device);
        const auto & bound = operation.source_views[i];
        if (current.type != bound.type || current.data != bound.data || current.device != bound.device ||
                current.buffer_base != bound.buffer_base || current.buffer_bytes != bound.buffer_bytes ||
                current.buffer_identity != bound.buffer_identity || memcmp(current.ne, bound.ne, sizeof(bound.ne)) ||
                memcmp(current.nb, bound.nb, sizeof(bound.nb))) { return false; }
    }
    return true;
}

static bool weighted_binding_prepare(prepared_operation & operation, int device, bool & alias) {
    const auto & match = operation.weighted;
    const ggml_tensor * tensors[] = {match.experts, match.weights, match.dst, match.expert_scale};
    const uint32_t count = match.expert_scale ? 4 : 3;
    size_t bytes[4] = {};
    for (uint32_t i = 0; i < count; ++i) {
        const auto * tensor = tensors[i];
        if (!tensor || tensor->type != GGML_TYPE_F32) { return false; }
        size_t contiguous = sizeof(float);
        for (int k = 0; k < 4; ++k) {
            if (tensor->ne[k] <= 0 || tensor->nb[k] != contiguous ||
                    uint64_t(tensor->ne[k]) > SIZE_MAX / contiguous) { return false; }
            contiguous *= size_t(tensor->ne[k]);
        }
        bool unavailable = false;
        if (!tensor_storage(tensor, device, unavailable)) { return false; }
        const auto pointer = reinterpret_cast<uintptr_t>(tensor->data);
        if (pointer % alignof(float) || contiguous > UINTPTR_MAX - pointer) { return false; }
        bytes[i] = contiguous;
        operation.source_tensors[i] = tensor;
        operation.source_views[i] = tensor_view(tensor, device);
    }
    const auto * experts = match.experts;
    if (experts->ne[0] > INT64_MAX - 255 || experts->ne[1] > INT_MAX ||
            uint64_t(experts->ne[2]) > INT64_MAX / uint64_t(experts->ne[3])) { return false; }
    const int64_t rows = experts->ne[2] * experts->ne[3];
    for (uint32_t i : {uint32_t(1), uint32_t(3)}) {
        if (i >= count) { continue; }
        if (tensors[i]->ne[0] != 1 || tensors[i]->ne[1] != experts->ne[1] ||
                tensors[i]->ne[2] != experts->ne[2] || tensors[i]->ne[3] != experts->ne[3]) { return false; }
    }
    if (match.dst->ne[0] != experts->ne[0] || match.dst->ne[1] != rows ||
            match.dst->ne[2] != 1 || match.dst->ne[3] != 1) { return false; }
    const auto output = reinterpret_cast<uintptr_t>(match.dst->data);
    // The expert output is a synthetic NONE node; it still needs a range check.
    for (uint32_t i = 0; i < count; ++i) {
        if (i == 2) { continue; }
        const auto input = reinterpret_cast<uintptr_t>(tensors[i]->data);
        if (output < input + bytes[i] && input < output + bytes[2]) { alias = true; return false; }
    }
    cudaDeviceProp properties;
    if (cudaGetDeviceProperties(&properties, device) != cudaSuccess || properties.maxThreadsPerBlock < 256 ||
            rows > properties.maxGridSize[0] || (experts->ne[0] + 255) / 256 > properties.maxGridSize[1]) { return false; }
    return true;
}

bool core_session::prepare_operations(const ggml_cgraph * graph, const std::vector<ggml_moe_source_operation> & input,
        std::vector<prepared_operation> & output) {
    output.reserve(input.size());
    for (size_t i = 0; i < input.size();) {
        prepared_operation prepared;
        prepared.operation = input[i];
        auto * node = input[i].tensor;
        if (input[i].effect == ggml_moe_source_effect::metadata) { output.push_back(std::move(prepared)); ++i; continue; }
        const auto rejected = [&](const char * stage, const ggml_tensor * tensor) {
            if (replay_diagnostic) {
                fprintf(stderr, "moe-source-core-preparation-rejected: stage=%s index=%llu name=%s op=%s type=%s ne=%lld,%lld,%lld,%lld\n",
                    stage, (unsigned long long) input[i].original_index, tensor->name, ggml_op_name(tensor->op), ggml_type_name(tensor->type),
                    (long long) tensor->ne[0], (long long) tensor->ne[1], (long long) tensor->ne[2], (long long) tensor->ne[3]);
            }
            return false;
        };
        if (!tensor_storage(node, parent->device, unsupported)) { return rejected("output_storage", node); }
        for (const auto * source : node->src) {
            if (source && !tensor_storage(source, parent->device, unsupported)) { return rejected("input_storage", source); }
        }
        if (!ggml_backend_dev_supports_op(ggml_backend_get_device(config.backend), node)) { unsupported = true; return rejected("backend_operation", node); }
        if (weighted_enabled && node->op == GGML_OP_MUL && input[i].original_index < uint64_t(graph->n_nodes) &&
                graph->use_counts && graph->visited_hash_set.size && graph->visited_hash_set.used && graph->visited_hash_set.keys &&
                node->ne[1] > 0 && node->ne[1] <= INT_MAX && node->ne[2] > 0 && node->ne[3] > 0 &&
                uint64_t(node->ne[2]) <= INT64_MAX / uint64_t(node->ne[3])) {
            ggml_cuda_moe_weighted_reduction_match match;
            if (ggml_cuda_match_moe_weighted_reduction(graph, int(input[i].original_index), match)) {
                ++weighted_matches;
                bool complete = match.node_count > 0 && size_t(match.node_count) <= input.size() - i;
                std::vector<const ggml_tensor *> cut;
                for (int k = 0; complete && k < match.node_count; ++k) {
                    const auto & operation = input[i + k];
                    complete = operation.original_index == input[i].original_index + k &&
                        operation.original == graph->nodes[operation.original_index] &&
                        operation.effect != ggml_moe_source_effect::owner_write;
                    cut.push_back(operation.tensor);
                }
                if (complete) {
                    ++weighted_complete;
                    prepared.weighted = {program->find(match.experts), program->find(match.expert_scale),
                        program->find(match.weights), program->find(match.dst), match.node_count};
                    const ggml_tensor * retained[] = {prepared.weighted.dst};
                    if ((!match.expert_scale || prepared.weighted.expert_scale) &&
                            program->closed_cut(cut.data(), cut.size(), retained, 1)) {
                        ++weighted_closed;
                        bool alias = false;
                        if (weighted_binding_prepare(prepared, parent->device, alias)) {
                            prepared.source_kind = 4; prepared.consumed = uint32_t(match.node_count); ++weighted_ready;
                        }
                        weighted_aliases += alias;
                    }
                }
            }
        }
        if (node->op == GGML_OP_CUSTOM) {
            if (!ggml_cuda_staged_input_prepare_source_view(parent->device, node, prepared.staged)) { unsupported = true; return rejected("staged_input", node); }
            prepared.source_kind = 3;
        }
        const size_t unit = ggml_moe_source_operation_group_size(input, i);
        if (!fusion_graph && !prepared.source_kind && unit == 2 && node->op == GGML_OP_RMS_NORM) {
            auto * multiply = input[i + 1].tensor;
            const ggml_tensor * cut[] = {node, multiply};
            const ggml_tensor * retained[] = {multiply};
            if (multiply->op == GGML_OP_MUL && multiply->src[0] == node && node->src[0] && multiply->src[1] &&
                    program->closed_cut(cut, 2, retained, 1)) {
                prepared.source_views[0] = tensor_view(node->src[0], parent->device);
                prepared.source_views[1] = tensor_view(multiply->src[1], parent->device);
                prepared.source_views[2] = tensor_view(multiply, parent->device);
                prepared.source_tensors[0] = node->src[0]; prepared.source_tensors[1] = multiply->src[1]; prepared.source_tensors[2] = multiply;
                float epsilon;
                memcpy(&epsilon, node->op_params, sizeof(epsilon));
                const auto status = ggml_cuda_moe_source_rms_binding_prepare(prepared.source_views, epsilon, prepared.source_binding);
                if (status == GGML_CUDA_MOE_SOURCE_NORM_INVALID) { return false; }
                if (status == GGML_CUDA_MOE_SOURCE_NORM_READY) { prepared.source_kind = 1; prepared.consumed = 2; }
            }
        }
        if (!prepared.source_kind && unit == 9) {
            ggml_tensor * n[9];
            const ggml_op expected[] = {GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_MUL,
                GGML_OP_RESHAPE, GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_UNARY};
            bool match = true;
            for (int k = 0; k < 9; ++k) { n[k] = input[i + k].tensor; match &= n[k]->op == expected[k]; }
            if (match && n[1]->src[0] == n[0] && n[3]->src[0] == n[2] && n[5]->src[0] == n[4] &&
                    n[7]->src[0] == n[6] && n[8]->src[0] == n[7] && n[0]->src[1] == n[6]->src[1] &&
                    ggml_get_unary_op(n[3]) == GGML_UNARY_OP_SOFTPLUS && ggml_get_unary_op(n[8]) == GGML_UNARY_OP_SIGMOID &&
                    (n[2]->src[0] == n[1] || n[2]->src[1] == n[1]) && (n[4]->src[0] == n[3] || n[4]->src[1] == n[3])) {
                const ggml_tensor * cut[9];
                for (int k = 0; k < 9; ++k) { cut[k] = n[k]; }
                const ggml_tensor * retained[] = {n[4], n[5], n[8]};
                const bool closed_cut = program->closed_cut(cut, 9, retained, 3);
                const auto * dt = n[2]->src[n[2]->src[0] == n[1] ? 1 : 0];
                const auto * a = n[4]->src[n[4]->src[0] == n[3] ? 1 : 0];
                const ggml_tensor * views[] = {n[0]->src[1], n[0]->src[0], n[6]->src[0], dt, a, n[5], n[8]};
                for (int k = 0; k < 7; ++k) {
                    prepared.source_tensors[k] = views[k];
                    prepared.source_views[k] = tensor_view(views[k], parent->device);
                }
                if (closed_cut) {
                    const auto status = ggml_cuda_moe_source_gdn_binding_prepare(prepared.source_views, prepared.source_binding);
                    if (status == GGML_CUDA_MOE_SOURCE_NORM_INVALID) { return false; }
                    if (status == GGML_CUDA_MOE_SOURCE_NORM_READY) { prepared.source_kind = 2; prepared.consumed = 9; }
                }
            }
        }
        if (!prepared.source_kind) {
            if (node->op == GGML_OP_MUL_MAT) {
                prepared.matmul_estimated = ggml_cuda_moe_fidelity_mul_mat_requirements(parent->device, node, prepared.matmul);
                if (prepared.matmul_estimated) {
                    pool_bytes = std::max(pool_bytes, prepared.matmul.pool_bytes);
                    cublas_bytes = std::max(cublas_bytes, prepared.matmul.cublas_workspace_bytes);
                }
            } else if (node->op == GGML_OP_FLASH_ATTN_EXT) {
                if (!ggml_cuda_flash_attn_ext_prepare_resources(*compute, node, prepared.attention)) { unsupported = true; return rejected("attention_resources", node); }
                pool_bytes = std::max(pool_bytes, prepared.attention.pool_bytes);
            } else if (node->op == GGML_OP_TOP_K || node->op == GGML_OP_ARGSORT) {
                ggml_cuda_source_sort_resources resources;
                const bool supported = node->op == GGML_OP_TOP_K ? ggml_cuda_top_k_prepare_resources(parent->device, node, resources) :
                    ggml_cuda_argsort_prepare_resources(parent->device, node, resources);
                if (!supported) { unsupported = true; return rejected("sort_resources", node); }
                prepared.sorter = resources;
                pool_bytes = std::max(pool_bytes, resources.pool_bytes);
            } else if (node->op == GGML_OP_SOFT_MAX) {
                if (!ggml_cuda_softmax_prepare_resources(parent->device, node, prepared.softmax)) { unsupported = true; return rejected("softmax_resources", node); }
                pool_bytes = std::max(pool_bytes, prepared.softmax.pool_bytes);
            }
        }
        i += prepared.consumed;
        output.push_back(std::move(prepared));
    }
    if (fusion_graph) {
        for (size_t i = 0; i < output.size();) {
            if (output[i].source_kind) { ++i; continue; }
            const size_t first = output[i].operation.original_index;
            size_t end = i + 1;
            while (end < output.size() && !output[end].source_kind &&
                    output[end].operation.original_index == first + end - i) { ++end; }
            size_t bytes = 0;
            if (!ggml_cuda_moe_source_image_resources(*compute, fusion_graph, int(first), int(first + end - i), bytes)) { return false; }
            image_bytes = std::max(image_bytes, bytes);
            i = end;
        }
    }
    return true;
}

bool core_session::emit_operations(const std::vector<prepared_operation> & operations) {
    for (size_t i = 0; i < operations.size(); ++i) {
        const auto & operation = operations[i];
        if (fusion_graph && !operation.source_kind && operation.operation.original_index < size_t(fusion_graph->n_nodes)) {
            const size_t first = operation.operation.original_index;
            size_t end = i + 1;
            while (end < operations.size() && !operations[end].source_kind &&
                    operations[end].operation.original_index == first + end - i) { ++end; }
            try {
                if (!ggml_cuda_moe_source_compute_range(*compute, fusion_graph, int(first), int(first + end - i), fused_nodes, image_groups, emitted_images)) { return false; }
            } catch (const std::bad_alloc &) {
                const auto & pool = static_cast<const ggml_cuda_pool_buffer &>(*compute->pools[parent->device][0]);
                unsupported |= pool.required <= pool.capacity;
                return false;
            }
            i = end - 1;
            continue;
        }
        if (operation.operation.effect == ggml_moe_source_effect::metadata || ggml_is_empty(operation.operation.tensor)) { continue; }
        if (operation.source_kind == 3) {
#ifdef GGML_MOE_SOURCE_GRAPH
            const auto & view = operation.staged;
            auto * flag = static_cast<uint32_t *>(view.device_flag);
            if (!cancel_alias || view.bytes != ggml_nbytes(operation.operation.tensor)) { return false; }
            auto * control = reinterpret_cast<source_control *>(cancel_alias + layers[0].control_offset);
            copy_input<<<1, 256, 0, stream>>>(static_cast<uint8_t *>(operation.operation.tensor->data),
                static_cast<const uint8_t *>(view.device_alias), view.bytes, flag, &control->stop, runtime());
#else
            return false;
#endif
        } else if (operation.source_kind == 4) {
            if (!weighted_binding_matches(operation, parent->device)) { return false; }
            const auto & match = operation.weighted;
            ggml_cuda_op_moe_weighted_reduction(*compute, match.experts, match.expert_scale, match.weights, match.dst);
        } else if (operation.source_kind) {
            ggml_cuda_moe_source_tensor current[7];
            const uint32_t count = operation.source_kind == 1 ? 3 : 7;
            for (uint32_t k = 0; k < count; ++k) { current[k] = tensor_view(operation.source_tensors[k], parent->device); }
            if (!ggml_cuda_moe_source_binding_matches(operation.source_binding, current, count) ||
                    !ggml_cuda_moe_source_binding_emit(operation.source_binding, parent->device, stream)) { return false; }
        } else {
            auto * tensor = operation.operation.tensor;
            if (!ggml_cuda_moe_router_compute(*compute, tensor)) { return false; }
        }
    }
    return cudaGetLastError() == cudaSuccess;
}

bool core_session::validate_operations(const std::vector<prepared_operation> & operations) {
    for (const auto & operation : operations) {
        if (operation.operation.effect == ggml_moe_source_effect::metadata) { continue; }
        const auto * tensor = operation.operation.tensor;
        if (operation.source_kind == 4 && !weighted_binding_matches(operation, parent->device)) { return false; }
        if (operation.source_kind == 3) {
            ggml_cuda_source_staged_input_view current;
            if (!ggml_cuda_staged_input_prepare_source_view(parent->device, tensor, current) ||
                    current.identity != operation.staged.identity || current.host != operation.staged.host ||
                    current.device_alias != operation.staged.device_alias || current.host_flag != operation.staged.host_flag ||
                    current.device_flag != operation.staged.device_flag || current.bytes != operation.staged.bytes ||
                    current.device != operation.staged.device) { return false; }
        }
        if (!operation.source_kind && operation.matmul_estimated) {
            ggml_cuda_moe_fidelity_mul_mat_resources current;
            if (!ggml_cuda_moe_fidelity_mul_mat_requirements(parent->device, tensor, current) ||
                    current.pool_bytes != operation.matmul.pool_bytes || current.cublas_workspace_bytes != operation.matmul.cublas_workspace_bytes ||
                    current.dispatch != operation.matmul.dispatch || current.compute_type != operation.matmul.compute_type) { return false; }
        }
        if (tensor->op == GGML_OP_FLASH_ATTN_EXT) {
            ggml_cuda_fattn_resources current;
            if (!ggml_cuda_flash_attn_ext_prepare_resources(*compute, tensor, current) ||
                    current.identity != operation.attention.identity || current.pool_bytes != operation.attention.pool_bytes) { return false; }
        }
        if (tensor->op == GGML_OP_SOFT_MAX) {
            ggml_cuda_source_softmax_resources current;
            if (!ggml_cuda_softmax_prepare_resources(parent->device, tensor, current) ||
                    current.identity != operation.softmax.identity || current.pool_bytes != operation.softmax.pool_bytes) { return false; }
        }
        if (operation.source_kind || (tensor->op != GGML_OP_TOP_K && tensor->op != GGML_OP_ARGSORT)) { continue; }
        ggml_cuda_source_sort_resources current;
        const bool valid = tensor->op == GGML_OP_TOP_K ? ggml_cuda_top_k_prepare_resources(parent->device, tensor, current) :
            ggml_cuda_argsort_prepare_resources(parent->device, tensor, current);
        if (!valid || current.identity != operation.sorter.identity || current.pool_bytes != operation.sorter.pool_bytes) { return false; }
    }
    return true;
}

bool core_session::acquire_owner(const ggml_cgraph * graph, bool first) {
    source_host_timer timer(replay_acquire_map_wall_ns, !first);
    const auto reject = [&](const char * stage, const core_layer * layer = nullptr) {
        fprintf(stderr, "moe-source-core-fault: stage=owner_%s first=%d epoch=%llu layer=%zu group_state=%d strategy=%u\n",
            stage, int(first), (unsigned long long) epoch, layer ? size_t(layer - layers.data()) : SIZE_MAX,
            layer && layer->owner_group ? int(layer->owner_group->state) : -1,
            layer && layer->owner_group ? uint32_t(layer->owner_group->strategy) : 0);
        return false;
    };
    if (owner_execution.outcome() != GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED) { return reject("plan"); }
    if (dispatch || !owner().begin_graph_dispatch(&owner_execution, GGML_CUDA_MOE_GRAPH_DISPATCH_DIRECT, config.profile_adaptation == 2)) { return reject("begin"); }
    dispatch = true;
    resources_live = true;
    bool maps_changed = false;
    for (auto & layer : layers) {
        const auto & expert = *layer.descriptor->expert;
        ggml_cuda_moe_graph_binding binding;
        layer.owner_group = owner_execution.find_group(graph->nodes[expert.region.first_node], &binding);
        if (!layer.owner_group) { return reject("group_binding", &layer); }
        if (hook) {
            ggml_cuda_moe_candidate_group_info info = {};
            if (!owner().get_group(layer.owner_group->key.candidate, &info)) { return reject("probe_domain", &layer); }
            layer.probe_domain = info.domain;
        }
        const auto previous_residency = layer.residency_token;
        auto previous = std::find_if(layers.begin(), layers.begin() + (&layer - layers.data()),
            [&](const core_layer & candidate) { return candidate.owner_group == layer.owner_group; });
        if (previous != layers.begin() + (&layer - layers.data())) {
            layer.residency_token = previous->residency_token;
        } else if (!owner().prepare_source_group(layer.owner_group, stream,
                reinterpret_cast<int32_t *>(host + layer.map_offset), expert.region.geometry.expert_count,
                reinterpret_cast<int32_t *>(host + layer.owners_offset), expert.region.geometry.expert_count, first,
                &layer.residency_token)) { return reject("prepare_group", &layer); }
        uint32_t mask = 0;
        for (uint32_t b = 0; b < expert.banks.size(); ++b) {
            if (!first) {
                if (!owner().source_transport_matches(transport, layer.transport_indices[b],
                        layer.owner_group->transaction, layer.bank_indices[b])) { return reject("transport_binding", &layer); }
                continue;
            }
            const auto & source = expert.banks[b];
            ggml_cuda_moe_grouped_bank_descriptor current;
            uint32_t bank = 0;
            for (; bank < layer.owner_group->key.n_banks; ++bank) {
                if (!owner().get_group_resource_bank(layer.owner_group->transaction, bank, &current)) { return reject("bank_reader", &layer); }
                if (current.tensor == source.source.witness) { break; }
            }
            if (bank == layer.owner_group->key.n_banks || bank >= 32 || (mask & (1u << bank)) ||
                    current.source_data != source.source.data || current.byte_extent != source.source.bytes ||
                    current.expert_stride != source.source.expert_stride || current.type != uint32_t(source.metadata.type) ||
                    current.encoding != GGML_CUDA_MOE_CANDIDATE_ENCODING_PLAIN ||
                    current.movement != GGML_CUDA_MOE_CANDIDATE_MOVEMENT_SLOT_BOUND || !layer.owner_group->bank_data[bank] ||
                    memcmp(current.ne, source.metadata.ne, sizeof(current.ne)) || memcmp(current.nb, source.metadata.nb, sizeof(current.nb)) ||
                    (expert.routed_operation && (expert.banks.size() != 1 || binding.bank_index != bank))) { return reject("bank_witness", &layer); }
            mask |= 1u << bank;
            layer.bank_indices[b] = bank;
            layer.banks[b] = current;
        }
        if (first && !expert.routed_operation && (layer.owner_group->key.n_banks >= 32 || mask != (1u << layer.owner_group->key.n_banks) - 1)) { return reject("complete_banks", &layer); }
        if (layer.residency_token != previous_residency) {
            maps_changed = true;
            map_bytes += (size_t(expert.region.geometry.expert_count) + layer.owner_group->n_slots) * sizeof(int32_t);
        }
    }
    if (first) {
        for (const auto & layer : layers) {
            if (!layer.descriptor->expert->routed_operation) { continue; }
            uint32_t mask = 0;
            for (const auto & candidate : layers) {
                if (candidate.owner_group != layer.owner_group) { continue; }
                for (const auto bank : candidate.bank_indices) { mask |= 1u << bank; }
            }
            if (layer.owner_group->key.n_banks >= 32 || mask != (1u << layer.owner_group->key.n_banks) - 1) { return reject("complete_banks", &layer); }
        }
    }
    uint64_t identity = 0;
    if (!owner().graph_resource_fingerprint(owner_execution, stream, &identity, first ? &resource_leases : nullptr) ||
            (resource_identity && resource_identity != identity)) { return reject("resource_identity"); }
    resource_identity = identity;
    if (maps_changed && !wait_stream(stream, now_ns() + 5'000'000'000ull)) { return reject("map_completion"); }
    for (auto & layer : layers) {
        auto previous = std::find_if(layers.begin(), layers.begin() + (&layer - layers.data()),
            [&](const core_layer & candidate) { return candidate.owner_group == layer.owner_group; });
        if (previous == layers.begin() + (&layer - layers.data())) { continue; }
        const auto count = layer.descriptor->expert->region.geometry.expert_count;
        memcpy(host + layer.map_offset, host + previous->map_offset, size_t(count) * sizeof(int32_t));
        memcpy(host + layer.owners_offset, host + previous->owners_offset, size_t(layer.owner_group->n_slots) * sizeof(int32_t));
    }
    if (!first && !apply_profiles(false)) { return reject("profile_application"); }
    if (first && (profiles.empty() && statistics.empty()) && std::getenv("GGML_MOE_SOURCE_MAP_DIAGNOSTIC")) {
        for (const auto & layer : layers) {
            const auto & expert = *layer.descriptor->expert;
            size_t payload = 0;
            for (const auto & bank : expert.banks) { payload += bank.expert_bytes; }
            const auto * map = reinterpret_cast<const int32_t *>(host + layer.map_offset);
            const auto * owners = reinterpret_cast<const int32_t *>(host + layer.owners_offset);
            fprintf(stderr, "moe-source-map: program=%llu down=%s experts=%u slots=%u payload_per_slot=%zu map=",
                (unsigned long long) program_instance, expert.region.down->name,
                expert.region.geometry.expert_count, layer.owner_group->n_slots, payload);
            for (uint32_t i = 0; i < expert.region.geometry.expert_count; ++i) { fprintf(stderr, "%s%d", i ? "," : "", map[i]); }
            fprintf(stderr, " owners=");
            for (uint32_t i = 0; i < layer.owner_group->n_slots; ++i) { fprintf(stderr, "%s%d", i ? "," : "", owners[i]); }
            fprintf(stderr, "\n");
        }
    }
    return true;
}

bool core_session::prepare_profiles() {
    if ((profiles.empty() && statistics.empty())) { return true; }
    for (uint32_t i = 0; i < layers.size(); ++i) {
        const auto & layer = layers[i];
        if (!layer.owner_group) { return false; }
        auto found = std::find_if(profile_groups.begin(), profile_groups.end(), [&](const core_profile_group & group) {
            return layers[group.representative].owner_group == layer.owner_group;
        });
        if (found == profile_groups.end()) {
            ggml_cuda_moe_candidate_group_info info;
            if (!owner().get_group(layer.owner_group->key.candidate, &info) || !info.down ||
                    !info.n_resource_banks || info.n_resource_banks != layer.owner_group->key.n_banks) { return false; }
            const auto seed = profiles.find(info.down);
            if (statistics.empty() && (seed == profiles.end() || seed->second.size() < layer.owner_group->n_slots ||
                    seed->second.size() > layer.descriptor->expert->region.geometry.expert_count)) { return false; }
            core_profile_group group;
            group.representative = i;
            group.domain = info.domain;
            if (statistics.empty()) { group.seed = &seed->second; }
            else { group.statistics.resize(info.n_resource_banks); }
            group.source = info.down;
            group.bindings.resize(info.n_resource_banks, UINT32_MAX);
            profile_groups.push_back(std::move(group));
            found = profile_groups.end() - 1;
        }
        auto & group = *found;
        const auto & expert = *layer.descriptor->expert;
        const auto & geometry = expert.region.geometry;
        if (geometry.expert_count != layers[group.representative].descriptor->expert->region.geometry.expert_count) { return false; }
        group.members.push_back(i);
        for (uint32_t bank = 0; bank < layer.banks.size(); ++bank) {
            const uint32_t index = layer.bank_indices[bank];
            if (index >= group.bindings.size()) { return false; }
            const auto seed = profiles.find(layer.banks[bank].tensor);
            if (seed != profiles.end() && (!group.seed || seed->second != *group.seed)) { return false; }
            if (!statistics.empty()) {
                const auto source = statistics.find(layer.banks[bank].tensor);
                if (source == statistics.end()) { return false; }
                const auto statistic = std::find_if(source->second.begin(), source->second.end(), [&](const core_source_statistics & value) {
                    return value.domain == group.domain;
                });
                if (statistic == source->second.end() || statistic->counts.size() != geometry.expert_count) { return false; }
                auto & view = group.statistics[index];
                if (view.counts && (view.counts != statistic->counts.data() || view.payload_bytes != layer.banks[bank].expert_stride)) { return false; }
                view = {statistic->counts.data(), statistic->observations, layer.banks[bank].expert_stride, geometry.expert_count, statistic->scores.empty() ? nullptr : statistic->scores.data()};
            }
            auto & binding = group.bindings[index];
            if (binding != UINT32_MAX && binding != layer.transport_indices[bank]) { return false; }
            if (binding == UINT32_MAX) {
                if (layer.banks[bank].expert_stride > SIZE_MAX - group.payload_bytes) { return false; }
                group.payload_bytes += layer.banks[bank].expert_stride;
                binding = layer.transport_indices[bank];
            }
        }
        if (!config.profile_adaptation) { continue; }
        // Count one routing decision when cuts read the same original IDs tensor.
        const auto observation = std::find_if(group.observations.begin(), group.observations.end(), [&](uint32_t index) {
            return layers[index].descriptor->expert->region.ids == expert.region.ids;
        });
        if (observation != group.observations.end()) {
            const auto & previous = layers[*observation].descriptor->expert->region.geometry;
            if (previous.route_capacity != geometry.route_capacity || previous.routes_per_row != geometry.routes_per_row ||
                    previous.row_capacity != geometry.row_capacity) { return false; }
            continue;
        }
        if (!geometry.routes_per_row || !geometry.row_capacity ||
                uint64_t(geometry.routes_per_row) * geometry.row_capacity != geometry.route_capacity ||
                layer.ids_stride < size_t(geometry.routes_per_row) * sizeof(int32_t) ||
                geometry.route_capacity > UINT32_MAX - group.routes.size()) { return false; }
        group.observations.push_back(i);
        group.routes.resize(group.routes.size() + geometry.route_capacity);
    }
    profile_updates.resize(profile_groups.size());
    for (auto & group : profile_groups) {
        if (!statistics.empty()) {
            if (!ggml_moe_source_rank_statistics(group.statistics, group.derived_seed)) { return false; }
            group.seed = &group.derived_seed;
        }
        const auto slots = layers[group.representative].owner_group->n_slots;
        if (slots && group.payload_bytes > SIZE_MAX / slots) { return false; }
        if (std::find(group.bindings.begin(), group.bindings.end(), UINT32_MAX) != group.bindings.end() ||
                (config.profile_adaptation && group.observations.empty())) { return false; }
    }
    return !profile_groups.empty();
}

bool core_session::apply_profiles(bool first, bool maintenance) {
    if ((profiles.empty() && statistics.empty())) { return !maintenance; }
    if (profile_groups.empty() || profile_updates.size() != profile_groups.size()) { return false; }
    for (size_t i = 0; i < profile_groups.size(); ++i) {
        auto & group = profile_groups[i];
        auto & layer = layers[group.representative];
        if (!layer.owner_group || !group.seed || group.seed->size() < layer.owner_group->n_slots) { return false; }
        for (const uint32_t member : group.members) {
            if (layers[member].owner_group != layer.owner_group) { return false; }
        }
        auto & update = profile_updates[i];
        update = {};
        update.group = layer.owner_group;
        update.seed = group.seed->data();
        update.n_seed = group.seed->size();
        update.map = reinterpret_cast<int32_t *>(host + layer.map_offset);
        update.owners = reinterpret_cast<int32_t *>(host + layer.owners_offset);
        update.bindings = group.bindings.data();
        update.n_bindings = group.bindings.size();
        if (config.profile_adaptation) {
            size_t offset = 0;
            for (const uint32_t observation : group.observations) {
                const auto & current = layers[observation];
                const auto & geometry = current.descriptor->expert->region.geometry;
                if (maintenance) {
                    for (uint32_t row = 0; row < geometry.row_capacity; ++row) {
                        memcpy(group.routes.data() + offset + size_t(row) * geometry.routes_per_row,
                            host + current.ids_offset + size_t(row) * current.ids_stride,
                            size_t(geometry.routes_per_row) * sizeof(int32_t));
                    }
                }
                offset += geometry.route_capacity;
            }
            if (offset != group.routes.size()) { return false; }
            update.routes = group.routes.data();
            update.n_routes = group.routes.size();
        }
    }
    if (config.profile_adaptation && (canceled.load() || !owner().update_source_profiles(
            profile_updates.data(), profile_updates.size(), transport, io,
            maintenance ? deadline : now_ns() + 30'000'000'000ull, maintenance,
            config.profile_adaptation == 2, &source_owner, hook, hook_data))) { return false; }
    for (size_t i = 0; i < profile_groups.size(); ++i) {
        const auto & group = profile_groups[i];
        auto & update = profile_updates[i];
        auto & layer = layers[group.representative];
        if (!config.profile_adaptation && (canceled.load() || !owner().apply_source_profile(
                *update.group, update.seed, update.group->n_slots, update.map, update.owners,
                transport, update.bindings, update.n_bindings, io, now_ns() + 30'000'000'000ull,
                &update.copied_bytes))) { return false; }
        for (const uint32_t member : group.members) {
            if (member == group.representative) { continue; }
            memcpy(host + layers[member].map_offset, update.map,
                size_t(layer.descriptor->expert->region.geometry.expert_count) * sizeof(int32_t));
            memcpy(host + layers[member].owners_offset, update.owners, size_t(update.group->n_slots) * sizeof(int32_t));
        }
        if (first || update.copied_bytes) {
            fprintf(stderr, "moe-profile-map: program=%llu down=%s slots=%u resident_bytes=%zu copied_bytes=%llu owners=",
                (unsigned long long) program_instance, group.source->name, update.group->n_slots,
                group.payload_bytes * update.group->n_slots, (unsigned long long) update.copied_bytes);
            for (uint32_t slot = 0; slot < update.group->n_slots; ++slot) { fprintf(stderr, "%s%d", slot ? "," : "", update.owners[slot]); }
            fprintf(stderr, "\n");
        }
    }
    return true;
}

bool core_session::finish_owner() {
    if (!dispatch) { return true; }
    if (!owner().finish_source_dispatch(&owner_execution)) { return false; }
    dispatch = false;
    for (auto & layer : layers) { layer.owner_group = nullptr; }
    return true;
}

bool core_session::allocate(const ggml_cgraph * graph, const std::vector<core_region *> & regions) try {
    source_host_timer timer(prepare_wall_ns);
#ifndef GGML_MOE_SOURCE_GRAPH
    GGML_UNUSED(graph); GGML_UNUSED(regions);
    return false;
#else
    if (prepared || !parent || !parent->moe_grouped_context || regions.empty() ||
            !std::atomic<uint32_t>::is_always_lock_free || sizeof(std::atomic<uint32_t>) != sizeof(uint32_t)) { return false; }
    ggml_cuda_set_device(parent->device);
    std::vector<const ggml_moe_source_expert *> experts;
    for (const auto * region : regions) {
        if (!region || region->owner != this || region->cpu_api != cpu_api || region->cpu_service != cpu_service) { return false; }
        experts.push_back(&region->expert);
    }
    program = std::make_unique<ggml_moe_source_program>();
    const auto option = [](const char * name, bool & value) {
        const auto * text = getenv(name);
        if (text && strcmp(text, "0") && strcmp(text, "1")) { return false; }
        if (text) { value = !strcmp(text, "1"); }
        return true;
    };
    if (!option("GGML_MOE_SOURCE_SHARED_OVERLAP", overlap_enabled) ||
            !option("GGML_MOE_SOURCE_TEST_OVERLAP_PROBE", overlap_probe) || (overlap_probe && !overlap_enabled) ||
            !option("GGML_MOE_SOURCE_KERNEL_COPY", kernel_copy_enabled) ||
            !option("GGML_MOE_SOURCE_GRAPH_FUSION", fusion_enabled) ||
            !option("GGML_MOE_CPU_PROFILE", cpu_profile) ||
            !option("GGML_MOE_SOURCE_DEVICE_TAIL", device_tail) ||
            !option("GGML_MOE_SOURCE_PUBLICATION_PACKET", publication_packet)) { return false; }
    const auto * disable_fusion = getenv("GGML_CUDA_DISABLE_FUSION");
    weighted_enabled = !disable_fusion || !std::atoi(disable_fusion);
    ggml_moe_source_schedule_options schedule;
    schedule.overlap_independent_ordinary = overlap_enabled;
    schedule.leaf_context = this;
    schedule.bind_leaf = [](void * context, const ggml_tensor * original, ggml_backend_buffer_t * buffer, void ** data) {
        auto & session = *static_cast<core_session *>(context);
        bool unavailable = false;
        if (tensor_storage(original, session.parent->device, unavailable)) { return true; }
        std::shared_ptr<void> lease;
        if (!session.owner().prepare_source_leaf(original, session.parent->stream(), session.source_owner, buffer, data, &lease)) { return false; }
        if (lease) { session.leaf_resource_leases.push_back(std::move(lease)); }
        return true;
    };
    if (weighted_enabled && graph->use_counts && graph->visited_hash_set.size &&
            graph->visited_hash_set.used && graph->visited_hash_set.keys) {
        const ggml_backend_graph_optimize_params params{
            [](void * user, ggml_tensor * tensor, ggml_tensor * until) {
                auto * dependencies = static_cast<std::vector<ggml_moe_source_allocation_dependency> *>(user);
                dependencies->push_back({tensor, until});
            }, &schedule.allocation_dependencies};
        if (fusion_enabled) {
            for (int first = 0; first < graph->n_nodes;) {
                int end = graph->n_nodes;
                bool owned = false;
                for (const auto * expert : experts) {
                    if (uint64_t(first) >= expert->region.first_node && uint64_t(first) <= expert->region.last_node) {
                        first = int(expert->region.last_node + 1); owned = true; break;
                    }
                    if (expert->region.first_node > uint64_t(first)) { end = std::min(end, int(expert->region.first_node)); }
                }
                if (owned) { continue; }
                auto range = ggml_graph_view(const_cast<ggml_cgraph *>(graph), first, end);
                ggml_cuda_moe_source_image_alloc_deps(&range, parent->device, &params);
                first = end;
            }
        }
        for (int i = 0; i < graph->n_nodes; ++i) {
            ggml_cuda_moe_weighted_reduction_match match;
            if (!ggml_cuda_match_moe_weighted_reduction(graph, i, match)) { continue; }
            bool ordinary = true;
            for (const auto * expert : experts) {
                ordinary &= uint64_t(i + match.node_count - 1) < expert->region.first_node || uint64_t(i) > expert->region.last_node;
            }
            if (ordinary) { ggml_cuda_moe_weighted_reduction_alloc_deps(match, &params); }
            i += match.node_count - 1;
        }
    }
    if (retain_routed_outputs) {
        const auto * until = graph->nodes[graph->n_nodes - 1];
        for (const auto * expert : experts) {
            if (expert->routed_operation && expert->region.output != until) {
                schedule.allocation_dependencies.push_back({expert->region.output, until});
            }
        }
    }
    if (!program->prepare(graph, experts, ggml_backend_get_default_buffer_type(config.backend), schedule)) {
        program.reset();
        return false;
    }
    if (!program->allocate()) { return false; }
    if (fusion_enabled && graph->use_counts && graph->visited_hash_set.size &&
            graph->visited_hash_set.used && graph->visited_hash_set.keys) {
        fusion_context = ggml_init({ggml_graph_overhead_custom(graph->n_nodes, false), nullptr, true});
        if (!fusion_context) { return false; }
        fusion_graph = ggml_new_graph_custom(fusion_context, graph->n_nodes, false);
        fusion_graph->n_nodes = graph->n_nodes;
        for (int i = 0; i < graph->n_nodes; ++i) {
            auto * tensor = program->find(graph->nodes[i]);
            fusion_graph->nodes[i] = tensor;
            if (!tensor) { continue; }
            const size_t slot = ggml_hash_insert(&fusion_graph->visited_hash_set, tensor);
            if (slot == GGML_HASHSET_ALREADY_EXISTS || slot == GGML_HASHSET_FULL) { return false; }
            // Keep consumers outside the emitted phase in the fusion use counts.
            fusion_graph->use_counts[slot] = ggml_node_get_use_count(graph, i);
        }
    }
    for (const auto * original : program->public_outputs()) {
        const auto * output = program->find(original);
        if (!output || !tensor_storage(original, parent->device, unsupported) ||
                !tensor_storage(output, parent->device, unsupported)) { return false; }
        if (output->data != original->data && (!ggml_is_contiguous(original) || !ggml_is_contiguous(output))) { unsupported = true; return false; }
    }
    compute = std::make_unique<ggml_backend_cuda_context>(parent->device);
    stream = parent->stream();
    compute->borrowed_stream = stream;
    compute->streams[parent->device][0] = stream;
    compute->decode_boundary_overlap = parent->decode_boundary_overlap;
    if (cudaStreamCreateWithFlags(&io, cudaStreamNonBlocking) != cudaSuccess) { return false; }
    noalias = getenv("GGML_MOE_FIDELITY_NO_HOST_ALIAS") && strcmp(getenv("GGML_MOE_FIDELITY_NO_HOST_ALIAS"), "0");
    const auto * arm = getenv("GGML_MOE_FIDELITY_ARM");
    if (arm && strcmp(arm, "poll") && strcmp(arm, "segmented")) { return false; }
    segmented = noalias || (arm && !strcmp(arm, "segmented"));
    size_t scratch_bytes = 0, quant_bytes = 0, selected_bytes = 0, cpu_bytes = 0, tile_bytes = 0;
#if !defined(GGML_CUDA_MOE_DEVICE_TAIL)
    device_tail = false;
#endif
    device_tail &= !segmented;
#if defined(GGML_CUDA_MOE_DEVICE_TAIL)
    size_t tail_bytes = 0;
    if (device_tail && (experts.size() > SIZE_MAX - 2 ||
            !product(experts.size() + 2, sizeof(ggml_cuda_moe_source_tail_dispatch), tail_bytes) ||
            !reserve(device_bytes, tail_bytes, tail_offset) ||
            !reserve(pinned_bytes, tail_bytes, host_tail_offset))) { return false; }
#endif
    layers.resize(experts.size());
    for (uint32_t i = 0; i < layers.size(); ++i) {
        auto & layer = layers[i];
        layer.descriptor = &program->layers()[i];
        const auto & expert = *experts[i];
        const auto & geometry = expert.region.geometry;
        const ggml_moe_reference_gpu_layout layout{int(expert.roles[0].type), int(expert.roles[2].type),
            int(expert.input_width), int(expert.hidden_width), int(expert.output_width),
            expert.roles[0].row_stride, expert.roles[1].row_stride, expert.roles[2].row_stride};
        if ((!expert.routed_operation && !expert.generic_body && (!ggml_moe_reference_gpu_supported(layout) || geometry.route_capacity > 65535)) || !geometry.route_capacity ||
                !prepare_operations(graph, layer.descriptor->prelude, layer.prelude) ||
                !prepare_operations(graph, layer.descriptor->overlap, layer.overlap)) { return false; }
        if (overlap_probe && !layer.overlap.empty() && !reserve(pinned_bytes, sizeof(source_flag), layer.overlap_probe_offset)) { return false; }
        size_t input_bytes, ids_bytes, output_bytes, quant = 0;
        layer.ids_stride = expert.routed_operation ? layer.descriptor->ids->nb[1] : size_t(geometry.routes_per_row) * sizeof(int32_t);
        if (expert.routed_operation) {
            layer.routed_output = *expert.region.output;
            layer.routed_output.src[0] = const_cast<ggml_tensor *>(&expert.banks[0].metadata);
            layer.routed_output.src[1] = layer.descriptor->activation;
            layer.routed_output.src[2] = layer.descriptor->ids;
            layer.routed_output.data = layer.descriptor->output->data;
            layer.routed_output.buffer = layer.descriptor->output->buffer;
            if (!cpu_api->execute_routed || !ggml_cuda_mmid_requirements(parent->device, &layer.routed_output, layer.routed_resources)) { return false; }
            input_bytes = ggml_nbytes(layer.descriptor->activation);
            if (!product(geometry.route_capacity, layer.descriptor->output->nb[1], output_bytes)) { return false; }
            pool_bytes = std::max(pool_bytes, layer.routed_resources.pool_bytes);
        } else if (!product(geometry.row_capacity, size_t(expert.input_width) * sizeof(float), input_bytes) ||
                !product(geometry.route_capacity, size_t(expert.output_width) * sizeof(float), output_bytes) ||
                (!expert.generic_body && !product(geometry.row_capacity, size_t(expert.input_width) / 32 * sizeof(block_q8_1), quant))) { return false; }
        if (expert.generic_body) {
            layer.body.reserve(expert.region.last_node - expert.region.first_node + 1);
            for (uint32_t n = expert.region.first_node; n <= expert.region.last_node; ++n) {
                const auto * original = graph->nodes[n];
                auto * tensor = program->find(original);
                if (!tensor || !tensor_storage(tensor, parent->device, unsupported)) { return false; }
                // Ordinary activations read full rows, including routes skipped by the owned matrix operation.
                if (!ggml_is_view(original) && cudaMemsetAsync(tensor->data, 0, ggml_nbytes(tensor), stream) != cudaSuccess) { return false; }
                size_t bank = SIZE_MAX;
                if (original->op == GGML_OP_MUL_MAT_ID) {
                    bank = 0;
                    while (bank < expert.banks.size() && expert.banks[bank].source.witness != original->src[0]) { ++bank; }
                    if (bank == expert.banks.size()) { return false; }
                    tensor->op = GGML_OP_MUL_MAT_ID;
                    tensor->src[0] = const_cast<ggml_tensor *>(&expert.banks[bank].metadata);
                    ggml_cuda_mmid_resources resources;
                    if (!ggml_cuda_mmid_requirements(parent->device, tensor, resources)) { return false; }
                    pool_bytes = std::max(pool_bytes, resources.pool_bytes);
                } else if (!ggml_is_view(original) && original->op != GGML_OP_GLU && original->op != GGML_OP_SQR &&
                        !(original->op == GGML_OP_UNARY && ggml_get_unary_op(original) == GGML_UNARY_OP_RELU)) { return false; }
                layer.body.push_back({tensor, bank});
            }
        }
        layer.input_bytes = input_bytes; layer.output_bytes = output_bytes;
        if (!product(geometry.row_capacity, layer.ids_stride, ids_bytes) ||
                !reserve(pinned_bytes, sizeof(source_control), layer.control_offset) ||
                !reserve(pinned_bytes, input_bytes, layer.input_offset) ||
                !reserve(pinned_bytes, ids_bytes, layer.ids_offset) ||
                !reserve(pinned_bytes, output_bytes, layer.cpu_offset) ||
                !reserve(pinned_bytes, size_t(geometry.expert_count) * sizeof(int32_t), layer.map_offset) ||
                !reserve(pinned_bytes, size_t(geometry.expert_count) * sizeof(int32_t), layer.owners_offset)) { return false; }
        layer.plan_bytes = 2 * group_bytes(geometry.route_capacity);
        if (kernel_copy_enabled) {
            size_t jobs, bytes;
            if (!product(geometry.route_capacity, expert.banks.size(), jobs) || jobs > UINT32_MAX ||
                    !product(jobs, sizeof(source_copy_job), bytes) || bytes > SIZE_MAX - sizeof(source_copy_plan) ||
                    bytes + sizeof(source_copy_plan) > SIZE_MAX - layer.plan_bytes) { return false; }
            layer.plan_bytes += sizeof(source_copy_plan) + bytes;
        }
        if (!expert.routed_operation && device_tail && expert.output_width % ggml_cuda_moe_source_import_threads == 0) {
            const size_t bytes = ((size_t(geometry.route_capacity) + 31) / 32) * sizeof(uint32_t);
            if (bytes > SIZE_MAX - layer.plan_bytes) { return false; }
            layer.gpu_mask_offset = layer.plan_bytes;
            layer.plan_bytes += bytes;
        }
        if (expert.routed_operation || expert.generic_body) {
            size_t pointer_bytes, pointers;
            if (!product(geometry.expert_count, expert.banks.size(), pointers) ||
                    !product(pointers, sizeof(void *), pointer_bytes) ||
                    !reserve(layer.plan_bytes, pointer_bytes, layer.source_image_offset) ||
                    !reserve(layer.plan_bytes, geometry.route_capacity, layer.ownership_offset)) { return false; }
            layer.cpu_ownership.resize(geometry.route_capacity);
        }
        if (!reserve(pinned_bytes, layer.plan_bytes, layer.host_plan_offset) ||
                !reserve(device_bytes, layer.plan_bytes, layer.device_plan_offset)) { return false; }
        layer.bank_indices.resize(expert.banks.size());
        layer.transport_indices.resize(expert.banks.size());
        layer.banks.resize(expert.banks.size());
        layer.selected_offsets.resize(expert.banks.size());
        size_t all_banks = 0;
        const uint32_t selected_capacity = uint64_t(std::min(geometry.route_capacity, geometry.expert_count)) * ggml_moe_fidelity_selection().pcie_num / 256;
        for (size_t b = 0; b < expert.banks.size(); ++b) {
            layer.selected_offsets[b] = all_banks;
            size_t bytes;
            if (!product(selected_capacity, expert.banks[b].expert_bytes, bytes) || bytes > SIZE_MAX - all_banks) { return false; }
            all_banks += bytes;
            tile_bytes = std::max(tile_bytes, expert.banks[b].expert_bytes);
        }
        layer.selected_bytes = all_banks;
        selected_bytes = std::max(selected_bytes, all_banks);
        quant_bytes = std::max(quant_bytes, quant);
        cpu_bytes = std::max(cpu_bytes, output_bytes);
        if (!expert.routed_operation && !expert.generic_body) { scratch_bytes = std::max(scratch_bytes, ggml_moe_reference_gpu_scratch(geometry.route_capacity, expert.hidden_width)); }
        layer.distinct.reserve(geometry.route_capacity);
        layer.slots.reserve(geometry.route_capacity);
        layer.classes.reserve(geometry.route_capacity);
        layer.selected.reserve(geometry.route_capacity);
        layer.weights.resize(geometry.route_capacity);
        layer.counts.resize(geometry.route_capacity);
        layer.cpu_ids.reserve(geometry.route_capacity);
        layer.cpu_rows.reserve(geometry.route_capacity);
        layer.cpu_destinations.reserve(geometry.route_capacity);
        for (auto * event : {&layer.ready, &layer.copied, &layer.cpu_copied, &layer.plan_ready}) {
            if (cudaEventCreateWithFlags(event, cudaEventDisableTiming) != cudaSuccess) { return false; }
        }
    }
    if (!prepare_operations(graph, program->epilogue(), epilogue) || pool_bytes > SIZE_MAX - 255 ||
            image_bytes > SIZE_MAX - GGML_PAD(pool_bytes, 256)) { return false; }
    pool_bytes = GGML_PAD(pool_bytes, 256) + image_bytes;
    if (!reserve(pinned_bytes, sizeof(source_runtime), host_runtime_offset) ||
            !reserve(device_bytes, sizeof(source_runtime), runtime_offset) ||
            !reserve(device_bytes, scratch_bytes, scratch_offset) ||
            !reserve(device_bytes, quant_bytes, quant_offset) ||
            !reserve(device_bytes, selected_bytes, selected_offset) ||
            !reserve(device_bytes, cpu_bytes, cpu_device_offset) ||
            !reserve(device_bytes, pool_bytes, pool_offset) ||
            !reserve(device_bytes, cublas_bytes, cublas_offset) ||
            program->storage_bytes() > SIZE_MAX - device_bytes ||
            (config.device_bytes && device_bytes + program->storage_bytes() > config.device_bytes) ||
            (config.pinned_bytes && pinned_bytes > config.pinned_bytes)) { return false; }
    arena = ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(config.backend), device_bytes);
    if (!arena || cudaHostAlloc(&host, pinned_bytes, cudaHostAllocMapped) != cudaSuccess) { return false; }
    memset(host, 0, pinned_bytes);
    for (uint32_t i = 0; i < layers.size(); ++i) {
        new (control(i)) source_control();
        if (layers[i].overlap_probe_offset != SIZE_MAX) { new (host + layers[i].overlap_probe_offset) source_flag(); }
    }
    controls_ready.store(true, std::memory_order_release);
    if (cudaHostGetDevicePointer(&cancel_alias, host, 0) != cudaSuccess) { cancel_alias = nullptr; (void) cudaGetLastError(); }
    alias = noalias ? nullptr : cancel_alias;
    if (!cancel_alias) {
        for (const auto & layer : layers) { for (const auto & operation : layer.prelude) { if (operation.source_kind == 3) { unsupported = true; return false; } } }
        for (const auto & operation : epilogue) { if (operation.source_kind == 3) { unsupported = true; return false; } }
    }
    if (!alias) { noalias = true; segmented = true; }
    device_tail &= !segmented;
    compute->pools[parent->device][0] = std::make_unique<ggml_cuda_pool_buffer>(data() + pool_offset, pool_bytes);
    if (cublas_bytes && !ggml_cuda_moe_fidelity_prepare_cublas(*compute, data() + cublas_offset, cublas_bytes)) { return false; }
    uint32_t mmids = 0;
    uint64_t fingerprint = 0;
    const auto coverage = owner().certify_graph_coverage(graph, &mmids, &fingerprint);
    if (!coverage || !mmids || !fingerprint || owner().prepare_graph_execution(graph, graph->uid,
            GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &owner_plan, &owner_execution, coverage,
            graph->nodes, mmids, fingerprint, true) == GGML_CUDA_MOE_GRAPH_PREPARE_UNAVAILABLE ||
            !owner_execution.resolve_streams([](void * context, const ggml_tensor *) {
                return static_cast<core_session *>(context)->stream;
            }, this) || !acquire_owner(graph, true)) { return false; }
    for (auto & layer : layers) {
        for (uint32_t b = 0; b < layer.banks.size(); ++b) {
            if (!owner().prepare_source_transport(layer.owner_group->transaction, layer.bank_indices[b], tile_bytes,
                    &transport, &layer.transport_indices[b], kernel_copy_enabled && !segmented)) { return false; }
        }
    }
    if (!prepare_profiles() || !apply_profiles(true) || !prepare_capture_resources() ||
            !wait_stream(stream, now_ns() + 5'000'000'000ull) || !finish_owner()) { return false; }
    resources_live = false;
    prepared = true;
    counters.window_fused_nodes = fused_nodes;
    counters.device_bytes = device_bytes + program->storage_bytes();
    counters.pinned_bytes = pinned_bytes;
    supervisor = std::thread([this] { supervise(); });
    return true;
#endif
} catch (...) { return false; }

bool core_session::build_plan(uint32_t i) {
    source_host_timer timer(compact_plan_wall_ns);
    auto & layer = layers[i];
    const auto & expert = *layer.descriptor->expert;
    const auto & geometry = expert.region.geometry;
    const auto * ids = reinterpret_cast<const int32_t *>(host + layer.ids_offset);
    const auto * slots = reinterpret_cast<const int32_t *>(host + layer.map_offset);
    const auto * owners = reinterpret_cast<const int32_t *>(host + layer.owners_offset);
    layer.distinct.clear(); layer.slots.clear(); layer.classes.clear(); layer.selected.clear();
    layer.cpu_ids.clear(); layer.cpu_rows.clear(); layer.cpu_destinations.clear();
    layer.cpu_count = layer.resident_count = layer.selected_count = 0;
    layer.branch_expected = layer.branch_completed = 0;
    memset(host + layer.host_plan_offset, 0, layer.plan_bytes);
    for (uint32_t route = 0; route < geometry.route_capacity; ++route) {
        const auto * row = reinterpret_cast<const int32_t *>(reinterpret_cast<const uint8_t *>(ids) + size_t(route / geometry.routes_per_row) * layer.ids_stride);
        const int32_t id = row[route % geometry.routes_per_row];
        if (expert.routed_operation) {
            layer.cpu_ids.push_back(id);
            layer.cpu_rows.push_back(route / geometry.routes_per_row);
            layer.cpu_destinations.push_back(route);
        }
        if (id < 0 || uint32_t(id) >= geometry.expert_count) { return false; }
        const auto it = std::find(layer.distinct.begin(), layer.distinct.end(), id);
        const uint32_t weight = uint32_t(it - layer.distinct.begin());
        if (it == layer.distinct.end()) {
            const int32_t slot = slots[id];
            if (slot < -1 || (slot >= 0 && (uint32_t(slot) >= layer.owner_group->n_slots || owners[slot] != id))) { return false; }
            layer.distinct.push_back(id);
            layer.slots.push_back(slot);
            layer.classes.push_back(slot >= 0 ? 0 : 2);
            layer.counts[weight] = 0;
        }
        layer.weights[route] = weight;
        ++layer.counts[weight];
    }
    uint32_t misses = 0;
    for (auto kind : layer.classes) { misses += kind == 2; }
    uint32_t selected = uint64_t(misses) * ggml_moe_fidelity_selection().pcie_num / 256;
    for (uint32_t w = uint32_t(layer.distinct.size()); w > 0 && selected; --w) {
        if (layer.classes[w - 1] == 2) { layer.classes[w - 1] = 1; --selected; }
    }
    source_group_view plans[] = {group_view(host + layer.host_plan_offset, geometry.route_capacity),
        group_view(host + layer.host_plan_offset + group_bytes(geometry.route_capacity), geometry.route_capacity)};
    uint32_t groups[2] = {}, entries[2] = {};
    for (uint32_t w = 0; w < layer.distinct.size(); ++w) {
        const auto kind = layer.classes[w];
        if (kind == 2) {
            for (uint32_t route = 0; route < geometry.route_capacity; ++route) {
                if (layer.weights[route] != w) { continue; }
                if (!expert.routed_operation) {
                    layer.cpu_ids.push_back(layer.distinct[w]);
                    layer.cpu_rows.push_back(route / geometry.routes_per_row);
                    layer.cpu_destinations.push_back(route);
                } else { ++layer.cpu_count; }
            }
            continue;
        }
        auto & plan = plans[kind];
        const uint32_t group = groups[kind]++;
        const uint32_t selected_index = uint32_t(layer.selected.size());
        if (kind == 1) { layer.selected.push_back(layer.distinct[w]); }
        uint64_t * pointers[] = {plan.gate, plan.up, plan.down};
        for (uint32_t r = 0; r < (expert.routed_operation ? 1u : 3u); ++r) {
            const auto & role = expert.roles[r];
            const auto & bank = expert.banks[role.bank];
            const auto * base = kind == 0 ? static_cast<const uint8_t *>(layer.owner_group->bank_data[layer.bank_indices[role.bank]]) +
                size_t(layer.slots[w]) * bank.source.expert_stride : data() + selected_offset + layer.selected_offsets[role.bank] +
                size_t(selected_index) * bank.expert_bytes;
            pointers[r][group] = reinterpret_cast<uint64_t>(base + role.offset);
        }
        plan.starts[group] = entries[kind];
        for (uint32_t route = 0; route < geometry.route_capacity; ++route) {
            if (layer.weights[route] != w) { continue; }
            plan.tokens[entries[kind]] = route / geometry.routes_per_row;
            plan.destinations[entries[kind]++] = route;
            if (layer.gpu_mask_offset != SIZE_MAX) {
                auto * mask = reinterpret_cast<uint32_t *>(host + layer.host_plan_offset + layer.gpu_mask_offset);
                mask[route / 32] |= uint32_t(1) << (route % 32);
            }
        }
        plan.starts[group + 1] = entries[kind];
    }
    *plans[0].count = groups[0]; *plans[1].count = groups[1];
    if (expert.routed_operation || expert.generic_body) {
        auto ** sources = reinterpret_cast<const void **>(host + layer.host_plan_offset + layer.source_image_offset);
        auto * ownership = host + layer.host_plan_offset + layer.ownership_offset;
        for (uint32_t w = 0, selected_index = 0; w < layer.distinct.size(); ++w) {
            const auto kind = layer.classes[w];
            if (kind == 2) { continue; }
            for (size_t b = 0; b < expert.banks.size(); ++b) {
                const auto & bank = expert.banks[b];
                sources[b * geometry.expert_count + layer.distinct[w]] = kind == 0 ?
                    static_cast<const uint8_t *>(layer.owner_group->bank_data[layer.bank_indices[b]]) +
                    size_t(layer.slots[w]) * bank.source.expert_stride : data() + selected_offset + layer.selected_offsets[b] +
                    size_t(selected_index) * bank.expert_bytes;
            }
            selected_index += kind == 1;
        }
        for (uint32_t route = 0; route < geometry.route_capacity; ++route) {
            ownership[route] = layer.classes[layer.weights[route]];
            layer.cpu_ownership[route] = ownership[route] == 2;
        }
    }
    if (!expert.routed_operation) { layer.cpu_count = uint32_t(layer.cpu_ids.size()); }
    layer.resident_count = entries[0]; layer.selected_count = entries[1];
    layer.kernel_copy = kernel_copy_enabled && !segmented;
    if (layer.kernel_copy) {
        auto * plan = copy_plan(host + layer.host_plan_offset, geometry.route_capacity);
        auto * jobs = copy_jobs(plan);
        for (uint32_t selected = 0; selected < layer.selected.size(); ++selected) {
            for (uint32_t b = 0; b < expert.banks.size(); ++b) {
                const auto & bank = expert.banks[b];
                const auto * input = static_cast<const uint8_t *>(bank.source.data) + size_t(layer.selected[selected]) * bank.source.expert_stride;
                const void * alias = nullptr;
                if (canceled.load() || !owner().source_transport_alias(transport, layer.transport_indices[b],
                        layer.banks[b].tensor, input, bank.expert_bytes, &alias)) { return false; }
                layer.kernel_copy &= alias != nullptr;
                if (bank.expert_bytes > UINT64_MAX - plan->bytes) { return false; }
                jobs[plan->jobs++] = {static_cast<const uint8_t *>(alias),
                    data() + selected_offset + layer.selected_offsets[b] + size_t(selected) * bank.expert_bytes, bank.expert_bytes};
                plan->bytes += bank.expert_bytes;
            }
        }
        plan->kernel = layer.kernel_copy;
        if (!layer.kernel_copy) { plan->jobs = 0; plan->bytes = 0; }
        kernel_jobs_expected += plan->jobs;
        kernel_bytes_expected += plan->bytes;
    }
    layer.branch_expected = (layer.cpu_count != 0) + (layer.resident_count != 0) + (layer.selected_count != 0);
    gpu_expected += (layer.resident_count != 0) + (layer.selected_count != 0);
    copy_expected += layer.selected_count != 0;
    counters.distinct_experts += layer.distinct.size();
    counters.resident_experts += groups[0]; counters.transfer_experts += groups[1];
    counters.cpu_experts += layer.distinct.size() - groups[0] - groups[1];
    counters.resident_routes += entries[0]; counters.transfer_routes += entries[1]; counters.cpu_routes += layer.cpu_count;
    return true;
}

static void CUDART_CB selected_ready(void * context) {
    static_cast<source_control *>(context)->copied.store(1);
}

bool core_session::serve(uint32_t i) {
    auto & layer = layers[i];
    const auto & expert = *layer.descriptor->expert;
    const auto & geometry = expert.region.geometry;
    source_cpu_range service_trace(cpu_profile, "generic.service", epoch, i);
    source_cpu_range plan_trace(cpu_profile, "generic.plan", epoch, i);
    if (canceled.load(std::memory_order_acquire) || control(i)->epoch != epoch || !build_plan(i)) { return false; }
    if (segmented || !layer.kernel_copy || hook) {
        ggml_cuda_set_device(parent->device);
    }
    if (segmented) {
        if (cudaMemcpyAsync(data() + layer.device_plan_offset, host + layer.host_plan_offset, layer.plan_bytes,
                cudaMemcpyHostToDevice, io) != cudaSuccess || cudaEventRecord(layer.plan_ready, io) != cudaSuccess ||
                cudaStreamWaitEvent(stream, layer.plan_ready, 0) != cudaSuccess ||
                !launch_graph(size_t(i) * 4 + 1)) { return false; }
    } else { control(i)->plan.store(1); }
    plan_trace.finish();
    service_trace.mark("generic.plan-ready");
    if (!layer.kernel_copy && !layer.selected.empty()) {
        source_host_timer timer(selected_copy_submit_wall_ns);
        for (uint32_t selected = 0; selected < layer.selected.size(); ++selected) {
            for (uint32_t b = 0; b < expert.banks.size(); ++b) {
                const auto & bank = expert.banks[b];
                auto * destination = data() + selected_offset + layer.selected_offsets[b] + size_t(selected) * bank.expert_bytes;
                const auto * input = static_cast<const uint8_t *>(bank.source.data) + size_t(layer.selected[selected]) * bank.source.expert_stride;
                if (canceled.load() || !owner().copy_source_transport(transport, layer.transport_indices[b], layer.banks[b].tensor, destination, input,
                        bank.expert_bytes, io, deadline)) { return false; }
                counters.h2d_bytes += bank.expert_bytes;
            }
        }
    }
    if (layer.kernel_copy) { control(i)->copied.store(1); }
    else if (cudaEventRecord(layer.copied, io) != cudaSuccess ||
            (!segmented && cudaLaunchHostFunc(io, selected_ready, control(i)) != cudaSuccess)) { return false; }
    layer.copy_submitted = true;
    if (hook && !hook(hook_data, GGML_BACKEND_MOE_HYBRID_TEST_GPU_ENQUEUED, epoch, stream)) { return false; }
    if (hook && cancel_alias && layer.overlap_probe_offset != SIZE_MAX) {
        const auto * flag = reinterpret_cast<const source_flag *>(host + layer.overlap_probe_offset);
        ggml_cuda_moe_source_overlap_probe probe{i, &flag->value};
        if (!hook(hook_data, GGML_BACKEND_MOE_HYBRID_TEST_SOURCE_OVERLAP_PROBE, epoch, &probe)) { return false; }
    }
    int32_t status = GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
    // Additive import requires zero CPU values on GPU routes.
    if (!device_tail || layer.gpu_mask_offset == SIZE_MAX) {
        for (uint32_t route = 0; route < geometry.route_capacity;) {
            if (layer.classes[layer.weights[route]] == 2) { ++route; continue; }
            const uint32_t first = route++;
            while (route < geometry.route_capacity && layer.classes[layer.weights[route]] != 2) { ++route; }
            const size_t stride = layer.output_bytes / geometry.route_capacity;
            memset(host + layer.cpu_offset + size_t(first) * stride, 0, size_t(route - first) * stride);
        }
    }
    if (layer.cpu_count) {
        if (hook && !hook(hook_data, GGML_BACKEND_MOE_HYBRID_TEST_CPU_QUEUED, epoch, nullptr)) { return false; }
        ggml_backend_moe_cpu_region_binding_v1 binding{sizeof(binding),
            expert.routed_operation ? geometry.row_capacity : layer.cpu_count,
            expert.routed_operation ? geometry.route_capacity : layer.cpu_count,
            layer.cpu_ids.data(), layer.cpu_rows.data(), layer.cpu_destinations.data()};
        ggml_backend_moe_cpu_dynamic_input_v1 inputs[2] = {{host + layer.input_offset,
            layer.input_bytes, layer.descriptor->activation->nb[2]}, {}};
        ggml_backend_moe_cpu_output_v1 output{host + layer.cpu_offset,
            layer.output_bytes, layer.descriptor->output->nb[1]};
        ggml_backend_moe_cpu_execute_v1 job = {};
        job.struct_size = sizeof(job); job.epoch = cpu_epoch;
        job.graph_uid = expert.cpu_graph_uid; job.graph_generation = expert.cpu_graph_generation;
        job.source_generation = expert.cpu_source_generation;
        job.binding = &binding; job.dynamic_inputs = inputs; job.n_dynamic_inputs = 2;
        job.outputs = &output; job.n_outputs = 1;
        job.flags = expert.routed_operation ? 0 : GGML_BACKEND_MOE_CPU_EXECUTE_FLAG_V1_PRIVATE_OUTPUTS;
        ggml_backend_moe_cpu_execute_result_v1 result = {};
        result.struct_size = sizeof(result);
        const auto started = now_ns();
        ++counters.cpu_jobs;
        {
            source_cpu_range execute_trace(cpu_profile, "generic.cpu", epoch, i);
            status = expert.routed_operation ? cpu_api->execute_routed(cpu_service, expert.cpu_region, &job,
                layer.cpu_ownership.data(), geometry.route_capacity, &result) : cpu_api->execute(cpu_service, expert.cpu_region, &job, &result);
            if (!status && result.published_routes != layer.cpu_count) { status = GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_COMPUTE_FAILED; }
        }
        counters.cpu_us += (now_ns() - started) / 1000;
        ++counters.cpu_execute_calls;
        if (!status) { ++cpu_publications; ++scatter_completions; ++layer.branch_completed; }
    }
    control(i)->status = status;
    if (status || canceled.load(std::memory_order_acquire)) { return false; }
    if (hook && expert.routed_operation && layer.cpu_count) {
        auto probe = cpu_probe(i);
        if (!hook(hook_data, GGML_BACKEND_MOE_HYBRID_TEST_SOURCE_ROUTED_CPU_PROBE, epoch, &probe)) { return false; }
    }
    if (segmented) {
        if (cudaStreamWaitEvent(stream, layer.copied, 0) != cudaSuccess ||
                !launch_graph(size_t(i) * 4 + 2) ||
                cudaMemcpyAsync(data() + cpu_device_offset, host + layer.cpu_offset,
                    layer.output_bytes, cudaMemcpyHostToDevice, io) != cudaSuccess ||
                cudaEventRecord(layer.cpu_copied, io) != cudaSuccess || cudaStreamWaitEvent(stream, layer.cpu_copied, 0) != cudaSuccess ||
                !launch_graph(size_t(i) * 4 + 3)) { return false; }
        counters.cpu_upload_bytes += layer.output_bytes;
    }
    control(i)->cpu.store(1);
    service_trace.mark("generic.cpu-ready");
    if (hook && !hook(hook_data, GGML_BACKEND_MOE_HYBRID_TEST_CPU_JOINED, epoch, nullptr)) { return false; }
    return true;
}

bool core_session::execute_experts(uint32_t i, uint32_t kind) {
    const auto & layer = layers[i];
    const auto & expert = *layer.descriptor->expert;
    const auto capacity = expert.region.geometry.route_capacity;
    if (expert.generic_body) {
        const auto & geometry = expert.region.geometry;
        ggml_cuda_mmid_execution execution;
        execution.route_owners = data() + layer.device_plan_offset + layer.ownership_offset;
        execution.status = &runtime()->failed;
        execution.row_capacity = geometry.row_capacity; execution.routes_per_row = geometry.routes_per_row;
        execution.expert_count = geometry.expert_count; execution.owner = kind;
        for (const auto & operation : layer.body) {
            if (operation.bank != SIZE_MAX) {
                execution.expert_sources = reinterpret_cast<const void * const *>(data() + layer.device_plan_offset + layer.source_image_offset) +
                    operation.bank * geometry.expert_count;
                if (!ggml_cuda_mmid_execution_compute(*compute, operation.tensor, execution)) { return false; }
            } else if (!ggml_is_view(operation.tensor) && !ggml_cuda_moe_router_compute(*compute, operation.tensor)) { return false; }
        }
        return true;
    }
    if (expert.routed_operation) {
        const auto & geometry = expert.region.geometry;
        ggml_cuda_mmid_execution execution;
        execution.route_owners = data() + layer.device_plan_offset + layer.ownership_offset;
        execution.expert_sources = reinterpret_cast<const void * const *>(data() + layer.device_plan_offset + layer.source_image_offset);
        execution.status = &runtime()->failed;
        execution.row_capacity = geometry.row_capacity; execution.routes_per_row = geometry.routes_per_row;
        execution.expert_count = geometry.expert_count; execution.owner = kind;
        return ggml_cuda_mmid_execution_compute(*compute, const_cast<ggml_tensor *>(&layer.routed_output), execution);
    }
    auto plan = group_view(data() + layer.device_plan_offset + kind * group_bytes(capacity), capacity);
    const ggml_moe_reference_gpu_layout layout{int(expert.roles[0].type), int(expert.roles[2].type),
        int(expert.input_width), int(expert.hidden_width), int(expert.output_width),
        expert.roles[0].row_stride, expert.roles[1].row_stride, expert.roles[2].row_stride};
    const ggml_moe_reference_gpu_group group{plan.gate, plan.up, plan.down, plan.starts, plan.count,
        plan.tokens, plan.destinations};
    return ggml_moe_reference_gpu_execute(layout, group, capacity, capacity, data() + quant_offset,
        data() + scratch_offset, static_cast<float *>(layer.descriptor->output->data), stream);
}

bool core_session::prepare_capture_resources() {
#ifndef GGML_MOE_SOURCE_GRAPH
    return false;
#else
    const auto saved_captures = captures, saved_publication_captures = publication_captures;
    const auto saved_fused = fused_nodes, saved_images = image_groups, saved_emitted = emitted_images;
    for (;;) {
        ggml_cuda_set_device(parent->device);
        std::optional<ggml_cuda_cublas_request> request;
        try { if (capture_program()) { break; } }
        catch (const ggml_cuda_cublas_request & resource) { request = resource; }
        const auto & pool = static_cast<const ggml_cuda_pool_buffer &>(*compute->pools[parent->device][0]);
        const size_t required = pool.required;
        const bool grow = required > pool.capacity;
        if (unsupported || (!grow && !request) || pool.used || !wait_stream(stream, now_ns() + 5'000'000'000ull)) { return false; }
        for (auto & executable : executables) {
            if (executable && cudaGraphExecDestroy(executable) != cudaSuccess) { return false; }
            executable = nullptr;
        }
        for (auto & graph : graphs) {
            if (graph && cudaGraphDestroy(graph) != cudaSuccess) { return false; }
            graph = nullptr;
        }
        executables.clear(); graphs.clear(); publication_targets.clear(); publication_graph = SIZE_MAX;
        captures = saved_captures; publication_captures = saved_publication_captures;
        fused_nodes = saved_fused; image_groups = saved_images; emitted_images = saved_emitted;
        counters.window_captures = captures;
        if (grow) {
            const size_t retained = device_bytes - pool_arena_bytes;
            if (required > SIZE_MAX - retained || program->storage_bytes() > SIZE_MAX - retained - required ||
                    (config.device_bytes && retained + required + program->storage_bytes() > config.device_bytes)) { return false; }
            compute->pools[parent->device][0].reset();
            if (pool_arena) { ggml_backend_buffer_free(pool_arena); pool_arena = nullptr; }
            pool_arena_bytes = 0; device_bytes = retained;
            std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
                ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(config.backend), required), ggml_backend_buffer_free);
            if (!buffer) { return false; }
            const size_t allocated = ggml_backend_buffer_get_size(buffer.get());
            if (allocated < required || allocated > SIZE_MAX - retained || program->storage_bytes() > SIZE_MAX - retained - allocated ||
                    (config.device_bytes && retained + allocated + program->storage_bytes() > config.device_bytes)) { return false; }
            compute->pools[parent->device][0] = std::make_unique<ggml_cuda_pool_buffer>(
                static_cast<uint8_t *>(ggml_backend_buffer_get_base(buffer.get())), allocated);
            pool_arena = buffer.release(); pool_arena_bytes = allocated; device_bytes = retained + allocated;
        }
        if (request) {
            const int device = request->device, stream_no = request->stream_no;
            if (device < 0 || device >= ggml_cuda_info().device_count || stream_no < 0 || stream_no >= GGML_CUDA_MAX_STREAMS) { return false; }
            const size_t bytes = ggml_cuda_cublas_workspace_size(ggml_cuda_info().devices[device].cc);
            if (bytes > SIZE_MAX - device_bytes || program->storage_bytes() > SIZE_MAX - device_bytes - bytes ||
                    (config.device_bytes && device_bytes + bytes + program->storage_bytes() > config.device_bytes)) { return false; }
            ggml_backend_buffer_ptr buffer(bytes ? ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(device), bytes) : nullptr);
            if (bytes && !buffer) { return false; }
            const size_t allocated = buffer ? ggml_backend_buffer_get_size(buffer.get()) : 0;
            if (allocated < bytes || allocated > SIZE_MAX - device_bytes || program->storage_bytes() > SIZE_MAX - device_bytes - allocated ||
                    (config.device_bytes && device_bytes + allocated + program->storage_bytes() > config.device_bytes)) { return false; }
            auto * workspace = buffer ? ggml_backend_buffer_get_base(buffer.get()) : nullptr;
            if (buffer) { library_workspaces.push_back(std::move(buffer)); }
            device_bytes += allocated; library_bytes += allocated;
            if (!ggml_cuda_prepare_cublas(*compute, device, stream_no, workspace, bytes)) { return false; }
            ++library_preparations;
        }
        ++resource_recaptures;
        (void) cudaGetLastError();
    }
    const auto & pool = static_cast<const ggml_cuda_pool_buffer &>(*compute->pools[parent->device][0]);
    fprintf(stderr, "moe-source-core-resources: program_instance=%llu pool_estimate=%zu pool_capacity=%zu pool_peak=%zu recaptures=%llu extra_bytes=%zu library_preparations=%llu library_bytes=%zu\n",
        (unsigned long long) program_instance, pool_bytes, pool.capacity, pool.peak,
        (unsigned long long) resource_recaptures, pool_arena_bytes, (unsigned long long) library_preparations, library_bytes);
    return true;
#endif
}

bool core_session::capture_program() {
#ifndef GGML_MOE_SOURCE_GRAPH
    return false;
#else
    struct capture_guard {
        ggml_backend_cuda_context & context;
        bool active = ggml_cuda_moe_fidelity_capture_enter();
        bool previous;
        capture_guard(ggml_backend_cuda_context & context) : context(context), previous(context.capture_resource_requests) {
            if (active) { context.capture_resource_requests = true; }
        }
        ~capture_guard() {
            context.capture_resource_requests = previous;
            if (active) { ggml_cuda_moe_fidelity_capture_leave(); }
        }
    } guard(*compute);
    if (!guard.active) { return false; }
    const size_t graph_count = segmented ? layers.size() * 4 + 1 : 1;
    graphs.reserve(graph_count);
    executables.reserve(graph_count);
    for (const auto & layer : layers) {
        if (!validate_operations(layer.prelude) || !validate_operations(layer.overlap)) { return false; }
    }
    if (!validate_operations(epilogue)) { return false; }
    bool tail_rejected = false;
    const auto make = [&](const auto & emit, bool publication = false, bool from_device = false) {
        cudaGraph_t graph = nullptr;
        if (cudaGraphCreate(&graph, 0) != cudaSuccess) { return false; }
        graphs.push_back(graph);
        std::vector<cudaGraphNode_t> dependencies;
        if (!emit(graph, dependencies)) { return false; }
        cudaGraphExec_t executable = nullptr;
        const auto instantiated = cudaGraphInstantiate(&executable, graph, from_device ? cudaGraphInstantiateFlagDeviceLaunch : 0);
        if (instantiated != cudaSuccess) {
            tail_rejected |= from_device && (instantiated == cudaErrorNotSupported || instantiated == cudaErrorInvalidValue);
            unsupported |= !from_device && instantiated == cudaErrorNotSupported;
            fprintf(stderr, "moe-source-core-preparation: stage=instantiate status=%d\n", int(instantiated));
            return false;
        }
        executables.push_back(executable);
        if (publication) { ++publication_captures; }
        else { ++captures; counters.window_captures = captures; }
        return cudaGraphUpload(executable, stream) == cudaSuccess;
    };
    const auto finish = [&](bool ok) {
        if (!ok || !publication_packet) { return ok; }
        for (const auto * original : program->public_outputs()) {
            if (program->find(original)->data != original->data) { publication_targets.emplace_back(original, original->data); }
        }
        if (publication_targets.empty()) { return true; }
        const auto index = executables.size();
        const bool ready = make([&](cudaGraph_t graph, std::vector<cudaGraphNode_t> & dependencies) {
            return capture(graph, stream, dependencies, [&] {
                for (const auto * original : program->public_outputs()) {
                    const auto * output = program->find(original);
                    if (output->data == original->data) { continue; }
                    if (cudaMemcpyAsync(original->data, output->data, ggml_nbytes(output), cudaMemcpyDeviceToDevice, stream) != cudaSuccess) { return false; }
                }
                return true;
            });
        }, true);
        if (ready) { publication_graph = index; }
        return ready;
    };
    const auto clear_output = [&](uint32_t i) {
        const auto & layer = layers[i];
        if (device_tail && layer.gpu_mask_offset != SIZE_MAX) { return true; }
        return cudaMemsetAsync(layer.descriptor->output->data, 0, ggml_nbytes(layer.descriptor->output), stream) == cudaSuccess;
    };
    const auto overlap = [&](uint32_t i) {
        const auto & layer = layers[i];
        if (!emit_operations(layer.overlap)) { return false; }
        if (!layer.overlap.empty() && !clear_output(i)) { return false; }
        if (cancel_alias && layer.overlap_probe_offset != SIZE_MAX) {
            publish_overlap<<<1, 1, 0, stream>>>(reinterpret_cast<source_flag *>(cancel_alias + layer.overlap_probe_offset), runtime());
        }
        return cudaGetLastError() == cudaSuccess;
    };
    const auto prefix = [&](uint32_t i) {
        const auto & layer = layers[i];
        const auto & expert = *layer.descriptor->expert;
        const auto & geometry = expert.region.geometry;
        if (!emit_operations(layer.prelude) || (layer.overlap.empty() && !clear_output(i))) { return false; }
        if (!segmented) {
            publish_ready<<<1, 1024, 0, stream>>>(control(i, true), static_cast<const float *>(layer.descriptor->activation->data),
                layer.input_bytes / sizeof(float), static_cast<const int32_t *>(layer.descriptor->ids->data),
                layer.descriptor->ids->nb[1], geometry.routes_per_row, geometry.route_capacity,
                reinterpret_cast<float *>(alias + layer.input_offset), reinterpret_cast<int32_t *>(alias + layer.ids_offset), layer.ids_stride);
            return cudaGetLastError() == cudaSuccess;
        }
        return cudaMemcpyAsync(host + layer.input_offset, layer.descriptor->activation->data,
                layer.input_bytes, cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
            cudaMemcpy2DAsync(host + layer.ids_offset, layer.ids_stride,
                layer.descriptor->ids->data, layer.descriptor->ids->nb[1],
                size_t(geometry.routes_per_row) * sizeof(int32_t), geometry.row_capacity,
                cudaMemcpyDeviceToHost, stream) == cudaSuccess;
    };
    const auto quantize = [&](uint32_t i) {
        const auto & layer = layers[i];
        const auto & expert = *layer.descriptor->expert;
        if (expert.routed_operation || expert.generic_body) { return true; }
        return ggml_moe_reference_gpu_quantize(static_cast<const float *>(layer.descriptor->activation->data),
            data() + quant_offset, size_t(expert.region.geometry.row_capacity) * expert.input_width, stream);
    };
    const auto join_cpu = [&](uint32_t i, const float * input, source_control * next, cudaGraphConditionalHandle handle) {
        const auto & layer = layers[i];
        const auto & expert = *layer.descriptor->expert;
        const auto & geometry = expert.region.geometry;
        const auto resident = group_view(data() + layer.device_plan_offset, geometry.route_capacity);
        const auto selected = group_view(data() + layer.device_plan_offset + group_bytes(geometry.route_capacity), geometry.route_capacity);
        const size_t values = size_t(geometry.route_capacity) * expert.output_width;
        if (expert.routed_operation) {
            import_routed_cpu<<<unsigned((values + 255) / 256), 256, 0, stream>>>(static_cast<float *>(layer.descriptor->output->data), input,
                values, expert.output_width, geometry.routes_per_row, layer.descriptor->output->nb[1] / sizeof(float),
                layer.descriptor->output->nb[2] / sizeof(float), runtime(), resident.count, selected.count, next, handle);
        } else {
            import_cpu<<<unsigned((values + 255) / 256), 256, 0, stream>>>(static_cast<float *>(layer.descriptor->output->data), input,
                values, runtime(), resident.count, selected.count, next, handle);
        }
        return cudaGetLastError() == cudaSuccess;
    };
    const auto epilogue_body = [&](cudaGraph_t graph, std::vector<cudaGraphNode_t> & dependencies, cudaGraphConditionalHandle handle = 0) {
        const auto emit = [&] { return emit_operations(epilogue); };
        return (segmented ? capture(graph, stream, dependencies, emit) :
            phase(graph, stream, dependencies, runtime(), nullptr, emit, handle)) &&
            capture(graph, stream, dependencies, [&] {
                return cudaMemcpyAsync(result(), runtime(), sizeof(source_runtime), cudaMemcpyDeviceToHost, stream) == cudaSuccess;
            });
    };
#if defined(GGML_CUDA_MOE_DEVICE_TAIL)
    if (device_tail) {
        auto * dispatch = reinterpret_cast<ggml_cuda_moe_source_tail_dispatch *>(data() + tail_offset);
        const auto old_captures = captures, old_fused = fused_nodes;
        const bool ready = [&] {
            if (!make([&](cudaGraph_t graph, std::vector<cudaGraphNode_t> & dependencies) {
                    return capture(graph, stream, dependencies, [&] {
                        return ggml_cuda_moe_source_tail_start(runtime(), control(0, true), dispatch, stream) == cudaSuccess;
                    });
                })) { return false; }
            for (uint32_t i = 0; i < layers.size(); ++i) {
                if (!make([&](cudaGraph_t graph, std::vector<cudaGraphNode_t> & dependencies) {
                        return capture(graph, stream, dependencies, [&] {
                            const auto & layer = layers[i];
                            const auto & expert = *layer.descriptor->expert;
                            const auto capacity = expert.region.geometry.route_capacity;
                            if (!prefix(i) || !overlap(i) || !quantize(i)) { return false; }
                            wait_phase<<<1, 1, 0, stream>>>(control(i, true), runtime(), 0);
                            import_plan<<<unsigned((layer.plan_bytes + 255) / 256), 256, 0, stream>>>(data() + layer.device_plan_offset,
                                alias + layer.host_plan_offset, layer.plan_bytes, control(i, true), runtime());
                            if (!execute_experts(i, 0)) { return false; }
                            const auto resident = group_view(data() + layer.device_plan_offset, capacity);
                            wait_phase<<<1, 1, 0, stream>>>(control(i, true), runtime(), 1);
                            const auto selected = group_view(data() + layer.device_plan_offset + group_bytes(capacity), capacity);
                            if (kernel_copy_enabled) {
                                auto * plan = copy_plan(data() + layer.device_plan_offset, capacity);
                                copy_selected<<<unsigned(ggml_cuda_info().devices[parent->device].nsm) * 4, 256, 0, stream>>>(plan, runtime(), selected.count);
                            } else { guard_count<<<1, 1, 0, stream>>>(selected.count, runtime()); }
                            if (!execute_experts(i, 1)) { return false; }
                            wait_phase<<<1, 1, 0, stream>>>(control(i, true), runtime(), 2);
                            const size_t values = size_t(capacity) * expert.output_width;
                            if (expert.routed_operation) {
                                return ggml_cuda_moe_source_tail_import(static_cast<float *>(layer.descriptor->output->data),
                                    reinterpret_cast<const float *>(alias + layer.cpu_offset), values, runtime(), resident.count, selected.count,
                                    control(std::min(size_t(i) + 1, layers.size() - 1), true), dispatch + i + 1, stream,
                                    expert.output_width, expert.region.geometry.routes_per_row, layer.descriptor->output->nb[1] / sizeof(float),
                                    layer.descriptor->output->nb[2] / sizeof(float)) == cudaSuccess;
                            }
                            return ggml_cuda_moe_source_tail_import(static_cast<float *>(layer.descriptor->output->data),
                                reinterpret_cast<const float *>(alias + layer.cpu_offset), values, runtime(), resident.count, selected.count,
                                control(std::min(size_t(i) + 1, layers.size() - 1), true), dispatch + i + 1, stream,
                                layer.gpu_mask_offset == SIZE_MAX ? 0 : expert.output_width, 0, 0, 0,
                                layer.gpu_mask_offset == SIZE_MAX ? nullptr : reinterpret_cast<const uint32_t *>(data() + layer.device_plan_offset + layer.gpu_mask_offset)) == cudaSuccess;
                        });
                    }, false, true)) { return false; }
            }
            if (!make([&](cudaGraph_t graph, std::vector<cudaGraphNode_t> & dependencies) {
                    return capture(graph, stream, dependencies, [&] {
                        return emit_operations(epilogue) && ggml_cuda_moe_source_tail_finish(runtime(), dispatch + layers.size() + 1, stream) == cudaSuccess;
                    });
                }, false, true) || !make([&](cudaGraph_t graph, std::vector<cudaGraphNode_t> & dependencies) {
                    return capture(graph, stream, dependencies, [&] {
                        return cudaMemcpyAsync(result(), runtime(), sizeof(source_runtime), cudaMemcpyDeviceToHost, stream) == cudaSuccess;
                    });
                }, false, true)) { return false; }
            auto * handles = reinterpret_cast<ggml_cuda_moe_source_tail_dispatch *>(host + host_tail_offset);
            for (size_t i = 0; i < layers.size() + 2; ++i) {
                new (handles + i) ggml_cuda_moe_source_tail_dispatch{executables[i + 1], executables.back()};
            }
            if (cudaMemcpyAsync(dispatch, handles, (layers.size() + 2) * sizeof(handles[0]), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
                    !wait_stream(stream, now_ns() + 5'000'000'000ull)) { return false; }
            return finish(true);
        }();
        if (ready) {
            fprintf(stderr, "moe-source-core-dispatch: program_instance=%llu mode=device-tail phases=%zu graphs=%zu\n",
                (unsigned long long) program_instance, layers.size(), executables.size());
            return true;
        }
        if (!tail_rejected || !wait_stream(stream, now_ns() + 5'000'000'000ull)) { return false; }
        for (auto & executable : executables) {
            if (executable && cudaGraphExecDestroy(executable) != cudaSuccess) { return false; }
            executable = nullptr;
        }
        for (auto & graph : graphs) {
            if (graph && cudaGraphDestroy(graph) != cudaSuccess) { return false; }
            graph = nullptr;
        }
        executables.clear(); graphs.clear(); captures = old_captures; fused_nodes = old_fused;
        counters.window_captures = captures;
        (void) cudaGetLastError();
        device_tail = false;
        fprintf(stderr, "moe-source-core-dispatch: program_instance=%llu mode=conditional reason=device-tail-capability\n", (unsigned long long) program_instance);
    }
#endif
    if (segmented) {
        for (uint32_t i = 0; i < layers.size(); ++i) {
            if (!make([&](cudaGraph_t graph, std::vector<cudaGraphNode_t> & dependencies) {
                    return capture(graph, stream, dependencies, [&] { return prefix(i); });
                }) || !make([&](cudaGraph_t graph, std::vector<cudaGraphNode_t> & dependencies) {
                    return capture(graph, stream, dependencies, [&] {
                        if (!overlap(i) || !quantize(i) || !execute_experts(i, 0)) { return false; }
                        return cudaGetLastError() == cudaSuccess;
                    });
                }) || !make([&](cudaGraph_t graph, std::vector<cudaGraphNode_t> & dependencies) {
                    return capture(graph, stream, dependencies, [&] {
                        if (!execute_experts(i, 1)) { return false; }
                        return cudaGetLastError() == cudaSuccess;
                    });
                }) || !make([&](cudaGraph_t graph, std::vector<cudaGraphNode_t> & dependencies) {
                    return capture(graph, stream, dependencies, [&] {
                        const auto & layer = layers[i];
                        return join_cpu(i, reinterpret_cast<const float *>(data() + cpu_device_offset), nullptr, 0);
                    });
                })) { return false; }
        }
        return finish(make(epilogue_body));
    }
    return finish(make([&](cudaGraph_t graph, std::vector<cudaGraphNode_t> & dependencies) {
        std::vector<cudaGraphConditionalHandle> handles(layers.size());
        for (auto & handle : handles) {
            if (cudaGraphConditionalHandleCreate(&handle, graph, 0, cudaGraphCondAssignDefault) != cudaSuccess) { return false; }
        }
        for (uint32_t i = 0; i < layers.size(); ++i) {
            if (!phase(graph, stream, dependencies, runtime(), control(i, true), [&] {
                const auto & layer = layers[i];
                const auto & expert = *layer.descriptor->expert;
                const auto capacity = expert.region.geometry.route_capacity;
                if (!prefix(i)) { return false; }
                if (!overlap(i) || !quantize(i)) { return false; }
                wait_phase<<<1, 1, 0, stream>>>(control(i, true), runtime(), 0);
                import_plan<<<unsigned((layer.plan_bytes + 255) / 256), 256, 0, stream>>>(data() + layer.device_plan_offset,
                    alias + layer.host_plan_offset, layer.plan_bytes, control(i, true), runtime());
                if (!execute_experts(i, 0)) { return false; }
                auto resident = group_view(data() + layer.device_plan_offset, capacity);
                wait_phase<<<1, 1, 0, stream>>>(control(i, true), runtime(), 1);
                auto selected = group_view(data() + layer.device_plan_offset + group_bytes(capacity), capacity);
                if (kernel_copy_enabled) {
                    auto * plan = copy_plan(data() + layer.device_plan_offset, capacity);
                    copy_selected<<<unsigned(ggml_cuda_info().devices[parent->device].nsm) * 4, 256, 0, stream>>>(plan, runtime(), selected.count);
                } else { guard_count<<<1, 1, 0, stream>>>(selected.count, runtime()); }
                if (!execute_experts(i, 1)) { return false; }
                wait_phase<<<1, 1, 0, stream>>>(control(i, true), runtime(), 2);
                return join_cpu(i, reinterpret_cast<const float *>(alias + layer.cpu_offset),
                    control(std::min(size_t(i) + 1, layers.size() - 1), true), handles[i]);
            }, i ? handles[i - 1] : 0, i == 0)) { return false; }
        }
        return epilogue_body(graph, dependencies, handles.back());
    }));
#endif
}

static void CUDART_CB input_ready(void * context) { static_cast<source_control *>(context)->ready.store(1); }

bool core_session::launch(uint32_t segment) {
    ggml_cuda_set_device(parent->device);
    if (!segmented) { return segment == 0 && launch_graph(0); }
    if (segment == layers.size()) { return launch_graph(size_t(layers.size()) * 4); }
    auto & layer = layers[segment];
    return launch_graph(size_t(segment) * 4) &&
        cudaEventRecord(layer.ready, stream) == cudaSuccess && cudaStreamWaitEvent(io, layer.ready, 0) == cudaSuccess &&
        cudaLaunchHostFunc(io, input_ready, control(segment)) == cudaSuccess;
}

ggml_moe_caller_progress core_session::progress() const {
    const auto main_status = cudaStreamQuery(stream);
    const auto io_status = cudaStreamQuery(io);
    if ((main_status != cudaSuccess && main_status != cudaErrorNotReady) ||
            (io_status != cudaSuccess && io_status != cudaErrorNotReady)) { return ggml_moe_caller_progress::failed; }
    return main_status == cudaSuccess && io_status == cudaSuccess ? ggml_moe_caller_progress::complete : ggml_moe_caller_progress::pending;
}

int32_t core_session::drain(bool internal) {
    source_host_timer timer(internal ? internal_drain_wall_ns : external_drain_wall_ns);
    const uint64_t expiry = now_ns() + 5'000'000'000ull;
    {
        std::unique_lock<std::mutex> lock(mutex);
        ++drains;
        if (internal && (!caller_active || caller_id != std::this_thread::get_id())) { return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_INVALID; }
        if (!internal) {
            if (caller_active && caller_id == std::this_thread::get_id()) { return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_BUSY; }
            ++external_drains;
            const bool exited = condition.wait_until(lock, std::chrono::steady_clock::time_point(std::chrono::nanoseconds(expiry)),
                [&] { return !caller_active; });
            if (!exited) { --external_drains; ++drain_failures; return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_TIMEOUT; }
        }
    }
    const auto done = [&](int32_t status) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!internal) { --external_drains; }
        if (status) { ++drain_failures; }
        return status;
    };
    std::unique_lock<std::timed_mutex> terminal_lock(terminal_mutex, std::defer_lock);
    if (!terminal_lock.try_lock_until(std::chrono::steady_clock::time_point(std::chrono::nanoseconds(expiry)))) {
        return done(GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_TIMEOUT);
    }
    if (prepared && !resources_live && !dispatch && !hook) { return done(GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK); }
    if (parent) { ggml_cuda_set_device(parent->device); }
    if (!wait_stream(stream, expiry) || !wait_stream(io, expiry)) { return done(GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_TIMEOUT); }
    if (!internal && cpu_api && cpu_service) {
        ggml_backend_moe_cpu_service_state_v1 state = {};
        state.struct_size = sizeof(state);
        if (cpu_api->state(cpu_service, &state) || state.active_jobs || cpu_api->drain(cpu_service)) {
            return done(GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_FAILED);
        }
    }
    if (!finish_owner()) { return done(GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_FAILED); }
    resources_live = false;
    return done(GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK);
}

bool core_session::metadata_valid(const ggml_cgraph * graph, const ggml_graph_execution_certificate & certificate,
        void * const * regions, uint32_t count, bool check_program) const {
    source_host_timer timer(metadata_validation_wall_ns, metadata_diagnostic);
    if (!graph || !regions || !count || count > config.max_regions || count > compute_regions.capacity()) { return false; }
    if (check_program && prepared && (!program || !program->matches(graph, certificate))) {
        fprintf(stderr, "moe-source-core-fault: stage=program_binding epoch=%llu split_uid=%llu\n",
            (unsigned long long) epoch, (unsigned long long) graph->uid);
        return false;
    }
    if (!source_certificate_valid(certificate, graph->uid)) {
        fprintf(stderr, "moe-source-core-fault: stage=dispatch_certificate epoch=%llu split_uid=%llu\n",
            (unsigned long long) epoch, (unsigned long long) graph->uid);
        return false;
    }
    for (uint32_t i = 0; i < count; ++i) {
        const auto * region = static_cast<const core_region *>(regions[i]);
        if (!region || region->owner != this) { return false; }
        const auto & descriptor = region->expert.region;
        if (memcmp(&descriptor.certificate, &certificate, sizeof(certificate)) ||
                descriptor.source_graph_uid != certificate.source_graph_uid || descriptor.split_graph_uid != graph->uid ||
                descriptor.owner_generation != certificate.owner_generation || !descriptor.geometry.row_capacity ||
                descriptor.geometry.row_capacity > certificate.n_rows ||
                !descriptor.allocator_generation || descriptor.allocator_generation != region->expert.cpu_graph_generation) {
            fprintf(stderr, "moe-source-core-fault: stage=region_certificate epoch=%llu index=%u prepared_magic=%u dispatched_magic=%u prepared_owner=%llu dispatched_owner=%llu prepared_split=%llu dispatched_split=%llu allocator=%llu cpu_graph_generation=%llu region_rows=%u prepared_rows=%u dispatched_rows=%u prepared_sequences=%u dispatched_sequences=%u\n",
                (unsigned long long) epoch, i, descriptor.certificate.magic, certificate.magic,
                (unsigned long long) descriptor.certificate.owner_generation, (unsigned long long) certificate.owner_generation,
                (unsigned long long) descriptor.split_graph_uid, (unsigned long long) graph->uid,
                (unsigned long long) descriptor.allocator_generation, (unsigned long long) region->expert.cpu_graph_generation,
                descriptor.geometry.row_capacity, descriptor.certificate.n_rows, certificate.n_rows,
                descriptor.certificate.n_sequences, certificate.n_sequences);
            return false;
        }
    }
    return true;
}

ggml_status core_session::replay(ggml_cgraph * graph, void * const * regions, uint32_t count) {
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (!caller_active || caller_id != std::this_thread::get_id() || closed) {
            fprintf(stderr, "moe-source-core-fault: stage=replay_admission closed=%d caller_active=%d\n", int(closed), int(caller_active));
            return GGML_STATUS_FAILED;
        }
        {
            source_host_timer timer(binding_validation_wall_ns, metadata_diagnostic);
            if (prepared && (!program || !program->matches(graph))) {
                rejected_before_effects = !resources_live && !dispatch && !effect_failure;
                fprintf(stderr, "moe-source-core-fault: stage=program_binding epoch=%llu split_uid=%llu rejection_safe=%d\n",
                    (unsigned long long) epoch, (unsigned long long) graph->uid, int(rejected_before_effects));
                return GGML_STATUS_FAILED;
            }
        }
        ++epoch;
        if (!epoch) { --epoch; return GGML_STATUS_FAILED; }
        // Cancellation epochs must not restart when a shared service changes programs.
        static std::atomic<uint64_t> next_cpu_epoch{1};
        auto next = next_cpu_epoch.load(std::memory_order_relaxed);
        do {
            if (next == UINT64_MAX) { return GGML_STATUS_FAILED; }
        } while (!next_cpu_epoch.compare_exchange_weak(next, next + 1, std::memory_order_relaxed));
        cpu_epoch = next;
    }
    // The program was checked above before advancing the epoch.
    if (!metadata_valid(graph, graph->execution_certificate, regions, count, false)) { return reject_before_effects(); }
    compute_regions.clear();
    for (uint32_t i = 0; i < count; ++i) { compute_regions.push_back(static_cast<core_region *>(regions[i])); }
    ggml_cuda_set_device(parent->device);
    if (normal_fallback) { resources_live = true; return ggml_backend_graph_compute_async(config.backend, graph); }
    if (!prepared && !allocate(graph, compute_regions)) {
        if (unsupported && drain(true) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK) {
            normal_fallback = true;
            resources_live = true;
            fprintf(stderr, "moe-source-core-admission: provider=normal reason=unsupported-capability split_uid=%llu effects_started=0\n",
                (unsigned long long) graph->uid);
            return ggml_backend_graph_compute_async(config.backend, graph);
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            closed = true; effect_failure = true; stop();
        }
        (void) drain(true);
        return GGML_STATUS_FAILED;
    }
    source_host_timer replay_timer(replay_wall_ns);
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (closed || canceled.load()) { return GGML_STATUS_FAILED; }
        deadline = now_ns() + 2'000'000'000ull;
        if (supervisor_needs_wake) { condition.notify_all(); }
    }
    bool ok = acquire_owner(graph, false);
    for (uint32_t i = 0; i < layers.size() && ok; ++i) {
        auto & layer = layers[i];
        auto * bell = control(i);
        bell->ready.store(0); bell->plan.store(0); bell->copied.store(0); bell->cpu.store(0);
        bell->status = 0; bell->epoch = epoch;
        bell->stop.store(canceled.load(std::memory_order_acquire) ? 1 : 0);
        if (layer.overlap_probe_offset != SIZE_MAX) {
            reinterpret_cast<source_flag *>(host + layer.overlap_probe_offset)->store(0);
        }
        layer.epoch = epoch;
        layer.branch_expected = layer.branch_completed = 0;
        layer.copy_submitted = false;
    }
    *result() = {};
    result()->epoch = epoch;
    if (ok && segmented) { ok = cudaMemcpyAsync(runtime(), result(), sizeof(source_runtime), cudaMemcpyHostToDevice, stream) == cudaSuccess; }
    ggml_moe_caller_sample sample;
    if (ok) {
        const ggml_moe_caller_schedule schedule{this, uint32_t(layers.size()), segmented, deadline,
            [](void * s, uint32_t n) { return static_cast<core_session *>(s)->launch(n); },
            [](void * s, uint32_t n) { return static_cast<core_session *>(s)->ready(n); },
            [](void * s) { return static_cast<core_session *>(s)->canceled.load(std::memory_order_acquire); },
            [](void * s) { return static_cast<core_session *>(s)->progress(); },
            [](void * s, uint32_t n) { return static_cast<core_session *>(s)->serve(n); }};
        const auto status = ggml_moe_caller_run(schedule, sample);
        ok = status == ggml_moe_caller_status::complete;
        if (!ok) {
            fprintf(stderr, "moe-source-core-fault: stage=caller status=%d epoch=%llu jobs=%llu progress_probes=%llu launch_ns=%llu observation_ns=%llu\n",
                int(status), (unsigned long long) epoch, (unsigned long long) sample.jobs,
                (unsigned long long) sample.progress_probes, (unsigned long long) sample.launch_ns,
                (unsigned long long) sample.observation_ns);
        }
    }
    if (ok && config.profile_adaptation) {
        ok = wait_stream(stream, deadline) && wait_stream(io, deadline) &&
            !canceled.load(std::memory_order_acquire) && result()->epoch == epoch &&
            !result()->failed && result()->joined == layers.size() && apply_profiles(false, true);
        if (!ok) { fprintf(stderr, "moe-source-core-fault: stage=adapt epoch=%llu\n", (unsigned long long) epoch); }
    }
    if (!ok) { stop(); }
    const auto drained = drain(true);
    if (!drained) { ++counters.window_waits; }
    if (!drained) {
        gpu_completions += result()->resident + result()->selected;
        copy_completions += result()->kernel_layers;
        kernel_jobs_completed += result()->copy_jobs;
        kernel_bytes_completed += result()->copy_bytes;
        counters.h2d_bytes += result()->copy_bytes;
        for (const auto & layer : layers) {
            if (!layer.kernel_copy && layer.selected_count && layer.copy_submitted) { ++copy_completions; }
        }
    }
    if (ok && drained) {
        fprintf(stderr, "moe-source-core-fault: stage=drain epoch=%llu status=%d\n", (unsigned long long) epoch, drained);
        ok = false;
    }
    if (ok && (canceled.load(std::memory_order_acquire) || result()->epoch != epoch ||
            result()->failed || result()->joined != layers.size())) {
        fprintf(stderr, "moe-source-core-fault: stage=join epoch=%llu result_epoch=%llu canceled=%d failed=%u joined=%u layers=%zu\n",
            (unsigned long long) epoch, (unsigned long long) result()->epoch,
            int(canceled.load(std::memory_order_acquire)), result()->failed, result()->joined, layers.size());
        ok = false;
    }
    if (ok && hook) {
        for (uint32_t i = 0; i < layers.size() && ok; ++i) {
            const auto & layer = layers[i];
            const auto & expert = *layer.descriptor->expert;
            for (const auto & bank : expert.banks) {
                ggml_backend_moe_source_access_v1 access = {i, uint32_t(layers.size()), bank.source.tensor,
                    static_cast<const ggml_tensor *>(bank.source.witness), layer.probe_domain, layer.descriptor->ids,
                    layer.weights.data(), layer.distinct.data(), layer.classes.data(),
                    uint32_t(layer.distinct.size()), expert.region.geometry.route_capacity};
                if (!hook(hook_data, GGML_BACKEND_MOE_HYBRID_TEST_SOURCE_ACCESS, epoch, &access)) { ok = false; break; }
            }
        }
    }
    if (ok && retain_routed_outputs && hook) {
        for (uint32_t i = 0; i < layers.size() && ok; ++i) {
            if (!layers[i].descriptor->expert->routed_operation) { continue; }
            if (layers[i].cpu_count) {
                auto probe = cpu_probe(i);
                ok = hook(hook_data, GGML_BACKEND_MOE_HYBRID_TEST_SOURCE_ROUTED_PUBLICATION_PROBE, epoch, &probe);
            }
            if (ok) {
                auto probe = gpu_probe(i);
                ok = hook(hook_data, GGML_BACKEND_MOE_HYBRID_TEST_SOURCE_ROUTED_GPU_PUBLICATION_PROBE, epoch, &probe);
            }
        }
    }
    if (ok) {
        if (hook && !hook(hook_data, GGML_BACKEND_MOE_HYBRID_TEST_BEFORE_PUBLISH, epoch, stream)) { ok = false; }
    }
    if (ok) {
        uint32_t resident = 0, selected = 0;
        uint32_t kernel_layers = 0;
        uint64_t jobs = 0, bytes = 0;
        for (auto & layer : layers) {
            if (layer.kernel_copy) {
                const auto * plan = copy_plan(host + layer.host_plan_offset, layer.descriptor->expert->region.geometry.route_capacity);
                kernel_layers += plan->jobs != 0;
                jobs += plan->jobs;
                bytes += plan->bytes;
            }
            if (cancel_alias && layer.overlap_probe_offset != SIZE_MAX &&
                    reinterpret_cast<source_flag *>(host + layer.overlap_probe_offset)->load() != 1) { ok = false; break; }
            if (layer.selected_count) { ++selected; ++layer.branch_completed; }
            if (layer.resident_count) { ++resident; ++layer.branch_completed; }
            if (layer.branch_expected != layer.branch_completed) { ok = false; break; }
        }
        if (result()->resident != resident || result()->selected != selected) { ok = false; }
        if (result()->kernel_layers != kernel_layers || result()->copy_jobs != jobs || result()->copy_bytes != bytes) { ok = false; }
    }
    if (ok) {
        std::lock_guard<std::mutex> lock(mutex);
        ok = !closed && !canceled.load(std::memory_order_acquire);
    }
    if (ok) {
        source_host_timer timer(public_copy_wait_wall_ns);
        bool copied = false;
        if (publication_graph != SIZE_MAX) {
            for (const auto & target : publication_targets) {
                if (target.first->data != target.second || !target.first->buffer) { ok = false; break; }
            }
            if (ok) { resources_live = true; ok = launch_graph(publication_graph); copied = true; }
        } else for (const auto * original : program->public_outputs()) {
            const auto * output = program->find(original);
            if (output && output->data == original->data) { continue; }
            resources_live = true;
            if (!output || !output->data || !original->data || !original->buffer ||
                    cudaMemcpyAsync(original->data, output->data, ggml_nbytes(output), cudaMemcpyDeviceToDevice, stream) != cudaSuccess) { ok = false; break; }
            copied = true;
        }
        if (ok && copied) {
            ok = wait_stream(stream, now_ns() + 5'000'000'000ull);
            if (ok) { resources_live = false; }
        }
    }
    if (ok) { ++publications; }
    else {
        std::lock_guard<std::mutex> lock(mutex);
        effect_failure = true; closed = true; ++failures; ++counters.errors;
        stop();
    }
    counters.last_epoch = epoch;
    replay_timer.finish();
    snapshot_transport();
    if (!ok || replay_diagnostic) { summary(); }
    return ok ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
}

int32_t core_session::dispose(bool retire) {
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (!closed || caller_active || external_drains || (!retire && regions)) { return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_BUSY; }
    }
    const auto status = drain();
    if (status) {
        std::lock_guard<std::mutex> lock(mutex);
        summary("drain", status);
        return status;
    }
    if (!owner().complete_source_adaptation(true, now_ns() + 5'000'000'000ull)) { return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_TIMEOUT; }
    if (cpu_api && cpu_service && cpu_api->set_test_hook && cpu_api->set_test_hook(cpu_service, nullptr, nullptr)) {
        return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_FAILED;
    }
    {
        std::lock_guard<std::mutex> lock(mutex);
        stopping = true;
        condition.notify_all();
    }
    if (supervisor.joinable()) { supervisor.join(); }
    ggml_cuda_set_device(parent->device);
    for (auto & executable : executables) {
        if (executable && cudaGraphExecDestroy(executable) != cudaSuccess) { return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_FAILED; }
        executable = nullptr;
    }
    for (auto & graph : graphs) {
        if (graph && cudaGraphDestroy(graph) != cudaSuccess) { return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_FAILED; }
        graph = nullptr;
    }
    for (auto & layer : layers) {
        for (auto * event : {&layer.ready, &layer.copied, &layer.cpu_copied, &layer.plan_ready}) {
            if (*event && cudaEventDestroy(*event) != cudaSuccess) { return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_FAILED; }
            *event = nullptr;
        }
    }
    snapshot_transport();
    if (!owner().release_source_transport(&transport) || (compute && !ggml_cuda_moe_fidelity_release_cublas(*compute))) {
        return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_FAILED;
    }
    {
        std::lock_guard<std::mutex> lock(mutex);
        summary("drain", GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK);
    }
    compute.reset();
    library_workspaces.clear();
    if (pool_arena) { ggml_backend_buffer_free(pool_arena); pool_arena = nullptr; }
    ggml_free(fusion_context); fusion_context = nullptr; fusion_graph = nullptr;
    program.reset();
    leaf_resource_leases.clear(); resource_leases.clear(); owner_plan.reset();
    if (io && cudaStreamDestroy(io) != cudaSuccess) { return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_FAILED; }
    io = nullptr;
    if (host && cudaFreeHost(host) != cudaSuccess) { return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_FAILED; }
    host = alias = cancel_alias = nullptr; controls_ready.store(false);
    if (arena) { ggml_backend_buffer_free(arena); arena = nullptr; }
    layers.clear(); epilogue.clear();
    prepared = false;
    return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK;
}

void core_session::summary(const char * kind, int32_t drain_status) const {
    if (!program) { return; }
    source_host_timer timer(summary_wall_ns);
    if (metadata_diagnostic) {
        fprintf(stderr, "moe-source-metadata-summary: program_instance=%llu epochs=%llu preflight_wall_ns=%llu binding_validation_wall_ns=%llu metadata_validation_wall_ns=%llu\n",
            (unsigned long long) program_instance, (unsigned long long) epoch,
            (unsigned long long) preflight_wall_ns.load(std::memory_order_relaxed),
            (unsigned long long) binding_validation_wall_ns.load(std::memory_order_relaxed),
            (unsigned long long) metadata_validation_wall_ns.load(std::memory_order_relaxed));
    }
    const auto transport_snapshot = transport_stats;
    fprintf(stderr, "moe-source-publication-summary: program_instance=%llu packet=%u captures=%llu launches=%llu\n",
        (unsigned long long) program_instance, unsigned(publication_graph != SIZE_MAX),
        (unsigned long long) publication_captures, (unsigned long long) publication_launches);
    fprintf(stderr, "moe-source-fusion-summary: scope=program program_instance=%llu enabled=%u fused_nodes=%llu captures=%llu\n",
        (unsigned long long) program_instance, unsigned(fusion_graph != nullptr), (unsigned long long) fused_nodes, (unsigned long long) captures);
    fprintf(stderr, "moe-source-core-images: program_instance=%llu image_groups=%llu emitted_images=%llu workspace_bytes=%zu\n",
        (unsigned long long) program_instance, (unsigned long long) image_groups, (unsigned long long) emitted_images, image_bytes);
    fprintf(stderr, "moe-source-weighted-summary: scope=program program_instance=%llu enabled=%u matches=%llu complete=%llu closed=%llu ready=%llu aliases=%llu\n",
        (unsigned long long) program_instance, unsigned(weighted_enabled), (unsigned long long) weighted_matches,
        (unsigned long long) weighted_complete, (unsigned long long) weighted_closed,
        (unsigned long long) weighted_ready, (unsigned long long) weighted_aliases);
    fprintf(stderr, "moe-source-overlap-summary: scope=program program_instance=%llu enabled=%u layers=%zu operations=%zu epochs=%llu probe=%u probe_layers=%zu storage_bytes=%zu\n",
        (unsigned long long) program_instance, unsigned(overlap_enabled), program->overlap_layer_count(),
        program->overlap_operation_count(), (unsigned long long) (overlap_enabled ? publications : 0),
        unsigned(overlap_probe), overlap_probe ? program->overlap_layer_count() : 0, program->storage_bytes());
    fprintf(stderr, "moe-source-core-summary: scope=program kind=%s provider=source-core program_instance=%llu device=%d split_uid=%llu domain=%u arm=%s no_alias=%u rows=%u layers=%zu ordinary=%zu state_effects=%zu public_outputs=%zu epochs=%llu cpu_calls=%llu cpu_routes=%llu resident_routes=%llu selected_routes=%llu selected_bytes=%llu map_bytes=%llu cpu_published=%llu scatter_completed=%llu copy_expected=%llu copy_completed=%llu gpu_expected=%llu gpu_completed=%llu public_published=%llu captures=%llu old_window_calls=0 old_emitter_calls=0 failures=%llu drains=%llu drain_failures=%llu drain_status=%d retained=%u caller_active=%u coordinator=0 cpu_us=%llu prepare_wall_ns=%llu replay_wall_ns=%llu replay_acquire_map_wall_ns=%llu internal_drain_wall_ns=%llu public_copy_wait_wall_ns=%llu external_drain_wall_ns=%llu summary_wall_ns=%llu compact_plan_wall_ns=%llu selected_copy_submit_wall_ns=%llu transport_direct_calls=%llu transport_direct_bytes=%llu transport_catalog_staged_calls=%llu transport_catalog_staged_bytes=%llu transport_null_staged_calls=%llu transport_null_staged_bytes=%llu transport_tile_wait_wall_ns=%llu transport_materialize_wall_ns=%llu kernel_copy_enabled=%u kernel_copy_jobs_expected=%llu kernel_copy_jobs_completed=%llu kernel_copy_bytes_expected=%llu kernel_copy_bytes_completed=%llu\n",
        kind, (unsigned long long) program_instance, parent->device,
        (unsigned long long) program->certificate().split_graph_uid, program->certificate().domain, segmented ? "segmented" : "poll",
        unsigned(noalias), program->certificate().n_rows, layers.size(), program->operation_count(), program->owner_write_count(),
        program->public_outputs().size(), (unsigned long long) epoch, (unsigned long long) counters.cpu_execute_calls,
        (unsigned long long) counters.cpu_routes, (unsigned long long) counters.resident_routes,
        (unsigned long long) counters.transfer_routes, (unsigned long long) counters.h2d_bytes, (unsigned long long) map_bytes,
        (unsigned long long) cpu_publications, (unsigned long long) scatter_completions, (unsigned long long) copy_expected,
        (unsigned long long) copy_completions, (unsigned long long) gpu_expected,
        (unsigned long long) gpu_completions, (unsigned long long) publications, (unsigned long long) captures,
        (unsigned long long) failures, (unsigned long long) drains, (unsigned long long) drain_failures,
        drain_status, unsigned(resources_live), unsigned(caller_active), (unsigned long long) counters.cpu_us,
        (unsigned long long) prepare_wall_ns.load(std::memory_order_relaxed), (unsigned long long) replay_wall_ns.load(std::memory_order_relaxed),
        (unsigned long long) replay_acquire_map_wall_ns.load(std::memory_order_relaxed), (unsigned long long) internal_drain_wall_ns.load(std::memory_order_relaxed),
        (unsigned long long) public_copy_wait_wall_ns.load(std::memory_order_relaxed), (unsigned long long) external_drain_wall_ns.load(std::memory_order_relaxed),
        (unsigned long long) summary_wall_ns.load(std::memory_order_relaxed),
        (unsigned long long) compact_plan_wall_ns.load(std::memory_order_relaxed), (unsigned long long) selected_copy_submit_wall_ns.load(std::memory_order_relaxed),
        (unsigned long long) transport_snapshot.direct_calls, (unsigned long long) transport_snapshot.direct_bytes,
        (unsigned long long) transport_snapshot.catalog_staged_calls, (unsigned long long) transport_snapshot.catalog_staged_bytes,
        (unsigned long long) transport_snapshot.null_staged_calls, (unsigned long long) transport_snapshot.null_staged_bytes,
        (unsigned long long) transport_snapshot.tile_wait_wall_ns, (unsigned long long) transport_snapshot.materialize_wall_ns, unsigned(kernel_copy_enabled),
        (unsigned long long) kernel_jobs_expected, (unsigned long long) kernel_jobs_completed,
        (unsigned long long) kernel_bytes_expected, (unsigned long long) kernel_bytes_completed);
    fflush(stderr);
}

static int32_t source_create(const ggml_backend_moe_hybrid_config_v1 * config, void ** output) try {
    if (!output) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT; }
    *output = nullptr;
    const auto & selection = ggml_moe_fidelity_selection();
    const auto * arm = getenv("GGML_MOE_FIDELITY_ARM");
    if (!config || config->struct_size != sizeof(*config) || !config->backend || !config->source_owner ||
            config->executor != GGML_BACKEND_MOE_HYBRID_EXECUTOR_V1_FIDELITY || config->admission_quota ||
            config->demand_admission || config->profile_adaptation > 2 ||
            (config->profile_adaptation && !config->n_profiles && !config->n_statistics) || !config->n_threads || !config->max_regions || !selection.valid ||
            !selection.source_pool || (arm && strcmp(arm, "poll") && strcmp(arm, "segmented"))) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
    }
#ifndef GGML_MOE_SOURCE_GRAPH
    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
#else
    auto session = std::make_unique<core_session>();
    const auto * metadata_diagnostic = getenv("GGML_MOE_SOURCE_METADATA_DIAGNOSTIC");
    session->metadata_diagnostic = metadata_diagnostic && !strcmp(metadata_diagnostic, "1");
    static std::atomic<uint64_t> next_instance{1};
    session->program_instance = next_instance.fetch_add(1, std::memory_order_relaxed);
    session->config = *config;
    if (config->n_profiles > GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS || (config->n_profiles && !config->profiles) ||
            (config->n_profiles && config->n_statistics) || !ggml_moe_source_scores_valid(config->statistics, config->statistics_scores, config->n_statistics)) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
    }
    uint64_t profile_experts = 0;
    for (uint32_t i = 0; i < config->n_profiles; ++i) {
        const auto & profile = config->profiles[i];
        if (!profile.down || !profile.experts || !profile.n_experts || profile.n_experts > 65536 ||
                profile.down->ne[2] <= 0 || profile.down->ne[2] > 65536 ||
                profile.n_experts > uint64_t(profile.down->ne[2]) || session->profiles.count(profile.down)) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
        }
        profile_experts += profile.n_experts;
        if (profile_experts > (1u << 22)) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY; }
        std::vector<uint8_t> seen(profile.down->ne[2], 0);
        for (uint32_t j = 0; j < profile.n_experts; ++j) {
            const int32_t expert = profile.experts[j];
            if (expert < 0 || expert >= profile.down->ne[2] || seen[expert]++) {
                return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
            }
        }
        session->profiles.emplace(profile.down, std::vector<int32_t>(profile.experts, profile.experts + profile.n_experts));
    }
    session->config.profiles = nullptr;
    for (uint32_t i = 0; i < config->n_statistics; ++i) {
        const auto & source = config->statistics[i];
        session->statistics[source.tensor].push_back({source.domain, source.observations,
            std::vector<uint64_t>(source.counts, source.counts + source.n_experts), config->statistics_scores ?
                std::vector<double>(config->statistics_scores[i], config->statistics_scores[i] + source.n_experts) : std::vector<double>{}});
    }
    session->config.statistics = nullptr;
    session->config.statistics_scores = nullptr;
    const auto * diagnostic = getenv("GGML_MOE_SOURCE_REPLAY_DIAGNOSTIC");
    session->replay_diagnostic = diagnostic && strcmp(diagnostic, "0");
    const auto * retain = getenv("GGML_TEST_MOE_RETAIN_ROUTED_OUTPUTS");
    session->retain_routed_outputs = retain && !strcmp(retain, "1");
    session->source_owner = *config->source_owner;
    session->config.source_owner = &session->source_owner;
    session->parent = static_cast<ggml_backend_cuda_context *>(config->backend->context);
    session->compute_regions.reserve(config->max_regions);
    if (!session->parent || !session->parent->moe_grouped_context) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_MISSING_SOURCE; }
    ggml_cuda_set_device(session->parent->device);
    session->stream = session->parent->stream();
    *output = session.release();
    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
#endif
} catch (...) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY; }

static int32_t source_prepare(void * opaque, const ggml_backend_moe_hybrid_region_v1 * descriptor,
        const ggml_backend_moe_cpu_region_service_api_v1 * cpu_api, ggml_backend_moe_cpu_service_v1_t cpu,
        const ggml_backend_moe_cpu_prepared_region_v1_t * cpu_regions, void ** output) try {
    if (!output) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT; }
    *output = nullptr;
    auto * session = static_cast<core_session *>(opaque);
    if (!session || !descriptor || !cpu_api || !cpu || !cpu_regions) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT; }
    std::lock_guard<std::mutex> lock(session->mutex);
    if (session->closed || session->caller_active || session->external_drains || session->prepared ||
            session->regions >= session->config.max_regions) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CLOSED; }
    auto region = std::make_unique<core_region>();
    region->owner = session;
    region->cpu_api = cpu_api; region->cpu_service = cpu;
    if (!ggml_moe_source_expert_prepare(*descriptor, cpu_regions, region->expert)) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
    }
    auto & expert = region->expert;
    const ggml_moe_reference_gpu_layout layout{int(expert.roles[0].type), int(expert.roles[2].type),
        int(expert.input_width), int(expert.hidden_width), int(expert.output_width), expert.roles[0].row_stride,
        expert.roles[1].row_stride, expert.roles[2].row_stride};
    expert.generic_body |= !expert.routed_operation && (!ggml_moe_reference_gpu_supported(layout) || descriptor->geometry.route_capacity > 65535);
    ggml_cuda_mmid_resources resources;
    bool supported = expert.routed_operation ? cpu_api->struct_size == sizeof(*cpu_api) && cpu_api->execute_routed &&
        ggml_cuda_mmid_requirements(session->parent->device, expert.region.output, resources) :
        expert.generic_body || (ggml_moe_reference_gpu_supported(layout) && descriptor->geometry.route_capacity <= 65535);
    if (supported && expert.generic_body) {
        const auto & query = *(descriptor->body_query ? descriptor->body_query : descriptor->query);
        for (uint32_t i = 0; i < query.n_body_nodes && supported; ++i) {
            const auto * node = query.body_nodes[i];
            if (node->op == GGML_OP_MUL_MAT_ID) {
                supported = ggml_cuda_mmid_requirements(session->parent->device, node, resources);
            } else {
                supported = ggml_is_view(node) || ((node->op == GGML_OP_GLU || node->op == GGML_OP_SQR ||
                    (node->op == GGML_OP_UNARY && ggml_get_unary_op(node) == GGML_UNARY_OP_RELU)) &&
                    ggml_backend_dev_supports_op(ggml_backend_get_device(session->config.backend), node));
            }
        }
    }
    if (!supported ||
            (session->cpu_api && (session->cpu_api != cpu_api || session->cpu_service != cpu))) {
        fprintf(stderr, "moe-source-core-preparation: stage=expert_capability gate_type=%d down_type=%d input=%u hidden=%u output=%u route_capacity=%u\n",
            int(expert.roles[0].type), int(expert.roles[2].type), expert.input_width, expert.hidden_width,
            expert.output_width, descriptor->geometry.route_capacity);
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
    }
    session->cpu_api = cpu_api; session->cpu_service = cpu;
    ++session->regions;
    *output = region.release();
    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
} catch (...) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY; }

static void source_cpu_hook(void * opaque, uint32_t phase) {
    auto & source = *static_cast<core_session *>(opaque);
    const uint32_t event = phase == GGML_BACKEND_MOE_CPU_TEST_PHASE_V1_ADMITTED ? GGML_BACKEND_MOE_HYBRID_TEST_CPU_ADMITTED :
        phase == GGML_BACKEND_MOE_CPU_TEST_PHASE_V1_BEFORE_COMMIT ? GGML_BACKEND_MOE_HYBRID_TEST_CPU_BEFORE_COMMIT :
        GGML_BACKEND_MOE_HYBRID_TEST_CPU_AFTER_COMMIT;
    if (source.hook && !source.hook(source.hook_data, event, source.epoch, nullptr)) { source.stop(); }
}

static ggml_status source_preflight(void * opaque, const ggml_cgraph * graph,
        const ggml_graph_execution_certificate * certificate, void * const * prepared, uint32_t count) {
    auto * session = static_cast<core_session *>(opaque);
    if (!session) { return GGML_STATUS_FAILED; }
    std::lock_guard<std::mutex> lock(session->mutex);
    source_host_timer timer(session->preflight_wall_ns, session->metadata_diagnostic);
    session->rejected_before_effects = false;
    if (session->closed || session->effect_failure || session->caller_active || session->external_drains) { return GGML_STATUS_FAILED; }
    if (certificate && session->metadata_valid(graph, *certificate, prepared, count)) { return GGML_STATUS_SUCCESS; }
    session->rejected_before_effects = !session->resources_live && !session->dispatch;
    return GGML_STATUS_FAILED;
}

static ggml_status source_compute(void * opaque, ggml_cgraph * graph, void * const * prepared, uint32_t count) {
    auto * session = static_cast<core_session *>(opaque);
    if (!session) { return GGML_STATUS_FAILED; }
    if (!graph || !prepared || !count) {
        std::lock_guard<std::mutex> lock(session->mutex);
        session->rejected_before_effects = false;
        return GGML_STATUS_FAILED;
    }
    if (!session->admit_caller()) { return GGML_STATUS_FAILED; }
    struct caller_guard {
        core_session & session;
        ~caller_guard() { session.leave_caller(); }
    } guard{*session};
    try {
        // The shared worker service must refer to the active program only.
        if (session->cpu_api && session->cpu_service) {
            const auto setter = session->cpu_api->set_test_hook;
            if ((!setter && session->hook) || (setter && setter(session->cpu_service,
                    session->hook ? source_cpu_hook : nullptr, session->hook ? session : nullptr))) {
                return session->reject_before_effects();
            }
        }
        if (session->hook && !session->hook(session->hook_data, GGML_BACKEND_MOE_HYBRID_TEST_SOURCE_COMPUTE_ENTERED, session->epoch, nullptr)) {
            std::lock_guard<std::mutex> lock(session->mutex);
            session->closed = true; session->stop();
            return GGML_STATUS_FAILED;
        }
        const auto status = session->replay(graph, prepared, count);
        if (status != GGML_STATUS_SUCCESS) {
            std::lock_guard<std::mutex> lock(session->mutex);
            if (!session->rejected_before_effects) { session->closed = true; session->effect_failure = true; session->stop(); }
        }
        if (session->hook && !session->hook(session->hook_data, GGML_BACKEND_MOE_HYBRID_TEST_SOURCE_COMPUTE_RETURNING, session->epoch, session->stream)) {
            std::lock_guard<std::mutex> lock(session->mutex);
            session->closed = true; session->stop();
            return GGML_STATUS_FAILED;
        }
        return status;
    }
    catch (...) {
        {
            std::lock_guard<std::mutex> lock(session->mutex);
            session->closed = true; session->effect_failure = true; session->stop();
        }
        (void) session->drain(true);
        return GGML_STATUS_FAILED;
    }
}

static int32_t source_close(void * opaque) {
    auto * session = static_cast<core_session *>(opaque);
    if (!session) { return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_INVALID; }
    std::lock_guard<std::mutex> lock(session->mutex);
    session->closed = true;
    return session->stop() ? GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK : GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_FAILED;
}

static int32_t source_drain(void * opaque) {
    auto * session = static_cast<core_session *>(opaque);
    if (!session) { return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_INVALID; }
    const auto status = session->drain();
    std::lock_guard<std::mutex> lock(session->mutex);
    if (!session->caller_active && (status || session->replay_diagnostic)) { session->summary("drain", status); }
    return status;
}

static int32_t source_release_region(void * opaque, void ** prepared) {
    auto * session = static_cast<core_session *>(opaque);
    if (!session || !prepared) { return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_INVALID; }
    if (!*prepared) { return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK; }
    auto * region = static_cast<core_region *>(*prepared);
    if (region->owner != session) { return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_INVALID; }
    const auto status = session->dispose(true);
    if (status) { return status; }
    std::lock_guard<std::mutex> lock(session->mutex);
    if (!session->regions) { return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_INVALID; }
    delete region;
    --session->regions;
    *prepared = nullptr;
    return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK;
}

static int32_t source_release(void ** opaque) {
    if (!opaque) { return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_INVALID; }
    if (!*opaque) { return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK; }
    auto * session = static_cast<core_session *>(*opaque);
    const auto status = session->dispose();
    if (status) { return status; }
    delete session;
    *opaque = nullptr;
    return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK;
}

static bool source_state(void * opaque, ggml_backend_moe_hybrid_state_v1 * state) {
    auto * session = static_cast<core_session *>(opaque);
    if (!session || !state || state->struct_size != sizeof(*state)) { return false; }
    std::lock_guard<std::mutex> lock(session->mutex);
    if (session->caller_active) {
        *state = {};
        state->struct_size = sizeof(*state);
        state->dispatch_active = 1;
    } else { *state = session->counters; state->struct_size = sizeof(*state); }
    state->quiescing = session->closed;
    state->ticket_state = session->rejected_before_effects && !session->closed && !session->effect_failure &&
        !session->caller_active && !session->resources_live && !session->dispatch ?
        GGML_BACKEND_MOE_SOURCE_CORE_TICKET_V1_REJECTED_BEFORE_EFFECTS : 0;
    if (session->cpu_api && session->cpu_service) {
        ggml_backend_moe_cpu_service_state_v1 cpu = {};
        cpu.struct_size = sizeof(cpu);
        if (session->cpu_api->state(session->cpu_service, &cpu)) { return false; }
        state->cpu_active_jobs = cpu.active_jobs;
        state->prepared_cpu_bytes = cpu.prepared_payload_bytes;
    }
    return true;
}

static bool source_set_hook(void * opaque, ggml_backend_moe_hybrid_test_hook_v1_t hook, void * data) {
    auto * session = static_cast<core_session *>(opaque);
    if (!session) { return false; }
    std::lock_guard<std::mutex> lock(session->mutex);
    if (session->caller_active || session->external_drains || session->closed) { return false; }
    if (hook && session->cpu_api && !session->cpu_api->set_test_hook) { return false; }
    session->hook = hook; session->hook_data = data;
    return true;
}

} // namespace

const ggml_backend_moe_source_core_api_v1 * ggml_cuda_moe_source_core_api() {
    static const ggml_backend_moe_source_core_api_v1 api{sizeof(api), 1, source_create, source_prepare,
        source_compute, source_close, source_drain, source_release_region, source_release, source_state, source_set_hook, source_preflight};
    return &api;
}

bool ggml_cuda_mmid_pool_compute_for_test(ggml_backend_t backend, ggml_tensor * dst,
        const ggml_cuda_mmid_execution & execution, void * workspace, size_t bytes, size_t * peak) {
    ggml_cuda_mmid_resources resources;
    if (!ggml_backend_is_cuda(backend) || !backend->context || !peak ||
            !ggml_cuda_mmid_execution_valid(dst, execution)) { return false; }
    auto * context = static_cast<ggml_backend_cuda_context *>(backend->context);
    if (!ggml_cuda_mmid_requirements(context->device, dst, resources) || bytes < resources.pool_bytes ||
            (resources.pool_bytes && (!workspace || reinterpret_cast<uintptr_t>(workspace) % 256))) { return false; }
    auto prepared = std::make_unique<ggml_cuda_pool_buffer>(static_cast<uint8_t *>(workspace), resources.pool_bytes);
    auto * measured = prepared.get();
    auto & current = context->pools[context->device][context->curr_stream_no];
    std::unique_ptr<ggml_cuda_pool> borrowed = std::move(prepared);
    borrowed.swap(current);
    bool ok = false;
    try { ok = ggml_cuda_mmid_execution_compute(backend, dst, execution); } catch (...) {}
    ok &= measured->used == 0;
    *peak = measured->peak;
    borrowed.swap(current);
    return ok;
}
