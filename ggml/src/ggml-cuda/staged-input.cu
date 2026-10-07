#include "common.cuh"
#include "staged-input.cuh"
#include "ggml-backend-impl.h"

#include <atomic>
#include <cstring>
#include <new>

// Windows uses host inputs until the host-mapped memop path is validated.
#if !defined(_WIN32) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 12000
#include <cudaTypedefs.h>

namespace {
struct staged_input {
    void * host = nullptr;
    std::atomic<uint32_t> * flag = nullptr;
    CUdeviceptr device_flag = 0;
    size_t bytes = 0;
    int device = -1;
    PFN_cuStreamWaitValue32_v11070 wait = nullptr;
    PFN_cuStreamWriteValue32_v11070 write = nullptr;

    ~staged_input() {
        if (host) { cudaFreeHost(host); }
        if (flag) { cudaFreeHost(flag); }
    }
};

[[noreturn]] static void marker(ggml_tensor *, int, int, void *) {
    GGML_ABORT("staged input requires its CUDA backend");
}

static bool enqueue(staged_input & input, cudaStream_t stream, void * dst) {
    return input.wait(stream, input.device_flag, 1, CU_STREAM_WAIT_VALUE_EQ) == CUDA_SUCCESS &&
        cudaMemcpyAsync(dst, input.host, input.bytes, cudaMemcpyHostToDevice, stream) == cudaSuccess &&
        input.write(stream, input.device_flag, 0, CU_STREAM_WRITE_VALUE_DEFAULT) == CUDA_SUCCESS;
}

static void * create(ggml_backend_t backend, size_t bytes) {
    if (!bytes || bytes > 1024*1024) { return nullptr; }
    // Resolve stream memops without adding a CUDA driver link dependency.
    auto input = std::make_unique<staged_input>();
    const auto resolve = [](const char * name, void ** function) {
        cudaDriverEntryPointQueryResult query;
#if CUDART_VERSION >= 12050
        const auto error = cudaGetDriverEntryPointByVersion(name, function, 11070, cudaEnableDefault, &query);
#else
        const auto error = cudaGetDriverEntryPoint(name, function, cudaEnableDefault, &query);
#endif
        return error == cudaSuccess && query == cudaDriverEntryPointSuccess && *function != nullptr;
    };
    if (!resolve("cuStreamWaitValue32", reinterpret_cast<void **>(&input->wait)) ||
        !resolve("cuStreamWriteValue32", reinterpret_cast<void **>(&input->write))) {
        cudaGetLastError();
        return nullptr;
    }
    auto * ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    ggml_cuda_set_device(ctx->device);
    input->bytes = bytes;
    input->device = ctx->device;
    if (cudaHostAlloc(&input->host, bytes, cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc(reinterpret_cast<void **>(&input->flag), sizeof(*input->flag), cudaHostAllocMapped) != cudaSuccess ||
        cudaHostGetDevicePointer(reinterpret_cast<void **>(&input->device_flag), input->flag, 0) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    new (input->flag) std::atomic<uint32_t>(1);
    std::memset(input->host, 0, bytes);

    // Probe eager execution and replay before the model graph can contain a wait.
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    void * dst = nullptr;
    bool ok = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess && cudaMalloc(&dst, bytes) == cudaSuccess;
    if (ok) {
        ok = enqueue(*input, stream, dst) && cudaStreamSynchronize(stream) == cudaSuccess &&
            input->flag->load(std::memory_order_acquire) == 0;
    }
    if (ok) {
        ok = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
        if (ok) {
            ok = enqueue(*input, stream, dst);
            const auto end = cudaStreamEndCapture(stream, &graph);
            ok = ok && end == cudaSuccess && graph;
        }
    }
    if (ok) { ok = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) == cudaSuccess; }
    for (int replay = 0; ok && replay < 2; ++replay) {
        input->flag->store(1, std::memory_order_release);
        ok = cudaGraphLaunch(exec, stream) == cudaSuccess && cudaStreamSynchronize(stream) == cudaSuccess &&
            input->flag->load(std::memory_order_acquire) == 0;
    }
    if (exec) { cudaGraphExecDestroy(exec); }
    if (graph) { cudaGraphDestroy(graph); }
    if (dst) { cudaFree(dst); }
    if (stream) { cudaStreamDestroy(stream); }
    if (!ok) { cudaGetLastError(); return nullptr; }
    return input.release();
}

static ggml_tensor * build(void * opaque, ggml_context * ctx, ggml_tensor * dependency, int64_t elements) {
    auto * input = static_cast<staged_input *>(opaque);
    GGML_ASSERT(elements > 0 && size_t(elements)*sizeof(float) == input->bytes);
    return ggml_custom_4d(ctx, GGML_TYPE_F32, elements, 1, 1, 1, &dependency, dependency ? 1 : 0, marker, 1, input);
}
}

bool ggml_cuda_staged_input_supports(const ggml_tensor * tensor) {
    if (tensor->op != GGML_OP_CUSTOM) { return false; }
    ggml_custom_op_params params;
    memcpy(&params, tensor->op_params, sizeof(params));
    return params.fun == marker && params.userdata && tensor->type == GGML_TYPE_F32;
}

bool ggml_cuda_staged_input_compute(ggml_backend_cuda_context & ctx, ggml_tensor * tensor) {
    if (!ggml_cuda_staged_input_supports(tensor)) { return false; }
    ggml_custom_op_params params;
    memcpy(&params, tensor->op_params, sizeof(params));
    auto & input = *static_cast<staged_input *>(params.userdata);
    GGML_ASSERT(ggml_nbytes(tensor) == input.bytes);
    return enqueue(input, ctx.stream(), tensor->data);
}

const ggml_staged_input_api * ggml_cuda_staged_input_api() {
    static const ggml_staged_input_api api = {
        create,
        [](void * input) { delete static_cast<staged_input *>(input); },
        [](void * input) -> void * { return static_cast<staged_input *>(input)->host; },
        [](void * input) { static_cast<staged_input *>(input)->flag->store(1, std::memory_order_release); },
        build,
    };
    return &api;
}

bool ggml_cuda_staged_input_pending_for_test(void * input) {
    return static_cast<staged_input *>(input)->flag->load(std::memory_order_acquire) != 0;
}
bool ggml_cuda_staged_input_prepare_source_view(int device, const ggml_tensor * node, ggml_cuda_source_staged_input_view & view) {
    view = {};
    int selected = -1;
    if (device < 0 || device >= ggml_cuda_info().device_count || cudaGetDevice(&selected) != cudaSuccess || selected != device ||
            !node || node->op != GGML_OP_CUSTOM || node->type != GGML_TYPE_F32 || node->ne[0] <= 0 ||
            uint64_t(node->ne[0]) > SIZE_MAX / sizeof(float) || uint64_t(node->ne[0]) > uint64_t(INT64_MAX) / sizeof(float) ||
            node->view_src || node->view_offs) { return false; }
    const size_t bytes = size_t(node->ne[0]) * sizeof(float);
    if (node->nb[0] != sizeof(float)) { return false; }
    for (int i = 1; i < 4; ++i) { if (node->ne[i] != 1 || node->nb[i] != bytes) { return false; } }
    ggml_custom_op_params params;
    memcpy(&params, node->op_params, sizeof(params));
    if (params.fun != marker || params.n_tasks != 1 || !params.userdata ||
            reinterpret_cast<uintptr_t>(params.userdata) % alignof(staged_input)) { return false; }
    const auto & input = *static_cast<const staged_input *>(params.userdata);
    if (input.device != device || input.bytes != bytes || !input.host || !input.flag || !input.device_flag ||
            sizeof(std::atomic<uint32_t>) != sizeof(uint32_t) || !std::atomic<uint32_t>::is_always_lock_free ||
            reinterpret_cast<uintptr_t>(input.host) % alignof(float) ||
            reinterpret_cast<uintptr_t>(input.flag) % alignof(std::atomic<uint32_t>)) { return false; }
    void * payload_alias = nullptr;
    void * flag_alias = nullptr;
    cudaPointerAttributes payload_attributes = {}, flag_attributes = {};
    if (cudaHostGetDevicePointer(&payload_alias, input.host, 0) != cudaSuccess || !payload_alias ||
            cudaHostGetDevicePointer(&flag_alias, input.flag, 0) != cudaSuccess || !flag_alias ||
            reinterpret_cast<CUdeviceptr>(flag_alias) != input.device_flag ||
            cudaPointerGetAttributes(&payload_attributes, input.host) != cudaSuccess ||
            cudaPointerGetAttributes(&flag_attributes, input.flag) != cudaSuccess ||
            payload_attributes.type != cudaMemoryTypeHost || payload_attributes.device != device ||
            payload_attributes.hostPointer != input.host || payload_attributes.devicePointer != payload_alias ||
            flag_attributes.type != cudaMemoryTypeHost || flag_attributes.device != device ||
            flag_attributes.hostPointer != input.flag || flag_attributes.devicePointer != flag_alias ||
            reinterpret_cast<uintptr_t>(payload_alias) % alignof(float) || reinterpret_cast<uintptr_t>(flag_alias) % alignof(uint32_t)) { return false; }
    const auto host = reinterpret_cast<uintptr_t>(input.host), flag = reinterpret_cast<uintptr_t>(input.flag);
    const auto payload = reinterpret_cast<uintptr_t>(payload_alias), device_flag = reinterpret_cast<uintptr_t>(flag_alias);
    if (bytes > UINTPTR_MAX - host || bytes > UINTPTR_MAX - payload || sizeof(uint32_t) > UINTPTR_MAX - flag ||
            sizeof(uint32_t) > UINTPTR_MAX - device_flag ||
            (host < flag + sizeof(uint32_t) && flag < host + bytes) ||
            (payload < device_flag + sizeof(uint32_t) && device_flag < payload + bytes)) { return false; }
    uint64_t identity = 14695981039346656037ULL;
    const auto mix = [&identity](uint64_t value) { identity = (identity ^ value) * 1099511628211ULL; };
    for (uint64_t value : {uint64_t(host), uint64_t(payload), uint64_t(flag), uint64_t(device_flag), uint64_t(bytes),
            uint64_t(device), uint64_t(node->op), uint64_t(node->type), uint64_t(node->flags), uint64_t(node->view_offs)}) { mix(value); }
    for (int i = 0; i < 4; ++i) { mix(uint64_t(node->ne[i])); mix(node->nb[i]); }
    for (const auto * source : node->src) { mix(reinterpret_cast<uintptr_t>(source)); }
    const auto * raw_params = reinterpret_cast<const unsigned char *>(node->op_params);
    for (size_t i = 0; i < sizeof(node->op_params); ++i) { mix(raw_params[i]); }
    view.host = input.host; view.device_alias = payload_alias; view.host_flag = input.flag; view.device_flag = flag_alias;
    view.bytes = bytes; view.device = device; view.identity = identity;
    return true;
}
#else
bool ggml_cuda_staged_input_supports(const ggml_tensor *) { return false; }
bool ggml_cuda_staged_input_compute(ggml_backend_cuda_context &, ggml_tensor *) { return false; }
const ggml_staged_input_api * ggml_cuda_staged_input_api() { return nullptr; }
bool ggml_cuda_staged_input_pending_for_test(void *) { return false; }
bool ggml_cuda_staged_input_prepare_source_view(int, const ggml_tensor *, ggml_cuda_source_staged_input_view & view) { view = {}; return false; }
#endif
