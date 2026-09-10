#include "staged-input.cuh"
#include "ggml-backend-impl.h"

#include <atomic>
#include <cstring>
#include <new>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 12000
namespace {
struct staged_input {
    void * host = nullptr;
    std::atomic<uint32_t> * flag = nullptr;
    CUdeviceptr device_flag = 0;
    size_t bytes = 0;
    decltype(&cuStreamWaitValue32) wait = nullptr;
    decltype(&cuStreamWriteValue32) write = nullptr;

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
    cudaDriverEntryPointQueryResult query;
    if (cudaGetDriverEntryPoint("cuStreamWaitValue32", reinterpret_cast<void **>(&input->wait), cudaEnableDefault, &query) != cudaSuccess || !input->wait ||
        cudaGetDriverEntryPoint("cuStreamWriteValue32", reinterpret_cast<void **>(&input->write), cudaEnableDefault, &query) != cudaSuccess || !input->write) {
        cudaGetLastError();
        return nullptr;
    }
    auto * ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    ggml_cuda_set_device(ctx->device);
    input->bytes = bytes;
    if (cudaHostAlloc(&input->host, bytes, cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc(reinterpret_cast<void **>(&input->flag), sizeof(*input->flag), cudaHostAllocMapped) != cudaSuccess ||
        cudaHostGetDevicePointer(reinterpret_cast<void **>(&input->device_flag), input->flag, 0) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    new (input->flag) std::atomic<uint32_t>(1);
    std::memset(input->host, 0, bytes);

    // Probe capture and replay before the model graph can contain a wait.
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    void * dst = nullptr;
    bool ok = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess && cudaMalloc(&dst, bytes) == cudaSuccess;
    if (ok) {
        ok = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
        if (ok) {
            ok = enqueue(*input, stream, dst);
            const auto end = cudaStreamEndCapture(stream, &graph);
            ok = ok && end == cudaSuccess && graph;
        }
    }
    if (ok) { ok = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) == cudaSuccess; }
    if (ok) { ok = cudaGraphLaunch(exec, stream) == cudaSuccess && cudaStreamSynchronize(stream) == cudaSuccess; }
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
#else
bool ggml_cuda_staged_input_supports(const ggml_tensor *) { return false; }
bool ggml_cuda_staged_input_compute(ggml_backend_cuda_context &, ggml_tensor *) { return false; }
const ggml_staged_input_api * ggml_cuda_staged_input_api() { return nullptr; }
#endif
