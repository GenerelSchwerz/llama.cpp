#include "moe-source-tail.cuh"

#if defined(GGML_CUDA_MOE_DEVICE_TAIL)
#include <cuda/atomic>

static __device__ bool stopped(const ggml_cuda_moe_source_control * control, cuda::memory_order order = cuda::memory_order_acquire) {
    auto * flag = reinterpret_cast<uint32_t *>(const_cast<std::atomic<uint32_t> *>(&control->stop.value));
    return cuda::atomic_ref<uint32_t, cuda::thread_scope_system>(*flag).load(order) != 0;
}

static __device__ void dispatch_tail(ggml_cuda_moe_source_runtime * runtime,
        const ggml_cuda_moe_source_tail_dispatch * dispatch, bool complete) {
    const auto graph = complete ? dispatch->next : dispatch->finalize;
    if (cudaGraphLaunch(graph, cudaStreamGraphTailLaunch) != cudaSuccess) {
        runtime->epoch = 0;
        if (graph != dispatch->finalize) { (void) cudaGraphLaunch(dispatch->finalize, cudaStreamGraphTailLaunch); }
    }
}

static __global__ void start_tail(ggml_cuda_moe_source_runtime * runtime,
        ggml_cuda_moe_source_control * control, const ggml_cuda_moe_source_tail_dispatch * dispatch) {
    const bool canceled = stopped(control);
    *runtime = {};
    runtime->epoch = control->epoch;
    runtime->failed = canceled;
    dispatch_tail(runtime, dispatch, !canceled);
}

static __global__ void import_join_tail(float * output, const float * input, size_t values, size_t width, const uint32_t * gpu_mask,
        ggml_cuda_moe_source_runtime * runtime, const int32_t * resident, const int32_t * selected,
        ggml_cuda_moe_source_control * next_control, const ggml_cuda_moe_source_tail_dispatch * dispatch) {
    if (!runtime->failed) {
        const uint32_t route = blockIdx.y;
        const bool gpu = (gpu_mask[route / 32] & (uint32_t(1) << (route % 32))) != 0;
        const size_t column = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (column < width) {
            const size_t i = size_t(route) * width + column;
            output[i] = gpu ? __fadd_rn(output[i], 0.0f) : __fadd_rn(0.0f, input[i]);
        }
    }
    if (!blockIdx.x && !blockIdx.y && !threadIdx.x) {
        if (!runtime->failed) {
            ++runtime->joined;
            runtime->resident += *resident > 0;
            runtime->selected += *selected > 0;
        }
        // This flag selects the next graph and publishes no input data.
        dispatch_tail(runtime, dispatch, !runtime->failed && !stopped(next_control, cuda::memory_order_relaxed));
    }
}

template<bool routed>
static __global__ void import_tail(float * output, const float * input, size_t values,
        ggml_cuda_moe_source_runtime * runtime, const int32_t * resident, const int32_t * selected,
        ggml_cuda_moe_source_control * next_control, const ggml_cuda_moe_source_tail_dispatch * dispatch,
        size_t width, uint32_t routes_per_row, size_t output_column, size_t output_row) {
    if (!runtime->failed) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < values; i += size_t(gridDim.x) * blockDim.x) {
            if constexpr (routed) {
                const size_t route = i / width, column = i % width;
                output[(route / routes_per_row) * output_row + (route % routes_per_row) * output_column + column] += input[route * output_column + column];
            } else {
                output[i] += input[i];
            }
        }
    }
    if (!blockIdx.x && !threadIdx.x) {
        if (!runtime->failed) {
            ++runtime->joined;
            runtime->resident += *resident > 0;
            runtime->selected += *selected > 0;
        }
        // This flag selects the next graph and publishes no input data.
        dispatch_tail(runtime, dispatch, !runtime->failed && !stopped(next_control, cuda::memory_order_relaxed));
    }
}

static __global__ void finish_tail(ggml_cuda_moe_source_runtime * runtime,
        const ggml_cuda_moe_source_tail_dispatch * dispatch) {
    dispatch_tail(runtime, dispatch, true);
}

cudaError_t ggml_cuda_moe_source_tail_start(ggml_cuda_moe_source_runtime * runtime,
        ggml_cuda_moe_source_control * control, const ggml_cuda_moe_source_tail_dispatch * dispatch, cudaStream_t stream) {
    start_tail<<<1, 1, 0, stream>>>(runtime, control, dispatch);
    return cudaGetLastError();
}

cudaError_t ggml_cuda_moe_source_tail_import(float * output, const float * input, size_t values,
        ggml_cuda_moe_source_runtime * runtime, const int32_t * resident, const int32_t * selected,
        ggml_cuda_moe_source_control * next_control, const ggml_cuda_moe_source_tail_dispatch * dispatch, cudaStream_t stream,
        size_t width, uint32_t routes_per_row, size_t output_column, size_t output_row, const uint32_t * gpu_mask) {
    if (gpu_mask) {
        if (routes_per_row || output_column || output_row || !width || width % ggml_cuda_moe_source_import_threads ||
                values % width || !values || values / width > 65535 || width > UINT32_MAX) { return cudaErrorInvalidValue; }
        import_join_tail<<<dim3(unsigned(width / ggml_cuda_moe_source_import_threads), unsigned(values / width)), ggml_cuda_moe_source_import_threads, 0, stream>>>(output, input, values,
            width, gpu_mask, runtime, resident, selected, next_control, dispatch);
    } else if (width) {
        if (!routes_per_row) { return cudaErrorInvalidValue; }
        import_tail<true><<<unsigned((values + 255) / 256), 256, 0, stream>>>(output, input, values, runtime,
            resident, selected, next_control, dispatch, width, routes_per_row, output_column, output_row);
    } else {
        import_tail<false><<<unsigned((values + 255) / 256), 256, 0, stream>>>(output, input, values, runtime,
            resident, selected, next_control, dispatch, 0, 0, 0, 0);
    }
    return cudaGetLastError();
}

cudaError_t ggml_cuda_moe_source_tail_finish(ggml_cuda_moe_source_runtime * runtime,
        const ggml_cuda_moe_source_tail_dispatch * dispatch, cudaStream_t stream) {
    finish_tail<<<1, 1, 0, stream>>>(runtime, dispatch);
    return cudaGetLastError();
}
#endif
