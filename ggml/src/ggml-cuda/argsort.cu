#include "argsort.cuh"
#include "top-k.cuh"
#include "ggml-backend-impl.h"

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 1)
#        define STRIDED_ITERATOR_AVAILABLE
#        include <cuda/iterator>
#    endif
using namespace cub;
#endif  // GGML_CUDA_USE_CUB

static __global__ void init_indices(int * indices, const int ncols, const int nrows) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;

    if (col < ncols && row < nrows) {
        indices[row * ncols + col] = col;
    }
}

#ifndef STRIDED_ITERATOR_AVAILABLE
static __global__ void init_offsets(int * offsets, const int ncols, const int nrows) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= nrows) {
        offsets[idx] = idx * ncols;
    }
}
#endif  // STRIDED_ITERATOR_AVAILABLE

#ifdef GGML_CUDA_USE_CUB

// returns the suggested maximum number of rows to process during one argsort_f32_i32_cuda_cub() call
int argsort_f32_i32_cuda_cub_chunk_nrows(const size_t nb01, const int64_t nrows) {
    // perform argsort in chunks up to approximately this size (currently 64MB)
    // to avoid excessive temporary buffers memory usage
    const int chunk_bytes = 1 << 26;

    // calculate how many rows will fit in one chunk (must be at least one)
    const int chunk_nrows = std::max((int) (chunk_bytes / nb01), 1);

    // limit the resulting amount to total nrows
    return std::min((int64_t) chunk_nrows, nrows);
}

void argsort_f32_i32_cuda_cub(ggml_cuda_pool & pool,
                              const float *    x,
                              int *            dst,
                              const int        ncols,
                              const int        nrows,
                              ggml_sort_order  order,
                              cudaStream_t     stream) {
    ggml_cuda_pool_alloc<int>   temp_indices_alloc(pool, ncols * nrows);
    ggml_cuda_pool_alloc<float> temp_keys_alloc(pool, ncols * nrows);
    // Device*Sort algorithms currently do not allow for in-place sorting/aliasing of input/outputs
    ggml_cuda_pool_alloc<float> temp_keys_out_alloc(pool, ncols * nrows);

    int *   temp_indices = temp_indices_alloc.get();
    float * temp_keys    = temp_keys_alloc.get();
    float * temp_keys_out = temp_keys_out_alloc.get();

    static const int block_size = 256;
    const dim3 grid_size((ncols + block_size - 1) / block_size, nrows);
    init_indices<<<grid_size, block_size, 0, stream>>>(temp_indices, ncols, nrows);

#ifdef STRIDED_ITERATOR_AVAILABLE
    auto offset_iterator = cuda::make_strided_iterator(cuda::make_counting_iterator(0), ncols);
#else
    // offset_iterator needs to populate nrows + 1 elements, so we also have to ceildiv nrows + 1 by block_size
    const int                 nrows_offset = nrows + 1;
    ggml_cuda_pool_alloc<int> offsets_alloc(pool, nrows_offset);
    int *                     offset_iterator = offsets_alloc.get();
    const dim3                offset_grid((nrows_offset + block_size - 1) / block_size);
    init_offsets<<<offset_grid, block_size, 0, stream>>>(offset_iterator, ncols, nrows);
#endif
    CUDA_CHECK(cudaMemcpyAsync(temp_keys, x, ncols * nrows * sizeof(float), cudaMemcpyDeviceToDevice, stream));

    size_t temp_storage_bytes = 0;

    bool is_capturing = false;
#ifdef USE_CUDA_GRAPH
    // Currently (confirmed for CCCL <= 3.2) DeviceSegmentedSort does not support stream capture, while DeviceSegmentedRadixSort does.
    // See https://github.com/NVIDIA/cccl/issues/5661#issuecomment-3229037149
    // TODO: constrain this to the CCCL versions that have this issue once it's resolved in a future CCCL release.
    cudaStreamCaptureStatus capture_status;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
    is_capturing = (capture_status != cudaStreamCaptureStatusNone);
#endif  // USE_CUDA_GRAPH

    if (order == GGML_SORT_ORDER_ASC) {
        if (nrows == 1) {
            CUDA_CHECK(DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, temp_keys, temp_keys_out,  // keys in, keys out
                                                  temp_indices, dst,  // values (indices)
                                                  ncols, 0, sizeof(float) * 8, stream));
        } else if (is_capturing) {
            CUDA_CHECK(DeviceSegmentedRadixSort::SortPairs(
                nullptr, temp_storage_bytes, temp_keys, temp_keys_out,  // keys in, keys out
                temp_indices, dst,                                  // values (indices)
                ncols * nrows, nrows,                               // num items, num segments
                offset_iterator, offset_iterator + 1, 0, sizeof(float) * 8, stream));
        } else {
            CUDA_CHECK(DeviceSegmentedSort::SortPairs(nullptr, temp_storage_bytes, temp_keys,
                                                      temp_keys_out, // keys out
                                                      temp_indices, dst,     // values (indices)
                                                      ncols * nrows, nrows,  // num items, num segments
                                                      offset_iterator, offset_iterator + 1, stream));
        }
    } else {
        if (nrows == 1) {
            CUDA_CHECK(DeviceRadixSort::SortPairsDescending(nullptr, temp_storage_bytes, temp_keys,
                                                            temp_keys_out, // keys out
                                                            temp_indices, dst,  // values (indices)
                                                            ncols, 0, sizeof(float) * 8, stream));
        } else if (is_capturing) {
            CUDA_CHECK(DeviceSegmentedRadixSort::SortPairsDescending(
                nullptr, temp_storage_bytes, temp_keys, temp_keys_out, temp_indices, dst, ncols * nrows, nrows,
                offset_iterator, offset_iterator + 1, 0, sizeof(float) * 8, stream));
        } else {
            CUDA_CHECK(DeviceSegmentedSort::SortPairsDescending(nullptr, temp_storage_bytes, temp_keys, temp_keys_out,
                                                                temp_indices, dst, ncols * nrows, nrows,
                                                                offset_iterator, offset_iterator + 1, stream));
        }
    }

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    if (order == GGML_SORT_ORDER_ASC) {
        if (nrows == 1) {
            CUDA_CHECK(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, temp_keys,
                                                  temp_keys_out, // keys out
                                                  temp_indices, dst,  // values (indices)
                                                  ncols, 0, sizeof(float) * 8, stream));
        } else if (is_capturing) {
            CUDA_CHECK(DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, temp_keys, temp_keys_out,
                                                           temp_indices, dst, ncols * nrows, nrows, offset_iterator,
                                                           offset_iterator + 1, 0, sizeof(float) * 8, stream));
        } else {
            CUDA_CHECK(DeviceSegmentedSort::SortPairs(d_temp_storage, temp_storage_bytes, temp_keys, temp_keys_out,
                                                      temp_indices, dst, ncols * nrows, nrows, offset_iterator,
                                                      offset_iterator + 1, stream));
        }
    } else {
        if (nrows == 1) {
            CUDA_CHECK(DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, temp_keys,
                                                            temp_keys_out, // keys out
                                                            temp_indices, dst,  // values (indices)
                                                            ncols, 0, sizeof(float) * 8, stream));
        } else if (is_capturing) {
            CUDA_CHECK(DeviceSegmentedRadixSort::SortPairsDescending(
                d_temp_storage, temp_storage_bytes, temp_keys, temp_keys_out, temp_indices, dst, ncols * nrows, nrows,
                offset_iterator, offset_iterator + 1, 0, sizeof(float) * 8, stream));
        } else {
            CUDA_CHECK(DeviceSegmentedSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, temp_keys,
                                                                temp_keys_out, temp_indices, dst, ncols * nrows, nrows,
                                                                offset_iterator, offset_iterator + 1, stream));
        }
    }
}
#endif  // GGML_CUDA_USE_CUB

// Bitonic sort implementation
template<typename T>
static inline __device__ void ggml_cuda_swap(T & a, T & b) {
    T tmp = a;
    a = b;
    b = tmp;
}

// One compare-exchange of the bitonic network at (k, j) for column col.
template<ggml_sort_order order>
static inline __device__ void bitonic_step(const float * x_row, int * dst_row, const int ncols, const int col, const int k, const int j) {
    const int ixj = col ^ j;
    if (ixj <= col) {
        return;
    }
    if ((col & k) == 0) {
        if (dst_row[col] >= ncols ||
            (dst_row[ixj] < ncols && (order == GGML_SORT_ORDER_ASC ?
                x_row[dst_row[col]] > x_row[dst_row[ixj]] :
                x_row[dst_row[col]] < x_row[dst_row[ixj]]))
        ) {
            ggml_cuda_swap(dst_row[col], dst_row[ixj]);
        }
    } else {
        if (dst_row[ixj] >= ncols ||
            (dst_row[col] < ncols && (order == GGML_SORT_ORDER_ASC ?
                x_row[dst_row[col]] < x_row[dst_row[ixj]] :
                x_row[dst_row[col]] > x_row[dst_row[ixj]]))
        ) {
            ggml_cuda_swap(dst_row[col], dst_row[ixj]);
        }
    }
}

// Bitonic sort of one row per block. Each thread owns the columns
// threadIdx.x + i * blockDim.x, so rows wider than the block (up to the
// shared memory limit) sort with several columns per thread. Every
// (k, j) stage runs all owned columns before the barrier; a pair
// (col, col ^ j) is exchanged by the owner of its lower index only.
template<ggml_sort_order order>
static __global__ void k_argsort_f32_i32(const float * x, int * dst, const int ncols, int ncols_pad) {
    const int row = blockIdx.x;

    const float * x_row = x + row * ncols;
    extern __shared__ int dst_row[];

    // initialize indices
    for (int col = threadIdx.x; col < ncols_pad; col += blockDim.x) {
        dst_row[col] = col;
    }

    __syncthreads();

    for (int k = 2; k <= ncols_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            for (int col = threadIdx.x; col < ncols_pad; col += blockDim.x) {
                bitonic_step<order>(x_row, dst_row, ncols, col, k, j);
            }
            __syncthreads();
        }
    }

    // copy the result to dst without the padding
    for (int col = threadIdx.x; col < ncols; col += blockDim.x) {
        dst[row * ncols + col] = dst_row[col];
    }
}

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

void argsort_f32_i32_cuda_bitonic(const float *   x,
                                  int *           dst,
                                  const int       ncols,
                                  const int       nrows,
                                  ggml_sort_order order,
                                  cudaStream_t    stream) {
    // bitonic sort requires ncols to be power of 2
    const int ncols_pad = next_power_of_2(ncols);

    // one thread per column up to the block limit, several columns per
    // thread beyond it; shared memory is the remaining bound
    const dim3 block_dims(ncols_pad < CUDA_ARGSORT_BLOCK_SIZE ? ncols_pad : CUDA_ARGSORT_BLOCK_SIZE, 1, 1);
    const dim3 block_nums(nrows, 1, 1);
    const size_t shared_mem = ncols_pad * sizeof(int);

    // FIXME: this limit could be raised by ~2-4x on Ampere or newer
    GGML_ASSERT(shared_mem <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);

    if (order == GGML_SORT_ORDER_ASC) {
        k_argsort_f32_i32<GGML_SORT_ORDER_ASC>
            <<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad);
    } else if (order == GGML_SORT_ORDER_DESC) {
        k_argsort_f32_i32<GGML_SORT_ORDER_DESC>
            <<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad);
    } else {
        GGML_ABORT("fatal error");
    }
}

void ggml_cuda_op_argsort(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    enum ggml_sort_order order = (enum ggml_sort_order) dst->op_params[0];

#ifdef GGML_CUDA_USE_CUB
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;

    // early return if we can use bitonic argsort
    if (shared_mem <= max_shared_mem && ncols <= 1024) {
        argsort_f32_i32_cuda_bitonic(src0_d, (int *) dst_d, ncols, nrows, order, stream);
        return;
    }

    const int chunk_nrows = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);

    ggml_cuda_pool & pool = ctx.pool();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        argsort_f32_i32_cuda_cub(pool, src0_d, (int *) dst_d, ncols, iter_nrows, order, stream);

        src0_d += ncols * iter_nrows;
        dst_d  += ncols * iter_nrows;
    }
#else
    argsort_f32_i32_cuda_bitonic(src0_d, (int *) dst_d, ncols, nrows, order, stream);
#endif
}

namespace ggml_cuda_source_sort_detail {

bool metadata(int device, const ggml_tensor * dst, ggml_op op, ggml_cuda_source_sort_resources & resources, int & ncols, int & nrows) {
    resources = {};
    int selected = -1;
    if (device < 0 || device >= ggml_cuda_info().device_count || cudaGetDevice(&selected) != cudaSuccess || selected != device ||
            !dst || dst->op != op || !dst->src[0] || dst->type != GGML_TYPE_I32 || dst->src[0]->type != GGML_TYPE_F32) { return false; }
    const auto * src = dst->src[0];
    uint64_t elements[2] = {1, 1};
    for (int t = 0; t < 2; ++t) {
        const auto * tensor = t ? dst : src;
        size_t stride = sizeof(int32_t);
        for (int i = 0; i < 4; ++i) {
            if (tensor->ne[i] <= 0 || tensor->ne[i] > INT32_MAX || tensor->nb[i] != stride ||
                    uint64_t(tensor->ne[i]) > uint64_t(INT32_MAX) / elements[t] ||
                    size_t(tensor->ne[i]) > SIZE_MAX / stride) { return false; }
            elements[t] *= uint64_t(tensor->ne[i]); stride *= size_t(tensor->ne[i]);
        }
    }
    if (src->ne[0] > (1 << 30) || dst->ne[0] > src->ne[0]) { return false; }
    for (int i = 1; i < 4; ++i) {
        if (src->ne[i] != dst->ne[i]) { return false; }
    }
    ncols = int(src->ne[0]); nrows = int(elements[0] / uint64_t(ncols));
    const auto & info = ggml_cuda_info().devices[device];
    uint64_t identity = 14695981039346656037ULL;
    for (uint64_t value : {uint64_t(device), uint64_t(info.cc), uint64_t(info.smpb), uint64_t(info.nsm),
            uint64_t(op), uint64_t(src->type), uint64_t(dst->type)}) { mix(identity, value); }
    for (const auto * tensor : {src, dst}) {
        for (int i = 0; i < 4; ++i) { mix(identity, uint64_t(tensor->ne[i])); mix(identity, tensor->nb[i]); }
        const auto * params = reinterpret_cast<const unsigned char *>(tensor->op_params);
        for (size_t i = 0; i < sizeof(tensor->op_params); ++i) { mix(identity, params[i]); }
    }
#ifdef GGML_CUDA_USE_CUB
    mix(identity, 1);
    mix(identity, CCCL_MAJOR_VERSION); mix(identity, CCCL_MINOR_VERSION);
#else
    mix(identity, 0);
#endif
#ifdef USE_CUDA_GRAPH
    mix(identity, 1);
#else
    mix(identity, 0);
#endif
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess || nrows > prop.maxGridSize[0]) { return false; }
    for (uint64_t value : {uint64_t(prop.maxThreadsPerBlock), uint64_t(prop.maxGridSize[0]), uint64_t(prop.maxGridSize[1]),
            uint64_t(prop.sharedMemPerBlock), uint64_t(prop.warpSize)}) { mix(identity, value); }
    resources.identity = identity;
    return true;
}

} // namespace ggml_cuda_source_sort_detail

bool ggml_cuda_argsort_prepare_resources(int device, const ggml_tensor * dst, ggml_cuda_source_sort_resources & resources) {
    using namespace ggml_cuda_source_sort_detail;
    resources = {};
    ggml_cuda_source_sort_resources measured;
    int ncols = 0, nrows = 0;
    if (!metadata(device, dst, GGML_OP_ARGSORT, measured, ncols, nrows) || dst->ne[0] != ncols) { return false; }
    const auto order = ggml_sort_order(dst->op_params[0]);
    if (order != GGML_SORT_ORDER_ASC && order != GGML_SORT_ORDER_DESC) { return false; }
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) { return false; }
    const int padded = next_power_of_2(ncols);
    const bool bitonic = ncols <= 1024 && size_t(padded) * sizeof(int) <= ggml_cuda_info().devices[device].smpb;
    mix(measured.identity, bitonic);
    if (bitonic) {
        if (padded > prop.maxThreadsPerBlock || size_t(padded) * sizeof(int) > prop.sharedMemPerBlock) { return false; }
        resources = measured; return true;
    }
#ifdef GGML_CUDA_USE_CUB
    const int chunk = argsort_f32_i32_cuda_cub_chunk_nrows(dst->src[0]->nb[1], nrows);
    if (chunk <= 0 || ncols > INT32_MAX / chunk || chunk > prop.maxGridSize[1] ||
            (ncols + 255) / 256 > prop.maxGridSize[0]) { return false; }
    mix(measured.identity, chunk);
    const int remainder = nrows % chunk;
    for (int rows : {chunk, remainder}) {
        if (!rows) { continue; }
        const int items = ncols * rows;
        size_t arrays = 0;
        for (int i = 0; i < 3; ++i) { if (!add(arrays, size_t(items) * sizeof(int32_t))) { return false; } }
#ifdef STRIDED_ITERATOR_AVAILABLE
        auto offsets = cuda::make_strided_iterator(cuda::make_counting_iterator(0), ncols);
#else
        if (rows == INT32_MAX || !add(arrays, size_t(rows + 1) * sizeof(int))) { return false; }
        const int * offsets = reinterpret_cast<const int *>(uintptr_t(256));
#endif
        for (bool capture : {false, true}) {
#ifndef USE_CUDA_GRAPH
            if (capture) { continue; }
#endif
            size_t temp = 0;
            cudaError_t result;
            float * keys = nullptr;
            int * indices = nullptr;
            if (rows == 1) {
                result = order == GGML_SORT_ORDER_ASC ?
                    cub::DeviceRadixSort::SortPairs(nullptr, temp, keys, keys, indices, indices, ncols, 0, sizeof(float)*8) :
                    cub::DeviceRadixSort::SortPairsDescending(nullptr, temp, keys, keys, indices, indices, ncols, 0, sizeof(float)*8);
            } else if (capture) {
                result = order == GGML_SORT_ORDER_ASC ?
                    cub::DeviceSegmentedRadixSort::SortPairs(nullptr, temp, keys, keys, indices, indices, items, rows, offsets, offsets + 1, 0, sizeof(float)*8) :
                    cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr, temp, keys, keys, indices, indices, items, rows, offsets, offsets + 1, 0, sizeof(float)*8);
            } else {
                result = order == GGML_SORT_ORDER_ASC ?
                    cub::DeviceSegmentedSort::SortPairs(nullptr, temp, keys, keys, indices, indices, items, rows, offsets, offsets + 1) :
                    cub::DeviceSegmentedSort::SortPairsDescending(nullptr, temp, keys, keys, indices, indices, items, rows, offsets, offsets + 1);
            }
            if (result != cudaSuccess) { return false; }
            size_t peak = arrays;
            if (!add(peak, temp)) { return false; }
            measured.pool_bytes = std::max(measured.pool_bytes, peak);
            mix(measured.identity, rows); mix(measured.identity, capture); mix(measured.identity, temp);
        }
    }
    resources = measured; return true;
#else
    return false;
#endif
}

namespace {
struct source_sort_test_pool : ggml_cuda_pool {
    char * data;
    size_t capacity, used = 0, peak = 0;
    source_sort_test_pool(void * data, size_t capacity) : data(static_cast<char *>(data)), capacity(capacity) {}
    void * alloc(size_t bytes, size_t * actual) override {
        size_t aligned = 0;
        if (!ggml_cuda_source_sort_detail::add(aligned, bytes) || aligned > capacity - used) { throw std::bad_alloc(); }
        *actual = aligned;
        void * result = data + used;
        used += aligned; peak = std::max(peak, used);
        return result;
    }
    void free(void * ptr, size_t bytes) override {
        GGML_ASSERT(bytes <= used && ptr == data + used - bytes);
        used -= bytes;
    }
};
}

bool ggml_cuda_source_sort_capture_for_test(int device, ggml_tensor * dst,
        ggml_backend_buffer * scratch, size_t capacity, ggml_cuda_source_sort_test_result & result) {
    result = {};
    ggml_cuda_source_sort_resources resources;
    if (!dst || (dst->op != GGML_OP_ARGSORT && dst->op != GGML_OP_TOP_K) ||
            !(dst->op == GGML_OP_TOP_K ? ggml_cuda_top_k_prepare_resources(device, dst, resources) :
                ggml_cuda_argsort_prepare_resources(device, dst, resources)) ||
            capacity < resources.pool_bytes || !scratch || scratch->buft != ggml_backend_cuda_buffer_type(device) ||
            capacity > ggml_backend_buffer_get_size(scratch) ||
            reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(scratch)) % 256) { return false; }
    for (const auto * tensor : {static_cast<const ggml_tensor *>(dst->src[0]), static_cast<const ggml_tensor *>(dst)}) {
        if (!tensor->buffer || tensor->buffer->buft != ggml_backend_cuda_buffer_type(device)) { return false; }
        const auto base = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(tensor->buffer));
        const auto p = reinterpret_cast<uintptr_t>(tensor->data);
        const auto bytes = ggml_backend_buffer_get_size(tensor->buffer);
        if (!p || p < base || p - base > bytes || ggml_nbytes(tensor) > bytes - (p - base)) { return false; }
    }
    const auto in = reinterpret_cast<uintptr_t>(dst->src[0]->data), out = reinterpret_cast<uintptr_t>(dst->data);
    const auto work = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(scratch));
    const auto in_bytes = ggml_nbytes(dst->src[0]), out_bytes = ggml_nbytes(dst);
    if (in_bytes > UINTPTR_MAX - in || out_bytes > UINTPTR_MAX - out || capacity > UINTPTR_MAX - work ||
            (in < out + out_bytes && out < in + in_bytes) ||
            (work < in + in_bytes && in < work + capacity) ||
            (work < out + out_bytes && out < work + capacity)) { return false; }
    ggml_backend_cuda_context context(device);
    auto pool = std::make_unique<source_sort_test_pool>(reinterpret_cast<void *>(work), capacity);
    auto * measured = pool.get();
    context.pools[device][0] = std::move(pool);
    const auto stream = context.stream();
    const auto compute = [&] { dst->op == GGML_OP_TOP_K ? ggml_cuda_op_top_k(context, dst) : ggml_cuda_op_argsort(context, dst); };
    compute();
    if (cudaStreamSynchronize(stream) != cudaSuccess || measured->used) { return false; }
    result.eager_peak = measured->peak; measured->peak = 0;
#ifdef USE_CUDA_GRAPH
    struct capture_guard {
        cudaStream_t stream;
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        bool capturing = false;
        ~capture_guard() {
            if (capturing) { (void) cudaStreamEndCapture(stream, &graph); }
            if (exec) { (void) cudaGraphExecDestroy(exec); }
            if (graph) { (void) cudaGraphDestroy(graph); }
        }
    } capture{stream};
    if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { return false; }
    capture.capturing = true;
    compute();
    const auto ended = cudaStreamEndCapture(stream, &capture.graph);
    capture.capturing = false;
    if (ended != cudaSuccess || !capture.graph || measured->used ||
            cudaGraphInstantiate(&capture.exec, capture.graph, nullptr, nullptr, 0) != cudaSuccess) { return false; }
    result.capture_peak = measured->peak;
    for (int replay = 0; replay < 2; ++replay) {
        if (cudaGraphLaunch(capture.exec, stream) != cudaSuccess) { return false; }
        ++result.replays;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) { return false; }
#endif
    if (capacity <= SIZE_MAX - 256) {
        size_t actual = 0;
        bool rejected = false;
        try { (void) measured->alloc(capacity + 256, &actual); } catch (const std::bad_alloc &) { rejected = true; }
        if (!rejected || measured->used) { return false; }
    }
    return result.eager_peak <= capacity && result.capture_peak <= capacity;
}
