#include "ggml-cuda.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-cuda/allreduce.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/reuse.cuh"
#include "ggml-cuda/acc.cuh"
#include "ggml-cuda/add-id.cuh"
#include "ggml-cuda/arange.cuh"
#include "ggml-cuda/argmax.cuh"
#include "ggml-cuda/argsort.cuh"
#include "ggml-cuda/binbcast.cuh"
#include "ggml-cuda/clamp.cuh"
#include "ggml-cuda/col2im-1d.cuh"
#include "ggml-cuda/concat.cuh"
#include "ggml-cuda/conv-transpose-1d.cuh"
#include "ggml-cuda/conv2d.cuh"
#include "ggml-cuda/conv2d-dw.cuh"
#include "ggml-cuda/conv2d-transpose.cuh"
#include "ggml-cuda/conv3d.cuh"
#include "ggml-cuda/convert.cuh"
#include "ggml-cuda/count-equal.cuh"
#include "ggml-cuda/cpy.cuh"
#include "ggml-cuda/cross-entropy-loss.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/diagmask.cuh"
#include "ggml-cuda/diag.cuh"
#include "ggml-cuda/fattn.cuh"
#include "ggml-cuda/fwht.cuh"
#include "ggml-cuda/getrows.cuh"
#include "ggml-cuda/im2col.cuh"
#include "ggml-cuda/mmf.cuh"
#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/mmvf.cuh"
#include "ggml-cuda/mmvq.cuh"
#include "ggml-cuda/moe-weighted-reduction.cuh"
#include "ggml-cuda/norm.cuh"
#include "ggml-cuda/opt-step-adamw.cuh"
#include "ggml-cuda/opt-step-sgd.cuh"
#include "ggml-cuda/out-prod.cuh"
#include "ggml-cuda/pad.cuh"
#include "ggml-cuda/pool2d.cuh"
#include "ggml-cuda/pool1d.cuh"
#include "ggml-cuda/quantize.cuh"
#include "ggml-cuda/rope.cuh"
#include "ggml-cuda/roll.cuh"
#include "ggml-cuda/scale.cuh"
#include "ggml-cuda/snake.cuh"
#include "ggml-cuda/softcap.cuh"
#include "ggml-cuda/softmax.cuh"
#include "ggml-cuda/ssm-conv.cuh"
#include "ggml-cuda/ssm-scan.cuh"
#include "ggml-cuda/sum.cuh"
#include "ggml-cuda/sumrows.cuh"
#include "ggml-cuda/top-k.cuh"
#include "ggml-cuda/mean.cuh"
#include "ggml-cuda/tsembd.cuh"
#include "ggml-cuda/topk-moe.cuh"
#include "ggml-cuda/unary.cuh"
#include "ggml-cuda/upscale.cuh"
#include "ggml-cuda/wkv.cuh"
#include "ggml-cuda/gla.cuh"
#include "ggml-cuda/gated_delta_net.cuh"
#include "ggml-cuda/dsv4-hc.cuh"
#include "ggml-cuda/set.cuh"
#include "ggml-cuda/set-rows.cuh"
#include "ggml-cuda/pad_reflect_1d.cuh"
#include "ggml-cuda/solve_tri.cuh"
#include "ggml-cuda/tri.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/fill.cuh"
#include "ggml-cuda/lightning-indexer.cuh"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cinttypes>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cfloat>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static_assert(sizeof(half) == sizeof(ggml_fp16_t), "wrong fp16 size");

#define GGML_LOG_WARN_ONCE(str) \
    { static std::once_flag warn_flag; std::call_once(warn_flag, []() { GGML_LOG_WARN(str); }); }

[[noreturn]]
void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    int id = -1; // in case cudaGetDevice fails
    (void)cudaGetDevice(&id);

    GGML_LOG_ERROR(GGML_CUDA_NAME " error: %s\n", msg);
    GGML_LOG_ERROR("  current device: %d, in function %s at %s:%d\n", id, func, file, line);
    GGML_LOG_ERROR("  %s\n", stmt);
    // abort with GGML_ABORT to get a stack trace
    GGML_ABORT(GGML_CUDA_NAME " error");
}

// map a (possibly virtual) device id to the physical CUDA device that backs it
static int ggml_cuda_get_physical_device(int device) {
    const ggml_cuda_device_info & info = ggml_cuda_info();
    GGML_ASSERT(device >= 0 && device < info.device_count);
    return info.devices[device].physical_device;
}

// this is faster on Windows
// probably because the Windows CUDA libraries forget to make this check before invoking the drivers
void ggml_cuda_set_device(int device) {
    // translate the (possibly virtual) device id to the physical CUDA device that backs it
    const int physical_device = ggml_cuda_get_physical_device(device);

    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));

    if (physical_device == current_device) {
        return;
    }

    CUDA_CHECK(cudaSetDevice(physical_device));
}

int ggml_cuda_get_device() {
    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

static cudaError_t ggml_cuda_device_malloc(void ** ptr, size_t size, int device) {
    ggml_cuda_set_device(device);
    cudaError_t err;
    if (getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != nullptr) {
        err = cudaMallocManaged(ptr, size);
#if defined(GGML_USE_HIP)
        if (err == hipSuccess) {
            // hipMemAdviseSetCoarseGrain is an optional performance hint;
            // ignore errors (e.g. hipErrorInvalidValue on some APU/iGPU configs).
            (void)cudaMemAdvise(*ptr, size, hipMemAdviseSetCoarseGrain, device);
            (void)hipGetLastError(); // clear any error
        }

        // fall back to cudaMalloc if not supported (e.g. on Windows)
        if (err == hipErrorNotSupported) {
            static bool warned_unsupported = false;
            if (!warned_unsupported) {
                GGML_LOG_WARN("hipMallocManaged unsupported, falling back to hipMalloc.\n");
                warned_unsupported = true;
            }

            err = cudaMalloc(ptr, size);
        }
#endif // defined(GGML_USE_HIP)
    } else {
        err = cudaMalloc(ptr, size);
    }
    return err;
}

#if defined(GGML_USE_HIP)
static int ggml_cuda_parse_id(char devName[]) {
    // A list of possible Target IDs can be found under the rocclr/clr repo in device.cpp
    // these values are not stable so this is susceptible to breakage
    // https://github.com/ROCm/clr/blob/amd-staging/rocclr/device/device.cpp
    int archMajor = 0x0;
    int archMinor = 0x0;
    int archNum = GGML_CUDA_CC_OFFSET_AMD;
    int archLen = strlen(devName);
    char archName[archLen + 1];

    // strip leading 'gfx' while copying into our buffer
    if (archLen > 3) {
        strcpy(archName, &devName[3]);
        archLen -= 3;
    }

    // trim trailing :xnack- or :sramecc- statuses
    archLen = strcspn(archName, ":");
    archName[archLen] = '\0';

    // tease out the version information
    if (archLen > 8) {
        // versions labeled generic use '-' as delimiter
        // strip the trailing "-generic" then iterate through what remains
        if ((strstr(archName, "-generic"))) {
            archName[archLen - 8] = '\0';
            char * pch;
            if ((pch = strtok(archName, "-"))) {
                archMajor = (int)strtoul(pch, 0, 16);
                if ((pch = strtok(NULL, "-"))) {
                    archMinor = 0x10 * (int)strtoul(pch, 0, 16);
                }
            }
        }
    } else if (archLen >= 3) {
        // last two digits should be the minor * 0x10 + stepping
        archMinor = (int)strtoul(&archName[archLen - 2], 0, 16);
        archName[archLen - 2] = '\0';

        // only the major version remains
        archMajor = (int)strtoul(archName, 0, 16);
    }
    archNum += archMajor * 0x100;
    archNum += archMinor;

    return archNum;
}
#endif // defined(GGML_USE_HIP)

static ggml_cuda_device_info ggml_cuda_init() {
    ggml_cuda_device_info info = {};

    cudaError_t err = cudaGetDeviceCount(&info.physical_device_count);
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: failed to initialize " GGML_CUDA_NAME ": %s\n", __func__, cudaGetErrorString(err));
        return info;
    }

    GGML_ASSERT(info.physical_device_count <= GGML_CUDA_MAX_DEVICES);

    // by default expose exactly the physical devices; GGML_CUDA_DEVICES can request a different
    // number of (virtual) devices to emulate multi-GPU systems on a machine with fewer GPUs
    info.device_count = info.physical_device_count;

    const char * devices_env = getenv("GGML_CUDA_DEVICES");
    if (devices_env != nullptr && info.physical_device_count > 0) {
        const int requested = atoi(devices_env);
        if (requested > 0) {
            info.device_count = requested;
        } else {
            GGML_LOG_WARN("%s: ignoring invalid GGML_CUDA_DEVICES=\"%s\"\n", __func__, devices_env);
        }
    }

    if (info.device_count > GGML_CUDA_MAX_DEVICES) {
        GGML_LOG_WARN("%s: requested %d devices, clamping to GGML_CUDA_MAX_DEVICES=%d\n",
                      __func__, info.device_count, GGML_CUDA_MAX_DEVICES);
        info.device_count = GGML_CUDA_MAX_DEVICES;
    }

    // map each (virtual) device to a backing physical device (round-robin), assign each its index
    // among the (virtual) devices sharing that physical GPU, and store the per-physical share count
    int physical_share_count[GGML_CUDA_MAX_DEVICES] = {};
    GGML_ASSERT(info.device_count == 0 || info.physical_device_count > 0);
    for (int id = 0; id < info.device_count; ++id) {
        info.devices[id].physical_device = id % info.physical_device_count;
        info.devices[id].virtual_index  = physical_share_count[info.devices[id].physical_device]++;
    }

    int64_t total_vram = 0;
    for (int id = 0; id < info.physical_device_count; ++id) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
        total_vram += prop.totalGlobalMem;
    }
    GGML_LOG_INFO("%s: found %d " GGML_CUDA_NAME " devices (Total VRAM: %zu MiB):\n",
                  __func__, info.physical_device_count, (size_t)(total_vram / (1024 * 1024)));
    if (info.device_count != info.physical_device_count) {
        GGML_LOG_INFO("%s: emulating %d virtual device(s) on %d physical device(s) (GGML_CUDA_DEVICES)\n",
                      __func__, info.device_count, info.physical_device_count);
    }
    total_vram = 0;

    std::vector<std::pair<int, std::string>> turing_devices_without_mma;
    for (int id = 0; id < info.device_count; ++id) {
        const int physical_id = info.devices[id].physical_device;

        int device_vmm = 0;

#if defined(GGML_USE_VMM)
        CUdevice device;
        CU_CHECK(cuDeviceGet(&device, physical_id));
        CU_CHECK(cuDeviceGetAttribute(&device_vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device));

        if (device_vmm) {
            CUmemAllocationProp alloc_prop = {};
            alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            alloc_prop.location.id = physical_id;
            CU_CHECK(cuMemGetAllocationGranularity(&info.devices[id].vmm_granularity, &alloc_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        }
#endif // defined(GGML_USE_VMM)
        info.devices[id].vmm = !!device_vmm;

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, physical_id));

        // a virtual device owns only a share of its physical GPU's memory; report that share so the
        // logged per-device VRAM sums to the physical total above.
        GGML_ASSERT(physical_share_count[physical_id] > 0);
        info.devices[id].physical_share_count = physical_share_count[physical_id];
        const size_t device_vram = prop.totalGlobalMem / info.devices[id].physical_share_count;
        const size_t device_vram_mib = device_vram / (1024 * 1024);

        info.default_tensor_split[id] = total_vram;
        total_vram += device_vram;
        info.devices[id].integrated = false; // Temporarily disabled due to issues with corrupted output (e.g. #15034)
        info.devices[id].nsm        = prop.multiProcessorCount;
        info.devices[id].smpb       = prop.sharedMemPerBlock;
        info.devices[id].warp_size  = prop.warpSize;

        int supports_coop_launch = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&supports_coop_launch, cudaDevAttrCooperativeLaunch, physical_id));
        info.devices[id].supports_cooperative_launch = !!supports_coop_launch;

#if defined(GGML_USE_HIP)
        info.devices[id].smpbo = prop.sharedMemPerBlock;

        info.devices[id].cc = ggml_cuda_parse_id(prop.gcnArchName);
        if ((info.devices[id].cc & 0xff00) == 0x0) {
            GGML_LOG_WARN("invalid architecture ID received for device %d %s: %s  cc %d.%d\n",
                            id, prop.name, prop.gcnArchName, prop.major, prop.minor);

            // Fallback to prop.major and prop.minor
            if (prop.major > 0) {
                info.devices[id].cc = GGML_CUDA_CC_OFFSET_AMD + prop.major * 0x100;
                info.devices[id].cc += prop.minor * 0x10;
            }
        }
        GGML_LOG_INFO("  Device %d: %s, %s (0x%x), VMM: %s, Wave Size: %d, VRAM: %zu MiB\n",
                      id, prop.name, prop.gcnArchName, info.devices[id].cc & 0xffff,
                      device_vmm ? "yes" : "no", prop.warpSize,
                      device_vram_mib);
#elif defined(GGML_USE_MUSA)
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = GGML_CUDA_CC_OFFSET_MTHREADS + prop.major * 0x100;
        info.devices[id].cc += prop.minor * 0x10;
        GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s, VRAM: %zu MiB\n",
                      id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no",
                      device_vram_mib);
#else
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = 100*prop.major + 10*prop.minor;
        GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s, VRAM: %zu MiB\n",
                      id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no",
                      device_vram_mib);
        std::string device_name(prop.name);
        if (device_name == "NVIDIA GeForce MX450") {
            turing_devices_without_mma.push_back({ id, device_name });
        } else if (device_name == "NVIDIA GeForce MX550") {
            turing_devices_without_mma.push_back({ id, device_name });
        } else if (device_name.substr(0, 21) == "NVIDIA GeForce GTX 16") {
            turing_devices_without_mma.push_back({ id, device_name });
        }

        // Temporary performance fix:
        // Setting device scheduling strategy for iGPUs with cc121 to "spinning" to avoid delays in cuda synchronize calls.
        // TODO: Check for future drivers the default scheduling strategy and
        // remove this call again when cudaDeviceScheduleSpin is default.
        if (prop.major == 12 && prop.minor == 1) {
            CUDA_CHECK(cudaSetDevice(physical_id));
            CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
        }

#endif  // defined(GGML_USE_HIP)
    }

    if (ggml_cuda_highest_compiled_arch(GGML_CUDA_CC_TURING) >= GGML_CUDA_CC_TURING && !turing_devices_without_mma.empty()) {
        GGML_LOG_INFO("The following devices will have suboptimal performance due to a lack of tensor cores:\n");
        for (size_t device_pos = 0; device_pos < turing_devices_without_mma.size(); device_pos++) {
            GGML_LOG_INFO(
                "  Device %d: %s\n", turing_devices_without_mma[device_pos].first, turing_devices_without_mma[device_pos].second.c_str());
        }
        GGML_LOG_INFO(
            "Consider compiling with CMAKE_CUDA_ARCHITECTURES=61-virtual;80-virtual and DGGML_CUDA_FORCE_MMQ to force the use of the Pascal code for Turing.\n");
    }

    for (int id = 0; id < info.device_count; ++id) {
        info.default_tensor_split[id] /= total_vram;
    }

    // configure logging to stdout
    // CUBLAS_CHECK(cublasLoggerConfigure(1, 1, 0, nullptr));

    if (getenv("GGML_CUDA_P2P") != nullptr) {
        for (int id = 0; id < info.physical_device_count; ++id) {
            CUDA_CHECK(cudaSetDevice(id));
            for (int id_other = 0; id_other < info.physical_device_count; ++id_other) {
                if (id == id_other) {
                    continue;
                }
                int can_access_peer;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id, id_other));
                if (can_access_peer) {
                    CUDA_CHECK(cudaDeviceEnablePeerAccess(id_other, 0));
                }
            }
        }
    }

    return info;
}

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = ggml_cuda_init();
    return info;
}

// #define DEBUG_CUDA_MALLOC

// buffer pool for cuda (legacy)
struct ggml_cuda_pool_leg : public ggml_cuda_pool {
    static const int MAX_BUFFERS = 256;

    int device;
    struct ggml_cuda_buffer {
        void * ptr = nullptr;
        size_t size = 0;
    };

    ggml_cuda_buffer buffer_pool[MAX_BUFFERS] = {};
    size_t pool_size = 0;

    explicit ggml_cuda_pool_leg(int device) :
        device(device) {
    }

    ~ggml_cuda_pool_leg() {
        clear_pool();
        GGML_ASSERT(pool_size == 0);
    }

    void clear_pool() {
        ggml_cuda_set_device(device);
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
                CUDA_CHECK(cudaFree(b.ptr));
                pool_size -= b.size;
                b.ptr  = nullptr;
                b.size = 0;
            }
        }
    }

    void * alloc(size_t size, size_t * actual_size) override {
#ifdef DEBUG_CUDA_MALLOC
        int nnz = 0;
        size_t max_size = 0;
#endif
        size_t best_diff = 1ull << 36;
        int ibest = -1;
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr != nullptr) {
#ifdef DEBUG_CUDA_MALLOC
                ++nnz;
                if (b.size > max_size) max_size = b.size;
#endif
                if (b.size >= size) {
                    size_t diff = b.size - size;
                    if (diff < best_diff) {
                        best_diff = diff;
                        ibest = i;
                        if (!best_diff) {
                            void * ptr = b.ptr;
                            *actual_size = b.size;
                            b.ptr = nullptr;
                            b.size = 0;
                            return ptr;
                        }
                    }
                }
            }
        }
        if (ibest >= 0) {
            ggml_cuda_buffer& b = buffer_pool[ibest];
            void * ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            return ptr;
        }
        void * ptr;
        size_t look_ahead_size = (size_t) (1.05 * size);
        look_ahead_size = 256 * ((look_ahead_size + 255)/256);
        ggml_cuda_set_device(device);
        cudaError_t err = ggml_cuda_device_malloc(&ptr, look_ahead_size, device);
        if (err == cudaErrorMemoryAllocation) {
            (void)cudaGetLastError();
            const size_t cached_bytes = pool_size;
            GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: alloc of %.2f MiB failed, flushing %.2f MiB of cached buffers and retrying\n",
                           device, look_ahead_size/1024.0/1024.0, cached_bytes/1024.0/1024.0);
            CUDA_CHECK(cudaDeviceSynchronize());
            clear_pool();
            err = ggml_cuda_device_malloc(&ptr, look_ahead_size, device);
            if (err == cudaSuccess) {
                GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: retry succeeded\n", device);
            }
        }
        CUDA_CHECK(err);
        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;
#ifdef DEBUG_CUDA_MALLOC
        GGML_LOG_INFO("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, device, nnz,
                           (uint32_t)(max_size / 1024 / 1024), (uint32_t)(pool_size / 1024 / 1024), (uint32_t)(size / 1024 / 1024));
#endif
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr = ptr;
                b.size = size;
                return;
            }
        }
        GGML_LOG_DEBUG(GGML_CUDA_NAME " buffer pool full, increase MAX_CUDA_BUFFERS\n");
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaFree(ptr));
        pool_size -= size;
    }
};

// pool with virtual memory
#if defined(GGML_USE_VMM)
struct ggml_cuda_pool_vmm : public ggml_cuda_pool {
    static const size_t CUDA_POOL_VMM_MAX_SIZE = 1ull << 35; // 32 GB

    int device;
    int physical_device;
    CUdeviceptr pool_addr = 0;
    size_t pool_used = 0;
    size_t pool_size = 0;
    size_t granularity;
#if defined(GGML_USE_HIP)
    std::vector<std::pair<CUdeviceptr, size_t>> mappings;
#endif

    explicit ggml_cuda_pool_vmm(int device) :
        device(device),
        physical_device(ggml_cuda_get_physical_device(device)),
        granularity(ggml_cuda_info().devices[device].vmm_granularity) {
    }

    ~ggml_cuda_pool_vmm() {
        if (pool_addr != 0) {
#if defined(GGML_USE_HIP)
            // Workaround for https://github.com/ROCm/ROCR-Runtime/issues/285
            for (std::pair<CUdeviceptr, size_t> & mapping : mappings) {
                CU_CHECK(cuMemUnmap(mapping.first, mapping.second));
            }
#else
            CU_CHECK(cuMemUnmap(pool_addr, pool_size));
#endif
            CU_CHECK(cuMemAddressFree(pool_addr, CUDA_POOL_VMM_MAX_SIZE));
        }
    }

    void * alloc(size_t size, size_t * actual_size) override {
        // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
        const size_t alignment = 128;
        size = alignment * ((size + alignment - 1) / alignment);

        size_t avail = pool_size - pool_used;

        if (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_size = size - avail;
            reserve_size = granularity * ((reserve_size + granularity - 1) / granularity);

            GGML_ASSERT(pool_size + reserve_size <= CUDA_POOL_VMM_MAX_SIZE);

            // allocate more physical memory
            CUmemAllocationProp prop = {};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = physical_device;
            CUmemGenericAllocationHandle handle;
            CU_CHECK(cuMemCreate(&handle, reserve_size, &prop, 0));

            // reserve virtual address space (if not already reserved)
            if (pool_addr == 0) {
                CU_CHECK(cuMemAddressReserve(&pool_addr, CUDA_POOL_VMM_MAX_SIZE, 0, 0, 0));
            }

            // map at the end of the pool
            CUdeviceptr start_ptr = (CUdeviceptr)((char *)(pool_addr) + pool_size);
            CU_CHECK(cuMemMap(start_ptr, reserve_size, 0, handle, 0));
#if defined(GGML_USE_HIP)
            mappings.push_back({start_ptr, reserve_size});
#endif

            // the memory allocation handle is no longer needed after mapping
            CU_CHECK(cuMemRelease(handle));

            // VMM Bug fix for P2P access if GGML_CUDA_P2P is set, or if NCCL build
            bool use_peer_access = getenv("GGML_CUDA_P2P") != nullptr;
#if defined(GGML_USE_NCCL)
            use_peer_access = true;
#endif // defined(GGML_USE_NCCL)

            if (use_peer_access) {
                // NCCL implicitly enables peer access (cudaDeviceEnablePeerAccess), and
                // GGML_CUDA_P2P enables it explicitly. Unlike cudaMalloc buffers, VMM
                // allocations do not become peer-accessible from that alone, so access
                // must be granted explicitly here. With virtual devices, grant access
                // on the backing *physical* devices (deduplicated, since several
                // virtual devices can map to the same physical GPU).
                std::vector<CUmemAccessDesc> access_descs;
                bool physical_seen[GGML_CUDA_MAX_DEVICES] = {};
                const int device_count = ggml_cuda_info().device_count;
                for (int id = 0; id < device_count; ++id) {
                    const int id_physical = ggml_cuda_get_physical_device(id);
                    if (id_physical != physical_device) {
                        int can_access_peer = 0;
                        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id_physical, physical_device));
                        if (!can_access_peer) {
                            continue;
                        }
                    }
                    if (physical_seen[id_physical]) {
                        continue;
                    }
                    physical_seen[id_physical] = true;
                    CUmemAccessDesc access = {};
                    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                    access.location.id = id_physical;
                    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                    access_descs.push_back(access);
                }
                CU_CHECK(cuMemSetAccess(start_ptr, reserve_size, access_descs.data(), access_descs.size()));
            } else {
                // set access for non P2P
                CUmemAccessDesc access = {};
                access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                access.location.id = physical_device;
                access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                CU_CHECK(cuMemSetAccess(start_ptr, reserve_size, &access, 1));
            }

            // add to the pool
            pool_size += reserve_size;

            //printf("cuda pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
            //       device, (unsigned long long) (pool_size/1024/1024),
            //       (unsigned long long) (reserve_size/1024/1024));
        }

        GGML_ASSERT(pool_addr != 0);

        void * ptr = (void *) ((CUdeviceptr)((char *)(pool_addr) + pool_used));
        *actual_size = size;
        pool_used += size;

#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: allocated %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        return ptr;
    }

    void free(void * ptr, size_t size) override {
#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: freed %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        pool_used -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == (void *) ((char *)(pool_addr) + pool_used));
    }
};
#endif // defined(GGML_USE_VMM)

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int                  device,
                                                                               [[maybe_unused]] int stream_no) {
#if defined(GGML_USE_VMM)
    if (ggml_cuda_info().devices[device].vmm) {
        return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_vmm(device));
    }
#endif // defined(GGML_USE_VMM)
    return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_leg(device));
}

// destroying a cuBLAS handle while a graph is being captured in a different thread can result in a CUDA error
// this lock is used to ensure that no cuBLAS handle is destroyed while a graph is being captured

static std::mutex ggml_cuda_lock;
static std::condition_variable ggml_cuda_lock_cv;
static std::atomic<int> ggml_cuda_lock_counter;

ggml_backend_cuda_context::~ggml_backend_cuda_context() {
    std::unique_lock<std::mutex> lock(ggml_cuda_lock);
    ggml_cuda_lock_cv.wait(lock, []{ return ggml_cuda_lock_counter.load(std::memory_order_relaxed) == 0; });

    if (copy_event != nullptr) {
        CUDA_CHECK(cudaEventDestroy(copy_event));
    }
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        for (int j = 0; j < GGML_CUDA_MAX_STREAMS; ++j) {
            if (streams[i][j] != nullptr) {
                CUDA_CHECK(cudaStreamDestroy(streams[i][j]));
            }
            if (cublas_handles[i][j] != nullptr) {
                CUBLAS_CHECK(cublasDestroy(cublas_handles[i][j]));
            }
            if (cublas_workspaces[i][j] != nullptr) {
                CUDA_CHECK(cudaFree(cublas_workspaces[i][j]));
            }
        }
    }
}


// cuda buffer

struct ggml_backend_cuda_buffer_context {
    int device;
    void * dev_ptr = nullptr;
    std::string name;

    ggml_backend_cuda_buffer_context(int device, void * dev_ptr) :
        device(device), dev_ptr(dev_ptr),
        name(GGML_CUDA_NAME + std::to_string(device)) {
    }

    ~ggml_backend_cuda_buffer_context() {
        CUDA_CHECK(cudaFree(dev_ptr));
    }
};

static void ggml_backend_cuda_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    delete ctx;
}

static bool ggml_backend_buffer_is_cuda(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_cuda_buffer_free_buffer;
}

static void * ggml_backend_cuda_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    return ctx->dev_ptr;
}

static enum ggml_status ggml_backend_cuda_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    if (tensor->view_src != NULL) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        return GGML_STATUS_SUCCESS;
    }

    if (ggml_is_quantized(tensor->type) && tensor->view_src == nullptr && ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        // initialize padding to 0 to avoid possible NaN values
        const size_t original_size = ggml_nbytes(tensor);
        const size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size) {
            ggml_cuda_set_device(ctx->device);
            CUDA_CHECK(cudaMemset((char *)tensor->data + original_size, 0, padded_size - original_size));
        }
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cuda_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync((char *) tensor->data + offset, value, size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpy2DAsync(
        (char *) tensor->data + offset, stride_tensor, data, stride_data, size, n_copies, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpy2DAsync(
        data, stride_data, (const char *) tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static bool ggml_backend_cuda_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_cuda(src->buffer)) {
        ggml_backend_cuda_buffer_context * src_ctx = (ggml_backend_cuda_buffer_context *)src->buffer->context;
        ggml_backend_cuda_buffer_context * dst_ctx = (ggml_backend_cuda_buffer_context *)dst->buffer->context;
        // compare the backing physical devices: distinct virtual devices may share one physical GPU,
        // in which case a same-device copy (not a peer copy) is required
        const int src_physical = ggml_cuda_get_physical_device(src_ctx->device);
        const int dst_physical = ggml_cuda_get_physical_device(dst_ctx->device);
        if (src_physical == dst_physical) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(src), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, ggml_nbytes(src), cudaStreamPerThread));
#endif
        }
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cuda_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync(ctx->dev_ptr, value, buffer->size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static const ggml_backend_buffer_i ggml_backend_cuda_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_cuda_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cuda_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cuda_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_cuda_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cuda_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cuda_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_cuda_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_cuda_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_cuda_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cuda_buffer_clear,
    /* .reset           = */ NULL,
};

// cuda buffer type
struct ggml_backend_cuda_buffer_type_context {
    int device;
    std::string name;
};

static const char * ggml_backend_cuda_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_buffer_type_context * ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static bool ggml_backend_buft_is_cuda(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_buffer_type_get_name;
}

static ggml_backend_buffer_t ggml_backend_cuda_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    ggml_cuda_set_device(buft_ctx->device);

    void * dev_ptr;
    cudaError_t err = ggml_cuda_device_malloc(&dev_ptr, size, buft_ctx->device);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d: cudaMalloc failed: %s\n", __func__, size / 1024.0 / 1024.0, buft_ctx->device, cudaGetErrorString(err));
        return nullptr;
    }

    ggml_backend_cuda_buffer_context * ctx = new ggml_backend_cuda_buffer_context(buft_ctx->device, dev_ptr);

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, size);
}

static size_t ggml_backend_cuda_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_cuda_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *) buft->context;

    size_t size = tensor->op == GGML_OP_FLASH_ATTN_EXT
        ? ggml_cuda_flash_attn_ext_get_alloc_size(buft_ctx->device, tensor)
        : ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    // [TAG_ALLOC_SIZE_EXPAND]
    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            GGML_ASSERT(tensor->nb[0] == ggml_element_size(tensor));
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;
}

static const ggml_backend_buffer_type_i ggml_backend_cuda_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_cuda_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_cuda_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_cuda_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_cuda_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_cuda_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type ggml_backend_cuda_buffer_types[GGML_CUDA_MAX_DEVICES];

    static bool ggml_backend_cuda_buffer_type_initialized = false;

    if (!ggml_backend_cuda_buffer_type_initialized) {
        for (int i = 0; i < ggml_backend_cuda_get_device_count(); i++) {
            ggml_backend_cuda_buffer_types[i] = {
                /* .iface    = */ ggml_backend_cuda_buffer_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), i),
                /* .context  = */ new ggml_backend_cuda_buffer_type_context{i, GGML_CUDA_NAME + std::to_string(i)},
            };
        }
        ggml_backend_cuda_buffer_type_initialized = true;
    }

    return &ggml_backend_cuda_buffer_types[device];
}

// Communication context for multi-GPU AllReduce during tensor parallelism.
//
// Created once per meta backend instance.  Resources for the selected mode
// (NCCL communicators or the internal AllReduce pipeline) are initialised
// eagerly during comm_init so any init failure surfaces at startup rather
// than mid-run.
struct ggml_backend_cuda_comm_context {
    using try_allreduce_fn = bool(*)(ggml_backend_cuda_comm_context *, struct ggml_tensor **);

    std::vector<ggml_backend_t> backends;
    std::vector<int>            dev_ids;

    // Set by the init chain (comm_init_{nccl, internal, none}) to one of
    // try_allreduce_{nccl, internal, butterfly}.  nccl needs `comms`,
    // internal needs `ar_pipeline`, butterfly needs nothing.  Per-call
    // failures return false; the meta backend's generic implementation then
    // handles that call.
    try_allreduce_fn            try_allreduce = nullptr;

    ggml_cuda_ar_pipeline *     ar_pipeline = nullptr;

#ifdef GGML_USE_NCCL
    std::vector<ncclComm_t>     comms;
#endif // GGML_USE_NCCL

    ~ggml_backend_cuda_comm_context() {
#ifdef GGML_USE_NCCL
        for (ncclComm_t comm : comms) {
            NCCL_CHECK(ncclCommDestroy(comm));
        }
#endif // GGML_USE_NCCL
        ggml_cuda_ar_pipeline_free(ar_pipeline);
    }
};

#ifdef GGML_USE_NCCL
// AllReduce via NCCL. Reduces as FP32 for small tensors and BF16 for large
// tensors (bandwidth-bound), then converts back to FP32.
static bool ggml_backend_cuda_comm_allreduce_nccl(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    const int64_t ne = ggml_nelements(tensors[0]);
    // FIXME the input of llm_graph_context::build_in_out_ids can produce a tensor with 0 elements if n_outputs == 0
    // This then causes a crash in this function
    if (ne == 0) {
        return true;
    }

    const size_t n_backends = comm_ctx->backends.size();

    for (size_t i = 0; i < n_backends; ++i) {
        GGML_ASSERT(tensors[i] != nullptr);
        GGML_ASSERT(ggml_nelements(tensors[i]) == ne);
        GGML_ASSERT(ggml_is_contiguously_allocated(tensors[i]));
    }

    // For small tensors, simply reduce them as FP32.
    // The following heuristic for how "small" a tensor should be is based on RTX 4090s connected via 16x PCIe 4.0.
    if ((n_backends <= 2 && ne < 32768) || (n_backends == 3 && ne < 131072) || (n_backends >= 4 && ne < 262144)) {
        for (size_t i = 0; i < n_backends; ++i) {
            if ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
                ggml_cuda_set_device(cuda_ctx->device);
                CUDA_CHECK(cudaMemsetAsync(tensors[i]->data, 0, ggml_nbytes(tensors[i]), cuda_ctx->stream()));
            }
        }
        NCCL_CHECK(ncclGroupStart());
        for (size_t i = 0; i < n_backends; ++i) {
            ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
            NCCL_CHECK(ncclAllReduce(tensors[i]->data, tensors[i]->data, ne, ncclFloat, ncclSum, comm_ctx->comms[i], cuda_ctx->stream()));
        }
        NCCL_CHECK(ncclGroupEnd());
        return true;
    }

    // For large tensors it's faster to compress them to BF16 for the reduction:
    to_bf16_cuda_t to_bf16 = ggml_get_to_bf16_cuda(GGML_TYPE_F32);
    to_fp32_cuda_t to_fp32 = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);

    ggml_cuda_pool_alloc<nv_bfloat16> tmp[GGML_CUDA_MAX_DEVICES];
    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
        tmp[i].pool = &cuda_ctx->pool();
        tmp[i].alloc(ne);

        ggml_cuda_set_device(cuda_ctx->device);
        if (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) {
            to_bf16(tensors[i]->data, tmp[i].get(), ne, cuda_ctx->stream());
        } else {
            CUDA_CHECK(cudaMemsetAsync(tmp[i].get(), 0, ne * sizeof(nv_bfloat16), cuda_ctx->stream()));
        }
        CUDA_CHECK(cudaGetLastError());
    }

    NCCL_CHECK(ncclGroupStart());
    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
        NCCL_CHECK(ncclAllReduce(tmp[i].get(), tmp[i].get(), ne, ncclBfloat16, ncclSum, comm_ctx->comms[i], cuda_ctx->stream()));
    }
    NCCL_CHECK(ncclGroupEnd());

    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;

        ggml_cuda_set_device(cuda_ctx->device);
        to_fp32(tmp[i].get(), (float *) tensors[i]->data, ne, cuda_ctx->stream());
        CUDA_CHECK(cudaGetLastError());
    }

    return true;
}
#endif // GGML_USE_NCCL

// Run the internal AR pipeline.  Returns false on unsupported / failed input
// -- the caller decides whether to abort (env-forced) or fall back silently.
static bool ggml_backend_cuda_comm_allreduce_internal(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    GGML_ASSERT(comm_ctx->ar_pipeline != nullptr);

    const size_t n_backends = comm_ctx->backends.size();
    GGML_ASSERT(n_backends == 2);
    GGML_ASSERT(tensors[0] != nullptr);

    const int64_t   ne   = ggml_nelements(tensors[0]);
    const ggml_type type = tensors[0]->type;

    if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_BF16) {
        GGML_LOG_DEBUG("%s: internal unsupported: type=%d\n", __func__, (int) type);
        return false;
    }

    if (ne == 0) {
        return true;
    }

    for (size_t i = 0; i < n_backends; ++i) {
        if (tensors[i] == nullptr) {
            GGML_LOG_ERROR("%s: internal failed: tensor[%zu] is null\n", __func__, i);
            return false;
        }
        if (ggml_nelements(tensors[i]) != ne || tensors[i]->type != type) {
            GGML_LOG_ERROR("%s: internal failed: tensor[%zu] ne=%" PRId64 " type=%d expected ne=%" PRId64 " type=%d\n",
                           __func__, i, ggml_nelements(tensors[i]), (int) tensors[i]->type, ne, (int) type);
            return false;
        }
        if (!ggml_is_contiguously_allocated(tensors[i])) {
            GGML_LOG_DEBUG("%s: internal unsupported: tensor[%zu] is not contiguously allocated: ne=%" PRId64 " nbytes=%zu packed=%zu type=%d\n",
                           __func__, i, ne, ggml_nbytes(tensors[i]),
                           (size_t) ne * ggml_type_size(type) / ggml_blck_size(type), (int) type);
            return false;
        }
        if (((uintptr_t) tensors[i]->data & 0xF) != 0) {
            GGML_LOG_DEBUG("%s: internal unsupported: tensor[%zu] data pointer is not 16-byte aligned: %p type=%d ne=%" PRId64 "\n",
                           __func__, i, tensors[i]->data, (int) type, ne);
            return false;
        }
        GGML_ASSERT((ggml_nbytes(tensors[i]) & 0xF) == 0);
    }

    return ggml_cuda_ar_allreduce(comm_ctx->ar_pipeline, comm_ctx->backends.data(), tensors);
}

// ---------------------------------------------------------------------------
// Per-call dispatch -- three variants, one per backend.  Each is set as
// comm_ctx->try_allreduce by the matching init step.  Per-call failure
// returns false; the meta backend's generic implementation handles that call.
// ---------------------------------------------------------------------------

#ifdef GGML_USE_NCCL
static bool ggml_backend_cuda_comm_try_allreduce_nccl(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    return ggml_backend_cuda_comm_allreduce_nccl(comm_ctx, tensors);
}
#endif // GGML_USE_NCCL

static bool ggml_backend_cuda_comm_try_allreduce_internal(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    return ggml_backend_cuda_comm_allreduce_internal(comm_ctx, tensors);
}

static bool ggml_backend_cuda_comm_try_allreduce_butterfly(
        ggml_backend_cuda_comm_context *, struct ggml_tensor **) {
    return false;
}

static void ggml_backend_cuda_comm_free(void * comm_ctx_v) {
    if (comm_ctx_v == nullptr) {
        return;
    }
    delete static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
}

// ---------------------------------------------------------------------------
// Init -- chained nccl -> internal -> none.  Each step tries to bring up its
// resource; on failure it warns and recurses into the next step.
// ---------------------------------------------------------------------------
static void ggml_backend_cuda_comm_init_none(ggml_backend_cuda_comm_context * ret) {
    ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_butterfly;
}

static void ggml_backend_cuda_comm_init_internal(ggml_backend_cuda_comm_context * ret) {
    ret->ar_pipeline = ggml_cuda_ar_pipeline_init(ret->dev_ids.data(), ret->dev_ids.size());
    if (ret->ar_pipeline) {
        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_internal;
        return;
    }

    // Clear sticky CUDA error from the failed init.
    (void) cudaGetLastError();
    GGML_LOG_WARN("internal AllReduce init failed (n_devices != 2?); "
                  "falling back to meta-backend butterfly\n");
    ggml_backend_cuda_comm_init_none(ret);
}

static void ggml_backend_cuda_comm_init_nccl(ggml_backend_cuda_comm_context * ret) {
#ifdef GGML_USE_NCCL
    // Disabling NCCL path when CUDA virtual devices are in use since NCCL requires one distinct physical GPU per rank.
    const ggml_cuda_device_info & info = ggml_cuda_info();
    if (info.device_count > info.physical_device_count) {
        GGML_LOG_WARN("NCCL disabled: virtual devices in use; "
                      "falling back to internal AllReduce\n");
        ggml_backend_cuda_comm_init_internal(ret);
        return;
    }

    const size_t n = ret->dev_ids.size();
    ret->comms.resize(n);
    ncclResult_t rc = ncclCommInitAll(ret->comms.data(), (int) n, ret->dev_ids.data());
    if (rc == ncclSuccess) {
        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_nccl;
        return;
    }

    ret->comms.clear();
    GGML_LOG_WARN("NCCL init failed (%s); falling back to internal AllReduce\n",
                  ncclGetErrorString(rc));
#else // GGML_USE_NCCL
#ifndef GGML_USE_HIP
    GGML_LOG_WARN("NCCL not compiled in; falling back to internal AllReduce.  "
                  "Recompile with -DGGML_CUDA_NCCL=ON for best multi-GPU performance.\n");
#endif // !GGML_USE_HIP
#endif // GGML_USE_NCCL

    ggml_backend_cuda_comm_init_internal(ret);
}

// Top-level init.  Picks one of the three init paths based on
// GGML_CUDA_ALLREDUCE (or the platform default) and lets the chain handle
// any fallback.  Unrecognised env values warn and fall through to the
// platform default.
static void * ggml_backend_cuda_comm_init(ggml_backend_t * backends, size_t n_backends) {
    for (size_t i = 0; i < n_backends; i++) {
        if (!ggml_backend_is_cuda(backends[i])) {
            return nullptr;
        }
    }

    auto * ret = new ggml_backend_cuda_comm_context;
    ret->backends.assign(backends, backends + n_backends);
    ret->dev_ids.reserve(n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        ret->dev_ids.push_back(static_cast<ggml_backend_cuda_context *>(backends[i]->context)->device);
    }

    const char * env = getenv("GGML_CUDA_ALLREDUCE");
    if (!env) {
        // Platform default: Linux uses NCCL, otherwise (generally Windows) internal
#if defined(__linux__)
        ggml_backend_cuda_comm_init_nccl(ret);
#else
        ggml_backend_cuda_comm_init_internal(ret);
#endif // defined(__linux__)
    } else {
        std::string env_str(env);
        if (env_str == "nccl") {
            ggml_backend_cuda_comm_init_nccl(ret);
        } else if (env_str == "internal") {
            ggml_backend_cuda_comm_init_internal(ret);
        } else if (env_str == "none") {
            ggml_backend_cuda_comm_init_none(ret);
        } else {
            GGML_LOG_WARN("unknown GGML_CUDA_ALLREDUCE value: %s\n", env);
            ggml_backend_cuda_comm_init_none(ret);
        }
    }

    return ret;
}

// Top-level dispatch -- calls the function pointer chosen by comm_init.
// Returns false to let the meta-backend's butterfly run.
static bool ggml_backend_cuda_comm_allreduce_tensor(void * comm_ctx_v, struct ggml_tensor ** tensors) {
    if (comm_ctx_v == nullptr) {
        return false;
    }
    auto * comm_ctx = static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
    return comm_ctx->try_allreduce(comm_ctx, tensors);
}

// host buffer type

static const char * ggml_backend_cuda_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_CUDA_NAME "_Host";

    GGML_UNUSED(buft);
}

static bool ggml_backend_buft_is_cuda_host(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
}

static void ggml_backend_cuda_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    CUDA_CHECK(cudaFreeHost(buffer->context));
}

static void * ggml_cuda_host_malloc(size_t size) {
    if (getenv("GGML_CUDA_NO_PINNED") != nullptr) {
        return nullptr;
    }

    void * ptr = nullptr;
    cudaError_t err = cudaMallocHost((void **) &ptr, size);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_DEBUG("%s: failed to allocate %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return nullptr;
    }

    return ptr;
}

static ggml_backend_buffer_t ggml_backend_cuda_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = ggml_cuda_host_malloc(size);

    if (ptr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_cuda_host_buffer_free_buffer;

    return buffer;
}

ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_cuda_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_cuda_host_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_cuda_host_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), 0),
        /* .context  = */ nullptr,
    };

    return &ggml_backend_cuda_buffer_type_host;
}

//static bool ggml_backend_buffer_is_cuda_host(ggml_backend_buffer_t buffer) {
//    return buffer->buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
//}

/// kernels

typedef void (*ggml_cuda_op_mul_mat_t)(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

static __global__ void k_compute_batched_ptrs(
        const void * src0_as_f16, const void * src1_as_f16, char * dst,
        const void ** ptrs_src, void ** ptrs_dst,
        int64_t ne12, int64_t ne13,
        int64_t ne23,
        size_t  nb02, size_t  nb03,
        size_t  nb12, size_t  nb13,
        size_t  nbd2, size_t  nbd3,
        int64_t r2,   int64_t r3) {
    const int64_t i13 = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t i12 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i13 >= ne13 || i12 >= ne12) {
        return;
    }

    const int64_t i03 = i13 / r3;
    const int64_t i02 = i12 / r2;

    ptrs_src[0*ne23 + i12 + i13*ne12] = (const char *) src0_as_f16 + i02*nb02 + i03*nb03;
    ptrs_src[1*ne23 + i12 + i13*ne12] = (const char *) src1_as_f16 + i12*nb12 + i13*nb13;
    ptrs_dst[0*ne23 + i12 + i13*ne12] = (      char *)         dst + i12*nbd2 + i13*nbd3;
}

// Type traits for mapping ggml types to CUDA/cuBLAS types
template<ggml_type T>
struct batched_mul_mat_traits;

template<>
struct batched_mul_mat_traits<GGML_TYPE_F32> {
    using cuda_type = float;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_32F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F32;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_fp32_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_fp32_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_BF16> {
    using cuda_type = nv_bfloat16;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_16BF;
    static inline const ggml_type ggml_type_val = GGML_TYPE_BF16;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_bf16_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_bf16_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_F16> {
    using cuda_type = half;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
    static inline const cudaDataType_t data_type = CUDA_R_16F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F16;
    static inline const half alpha = 1.0;
    static inline const half beta = 0.0;
    static inline const void* get_alpha() { static const half val = alpha; return &val; }
    static inline const void* get_beta() { static const half val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_fp16_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_fp16_nc_cuda(src_type); }
};

static bool ggml_cuda_cublas_prefer_f32_output(ggml_type type, int cc) {
    if (type == GGML_TYPE_F16) {
        return cc == GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_CDNA(cc);
    }
    if (type == GGML_TYPE_BF16) {
        return !GGML_CUDA_CC_IS_RDNA3(cc) && !GGML_CUDA_CC_IS_CDNA(cc);
    }
    return false;
}

template<ggml_type compute_type>
static void ggml_cuda_mul_mat_cublas_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const void * prepared_src1 = nullptr, ggml_tensor * hc_dst = nullptr, const ggml_cuda_scaled_unary_args * unary = nullptr, const ggml_cuda_affine_unary_ops * affine = nullptr) {
    using traits = batched_mul_mat_traits<compute_type>;
    using cuda_t = typename traits::cuda_type;

    GGML_ASSERT(ggml_is_contiguous(dst));

    // Byte offsets and tensor dimensions are currently used in an inconsistent way for dst.
    // As long as dst is contiguous this does not matter though.

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t ne_dst = ggml_nelements(dst);
    cudaStream_t main_stream = ctx.stream();
    cublasHandle_t cublas_h = ctx.cublas_handle();

    const size_t src0_ts = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == src0_ts);
    int64_t s01 = nb01 / src0_ts;
    int64_t s02 = nb02 / src0_ts;
    int64_t s03 = nb03 / src0_ts;

    const size_t src1_ts = ggml_type_size(src1->type);
    GGML_ASSERT(nb10 == src1_ts);
    int64_t s11 = nb11 / src1_ts;
    int64_t s12 = nb12 / src1_ts;
    int64_t s13 = nb13 / src1_ts;

    float * dst_ddf = (float *) dst->data;

    const cuda_t * src0_ptr = nullptr;
    const cuda_t * src1_ptr = nullptr;

    ggml_cuda_pool_alloc<cuda_t> src0_alloc(ctx.pool());
    ggml_cuda_pool_alloc<cuda_t> src1_alloc(ctx.pool());

    bool is_src0_cont_2 = ggml_is_contiguous_2(src0);
    bool is_src1_cont_2 = ggml_is_contiguous_2(src1);

    if (src0->type == compute_type) {
        src0_ptr = (const cuda_t *) src0->data;
    } else {
        src0_alloc.alloc(ggml_nelements(src0));

        if (ggml_is_contiguously_allocated(src0)) {
            const auto convert_func = traits::convert(src0->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src0->data, src0_alloc.get(), ggml_nelements(src0), main_stream);
            const size_t src0_bs = ggml_blck_size(src0->type);
            s01 *= src0_bs;
            s02 *= src0_bs;
            s03 *= src0_bs;
        } else {
            const auto convert_func = traits::convert_nc(src0->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src0->data, src0_alloc.get(), ne00, ne01, ne02, ne03, s01, s02, s03, main_stream);
            s01 = ne00;
            s02 = ne01*s01;
            s03 = ne02*s02;
            is_src0_cont_2 = true;
        }
        src0_ptr = src0_alloc.get();
    }

    if (prepared_src1) {
        GGML_ASSERT(src1->type != compute_type);
        src1_ptr = (const cuda_t *) prepared_src1;
        if (!ggml_is_contiguously_allocated(src1)) {
            s11 = ne10;
            s12 = ne11*s11;
            s13 = ne12*s12;
            is_src1_cont_2 = true;
        }
    } else if (src1->type == compute_type) {
        src1_ptr = (const cuda_t *) src1->data;
    } else {
        src1_alloc.alloc(ggml_nelements(src1));

        if (ggml_is_contiguously_allocated(src1)) {
            const auto convert_func = traits::convert(src1->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src1->data, src1_alloc.get(), ggml_nelements(src1), main_stream);
            const size_t src1_bs = ggml_blck_size(src1->type);
            s11 *= src1_bs;
            s12 *= src1_bs;
            s13 *= src1_bs;
        } else {
            const auto convert_func = traits::convert_nc(src1->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src1->data, src1_alloc.get(), ne10, ne11, ne12, ne13, s11, s12, s13, main_stream);
            s11 = ne10;
            s12 = ne11*s11;
            s13 = ne12*s12;
            is_src1_cont_2 = true;
        }
        src1_ptr = src1_alloc.get();
    }

    ggml_cuda_pool_alloc<cuda_t> dst_temp(ctx.pool());
    char * dst_ptr;
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    cublasComputeType_t cu_compute_type = traits::compute_type;
    cudaDataType_t cu_data_type = traits::data_type;
    cudaDataType_t cu_data_type_a = traits::data_type;
    cudaDataType_t cu_data_type_b = traits::data_type;
    const void * alpha = traits::get_alpha();
    const void * beta = traits::get_beta();

    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const bool prefer_f32_output = ggml_cuda_cublas_prefer_f32_output(compute_type, cc);

    if (prefer_f32_output) {
        dst_ptr = (char *) dst_ddf;
        cu_compute_type = batched_mul_mat_traits<GGML_TYPE_F32>::compute_type;
        cu_data_type = batched_mul_mat_traits<GGML_TYPE_F32>::data_type;
        alpha = batched_mul_mat_traits<GGML_TYPE_F32>::get_alpha();
        beta = batched_mul_mat_traits<GGML_TYPE_F32>::get_beta();
    } else {
        if constexpr (compute_type == GGML_TYPE_F32) {
            dst_ptr = (char *) dst_ddf;  // Direct F32 output
        } else {
            dst_ptr = (char *) dst_temp.alloc(ne_dst);
            nbd2 /= sizeof(float) / sizeof(cuda_t);
            nbd3 /= sizeof(float) / sizeof(cuda_t);
        }
    }

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

    // Theoretically cublasGemmStridedBatchedEx would always work, even for a single matrix.
    // However, for some old NVIDIA and AMD GPUs the strided/Ex GEMM is much slower,
    //     probably because the internal kernel selection logic is suboptimal.
    if (compute_type == GGML_TYPE_F32 && ne12 == 1 && ne13 == 1) {
        CUBLAS_CHECK(
            cublasSgemm(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                    ne01, ne11, ne10,
                    (const float *) alpha, (const float *) src0_ptr, s01,
                                           (const float *) src1_ptr, s11,
                    (const float *) beta,  (float       *)  dst_ptr, ne0));
    } else if (ne12 == 1 && ne13 == 1) {
        CUBLAS_CHECK(
            cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                    ne01, ne11, ne10,
                    alpha, src0_ptr, cu_data_type_a, s01,
                           src1_ptr, cu_data_type_b, s11,
                    beta,   dst_ptr, cu_data_type,   ne0,
                    cu_compute_type,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else if (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
        // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be determined from dim 3:
        const int64_t sma = ne02 == 1 ? s03 : s02;
        const int64_t smb = ne12 == 1 ? s13 : s12;

        // there is no broadcast and src0, src1 are contiguous across dims 2, 3
        // use cublasGemmStridedBatchedEx
        CUBLAS_CHECK(
        cublasGemmStridedBatchedEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, src0_ptr, cu_data_type_a, s01, sma,     // strideA
                       src1_ptr, cu_data_type_b, s11, smb,     // strideB
                beta,   dst_ptr, cu_data_type,   ne0, ne1*ne0, // strideC
                ne12*ne13,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
        // use cublasGemmBatchedEx
        const int64_t ne23 = ne12*ne13;

        ggml_cuda_pool_alloc<const void *> ptrs_src(ctx.pool(), 2*ne23);
        ggml_cuda_pool_alloc<      void *> ptrs_dst(ctx.pool(), 1*ne23);

        const size_t src_type_size = sizeof(cuda_t);

        const int threads_x = 16;
        const int threads_y = 16;
        const dim3 block_dims(threads_x, threads_y);

        const dim3 grid_dims(
            (ne13 + threads_x - 1) / threads_x,
            (ne12 + threads_y - 1) / threads_y
        );
        k_compute_batched_ptrs<<<grid_dims, block_dims, 0, main_stream>>>(
                src0_ptr, src1_ptr, dst_ptr,
                ptrs_src.get(), ptrs_dst.get(),
                ne12, ne13,
                ne23,
                s02*src_type_size, s03*src_type_size,
                s12*src_type_size, s13*src_type_size,
                nbd2, nbd3,
                r2, r3);

        CUDA_CHECK(cudaGetLastError());

        CUBLAS_CHECK(
        cublasGemmBatchedEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, (const void **) (ptrs_src.get() + 0*ne23), cu_data_type_a, s01,
                       (const void **) (ptrs_src.get() + 1*ne23), cu_data_type_b, s11,
                beta,  (      void **) (ptrs_dst.get() + 0*ne23), cu_data_type,   ne0,
                ne23,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // Convert output back to F32 if needed
    if (cu_data_type != CUDA_R_32F) {
        if (hc_dst) {
            ggml_cuda_op_dsv4_hc_pre_convert(ctx, traits::ggml_type_val, dst_temp.get(), dst, hc_dst);
        } else if (unary) {
            ggml_cuda_op_scaled_unary_convert(ctx, traits::ggml_type_val, dst_temp.get(), dst, *unary);
        } else if (affine) {
            ggml_cuda_op_affine_unary_convert(ctx, traits::ggml_type_val, dst_temp.get(), dst, *affine);
        } else {
            const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(traits::ggml_type_val);
            to_fp32_cuda(dst_temp.get(), dst_ddf, ne_dst, main_stream);
        }
    }
}

static ggml_type ggml_cuda_mul_mat_cublas_compute_type(int cc, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, bool warn = true) {
    const ggml_prec prec = (ggml_prec) ggml_get_op_params_i32(dst, 0);
    ggml_type compute_type = src0->type;
    if (ggml_is_quantized(compute_type)) {
        compute_type = fast_fp16_hardware_available(cc) ? GGML_TYPE_F16 : GGML_TYPE_F32;
    } else if (compute_type == GGML_TYPE_F16 && !fast_fp16_hardware_available(cc)) {
        compute_type = GGML_TYPE_F32;
    } else if (compute_type == GGML_TYPE_BF16 && !fast_bf16_hardware_available(cc)) {
        if (GGML_CUDA_CC_IS_AMD(cc) && src1->ne[1] > 32) {
            compute_type = GGML_TYPE_F32;
        }
        if (GGML_CUDA_CC_IS_NVIDIA(cc) && src1->ne[1] > (cc >= GGML_CUDA_CC_VOLTA ? 8 : 128)) {
            compute_type = GGML_TYPE_F32;
        }
    }
    // F16 is the only compute type that can not satisfy a request for BF16
    if (prec == GGML_PREC_BF16 && compute_type == GGML_TYPE_F16) {
        compute_type = fast_bf16_hardware_available(cc) ? GGML_TYPE_BF16 : GGML_TYPE_F32;
    } else if (prec == GGML_PREC_F32) {
        compute_type = GGML_TYPE_F32;
    }

    const char * env_c = getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE");
    if (env_c != nullptr) {
        std::string env_cpp = env_c;
        for (char & c : env_cpp) {
            c = std::tolower(c);
        }
        if (env_cpp == "f32" || env_cpp == "fp32") {
            compute_type = GGML_TYPE_F32;
        } else if (env_cpp == "f16" || env_cpp == "fp16") {
            compute_type = GGML_TYPE_F16;
        } else if (env_cpp == "bf16") {
            compute_type = GGML_TYPE_BF16;
        } else if (warn && env_cpp != "auto") {
            GGML_LOG_WARN("%s: unknown value for GGML_CUDA_CUBLAS_COMPUTE_TYPE: %s", "ggml_cuda_mul_mat_cublas", env_cpp.c_str());
        }
    }

    return compute_type;
}

static void ggml_cuda_mul_mat_cublas(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const void * prepared_src1 = nullptr, ggml_tensor * hc_dst = nullptr, const ggml_cuda_scaled_unary_args * unary = nullptr, const ggml_cuda_affine_unary_ops * affine = nullptr) {
    const ggml_type compute_type = ggml_cuda_mul_mat_cublas_compute_type(ggml_cuda_info().devices[ctx.device].cc, src0, src1, dst, true);
    switch (compute_type) {
        case GGML_TYPE_F32:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F32>(ctx, src0, src1, dst, prepared_src1, hc_dst, unary, affine);
            break;
        case GGML_TYPE_BF16:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_BF16>(ctx, src0, src1, dst, prepared_src1, hc_dst, unary, affine);
            break;
        case GGML_TYPE_F16:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F16>(ctx, src0, src1, dst, prepared_src1, hc_dst, unary, affine);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

static bool ggml_cuda_should_fuse_mul_mat(const ggml_tensor * ffn_up,
                                          const ggml_tensor * ffn_gate,
                                          const ggml_tensor * glu,
                                          const ggml_tensor * ffn_up_bias = nullptr,
                                          const ggml_tensor * ffn_gate_bias = nullptr,
                                          const ggml_tensor * ffn_up_scale = nullptr,
                                          const ggml_tensor * ffn_gate_scale = nullptr) {
    const bool has_bias = ffn_up_bias != nullptr || ffn_gate_bias != nullptr;
    const bool has_scale = ffn_up_scale != nullptr || ffn_gate_scale != nullptr;

    if (has_bias && (!ffn_up_bias || !ffn_gate_bias)) {
        return false;
    }
    if (has_scale && (!ffn_up_scale || !ffn_gate_scale)) {
        return false;
    }

    const bool is_mul_mat     = ffn_up->op == GGML_OP_MUL_MAT     && ffn_gate->op == GGML_OP_MUL_MAT     && glu->op == GGML_OP_GLU;
    const bool is_mul_mat_id  = ffn_up->op == GGML_OP_MUL_MAT_ID  && ffn_gate->op == GGML_OP_MUL_MAT_ID  && glu->op == GGML_OP_GLU;

    GGML_ASSERT(ffn_up && ffn_gate && glu);

    if (!is_mul_mat && !is_mul_mat_id) {
        return false;
    }

    const ggml_op expected_bias_op = is_mul_mat ? GGML_OP_ADD : GGML_OP_ADD_ID;
    const ggml_tensor * ffn_up_bias_src   = has_scale ? ffn_up_scale   : ffn_up;
    const ggml_tensor * ffn_gate_bias_src = has_scale ? ffn_gate_scale : ffn_gate;
    const ggml_tensor * ffn_up_out        = has_bias ? ffn_up_bias     : ffn_up_bias_src;
    const ggml_tensor * ffn_gate_out      = has_bias ? ffn_gate_bias   : ffn_gate_bias_src;

    if (glu->src[0] != ffn_gate_out || glu->src[1] != ffn_up_out) {
        return false;
    }

    if (has_scale) {
        if (ffn_up_scale->op != GGML_OP_MUL || ffn_gate_scale->op != GGML_OP_MUL) {
            return false;
        }
        const bool up_has_mm   = ffn_up_scale->src[0] == ffn_up || ffn_up_scale->src[1] == ffn_up;
        const bool gate_has_mm = ffn_gate_scale->src[0] == ffn_gate || ffn_gate_scale->src[1] == ffn_gate;
        if (!up_has_mm || !gate_has_mm) {
            return false;
        }
    }

    if (has_bias) {
        if (ffn_up_bias->op != expected_bias_op || ffn_gate_bias->op != expected_bias_op) {
            return false;
        }

        if (expected_bias_op == GGML_OP_ADD) {
            const bool up_has_mul   = ffn_up_bias->src[0] == ffn_up_bias_src || ffn_up_bias->src[1] == ffn_up_bias_src;
            const bool gate_has_mul = ffn_gate_bias->src[0] == ffn_gate_bias_src || ffn_gate_bias->src[1] == ffn_gate_bias_src;
            if (!up_has_mul || !gate_has_mul) {
                return false;
            }
        } else { // GGML_OP_ADD_ID
            if (ffn_up_bias->src[0] != ffn_up_bias_src || ffn_gate_bias->src[0] != ffn_gate_bias_src) {
                return false;
            }
            if (ffn_up_bias->src[2] != ffn_up->src[2] || ffn_gate_bias->src[2] != ffn_gate->src[2]) {
                return false;
            }
        }
    }

    if (ffn_up->src[0]->type != ffn_gate->src[0]->type || !ggml_are_same_shape(ffn_up->src[0], ffn_gate->src[0]) ||
        !ggml_are_same_stride(ffn_up->src[0], ffn_gate->src[0])) {
        return false;
    }

    if (ffn_up->src[1] != ffn_gate->src[1]) {
        return false;
    }

    if (is_mul_mat_id && ffn_up->src[2] != ffn_gate->src[2]) {
        return false;
    }

    static constexpr std::array<ggml_glu_op, 4> valid_glu_ops = { GGML_GLU_OP_SWIGLU, GGML_GLU_OP_GEGLU, GGML_GLU_OP_SWIGLU_OAI, GGML_GLU_OP_SWIGLU_CLAMP };

    if (std::find(valid_glu_ops.begin(), valid_glu_ops.end(), ggml_get_glu_op(glu)) == valid_glu_ops.end()) {
        return false;
    }

    if (const bool swapped = ggml_get_op_params_i32(glu, 1); swapped) {
        return false;
    }

    return true;
}

static bool ggml_cuda_should_fuse_mul_mat_vec_f(const ggml_tensor * tensor) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool is_mul_mat_id = tensor->op == GGML_OP_MUL_MAT_ID;

    bool use_mul_mat_vec_f =
        (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16) &&
        src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

    const int cc      = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    use_mul_mat_vec_f = use_mul_mat_vec_f && ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, is_mul_mat_id ? src1->ne[2] : src1->ne[1]);

    //we only support fusion for ncols_dst = 1
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] != 1) {
        return false;
    }


    return use_mul_mat_vec_f;
}

static bool ggml_cuda_should_fuse_mul_mat_vec_q(const ggml_tensor * tensor) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
                                   ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) &&
                                   src0->view_src;

    bool use_mul_mat_vec_q = ggml_is_quantized(src0->type) && !bad_padding_clear && src1->type == GGML_TYPE_F32 &&
                             dst->type == GGML_TYPE_F32 && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;

    // fusion is not universally faster on Pascal
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (cc <= GGML_CUDA_CC_PASCAL) {
        return false;
    }
    //we only support fusion for ncols_dst = 1
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] > get_mmvq_mmid_max_batch(src0->type, cc)) {
        return false;
    }

    return use_mul_mat_vec_q;
}

enum ggml_cuda_mm_kernel {
    GGML_CUDA_MM_MMVF,
    GGML_CUDA_MM_MMVF_TRANSPOSE,
    GGML_CUDA_MM_MMF,
    GGML_CUDA_MM_MMVQ,
    GGML_CUDA_MM_MMQ,
    GGML_CUDA_MM_CUBLAS,
};

static ggml_cuda_mm_kernel ggml_cuda_mul_mat_kernel(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int device) {
    GGML_TENSOR_BINARY_OP_LOCALS
    // If src0 is a temporary compute buffer it may have some padding that needs to be cleared for mul_mat_vec_q or mul_mat_q.
    // But if src0 is also a view of another tensor then this cannot be done safely because it may overwrite valid tensor data.
    // Therefore, in such cases use cuBLAS.
    const bool bad_padding_clear = src0->buffer && ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE
        && ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) && src0->view_src;
    if (bad_padding_clear || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return GGML_CUDA_MM_CUBLAS;
    }

    const int cc        = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;

    if (ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, ne11)) {
        // The custom F16 vector kernel can be used over batched cuBLAS GEMM.
        // But this is only faster for GPUs without tensor cores or with a thin src0 matrix (particularly KQV in attention)
        return GGML_CUDA_MM_MMVF;
    }
    // A transposed vector can still use MMVQ (i.e. ne01 == 1)
    if (ne01 == 1 && ne11 > MMVF_MAX_BATCH_SIZE && ne2 == 1 && ne3 == 1
            && src0->type == GGML_TYPE_F32
            && ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst)
            && ggml_cuda_should_use_mmvf(src1->type, cc, src1->ne, src1->nb, /*ne11 =*/ 1)) {
        return GGML_CUDA_MM_MMVF_TRANSPOSE;
    }
    if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11, /*mul_mat_id =*/ false)) {
        return GGML_CUDA_MM_MMF;
    }
    if (ggml_cuda_should_use_mmvq(src0->type, cc, ne11)) {
        return GGML_CUDA_MM_MMVQ;
    }
    if (ggml_cuda_should_use_mmq(src0->type, cc, ne11, /*n_experts =*/ 0)) {
        return GGML_CUDA_MM_MMQ;
    }
    return GGML_CUDA_MM_CUBLAS;
}

static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const void * prepared_src1 = nullptr) {
    if (ggml_cuda_op_mul_mat_use_fwht(dst) && ggml_cuda_op_fwht(ctx, src1, dst)) {
        return;
    }
    switch (ggml_cuda_mul_mat_kernel(src0, src1, dst, ctx.device)) {
        case GGML_CUDA_MM_MMVF:
            ggml_cuda_mul_mat_vec_f(ctx, src0, src1, nullptr, dst);
            break;
        case GGML_CUDA_MM_MMVF_TRANSPOSE: {
            ggml_tensor dst_vec = *dst;
            dst_vec.ne[0] = src1->ne[1];
            dst_vec.ne[1] = 1;
            dst_vec.nb[1] = dst_vec.nb[0]*src1->ne[1];
            dst_vec.nb[2] = dst_vec.nb[1];
            dst_vec.nb[3] = dst_vec.nb[1];
            ggml_cuda_mul_mat_vec_f(ctx, src1, src0, nullptr, &dst_vec);
            break;
        }
        case GGML_CUDA_MM_MMF:
            ggml_cuda_mul_mat_f(ctx, src0, src1, nullptr, dst);
            break;
        case GGML_CUDA_MM_MMVQ:
            ggml_cuda_mul_mat_vec_q(ctx, src0, src1, nullptr, dst);
            break;
        case GGML_CUDA_MM_MMQ:
            ggml_cuda_mul_mat_q(ctx, src0, src1, nullptr, dst);
            break;
        case GGML_CUDA_MM_CUBLAS:
            ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst, prepared_src1);
            break;
    }
}

// returns true when ggml_cuda_mul_mat_id takes the fallback path that requires stream synchronization
// [TAG_MUL_MAT_ID_CUDA_GRAPHS]
static bool ggml_cuda_mul_mat_id_needs_sync(const ggml_tensor * dst, const int cc) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return true;
    }

    if (dst->ne[2] <= MMVQ_MAX_BATCH_SIZE) {
        if (ggml_is_quantized(src0->type)) {
            if (dst->ne[2] <= get_mmvq_mmid_max_batch(src0->type, cc)) {
                return false;
            }
        } else if (GGML_CUDA_CC_IS_AMD(cc)) {
            return false;
        }
    }

    if (ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[2], /*n_experts=*/src0->ne[2])) {
        return false;
    }

    if (ggml_cuda_should_use_mmf(src0->type, cc, WARP_SIZE, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
        return false;
    }

    return true;
}

static void ggml_cuda_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
    if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        static_assert(MMVQ_MAX_BATCH_SIZE == MMVF_MAX_BATCH_SIZE);
        if (ne2 <= MMVQ_MAX_BATCH_SIZE) {
            if (ggml_is_quantized(src0->type)) {
                const int mmvq_mmid_max = get_mmvq_mmid_max_batch(src0->type, cc);
                if (ne2 <= mmvq_mmid_max) {
                    ggml_cuda_mul_mat_vec_q(ctx, src0, src1, ids, dst);
                    return;
                }
            } else {
                if (GGML_CUDA_CC_IS_AMD(cc)) {
                    ggml_cuda_mul_mat_vec_f(ctx, src0, src1, ids, dst);
                    return;
                }
            }
        }

        if (ggml_cuda_should_use_mmq(src0->type, cc, ne12, /*n_experts=*/ne02)) {
            ggml_cuda_mul_mat_q(ctx, src0, src1, ids, dst);
            return;
        }

        if (ggml_cuda_should_use_mmf(src0->type, cc, WARP_SIZE, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
            ggml_cuda_mul_mat_f(ctx, src0, src1, ids, dst);
            return;
        }
    }

    // note: this path should not be reached when recording CUDA graphs, because it requires stream synchronization
    GGML_ASSERT(ggml_cuda_mul_mat_id_needs_sync(dst, cc));
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const ggml_type type_src1_sorted = (src0->type == GGML_TYPE_F16 && !fast_fp16_hardware_available(cc))
        || ggml_is_quantized(src0->type) ? GGML_TYPE_F32 : src0->type;
    const ggml_type type_dst_sorted  = GGML_TYPE_F32;
    const size_t ts_src1_sorted = ggml_type_size(type_src1_sorted);
    const size_t ts_dst_sorted  = ggml_type_size(type_dst_sorted);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;

    std::vector<int32_t> ids_to_sorted_host;
    ids_to_sorted_host.reserve(2*ne_get_rows);
    std::vector<int32_t> ids_from_sorted_host(ne_get_rows);

    ggml_cuda_pool_alloc<int32_t> ids_buf_dev(ctx.pool(), 2*ne_get_rows);

    std::vector<int32_t> tokens_per_expert(ne02);

    ggml_cuda_pool_alloc<char> src1_sorted(ctx.pool(), ne12*n_expert_used*ne10*ts_src1_sorted);
    ggml_cuda_pool_alloc<char>  dst_sorted(ctx.pool(), ne2 *n_expert_used* ne0*ts_dst_sorted);

    std::vector<char> ids_host(ggml_nbytes(ids));
    CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids->data, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int64_t i02 = 0; i02 < ne02; ++i02) { // expert matrices
        for (int64_t i12 = 0; i12 < ne12; ++i12) { // tokens
            for (int64_t iex = 0; iex < n_expert_used; ++iex) {
                const int32_t expert_to_use = *(const int32_t *)(ids_host.data() + i12*ids->nb[1] + iex*ids->nb[0]);
                assert(expert_to_use >= 0 && expert_to_use < ne02);
                if (expert_to_use == i02) {
                    ids_from_sorted_host[i12*n_expert_used + iex] = ids_to_sorted_host.size();
                    ids_to_sorted_host.push_back(i12*ne11 + iex % ne11);
                    tokens_per_expert[i02]++;
                    break;
                }
            }
        }
    }
    GGML_ASSERT(ids_to_sorted_host.size() == size_t(ne_get_rows));

    ids_to_sorted_host.insert(ids_to_sorted_host.end(), ids_from_sorted_host.begin(), ids_from_sorted_host.end());

    CUDA_CHECK(cudaMemcpyAsync(ids_buf_dev.ptr, ids_to_sorted_host.data(), 2*ne_get_rows*sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const int32_t * ids_to_sorted   = ids_buf_dev.ptr + 0*ne_get_rows;
    const int32_t * ids_from_sorted = ids_buf_dev.ptr + 1*ne_get_rows;

    get_rows_cuda(src1->data, src1->type, ids_to_sorted, src1_sorted.ptr, type_src1_sorted,
        ne10, nb11, nb12, nb13,
        ne_get_rows, 1, 1, sizeof(int32_t), ne_get_rows*sizeof(int32_t), ne_get_rows*sizeof(int32_t),
        ne10*ts_src1_sorted, ne_get_rows*ne10*ts_src1_sorted, ne_get_rows*ne10*ts_src1_sorted, stream);
    CUDA_CHECK(cudaGetLastError());

    char * src1_data_cur = (char *) src1_sorted.ptr;
    char *  dst_data_cur = (char *)  dst_sorted.ptr;
    for (int64_t i02 = 0; i02 < ne02; ++i02) {
        if (tokens_per_expert[i02] == 0) {
            continue;
        }

        ggml_tensor src0_slice = *src0;
        src0_slice.ne[2]    = 1;
        src0_slice.nb[3]    = src0_slice.nb[2];
        src0_slice.op       = GGML_OP_VIEW;
        src0_slice.view_src = dst->src[0]; // non-const pointer to src0
        src0_slice.data     = (char *) src0->data + i02*nb02;

        ggml_tensor src1_slice;
        memset(&src1_slice, 0, sizeof(src1_slice));
        src1_slice.buffer = src1->buffer;
        src1_slice.type   = type_src1_sorted;
        src1_slice.ne[0]  = ne10;
        src1_slice.ne[1]  = tokens_per_expert[i02];
        src1_slice.ne[2]  = 1;
        src1_slice.ne[3]  = 1;
        src1_slice.nb[0]  = ts_src1_sorted;
        src1_slice.nb[1]  = src1_slice.ne[0] * src1_slice.nb[0];
        src1_slice.nb[2]  = src1_slice.ne[1] * src1_slice.nb[1];
        src1_slice.nb[3]  = src1_slice.ne[2] * src1_slice.nb[2];
        src1_slice.data   = src1_data_cur;

        ggml_tensor dst_slice;
        memset(&dst_slice, 0, sizeof(dst_slice));
        memcpy(dst_slice.op_params, dst->op_params, sizeof(dst_slice.op_params));
        dst_slice.buffer = dst->buffer;
        dst_slice.type   = type_dst_sorted;
        dst_slice.ne[0]  = ne0;
        dst_slice.ne[1]  = tokens_per_expert[i02];
        dst_slice.ne[2]  = 1;
        dst_slice.ne[3]  = 1;
        dst_slice.nb[0]  = ts_dst_sorted;
        dst_slice.nb[1]  = dst_slice.ne[0] * dst_slice.nb[0];
        dst_slice.nb[2]  = dst_slice.ne[1] * dst_slice.nb[1];
        dst_slice.nb[3]  = dst_slice.ne[2] * dst_slice.nb[2];
        dst_slice.data   = dst_data_cur;

        ggml_cuda_mul_mat(ctx, &src0_slice, &src1_slice, &dst_slice);
        CUDA_CHECK(cudaGetLastError());

        src1_data_cur += src1_slice.nb[2];
        dst_data_cur  +=  dst_slice.nb[2];
    }

    get_rows_cuda(dst_sorted.ptr, type_dst_sorted, ids_from_sorted, dst->data, dst->type,
        ne0, ne0*ts_dst_sorted, ne_get_rows*ne0*ts_dst_sorted, ne_get_rows*ne0*ts_dst_sorted,
        ne_get_rows, 1, 1, sizeof(int32_t), ne_get_rows*sizeof(int32_t), ne_get_rows*sizeof(int32_t),
        nb1, nb2, nb3, stream);
}

static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst, const void * prepared_src1 = nullptr) {
    switch (dst->op) {
        case GGML_OP_ARGMAX:
            ggml_cuda_argmax(ctx, dst);
            break;
        case GGML_OP_COUNT_EQUAL:
            ggml_cuda_count_equal(ctx, dst);
            break;
        case GGML_OP_REPEAT:
            ggml_cuda_op_repeat(ctx, dst);
            break;
        case GGML_OP_REPEAT_BACK:
            ggml_cuda_op_repeat_back(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
            ggml_cuda_op_get_rows(ctx, dst);
            break;
        case GGML_OP_GET_ROWS_BACK:
            ggml_cuda_op_get_rows_back(ctx, dst);
            break;
        case GGML_OP_SET_ROWS:
            ggml_cuda_op_set_rows(ctx, dst);
            break;
        case GGML_OP_SET:
            ggml_cuda_op_set(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_CPY:
            ggml_cuda_cpy(ctx, dst->src[0], dst->src[1]);
            break;
        case GGML_OP_CONT:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_ADD:
        case GGML_OP_ADD1: // TODO: more efficient implementation
            ggml_cuda_op_add(ctx, dst);
            break;
        case GGML_OP_ADD_ID:
            ggml_cuda_op_add_id(ctx, dst);
            break;
        case GGML_OP_SUB:
            ggml_cuda_op_sub(ctx, dst);
            break;
        case GGML_OP_ACC:
            ggml_cuda_op_acc(ctx, dst);
            break;
        case GGML_OP_MUL:
            ggml_cuda_op_mul(ctx, dst);
            break;
        case GGML_OP_DIV:
            ggml_cuda_op_div(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_ABS:
                    ggml_cuda_op_abs(ctx, dst);
                    break;
                case GGML_UNARY_OP_SGN:
                    ggml_cuda_op_sgn(ctx, dst);
                    break;
                case GGML_UNARY_OP_NEG:
                    ggml_cuda_op_neg(ctx, dst);
                    break;
                case GGML_UNARY_OP_STEP:
                    ggml_cuda_op_step(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU:
                    ggml_cuda_op_gelu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SILU:
                    ggml_cuda_op_silu(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_ERF:
                    ggml_cuda_op_gelu_erf(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    ggml_cuda_op_gelu_quick(ctx, dst);
                    break;
                case GGML_UNARY_OP_TANH:
                    ggml_cuda_op_tanh(ctx, dst);
                    break;
                case GGML_UNARY_OP_RELU:
                    ggml_cuda_op_relu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SIGMOID:
                    ggml_cuda_op_sigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    ggml_cuda_op_hardsigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    ggml_cuda_op_hardswish(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXP:
                    ggml_cuda_op_exp(ctx, dst);
                    break;
                case GGML_UNARY_OP_ELU:
                    ggml_cuda_op_elu(ctx, dst);
                    break;
                case GGML_UNARY_OP_XIELU:
                    ggml_cuda_op_xielu(ctx, dst);
                    break;
                case GGML_UNARY_OP_FLOOR:
                    ggml_cuda_op_floor(ctx, dst);
                    break;
                case GGML_UNARY_OP_CEIL:
                    ggml_cuda_op_ceil(ctx, dst);
                    break;
                case GGML_UNARY_OP_ROUND:
                    ggml_cuda_op_round(ctx, dst);
                    break;
                case GGML_UNARY_OP_TRUNC:
                    ggml_cuda_op_trunc(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXPM1:
                    ggml_cuda_op_expm1(ctx, dst);
                    break;
                case GGML_UNARY_OP_SOFTPLUS:
                    ggml_cuda_op_softplus(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(dst)) {
                case GGML_GLU_OP_REGLU:
                    ggml_cuda_op_reglu(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU:
                    ggml_cuda_op_geglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU:
                    ggml_cuda_op_swiglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    ggml_cuda_op_swiglu_oai(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_ERF:
                    ggml_cuda_op_geglu_erf(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_QUICK:
                    ggml_cuda_op_geglu_quick(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    ggml_cuda_op_swiglu_clamp(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            ggml_cuda_op_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_cuda_op_group_norm(ctx, dst);
            break;
        case GGML_OP_L2_NORM:
            ggml_cuda_op_l2_norm(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            ggml_cuda_op_concat(ctx, dst);
            break;
        case GGML_OP_UPSCALE:
            ggml_cuda_op_upscale(ctx, dst);
            break;
        case GGML_OP_PAD:
            ggml_cuda_op_pad(ctx, dst);
            break;
        case GGML_OP_PAD_REFLECT_1D:
            ggml_cuda_op_pad_reflect_1d(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            ggml_cuda_op_arange(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_cuda_op_timestep_embedding(ctx, dst);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_cuda_op_leaky_relu(ctx, dst);
            break;
        case GGML_OP_SILU_BACK:
            ggml_cuda_op_silu_back(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_cuda_op_rms_norm(ctx, dst);
            break;
        case GGML_OP_RMS_NORM_BACK:
            ggml_cuda_op_rms_norm_back(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            ggml_cuda_mul_mat(ctx, dst->src[0], dst->src[1], dst, prepared_src1);
            break;
        case GGML_OP_MUL_MAT_ID:
            ggml_cuda_mul_mat_id(ctx, dst);
            break;
        case GGML_OP_OUT_PROD:
            ggml_cuda_out_prod(ctx, dst);
            break;
        case GGML_OP_SCALE:
            ggml_cuda_op_scale(ctx, dst);
            break;
        case GGML_OP_SQR:
            ggml_cuda_op_sqr(ctx, dst);
            break;
        case GGML_OP_SQRT:
            ggml_cuda_op_sqrt(ctx, dst);
            break;
        case GGML_OP_SIN:
            ggml_cuda_op_sin(ctx, dst);
            break;
        case GGML_OP_COS:
            ggml_cuda_op_cos(ctx, dst);
            break;
        case GGML_OP_CLAMP:
            ggml_cuda_op_clamp(ctx, dst);
            break;
        case GGML_OP_LOG:
            ggml_cuda_op_log(ctx, dst);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
                break;
        case GGML_OP_DIAG:
            ggml_cuda_op_diag(ctx, dst);
            break;
        case GGML_OP_DIAG_MASK_INF:
            ggml_cuda_op_diag_mask_inf(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX:
            ggml_cuda_op_soft_max(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX_BACK:
            ggml_cuda_op_soft_max_back(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggml_cuda_op_rope(ctx, dst);
            break;
        case GGML_OP_ROPE_BACK:
            ggml_cuda_op_rope_back(ctx, dst);
            break;
        case GGML_OP_ROLL:
            ggml_cuda_op_roll(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggml_cuda_op_im2col(ctx, dst);
            break;
        case GGML_OP_IM2COL_3D:
            ggml_cuda_op_im2col_3d(ctx, dst);
            break;
        case GGML_OP_CONV_2D:
            ggml_cuda_op_conv2d(ctx, dst);
            break;
        case GGML_OP_CONV_3D:
            ggml_cuda_op_conv3d(ctx, dst);
            break;
        case GGML_OP_CONV_2D_DW:
            ggml_cuda_op_conv2d_dw(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            ggml_cuda_conv_2d_transpose_p0(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            ggml_cuda_op_conv_transpose_1d(ctx,dst);
            break;
        case GGML_OP_COL2IM_1D:
            ggml_cuda_op_col2im_1d(ctx, dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_cuda_op_pool2d(ctx, dst);
            break;
        case GGML_OP_POOL_1D:
            ggml_cuda_op_pool1d(ctx, dst);
            break;
        case GGML_OP_SUM:
            ggml_cuda_op_sum(ctx, dst);
            break;
        case GGML_OP_CUMSUM:
            ggml_cuda_op_cumsum(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            ggml_cuda_op_sum_rows(ctx, dst);
            break;
        case GGML_OP_MEAN:
            ggml_cuda_op_mean(ctx, dst);
            break;
        case GGML_OP_SSM_CONV:
            ggml_cuda_op_ssm_conv(ctx, dst);
            break;
        case GGML_OP_SSM_SCAN:
            ggml_cuda_op_ssm_scan(ctx, dst);
            break;
        case GGML_OP_TOP_K:
            ggml_cuda_op_top_k(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            ggml_cuda_op_argsort(ctx, dst);
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_cuda_flash_attn_ext(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_cuda_cross_entropy_loss(ctx, dst);
            break;
        case GGML_OP_TRI:
            ggml_cuda_op_tri(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV6:
            ggml_cuda_op_rwkv_wkv6(ctx, dst);
            break;
        case GGML_OP_GATED_LINEAR_ATTN:
            ggml_cuda_op_gated_linear_attn(ctx, dst);
            break;
        case GGML_OP_GATED_DELTA_NET:
            ggml_cuda_op_gated_delta_net(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_COMB:
            ggml_cuda_op_dsv4_hc_comb(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_PRE:
            ggml_cuda_op_dsv4_hc_pre(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_POST:
            ggml_cuda_op_dsv4_hc_post(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV7:
            ggml_cuda_op_rwkv_wkv7(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            ggml_cuda_cross_entropy_loss_back(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_ADAMW:
            ggml_cuda_opt_step_adamw(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_SGD:
            ggml_cuda_opt_step_sgd(ctx, dst);
            break;
        case GGML_OP_SOLVE_TRI:
            ggml_cuda_op_solve_tri(ctx, dst);
            break;
        case GGML_OP_FILL:
            ggml_cuda_op_fill(ctx, dst);
            break;
        case GGML_OP_LIGHTNING_INDEXER:
            ggml_cuda_lightning_indexer(ctx, dst);
            break;
        default:
            return false;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: %s failed\n", __func__, ggml_op_desc(dst));
        CUDA_CHECK(err);
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////

// backend

static const char * ggml_backend_cuda_get_name(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    return cuda_ctx->name.c_str();
}

static void ggml_backend_cuda_free(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    delete cuda_ctx;
    delete backend;
}

static void ggml_backend_cuda_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cuda_ctx->stream()));
}

static void ggml_backend_cuda_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

static void ggml_backend_cuda_set_tensor_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpy2DAsync(
        (char *) tensor->data + offset, stride_tensor, data, stride_data, size, n_copies, cudaMemcpyHostToDevice, cuda_ctx->stream()));
}

static void ggml_backend_cuda_get_tensor_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpy2DAsync(
        data, stride_data, (const char *) tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

static bool ggml_backend_cuda_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    if (!ggml_backend_is_cuda(backend_src) || !ggml_backend_is_cuda(backend_dst)) {
        return false;
    }

    if (!ggml_backend_buffer_is_cuda(buf_src) || !ggml_backend_buffer_is_cuda(buf_dst)) {
        return false;
    }

    // device -> device copy
    ggml_backend_cuda_context * cuda_ctx_src = (ggml_backend_cuda_context *) backend_src->context;
    ggml_backend_cuda_context * cuda_ctx_dst = (ggml_backend_cuda_context *) backend_dst->context;

    ggml_backend_cuda_buffer_context * buf_ctx_src = (ggml_backend_cuda_buffer_context *) buf_src->context;
    ggml_backend_cuda_buffer_context * buf_ctx_dst = (ggml_backend_cuda_buffer_context *) buf_dst->context;

    if (cuda_ctx_src->device != buf_ctx_src->device || cuda_ctx_dst->device != buf_ctx_dst->device) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: backend and buffer devices do not match\n", __func__);
#endif // NDEBUG
        return false;
    }

    if (backend_src != backend_dst) {
        // copy on src stream
        // compare the backing physical devices: distinct virtual devices may share one physical GPU,
        // in which case a same-device copy (not a peer copy) is required
        const int src_physical = ggml_cuda_get_physical_device(cuda_ctx_src->device);
        const int dst_physical = ggml_cuda_get_physical_device(cuda_ctx_dst->device);
        if (src_physical == dst_physical) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, ggml_nbytes(dst), cuda_ctx_src->stream()));
#endif // GGML_CUDA_NO_PEER_COPY
        }

        // record event on src stream after the copy
        if (!cuda_ctx_src->copy_event) {
            ggml_cuda_set_device(cuda_ctx_src->device);
            CUDA_CHECK(cudaEventCreateWithFlags(&cuda_ctx_src->copy_event, cudaEventDisableTiming));
        }

        CUDA_CHECK(cudaEventRecord(cuda_ctx_src->copy_event, cuda_ctx_src->stream()));

        // wait on dst stream for the copy to complete
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx_dst->stream(), cuda_ctx_src->copy_event, 0));
    } else {
        // src and dst are on the same backend
        CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
    }
    return true;
}

static void ggml_backend_cuda_synchronize(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));

    GGML_UNUSED(backend);
}

static bool ggml_cuda_is_view_or_noop(const ggml_tensor * t) {
    return ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
}

#ifdef USE_CUDA_GRAPH
static bool ggml_cuda_graph_check_compability(ggml_cgraph * cgraph) {

    bool use_cuda_graph = true;
    // Loop over nodes in GGML graph to obtain info needed for CUDA graph

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (ggml_cuda_is_view_or_noop(node)) {
            continue;
        }

        // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
        if (node->op == GGML_OP_MUL_MAT_ID) {
            const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
            if (ggml_cuda_mul_mat_id_needs_sync(node, cc)) {
                // the mul_mat_id fallback path synchronizes the stream, so we cannot use CUDA graphs
                // ref: https://github.com/ggml-org/llama.cpp/pull/18958
                use_cuda_graph = false;
#ifndef NDEBUG
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to unsupported node type\n", __func__);
#endif
            }
        }

        if (!use_cuda_graph) {
            break;
        }
    }

    return use_cuda_graph;
}

static const void * ggml_cuda_graph_get_key(ggml_cgraph * cgraph) {
    return cgraph->nodes[0];
}

static bool ggml_cuda_graph_update_required(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph) {
    bool res = false;

    const void * graph_key = ggml_cuda_graph_get_key(cgraph);
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

    if (cgraph->uid != 0 &&
        cgraph->uid == graph->uid) {
        GGML_LOG_DEBUG("CUDA Graph id %zu reused\n", cgraph->uid);
        GGML_ASSERT((int)graph->node_props.size() == cgraph->n_nodes);
        return false;
    }

    graph->uid = cgraph->uid;

    // Check if the graph size has changed
    if ((int)graph->node_props.size() != cgraph->n_nodes) {
        res = true;
        graph->node_props.resize(cgraph->n_nodes);
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_cuda_graph::node_properties prop = {};
        memcpy(&prop.node, cgraph->nodes[i], sizeof(ggml_tensor));

        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (cgraph->nodes[i]->src[j]) {
                prop.node_src_data_ptrs[j] = cgraph->nodes[i]->src[j]->data;
                memcpy(prop.node_src_ne[j], cgraph->nodes[i]->src[j]->ne, sizeof(prop.node_src_ne[j]));
                memcpy(prop.node_src_nb[j], cgraph->nodes[i]->src[j]->nb, sizeof(prop.node_src_nb[j]));
            }
        }

        if (res || memcmp(&graph->node_props[i], &prop, sizeof(prop)) != 0) {
            graph->node_props[i] = prop;
            res = true;
        }
    }

    return res;
}

static void ggml_cuda_graph_update_executable(ggml_backend_cuda_context * cuda_ctx, const void * graph_key) {
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

#if CUDART_VERSION >= 12000
    cudaGraphExecUpdateResultInfo result_info;
    cudaError_t stat = cudaGraphExecUpdate(graph->instance, graph->graph, &result_info);
#else
    cudaGraphNode_t errorNode;
    cudaGraphExecUpdateResult result_info;
    cudaError_t stat = cudaGraphExecUpdate(graph->instance, graph->graph, &errorNode, &result_info);
#endif // CUDART_VERSION >= 12000

    if (stat == cudaErrorGraphExecUpdateFailure) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: CUDA graph update failed\n", __func__);
#endif

        // The pre-existing graph exec cannot be updated due to violated constraints
        // so instead clear error and re-instantiate
        (void)cudaGetLastError();
        CUDA_CHECK(cudaGraphExecDestroy(graph->instance));
        graph->instance = nullptr;
        CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));
    } else {
        GGML_ASSERT(stat == cudaSuccess);
    }
}
#endif // USE_CUDA_GRAPH

static bool ggml_cuda_should_fuse_rope_set_rows(const ggml_tensor * rope,
                                                const ggml_tensor * view,
                                                const ggml_tensor * set_rows) {

    if (rope->op != GGML_OP_ROPE || view->op != GGML_OP_VIEW || set_rows->op != GGML_OP_SET_ROWS) {
        return false;
    }
    // ne3 not tested
    if (rope->src[0]->ne[3] != 1) {
        return false;
    }

    if (set_rows->type != GGML_TYPE_F32 && set_rows->type != GGML_TYPE_F16) {
        return false;
    }

    if (set_rows->src[1]->type != GGML_TYPE_I64) {
        return false;
    }

    // The view should flatten two dims of rope into one dim
    if (!ggml_is_contiguous(view) || view->ne[0] != rope->ne[0] * rope->ne[1]) {
        return false;
    }

    // Only norm/neox shaders have the fusion code
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) {
        return false;
    }

    return true;
}

static bool ggml_cuda_should_fuse_rms_norm_mul_rope(const ggml_tensor * rms_norm,
                                                    const ggml_tensor * mul,
                                                    const ggml_tensor * rope) {
    if (rms_norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || rope->op != GGML_OP_ROPE) {
        return false;
    }

    if (rms_norm->src[0]->type != GGML_TYPE_F32 || rms_norm->type != GGML_TYPE_F32 ||
        mul->src[0]->type != GGML_TYPE_F32 || mul->src[1]->type != GGML_TYPE_F32 ||
        mul->type != GGML_TYPE_F32 || rope->type != GGML_TYPE_F32) {
        return false;
    }

    if (rope->src[0] != mul) {
        return false;
    }

    //if rms norm is the B operand, then we don't handle broadcast
    if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
        return false;
    }

    if (!ggml_are_same_shape(rms_norm, mul)) {
        return false;
    }

    //rms_norm kernel assumes contiguous rows
    if (!ggml_is_contiguous_rows(rms_norm->src[0]) ||
        !ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
        return false;
    }

    // the fused kernel handles the norm/neox rope modes only
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) {
        return false;
    }

    const int n_dims = ((const int32_t *) rope->op_params)[1];
    if (n_dims % 2 != 0 || rope->src[0]->ne[0] % 2 != 0) {
        return false;
    }

    // ggml_rope_set_offset is not yet supported in the fused kernel
    const int n_offs = ((const int32_t *) rope->op_params)[15];
    if (n_offs != 0) {
        return false;
    }

    return true;
}

// match gated_delta_net + the strided cpy that scatters its state snapshots into the cache
// (slot i -> rollback group i, slot 0 newest), so the kernel can write them and skip the cpy.
static int ggml_cuda_try_gdn_cache_fusion(
        const ggml_cgraph * cgraph, int node_idx, ggml_cuda_gated_delta_net_fused_cache & fused_state_cpy) {
    const ggml_tensor * gdn = cgraph->nodes[node_idx];
    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    if (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 ||
        (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }

    const ggml_tensor * src_v     = gdn->src[2];
    const int64_t       S_v       = src_v->ne[0];
    const int64_t       H         = src_v->ne[1];
    const int64_t       n_tokens  = src_v->ne[2];
    const int64_t       n_seqs    = src_v->ne[3];
    const int64_t       D         = S_v * S_v * H;
    const int64_t       K         = ggml_get_op_params_i32(gdn, 0); // snapshot slot count
    const int64_t       n_written = std::min<int64_t>(n_tokens, K); // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    const size_t tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    // snapshot cpy is the first real node after the gdn (skip views/no-ops)
    const ggml_tensor * cpy  = nullptr;
    int                 skip = 0;
    for (int j = node_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        if (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
        cpy  = n;
        skip = j - node_idx;
    }
    if (cpy == nullptr) {
        return 0;
    }

    const ggml_tensor * src = cpy->src[0]; // view of the gdn snapshot tail
    const ggml_tensor * dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    if (src->op != GGML_OP_VIEW || src->view_src != gdn || src->view_offs != tail_off ||
        !ggml_is_contiguous(src)) {
        return 0;
    }

    // dst is the [D, n_seqs, n_written] cache view; require nb[1] == D (the per-seq stride the kernel
    // assumes). ggml_cpy pins src to the same element count.
    const std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    if (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->data == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) || dst->nb[1] != (size_t) ggml_row_size(GGML_TYPE_F32, D)) {
        return 0;
    }

    fused_state_cpy.data        = (float *) dst->data; // rollback group 0 (newest)
    fused_state_cpy.slot_stride = K > 1 ? (int64_t) (dst->nb[2] / sizeof(float)) : 0;
    return skip;
}

static bool ggml_cuda_topk_moe_fusion(const struct ggml_cgraph * cgraph, int node_idx, ggml_cuda_topk_moe_args & args) {
    args.sigmoid         = false;
    args.sqrt_softplus   = false;
    args.softmax         = false;
    args.delayed_softmax = false;
    args.prob_bias       = false;
    args.norm            = false;

    const int      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    if (nodes[node_idx]->op == GGML_OP_SOFT_MAX) {
        args.softmax = true;
    }

    if (nodes[node_idx]->op == GGML_OP_UNARY) {
        const ggml_unary_op unary_op = ggml_get_unary_op(nodes[node_idx]);
        if (unary_op == GGML_UNARY_OP_SIGMOID) {
            args.sigmoid = true;
        } else if (unary_op == GGML_UNARY_OP_SOFTPLUS && node_idx + 1 < n_nodes &&
                   nodes[node_idx + 1]->op == GGML_OP_SQRT && nodes[node_idx + 1]->src[0] == nodes[node_idx]) {
            // sqrt(softplus(x)) scoring (DeepSeek-V4)
            args.sqrt_softplus = true;
            node_idx++;
        } else {
            return false;
        }
    }

    if (nodes[node_idx]->op == GGML_OP_ARGSORT) {
        args.delayed_softmax = true;
    }

    node_idx++;

    if (args.sigmoid || args.sqrt_softplus || args.softmax) {
        // SOFTMAX -> RESHAPE
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_RESHAPE ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx];
        node_idx++;

        if (node_idx >= n_nodes) {
            return false;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        if (nodes[node_idx]->op == GGML_OP_ADD && nodes[node_idx]->src[0] == nodes[node_idx - 2]) {
            args.prob_bias = true;
            node_idx++;
        }
        // RESHAPE/ADD -> ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_ARGSORT) {
            return false;
        }

        if (args.prob_bias && nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        } else if (!args.prob_bias && nodes[node_idx]->src[0] != nodes[node_idx - 2]) {
            return false;
        }

        node_idx++;

        // ARGSORT-> VIEW
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;

        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_GET_ROWS) {
            return false;
        }

        // GET_ROWS
        if (nodes[node_idx]->src[0] != probs_reshaped || nodes[node_idx]->src[1] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;
    } else if (args.delayed_softmax) {
        if (node_idx - 2 < 0) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx - 2];

        // VIEW->ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
            nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;

        // GET_ROWS
        if (node_idx >= n_nodes || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
                nodes[node_idx]->src[0] != probs_reshaped) {
            return false;
        }
        node_idx++;

        static const std::vector<ggml_op> remaining_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        for (const ggml_op op : remaining_ops) {
            if (node_idx >= n_nodes || nodes[node_idx]->op != op || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
                return false;
            }
            node_idx++;
        }
    }

    // At this point we can check for norm + scale. Everything is now at least valid till the norm
    if (node_idx >= n_nodes) {
        return true;
    }

    if (nodes[node_idx]->op == GGML_OP_RESHAPE) {
        //check RESHAPE->SUM_ROWS->CLAMP->DIV->RESHAPE
        static const std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP };

        args.norm = true;
        for (const ggml_op op : norm_ops) {
            if (nodes[node_idx]->op == op && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
                node_idx++;
            } else {
                args.norm = false;
                return true;
            }
        }

        // DIV <- CLAMP, RESHAPE
        if (nodes[node_idx]->op != GGML_OP_DIV || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
            nodes[node_idx]->src[0] != nodes[node_idx - 3]) {
            args.norm = false;
            return true;
        }
        node_idx++;

        if (nodes[node_idx]->op != GGML_OP_RESHAPE || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            args.norm = false;
            return true;
        }

        node_idx++;
    }

    if (nodes[node_idx]->op == GGML_OP_SCALE && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
        args.scale = true;
    }

    return true;
}

// returns whether the write (out) nodes overwrite the read nodes in operation
static bool ggml_cuda_check_fusion_memory_ranges(const ggml_cgraph * cgraph,
                                                 const int           node_idx,
                                                 const int           node_count,
                                                 const int *         out_nodes,
                                                 const int           out_count,
                                                 const bool          is_topk_moe = false) {
    auto nodes_overlap = [&](const ggml_tensor * a, const ggml_tensor * b) {
        const int64_t a_start = (int64_t) a->data;
        const int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);

        const int64_t b_start = (int64_t) b->data;
        const int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        if ((b_start <= a_start && a_start < b_end) || (a_start <= b_start && b_start < a_end)) {
            return true;
        }

        return false;
    };

    bool is_ok = true;
    // one block reads all logits before it writes, so logits may alias the out nodes
    const ggml_tensor * logits_may_alias = nullptr;
    if (is_topk_moe && ggml_nrows(cgraph->nodes[node_idx]) <= TOPK_MOE_ROWS_PER_BLOCK) {
        logits_may_alias = cgraph->nodes[node_idx]->src[0];
    }

    for (int i = 0; i < out_count; ++i) {
        const ggml_tensor * dst = cgraph->nodes[out_nodes[i]];

        for (int j = node_idx; j < node_idx + node_count; ++j) {
            // Loop over all srcs of all nodes in the fusion. If the src overlaps
            // the destination and the src is not an intermediate node that's being
            // elided, then disable fusion.

            for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                const ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                if (!src || src->op == GGML_OP_NONE || src == logits_may_alias) {
                    continue;
                }

                if (nodes_overlap(dst, src)) {
                    bool found = false;

                    for (int k = node_idx; k < j; ++k) {
                        if (cgraph->nodes[k] == src) {
                            found = true;
                            break;
                        }
                    }

                    if (!found) {
                        is_ok = false;
                        break;
                    }
                }
            }
        }
    }

    return is_ok;
}

static bool ggml_cuda_can_share_mmq_input(const ggml_tensor * node, int device) {
    if (node->op != GGML_OP_MUL_MAT || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || node->view_src ||
            node->type != GGML_TYPE_F32 || !ggml_is_contiguous(node)) {
        return false;
    }
    const ggml_tensor * weight = node->src[0];
    const ggml_tensor * input = node->src[1];
    const int cc = ggml_cuda_info().devices[device].cc;
    if (input->type != GGML_TYPE_F32 || input->nb[0] != sizeof(float) || input->ne[1] < 1 ||
            input->ne[2] != 1 || input->ne[3] != 1 || weight->ne[2] != 1 || weight->ne[3] != 1 ||
            !ggml_is_contiguous(weight) || ggml_cuda_should_use_mmvq(weight->type, cc, input->ne[1]) ||
            !ggml_cuda_should_use_mmq(weight->type, cc, input->ne[1], /*n_experts =*/ 0) || ggml_cuda_op_mul_mat_use_fwht(node)) {
        return false;
    }
    for (const ggml_tensor * tensor : { weight, input, node }) {
        if (!tensor->buffer || !tensor->data || tensor->buffer->buft != ggml_backend_cuda_buffer_type(device)) {
            return false;
        }
    }
    return !(weight->view_src && ggml_backend_buffer_get_usage(weight->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
        ggml_nbytes(weight) != ggml_backend_buffer_get_alloc_size(weight->buffer, weight));
}

static bool ggml_cuda_mmq_input_overwritten(const ggml_tensor * node, const ggml_tensor * input) {
    if (node->op == GGML_OP_OPT_STEP_ADAMW || node->op == GGML_OP_OPT_STEP_SGD) {
        return true;
    }
    if (ggml_cuda_is_view_or_noop(node) || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        return false;
    }
    const uintptr_t start = (uintptr_t) input->data;
    const uintptr_t end = start + ggml_backend_buffer_get_alloc_size(input->buffer, input);
    const auto overlaps = [&](const void * data, size_t size) {
        const uintptr_t other = (uintptr_t) data;
        return start < other + size && other < end;
    };
    if (!node->buffer || !node->data || overlaps(node->data, ggml_backend_buffer_get_alloc_size(node->buffer, node))) {
        return true;
    }
    if (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID) {
        const ggml_tensor * weight = node->src[0];
        if (weight->buffer && ggml_backend_buffer_get_usage(weight->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
            const size_t size = ggml_nbytes(weight);
            const size_t alloc = ggml_backend_buffer_get_alloc_size(weight->buffer, weight);
            if (alloc > size && overlaps((const char *) weight->data + size, alloc - size)) {
                return true;
            }
        }
    }
    return false;
}

// The long form spans 2*k + 1 nodes. ggml_can_fuse_subgraph() accepts at most
// 31 nodes, so k <= 15; larger values use the per-operation path.
static constexpr int MOE_WEIGHTED_REDUCTION_MAX_EXPERTS = 15;

struct ggml_cuda_moe_weighted_reduction_match {
    const ggml_tensor * experts      = nullptr;
    const ggml_tensor * expert_scale = nullptr;
    const ggml_tensor * weights      = nullptr;
    ggml_tensor *       dst          = nullptr;
    int                 node_count   = 0;
};

static bool ggml_cuda_match_moe_weighted_reduction(
        const ggml_cgraph * cgraph,
        int node_idx,
        ggml_cuda_moe_weighted_reduction_match & match) {
    const ggml_tensor * first = cgraph->nodes[node_idx];
    if (first->op != GGML_OP_MUL || first->type != GGML_TYPE_F32 || !ggml_is_contiguous(first)) {
        return false;
    }

    auto split_mul = [](const ggml_tensor * mul, const ggml_tensor *& full, const ggml_tensor *& broadcast) {
        auto is_weights = [mul](const ggml_tensor * tensor) {
            return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) && tensor->ne[0] == 1 &&
                tensor->ne[1] == mul->ne[1] && tensor->ne[2] == mul->ne[2] && tensor->ne[3] == mul->ne[3];
        };
        auto is_experts = [mul](const ggml_tensor * tensor) {
            return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) &&
                ggml_are_same_shape(tensor, mul);
        };

        if (is_experts(mul->src[0]) && is_weights(mul->src[1])) {
            full      = mul->src[0];
            broadcast = mul->src[1];
            return true;
        }
        if (is_experts(mul->src[1]) && is_weights(mul->src[0])) {
            full      = mul->src[1];
            broadcast = mul->src[0];
            return true;
        }
        return false;
    };

    const ggml_tensor * weighted     = first;
    const ggml_tensor * experts      = nullptr;
    const ggml_tensor * expert_scale = nullptr;
    const ggml_tensor * weights      = nullptr;
    int                 mul_count    = 1;

    // Match both structural forms:
    //   (experts * expert_scale) * router_weight
    //   experts * router_weight
    // The matcher does not depend on the model or quantization type.
    if (node_idx + 1 < cgraph->n_nodes) {
        const ggml_tensor * second = cgraph->nodes[node_idx + 1];
        const ggml_tensor * scaled = nullptr;
        const ggml_tensor * route  = nullptr;
        const ggml_tensor * raw    = nullptr;
        const ggml_tensor * scale  = nullptr;
        if (second->op == GGML_OP_MUL && second->type == GGML_TYPE_F32 && ggml_is_contiguous(second) &&
                split_mul(second, scaled, route) && scaled == first && split_mul(first, raw, scale)) {
            weighted     = second;
            experts      = raw;
            expert_scale = scale;
            weights      = route;
            mul_count    = 2;
        }
    }

    if (experts == nullptr && !split_mul(first, experts, weights)) {
        return false;
    }

    const int     n_expert_used = (int) weighted->ne[1];
    const int64_t n_tokens      = weighted->ne[2] * weighted->ne[3];
    if (n_expert_used < 2 || n_expert_used > MOE_WEIGHTED_REDUCTION_MAX_EXPERTS || n_tokens <= 0) {
        return false;
    }

    const int node_count = 2 * n_expert_used + mul_count - 1;
    if (node_idx + node_count > cgraph->n_nodes) {
        return false;
    }

    std::vector<ggml_op> ops(node_count, GGML_OP_VIEW);
    ops[0] = GGML_OP_MUL;
    if (mul_count == 2) {
        ops[1] = GGML_OP_MUL;
    }
    std::vector<const ggml_tensor *> views;
    views.reserve(n_expert_used);
    const ggml_tensor * previous = nullptr;
    int n_adds = 0;
    for (int offset = mul_count; offset < node_count; ++offset) {
        const ggml_tensor * candidate = cgraph->nodes[node_idx + offset];
        ops[offset] = candidate->op;

        if (candidate->op == GGML_OP_VIEW) {
            const int expert = (int) views.size();
            if (expert >= n_expert_used || candidate->src[0] != weighted || candidate->view_src != weighted ||
                    candidate->type != GGML_TYPE_F32 || candidate->ne[0] != weighted->ne[0] ||
                    candidate->ne[1] != n_tokens || candidate->ne[2] != 1 || candidate->ne[3] != 1 ||
                    candidate->nb[0] != weighted->nb[0] || candidate->nb[1] != weighted->nb[2] ||
                    candidate->view_offs != (size_t) expert * weighted->nb[1]) {
                return false;
            }
            views.push_back(candidate);
            continue;
        }

        if (candidate->op != GGML_OP_ADD || views.size() < 2 || n_adds + 1 >= (int) views.size()) {
            return false;
        }
        const ggml_tensor * lhs = n_adds == 0 ? views[0] : previous;
        const ggml_tensor * rhs = views[n_adds + 1];
        if (candidate->src[0] != lhs || candidate->src[1] != rhs || candidate->type != GGML_TYPE_F32) {
            return false;
        }
        previous = candidate;
        ++n_adds;
    }

    if ((int) views.size() != n_expert_used || n_adds != n_expert_used - 1 || previous == nullptr) {
        return false;
    }
    if (!ggml_is_contiguous(previous) || previous->ne[0] != weighted->ne[0] ||
            previous->ne[1] != n_tokens || previous->ne[2] != 1 || previous->ne[3] != 1) {
        return false;
    }

    const int output_idx = node_idx + node_count - 1;
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, node_count, ops.data(), &output_idx, 1)) {
        return false;
    }

    match.experts      = experts;
    match.expert_scale = expert_scale;
    match.weights      = weights;
    match.dst          = cgraph->nodes[output_idx];
    match.node_count   = node_count;
    return true;
}


static bool ggml_cuda_can_fuse(const struct ggml_cgraph *                cgraph,
                               int                                       node_idx,
                               std::initializer_list<enum ggml_op>       ops,
                               std::initializer_list<enum ggml_unary_op> unary_ops) {
#ifndef NDEBUG
    const size_t num_unary = std::count(ops.begin(), ops.end(), GGML_OP_UNARY);
    GGML_ASSERT(unary_ops.size() == num_unary);
#endif

    const auto is_equal = [](const std::initializer_list<enum ggml_op> & list1,
                             const std::initializer_list<enum ggml_op> & list2) {
        return std::equal(list1.begin(), list1.end(), list2.begin(), list2.end());
    };

    std::initializer_list<enum ggml_op> mul_mat_bias_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_id_bias_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_GLU };

    std::initializer_list<enum ggml_op> mul_mat_id_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_MUL_MAT,    GGML_OP_GLU };

    if ((is_equal(mul_mat_bias_glu_ops, ops) || is_equal(mul_mat_id_bias_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
        const ggml_tensor * ffn_gate      = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_gate_bias = cgraph->nodes[node_idx + 1];
        const ggml_tensor * ffn_up        = cgraph->nodes[node_idx + 2];
        const ggml_tensor * ffn_up_bias   = cgraph->nodes[node_idx + 3];
        const ggml_tensor * glu           = cgraph->nodes[node_idx + 4];

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu, ffn_up_bias, ffn_gate_bias)) {
            int out_nodes[] = { node_idx + 4 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if ((is_equal(mul_mat_id_glu_ops, ops) || is_equal(mul_mat_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * ffn_gate = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_up   = cgraph->nodes[node_idx + 1];
        const ggml_tensor * glu      = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    std::initializer_list<enum ggml_op> rms_norm_mul_rope_ops          = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE };
    std::initializer_list<enum ggml_op> rms_norm_mul_rope_set_rows_ops = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rms_norm_mul_rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 3];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 4];

        if (ggml_check_edges(cgraph, node_idx, {{1, 0, 0}, {2, 0, 1}, {3, 0, 2}, {4, 0, 3}}) &&
            ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope) &&
            ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
            int out_nodes[] = { node_idx + 4 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if (is_equal(rms_norm_mul_rope_ops, ops) && ggml_can_fuse(cgraph, node_idx, ops)) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
        return false;
    }

    std::initializer_list<enum ggml_op> rope_set_rows_ops = { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * rope     = cgraph->nodes[node_idx];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 1];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if (!ggml_can_fuse(cgraph, node_idx, ops)) {
        return false;
    }

    if ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        const ggml_tensor *rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor *mul      = cgraph->nodes[node_idx+1];
        const ggml_tensor *add      = nullptr;

        if (ops.size() == 3 && ops.begin()[2] == GGML_OP_ADD) {
            add = cgraph->nodes[node_idx+2];
        }

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        //rms norm only supports F32
        if (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            return false;
        }

        if (add && (add->src[0]->type != GGML_TYPE_F32 ||
            add->src[1]->type != GGML_TYPE_F32 ||
            add->type != GGML_TYPE_F32) ) {
            return false;
        }

        //if rms norm is the B operand, then we don't handle broadcast
        if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
            return false;
        }

        //rms_norm kernel assumes contiguous rows
        if (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
            return false;
        }

        if (add && (!ggml_is_contiguous(add->src[0]) || !ggml_is_contiguous_rows(add->src[1]))) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_SCALE) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * scale    = cgraph->nodes[node_idx+1];

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        float bias;
        memcpy(&bias, (const float *) scale->op_params + 1, sizeof(float));

        return bias == 0.0f && scale->type == GGML_TYPE_F32;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_UNARY
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * silu     = cgraph->nodes[node_idx+1];
        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            return false;
        }

        if (ssm_conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_ADD
     && ops.begin()[2] == GGML_OP_UNARY && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * add      = cgraph->nodes[node_idx+1];
        const ggml_tensor * silu     = cgraph->nodes[node_idx+2];
        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            return false;
        }

        if (ssm_conv->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }

        // ADD must consume ssm_conv's output and broadcast a 1-D channel-wise bias.
        const ggml_tensor * bias = (add->src[0] == ssm_conv) ? add->src[1] : add->src[0];
        if (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias)) {
            return false;
        }
        if (ggml_nelements(bias) != ssm_conv->ne[0] || bias->ne[0] != ssm_conv->ne[0]) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL
     && unary_ops.size() == 1 && (unary_ops.begin()[0] == GGML_UNARY_OP_SILU || unary_ops.begin()[0] == GGML_UNARY_OP_SIGMOID || unary_ops.begin()[0] == GGML_UNARY_OP_SOFTPLUS)) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * mul   = cgraph->nodes[node_idx+1];

        if (ggml_get_unary_op(unary) != unary_ops.begin()[0]) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16 && unary->type != GGML_TYPE_BF16) {
            return false;
        }

        if (unary->type != mul->type) {
            return false;
        }

        const ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
        if (other->type != unary->type) {
            return false;
        }
        if (!ggml_is_contiguous_1(other) || !ggml_is_contiguous_1(unary->src[0]) || !ggml_are_same_shape(other, unary)) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_SQR
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_RELU) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * sqr   = cgraph->nodes[node_idx+1];

        if (ggml_get_unary_op(unary) != GGML_UNARY_OP_RELU) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
            return false;
        }

        if (unary->type != sqr->type) {
            return false;
        }

        if (!ggml_is_contiguous(unary->src[0])) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SCALE && ops.begin()[1] == GGML_OP_UNARY && ops.begin()[2] == GGML_OP_SCALE
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_TANH) {
        const ggml_tensor *scale  = cgraph->nodes[node_idx];
        const ggml_tensor *tanh   = cgraph->nodes[node_idx+1];
        const ggml_tensor *scale2 = cgraph->nodes[node_idx+2];

        // the fused softcap kernel is F32 only
        if (scale->src[0]->type != GGML_TYPE_F32 || scale->type != GGML_TYPE_F32) {
            return false;
        }

        if (ggml_get_unary_op(tanh) != GGML_UNARY_OP_TANH) {
            return false;
        }

        // Check for bias
        if (ggml_get_op_params_f32(scale, 1) != 0.0f || ggml_get_op_params_f32(scale2, 1) != 0.0f) {
            return false;
        }

        return true;
    }

    return false;
}

static bool ggml_cuda_can_share_mmvq_input(const ggml_tensor * node, int device) {
    if (node->op != GGML_OP_MUL_MAT || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || node->view_src ||
            node->type != GGML_TYPE_F32 || !ggml_is_contiguous(node)) {
        return false;
    }
    const ggml_tensor * weight = node->src[0];
    const ggml_tensor * input = node->src[1];
    const int cc = ggml_cuda_info().devices[device].cc;
    if (cc <= GGML_CUDA_CC_PASCAL || input->type != GGML_TYPE_F32 || input->nb[0] != sizeof(float) ||
            input->ne[1] < 1 || input->ne[2] != 1 || input->ne[3] != 1 || weight->ne[2] != 1 || weight->ne[3] != 1 ||
            !ggml_is_contiguous(weight) || !ggml_cuda_should_use_mmvq(weight->type, cc, input->ne[1]) ||
            ggml_cuda_op_mul_mat_use_fwht(node)) {
        return false;
    }
    for (const ggml_tensor * tensor : { weight, input, node }) {
        if (!tensor->buffer || !tensor->data || tensor->buffer->buft != ggml_backend_cuda_buffer_type(device)) {
            return false;
        }
    }
    return !(weight->view_src && ggml_backend_buffer_get_usage(weight->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
        ggml_nbytes(weight) != ggml_backend_buffer_get_alloc_size(weight->buffer, weight));
}

static bool ggml_cuda_mmvq_input_overwritten(const ggml_tensor * node, const ggml_tensor * input) {
    if (node->op == GGML_OP_OPT_STEP_ADAMW || node->op == GGML_OP_OPT_STEP_SGD) {
        return true;
    }
    if (ggml_cuda_is_view_or_noop(node) || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        return false;
    }
    const uintptr_t start = (uintptr_t) input->data;
    const uintptr_t end = start + ggml_backend_buffer_get_alloc_size(input->buffer, input);
    const auto overlaps = [&](const void * data, size_t size) {
        const uintptr_t other = (uintptr_t) data;
        return start < other + size && other < end;
    };
    if (!node->buffer || !node->data || overlaps(node->data, ggml_backend_buffer_get_alloc_size(node->buffer, node))) {
        return true;
    }
    if (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID) {
        const ggml_tensor * weight = node->src[0];
        if (weight->buffer && ggml_backend_buffer_get_usage(weight->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
            const size_t size = ggml_nbytes(weight);
            const size_t alloc = ggml_backend_buffer_get_alloc_size(weight->buffer, weight);
            if (alloc > size && overlaps((const char *) weight->data + size, alloc - size)) {
                return true;
            }
        }
    }
    return false;
}

static bool ggml_cuda_prepared_bytes(const ggml_tensor * tensor, size_t & bytes) {
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (tensor->ne[d] <= 0) { return false; }
    }
    const size_t block = ggml_blck_size(tensor->type);
    bytes = ggml_type_size(tensor->type);
    int first = 0;
    if (block > 1) {
        if (!tensor->nb[0] || uint64_t(tensor->ne[0]) > SIZE_MAX/tensor->nb[0] || tensor->ne[0] % block) {
            return false;
        }
        bytes = size_t(tensor->ne[0])*tensor->nb[0]/block;
        first = 1;
    }
    for (int d = first; d < GGML_MAX_DIMS; ++d) {
        const uint64_t count = tensor->ne[d] - 1;
        if (count && tensor->nb[d] > (SIZE_MAX - bytes)/count) { return false; }
        bytes += size_t(count)*tensor->nb[d];
    }
    return bytes && bytes == ggml_nbytes(tensor);
}

static bool ggml_cuda_prepared_range(const ggml_tensor * tensor, int device, uintptr_t & begin, uintptr_t & end) {
    if (!tensor->buffer || !tensor->data || tensor->buffer->buft != ggml_backend_cuda_buffer_type(device)) {
        return false;
    }
    const uintptr_t base = (uintptr_t) ggml_backend_buffer_get_base(tensor->buffer);
    const size_t size = ggml_backend_buffer_get_size(tensor->buffer);
    size_t bytes;
    if (!ggml_cuda_prepared_bytes(tensor, bytes)) { return false; }
    begin = (uintptr_t) tensor->data;
    if (begin < base || begin - base > size || bytes > size - (begin - base) || begin > UINTPTR_MAX - bytes) {
        return false;
    }
    if (tensor->view_src) {
        const ggml_tensor * root = tensor->view_src;
        const uintptr_t root_begin = (uintptr_t) root->data;
        size_t root_bytes;
        if (!ggml_cuda_prepared_bytes(root, root_bytes)) { return false; }
        if (root->view_src || root->buffer != tensor->buffer || !root->data || root_begin < base || root_begin - base > size ||
                root_bytes > size - (root_begin - base) || tensor->view_offs > root_bytes || bytes > root_bytes - tensor->view_offs ||
                root_begin > UINTPTR_MAX - tensor->view_offs || begin != root_begin + tensor->view_offs) {
            return false;
        }
    }
    end = begin + bytes;
    return true;
}

static int ggml_cuda_cublas_input_key(const ggml_tensor * node, int device, size_t & size) {
    if (node->op != GGML_OP_MUL_MAT || !(node->flags & GGML_TENSOR_FLAG_COMPUTE) || node->view_src || node->type != GGML_TYPE_F32 ||
            !node->src[0] || !node->src[1] || !ggml_is_contiguous(node) || ggml_cuda_op_mul_mat_use_fwht(node)) {
        return -1;
    }
    const ggml_tensor * weight = node->src[0], * input = node->src[1];
    for (const ggml_tensor * tensor : {weight, input, node}) {
        if (tensor != weight && tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_BF16) {
            return -1;
        }
        uintptr_t begin, end;
        if (!ggml_cuda_prepared_range(tensor, device, begin, end) || tensor->nb[0] != ggml_type_size(tensor->type) || (tensor != weight && begin % ggml_type_size(tensor->type))) {
            return -1;
        }
    }
    if (ggml_cuda_mul_mat_kernel(weight, input, node, device) != GGML_CUDA_MM_CUBLAS) {
        return -1;
    }
    const ggml_type type = ggml_cuda_mul_mat_cublas_compute_type(ggml_cuda_info().devices[device].cc, weight, input, node, false);
    if ((type != GGML_TYPE_F16 && type != GGML_TYPE_BF16) || type == input->type) {
        return -1;
    }
    size_t elements = 1;
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (input->ne[d] <= 0 || uint64_t(input->ne[d]) > SIZE_MAX/elements/sizeof(half)) {
            return -1;
        }
        elements *= input->ne[d];
    }
    size = elements*sizeof(half);
    return int(type);
}

static bool ggml_cuda_prepared_input_overwritten(const ggml_tensor * node, const ggml_tensor * input) {
    if (node->op == GGML_OP_OPT_STEP_ADAMW || node->op == GGML_OP_OPT_STEP_SGD) {
        return true;
    }
    if (ggml_cuda_is_view_or_noop(node) || !(node->flags & GGML_TENSOR_FLAG_COMPUTE)) {
        return false;
    }
    const int device = ((const ggml_backend_cuda_buffer_context *) input->buffer->context)->device;
    uintptr_t begin, end, other_begin, other_end;
    if (!ggml_cuda_prepared_range(input, device, begin, end) || !ggml_cuda_prepared_range(node, device, other_begin, other_end)) {
        return true;
    }
    if (node->op == GGML_OP_FLASH_ATTN_EXT) {
        size_t maximum = other_end - other_begin;
        for (int j = 1; j <= 2; ++j) {
            const ggml_tensor * tensor = node->src[j];
            size_t bytes, elements = 1;
            if (!tensor || !ggml_cuda_prepared_bytes(tensor, bytes)) { return true; }
            for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                if (uint64_t(tensor->ne[d]) > INT64_MAX/elements) { return true; }
                elements *= tensor->ne[d];
            }
            if (maximum > SIZE_MAX - 127 || elements > (SIZE_MAX - 127 - maximum)/sizeof(half)) { return true; }
            maximum += 127 + elements*sizeof(half);
        }
        if (other_begin > UINTPTR_MAX - maximum) { return true; }
    }
    const size_t allocated = ggml_backend_buffer_get_alloc_size(node->buffer, node);
    const uintptr_t base = (uintptr_t) ggml_backend_buffer_get_base(node->buffer);
    const size_t size = ggml_backend_buffer_get_size(node->buffer);
    if (allocated < other_end - other_begin || allocated > size - (other_begin - base) || other_begin > UINTPTR_MAX - allocated ||
            (begin < other_begin + allocated && other_begin < end)) {
        return true;
    }
    if (node->op == GGML_OP_SET) {
        if (!node->src[1]) { return true; }
        ggml_tensor slice = *node;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) { slice.ne[d] = node->src[1]->ne[d]; }
        slice.nb[0] = ggml_type_size(node->type);
        for (int d = 1; d < GGML_MAX_DIMS; ++d) {
            if (node->op_params[d - 1] < 0) { return true; }
            slice.nb[d] = node->op_params[d - 1];
        }
        size_t bytes;
        if (node->op_params[3] < 0 || !ggml_cuda_prepared_bytes(&slice, bytes) ||
                size_t(node->op_params[3]) > allocated || bytes > allocated - size_t(node->op_params[3])) {
            return true;
        }
    }
    if (node->op == GGML_OP_CPY) {
        if (!node->src[1] || !ggml_cuda_prepared_range(node->src[1], device, other_begin, other_end) ||
                (begin < other_end && other_begin < end)) {
            return true;
        }
    }
    if (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID) {
        const ggml_tensor * weight = node->src[0];
        if (weight->buffer && ggml_backend_buffer_get_usage(weight->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
            size_t bytes;
            if (!ggml_cuda_prepared_bytes(weight, bytes)) { return true; }
            const size_t allocated = ggml_backend_buffer_get_alloc_size(weight->buffer, weight);
            if (allocated > bytes) {
                if (!ggml_cuda_prepared_range(weight, device, other_begin, other_end) || other_begin > UINTPTR_MAX - allocated ||
                        (begin < other_begin + allocated && other_end < end)) {
                    return true;
                }
            }
        }
    }
    return false;
}

static void ggml_cuda_prepare_cublas_input(ggml_backend_cuda_context & ctx, const ggml_tensor * input, int key, ggml_cuda_pool_alloc<char> & prepared) {
    const size_t elements = ggml_nelements(input);
    prepared.alloc(elements*sizeof(half));
    const bool contiguous = ggml_is_contiguously_allocated(input);
    if (key == GGML_TYPE_BF16) {
        if (contiguous) {
            ggml_get_to_bf16_cuda(input->type)(input->data, (nv_bfloat16 *) prepared.get(), elements, ctx.stream());
        } else {
            ggml_get_to_bf16_nc_cuda(input->type)(input->data, (nv_bfloat16 *) prepared.get(), input->ne[0], input->ne[1], input->ne[2], input->ne[3],
                input->nb[1]/ggml_type_size(input->type), input->nb[2]/ggml_type_size(input->type), input->nb[3]/ggml_type_size(input->type), ctx.stream());
        }
    } else {
        if (contiguous) {
            ggml_get_to_fp16_cuda(input->type)(input->data, (half *) prepared.get(), elements, ctx.stream());
        } else {
            ggml_get_to_fp16_nc_cuda(input->type)(input->data, (half *) prepared.get(), input->ne[0], input->ne[1], input->ne[2], input->ne[3],
                input->nb[1]/ggml_type_size(input->type), input->nb[2]/ggml_type_size(input->type), input->nb[3]/ggml_type_size(input->type), ctx.stream());
        }
    }
}

struct ggml_cuda_hc_post_norm_match {
    ggml_tensor * post = nullptr;
    ggml_tensor * norm = nullptr;
    ggml_tensor * mul = nullptr;
    ggml_tensor * scale = nullptr;
    int count = 0;
};

static ggml_cuda_hc_post_norm_match ggml_cuda_match_hc_post_norm(ggml_cgraph * cgraph, int i) {
    ggml_cuda_hc_post_norm_match match;
    if (cgraph->nodes[i]->op != GGML_OP_DSV4_HC_POST || i + 1 >= cgraph->n_nodes) {
        return match;
    }
    ggml_tensor * post = cgraph->nodes[i];
    int norm_i = i + 1;
    const bool reshape = cgraph->nodes[norm_i]->op == GGML_OP_RESHAPE;
    if (reshape) {
        const ggml_tensor * view = cgraph->nodes[norm_i];
        if (view->src[0] != post || view->view_src != post || view->view_offs != 0 || !ggml_is_contiguous(view) || ++norm_i >= cgraph->n_nodes) {
            return match;
        }
    }
    ggml_tensor * norm = cgraph->nodes[norm_i];
    if (norm->op != GGML_OP_RMS_NORM || norm->src[0] != cgraph->nodes[norm_i - 1]) {
        return match;
    }
    if (ggml_cuda_can_fuse(cgraph, norm_i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {}) ||
            ggml_cuda_can_fuse(cgraph, norm_i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE }, {}) ||
            ggml_cuda_can_fuse(cgraph, norm_i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD }, {})) {
        return match;
    }
    const bool with_scale = ggml_cuda_can_fuse(cgraph, norm_i, { GGML_OP_RMS_NORM, GGML_OP_SCALE }, {});
    const bool weighted = ggml_cuda_can_fuse(cgraph, norm_i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {});
    ggml_tensor * mul = weighted ? cgraph->nodes[norm_i + 1] : nullptr;
    ggml_tensor * scale = with_scale ? cgraph->nodes[norm_i + 1] : nullptr;
    bool closed = reshape
        ? (weighted ? ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_DSV4_HC_POST, GGML_OP_RESHAPE, GGML_OP_RMS_NORM, GGML_OP_MUL }, { i, i + 1, i + 3 }) : ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_DSV4_HC_POST, GGML_OP_RESHAPE, GGML_OP_RMS_NORM }, { i, i + 1, i + 2 }))
        : (weighted ? ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_DSV4_HC_POST, GGML_OP_RMS_NORM, GGML_OP_MUL }, { i, i + 2 }) : ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_DSV4_HC_POST, GGML_OP_RMS_NORM }, { i, i + 1 }));
    if (scale) {
        closed = reshape
            ? ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_DSV4_HC_POST, GGML_OP_RESHAPE, GGML_OP_RMS_NORM, GGML_OP_SCALE }, { i, i + 1, i + 3 })
            : ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_DSV4_HC_POST, GGML_OP_RMS_NORM, GGML_OP_SCALE }, { i, i + 2 });
    }
    const bool supported = scale ? ggml_cuda_should_fuse_hc_post_norm_scale(post, norm, scale) : ggml_cuda_should_fuse_hc_post_norm(post, norm, mul);
    if (closed && supported) {
        match.post = post;
        match.norm = norm;
        match.mul = mul;
        match.scale = scale;
        match.count = norm_i - i + 1 + int(weighted) + int(with_scale);
    }
    return match;
}

static bool ggml_cuda_hc_post_norm_memory_ok(const ggml_cgraph * cgraph, int i, const ggml_cuda_hc_post_norm_match & match, int device) {
    const auto range = [device](const ggml_tensor * tensor, uintptr_t & begin, uintptr_t & end) {
        if (!tensor || !tensor->buffer || !tensor->data || !ggml_backend_buffer_is_cuda(tensor->buffer)) {
            return false;
        }
        const auto * ctx = (const ggml_backend_cuda_buffer_context *) tensor->buffer->context;
        if (ctx->device != device) {
            return false;
        }
        begin = (uintptr_t) tensor->data;
        const uintptr_t base = (uintptr_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t size = ggml_backend_buffer_get_size(tensor->buffer);
        const size_t alloc = ggml_backend_buffer_get_alloc_size(tensor->buffer, tensor);
        if (begin < base || begin - base > size || alloc < ggml_nbytes(tensor) || alloc > size - (begin - base) || begin > UINTPTR_MAX - alloc || begin % sizeof(float) != 0) {
            return false;
        }
        end = begin + alloc;
        if (tensor->view_src) {
            const ggml_tensor * root = tensor->view_src;
            if (root->view_src || root->buffer != tensor->buffer || !root->data || (uintptr_t) root->data < base ||
                    (uintptr_t) root->data - base > size || ggml_nbytes(root) > size - ((uintptr_t) root->data - base) || tensor->view_offs > ggml_nbytes(root) ||
                    begin != (uintptr_t) root->data + tensor->view_offs || ggml_nbytes(tensor) > ggml_nbytes(root) - tensor->view_offs) {
                return false;
            }
        }
        return true;
    };
    const ggml_tensor * writes[] = { match.post, match.scale ? match.scale : match.mul ? match.mul : match.norm };
    uintptr_t begin[2], end[2];
    for (int w = 0; w < 2; ++w) {
        if (!range(writes[w], begin[w], end[w])) {
            return false;
        }
    }
    for (int j = i + 1; j < i + match.count; ++j) {
        if (cgraph->nodes[j]->op == GGML_OP_RESHAPE) {
            uintptr_t lo, hi;
            const ggml_tensor * view = cgraph->nodes[j];
            if (!range(view, lo, hi) || view->buffer != match.post->buffer || lo != begin[0] || ggml_nbytes(view) != ggml_nbytes(match.post)) {
                return false;
            }
        }
    }
    const auto overlaps = [](uintptr_t a, uintptr_t b, uintptr_t c, uintptr_t d) { return a < d && c < b; };
    if (overlaps(begin[0], end[0], begin[1], end[1]) && (begin[0] != begin[1] || end[0] != end[1])) {
        return false;
    }
    for (int j = i; j < i + match.count; ++j) {
        for (const ggml_tensor * src : cgraph->nodes[j]->src) {
            if (!src) {
                continue;
            }
            bool internal = false;
            for (int k = i; k < j; ++k) {
                internal |= src == cgraph->nodes[k];
            }
            if (internal) {
                continue;
            }
            uintptr_t lo, hi;
            if (!range(src, lo, hi)) {
                return false;
            }
            for (int w = 0; w < 2; ++w) {
                if (!overlaps(begin[w], end[w], lo, hi)) {
                    continue;
                }
                if (src == match.post->src[1] && !match.post->src[3] &&
                        ggml_are_same_shape(match.post, src) && ggml_are_same_stride(match.post, src) && begin[w] == lo && end[w] == hi) {
                    continue;
                }
                return false;
            }
        }
    }
    return true;
}

// try and fuse nodes and return the number of nodes to skip
struct ggml_cuda_norm_emit_match {
    ggml_tensor * post = nullptr;
    ggml_tensor * norm = nullptr;
    ggml_tensor * mul = nullptr;
    ggml_tensor * add = nullptr;
    ggml_tensor * scale = nullptr;
    ggml_tensor * dst = nullptr;
    int last = -1;
    int f16 = -1;
    int bf16 = -1;
};

static ggml_cuda_norm_emit_match ggml_cuda_match_norm_emit(ggml_cgraph * graph, int i, int device) {
    if (graph->nodes[i]->op == GGML_OP_DSV4_HC_POST) {
        const auto hc = ggml_cuda_match_hc_post_norm(graph, i);
        if (!hc.count || !ggml_cuda_hc_post_norm_memory_ok(graph, i, hc, device)) { return {}; }
        ggml_cuda_norm_emit_match match;
        match.post = hc.post;
        match.norm = hc.norm;
        match.mul = hc.mul;
        match.scale = hc.scale;
        match.dst = hc.scale ? hc.scale : hc.mul ? hc.mul : hc.norm;
        match.last = i + hc.count - 1;
        if (!(match.norm->flags & GGML_TENSOR_FLAG_COMPUTE) || !(match.dst->flags & GGML_TENSOR_FLAG_COMPUTE)) { return {}; }
        return match;
    }
    ggml_tensor * norm = graph->nodes[i];
    if (norm->op != GGML_OP_RMS_NORM || !norm->src[0] || norm->type != GGML_TYPE_F32 || norm->src[0]->type != GGML_TYPE_F32 ||
            !(norm->flags & GGML_TENSOR_FLAG_COMPUTE)) { return {}; }
    if (ggml_cuda_can_fuse(graph, i, {GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS}, {}) ||
            ggml_cuda_can_fuse(graph, i, {GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE}, {})) { return {}; }
    ggml_cuda_norm_emit_match match;
    match.norm = match.dst = norm;
    match.last = i;
    if (ggml_cuda_can_fuse(graph, i, {GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD}, {})) {
        match.mul = graph->nodes[i + 1]; match.add = match.dst = graph->nodes[i + 2]; match.last = i + 2;
    } else if (ggml_cuda_can_fuse(graph, i, {GGML_OP_RMS_NORM, GGML_OP_MUL}, {})) {
        match.mul = match.dst = graph->nodes[i + 1]; match.last = i + 1;
    } else if (ggml_cuda_can_fuse(graph, i, {GGML_OP_RMS_NORM, GGML_OP_SCALE}, {})) {
        match.scale = match.dst = graph->nodes[i + 1]; match.last = i + 1;
    }
    const ggml_tensor * weight = match.mul ? (match.mul->src[0] == norm ? match.mul->src[1] : match.mul->src[0]) : nullptr;
    const ggml_tensor * bias = match.add ? (match.add->src[0] == match.mul ? match.add->src[1] : match.add->src[0]) : nullptr;
    uintptr_t begin, end;
    if (!(match.dst->flags & GGML_TENSOR_FLAG_COMPUTE) || !ggml_are_same_shape(norm->src[0], norm) || !ggml_are_same_shape(norm, match.dst) ||
            !ggml_is_contiguous(match.dst) || !ggml_cuda_prepared_range(match.dst, device, begin, end) ||
            begin % sizeof(float) || (end - begin)/sizeof(float) > INT_MAX) { return {}; }
    const ggml_tensor * reads[] = {norm->src[0], weight, bias};
    for (const ggml_tensor * read : reads) {
        if (!read) { continue; }
        uintptr_t read_begin, read_end;
        if (read->type != GGML_TYPE_F32 || read->nb[0] != sizeof(float) || !ggml_cuda_prepared_range(read, device, read_begin, read_end) ||
                read_begin % sizeof(float) || (read_end - read_begin)/sizeof(float) > INT_MAX ||
                (begin < read_end && read_begin < end)) { return {}; }
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            if (read->nb[d] % sizeof(float) || read->nb[d]/sizeof(float) > INT64_MAX || read->ne[d] > INT_MAX) { return {}; }
        }
    }
    if (norm->src[0]->ne[2] > 65535 || norm->src[0]->ne[3] > 65535) { return {}; }
    return match;
}

static std::vector<ggml_cuda_norm_emit_match> ggml_cuda_plan_norm_emit(ggml_cgraph * graph, int device,
        const std::vector<int> & keys, const std::vector<size_t> & sizes, ggml_cuda_reuse_plan & reuse) {
    std::unordered_map<const ggml_tensor *, std::vector<int>> readers;
    for (int i = 0; i < graph->n_nodes; ++i) {
        if (keys[i] != GGML_TYPE_F16 && keys[i] != GGML_TYPE_BF16) { continue; }
        const ggml_tensor * input = graph->nodes[i]->src[1];
        readers[input->view_src ? input->view_src : input].push_back(i);
    }
    if (readers.empty()) { return {}; }
    std::vector<ggml_cuda_norm_emit_match> emits(graph->n_nodes);
    bool moved = false;
    std::vector<bool> removed(reuse.groups.size(), false);
    for (int i = 0; i < graph->n_nodes; ++i) {
        auto match = ggml_cuda_match_norm_emit(graph, i, device);
        if (!match.norm) { continue; }
        const auto can_emit = [&](int node, int prepare) {
            const ggml_tensor * input = graph->nodes[node]->src[1];
            const ggml_tensor * root = input->view_src ? input->view_src : input;
            if (prepare <= match.last || root != match.dst || !ggml_is_contiguous(input) || input->data != match.dst->data ||
                    input->type != GGML_TYPE_F32 || ggml_nbytes(input) != ggml_nbytes(match.dst)) { return false; }
            for (int j = match.last + 1; j <= prepare; ++j) {
                if (ggml_cuda_prepared_input_overwritten(graph->nodes[j], input)) { return false; }
            }
            return true;
        };
        for (size_t g = 0; g < reuse.groups.size(); ++g) {
            auto & group = reuse.groups[g];
            const int key = keys[group.node];
            int & target = key == GGML_TYPE_F16 ? match.f16 : match.bf16;
            if (removed[g] || target >= 0 || (key != GGML_TYPE_F16 && key != GGML_TYPE_BF16) || !can_emit(group.node, group.prepare)) { continue; }
            target = int(g);
            group.prepare = i;
            group.after = true;
            moved = true;
        }
        const auto found = readers.find(match.dst);
        if (found != readers.end()) {
            for (const int node : found->second) {
                int & target = keys[node] == GGML_TYPE_F16 ? match.f16 : match.bf16;
                if (sizes[node] > SIZE_MAX - 255 || !can_emit(node, node)) { continue; }
                const int previous = reuse.nodes.empty() ? -1 : reuse.nodes[node];
                if (previous == target && target >= 0) { continue; }
                if (target >= 0) {
                    auto & group = reuse.groups[target];
                    if (sizes[node] != sizes[group.node]) { continue; }
                    if (previous >= 0) {
                        const auto & other = reuse.groups[previous];
                        if (keys[other.node] != keys[group.node] || !can_emit(other.node, other.prepare)) { continue; }
                        group.last = std::max(group.last, other.last);
                        group.size = std::max(group.size, other.size);
                        for (int & member : reuse.nodes) {
                            if (member == previous) { member = target; }
                        }
                        removed[previous] = true;
                    } else {
                        reuse.nodes[node] = target;
                        group.last = std::max(group.last, node);
                        group.size = std::max(group.size, GGML_PAD(sizes[node], 256));
                    }
                } else {
                    if (previous >= 0) { continue; }
                    if (reuse.nodes.empty()) {
                        reuse.nodes.assign(graph->n_nodes, -1);
                        reuse.starts.assign(graph->n_nodes, -1);
                    }
                    target = int(reuse.groups.size());
                    reuse.groups.push_back({node, i, node, -1, true, GGML_PAD(sizes[node], 256)});
                    removed.push_back(false);
                    reuse.nodes[node] = target;
                }
                moved = true;
            }
        }
        if (match.f16 >= 0 || match.bf16 >= 0) { emits[i] = match; }
    }
    if (moved) {
        std::vector<int> remap(reuse.groups.size(), -1);
        int next = 0;
        for (size_t g = 0; g < reuse.groups.size(); ++g) {
            if (removed[g]) { continue; }
            remap[g] = next;
            reuse.groups[next++] = reuse.groups[g];
        }
        reuse.groups.resize(next);
        for (int & group : reuse.nodes) { if (group >= 0) { group = remap[group]; } }
        for (auto & emit : emits) {
            if (emit.f16 >= 0) { emit.f16 = remap[emit.f16]; }
            if (emit.bf16 >= 0) { emit.bf16 = remap[emit.bf16]; }
        }
        reuse.pack();
    }
    if (reuse.groups.empty()) { return {}; }
    return emits;
}

struct ggml_cuda_norm_q8_match {
    ggml_tensor * post = nullptr;
    ggml_tensor * norm = nullptr;
    ggml_tensor * mul = nullptr;
    ggml_tensor * add = nullptr;
    ggml_tensor * scale = nullptr;
    ggml_tensor * dst = nullptr;
    int last = -1;
    int image = -1;
};

static ggml_cuda_norm_q8_match ggml_cuda_match_norm_q8(ggml_cgraph * graph, int i, int device) {
    if (graph->nodes[i]->op == GGML_OP_DSV4_HC_POST) {
        const auto hc = ggml_cuda_match_hc_post_norm(graph, i);
        if (!hc.count || hc.norm->ne[0] % QK8_1 || !ggml_cuda_hc_post_norm_memory_ok(graph, i, hc, device)) { return {}; }
        ggml_cuda_norm_q8_match match;
        match.post = hc.post;
        match.norm = hc.norm;
        match.mul = hc.mul;
        match.scale = hc.scale;
        match.dst = hc.scale ? hc.scale : hc.mul ? hc.mul : hc.norm;
        match.last = i + hc.count - 1;
        if (!(match.norm->flags & GGML_TENSOR_FLAG_COMPUTE) || !(match.dst->flags & GGML_TENSOR_FLAG_COMPUTE)) { return {}; }
        return match;
    }
    ggml_tensor * norm = graph->nodes[i];
    if (norm->op != GGML_OP_RMS_NORM || !norm->src[0] || norm->type != GGML_TYPE_F32 || norm->src[0]->type != GGML_TYPE_F32 ||
            !(norm->flags & GGML_TENSOR_FLAG_COMPUTE)) { return {}; }
    if (ggml_cuda_can_fuse(graph, i, {GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS}, {}) ||
            ggml_cuda_can_fuse(graph, i, {GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE}, {})) { return {}; }
    ggml_cuda_norm_q8_match match;
    match.norm = match.dst = norm;
    match.last = i;
    if (ggml_cuda_can_fuse(graph, i, {GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD}, {})) {
        match.mul = graph->nodes[i + 1]; match.add = match.dst = graph->nodes[i + 2]; match.last = i + 2;
    } else if (ggml_cuda_can_fuse(graph, i, {GGML_OP_RMS_NORM, GGML_OP_MUL}, {})) {
        match.mul = match.dst = graph->nodes[i + 1]; match.last = i + 1;
    } else if (ggml_cuda_can_fuse(graph, i, {GGML_OP_RMS_NORM, GGML_OP_SCALE}, {})) {
        match.scale = match.dst = graph->nodes[i + 1]; match.last = i + 1;
    }
    const ggml_tensor * weight = match.mul ? (match.mul->src[0] == norm ? match.mul->src[1] : match.mul->src[0]) : nullptr;
    const ggml_tensor * bias = match.add ? (match.add->src[0] == match.mul ? match.add->src[1] : match.add->src[0]) : nullptr;
    uintptr_t begin, end;
    if (!(match.dst->flags & GGML_TENSOR_FLAG_COMPUTE) || !ggml_are_same_shape(norm->src[0], norm) || !ggml_are_same_shape(norm, match.dst) ||
            !ggml_is_contiguous(match.dst) || !ggml_cuda_prepared_range(match.dst, device, begin, end) ||
            begin % sizeof(float) || (end - begin)/sizeof(float) > INT_MAX) { return {}; }
    const ggml_tensor * reads[] = {norm->src[0], weight, bias};
    for (const ggml_tensor * read : reads) {
        if (!read) { continue; }
        uintptr_t read_begin, read_end;
        if (read->type != GGML_TYPE_F32 || read->nb[0] != sizeof(float) || !ggml_cuda_prepared_range(read, device, read_begin, read_end) ||
                read_begin % sizeof(float) || (read_end - read_begin)/sizeof(float) > INT_MAX ||
                (begin < read_end && read_begin < end)) { return {}; }
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            if (read->nb[d] % sizeof(float) || read->nb[d]/sizeof(float) > INT64_MAX || read->ne[d] > INT_MAX) { return {}; }
        }
    }
    if (norm->ne[0] % QK8_1 || norm->src[0]->ne[2] > 65535 || norm->src[0]->ne[3] > 65535) { return {}; }
    return match;
}

static std::vector<ggml_cuda_norm_q8_match> ggml_cuda_plan_norm_q8(ggml_cgraph * graph, int device,
        const std::vector<int> & keys, const std::vector<size_t> & sizes, ggml_cuda_reuse_plan & reuse) {
    std::unordered_map<const ggml_tensor *, std::vector<int>> readers;
    for (int i = 0; i < graph->n_nodes; ++i) {
        if (keys[i] < 0) { continue; }
        const ggml_tensor * input = graph->nodes[i]->src[1];
        readers[input->view_src ? input->view_src : input].push_back(i);
    }
    if (readers.empty()) { return {}; }
    std::vector<ggml_cuda_norm_q8_match> emits(graph->n_nodes);
    bool moved = false;
    std::vector<bool> removed(reuse.groups.size(), false);
    for (int i = 0; i < graph->n_nodes; ++i) {
        auto match = ggml_cuda_match_norm_q8(graph, i, device);
        if (!match.norm) { continue; }
        const auto can_emit = [&](int node, int prepare) {
            const ggml_tensor * input = graph->nodes[node]->src[1];
            const ggml_tensor * root = input->view_src ? input->view_src : input;
            uintptr_t begin, end;
            if (prepare <= match.last || root != match.dst || !ggml_is_contiguous(input) || input->data != match.dst->data ||
                    input->type != GGML_TYPE_F32 || input->ne[0] % QK8_1 || ggml_nbytes(input) != ggml_nbytes(match.dst) ||
                    !ggml_cuda_prepared_range(input, device, begin, end)) { return false; }
            if (match.post && input->ne[0] % MATRIX_ROW_PADDING && match.norm->ne[0] >= input->ne[0]) {
                return false;
            }
            for (int j = match.last + 1; j <= prepare; ++j) {
                if (ggml_cuda_mmvq_input_overwritten(graph->nodes[j], input)) { return false; }
            }
            return true;
        };
        for (size_t g = 0; g < reuse.groups.size(); ++g) {
            auto & group = reuse.groups[g];
            if (removed[g] || !can_emit(group.node, group.prepare)) { continue; }
            match.image = int(g);
            group.prepare = i;
            group.after = true;
            moved = true;
            break;
        }
        const auto found = readers.find(match.dst);
        if (found != readers.end()) {
            for (const int node : found->second) {
                if (sizes[node] > SIZE_MAX - 255 || !can_emit(node, node)) { continue; }
                const int previous = reuse.nodes.empty() ? -1 : reuse.nodes[node];
                if (previous == match.image && match.image >= 0) { continue; }
                if (match.image >= 0) {
                    auto & group = reuse.groups[match.image];
                    const ggml_tensor * image = graph->nodes[group.node]->src[1];
                    const ggml_tensor * input = graph->nodes[node]->src[1];
                    if (input->ne[0] != image->ne[0] || input->ne[1] != image->ne[1]) {
                        if (input->ne[0] % MATRIX_ROW_PADDING || image->ne[0] % MATRIX_ROW_PADDING ||
                                sizes[node] != sizes[group.node]) { continue; }
                    }
                    if (previous >= 0) {
                        const auto & other = reuse.groups[previous];
                        if (!can_emit(other.node, other.prepare)) { continue; }
                        group.last = std::max(group.last, other.last);
                        group.size = std::max(group.size, other.size);
                        for (int & member : reuse.nodes) {
                            if (member == previous) { member = match.image; }
                        }
                        removed[previous] = true;
                    } else {
                        reuse.nodes[node] = match.image;
                        group.last = std::max(group.last, node);
                        group.size = std::max(group.size, GGML_PAD(sizes[node], 256));
                    }
                } else {
                    if (previous >= 0) { continue; }
                    if (reuse.nodes.empty()) {
                        reuse.nodes.assign(graph->n_nodes, -1);
                        reuse.starts.assign(graph->n_nodes, -1);
                    }
                    match.image = int(reuse.groups.size());
                    reuse.groups.push_back({node, i, node, -1, true, GGML_PAD(sizes[node], 256)});
                    removed.push_back(false);
                    reuse.nodes[node] = match.image;
                }
                moved = true;
            }
        }
        if (match.image >= 0) { emits[i] = match; }
    }
    if (moved) {
        std::vector<int> remap(reuse.groups.size(), -1);
        int next = 0;
        for (size_t g = 0; g < reuse.groups.size(); ++g) {
            if (removed[g]) { continue; }
            remap[g] = next;
            reuse.groups[next++] = reuse.groups[g];
        }
        reuse.groups.resize(next);
        for (int & group : reuse.nodes) { if (group >= 0) { group = remap[group]; } }
        for (auto & emit : emits) { if (emit.image >= 0) { emit.image = remap[emit.image]; } }
        reuse.pack();
    }
    if (reuse.groups.empty()) { return {}; }
    return emits;
}

struct ggml_cuda_norm_mmq_match {
    ggml_tensor * post = nullptr;
    ggml_tensor * norm = nullptr;
    ggml_tensor * mul = nullptr;
    ggml_tensor * add = nullptr;
    ggml_tensor * scale = nullptr;
    ggml_tensor * dst = nullptr;
    int last = -1;
    int image = -1;
};

static ggml_cuda_norm_mmq_match ggml_cuda_match_norm_mmq(ggml_cgraph * graph, int i, int device) {
    if (graph->nodes[i]->op == GGML_OP_DSV4_HC_POST) {
        const auto match = ggml_cuda_match_norm_emit(graph, i, device);
        if (!match.post || match.norm->ne[0] % QK8_1) { return {}; }
        ggml_cuda_norm_mmq_match mmq;
        mmq.post = match.post;
        mmq.norm = match.norm;
        mmq.mul = match.mul;
        mmq.scale = match.scale;
        mmq.dst = match.dst;
        mmq.last = match.last;
        return mmq;
    }
    ggml_tensor * norm = graph->nodes[i];
    if (norm->op != GGML_OP_RMS_NORM || !norm->src[0] || norm->type != GGML_TYPE_F32 || norm->src[0]->type != GGML_TYPE_F32 ||
            !(norm->flags & GGML_TENSOR_FLAG_COMPUTE)) { return {}; }
    if (ggml_cuda_can_fuse(graph, i, {GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS}, {}) ||
            ggml_cuda_can_fuse(graph, i, {GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE}, {})) { return {}; }
    ggml_cuda_norm_mmq_match match;
    match.norm = match.dst = norm;
    match.last = i;
    if (ggml_cuda_can_fuse(graph, i, {GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD}, {})) {
        match.mul = graph->nodes[i + 1]; match.add = match.dst = graph->nodes[i + 2]; match.last = i + 2;
    } else if (ggml_cuda_can_fuse(graph, i, {GGML_OP_RMS_NORM, GGML_OP_MUL}, {})) {
        match.mul = match.dst = graph->nodes[i + 1]; match.last = i + 1;
    } else if (ggml_cuda_can_fuse(graph, i, {GGML_OP_RMS_NORM, GGML_OP_SCALE}, {})) {
        match.scale = match.dst = graph->nodes[i + 1]; match.last = i + 1;
    }
    const ggml_tensor * weight = match.mul ? (match.mul->src[0] == norm ? match.mul->src[1] : match.mul->src[0]) : nullptr;
    const ggml_tensor * bias = match.add ? (match.add->src[0] == match.mul ? match.add->src[1] : match.add->src[0]) : nullptr;
    uintptr_t begin, end;
    if (!(match.dst->flags & GGML_TENSOR_FLAG_COMPUTE) || !ggml_are_same_shape(norm->src[0], norm) || !ggml_are_same_shape(norm, match.dst) ||
            !ggml_is_contiguous(match.dst) || !ggml_cuda_prepared_range(match.dst, device, begin, end) ||
            begin % sizeof(float) || (end - begin)/sizeof(float) > INT_MAX) { return {}; }
    const ggml_tensor * reads[] = {norm->src[0], weight, bias};
    for (const ggml_tensor * read : reads) {
        if (!read) { continue; }
        uintptr_t read_begin, read_end;
        if (read->type != GGML_TYPE_F32 || read->nb[0] != sizeof(float) || !ggml_cuda_prepared_range(read, device, read_begin, read_end) ||
                read_begin % sizeof(float) || (read_end - read_begin)/sizeof(float) > INT_MAX ||
                (begin < read_end && read_begin < end)) { return {}; }
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            if (read->nb[d] % sizeof(float) || read->nb[d]/sizeof(float) > INT64_MAX || read->ne[d] > INT_MAX) { return {}; }
        }
    }
    if (norm->ne[0] % QK8_1 || norm->src[0]->ne[2] > 65535 || norm->src[0]->ne[3] > 65535) { return {}; }
    return match;
}

static std::vector<ggml_cuda_norm_mmq_match> ggml_cuda_plan_norm_mmq(ggml_cgraph * graph, int device,
        const std::vector<int> & keys, const std::vector<size_t> & sizes, ggml_cuda_reuse_plan & reuse) {
    std::unordered_map<const ggml_tensor *, std::vector<int>> readers;
    for (int i = 0; i < graph->n_nodes; ++i) {
        if (keys[i] < 0) { continue; }
        const ggml_tensor * input = graph->nodes[i]->src[1];
        readers[input->view_src ? input->view_src : input].push_back(i);
    }
    if (readers.empty()) { return {}; }
    std::vector<ggml_cuda_norm_mmq_match> emits(graph->n_nodes);
    bool moved = false;
    std::vector<bool> removed(reuse.groups.size(), false);
    for (int i = 0; i < graph->n_nodes; ++i) {
        auto match = ggml_cuda_match_norm_mmq(graph, i, device);
        if (!match.norm) { continue; }
        const auto norm_group_invalid = [&](int node) {
            const int group = keys[node] == int(MMQ_Q8_1_DS_LAYOUT_D2S6) ? 64 : 32;
            return match.norm->ne[0] % group || graph->nodes[node]->src[1]->ne[0] % group;
        };
        const auto can_emit = [&](int node, int prepare) {
            const ggml_tensor * input = graph->nodes[node]->src[1];
            const ggml_tensor * root = input->view_src ? input->view_src : input;
            uintptr_t begin, end;
            if (keys[node] < 0 || keys[node] > 3 || (match.post && keys[node] > 2) ||
                    norm_group_invalid(node) || prepare <= match.last || root != match.dst || !ggml_is_contiguous(input) || input->data != match.dst->data ||
                    input->type != GGML_TYPE_F32 || input->ne[0] % QK8_1 || ggml_nbytes(input) != ggml_nbytes(match.dst) ||
                    !ggml_cuda_prepared_range(input, device, begin, end)) { return false; }
            for (int j = match.last + 1; j <= prepare; ++j) {
                if (ggml_cuda_mmq_input_overwritten(graph->nodes[j], input)) { return false; }
            }
            return true;
        };
        for (size_t g = 0; g < reuse.groups.size(); ++g) {
            auto & group = reuse.groups[g];
            if (removed[g] || !can_emit(group.node, group.prepare)) { continue; }
            match.image = int(g);
            group.prepare = i;
            group.after = true;
            moved = true;
            break;
        }
        const auto found = readers.find(match.dst);
        if (found != readers.end()) {
            for (const int node : found->second) {
                if (sizes[node] > SIZE_MAX - 255 || !can_emit(node, node)) { continue; }
                const int previous = reuse.nodes.empty() ? -1 : reuse.nodes[node];
                if (previous == match.image && match.image >= 0) { continue; }
                if (match.image >= 0) {
                    auto & group = reuse.groups[match.image];
                    const ggml_tensor * image = graph->nodes[group.node]->src[1];
                    const ggml_tensor * input = graph->nodes[node]->src[1];
                    if (keys[node] != keys[group.node] || input->ne[0] != image->ne[0] || input->ne[1] != image->ne[1]) { continue; }
                    if (previous >= 0) {
                        const auto & other = reuse.groups[previous];
                        if (!can_emit(other.node, other.prepare)) { continue; }
                        group.last = std::max(group.last, other.last);
                        group.size = std::max(group.size, other.size);
                        for (int & member : reuse.nodes) {
                            if (member == previous) { member = match.image; }
                        }
                        removed[previous] = true;
                    } else {
                        reuse.nodes[node] = match.image;
                        group.last = std::max(group.last, node);
                        group.size = std::max(group.size, GGML_PAD(sizes[node], 256));
                    }
                } else {
                    if (previous >= 0) { continue; }
                    if (reuse.nodes.empty()) {
                        reuse.nodes.assign(graph->n_nodes, -1);
                        reuse.starts.assign(graph->n_nodes, -1);
                    }
                    match.image = int(reuse.groups.size());
                    reuse.groups.push_back({node, i, node, -1, true, GGML_PAD(sizes[node], 256)});
                    removed.push_back(false);
                    reuse.nodes[node] = match.image;
                }
                moved = true;
            }
        }
        if (match.image >= 0) { emits[i] = match; }
    }
    if (moved) {
        std::vector<int> remap(reuse.groups.size(), -1);
        int next = 0;
        for (size_t g = 0; g < reuse.groups.size(); ++g) {
            if (removed[g]) { continue; }
            remap[g] = next;
            reuse.groups[next++] = reuse.groups[g];
        }
        reuse.groups.resize(next);
        for (int & group : reuse.nodes) { if (group >= 0) { group = remap[group]; } }
        for (auto & emit : emits) { if (emit.image >= 0) { emit.image = remap[emit.image]; } }
        reuse.pack();
    }
    if (reuse.groups.empty()) { return {}; }
    return emits;
}


struct ggml_cuda_scaled_unary_match {
    ggml_tensor * first = nullptr;
    ggml_tensor * unary = nullptr;
    ggml_tensor * last = nullptr;
    int count = 0;
};

static ggml_cuda_scaled_unary_match ggml_cuda_match_scaled_unary(ggml_cgraph * cgraph, int i, bool plain = false) {
    ggml_cuda_scaled_unary_match match;
    ggml_tensor * first = cgraph->nodes[i];
    const bool before = first->op == GGML_OP_SCALE;
    const int u = i + int(before);
    if (u >= cgraph->n_nodes || cgraph->nodes[u]->op != GGML_OP_UNARY ||
            (ggml_get_unary_op(cgraph->nodes[u]) != GGML_UNARY_OP_SILU && ggml_get_unary_op(cgraph->nodes[u]) != GGML_UNARY_OP_SIGMOID)) {
        return match;
    }
    if ((before || plain) && (ggml_cuda_can_fuse(cgraph, u, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SILU }) ||
                   ggml_cuda_can_fuse(cgraph, u, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SIGMOID }))) {
        return match;
    }
    if (before && cgraph->nodes[u]->src[0] != first) {
        return match;
    }
    const bool after = u + 1 < cgraph->n_nodes && cgraph->nodes[u + 1]->op == GGML_OP_SCALE && cgraph->nodes[u + 1]->src[0] == cgraph->nodes[u];
    int count = 0;
    if (before && after && ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE }, { i, i + 1, i + 2 })) {
        count = 3;
    } else if (before && ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY }, { i, i + 1 })) {
        count = 2;
    } else if (!before && after && ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_UNARY, GGML_OP_SCALE }, { i, i + 1 })) {
        count = 2;
    }
    if (!before && !after && plain && ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_UNARY }, { i })) { count = 1; }
    if (!count || !first->src[0] || (first->type != GGML_TYPE_F32 && first->type != GGML_TYPE_BF16)) {
        return match;
    }
    const ggml_tensor * src = first->src[0];
    size_t elements = 1;
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (src->ne[d] <= 0 || (size_t) src->ne[d] > (INT_MAX - CUDA_NEG_BLOCK_SIZE)/elements) {
            return match;
        }
        elements *= src->ne[d];
    }
    for (int j = i - 1; j < i + count; ++j) {
        const ggml_tensor * tensor = j < i ? src : cgraph->nodes[j];
        if (tensor->type != first->type || !ggml_is_contiguous(tensor) || !ggml_are_same_shape(src, tensor)) {
            return match;
        }
    }
    match.first = first;
    match.unary = cgraph->nodes[u];
    match.last = cgraph->nodes[i + count - 1];
    match.count = count;
    return match;
}

static bool ggml_cuda_scaled_unary_memory_ok(const ggml_cuda_scaled_unary_match & match, int device) {
    const auto range = [device](const ggml_tensor * tensor, uintptr_t & begin, uintptr_t & end) {
        if (!tensor->buffer || !tensor->data || !ggml_backend_buffer_is_cuda(tensor->buffer) ||
                ((const ggml_backend_cuda_buffer_context *) tensor->buffer->context)->device != device) {
            return false;
        }
        const uintptr_t base = (uintptr_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t size = ggml_backend_buffer_get_size(tensor->buffer);
        const size_t bytes = ggml_nbytes(tensor);
        begin = (uintptr_t) tensor->data;
        if (begin < base || begin - base > size || bytes > size - (begin - base) || begin > UINTPTR_MAX - bytes || begin % ggml_element_size(tensor)) {
            return false;
        }
        if (tensor->view_src) {
            const ggml_tensor * root = tensor->view_src;
            if (root->view_src || root->buffer != tensor->buffer || !root->data || tensor->view_offs > ggml_nbytes(root) ||
                    begin != (uintptr_t) root->data + tensor->view_offs || bytes > ggml_nbytes(root) - tensor->view_offs ||
                    (uintptr_t) root->data < base || (uintptr_t) root->data - base > size || ggml_nbytes(root) > size - ((uintptr_t) root->data - base)) {
                return false;
            }
        }
        end = begin + bytes;
        return true;
    };
    const ggml_tensor * tensors[] = { match.first->src[0], match.first, match.unary, match.last };
    uintptr_t begin[4], end[4];
    for (int j = 0; j < 4; ++j) {
        if (!range(tensors[j], begin[j], end[j])) {
            return false;
        }
        for (int k = 0; k < j; ++k) {
            if (begin[j] < end[k] && begin[k] < end[j] && (begin[j] != begin[k] || end[j] != end[k])) {
                return false;
            }
        }
    }
    return true;
}

static bool ggml_cuda_hc_injection_bytes(const ggml_tensor * tensor, size_t & bytes) {
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (tensor->ne[d] <= 0) { return false; }
    }
    const size_t block = ggml_blck_size(tensor->type);
    bytes = ggml_type_size(tensor->type);
    int first = 0;
    if (block > 1) {
        if (!tensor->nb[0] || uint64_t(tensor->ne[0]) > SIZE_MAX/tensor->nb[0] || tensor->ne[0] % block) {
            return false;
        }
        bytes = size_t(tensor->ne[0])*tensor->nb[0]/block;
        first = 1;
    }
    for (int d = first; d < GGML_MAX_DIMS; ++d) {
        const uint64_t count = tensor->ne[d] - 1;
        if (count && tensor->nb[d] > (SIZE_MAX - bytes)/count) { return false; }
        bytes += size_t(count)*tensor->nb[d];
    }
    return bytes && bytes == ggml_nbytes(tensor);
}

static bool ggml_cuda_hc_injection_range(const ggml_tensor * tensor, int device, uintptr_t & begin, uintptr_t & end) {
    if (!tensor->buffer || !tensor->data || tensor->buffer->buft != ggml_backend_cuda_buffer_type(device)) {
        return false;
    }
    const uintptr_t base = (uintptr_t) ggml_backend_buffer_get_base(tensor->buffer);
    const size_t size = ggml_backend_buffer_get_size(tensor->buffer);
    size_t bytes;
    if (!ggml_cuda_hc_injection_bytes(tensor, bytes)) { return false; }
    begin = (uintptr_t) tensor->data;
    if (begin % sizeof(float)) { return false; }
    if (begin < base || begin - base > size || bytes > size - (begin - base) || begin > UINTPTR_MAX - bytes) {
        return false;
    }
    if (tensor->view_src) {
        const ggml_tensor * root = tensor->view_src;
        const uintptr_t root_begin = (uintptr_t) root->data;
        size_t root_bytes;
        if (!ggml_cuda_hc_injection_bytes(root, root_bytes)) { return false; }
        if (root_begin % sizeof(float) || root->view_src || root->buffer != tensor->buffer || !root->data || root_begin < base || root_begin - base > size ||
                root_bytes > size - (root_begin - base) || tensor->view_offs > root_bytes || bytes > root_bytes - tensor->view_offs ||
                root_begin > UINTPTR_MAX - tensor->view_offs || begin != root_begin + tensor->view_offs) {
            return false;
        }
    }
    end = begin + bytes;
    return true;
}


struct ggml_cuda_hc_injection_match {
    ggml_tensor * first = nullptr;
    ggml_tensor * unary = nullptr;
    ggml_tensor * last = nullptr;
    ggml_tensor * post = nullptr;
    ggml_cuda_hc_post_norm_match hc;
    int count = 0;
};

static ggml_cuda_hc_injection_match ggml_cuda_match_hc_injection(ggml_backend_cuda_context & ctx, ggml_cgraph * graph, int i, bool allocated = true) {
    ggml_cuda_hc_injection_match match;
    ggml_tensor * first = graph->nodes[i];
    const bool before = first->op == GGML_OP_SCALE;
    const int u = i + int(before);
    if (u >= graph->n_nodes || graph->nodes[u]->op != GGML_OP_UNARY) { return match; }
    ggml_tensor * unary = graph->nodes[u];
    const ggml_unary_op op = ggml_get_unary_op(unary);
    if (op != GGML_UNARY_OP_SIGMOID && op != GGML_UNARY_OP_SILU) { return match; }
    const bool after = u + 1 < graph->n_nodes && graph->nodes[u + 1]->op == GGML_OP_SCALE;
    if (!before && !after) { return match; }
    const int weight_last = u + int(after);
    const int post_i = weight_last + 1;
    if (post_i >= graph->n_nodes) { return match; }
    ggml_tensor * last = graph->nodes[weight_last];
    ggml_tensor * post = graph->nodes[post_i];
    const ggml_tensor * raw = first->src[0];
    if (!raw || post->op != GGML_OP_DSV4_HC_POST || post->src[2] != last ||
            (before && unary->src[0] != first) || (after && last->src[0] != unary)) { return match; }
    for (const ggml_tensor * tensor : {raw, (const ggml_tensor *) first, (const ggml_tensor *) unary, (const ggml_tensor *) last}) {
        size_t bytes;
        if (tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor) || !ggml_are_same_shape(raw, tensor) ||
                !ggml_cuda_hc_injection_bytes(tensor, bytes) || bytes/sizeof(float) > INT_MAX) { return {}; }
    }
    const ggml_tensor * x = post->src[0];
    const ggml_tensor * residual = post->src[1];
    const ggml_tensor * comb = post->src[3];
    if (!x || !residual || x->type != GGML_TYPE_F32 || residual->type != GGML_TYPE_F32 || post->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(post) || !ggml_are_same_shape(post, residual) || x->ne[0] <= 0 || residual->ne[1] <= 0 || x->ne[1] <= 0 ||
            x->ne[2] != 1 || x->ne[3] != 1 || residual->ne[0] != x->ne[0] || residual->ne[2] != x->ne[1] || residual->ne[3] != 1 ||
            raw->ne[0] != residual->ne[1] || raw->ne[1] != x->ne[1] || raw->ne[2] != 1 || raw->ne[3] != 1 ||
            (comb && (comb->type != GGML_TYPE_F32 || comb->ne[0] != residual->ne[1] || comb->ne[1] != residual->ne[1] || comb->ne[2] != x->ne[1] || comb->ne[3] != 1))) { return {}; }
    size_t bytes;
    if (!ggml_cuda_hc_injection_bytes(post, bytes) || bytes/sizeof(float) > INT_MAX) { return {}; }
    auto hc = ggml_cuda_match_hc_post_norm(graph, post_i);
    if (hc.count && graph->nodes[post_i + hc.count - 1] != (hc.mul ? hc.mul : hc.norm)) { return {}; }
    if (hc.count && hc.norm->ne[0] != x->ne[0]) {
        if (comb && bytes/sizeof(float) <= 256) { return {}; }
        if (hc.norm->ne[0] < x->ne[0] && hc.norm->ne[0] > WARP_SIZE/2 && hc.norm->ne[0] < WARP_SIZE) { return {}; }
        const int block_size = hc.norm->ne[0] < 1024 ? 256 : 1024;
        const int64_t nweights = (hc.norm->ne[0] + x->ne[0] - 2)/x->ne[0] + 1;
        if ((hc.norm->ne[0] < x->ne[0] && hc.norm->ne[0] <= WARP_SIZE/2) || nweights >= block_size) { hc = {}; }
    }
    const int end = hc.count ? post_i + hc.count : post_i + 1;
    const int64_t ncols = hc.count ? hc.norm->ne[0] : 256;
    const size_t shared = ((hc.count ? 32 : 0) + (ncols + x->ne[0] - 2)/x->ne[0] + 1)*sizeof(float);
    if (shared > ggml_cuda_info().devices[ctx.device].smpb) { return {}; }
    std::vector<ggml_op> ops;
    std::vector<int> outputs;
    for (int j = i; j < end; ++j) {
        ops.push_back(graph->nodes[j]->op);
        if (j <= weight_last || j == post_i || j == end - 1) { outputs.push_back(j); }
    }
    if (!ggml_can_fuse_subgraph(graph, i, ops.size(), ops.data(), outputs.data(), outputs.size())) { return {}; }
    const ggml_tensor * gamma = hc.mul ? hc.mul->src[hc.mul->src[0] == hc.norm ? 1 : 0] : nullptr;
    const ggml_tensor * reads[] = {raw, x, residual, comb, gamma};
    for (const ggml_tensor * tensor : reads) {
        if (!tensor) { continue; }
        const ggml_tensor * root = tensor->view_src ? tensor->view_src : tensor;
        size_t span;
        if (!ggml_cuda_hc_injection_bytes(tensor, span) || span > INT64_MAX) { return {}; }
        for (int d = 0; d < GGML_MAX_DIMS; ++d) { if (tensor->nb[d] % sizeof(float)) { return {}; } }
        for (int j = i; j < end; ++j) { if (graph->nodes[j] == root) { return {}; } }
    }
    if (allocated) {
        if (hc.count && !ggml_cuda_hc_post_norm_memory_ok(graph, post_i, hc, ctx.device)) { return {}; }
        std::vector<const ggml_tensor *> writes = {first, unary, last, post};
        if (hc.count) { writes.push_back(hc.mul ? hc.mul : hc.norm); }
        std::vector<uintptr_t> wb(writes.size()), we(writes.size());
        for (size_t j = 0; j < writes.size(); ++j) {
            if (!ggml_cuda_hc_injection_range(writes[j], ctx.device, wb[j], we[j])) { return {}; }
            for (size_t k = 0; k < j; ++k) {
                if (wb[j] < we[k] && wb[k] < we[j] &&
                        !((j < 3 && k < 3 && wb[j] == wb[k] && we[j] == we[k]) || (j >= 3 && k >= 3 && wb[j] == wb[k] && we[j] == we[k]))) { return {}; }
            }
        }
        for (const ggml_tensor * tensor : reads) {
            if (!tensor) { continue; }
            uintptr_t lo, hi;
            if (!ggml_cuda_hc_injection_range(tensor, ctx.device, lo, hi)) { return {}; }
            for (size_t j = 0; j < writes.size(); ++j) {
                if (!(wb[j] < hi && lo < we[j])) { continue; }
                if (j >= 3 && tensor == residual && !comb && ggml_are_same_shape(writes[j], residual) &&
                        ggml_are_same_stride(writes[j], residual) && wb[j] == lo && we[j] == hi) { continue; }
                return {};
            }
        }
    }
    match.first = first;
    match.unary = unary;
    match.last = last;
    match.post = post;
    match.hc = hc;
    match.count = end - i;
    return match;
}

static bool ggml_cuda_hc_up_bytes(const ggml_tensor * tensor, size_t & bytes) {
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (tensor->ne[d] <= 0) { return false; }
    }
    const size_t block = ggml_blck_size(tensor->type);
    bytes = ggml_type_size(tensor->type);
    int first = 0;
    if (block > 1) {
        if (!tensor->nb[0] || uint64_t(tensor->ne[0]) > SIZE_MAX/tensor->nb[0] || tensor->ne[0] % block) {
            return false;
        }
        bytes = size_t(tensor->ne[0])*tensor->nb[0]/block;
        first = 1;
    }
    for (int d = first; d < GGML_MAX_DIMS; ++d) {
        const uint64_t count = tensor->ne[d] - 1;
        if (count && tensor->nb[d] > (SIZE_MAX - bytes)/count) { return false; }
        bytes += size_t(count)*tensor->nb[d];
    }
    return bytes && bytes == ggml_nbytes(tensor);
}

static bool ggml_cuda_hc_up_range(const ggml_tensor * tensor, int device, uintptr_t & begin, uintptr_t & end) {
    if (!tensor->buffer || !tensor->data || tensor->buffer->buft != ggml_backend_cuda_buffer_type(device)) {
        return false;
    }
    const uintptr_t base = (uintptr_t) ggml_backend_buffer_get_base(tensor->buffer);
    const size_t size = ggml_backend_buffer_get_size(tensor->buffer);
    size_t bytes;
    if (!ggml_cuda_hc_up_bytes(tensor, bytes)) { return false; }
    begin = (uintptr_t) tensor->data;
    if (begin < base || begin - base > size || bytes > size - (begin - base) || begin > UINTPTR_MAX - bytes) {
        return false;
    }
    if (tensor->view_src) {
        const ggml_tensor * root = tensor->view_src;
        const uintptr_t root_begin = (uintptr_t) root->data;
        size_t root_bytes;
        if (!ggml_cuda_hc_up_bytes(root, root_bytes)) { return false; }
        if (root->view_src || root->buffer != tensor->buffer || !root->data || root_begin < base || root_begin - base > size ||
                root_bytes > size - (root_begin - base) || tensor->view_offs > root_bytes || bytes > root_bytes - tensor->view_offs ||
                root_begin > UINTPTR_MAX - tensor->view_offs || begin != root_begin + tensor->view_offs) {
            return false;
        }
    }
    end = begin + bytes;
    return true;
}

struct ggml_cuda_hc_up_match {
    ggml_tensor * mm = nullptr;
    ggml_tensor * dst = nullptr;
    int count = 0;
    bool convert = false;
};

static ggml_cuda_hc_up_match ggml_cuda_match_hc_up_shape(ggml_backend_cuda_context & ctx, ggml_cgraph * graph, int i, bool convert = false) {
    ggml_tensor * mm = graph->nodes[i];
    if (mm->op != GGML_OP_MUL_MAT || !mm->src[0] || !mm->src[1] || mm->type != GGML_TYPE_F32 ||
            mm->src[1]->type != GGML_TYPE_F32 || mm->ne[2] != 1 || mm->ne[3] != 1 || (!convert && mm->ne[1] > MMVF_MAX_BATCH_SIZE) ||
            ggml_cuda_op_mul_mat_use_fwht(mm)) {
        return {};
    }
    int last = i + 1;
    while (last < graph->n_nodes && last < i + 3 && graph->nodes[last]->op == GGML_OP_RESHAPE) { ++last; }
    if (last >= graph->n_nodes || graph->nodes[last]->op != GGML_OP_DSV4_HC_PRE) { return {}; }
    ggml_tensor * dst = graph->nodes[last];
    const ggml_tensor * norm = dst->src[0];
    const ggml_tensor * view = dst->src[1];
    const ggml_tensor * weight = mm->src[0];
    const ggml_tensor * input = mm->src[1];
    if (!norm || !view || !ggml_get_op_params_i32(dst, 1) || norm->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            norm->ne[0] <= 0 || norm->ne[1] <= 0 || norm->ne[2] <= 0 || view->op != GGML_OP_RESHAPE || view->src[0] != mm || view->type != GGML_TYPE_F32 || !ggml_is_contiguous(view) ||
            !ggml_are_same_shape(view, norm) || norm->ne[3] != 1 || mm->ne[0]/norm->ne[0] != norm->ne[1] || mm->ne[0] % norm->ne[0] || mm->ne[1] != norm->ne[2] ||
            dst->ne[0] != norm->ne[0] || dst->ne[1] != norm->ne[2] || dst->ne[2] != 1 || dst->ne[3] != 1 ||
            weight->ne[2] != 1 || weight->ne[3] != 1 || input->ne[2] != 1 || input->ne[3] != 1 ||
            (!convert && weight->type != GGML_TYPE_F32 && weight->type != GGML_TYPE_F16 && weight->type != GGML_TYPE_BF16)) {
        return {};
    }
    if (convert) {
        const int cc = ggml_cuda_info().devices[ctx.device].cc;
        if (ggml_cuda_mul_mat_kernel(weight, input, mm, ctx.device) != GGML_CUDA_MM_CUBLAS) { return {}; }
        const ggml_type type = ggml_cuda_mul_mat_cublas_compute_type(cc, weight, input, mm, false);
        if ((type != GGML_TYPE_F16 && type != GGML_TYPE_BF16) || ggml_cuda_cublas_prefer_f32_output(type, cc)) { return {}; }
    } else if (!ggml_cuda_should_use_mmvf(weight->type, ggml_cuda_info().devices[ctx.device].cc, weight->ne, weight->nb, input->ne[1]) ||
            !ggml_cuda_should_fuse_hc_up(weight, norm, ctx.device)) { return {}; }
    const int count = last - i + 1;
    ggml_op ops[4]; int outputs[4];
    for (int j = 0; j < count; ++j) { ops[j] = graph->nodes[i + j]->op; outputs[j] = i + j; }
    if (!ggml_can_fuse_subgraph(graph, i, count, ops, outputs, count)) { return {}; }
    return {mm, dst, count, convert};
}

static ggml_cuda_hc_up_match ggml_cuda_match_hc_up(ggml_backend_cuda_context & ctx, ggml_cgraph * graph, int i) {
    auto match = ggml_cuda_match_hc_up_shape(ctx, graph, i);
    if (!match.count) { match = ggml_cuda_match_hc_up_shape(ctx, graph, i, true); }
    if (!match.count) { return {}; }
    ggml_tensor * mm = match.mm;
    ggml_tensor * dst = match.dst;
    const ggml_tensor * norm = dst->src[0];
    const ggml_tensor * view = dst->src[1];
    const ggml_tensor * weight = mm->src[0];
    const ggml_tensor * input = mm->src[1];
    const ggml_tensor * tensors[] = {weight, input, mm};
    for (const ggml_tensor * tensor : tensors) {
        uintptr_t begin, end;
        if (!ggml_cuda_hc_up_range(tensor, ctx.device, begin, end) || tensor->nb[0] != ggml_type_size(tensor->type) ||
                (end - begin)/ggml_type_size(tensor->type) > INT_MAX || begin % (match.convert ? sizeof(float) : 2*ggml_type_size(tensor->type))) { return {}; }
        for (int d = 1; d < GGML_MAX_DIMS; ++d) {
            if (tensor->ne[d] > INT_MAX || tensor->nb[d]/ggml_type_size(tensor->type) > INT_MAX ||
                    (tensor != mm && tensor->nb[d] % ((match.convert ? 1 : 2)*ggml_type_size(tensor->type)))) { return {}; }
        }
    }
    if (!(mm->flags & GGML_TENSOR_FLAG_COMPUTE) || !(dst->flags & GGML_TENSOR_FLAG_COMPUTE) || !ggml_is_contiguous(mm) ||
            !ggml_is_contiguous(dst) || (ggml_backend_buffer_get_usage(weight->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
            weight->view_src && ggml_nbytes(weight) != ggml_backend_buffer_get_alloc_size(weight->buffer, weight))) { return {}; }
    uintptr_t begin[2], end[2], reads_begin[3], reads_end[3], view_begin, view_end;
    const ggml_tensor * reads[] = {weight, input, norm};
    for (int j = 0; j < 3; ++j) {
        if (!ggml_cuda_hc_up_range(reads[j], ctx.device, reads_begin[j], reads_end[j]) || reads_begin[j] % ggml_type_size(reads[j]->type)) { return {}; }
    }
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (norm->nb[d] % sizeof(float) || norm->nb[d]/sizeof(float) > INT64_MAX) { return {}; }
    }
    if (!ggml_cuda_hc_up_range(mm, ctx.device, begin[0], end[0]) || !ggml_cuda_hc_up_range(dst, ctx.device, begin[1], end[1]) ||
            !ggml_cuda_hc_up_range(view, ctx.device, view_begin, view_end) || view_begin != begin[0] || view_end != end[0]) { return {}; }
    for (int j = 0; j < 2; ++j) {
        if (begin[j] % sizeof(float)) { return {}; }
        for (int k = 0; k < 3; ++k) {
            if (begin[j] < reads_end[k] && reads_begin[k] < end[j]) { return {}; }
        }
    }
    if (begin[0] < end[1] && begin[1] < end[0]) { return {}; }
    return match;
}

struct ggml_cuda_cublas_unary_match {
    ggml_tensor * mm = nullptr;
    ggml_cuda_scaled_unary_match unary;
    int count = 0;
};

static ggml_cuda_cublas_unary_match ggml_cuda_match_cublas_unary(ggml_backend_cuda_context & ctx, ggml_cgraph * graph, int i, bool allocated = true) {
    ggml_tensor * mm = graph->nodes[i];
    if (mm->op != GGML_OP_MUL_MAT || !mm->src[0] || !mm->src[1] || mm->type != GGML_TYPE_F32 ||
            mm->src[1]->type != GGML_TYPE_F32 || !ggml_is_contiguous(mm) || i + 1 >= graph->n_nodes || ggml_cuda_op_mul_mat_use_fwht(mm)) { return {}; }
    const auto unary = ggml_cuda_match_scaled_unary(graph, i + 1, true);
    if (!unary.count || unary.first->src[0] != mm || ggml_cuda_match_hc_injection(ctx, graph, i + 1, false).count) { return {}; }
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (ggml_cuda_mul_mat_kernel(mm->src[0], mm->src[1], mm, ctx.device) != GGML_CUDA_MM_CUBLAS) { return {}; }
    const ggml_type type = ggml_cuda_mul_mat_cublas_compute_type(cc, mm->src[0], mm->src[1], mm, false);
    if ((type != GGML_TYPE_F16 && type != GGML_TYPE_BF16) || ggml_cuda_cublas_prefer_f32_output(type, cc)) { return {}; }
    const int count = unary.count + 1;
    ggml_op ops[4]; int outputs[4];
    for (int j = 0; j < count; ++j) { ops[j] = graph->nodes[i + j]->op; outputs[j] = i + j; }
    if (!ggml_can_fuse_subgraph(graph, i, count, ops, outputs, count)) { return {}; }
    if (allocated) {
        if (!ggml_cuda_scaled_unary_memory_ok(unary, ctx.device)) { return {}; }
        uintptr_t read_begin[2], read_end[2];
        for (int j = 0; j < 2; ++j) {
            if (!ggml_cuda_hc_up_range(mm->src[j], ctx.device, read_begin[j], read_end[j])) { return {}; }
        }
        for (const ggml_tensor * tensor : {mm, unary.first, unary.unary, unary.last}) {
            uintptr_t begin, end;
            if (!(tensor->flags & GGML_TENSOR_FLAG_COMPUTE) || !ggml_cuda_hc_up_range(tensor, ctx.device, begin, end)) { return {}; }
            for (int j = 0; j < 2; ++j) { if (begin < read_end[j] && read_begin[j] < end) { return {}; } }
        }
    }
    return {mm, unary, count};
}

static bool ggml_cuda_affine_unary_bytes(const ggml_tensor * tensor, size_t & bytes) {
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (tensor->ne[d] <= 0) { return false; }
    }
    const size_t block = ggml_blck_size(tensor->type);
    bytes = ggml_type_size(tensor->type);
    int first = 0;
    if (block > 1) {
        if (!tensor->nb[0] || uint64_t(tensor->ne[0]) > SIZE_MAX/tensor->nb[0] || tensor->ne[0] % block) {
            return false;
        }
        bytes = size_t(tensor->ne[0])*tensor->nb[0]/block;
        first = 1;
    }
    for (int d = first; d < GGML_MAX_DIMS; ++d) {
        const uint64_t count = tensor->ne[d] - 1;
        if (count && tensor->nb[d] > (SIZE_MAX - bytes)/count) { return false; }
        bytes += size_t(count)*tensor->nb[d];
    }
    return bytes && bytes == ggml_nbytes(tensor);
}

static bool ggml_cuda_affine_unary_range(const ggml_tensor * tensor, int device, uintptr_t & begin, uintptr_t & end) {
    if (!tensor->buffer || !tensor->data || tensor->buffer->buft != ggml_backend_cuda_buffer_type(device)) {
        return false;
    }
    const uintptr_t base = (uintptr_t) ggml_backend_buffer_get_base(tensor->buffer);
    const size_t size = ggml_backend_buffer_get_size(tensor->buffer);
    size_t bytes;
    if (!ggml_cuda_affine_unary_bytes(tensor, bytes)) { return false; }
    begin = (uintptr_t) tensor->data;
    if (begin < base || begin - base > size || bytes > size - (begin - base) || begin > UINTPTR_MAX - bytes) {
        return false;
    }
    if (tensor->view_src) {
        const ggml_tensor * root = tensor->view_src;
        const uintptr_t root_begin = (uintptr_t) root->data;
        size_t root_bytes;
        if (!ggml_cuda_affine_unary_bytes(root, root_bytes)) { return false; }
        if (root->view_src || root->buffer != tensor->buffer || !root->data || root_begin < base || root_begin - base > size ||
                root_bytes > size - (root_begin - base) || tensor->view_offs > root_bytes || bytes > root_bytes - tensor->view_offs ||
                root_begin > UINTPTR_MAX - tensor->view_offs || begin != root_begin + tensor->view_offs) {
            return false;
        }
    }
    end = begin + bytes;
    return true;
}

struct ggml_cuda_affine_unary_match {
    ggml_tensor * mul = nullptr;
    ggml_tensor * add = nullptr;
    ggml_tensor * unary = nullptr;
    ggml_tensor * post = nullptr;
    ggml_tensor * tail = nullptr;
    int count = 0;
};

static bool ggml_cuda_affine_unary_binary_type(const ggml_tensor * node) {
    if (!node->src[0] || !node->src[1]) { return false; }
    const ggml_type a = node->src[0]->type;
    const ggml_type b = node->src[1]->type;
    return (a == GGML_TYPE_F32 && b == GGML_TYPE_F32 && node->type == GGML_TYPE_F32) ||
           (a == GGML_TYPE_F16 && b == GGML_TYPE_F16 && node->type == GGML_TYPE_F16) ||
           (a == GGML_TYPE_F16 && b == GGML_TYPE_F32 && (node->type == GGML_TYPE_F16 || node->type == GGML_TYPE_F32)) ||
           (a == GGML_TYPE_BF16 && (b == GGML_TYPE_F32 || b == GGML_TYPE_BF16) && node->type == GGML_TYPE_BF16);
}

static bool ggml_cuda_affine_unary_view_op(ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

static bool ggml_cuda_affine_unary_prior_view(ggml_cgraph * graph, int start, const ggml_tensor * tensor, bool metadata = false) {
    const ggml_tensor * root = tensor->view_src;
    size_t bytes, root_bytes;
    if ((tensor->op != GGML_OP_VIEW && (!metadata || !ggml_cuda_affine_unary_view_op(tensor->op))) || !root || root->view_src ||
            !ggml_cuda_affine_unary_bytes(tensor, bytes) || !ggml_cuda_affine_unary_bytes(root, root_bytes) ||
            tensor->view_offs > root_bytes || bytes > root_bytes - tensor->view_offs) {
        return false;
    }
    if (root->op == GGML_OP_NONE) { return true; }
    for (int i = 0; i < start; ++i) {
        if (graph->nodes[i] == root) { return true; }
    }
    return false;
}

static ggml_cuda_affine_unary_match ggml_cuda_match_affine_unary(ggml_backend_cuda_context & ctx, ggml_cgraph * graph, int i, bool allocated = true, bool with_tail = false) {
    ggml_cuda_affine_unary_match match;
    ggml_tensor * mul = graph->nodes[i];
    if (mul->op != GGML_OP_MUL || !ggml_cuda_affine_unary_binary_type(mul)) { return match; }
    const auto next_compute = [&](int index) {
        while (index < graph->n_nodes && graph->nodes[index]->op == GGML_OP_VIEW) {
            if (!ggml_cuda_affine_unary_prior_view(graph, i, graph->nodes[index])) { return graph->n_nodes; }
            uintptr_t begin, end;
            if (allocated && !ggml_cuda_affine_unary_range(graph->nodes[index], ctx.device, begin, end)) { return graph->n_nodes; }
            ++index;
        }
        return index;
    };
    const int a = next_compute(i + 1);
    if (a >= graph->n_nodes || graph->nodes[a]->op != GGML_OP_ADD || graph->nodes[a]->src[0] != mul ||
            !ggml_cuda_affine_unary_binary_type(graph->nodes[a])) { return match; }
    ggml_tensor * add = graph->nodes[a];
    const int u = next_compute(a + 1);
    if (u >= graph->n_nodes || graph->nodes[u]->op != GGML_OP_UNARY || graph->nodes[u]->src[0] != add ||
            graph->nodes[u]->type != add->type) { return match; }
    ggml_tensor * unary = graph->nodes[u];
    const ggml_unary_op op = ggml_get_unary_op(unary);
    if ((op != GGML_UNARY_OP_SIGMOID && op != GGML_UNARY_OP_SILU) ||
            ggml_cuda_can_fuse(graph, u, { GGML_OP_UNARY, GGML_OP_MUL }, { op })) { return match; }
    ggml_cuda_topk_moe_args topk{};
    if (op == GGML_UNARY_OP_SIGMOID && ggml_cuda_topk_moe_fusion(graph, u, topk)) { return match; }
    const int p = next_compute(u + 1);
    ggml_tensor * post = p < graph->n_nodes && graph->nodes[p]->op == GGML_OP_SCALE && graph->nodes[p]->src[0] == unary ? graph->nodes[p] : nullptr;
    if (post && ((post->type != GGML_TYPE_F32 && post->type != GGML_TYPE_BF16) || post->type != unary->type)) { return match; }
    const int q = with_tail && post ? next_compute(p + 1) : graph->n_nodes;
    ggml_tensor * tail = q < graph->n_nodes && graph->nodes[q]->op == GGML_OP_SCALE && graph->nodes[q]->src[0] == post ? graph->nodes[q] : nullptr;
    if (tail && (tail->type != post->type || !ggml_is_contiguous(tail) || !ggml_are_same_shape(mul, tail))) { tail = nullptr; }
    const int indices[] = { i, a, u, p, q };
    const ggml_op ops[] = { GGML_OP_MUL, GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_SCALE };
    const int n = tail ? 5 : post ? 4 : 3;
    if (!ggml_can_fuse_subgraph_ext(graph, indices, n, ops, indices, n)) { return match; }
    int64_t elements = 1;
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (mul->ne[d] <= 0 || mul->ne[d] > (INT_MAX - CUDA_NEG_BLOCK_SIZE)/elements) { return {}; }
        elements *= mul->ne[d];
    }
    const ggml_tensor * reads[] = { mul->src[0], mul->src[1], add->src[1] };
    uintptr_t read_begin[3], read_end[3], begin[5], end[5];
    for (int j = 0; j < 3; ++j) {
        const ggml_tensor * tensor = reads[j];
        size_t bytes;
        if (!ggml_cuda_affine_unary_bytes(tensor, bytes) ||
                (allocated && (!ggml_cuda_affine_unary_range(tensor, ctx.device, read_begin[j], read_end[j]) ||
                 read_begin[j] % ggml_type_size(tensor->type)))) { return {}; }
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            if (tensor->ne[d] > INT_MAX || tensor->nb[d] % ggml_type_size(tensor->type) ||
                    tensor->nb[d]/ggml_type_size(tensor->type) > UINT32_MAX ||
                    mul->ne[d] % tensor->ne[d]) { return {}; }
        }
        for (int k = 0; k < n; ++k) {
            if (tensor == graph->nodes[indices[k]] || tensor->view_src == graph->nodes[indices[k]]) { return {}; }
        }
    }
    if (!ggml_are_same_shape(reads[0], mul)) { return {}; }
    for (int j = 0; j < n; ++j) {
        const ggml_tensor * tensor = graph->nodes[indices[j]];
        size_t bytes;
        if (!ggml_cuda_affine_unary_bytes(tensor, bytes) || !ggml_is_contiguous(tensor) || !ggml_are_same_shape(mul, tensor) ||
                (allocated && (!ggml_cuda_affine_unary_range(tensor, ctx.device, begin[j], end[j]) ||
                 begin[j] % ggml_type_size(tensor->type)))) { return {}; }
        if (!allocated) { continue; }
        for (int k = 0; k < 3; ++k) {
            if (begin[j] < read_end[k] && read_begin[k] < end[j] &&
                    (k != 0 || begin[j] != read_begin[k] || end[j] != read_end[k] || tensor->type != reads[0]->type || !ggml_is_contiguous(reads[0]))) {
                return {};
            }
        }
        for (int k = 0; k < j; ++k) {
            if (begin[j] < end[k] && begin[k] < end[j] &&
                    (begin[j] != begin[k] || end[j] != end[k] || tensor->type != graph->nodes[indices[k]]->type)) { return {}; }
        }
    }
    match.mul = mul;
    match.add = add;
    match.unary = unary;
    match.post = post;
    match.tail = tail;
    match.count = indices[n - 1] - i + 1;
    return match;
}

static ggml_cuda_affine_unary_match ggml_cuda_match_mul_add(ggml_backend_cuda_context & ctx, ggml_cgraph * graph, int i, bool allocated = true, bool metadata = false) {
    ggml_tensor * mul = graph->nodes[i];
    if (mul->op != GGML_OP_MUL || !ggml_cuda_affine_unary_binary_type(mul)) { return {}; }
    int a = i + 1;
    while (a < graph->n_nodes && (metadata ? ggml_cuda_affine_unary_view_op(graph->nodes[a]->op) : graph->nodes[a]->op == GGML_OP_VIEW)) {
        uintptr_t begin, end;
        if (!ggml_cuda_affine_unary_prior_view(graph, i, graph->nodes[a], metadata) ||
                (allocated && !ggml_cuda_affine_unary_range(graph->nodes[a], ctx.device, begin, end))) { return {}; }
        ++a;
    }
    if (a >= graph->n_nodes || graph->nodes[a]->op != GGML_OP_ADD || !ggml_cuda_affine_unary_binary_type(graph->nodes[a])) { return {}; }
    ggml_tensor * add = graph->nodes[a];
    if ((add->src[0] == mul) == (add->src[1] == mul)) { return {}; }
    if (a + 1 < graph->n_nodes && ggml_can_fuse(graph, a, {GGML_OP_ADD, GGML_OP_ADD}) &&
            graph->nodes[a + 1]->src[0] == add && ggml_are_same_layout(add->src[1], graph->nodes[a + 1]->src[1])) { return {}; }
    const int indices[] = {i, a};
    const ggml_op ops[] = {GGML_OP_MUL, GGML_OP_ADD};
    if (!ggml_can_fuse_subgraph_ext(graph, indices, 2, ops, indices, 2)) { return {}; }
    int64_t elements = 1;
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (mul->ne[d] <= 0 || mul->ne[d] > (INT_MAX - CUDA_NEG_BLOCK_SIZE)/elements) { return {}; }
        elements *= mul->ne[d];
    }
    const ggml_tensor * reads[] = {mul->src[0], mul->src[1], add->src[add->src[0] == mul ? 1 : 0]};
    uintptr_t rb[3], re[3], wb[2], we[2];
    for (int j = 0; j < 3; ++j) {
        const ggml_tensor * tensor = reads[j];
        size_t bytes;
        if (!ggml_cuda_affine_unary_bytes(tensor, bytes) || (allocated &&
                (!ggml_cuda_affine_unary_range(tensor, ctx.device, rb[j], re[j]) || rb[j] % ggml_type_size(tensor->type)))) { return {}; }
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            if (tensor->ne[d] > INT_MAX || tensor->nb[d] % ggml_type_size(tensor->type) ||
                    tensor->nb[d]/ggml_type_size(tensor->type) > UINT32_MAX || mul->ne[d] % tensor->ne[d]) { return {}; }
        }
        if (tensor == mul || tensor == add || tensor->view_src == mul || tensor->view_src == add) { return {}; }
    }
    if (!ggml_are_same_shape(reads[0], mul)) { return {}; }
    for (int j = 0; j < 2; ++j) {
        const ggml_tensor * tensor = graph->nodes[indices[j]];
        size_t bytes;
        if (!ggml_cuda_affine_unary_bytes(tensor, bytes) || !ggml_is_contiguous(tensor) || !ggml_are_same_shape(mul, tensor) || (allocated &&
                (!ggml_cuda_affine_unary_range(tensor, ctx.device, wb[j], we[j]) || wb[j] % ggml_type_size(tensor->type)))) { return {}; }
        if (!allocated) { continue; }
        for (int k = 0; k < 3; ++k) {
            if (wb[j] < re[k] && rb[k] < we[j] && (k != 0 || wb[j] != rb[k] || we[j] != re[k] ||
                    tensor->type != reads[0]->type || !ggml_is_contiguous(reads[0]))) { return {}; }
        }
        if (j && wb[j] < we[0] && wb[0] < we[j] && (wb[j] != wb[0] || we[j] != we[0] || tensor->type != mul->type)) { return {}; }
    }
    ggml_cuda_affine_unary_match match;
    match.mul = mul;
    match.add = add;
    match.count = a - i + 1;
    return match;
}

struct ggml_cuda_hc_affine_injection_match {
    ggml_cuda_affine_unary_match affine;
    ggml_tensor * post = nullptr;
    ggml_tensor * comb = nullptr;
    ggml_cuda_hc_post_norm_match hc;
    int count = 0;
};

static bool ggml_cuda_hc_affine_tensor_ok(const ggml_tensor * tensor) {
    if (!tensor || tensor->type != GGML_TYPE_F32 || (tensor->view_src && tensor->view_src->type != GGML_TYPE_F32)) { return false; }
    size_t elements = 1;
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (tensor->ne[d] <= 0 || tensor->ne[d] > INT_MAX || tensor->nb[d] % sizeof(float) || elements > INT64_MAX/sizeof(float)/size_t(tensor->ne[d])) { return false; }
        elements *= tensor->ne[d];
    }
    size_t bytes;
    return ggml_cuda_affine_unary_bytes(tensor, bytes) && bytes <= INT64_MAX;
}

static bool ggml_cuda_hc_affine_prior_source(ggml_cgraph * graph, int start, const ggml_tensor * tensor) {
    if (!tensor) { return false; }
    if (tensor->view_src) { return ggml_cuda_affine_unary_prior_view(graph, start, tensor); }
    if (tensor->op == GGML_OP_NONE) { return true; }
    for (int i = 0; i < start; ++i) {
        if (graph->nodes[i] == tensor) { return true; }
    }
    return false;
}

static ggml_cuda_hc_affine_injection_match ggml_cuda_match_hc_affine_injection(ggml_backend_cuda_context & ctx, ggml_cgraph * graph, int i, bool allocated = true) {
    auto affine = ggml_cuda_match_affine_unary(ctx, graph, i, allocated);
    if (!affine.count || affine.mul->type != GGML_TYPE_F32 || affine.add->type != GGML_TYPE_F32 || affine.unary->type != GGML_TYPE_F32) { return {}; }
    ggml_tensor * last = affine.post ? affine.post : affine.unary;
    const ggml_tensor * gate_inputs[] = {affine.mul->src[0], affine.mul->src[1], affine.add->src[1]};
    for (const ggml_tensor * tensor : gate_inputs) {
        if (!ggml_cuda_hc_affine_tensor_ok(tensor) || tensor->ne[2] != 1 || tensor->ne[3] != 1 || affine.mul->ne[0] % tensor->ne[0] || affine.mul->ne[1] % tensor->ne[1]) { return {}; }
    }
    int post_i = i + affine.count;
    auto skip_views = [&]() {
        while (post_i < graph->n_nodes && graph->nodes[post_i]->op == GGML_OP_VIEW) {
            if (!ggml_cuda_affine_unary_prior_view(graph, i, graph->nodes[post_i])) { return false; }
            ++post_i;
        }
        return true;
    };
    if (!skip_views()) { return {}; }
    ggml_tensor * comb = nullptr;
    if (post_i < graph->n_nodes && graph->nodes[post_i]->op == GGML_OP_DSV4_HC_COMB) {
        comb = graph->nodes[post_i++];
        // The existing COMB operator supports four streams.
        if (comb->type != GGML_TYPE_F32 || comb->ne[0] != 4 || comb->ne[1] != 4 || comb->ne[3] != 1) { return {}; }
        const int64_t mix_dim = (comb->ne[0] + 2)*comb->ne[0];
        const ggml_tensor * mixes = comb->src[0];
        const ggml_tensor * scale = comb->src[1];
        const ggml_tensor * base = comb->src[2];
        if (!mixes || !scale || !base || mixes->ne[0] != mix_dim || mixes->ne[1] != comb->ne[2] || mixes->ne[2] != 1 || mixes->ne[3] != 1 ||
                scale->ne[0] < 3 || scale->ne[1] != 1 || scale->ne[2] != 1 || scale->ne[3] != 1 ||
                base->ne[0] != mix_dim || base->ne[1] != 1 || base->ne[2] != 1 || base->ne[3] != 1 || comb->op_params[1] <= 0) { return {}; }
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (comb->src[j] && (!ggml_cuda_hc_affine_tensor_ok(comb->src[j]) || !ggml_cuda_hc_affine_prior_source(graph, i, comb->src[j]))) { return {}; }
        }
        if (!skip_views()) { return {}; }
    }
    if (post_i >= graph->n_nodes) { return {}; }
    ggml_tensor * post = graph->nodes[post_i];
    const ggml_tensor * x = post->src[0];
    const ggml_tensor * residual = post->src[1];
    const ggml_tensor * post_comb = post->src[3];
    if (post->op != GGML_OP_DSV4_HC_POST || post->src[2] != last || !x || !residual ||
            x->type != GGML_TYPE_F32 || residual->type != GGML_TYPE_F32 || post->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(post) || !ggml_are_same_shape(post, residual) || x->ne[0] <= 0 || x->ne[1] <= 0 || residual->ne[1] <= 0 ||
            x->ne[2] != 1 || x->ne[3] != 1 || residual->ne[0] != x->ne[0] || residual->ne[2] != x->ne[1] || residual->ne[3] != 1 ||
            last->ne[0] != residual->ne[1] || last->ne[1] != x->ne[1] || last->ne[2] != 1 || last->ne[3] != 1 ||
            (comb && post_comb != comb) ||
            (post_comb && (post_comb->type != GGML_TYPE_F32 || post_comb->ne[0] != residual->ne[1] || post_comb->ne[1] != residual->ne[1] || post_comb->ne[2] != x->ne[1] || post_comb->ne[3] != 1))) { return {}; }
    for (const ggml_tensor * tensor : {x, residual, (const ggml_tensor *) post, post_comb}) {
        if (tensor && !ggml_cuda_hc_affine_tensor_ok(tensor)) { return {}; }
    }
    size_t bytes;
    if (!ggml_cuda_hc_injection_bytes(post, bytes) || bytes/sizeof(float) > INT_MAX) { return {}; }
    auto hc = ggml_cuda_match_hc_post_norm(graph, post_i);
    if (hc.count && graph->nodes[post_i + hc.count - 1] != (hc.scale ? hc.scale : hc.mul ? hc.mul : hc.norm)) { return {}; }
    if (hc.count && hc.norm->ne[0] != x->ne[0]) {
        if (post_comb && bytes/sizeof(float) <= 256) { return {}; }
        if (hc.norm->ne[0] < x->ne[0] && hc.norm->ne[0] > WARP_SIZE/2 && hc.norm->ne[0] < WARP_SIZE) { return {}; }
        const int block_size = hc.norm->ne[0] < 1024 ? 256 : 1024;
        const int64_t nweights = (hc.norm->ne[0] + x->ne[0] - 2)/x->ne[0] + 1;
        if ((hc.norm->ne[0] < x->ne[0] && hc.norm->ne[0] <= WARP_SIZE/2) || nweights >= block_size) { hc = {}; }
    }
    const int end = hc.count ? post_i + hc.count : post_i + 1;
    const int64_t ncols = hc.count ? hc.norm->ne[0] : 256;
    const size_t shared = ((hc.count ? 32 : 0) + (ncols + x->ne[0] - 2)/x->ne[0] + 1)*sizeof(float);
    if (shared > ggml_cuda_info().devices[ctx.device].smpb) { return {}; }
    std::vector<ggml_op> ops;
    std::vector<int> indices;
    std::vector<int> outputs;
    for (int j = i; j < end; ++j) {
        ops.push_back(graph->nodes[j]->op);
        indices.push_back(j);
        if ((!hc.mul && !hc.scale) || graph->nodes[j] != hc.norm) { outputs.push_back(j); }
    }
    if (!ggml_can_fuse_subgraph_ext(graph, indices.data(), indices.size(), ops.data(), outputs.data(), outputs.size())) { return {}; }
    if (allocated) {
        if (hc.count && !ggml_cuda_hc_post_norm_memory_ok(graph, post_i, hc, ctx.device)) { return {}; }
        std::vector<const ggml_tensor *> reads = {gate_inputs[0], gate_inputs[1], gate_inputs[2], x, residual, post_comb};
        if (hc.mul) { reads.push_back(hc.mul->src[hc.mul->src[0] == hc.norm ? 1 : 0]); }
        if (comb) {
            for (int j = 0; j < GGML_MAX_SRC; ++j) {
                if (comb->src[j]) { reads.push_back(comb->src[j]); }
            }
        }
        std::vector<const ggml_tensor *> writes = {affine.mul, affine.add, affine.unary};
        if (affine.post) { writes.push_back(affine.post); }
        writes.push_back(post);
        if (hc.count) { writes.push_back(hc.scale ? hc.scale : hc.mul ? hc.mul : hc.norm); }
        if (comb) { writes.push_back(comb); }
        std::vector<uintptr_t> rb(reads.size()), re(reads.size()), wb(writes.size()), we(writes.size());
        for (size_t j = 0; j < reads.size(); ++j) {
            if (reads[j] && (!ggml_cuda_hc_affine_tensor_ok(reads[j]) || !ggml_cuda_hc_injection_range(reads[j], ctx.device, rb[j], re[j]))) { return {}; }
        }
        for (size_t j = 0; j < writes.size(); ++j) {
            if (!(writes[j]->flags & GGML_TENSOR_FLAG_COMPUTE) || !ggml_is_contiguous(writes[j]) || !ggml_cuda_hc_affine_tensor_ok(writes[j]) || !ggml_cuda_hc_injection_range(writes[j], ctx.device, wb[j], we[j])) { return {}; }
            for (size_t k = 0; k < reads.size(); ++k) {
                // COMB is written before the affine kernel reads it.
                if (reads[k] && !(writes[j] == comb && reads[k] == comb) && wb[j] < re[k] && rb[k] < we[j]) { return {}; }
            }
            for (size_t k = 0; k < j; ++k) {
                if (wb[j] < we[k] && wb[k] < we[j]) { return {}; }
            }
        }
        for (int j = i; j < end; ++j) {
            const ggml_tensor * tensor = graph->nodes[j];
            if (tensor->op != GGML_OP_VIEW && tensor->op != GGML_OP_RESHAPE) { continue; }
            uintptr_t begin, stop;
            if (!ggml_cuda_hc_injection_range(tensor, ctx.device, begin, stop)) { return {}; }
        }
    }
    return {affine, post, comb, hc, end - i};
}

struct ggml_cuda_cublas_affine_match {
    ggml_tensor * mm = nullptr;
    ggml_cuda_affine_unary_match affine;
    int count = 0;
};

static bool ggml_cuda_cublas_affine_input(const ggml_tensor * mm, const ggml_tensor * input) {
    if (input == mm) { return true; }
    if (!input || input->op != GGML_OP_VIEW || input->type != GGML_TYPE_F32 || input->view_src != mm ||
            input->view_offs % sizeof(float) || input->nb[0] != sizeof(float) || input->ne[0] <= 0 || input->ne[0] > mm->ne[0] ||
            input->view_offs/sizeof(float) > uint64_t(mm->ne[0] - input->ne[0])) { return false; }
    for (int d = 1; d < GGML_MAX_DIMS; ++d) {
        if (input->ne[d] != mm->ne[d] || input->nb[d] != mm->nb[d]) { return false; }
    }
    return true;
}

static ggml_cuda_cublas_affine_match ggml_cuda_match_cublas_affine(ggml_backend_cuda_context & ctx, ggml_cgraph * graph, int i, bool allocated = true, bool with_tail = true) {
    ggml_tensor * mm = graph->nodes[i];
    if (mm->op != GGML_OP_MUL_MAT || !mm->src[0] || !mm->src[1] || mm->type != GGML_TYPE_F32 || mm->src[1]->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(mm) || ggml_nelements(mm) <= 0 || ggml_nelements(mm) > INT_MAX - CUDA_NEG_BLOCK_SIZE || i + 1 >= graph->n_nodes || ggml_cuda_op_mul_mat_use_fwht(mm)) { return {}; }
    int first = i + 1;
    while (first < graph->n_nodes && graph->nodes[first]->op == GGML_OP_VIEW) {
        if (!ggml_cuda_affine_unary_prior_view(graph, i, graph->nodes[first]) && !ggml_cuda_cublas_affine_input(mm, graph->nodes[first])) { return {}; }
        uintptr_t begin, end;
        if (allocated && !ggml_cuda_affine_unary_range(graph->nodes[first], ctx.device, begin, end)) { return {}; }
        ++first;
    }
    if (first >= graph->n_nodes) { return {}; }
    auto affine = ggml_cuda_match_affine_unary(ctx, graph, first, allocated, with_tail);
    if (!affine.count) { affine = ggml_cuda_match_affine_unary(ctx, graph, first, allocated); }
    const auto reject = [&]() {
        return with_tail && affine.tail ? ggml_cuda_match_cublas_affine(ctx, graph, i, allocated, false) : ggml_cuda_cublas_affine_match{};
    };
    if (!affine.count || !ggml_cuda_cublas_affine_input(mm, affine.mul->src[0]) || ggml_cuda_match_hc_affine_injection(ctx, graph, first, false).count) { return reject(); }
    if (ggml_cuda_mul_mat_kernel(mm->src[0], mm->src[1], mm, ctx.device) != GGML_CUDA_MM_CUBLAS) { return reject(); }
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const ggml_type type = ggml_cuda_mul_mat_cublas_compute_type(cc, mm->src[0], mm->src[1], mm, false);
    if ((type != GGML_TYPE_F16 && type != GGML_TYPE_BF16) || ggml_cuda_cublas_prefer_f32_output(type, cc)) { return reject(); }
    const int count = first - i + affine.count;
    std::vector<ggml_op> ops;
    std::vector<int> outputs;
    for (int j = i; j < i + count; ++j) { ops.push_back(graph->nodes[j]->op); outputs.push_back(j); }
    if (!ggml_can_fuse_subgraph_ext(graph, outputs.data(), count, ops.data(), outputs.data(), outputs.size())) { return reject(); }
    if (allocated) {
        uintptr_t rb[4], re[4], wb[6], we[6];
        const ggml_tensor * reads[] = {mm->src[0], mm->src[1], affine.mul->src[1], affine.add->src[1]};
        const ggml_tensor * writes[] = {mm, affine.mul, affine.add, affine.unary, affine.post, affine.tail};
        for (int j = 0; j < 4; ++j) {
            if (j < 2 ? !ggml_cuda_hc_up_range(reads[j], ctx.device, rb[j], re[j]) : !ggml_cuda_affine_unary_range(reads[j], ctx.device, rb[j], re[j])) { return reject(); }
        }
        for (int j = 0; j < 6; ++j) {
            if (!writes[j]) { continue; }
            if (!(writes[j]->flags & GGML_TENSOR_FLAG_COMPUTE) || !ggml_cuda_affine_unary_range(writes[j], ctx.device, wb[j], we[j])) { return reject(); }
            for (int k = 0; k < 4; ++k) { if (wb[j] < re[k] && rb[k] < we[j]) { return reject(); } }
            for (int k = 0; k < j; ++k) {
                if (writes[k] && wb[j] < we[k] && wb[k] < we[j] &&
                        (wb[j] != wb[k] || we[j] != we[k] || writes[j]->type != writes[k]->type)) { return reject(); }
            }
        }
    }
    return {mm, affine, count};
}

struct ggml_cuda_repeat_mul_add_match {
    ggml_tensor * repeat = nullptr;
    ggml_cuda_affine_unary_match binary;
    int count = 0;
};

static ggml_cuda_repeat_mul_add_match ggml_cuda_match_repeat_mul_add(ggml_backend_cuda_context & ctx, ggml_cgraph * graph, int i, bool allocated = true) {
    ggml_tensor * repeat = graph->nodes[i];
    if (repeat->op != GGML_OP_REPEAT || !repeat->src[0] || repeat->type != repeat->src[0]->type ||
            (repeat->type != GGML_TYPE_F32 && repeat->type != GGML_TYPE_F16)) { return {}; }
    int m = i + 1;
    while (m < graph->n_nodes && ggml_cuda_affine_unary_view_op(graph->nodes[m]->op)) {
        uintptr_t begin, end;
        if (!ggml_cuda_affine_unary_prior_view(graph, i, graph->nodes[m], true) ||
                (allocated && !ggml_cuda_affine_unary_range(graph->nodes[m], ctx.device, begin, end))) { return {}; }
        ++m;
    }
    if (m >= graph->n_nodes || (graph->nodes[m]->src[0] == repeat) == (graph->nodes[m]->src[1] == repeat)) { return {}; }
    if (ggml_cuda_match_hc_affine_injection(ctx, graph, m, false).count || ggml_cuda_match_affine_unary(ctx, graph, m, false).count) { return {}; }
    const auto binary = ggml_cuda_match_mul_add(ctx, graph, m, allocated, true);
    if (!binary.count || !ggml_are_same_shape(repeat, binary.mul) || !ggml_is_contiguous(repeat)) { return {}; }
    const int indices[] = {i, m, m + binary.count - 1};
    const ggml_op ops[] = {GGML_OP_REPEAT, GGML_OP_MUL, GGML_OP_ADD};
    if (!ggml_can_fuse_subgraph_ext(graph, indices, 3, ops, indices, 3)) { return {}; }
    const ggml_tensor * reads[] = {repeat->src[0], binary.mul->src[binary.mul->src[0] == repeat ? 1 : 0], binary.add->src[binary.add->src[0] == binary.mul ? 1 : 0]};
    const ggml_tensor * writes[] = {repeat, binary.mul, binary.add};
    uintptr_t rb[3], re[3], wb[3], we[3];
    for (int j = 0; j < 3; ++j) {
        size_t bytes;
        if (!ggml_cuda_affine_unary_bytes(reads[j], bytes) || (allocated &&
                (!ggml_cuda_affine_unary_range(reads[j], ctx.device, rb[j], re[j]) || rb[j] % ggml_type_size(reads[j]->type)))) { return {}; }
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            if (reads[j]->ne[d] > INT_MAX || reads[j]->nb[d] % ggml_type_size(reads[j]->type) ||
                    reads[j]->nb[d]/ggml_type_size(reads[j]->type) > UINT32_MAX || binary.mul->ne[d] % reads[j]->ne[d]) { return {}; }
        }
        for (const ggml_tensor * write : writes) {
            if (reads[j] == write || reads[j]->view_src == write) { return {}; }
        }
    }
    if (allocated) {
        for (int j = 0; j < 3; ++j) {
            if (!ggml_cuda_affine_unary_range(writes[j], ctx.device, wb[j], we[j]) || wb[j] % ggml_type_size(writes[j]->type)) { return {}; }
            for (int k = 0; k < 3; ++k) { if (wb[j] < re[k] && rb[k] < we[j]) { return {}; } }
            for (int k = 0; k < j; ++k) {
                if (wb[j] < we[k] && wb[k] < we[j] && (wb[j] != wb[k] || we[j] != we[k] || writes[j]->type != writes[k]->type)) { return {}; }
            }
        }
    }
    return {repeat, binary, indices[2] - i + 1};
}

static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i,
        const ggml_tensor * shared_input, const char * quantized, int prepared_hc_post = -1,
        const ggml_cuda_hc_affine_injection_match * affine_match = nullptr, bool * affine_emit = nullptr, const void * prepared_src1 = nullptr) {

    static bool disable_fusion = getenv("GGML_CUDA_DISABLE_FUSION") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_FUSION"));
    if (disable_fusion) {
        return 0;
    }

    const auto shared_quantized = [&](const ggml_tensor * mm_node) {
        return quantized && mm_node->src[1] == shared_input && ggml_cuda_can_share_mmvq_input(mm_node, cuda_ctx->device) &&
            !ggml_cuda_mmvq_input_overwritten(mm_node, shared_input) ? quantized : nullptr;
    };
    ggml_tensor * node = cgraph->nodes[i];

    if (node->op == GGML_OP_MUL) {
        ggml_cuda_moe_weighted_reduction_match match;
        if (ggml_cuda_match_moe_weighted_reduction(cgraph, i, match)) {
            const int output_idx = i + match.node_count - 1;
            if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, match.node_count, &output_idx, 1)) {
                ggml_cuda_op_moe_weighted_reduction(
                    *cuda_ctx, match.experts, match.expert_scale, match.weights, match.dst);
                return match.node_count - 1;
            }
        }
    }

    // gated_delta_net -> cpy: scatter recurrent-state snapshots into the cache
    if (node->op == GGML_OP_GATED_DELTA_NET) {
        ggml_cuda_gated_delta_net_fused_cache fused_state_cpy;
        const int nodes_to_skip = ggml_cuda_try_gdn_cache_fusion(cgraph, i, fused_state_cpy);
        if (nodes_to_skip > 0) {
#ifdef GGML_CUDA_DEBUG
            GGML_LOG_INFO("%s: fused gated_delta_net snapshot copies for %s (skipped %d nodes)\n",
                          __func__, node->name, nodes_to_skip);
#endif
            ggml_cuda_op_gated_delta_net_fused_cache(*cuda_ctx, node, fused_state_cpy);
            return nodes_to_skip;
        }
    }

    //topk-moe
    if (cgraph->nodes[i]->op == GGML_OP_UNARY || cgraph->nodes[i]->op == GGML_OP_SOFT_MAX ||
            cgraph->nodes[i]->op == GGML_OP_ARGSORT) {
        ggml_cuda_topk_moe_args args;
        const bool              can_fuse = ggml_cuda_topk_moe_fusion(cgraph, i, args);
        std::vector<ggml_op>    ops;
        ops.reserve(13);  // max ops; avoids gcc -Wstringop-overflow false positive

        if (can_fuse) {
            const ggml_tensor * logits  = node->src[0];
            ggml_tensor *       weights = nullptr;
            ggml_tensor *       ids     = nullptr;
            const ggml_tensor * bias    = nullptr;
            const ggml_tensor * clamp   = nullptr;
            const ggml_tensor * scale   = nullptr;

            if (!args.delayed_softmax) {
                int out_nodes[2];  // nodes which can't be elided

                if (args.sigmoid) {
                    ops.insert(ops.end(), { GGML_OP_UNARY });
                } else if (args.sqrt_softplus) {
                    ops.insert(ops.end(), { GGML_OP_UNARY, GGML_OP_SQRT });
                } else {
                    ops.insert(ops.end(), { GGML_OP_SOFT_MAX });
                }
                const int i_probs = i + (int) ops.size() - 1;  // last node of the gating activation

                if (args.prob_bias) {
                    bias = cgraph->nodes[i_probs + 2]->src[1];
                    ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW,
                                            GGML_OP_GET_ROWS });
                    out_nodes[0] = i_probs + 4;
                } else {
                    ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
                    out_nodes[0] = i_probs + 3;
                }
                ids = cgraph->nodes[out_nodes[0]];

                if (args.norm) {
                    ops.insert(ops.end(),
                               { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
                    clamp = cgraph->nodes[i + ops.size() - 3];
                }
                if (args.scale) {
                    ops.insert(ops.end(), { GGML_OP_SCALE });
                    scale = cgraph->nodes[i + ops.size() - 1];
                }

                weights      = cgraph->nodes[i + ops.size() - 1];
                out_nodes[1] = i + ops.size() - 1;

                if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                        ggml_cuda_should_use_topk_moe(node, logits, weights, ids) &&
                        ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, clamp, scale, bias, args);
                    return ops.size() - 1;
                }
            } else if (!args.norm && !args.prob_bias) {
                //special case gpt-oss, no norm, no bias.
                ops.insert(ops.end(), { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_RESHAPE,
                                        GGML_OP_SOFT_MAX, GGML_OP_RESHAPE });
                weights                     = cgraph->nodes[i + 5];
                ids                         = cgraph->nodes[i + 1];
                const ggml_tensor * softmax = cgraph->nodes[i + 4];

                int out_nodes[2] = { i + 1, i + 5 };
                if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                        ggml_cuda_should_use_topk_moe(softmax, logits, weights, ids) &&
                        ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, clamp, scale, bias, args);
                    return ops.size() - 1;
                }
            }
        }
    }

    //RoPE + view + set-rows
    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        ggml_tensor * rope     = cgraph->nodes[i];
        ggml_tensor * set_rows = cgraph->nodes[i + 2];

        ggml_cuda_op_rope_fused(*cuda_ctx, rope, set_rows);
        return 2;
    }

    // Snake activation: y = x + sin(a*x)^2 * inv_b
    // Naive 5-op decomposition emitted by frontends: mul -> sin -> sqr -> mul -> add
    if (ggml_can_fuse_subgraph(cgraph, i,
            { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ADD },
            { i + 4 })) {
        const ggml_tensor * mul0 = cgraph->nodes[i];
        const ggml_tensor * sqr  = cgraph->nodes[i + 2];
        const ggml_tensor * mul1 = cgraph->nodes[i + 3];
        ggml_tensor *       add  = cgraph->nodes[i + 4];

        // x carries the full activation shape, a is the broadcast operand
        const ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
        const ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

        // mul1 reads sqr and inv_b in either operand order
        const ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

        // closure check: the trailing add must read the same x as the leading mul
        const ggml_tensor * x_in_add = (add->src[0] == mul1) ? add->src[1] : add->src[0];

        // Kernel iterates over total = T * C, so x and add must be 2D and
        // a / inv_b must collapse to [1, C, 1, 1]. Higher dims are not handled.
        const bool dim_ok   = (x->ne[2]   == 1 && x->ne[3]   == 1) &&
                              (add->ne[2] == 1 && add->ne[3] == 1) &&
                              (a->ne[2]   == 1 && a->ne[3]   == 1);
        const bool shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];

        // x is in the supported whitelist and every chain intermediate shares
        // x's type. launch_snake reads a and inv_b as const float *, so they
        // stay F32.
        const ggml_tensor * sin1 = cgraph->nodes[i + 1];
        const bool types_ok = (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
                              (a->type    == GGML_TYPE_F32) && (inv_b->type == GGML_TYPE_F32) &&
                              (mul0->type == x->type) && (sin1->type  == x->type) &&
                              (sqr->type  == x->type) && (mul1->type  == x->type) &&
                              (add->type  == x->type);

        // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
        const bool contig_ok = ggml_is_contiguous(x) && ggml_is_contiguous(add) &&
                               ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

        if (types_ok && shape_ok && dim_ok && contig_ok && x_in_add == x) {
            ggml_cuda_op_snake_fused(*cuda_ctx, x, a, inv_b, add);
            return 4;
        }
    }

    // multi-(add or mul)
    if (node->op == GGML_OP_ADD || node->op == GGML_OP_MUL) {
        int     n_fuse = 0;
        ggml_op ops[8];
        std::fill(ops, ops + 8, node->op);

        for (; n_fuse <= 6; ++n_fuse) {
            if (!ggml_can_fuse(cgraph, i + n_fuse, ops + n_fuse, 2)) {
                break;
            }
            if (cgraph->nodes[i + n_fuse] != cgraph->nodes[i + n_fuse + 1]->src[0]) {
                break;
            }
            if (!ggml_are_same_layout(cgraph->nodes[i + n_fuse]->src[1], cgraph->nodes[i + n_fuse + 1]->src[1])) {
                break;
            }
        }

        n_fuse++;

        if (n_fuse > 1) {
            ggml_tensor fused_node;
            memcpy(&fused_node, node, sizeof(ggml_tensor));
            for (int j = 0; j < n_fuse - 1; ++j) {
                fused_node.src[j + 2] = cgraph->nodes[i + j + 1]->src[1];
            }
            fused_node.data = cgraph->nodes[i + n_fuse - 1]->data;
            if (node->op == GGML_OP_ADD) {
                ggml_cuda_op_fused_add(*cuda_ctx, &fused_node, n_fuse);
            } else {
                ggml_cuda_op_fused_mul(*cuda_ctx, &fused_node, n_fuse);
            }
            return n_fuse - 1;
        }
    }

    bool fused_mul_mat_vec = false;
    int  fused_node_count  = 0;

    auto get_mul_mat_scale = [](const ggml_tensor * scale_node, const ggml_tensor * mm_node) -> const ggml_tensor * {
        const bool scale_lhs_mm = scale_node->src[0] == mm_node;
        const bool scale_rhs_mm = scale_node->src[1] == mm_node;
        if (!scale_lhs_mm && !scale_rhs_mm) {
            return nullptr;
        }

        const ggml_tensor * scale = scale_lhs_mm ? scale_node->src[1] : scale_node->src[0];
        if (mm_node->src[0]->type != GGML_TYPE_NVFP4 || scale_node->type != GGML_TYPE_F32 ||
                scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_nelements(scale) != 1 ||
                !ggml_are_same_shape(scale_node, mm_node)) {
            return nullptr;
        }

        return scale;
    };

    auto get_mul_mat_id_scale = [](const ggml_tensor * reshape, const ggml_tensor * repeat, const ggml_tensor * getrows,
            const ggml_tensor * scale_node, const ggml_tensor * mm_node) -> const ggml_tensor * {
        if (repeat->src[0] != reshape || getrows->src[0] != repeat || getrows->src[1] != mm_node->src[2]) {
            return nullptr;
        }
        if (!((scale_node->src[0] == mm_node && scale_node->src[1] == getrows) ||
                (scale_node->src[0] == getrows && scale_node->src[1] == mm_node))) {
            return nullptr;
        }

        const ggml_tensor * scale = reshape->src[0];
        if (mm_node->src[0]->type != GGML_TYPE_NVFP4 || scale_node->type != GGML_TYPE_F32 ||
                scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_nelements(scale) != mm_node->src[0]->ne[2] ||
                !ggml_are_same_shape(scale_node, mm_node)) {
            return nullptr;
        }

        return scale;
    };

    auto get_bias_tensor = [](const ggml_tensor * bias_node, const ggml_tensor * mul_node, ggml_op op_bias) -> const ggml_tensor * {
        if (op_bias == GGML_OP_ADD) {
            if (bias_node->src[0] == mul_node) {
                return bias_node->src[1];
            }
            if (bias_node->src[1] == mul_node) {
                return bias_node->src[0];
            }
            return nullptr;
        }
        GGML_ASSERT(op_bias == GGML_OP_ADD_ID);
        GGML_ASSERT(bias_node->src[0] == mul_node);
        return bias_node->src[1];
    };

    // gate + glu + up, with optional scale/bias on both lanes.
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        if (op == GGML_OP_MUL_MAT) {
            for (const bool with_bias : { false, true }) {
                const int gate_idx       = i;
                const int gate_scale_idx = i + 1;
                const int gate_bias_idx  = with_bias ? i + 2 : -1;
                const int up_idx         = with_bias ? i + 3 : i + 2;
                const int up_scale_idx   = up_idx + 1;
                const int up_bias_idx    = with_bias ? up_idx + 2 : -1;
                const int glu_idx        = with_bias ? up_idx + 3 : up_idx + 2;

                const int out_nodes[] = { glu_idx };
                ggml_op ops[7];
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = bias_op;
                    ops[3] = op;
                    ops[4] = GGML_OP_MUL;
                    ops[5] = bias_op;
                    ops[6] = GGML_OP_GLU;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = op;
                    ops[3] = GGML_OP_MUL;
                    ops[4] = GGML_OP_GLU;
                }
                const int n_ops = with_bias ? 7 : 5;

                if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                        !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                    continue;
                }

                ggml_tensor * gate_n       = cgraph->nodes[gate_idx];
                ggml_tensor * gate_scale_n = cgraph->nodes[gate_scale_idx];
                ggml_tensor * gate_out_n   = with_bias ? cgraph->nodes[gate_bias_idx] : gate_scale_n;
                ggml_tensor * up_n         = cgraph->nodes[up_idx];
                ggml_tensor * up_scale_n   = cgraph->nodes[up_scale_idx];
                ggml_tensor * up_out_n     = with_bias ? cgraph->nodes[up_bias_idx] : up_scale_n;
                const ggml_tensor * glu = cgraph->nodes[glu_idx];

                if (!ggml_cuda_should_fuse_mul_mat(up_n, gate_n, glu,
                        with_bias ? up_out_n : nullptr, with_bias ? gate_out_n : nullptr, up_scale_n, gate_scale_n)) {
                    continue;
                }

                const ggml_tensor * gate_scale = get_mul_mat_scale(gate_scale_n, gate_n);
                const ggml_tensor * up_scale   = get_mul_mat_scale(up_scale_n, up_n);
                if (!gate_scale || !up_scale) {
                    continue;
                }

                const ggml_tensor * up_bias   = with_bias ? get_bias_tensor(up_out_n, up_scale_n, bias_op) : nullptr;
                const ggml_tensor * gate_bias = with_bias ? get_bias_tensor(gate_out_n, gate_scale_n, bias_op) : nullptr;
                if (with_bias && (!ggml_are_same_shape(gate_out_n->src[0], gate_out_n->src[1]) ||
                        !ggml_are_same_shape(up_out_n->src[0], up_out_n->src[1]))) {
                    continue;
                }

                const ggml_tensor * src0 = up_n->src[0];
                const ggml_tensor * src1 = up_n->src[1];
                const ggml_tensor * ids  = up_n->src[2];

                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate       = gate_n->src[0];
                fusion_data.x_bias     = up_bias;
                fusion_data.gate_bias  = gate_bias;
                fusion_data.x_scale    = up_scale;
                fusion_data.gate_scale = gate_scale;
                fusion_data.glu_op     = ggml_get_glu_op(glu);
                fusion_data.glu_limit  = ggml_get_op_params_f32(glu, 3);

                if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data, shared_quantized(up_n));
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }
            }

            if (fused_mul_mat_vec) {
                break;
            }
        } else {
            for (const bool with_bias : { false, true }) {
                const int gate_idx       = i;
                const int gate_scale_idx = i + 4;
                const int gate_bias_idx  = with_bias ? i + 5 : -1;
                const int up_idx         = with_bias ? i + 6 : i + 5;
                const int up_scale_idx   = up_idx + 4;
                const int up_bias_idx    = with_bias ? up_idx + 5 : -1;
                const int glu_idx        = with_bias ? up_idx + 6 : up_idx + 5;

                const int out_nodes[] = { glu_idx };
                ggml_op ops[13];
                if (with_bias) {
                    ops[0]  = op;
                    ops[1]  = GGML_OP_RESHAPE;
                    ops[2]  = GGML_OP_REPEAT;
                    ops[3]  = GGML_OP_GET_ROWS;
                    ops[4]  = GGML_OP_MUL;
                    ops[5]  = bias_op;
                    ops[6]  = op;
                    ops[7]  = GGML_OP_RESHAPE;
                    ops[8]  = GGML_OP_REPEAT;
                    ops[9]  = GGML_OP_GET_ROWS;
                    ops[10] = GGML_OP_MUL;
                    ops[11] = bias_op;
                    ops[12] = GGML_OP_GLU;
                } else {
                    ops[0]  = op;
                    ops[1]  = GGML_OP_RESHAPE;
                    ops[2]  = GGML_OP_REPEAT;
                    ops[3]  = GGML_OP_GET_ROWS;
                    ops[4]  = GGML_OP_MUL;
                    ops[5]  = op;
                    ops[6]  = GGML_OP_RESHAPE;
                    ops[7]  = GGML_OP_REPEAT;
                    ops[8]  = GGML_OP_GET_ROWS;
                    ops[9]  = GGML_OP_MUL;
                    ops[10] = GGML_OP_GLU;
                }
                const int n_ops = with_bias ? 13 : 11;

                if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                        !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                    continue;
                }

                ggml_tensor * gate_n       = cgraph->nodes[gate_idx];
                ggml_tensor * gate_scale_n = cgraph->nodes[gate_scale_idx];
                ggml_tensor * gate_out_n   = with_bias ? cgraph->nodes[gate_bias_idx] : gate_scale_n;
                ggml_tensor * up_n         = cgraph->nodes[up_idx];
                ggml_tensor * up_scale_n   = cgraph->nodes[up_scale_idx];
                ggml_tensor * up_out_n     = with_bias ? cgraph->nodes[up_bias_idx] : up_scale_n;
                const ggml_tensor * glu = cgraph->nodes[glu_idx];

                if (!ggml_cuda_should_fuse_mul_mat(up_n, gate_n, glu,
                        with_bias ? up_out_n : nullptr, with_bias ? gate_out_n : nullptr, up_scale_n, gate_scale_n)) {
                    continue;
                }

                const ggml_tensor * gate_scale = get_mul_mat_id_scale(cgraph->nodes[gate_idx + 1], cgraph->nodes[gate_idx + 2],
                        cgraph->nodes[gate_idx + 3], gate_scale_n, gate_n);
                const ggml_tensor * up_scale = get_mul_mat_id_scale(cgraph->nodes[up_idx + 1], cgraph->nodes[up_idx + 2],
                        cgraph->nodes[up_idx + 3], up_scale_n, up_n);
                if (!gate_scale || !up_scale) {
                    continue;
                }

                const ggml_tensor * up_bias   = with_bias ? get_bias_tensor(up_out_n, up_scale_n, bias_op) : nullptr;
                const ggml_tensor * gate_bias = with_bias ? get_bias_tensor(gate_out_n, gate_scale_n, bias_op) : nullptr;

                const ggml_tensor * src0 = up_n->src[0];
                const ggml_tensor * src1 = up_n->src[1];
                const ggml_tensor * ids  = up_n->src[2];

                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate       = gate_n->src[0];
                fusion_data.x_bias     = up_bias;
                fusion_data.gate_bias  = gate_bias;
                fusion_data.x_scale    = up_scale;
                fusion_data.gate_scale = gate_scale;
                fusion_data.glu_op     = ggml_get_glu_op(glu);
                fusion_data.glu_limit  = ggml_get_op_params_f32(glu, 3);

                if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data, shared_quantized(up_n));
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }
            }

            if (fused_mul_mat_vec) {
                break;
            }
        }

        if (ggml_cuda_can_fuse(cgraph, i, { op, bias_op, op, bias_op, GGML_OP_GLU }, {})) {
            ggml_tensor * glu         = cgraph->nodes[i + 4];
            ggml_tensor * gate_bias_n = glu->src[0];
            ggml_tensor * up_bias_n   = glu->src[1];

            //we don't assume the order for {gate, up}. Instead infer it from the bias tensor
            ggml_tensor * gate_n = nullptr;
            ggml_tensor * up_n   = nullptr;

            if (gate_bias_n->src[0] == cgraph->nodes[i] || gate_bias_n->src[1] == cgraph->nodes[i]) {
                gate_n = cgraph->nodes[i];
                up_n   = cgraph->nodes[i + 2];
            } else if (gate_bias_n->src[0] == cgraph->nodes[i + 2] || gate_bias_n->src[1] == cgraph->nodes[i + 2]) {
                gate_n = cgraph->nodes[i + 2];
                up_n   = cgraph->nodes[i];
            } else {
                continue;
            }

            const ggml_tensor * up_bias_tensor   = get_bias_tensor(up_bias_n, up_n, bias_op);
            const ggml_tensor * gate_bias_tensor = get_bias_tensor(gate_bias_n, gate_n, bias_op);

            if (!up_bias_tensor || !gate_bias_tensor) {
                continue;
            }

            // we don't support repeating adds
            if (bias_op == GGML_OP_ADD && (!ggml_are_same_shape(gate_bias_n->src[0], gate_bias_n->src[1]) ||
                                           !ggml_are_same_shape(up_bias_n->src[0], up_bias_n->src[1]))) {
                continue;
            }

            const ggml_tensor * src0 = up_n->src[0];
            const ggml_tensor * src1 = up_n->src[1];
            const ggml_tensor * ids  = up_n->src[2];

            if (ggml_cuda_should_fuse_mul_mat_vec_f(up_n)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate_n->src[0];
                fusion_data.x_bias    = up_bias_tensor;
                fusion_data.gate_bias = gate_bias_tensor;
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 5;
                break;
            }

            if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate_n->src[0];
                fusion_data.x_bias    = up_bias_tensor;
                fusion_data.gate_bias = gate_bias_tensor;
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data, shared_quantized(up_n));
                fused_mul_mat_vec = true;
                fused_node_count  = 5;
                break;
            }
        } else if (ggml_cuda_can_fuse(cgraph, i, { op, op, GGML_OP_GLU }, {})) {
            ggml_tensor * glu  = cgraph->nodes[i + 2];
            ggml_tensor * gate = glu->src[0];
            ggml_tensor * up   = glu->src[1];

            bool ok = (gate == cgraph->nodes[i] && up == cgraph->nodes[i + 1]) ||
                      (gate == cgraph->nodes[i + 1] && up == cgraph->nodes[i]);

            if (!ok) {
                continue;
            }

            const ggml_tensor * src0 = up->src[0];
            const ggml_tensor * src1 = up->src[1];
            const ggml_tensor * ids  = up->src[2];

            if (ggml_cuda_should_fuse_mul_mat_vec_f(up)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }

            if (ggml_cuda_should_fuse_mul_mat_vec_q(up)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data, shared_quantized(up));
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    fused_mul_mat_vec = false;
    fused_node_count  = 0;

    // mul_mat + scale + optional bias
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        for (const bool with_bias : { false, true }) {
            const int n_ops = op == GGML_OP_MUL_MAT ? (with_bias ? 3 : 2) : (with_bias ? 6 : 5);
            const int out_nodes[] = { i + n_ops - 1 };
            ggml_op ops[6];
            if (op == GGML_OP_MUL_MAT) {
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = bias_op;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                }
            } else {
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_RESHAPE;
                    ops[2] = GGML_OP_REPEAT;
                    ops[3] = GGML_OP_GET_ROWS;
                    ops[4] = GGML_OP_MUL;
                    ops[5] = bias_op;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_RESHAPE;
                    ops[2] = GGML_OP_REPEAT;
                    ops[3] = GGML_OP_GET_ROWS;
                    ops[4] = GGML_OP_MUL;
                }
            }

            if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                    !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                continue;
            }

            ggml_tensor * mm_node    = cgraph->nodes[i];
            ggml_tensor * scale_node = op == GGML_OP_MUL_MAT ? cgraph->nodes[i + 1] : cgraph->nodes[i + 4];
            ggml_tensor * out_node   = with_bias ? cgraph->nodes[i + n_ops - 1] : scale_node;

            const ggml_tensor * scale = nullptr;
            if (op == GGML_OP_MUL_MAT) {
                scale = get_mul_mat_scale(scale_node, mm_node);
            } else {
                scale = get_mul_mat_id_scale(cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 3], scale_node, mm_node);
            }
            if (!scale) {
                continue;
            }

            const ggml_tensor * bias = with_bias ? get_bias_tensor(out_node, scale_node, bias_op) : nullptr;
            if (with_bias && !bias) {
                continue;
            }
            if (with_bias && bias_op == GGML_OP_ADD && !ggml_are_same_shape(out_node->src[0], out_node->src[1])) {
                continue;
            }
            if (with_bias && bias_op == GGML_OP_ADD_ID && out_node->src[2] != mm_node->src[2]) {
                continue;
            }

            const ggml_tensor * src0 = mm_node->src[0];
            const ggml_tensor * src1 = mm_node->src[1];
            const ggml_tensor * ids  = mm_node->src[2];

            ggml_cuda_mm_fusion_args_host fusion_data{};
            fusion_data.x_bias  = bias;
            fusion_data.x_scale = scale;

            if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, out_node, &fusion_data, shared_quantized(mm_node));
                fused_mul_mat_vec = true;
                fused_node_count  = n_ops;
                break;
            }
        }
        if (fused_mul_mat_vec) {
            break;
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    // mul_mat + add
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        if (!ggml_can_fuse(cgraph, i, { op, bias_op })) {
            continue;
        }

        ggml_tensor * mm_node   = cgraph->nodes[i];
        ggml_tensor * bias_node = cgraph->nodes[i + 1];

        ggml_tensor * bias_tensor = nullptr;
        if (bias_op == GGML_OP_ADD) {
            if (bias_node->src[0] == mm_node) {
                bias_tensor = bias_node->src[1];
            } else if (bias_node->src[1] == mm_node) {
                bias_tensor = bias_node->src[0];
            } else {
                continue;
            }
        } else {
            if (bias_node->src[0] != mm_node) {
                continue;
            }
            bias_tensor = bias_node->src[1];
        }

        const ggml_tensor * src0 = mm_node->src[0];
        const ggml_tensor * src1 = mm_node->src[1];
        const ggml_tensor * ids  = mm_node->src[2];

        if (bias_op == GGML_OP_ADD_ID && bias_node->src[2] != ids) {
            continue;
        }

        if (bias_op == GGML_OP_ADD && !ggml_are_same_shape(bias_node->src[0], bias_node->src[1])) {
            continue;
        }

        ggml_cuda_mm_fusion_args_host fusion_data{};
        fusion_data.x_bias = bias_tensor;

        if (ggml_cuda_should_fuse_mul_mat_vec_f(mm_node)) {
            ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count  = 2;
            break;
        }

        if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
            ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data, shared_quantized(mm_node));
            fused_mul_mat_vec = true;
            fused_node_count  = 2;
            break;
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    if (cuda_ctx->curr_stream_no == 0 && cuda_ctx->stream_context().concurrent_events.empty()) {
        const auto injection = ggml_cuda_match_hc_injection(*cuda_ctx, cgraph, i);
        if (injection.count && (prepared_hc_post < 0 || injection.post != cgraph->nodes[prepared_hc_post] || (!injection.hc.norm && injection.post->src[3]))) {
            ggml_cuda_op_hc_injection(*cuda_ctx, injection.first, injection.unary, injection.last, injection.post, injection.hc.norm, injection.hc.mul);
            return injection.count - 1;
        }
    }

    if (node->op == GGML_OP_DSV4_HC_POST && cuda_ctx->curr_stream_no == 0 && cuda_ctx->stream_context().concurrent_events.empty()) {
        const auto match = ggml_cuda_match_hc_post_norm(cgraph, i);
        if (match.count != 0 && ggml_cuda_hc_post_norm_memory_ok(cgraph, i, match, cuda_ctx->device)) {
            if (match.scale) {
                ggml_cuda_op_hc_post_norm_scale(*cuda_ctx, match.post, match.norm, match.scale, nullptr, nullptr);
            } else {
                ggml_cuda_op_hc_post_norm(*cuda_ctx, match.post, match.norm, match.mul);
            }
            return match.count - 1;
        }
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 4]);
        return 4;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE }, {})) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], nullptr);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD }, {})) {
        ggml_cuda_op_rms_norm_fused_add(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
        ggml_cuda_op_rms_norm_fused(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_SCALE }, {})) {
        ggml_cuda_op_rms_norm_scale_fused(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_ADD, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        ggml_cuda_op_ssm_conv(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        ggml_cuda_op_ssm_conv(*cuda_ctx, node, /*bias_add_node=*/ nullptr, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SILU }) ||
        ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SIGMOID }) ||
        ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SOFTPLUS })) {
        ggml_cuda_op_unary_mul(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_SQR }, { GGML_UNARY_OP_RELU })) {
        ggml_cuda_op_relu_sqr(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE }, { GGML_UNARY_OP_TANH })) {
        ggml_cuda_op_softcap(*cuda_ctx, cgraph->nodes[i + 2], node);
        return 2;
    }

    if (cuda_ctx->curr_stream_no == 0 && cuda_ctx->stream_context().concurrent_events.empty()) {
        const auto match = ggml_cuda_match_scaled_unary(cgraph, i);
        if (match.count && ggml_cuda_scaled_unary_memory_ok(match, cuda_ctx->device)) {
            ggml_cuda_op_scaled_unary(*cuda_ctx, match.first, match.unary, match.last);
            return match.count - 1;
        }
    }

    if (cuda_ctx->curr_stream_no == 0 && cuda_ctx->stream_context().concurrent_events.empty()) {
        const auto match = ggml_cuda_match_cublas_affine(*cuda_ctx, cgraph, i);
        if (match.count) {
            const auto & affine = match.affine;
            const ggml_cuda_affine_unary_ops ops = {affine.mul, affine.add, affine.unary, affine.post, affine.tail};
            ggml_cuda_mul_mat_cublas(*cuda_ctx, match.mm->src[0], match.mm->src[1], match.mm, prepared_src1, nullptr, nullptr, &ops);
            return match.count - 1;
        }
    }

    if (cuda_ctx->curr_stream_no == 0 && cuda_ctx->stream_context().concurrent_events.empty()) {
        const auto match = ggml_cuda_match_cublas_unary(*cuda_ctx, cgraph, i);
        if (match.count) {
            const ggml_cuda_scaled_unary_args args = {match.unary.first, match.unary.unary, match.unary.last};
            ggml_cuda_mul_mat_cublas(*cuda_ctx, match.mm->src[0], match.mm->src[1], match.mm, prepared_src1, nullptr, &args);
            return match.count - 1;
        }
    }

    if (cuda_ctx->curr_stream_no == 0 && cuda_ctx->stream_context().concurrent_events.empty()) {
        const auto match = ggml_cuda_match_hc_up(*cuda_ctx, cgraph, i);
        if (match.count) {
            if (match.convert) {
                ggml_cuda_mul_mat_cublas(*cuda_ctx, match.mm->src[0], match.mm->src[1], match.mm, prepared_src1, match.dst);
            } else {
                ggml_cuda_mul_mat_vec_f_hc_pre(*cuda_ctx, match.mm, match.dst);
            }
            return match.count - 1;
        }
    }

    if (cuda_ctx->curr_stream_no == 0 && cuda_ctx->stream_context().concurrent_events.empty()) {
        const auto injection = affine_match ? *affine_match : ggml_cuda_match_hc_affine_injection(*cuda_ctx, cgraph, i);
        if (injection.count && affine_emit) {
            *affine_emit = true;
            return 0;
        }
        if (injection.count && (prepared_hc_post < 0 || injection.post != cgraph->nodes[prepared_hc_post] || (!injection.hc.norm && injection.post->src[3]))) {
            if (injection.comb) {
                const bool ok = ggml_cuda_compute_forward(*cuda_ctx, injection.comb);
                GGML_ASSERT(ok);
            }
            const auto & affine = injection.affine;
            ggml_cuda_op_hc_affine_injection(*cuda_ctx, affine.mul, affine.add, affine.unary, affine.post ? affine.post : affine.unary, injection.post, injection.hc.norm, injection.hc.mul, nullptr, injection.hc.scale);
            return injection.count - 1;
        }
    }

    if (cuda_ctx->curr_stream_no == 0 && cuda_ctx->stream_context().concurrent_events.empty()) {
        const auto repeated = ggml_cuda_match_repeat_mul_add(*cuda_ctx, cgraph, i);
        if (repeated.count) {
            ggml_cuda_op_repeat_mul_add(*cuda_ctx, repeated.repeat, repeated.binary.mul, repeated.binary.add);
            return repeated.count - 1;
        }
        auto affine = ggml_cuda_match_affine_unary(*cuda_ctx, cgraph, i, true, true);
        if (!affine.count) { affine = ggml_cuda_match_affine_unary(*cuda_ctx, cgraph, i); }
        if (affine.count) {
            ggml_cuda_op_affine_unary(*cuda_ctx, affine.mul, affine.add, affine.unary, affine.post, affine.tail);
            return affine.count - 1;
        }
        const auto binary = ggml_cuda_match_mul_add(*cuda_ctx, cgraph, i);
        if (binary.count) {
            ggml_cuda_op_mul_add(*cuda_ctx, binary.mul, binary.add);
            return binary.count - 1;
        }
    }


    return 0;
}

static void ggml_cuda_graph_evaluate_and_capture(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const bool use_cuda_graph, const bool cuda_graph_update_required, const void * graph_key) {
    bool graph_evaluated_or_captured = false;

    // flag used to determine whether it is an integrated_gpu
    const bool integrated            = ggml_cuda_info().devices[cuda_ctx->device].integrated;

    ggml_cuda_stream_context & stream_ctx = cuda_ctx->stream_context();
    bool                         is_concurrent_event_active = false;
    ggml_cuda_concurrent_event * concurrent_event           = nullptr;
    bool                         should_launch_concurrent_events = false;


    while (!graph_evaluated_or_captured) {
        // Only perform the graph execution if CUDA graphs are not enabled, or we are capturing the graph.
        // With the use of CUDA graphs, the execution will be performed by the graph launch.
        if (!use_cuda_graph || cuda_graph_update_required) {
            [[maybe_unused]] int prev_i = 0;

            if (stream_ctx.concurrent_events.size() > 0) {
                should_launch_concurrent_events = true;
                for (const auto & [tensor, event] : stream_ctx.concurrent_events) {
                    should_launch_concurrent_events = should_launch_concurrent_events && event.is_valid();
                }
            }

            if (should_launch_concurrent_events) {
                // Restore original node order within each concurrent region to enable fusion within streams

                std::unordered_map<const ggml_tensor *, int> node_to_idx;
                node_to_idx.reserve(cgraph->n_nodes);
                for (int i = 0; i < cgraph->n_nodes; ++i) {
                    node_to_idx[cgraph->nodes[i]] = i;
                }

                for (auto & [fork_node, event] : stream_ctx.concurrent_events) {
                    // Find positions of all nodes from this event in the current graph
                    std::vector<int> positions;
                    positions.reserve(event.original_order.size());

                    bool all_found = true;
                    for (const ggml_tensor * orig_node : event.original_order) {
                        auto it = node_to_idx.find(orig_node);
                        if (it != node_to_idx.end()) {
                            positions.push_back(it->second);
                        } else {
                            all_found = false;
                            break;
                        }
                    }

                    if (!all_found || positions.size() != event.original_order.size()) {
                        continue;
                    }

                    // Sort positions to get contiguous range
                    std::vector<int> sorted_positions = positions;
                    std::sort(sorted_positions.begin(), sorted_positions.end());

                    bool is_contiguous = true;
                    for (size_t i = 1; i < sorted_positions.size(); ++i) {
                        if (sorted_positions[i] != sorted_positions[i-1] + 1) {
                            is_contiguous = false;
                            break;
                        }
                    }

                    if (!is_contiguous) {
                        continue;
                    }

                    // Restore original order at the sorted positions
                    int start_pos = sorted_positions[0];
                    for (size_t i = 0; i < event.original_order.size(); ++i) {
                        cgraph->nodes[start_pos + i] = const_cast<ggml_tensor *>(event.original_order[i]);
                    }
                }
            } else {
                stream_ctx.concurrent_events.clear();
            }

            static const bool disable_reuse = getenv("GGML_CUDA_DISABLE_FUSION") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_FUSION"));
            std::vector<int> input_keys;
            std::vector<size_t> input_sizes;
            if (!disable_reuse) {
                input_keys.assign(cgraph->n_nodes, -1);
                input_sizes.resize(cgraph->n_nodes);
                for (int i = 0; i < cgraph->n_nodes; ++i) {
                    input_keys[i] = ggml_cuda_cublas_input_key(cgraph->nodes[i], cuda_ctx->device, input_sizes[i]);
                }
            }
            ggml_cuda_reuse_plan reuse(cgraph, stream_ctx, input_keys, input_sizes, ggml_cuda_prepared_input_overwritten);
            std::vector<ggml_cuda_norm_emit_match> norm_emits;
            if (!disable_reuse && stream_ctx.concurrent_events.empty()) {
                norm_emits = ggml_cuda_plan_norm_emit(cgraph, cuda_ctx->device, input_keys, input_sizes, reuse);
            }
            ggml_cuda_reuse_inputs<ggml_cuda_pool_alloc<char>> shared_inputs(cuda_ctx->pool(), reuse);
            const auto prepare_group = [&](int group) {
                if (shared_inputs[group].get()) { return; }
                const auto * node = cgraph->nodes[reuse.groups[group].node];
                ggml_cuda_prepare_cublas_input(*cuda_ctx, node->src[1], input_keys[reuse.groups[group].node], shared_inputs[group]);
            };
            const auto prepare_shared = [&](int i, bool after) {
                if (reuse.starts.empty()) { return; }
                for (int group = reuse.starts[i]; group >= 0; group = reuse.groups[group].next) {
                    if (reuse.groups[group].after == after) { prepare_group(group); }
                }
            };
            std::vector<int> mmvq_keys;
            std::vector<size_t> mmvq_sizes;
            if (!disable_reuse) {
                mmvq_keys.assign(cgraph->n_nodes, -1);
                mmvq_sizes.resize(cgraph->n_nodes);
                for (int i = 0; i < cgraph->n_nodes; ++i) {
                    const ggml_tensor * node = cgraph->nodes[i];
                    if (ggml_cuda_can_share_mmvq_input(node, cuda_ctx->device)) {
                        const ggml_tensor * input = node->src[1];
                        if (input->ne[0] <= 0 || input->ne[0] > INT64_MAX - MATRIX_ROW_PADDING + 1 ||
                                uint64_t(input->ne[0]) > SIZE_MAX - MATRIX_ROW_PADDING + 1) { continue; }
                        const size_t blocks = GGML_PAD(size_t(input->ne[0]), MATRIX_ROW_PADDING)/QK8_1;
                        if (blocks > SIZE_MAX/sizeof(block_q8_1)) { continue; }
                        const size_t row = blocks*sizeof(block_q8_1);
                        if (uint64_t(input->ne[1]) > SIZE_MAX/row) { continue; }
                        mmvq_keys[i] = 0;
                        mmvq_sizes[i] = size_t(input->ne[1])*row;
                    }
                }
            }
            ggml_cuda_reuse_plan q8_reuse(cgraph, stream_ctx, mmvq_keys, mmvq_sizes, ggml_cuda_mmvq_input_overwritten);
            std::vector<ggml_cuda_norm_q8_match> q8_emits;
            if (!disable_reuse && stream_ctx.concurrent_events.empty()) {
                q8_emits = ggml_cuda_plan_norm_q8(cgraph, cuda_ctx->device, mmvq_keys, mmvq_sizes, q8_reuse);
            }
            ggml_cuda_reuse_inputs<ggml_cuda_pool_alloc<char>> q8_inputs(cuda_ctx->pool(), q8_reuse);
            const auto prepare_q8_group = [&](int group) {
                if (q8_inputs[group].get()) { return; }
                const ggml_tensor * node = cgraph->nodes[q8_reuse.groups[group].node];
                ggml_cuda_quantize_mmvq_input(*cuda_ctx, node->src[0], node->src[1], q8_inputs[group]);
            };
            const auto prepare_q8_shared = [&](int i, bool after) {
                if (q8_reuse.starts.empty()) { return; }
                for (int group = q8_reuse.starts[i]; group >= 0; group = q8_reuse.groups[group].next) {
                    if (q8_reuse.groups[group].after == after) {
                        prepare_q8_group(group);
                    }
                }
            };
            std::vector<int> mmq_keys;
            std::vector<size_t> mmq_sizes;
            const int mmq_cc = ggml_cuda_info().devices[cuda_ctx->device].cc;
            const auto scale_size = [&](const ggml_tensor * node) {
                return ggml_cuda_mmq_get_prec_src1(node->src[0], node, mmq_cc) == GGML_PREC_Q4 && node->src[0]->type == GGML_TYPE_NVFP4 ?
                    GGML_PAD((size_t(node->src[1]->ne[1]) + 128)*sizeof(float), 256) : 0;
            };
            if (!disable_reuse) {
                mmq_keys.assign(cgraph->n_nodes, -1);
                mmq_sizes.resize(cgraph->n_nodes);
                for (int i = 0; i < cgraph->n_nodes; ++i) {
                    const ggml_tensor * node = cgraph->nodes[i];
                    if (ggml_cuda_can_share_mmq_input(node, cuda_ctx->device)) {
                        const ggml_tensor * input = node->src[1];
                        if (input->ne[0] <= 0 || input->ne[0] > INT64_MAX - MATRIX_ROW_PADDING + 1 ||
                                uint64_t(input->ne[0]) > SIZE_MAX - MATRIX_ROW_PADDING + 1) { continue; }
                        const bool fp4 = ggml_cuda_mmq_get_prec_src1(node->src[0], node, mmq_cc) == GGML_PREC_Q4;
                        const size_t blocks = GGML_PAD(size_t(input->ne[0]), MATRIX_ROW_PADDING)/(fp4 ? QK_FP4_MMQ : QK8_1_MMQ);
                        if (blocks > SIZE_MAX/sizeof(block_q8_1_mmq)) { continue; }
                        const size_t row = blocks*sizeof(block_q8_1_mmq);
                        // The kernel reads a full tile, including columns past the input.
                        const size_t tail = 128*sizeof(block_q8_1_mmq);
                        if (uint64_t(input->ne[1]) > (SIZE_MAX - tail - 255)/row ||
                                uint64_t(input->ne[1]) > (SIZE_MAX - 255)/sizeof(float) - 128) { continue; }
                        const size_t quantized = size_t(input->ne[1])*row + tail;
                        if (scale_size(node) > SIZE_MAX - GGML_PAD(quantized, 256)) { continue; }
                        mmq_keys[i] = ggml_cuda_mmq_get_prec_src1(node->src[0], node, mmq_cc) == GGML_PREC_Q4 ?
                            3 + (node->src[0]->type == GGML_TYPE_NVFP4) : int(mmq_get_q8_1_ds_layout(node->src[0]->type));
                        mmq_sizes[i] = GGML_PAD(quantized, 256) + scale_size(node);
                    }
                }
            }
            ggml_cuda_reuse_plan mmq_reuse(cgraph, stream_ctx, mmq_keys, mmq_sizes, ggml_cuda_mmq_input_overwritten);
            std::vector<ggml_cuda_norm_mmq_match> mmq_emits;
            if (!disable_reuse && stream_ctx.concurrent_events.empty()) {
                mmq_emits = ggml_cuda_plan_norm_mmq(cgraph, cuda_ctx->device, mmq_keys, mmq_sizes, mmq_reuse);
            }
            ggml_cuda_reuse_inputs<ggml_cuda_mmq_input> mmq_inputs(cuda_ctx->pool(), mmq_reuse);
            const auto prepare_mmq_group = [&](int group) {
                if (mmq_inputs[group].quantized.get()) { return; }
                const ggml_tensor * node = cgraph->nodes[mmq_reuse.groups[group].node];
                ggml_cuda_quantize_mmq_input(*cuda_ctx, node, mmq_reuse.groups[group].size - scale_size(node), mmq_inputs[group]);
            };
            const auto prepare_mmq_shared = [&](int i, bool after) {
                if (mmq_reuse.starts.empty()) { return; }
                for (int group = mmq_reuse.starts[i]; group >= 0; group = mmq_reuse.groups[group].next) {
                    if (mmq_reuse.groups[group].after == after) { prepare_mmq_group(group); }
                }
            };
            const auto try_launch_concurrent_event = [&](const ggml_tensor * node) {
                if (stream_ctx.concurrent_events.find(node) != stream_ctx.concurrent_events.end()) {
                    const int i = reuse.indices.empty() ? -1 : reuse.indices.at(node);
                    if (i >= 0) { prepare_shared(i, true); }
                    const int q8_i = q8_reuse.indices.empty() ? -1 : q8_reuse.indices.at(node);
                    if (q8_i >= 0) { prepare_q8_shared(q8_i, true); }
                    const int mmq_i = mmq_reuse.indices.empty() ? -1 : mmq_reuse.indices.at(node);
                    if (mmq_i >= 0) { prepare_mmq_shared(mmq_i, true); }
                    concurrent_event = &stream_ctx.concurrent_events[node];

                    is_concurrent_event_active = true;

                    GGML_LOG_DEBUG("Launching %d streams at %s\n", concurrent_event->n_streams, node->name);

                    cudaStream_t main_stream = cuda_ctx->stream();  // this should be stream 0
                    GGML_ASSERT(cuda_ctx->curr_stream_no == 0);
                    CUDA_CHECK(cudaEventRecord(concurrent_event->fork_event, main_stream));

                    for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                        cudaStream_t stream = cuda_ctx->stream(cuda_ctx->device, i);
                        CUDA_CHECK(cudaStreamWaitEvent(stream, concurrent_event->fork_event));
                    }
                }
            };

            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (is_concurrent_event_active) {
                    GGML_ASSERT(concurrent_event);

                    if (node == concurrent_event->join_node) {
                        cuda_ctx->curr_stream_no = 0;
                        for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                            // Wait on join events of forked streams in the main stream
                            CUDA_CHECK(cudaEventRecord(concurrent_event->join_events[i - 1],
                                                       cuda_ctx->stream(cuda_ctx->device, i)));
                            CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), concurrent_event->join_events[i - 1]));
                        }

                        is_concurrent_event_active = false;
                        concurrent_event           = nullptr;
                    } else {
                        GGML_ASSERT (concurrent_event->stream_mapping.find(node) != concurrent_event->stream_mapping.end());
                        cuda_ctx->curr_stream_no = concurrent_event->stream_mapping[node];
                        GGML_LOG_DEBUG("Setting stream no to %d for node %s\n", cuda_ctx->curr_stream_no, node->name);
                    }
                } else if (i - prev_i > 1) {
                    //the previous node was fused
                    const ggml_tensor * prev_node = cgraph->nodes[i - 1];
                    try_launch_concurrent_event(prev_node);

                    if (is_concurrent_event_active) {
                        cuda_ctx->curr_stream_no = concurrent_event->stream_mapping[node];
                        GGML_LOG_DEBUG("Setting stream no to %d for node %s\n", cuda_ctx->curr_stream_no, node->name);
                    }
                }

                prev_i = i;

                if (ggml_cuda_is_view_or_noop(node)) {
                    continue;
                }

                if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                    continue;
                }

                const int group = reuse.nodes.empty() ? -1 : reuse.nodes[i];
                prepare_shared(i, false);
                if (group >= 0) { prepare_group(group); }
                const void * prepared_src1 = group >= 0 ? shared_inputs[group].get() : nullptr;

                prepare_q8_shared(i, false);
                const int q8_group = q8_reuse.nodes.empty() ? -1 : q8_reuse.nodes[i];
                if (q8_group >= 0 && !q8_inputs[q8_group].get()) {
                    bool ready = !is_concurrent_event_active;
                    if (!ready && !q8_reuse.groups[q8_group].after) {
                        const auto first = concurrent_event->stream_mapping.find(cgraph->nodes[q8_reuse.groups[q8_group].node]);
                        ready = first != concurrent_event->stream_mapping.end() && first->second == cuda_ctx->curr_stream_no;
                    }
                    if (ready) { prepare_q8_group(q8_group); }
                }
                const ggml_tensor * shared_input = q8_group >= 0 ? node->src[1] : nullptr;
                const char * quantized = q8_group >= 0 ? q8_inputs[q8_group].get() : nullptr;
                int prepared_hc_post = -1;
                ggml_cuda_hc_affine_injection_match affine_injection;
                const ggml_cuda_hc_affine_injection_match * affine_match = nullptr;
                if (node->op == GGML_OP_MUL && cuda_ctx->curr_stream_no == 0 && cuda_ctx->stream_context().concurrent_events.empty() &&
                        (!norm_emits.empty() || !q8_emits.empty() || !mmq_emits.empty())) {
                    affine_injection = ggml_cuda_match_hc_affine_injection(*cuda_ctx, cgraph, i);
                    affine_match = &affine_injection;
                    if (affine_injection.count) {
                        const int post_i = i + affine_injection.count - (affine_injection.hc.count ? affine_injection.hc.count : 1);
                        if ((!norm_emits.empty() && norm_emits[post_i].post) || (!q8_emits.empty() && q8_emits[post_i].post) || (!mmq_emits.empty() && mmq_emits[post_i].post)) {
                            prepared_hc_post = post_i;
                        }
                    }
                }
                bool affine_emit = false;
                if (affine_match && affine_injection.hc.norm && prepared_hc_post >= 0) {
                    const auto fits = [&](const auto & emit) {
                        return !emit.norm || (emit.post == affine_injection.post && emit.norm == affine_injection.hc.norm &&
                            emit.mul == affine_injection.hc.mul && !emit.add && emit.scale == affine_injection.hc.scale &&
                            emit.dst == (emit.scale ? emit.scale : emit.mul ? emit.mul : emit.norm) && emit.last == i + affine_injection.count - 1);
                    };
                    if ((norm_emits.empty() || fits(norm_emits[prepared_hc_post])) &&
                            (q8_emits.empty() || fits(q8_emits[prepared_hc_post])) &&
                            (mmq_emits.empty() || fits(mmq_emits[prepared_hc_post]))) {
                        const int skipped = ggml_cuda_try_fuse(cuda_ctx, cgraph, i, shared_input, quantized, prepared_hc_post, affine_match, &affine_emit, prepared_src1);
                        if (skipped) {
                            i += skipped;
                            continue;
                        }
                        GGML_ASSERT(affine_emit);
                    }
                }
                const int emit_i = affine_emit ? prepared_hc_post : i;
                const auto launch_affine_emit = [&](const ggml_cuda_hc_affine_emit_data & data) {
                    if (affine_injection.comb) {
                        const bool ok = ggml_cuda_compute_forward(*cuda_ctx, affine_injection.comb);
                        GGML_ASSERT(ok);
                    }
                    const auto & affine = affine_injection.affine;
                    ggml_cuda_op_hc_affine_injection(*cuda_ctx, affine.mul, affine.add, affine.unary, affine.post ? affine.post : affine.unary,
                        affine_injection.post, affine_injection.hc.norm, affine_injection.hc.mul, &data, affine_injection.hc.scale);
                };
                if (!mmq_emits.empty() && mmq_emits[emit_i].post && (q8_emits.empty() || !q8_emits[emit_i].norm)) {
                    const auto & emit = mmq_emits[emit_i];
                    const auto * typed = !norm_emits.empty() && norm_emits[emit_i].norm ? &norm_emits[emit_i] : nullptr;
                    if (!typed || (typed->post == emit.post && typed->norm == emit.norm && typed->dst == emit.dst && typed->last == emit.last)) {
                        const int g = emit.image;
                        const int reader = mmq_reuse.groups[g].node;
                        const ggml_tensor * input = cgraph->nodes[reader]->src[1];
                        mmq_inputs[g].quantized.alloc(mmq_reuse.groups[g].size);
                        const auto image = [&](int group) -> void * {
                            if (group < 0) { return nullptr; }
                            shared_inputs[group].alloc(input_sizes[reuse.groups[group].node]);
                            return shared_inputs[group].get();
                        };
                        void * f16 = typed ? image(typed->f16) : nullptr;
                        void * bf16 = typed ? image(typed->bf16) : nullptr;
                        if (affine_emit) {
                            launch_affine_emit({f16, bf16, nullptr, mmq_inputs[g].quantized.get(), input->ne[0], GGML_PAD(input->ne[0], MATRIX_ROW_PADDING), input->ne[1], mmq_keys[reader]});
                        } else {
                            ggml_cuda_op_hc_post_norm_emit_mmq(*cuda_ctx, emit.post, emit.norm, emit.mul, emit.scale, f16, bf16,
                                mmq_inputs[g].quantized.get(), input->ne[0], GGML_PAD(input->ne[0], MATRIX_ROW_PADDING), input->ne[1], mmq_keys[reader]);
                        }
                        i = emit.last;
                        continue;
                    }
                }

                if (!q8_emits.empty() && q8_emits[emit_i].norm &&
                        ((!norm_emits.empty() && norm_emits[emit_i].norm) || q8_emits[emit_i].post)) {
                    const auto & emit = q8_emits[emit_i];
                    const int g = emit.image;
                    const ggml_tensor * input = cgraph->nodes[q8_reuse.groups[g].node]->src[1];
                    q8_inputs[g].alloc(mmvq_sizes[q8_reuse.groups[g].node]);
                    void * f16 = nullptr;
                    void * bf16 = nullptr;
                    if (!norm_emits.empty() && norm_emits[emit_i].norm) {
                        const auto & typed = norm_emits[emit_i];
                        GGML_ASSERT(typed.norm == emit.norm && typed.dst == emit.dst && typed.last == emit.last);
                        const auto image = [&](int group) -> void * {
                            if (group < 0) { return nullptr; }
                            shared_inputs[group].alloc(input_sizes[reuse.groups[group].node]);
                            return shared_inputs[group].get();
                        };
                        f16 = image(typed.f16);
                        bf16 = image(typed.bf16);
                    }
                    if (emit.post) {
                        if (emit.scale && !affine_emit) {
                            ggml_cuda_op_dsv4_hc_post(*cuda_ctx, emit.post);
                            ggml_cuda_op_rms_norm_emit_q8(*cuda_ctx, emit.norm, nullptr, nullptr, emit.scale, f16, bf16,
                                q8_inputs[g].get(), input->ne[0], GGML_PAD(input->ne[0], MATRIX_ROW_PADDING));
                        } else {
                            if (affine_emit) {
                                launch_affine_emit({f16, bf16, q8_inputs[g].get(), nullptr, input->ne[0], GGML_PAD(input->ne[0], MATRIX_ROW_PADDING), input->ne[1], 0});
                            } else {
                                ggml_cuda_op_hc_post_norm_emit_q8(*cuda_ctx, emit.post, emit.norm, emit.mul, f16, bf16,
                                    q8_inputs[g].get(), input->ne[0], GGML_PAD(input->ne[0], MATRIX_ROW_PADDING));
                            }
                        }
                    } else {
                        ggml_cuda_op_rms_norm_emit_q8(*cuda_ctx, emit.norm, emit.mul, emit.add, emit.scale, f16, bf16,
                            q8_inputs[g].get(), input->ne[0], GGML_PAD(input->ne[0], MATRIX_ROW_PADDING));
                    }
                    i = emit.last;
                    continue;
                }

                if (!norm_emits.empty() && norm_emits[emit_i].norm) {
                    const auto & emit = norm_emits[emit_i];
                    const auto image = [&](int g) -> void * {
                        if (g < 0) { return nullptr; }
                        shared_inputs[g].alloc(input_sizes[reuse.groups[g].node]);
                        return shared_inputs[g].get();
                    };
                    if (emit.post) {
                        if (emit.scale && !affine_emit) {
                            ggml_cuda_op_hc_post_norm_scale(*cuda_ctx, emit.post, emit.norm, emit.scale, image(emit.f16), image(emit.bf16));
                        } else {
                            if (affine_emit) {
                                launch_affine_emit({image(emit.f16), image(emit.bf16)});
                            } else {
                                ggml_cuda_op_hc_post_norm_emit(*cuda_ctx, emit.post, emit.norm, emit.mul, image(emit.f16), image(emit.bf16));
                            }
                        }
                    } else {
                        ggml_cuda_op_rms_norm_emit(*cuda_ctx, emit.norm, emit.mul, emit.add, emit.scale, image(emit.f16), image(emit.bf16));
                    }
                    i = emit.last;
                    continue;
                }

                if (!q8_emits.empty() && q8_emits[i].norm) {
                    const auto & emit = q8_emits[i];
                    const int g = emit.image;
                    const ggml_tensor * input = cgraph->nodes[q8_reuse.groups[g].node]->src[1];
                    q8_inputs[g].alloc(mmvq_sizes[q8_reuse.groups[g].node]);
                    ggml_cuda_op_rms_norm_q8(*cuda_ctx, emit.norm, emit.mul, emit.add, emit.scale,
                        q8_inputs[g].get(), input->ne[0], GGML_PAD(input->ne[0], MATRIX_ROW_PADDING));
                    i = emit.last;
                    continue;
                }

                prepare_mmq_shared(i, false);
                if (!mmq_emits.empty() && mmq_emits[i].norm) {
                    const auto & emit = mmq_emits[i];
                    const int g = emit.image;
                    const int reader = mmq_reuse.groups[g].node;
                    const ggml_tensor * input = cgraph->nodes[reader]->src[1];
                    mmq_inputs[g].quantized.alloc(mmq_reuse.groups[g].size);
                    ggml_cuda_op_rms_norm_mmq(*cuda_ctx, emit.norm, emit.mul, emit.add, emit.scale,
                        mmq_inputs[g].quantized.get(), input->ne[0], GGML_PAD(input->ne[0], MATRIX_ROW_PADDING), input->ne[1], mmq_keys[reader]);
                    i = emit.last;
                    continue;
                }

                const int mmq_group = mmq_reuse.nodes.empty() ? -1 : mmq_reuse.nodes[i];
                if (mmq_group >= 0 && !mmq_inputs[mmq_group].quantized.get()) {
                    bool ready = !is_concurrent_event_active;
                    if (!ready && !mmq_reuse.groups[mmq_group].after) {
                        const auto first = concurrent_event->stream_mapping.find(cgraph->nodes[mmq_reuse.groups[mmq_group].node]);
                        ready = first != concurrent_event->stream_mapping.end() && first->second == cuda_ctx->curr_stream_no;
                    }
                    if (ready) { prepare_mmq_group(mmq_group); }
                }

                if (node->op == GGML_OP_SCALE || node->op == GGML_OP_UNARY) {
                    for (int j = i + 1; j < cgraph->n_nodes && j - i <= 3; ++j) {
                        if ((!norm_emits.empty() && norm_emits[j].post) || (!q8_emits.empty() && q8_emits[j].post) || (!mmq_emits.empty() && mmq_emits[j].post)) {
                            prepared_hc_post = j;
                            break;
                        }
                    }
                }
                int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i, shared_input, quantized, prepared_hc_post, affine_match, nullptr, prepared_src1);

                if (nodes_to_skip != 0) {
#ifdef GGML_CUDA_DEBUG
                    const int last_fused = i + nodes_to_skip;
                    GGML_LOG_INFO("nodes_fused: %d, first: %s (%s), last: %s (%s)\n",
                            nodes_to_skip + 1, ggml_op_name(node->op), node->name,
                            ggml_op_name(cgraph->nodes[last_fused]->op), cgraph->nodes[last_fused]->name);
#endif
                    i += nodes_to_skip;
                    continue;
                }
#ifndef NDEBUG
                // On integrated GPUs (APUs, e.g. RDNA3.5) the scheduler may place a
                // node's output on the host-visible buffer, which the compute path
                // handles. Allow that here, mirroring the src-tensor check below.
                assert(node->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) ||
                       (integrated && ggml_backend_buft_is_cuda_host(node->buffer->buft)));
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    if (node->src[j] != nullptr) {
                        assert(node->src[j]->buffer);
                        assert(node->src[j]->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) ||
                               (integrated && ggml_backend_buft_is_cuda_host(node->src[j]->buffer->buft)));
                    }
                }
#else
                GGML_UNUSED(integrated);
#endif  // NDEBUG

                bool ok;
                if (mmq_group >= 0 && mmq_inputs[mmq_group].quantized.get()) {
                    ggml_cuda_mul_mat_q(*cuda_ctx, node->src[0], node->src[1], nullptr, node, &mmq_inputs[mmq_group]);
                    ok = true;
                } else if (quantized) {
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, node->src[0], node->src[1], nullptr, node, nullptr, quantized);
                    ok = true;
                } else {
                    ok = ggml_cuda_compute_forward(*cuda_ctx, node, prepared_src1);
                }
                if (!ok) {
                    GGML_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
                }
                GGML_ASSERT(ok);

                if (!is_concurrent_event_active) {
                    try_launch_concurrent_event(node);
               }
            }
        }

#ifdef USE_CUDA_GRAPH
        ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
        if (use_cuda_graph && cuda_graph_update_required) { // End CUDA graph capture
            if (graph->graph != nullptr) {
                CUDA_CHECK(cudaGraphDestroy(graph->graph));
                graph->graph = nullptr;
            }

            CUDA_CHECK(cudaStreamEndCapture(cuda_ctx->stream(), &graph->graph));
            graph_evaluated_or_captured = true; // CUDA graph has been captured

            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            if (ggml_cuda_lock_counter.fetch_sub(1, std::memory_order_relaxed) == 1) {
                ggml_cuda_lock_cv.notify_all();
            }
        } else {
            graph_evaluated_or_captured = true; // ggml graph has been directly evaluated
        }
    }

    if (use_cuda_graph) {
        ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
        if (graph->instance == nullptr) { // Create executable graph from captured graph.
            CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));
        }
        if (cuda_graph_update_required) { // Update graph executable
            ggml_cuda_graph_update_executable(cuda_ctx, graph_key);
        }
        // Launch graph
        CUDA_CHECK(cudaGraphLaunch(graph->instance, cuda_ctx->stream()));
#else
        GGML_UNUSED(graph_key);
        graph_evaluated_or_captured = true;
#endif  // USE_CUDA_GRAPH
    }
}

#ifdef USE_CUDA_GRAPH
static bool ggml_cuda_graph_set_enabled(ggml_backend_cuda_context * cuda_ctx, const void * graph_key) {
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

    if (graph->graph == nullptr) {
        if (ggml_cuda_info().devices[cuda_ctx->device].cc < GGML_CUDA_CC_VOLTA) {
            if (!graph->disable_due_to_gpu_arch) {
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to GPU architecture\n", __func__);
            }
            graph->disable_due_to_gpu_arch = true;
        }
    }

    return graph->is_enabled();
}
#endif // USE_CUDA_GRAPH

static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    ggml_cuda_set_device(cuda_ctx->device);

    bool use_cuda_graph             = false;
    bool cuda_graph_update_required = false;
    const void * graph_key = nullptr;

#ifdef USE_CUDA_GRAPH
    graph_key = ggml_cuda_graph_get_key(cgraph);

    ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);

    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
    if (graph->is_enabled()) {
        const bool graph_compatible = ggml_cuda_graph_check_compability(cgraph);
        if (graph_compatible) {
            const bool properties_changed = ggml_cuda_graph_update_required(cuda_ctx, cgraph);

            if (!graph->warmup_complete) {
                // Warmup: need at least 2 calls with no property change on the 2nd call
                if (!properties_changed) {
                    graph->warmup_complete = true;
                    GGML_LOG_DEBUG("%s: CUDA graph warmup complete\n", __func__);
                    use_cuda_graph = true;
                    cuda_graph_update_required = true;
                }
                // else: properties changed or first call - execute directly (use_cuda_graph stays false)
            } else {
                // Post-warmup: normal CUDA graph operation
                if (properties_changed) {
                    // Properties changed - reset warmup, execute directly until stable again
                    graph->warmup_complete = false;
                    GGML_LOG_DEBUG("%s: CUDA graph warmup reset\n", __func__);
                } else {
                    use_cuda_graph = true;
                    cuda_graph_update_required = graph->instance == nullptr;
                }
            }
        }
    }
#endif // USE_CUDA_GRAPH

    if (use_cuda_graph && cuda_graph_update_required) {
        // Start CUDA graph capture
        {
            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            ggml_cuda_lock_counter.fetch_add(1, std::memory_order_relaxed);
        }

        CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));
    }

    ggml_cuda_graph_evaluate_and_capture(cuda_ctx, cgraph, use_cuda_graph, cuda_graph_update_required, graph_key);

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cuda_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    CUDA_CHECK(cudaEventRecord((cudaEvent_t)event->context, cuda_ctx->stream()));
}

static void ggml_backend_cuda_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    if (ggml_backend_is_cuda(backend)) {
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), (cudaEvent_t)event->context, 0));
    } else {
#if 0
        // untested
        auto wait_fn = [](void * user_data) {
            ggml_backend_event_t event = (ggml_backend_event_t)user_data;
            ggml_backend_event_synchronize(event);
        };

        CUDA_CHECK(cudaLaunchHostFunc(cuda_ctx->stream(), wait_fn, event));
#endif
        GGML_ABORT("fatal error");
    }
}

static void ggml_backend_cuda_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    static const bool disable_fusion = getenv("GGML_CUDA_DISABLE_FUSION") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_FUSION"));

    auto add_alloc_deps = [&](size_t start, size_t last_node) {

        for (size_t i = start; i < last_node; ++i) {
            params->add_alloc_dep(params->user_data, cgraph->nodes[i], cgraph->nodes[last_node]);

            for (int j = 0; j < GGML_MAX_SRC; ++j) {
                if (cgraph->nodes[i]->src[j]) {
                    params->add_alloc_dep(params->user_data, cgraph->nodes[i]->src[j], cgraph->nodes[last_node]);
                }
            }
        }
    };

    if (!disable_fusion) {
        // add alloc deps for performance positive fusions. This may increase the overall compute buffer size.
        // TODO: consolidate fusion paths in graph_optimize and graph_compute
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            const auto cublas_affine = ggml_cuda_match_cublas_affine(*cuda_ctx, cgraph, i, false);
            if (cublas_affine.count) {
                ggml_tensor * terminal = cublas_affine.affine.tail ? cublas_affine.affine.tail : cublas_affine.affine.post ? cublas_affine.affine.post : cublas_affine.affine.unary;
                for (ggml_tensor * read : {cublas_affine.mm->src[0], cublas_affine.mm->src[1], cublas_affine.affine.mul->src[1], cublas_affine.affine.add->src[1]}) {
                    params->add_alloc_dep(params->user_data, read, terminal);
                }
            }
            const auto unary = ggml_cuda_match_cublas_unary(*cuda_ctx, cgraph, i, false);
            if (unary.count) {
                params->add_alloc_dep(params->user_data, unary.mm->src[0], unary.unary.last);
                params->add_alloc_dep(params->user_data, unary.mm->src[1], unary.unary.last);
            }
            auto hc_up = ggml_cuda_match_hc_up_shape(*cuda_ctx, cgraph, i);
            if (!hc_up.count) { hc_up = ggml_cuda_match_hc_up_shape(*cuda_ctx, cgraph, i, true); }
            if (hc_up.count) {
                params->add_alloc_dep(params->user_data, hc_up.mm->src[0], hc_up.dst);
                params->add_alloc_dep(params->user_data, hc_up.mm->src[1], hc_up.dst);
            }

            const auto repeated = ggml_cuda_match_repeat_mul_add(*cuda_ctx, cgraph, i, false);
            if (repeated.count) {
                for (ggml_tensor * read : {repeated.repeat->src[0], repeated.repeat, repeated.binary.mul->src[0], repeated.binary.mul->src[1], repeated.binary.add->src[repeated.binary.add->src[0] == repeated.binary.mul ? 1 : 0]}) {
                    params->add_alloc_dep(params->user_data, read, repeated.binary.add);
                }
                i += repeated.count - 1;
                continue;
            }
            const auto affine_injection = ggml_cuda_match_hc_affine_injection(*cuda_ctx, cgraph, i, false);
            if (affine_injection.count) {
                ggml_tensor * terminal = cgraph->nodes[i + affine_injection.count - 1];
                ggml_tensor * unused_norm = affine_injection.hc.mul || affine_injection.hc.scale ? affine_injection.hc.norm : nullptr;
                for (int j = i; j < i + affine_injection.count - 1; ++j) {
                    if (cgraph->nodes[j] != unused_norm) { params->add_alloc_dep(params->user_data, cgraph->nodes[j], terminal); }
                    for (int k = 0; k < GGML_MAX_SRC; ++k) {
                        if (cgraph->nodes[j]->src[k]) { params->add_alloc_dep(params->user_data, cgraph->nodes[j]->src[k], terminal); }
                    }
                }
                for (ggml_tensor * src : terminal->src) {
                    if (src && src != unused_norm) { params->add_alloc_dep(params->user_data, src, terminal); }
                }
                i += affine_injection.count - 1;
                continue;
            }
            auto affine = ggml_cuda_match_affine_unary(*cuda_ctx, cgraph, i, false, true);
            if (!affine.count) { affine = ggml_cuda_match_affine_unary(*cuda_ctx, cgraph, i, false); }
            if (affine.count) {
                ggml_tensor * last = cgraph->nodes[i + affine.count - 1];
                params->add_alloc_dep(params->user_data, affine.mul->src[0], last);
                params->add_alloc_dep(params->user_data, affine.mul->src[1], last);
                params->add_alloc_dep(params->user_data, affine.add->src[1], last);
                i += affine.count - 1;
                continue;
            }

            const auto binary = ggml_cuda_match_mul_add(*cuda_ctx, cgraph, i, false);
            if (binary.count) {
                for (ggml_tensor * read : {binary.mul->src[0], binary.mul->src[1], binary.add->src[binary.add->src[0] == binary.mul ? 1 : 0]}) {
                    params->add_alloc_dep(params->user_data, read, binary.add);
                }
                i += binary.count - 1;
                continue;
            }

            const auto injection = ggml_cuda_match_hc_injection(*cuda_ctx, cgraph, i, false);
            if (injection.count) {
                ggml_tensor * terminal = cgraph->nodes[i + injection.count - 1];
                params->add_alloc_dep(params->user_data, injection.first->src[0], terminal);
                for (ggml_tensor * tensor : {injection.first, injection.unary, injection.last}) {
                    if (tensor != terminal) { params->add_alloc_dep(params->user_data, tensor, terminal); }
                }
                for (ggml_tensor * tensor : {injection.post->src[0], injection.post->src[1], injection.post->src[3]}) {
                    if (tensor) { params->add_alloc_dep(params->user_data, tensor, terminal); }
                }
                if (injection.hc.mul) {
                    params->add_alloc_dep(params->user_data, injection.hc.mul->src[injection.hc.mul->src[0] == injection.hc.norm ? 1 : 0], terminal);
                }
                i += injection.count - 1;
                continue;
            }
            if (cgraph->nodes[i]->op == GGML_OP_DSV4_HC_POST) {
                const auto match = ggml_cuda_match_hc_post_norm(cgraph, i);
                if (match.count != 0) {
                    for (int j = i; j < i + match.count; ++j) {
                        for (const ggml_tensor * src : cgraph->nodes[j]->src) {
                            bool internal = false;
                            for (int k = i; src && k < j; ++k) {
                                internal |= src == cgraph->nodes[k];
                            }
                            if (src && !internal) {
                                params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(src), cgraph->nodes[i + match.count - 1]);
                            }
                        }
                    }
                    i += match.count - 1;
                    continue;
                }
            }
            ggml_cuda_moe_weighted_reduction_match match;
            if (ggml_cuda_match_moe_weighted_reduction(cgraph, i, match)) {
                params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(match.experts), match.dst);
                params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(match.weights), match.dst);
                if (match.expert_scale != nullptr) {
                    params->add_alloc_dep(
                        params->user_data, const_cast<ggml_tensor *>(match.expert_scale), match.dst);
                }
                i += match.node_count - 1;
            }

            if (cgraph->nodes[i]->op == GGML_OP_UNARY || cgraph->nodes[i]->op == GGML_OP_SOFT_MAX ||
                    cgraph->nodes[i]->op == GGML_OP_ARGSORT) {
                ggml_cuda_topk_moe_args args;
                const bool              can_fuse = ggml_cuda_topk_moe_fusion(cgraph, i, args);
                std::vector<ggml_op>    ops;
                ops.reserve(13);  // max ops; avoids gcc -Wstringop-overflow false positive

                const ggml_tensor * node = cgraph->nodes[i];

                if (can_fuse) {
                    const ggml_tensor * logits  = node->src[0];
                    ggml_tensor *       weights = nullptr;
                    ggml_tensor *       ids     = nullptr;

                    if (!args.delayed_softmax) {
                        int out_nodes[2];  // nodes which can't be elided

                        if (args.sigmoid) {
                            ops.insert(ops.end(), { GGML_OP_UNARY });
                        } else if (args.sqrt_softplus) {
                            ops.insert(ops.end(), { GGML_OP_UNARY, GGML_OP_SQRT });
                        } else {
                            ops.insert(ops.end(), { GGML_OP_SOFT_MAX });
                        }
                        const int i_probs = i + (int) ops.size() - 1;  // last node of the gating activation

                        if (args.prob_bias) {
                            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW,
                                                    GGML_OP_GET_ROWS });
                            out_nodes[0] = i_probs + 4;
                        } else {
                            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
                            out_nodes[0] = i_probs + 3;
                        }
                        ids = cgraph->nodes[out_nodes[0]];

                        if (args.norm) {
                            ops.insert(ops.end(),
                                       { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
                        }
                        if (args.scale) {
                            ops.insert(ops.end(), { GGML_OP_SCALE });
                        }

                        weights      = cgraph->nodes[i + ops.size() - 1];
                        out_nodes[1] = i + ops.size() - 1;

                        if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                                ggml_cuda_should_use_topk_moe(node, logits, weights, ids)) {

                            add_alloc_deps(i, i + ops.size());
                            i += ops.size() - 1;
                        }
                    }
                }
            }
        }
    }

#ifdef USE_CUDA_GRAPH
    const void * graph_key = ggml_cuda_graph_get_key(cgraph);
    const bool use_cuda_graph = ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);
#else
    const bool use_cuda_graph = false;
    GGML_UNUSED(cuda_ctx);
    GGML_UNUSED(cgraph);
#endif

    static bool enable_graph_optimization = [] {
        const char * env     = getenv("GGML_CUDA_GRAPH_OPT");
        return env != nullptr && atoi(env) == 1;
    }();

    if (!enable_graph_optimization) {
        return;
    }

    ggml_cuda_stream_context & stream_context = cuda_ctx->stream_context();
    stream_context.reset();

    if (!use_cuda_graph) {
        return;
    }

    ggml_cuda_set_device(cuda_ctx->device);

    // number of out-degrees for a particular node
    std::unordered_map<const ggml_tensor *, int> fan_out;
    // reverse mapping of node to index in the cgraph
    std::unordered_map<const ggml_tensor *, int> node_indices;

    const auto & is_noop = [](const ggml_tensor * node) -> bool {
        return ggml_is_empty(node) || node->op == GGML_OP_NONE || node->op == GGML_OP_RESHAPE ||
               node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE;
    };

    const auto & depends_on = [](const ggml_tensor * dst, const ggml_tensor * src) -> bool {
        for (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
            if (dst->src[s] == src) {
                return true;
            }
        }
        // implicit dependency if they view the same tensor
        const ggml_tensor * dst2 = dst->view_src ? dst->view_src : dst;
        const ggml_tensor * src2 = src->view_src ? src->view_src : src;
        if (dst2 == src2) {
            return true;
        }
        return false;
    };

    for (int node_idx = 0; node_idx < cgraph->n_nodes; node_idx++) {
        const ggml_tensor * node = cgraph->nodes[node_idx];
        node_indices[node]       = node_idx;

        if (is_noop(node)) {
            continue;
        }
        for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
            const ggml_tensor * src = cgraph->nodes[node_idx]->src[src_idx];
            //TODO: check why nrows > 1 fails
            if (node && !is_noop(node) && ggml_nrows(node) <= 1) {
                fan_out[src] += 1;
            }
        }
    }

    // Target Q, K, V for concurrency
    // this is a more general way to find nodes which can be candidates for concurrency (although it has not been tested for anything else):
    // 1. find fan-out (fork) nodes where the same input is used at least N times (in QKV, it would be "attn-norm")
    // 2. find the join node, where 2 or more of the outputs are required (in QKV, this would "KQ" or "flash-attn")
    // 3. account for all branches from the fork to the join
    // 4. To extend lifetimes of the tensors, we interleave the branches (see below for more details)
    // 5. save the original cgraph and restore it in graph_compute, to enable fusion within streams
    // See discussion: https://github.com/ggml-org/llama.cpp/pull/16991#issuecomment-3522620030

    const int min_fan_out = 3;
    const int max_fan_out = 3;

    // store {fork_idx, join_idx}
    std::vector<std::pair<int, int>> concurrent_node_ranges;

    for (const auto & [root_node, count] : fan_out) {
        if (count >= min_fan_out && count <= max_fan_out) {
            const int root_node_idx = node_indices[root_node];

            // only optimize for attn_norm
            // TODO: make this more generic
            if (!strstr(root_node->name, "attn_norm")) {
                continue;
            }

            bool is_part_of_event = false;
            for (const auto & [start, end] : concurrent_node_ranges) {
                if (root_node_idx >= start && root_node_idx <= end) {
                    is_part_of_event = true;
                }
            }

            if (is_part_of_event) {
                continue;
            }

            std::vector<std::vector<const ggml_tensor *>> nodes_per_branch;
            for (int i = root_node_idx + 1; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * node = cgraph->nodes[i];
                if (!is_noop(node) && depends_on(node, root_node)) {
                    nodes_per_branch.push_back({ node });
                }
            }

            GGML_ASSERT(nodes_per_branch.size() == (size_t) count);

            //find the join point
            const ggml_tensor * join_node = nullptr;

            const auto & belongs_to_branch = [&](const ggml_tensor *                      node,
                                                 const std::vector<const ggml_tensor *> & branch) -> bool {
                for (const ggml_tensor * n : branch) {
                    if (depends_on(node, n)) {
                        return true;
                    }
                }
                return false;
            };

            for (int i = root_node_idx + 1; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * curr_node = cgraph->nodes[i];

                int num_joins = 0;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    if (belongs_to_branch(curr_node, nodes_per_branch[branch_idx])) {
                        num_joins++;
                    }
                }

                if (num_joins >= 2) {
                    join_node = curr_node;
                    break;
                }

                bool found_branch = false;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    std::vector<const ggml_tensor *> & branch_vec = nodes_per_branch[branch_idx];
                    if (belongs_to_branch(curr_node, branch_vec)) {
                        //continue accumulating
                        if (std::find(branch_vec.begin(), branch_vec.end(), curr_node) == branch_vec.end()) {
                            branch_vec.push_back(curr_node);
                        }
                        found_branch = true;
                    }
                }

                if (!found_branch && is_noop(curr_node)) {
                    // we can put it in any branch because it will be ignored
                    nodes_per_branch[0].push_back({ curr_node });
                }
            }

            if (join_node) {
                //Create ggml_cuda_concurrent_event
                ggml_cuda_concurrent_event concurrent_event(nodes_per_branch.size());
                concurrent_event.join_node = join_node;

                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    for (const ggml_tensor * n : nodes_per_branch[branch_idx]) {
                        concurrent_event.stream_mapping[n] = branch_idx + 1;
                    }
                }

                int fork_node_idx = node_indices[root_node];
                int join_node_idx = node_indices[join_node];

                int       current_branch_idx = 0;
                int       current_node_idx   = fork_node_idx + 1;
                const int n_branches         = nodes_per_branch.size();

                int total_branch_nodes = 0;
                for (std::vector<const ggml_tensor *> branch_nodes : nodes_per_branch) {
                    total_branch_nodes += branch_nodes.size();
                }

                // there are other nodes in the middle which are unaccounted for
                // usually (cpy) nodes, then ignore this fork
                if (join_node_idx - fork_node_idx - 1 != total_branch_nodes) {
                    GGML_LOG_DEBUG(
                        "Skipping %s because the number of nodes in the middle is not equal to the total number of "
                        "branch nodes %d != %d\n",
                        root_node->name, join_node_idx - fork_node_idx - 1, total_branch_nodes);
                    continue;
                }

                // Save the original order of nodes in this region before interleaving
                // This is used later to restore grouping for fusion within streams
                concurrent_event.original_order.reserve(total_branch_nodes);
                for (int i = fork_node_idx + 1; i < join_node_idx; ++i) {
                    concurrent_event.original_order.push_back(cgraph->nodes[i]);
                }

                std::unordered_map<const ggml_tensor *, ggml_cuda_concurrent_event> & concurrent_events = cuda_ctx->stream_context().concurrent_events;
                GGML_ASSERT(concurrent_events.find(root_node) == concurrent_events.end());
                concurrent_events.emplace(root_node, std::move(concurrent_event));
                GGML_LOG_DEBUG("Adding stream at node %s %p\n", root_node->name, root_node);
                concurrent_node_ranges.emplace_back(fork_node_idx, join_node_idx);

                // interleave tensors to extend lifetimes so that ggml graph doesn't recycle them
                // example transformation:
                // [attn-norm, QMul, QNorm, QRope, KMul, KNorm, KRope, VMul, attn] ->
                // [attn-norm, QMul, KMul, VMul, QNorm, VNorm, QRope, KRope, attn]
                while (current_node_idx < join_node_idx) {
                    std::vector<const ggml_tensor *> & branch_nodes = nodes_per_branch[current_branch_idx];

                    bool has_node = false;
                    for (std::vector<const ggml_tensor *> branch_node : nodes_per_branch) {
                        has_node |= branch_node.size() > 0;
                    }

                    GGML_ASSERT(has_node);

                    if (branch_nodes.empty()) {
                        current_branch_idx = (current_branch_idx + 1) % n_branches;
                        continue;
                    }

                    cgraph->nodes[current_node_idx] = const_cast<ggml_tensor *>(branch_nodes.front());
                    current_node_idx++;
                    branch_nodes.erase(branch_nodes.begin());

                    // append all empty nodes
                    while (!branch_nodes.empty() && is_noop(branch_nodes.front())) {
                        cgraph->nodes[current_node_idx] = const_cast<ggml_tensor *>(branch_nodes.front());
                        current_node_idx++;
                        branch_nodes.erase(branch_nodes.begin());
                    }

                    current_branch_idx = (current_branch_idx + 1) % n_branches;
                }
            }
        }
    }
}

static const ggml_backend_i ggml_backend_cuda_interface = {
    /* .get_name                = */ ggml_backend_cuda_get_name,
    /* .free                    = */ ggml_backend_cuda_free,
    /* .set_tensor_async        = */ ggml_backend_cuda_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_cuda_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_cuda_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_cuda_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_cuda_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_cuda_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_cuda_graph_compute,
    /* .event_record            = */ ggml_backend_cuda_event_record,
    /* .event_wait              = */ ggml_backend_cuda_event_wait,
    /* .graph_optimize          = */ ggml_backend_cuda_graph_optimize,
};

static ggml_guid_t ggml_backend_cuda_guid() {
    static ggml_guid guid = { 0x2c, 0xdd, 0xe8, 0x1c, 0x65, 0xb3, 0x65, 0x73, 0x6a, 0x12, 0x88, 0x61, 0x1c, 0xc9, 0xdc, 0x25 };
    return &guid;
}

bool ggml_backend_is_cuda(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_cuda_guid());
}

int ggml_backend_cuda_get_device_count() {
    return ggml_cuda_info().device_count;
}

static std::string ggml_cuda_device_description(int device) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(device)));

    const ggml_cuda_device_info & info = ggml_cuda_info();
    std::string description = prop.name;
    if (info.device_count > info.physical_device_count) {
        description += " (dev p" + std::to_string(info.devices[device].physical_device) +
                       "/v" + std::to_string(info.devices[device].virtual_index) + ")";
    }
    return description;
}

void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size) {
    snprintf(description, description_size, "%s", ggml_cuda_device_description(device).c_str());
}

static int ggml_cuda_physical_device_share_count(int device) {
    const ggml_cuda_device_info & info = ggml_cuda_info();
    GGML_ASSERT(device >= 0 && device < info.device_count);
    return info.devices[device].physical_share_count;
}

void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total) {
    ggml_cuda_set_device(device);

    CUDA_CHECK(cudaMemGetInfo(free, total));

    // virtual devices sharing one physical GPU share its memory pool; split it between them
    const int share_count = ggml_cuda_physical_device_share_count(device);
    *free  /= share_count;
    *total /= share_count;
}

bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size) {
    if (getenv("GGML_CUDA_REGISTER_HOST") == nullptr) {
        return false;
    }

#if CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA) || defined(GGML_USE_HIP)
    cudaError_t err = cudaHostRegister(buffer, size, cudaHostRegisterPortable | cudaHostRegisterReadOnly);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();

        GGML_LOG_DEBUG("%s: failed to register %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return false;
    }
    return true;
#else
    GGML_UNUSED(buffer);
    GGML_UNUSED(size);
    return false;
#endif // CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA)
}

void ggml_backend_cuda_unregister_host_buffer(void * buffer) {
    if (getenv("GGML_CUDA_REGISTER_HOST") == nullptr) {
        return;
    }

    cudaError_t err = cudaHostUnregister(buffer);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
    }
}


// backend device

struct ggml_backend_cuda_device_context {
    int device;
    std::string name;
    std::string description;
    std::string pci_bus_id;
    int op_offload_min_batch_size;
};

static const char * ggml_backend_cuda_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_cuda_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->description.c_str();
}

#if defined(__linux__)
// Helper function to get available memory from /proc/meminfo for UMA systems
static bool ggml_backend_cuda_get_available_uma_memory(long * available_memory_kb, long * free_swap_kb) {
    FILE * meminfo_file = nullptr;
    // 2KB buffer for reading /proc/meminfo since it does not report size info, should be enough
    const size_t BUFFER_SIZE = 2048;
    auto file_buffer = std::make_unique<char[]>(BUFFER_SIZE);
    size_t bytes_read = 0;
    long huge_tlb_total_pages = -1;
    long huge_tlb_free_pages = -1;
    long huge_tlb_page_size = -1;

    if (available_memory_kb == nullptr || free_swap_kb == nullptr) {
        return false;
    }

    meminfo_file = fopen("/proc/meminfo", "r");
    if (meminfo_file == nullptr) {
        GGML_LOG_ERROR("%s: failed to open /proc/meminfo\n", __func__);
        return false;
    }

    // Read file into buffer
    bytes_read = fread(file_buffer.get(), 1, BUFFER_SIZE - 1, meminfo_file);
    fclose(meminfo_file);

    if (bytes_read == 0) {
        GGML_LOG_ERROR("%s: failed to read from /proc/meminfo\n", __func__);
        return false;
    }
    file_buffer[bytes_read] = '\0';

    *available_memory_kb = -1;
    *free_swap_kb = -1;

    // Parse the file buffer line by line
    char * line = file_buffer.get();
    char * line_next;
    while (line < file_buffer.get() + bytes_read) {
        // Find the end of the current line
        line_next = strchr(line, '\n');
        if (line_next != nullptr) {
            *line_next = '\0';
            line_next++;
        } else {
            line_next = file_buffer.get() + bytes_read;
        }

        long value;
        if (sscanf(line, "MemAvailable: %ld kB", &value) == 1) {
            *available_memory_kb = value;
        } else if (sscanf(line, "SwapFree: %ld kB", &value) == 1) {
            *free_swap_kb = value;
        } else if (sscanf(line, "HugePages_Total: %ld", &value) == 1) {
            huge_tlb_total_pages = value;
        } else if (sscanf(line, "HugePages_Free: %ld", &value) == 1) {
            huge_tlb_free_pages = value;
        } else if (sscanf(line, "Hugepagesize: %ld kB", &value) == 1) {
            huge_tlb_page_size = value;
        }

        line = line_next;
    }

    if (huge_tlb_total_pages != 0 && huge_tlb_total_pages != -1) {
        *available_memory_kb = huge_tlb_free_pages * huge_tlb_page_size;

        // Hugetlbfs pages are not swappable.
        *free_swap_kb = 0;
    }

    GGML_LOG_DEBUG("%s: final available_memory_kb: %ld\n", __func__, *available_memory_kb);
    return true;
}
#endif // defined(__linux__)

static void ggml_backend_cuda_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    ggml_cuda_set_device(ctx->device);
    cudaError_t err = cudaMemGetInfo(free, total);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        GGML_LOG_WARN("%s: cudaMemGetInfo failed (%s), returning 0/0\n", __func__, cudaGetErrorString(err));
        *free = 0;
        *total = 0;
        return;
    }

// ref: https://github.com/ggml-org/llama.cpp/pull/17368
#if defined(__linux__) && !defined(GGML_USE_HIP)
    // Check if this is a UMA (Unified Memory Architecture) system
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(ctx->device)));

    // Check if UMA is explicitly enabled via environment variable
    bool uma_env = getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != nullptr;
    bool is_uma = prop.integrated > 0 || uma_env;

    if (is_uma) {
        // For UMA systems (like DGX Spark), use system memory info
        long available_memory_kb = 0;
        long free_swap_kb = 0;

        if (ggml_backend_cuda_get_available_uma_memory(&available_memory_kb, &free_swap_kb) && available_memory_kb > 0) {
            *free = (size_t)available_memory_kb * 1024;
        } else {
            GGML_LOG_ERROR("%s: /proc/meminfo reading failed, using cudaMemGetInfo\n", __func__);
        }
    }
#endif // defined(__linux__) && !defined(GGML_USE_HIP)

    // virtual devices sharing one physical GPU share its memory pool; split it between them
    const int share_count = ggml_cuda_physical_device_share_count(ctx->device);
    *free  /= share_count;
    *total /= share_count;
}

static enum ggml_backend_dev_type ggml_backend_cuda_device_get_type(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *) dev->context;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(ctx->device)));

    return prop.integrated
        ? GGML_BACKEND_DEVICE_TYPE_IGPU
        : GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_cuda_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;

    props->name        = ggml_backend_cuda_device_get_name(dev);
    props->description = ggml_backend_cuda_device_get_description(dev);
    props->type        = ggml_backend_cuda_device_get_type(dev);
    props->device_id   = ctx->pci_bus_id.empty() ? nullptr : ctx->pci_bus_id.c_str();
    ggml_backend_cuda_device_get_memory(dev, &props->memory_free, &props->memory_total);

    bool host_buffer = getenv("GGML_CUDA_NO_PINNED") == nullptr;
#ifdef GGML_CUDA_NO_PEER_COPY
    bool events = false;
#else
    bool events = true;
#endif

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ host_buffer,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ events,
        /* .mmap_support          = */ props->type != GGML_BACKEND_DEVICE_TYPE_IGPU,
    };
}

static ggml_backend_t ggml_backend_cuda_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_backend_cuda_init(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cuda_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_backend_cuda_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cuda_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cuda_host_buffer_type();
}

// TODO: move these functions here
static bool ggml_backend_cuda_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    // check if all the sources are allocated on this device
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] && op->src[i]->buffer && ggml_backend_buft_is_cuda(op->src[i]->buffer->buft)) {
            ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)op->src[i]->buffer->buft->context;
            if (buft_ctx->device != dev_ctx->device) {
                return false;
            }
        }
    }

    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_XIELU:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                    if (op->src[0]->type == GGML_TYPE_BF16 && ggml_get_unary_op(op) == GGML_UNARY_OP_XIELU) {
                        return false;
                    }
                    // TODO: should become:
                    //return ggml_is_contiguous_rows(op->src[0]);
                    return ggml_is_contiguous(op->src[0]);
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    if (op->src[0]->type == GGML_TYPE_BF16 &&
                            (ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU_OAI || ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU_CLAMP)) {
                        return false;
                    }
                    return ggml_is_contiguous_1(op->src[0]);
                default:
                    return false;
            }
            break;
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            {
                struct ggml_tensor * a = op->src[0];
                struct ggml_tensor * b = op->src[1];
                if (a->nb[0] != ggml_element_size(a) || b->nb[0] != ggml_element_size(b)) {
                    return false; // TODO this could in principle be implemented though currently there is no use case.
                }
                if (b->type == GGML_TYPE_F16 && a->type != GGML_TYPE_F16 && !ggml_cuda_op_mul_mat_use_fwht(op)) {
                    return false;
                }
                if (op->op == GGML_OP_MUL_MAT_ID && ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
                    return false;
                }
#ifdef GGML_USE_MUSA
                const int cc = ggml_cuda_info().devices[dev_ctx->device].cc;
                if (b->ne[2]*b->ne[3] > 1 && !ggml_is_transposed(a) && !ggml_is_transposed(b)) {
                    if (GGML_CUDA_CC_IS_QY1(cc) && op->op == GGML_OP_MUL_MAT &&
                            a->type == GGML_TYPE_F16 && b->type == GGML_TYPE_F16) {
                        return false;
                    }
                    if (GGML_CUDA_CC_IS_QY2(cc) && op->op == GGML_OP_MUL_MAT_ID &&
                            a->type == GGML_TYPE_Q2_K && b->type == GGML_TYPE_F32) {
                        return false;
                    }
                }
#endif // GGML_USE_MUSA
                switch (a->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q2_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_MXFP4:
                    case GGML_TYPE_NVFP4:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_Q8_K:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_IQ4_XS:
                    case GGML_TYPE_BF16:
                        return true;
                    default:
                        return false;
                }
            } break;
        case GGML_OP_OUT_PROD:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_GET_ROWS:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F16:
                    case GGML_TYPE_F32:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_I32:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q2_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ4_XS:
                        return true;
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_MXFP4:
                        // 32-value sub-blocks, the row size does not guarantee
                        // the QK_K super-blocks the get_rows kernel iterates on
                        return op->src[0]->ne[0] % QK_K == 0;
                    default:
                        return false;
                }
            } break;
        case GGML_OP_GET_ROWS_BACK:
            {
                return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->ne[2] == 1 && op->ne[3] == 1;
            } break;
        case GGML_OP_SET_ROWS:
            {
                return (
                           (
                               (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_BF16 ||
                               op->type == GGML_TYPE_Q4_0 || op->type == GGML_TYPE_Q4_1 || op->type == GGML_TYPE_Q5_0 ||
                               op->type == GGML_TYPE_Q5_1 || op->type == GGML_TYPE_Q8_0 || op->type == GGML_TYPE_IQ4_NL) &&
                               op->src[0]->type == GGML_TYPE_F32
                           ) || (
                               op->type == GGML_TYPE_F16 && op->src[0]->type == GGML_TYPE_F16
                           )
                       ) &&
                       (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32);
            } break;
        case GGML_OP_SET:
            {
                const ggml_type t = op->type;
                return (t == GGML_TYPE_F32 || t == GGML_TYPE_I32) &&
                    t == op->src[0]->type &&
                    t == op->src[1]->type;
            } break;
        case GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if ((src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_BF16 || src0_type == GGML_TYPE_F16) &&
                    (src1_type == GGML_TYPE_F32 || src1_type == GGML_TYPE_BF16 || src1_type == GGML_TYPE_F16)
                ) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q8_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q8_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_IQ4_NL) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_I32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_I32) {
                    return true;
                }
                if (src0_type == src1_type && ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                    return true;
                }
                return false;
            } break;
        case GGML_OP_DUP:
                return true;
        case GGML_OP_ARGMAX:
        case GGML_OP_COUNT_EQUAL:
            {
                return true;
            } break;
        case GGML_OP_REPEAT:
            {
                // the CUDA REPEAT path only implements F32/F16; other types assert at runtime
                ggml_type src0_type = op->src[0]->type;
                return src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_F16;
            } break;
        case GGML_OP_REPEAT_BACK:
                return op->type == GGML_TYPE_F32 && (op->src[0]->ne[2]*op->src[0]->ne[3]) <= (1 << 15);
        case GGML_OP_CONCAT:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                const int32_t dim = op->op_params[0];
                return src0_type == src1_type &&
                       src0_type == op->type &&
                       (
                           (
                               ggml_is_quantized(src0_type) &&
                               (
                                   (
                                       dim == 3 &&
                                       ggml_is_contiguous(op->src[0]) &&
                                       ggml_is_contiguous(op->src[1])
                                   ) || (
                                       dim != 3 &&
                                       ggml_is_contiguous_to_3(op->src[0]) &&
                                       ggml_is_contiguous_to_3(op->src[1])
                                   )
                               ) &&
                               op->src[0]->ne[0] % ggml_blck_size(src0_type) == 0 &&
                               op->src[1]->ne[0] % ggml_blck_size(src0_type) == 0
                           ) || (
                               !ggml_is_quantized(src0_type) &&
                               ggml_blck_size(src0_type) == 1 &&
                               (
                                   ggml_type_size(src0_type) == 1 ||
                                   ggml_type_size(src0_type) == 2 ||
                                   ggml_type_size(src0_type) == 4 ||
                                   ggml_type_size(src0_type) == 8
                               )
                           )
                       );
            } break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                return false;
            } break;
        case GGML_OP_COL2IM_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                return (src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_F16 || src0_type == GGML_TYPE_BF16) &&
                    op->type == src0_type &&
                    ggml_is_contiguous(op->src[0]) &&
                    ggml_is_contiguous(op);
            } break;
        case GGML_OP_SILU_BACK:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
            break;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_L2_NORM:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_RMS_NORM_BACK:
            return ggml_is_contiguous(op->src[0]);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_ADD_ID:
        case GGML_OP_ADD1:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_CLAMP:
        case GGML_OP_LOG:
            return true;
        case GGML_OP_SCALE:
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_BF16) && op->type == op->src[0]->type;
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            if (op->src[0]->type == GGML_TYPE_BF16 || op->src[1]->type == GGML_TYPE_BF16 || op->type == GGML_TYPE_BF16) {
                return op->src[0]->type == GGML_TYPE_BF16 && op->type == GGML_TYPE_BF16 &&
                    (op->src[1]->type == GGML_TYPE_BF16 || op->src[1]->type == GGML_TYPE_F32);
            }
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) &&
                   (op->type         == GGML_TYPE_F32 || op->type         == GGML_TYPE_F16);
        case GGML_OP_SSM_SCAN: {
            const int32_t K = ggml_get_op_params_i32(op, 0);

            if (op->src[3]->ne[0] == 1) {
                // Mamba2
                // (kernel only supports (d_state == 96 || d_state == 128 || d_state == 256) && d_head % 16 == 0)
                const int64_t d_state = op->src[0]->ne[0];
                return (d_state == 96 || d_state == 128 || d_state == 256) && op->src[0]->ne[1] % 16 == 0;
            } else {
                if (K > 1) {
                    return false;
                }

                // Mamba
                // (kernel only supports d_state == 16, d_head == 1, n_head % 128 == 0, n_group == 1)
                return op->src[0]->ne[0] == 16 && op->src[0]->ne[1] == 1 && op->src[0]->ne[2] % 128 == 0 && op->src[4]->ne[1] == 1;
            }
        }
        case GGML_OP_SSM_CONV: {
            // assumes d_inner % threads == 0
            return op->src[0]->ne[1] % 128 == 0;
        }
        case GGML_OP_CONT:
            return true;
        case GGML_OP_DIAG_MASK_INF:
            return true;
        case GGML_OP_SOFT_MAX:
            return true;
        case GGML_OP_SOFT_MAX_BACK: {
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));
            return max_bias == 0.0f;
        }
        case GGML_OP_ROLL:
            if(op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0])) {
                return true;
            }
            return false;
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK: {
            return op->src[0]->nb[0] == ggml_type_size(op->src[0]->type) && ggml_is_contiguous_2(op->src[0]);
        }
        case GGML_OP_IM2COL:
        case GGML_OP_IM2COL_3D:
        case GGML_OP_CONV_2D:
            return (ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]));
        case GGML_OP_CONV_3D:
            return (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                   op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && ggml_is_contiguous(op);
        case GGML_OP_CONV_2D_DW:
            return (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                   op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_CONV_TRANSPOSE_2D:
        case GGML_OP_POOL_1D:
        case GGML_OP_POOL_2D:
            return true;
        case GGML_OP_ACC:
            // TODO: extend support like so:
            //return ggml_is_contiguous_rows(op->src[0]) && ggml_is_contiguous_rows(op->src[1]);
            return ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
        case GGML_OP_SUM:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_TOP_K:
#if defined(GGML_USE_HIP) || defined(GGML_CUDA_USE_CUB)
            return true;
#else
            return op->src[0]->ne[0] <= 1024;
#endif // defined(GGML_USE_HIP) || defined(GGML_CUDA_USE_CUB)
        case GGML_OP_ARGSORT:
#ifndef GGML_CUDA_USE_CUB
            {
                // bitonic path: the padded row must fit in shared memory
                int64_t ncols_pad = 1;
                while (ncols_pad < op->src[0]->ne[0]) {
                    ncols_pad *= 2;
                }
                return ncols_pad * sizeof(int) <= ggml_cuda_info().devices[dev_ctx->device].smpb;
            }
#else
            return true;
#endif
        case GGML_OP_SUM_ROWS:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_MEAN:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_GROUP_NORM:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_PAD:
            return true;
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_RWKV_WKV7:
            return true;
        case GGML_OP_GATED_DELTA_NET:
            return true;
        case GGML_OP_DSV4_HC_COMB:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_PRE:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_POST:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && (op->src[3] == nullptr || op->src[3]->type == GGML_TYPE_F32) &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_FLASH_ATTN_EXT:
            return ggml_cuda_flash_attn_ext_supported(dev_ctx->device, op);
        case GGML_OP_CROSS_ENTROPY_LOSS:
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
        case GGML_OP_FILL:
        case GGML_OP_CUMSUM:
        case GGML_OP_TRI:
        case GGML_OP_DIAG:
        case GGML_OP_SOLVE_TRI:
            return true;
        case GGML_OP_LIGHTNING_INDEXER:
            return ggml_cuda_lightning_indexer_supported(dev_ctx->device, op);

        default:
            return false;
    }
}

static bool ggml_backend_cuda_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;
    const bool integrated = ggml_cuda_info().devices[dev_ctx->device].integrated;
    return (ggml_backend_buft_is_cuda(buft) && buft->device == dev) || (integrated && ggml_backend_buft_is_cuda_host(buft));
}

static int64_t get_op_batch_size(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_GET_ROWS:
            return 0;
        case GGML_OP_MUL_MAT:
            return op->ne[1];
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            return op->ne[2];
        default:
            return ggml_nrows(op);
    }
}

static bool ggml_backend_cuda_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    return get_op_batch_size(op) >= dev_ctx->op_offload_min_batch_size;
}

static ggml_backend_event_t ggml_backend_cuda_device_event_new(ggml_backend_dev_t dev) {
#ifdef GGML_CUDA_NO_PEER_COPY
    GGML_UNUSED(dev);
    return nullptr;
#else
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *)dev->context;

    ggml_cuda_set_device(dev_ctx->device);

    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

    return new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event,
    };
#endif
}

static void ggml_backend_cuda_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);

    CUDA_CHECK(cudaEventDestroy((cudaEvent_t)event->context));
    delete event;
}

static void ggml_backend_cuda_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    CUDA_CHECK(cudaEventSynchronize((cudaEvent_t)event->context));
}

static const ggml_backend_device_i ggml_backend_cuda_device_interface = {
    /* .get_name                = */ ggml_backend_cuda_device_get_name,
    /* .get_description         = */ ggml_backend_cuda_device_get_description,
    /* .get_memory              = */ ggml_backend_cuda_device_get_memory,
    /* .get_type                = */ ggml_backend_cuda_device_get_type,
    /* .get_props               = */ ggml_backend_cuda_device_get_props,
    /* .init_backend            = */ ggml_backend_cuda_device_init_backend,
    /* .get_buffer_type         = */ ggml_backend_cuda_device_get_buffer_type,
    /* .get_host_buffer_type    = */ ggml_backend_cuda_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ NULL,
    /* .supports_op             = */ ggml_backend_cuda_device_supports_op,
    /* .supports_buft           = */ ggml_backend_cuda_device_supports_buft,
    /* .offload_op              = */ ggml_backend_cuda_device_offload_op,
    /* .event_new               = */ ggml_backend_cuda_device_event_new,
    /* .event_free              = */ ggml_backend_cuda_device_event_free,
    /* .event_synchronize       = */ ggml_backend_cuda_device_event_synchronize,
};

// backend reg

struct ggml_backend_cuda_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_cuda_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_CUDA_NAME;
}

static size_t ggml_backend_cuda_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_cuda_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static ggml_backend_feature * ggml_backend_cuda_get_features(ggml_backend_reg_t reg) {
    static std::vector<ggml_backend_feature> features = []() {
        std::vector<ggml_backend_feature> features;
    #define _STRINGIFY(...) #__VA_ARGS__
    #define STRINGIFY(...) _STRINGIFY(__VA_ARGS__)

    #ifdef __CUDA_ARCH_LIST__
        features.push_back({ "ARCHS", STRINGIFY(__CUDA_ARCH_LIST__) });
    #endif

    #ifdef GGML_CUDA_FORCE_MMQ
        features.push_back({ "FORCE_MMQ", "1" });
    #endif

    #ifdef GGML_CUDA_FORCE_CUBLAS
        features.push_back({ "FORCE_CUBLAS", "1" });
    #endif

    #ifndef GGML_USE_VMM
        features.push_back({ "NO_VMM", "1" });
    #endif

    #ifdef GGML_CUDA_NO_PEER_COPY
        features.push_back({ "NO_PEER_COPY", "1" });
    #endif

    #ifdef GGML_CUDA_USE_GRAPHS
        features.push_back({ "USE_GRAPHS", "1" });
    #endif

    #ifdef GGML_CUDA_FA_QUANTS
        features.push_back({ "FA_QUANTS", GGML_CUDA_FA_QUANTS });
    #endif

    {
        const auto & info = ggml_cuda_info();
        for (int id = 0; id < info.device_count; ++id) {
            if (blackwell_mma_available(info.devices[id].cc)) {
                features.push_back({ "BLACKWELL_NATIVE_FP4", "1"});
                break;
            }
        }
    }

    #undef _STRINGIFY
    #undef STRINGIFY

        features.push_back({ nullptr, nullptr });

        return features;
    }();

    return features.data();

    GGML_UNUSED(reg);
}

static void * ggml_backend_cuda_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void *)ggml_backend_cuda_comm_init;
    }
    if (strcmp(name, "ggml_backend_comm_free") == 0) {
        return (void *)ggml_backend_cuda_comm_free;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        return (void *)ggml_backend_cuda_comm_allreduce_tensor;
    }
    if (strcmp(name, "ggml_backend_register_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_register_host_buffer;
    }
    if (strcmp(name, "ggml_backend_unregister_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_unregister_host_buffer;
    }
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_cuda_get_features;
    }
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_cuda_reg_interface = {
    /* .get_name          = */ ggml_backend_cuda_reg_get_name,
    /* .get_device_count  = */ ggml_backend_cuda_reg_get_device_count,
    /* .get_device        = */ ggml_backend_cuda_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_cuda_reg_get_proc_address,
};

// backend registry
ggml_backend_reg_t ggml_backend_cuda_reg() {
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_backend_cuda_reg_context * ctx = new ggml_backend_cuda_reg_context;
            const int min_batch_size = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

            const ggml_cuda_device_info & info = ggml_cuda_info();
            const bool virtual_devices = info.device_count > info.physical_device_count;

            for (int i = 0; i < info.device_count; i++) {
                const int physical_id = info.devices[i].physical_device;

                ggml_backend_cuda_device_context * dev_ctx = new ggml_backend_cuda_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_CUDA_NAME + std::to_string(i);
                dev_ctx->description = ggml_cuda_device_description(i);

                char pci_bus_id[32] = {};
                CUDA_CHECK(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), physical_id));
                dev_ctx->pci_bus_id = pci_bus_id;
                if (virtual_devices) {
                    // make the pci bus id unique for virtual devices
                    dev_ctx->pci_bus_id += "-v" + std::to_string(i);
                }
                for (char & c : dev_ctx->pci_bus_id) {
                    c = std::tolower(c);
                }
                dev_ctx->op_offload_min_batch_size = min_batch_size;

                ggml_backend_dev_t dev = new ggml_backend_device {
                    /* .iface   = */ ggml_backend_cuda_device_interface,
                    /* .reg     = */ &reg,
                    /* .context = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_cuda_reg_interface,
                /* .context     = */ ctx
            };
        }

        initialized = true;
    }

    return &reg;
}

ggml_backend_t ggml_backend_cuda_init(int device) {
    if (device < 0 || device >= ggml_backend_cuda_get_device_count()) {
        GGML_LOG_ERROR("%s: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_cuda_context * ctx = new ggml_backend_cuda_context(device);
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: failed to allocate context\n", __func__);
        return nullptr;
    }

    ggml_backend_t cuda_backend = new ggml_backend {
        /* .guid    = */ ggml_backend_cuda_guid(),
        /* .iface   = */ ggml_backend_cuda_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), device),
        /* .context = */ ctx,
    };

    return cuda_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cuda_reg)
