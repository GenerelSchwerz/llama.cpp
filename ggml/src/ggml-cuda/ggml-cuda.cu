#include "ggml-cuda.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-cuda/allreduce.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/reuse.cuh"
#include "ggml-cuda/staged-input.cuh"
#include "ggml-cuda/moe-fidelity.cuh"
#include "ggml-cuda/moe-fidelity-executor.cuh"
#include "ggml-cuda/moe-source-core.cuh"
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
#include "ggml-cuda/mmid.cuh"
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
#include "ggml-cuda/moe-cache.cuh"
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
#include <cstring>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include <cuda/atomic>
#endif

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
    if (getenv("GGML_MOE_FIDELITY_TRACE")) {
        fprintf(stderr, GGML_CUDA_NAME " error: %s\n  current device: %d, in function %s at %s:%d\n  %s\n", msg, id, func, file, line, stmt);
        fflush(stderr);
    }
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
#ifndef GGML_USE_MUSA
        for (int dim = 0; dim < 3; ++dim) {
            info.devices[id].max_grid_size[dim] = prop.maxGridSize[dim];
        }
#endif

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
            if (pool_size > 0) {
                CU_CHECK(cuMemUnmap(pool_addr, pool_size));
            }
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

    size_t trim() override {
        if (pool_addr == 0 || pool_size == 0) {
            return 0;
        }

#if defined(GGML_USE_HIP)
        if (pool_used != 0) {
            return 0;
        }
        ggml_cuda_set_device(device);
        for (std::pair<CUdeviceptr, size_t> & mapping : mappings) {
            CU_CHECK(cuMemUnmap(mapping.first, mapping.second));
        }
        mappings.clear();
        const size_t released = pool_size;
        pool_size = 0;
#else
        const size_t keep_size = granularity * ((pool_used + granularity - 1) / granularity);
        if (keep_size >= pool_size) {
            return 0;
        }
        const size_t released = pool_size - keep_size;
        ggml_cuda_set_device(device);
        CU_CHECK(cuMemUnmap(pool_addr + keep_size, released));
        pool_size = keep_size;
#endif
        return released;
    }
};
#endif // defined(GGML_USE_VMM)

static uint64_t ggml_backend_cuda_trim_transient_pools(ggml_backend_t backend) {
    auto * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    // Pool frees precede this call, but kernels and copies are asynchronous.
    for (int device = 0; device < GGML_CUDA_MAX_DEVICES; ++device) {
        for (int stream = 0; stream < GGML_CUDA_MAX_STREAMS; ++stream) {
            if (cuda_ctx->streams[device][stream] != nullptr) {
                ggml_cuda_set_device(device);
                CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->streams[device][stream]));
            }
        }
    }

    // Cached graphs may reference pool mappings released by trim().
    ggml_cuda_set_device(cuda_ctx->device);
    cuda_ctx->cuda_graphs.clear();
    ++cuda_ctx->workspace_generation;

    uint64_t released = 0;
    for (int device = 0; device < GGML_CUDA_MAX_DEVICES; ++device) {
        for (int stream = 0; stream < GGML_CUDA_MAX_STREAMS; ++stream) {
            if (cuda_ctx->pools[device][stream] != nullptr) {
                released += cuda_ctx->pools[device][stream]->trim();
            }
        }
    }
    return released;
}

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
            int can_access_peer = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, src_physical, dst_physical));
            if (!can_access_peer) {
                return false;
            }
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

static bool ggml_cuda_mul_mat_cublas_f32_output(int cc, ggml_type compute_type) {
    if (compute_type == GGML_TYPE_F16) {
        return cc == GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_CDNA(cc);
    }
    if (compute_type == GGML_TYPE_BF16) {
        return !GGML_CUDA_CC_IS_RDNA3(cc) && !GGML_CUDA_CC_IS_CDNA(cc);
    }
    return false;
}

template<ggml_type compute_type>
static void ggml_cuda_mul_mat_cublas_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const void * prepared_src1 = nullptr) {
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
    const bool prefer_f32_output = ggml_cuda_mul_mat_cublas_f32_output(cc, compute_type);

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
        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(traits::ggml_type_val);
        to_fp32_cuda(dst_temp.get(), dst_ddf, ne_dst, main_stream);
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
        const auto matches = [env_c](const char * value) {
            for (size_t i = 0; ; ++i) {
                if (std::tolower(static_cast<unsigned char>(env_c[i])) != value[i]) { return false; }
                if (!value[i]) { return true; }
            }
        };
        if (matches("f32") || matches("fp32")) {
            compute_type = GGML_TYPE_F32;
        } else if (matches("f16") || matches("fp16")) {
            compute_type = GGML_TYPE_F16;
        } else if (matches("bf16")) {
            compute_type = GGML_TYPE_BF16;
        } else if (warn && !matches("auto")) {
            GGML_LOG_WARN("%s: unknown value for GGML_CUDA_CUBLAS_COMPUTE_TYPE: %s", "ggml_cuda_mul_mat_cublas", env_c);
        }
    }

    return compute_type;
}

static void ggml_cuda_mul_mat_cublas(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const void * prepared_src1 = nullptr) {
    const ggml_type compute_type = ggml_cuda_mul_mat_cublas_compute_type(ggml_cuda_info().devices[ctx.device].cc, src0, src1, dst, true);
    switch (compute_type) {
        case GGML_TYPE_F32:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F32>(ctx, src0, src1, dst, prepared_src1);
            break;
        case GGML_TYPE_BF16:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_BF16>(ctx, src0, src1, dst, prepared_src1);
            break;
        case GGML_TYPE_F16:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F16>(ctx, src0, src1, dst, prepared_src1);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}



template <ggml_type compute_type>
static void ggml_cuda_gdn_cublas_pair(ggml_backend_cuda_context & ctx, ggml_tensor * alpha, ggml_tensor * beta) {
    using traits = batched_mul_mat_traits<compute_type>;
    using cuda_t = typename traits::cuda_type;
    const ggml_tensor * x = alpha->src[1];
    (void) ctx.cublas_handle();
    ggml_cuda_pool_alloc<cuda_t> prepared(ctx.pool(), ggml_nelements(x));
    if (ggml_is_contiguously_allocated(x)) {
        traits::convert(x->type)(x->data, prepared.get(), ggml_nelements(x), ctx.stream());
    } else {
        traits::convert_nc(x->type)(x->data, prepared.get(), x->ne[0], x->ne[1], x->ne[2], x->ne[3],
            x->nb[1] / sizeof(float), x->nb[2] / sizeof(float), x->nb[3] / sizeof(float), ctx.stream());
    }
    ggml_cuda_mul_mat_cublas_impl<compute_type>(ctx, alpha->src[0], x, alpha, prepared.get());
    ggml_cuda_mul_mat_cublas_impl<compute_type>(ctx, beta->src[0], x, beta, prepared.get());
}

static bool ggml_cuda_should_fuse_mul_mat(const ggml_tensor * ffn_up,
                                          const ggml_tensor * ffn_gate,
                                          const ggml_tensor * glu,
                                          const ggml_tensor * ffn_up_bias = nullptr,
                                          const ggml_tensor * ffn_gate_bias = nullptr,
                                          const ggml_tensor * ffn_up_scale = nullptr,
                                          const ggml_tensor * ffn_gate_scale = nullptr,
                                          bool allow_cached = false) {
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

    const bool up_cached = ggml_backend_buft_is_cuda_moe_cached(ffn_up->src[0]->buffer->buft);
    const bool gate_cached = ggml_backend_buft_is_cuda_moe_cached(ffn_gate->src[0]->buffer->buft);
    if (!allow_cached && (up_cached || gate_cached)) {
        return false;
    }

    return true;
}



static int ggml_cuda_match_mmf_id_pair(const ggml_cgraph * cgraph, const int i, const int device) {
    const ggml_tensor * first = cgraph->nodes[i];
    if (first->op != GGML_OP_MUL_MAT_ID || !first->src[0] || !first->src[1] ||
            first->src[1]->ne[2] <= 0 || first->src[1]->ne[2] > INT_MAX ||
            !ggml_cuda_should_use_mmf(first->src[0]->type, ggml_cuda_info().devices[device].cc, WARP_SIZE,
                first->src[0]->ne, first->src[0]->nb, first->src[1]->ne[2], true)) {
        return 0;
    }
    int count = 3;
    if (!ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU }, { i, i + 1, i + 2 })) {
        count = 4;
        if (!ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_MUL_MAT_ID, GGML_OP_VIEW, GGML_OP_MUL_MAT_ID, GGML_OP_GLU }, { i, i + 1, i + 2, i + 3 })) {
            return 0;
        }
        const ggml_tensor * view = cgraph->nodes[i + 1];
        const ggml_tensor * root = view->view_src;
        if (view != cgraph->nodes[i + 2]->src[0] || !root || root->op != GGML_OP_NONE || !root->buffer ||
                ggml_backend_buffer_get_usage(root->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
                (root->flags & GGML_TENSOR_FLAG_PARAM) || (view->flags & GGML_TENSOR_FLAG_OUTPUT) ||
                ggml_node_get_use_count(cgraph, i + 1) != 1) {
            return 0;
        }
    }
    const ggml_tensor * glu  = cgraph->nodes[i + count - 1];
    const ggml_tensor * gate = glu->src[0];
    const ggml_tensor * up   = glu->src[1];
    if (!gate || !up || !((gate == cgraph->nodes[i] && up == cgraph->nodes[i + count - 2]) ||
                          (up == cgraph->nodes[i] && gate == cgraph->nodes[i + count - 2]))) {
        return 0;
    }
    if (!up->src[0] || !gate->src[0] || !up->src[0]->buffer || !gate->src[0]->buffer ||
            up->src[0]->buffer->buft != ggml_backend_cuda_buffer_type(device) ||
            gate->src[0]->buffer->buft != ggml_backend_cuda_buffer_type(device)) {
        return 0;
    }
    return ggml_cuda_should_pair_mmf_id(up, gate, glu, device) ? count : 0;
}

static bool ggml_cuda_mmf_id_pair_memory_ok(const ggml_cgraph * cgraph, const int i, const int count, const int device) {
    const auto buffer_range = [device](const ggml_tensor * tensor, uintptr_t & begin, uintptr_t & end) {
        if (!tensor || !tensor->buffer || !tensor->data || !ggml_backend_buffer_is_cuda(tensor->buffer)) {
            return false;
        }
        const auto * buffer_ctx = (const ggml_backend_cuda_buffer_context *) tensor->buffer->context;
        if (buffer_ctx->device != device) {
            return false;
        }
        const size_t ts = ggml_type_size(tensor->type);
        const size_t block = ggml_blck_size(tensor->type);
        if (tensor->ne[0] <= 0 || tensor->ne[0] % block != 0) {
            return false;
        }
        size_t span = ts;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            const size_t elements = d == 0 ? tensor->ne[d]/block : tensor->ne[d];
            if (tensor->ne[d] <= 0 || (tensor->nb[d] != 0 && elements - 1 > (SIZE_MAX - span)/tensor->nb[d])) {
                return false;
            }
            span += (elements - 1)*tensor->nb[d];
        }
        begin = (uintptr_t) tensor->data;
        const uintptr_t base = (uintptr_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t size = ggml_backend_buffer_get_size(tensor->buffer);
        const size_t alloc = ggml_backend_buffer_get_alloc_size(tensor->buffer, tensor);
        if (begin < base || begin - base > size || alloc < span || alloc > size - (begin - base) || begin > UINTPTR_MAX - alloc) {
            return false;
        }
        end = begin + alloc;
        return true;
    };
    const auto allocation_range = [&buffer_range](const ggml_tensor * tensor, uintptr_t & begin, uintptr_t & end) {
        if (!buffer_range(tensor, begin, end)) {
            return false;
        }
        if (tensor->view_src) {
            uintptr_t root_begin, root_end;
            if (tensor->view_src->view_src || !buffer_range(tensor->view_src, root_begin, root_end) || tensor->view_offs > root_end - root_begin ||
                    begin != root_begin + tensor->view_offs || end > root_end) {
                return false;
            }
        }
        return true;
    };
    const ggml_tensor * dst = cgraph->nodes[i + count - 1];
    const ggml_tensor * up = dst->src[1];
    const ggml_tensor * gate = dst->src[0];
    uintptr_t begin[3], end[3];
    const ggml_tensor * writes[] = { up, gate, dst };
    for (int w = 0; w < 3; ++w) {
        if (!allocation_range(writes[w], begin[w], end[w])) {
            return false;
        }
    }
    const auto overlaps = [&begin, &end](const int w, const uintptr_t lo, const uintptr_t hi) {
        return begin[w] < hi && lo < end[w];
    };
    if (overlaps(0, begin[1], end[1])) {
        return false;
    }
    for (int w = 0; w < 2; ++w) {
        if (overlaps(w, begin[2], end[2]) && (begin[w] != begin[2] || end[w] != end[2])) {
            return false;
        }
    }
    for (int j = i; j < i + count; ++j) {
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
            uintptr_t src_begin, src_end;
            if (!allocation_range(src, src_begin, src_end)) {
                return false;
            }
            for (int w = 0; w < 3; ++w) {
                if (overlaps(w, src_begin, src_end)) {
                    return false;
                }
            }
        }
    }
    for (const ggml_tensor * weight : { up->src[0], gate->src[0] }) {
        if ((uintptr_t) weight->data % (2*ggml_type_size(weight->type)) != 0) {
            return false;
        }
    }
    return (uintptr_t) up->src[1]->data % (2*sizeof(float)) == 0;
}

static int ggml_cuda_match_mmq_id_pair(const ggml_cgraph * cgraph, const int i, const int device) {
    int count = 3;
    if (!ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU }, { i + 2 })) {
        count = 4;
        if (!ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_MUL_MAT_ID, GGML_OP_VIEW, GGML_OP_MUL_MAT_ID, GGML_OP_GLU }, { i + 1, i + 3 })) {
            return 0;
        }
        const ggml_tensor * view = cgraph->nodes[i + 1];
        const ggml_tensor * root = view->view_src;
        if (view != cgraph->nodes[i + 2]->src[0] || !root || root->op != GGML_OP_NONE || !root->buffer ||
                ggml_backend_buffer_get_usage(root->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
                (root->flags & GGML_TENSOR_FLAG_PARAM) || (view->flags & GGML_TENSOR_FLAG_OUTPUT) ||
                ggml_node_get_use_count(cgraph, i + 1) != 1) {
            return 0;
        }
    }
    const ggml_tensor * glu  = cgraph->nodes[i + count - 1];
    const ggml_tensor * gate = glu->src[0];
    const ggml_tensor * up   = glu->src[1];
    if (!gate || !up || !((gate == cgraph->nodes[i] && up == cgraph->nodes[i + count - 2]) ||
                          (up == cgraph->nodes[i] && gate == cgraph->nodes[i + count - 2]))) {
        return 0;
    }
    if (!up->src[0] || !gate->src[0] || !up->src[0]->buffer || !gate->src[0]->buffer ||
            up->src[0]->buffer->buft != ggml_backend_cuda_buffer_type(device) ||
            gate->src[0]->buffer->buft != ggml_backend_cuda_buffer_type(device)) {
        return 0;
    }
    return ggml_cuda_should_fuse_mmq_id(up, gate, glu, ggml_cuda_info().devices[device].cc, ggml_cuda_info().devices[device].smpbo) ? count : 0;
}

static bool ggml_cuda_mmq_id_pair_memory_ok(const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * dst) {
    const uintptr_t dst_start = (uintptr_t) dst->data;
    const size_t dst_size = ggml_backend_buffer_get_alloc_size(dst->buffer, dst);
    if (dst_start > UINTPTR_MAX - dst_size) {
        return false;
    }
    for (const ggml_tensor * src : { up->src[0], gate->src[0], up->src[1], up->src[2] }) {
        const uintptr_t src_start = (uintptr_t) src->data;
        const size_t src_size = ggml_backend_buffer_get_alloc_size(src->buffer, src);
        if (src_start > UINTPTR_MAX - src_size || (dst_start < src_start + src_size && src_start < dst_start + dst_size)) {
            return false;
        }
    }
    for (const ggml_tensor * weight : { up->src[0], gate->src[0] }) {
        if (ggml_backend_buffer_get_usage(weight->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
                ggml_backend_buffer_get_alloc_size(weight->buffer, weight) != ggml_nbytes(weight)) {
            return false;
        }
    }
    return true;
}

static bool ggml_cuda_should_fuse_mul_mat_vec_f(const ggml_tensor * tensor) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool is_mul_mat_id = tensor->op == GGML_OP_MUL_MAT_ID;

    if (is_mul_mat_id && ggml_backend_buft_is_cuda_moe_cached(src0->buffer->buft)) {
        return false;
    }

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

static bool ggml_cuda_should_fuse_mul_mat_vec_q(const ggml_tensor * tensor, bool allow_cached = false) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    if (!allow_cached && tensor->op == GGML_OP_MUL_MAT_ID && ggml_backend_buft_is_cuda_moe_cached(src0->buffer->buft)) {
        return false;
    }

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

enum class ggml_cuda_mul_mat_dispatch : uint32_t { cublas, mmvf, mmvf_transposed, mmf, mmvq, mmq };

static ggml_cuda_mul_mat_dispatch ggml_cuda_mul_mat_select(int device, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS

    // If src0 is a temporary compute buffer it may have some padding that needs to be cleared for mul_mat_vec_q or mul_mat_q.
    // But if src0 is also a view of another tensor then this cannot be done safely because it may overwrite valid tensor data.
    // Therefore, in such cases use cuBLAS.
    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE
        && ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) && src0->view_src;
    if (bad_padding_clear || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return ggml_cuda_mul_mat_dispatch::cublas;
    }

    const int cc        = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;

    if (ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, ne11)) {
        // The custom F16 vector kernel can be used over batched cuBLAS GEMM.
        // But this is only faster for GPUs without tensor cores or with a thin src0 matrix (particularly KQV in attention)
        return ggml_cuda_mul_mat_dispatch::mmvf;
    }
    // A transposed vector can still use MMVQ (i.e. ne01 == 1)
    if (ne01 == 1 && ne11 > MMVF_MAX_BATCH_SIZE && ne2 == 1 && ne3 == 1
            && src0->type == GGML_TYPE_F32
            && ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst)
            && ggml_cuda_should_use_mmvf(src1->type, cc, src1->ne, src1->nb, /*ne11 =*/ 1)) {
        return ggml_cuda_mul_mat_dispatch::mmvf_transposed;
    }
    if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11, /*mul_mat_id =*/ false)) {
        return ggml_cuda_mul_mat_dispatch::mmf;
    }
    if (ggml_cuda_should_use_mmvq(src0->type, cc, ne11)) {
        return ggml_cuda_mul_mat_dispatch::mmvq;
    }
    if (ggml_cuda_should_use_mmq(src0->type, cc, ne11, /*n_experts =*/ 0, ggml_cuda_info().devices[device].smpbo)) {
        return ggml_cuda_mul_mat_dispatch::mmq;
    }
    return ggml_cuda_mul_mat_dispatch::cublas;
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
    switch (ggml_cuda_mul_mat_select(device, src0, src1, dst)) {
        case ggml_cuda_mul_mat_dispatch::mmvf: return GGML_CUDA_MM_MMVF;
        case ggml_cuda_mul_mat_dispatch::mmvf_transposed: return GGML_CUDA_MM_MMVF_TRANSPOSE;
        case ggml_cuda_mul_mat_dispatch::mmf: return GGML_CUDA_MM_MMF;
        case ggml_cuda_mul_mat_dispatch::mmvq: return GGML_CUDA_MM_MMVQ;
        case ggml_cuda_mul_mat_dispatch::mmq: return GGML_CUDA_MM_MMQ;
        case ggml_cuda_mul_mat_dispatch::cublas: return GGML_CUDA_MM_CUBLAS;
    }
    GGML_ABORT("fatal error");
}

static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const void * prepared_src1 = nullptr) {
    if (ggml_cuda_op_mul_mat_use_fwht(dst) && ggml_cuda_op_fwht(ctx, src1, dst)) {
        return;
    }
    switch (ggml_cuda_mul_mat_select(ctx.device, src0, src1, dst)) {
        case ggml_cuda_mul_mat_dispatch::cublas:
            ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst, prepared_src1);
            break;
        case ggml_cuda_mul_mat_dispatch::mmvf:
            ggml_cuda_mul_mat_vec_f(ctx, src0, src1, nullptr, dst);
            break;
        case ggml_cuda_mul_mat_dispatch::mmvf_transposed: {
            ggml_tensor dst_vec = *dst;
            dst_vec.ne[0] = src1->ne[1];
            dst_vec.ne[1] = 1;
            dst_vec.nb[1] = dst_vec.nb[0]*src1->ne[1];
            dst_vec.nb[2] = dst_vec.nb[1];
            dst_vec.nb[3] = dst_vec.nb[1];
            ggml_cuda_mul_mat_vec_f(ctx, src1, src0, nullptr, &dst_vec);
        } break;
        case ggml_cuda_mul_mat_dispatch::mmf:
            ggml_cuda_mul_mat_f(ctx, src0, src1, nullptr, dst);
            break;
        case ggml_cuda_mul_mat_dispatch::mmvq:
            ggml_cuda_mul_mat_vec_q(ctx, src0, src1, nullptr, dst);
            break;
        case ggml_cuda_mul_mat_dispatch::mmq:
            ggml_cuda_mul_mat_q(ctx, src0, src1, nullptr, dst);
            break;
    }
}

bool ggml_cuda_moe_fidelity_mul_mat_requirements(int device, const ggml_tensor * node, ggml_cuda_moe_fidelity_mul_mat_resources & resources) {
    const auto & info = ggml_cuda_info();
    if (device < 0 || device >= info.device_count || !node || node->op != GGML_OP_MUL_MAT ||
            !node->src[0] || !node->src[1] || !node->src[0]->buffer || node->type != GGML_TYPE_F32 ||
            ggml_get_op_params_i32(node, 1) == GGML_HINT_SRC0_IS_HADAMARD) { return false; }
    const auto * src0 = node->src[0];
    const auto * src1 = node->src[1];
    const auto elements = [](const ggml_tensor * tensor, size_t & count) {
        count = 1;
        for (const auto n : tensor->ne) {
            if (n <= 0 || n > INT32_MAX || size_t(n) > SIZE_MAX / count) { return false; }
            count *= size_t(n);
        }
        return true;
    };
    size_t n0, n1, nd;
    if (!elements(src0, n0) || !elements(src1, n1) || !elements(node, nd) ||
            src0->ne[0] != src1->ne[0] || node->ne[0] != src0->ne[1] || node->ne[1] != src1->ne[1] ||
            node->ne[2] != src1->ne[2] || node->ne[3] != src1->ne[3] ||
            src1->ne[2] % src0->ne[2] || src1->ne[3] % src0->ne[3]) { return false; }
    ggml_cuda_moe_fidelity_mul_mat_resources measured;
    const auto dispatch = ggml_cuda_mul_mat_select(device, src0, src1, node);
    measured.dispatch = uint32_t(dispatch);
    const auto add = [&](size_t count, size_t bytes) {
        if (count > SIZE_MAX / bytes) { return false; }
        const size_t size = count * bytes;
        if (size > SIZE_MAX - 255) { return false; }
        const size_t aligned = (size + 255) & ~size_t(255);
        if (aligned > SIZE_MAX - measured.pool_bytes) { return false; }
        measured.pool_bytes += aligned;
        return true;
    };
    if (dispatch == ggml_cuda_mul_mat_dispatch::mmq) { return false; }
    if (dispatch == ggml_cuda_mul_mat_dispatch::mmvq) {
        const size_t width = size_t(src1->ne[0]);
        if (width > SIZE_MAX - (MATRIX_ROW_PADDING - 1)) { return false; }
        const size_t padded = (width + MATRIX_ROW_PADDING - 1) / MATRIX_ROW_PADDING * MATRIX_ROW_PADDING;
        const size_t rows = n1 / width;
        if (padded / QK8_1 > SIZE_MAX / rows || !add(padded / QK8_1 * rows, sizeof(block_q8_1))) { return false; }
    } else if (dispatch == ggml_cuda_mul_mat_dispatch::cublas) {
        if (!ggml_is_contiguous(node) || src0->nb[0] != ggml_type_size(src0->type) ||
                src1->nb[0] != ggml_type_size(src1->type)) { return false; }
        const int cc = info.devices[device].cc;
        const auto type = ggml_cuda_mul_mat_cublas_compute_type(cc, src0, src1, node);
        if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_BF16) { return false; }
        measured.compute_type = type;
        measured.cublas_workspace_bytes = ggml_cuda_cublas_workspace_size(cc);
        if (!measured.cublas_workspace_bytes) { return false; }
        const auto converts = [type](const ggml_tensor * tensor) {
            switch (type) {
                case GGML_TYPE_F32: return ggml_is_contiguously_allocated(tensor) ?
                    ggml_get_to_fp32_cuda(tensor->type) != nullptr : ggml_get_to_fp32_nc_cuda(tensor->type) != nullptr;
                case GGML_TYPE_F16: return ggml_is_contiguously_allocated(tensor) ?
                    ggml_get_to_fp16_cuda(tensor->type) != nullptr : ggml_get_to_fp16_nc_cuda(tensor->type) != nullptr;
                case GGML_TYPE_BF16: return ggml_is_contiguously_allocated(tensor) ?
                    ggml_get_to_bf16_cuda(tensor->type) != nullptr : ggml_get_to_bf16_nc_cuda(tensor->type) != nullptr;
                default: return false;
            }
        };
        const size_t bytes = ggml_type_size(type);
        const auto leading_dimension = [type](const ggml_tensor * tensor) {
            size_t leading = tensor->ne[0];
            if (tensor->type == type || ggml_is_contiguously_allocated(tensor)) {
                const size_t element = ggml_type_size(tensor->type);
                if (tensor->nb[1] % element) { return false; }
                leading = tensor->nb[1] / element;
                if (tensor->type != type) {
                    const size_t block = ggml_blck_size(tensor->type);
                    if (leading > SIZE_MAX / block) { return false; }
                    leading *= block;
                }
            }
            return leading > 0 && leading <= INT32_MAX;
        };
        if (!leading_dimension(src0) || !leading_dimension(src1)) { return false; }
        if ((src0->type != type && (!converts(src0) || !add(n0, bytes))) ||
                (src1->type != type && (!converts(src1) || !add(n1, bytes)))) { return false; }
        if (type != GGML_TYPE_F32 && !ggml_cuda_mul_mat_cublas_f32_output(cc, type) && !add(nd, bytes)) { return false; }
        const bool contiguous0 = ggml_is_contiguous_2(src0) || (src0->type != type && !ggml_is_contiguously_allocated(src0));
        const bool contiguous1 = ggml_is_contiguous_2(src1) || (src1->type != type && !ggml_is_contiguously_allocated(src1));
        if (!(src1->ne[2] == 1 && src1->ne[3] == 1) &&
                !(src1->ne[2] == src0->ne[2] && src1->ne[3] == src0->ne[3] && contiguous0 && contiguous1)) {
            const size_t matrices = size_t(src1->ne[2]) * size_t(src1->ne[3]);
            if (matrices > INT32_MAX) { return false; }
            if (!add(matrices, 2 * sizeof(void *)) || !add(matrices, sizeof(void *))) { return false; }
        } else if (size_t(src1->ne[2]) * size_t(src1->ne[3]) > INT32_MAX) {
            return false;
        }
    }
    resources = measured;
    return true;
}

bool ggml_cuda_prepare_cublas(ggml_backend_cuda_context & context, int device, int stream_no, void * workspace, size_t bytes) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (device < 0 || device >= ggml_cuda_info().device_count || stream_no < 0 || stream_no >= GGML_CUDA_MAX_STREAMS ||
            context.capture_resource_requests || context.cublas_handles[device][stream_no] ||
            context.cublas_workspaces[device][stream_no] || bytes != ggml_cuda_cublas_workspace_size(ggml_cuda_info().devices[device].cc) ||
            (bytes && (!workspace || reinterpret_cast<uintptr_t>(workspace) % 256))) { return false; }
    std::unique_lock<std::mutex> lock(ggml_cuda_lock, std::try_to_lock);
    if (!lock.owns_lock() || ggml_cuda_lock_counter.load(std::memory_order_relaxed) != 0) { return false; }
    ggml_cuda_set_device(device);
    const auto stream = context.stream(device, stream_no);
    cudaStreamCaptureStatus capture;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess || capture != cudaStreamCaptureStatusNone) { return false; }
    cublasHandle_t handle = nullptr;
    const auto created = cublasCreate(&handle);
    context.cublas_handles[device][stream_no] = handle;
    if (created != CUBLAS_STATUS_SUCCESS || !handle ||
            cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH) != CUBLAS_STATUS_SUCCESS ||
            cublasSetStream(handle, stream) != CUBLAS_STATUS_SUCCESS) { return false; }
#if CUBLAS_VER_MAJOR > 11 || (CUBLAS_VER_MAJOR == 11 && CUBLAS_VER_MINOR >= 2)
    if (cublasSetWorkspace(handle, workspace, bytes) != CUBLAS_STATUS_SUCCESS) { return false; }
#endif
    context.cublas_workspace_sizes[device] = bytes;
    return true;
#else
    GGML_UNUSED(context); GGML_UNUSED(device); GGML_UNUSED(stream_no); GGML_UNUSED(workspace); GGML_UNUSED(bytes);
    return false;
#endif
}

bool ggml_cuda_moe_fidelity_prepare_cublas(ggml_backend_cuda_context & context, void * workspace, size_t bytes) {
    const int device = context.device;
    if (device < 0 || device >= ggml_cuda_info().device_count || context.curr_stream_no != 0 || !context.borrowed_stream ||
            context.streams[device][0] != context.borrowed_stream) { return false; }
    return ggml_cuda_prepare_cublas(context, device, 0, workspace, bytes);
}

bool ggml_cuda_moe_fidelity_release_cublas(ggml_backend_cuda_context & context) {
    const int device = context.device;
    if (device < 0 || device >= ggml_cuda_info().device_count || context.curr_stream_no != 0 ||
            context.cublas_workspaces[device][0]) { return false; }
    auto & handle = context.cublas_handles[device][0];
    if (handle) {
        std::unique_lock<std::mutex> lock(ggml_cuda_lock, std::try_to_lock);
        if (!lock.owns_lock() || ggml_cuda_lock_counter.load(std::memory_order_relaxed) != 0 ||
                cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS) { return false; }
    }
    handle = nullptr;
    context.cublas_workspace_sizes[device] = 0;
    return true;
}

bool ggml_cuda_moe_fidelity_capture_enter() {
    std::lock_guard<std::mutex> lock(ggml_cuda_lock);
    if (ggml_cuda_lock_counter.load(std::memory_order_relaxed) == INT_MAX) { return false; }
    ggml_cuda_lock_counter.fetch_add(1, std::memory_order_relaxed);
    return true;
}

void ggml_cuda_moe_fidelity_capture_leave() {
    std::lock_guard<std::mutex> lock(ggml_cuda_lock);
    if (ggml_cuda_lock_counter.fetch_sub(1, std::memory_order_relaxed) == 1) {
        ggml_cuda_lock_cv.notify_all();
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

struct ggml_cuda_mul_mat_id_host_route {
    const char * data;
    size_t nb0;
    size_t nb1;
    const int32_t * expert_map;
    void * secondary_data = nullptr;
    int32_t secondary_begin = 0;
    int32_t secondary_count = 0;
    const int32_t * source_wait_class = nullptr;
    int n_wait_classes = 1;
    const uint32_t * stage_ready = nullptr;
};

// Shared by the regular and cached-buffer dispatch paths.
static bool ggml_cuda_mul_mat_id_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, bool use_mmq,
        const ggml_cuda_mul_mat_id_host_route * host_route = nullptr,
        ggml_cuda_mmid_consumer required_consumer = GGML_CUDA_MMID_CONSUMER_UNSUPPORTED,
        const ggml_cuda_mmid_direct_source_view * direct_source_view = nullptr,
        const ggml_cuda_mmid_execution * execution = nullptr) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (execution && (host_route || direct_source_view || !ggml_cuda_mmid_execution_valid(dst, *execution))) { return false; }
    const bool mapped_experts = host_route != nullptr && host_route->expert_map != nullptr;
    if (direct_source_view != nullptr && (mapped_experts ||
            !ggml_cuda_mmid_direct_source_view_valid(src0, *direct_source_view, required_consumer))) {
        return false;
    }
    const int64_t chooser_ne02 = direct_source_view != nullptr ? direct_source_view->logical_n_experts : ne02;
    const auto source_capability = ggml_cuda_mmid_source_capability_for(src0->type);
    const bool mapped_mmq = !mapped_experts || (source_capability.flags & GGML_CUDA_MMID_SOURCE_MAPPED_MMQ) != 0;

    // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
    if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        static_assert(MMVQ_MAX_BATCH_SIZE == MMVF_MAX_BATCH_SIZE);
        if (!mapped_experts && ne2 <= MMVQ_MAX_BATCH_SIZE) {
            if (ggml_is_quantized(src0->type) &&
                    (required_consumer == GGML_CUDA_MMID_CONSUMER_UNSUPPORTED ||
                        required_consumer == GGML_CUDA_MMID_CONSUMER_MMVQ)) {
                const int mmvq_mmid_max = get_mmvq_mmid_max_batch(src0->type, cc);
                if (ne2 <= mmvq_mmid_max) {
                    ggml_cuda_mul_mat_vec_q(ctx, src0, src1, ids, dst, nullptr, nullptr, execution);
                    return true;
                }
            } else if (required_consumer == GGML_CUDA_MMID_CONSUMER_UNSUPPORTED ||
                    required_consumer == GGML_CUDA_MMID_CONSUMER_MMVF) {
                if (GGML_CUDA_CC_IS_AMD(cc)) {
                    ggml_cuda_mul_mat_vec_f(ctx, src0, src1, ids, dst, nullptr, false, execution);
                    return true;
                }
            }
        }

        if ((required_consumer == GGML_CUDA_MMID_CONSUMER_UNSUPPORTED ||
                required_consumer == GGML_CUDA_MMID_CONSUMER_MMQ) &&
                use_mmq && mapped_mmq && ggml_cuda_should_use_mmq(src0->type, cc, ne12, /*n_experts=*/chooser_ne02)) {
            if (mapped_experts) {
                int32_t source_split = host_route->secondary_data ? host_route->secondary_begin : 0;
                if (source_split == 0) {
                    for (int64_t i02 = 0; i02 < ne02; ++i02) {
                        source_split = std::max(source_split, host_route->expert_map[i02] + 1);
                    }
                }
                GGML_ASSERT(source_split > 0);

                ggml_cuda_pool_alloc<int32_t> expert_map(ctx.pool(), ne02);
                ggml_cuda_pool_alloc<int32_t> source_wait_class(ctx.pool());
                CUDA_CHECK(cudaMemcpyAsync(expert_map.get(), host_route->expert_map,
                                           ne02 * sizeof(int32_t), cudaMemcpyHostToDevice, ctx.stream()));
                if (host_route->source_wait_class != nullptr) {
                    source_wait_class.alloc(ne02);
                    CUDA_CHECK(cudaMemcpyAsync(source_wait_class.get(), host_route->source_wait_class,
                                               ne02 * sizeof(int32_t), cudaMemcpyHostToDevice, ctx.stream()));
                }

                ggml_cuda_mul_mat_q_mapped(
                    ctx, src0, host_route->secondary_data ? host_route->secondary_data : src0->data,
                    src1, ids, dst, expert_map.get(), source_split,
                    source_wait_class.get(), host_route->stage_ready);
            } else {
                ggml_cuda_mul_mat_q(ctx, src0, src1, ids, dst, nullptr, nullptr, execution);
            }
            return true;
        }

        int64_t chooser_ne[GGML_MAX_DIMS];
        memcpy(chooser_ne, src0->ne, sizeof(chooser_ne));
        chooser_ne[2] = chooser_ne02;
        if ((required_consumer == GGML_CUDA_MMID_CONSUMER_UNSUPPORTED ||
                required_consumer == GGML_CUDA_MMID_CONSUMER_MMF) && !mapped_experts &&
                ggml_cuda_should_use_mmf(src0->type, cc, WARP_SIZE, chooser_ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
            ggml_cuda_mul_mat_f(ctx, src0, src1, ids, dst, nullptr, nullptr, execution);
            return true;
        }
    }

    if (required_consumer != GGML_CUDA_MMID_CONSUMER_UNSUPPORTED &&
            required_consumer != GGML_CUDA_MMID_CONSUMER_GENERIC) {
        return false;
    }
    if (execution) { return false; }

    // note: this path should not be reached when recording CUDA graphs, because it requires stream synchronization
    GGML_ASSERT(mapped_experts || !use_mmq || ggml_cuda_mul_mat_id_needs_sync(dst, cc));
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
    const size_t smpbo = ggml_cuda_info().devices[ggml_cuda_get_device()].smpbo;
    const bool ids_helper_fits = ne12 > 1 && ne12 < (1 << 22) && (size_t) ne12*sizeof(uint32_t) <= smpbo &&
        n_expert_used < (1 << 10) && ne02 <= INT_MAX;
    const bool use_device_ids = host_route != nullptr && ids->type == GGML_TYPE_I32 &&
        ids->ne[1] == ne12 && ids->ne[2] == 1 && ids->ne[3] == 1 &&
        ids->nb[0] == ggml_element_size(ids) && host_route->nb0 == sizeof(int32_t) &&
        ids_helper_fits;

    std::vector<int32_t> ids_to_sorted_host;
    if (!use_device_ids) {
        ids_to_sorted_host.resize(2*ne_get_rows);
    }

    ggml_cuda_pool_alloc<int32_t> ids_buf_dev(ctx.pool(), (use_device_ids ? 3 : 2)*ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool());
    if (use_device_ids) {
        expert_bounds.alloc(ne02 + 1);
    }

    std::vector<int32_t> tokens_per_expert(ne02);

    ggml_cuda_pool_alloc<char> src1_sorted(ctx.pool(), ne12*n_expert_used*ne10*ts_src1_sorted);
    ggml_cuda_pool_alloc<char>  dst_sorted(ctx.pool(), ne2 *n_expert_used* ne0*ts_dst_sorted);

    std::vector<char> ids_host;
    const char * ids_host_data = host_route ? host_route->data : nullptr;
    size_t ids_host_nb0 = host_route ? host_route->nb0 : 0;
    size_t ids_host_nb1 = host_route ? host_route->nb1 : 0;
    if (!use_device_ids) {
        ids_host.resize(ggml_nbytes(ids));
        CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids->data, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        ids_host_data = ids_host.data();
        ids_host_nb0 = ids->nb[0];
        ids_host_nb1 = ids->nb[1];
    }

    GGML_ASSERT(ne_get_rows <= INT_MAX && ne12 * ne11 <= INT_MAX);
    for (int64_t i12 = 0; i12 < ne12; ++i12) {
        for (int64_t iex = 0; iex < n_expert_used; ++iex) {
            const int32_t expert = *(const int32_t *)(ids_host_data + i12*ids_host_nb1 + iex*ids_host_nb0);
            GGML_ASSERT(expert >= 0 && expert < ne02);
            ++tokens_per_expert[expert];
        }
    }

    std::vector<size_t> expert_row_offset(ne02);
    size_t row_offset = 0;
    for (int64_t i02 = 0; i02 < ne02; ++i02) {
        expert_row_offset[i02] = row_offset;
        row_offset += tokens_per_expert[i02];
    }
    GGML_ASSERT(row_offset == (size_t) ne_get_rows);
    if (!use_device_ids) {
        auto cursor = expert_row_offset;
        for (int64_t i12 = 0; i12 < ne12; ++i12) {
            for (int64_t iex = 0; iex < n_expert_used; ++iex) {
                const int32_t expert = *(const int32_t *)(ids_host_data + i12*ids_host_nb1 + iex*ids_host_nb0);
                const size_t sorted = cursor[expert]++;
                ids_to_sorted_host[sorted] = i12 * ne11 + iex % ne11;
                ids_to_sorted_host[ne_get_rows + i12 * n_expert_used + iex] = sorted;
            }
        }
    }

    const int32_t * ids_to_sorted   = ids_buf_dev.ptr + 0*ne_get_rows;
    const int32_t * ids_from_sorted = ids_buf_dev.ptr + 1*ne_get_rows;
    if (use_device_ids) {
        int32_t * ids_dst = ids_buf_dev.ptr + 2*ne_get_rows;
        const int si1 = ids->nb[1] / ggml_element_size(ids);
        const int sis1 = nb12 / nb11;
        ggml_cuda_launch_mm_ids_helper(
            (const int32_t *) ids->data, ids_buf_dev.ptr, ids_dst, expert_bounds.get(),
            ne02, ne12, n_expert_used, ne11, si1, sis1, false, stream);
        ggml_cuda_launch_mm_ids_helper(
            (const int32_t *) ids->data, ids_buf_dev.ptr + ne_get_rows, ids_dst, expert_bounds.get(),
            ne02, ne12, n_expert_used, ne11, si1, sis1, true, stream);
        CUDA_CHECK(cudaGetLastError());
    } else {
        CUDA_CHECK(cudaMemcpyAsync(ids_buf_dev.ptr, ids_to_sorted_host.data(), 2*ne_get_rows*sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    get_rows_cuda(src1->data, src1->type, ids_to_sorted, src1_sorted.ptr, type_src1_sorted,
        ne10, nb11, nb12, nb13,
        ne_get_rows, 1, 1, sizeof(int32_t), ne_get_rows*sizeof(int32_t), ne_get_rows*sizeof(int32_t),
        ne10*ts_src1_sorted, ne_get_rows*ne10*ts_src1_sorted, ne_get_rows*ne10*ts_src1_sorted, stream);
    CUDA_CHECK(cudaGetLastError());

    const int n_wait_classes = host_route && host_route->source_wait_class ? host_route->n_wait_classes : 1;
    GGML_ASSERT(n_wait_classes > 0);
    for (int wait_class = 0; wait_class < n_wait_classes; ++wait_class) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && !defined(GGML_CUDA_NO_VMM)
        if (wait_class > 0) {
            GGML_ASSERT(host_route->stage_ready != nullptr);
            CU_CHECK(cuStreamWaitValue32(
                stream, (CUdeviceptr)(host_route->stage_ready + wait_class - 1), 1, CU_STREAM_WAIT_VALUE_EQ));
        }
#else
        GGML_ASSERT(wait_class == 0);
#endif

        for (int64_t i02 = 0; i02 < ne02; ++i02) {
            if (tokens_per_expert[i02] == 0 ||
                (host_route && host_route->source_wait_class && host_route->source_wait_class[i02] != wait_class)) {
                continue;
            }

            ggml_tensor src0_slice = *src0;
            src0_slice.ne[2]    = 1;
            src0_slice.nb[3]    = src0_slice.nb[2];
            src0_slice.op       = GGML_OP_VIEW;
            src0_slice.view_src = dst->src[0]; // non-const pointer to src0
            int32_t physical_expert = host_route && host_route->expert_map ? host_route->expert_map[i02] : i02;
            GGML_ASSERT(physical_expert >= 0);
            char * expert_base = (char *) src0->data;
            if (host_route && host_route->secondary_data && physical_expert >= host_route->secondary_begin) {
                physical_expert -= host_route->secondary_begin;
                GGML_ASSERT(physical_expert < host_route->secondary_count);
                expert_base = (char *) host_route->secondary_data;
            }
            src0_slice.data = expert_base + physical_expert*nb02;

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
            src1_slice.data   = (char *) src1_sorted.ptr + expert_row_offset[i02]*src1_slice.nb[1];

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
            dst_slice.data   = (char *) dst_sorted.ptr + expert_row_offset[i02]*dst_slice.nb[1];

            ggml_cuda_mul_mat(ctx, &src0_slice, &src1_slice, &dst_slice);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    get_rows_cuda(dst_sorted.ptr, type_dst_sorted, ids_from_sorted, dst->data, dst->type,
        ne0, ne0*ts_dst_sorted, ne_get_rows*ne0*ts_dst_sorted, ne_get_rows*ne0*ts_dst_sorted,
        ne_get_rows, 1, 1, sizeof(int32_t), ne_get_rows*sizeof(int32_t), ne_get_rows*sizeof(int32_t),
        nb1, nb2, nb3, stream);
    return true;
}

bool ggml_cuda_mmid_execution_compute(ggml_backend_cuda_context & context, ggml_tensor * dst, const ggml_cuda_mmid_execution & execution) {
    ggml_cuda_mmid_resources resources;
    if (!ggml_cuda_mmid_execution_valid(dst, execution) || !ggml_cuda_mmid_requirements(context.device, dst, resources)) { return false; }
    ggml_cuda_set_device(context.device);
    if (resources.consumer == GGML_CUDA_MMID_CONSUMER_MMVF) {
        ggml_cuda_mul_mat_vec_f(context, dst->src[0], dst->src[1], dst->src[2], dst, nullptr, false, &execution);
        return true;
    }
    return ggml_cuda_mul_mat_id_impl(context, dst, resources.consumer == GGML_CUDA_MMID_CONSUMER_MMQ,
        nullptr, resources.consumer, nullptr, &execution);
}

bool ggml_cuda_mmid_execution_compute(ggml_backend_t backend, ggml_tensor * dst, const ggml_cuda_mmid_execution & execution) {
    if (!ggml_backend_is_cuda(backend) || !backend->context || !ggml_cuda_mmid_execution_valid(dst, execution) ||
            !ggml_backend_dev_supports_op(ggml_backend_get_device(backend), dst)) { return false; }
    return ggml_cuda_mmid_execution_compute(*static_cast<ggml_backend_cuda_context *>(backend->context), dst, execution);
}

bool ggml_cuda_mmid_vector_compute_for_test(ggml_backend_t backend, ggml_tensor * dst) {
    if (!ggml_backend_is_cuda(backend) || !backend->context || !dst || dst->op != GGML_OP_MUL_MAT_ID ||
            !dst->src[0] || !dst->src[1] || !dst->src[2] || dst->ne[2] <= 0 || dst->ne[2] > MMVF_MAX_BATCH_SIZE ||
            !ggml_backend_dev_supports_op(ggml_backend_get_device(backend), dst)) { return false; }
    auto & context = *static_cast<ggml_backend_cuda_context *>(backend->context);
    ggml_cuda_set_device(context.device);
    if (ggml_is_quantized(dst->src[0]->type)) {
        ggml_cuda_mul_mat_vec_q(context, dst->src[0], dst->src[1], dst->src[2], dst);
    } else {
        ggml_cuda_mul_mat_vec_f(context, dst->src[0], dst->src[1], dst->src[2], dst);
    }
    return true;
}

cudaStream_t ggml_cuda_mmid_execution_stream_for_test(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_cuda(backend) && backend->context);
    return static_cast<ggml_backend_cuda_context *>(backend->context)->stream();
}

bool ggml_cuda_mmid_direct_source_view_compute_for_test(
        ggml_backend_t backend,
        ggml_tensor * dst,
        const ggml_cuda_mmid_direct_source_view * view,
        ggml_cuda_mmid_consumer consumer) {
    if (!ggml_backend_is_cuda(backend) || backend->context == nullptr || dst == nullptr ||
            dst->op != GGML_OP_MUL_MAT_ID || dst->src[0] == nullptr || dst->src[1] == nullptr) {
        return false;
    }
    auto * context = static_cast<ggml_backend_cuda_context *>(backend->context);
    ggml_cuda_set_device(context->device);
    return ggml_cuda_mul_mat_id_impl(
        *context, dst, ggml_cuda_moe_use_mmq(dst->src[0], dst->src[1]->ne[2]), nullptr, consumer, view);
}

bool ggml_cuda_mmid_bounded_compute_for_test(
        ggml_backend_t backend, ggml_tensor * dst, const uint32_t * active_channels, uint32_t * status) {
    if (!ggml_backend_is_cuda(backend) || backend->context == nullptr || dst == nullptr || dst->op != GGML_OP_MUL_MAT_ID) {
        return false;
    }
    auto * context = static_cast<ggml_backend_cuda_context *>(backend->context);
    ggml_cuda_set_device(context->device);
    return ggml_cuda_mul_mat_vec_q_bounded(*context, dst->src[0], dst->src[1], dst->src[2], dst, active_channels, status);
}

struct ggml_cuda_moe_ids_cache_key {
    const void * data;
    size_t       nbytes;
    int64_t      ne0;
    int64_t      ne1;
    int64_t      ne2;
    int64_t      ne3;
    size_t       nb0;
    size_t       nb1;
    size_t       nb2;
    size_t       nb3;
    int          layer;
};

static bool operator==(const ggml_cuda_moe_ids_cache_key & a, const ggml_cuda_moe_ids_cache_key & b) {
    return a.data == b.data && a.nbytes == b.nbytes &&
        a.ne0 == b.ne0 && a.ne1 == b.ne1 && a.ne2 == b.ne2 && a.ne3 == b.ne3 &&
        a.nb0 == b.nb0 && a.nb1 == b.nb1 && a.nb2 == b.nb2 && a.nb3 == b.nb3 &&
        a.layer == b.layer;
}

enum ggml_cuda_moe_ids_kind {
    GGML_CUDA_MOE_IDS_KIND_NONE,
    GGML_CUDA_MOE_IDS_KIND_GATE,
    GGML_CUDA_MOE_IDS_KIND_UP,
    GGML_CUDA_MOE_IDS_KIND_GATE_UP,
    GGML_CUDA_MOE_IDS_KIND_GATE_AND_UP,
    GGML_CUDA_MOE_IDS_KIND_DOWN,
};

struct ggml_cuda_moe_ids_cache_state {
    static constexpr size_t MAX_PUBLISHED = GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS;

    int device = -1;
    bool pending = false;
    ggml_cuda_moe_ids_kind kind = GGML_CUDA_MOE_IDS_KIND_NONE;
    ggml_cuda_moe_ids_cache_key key = {};
    std::vector<char> bytes;
    size_t nb0 = 0;
    size_t nb1 = 0;
    size_t nb2 = 0;
    uint64_t dispatch_id = 0;

    struct published_entry {
        int32_t * host_ids = nullptr;
        int32_t * device_ids = nullptr;
        uint32_t * host_ready = nullptr;
        uint32_t * device_ready = nullptr;
        size_t count = 0;
        uint64_t dispatch_id = 0;
        bool unconsumed = false;
        bool readable = false;
    };
    struct published_key {
        // Scratch addresses can alias distinct route producers in the same dispatch.
        const ggml_tensor * tensor;
        const void * data;
        cudaStream_t stream;

        bool operator==(const published_key & other) const {
            return tensor == other.tensor && data == other.data && stream == other.stream;
        }
    };
    struct published_key_hash {
        size_t operator()(const published_key & key) const {
            size_t h = std::hash<const void *>{}(key.data);
            h ^= std::hash<const void *>{}(key.tensor) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<void *>{}((void *) key.stream) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };
    // Keep mapped addresses stable for retained CUDA graphs until backend stream teardown.
    std::unordered_map<published_key, published_entry, published_key_hash> published;

    void clear_published() {
        for (auto & item : published) {
            if (item.second.host_ids) {
                cudaFreeHost(item.second.host_ids);
            }
        }
        published.clear();
    }

    ~ggml_cuda_moe_ids_cache_state() {
        clear_published();
    }
};

ggml_backend_cuda_context::ggml_backend_cuda_context(int device) :
    device(device), name(GGML_CUDA_NAME + std::to_string(device)) {
}

ggml_backend_cuda_context::~ggml_backend_cuda_context() {
    moe_router_contexts.clear();
    std::unique_lock<std::mutex> lock(ggml_cuda_lock);
    ggml_cuda_lock_cv.wait(lock, []{ return ggml_cuda_lock_counter.load(std::memory_order_relaxed) == 0; });

    cuda_graphs.clear();
    delete moe_grouped_context;

    if (copy_event != nullptr) {
        CUDA_CHECK(cudaEventDestroy(copy_event));
    }
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        for (int j = 0; j < GGML_CUDA_MAX_STREAMS; ++j) {
            if (streams[i][j] != nullptr) {
                if (moe_ids_cache != nullptr || streams[i][j] == borrowed_stream) {
                    CUDA_CHECK(cudaStreamSynchronize(streams[i][j]));
                }
                if (streams[i][j] != borrowed_stream) {
                    CUDA_CHECK(cudaStreamDestroy(streams[i][j]));
                }
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

static ggml_cuda_moe_ids_cache_state & ggml_cuda_moe_ids_cache_get(ggml_backend_cuda_context & ctx) {
    if (ctx.moe_ids_cache == nullptr) {
        ctx.moe_ids_cache = std::make_unique<ggml_cuda_moe_ids_cache_state>();
    }
    return *ctx.moe_ids_cache;
}

size_t ggml_cuda_moe_ids_cache_count_for_test(ggml_backend_t backend) {
    if (backend == nullptr || !ggml_backend_is_cuda(backend)) {
        return 0;
    }
    auto * ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    return ctx->moe_ids_cache != nullptr ? ctx->moe_ids_cache->published.size() : 0;
}

struct ggml_cuda_moe_ids_host {
    const std::vector<char> * bytes = nullptr;
    size_t nb0 = 0;
    size_t nb1 = 0;
    size_t nb2 = 0;
    uint64_t d2h_time_us = 0;
    uint64_t d2h_sync_count = 0;
    bool cache_hit = false;
    bool group_pending = false;
};

struct ggml_cuda_moe_ids_publish {
    int32_t * ids = nullptr;
    uint32_t * ready = nullptr;
};

static void ggml_cuda_moe_ids_cache_clear_pending(ggml_cuda_moe_ids_cache_state & state) {
    state.device = -1;
    state.pending = false;
    state.kind = GGML_CUDA_MOE_IDS_KIND_NONE;
    state.key = {};
}

static ggml_cuda_moe_ids_kind ggml_cuda_moe_ids_kind_from_name(const char * name) {
    if (name == nullptr) {
        return GGML_CUDA_MOE_IDS_KIND_NONE;
    }
    if (strstr(name, "_gate_up_") != nullptr) {
        return GGML_CUDA_MOE_IDS_KIND_GATE_UP;
    }
    if (strstr(name, "_gate_") != nullptr) {
        return GGML_CUDA_MOE_IDS_KIND_GATE;
    }
    if (strstr(name, "_up_") != nullptr) {
        return GGML_CUDA_MOE_IDS_KIND_UP;
    }
    if (strstr(name, "_down_") != nullptr) {
        return GGML_CUDA_MOE_IDS_KIND_DOWN;
    }
    return GGML_CUDA_MOE_IDS_KIND_NONE;
}

bool ggml_cuda_moe_use_mmq(const ggml_tensor * src0, int64_t n_tokens) {
    const ggml_cuda_moe_ids_kind kind = ggml_cuda_moe_ids_kind_from_name(src0->name);
    return kind != GGML_CUDA_MOE_IDS_KIND_NONE && n_tokens > 1;
}

static bool ggml_cuda_moe_use_compact_mmvq(const ggml_tensor * dst, int64_t n_compact_experts) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    ggml_cuda_mmid_capability_query query;
    query.source_type = src0->type;
    query.input_type = src1->type;
    query.output_type = dst->type;
    memcpy(query.source_ne, src0->ne, sizeof(query.source_ne));
    memcpy(query.source_nb, src0->nb, sizeof(query.source_nb));
    query.n_tokens = src1->ne[2];
    query.n_experts = src0->ne[2];
    query.phase = query.n_tokens == 1 ? GGML_CUDA_MMID_PHASE_DECODE : GGML_CUDA_MMID_PHASE_PREFILL;
    query.mapping = GGML_CUDA_MMID_MAPPING_DIRECT;
    query.use_mmq = ggml_cuda_moe_use_mmq(src0, query.n_tokens);
    const auto & info = ggml_cuda_info();
    const int device = ggml_cuda_get_device();
    if (device < 0 || device >= info.device_count) {
        return false;
    }
    query.cc = info.devices[device].cc;
    query.warp_size = info.devices[device].warp_size;
    query.smpbo = info.devices[device].smpbo;
    return ggml_cuda_mmid_can_use_compact_mmvq(query, n_compact_experts);
}

static int ggml_cuda_moe_layer_from_name(const char * name) {
    if (name == nullptr) {
        return -1;
    }

    const char * blk = strstr(name, "blk.");
    if (blk == nullptr) {
        return -1;
    }

    blk += 4;
    int layer = 0;
    bool found_digit = false;
    while (*blk >= '0' && *blk <= '9') {
        found_digit = true;
        layer = 10 * layer + (*blk - '0');
        ++blk;
    }

    return found_digit ? layer : -1;
}

static ggml_cuda_moe_ids_cache_key ggml_cuda_moe_ids_cache_make_key(const ggml_tensor * ids, int layer) {
    return {
        ids->data,
        ggml_nbytes(ids),
        ids->ne[0],
        ids->ne[1],
        ids->ne[2],
        ids->ne[3],
        ids->nb[0],
        ids->nb[1],
        ids->nb[2],
        ids->nb[3],
        layer,
    };
}

// The GPU publishes this aligned flag in mapped host memory.
static uint32_t ggml_cuda_moe_ready_load(uint32_t * ready) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    return cuda::atomic_ref<uint32_t, cuda::thread_scope_system>(*ready).load(cuda::memory_order_acquire);
#else
    return __atomic_load_n(ready, __ATOMIC_ACQUIRE);
#endif
}

static void ggml_cuda_moe_ready_clear(uint32_t * ready) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    cuda::atomic_ref<uint32_t, cuda::thread_scope_system>(*ready).store(0, cuda::memory_order_release);
#else
    __atomic_store_n(ready, 0, __ATOMIC_RELEASE);
#endif
}

static ggml_cuda_moe_ids_publish ggml_cuda_moe_prepare_ids_publish(
        ggml_backend_cuda_context & ctx,
        const ggml_cgraph * graph,
        int next_node,
        const ggml_tensor * ids) {
    const size_t count = (size_t) ids->ne[0];
    if (ctx.moe_grouped_context == nullptr || ctx.moe_grouped_context->state().n_slots == 0 || ids->ne[1] * ids->ne[2] != 1 ||
            ids->nb[0] != sizeof(int32_t) || ggml_nbytes(ids) != count*sizeof(int32_t)) {
        return {};
    }

    // Only the next cached MMID can consume this host publication.
    const ggml_tensor * consumer = nullptr;
    for (int i = next_node; i < graph->n_nodes; ++i) {
        const ggml_tensor * node = graph->nodes[i];
        if (node->op == GGML_OP_MUL_MAT_ID && !ggml_is_empty(node) && (node->flags & GGML_TENSOR_FLAG_COMPUTE) != 0) {
            consumer = node;
            break;
        }
    }
    if (consumer == nullptr || consumer->src[2] != ids || consumer->src[0] == nullptr ||
            consumer->src[0]->buffer == nullptr ||
            !ggml_backend_buft_is_cuda_moe_cached(consumer->src[0]->buffer->buft)) {
        return {};
    }

    auto & ggml_cuda_moe_ids_cache = ggml_cuda_moe_ids_cache_get(ctx);
    const ggml_cuda_moe_ids_cache_state::published_key key{ids, ids->data, ctx.stream()};
    auto & published = ggml_cuda_moe_ids_cache.published;
    if (published.find(key) == published.end() && published.size() >= ggml_cuda_moe_ids_cache_state::MAX_PUBLISHED) {
        return {};
    }
    auto & entry = published[key];
    if (entry.host_ids && entry.count != count) {
        // Captured graphs can still write to this allocation.
        return {};
    }
    if (!entry.host_ids) {
        void * host = nullptr;
        const size_t bytes = count*sizeof(int32_t) + sizeof(uint32_t);
        cudaError_t err = cudaHostAlloc(&host, bytes, cudaHostAllocMapped);
        if (err != cudaSuccess) {
            cudaGetLastError();
            return {};
        }
        void * device_ptr = nullptr;
        err = cudaHostGetDevicePointer(&device_ptr, host, 0);
        if (err != cudaSuccess || !device_ptr) {
            cudaGetLastError();
            cudaFreeHost(host);
            return {};
        }
        entry.host_ids = (int32_t *) host;
        entry.device_ids = (int32_t *) device_ptr;
        entry.host_ready = (uint32_t *) (entry.host_ids + count);
        entry.device_ready = (uint32_t *) (entry.device_ids + count);
        entry.count = count;
        ggml_cuda_moe_ready_clear(entry.host_ready);
    }
    // A failed/skipped consumer can leave this producer's previous publication in flight.
    // Do not clear its flag here: that older producer can still set it after this call.
    entry.readable = !entry.unconsumed;
    entry.unconsumed = true;
    entry.dispatch_id = ggml_cuda_moe_ids_cache.dispatch_id;
    return {entry.device_ids, entry.device_ready};
}

static ggml_cuda_moe_ids_host ggml_cuda_moe_read_ids(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * ids,
        const char * tensor_name,
        std::vector<char> & storage) {
    ggml_cuda_moe_ids_host result;
    const int device = ggml_cuda_get_device();
    auto & ggml_cuda_moe_ids_cache = ggml_cuda_moe_ids_cache_get(ctx);
    const int layer = ggml_cuda_moe_layer_from_name(tensor_name);
    const ggml_cuda_moe_ids_cache_key key = ggml_cuda_moe_ids_cache_make_key(ids, layer);
    const ggml_cuda_moe_ids_kind kind = ggml_cuda_moe_ids_kind_from_name(tensor_name);

    if (layer < 0) {
        ggml_cuda_moe_ids_cache_clear_pending(ggml_cuda_moe_ids_cache);
    } else if (ggml_cuda_moe_ids_cache.pending &&
            (ggml_cuda_moe_ids_cache.device != device || !(ggml_cuda_moe_ids_cache.key == key))) {
        ggml_cuda_moe_ids_cache_clear_pending(ggml_cuda_moe_ids_cache);
    }

    if (ggml_cuda_moe_ids_cache.pending) {
        if ((kind == GGML_CUDA_MOE_IDS_KIND_GATE && ggml_cuda_moe_ids_cache.kind == GGML_CUDA_MOE_IDS_KIND_UP) ||
            (kind == GGML_CUDA_MOE_IDS_KIND_UP && ggml_cuda_moe_ids_cache.kind == GGML_CUDA_MOE_IDS_KIND_GATE)) {
            ggml_cuda_moe_ids_cache.kind = GGML_CUDA_MOE_IDS_KIND_GATE_AND_UP;
            result.cache_hit = true;
        } else if (kind == GGML_CUDA_MOE_IDS_KIND_DOWN &&
                (ggml_cuda_moe_ids_cache.kind == GGML_CUDA_MOE_IDS_KIND_UP ||
                 ggml_cuda_moe_ids_cache.kind == GGML_CUDA_MOE_IDS_KIND_GATE_UP ||
                 ggml_cuda_moe_ids_cache.kind == GGML_CUDA_MOE_IDS_KIND_GATE_AND_UP)) {
            result.cache_hit = true;
            ggml_cuda_moe_ids_cache_clear_pending(ggml_cuda_moe_ids_cache);
        } else {
            ggml_cuda_moe_ids_cache_clear_pending(ggml_cuda_moe_ids_cache);
        }

        if (result.cache_hit) {
            result.bytes = &ggml_cuda_moe_ids_cache.bytes;
            result.nb0 = ggml_cuda_moe_ids_cache.nb0;
            result.nb1 = ggml_cuda_moe_ids_cache.nb1;
            result.nb2 = ggml_cuda_moe_ids_cache.nb2;
        }
    }

    if (!result.cache_hit) {
        const bool starts_group = layer >= 0 && (kind == GGML_CUDA_MOE_IDS_KIND_GATE ||
            kind == GGML_CUDA_MOE_IDS_KIND_UP || kind == GGML_CUDA_MOE_IDS_KIND_GATE_UP);
        std::vector<char> * target = starts_group ? &ggml_cuda_moe_ids_cache.bytes : &storage;
        const int64_t start_us = ggml_time_us();
        const size_t row_bytes = ids->ne[0] * sizeof(int32_t);
        const bool compact_ids = ids->type == GGML_TYPE_I32 && ids->ne[3] == 1 &&
            ids->nb[0] == sizeof(int32_t) && ids->nb[1] >= row_bytes &&
            ids->nb[2] == ids->nb[1] * (size_t) ids->ne[1];
        if (compact_ids) {
            const size_t rows = (size_t) ids->ne[1] * ids->ne[2];
            target->resize(row_bytes * rows);
            result.nb0 = sizeof(int32_t);
            result.nb1 = row_bytes;
            result.nb2 = row_bytes * (size_t) ids->ne[1];
        } else {
            target->resize(ggml_nbytes(ids));
            result.nb0 = ids->nb[0];
            result.nb1 = ids->nb[1];
            result.nb2 = ids->nb[2];
        }

        const ggml_cuda_moe_ids_cache_state::published_key published_key{ids, ids->data, ctx.stream()};
        auto published = ggml_cuda_moe_ids_cache.published.find(published_key);
        // A retained allocation is not evidence that this dispatch produced these IDs.
        const bool can_use_published = ids->ne[1] * ids->ne[2] == 1 && published != ggml_cuda_moe_ids_cache.published.end() &&
            published->second.count == (size_t) ids->ne[0] && target->size() == published->second.count*sizeof(int32_t) &&
            published->second.dispatch_id == ggml_cuda_moe_ids_cache.dispatch_id &&
            published->second.unconsumed;
        bool ready = false;
        if (can_use_published && published->second.readable) {
            const int64_t spin_start_us = ggml_time_us();
            do {
                for (int spin = 0; spin < 1024; ++spin) {
                    if (ggml_cuda_moe_ready_load(published->second.host_ready) != 0) {
                        ready = true;
                        break;
                    }
                }
            } while (!ready && ggml_time_us() - spin_start_us < 1000);
            if (ready) {
                memcpy(target->data(), published->second.host_ids, target->size());
            }
        }
        if (!ready) {
            if (compact_ids) {
                const size_t rows = (size_t) ids->ne[1] * ids->ne[2];
                CUDA_CHECK(cudaMemcpy2DAsync(
                    target->data(), row_bytes, ids->data, ids->nb[1],
                    row_bytes, rows, cudaMemcpyDeviceToHost, ctx.stream()));
            } else {
                CUDA_CHECK(cudaMemcpyAsync(target->data(), ids->data, ggml_nbytes(ids), cudaMemcpyDeviceToHost, ctx.stream()));
            }
            CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
            result.d2h_sync_count = 1;
        }
        if (can_use_published) {
            ggml_cuda_moe_ready_clear(published->second.host_ready);
            published->second.unconsumed = false;
        }
        result.d2h_time_us = (uint64_t) (ggml_time_us() - start_us);
        result.bytes = target;

        if (starts_group) {
            ggml_cuda_moe_ids_cache.device = device;
            ggml_cuda_moe_ids_cache.key = key;
            ggml_cuda_moe_ids_cache.kind = kind;
            ggml_cuda_moe_ids_cache.pending = true;
            ggml_cuda_moe_ids_cache.nb0 = result.nb0;
            ggml_cuda_moe_ids_cache.nb1 = result.nb1;
            ggml_cuda_moe_ids_cache.nb2 = result.nb2;
        } else {
            ggml_cuda_moe_ids_cache_clear_pending(ggml_cuda_moe_ids_cache);
        }
    }

    result.group_pending = ggml_cuda_moe_ids_cache.pending &&
        ggml_cuda_moe_ids_cache.device == device && ggml_cuda_moe_ids_cache.key == key;
    return result;
}

struct ggml_cuda_moe_host_staging_runtime_capability {
    ggml_cuda_mmid_consumer consumer = GGML_CUDA_MMID_CONSUMER_UNSUPPORTED;
    ggml_cuda_mmid_mapping mapping = GGML_CUDA_MMID_MAPPING_DIRECT;
    int64_t materialized_experts = 0;
};

static bool ggml_cuda_moe_host_staging_capability(
        const ggml_tensor * dst,
        const ggml_cuda_moe_graph_capability_witness & witness,
        int64_t n_unique,
        ggml_cuda_moe_host_staging_runtime_capability * runtime) {
    if (dst == nullptr || dst->src[0] == nullptr || dst->src[1] == nullptr || runtime == nullptr || n_unique <= 0 ||
            witness.tensor != dst->src[0] || witness.strategy != GGML_CUDA_MOE_EXECUTION_STRATEGY_HOST_STAGED ||
            witness.consumer == GGML_CUDA_MMID_CONSUMER_UNSUPPORTED ||
            (witness.materialized_mapping != GGML_CUDA_MMID_MAPPING_DIRECT &&
                witness.materialized_mapping != GGML_CUDA_MMID_MAPPING_SOURCE_MAP)) {
        return false;
    }

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const auto & info = ggml_cuda_info();
    const int device = ggml_cuda_get_device();
    if (device < 0 || device >= info.device_count || src0->ne[2] <= 0 || n_unique > src0->ne[2]) {
        return false;
    }

    ggml_cuda_mmid_capability_query native;
    native.source_type = src0->type;
    native.input_type = src1->type;
    native.output_type = dst->type;
    memcpy(native.source_ne, src0->ne, sizeof(native.source_ne));
    memcpy(native.source_nb, src0->nb, sizeof(native.source_nb));
    native.n_tokens = src1->ne[2];
    native.n_experts = src0->ne[2];
    native.cc = info.devices[device].cc;
    native.warp_size = info.devices[device].warp_size;
    native.smpbo = info.devices[device].smpbo;
    native.phase = native.n_tokens == 1 ? GGML_CUDA_MMID_PHASE_DECODE : GGML_CUDA_MMID_PHASE_PREFILL;
    native.mapping = GGML_CUDA_MMID_MAPPING_DIRECT;
    native.use_mmq = ggml_cuda_moe_use_mmq(src0, native.n_tokens);
    if (witness.n_rows > 1 && !ggml_cuda_mmid_can_use_compact_mmvq(native, witness.n_slots)) {
        native.mapping = GGML_CUDA_MMID_MAPPING_SOURCE_MAP;
    }
    const auto native_capability = ggml_cuda_mmid_get_capability(native);
    if (native_capability.reason != GGML_CUDA_MMID_CAPABILITY_OK ||
            native_capability.selection != witness.consumer || witness.phase != native.phase ||
            witness.mapping != native.mapping || witness.n_tokens != native.n_tokens ||
            witness.n_experts != native.n_experts || witness.use_mmq != static_cast<uint32_t>(native.use_mmq)) {
        return false;
    }

    ggml_cuda_mmid_capability_query materialized = native;
    materialized.phase = static_cast<ggml_cuda_mmid_phase>(witness.materialized_phase);
    materialized.mapping = static_cast<ggml_cuda_mmid_mapping>(witness.materialized_mapping);
    materialized.preferred_consumer = static_cast<ggml_cuda_mmid_consumer>(witness.consumer);
    materialized.independent_rows = witness.row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT;
    const auto expected_phase = materialized.independent_rows ? GGML_CUDA_MMID_PHASE_DECODE : native.phase;
    if (materialized.phase != expected_phase) {
        return false;
    }
    if (materialized.mapping == GGML_CUDA_MMID_MAPPING_DIRECT) {
        if (materialized.source_nb[2] > SIZE_MAX / static_cast<size_t>(n_unique)) {
            return false;
        }
        materialized.n_experts = n_unique;
        materialized.source_ne[2] = n_unique;
        materialized.source_nb[3] = materialized.source_nb[2] * static_cast<size_t>(n_unique);
    }
    const auto capability = ggml_cuda_mmid_get_capability(materialized);
    if (capability.reason != GGML_CUDA_MMID_CAPABILITY_OK || capability.selection != witness.consumer) {
        return false;
    }
    runtime->consumer = capability.selection;
    runtime->mapping = materialized.mapping;
    runtime->materialized_experts = materialized.mapping == GGML_CUDA_MMID_MAPPING_DIRECT ? n_unique : src0->ne[2];
    return true;
}

static void ggml_cuda_moe_record_legacy_op(
        ggml_cuda_moe_grouped_context * owner,
        bool is_decode,
        bool staged,
        bool split_staged,
        bool overflow,
        uint64_t unique_experts,
        uint64_t ids_bytes,
        uint64_t ids_d2h_time_us,
        uint64_t ids_d2h_sync_count,
        uint64_t acquire_time_us,
        uint64_t remap_time_us,
        uint64_t copy_wait_event_count,
        uint64_t copy_wait_event_time_us,
        uint64_t total_time_us,
        bool ids_cache_hit) {
    if (owner != nullptr) {
        owner->record_legacy_op(
            is_decode, staged, split_staged, overflow, unique_experts, ids_bytes, ids_d2h_time_us, ids_d2h_sync_count,
            acquire_time_us, remap_time_us, copy_wait_event_count, copy_wait_event_time_us, total_time_us, ids_cache_hit);
    } else {
        ggml_cuda_moe_record_op_stats(
            is_decode, staged, split_staged, overflow, unique_experts, ids_bytes, ids_d2h_time_us, ids_d2h_sync_count,
            acquire_time_us, remap_time_us, copy_wait_event_count, copy_wait_event_time_us, total_time_us, ids_cache_hit);
    }
}

// Per-op staging fallback for the cached path. Used when the persistent cache
// either isn't initialized or its slot size doesn't match this op's expert
// stride. Same approach as iteration 1: stage every unique expert touched by
// this op into a pool scratch buffer, run the impl, free the scratch.
static void ggml_cuda_mul_mat_id_staged(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                        const std::vector<char> & ids_host_bytes,
                                        size_t ids_host_nb0,
                                        size_t ids_host_nb1,
                                        size_t ids_host_nb2,
                                        bool single_row,
                                        bool telemetry_is_decode,
                                        bool overflow,
                                        ggml_cuda_moe_cache * cache,
                                        ggml_cuda_moe_grouped_context * owner,
                                        bool ids_cache_hit,
                                        const void * resident_pool = nullptr,
                                        const std::vector<int32_t> * resident_slots = nullptr,
                                        bool record_legacy_telemetry = true) {
    const int64_t op_start_us = ggml_time_us();
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * ids  = dst->src[2];
    cudaStream_t  stream = ctx.stream();
    const bool    use_mmq = ggml_cuda_moe_use_mmq(src0, dst->src[1]->ne[2]);

    const int64_t n_experts_total = src0->ne[2];
    const size_t  expert_stride   = src0->nb[2];

    const int64_t ids_ne0 = ids->ne[0];
    const int64_t ids_ne1 = ids->ne[1];
    const int64_t ids_ne2 = ids->ne[2];
    std::vector<int32_t> expert_to_pos(n_experts_total, -1);
    std::vector<int32_t> unique_experts;
    unique_experts.reserve(std::min<int64_t>(n_experts_total, 64));

    for (int64_t i2 = 0; i2 < ids_ne2; ++i2) {
        for (int64_t i1 = 0; i1 < ids_ne1; ++i1) {
            for (int64_t i0 = 0; i0 < ids_ne0; ++i0) {
                const int32_t eid = *(const int32_t *)(ids_host_bytes.data()
                    + i2*ids_host_nb2 + i1*ids_host_nb1 + i0*ids_host_nb0);
                GGML_ASSERT(eid >= 0 && eid < n_experts_total);
                if (expert_to_pos[eid] < 0) {
                    expert_to_pos[eid] = (int32_t)unique_experts.size();
                    unique_experts.push_back(eid);
                }
            }
        }
    }

    const int n_unique = (int)unique_experts.size();
    GGML_ASSERT(n_unique > 0);
    const bool compact_mmvq = !single_row && ggml_cuda_moe_use_compact_mmvq(dst, n_unique);
    const auto source_capability = ggml_cuda_mmid_source_capability_for(src0->type);
    const bool compact_mmq = !single_row && !compact_mmvq && use_mmq &&
        (source_capability.flags & GGML_CUDA_MMID_SOURCE_MMQ) != 0 &&
        (source_capability.flags & GGML_CUDA_MMID_SOURCE_MAPPED_MMQ) == 0 &&
        ggml_cuda_should_use_mmq(
            src0->type, ggml_cuda_info().devices[ggml_cuda_get_device()].cc, dst->src[1]->ne[2], n_unique);
    const bool compact_source = single_row || compact_mmvq || compact_mmq;

    // Match CUDA tensor tail padding for staged quantized sources.
    const int64_t row_remainder = src0->ne[0] % MATRIX_ROW_PADDING;
    const size_t source_padding = ggml_is_quantized(src0->type) && row_remainder != 0 ?
        ggml_row_size(src0->type, MATRIX_ROW_PADDING - row_remainder) : 0;
    GGML_ASSERT(expert_stride > 0 && (size_t) n_unique <= (SIZE_MAX - source_padding) / expert_stride);
    const size_t staged_expert_bytes = (size_t) n_unique * expert_stride;
    const size_t staged_source_bytes = staged_expert_bytes + source_padding;
    ggml_cuda_pool_alloc<char> scratch_experts(ctx.pool(), staged_source_bytes);

    const char * src_base = (const char *)src0->data;
    std::vector<const void *> host_ptrs;
    if (overflow && cache) {
        host_ptrs.reserve(n_unique);
        for (int32_t eid : unique_experts) {
            host_ptrs.push_back(src_base + (size_t)eid * expert_stride);
        }
    }

    std::vector<int> split_slot_ids(n_unique, -1);
    std::vector<int32_t> source_wait_class_host;
    ggml_cuda_pool_alloc<uint32_t> stage_ready(ctx.pool());
    int stage_ready_capacity = 0;
    if (overflow && !compact_source && ggml_cuda_moe_cache_can_overlap_staging(cache)) {
        const int n_slots = ggml_cuda_moe_cache_n_slots(cache);
        GGML_ASSERT(n_slots > 0);
        const size_t n_staging_waves = 1 + ((size_t) n_unique - 1) / (size_t) n_slots;
        GGML_ASSERT(n_staging_waves < (size_t) std::numeric_limits<int>::max());
        stage_ready_capacity = 1 + (int) n_staging_waves;
        source_wait_class_host.resize(n_unique);
        stage_ready.alloc(stage_ready_capacity);
    }
    int n_resident = 0;
    int n_wait_classes = 1;
    bool split_staged = false;
    if (overflow && cache && src0->type != GGML_TYPE_MXFP4 && src0->type != GGML_TYPE_NVFP4) {
        const int n_slots = ggml_cuda_moe_cache_n_slots(cache);
        const int min_resident = n_slots / 6 + (n_slots % 6 != 0);
        if (!source_wait_class_host.empty()) {
            if (owner != nullptr && ggml_cuda_moe_take_split_staging_poison_for_test(owner)) {
                CUDA_CHECK(cudaMemsetAsync(scratch_experts.get(), 0xff, staged_source_bytes, stream));
            }
            split_staged = ggml_cuda_moe_cache_prepare_split_staging(
                cache, host_ptrs.data(), n_unique, expert_stride, source_padding, min_resident,
                split_slot_ids.data(), source_wait_class_host.empty() ? nullptr : source_wait_class_host.data(),
                &n_resident, scratch_experts.get(), stage_ready.get(), stage_ready_capacity, &n_wait_classes, stream,
                telemetry_is_decode);
        }
    }

    bool cache_staged = false;
    if (overflow && cache && !split_staged) {
        cache_staged = ggml_cuda_moe_cache_copy_to_staging(
            cache, host_ptrs.data(), n_unique, expert_stride, scratch_experts.get(), stream, telemetry_is_decode);
    }

    if (!cache_staged && !split_staged) {
        char * dst_base = (char *)scratch_experts.get();
        for (int i = 0; i < n_unique; ++i) {
            const int32_t eid = unique_experts[i];
            const int32_t slot = resident_slots != nullptr && static_cast<size_t>(eid) < resident_slots->size() ?
                (*resident_slots)[eid] : -1;
            if (slot >= 0 && resident_pool != nullptr) {
                CUDA_CHECK(cudaMemcpyAsync(
                    dst_base + (size_t)i * expert_stride,
                    static_cast<const char *>(resident_pool) + (size_t)slot * expert_stride,
                    expert_stride,
                    cudaMemcpyDeviceToDevice, stream));
            } else {
                if (owner != nullptr) {
                    GGML_ASSERT(owner->copy_staged_source(
                        src0, dst_base + (size_t)i * expert_stride,
                        src_base + (size_t)eid * expert_stride, expert_stride, stream));
                } else {
                    CUDA_CHECK(cudaMemcpyAsync(
                        dst_base + (size_t)i * expert_stride,
                        src_base + (size_t)eid * expert_stride,
                        expert_stride,
                        cudaMemcpyHostToDevice, stream));
                }
            }
        }
    }
    if (!split_staged && source_padding > 0) {
        CUDA_CHECK(cudaMemsetAsync(
            scratch_experts.get() + staged_expert_bytes, 0, source_padding, stream));
    }

    const size_t ids_total_elems = (size_t)ids_ne0 * ids_ne1 * ids_ne2;
    if (split_staged) {
        const int n_slots = ggml_cuda_moe_cache_n_slots(cache);
        std::vector<int32_t> expert_source_host(n_experts_total, -1);
        std::vector<int32_t> expert_wait_class_host(n_experts_total, -1);
        int n_misses = 0;
        for (int i = 0; i < n_unique; ++i) {
            const int32_t eid = unique_experts[i];
            if (split_slot_ids[i] < 0) {
                expert_source_host[eid] = n_slots + n_misses++;
            } else {
                expert_source_host[eid] = split_slot_ids[i];
            }
            expert_wait_class_host[eid] = source_wait_class_host[i];
        }
        GGML_ASSERT(n_resident > 0 && n_misses > 0 && n_resident + n_misses == n_unique);

        ggml_tensor src0_synth = *src0;
        src0_synth.ne[2] = n_experts_total;
        src0_synth.nb[3] = src0_synth.nb[2] * (size_t)n_experts_total;
        src0_synth.data = ggml_cuda_moe_cache_slot_ptr(cache, 0);

        ggml_tensor * orig_src0 = dst->src[0];
        dst->src[0] = &src0_synth;
        const ggml_cuda_mul_mat_id_host_route host_route = {
            ids_host_bytes.data(), ids_host_nb0, ids_host_nb1, expert_source_host.data(),
            scratch_experts.get(), n_slots, n_misses, expert_wait_class_host.data(), n_wait_classes, stage_ready.get(),
        };
        ggml_cuda_mul_mat_id_impl(ctx, dst, use_mmq, &host_route);
        dst->src[0] = orig_src0;

        if (stage_ready.get() != nullptr) {
            const bool staging_finished = ggml_cuda_moe_cache_finish_split_staging(cache, stream);
            GGML_ASSERT(staging_finished);
        }

        const bool slots_released = ggml_cuda_moe_cache_release_split_slots(
            cache, split_slot_ids.data(), n_unique, stream);
        GGML_ASSERT(slots_released);
        if (record_legacy_telemetry) {
            ggml_cuda_moe_record_legacy_op(owner,
                telemetry_is_decode, true, true, overflow, (uint64_t)n_unique, (uint64_t)ggml_nbytes(ids),
                0, 0, 0, 0, 0, 0, (uint64_t)(ggml_time_us() - op_start_us), ids_cache_hit);
        }
        return;
    }

    if (compact_source) {
        std::vector<int32_t> remapped_ids_host;
        remapped_ids_host.reserve(ids_total_elems);
        for (int64_t i2 = 0; i2 < ids_ne2; ++i2) {
            for (int64_t i1 = 0; i1 < ids_ne1; ++i1) {
                for (int64_t i0 = 0; i0 < ids_ne0; ++i0) {
                    const int32_t eid = *(const int32_t *)(ids_host_bytes.data()
                        + i2*ids_host_nb2 + i1*ids_host_nb1 + i0*ids_host_nb0);
                    remapped_ids_host.push_back(expert_to_pos[eid]);
                }
            }
        }

        ggml_cuda_pool_alloc<int32_t> scratch_ids(ctx.pool(), ids_total_elems);
        CUDA_CHECK(cudaMemcpyAsync(scratch_ids.get(), remapped_ids_host.data(),
                                   ids_total_elems * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

        ggml_tensor src0_synth = *src0;
        src0_synth.ne[2] = n_unique;
        src0_synth.nb[3] = src0_synth.nb[2] * (size_t) n_unique;
        src0_synth.data = scratch_experts.get();

        ggml_tensor ids_synth = *ids;
        ids_synth.data = scratch_ids.get();
        ids_synth.nb[0] = sizeof(int32_t);
        ids_synth.nb[1] = (size_t)ids_ne0 * sizeof(int32_t);
        ids_synth.nb[2] = (size_t)ids_ne0 * ids_ne1 * sizeof(int32_t);
        ids_synth.nb[3] = ids_synth.nb[2];

        ggml_tensor * orig_src0 = dst->src[0];
        ggml_tensor * orig_ids = dst->src[2];
        dst->src[0] = &src0_synth;
        dst->src[2] = &ids_synth;
        const bool dispatched = ggml_cuda_mul_mat_id_impl(
            ctx, dst, use_mmq, nullptr,
            compact_mmvq ? GGML_CUDA_MMID_CONSUMER_MMVQ :
                compact_mmq ? GGML_CUDA_MMID_CONSUMER_MMQ : GGML_CUDA_MMID_CONSUMER_UNSUPPORTED);
        dst->src[0] = orig_src0;
        dst->src[2] = orig_ids;
        GGML_ASSERT(dispatched);

        if (record_legacy_telemetry) {
            ggml_cuda_moe_record_legacy_op(owner,
                telemetry_is_decode, true, false, overflow, (uint64_t) n_unique, (uint64_t) ggml_nbytes(ids),
                0, 0, 0, 0, 0, 0, (uint64_t) (ggml_time_us() - op_start_us), ids_cache_hit);
        }
        return;
    }

    ggml_tensor src0_synth = *src0;
    src0_synth.ne[2] = n_experts_total;
    src0_synth.nb[3] = src0_synth.nb[2] * (size_t) n_experts_total;
    src0_synth.data  = scratch_experts.get();

    ggml_tensor * orig_src0 = dst->src[0];
    dst->src[0] = &src0_synth;

    const ggml_cuda_mul_mat_id_host_route host_route = {
        ids_host_bytes.data(), ids_host_nb0, ids_host_nb1, expert_to_pos.data(),
    };
    ggml_cuda_mul_mat_id_impl(ctx, dst, use_mmq, &host_route);

    dst->src[0] = orig_src0;

    if (record_legacy_telemetry) {
        ggml_cuda_moe_record_legacy_op(owner,
            telemetry_is_decode, true, false, overflow, (uint64_t) n_unique, (uint64_t) ggml_nbytes(ids),
            0, 0, 0, 0, 0, 0, (uint64_t) (ggml_time_us() - op_start_us), ids_cache_hit);
    }
}

static bool ggml_cuda_mul_mat_id_grouped_host_staged(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        ggml_cuda_moe_graph_group_dispatch * group,
        const ggml_cuda_moe_graph_binding & binding) {
    auto * owner = ctx.moe_grouped_context;
    if (owner == nullptr || group == nullptr ||
            (group->state != GGML_CUDA_MOE_GRAPH_GROUP_HOST_STAGED_ARMED &&
                group->state != GGML_CUDA_MOE_GRAPH_GROUP_HOST_STAGED_ACTIVE)) {
        return false;
    }

    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * ids = dst->src[2];
    if (src0 == nullptr || ids == nullptr || ids->type != GGML_TYPE_I32 || src0->ne[2] <= 0 ||
            ids->ne[0] <= 0 || ids->ne[1] <= 0 || ids->ne[2] != 1 || ids->ne[3] != 1 ||
            static_cast<uint64_t>(ids->ne[0]) > SIZE_MAX / static_cast<uint64_t>(ids->ne[1])) {
        return false;
    }
    const size_t n_ids = static_cast<size_t>(ids->ne[0]) * static_cast<size_t>(ids->ne[1]);
    if (n_ids == 0 || n_ids > SIZE_MAX / sizeof(int32_t)) {
        return false;
    }

    std::vector<char> ids_host_storage;
    const ggml_cuda_moe_ids_host ids_host = ggml_cuda_moe_read_ids(ctx, ids, src0->name, ids_host_storage);
    if (ids_host.bytes == nullptr) {
        return false;
    }
    const int64_t n_experts = src0->ne[2];
    std::vector<int32_t> expert_to_pos(static_cast<size_t>(n_experts), -1);
    std::vector<int32_t> unique_experts;
    unique_experts.reserve(std::min<int64_t>(n_experts, static_cast<int64_t>(n_ids)));
    for (int64_t i1 = 0; i1 < ids->ne[1]; ++i1) {
        for (int64_t i0 = 0; i0 < ids->ne[0]; ++i0) {
            int32_t expert = -1;
            memcpy(&expert, ids_host.bytes->data() + i1 * ids_host.nb1 + i0 * ids_host.nb0, sizeof(expert));
            if (expert < 0 || expert >= n_experts) {
                return false;
            }
            if (expert_to_pos[expert] < 0) {
                expert_to_pos[expert] = static_cast<int32_t>(unique_experts.size());
                unique_experts.push_back(expert);
            }
        }
    }
    if (unique_experts.empty() || unique_experts.size() > static_cast<size_t>(UINT32_MAX)) {
        return false;
    }

    ggml_cuda_moe_host_staging_runtime_capability runtime;
    if (!ggml_cuda_moe_host_staging_capability(dst, group->capabilities[binding.slot_index], unique_experts.size(), &runtime)) {
        return false;
    }
    if (group->state == GGML_CUDA_MOE_GRAPH_GROUP_HOST_STAGED_ARMED) {
        if (owner->prepare_host_staged_group(
                group, binding, dst, ctx.stream(), unique_experts.data(), unique_experts.size()) !=
                GGML_CUDA_MOE_GROUPED_DECODE_READY) {
            return false;
        }
        if (ggml_cuda_moe_take_host_staged_evaluator_failure_for_test(owner)) {
            return false;
        }
    }
    if (group->state != GGML_CUDA_MOE_GRAPH_GROUP_HOST_STAGED_ACTIVE ||
            binding.slot_index >= group->key.n_banks || group->bank_data[binding.slot_index] == nullptr ||
            group->prefill_slot_for_expert.size() != static_cast<size_t>(n_experts)) {
        return false;
    }

    const bool single_row = ids->ne[1] * ids->ne[2] == 1;
    const bool telemetry_is_decode = group->authority.legacy_telemetry_is_decode(single_row);
    ggml_cuda_mul_mat_id_staged(
        ctx, dst, *ids_host.bytes, ids_host.nb0, ids_host.nb1, ids_host.nb2,
        single_row, telemetry_is_decode, false, nullptr, owner, ids_host.cache_hit,
        group->bank_data[binding.slot_index], &group->prefill_slot_for_expert, false);
    owner->record_host_staged_op(false);
    return true;
}


static void ggml_cuda_mul_mat_id_cached(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        const ggml_cuda_moe_group_call_lease * authority) {
    const int64_t op_start_us = ggml_time_us();
    ggml_tensor * src0 = dst->src[0];   // experts in CPU pinned
    ggml_tensor * ids  = dst->src[2];   // routing decision

    cudaStream_t stream = ctx.stream();
    const bool   single_row = ids->ne[1] * ids->ne[2] == 1;
    const bool   telemetry_is_decode = authority != nullptr ?
        authority->legacy_telemetry_is_decode(single_row) : single_row;
    const bool   use_mmq = ggml_cuda_moe_use_mmq(src0, dst->src[1]->ne[2]);
    auto * owner = ctx.moe_grouped_context;
    auto owner_lease = owner != nullptr ? owner->begin_legacy_operation() : ggml_cuda_moe_legacy_operation_lease{};
    auto * leased_owner = owner_lease ? owner : nullptr;
    const uint32_t top_k = ids->ne[0] > 0 && ids->ne[0] <= UINT32_MAX ? static_cast<uint32_t>(ids->ne[0]) : 1;
    auto cache_lease = leased_owner != nullptr ? leased_owner->acquire_legacy_cache(src0, nullptr, authority, stream, top_k) : ggml_cuda_moe_legacy_cache_lease{};
    ggml_cuda_moe_cache * cache = cache_lease.get();

    std::vector<char> ids_host_storage;
    const ggml_cuda_moe_ids_host ids_host = ggml_cuda_moe_read_ids(ctx, ids, src0->name, ids_host_storage);
    const std::vector<char> * ids_host_bytes = ids_host.bytes;
    const size_t ids_host_nb0 = ids_host.nb0;
    const size_t ids_host_nb1 = ids_host.nb1;
    const size_t ids_host_nb2 = ids_host.nb2;
    const uint64_t ids_d2h_time_us = ids_host.d2h_time_us;
    const uint64_t ids_d2h_sync_count = ids_host.d2h_sync_count;
    const bool ids_cache_hit = ids_host.cache_hit;
    const size_t expert_stride = src0->nb[2];

    if (cache == nullptr) {
        ggml_cuda_mul_mat_id_staged(
            ctx, dst, *ids_host_bytes, ids_host_nb0, ids_host_nb1, ids_host_nb2,
            single_row, telemetry_is_decode, false, nullptr, leased_owner, ids_cache_hit);
        return;
    }

    // 3. First pass: count unique experts referenced by this op. If the
    //    count exceeds the cache's slot count, eviction would happen
    //    mid-op and orphan earlier acquires in our local map (we would
    //    record expert E0 -> slot 0 but a later acquire of E18 evicts E0
    //    from slot 0 to make room, leaving our map stale). Detect this
    //    and stage instead so the kernel reads correct data.
    const int64_t n_experts_total = src0->ne[2];
    const int64_t ids_ne0 = ids->ne[0];
    const int64_t ids_ne1 = ids->ne[1];
    const int64_t ids_ne2 = ids->ne[2];

    // Single dedup pass: build unique_eids with -2 sentinel for "seen".
    // If we exceed the cache's slot count, fall back to staging.
    const int slot_capacity = ggml_cuda_moe_cache_n_slots(cache);
    std::vector<int32_t> expert_to_slot(n_experts_total, -1);
    std::vector<int32_t> unique_eids;
    unique_eids.reserve(std::min<int64_t>(n_experts_total, 64));
    bool overflow = false;
    for (int64_t i2 = 0; i2 < ids_ne2 && !overflow; ++i2) {
        for (int64_t i1 = 0; i1 < ids_ne1 && !overflow; ++i1) {
            for (int64_t i0 = 0; i0 < ids_ne0 && !overflow; ++i0) {
                const int32_t eid = *(const int32_t *)(ids_host_bytes->data()
                    + i2*ids_host_nb2 + i1*ids_host_nb1 + i0*ids_host_nb0);
                GGML_ASSERT(eid >= 0 && eid < n_experts_total);
                if (expert_to_slot[eid] == -2) continue;
                if ((int)unique_eids.size() >= slot_capacity) {
                    overflow = true;
                    break;
                }
                expert_to_slot[eid] = -2;
                unique_eids.push_back(eid);
            }
        }
    }

    if (overflow) {
        // More unique experts than the cache can hold simultaneously.
        // Stage so no slot gets overwritten mid-op.
        ggml_cuda_mul_mat_id_staged(
            ctx, dst, *ids_host_bytes, ids_host_nb0, ids_host_nb1, ids_host_nb2,
            single_row, telemetry_is_decode, true, cache, leased_owner, ids_cache_hit);
        return;
    }

    const bool compact_mmvq = !single_row && ggml_cuda_moe_use_compact_mmvq(dst, slot_capacity);
    const bool compact_source = single_row || compact_mmvq;

    const int64_t acquire_start_us = ggml_time_us();

    // 5. Reset sentinels and acquire each unique expert on this tensor's cache.
    //    Acquires issue H2D on the cache's dedicated copy stream so they
    //    pipeline with the compute stream's previous-op kernel.
    for (int32_t eid : unique_eids) expert_to_slot[eid] = -1;
    const char * src_base = (const char *)src0->data;
    cudaStream_t copy_stream = ggml_cuda_moe_cache_copy_stream(cache);
    bool any_cache_failure = false;
    std::vector<int> pinned_slots;
    pinned_slots.reserve(unique_eids.size());

    for (int64_t i2 = 0; i2 < ids_ne2; ++i2) {
        for (int64_t i1 = 0; i1 < ids_ne1; ++i1) {
            for (int64_t i0 = 0; i0 < ids_ne0; ++i0) {
                const int32_t eid = *(const int32_t *)(ids_host_bytes->data()
                    + i2*ids_host_nb2 + i1*ids_host_nb1 + i0*ids_host_nb0);
                if (expert_to_slot[eid] >= 0) continue;

                const void * host_ptr = src_base + (size_t)eid * expert_stride;
                int slot = ggml_cuda_moe_cache_acquire(
                    cache, host_ptr, expert_stride, copy_stream, telemetry_is_decode, false, true);
                if (slot < 0) {
                    any_cache_failure = true;
                    break;
                }
                expert_to_slot[eid] = slot;
                pinned_slots.push_back(slot);
            }
            if (any_cache_failure) break;
        }
        if (any_cache_failure) break;
    }

    if (any_cache_failure) {
        ggml_cuda_moe_cache_release_slots(cache, pinned_slots.data(), (int) pinned_slots.size());
        // Cache had an internal failure (e.g., cudaMemcpyAsync error mid-op).
        // Fall back so we still produce correct output.
        ggml_cuda_mul_mat_id_staged(
            ctx, dst, *ids_host_bytes, ids_host_nb0, ids_host_nb1, ids_host_nb2,
            single_row, telemetry_is_decode, false, nullptr, leased_owner, ids_cache_hit);
        return;
    }
    const uint64_t acquire_time_us = (uint64_t) (ggml_time_us() - acquire_start_us);
    uint64_t copy_wait_event_time_us = 0;

    // Compute stream must wait for all pending H2D copies on copy_stream
    // before reading the slot pool. Use an event to express the dependency.
    {
        const int64_t event_start_us = ggml_time_us();
        cudaEvent_t copy_done;
        CUDA_CHECK(cudaEventCreateWithFlags(&copy_done, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(copy_done, copy_stream));
        CUDA_CHECK(cudaStreamWaitEvent(stream, copy_done, 0));
        CUDA_CHECK(cudaEventDestroy(copy_done));
        copy_wait_event_time_us = (uint64_t) (ggml_time_us() - event_start_us);
    }

    void * pool_d = ggml_cuda_moe_cache_slot_ptr(cache, 0);
    uint64_t remap_time_us = 0;
    if (compact_source) {
        const int64_t remap_start_us = ggml_time_us();
        const size_t ids_total_elems = (size_t)ids_ne0 * ids_ne1 * ids_ne2;
        std::vector<int32_t> remapped_ids_host;
        remapped_ids_host.reserve(ids_total_elems);
        for (int64_t i2 = 0; i2 < ids_ne2; ++i2) {
            for (int64_t i1 = 0; i1 < ids_ne1; ++i1) {
                for (int64_t i0 = 0; i0 < ids_ne0; ++i0) {
                    const int32_t eid = *(const int32_t *)(ids_host_bytes->data()
                        + i2*ids_host_nb2 + i1*ids_host_nb1 + i0*ids_host_nb0);
                    remapped_ids_host.push_back(expert_to_slot[eid]);
                }
            }
        }

        ggml_cuda_pool_alloc<int32_t> scratch_ids(ctx.pool(), ids_total_elems);
        CUDA_CHECK(cudaMemcpyAsync(scratch_ids.get(), remapped_ids_host.data(),
                                   ids_total_elems * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        remap_time_us = (uint64_t) (ggml_time_us() - remap_start_us);

        const int n_slots = ggml_cuda_moe_cache_n_slots(cache);
        ggml_tensor src0_synth = *src0;
        src0_synth.ne[2] = n_slots;
        src0_synth.nb[3] = src0_synth.nb[2] * (size_t)n_slots;
        src0_synth.data = pool_d;

        ggml_tensor ids_synth = *ids;
        ids_synth.data = scratch_ids.get();
        ids_synth.nb[0] = sizeof(int32_t);
        ids_synth.nb[1] = (size_t)ids_ne0 * sizeof(int32_t);
        ids_synth.nb[2] = (size_t)ids_ne0 * ids_ne1 * sizeof(int32_t);
        ids_synth.nb[3] = ids_synth.nb[2];

        ggml_tensor * orig_src0 = dst->src[0];
        ggml_tensor * orig_ids = dst->src[2];
        dst->src[0] = &src0_synth;
        dst->src[2] = &ids_synth;
        const bool dispatched = ggml_cuda_mul_mat_id_impl(
            ctx, dst, use_mmq, nullptr,
            compact_mmvq ? GGML_CUDA_MMID_CONSUMER_MMVQ : GGML_CUDA_MMID_CONSUMER_UNSUPPORTED);
        dst->src[0] = orig_src0;
        dst->src[2] = orig_ids;
        GGML_ASSERT(dispatched);
    } else {
        ggml_tensor src0_synth = *src0;
        src0_synth.ne[2] = n_experts_total;
        src0_synth.nb[3] = src0_synth.nb[2] * (size_t)n_experts_total;
        src0_synth.data = pool_d;

        ggml_tensor * orig_src0 = dst->src[0];
        dst->src[0] = &src0_synth;
        const ggml_cuda_mul_mat_id_host_route host_route = {
            ids_host_bytes->data(), ids_host_nb0, ids_host_nb1, expert_to_slot.data(),
        };
        ggml_cuda_mul_mat_id_impl(ctx, dst, use_mmq, &host_route);
        dst->src[0] = orig_src0;
    }
    if (!ggml_cuda_moe_cache_mark_used(cache, stream)) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    ggml_cuda_moe_cache_release_slots(cache, pinned_slots.data(), (int) pinned_slots.size());

    ggml_cuda_moe_record_legacy_op(leased_owner,
        telemetry_is_decode, false, false, false, (uint64_t) unique_eids.size(), (uint64_t) ggml_nbytes(ids),
        ids_d2h_time_us, ids_d2h_sync_count, acquire_time_us, remap_time_us,
        1, copy_wait_event_time_us, (uint64_t) (ggml_time_us() - op_start_us), ids_cache_hit);
}

static void ggml_cuda_moe_shadow_probe_registered(
        ggml_cuda_moe_grouped_context & registry,
        const ggml_tensor * weight,
        const ggml_tensor * ids,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    ggml_cuda_moe_candidate_probe_input input = {};
    input.n_banks = fusion != nullptr && fusion->gate != nullptr ? 2 : 1;
    input.exact_auxiliaries = fusion != nullptr &&
        (fusion->x_scale != nullptr || fusion->gate_scale != nullptr || fusion->x_bias != nullptr || fusion->gate_bias != nullptr);
    input.banks[0] = {
        weight,
        ids,
        fusion != nullptr ? fusion->x_scale : nullptr,
        fusion != nullptr ? fusion->x_bias : nullptr,
        static_cast<uint32_t>(input.n_banks == 2 ? GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT : GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID),
    };
    if (input.n_banks == 2) {
        input.banks[1] = {
            fusion->gate,
            fusion->gate_ids != nullptr ? fusion->gate_ids : ids,
            fusion->gate_scale,
            fusion->gate_bias,
            GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT,
        };
    }
    (void) registry.probe(input, nullptr);
}

static inline void ggml_cuda_moe_shadow_probe(
        ggml_backend_cuda_context & ctx,
        const ggml_cuda_moe_graph_execution * execution,
        const ggml_tensor * mmid_node,
        const ggml_tensor * weight,
        const ggml_tensor * ids,
        const ggml_cuda_mm_fusion_args_host * fusion = nullptr,
        bool is_mmid = true) {
    auto * registry = ctx.moe_grouped_context;
    if (registry == nullptr || !is_mmid) {
        return;
    }
    if (execution != nullptr) {
        (void) execution->find(mmid_node, nullptr);
    }
    ggml_cuda_moe_shadow_probe_registered(*registry, weight, ids, fusion);
}

static bool ggml_cuda_moe_group_views(
        const ggml_cuda_moe_graph_group_dispatch & group,
        const ggml_cuda_moe_graph_binding & binding,
        const ggml_tensor * weight,
        ggml_tensor & bank_view,
        ggml_tensor & ids_view) {
    if (group.state != GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_ACTIVE || weight == nullptr || group.capabilities == nullptr ||
            binding.slot_index >= group.key.n_banks || binding.slot_index >= GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS ||
            group.capabilities[binding.slot_index].tensor != weight ||
            group.capabilities[binding.slot_index].role != binding.role || group.remapped_ids == nullptr) {
        return false;
    }
    if (group.bank_data[binding.slot_index] == nullptr) {
        return false;
    }
    bank_view = *weight;
    bank_view.data = const_cast<void *>(group.bank_data[binding.slot_index]);
    bank_view.ne[2] = group.n_slots;
    bank_view.nb[3] = bank_view.nb[2] * group.n_slots;
    ids_view = *group.key.ids.tensor;
    ids_view.data = const_cast<int32_t *>(group.remapped_ids);
    ids_view.nb[0] = sizeof(int32_t);
    ids_view.nb[1] = ids_view.ne[0] * sizeof(int32_t);
    ids_view.nb[2] = ids_view.nb[1] * ids_view.ne[1];
    ids_view.nb[3] = ids_view.nb[2] * ids_view.ne[2];
    return true;
}

static bool ggml_cuda_moe_grouped_mmvq(
        const ggml_cuda_moe_graph_group_dispatch * group,
        const ggml_cuda_moe_graph_binding & binding) {
    if (group == nullptr || group->capabilities == nullptr || binding.slot_index >= group->key.n_banks) {
        return false;
    }
    const auto & capability = group->capabilities[binding.slot_index];
    return capability.reason == GGML_CUDA_MMID_CAPABILITY_OK &&
        capability.equivalence_reason == GGML_CUDA_MMID_CAPABILITY_OK &&
        capability.consumer == GGML_CUDA_MMID_CONSUMER_MMVQ;
}

static bool ggml_cuda_moe_grouped_b1_scale_fusion(
        const ggml_cuda_moe_graph_group_dispatch * group,
        const ggml_cuda_moe_graph_binding & binding) {
    return group != nullptr && group->key.layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE &&
        group->key.n_banks == 3 && group->key.ids.ne[1] == 1 && group->key.ids.ne[2] == 1 && group->key.ids.ne[3] == 1 &&
        (binding.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT ||
            binding.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT ||
            binding.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT) &&
        ggml_cuda_moe_grouped_mmvq(group, binding) &&
        group->capabilities[binding.slot_index].source_type == GGML_TYPE_NVFP4;
}

static bool ggml_cuda_moe_group_scale_view(
        const ggml_cuda_moe_graph_group_dispatch & group,
        const ggml_cuda_moe_graph_binding & binding,
        const ggml_tensor * scale,
        ggml_tensor & scale_view) {
    if (group.state != GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_ACTIVE || !ggml_cuda_moe_grouped_b1_scale_fusion(&group, binding) ||
            group.n_auxiliary_shadows != 3 || group.n_slots == 0 || scale == nullptr) {
        return false;
    }
    const uint32_t scale_role = binding.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT ?
        GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_SCALE :
        binding.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT ?
            GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_SCALE : GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_SCALE;
    uint32_t auxiliary_index = 0;
    while (auxiliary_index < group.n_auxiliary_shadows &&
            (group.auxiliary_roles[auxiliary_index] != scale_role || group.auxiliary_tensors[auxiliary_index] != scale)) {
        ++auxiliary_index;
    }
    if (auxiliary_index == group.n_auxiliary_shadows || group.auxiliary_data[auxiliary_index] == nullptr ||
            scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_n_dims(scale) != 1 ||
            ggml_nelements(scale) != group.capabilities[binding.slot_index].n_experts) {
        return false;
    }
    scale_view = *scale;
    scale_view.data = const_cast<float *>(group.auxiliary_data[auxiliary_index]);
    scale_view.ne[0] = group.n_slots;
    scale_view.ne[1] = 1;
    scale_view.ne[2] = 1;
    scale_view.ne[3] = 1;
    scale_view.nb[0] = sizeof(float);
    scale_view.nb[1] = group.n_slots * sizeof(float);
    scale_view.nb[2] = scale_view.nb[1];
    scale_view.nb[3] = scale_view.nb[2];
    return true;
}

static uint32_t ggml_cuda_moe_bias_role(uint32_t weight_role) {
    switch (weight_role) {
        case GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT:
            return GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_BIAS;
        case GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT:
            return GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_BIAS;
        case GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT:
            return GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_BIAS;
        case GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT:
            return GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_BIAS;
        default:
            return GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID;
    }
}

static bool ggml_cuda_moe_group_bias_view(
        const ggml_cuda_moe_graph_group_dispatch & group,
        const ggml_cuda_moe_graph_binding & binding,
        const ggml_tensor * bias,
        ggml_tensor & bias_view) {
    if (group.state != GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_ACTIVE || group.n_slots == 0 || bias == nullptr ||
            binding.slot_index >= group.key.n_banks || !ggml_cuda_moe_grouped_mmvq(&group, binding)) {
        return false;
    }
    const uint32_t bias_role = ggml_cuda_moe_bias_role(binding.role);
    uint32_t auxiliary_index = 0;
    while (auxiliary_index < group.n_auxiliary_shadows &&
            (group.auxiliary_roles[auxiliary_index] != bias_role || group.auxiliary_tensors[auxiliary_index] != bias)) {
        ++auxiliary_index;
    }
    if (bias_role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID || auxiliary_index == group.n_auxiliary_shadows ||
            group.auxiliary_data[auxiliary_index] == nullptr || bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias) ||
            ggml_n_dims(bias) != 2 || bias->ne[0] <= 0 ||
            bias->ne[1] != group.capabilities[binding.slot_index].n_experts) {
        return false;
    }
    bias_view = *bias;
    bias_view.data = const_cast<float *>(group.auxiliary_data[auxiliary_index]);
    bias_view.ne[1] = group.n_slots;
    bias_view.ne[2] = 1;
    bias_view.ne[3] = 1;
    bias_view.nb[0] = sizeof(float);
    bias_view.nb[1] = bias_view.ne[0] * sizeof(float);
    bias_view.nb[2] = bias_view.nb[1] * group.n_slots;
    bias_view.nb[3] = bias_view.nb[2];
    return true;
}

// Public entry point for GGML_OP_MUL_MAT_ID. Routes cached-buffer tensors
// through the staging path; everything else goes straight to the regular
// implementation, preserving existing behavior bit-for-bit.
static bool ggml_cuda_mul_mat_id(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        ggml_cuda_moe_graph_execution * execution) {
    const ggml_tensor * src0 = dst->src[0];
    ggml_cuda_moe_graph_binding binding;
    auto * group = execution != nullptr ? execution->find_group(dst, &binding) : nullptr;
    if (execution != nullptr && execution->rejects_cached_mmid(dst)) {
        return false;
    }

    ggml_cuda_moe_shadow_probe(ctx, execution, dst, src0, dst->src[2]);

    if (group != nullptr && (group->state == GGML_CUDA_MOE_GRAPH_GROUP_PREFILL_ARMED ||
            group->state == GGML_CUDA_MOE_GRAPH_GROUP_PREFILL_ACTIVE)) {
        std::vector<char> ids_host_storage;
        const ggml_cuda_moe_ids_host ids_host = ggml_cuda_moe_read_ids(ctx, dst->src[2], src0->name, ids_host_storage);
        const int64_t n_experts_total = src0->ne[2];
        std::vector<uint8_t> seen((size_t)n_experts_total, 0);
        std::vector<int32_t> unique_experts;
        unique_experts.reserve(std::min<int64_t>(n_experts_total, 64));
        for (int64_t i2 = 0; i2 < dst->src[2]->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < dst->src[2]->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < dst->src[2]->ne[0]; ++i0) {
                    const int32_t expert = *(const int32_t *)(ids_host.bytes->data() +
                        i2*ids_host.nb2 + i1*ids_host.nb1 + i0*ids_host.nb0);
                    if (expert < 0 || expert >= n_experts_total) {
                        return false;
                    }
                    if (!seen[expert]) {
                        seen[expert] = 1;
                        unique_experts.push_back(expert);
                    }
                }
            }
        }
        if (unique_experts.empty()) {
            return false;
        }
        if (group->state == GGML_CUDA_MOE_GRAPH_GROUP_PREFILL_ARMED &&
                ctx.moe_grouped_context->prepare_prefill_group(
                    group, binding, dst, ctx.stream(), unique_experts.data(), unique_experts.size()) !=
                    GGML_CUDA_MOE_GROUPED_DECODE_READY) {
            return false;
        }
        if (group->state != GGML_CUDA_MOE_GRAPH_GROUP_PREFILL_ACTIVE ||
                binding.slot_index >= group->key.n_banks || group->bank_data[binding.slot_index] == nullptr ||
                group->prefill_slot_for_expert.size() != static_cast<size_t>(n_experts_total)) {
            return false;
        }
        const auto bounded = ctx.moe_grouped_context->execute_bounded_prefill_mmq(
            ctx, group, binding, dst, ctx.stream(), unique_experts.data(), unique_experts.size());
        if (bounded == GGML_CUDA_MOE_GROUPED_DECODE_ERROR) {
            return false;
        }
        if (bounded == GGML_CUDA_MOE_GROUPED_DECODE_FALLBACK) {
            ggml_cuda_mul_mat_id_staged(
                ctx, dst, *ids_host.bytes, ids_host.nb0, ids_host.nb1, ids_host.nb2,
                false, false, false, nullptr, ctx.moe_grouped_context, ids_host.cache_hit,
                group->bank_data[binding.slot_index], &group->prefill_slot_for_expert, false);
        }
        if (dst == group->last_reader && !ctx.moe_grouped_context->finish_prefill_group(
                group, binding, dst, ctx.stream())) {
            return false;
        }
        return true;
    }

    if (group != nullptr && (group->state == GGML_CUDA_MOE_GRAPH_GROUP_HOST_STAGED_ARMED ||
            group->state == GGML_CUDA_MOE_GRAPH_GROUP_HOST_STAGED_ACTIVE)) {
        if (!ggml_cuda_mul_mat_id_grouped_host_staged(ctx, dst, group, binding)) {
            return false;
        }
        if (dst == group->last_reader &&
                !ctx.moe_grouped_context->finish_host_staged_group(group, binding, dst, ctx.stream())) {
            return false;
        }
        return true;
    }
    if (group != nullptr && group->state == GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_ARMED) {
        const auto result = ctx.moe_grouped_context->prepare_graph_group(group, binding, dst, ctx.stream());
        if (result != GGML_CUDA_MOE_GROUPED_DECODE_READY) {
            return false;
        }
    }
    if (group != nullptr && group->state == GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_ACTIVE) {
        ggml_tensor bank_view;
        ggml_tensor ids_view;
        if (group->stream != ctx.stream() || dst->src[2] != group->key.ids.tensor ||
                !ggml_cuda_moe_group_views(*group, binding, src0, bank_view, ids_view)) {
            return false;
        }
        if (binding.slot_index >= group->key.n_banks || group->capabilities == nullptr) {
            return false;
        }
        const auto & capability = group->capabilities[binding.slot_index];
        const bool use_mmq = ggml_cuda_moe_use_mmq(src0, dst->src[1]->ne[2]);
        if (capability.use_mmq != static_cast<uint32_t>(use_mmq) ||
                capability.consumer == GGML_CUDA_MMID_CONSUMER_UNSUPPORTED ||
                capability.consumer == GGML_CUDA_MMID_CONSUMER_GENERIC) {
            return false;
        }
        ggml_tensor * original_weight = dst->src[0];
        ggml_tensor * original_ids = dst->src[2];
        dst->src[0] = &bank_view;
        dst->src[2] = &ids_view;
        const ggml_cuda_mmid_direct_source_view direct_source_view = {
            bank_view.type,
            capability.n_slots,
            bank_view.ne[2],
            group->n_slots,
            bank_view.nb[2],
        };
        const bool dispatched = ggml_cuda_mul_mat_id_impl(
            ctx, dst, use_mmq, nullptr, static_cast<ggml_cuda_mmid_consumer>(capability.consumer),
            &direct_source_view);
        dst->src[0] = original_weight;
        dst->src[2] = original_ids;
        if (!dispatched) {
            return false;
        }
        if (dst == group->last_reader &&
                !ctx.moe_grouped_context->finish_graph_group(group, binding, dst, ctx.stream())) {
            return false;
        }
        return true;
    }
    if (group != nullptr && group->state != GGML_CUDA_MOE_GRAPH_GROUP_DETACHED_STAGED) {
        return false;
    }

    const bool src0_cached = src0 && src0->buffer && ggml_backend_buft_is_cuda_moe_cached(src0->buffer->buft);
    if (src0_cached) {
        static std::once_flag once;
        std::call_once(once, [&]() {
            const char * buft_name = "(null)";
            if (src0 && src0->buffer && src0->buffer->buft && src0->buffer->buft->iface.get_name) {
                buft_name = src0->buffer->buft->iface.get_name(src0->buffer->buft);
            }
            GGML_LOG_INFO("moe-cache: first mul_mat_id  src0=%s  buft=%s  is_cached=%d\n",
                          src0 ? src0->name : "(null)",
                          buft_name,
                          1);
        });
    }

    if (src0_cached) {
        const auto outcome = execution != nullptr ? execution->outcome() : GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR;
        const bool detached_staging = execution != nullptr && execution->requires_dispatch() &&
            outcome != GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR &&
            (execution->dispatch_mode() == GGML_CUDA_MOE_GRAPH_DISPATCH_STAGED ||
                outcome == GGML_CUDA_MOE_GRAPH_OUTCOME_PREFILL_STAGED ||
                outcome == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_STAGED);
        if (detached_staging) {
            auto * owner = ctx.moe_grouped_context;
            if (owner != nullptr && !owner->track_staged_stream(execution, ctx.stream())) {
                return false;
            }
            std::vector<char> ids_host_storage;
            const ggml_cuda_moe_ids_host ids_host = ggml_cuda_moe_read_ids(ctx, dst->src[2], src0->name, ids_host_storage);
            if (ids_host.bytes == nullptr) {
                return false;
            }
            const bool single_row = dst->src[2]->ne[1] * dst->src[2]->ne[2] == 1;
            ggml_cuda_mul_mat_id_staged(
                ctx, dst, *ids_host.bytes, ids_host.nb0, ids_host.nb1, ids_host.nb2,
                single_row, outcome == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_STAGED || single_row,
                false, nullptr, owner, ids_host.cache_hit);
            return true;
        }
        const auto * authority = group != nullptr && group->authority ? &group->authority :
            (execution != nullptr ? execution->find_authority(dst) : nullptr);
        ggml_cuda_mul_mat_id_cached(ctx, dst, authority);
        return true;
    }
    ggml_cuda_mul_mat_id_impl(ctx, dst, true);
    return true;
}

static bool ggml_cuda_compute_forward(
        ggml_backend_cuda_context & ctx,
        struct ggml_tensor * dst,
        ggml_cuda_moe_graph_execution * execution, const void * prepared_src1 = nullptr) {
    if (execution != nullptr && execution->requires_dispatch() && ctx.moe_grouped_context != nullptr &&
            execution->dispatch_mode() == GGML_CUDA_MOE_GRAPH_DISPATCH_STAGED &&
            !ctx.moe_grouped_context->track_staged_stream(execution, ctx.stream())) {
        return false;
    }
    if (execution != nullptr && execution->requires_dispatch() && ctx.moe_grouped_context != nullptr &&
            (execution->outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED ||
                execution->outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_PREFILL_STAGED ||
                execution->outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_STAGED) &&
            (dst->op == GGML_OP_REPEAT || dst->op == GGML_OP_ADD_ID)) {
        const float * source = nullptr;
        if (!ctx.moe_grouped_context->original_auxiliary_source(*execution, dst, ctx.stream(), &source)) {
            return false;
        }
        const int index = dst->op == GGML_OP_REPEAT ? 0 : 1;
        ggml_tensor source_view = *dst->src[index];
        ggml_tensor dst_view = *dst;
        source_view.data = const_cast<float *>(source);
        dst_view.src[index] = &source_view;
        if (dst->op == GGML_OP_REPEAT) {
            ggml_cuda_op_repeat(ctx, &dst_view);
        } else {
            ggml_cuda_op_add_id(ctx, &dst_view);
        }
        return true;
    }
    switch (dst->op) {
        case GGML_OP_CUSTOM:
            return ggml_cuda_staged_input_compute(ctx, dst);
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
        case GGML_OP_ADD_ID: {
            const float * prefill_source = nullptr;
            if (execution != nullptr && ctx.moe_grouped_context != nullptr &&
                    ctx.moe_grouped_context->prefill_add_id_source(
                        *execution, dst, ctx.stream(), &prefill_source)) {
                ggml_tensor source_view = *dst->src[1];
                ggml_tensor dst_view = *dst;
                source_view.data = const_cast<float *>(prefill_source);
                dst_view.src[1] = &source_view;
                ggml_cuda_op_add_id(ctx, &dst_view);
                if (!ctx.moe_grouped_context->finish_prefill_add_id(
                        *execution, dst, ctx.stream(), prefill_source)) {
                    return false;
                }
            } else {
                ggml_cuda_op_add_id(ctx, dst);
            }
            break;
        }
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
            if (!ggml_cuda_mul_mat_id(ctx, dst, execution)) {
                return false;
            }
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

bool ggml_cuda_moe_router_compute(ggml_backend_cuda_context & context, ggml_tensor * node) {
    return ggml_cuda_compute_forward(context, node, nullptr);
}

static bool ggml_cuda_is_view_or_noop(const ggml_tensor * t) {
    return ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
}

bool ggml_cuda_moe_fidelity_emit_node(ggml_backend_cuda_context & context, const ggml_cgraph * graph,
        uint64_t graph_uid, uint32_t index, ggml_tensor * node) {
    if (!graph || graph->uid != graph_uid || index >= uint32_t(graph->n_nodes) || !node) { return false; }
    const auto * original = graph->nodes[index];
    if (node == original || node->op != original->op || node->type != original->type || node->flags != original->flags ||
            memcmp(node->ne, original->ne, sizeof(node->ne)) || memcmp(node->nb, original->nb, sizeof(node->nb)) ||
            memcmp(node->op_params, original->op_params, sizeof(node->op_params))) { return false; }
    if (ggml_cuda_is_view_or_noop(node) || !(node->flags & GGML_TENSOR_FLAG_COMPUTE)) { return true; }
    if (!node->view_src && node->data == original->data) { return false; }
    return ggml_cuda_compute_forward(context, node, nullptr);
}

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
            int can_access_peer = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, src_physical, dst_physical));
            if (!can_access_peer) {
                return false;
            }
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

static uint64_t ggml_cuda_graph_get_key(ggml_cgraph * cgraph) {
    if (cgraph->n_nodes <= 0) {
        return 0;
    }

    uint64_t key = (uint64_t) (uintptr_t) cgraph->nodes[0];

    auto mix = [&key](uint64_t v) {
        key = (key ^ v) * 0x100000001b3ull;
    };

    mix(cgraph->n_nodes);

    for (int d = 0; d < GGML_MAX_DIMS; d++) {
        mix(cgraph->nodes[0]->ne[d]);
        mix(cgraph->nodes[cgraph->n_nodes - 1]->ne[d]);
    }

    return key;
}

static void ggml_cuda_moe_clear_graph_coverage(ggml_cuda_graph * graph) {
    graph->moe_graph_plan.reset();
    graph->moe_coverage_nodes = nullptr;
    graph->moe_coverage_epoch = 0;
    graph->moe_coverage_mmid_fingerprint = 0;
    graph->moe_coverage_n_nodes = 0;
    graph->moe_coverage_mmid_count = 0;
}

void ggml_backend_cuda_context::certify_moe_graph(ggml_cgraph * cgraph) {
    ggml_cuda_moe_graph_span span;
    if (moe_grouped_context == nullptr || cgraph == nullptr ||
            !ggml_cuda_moe_graph_span_bounds(cgraph->nodes, cgraph->n_nodes, &span)) {
        return;
    }

    const uint64_t key = ggml_cuda_graph_get_key(cgraph);
    uint32_t coverage_mmid_count = 0;
    uint64_t coverage_mmid_fingerprint = 0;
    const uint64_t coverage_epoch = moe_grouped_context->certify_graph_coverage(
        cgraph, &coverage_mmid_count, &coverage_mmid_fingerprint);
    ggml_cuda_graph * graph = cuda_graph(key);
    ggml_cuda_moe_clear_graph_coverage(graph);
    if (coverage_epoch == 0) {
        return;
    }

    graph->moe_coverage_nodes = cgraph->nodes;
    graph->moe_coverage_epoch = coverage_epoch;
    graph->moe_coverage_mmid_fingerprint = coverage_mmid_fingerprint;
    graph->moe_coverage_n_nodes = cgraph->n_nodes;
    graph->moe_coverage_mmid_count = coverage_mmid_count;
}

bool ggml_backend_cuda_context::recover_moe_graph(ggml_cgraph * cgraph, ggml_cuda_graph * graph) {
    if (graph == nullptr) {
        return false;
    }
    ggml_cuda_moe_clear_graph_coverage(graph);
    if (moe_grouped_context == nullptr || cgraph == nullptr) {
        return false;
    }
    uint64_t coverage_epoch = 0;
    uint32_t coverage_mmid_count = 0;
    uint64_t coverage_mmid_fingerprint = 0;
    if (!moe_grouped_context->recover_graph_coverage(
            cgraph, &coverage_epoch, &coverage_mmid_count, &coverage_mmid_fingerprint)) {
        return false;
    }
    graph->moe_coverage_nodes = cgraph->nodes;
    graph->moe_coverage_epoch = coverage_epoch;
    graph->moe_coverage_mmid_fingerprint = coverage_mmid_fingerprint;
    graph->moe_coverage_n_nodes = cgraph->n_nodes;
    graph->moe_coverage_mmid_count = coverage_mmid_count;
    return true;
}

#ifdef USE_CUDA_GRAPH
static bool ggml_cuda_graph_check_compability(ggml_cgraph * cgraph, bool * has_cached_mmid) {

    bool use_cuda_graph = true;
    *has_cached_mmid = false;
    // Loop over nodes in GGML graph to obtain info needed for CUDA graph

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (ggml_cuda_is_view_or_noop(node)) {
            continue;
        }

        // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
        if (node->op == GGML_OP_MUL_MAT_ID) {
            if (node->src[0] && node->src[0]->buffer &&
                    ggml_backend_buft_is_cuda_moe_cached(node->src[0]->buffer->buft)) {
                *has_cached_mmid = true;
            }

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

struct ggml_cuda_graph_property_probe {
    bool changed;
    bool uid_match;
};

static ggml_cuda_graph::node_properties ggml_cuda_graph_node_properties(const ggml_tensor * node) {
    ggml_cuda_graph::node_properties prop = {};
    memcpy(&prop.node, node, sizeof(ggml_tensor));

    for (int j = 0; j < GGML_MAX_SRC; ++j) {
        if (node->src[j]) {
            prop.node_src_data_ptrs[j] = node->src[j]->data;
            prop.node_src_type[j] = node->src[j]->type;
            memcpy(prop.node_src_ne[j], node->src[j]->ne, sizeof(prop.node_src_ne[j]));
            memcpy(prop.node_src_nb[j], node->src[j]->nb, sizeof(prop.node_src_nb[j]));
        }
    }
    return prop;
}

static ggml_cuda_graph_property_probe ggml_cuda_graph_probe_properties(
        ggml_backend_cuda_context * cuda_ctx,
        ggml_cgraph * cgraph) {
    const uint64_t graph_key = ggml_cuda_graph_get_key(cgraph);
    const ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
    const uint64_t execution_semantic_key = ggml_cuda_moe_execution_semantic_key(cgraph);

    if (cgraph->uid != 0 &&
        cgraph->uid == graph->uid && execution_semantic_key == graph->execution_semantic_key) {
        GGML_LOG_DEBUG("CUDA Graph id %zu reused\n", cgraph->uid);
        GGML_ASSERT((int)graph->node_props.size() == cgraph->n_nodes);
        return {false, true};
    }

    if (execution_semantic_key != graph->execution_semantic_key ||
            (int) graph->node_props.size() != cgraph->n_nodes) {
        return {true, false};
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        const auto prop = ggml_cuda_graph_node_properties(cgraph->nodes[i]);
        if (memcmp(&graph->node_props[i], &prop, sizeof(prop)) != 0) {
            return {true, false};
        }
    }

    return {false, false};
}

static void ggml_cuda_graph_commit_properties(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph) {
    const uint64_t graph_key = ggml_cuda_graph_get_key(cgraph);
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
    graph->uid = cgraph->uid;
    graph->execution_semantic_key = ggml_cuda_moe_execution_semantic_key(cgraph);
    graph->node_props.resize(cgraph->n_nodes);
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        graph->node_props[i] = ggml_cuda_graph_node_properties(cgraph->nodes[i]);
    }
}

static bool ggml_cuda_graph_update_executable(ggml_backend_cuda_context * cuda_ctx, uint64_t graph_key) {
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
        return false;
    } else {
        GGML_ASSERT(stat == cudaSuccess);
        return true;
    }
}
#endif // USE_CUDA_GRAPH

bool ggml_cuda_graph_capture_state_query_for_test(
        ggml_backend_t backend,
        const ggml_cgraph * cgraph,
        ggml_cuda_graph_capture_state_for_test * state) {
    if (state == nullptr) {
        return false;
    }
    *state = {};
    if (!ggml_backend_is_cuda(backend) || backend->context == nullptr ||
            cgraph == nullptr || cgraph->n_nodes == 0) {
        return false;
    }
    auto * context = static_cast<ggml_backend_cuda_context *>(backend->context);
    const auto graph = context->cuda_graphs.find(ggml_cuda_graph_get_key(const_cast<ggml_cgraph *>(cgraph)));
    if (graph == context->cuda_graphs.end() || graph->second == nullptr) {
        return false;
    }
#ifdef USE_CUDA_GRAPH
    ggml_cuda_set_device(context->device);
    bool has_cached_mmid = false;
    state->capture_available = graph->second->is_enabled() &&
        ggml_cuda_info().devices[context->device].cc >= GGML_CUDA_CC_VOLTA &&
        ggml_cuda_graph_check_compability(const_cast<ggml_cgraph *>(cgraph), &has_cached_mmid);
    state->graph = reinterpret_cast<uintptr_t>(graph->second->graph);
    state->instance = reinterpret_cast<uintptr_t>(graph->second->instance);
    state->execution_semantic_key = graph->second->execution_semantic_key;
    state->moe_resource_fingerprint = graph->second->moe_resource_fingerprint;
    state->warmup_complete = graph->second->warmup_complete;
#endif
    return true;
}

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
    if (!ggml_gated_delta_net_validate(gdn) ||
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
    const int32_t       trailing  = ggml_get_op_params_i32(gdn, 1);
    const int32_t       selected  = ggml_get_op_params_i32(gdn, 2);
    const bool          reserve   = ggml_get_op_params_i32(gdn, 3) != 0;
    const int64_t       n_written = selected >= 0 ? 1 : reserve ? trailing : std::min<int64_t>(n_tokens, K);

    // snapshot tail starts right after the attention scores
    const size_t tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);
    const size_t state_size = ggml_row_size(GGML_TYPE_F32, D * n_seqs);

    const auto find_cpy = [&](int first, const ggml_tensor ** cpy, int * cpy_idx) {
        for (int j = first; j < cgraph->n_nodes; ++j) {
            const ggml_tensor * node = cgraph->nodes[j];
            if (ggml_cuda_is_view_or_noop(node)) {
                continue;
            }
            if (node->op != GGML_OP_CPY || (node->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                return false;
            }
            *cpy = node;
            *cpy_idx = j;
            return true;
        }
        return false;
    };
    const auto match_cpy = [&](const ggml_tensor * cpy, int64_t first_slot, int64_t count) {
        const ggml_tensor * src = cpy->src[0];
        const ggml_tensor * dst = cpy->src[1];
        const std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, count, 1 };
        return src != nullptr && dst != nullptr && src->op == GGML_OP_VIEW && src->view_src == gdn &&
            src->view_offs == tail_off + first_slot * state_size &&
            std::equal(expected_ne.begin(), expected_ne.end(), src->ne) && ggml_is_contiguous(src) &&
            dst->op == GGML_OP_VIEW && dst->type == GGML_TYPE_F32 && dst->data != nullptr &&
            std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) &&
            dst->nb[0] == ggml_type_size(GGML_TYPE_F32) && dst->nb[1] == (size_t) ggml_row_size(GGML_TYPE_F32, D) &&
            dst->nb[2] >= state_size && dst->nb[2] % sizeof(float) == 0;
    };

    const ggml_tensor * cpy = nullptr;
    int cpy_idx = 0;
    if (!find_cpy(node_idx + 1, &cpy, &cpy_idx) || !match_cpy(cpy, 0, n_written)) {
        return 0;
    }

    const ggml_tensor * dst = cpy->src[1];
    fused_state_cpy.data        = (float *) dst->data;
    fused_state_cpy.slot_stride = K > 1 ? (int64_t) (dst->nb[2] / sizeof(float)) : 0;

    if (reserve) {
        const ggml_tensor * reserve_cpy = nullptr;
        int reserve_cpy_idx = 0;
        if (!find_cpy(cpy_idx + 1, &reserve_cpy, &reserve_cpy_idx) || !match_cpy(reserve_cpy, K - 1, 1)) {
            return 0;
        }
        const ggml_tensor * reserve_dst = reserve_cpy->src[1];
        if ((float *) reserve_dst->data != fused_state_cpy.data + (K - 1) * fused_state_cpy.slot_stride ||
            reserve_dst->nb[2] != dst->nb[2]) {
            return 0;
        }
        cpy_idx = reserve_cpy_idx;
    }

    return cpy_idx - node_idx;
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
                                                 const bool          is_topk_moe = false,
                                                 const bool          check_leaf_inputs = false) {
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

                if (!src || (!check_leaf_inputs && src->op == GGML_OP_NONE) || src == logits_may_alias) {
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

static int ggml_cuda_get_graph_stream(const ggml_cuda_stream_context & stream_ctx, const ggml_tensor * node) {
    int stream = 0;

    for (const auto & entry : stream_ctx.concurrent_events) {
        const auto it = entry.second.stream_mapping.find(node);
        if (it == entry.second.stream_mapping.end()) {
            continue;
        }
        if (stream != 0 && stream != it->second) {
            return -1;
        }
        stream = it->second;
    }

    return stream;
}

static cudaStream_t ggml_cuda_moe_graph_stream(void * data, const ggml_tensor * node) {
    auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(data);
    const int stream = ggml_cuda_get_graph_stream(cuda_ctx->stream_context(), node);
    return stream >= 0 ? cuda_ctx->stream(cuda_ctx->device, stream) : nullptr;
}

static bool ggml_cuda_can_fuse_f32_q8_0_cpy_pair(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, int node_idx) {
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, { GGML_OP_CPY, GGML_OP_CPY }, { node_idx, node_idx + 1 })) {
        return false;
    }

    const ggml_tensor * dst0 = cgraph->nodes[node_idx];
    const ggml_tensor * dst1 = cgraph->nodes[node_idx + 1];
    const ggml_tensor * src0 = dst0->src[0];
    const ggml_tensor * src1 = dst1->src[0];

    if (!src0 || !src1 || dst0->src[1] != dst0 || dst1->src[1] != dst1 || src0 == src1) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 ||
        dst0->type != GGML_TYPE_Q8_0 || dst1->type != GGML_TYPE_Q8_0) {
        return false;
    }
    if (!ggml_are_same_shape(src0, src1) || !ggml_are_same_shape(src0, dst0) || !ggml_are_same_shape(src1, dst1)) {
        return false;
    }
    if (dst0->ne[0] <= 0 || dst0->ne[0] % QK8_0 != 0 || dst0->ne[1] <= 0 || dst0->ne[2] != 1 || dst0->ne[3] != 1) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst0) || !ggml_is_contiguous(dst1)) {
        return false;
    }
    if ((uintptr_t) src0->data % alignof(float) != 0 || (uintptr_t) src1->data % alignof(float) != 0 ||
        (uintptr_t) dst0->data % alignof(block_q8_0) != 0 || (uintptr_t) dst1->data % alignof(block_q8_0) != 0) {
        return false;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_cuda_buffer_type(cuda_ctx->device);
    const ggml_tensor * tensors[4] = { src0, dst0, src1, dst1 };
    for (const ggml_tensor * tensor : tensors) {
        if (!tensor->data || !tensor->buffer || tensor->buffer->buft != buft) {
            return false;
        }
    }

    auto overlaps = [](const ggml_tensor * a, const ggml_tensor * b) {
        const uintptr_t a_begin = (uintptr_t) a->data;
        const uintptr_t b_begin = (uintptr_t) b->data;
        const size_t a_size = ggml_nbytes(a);
        const size_t b_size = ggml_nbytes(b);
        if (a_size > UINTPTR_MAX - a_begin || b_size > UINTPTR_MAX - b_begin) {
            return true;
        }
        const uintptr_t a_end = a_begin + a_size;
        const uintptr_t b_end = b_begin + b_size;
        return a_begin < b_end && b_begin < a_end;
    };
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            if (overlaps(tensors[i], tensors[j])) {
                return false;
            }
        }
    }

    const ggml_cuda_stream_context & stream_ctx = cuda_ctx->stream_context();
    const int stream0 = ggml_cuda_get_graph_stream(stream_ctx, dst0);
    const int stream1 = ggml_cuda_get_graph_stream(stream_ctx, dst1);
    return stream0 >= 0 && stream0 == stream1 && stream0 == cuda_ctx->curr_stream_no;
}

// The long form spans 2*k + 1 nodes. ggml_can_fuse_subgraph() accepts at most
// 31 nodes, so k <= 15; larger values use the per-operation path.
static constexpr int MOE_WEIGHTED_REDUCTION_MAX_EXPERTS = 15;

void ggml_cuda_moe_weighted_reduction_alloc_deps(const ggml_cuda_moe_weighted_reduction_match & match,
        const ggml_backend_graph_optimize_params * params) {
    params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(match.experts), match.dst);
    params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(match.weights), match.dst);
    if (match.expert_scale) {
        params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(match.expert_scale), match.dst);
    }
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
            !ggml_cuda_should_use_mmq(weight->type, cc, input->ne[1], /*n_experts =*/ 0, ggml_cuda_info().devices[device].smpbo) || ggml_cuda_op_mul_mat_use_fwht(node)) {
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

static bool ggml_cuda_can_share_mmq_id_input(const ggml_tensor * node, int device) {
    if (node->op != GGML_OP_MUL_MAT_ID || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || node->view_src ||
            node->type != GGML_TYPE_F32 || !ggml_is_contiguous(node)) {
        return false;
    }
    const ggml_tensor * weight = node->src[0];
    const ggml_tensor * input = node->src[1];
    const ggml_tensor * ids = node->src[2];
    const int cc = ggml_cuda_info().devices[device].cc;
    if (input->type != GGML_TYPE_F32 || input->nb[0] != sizeof(float) || input->ne[3] != 1 ||
            weight->ne[3] != 1 || !ggml_is_contiguous_rows(weight) || ids->type != GGML_TYPE_I32 ||
            ids->nb[0] != sizeof(int32_t) || input->nb[2] % input->nb[1] != 0 ||
            (node->ne[2] <= MMVQ_MAX_BATCH_SIZE && ggml_is_quantized(weight->type) &&
                node->ne[2] <= get_mmvq_mmid_max_batch(weight->type, cc)) ||
            !ggml_cuda_should_use_mmq(weight->type, cc, input->ne[2], weight->ne[2]) ||
            ggml_cuda_op_mul_mat_use_fwht(node)) {
        return false;
    }
    for (const ggml_tensor * tensor : { weight, input, ids, node }) {
        if (!tensor->buffer || !tensor->data || tensor->buffer->buft != ggml_backend_cuda_buffer_type(device)) {
            return false;
        }
    }
    return !(weight->view_src && ggml_backend_buffer_get_usage(weight->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
        ggml_nbytes(weight) != ggml_backend_buffer_get_alloc_size(weight->buffer, weight));
}

static bool ggml_cuda_mmq_id_input_overwritten(const ggml_tensor * node, const ggml_tensor * input) {
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

bool ggml_cuda_match_moe_weighted_reduction(
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




struct ggml_cuda_rms_norm_gated_match {
    ggml_tensor * norm;
    ggml_tensor * mul;
    ggml_tensor * gate;
    ggml_tensor * dst;
    int node_count;
    ggml_unary_op gate_op;
};

static bool ggml_cuda_match_rms_norm_gated(const ggml_cgraph * cgraph, int i, ggml_cuda_rms_norm_gated_match & match) {
    if (!ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_UNARY, GGML_OP_MUL }, { i + 3 })) {
        return false;
    }
    ggml_tensor * norm  = cgraph->nodes[i];
    ggml_tensor * mul   = cgraph->nodes[i + 1];
    ggml_tensor * unary = cgraph->nodes[i + 2];
    ggml_tensor * dst   = cgraph->nodes[i + 3];
    const ggml_unary_op gate_op = ggml_get_unary_op(unary);
    if ((gate_op != GGML_UNARY_OP_SILU && gate_op != GGML_UNARY_OP_SIGMOID) || norm->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32 || unary->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            norm->src[0]->type != GGML_TYPE_F32 || unary->src[0]->type != GGML_TYPE_F32 ||
            norm->view_src || mul->view_src || !ggml_is_contiguous_rows(norm->src[0]) ||
            !ggml_is_contiguous_rows(unary->src[0]) || !ggml_is_contiguous(dst) ||
            !ggml_are_same_shape(norm, mul) || !ggml_are_same_shape(norm, unary) || !ggml_are_same_shape(norm, dst) ||
            !((dst->src[0] == mul && dst->src[1] == unary) || (dst->src[0] == unary && dst->src[1] == mul))) {
        return false;
    }
    const ggml_tensor * weight = mul->src[0] == norm ? mul->src[1] : mul->src[1] == norm ? mul->src[0] : nullptr;
    ggml_tensor * gate = unary->src[0];
    if (!weight || weight == norm || weight->type != GGML_TYPE_F32 || !ggml_is_contiguous_rows(weight) ||
            gate == norm || gate == mul || gate->view_src == norm || gate->view_src == mul) {
        return false;
    }
    match = { norm, mul, gate, dst, 4, gate_op };
    return true;
}


static bool ggml_cuda_match_ssm_conv_qk(const ggml_cgraph * cgraph, int i, bool check_memory = false) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    return false;
#endif
    const ggml_op ops[] = { GGML_OP_SSM_CONV, GGML_OP_UNARY, GGML_OP_VIEW, GGML_OP_RMS_NORM,
                           GGML_OP_SCALE, GGML_OP_VIEW, GGML_OP_RMS_NORM, GGML_OP_SCALE };
    const int outputs[] = { i + 1, i + 4, i + 7 };
    if (i + 8 > cgraph->n_nodes || !ggml_can_fuse_subgraph(cgraph, i, 8, ops, outputs, 3)) {
        return false;
    }
    const ggml_tensor * conv = cgraph->nodes[i];
    const ggml_tensor * silu = cgraph->nodes[i + 1];
    const ggml_tensor * x = conv->src[0];
    const ggml_tensor * w = conv->src[1];
    switch (w->ne[0]) {
        case 3: case 4: case 5: case 9: case 15: break;
        default: return false;
    }
    const ggml_tensor * q = cgraph->nodes[i + 2];
    const int64_t d = q->ne[0];
    const int64_t h = q->ne[1];
    const int64_t t = silu->ne[1];
    const int64_t s = silu->ne[2];
    const int64_t c = silu->ne[0];
    if (conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32 ||
            x->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(conv) || !ggml_is_contiguous(silu) || !ggml_is_contiguous(w) ||
            x->nb[0] != sizeof(float) || x->nb[1] != x->ne[0]*sizeof(float) ||
            silu->src[0] != conv || ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU ||
            conv->view_src || silu->view_src || (silu->flags & GGML_TENSOR_FLAG_OUTPUT) ||
            !ggml_are_same_shape(conv, silu) || d <= 0 || d > 256 || c > INT_MAX || h <= 0 || t <= 0 || t > 65535 || s <= 0 || s > 65535 ||
            silu->ne[3] != 1 || c % 128 != 0 || c / 128 > 65535 || h > c / (2*d) || 2*d*h >= c ||
            x->ne[1] != c || x->ne[2] != s || x->ne[3] != 1 || x->ne[0] != t + w->ne[0] - 1 ||
            w->ne[1] != c || w->ne[2] != 1 || w->ne[3] != 1) {
        return false;
    }
    if (t > 32 && (t < 2048 || d < 128 || d > 192)) {
        return false;
    }
    const size_t sequence_bytes = x->nb[1]*c;
    if (x->nb[2] > INT_MAX || sequence_bytes > INT_MAX || x->nb[2] < sequence_bytes ||
            x->nb[2] % sizeof(float) != 0 || (size_t) (s - 1) > (INT_MAX - sequence_bytes)/x->nb[2]) {
        return false;
    }
    for (int j : { 2, 5 }) {
        const ggml_tensor * view = cgraph->nodes[i + j];
        const ggml_tensor * norm = cgraph->nodes[i + j + 1];
        const ggml_tensor * scale = cgraph->nodes[i + j + 2];
        float eps, params[2];
        memcpy(&eps, norm->op_params, sizeof(eps));
        memcpy(params, scale->op_params, sizeof(params));
        if (view->type != GGML_TYPE_F32 || norm->type != GGML_TYPE_F32 || scale->type != GGML_TYPE_F32 ||
                view->src[0] != silu || view->view_src != silu || norm->src[0] != view || scale->src[0] != norm ||
                !ggml_are_same_shape(view, q) || !ggml_are_same_shape(view, norm) || !ggml_are_same_shape(view, scale) ||
                view->ne[2] != t || view->ne[3] != s || view->nb[0] != sizeof(float) ||
                view->nb[1] != d*sizeof(float) || view->nb[2] != silu->nb[1] || view->nb[3] != silu->nb[2] ||
                view->view_offs != (j == 2 ? 0 : d*h*sizeof(float)) || norm->view_src || scale->view_src ||
                !ggml_is_contiguous(norm) || !ggml_is_contiguous(scale) || !std::isfinite(eps) || eps < 0.0f ||
                params[0] != 1.0f/sqrtf(float(d)) || params[1] != 0.0f) {
            return false;
        }
    }
    // Keep the value slice in the original SiLU tensor.
    int value_views = 0;
    for (int j = i + 8; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * view = cgraph->nodes[j];
        for (int k = 0; k < GGML_MAX_SRC; ++k) {
            if (view->src[k] != silu) {
                continue;
            }
            if (k != 0 || view->op != GGML_OP_VIEW || view->view_src != silu || view->type != GGML_TYPE_F32 ||
                    view->ne[0] <= 0 || view->ne[1] <= 0 || view->ne[0] > c - 2*d*h ||
                    view->ne[1] != (c - 2*d*h) / view->ne[0] || (c - 2*d*h) % view->ne[0] != 0 ||
                    view->ne[2] != t || view->ne[3] != s || view->nb[0] != sizeof(float) ||
                    view->nb[1] != view->ne[0]*sizeof(float) || view->nb[2] != silu->nb[1] ||
                    view->nb[3] != silu->nb[2] || view->view_offs != 2*d*h*sizeof(float)) {
                return false;
            }
            ++value_views;
        }
    }
    if (value_views != 1) {
        return false;
    }
    if (!check_memory) {
        return true;
    }
    if (!ggml_cuda_check_fusion_memory_ranges(cgraph, i, 8, outputs, 3)) {
        return false;
    }
    auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        const uintptr_t a0 = (uintptr_t) a->data;
        const uintptr_t b0 = (uintptr_t) b->data;
        return a0 <= b0 ? b0 - a0 < ggml_nbytes(a) : a0 - b0 < ggml_nbytes(b);
    };
    for (int j = 0; j < 3; ++j) {
        const ggml_tensor * out = cgraph->nodes[outputs[j]];
        if (!out->data || !out->buffer || overlap(out, x) || overlap(out, w)) {
            return false;
        }
        for (int k = 0; k < j; ++k) {
            if (overlap(out, cgraph->nodes[outputs[k]])) {
                return false;
            }
        }
        for (int k = 0; k < cgraph->n_leafs; ++k) {
            const ggml_tensor * leaf = cgraph->leafs[k];
            if (leaf->data && overlap(out, leaf)) {
                return false;
            }
        }
    }
    return true;
}

static bool ggml_cuda_match_add_softplus_mul(const ggml_cgraph * graph, int i) {
    const ggml_op ops[] = { GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_MUL };
    const int output = i + 2;
    if (i > graph->n_nodes - 3 || !ggml_can_fuse_subgraph(graph, i, 3, ops, &output, 1)) {
        return false;
    }
    const ggml_tensor * add = graph->nodes[i];
    const ggml_tensor * unary = graph->nodes[i + 1];
    const ggml_tensor * mul = graph->nodes[i + 2];
    const ggml_tensor * scale = mul->src[0] == unary ? mul->src[1] : mul->src[0];
    if (!add->src[0] || !add->src[1] || unary->src[0] != add || ggml_get_unary_op(unary) != GGML_UNARY_OP_SOFTPLUS ||
            (mul->src[0] != unary && mul->src[1] != unary) || !scale || scale == add || scale == unary) {
        return false;
    }
    const ggml_tensor * tensors[] = { add, unary, mul, add->src[0], add->src[1], scale };
    for (const ggml_tensor * tensor : tensors) {
        uint64_t elements = 1;
        size_t span = sizeof(float);
        if (tensor->type != GGML_TYPE_F32 || tensor->nb[0] != sizeof(float)) {
            return false;
        }
        for (int j = 0; j < GGML_MAX_DIMS; ++j) {
            if (tensor->ne[j] <= 0 || uint64_t(tensor->ne[j]) > uint64_t(INT_MAX) / elements ||
                    tensor->nb[j] % sizeof(float) ||
                    (tensor->ne[j] > 1 && tensor->nb[j] > (SIZE_MAX - span) / size_t(tensor->ne[j] - 1))) {
                return false;
            }
            elements *= uint64_t(tensor->ne[j]);
            span += size_t(tensor->ne[j] - 1) * tensor->nb[j];
        }
    }
    if (!ggml_are_same_shape(add, unary) || !ggml_are_same_shape(add, mul) ||
            !ggml_are_same_shape(add->src[0], add) || !ggml_can_repeat(add->src[1], add) || !ggml_can_repeat(scale, mul)) {
        return false;
    }
    if (!ggml_is_contiguous(add) || !ggml_is_contiguous(unary) || !ggml_is_contiguous(mul) || mul->view_src) {
        return false;
    }
    return true;
}

static bool ggml_cuda_gdn_allocation_range(const ggml_tensor * tensor, uintptr_t & begin, uintptr_t & end) {
    if (!tensor->data || !tensor->buffer) {
        return false;
    }
    size_t span = ggml_type_size(tensor->type);
    const int64_t block = ggml_blck_size(tensor->type);
    if (tensor->ne[0] <= 0 || tensor->ne[0] % block) {
        return false;
    }
    for (int j = 0; j < GGML_MAX_DIMS; ++j) {
        if (tensor->ne[j] <= 0) {
            return false;
        }
        const size_t intervals = size_t(j == 0 ? tensor->ne[j] / block - 1 : tensor->ne[j] - 1);
        if (intervals && tensor->nb[j] > (SIZE_MAX - span) / intervals) {
            return false;
        }
        span += intervals * tensor->nb[j];
    }
    const uintptr_t base = (uintptr_t) ggml_backend_buffer_get_base(tensor->buffer);
    const size_t size = ggml_backend_buffer_get_size(tensor->buffer);
    const size_t allocation = ggml_backend_buffer_get_alloc_size(tensor->buffer, tensor);
    begin = (uintptr_t) tensor->data;
    if (size > UINTPTR_MAX - base || begin < base || begin - base > size || allocation < span ||
            allocation > size - (begin - base) || allocation > UINTPTR_MAX - begin) {
        return false;
    }
    end = begin + allocation;
    return true;
}

static bool ggml_cuda_gdn_writes_safe(const ggml_cgraph * graph, int i, int count, int device) {
    if (i < 0 || count <= 0 || count > graph->n_nodes || i > graph->n_nodes - count) {
        return false;
    }
    for (int j = i; j < i + count; ++j) {
        const ggml_tensor * write = graph->nodes[j];
        if (write->op == GGML_OP_RESHAPE || write->op == GGML_OP_VIEW) {
            continue;
        }
        uintptr_t write_begin, write_end;
        if (!write->buffer || write->buffer->buft != ggml_backend_cuda_buffer_type(device) ||
                !ggml_cuda_gdn_allocation_range(write, write_begin, write_end)) {
            return false;
        }
        for (int k = i; k < i + count; ++k) {
            for (const ggml_tensor * read : graph->nodes[k]->src) {
                if (!read) {
                    continue;
                }
                bool internal = false;
                for (int l = i; l < k; ++l) {
                    internal |= read == graph->nodes[l];
                }
                uintptr_t read_begin, read_end;
                if (!internal && (!ggml_cuda_gdn_allocation_range(read, read_begin, read_end) ||
                        (write_begin < read_end && read_begin < write_end))) {
                    return false;
                }
            }
        }
        for (int k = 0; k < graph->n_leafs; ++k) {
            const ggml_tensor * leaf = graph->leafs[k];
            uintptr_t begin, end;
            if (leaf->data && (!ggml_cuda_gdn_allocation_range(leaf, begin, end) ||
                    (write_begin < end && begin < write_end))) {
                return false;
            }
        }
        for (int k = 0; k < graph->n_nodes; ++k) {
            const ggml_tensor * read = graph->nodes[k];
            uintptr_t begin, end;
            if (read->op == GGML_OP_NONE && read->data && (!ggml_cuda_gdn_allocation_range(read, begin, end) ||
                    (write_begin < end && begin < write_end))) {
                return false;
            }
        }
    }
    return true;
}

struct ggml_cuda_gdn_packed_match {
    const ggml_tensor * packed;
    const ggml_tensor * bias;
    const ggml_tensor * scale;
    ggml_tensor * gate;
    ggml_tensor * beta;
    int64_t heads_per_group;
    int count;
    bool use_mmf = false;
};

static bool ggml_cuda_match_gdn_packed(const ggml_cgraph * graph, int i, ggml_cuda_gdn_packed_match & match, int device = -1) {
    if (i < 0 || i >= graph->n_nodes || graph->nodes[i]->op != GGML_OP_CONT) {
        return false;
    }
    auto valid_elements = [](const ggml_tensor * value) {
        uint64_t elements = 1;
        if (!value || value->type != GGML_TYPE_F32) {
            return false;
        }
        for (int j = 0; j < GGML_MAX_DIMS; ++j) {
            if (value->ne[j] <= 0 || uint64_t(value->ne[j]) > uint64_t(INT_MAX) / elements) {
                return false;
            }
            elements *= uint64_t(value->ne[j]);
        }
        return true;
    };
    const ggml_tensor * alpha = graph->nodes[i];
    const ggml_tensor * a = alpha->src[0];
    if (!valid_elements(alpha) || !valid_elements(a) || a->op != GGML_OP_VIEW || !valid_elements(a->src[0]) || !a->view_src ||
            !ggml_is_contiguous(a->src[0]) || a->view_offs % sizeof(float)) {
        return false;
    }
    const ggml_tensor * packed = a->view_src;
    if (!valid_elements(packed) || !ggml_is_contiguous(packed) || packed->ne[0] <= 0 || packed->ne[0] % 2 ||
            ggml_nelements(a->src[0]) != ggml_nelements(packed)) {
        return false;
    }
    const int64_t heads = packed->ne[0] / 2;
    const int64_t lanes = a->view_offs / sizeof(float);
    if (lanes <= 0 || lanes > heads || heads % lanes || a->nb[0] != sizeof(float)) {
        return false;
    }
    const int64_t groups = heads / lanes;
    const bool grouped = a->ne[0] == lanes && a->ne[1] == groups && a->nb[1] == 2 * lanes * sizeof(float) &&
        a->nb[2] == 2 * heads * sizeof(float) && a->nb[3] == a->nb[2] * a->ne[2];
    const bool flat = groups == 1 && a->ne[0] == heads && a->ne[2] == 1 && a->ne[3] == 1 &&
        a->nb[1] == 2 * heads * sizeof(float);
    if ((!grouped && !flat) || ggml_nelements(a) != ggml_nelements(alpha) ||
            ggml_nelements(alpha) > INT_MAX || ggml_nelements(alpha) != ggml_nelements(packed) / 2 || alpha->ne[0] != heads) {
        return false;
    }
    int cursor = i + 1;
    ggml_op ops[10] = { GGML_OP_CONT };
    int outputs[5], output_count = 0;
    auto take = [&](ggml_op op) -> ggml_tensor * {
        if (cursor >= graph->n_nodes || cursor - i >= 10 || graph->nodes[cursor]->op != op) {
            return nullptr;
        }
        ops[cursor - i] = op;
        return graph->nodes[cursor++];
    };
    const ggml_tensor * alpha_input = alpha;
    if (cursor < graph->n_nodes && graph->nodes[cursor]->op == GGML_OP_RESHAPE) {
        alpha_input = take(GGML_OP_RESHAPE);
        if (alpha_input->src[0] != alpha || alpha_input->view_src != alpha || alpha_input->view_offs) {
            return false;
        }
    }
    ggml_tensor * add = take(GGML_OP_ADD);
    ggml_tensor * unary = take(GGML_OP_UNARY);
    ggml_tensor * gate = take(GGML_OP_MUL);
    if (!valid_elements(add) || !valid_elements(unary) || !valid_elements(gate) || add->src[0] != alpha_input || unary->src[0] != add ||
            ggml_get_unary_op(unary) != GGML_UNARY_OP_SOFTPLUS || (gate->src[0] != unary && gate->src[1] != unary)) {
        return false;
    }
    outputs[output_count++] = cursor - 1;
    const ggml_tensor * bias = add->src[1];
    const ggml_tensor * scale = gate->src[0] == unary ? gate->src[1] : gate->src[0];
    if (!valid_elements(bias) || !valid_elements(scale) || bias->ne[0] != heads || scale->ne[0] != heads || ggml_nelements(bias) != heads ||
            ggml_nelements(scale) != heads || !ggml_can_repeat(bias, add) || !ggml_can_repeat(scale, gate)) {
        return false;
    }
    if (cursor < graph->n_nodes && graph->nodes[cursor]->op == GGML_OP_RESHAPE) {
        const ggml_tensor * view = take(GGML_OP_RESHAPE);
        if (!valid_elements(view) || view->src[0] != gate || view->view_src != gate || view->view_offs || !ggml_is_contiguous(view) ||
                ggml_nelements(view) != ggml_nelements(gate)) {
            return false;
        }
        outputs[output_count++] = cursor - 1;
    }
    const ggml_tensor * b = nullptr;
    if (cursor < graph->n_nodes && graph->nodes[cursor]->op == GGML_OP_VIEW) {
        b = take(GGML_OP_VIEW);
        outputs[output_count++] = cursor - 1;
    }
    ggml_tensor * beta_input = take(GGML_OP_CONT);
    ggml_tensor * beta = take(GGML_OP_UNARY);
    if (!valid_elements(beta_input) || !valid_elements(beta) || beta->src[0] != beta_input || ggml_get_unary_op(beta) != GGML_UNARY_OP_SIGMOID ||
            (b && beta_input->src[0] != b)) {
        return false;
    }
    outputs[output_count++] = cursor - 1;
    b = beta_input->src[0];
    if (!valid_elements(b) || b->op != GGML_OP_VIEW || b->src[0] != a->src[0] || b->view_src != packed || b->view_offs ||
            !ggml_are_same_shape(a, b) || !ggml_are_same_stride(a, b)) {
        return false;
    }
    if (cursor < graph->n_nodes && graph->nodes[cursor]->op == GGML_OP_RESHAPE) {
        const ggml_tensor * view = take(GGML_OP_RESHAPE);
        if (!valid_elements(view) || view->src[0] != beta || view->view_src != beta || view->view_offs || !ggml_is_contiguous(view) ||
                ggml_nelements(view) != ggml_nelements(beta)) {
            return false;
        }
        outputs[output_count++] = cursor - 1;
    }
    const ggml_tensor * values[] = { alpha, alpha_input, add, unary, gate, beta_input, beta, bias, scale, packed };
    for (const ggml_tensor * value : values) {
        if (!valid_elements(value) || !ggml_is_contiguous(value) ||
                (value != bias && value != scale && value != packed && ggml_nelements(value) != ggml_nelements(alpha))) {
            return false;
        }
    }
    if (!ggml_are_same_shape(alpha_input, add) || !ggml_are_same_shape(add, unary) || !ggml_are_same_shape(unary, gate) ||
            gate->view_src || beta->view_src || !ggml_are_same_shape(beta_input, beta) ||
            !ggml_can_fuse_subgraph(graph, i, cursor - i, ops, outputs, output_count)) {
        return false;
    }
    const ggml_tensor * parameters[] = { bias, scale };
    for (const ggml_tensor * parameter : parameters) {
        for (int j = i; j < cursor; ++j) {
            if (parameter == graph->nodes[j]) {
                return false;
            }
        }
    }
    match = { packed, bias, scale, gate, beta, lanes, cursor - i };
    if (device < 0) {
        return true;
    }
    if (ggml_cuda_get_device() != device) {
        return false;
    }
    auto range = [&](const ggml_tensor * value, uintptr_t & begin, uintptr_t & end, bool kernel_tensor) {
        if (!value->data || !value->buffer || (kernel_tensor && value->buffer->buft != ggml_backend_cuda_buffer_type(device))) {
            return false;
        }
        const uintptr_t base = (uintptr_t) ggml_backend_buffer_get_base(value->buffer);
        const size_t size = ggml_backend_buffer_get_size(value->buffer);
        const size_t allocation = ggml_backend_buffer_get_alloc_size(value->buffer, value);
        const int64_t block = ggml_blck_size(value->type);
        size_t span = ggml_type_size(value->type);
        if (value->ne[0] <= 0 || value->ne[0] % block) {
            return false;
        }
        for (int k = 0; k < GGML_MAX_DIMS; ++k) {
            if (value->ne[k] <= 0) {
                return false;
            }
            const size_t intervals = size_t(k == 0 ? value->ne[k] / block - 1 : value->ne[k] - 1);
            if (intervals && value->nb[k] > (SIZE_MAX - span) / intervals) {
                return false;
            }
            span += intervals * value->nb[k];
        }
        begin = (uintptr_t) value->data;
        if ((kernel_tensor && begin % sizeof(float)) || size > UINTPTR_MAX - base || begin < base || begin - base > size || allocation < span ||
                allocation > size - (begin - base) || allocation > UINTPTR_MAX - begin) {
            return false;
        }
        end = begin + allocation;
        return true;
    };
    const ggml_tensor * reads[] = { packed, bias, scale };
    const ggml_tensor * writes[] = { gate, beta };
    uintptr_t write_begin[2], write_end[2];
    for (int j = 0; j < 2; ++j) {
        if (!range(writes[j], write_begin[j], write_end[j], true)) {
            return false;
        }
        for (const ggml_tensor * read : reads) {
            uintptr_t begin, end;
            if (!range(read, begin, end, true) || (write_begin[j] < end && begin < write_end[j])) {
                return false;
            }
        }
        for (int k = 0; k < graph->n_leafs; ++k) {
            const ggml_tensor * leaf = graph->leafs[k];
            uintptr_t begin, end;
            if (leaf->data && (!range(leaf, begin, end, false) || (write_begin[j] < end && begin < write_end[j]))) {
                return false;
            }
        }
    }
    for (int j = 0; j < graph->n_nodes; ++j) {
        const ggml_tensor * value = graph->nodes[j];
        if (value->op != GGML_OP_NONE || !value->data) {
            continue;
        }
        uintptr_t begin, end;
        if (!range(value, begin, end, false) || (write_begin[0] < end && begin < write_end[0]) ||
                (write_begin[1] < end && begin < write_end[1])) {
            return false;
        }
    }
    if (write_begin[0] < write_end[1] && write_begin[1] < write_end[0]) {
        return false;
    }
    for (int j = i; j < cursor; ++j) {
        const ggml_tensor * value = graph->nodes[j];
        if (value->op == GGML_OP_RESHAPE && value->data != value->src[0]->data) {
            return false;
        }
    }
    if (a->src[0]->data != packed->data || a->data != (const char *) packed->data + a->view_offs || b->data != packed->data) {
        return false;
    }
    return ggml_cuda_gdn_writes_safe(graph, i, cursor - i, device);
}

static bool ggml_cuda_match_gdn_packed_projection(const ggml_cgraph * graph, int i, int device,
        ggml_cuda_gdn_packed_match & match, bool allocated = false) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(graph, i, device, match, allocated);
    return false;
#else
    if (i < 0 || i > graph->n_nodes - 3 || graph->nodes[i]->op != GGML_OP_MUL_MAT) {
        return false;
    }
    const ggml_tensor * raw = graph->nodes[i];
    int cursor = i + 1;
    const ggml_tensor * parent = raw;
    if (graph->nodes[cursor]->op == GGML_OP_RESHAPE) {
        parent = graph->nodes[cursor++];
        if (parent->src[0] != raw || parent->view_src != raw || parent->view_offs) {
            return false;
        }
    }
    const ggml_tensor * view = graph->nodes[cursor++];
    if (view->op != GGML_OP_VIEW || view->src[0] != parent || view->view_src != raw ||
            !ggml_cuda_match_gdn_packed(graph, cursor, match, allocated ? device : -1) || match.packed != raw ||
            graph->nodes[cursor]->src[0] != view || raw->view_src || !(raw->flags & GGML_TENSOR_FLAG_COMPUTE) ||
            !raw->src[0] || !raw->src[1] || ggml_cuda_op_mul_mat_use_fwht(raw)) {
        return false;
    }
    const int count = cursor - i + match.count;
    ggml_op ops[14];
    int outputs[4], output_count = 0;
    if (count > 14) {
        return false;
    }
    for (int j = i; j < i + count; ++j) {
        const ggml_tensor * value = graph->nodes[j];
        ops[j - i] = value->op;
        if (value == match.gate || value == match.beta ||
                (value->op == GGML_OP_RESHAPE && (value->view_src == match.gate || value->view_src == match.beta))) {
            if (output_count == 4) {
                return false;
            }
            outputs[output_count++] = j;
        }
        if (value == match.bias || value == match.scale) {
            return false;
        }
    }
    if (!ggml_can_fuse_subgraph(graph, i, count, ops, outputs, output_count)) {
        return false;
    }
    const ggml_tensor * bank = raw->src[0];
    const ggml_tensor * x = raw->src[1];
    const bool quantized = ggml_is_quantized(bank->type);
    const int cc = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || bank->op != GGML_OP_NONE || bank->view_src ||
            (!quantized && bank->type != GGML_TYPE_F32 && bank->type != GGML_TYPE_F16 && bank->type != GGML_TYPE_BF16) ||
            x->type != GGML_TYPE_F32 || bank->ne[0] != x->ne[0] || bank->ne[1] != raw->ne[0] ||
            raw->ne[1] != x->ne[1] || raw->ne[2] != x->ne[2] || raw->ne[3] != x->ne[3]) {
        return false;
    }
    const ggml_tensor * inputs[] = { bank, x };
    for (const ggml_tensor * input : inputs) {
        const size_t unit = ggml_type_size(input->type);
        const int64_t block = ggml_blck_size(input->type);
        uint64_t elements = 1;
        size_t span = unit;
        if (input->ne[0] <= 0 || input->ne[0] % block || input->nb[0] != unit) {
            return false;
        }
        for (int j = 0; j < GGML_MAX_DIMS; ++j) {
            const uint64_t dimension = j == 0 ? input->ne[j] / block : input->ne[j];
            if (input->ne[j] <= 0 || input->ne[j] > INT_MAX || dimension > uint64_t(INT_MAX) / elements ||
                    input->nb[j] % unit || input->nb[j] / unit > INT_MAX) {
                return false;
            }
            elements *= dimension;
            if (dimension > 1 && input->nb[j] > (SIZE_MAX - span) / (dimension - 1)) {
                return false;
            }
            span += (dimension - 1) * input->nb[j];
        }
        if (span / unit > INT_MAX) {
            return false;
        }
    }
    if (x->ne[2] > 65535 || x->ne[3] > 65535 || (quantized && x->ne[2] > 65535 / x->ne[3]) ||
            bank->ne[2] > x->ne[2] || x->ne[2] % bank->ne[2] || bank->ne[3] > x->ne[3] || x->ne[3] % bank->ne[3]) {
        return false;
    }
    const size_t alignment = quantized ? (ggml_type_size(bank->type) % sizeof(uint32_t) ?
        (ggml_type_size(bank->type) % sizeof(uint16_t) ? 1 : sizeof(uint16_t)) : sizeof(uint32_t)) : 2 * ggml_type_size(bank->type);
    for (int j = 1; j < GGML_MAX_DIMS; ++j) {
        if (bank->nb[j] % alignment || x->nb[j] % (quantized ? sizeof(float) : 2 * sizeof(float))) {
            return false;
        }
    }
    const bool vector = quantized ? ggml_cuda_should_use_mmvq(bank->type, cc, x->ne[1]) :
        ggml_cuda_should_use_mmvf(bank->type, cc, bank->ne, bank->nb, x->ne[1]);
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const bool mmf = !quantized && !vector && ggml_cuda_should_use_mmf(bank->type, cc, warp_size, bank->ne, bank->nb, x->ne[1], false);
    if ((!vector && !mmf) || (mmf && (bank->nb[1] % (2 * sizeof(float)) || x->nb[1] % ((bank->type == GGML_TYPE_F32 ? 2 : 4) * sizeof(float))))) {
        return false;
    }
    match.count = count;
    match.use_mmf = mmf;
    if (!allocated) {
        return true;
    }
    const ggml_tensor * metadata[] = { raw, parent, view };
    for (const ggml_tensor * value : metadata) {
        uintptr_t begin, end;
        if (!value->buffer || value->buffer->buft != ggml_backend_cuda_buffer_type(device) ||
                !ggml_cuda_gdn_allocation_range(value, begin, end) ||
                (value->op == GGML_OP_RESHAPE && value->data != raw->data)) {
            return false;
        }
    }
    for (const ggml_tensor * input : inputs) {
        uintptr_t begin, end;
        if (!input->buffer || input->buffer->buft != ggml_backend_cuda_buffer_type(device) ||
                !ggml_cuda_gdn_allocation_range(input, begin, end) || begin % (input == bank ? alignment : (quantized ? sizeof(float) : 2 * sizeof(float)))) {
            return false;
        }
    }
    if (quantized && ggml_backend_buffer_get_usage(bank->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
            ggml_backend_buffer_get_alloc_size(bank->buffer, bank) != ggml_nbytes(bank)) {
        return false;
    }
    return ggml_cuda_gdn_writes_safe(graph, i, count, device);
#endif
}

struct ggml_cuda_gdn_post_match {
    ggml_tensor * producer;
    ggml_tensor * beta;
    int count;
    const ggml_tensor * beta_scale = nullptr;
};

static bool ggml_cuda_match_gdn_post(const ggml_cgraph * graph, int i, ggml_cuda_gdn_post_match & match,
        int device = -1, bool cublas = false, bool projection_scales = false) {
    if (!ggml_cuda_match_add_softplus_mul(graph, i) || i > graph->n_nodes - 6) {
        return false;
    }
    const ggml_tensor * add = graph->nodes[i];
    const ggml_tensor * unary = graph->nodes[i + 1];
    const ggml_tensor * mul = graph->nodes[i + 2];
    const ggml_tensor * gate = graph->nodes[i + 3];
    const ggml_tensor * scale = mul->src[0] == unary ? mul->src[1] : mul->src[0];
    ggml_tensor * producer = graph->nodes[i + 4];
    int last = i + 6;
    const ggml_tensor * beta_scale = nullptr;
    ggml_op ops[8] = { GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_RESHAPE, GGML_OP_CONT, GGML_OP_UNARY, GGML_OP_UNARY, GGML_OP_UNARY };
    if (producer->op == GGML_OP_MUL_MAT) {
        if (i > graph->n_nodes - 7) {
            return false;
        }
        ops[4] = GGML_OP_MUL_MAT;
        ops[5] = GGML_OP_RESHAPE;
        const ggml_tensor * input = producer;
        if (projection_scales && graph->nodes[i + 5]->op == GGML_OP_MUL) {
            if (i > graph->n_nodes - 8) {
                return false;
            }
            input = graph->nodes[i + 5];
            beta_scale = input->src[0] == producer ? input->src[1] : input->src[0];
            if (!beta_scale || (input->src[0] != producer && input->src[1] != producer) || input->view_src ||
                    input->type != GGML_TYPE_F32 || !(input->flags & GGML_TENSOR_FLAG_COMPUTE) || !ggml_is_contiguous(input) || !ggml_are_same_shape(input, producer)) {
                return false;
            }
            ops[5] = GGML_OP_MUL;
            ops[6] = GGML_OP_RESHAPE;
            last = i + 7;
        }
        const ggml_tensor * view = graph->nodes[last - 1];
        if (view->src[0] != input || view->view_src != input || view->view_offs != 0 ||
                !ggml_are_same_shape(view, gate)) {
            return false;
        }
    } else if (producer->op == GGML_OP_VIEW) {
        if (i > graph->n_nodes - 7) {
            return false;
        }
        ops[4] = GGML_OP_VIEW;
        ops[5] = GGML_OP_CONT;
        const ggml_tensor * view = producer;
        producer = graph->nodes[i + 5];
        if (producer->src[0] != view || !view->src[0] ||
                view->view_src != (view->src[0]->view_src ? view->src[0]->view_src : view->src[0])) {
            return false;
        }
    } else if (producer->op == GGML_OP_CONT) {
        last = i + 5;
    } else {
        return false;
    }
    ggml_tensor * beta = graph->nodes[last];
    const ggml_tensor * beta_input = graph->nodes[last - 1];
    const int outputs[] = { i + 2, i + 3, last, i + 4 };
    const int output_count = ops[4] == GGML_OP_VIEW ? 4 : 3;
    const ggml_tensor * results[] = { gate, producer, beta_input, beta };
    for (const ggml_tensor * result : results) {
        uint64_t elements = 1;
        for (int j = 0; j < GGML_MAX_DIMS; ++j) {
            if (result->ne[j] <= 0 || uint64_t(result->ne[j]) > uint64_t(INT_MAX) / elements || result->nb[j] > INT_MAX) {
                return false;
            }
            elements *= uint64_t(result->ne[j]);
        }
        if (result->type != GGML_TYPE_F32 || !ggml_is_contiguous(result) || elements != uint64_t(ggml_nelements(mul))) {
            return false;
        }
    }
    if (!ggml_can_fuse_subgraph(graph, i, last - i + 1, ops, outputs, output_count) ||
            gate->src[0] != mul || gate->view_src != mul || gate->view_offs != 0 || !ggml_is_contiguous(gate) ||
            beta->src[0] != beta_input || ggml_get_unary_op(beta) != GGML_UNARY_OP_SIGMOID ||
            producer->type != GGML_TYPE_F32 || beta->type != GGML_TYPE_F32 || !ggml_is_contiguous(producer) ||
            !ggml_is_contiguous(beta) || beta->view_src || ggml_nelements(producer) != ggml_nelements(mul) ||
            ggml_nelements(beta) != ggml_nelements(mul) || !producer->src[0] ||
            !(producer->flags & GGML_TENSOR_FLAG_COMPUTE) ||
            (producer->op == GGML_OP_MUL_MAT && !producer->src[1])) {
        return false;
    }
    const ggml_tensor * reads[] = { add->src[0], add->src[1], scale, producer->src[0], producer->src[1], beta_scale };
    for (const ggml_tensor * read : reads) {
        if (!read) {
            continue;
        }
        uint64_t elements = 1;
        size_t span = ggml_type_size(read->type);
        const int64_t block = ggml_blck_size(read->type);
        if (read->ne[0] <= 0 || read->ne[0] % block || read->nb[0] != ggml_type_size(read->type)) {
            return false;
        }
        for (int j = 0; j < GGML_MAX_DIMS; ++j) {
            if (read->ne[j] <= 0 || uint64_t(read->ne[j]) > uint64_t(INT_MAX) / elements || read->nb[j] > INT_MAX) {
                return false;
            }
            elements *= uint64_t(read->ne[j]);
            const size_t intervals = size_t(j == 0 ? read->ne[j] / block - 1 : read->ne[j] - 1);
            if (intervals && read->nb[j] > (SIZE_MAX - span) / intervals) {
                return false;
            }
            span += intervals * read->nb[j];
        }
        for (int j = i; j <= last; ++j) {
            if (read != graph->nodes[j]) {
                continue;
            }
            if (producer->op != GGML_OP_CONT || producer != graph->nodes[i + 5] ||
                    read != graph->nodes[i + 4] || read->op != GGML_OP_VIEW) {
                return false;
            }
            for (int k = i; k <= last; ++k) {
                if (read->src[0] == graph->nodes[k] || read->view_src == graph->nodes[k]) {
                    return false;
                }
            }
        }
    }
    match = { producer, beta, last - i + 1, beta_scale };
    if (device < 0) {
        return true;
    }
    if (ggml_cuda_get_device() != device || !mul->data || !beta->data ||
            (beta_scale ? graph->nodes[i + 5]->data : producer->data) != beta_input->data || gate->data != mul->data) {
        return false;
    }
    auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        const uintptr_t x = (uintptr_t) a->data, y = (uintptr_t) b->data;
        return x <= y ? y - x < ggml_nbytes(a) : x - y < ggml_nbytes(b);
    };
    const ggml_tensor * tensors[] = { mul, beta, producer, add->src[0], add->src[1], scale,
        producer->src[0], producer->src[1], beta_scale, beta_scale ? graph->nodes[i + 5] : nullptr };
    for (const ggml_tensor * tensor : tensors) {
        if (!tensor) {
            continue;
        }
        if (!tensor->data || !tensor->buffer || tensor->buffer->buft != ggml_backend_cuda_buffer_type(device)) {
            return false;
        }
        const uintptr_t base = (uintptr_t) ggml_backend_buffer_get_base(tensor->buffer);
        const uintptr_t data = (uintptr_t) tensor->data;
        const size_t size = ggml_backend_buffer_get_size(tensor->buffer);
        const size_t span = ggml_nbytes(tensor);
        if (size > UINTPTR_MAX - base || data < base || data - base > size ||
                span > size - (data - base) || span > UINTPTR_MAX - data) {
            return false;
        }
    }
    if (!cublas && producer->op == GGML_OP_MUL_MAT && ggml_is_quantized(producer->src[0]->type)) {
        const ggml_tensor * weights = producer->src[0];
        if (ggml_backend_buffer_get_usage(weights->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
                ggml_backend_buffer_get_alloc_size(weights->buffer, weights) != ggml_nbytes(weights)) {
            return false;
        }
    }
    if (overlap(mul, beta) || overlap(mul, producer) ||
            (overlap(beta, producer) && beta->data != producer->data) ||
            (beta_scale && (overlap(mul, beta_input) ||
                (overlap(beta, beta_input) && beta->data != beta_input->data)))) {
        return false;
    }
    for (const ggml_tensor * read : reads) {
        if (read && (overlap(mul, read) || overlap(beta, read) || overlap(producer, read))) {
            return false;
        }
    }
    const ggml_tensor * writes[] = { mul, gate, beta, producer };
    for (const ggml_tensor * write : writes) {
        uintptr_t begin, end;
        if (!ggml_cuda_gdn_allocation_range(write, begin, end)) {
            return false;
        }
        for (int j = i; j <= last; ++j) {
            for (const ggml_tensor * source : graph->nodes[j]->src) {
                if (!source) {
                    continue;
                }
                bool internal = false;
                for (int k = i; k < j; ++k) {
                    internal |= source == graph->nodes[k];
                }
                uintptr_t source_begin, source_end;
                if (!internal && (!ggml_cuda_gdn_allocation_range(source, source_begin, source_end) ||
                        (begin < source_end && source_begin < end))) {
                    return false;
                }
            }
        }
        for (int j = 0; j < graph->n_leafs; ++j) {
            const ggml_tensor * leaf = graph->leafs[j];
            uintptr_t leaf_begin, leaf_end;
            if (leaf->data && (!ggml_cuda_gdn_allocation_range(leaf, leaf_begin, leaf_end) ||
                    (begin < leaf_end && leaf_begin < end))) {
                return false;
            }
        }
        if (projection_scales) {
            for (int j = 0; j < graph->n_nodes; ++j) {
                const ggml_tensor * value = graph->nodes[j];
                uintptr_t other_begin, other_end;
                if (value->op == GGML_OP_NONE && value->data && (!ggml_cuda_gdn_allocation_range(value, other_begin, other_end) ||
                        (begin < other_end && other_begin < end))) {
                    return false;
                }
            }
        }
    }
    return ggml_cuda_gdn_writes_safe(graph, i, last - i + 1, device);
}

enum class ggml_cuda_gdn_projection { VECTOR, MMF, BF16_ROUNDED, CUBLAS };

struct ggml_cuda_gdn_projection_match {
    const ggml_tensor * beta;
    const ggml_tensor * bias;
    const ggml_tensor * scale;
    const ggml_tensor * alpha_scale;
    const ggml_tensor * beta_scale;
    ggml_tensor * gate;
    ggml_tensor * beta_output;
    int count;
    bool use_mmf;
};

static bool ggml_cuda_match_gdn_projections(const ggml_cgraph * graph, int i, int device,
        ggml_cuda_gdn_projection kind, bool allocated = false, ggml_cuda_gdn_projection_match * match = nullptr) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(graph, i, device, kind, allocated, match);
    return false;
#else
    if (i < 0 || i > graph->n_nodes - 9) {
        return false;
    }
    const bool cublas = kind == ggml_cuda_gdn_projection::CUBLAS;
    const bool bf16_rounded = kind == ggml_cuda_gdn_projection::BF16_ROUNDED;
    const bool mmf = kind == ggml_cuda_gdn_projection::MMF;
    const ggml_tensor * alpha = graph->nodes[i];
    const ggml_tensor * alpha_input = alpha;
    const ggml_tensor * alpha_scale = nullptr;
    if (!cublas && graph->nodes[i + 1]->op == GGML_OP_MUL) {
        alpha_input = graph->nodes[i + 1];
        alpha_scale = alpha_input->src[0] == alpha ? alpha_input->src[1] : alpha_input->src[0];
        if (!alpha_scale || (alpha_input->src[0] != alpha && alpha_input->src[1] != alpha) || alpha_input->view_src ||
                alpha_input->type != GGML_TYPE_F32 || !(alpha_input->flags & GGML_TENSOR_FLAG_COMPUTE) || !ggml_is_contiguous(alpha_input) || !ggml_are_same_shape(alpha_input, alpha)) {
            return false;
        }
    }
    const int add_index = i + (alpha_scale ? 3 : 2);
    const ggml_tensor * reshaped = graph->nodes[add_index - 1];
    const ggml_tensor * add = graph->nodes[add_index];
    ggml_tensor * mul = graph->nodes[add_index + 2];
    const ggml_tensor * scale = mul->src[0] == graph->nodes[add_index + 1] ? mul->src[1] : mul->src[0];
    ggml_cuda_gdn_post_match post;
    if (!ggml_cuda_match_gdn_post(graph, add_index, post, allocated ? device : -1, cublas, !cublas) ||
            post.producer->op != GGML_OP_MUL_MAT || (cublas && post.count != 7)) {
        return false;
    }
    const ggml_tensor * beta = post.producer;
    const int count = add_index - i + post.count;
    ggml_op ops[11] = { GGML_OP_MUL_MAT };
    int cursor = 1;
    if (alpha_scale) {
        ops[cursor++] = GGML_OP_MUL;
    }
    ops[cursor++] = GGML_OP_RESHAPE;
    ops[cursor++] = GGML_OP_ADD;
    ops[cursor++] = GGML_OP_UNARY;
    ops[cursor++] = GGML_OP_MUL;
    ops[cursor++] = GGML_OP_RESHAPE;
    ops[cursor++] = GGML_OP_MUL_MAT;
    if (post.beta_scale) {
        ops[cursor++] = GGML_OP_MUL;
    }
    ops[cursor++] = GGML_OP_RESHAPE;
    ops[cursor++] = GGML_OP_UNARY;
    const int outputs[] = { add_index + 2, add_index + 3, i + count - 1 };
    if (cursor != count || !ggml_can_fuse_subgraph(graph, i, count, ops, outputs, 3) ||
            !alpha->src[0] || !alpha->src[1] || !beta->src[0] || beta->src[1] != alpha->src[1] ||
            alpha->type != GGML_TYPE_F32 || !(alpha->flags & GGML_TENSOR_FLAG_COMPUTE) || alpha->view_src || beta->view_src ||
            reshaped->src[0] != alpha_input || reshaped->view_src != alpha_input || reshaped->view_offs || add->src[0] != reshaped ||
            !ggml_are_same_shape(alpha, beta) || !ggml_are_same_stride(alpha, beta) || !ggml_is_contiguous(alpha) ||
            ggml_get_op_params_i32(alpha, 0) != ggml_get_op_params_i32(beta, 0)) {
        return false;
    }
    const ggml_tensor * wa = alpha->src[0];
    const ggml_tensor * wb = beta->src[0];
    const ggml_tensor * x = alpha->src[1];
    const bool quantized = ggml_is_quantized(wa->type);
    size_t wa_span = ggml_type_size(wa->type);
    if (cublas) {
        const int64_t block = ggml_blck_size(wa->type);
        uint64_t elements = 1;
        if (wa->ne[0] <= 0 || wa->ne[0] % block || wa->nb[0] != ggml_type_size(wa->type)) {
            return false;
        }
        for (int j = 0; j < GGML_MAX_DIMS; ++j) {
            if (wa->ne[j] <= 0 || uint64_t(wa->ne[j]) > uint64_t(INT_MAX) / elements || wa->nb[j] > INT_MAX) {
                return false;
            }
            elements *= uint64_t(wa->ne[j]);
            const size_t intervals = size_t(j == 0 ? wa->ne[j] / block - 1 : wa->ne[j] - 1);
            if (intervals && wa->nb[j] > (SIZE_MAX - wa_span) / intervals) {
                return false;
            }
            wa_span += intervals * wa->nb[j];
        }
    }
    if ((!quantized && wa->type != GGML_TYPE_F32 && wa->type != GGML_TYPE_F16 && wa->type != GGML_TYPE_BF16) ||
            (!cublas && (wa->op != GGML_OP_NONE || wb->op != GGML_OP_NONE || wa->view_src || wb->view_src)) ||
            !ggml_are_same_shape(wa, wb) || (!cublas && !ggml_are_same_stride(wa, wb)) || wa->type != wb->type ||
            (!cublas && (wa->ne[2] != 1 || wa->ne[3] != 1)) ||
            alpha->ne[1] != x->ne[1] || alpha->ne[2] != x->ne[2] || alpha->ne[3] != x->ne[3] ||
            (!cublas && (x->ne[2] > 65535 || x->ne[3] > 65535 ||
                (quantized && x->ne[2] > 65535 / x->ne[3]) || ggml_nbytes(x) / sizeof(float) > INT_MAX)) ||
            wa->ne[0] != x->ne[0] || wa->ne[1] != alpha->ne[0] || x->type != GGML_TYPE_F32 ||
            !add->src[1] || !scale || !ggml_is_contiguous(add->src[1]) || !ggml_is_contiguous(scale) ||
            ggml_nelements(add->src[1]) != alpha->ne[0] || ggml_nelements(scale) != alpha->ne[0] ||
            ggml_cuda_op_mul_mat_use_fwht(alpha) || ggml_cuda_op_mul_mat_use_fwht(beta)) {
        return false;
    }
    const ggml_tensor * bank_scales[] = { alpha_scale, post.beta_scale };
    for (const ggml_tensor * bank_scale : bank_scales) {
        if (!bank_scale) {
            continue;
        }
        if (cublas || bank_scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(bank_scale) ||
                (bank_scale->ne[0] != 1 && bank_scale->ne[0] != alpha->ne[0]) || bank_scale->ne[1] != 1 ||
                bank_scale->ne[2] != 1 || bank_scale->ne[3] != 1 || !ggml_can_repeat(bank_scale, alpha)) {
            return false;
        }
        for (int j = i; j < i + count; ++j) {
            if (bank_scale == graph->nodes[j]) {
                return false;
            }
        }
    }
    const ggml_tensor * parameters[] = { add->src[1], scale };
    for (const ggml_tensor * parameter : parameters) {
        for (int j = i; j < i + count; ++j) {
            if (parameter == graph->nodes[j]) {
                return false;
            }
        }
    }
    const int cc = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return false;
    }
    // Limit repeated bank scans to two column tiles.
    if (bf16_rounded && (!fast_bf16_hardware_available(cc) || wa->type != GGML_TYPE_BF16 || x->ne[1] <= 0 || x->ne[1] > 2 * MMVF_MAX_BATCH_SIZE ||
            wa->ne[0] <= 0 || wa->ne[0] % 2 || wa->nb[0] != sizeof(nv_bfloat16) || x->nb[0] != sizeof(float))) {
        return false;
    }
    if (cublas || bf16_rounded) {
        if (allocated && (!wa->buffer || !wb->buffer)) {
            return false;
        }
        const int warp_size = ggml_cuda_info().devices[device].warp_size;
        const ggml_tensor * projections[] = { alpha, beta };
        for (const ggml_tensor * projection : projections) {
            const ggml_tensor * bank = projection->src[0];
            const bool bad_padding_clear = bank->view_src && (!allocated ||
                (ggml_backend_buffer_get_usage(bank->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
                ggml_nbytes(bank) != ggml_backend_buffer_get_alloc_size(bank->buffer, bank)));
            const bool transposed_vector = bank->ne[1] == 1 && x->ne[1] > MMVF_MAX_BATCH_SIZE &&
                projection->ne[2] == 1 && projection->ne[3] == 1 && bank->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(bank) && ggml_is_contiguous(x) && ggml_is_contiguous(projection) &&
                ggml_cuda_should_use_mmvf(x->type, cc, x->ne, x->nb, 1);
            if (!bad_padding_clear && (transposed_vector ||
                    ggml_cuda_should_use_mmvf(bank->type, cc, bank->ne, bank->nb, x->ne[1]) ||
                    ggml_cuda_should_use_mmf(bank->type, cc, warp_size, bank->ne, bank->nb, x->ne[1], false) ||
                    ggml_cuda_should_use_mmvq(bank->type, cc, x->ne[1]) ||
                    ggml_cuda_should_use_mmq(bank->type, cc, x->ne[1], 0))) {
                return false;
            }
        }
        const ggml_type compute_type = ggml_cuda_mul_mat_cublas_compute_type(cc, wa, x, alpha);
        if ((compute_type != GGML_TYPE_F16 && compute_type != GGML_TYPE_BF16) ||
                (bf16_rounded && compute_type != GGML_TYPE_BF16) ||
                compute_type != ggml_cuda_mul_mat_cublas_compute_type(cc, wb, x, beta)) {
            return false;
        }
    } else {
        if (mmf) {
            const int warp_size = ggml_cuda_info().devices[device].warp_size;
            const ggml_tensor * banks[] = { wa, wb };
            for (const ggml_tensor * bank : banks) {
                if (ggml_cuda_should_use_mmvf(bank->type, cc, bank->ne, bank->nb, x->ne[1]) ||
                        !ggml_cuda_should_use_mmf(bank->type, cc, warp_size, bank->ne, bank->nb, x->ne[1], false) ||
                        bank->nb[1] % (2*sizeof(float)) || bank->ne[1] > INT_MAX/2) {
                    return false;
                }
            }
            if (x->nb[1] % ((wa->type == GGML_TYPE_F32 ? 2 : 4)*sizeof(float))) {
                return false;
            }
        } else if (quantized ? !ggml_cuda_should_use_mmvq(wa->type, cc, x->ne[1]) :
                !ggml_cuda_should_use_mmvf(wa->type, cc, wa->ne, wa->nb, x->ne[1])) {
            return false;
        }
    }
    if (!cublas) {
        for (int j = 1; j < GGML_MAX_DIMS; ++j) {
            if (x->nb[j] % (quantized ? sizeof(float) : 2 * sizeof(float)) ||
                    wa->nb[j] % (bf16_rounded ? 2 * sizeof(nv_bfloat16) : ggml_type_size(wa->type)) ||
                    (bf16_rounded && (wa->nb[j] / sizeof(nv_bfloat16) > INT_MAX || x->nb[j] / sizeof(float) > INT_MAX))) {
                return false;
            }
        }
        if (ggml_nbytes(wa) / ggml_type_size(wa->type) > INT_MAX) {
            return false;
        }
    }
    const size_t weight_alignment = quantized ?
        (ggml_type_size(wa->type) % sizeof(uint32_t) ? (ggml_type_size(wa->type) % sizeof(uint16_t) ? 1 : sizeof(uint16_t)) : sizeof(uint32_t)) : (cublas ? ggml_type_size(wa->type) : 2 * ggml_type_size(wa->type));
    if (quantized && !cublas) {
        for (int j = 1; j < GGML_MAX_DIMS; ++j) {
            if (wa->nb[j] % weight_alignment) {
                return false;
            }
        }
    }
    if (match) {
        *match = { beta, add->src[1], scale, alpha_scale, post.beta_scale, mul, post.beta, count, mmf };
    }
    if (!allocated) {
        return true;
    }
    if (!wa->data || !wa->buffer || wa->buffer->buft != ggml_backend_cuda_buffer_type(device) ||
            reshaped->data != alpha_input->data || (uintptr_t) mul->data % sizeof(float) ||
            (uintptr_t) post.beta->data % sizeof(float) || (uintptr_t) wa->data % weight_alignment ||
            (uintptr_t) wb->data % weight_alignment || (uintptr_t) x->data % (cublas ? sizeof(float) : 2 * sizeof(float))) {
        return false;
    }
    const uintptr_t begin = (uintptr_t) wa->data;
    const uintptr_t base = (uintptr_t) ggml_backend_buffer_get_base(wa->buffer);
    const size_t size = ggml_backend_buffer_get_size(wa->buffer);
    const size_t allocation = ggml_backend_buffer_get_alloc_size(wa->buffer, wa);
    if (size > UINTPTR_MAX - base || begin < base || begin - base > size || allocation < (cublas ? wa_span : ggml_nbytes(wa)) ||
            allocation > size - (begin - base) || allocation > UINTPTR_MAX - begin) {
        return false;
    }
    if (quantized && !cublas) {
        const ggml_tensor * banks[] = { wa, wb };
        for (const ggml_tensor * bank : banks) {
            if (ggml_backend_buffer_get_usage(bank->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
                    ggml_backend_buffer_get_alloc_size(bank->buffer, bank) != ggml_nbytes(bank)) {
                return false;
            }
        }
    }
    const ggml_tensor * writes[] = { mul, post.beta };
    for (const ggml_tensor * write : writes) {
        const uintptr_t other = (uintptr_t) write->data;
        const size_t extent = ggml_backend_buffer_get_alloc_size(write->buffer, write);
        if (begin < other + extent && other < begin + allocation) {
            return false;
        }
    }
    for (const ggml_tensor * bank_scale : bank_scales) {
        if (!bank_scale) {
            continue;
        }
        if (!bank_scale->data || !bank_scale->buffer || bank_scale->buffer->buft != ggml_backend_cuda_buffer_type(device) ||
                (uintptr_t) bank_scale->data % sizeof(float)) {
            return false;
        }
        const uintptr_t scale_begin = (uintptr_t) bank_scale->data;
        const uintptr_t scale_base = (uintptr_t) ggml_backend_buffer_get_base(bank_scale->buffer);
        const size_t scale_size = ggml_backend_buffer_get_size(bank_scale->buffer);
        const size_t scale_span = ggml_backend_buffer_get_alloc_size(bank_scale->buffer, bank_scale);
        if (scale_size > UINTPTR_MAX - scale_base || scale_begin < scale_base || scale_begin - scale_base > scale_size ||
                scale_span < ggml_nbytes(bank_scale) || scale_span > scale_size - (scale_begin - scale_base) || scale_span > UINTPTR_MAX - scale_begin) {
            return false;
        }
        for (const ggml_tensor * write : writes) {
            const uintptr_t write_begin = (uintptr_t) write->data;
            const size_t write_span = ggml_backend_buffer_get_alloc_size(write->buffer, write);
            if (scale_begin < write_begin + write_span && write_begin < scale_begin + scale_span) {
                return false;
            }
        }
    }
    if (!ggml_cuda_gdn_writes_safe(graph, i, count, device)) {
        return false;
    }
    if (cublas) {
        const ggml_tensor * raw_writes[] = { alpha, beta };
        const ggml_tensor * reads[] = { wa, wb, x, add->src[1], scale };
        for (const ggml_tensor * write : raw_writes) {
            if (!write->data || !write->buffer || write->buffer->buft != ggml_backend_cuda_buffer_type(device)) {
                return false;
            }
            const uintptr_t write_begin = (uintptr_t) write->data;
            const size_t write_span = ggml_backend_buffer_get_alloc_size(write->buffer, write);
            const uintptr_t write_base = (uintptr_t) ggml_backend_buffer_get_base(write->buffer);
            const size_t write_size = ggml_backend_buffer_get_size(write->buffer);
            if (write_span < ggml_nbytes(write) || write_size > UINTPTR_MAX - write_base || write_begin < write_base ||
                    write_begin - write_base > write_size || write_span > write_size - (write_begin - write_base) ||
                    write_span > UINTPTR_MAX - write_begin) {
                return false;
            }
            for (const ggml_tensor * read : reads) {
                const uintptr_t read_begin = (uintptr_t) read->data;
                const size_t read_span = ggml_backend_buffer_get_alloc_size(read->buffer, read);
                if (read_span > UINTPTR_MAX - read_begin ||
                        (write_begin < read_begin + read_span && read_begin < write_begin + write_span)) {
                    return false;
                }
            }
            for (int j = 0; j < graph->n_leafs; ++j) {
                const ggml_tensor * leaf = graph->leafs[j];
                if (!leaf->data) {
                    continue;
                }
                const uintptr_t leaf_begin = (uintptr_t) leaf->data;
                const size_t leaf_span = ggml_backend_buffer_get_alloc_size(leaf->buffer, leaf);
                if (leaf_span > UINTPTR_MAX - leaf_begin ||
                        (write_begin < leaf_begin + leaf_span && leaf_begin < write_begin + write_span)) {
                    return false;
                }
            }
        }
        const uintptr_t alpha_begin = (uintptr_t) alpha->data;
        const uintptr_t beta_begin = (uintptr_t) beta->data;
        if (alpha_begin < beta_begin + ggml_backend_buffer_get_alloc_size(beta->buffer, beta) &&
                beta_begin < alpha_begin + ggml_backend_buffer_get_alloc_size(alpha->buffer, alpha)) {
            return false;
        }
    }
    return true;
#endif
}

static bool ggml_cuda_can_fuse(const struct ggml_cgraph *                cgraph,
                               int                                       node_idx,
                               std::initializer_list<enum ggml_op>       ops,
                               std::initializer_list<enum ggml_unary_op> unary_ops,
                               bool                                      allow_cached = false) {
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

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu, ffn_up_bias, ffn_gate_bias,
                nullptr, nullptr, allow_cached)) {
            int out_nodes[] = { node_idx + 4 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if ((is_equal(mul_mat_id_glu_ops, ops) || is_equal(mul_mat_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * ffn_gate = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_up   = cgraph->nodes[node_idx + 1];
        const ggml_tensor * glu      = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu, nullptr, nullptr, nullptr, nullptr, allow_cached)) {
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

// try and fuse nodes and return the number of nodes to skip


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

static ggml_cuda_norm_emit_match ggml_cuda_match_norm_emit(ggml_cgraph * graph, int i, int device, bool check_storage = true) {
    if (graph->nodes[i]->op == GGML_OP_DSV4_HC_POST) {
        const auto hc = ggml_cuda_match_hc_post_norm(graph, i);
        if (!hc.count || (check_storage && !ggml_cuda_hc_post_norm_memory_ok(graph, i, hc, device))) { return {}; }
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
    if (!check_storage) { return match; }
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

void ggml_cuda_moe_source_image_alloc_deps(ggml_cgraph * graph, int device,
        const ggml_backend_graph_optimize_params * params) {
    for (int i = 0; i < graph->n_nodes; ++i) {
        const auto match = ggml_cuda_match_norm_emit(graph, i, device, false);
        if (!match.norm || match.norm->view_src || match.dst->view_src) { continue; }
        bool reader = false;
        for (int j = match.last + 1; j < graph->n_nodes; ++j) {
            const auto * node = graph->nodes[j];
            if (node->op != GGML_OP_MUL_MAT || !node->src[0] || !node->src[1]) { continue; }
            const auto * input = node->src[1];
            const auto * root = input->view_src ? input->view_src : input;
            reader |= root == match.dst && (ggml_is_quantized(node->src[0]->type) ||
                node->src[0]->type == GGML_TYPE_F16 || node->src[0]->type == GGML_TYPE_BF16);
        }
        if (reader) { params->add_alloc_dep(params->user_data, match.norm->src[0], match.dst); }
        i = match.last;
    }
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


struct ggml_cuda_scaled_unary_match {
    ggml_tensor * first = nullptr;
    ggml_tensor * unary = nullptr;
    ggml_tensor * last = nullptr;
    int count = 0;
};

static ggml_cuda_scaled_unary_match ggml_cuda_match_scaled_unary(ggml_cgraph * cgraph, int i) {
    ggml_cuda_scaled_unary_match match;
    ggml_tensor * first = cgraph->nodes[i];
    const bool before = first->op == GGML_OP_SCALE;
    const int u = i + int(before);
    if (u >= cgraph->n_nodes || cgraph->nodes[u]->op != GGML_OP_UNARY ||
            (ggml_get_unary_op(cgraph->nodes[u]) != GGML_UNARY_OP_SILU && ggml_get_unary_op(cgraph->nodes[u]) != GGML_UNARY_OP_SIGMOID)) {
        return match;
    }
    if (before && (ggml_cuda_can_fuse(cgraph, u, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SILU }) ||
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

static bool ggml_cuda_mmvf_postop_bytes(const ggml_tensor * tensor, size_t & bytes) {
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

static bool ggml_cuda_mmvf_postop_range(const ggml_tensor * tensor, int device, uintptr_t & begin, uintptr_t & end) {
    if (!tensor->buffer || !tensor->data || tensor->buffer->buft != ggml_backend_cuda_buffer_type(device)) {
        return false;
    }
    const uintptr_t base = (uintptr_t) ggml_backend_buffer_get_base(tensor->buffer);
    const size_t size = ggml_backend_buffer_get_size(tensor->buffer);
    size_t bytes;
    if (!ggml_cuda_mmvf_postop_bytes(tensor, bytes)) { return false; }
    begin = (uintptr_t) tensor->data;
    if (begin < base || begin - base > size || bytes > size - (begin - base) || begin > UINTPTR_MAX - bytes) {
        return false;
    }
    if (tensor->view_src) {
        const ggml_tensor * root = tensor->view_src;
        const uintptr_t root_begin = (uintptr_t) root->data;
        size_t root_bytes;
        if (!ggml_cuda_mmvf_postop_bytes(root, root_bytes)) { return false; }
        if (root->view_src || root->buffer != tensor->buffer || !root->data || root_begin < base || root_begin - base > size ||
                root_bytes > size - (root_begin - base) || tensor->view_offs > root_bytes || bytes > root_bytes - tensor->view_offs ||
                root_begin > UINTPTR_MAX - tensor->view_offs || begin != root_begin + tensor->view_offs) {
            return false;
        }
    }
    end = begin + bytes;
    return true;
}

struct ggml_cuda_mmvf_postop_match {
    ggml_tensor * mm = nullptr;
    ggml_tensor * pre = nullptr;
    ggml_tensor * unary = nullptr;
    ggml_tensor * post = nullptr;
    int count = 0;
};

static ggml_cuda_mmvf_postop_match ggml_cuda_match_mmvf_postop(ggml_backend_cuda_context & ctx, ggml_cgraph * graph, int i) {
    ggml_cuda_mmvf_postop_match match;
    ggml_tensor * mm = graph->nodes[i];
    if (mm->op != GGML_OP_MUL_MAT || !mm->src[0] || !mm->src[1] || mm->type != GGML_TYPE_F32 ||
            mm->src[1]->type != GGML_TYPE_F32 || ggml_cuda_op_mul_mat_use_fwht(mm)) {
        return match;
    }
    const ggml_tensor * weight = mm->src[0];
    const ggml_tensor * input = mm->src[1];
    if (weight->type != GGML_TYPE_F32 && weight->type != GGML_TYPE_F16 && weight->type != GGML_TYPE_BF16) {
        return match;
    }
    const ggml_tensor * tensors[] = {weight, input, mm};
    for (const ggml_tensor * tensor : tensors) {
        uintptr_t begin, end;
        if (!ggml_cuda_mmvf_postop_range(tensor, ctx.device, begin, end) || tensor->nb[0] != ggml_type_size(tensor->type) ||
                (end - begin)/ggml_type_size(tensor->type) > INT_MAX || begin % (2*ggml_type_size(tensor->type)) ||
                (tensor == mm && (end - begin)/sizeof(float) > INT_MAX - CUDA_NEG_BLOCK_SIZE)) {
            return match;
        }
        for (int d = 1; d < GGML_MAX_DIMS; ++d) {
            if (tensor->nb[d]/ggml_type_size(tensor->type) > INT_MAX || tensor->ne[d] > INT_MAX ||
                    (tensor != mm && tensor->nb[d] % (2*ggml_type_size(tensor->type)))) {
                return match;
            }
        }
    }
    if (mm->ne[2] > 65535 || mm->ne[3] > 65535 ||
            (ggml_backend_buffer_get_usage(weight->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE && weight->view_src &&
             ggml_nbytes(weight) != ggml_backend_buffer_get_alloc_size(weight->buffer, weight)) ||
            !ggml_cuda_should_use_mmvf(weight->type, ggml_cuda_info().devices[ctx.device].cc, weight->ne, weight->nb, input->ne[1])) {
        return match;
    }
    int u = i + 1;
    if (u < graph->n_nodes && graph->nodes[u]->op == GGML_OP_SCALE && graph->nodes[u]->src[0] == mm) {
        match.pre = graph->nodes[u++];
    }
    if (u >= graph->n_nodes || graph->nodes[u]->op != GGML_OP_UNARY ||
            graph->nodes[u]->src[0] != (match.pre ? match.pre : mm) ||
            (ggml_get_unary_op(graph->nodes[u]) != GGML_UNARY_OP_SILU && ggml_get_unary_op(graph->nodes[u]) != GGML_UNARY_OP_SIGMOID)) {
        return {};
    }
    if (ggml_cuda_can_fuse(graph, u, { GGML_OP_UNARY, GGML_OP_MUL }, { ggml_get_unary_op(graph->nodes[u]) })) {
        return {};
    }
    match.unary = graph->nodes[u];
    int last = u;
    if (u + 1 < graph->n_nodes && graph->nodes[u + 1]->op == GGML_OP_SCALE && graph->nodes[u + 1]->src[0] == match.unary) {
        match.post = graph->nodes[++last];
    }
    const int count = last - i + 1;
    ggml_op ops[4];
    int outputs[4];
    uintptr_t begin[4], end[4], read_begin[2], read_end[2];
    if (!ggml_cuda_mmvf_postop_range(weight, ctx.device, read_begin[0], read_end[0]) ||
            !ggml_cuda_mmvf_postop_range(input, ctx.device, read_begin[1], read_end[1])) {
        return {};
    }
    for (int j = 0; j < count; ++j) {
        const ggml_tensor * tensor = graph->nodes[i + j];
        if (tensor->type != GGML_TYPE_F32 || !(tensor->flags & GGML_TENSOR_FLAG_COMPUTE) || !ggml_is_contiguous(tensor) ||
                !ggml_are_same_shape(mm, tensor) || !ggml_cuda_mmvf_postop_range(tensor, ctx.device, begin[j], end[j]) || begin[j] % sizeof(float)) {
            return {};
        }
        for (int k = 0; k < 2; ++k) {
            if (begin[j] < read_end[k] && read_begin[k] < end[j]) {
                return {};
            }
        }
        for (int k = 0; k < j; ++k) {
            if (begin[j] < end[k] && begin[k] < end[j] && (begin[j] != begin[k] || end[j] != end[k])) {
                return {};
            }
        }
        ops[j] = tensor->op;
        outputs[j] = i + j;
    }
    if (!ggml_can_fuse_subgraph(graph, i, count, ops, outputs, count)) {
        return {};
    }
    match.mm = mm;
    match.count = count;
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
};

static ggml_cuda_hc_up_match ggml_cuda_match_hc_up_shape(ggml_backend_cuda_context & ctx, ggml_cgraph * graph, int i) {
    ggml_tensor * mm = graph->nodes[i];
    if (mm->op != GGML_OP_MUL_MAT || !mm->src[0] || !mm->src[1] || mm->type != GGML_TYPE_F32 ||
            mm->src[1]->type != GGML_TYPE_F32 || mm->ne[2] != 1 || mm->ne[3] != 1 || mm->ne[1] > MMVF_MAX_BATCH_SIZE ||
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
            (weight->type != GGML_TYPE_F32 && weight->type != GGML_TYPE_F16 && weight->type != GGML_TYPE_BF16)) {
        return {};
    }
    if (!ggml_cuda_should_use_mmvf(weight->type, ggml_cuda_info().devices[ctx.device].cc, weight->ne, weight->nb, input->ne[1]) ||
            !ggml_cuda_should_fuse_hc_up(weight, norm, ctx.device)) { return {}; }
    const int count = last - i + 1;
    ggml_op ops[4]; int outputs[4];
    for (int j = 0; j < count; ++j) { ops[j] = graph->nodes[i + j]->op; outputs[j] = i + j; }
    if (!ggml_can_fuse_subgraph(graph, i, count, ops, outputs, count)) { return {}; }
    return {mm, dst, count};
}

static ggml_cuda_hc_up_match ggml_cuda_match_hc_up(ggml_backend_cuda_context & ctx, ggml_cgraph * graph, int i) {
    const auto match = ggml_cuda_match_hc_up_shape(ctx, graph, i);
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
                (end - begin)/ggml_type_size(tensor->type) > INT_MAX || begin % (2*ggml_type_size(tensor->type))) { return {}; }
        for (int d = 1; d < GGML_MAX_DIMS; ++d) {
            if (tensor->ne[d] > INT_MAX || tensor->nb[d]/ggml_type_size(tensor->type) > INT_MAX ||
                    (tensor != mm && tensor->nb[d] % (2*ggml_type_size(tensor->type)))) { return {}; }
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

// try and fuse nodes and return the number of nodes to skip
static int ggml_cuda_try_fuse(
        ggml_backend_cuda_context * cuda_ctx,
        ggml_cgraph * cgraph,
        int i,
        ggml_cuda_moe_graph_execution * execution,
        const ggml_tensor * shared_input = nullptr, const char * quantized = nullptr) {

    static bool disable_fusion = getenv("GGML_CUDA_DISABLE_FUSION") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_FUSION"));
    if (disable_fusion) {
        return 0;
    }

    const auto shared_quantized = [&](const ggml_tensor * mm_node) {
        return quantized && mm_node->src[1] == shared_input && ggml_cuda_can_share_mmvq_input(mm_node, cuda_ctx->device) &&
            !ggml_cuda_mmvq_input_overwritten(mm_node, shared_input) ? quantized : nullptr;
    };
    ggml_tensor * node = cgraph->nodes[i];

    if (node->op == GGML_OP_CPY && ggml_cuda_can_fuse_f32_q8_0_cpy_pair(cuda_ctx, cgraph, i)) {
        ggml_cuda_cpy_f32_q8_0_pair(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

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

    if (node->op == GGML_OP_RMS_NORM) {
        ggml_cuda_rms_norm_gated_match match;
        if (ggml_cuda_match_rms_norm_gated(cgraph, i, match)) {
            const int output_idx = i + match.node_count - 1;
            if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, match.node_count, &output_idx, 1, false, true)) {
                ggml_cuda_op_rms_norm_fused(*cuda_ctx, match.norm, match.mul, match.gate, match.dst, match.gate_op);
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
        const bool grouped_device_ids = execution != nullptr &&
            execution->outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED &&
            execution->dispatch_mode() != GGML_CUDA_MOE_GRAPH_DISPATCH_STAGED;
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
                    const ggml_cuda_moe_ids_publish publish = grouped_device_ids ?
                        ggml_cuda_moe_ids_publish{} : ggml_cuda_moe_prepare_ids_publish(*cuda_ctx, cgraph, i + ops.size(), ids);
                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, clamp, scale, bias, args, publish.ids, publish.ready);
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
                    const ggml_cuda_moe_ids_publish publish = grouped_device_ids ?
                        ggml_cuda_moe_ids_publish{} : ggml_cuda_moe_prepare_ids_publish(*cuda_ctx, cgraph, i + ops.size(), ids);
                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, clamp, scale, bias, args, publish.ids, publish.ready);
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

    ggml_cuda_gdn_packed_match packed_projection;
    if (node->op == GGML_OP_MUL_MAT && cuda_ctx->stream_context().concurrent_events.empty() &&
            ggml_cuda_match_gdn_packed_projection(cgraph, i, cuda_ctx->device, packed_projection, true)) {
        const ggml_cuda_gdn_packed_args_host packed = { packed_projection.bias, packed_projection.scale,
            packed_projection.gate, packed_projection.beta, int(packed_projection.heads_per_group) };
        ggml_cuda_mm_fusion_args_host fusion{};
        fusion.packed = &packed;
        if (ggml_is_quantized(node->src[0]->type)) {
            ggml_cuda_mul_mat_vec_q(*cuda_ctx, node->src[0], node->src[1], nullptr, node, &fusion);
        } else if (packed_projection.use_mmf) {
            ggml_cuda_mul_mat_f(*cuda_ctx, node->src[0], node->src[1], nullptr, node, &packed);
        } else {
            ggml_cuda_mul_mat_vec_f(*cuda_ctx, node->src[0], node->src[1], nullptr, node, &fusion);
        }
        return packed_projection.count - 1;
    }

    ggml_cuda_gdn_projection_match projection;
    bool bf16_rounded = false;
    if (node->op == GGML_OP_MUL_MAT && cuda_ctx->stream_context().concurrent_events.empty() &&
            (ggml_cuda_match_gdn_projections(cgraph, i, cuda_ctx->device, ggml_cuda_gdn_projection::VECTOR, true, &projection) ||
             ggml_cuda_match_gdn_projections(cgraph, i, cuda_ctx->device, ggml_cuda_gdn_projection::MMF, true, &projection) ||
             (bf16_rounded = ggml_cuda_match_gdn_projections(cgraph, i, cuda_ctx->device, ggml_cuda_gdn_projection::BF16_ROUNDED, true, &projection)))) {
        ggml_cuda_mm_fusion_args_host fusion{};
        fusion.gate = projection.beta->src[0];
        fusion.x_bias = projection.bias;
        fusion.x_scale = projection.alpha_scale;
        fusion.gate_scale = projection.beta_scale;
        fusion.post_scale = projection.scale;
        fusion.second_output = projection.beta_output;
        ggml_tensor dst = *node;
        dst.data = projection.gate->data;
        dst.buffer = projection.gate->buffer;
        if (ggml_is_quantized(node->src[0]->type)) {
            ggml_cuda_mul_mat_vec_q(*cuda_ctx, node->src[0], node->src[1], nullptr, &dst, &fusion);
        } else if (projection.use_mmf) {
            ggml_cuda_mul_mat_f(*cuda_ctx, node->src[0], node->src[1], nullptr, &dst, nullptr, &fusion);
        } else {
            ggml_cuda_mul_mat_vec_f(*cuda_ctx, node->src[0], node->src[1], nullptr, &dst, &fusion, bf16_rounded);
        }
        return projection.count - 1;
    }

    const int mmq_pair_count = node->op == GGML_OP_MUL_MAT_ID && cuda_ctx->curr_stream_no == 0 && cuda_ctx->stream_context().concurrent_events.empty()
            ? ggml_cuda_match_mmq_id_pair(cgraph, i, cuda_ctx->device) : 0;
    if (mmq_pair_count != 0) {
        ggml_tensor * glu = cgraph->nodes[i + mmq_pair_count - 1];
        const int output_idx = i + mmq_pair_count - 1;
        if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, mmq_pair_count, &output_idx, 1) &&
                ggml_cuda_mmq_id_pair_memory_ok(glu->src[1], glu->src[0], glu)) {
            ggml_cuda_mul_mat_id_q_pair(*cuda_ctx, glu->src[1], glu->src[0], glu);
            return mmq_pair_count - 1;
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

                ggml_cuda_moe_graph_binding up_binding;
                ggml_cuda_moe_graph_binding gate_binding;
                auto * up_group = execution != nullptr ? execution->find_group(up_n, &up_binding) : nullptr;
                auto * gate_group = execution != nullptr ? execution->find_group(gate_n, &gate_binding) : nullptr;
                const bool grouped_pair = !with_bias && up_group != nullptr && gate_group == up_group &&
                    (up_group->state == GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_ARMED ||
                        up_group->state == GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_ACTIVE);
                const bool grouped_fusion_equivalent = grouped_pair &&
                    ggml_cuda_moe_grouped_b1_scale_fusion(up_group, up_binding) &&
                    ggml_cuda_moe_grouped_b1_scale_fusion(gate_group, gate_binding);

                if (!ggml_cuda_should_fuse_mul_mat(up_n, gate_n, glu,
                        with_bias ? up_out_n : nullptr, with_bias ? gate_out_n : nullptr,
                        up_scale_n, gate_scale_n, grouped_pair)) {
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

                if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n, grouped_pair) &&
                        (!grouped_pair || grouped_fusion_equivalent)) {
                    ggml_cuda_moe_shadow_probe(*cuda_ctx, execution, up_n, src0, ids, &fusion_data);
                    if (grouped_pair && up_group->state == GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_ARMED) {
                        const auto result = cuda_ctx->moe_grouped_context->prepare_graph_group(
                            up_group, gate_binding, gate_n, cuda_ctx->stream());
                        if (result != GGML_CUDA_MOE_GROUPED_DECODE_READY) {
                            return -1;
                        }
                    }
                    if (grouped_pair && up_group->state == GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_ACTIVE) {
                        ggml_tensor up_view;
                        ggml_tensor gate_view;
                        ggml_tensor ids_view;
                        ggml_tensor gate_ids_view;
                        ggml_tensor up_scale_view;
                        ggml_tensor gate_scale_view;
                        if (up_n->src[2] != up_group->key.ids.tensor || gate_n->src[2] != up_group->key.ids.tensor ||
                                up_group->stream != cuda_ctx->stream() ||
                                !ggml_cuda_moe_group_views(*up_group, up_binding, src0, up_view, ids_view) ||
                                !ggml_cuda_moe_group_views(*up_group, gate_binding, gate_n->src[0], gate_view, gate_ids_view) ||
                                ids_view.data != gate_ids_view.data ||
                                !ggml_cuda_moe_group_scale_view(*up_group, up_binding, up_scale, up_scale_view) ||
                                !ggml_cuda_moe_group_scale_view(*up_group, gate_binding, gate_scale, gate_scale_view)) {
                            return -1;
                        }
                        fusion_data.gate = &gate_view;
                        fusion_data.gate_ids = &ids_view;
                        fusion_data.x_scale = &up_scale_view;
                        fusion_data.gate_scale = &gate_scale_view;
                        ggml_cuda_mul_mat_vec_q(*cuda_ctx, &up_view, src1, &ids_view, cgraph->nodes[glu_idx], &fusion_data);
                        fused_mul_mat_vec = true;
                        fused_node_count = n_ops;
                        break;
                    }
                    if (ggml_backend_buft_is_cuda_moe_cached(src0->buffer->buft)) {
                        continue;
                    }
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

        if (ggml_cuda_can_fuse(cgraph, i, { op, bias_op, op, bias_op, GGML_OP_GLU }, {},
                op == GGML_OP_MUL_MAT_ID)) {
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

            ggml_cuda_moe_graph_binding up_binding;
            ggml_cuda_moe_graph_binding gate_binding;
            auto * up_group = op == GGML_OP_MUL_MAT_ID && execution != nullptr ? execution->find_group(up_n, &up_binding) : nullptr;
            auto * gate_group = op == GGML_OP_MUL_MAT_ID && execution != nullptr ? execution->find_group(gate_n, &gate_binding) : nullptr;
            const bool grouped_pair = up_group != nullptr && gate_group == up_group &&
                (up_group->state == GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_ARMED ||
                    up_group->state == GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_ACTIVE) &&
                up_binding.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT &&
                gate_binding.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT &&
                ggml_cuda_moe_grouped_mmvq(up_group, up_binding) && ggml_cuda_moe_grouped_mmvq(up_group, gate_binding);

            if (ggml_cuda_should_fuse_mul_mat_vec_f(up_n)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate_n->src[0];
                fusion_data.x_bias    = up_bias_tensor;
                fusion_data.gate_bias = gate_bias_tensor;
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_moe_shadow_probe(*cuda_ctx, execution, up_n, src0, ids, &fusion_data, op == GGML_OP_MUL_MAT_ID);
                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 5;
                break;
            }

            if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n, grouped_pair)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate_n->src[0];
                fusion_data.x_bias    = up_bias_tensor;
                fusion_data.gate_bias = gate_bias_tensor;
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_moe_shadow_probe(*cuda_ctx, execution, up_n, src0, ids, &fusion_data, op == GGML_OP_MUL_MAT_ID);
                if (grouped_pair && up_group->state == GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_ARMED) {
                    if (up_group->first_reader != gate_n && up_group->first_reader != up_n) {
                        return -1;
                    }
                    const bool gate_first = up_group->first_reader == gate_n;
                    const auto result = cuda_ctx->moe_grouped_context->prepare_graph_group(
                        up_group, gate_first ? gate_binding : up_binding, gate_first ? gate_n : up_n, cuda_ctx->stream());
                    if (result != GGML_CUDA_MOE_GROUPED_DECODE_READY) {
                        return -1;
                    }
                }
                if (grouped_pair && up_group->state == GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_ACTIVE) {
                    ggml_tensor up_view;
                    ggml_tensor gate_view;
                    ggml_tensor ids_view;
                    ggml_tensor gate_ids_view;
                    ggml_tensor up_bias_view;
                    ggml_tensor gate_bias_view;
                    if (up_n->src[2] != up_group->key.ids.tensor || gate_n->src[2] != up_group->key.ids.tensor ||
                            up_group->stream != cuda_ctx->stream() ||
                            !ggml_cuda_moe_group_views(*up_group, up_binding, src0, up_view, ids_view) ||
                            !ggml_cuda_moe_group_views(*up_group, gate_binding, gate_n->src[0], gate_view, gate_ids_view) ||
                            ids_view.data != gate_ids_view.data ||
                            !ggml_cuda_moe_group_bias_view(*up_group, up_binding, up_bias_tensor, up_bias_view) ||
                            !ggml_cuda_moe_group_bias_view(*up_group, gate_binding, gate_bias_tensor, gate_bias_view)) {
                        return -1;
                    }
                    fusion_data.gate = &gate_view;
                    fusion_data.gate_ids = &gate_ids_view;
                    fusion_data.x_bias = &up_bias_view;
                    fusion_data.gate_bias = &gate_bias_view;
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, &up_view, src1, &ids_view, glu, &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count = 5;
                    break;
                }
                if (ggml_backend_buft_is_cuda_moe_cached(src0->buffer->buft)) {
                    continue;
                }
                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data, shared_quantized(up_n));
                fused_mul_mat_vec = true;
                fused_node_count  = 5;
                break;
            }
        } else if (ggml_cuda_can_fuse(cgraph, i, { op, op, GGML_OP_GLU }, {}, op == GGML_OP_MUL_MAT_ID)) {
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

                ggml_cuda_moe_shadow_probe(*cuda_ctx, execution, up, src0, ids, &fusion_data, op == GGML_OP_MUL_MAT_ID);
                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }

            ggml_cuda_moe_graph_binding up_binding;
            ggml_cuda_moe_graph_binding gate_binding;
            auto * up_group = op == GGML_OP_MUL_MAT_ID && execution != nullptr ? execution->find_group(up, &up_binding) : nullptr;
            auto * gate_group = op == GGML_OP_MUL_MAT_ID && execution != nullptr ? execution->find_group(gate, &gate_binding) : nullptr;
            const bool grouped_pair = up_group != nullptr && gate_group == up_group &&
                (up_group->state == GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_ARMED ||
                    up_group->state == GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_ACTIVE);
            const bool grouped_fusion_equivalent = grouped_pair &&
                ggml_cuda_moe_grouped_mmvq(up_group, up_binding) && ggml_cuda_moe_grouped_mmvq(gate_group, gate_binding);
            if (ggml_cuda_should_fuse_mul_mat_vec_q(up, grouped_pair) &&
                    (!grouped_pair || grouped_fusion_equivalent)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_moe_shadow_probe(*cuda_ctx, execution, up, src0, ids, &fusion_data, op == GGML_OP_MUL_MAT_ID);
                if (up_group != nullptr && gate_group == up_group && up_group->state == GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_ARMED) {
                    ggml_tensor * first_reader = cgraph->nodes[i];
                    const auto & first_binding = first_reader == up ? up_binding : gate_binding;
                    const auto result = cuda_ctx->moe_grouped_context->prepare_graph_group(
                        up_group, first_binding, first_reader, cuda_ctx->stream());
                    if (result != GGML_CUDA_MOE_GROUPED_DECODE_READY) {
                        return -1;
                    }
                }
                if (up_group != nullptr && gate_group == up_group && up_group->state == GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_ACTIVE) {
                    ggml_tensor up_view;
                    ggml_tensor gate_view;
                    ggml_tensor ids_view;
                    ggml_tensor gate_ids_view;
                    if (up->src[2] != up_group->key.ids.tensor || gate->src[2] != up_group->key.ids.tensor ||
                            up_group->stream != cuda_ctx->stream() ||
                            !ggml_cuda_moe_group_views(*up_group, up_binding, src0, up_view, ids_view) ||
                            !ggml_cuda_moe_group_views(*up_group, gate_binding, gate->src[0], gate_view, gate_ids_view) ||
                            ids_view.data != gate_ids_view.data) {
                        return -1;
                    }
                    fusion_data.gate = &gate_view;
                    fusion_data.gate_ids = &ids_view;
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, &up_view, src1, &ids_view, glu, &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count = 3;
                    break;
                }
                if (ggml_backend_buft_is_cuda_moe_cached(src0->buffer->buft)) {
                    continue;
                }
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

    const int mmf_pair_count = node->op == GGML_OP_MUL_MAT_ID && cuda_ctx->curr_stream_no == 0 && cuda_ctx->stream_context().concurrent_events.empty()
            ? ggml_cuda_match_mmf_id_pair(cgraph, i, cuda_ctx->device) : 0;
    if (mmf_pair_count != 0 && ggml_cuda_mmf_id_pair_memory_ok(cgraph, i, mmf_pair_count, cuda_ctx->device)) {
        ggml_tensor * glu = cgraph->nodes[i + mmf_pair_count - 1];
        ggml_cuda_mul_mat_id_f_pair(*cuda_ctx, glu->src[1], glu->src[0], glu);
        return mmf_pair_count - 1;
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

            if (op == GGML_OP_MUL_MAT) {
                const ggml_tensor * parameters[] = { scale, bias };
                bool external = true;
                for (const ggml_tensor * parameter : parameters) {
                    for (int j = i; parameter && j < i + n_ops; ++j) {
                        external &= parameter != cgraph->nodes[j];
                    }
                }
                if (!external || !ggml_cuda_gdn_writes_safe(cgraph, i, n_ops, cuda_ctx->device)) {
                    continue;
                }
            }

            const ggml_tensor * src0 = mm_node->src[0];
            const ggml_tensor * src1 = mm_node->src[1];
            const ggml_tensor * ids  = mm_node->src[2];

            ggml_cuda_moe_graph_binding binding;
            auto * group = op == GGML_OP_MUL_MAT_ID && execution != nullptr ? execution->find_group(mm_node, &binding) : nullptr;
            const bool grouped_down = !with_bias && group != nullptr &&
                group->state == GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_ACTIVE &&
                binding.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT &&
                ggml_cuda_moe_grouped_b1_scale_fusion(group, binding);

            ggml_cuda_mm_fusion_args_host fusion_data{};
            fusion_data.x_bias  = bias;
            fusion_data.x_scale = scale;

            if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node, grouped_down)) {
                ggml_cuda_moe_shadow_probe(*cuda_ctx, execution, mm_node, src0, ids, &fusion_data, op == GGML_OP_MUL_MAT_ID);
                if (grouped_down) {
                    ggml_tensor bank_view;
                    ggml_tensor ids_view;
                    ggml_tensor scale_view;
                    if (group->stream != cuda_ctx->stream() || ids != group->key.ids.tensor ||
                            !ggml_cuda_moe_group_views(*group, binding, src0, bank_view, ids_view) ||
                            !ggml_cuda_moe_group_scale_view(*group, binding, scale, scale_view)) {
                        return -1;
                    }
                    fusion_data.x_scale = &scale_view;
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, &bank_view, src1, &ids_view, out_node, &fusion_data);
                    if (!cuda_ctx->moe_grouped_context->finish_graph_group(
                            group, binding, mm_node, cuda_ctx->stream())) {
                        return -1;
                    }
                    fused_mul_mat_vec = true;
                    fused_node_count = n_ops;
                    break;
                }
                if (ggml_backend_buft_is_cuda_moe_cached(src0->buffer->buft)) {
                    continue;
                }
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

        ggml_cuda_moe_graph_binding binding;
        auto * group = op == GGML_OP_MUL_MAT_ID && execution != nullptr ? execution->find_group(mm_node, &binding) : nullptr;
        const bool grouped_down = group != nullptr && group->state == GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_ACTIVE &&
            binding.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT && ggml_cuda_moe_grouped_mmvq(group, binding);

        if (bias_op == GGML_OP_ADD_ID && bias_node->src[2] != ids) {
            continue;
        }

        if (bias_op == GGML_OP_ADD && !ggml_are_same_shape(bias_node->src[0], bias_node->src[1])) {
            continue;
        }

        ggml_cuda_mm_fusion_args_host fusion_data{};
        fusion_data.x_bias = bias_tensor;

        if (ggml_cuda_should_fuse_mul_mat_vec_f(mm_node)) {
            ggml_cuda_moe_shadow_probe(*cuda_ctx, execution, mm_node, src0, ids, &fusion_data, op == GGML_OP_MUL_MAT_ID);
            ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count  = 2;
            break;
        }

        if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node, grouped_down)) {
            ggml_cuda_moe_shadow_probe(*cuda_ctx, execution, mm_node, src0, ids, &fusion_data, op == GGML_OP_MUL_MAT_ID);
            if (grouped_down) {
                ggml_tensor bank_view;
                ggml_tensor ids_view;
                ggml_tensor bias_view;
                if (group->stream != cuda_ctx->stream() || ids != group->key.ids.tensor ||
                        !ggml_cuda_moe_group_views(*group, binding, src0, bank_view, ids_view) ||
                        !ggml_cuda_moe_group_bias_view(*group, binding, bias_tensor, bias_view)) {
                    return -1;
                }
                fusion_data.x_bias = &bias_view;
                ggml_cuda_mul_mat_vec_q(*cuda_ctx, &bank_view, src1, &ids_view, bias_node, &fusion_data);
                if (!cuda_ctx->moe_grouped_context->finish_graph_group(group, binding, mm_node, cuda_ctx->stream())) {
                    return -1;
                }
                fused_mul_mat_vec = true;
                fused_node_count = 2;
                break;
            }
            if (ggml_backend_buft_is_cuda_moe_cached(src0->buffer->buft)) {
                continue;
            }
            ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data, shared_quantized(mm_node));
            fused_mul_mat_vec = true;
            fused_node_count  = 2;
            break;
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
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

    if (node->op == GGML_OP_SSM_CONV && cuda_ctx->stream_context().concurrent_events.empty() &&
            ggml_cuda_match_ssm_conv_qk(cgraph, i, true)) {
        ggml_cuda_op_ssm_conv_qk(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 3],
            cgraph->nodes[i + 4], cgraph->nodes[i + 6], cgraph->nodes[i + 7]);
        return 7;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_ADD, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        ggml_cuda_op_ssm_conv(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        ggml_cuda_op_ssm_conv(*cuda_ctx, node, /*bias_add_node=*/ nullptr, cgraph->nodes[i + 1]);
        return 1;
    }

    if (node->op == GGML_OP_MUL_MAT && cuda_ctx->stream_context().concurrent_events.empty() &&
            ggml_cuda_match_gdn_projections(cgraph, i, cuda_ctx->device, ggml_cuda_gdn_projection::CUBLAS, true)) {
        ggml_tensor * beta = cgraph->nodes[i + 6];
        const ggml_type compute_type = ggml_cuda_mul_mat_cublas_compute_type(ggml_cuda_info().devices[cuda_ctx->device].cc, node->src[0], node->src[1], node);
        if (compute_type == GGML_TYPE_BF16) {
            ggml_cuda_gdn_cublas_pair<GGML_TYPE_BF16>(*cuda_ctx, node, beta);
        } else {
            ggml_cuda_gdn_cublas_pair<GGML_TYPE_F16>(*cuda_ctx, node, beta);
        }
        ggml_cuda_op_gdn_post(*cuda_ctx, cgraph->nodes[i + 2], cgraph->nodes[i + 3], cgraph->nodes[i + 4],
            beta, cgraph->nodes[i + 8]);
        return 8;
    }

    ggml_cuda_gdn_packed_match packed_post;
    if (node->op == GGML_OP_CONT && cuda_ctx->stream_context().concurrent_events.empty() &&
            ggml_cuda_match_gdn_packed(cgraph, i, packed_post, cuda_ctx->device)) {
        ggml_cuda_op_gdn_packed_post(*cuda_ctx, packed_post.packed, packed_post.bias, packed_post.scale,
            packed_post.gate, packed_post.beta, packed_post.heads_per_group);
        return packed_post.count - 1;
    }

    ggml_cuda_gdn_post_match gdn_post;
    if (node->op == GGML_OP_ADD && cuda_ctx->stream_context().concurrent_events.empty() &&
            (ggml_cuda_match_gdn_post(cgraph, i, gdn_post, cuda_ctx->device) ||
             ggml_cuda_match_gdn_post(cgraph, i, gdn_post, cuda_ctx->device, false, true))) {
        GGML_ASSERT(ggml_cuda_compute_forward(*cuda_ctx, gdn_post.producer, execution));
        ggml_tensor * beta_input = gdn_post.producer;
        if (gdn_post.beta_scale) {
            beta_input = cgraph->nodes[i + 5];
            GGML_ASSERT(ggml_cuda_compute_forward(*cuda_ctx, beta_input, execution));
        }
        ggml_cuda_op_gdn_post(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], beta_input, gdn_post.beta);
        return gdn_post.count - 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SILU }) ||
        ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SIGMOID }) ||
        (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SOFTPLUS }) &&
         ggml_cuda_gdn_writes_safe(cgraph, i, 2, cuda_ctx->device))) {
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
        const auto match = ggml_cuda_match_mmvf_postop(*cuda_ctx, cgraph, i);
        if (match.count) {
            ggml_cuda_mul_mat_vec_f_postop(*cuda_ctx, match.mm, match.pre, match.unary, match.post);
            return match.count - 1;
        }
    }

    if (cuda_ctx->curr_stream_no == 0 && cuda_ctx->stream_context().concurrent_events.empty()) {
        const auto match = ggml_cuda_match_hc_up(*cuda_ctx, cgraph, i);
        if (match.count) {
            ggml_cuda_mul_mat_vec_f_hc_pre(*cuda_ctx, match.mm, match.dst);
            return match.count - 1;
        }
    }

    return 0;
}

static int ggml_cuda_moe_source_fusion_end(ggml_cgraph * graph, int first) {
    std::vector<const ggml_tensor *> inputs;
    for (int i = first; i < graph->n_nodes; ++i) {
        const auto * node = graph->nodes[i];
        for (const auto * src : node->src) {
            if (src && !ggml_is_empty(src) && src->op == GGML_OP_NONE &&
                    src->buffer && src->data &&
                    (ggml_backend_buffer_get_usage(src->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
                     (src->flags & GGML_TENSOR_FLAG_PARAM)) &&
                    std::find(inputs.begin(), inputs.end(), src) == inputs.end()) { inputs.push_back(src); }
        }
        if (ggml_cuda_is_view_or_noop(node) || !(node->flags & GGML_TENSOR_FLAG_COMPUTE)) { continue; }
        const auto dst = reinterpret_cast<uintptr_t>(node->data);
        const auto dst_bytes = ggml_backend_buffer_get_alloc_size(node->buffer, node);
        for (const auto * src : inputs) {
            const auto input = reinterpret_cast<uintptr_t>(src->data);
            const auto bytes = ggml_backend_buffer_get_alloc_size(src->buffer, src);
            if (dst_bytes > UINTPTR_MAX - dst || bytes > UINTPTR_MAX - input ||
                    (dst < input + bytes && input < dst + dst_bytes)) { return std::max(first + 1, i); }
        }
    }
    return graph->n_nodes;
}

static bool ggml_cuda_compute_graph_nodes(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph,
    ggml_cuda_moe_graph_execution * moe_execution, bool use_cuda_graph, bool cuda_graph_update_required,
    uint64_t * fused_nodes, size_t * scratch_bytes, uint64_t * image_groups, uint64_t * emitted_images);

bool ggml_cuda_moe_source_image_resources(ggml_backend_cuda_context & context, ggml_cgraph * graph,
        int first, int end, size_t & bytes) {
    if (!graph || first < 0 || first > end || end > graph->n_nodes) { return false; }
    auto range = ggml_graph_view(graph, first, end);
    uint64_t ignored = 0;
    return ggml_cuda_compute_graph_nodes(&context, &range, nullptr, false, false, &ignored, &bytes, nullptr, nullptr);
}

bool ggml_cuda_moe_source_compute_range(ggml_backend_cuda_context & context, ggml_cgraph * graph,
        int first, int end, uint64_t & fused_nodes, uint64_t & image_groups, uint64_t & emitted_images) {
    if (!graph || first < 0 || first > end || end > graph->n_nodes) { return false; }
    auto range = ggml_graph_view(graph, first, end);
    return ggml_cuda_compute_graph_nodes(&context, &range, nullptr, false, false,
        &fused_nodes, nullptr, &image_groups, &emitted_images);
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
                if (ggml_cuda_mmvq_input_overwritten(graph->nodes[j], input) || ggml_cuda_prepared_input_overwritten(graph->nodes[j], input)) { return false; }
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

struct ggml_cuda_moe_hybrid_bucket {
    std::vector<ggml_tensor> tensors;
    std::vector<ggml_tensor *> nodes;
};

#if defined(USE_CUDA_GRAPH) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 12030
#define GGML_CUDA_MOE_HYBRID_WINDOW

static bool ggml_cuda_moe_hybrid_body_supported(cudaGraph_t graph) {
    size_t count = 0;
    if (cudaGraphGetNodes(graph, nullptr, &count) != cudaSuccess) {
        return false;
    }
    std::vector<cudaGraphNode_t> nodes(count);
    if (cudaGraphGetNodes(graph, nodes.data(), &count) != cudaSuccess) {
        return false;
    }
    for (auto node : nodes) {
        cudaGraphNodeType type;
        if (cudaGraphNodeGetType(node, &type) != cudaSuccess) {
            return false;
        }
        switch (type) {
            case cudaGraphNodeTypeKernel:
            case cudaGraphNodeTypeMemcpy:
            case cudaGraphNodeTypeMemset:
            case cudaGraphNodeTypeEmpty:
                break;
            case cudaGraphNodeTypeGraph: {
                cudaGraph_t child = nullptr;
                if (cudaGraphChildGraphNodeGetGraph(node, &child) != cudaSuccess ||
                        !ggml_cuda_moe_hybrid_body_supported(child)) {
                    return false;
                }
                break;
            }
            default:
                fprintf(stderr, "moe-hybrid-window: unsupported body node type=%u\n", unsigned(type));
                return false;
        }
    }
    return true;
}

static cudaError_t ggml_cuda_moe_hybrid_add_if(cudaGraph_t graph, cudaGraphConditionalHandle handle,
        const std::vector<cudaGraphNode_t> & dependencies, cudaGraphNode_t & node, cudaGraph_t & body) {
    cudaGraphNodeParams params = {};
    params.type = cudaGraphNodeTypeConditional;
    params.conditional.handle = handle;
    params.conditional.type = cudaGraphCondTypeIf;
    params.conditional.size = 1;
#if CUDART_VERSION >= 13000
    const auto status = cudaGraphAddNode(&node, graph, dependencies.data(), nullptr, dependencies.size(), &params);
#else
    const auto status = cudaGraphAddNode(&node, graph, dependencies.data(), dependencies.size(), &params);
#endif
    if (status == cudaSuccess) {
        body = params.conditional.phGraph_out[0];
    }
    return status;
}

template<typename F>
static cudaError_t ggml_cuda_moe_hybrid_capture(cudaGraph_t graph, cudaStream_t stream,
        std::vector<cudaGraphNode_t> & dependencies, F emit) {
    auto status = cudaStreamBeginCaptureToGraph(stream, graph, dependencies.data(), nullptr,
        dependencies.size(), cudaStreamCaptureModeRelaxed);
    if (status != cudaSuccess) {
        return status;
    }
    cudaStreamCaptureStatus capture_status;
    const cudaGraphNode_t * frontier = nullptr;
    size_t count = 0;
    try {
        status = emit();
        if (status == cudaSuccess) {
#if CUDART_VERSION >= 13000
            status = cudaStreamGetCaptureInfo(stream, &capture_status, nullptr, nullptr, &frontier, nullptr, &count);
#else
            status = cudaStreamGetCaptureInfo(stream, &capture_status, nullptr, nullptr, &frontier, &count);
#endif
            if (status == cudaSuccess) {
                dependencies.clear();
                if (count != 0) {
                    dependencies.assign(frontier, frontier + count);
                }
            }
        }
    } catch (...) {
        cudaGraph_t captured = nullptr;
        (void) cudaStreamEndCapture(stream, &captured);
        throw;
    }
    cudaGraph_t captured = nullptr;
    const auto ended = cudaStreamEndCapture(stream, &captured);
    return status != cudaSuccess ? status : ended != cudaSuccess ? ended :
        captured == graph ? cudaSuccess : cudaErrorInvalidValue;
}

cudaError_t ggml_cuda_moe_fidelity_add_if(cudaGraph_t graph, cudaGraphConditionalHandle handle,
        const std::vector<cudaGraphNode_t> & dependencies, cudaGraphNode_t & node, cudaGraph_t & body) {
    return ggml_cuda_moe_hybrid_add_if(graph, handle, dependencies, node, body);
}

cudaError_t ggml_cuda_moe_fidelity_capture_to_graph(cudaGraph_t graph, cudaStream_t stream,
        std::vector<cudaGraphNode_t> & dependencies, cudaError_t (*emit)(void *), void * opaque) {
    if (!emit) { return cudaErrorInvalidValue; }
    return ggml_cuda_moe_hybrid_capture(graph, stream, dependencies, [&] { return emit(opaque); });
}

bool ggml_cuda_moe_fidelity_body_supported(cudaGraph_t graph) {
    return ggml_cuda_moe_hybrid_body_supported(graph);
}

struct ggml_cuda_moe_hybrid_probe_record {
    uint32_t epoch;
    uint32_t fail;
    uint32_t failed;
    uint32_t accepted;
    uint32_t output[2];
};

static __global__ void ggml_cuda_moe_hybrid_probe_decide(ggml_cuda_moe_hybrid_probe_record * record,
        cudaGraphConditionalHandle handle) {
    const bool accepted = record->accepted == record->epoch && !record->failed;
    record->failed |= !accepted;
    cudaGraphSetConditional(handle, accepted);
}

static void CUDART_CB ggml_cuda_moe_hybrid_probe_host(void * opaque) {
    auto & record = *static_cast<ggml_cuda_moe_hybrid_probe_record *>(opaque);
    record.accepted = !record.failed && !record.fail ? record.epoch : 0;
    record.fail = 0;
}
#endif

static int32_t ggml_cuda_moe_hybrid_window_probe(ggml_backend_t backend, uint32_t replays) {
#ifndef GGML_CUDA_MOE_HYBRID_WINDOW
    GGML_UNUSED(backend);
    GGML_UNUSED(replays);
    return 0;
#else
    if (backend == nullptr || !ggml_backend_is_cuda(backend) || replays == 0) {
        return -1;
    }
    auto & ctx = *static_cast<ggml_backend_cuda_context *>(backend->context);
    ggml_cuda_set_device(ctx.device);
    struct resources {
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t instance = nullptr;
        cudaStream_t stream = nullptr;
        ggml_cuda_moe_hybrid_probe_record * host = nullptr;
        ggml_cuda_moe_hybrid_probe_record * device = nullptr;
        ~resources() {
            if (stream) { CUDA_CHECK(cudaStreamSynchronize(stream)); }
            if (instance) { CUDA_CHECK(cudaGraphExecDestroy(instance)); }
            if (graph) { CUDA_CHECK(cudaGraphDestroy(graph)); }
            if (stream) { CUDA_CHECK(cudaStreamDestroy(stream)); }
            if (device) { CUDA_CHECK(cudaFree(device)); }
            if (host) { CUDA_CHECK(cudaFreeHost(host)); }
        }
    } probe;
    if (cudaStreamCreateWithFlags(&probe.stream, cudaStreamNonBlocking) != cudaSuccess ||
            cudaMallocHost(&probe.host, sizeof(*probe.host)) != cudaSuccess ||
            cudaMalloc(&probe.device, sizeof(*probe.device)) != cudaSuccess ||
            cudaGraphCreate(&probe.graph, 0) != cudaSuccess) {
        return -1;
    }
    std::vector<cudaGraphNode_t> dependencies;
    for (uint32_t r = 0; r < 2; ++r) {
        cudaGraphConditionalHandle handle;
        auto status = cudaGraphConditionalHandleCreate(&handle, probe.graph, 0, cudaGraphCondAssignDefault);
        if (status != cudaSuccess) {
            (void) cudaGetLastError();
            return status == cudaErrorNotSupported ? 0 : -1;
        }
        status = ggml_cuda_moe_hybrid_capture(probe.graph, probe.stream, dependencies, [&] {
            auto result = r == 0 ? cudaMemcpyAsync(probe.device, probe.host, sizeof(*probe.host),
                cudaMemcpyHostToDevice, probe.stream) : cudaSuccess;
            if (result == cudaSuccess) {
                result = cudaMemcpyAsync(probe.host, probe.device, sizeof(*probe.host), cudaMemcpyDeviceToHost, probe.stream);
            }
            if (result == cudaSuccess) {
                result = cudaLaunchHostFunc(probe.stream, ggml_cuda_moe_hybrid_probe_host, probe.host);
            }
            if (result == cudaSuccess) {
                result = cudaMemcpyAsync(probe.device, probe.host, sizeof(*probe.host), cudaMemcpyHostToDevice, probe.stream);
            }
            if (result == cudaSuccess) {
                ggml_cuda_moe_hybrid_probe_decide<<<1, 1, 0, probe.stream>>>(probe.device, handle);
                result = cudaGetLastError();
            }
            return result;
        });
        cudaGraphNode_t node = nullptr;
        cudaGraph_t body = nullptr;
        if (status != cudaSuccess ||
                ggml_cuda_moe_hybrid_add_if(probe.graph, handle, dependencies, node, body) != cudaSuccess) {
            return -1;
        }
        std::vector<cudaGraphNode_t> body_dependencies;
        if (ggml_cuda_moe_hybrid_capture(body, probe.stream, body_dependencies, [&] {
                return cudaMemcpyAsync(&probe.device->output[r], &probe.device->epoch, sizeof(uint32_t),
                    cudaMemcpyDeviceToDevice, probe.stream);
            }) != cudaSuccess || !ggml_cuda_moe_hybrid_body_supported(body)) {
            return -1;
        }
        dependencies.assign(1, node);
    }
    if (ggml_cuda_moe_hybrid_capture(probe.graph, probe.stream, dependencies, [&] {
            return cudaMemcpyAsync(probe.host, probe.device, sizeof(*probe.host), cudaMemcpyDeviceToHost, probe.stream);
        }) != cudaSuccess || cudaGraphInstantiate(&probe.instance, probe.graph, nullptr, nullptr, 0) != cudaSuccess) {
        return -1;
    }
    for (uint32_t iteration = 0; iteration < replays; ++iteration) {
        const uint32_t replay = iteration + 1;
        const bool fail = replay % 2 == 0;
        *probe.host = {replay, uint32_t(fail), 0, 0, {0, 0}};
        if (cudaGraphLaunch(probe.instance, probe.stream) != cudaSuccess || cudaStreamSynchronize(probe.stream) != cudaSuccess ||
                probe.host->failed != uint32_t(fail) || probe.host->output[0] != (fail ? 0 : replay) ||
                probe.host->output[1] != (fail ? 0 : replay)) {
            return -1;
        }
    }
    return 1;
#endif
}

static constexpr size_t ggml_cuda_moe_hybrid_align(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

static bool ggml_cuda_moe_hybrid_align_checked(size_t value, size_t alignment, size_t & result) {
    if (value > SIZE_MAX - (alignment - 1)) {
        return false;
    }
    result = ggml_cuda_moe_hybrid_align(value, alignment);
    return true;
}

struct ggml_cuda_norm_mmq_match {
    ggml_tensor * norm = nullptr;
    ggml_tensor * mul = nullptr;
    ggml_tensor * add = nullptr;
    ggml_tensor * scale = nullptr;
    ggml_tensor * dst = nullptr;
    int last = -1;
    int image = -1;
};

static ggml_cuda_norm_mmq_match ggml_cuda_match_norm_mmq(ggml_cgraph * graph, int i, int device) {
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
            if (keys[node] < 0 || keys[node] > 3 ||
                    norm_group_invalid(node) || prepare <= match.last || root != match.dst || !ggml_is_contiguous(input) || input->data != match.dst->data ||
                    input->type != GGML_TYPE_F32 || input->ne[0] % QK8_1 || ggml_nbytes(input) != ggml_nbytes(match.dst) ||
                    !ggml_cuda_prepared_range(input, device, begin, end)) { return false; }
            for (int j = match.last + 1; j <= prepare; ++j) {
                if (ggml_cuda_mmq_input_overwritten(graph->nodes[j], input) || ggml_cuda_prepared_input_overwritten(graph->nodes[j], input)) { return false; }
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

static constexpr size_t GGML_CUDA_MOE_HYBRID_ARENA_ALIGNMENT = 256;
static_assert((GGML_CUDA_MOE_HYBRID_ARENA_ALIGNMENT & (GGML_CUDA_MOE_HYBRID_ARENA_ALIGNMENT - 1)) == 0);

struct ggml_cuda_moe_hybrid_control_layout {
    size_t slots_offset;
    size_t resident_slots_offset;
    size_t scalar_ids_offset;
    size_t transfer_ids_offset;
    size_t consumer_status_offset;
    size_t bytes;
};

static bool ggml_cuda_moe_hybrid_measure_control(uint32_t routes, ggml_cuda_moe_hybrid_control_layout & layout) {
    const size_t header = ggml_cuda_moe_hybrid_align(sizeof(ggml_cuda_moe_hybrid_selection), alignof(int32_t));
    if (routes > (SIZE_MAX - header - 2 * sizeof(uint32_t)) / (GGML_CUDA_MOE_HYBRID_PACKET_ARRAYS * sizeof(int32_t))) {
        return false;
    }
    layout.slots_offset = header;
    layout.resident_slots_offset = header + size_t(routes) * sizeof(int32_t);
    layout.transfer_ids_offset = header + size_t(routes) * 3 * sizeof(int32_t);
    layout.scalar_ids_offset = header + size_t(routes) * GGML_CUDA_MOE_HYBRID_PACKET_ARRAYS * sizeof(int32_t);
    layout.consumer_status_offset = layout.scalar_ids_offset + sizeof(int32_t);
    return ggml_cuda_moe_hybrid_align_checked(
        layout.consumer_status_offset + sizeof(uint32_t), GGML_CUDA_MOE_HYBRID_ARENA_ALIGNMENT, layout.bytes);
}

struct ggml_cuda_moe_hybrid_region {
    struct copy {
        ggml_cuda_moe_hybrid_region * region;
        uint32_t bank;
        uint32_t row;
        uint32_t tile;
        size_t offset;
        size_t bytes;
    };
    ggml_cuda_moe_hybrid_session * session = nullptr;
    ggml_cuda_moe_graph_group_dispatch * group = nullptr;
    ggml_cuda_moe_hybrid_runtime * host_runtime = nullptr;
    ggml_cuda_moe_hybrid_runtime * device_runtime = nullptr;
    size_t admission_stride = 0;
    uint64_t admission_auxiliary_bytes = 0;
    uint64_t packet_epoch = 0;
    int64_t readback_started = 0;
    std::atomic<uint32_t> packet_status{0};
    std::atomic<uint32_t> packet_admissions{0};
    std::vector<copy> copies;
    ggml_backend_moe_hybrid_region_v1 descriptor = {};
    const ggml_backend_moe_cpu_region_service_api_v1 * cpu_api = nullptr;
    ggml_backend_moe_cpu_service_v1_t cpu_service = nullptr;
    std::vector<ggml_backend_moe_cpu_prepared_region_v1_t> cpu_regions;
    std::vector<uint64_t> graph_uids;
    std::vector<ggml_backend_moe_cpu_prepared_region_v1_t> cpu_batch_regions;
    std::vector<uint64_t> batch_graph_uids;
    uint64_t graph_generation = 0;
    uint64_t source_generation = 0;
    std::vector<ggml_tensor> tensors;
    std::vector<ggml_tensor *> nodes;
    std::vector<ggml_backend_moe_cpu_region_source_v1> sources;
    std::vector<ggml_cuda_moe_grouped_bank_descriptor> source_banks;
    std::vector<uint32_t> bank_indices;
    std::vector<size_t> source_offsets;
    std::vector<ggml_cuda_moe_hybrid_bucket> resident_buckets;
    ggml_cuda_moe_candidate_group_key resident_key;
    uint32_t resident_slots = 0;
    bool combined_gpu = false;
    bool direct_gather = false;
    bool fused_experts = false;
    uint32_t transfer_capacity = 0;
    bool multi_rows = false;
    ggml_cuda_moe_hybrid_rows_layout rows_layout;
    size_t rows_offset = 0;
    size_t host_rows_offset = 0;
    uint32_t n_dynamic = 0;
    uint32_t expert_count = 0;
    size_t raw_offset = 0;
    size_t cpu_offset = 0;
    size_t host_slots_offset = 0;
    size_t host_ids_offset = 0;
    size_t host_input_offset = 0;
    size_t host_raw_offset = 0;
    size_t host_staging_offset = 0;
    size_t host_staging_bytes = 0;
    std::vector<int32_t> distinct_ids;
    std::vector<int32_t> slots;
    std::vector<int32_t> cpu_ids;
    std::vector<uint32_t> classes;
    std::vector<uint32_t> first_ranks;
    std::vector<uint32_t> fanout;
    std::vector<uint32_t> cpu_ranks;
    std::vector<uint32_t> cpu_rows;
    std::vector<uint32_t> resident_rows;
    std::vector<uint32_t> transfer_rows;
};

struct ggml_cuda_moe_hybrid_pool : ggml_cuda_pool {
    uint8_t * data;
    size_t capacity;
    size_t used = 0;
    size_t peak = 0;

    ggml_cuda_moe_hybrid_pool(uint8_t * data, size_t capacity) : data(data), capacity(capacity) {}

    void * alloc(size_t bytes, size_t * actual) override {
        if (bytes > SIZE_MAX - (GGML_CUDA_MOE_HYBRID_ARENA_ALIGNMENT - 1) ||
                ggml_cuda_moe_hybrid_align(bytes, GGML_CUDA_MOE_HYBRID_ARENA_ALIGNMENT) > capacity - used) {
            throw std::bad_alloc();
        }
        *actual = ggml_cuda_moe_hybrid_align(bytes, GGML_CUDA_MOE_HYBRID_ARENA_ALIGNMENT);
        void * result = data + used;
        used += *actual;
        peak = std::max(peak, used);
        return result;
    }

    void free(void * ptr, size_t bytes) override {
        GGML_ASSERT(bytes <= used && ptr == data + used - bytes);
        used -= bytes;
    }
};

struct ggml_cuda_moe_hybrid_session {
    ggml_backend_t backend = nullptr;
    ggml_backend_cuda_context * ctx = nullptr;
    std::unique_ptr<ggml_backend_cuda_context> compute;
    ggml_backend_buffer_t arena = nullptr;
    ggml_backend_buffer_t control = nullptr;
    uint8_t * host = nullptr;
    uint8_t * host_alias = nullptr;
    size_t device_limit = 0;
    size_t pinned_limit = 0;
    size_t arena_required = 0;
    size_t control_bytes = 0;
    uint32_t route_capacity = 0;
    size_t device_bytes = 0;
    size_t work_bytes = 0;
    size_t pinned_bytes = 0;
    cudaEvent_t tile_done[2] = {};
    cudaEvent_t transfer_ready = nullptr;
    cudaEvent_t cpu_ready = nullptr;
    cudaEvent_t gpu_started = nullptr;
    cudaEvent_t gpu_done = nullptr;
    cudaEvent_t plan_ready = nullptr;
    cudaEvent_t producer_done = nullptr;
    cudaStream_t io_stream = nullptr;
    bool tile_pending[2] = {};
    uint32_t next_tile = 0;
    bool fail_before_publish = false;
    bool diagnostic = false;
    uint32_t gpu_miss_quota = 0;
    uint32_t admission_quota = 0;
    bool demand_admission = false;
    bool resident_batch = false;
    bool test_no_host_alias = false;
    bool window_requested = false;
    bool window_cpu_pipeline = true;
    bool window_combine_gpu = true;
    bool window_direct_gather = false;
    bool window_fuse_experts = false;
    bool window_active = false;
    bool window_unavailable = false;
    uint32_t * window_failed = nullptr;
    uint32_t * window_host_failed = nullptr;
    ggml_cuda_moe_hybrid_runtime * window_host = nullptr;
    ggml_cuda_moe_hybrid_runtime * window_device = nullptr;
    uint32_t window_capacity = 0;
    size_t window_bytes = 0;
    void * window_input = nullptr;
    size_t window_input_bytes = 0;
    uint64_t window_identity = 0;
#ifdef GGML_CUDA_MOE_HYBRID_WINDOW
    std::unique_ptr<ggml_cuda_graph> window_graph;
#endif
    std::atomic<uint64_t> window_launches{0}, window_captures{0}, window_waits{0}, window_fallbacks{0};
    std::atomic<uint64_t> window_fused_nodes{0};
    std::atomic<uint64_t> window_combined_regions{0};
    std::atomic<uint64_t> window_direct_regions{0}, window_compact_select_regions{0};
    std::atomic<uint64_t> window_fused_expert_bodies{0};
    void * const * regions = nullptr;
    uint32_t n_regions = 0;
    uint32_t completed = 0;
    uint64_t epoch = 0;
    uint64_t packet_epoch = 0;
    std::atomic<uint64_t> packet_regions{0}, producer_fences{0}, fixed_copy_bytes{0}, empty_copy_bytes{0};
    std::atomic<uint64_t> producer_events{0}, producer_drain_events{0};
    std::atomic<uint64_t> resident_routes{0}, transferred_routes{0}, cpu_routes{0}, h2d_bytes{0};
    std::atomic<uint64_t> distinct_experts{0}, resident_experts{0}, transfer_experts{0}, cpu_experts{0}, cpu_jobs{0};
    std::atomic<uint64_t> cpu_row_jobs{0};
    std::atomic<uint64_t> cpu_execute_calls{0};
    std::atomic<uint64_t> cpu_upload_bytes{0}, cpu_us{0}, gpu_enqueue_us{0}, join_us{0};
    std::atomic<uint64_t> errors{0}, capacity_errors{0}, cancellations{0}, work_peak{0};
    std::atomic<uint64_t> submit_to_start_us{0}, gpu_branch_us{0};
    std::atomic<uint64_t> admission_reserved{0}, admission_committed{0}, admission_replacements{0};
    std::atomic<uint64_t> admission_no_slot{0}, admission_bytes{0}, admission_aborted{0};
    std::atomic<uint64_t> resident_batches{0}, resident_body_submissions{0}, gpu_body_submissions{0};
    std::atomic<uint64_t> readback_us{0}, publication_us{0}, admission_fence_us{0}, prepared_device_bytes{0};
    std::atomic<uint64_t> last_submit_us{0}, last_cpu_start_us{0}, last_cpu_done_us{0};
    std::atomic<uint64_t> last_gpu_enqueue_done_us{0}, last_gpu_done_observed_us{0};
    std::atomic<uint64_t> last_join_start_us{0}, last_join_done_us{0};
    std::mutex mutex;
    std::condition_variable ready;
    std::thread coordinator;
    bool stopping = false;
    bool quiescing = false;
    bool dispatch_active = false;
    std::atomic<bool> canceled{false};
    const ggml_backend_moe_cpu_region_service_api_v1 * cpu_api = nullptr;
    ggml_backend_moe_cpu_service_v1_t cpu_service = nullptr;
    std::vector<ggml_cuda_moe_hybrid_region *> prepared_regions;
    ggml_backend_moe_hybrid_test_hook_v1_t test_hook = nullptr;
    void * test_hook_data = nullptr;
    struct {
        uint32_t state = 0; // idle, queued, running, done
        int32_t status = GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
        ggml_backend_moe_cpu_prepared_region_v1_t region = 0;
        ggml_cuda_moe_hybrid_region * rows_region = nullptr;
        ggml_backend_moe_cpu_region_binding_v1 binding = {};
        ggml_backend_moe_cpu_dynamic_input_v1 inputs[2] = {};
        ggml_backend_moe_cpu_output_v1 output = {};
        ggml_backend_moe_cpu_execute_v1 work = {};
        ggml_backend_moe_cpu_execute_result_v1 result = {};
    } ticket;

    bool hook(uint32_t phase, void * stream = nullptr) noexcept {
        try {
            if (test_hook == nullptr) {
                return true;
            }
            uint64_t epoch;
            {
                std::lock_guard<std::mutex> lock(mutex);
                epoch = ticket.work.epoch;
            }
            return test_hook(test_hook_data, phase, epoch, stream);
        } catch (...) {
            return false;
        }
    }

    static void cpu_hook(void * opaque, uint32_t phase) {
        auto & session = *static_cast<ggml_cuda_moe_hybrid_session *>(opaque);
        const uint32_t hybrid_phase = phase == GGML_BACKEND_MOE_CPU_TEST_PHASE_V1_ADMITTED ?
            GGML_BACKEND_MOE_HYBRID_TEST_CPU_ADMITTED : GGML_BACKEND_MOE_HYBRID_TEST_CPU_BEFORE_COMMIT;
        if (phase != GGML_BACKEND_MOE_CPU_TEST_PHASE_V1_AFTER_COMMIT && !session.hook(hybrid_phase)) {
            session.cpu_api->cancel(session.cpu_service, session.ticket.work.epoch);
        }
    }

    void run() noexcept {
        std::unique_lock<std::mutex> lock(mutex);
        for (;;) {
            ready.wait(lock, [&] { return stopping || ticket.state == 1; });
            if (stopping) {
                return;
            }
            lock.unlock();
            int32_t status = GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CANCELED;
            const bool admitted = hook(GGML_BACKEND_MOE_HYBRID_TEST_CPU_QUEUED);
            lock.lock();
            ticket.state = 2;
            lock.unlock();
            const int64_t started = ggml_time_us();
            last_cpu_start_us.store(started);
            submit_to_start_us += started - last_submit_us.load();
            if (ticket.rows_region != nullptr) {
                auto & region = *ticket.rows_region;
                const auto packet = ggml_cuda_moe_hybrid_rows_packet(host + region.host_rows_offset, region.rows_layout.geometry);
                status = admitted && !canceled.load() ? GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK :
                    GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CANCELED;
                if (status == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK && !canceled.load()) {
                    try {
                        status = cpu_api->execute(cpu_service, ticket.region, &ticket.work, &ticket.result);
                        ++cpu_execute_calls;
                    } catch (...) {
                        status = GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_COMPUTE_FAILED;
                    }
                    cpu_row_jobs += packet.ticket->active_rows;
                    if (status == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK &&
                            (ticket.result.flags != GGML_BACKEND_MOE_CPU_EXECUTE_RESULT_FLAG_V1_PUBLISHED ||
                             ticket.result.epoch != ticket.work.epoch ||
                             ticket.result.published_routes != packet.ticket->cpu_routes ||
                             ticket.result.published_outputs != 1)) {
                        status = GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_COMPUTE_FAILED;
                    }
                } else if (status == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK) {
                    status = GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CANCELED;
                }
                for (uint32_t row = 0; row < packet.ticket->active_rows; ++row) {
                    const uint32_t count = packet.expected_routes[2 + row];
                    if (count == 0) { continue; }
                    packet.producers[2 + row] = {ticket.work.epoch,
                        status == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK ? count : 0, status, 1};
                }
                if (status != GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK && diagnostic) {
                    fprintf(stderr, "moe-hybrid-rows: CPU batch rows=%u routes=%u epoch=%llu status=%d result=%u/%u\n",
                        packet.ticket->active_rows, packet.ticket->cpu_routes, (unsigned long long) ticket.work.epoch, status,
                        ticket.result.published_routes, ticket.result.published_outputs);
                }
            } else if (admitted && !canceled.load()) {
                try {
                    status = cpu_api->execute(cpu_service, ticket.region, &ticket.work, &ticket.result);
                    ++cpu_execute_calls;
                } catch (...) {
                    status = GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_COMPUTE_FAILED;
                }
            }
            const int64_t done = ggml_time_us();
            last_cpu_done_us.store(done);
            cpu_us += done - started;
            if (status == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY) {
                ++capacity_errors;
            }
            lock.lock();
            ticket.status = status;
            ticket.state = 3;
            ready.notify_all();
        }
    }

    bool submit(ggml_cuda_moe_hybrid_region & region, const int32_t * ids, const uint32_t * ranks, uint32_t count) {
        std::lock_guard<std::mutex> lock(mutex);
        if (quiescing || canceled.load() || ticket.state != 0 || epoch == UINT64_MAX || count == 0 || count > region.descriptor.n_cpu_queries) {
            return false;
        }
        ticket.binding = {};
        ticket.rows_region = nullptr;
        ticket.inputs[0] = {};
        ticket.inputs[1] = {};
        ticket.output = {};
        ticket.work = {};
        ticket.result = {};
        ticket.region = region.cpu_regions[count - 1];
        std::fill_n(region.cpu_rows.data(), count, 0);
        ticket.binding = {sizeof(ticket.binding), 1, count, ids, region.cpu_rows.data(), ranks};
        ticket.inputs[0] = {host + region.host_input_offset, ggml_nbytes(&region.tensors[0]), region.tensors[0].nb[2]};
        ticket.output = {host + region.host_raw_offset, ggml_nbytes(region.descriptor.output), region.descriptor.output->nb[1]};
        ticket.work.struct_size = sizeof(ticket.work);
        ticket.work.epoch = ++epoch;
        ticket.work.graph_uid = region.graph_uids[count - 1];
        ticket.work.graph_generation = region.graph_generation;
        ticket.work.source_generation = region.source_generation;
        ticket.work.binding = &ticket.binding;
        ticket.work.dynamic_inputs = ticket.inputs;
        ticket.work.n_dynamic_inputs = 2;
        ticket.work.n_outputs = 1;
        ticket.work.outputs = &ticket.output;
        ticket.result.struct_size = sizeof(ticket.result);
        last_submit_us.store(ggml_time_us());
        ticket.state = 1;
        ++cpu_jobs;
        ready.notify_all();
        return true;
    }

    bool submit_rows(ggml_cuda_moe_hybrid_region & region) {
        std::lock_guard<std::mutex> lock(mutex);
        const auto & geometry = region.rows_layout.geometry;
        const auto packet = ggml_cuda_moe_hybrid_rows_packet(host + region.host_rows_offset, geometry);
        const auto & layer = *packet.ticket;
        if (quiescing || canceled.load() || ticket.state != 0 || !region.multi_rows || layer.status != 0 ||
                layer.epoch != region.packet_epoch || layer.identity != region.rows_layout.identity ||
                layer.active_rows == 0 || layer.active_rows > geometry.row_capacity ||
                layer.route_count != uint64_t(layer.active_rows) * geometry.routes_per_row ||
                layer.weight_count > geometry.weight_capacity || region.cpu_ids.size() < geometry.route_capacity ||
                region.cpu_rows.size() < geometry.route_capacity || region.cpu_ranks.size() < geometry.route_capacity) {
            return false;
        }
        uint32_t routes = 0;
        for (uint32_t row = 0; row < layer.active_rows; ++row) {
            const uint32_t count = packet.expected_routes[2 + row];
            if (count > geometry.routes_per_row) { return false; }
            for (uint32_t i = 0; i < count; ++i) {
                const size_t offset = size_t(row) * geometry.routes_per_row + i;
                const uint32_t route = packet.cpu_routes[offset];
                if (route >= layer.route_count) { return false; }
                const auto & record = packet.routes[route];
                if (record.source_row != row || record.source_route != route % geometry.routes_per_row ||
                        record.scatter_destination != route || record.weight_index >= layer.weight_count ||
                        packet.weight_classes[record.weight_index] != GGML_CUDA_MOE_HYBRID_CPU) { return false; }
                const int32_t expert = packet.weight_experts[record.weight_index];
                if (expert < 0 || uint32_t(expert) >= geometry.expert_count) { return false; }
                region.cpu_ids[routes] = expert;
                region.cpu_rows[routes] = row;
                region.cpu_ranks[routes] = route;
                ++routes;
            }
        }
        if (routes != layer.cpu_routes) { return false; }
        if (routes == 0) { return true; }
        if (region.cpu_batch_regions.size() != 1 || region.batch_graph_uids.size() != 1) { return false; }
        ticket.rows_region = &region;
        ticket.region = region.cpu_batch_regions[0];
        ticket.binding = {sizeof(ticket.binding), routes, routes, region.cpu_ids.data(),
            region.cpu_rows.data(), region.cpu_ranks.data()};
        ticket.inputs[0] = {host + region.host_input_offset, ggml_nbytes(region.descriptor.activation), region.descriptor.activation->nb[2]};
        ticket.inputs[1] = {};
        ticket.output = {host + region.host_raw_offset, ggml_nbytes(region.descriptor.output), region.descriptor.output->nb[1]};
        ticket.work = {};
        ticket.work.struct_size = sizeof(ticket.work);
        ticket.work.epoch = layer.epoch;
        ticket.work.graph_uid = region.batch_graph_uids[0];
        ticket.work.graph_generation = region.graph_generation;
        ticket.work.source_generation = region.source_generation;
        ticket.work.binding = &ticket.binding;
        ticket.work.dynamic_inputs = ticket.inputs;
        ticket.work.n_dynamic_inputs = 2;
        ticket.work.outputs = &ticket.output;
        ticket.work.n_outputs = 1;
        ticket.result = {};
        ticket.result.struct_size = sizeof(ticket.result);
        epoch = std::max(epoch, layer.epoch);
        last_submit_us.store(ggml_time_us());
        ticket.state = 1;
        ++cpu_jobs;
        ready.notify_all();
        return true;
    }

    bool join(bool cancel) {
        std::unique_lock<std::mutex> lock(mutex);
        if (ticket.state == 0) {
            return true;
        }
        if (cancel) {
            cpu_api->cancel(cpu_service, ticket.work.epoch);
            ++cancellations;
        }
        const int64_t started = ggml_time_us();
        last_join_start_us.store(started);
        ready.wait(lock, [&] { return ticket.state == 3; });
        const int64_t done = ggml_time_us();
        last_join_done_us.store(done);
        join_us += done - started;
        const bool ok = ticket.status == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK &&
            (ticket.rows_region != nullptr || (ticket.result.flags == GGML_BACKEND_MOE_CPU_EXECUTE_RESULT_FLAG_V1_PUBLISHED &&
            ticket.result.epoch == ticket.work.epoch && ticket.result.published_routes == ticket.binding.n_routes &&
            ticket.result.published_outputs == 1));
        ticket.state = 0;
        ticket.rows_region = nullptr;
        if (cancel) {
            GGML_ASSERT(cpu_api->drain(cpu_service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        }
        ready.notify_all();
        return ok;
    }

    void quiesce() {
        std::unique_lock<std::mutex> lock(mutex);
        quiescing = true;
        canceled.store(true);
        if (ticket.state != 0) {
            cpu_api->cancel(cpu_service, ticket.work.epoch);
        }
        ready.wait(lock, [&] { return !dispatch_active && ticket.state == 0; });
        if (cpu_service != nullptr) {
            GGML_ASSERT(cpu_api->drain(cpu_service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        }
    }

    void cancel_pending() {
        std::lock_guard<std::mutex> lock(mutex);
        canceled.store(true);
        if (ticket.state != 0) {
            cpu_api->cancel(cpu_service, ticket.work.epoch);
        }
    }

    ~ggml_cuda_moe_hybrid_session() {
        quiesce();
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
            ready.notify_all();
        }
        if (coordinator.joinable()) {
            coordinator.join();
        }
        if (ctx != nullptr) {
            ggml_cuda_set_device(ctx->device);
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream()));
            if (io_stream != nullptr) {
                CUDA_CHECK(cudaStreamSynchronize(io_stream));
            }
        }
#ifdef GGML_CUDA_MOE_HYBRID_WINDOW
        window_graph.reset();
#endif
        if (window_device != nullptr) {
            CUDA_CHECK(cudaFree(window_device));
        }
        if (window_input != nullptr) {
            CUDA_CHECK(cudaFree(window_input));
        }
        if (window_host != nullptr) {
            CUDA_CHECK(cudaFreeHost(window_host));
        }
        for (auto event : tile_done) {
            if (event != nullptr) {
                CUDA_CHECK(cudaEventDestroy(event));
            }
        }
        if (transfer_ready != nullptr) {
            CUDA_CHECK(cudaEventDestroy(transfer_ready));
        }
        if (cpu_ready != nullptr) {
            CUDA_CHECK(cudaEventDestroy(cpu_ready));
        }
        if (gpu_started != nullptr) {
            CUDA_CHECK(cudaEventDestroy(gpu_started));
        }
        if (gpu_done != nullptr) {
            CUDA_CHECK(cudaEventDestroy(gpu_done));
        }
        if (plan_ready != nullptr) {
            CUDA_CHECK(cudaEventDestroy(plan_ready));
        }
        if (producer_done != nullptr) {
            CUDA_CHECK(cudaEventDestroy(producer_done));
        }
        if (io_stream != nullptr) {
            CUDA_CHECK(cudaStreamDestroy(io_stream));
        }
        if (host != nullptr) {
            CUDA_CHECK(cudaFreeHost(host));
        }
        compute.reset();
        ggml_backend_buffer_free(arena);
        ggml_backend_buffer_free(control);
        if (distinct_experts.load() != 0 || window_requested) {
            ggml_backend_moe_cpu_service_state_v1 cpu = {};
            cpu.struct_size = sizeof(cpu);
            if (cpu_service != nullptr) {
                GGML_ASSERT(cpu_api->state(cpu_service, &cpu) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
            }
            fprintf(stderr, "moe-hybrid-summary: distinct=%llu resident=%llu transferred=%llu cpu=%llu cpu_jobs=%llu weight_h2d=%llu cpu_h2d=%llu submit_to_start_us=%llu cpu_us=%llu gpu_enqueue_us=%llu gpu_branch_us=%llu join_us=%llu errors=%llu capacity=%llu cancellations=%llu admission_reserved=%llu admission_committed=%llu admission_replacements=%llu admission_no_slot=%llu admission_bytes=%llu admission_aborted=%llu resident_batches=%llu resident_bodies=%llu gpu_bodies=%llu readback_us=%llu publication_us=%llu admission_fence_us=%llu prepared_device_bytes=%llu allocated_device_bytes=%zu pinned_bytes=%zu prepared_cpu_bytes=%llu\n",
                (unsigned long long) distinct_experts.load(), (unsigned long long) resident_experts.load(),
                (unsigned long long) transfer_experts.load(), (unsigned long long) cpu_experts.load(),
                (unsigned long long) cpu_jobs.load(), (unsigned long long) h2d_bytes.load(),
                (unsigned long long) cpu_upload_bytes.load(), (unsigned long long) submit_to_start_us.load(),
                (unsigned long long) cpu_us.load(), (unsigned long long) gpu_enqueue_us.load(),
                (unsigned long long) gpu_branch_us.load(), (unsigned long long) join_us.load(),
                (unsigned long long) errors.load(), (unsigned long long) capacity_errors.load(),
                (unsigned long long) cancellations.load(), (unsigned long long) admission_reserved.load(),
                (unsigned long long) admission_committed.load(), (unsigned long long) admission_replacements.load(),
                (unsigned long long) admission_no_slot.load(), (unsigned long long) admission_bytes.load(),
                (unsigned long long) admission_aborted.load(), (unsigned long long) resident_batches.load(),
                (unsigned long long) resident_body_submissions.load(), (unsigned long long) gpu_body_submissions.load(),
                (unsigned long long) readback_us.load(), (unsigned long long) publication_us.load(),
                (unsigned long long) admission_fence_us.load(), (unsigned long long) prepared_device_bytes.load(),
                control_bytes + device_bytes + work_bytes + window_bytes + window_input_bytes, pinned_bytes + window_bytes, (unsigned long long) cpu.prepared_payload_peak);
            fprintf(stderr, "moe-hybrid-packet: regions=%llu producer_fences=%llu producer_events=%llu drain_events=%llu fixed_copy_bytes=%llu empty_copy_bytes=%llu\n",
                (unsigned long long) packet_regions.load(), (unsigned long long) producer_fences.load(),
                (unsigned long long) producer_events.load(), (unsigned long long) producer_drain_events.load(),
                (unsigned long long) fixed_copy_bytes.load(), (unsigned long long) empty_copy_bytes.load());
            fprintf(stderr, "moe-hybrid-window: launches=%llu captures=%llu waits=%llu eager_s5b=%llu control_bytes=%zu input_bytes=%zu cpu_pipeline=%u observation_us=%llu capture_fused_nodes=%llu combined_regions=%llu direct_regions=%llu compact_select_regions=%llu fused_expert_bodies=%llu\n",
                (unsigned long long) window_launches.load(), (unsigned long long) window_captures.load(),
                (unsigned long long) window_waits.load(), (unsigned long long) window_fallbacks.load(), window_bytes, window_input_bytes,
                unsigned(window_cpu_pipeline), (unsigned long long) publication_us.load(), (unsigned long long) window_fused_nodes.load(),
                (unsigned long long) window_combined_regions.load(), (unsigned long long) window_direct_regions.load(),
                (unsigned long long) window_compact_select_regions.load(), (unsigned long long) window_fused_expert_bodies.load());
            fprintf(stderr, "moe-hybrid-cpu: tickets=%llu execute_calls=%llu source_rows=%llu\n",
                (unsigned long long) cpu_jobs.load(), (unsigned long long) cpu_execute_calls.load(),
                (unsigned long long) cpu_row_jobs.load());
        }
    }

    uint8_t * data() const {
        return arena == nullptr ? nullptr : static_cast<uint8_t *>(ggml_backend_buffer_get_base(arena));
    }

    uint8_t * control_data() const {
        return control == nullptr ? nullptr : static_cast<uint8_t *>(ggml_backend_buffer_get_base(control));
    }

    bool wait_producers(cudaStream_t stream, bool drain = false) {
        if (cudaEventRecord(producer_done, stream) != cudaSuccess) {
            return false;
        }
        if (drain) {
            ++producer_drain_events;
        } else {
            ++producer_events;
            ++producer_fences;
        }
        return cudaEventSynchronize(producer_done) == cudaSuccess;
    }

    bool copy_source(void * destination, const void * source, size_t bytes, uint32_t source_path,
            size_t staging_offset, size_t staging_bytes) {
        if (source_path == MOE_GROUPED_SOURCE_DIRECT_REGISTERED || source_path == MOE_GROUPED_SOURCE_MAPPED ||
                source_path == MOE_GROUPED_SOURCE_DEVICE) {
            const cudaMemcpyKind kind = source_path == MOE_GROUPED_SOURCE_DEVICE ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
            if (cudaMemcpyAsync(destination, source, bytes, kind, io_stream) != cudaSuccess) {
                return false;
            }
            h2d_bytes += bytes;
            return true;
        }
        if (source_path != MOE_GROUPED_SOURCE_PAGEABLE_STAGED) {
            return false;
        }
        if (staging_offset > pinned_bytes || staging_bytes == 0 || staging_bytes > (pinned_bytes - staging_offset) / 2) {
            return false;
        }
        for (size_t offset = 0; offset < bytes;) {
            const uint32_t tile = next_tile;
            if (tile_pending[tile] && cudaEventSynchronize(tile_done[tile]) != cudaSuccess) {
                return false;
            }
            tile_pending[tile] = false;
            const size_t count = std::min(staging_bytes, bytes - offset);
            auto * staging = host + staging_offset + tile * staging_bytes;
            memcpy(staging, static_cast<const uint8_t *>(source) + offset, count);
            if (cudaMemcpyAsync(static_cast<uint8_t *>(destination) + offset, staging, count,
                    cudaMemcpyHostToDevice, io_stream) != cudaSuccess) {
                return false;
            }
            if (cudaEventRecord(tile_done[tile], io_stream) != cudaSuccess) {
                (void) cudaStreamSynchronize(io_stream);
                return false;
            }
            tile_pending[tile] = true;
            next_tile = (tile + 1) % 2;
            offset += count;
            h2d_bytes += count;
        }
        return true;
    }
};

struct ggml_cuda_moe_hybrid_layout {
    size_t device_bytes = 0;
    size_t work_bytes = 0;
    size_t host_slots_offset = 0;
    size_t host_ids_offset = 0;
    size_t host_input_offset = 0;
    size_t host_raw_offset = 0;
    size_t host_staging_offset = 0;
    size_t host_staging_bytes = 0;
};

static uint32_t ggml_cuda_moe_hybrid_route_capacity(const ggml_cuda_moe_hybrid_region & region) {
    return region.multi_rows ? region.rows_layout.geometry.route_capacity : region.descriptor.n_cpu_queries;
}

static ggml_cuda_moe_hybrid_rows_view ggml_cuda_moe_hybrid_device_rows(const ggml_cuda_moe_hybrid_region & region) {
    return ggml_cuda_moe_hybrid_rows_packet(
        region.session->data() + region.rows_offset + region.rows_layout.control_offset, region.rows_layout.geometry);
}

static void ggml_cuda_moe_hybrid_submit_rows(ggml_cuda_moe_hybrid_region & region) {
    auto & session = *region.session;
    const auto rows = ggml_cuda_moe_hybrid_rows_packet(session.host + region.host_rows_offset, region.rows_layout.geometry);
    const auto & ticket = *rows.ticket;
    if (ticket.status != 0 || ticket.epoch != region.packet_epoch || ticket.identity != region.rows_layout.identity ||
            ticket.weight_count > region.rows_layout.geometry.weight_capacity ||
            ticket.route_count > region.rows_layout.geometry.route_capacity ||
            ticket.transfer_count > region.transfer_capacity) {
        if (session.diagnostic) {
            fprintf(stderr, "moe-hybrid-rows: packet rejected status=%u epoch=%llu/%llu routes=%u weights=%u transfers=%u\n",
                ticket.status, (unsigned long long) ticket.epoch, (unsigned long long) region.packet_epoch,
                ticket.route_count, ticket.weight_count, ticket.transfer_count);
        }
        region.packet_status.store(1);
        return;
    }
    auto packet = ggml_cuda_moe_hybrid_packet(session.host, ggml_cuda_moe_hybrid_route_capacity(region));
    *packet.header = ticket.selection;
    for (uint32_t row = 0; row < region.rows_layout.geometry.row_capacity; ++row) { rows.producers[2 + row] = {}; }
    for (uint32_t weight = 0; weight < ticket.weight_count; ++weight) {
        if (rows.weight_classes[weight] == GGML_CUDA_MOE_HYBRID_TRANSFER) {
            const uint32_t row = rows.weight_storage[weight];
            if (row >= ticket.transfer_count) { region.packet_status.store(1); return; }
            packet.transfers[row] = rows.weight_experts[weight];
        }
    }
    session.distinct_experts += ticket.weight_count;
    session.resident_experts += ticket.selection.resident_count;
    session.transfer_experts += ticket.transfer_count;
    session.cpu_experts += ticket.selection.cpu_count;
    session.resident_routes += ticket.resident_lanes;
    session.transferred_routes += ticket.transfer_lanes;
    session.cpu_routes += ticket.cpu_routes;
    session.cpu_upload_bytes += size_t(ticket.cpu_routes) * region.descriptor.output->nb[1];
    session.admission_reserved += ticket.selection.admissions;
    session.admission_replacements += ticket.selection.replacements;
    region.packet_admissions.store(ticket.selection.admissions);
    if (session.demand_admission) { session.admission_no_slot += ticket.transfer_count - ticket.selection.admissions; }
    for (const auto & source : region.sources) { session.h2d_bytes += source.expert_stride * ticket.transfer_count; }
    if (session.canceled.load() || !session.submit_rows(region)) {
        if (session.diagnostic) {
            fprintf(stderr, "moe-hybrid-rows: CPU batch submission rejected epoch=%llu canceled=%u\n",
                (unsigned long long) ticket.epoch, unsigned(session.canceled.load()));
        }
        region.packet_status.store(1);
    }
}

static bool ggml_cuda_moe_hybrid_validate_packet(const ggml_cuda_moe_hybrid_region & region,
        const ggml_cuda_moe_hybrid_packet_view & packet) {
    const auto & h = *packet.header;
    const uint32_t routes = region.descriptor.n_cpu_queries;
    if (h.status != 0 || h.epoch != region.packet_epoch || h.route_count != routes ||
            h.resident_count > routes || h.transfer_count > region.transfer_capacity || h.cpu_count > routes ||
            h.transfer_count > region.session->gpu_miss_quota || h.distinct_count > routes ||
            h.gpu_count > routes || h.gpu_count != uint64_t(h.resident_count) + h.transfer_count ||
            uint64_t(h.resident_count) + h.transfer_count + h.cpu_count != h.distinct_count ||
            h.admissions > h.transfer_count || h.replacements > h.admissions) {
        return false;
    }
    uint32_t admissions = 0;
    for (uint32_t row = 0; row < h.resident_count; ++row) {
        if (packet.residents[row] < 0 || uint32_t(packet.residents[row]) >= region.resident_slots) {
            return false;
        }
    }
    for (uint32_t row = 0; row < h.transfer_count; ++row) {
        const int32_t slot = packet.transfer_slots[row];
        if (packet.transfers[row] < 0 || uint32_t(packet.transfers[row]) >= region.expert_count ||
                packet.transfer_ids[row] != int32_t(row) || slot < -1 ||
                (slot >= 0 && uint32_t(slot) >= region.resident_slots)) {
            return false;
        }
        admissions += slot >= 0;
    }
    if (admissions != h.admissions || (!region.session->demand_admission && admissions != 0)) {
        return false;
    }
    if (region.combined_gpu) {
        if (h.admissions != h.transfer_count) {
            return false;
        }
        for (uint32_t row = 0; row < h.gpu_count; ++row) {
            const int32_t slot = packet.residents[row];
            if (slot < 0 || uint32_t(slot) >= region.resident_slots ||
                    (row >= h.resident_count && slot != packet.transfer_slots[row - h.resident_count])) {
                return false;
            }
            for (uint32_t previous = 0; previous < row; ++previous) {
                if (packet.residents[previous] == slot) {
                    return false;
                }
            }
        }
    }
    for (uint32_t row = 0; row < h.cpu_count; ++row) {
        const int32_t rank = packet.cpu_ranks[row];
        if (packet.cpu_ids[row] < 0 || uint32_t(packet.cpu_ids[row]) >= region.expert_count ||
                rank < 0 || uint32_t(rank) >= routes || packet.classes[rank] != 2 || packet.rows[rank] != rank) {
            return false;
        }
    }
    for (uint32_t rank = 0; rank < routes; ++rank) {
        const int32_t kind = packet.classes[rank], row = packet.rows[rank];
        if (row < 0 || kind < 0 || kind > 2 ||
                (kind == 0 && (uint32_t(row) >= h.resident_count || packet.slots[rank] != packet.residents[row])) ||
                (kind == 1 && (uint32_t(row) >= h.transfer_count ||
                    packet.slots[rank] != (packet.transfer_slots[row] < 0 ? -1 : -2 - packet.transfer_slots[row])))) {
            return false;
        }
        if (kind == 2) {
            bool found = false;
            for (uint32_t cpu = 0; cpu < h.cpu_count; ++cpu) {
                found |= packet.cpu_ranks[cpu] == row;
            }
            if (!found || uint32_t(row) > rank || packet.slots[rank] != -1) {
                return false;
            }
        }
    }
    return true;
}

static void ggml_cuda_moe_hybrid_submit_packet(void * opaque) noexcept {
    auto & region = *static_cast<ggml_cuda_moe_hybrid_region *>(opaque);
    auto & session = *region.session;
    try {
        if (session.window_active) {
            memset(session.host + region.host_raw_offset, 0, ggml_nbytes(region.descriptor.output));
        }
        if (region.multi_rows) {
            ggml_cuda_moe_hybrid_submit_rows(region);
            return;
        }
        const auto packet = ggml_cuda_moe_hybrid_packet(session.host, region.descriptor.n_cpu_queries);
        if (!ggml_cuda_moe_hybrid_validate_packet(region, packet)) {
            region.packet_status.store(1);
            return;
        }
        const auto & h = *packet.header;
        if (!session.window_active) {
            session.readback_us += ggml_time_us() - region.readback_started;
        }
        session.distinct_experts += h.distinct_count;
        session.resident_experts += h.resident_count;
        session.transfer_experts += h.transfer_count;
        session.cpu_experts += h.cpu_count;
        session.admission_reserved += h.admissions;
        region.packet_admissions.store(h.admissions);
        session.admission_replacements += h.replacements;
        if (session.demand_admission) {
            session.admission_no_slot += h.transfer_count - h.admissions;
        }
        for (uint32_t rank = 0; rank < h.route_count; ++rank) {
            session.resident_routes += packet.classes[rank] == 0;
            session.transferred_routes += packet.classes[rank] == 1;
            session.cpu_routes += packet.classes[rank] == 2;
            session.cpu_upload_bytes += packet.classes[rank] == 2 ? region.descriptor.output->nb[1] : 0;
        }
        for (const auto & source : region.sources) {
            session.h2d_bytes += source.expert_stride * h.transfer_count;
        }
        if (session.canceled.load() || (h.cpu_count != 0 && !session.submit(region, packet.cpu_ids,
                reinterpret_cast<const uint32_t *>(packet.cpu_ranks), h.cpu_count))) {
            region.packet_status.store(1);
        }
    } catch (...) {
        region.packet_status.store(1);
    }
}

static void ggml_cuda_moe_hybrid_fill_tile(void * opaque) noexcept {
    const auto & copy = *static_cast<const ggml_cuda_moe_hybrid_region::copy *>(opaque);
    auto & region = *copy.region;
    auto & session = *region.session;
    const auto packet = ggml_cuda_moe_hybrid_packet(session.host, ggml_cuda_moe_hybrid_route_capacity(region));
    auto * tile = session.host + region.host_staging_offset + copy.tile * region.host_staging_bytes;
    const bool active = !session.canceled.load() && region.packet_status.load() == 0 &&
        packet.header->epoch == region.packet_epoch && copy.row < packet.header->transfer_count;
    if (active) {
        const auto & source = region.sources[copy.bank];
        memcpy(tile, static_cast<const uint8_t *>(source.data) +
            size_t(packet.transfers[copy.row]) * source.expert_stride + copy.offset, copy.bytes);
    } else if (session.host_alias == nullptr) {
        memset(tile, 0, copy.bytes);
    }
    if (session.host_alias == nullptr) {
        session.fixed_copy_bytes += copy.bytes;
        session.empty_copy_bytes += active ? 0 : copy.bytes;
    }
}

static void ggml_cuda_moe_hybrid_complete_packet(void * opaque) noexcept {
    auto & region = *static_cast<ggml_cuda_moe_hybrid_region *>(opaque);
    auto & session = *region.session;
    if (region.host_runtime != nullptr && session.window_active) {
        const auto packet = ggml_cuda_moe_hybrid_packet(session.host, ggml_cuda_moe_hybrid_route_capacity(region));
        auto & completion = *region.host_runtime;
        completion.cpu_epoch = region.packet_epoch;
        completion.cpu_routes = packet.header->cpu_count;
        completion.cpu_status = region.packet_status.load() != 0 || session.fail_before_publish ||
            !session.hook(GGML_BACKEND_MOE_HYBRID_TEST_BEFORE_PUBLISH) || session.canceled.load();
        if (region.multi_rows) {
            auto rows = ggml_cuda_moe_hybrid_rows_packet(session.host + region.host_rows_offset, region.rows_layout.geometry);
            rows.runtime->cancel_epoch = completion.cpu_status != 0 ? region.packet_epoch : 0;
        }
    }
}

static void ggml_cuda_moe_hybrid_join_packet(void * opaque) noexcept {
    auto & region = *static_cast<ggml_cuda_moe_hybrid_region *>(opaque);
    auto & session = *region.session;
    try {
        if (!session.join(false) || session.canceled.load()) {
            region.packet_status.store(1);
        }
        if (session.host_alias == nullptr) {
            const auto packet = ggml_cuda_moe_hybrid_packet(session.host, ggml_cuda_moe_hybrid_route_capacity(region));
            const size_t bytes = ggml_nbytes(region.descriptor.output);
            session.fixed_copy_bytes += bytes;
            const uint32_t cpu = region.multi_rows ? ggml_cuda_moe_hybrid_rows_packet(
                session.host + region.host_rows_offset, region.rows_layout.geometry).ticket->cpu_routes :
                std::min(packet.header->cpu_count, region.descriptor.n_cpu_queries);
            session.empty_copy_bytes += bytes - size_t(cpu) * region.descriptor.output->nb[1];
        }
    } catch (...) {
        region.packet_status.store(1);
    }
    if (!session.window_active || !session.window_cpu_pipeline) {
        ggml_cuda_moe_hybrid_complete_packet(opaque);
    } else if (!session.hook(GGML_BACKEND_MOE_HYBRID_TEST_CPU_JOINED)) {
        region.packet_status.store(1);
    }
}

static __global__ void ggml_cuda_moe_hybrid_gather(
        ggml_cuda_moe_hybrid_packet_view packet, uint64_t epoch, uint32_t capacity, uint32_t n_experts,
        const uint4 * source, uint4 * destination, size_t words, uint32_t * status) {
    epoch = packet.runtime != nullptr ? packet.runtime->epoch : epoch;
    if (packet.header->status != 0 || packet.header->epoch != epoch || packet.header->transfer_count > capacity) {
        atomicExch(status, 1u);
        return;
    }
    for (size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            index < words * packet.header->transfer_count; index += size_t(gridDim.x) * blockDim.x) {
        const uint32_t row = index / words;
        const int32_t expert = packet.transfers[row];
        if (expert < 0 || uint32_t(expert) >= n_experts) {
            atomicExch(status, 1u);
            return;
        }
        destination[index] = source[size_t(expert) * words + index % words];
    }
}

static __global__ void ggml_cuda_moe_hybrid_copy_tile(
        ggml_cuda_moe_hybrid_packet_view packet, uint64_t epoch, uint32_t capacity, uint32_t row,
        const uint8_t * source, uint8_t * destination, size_t bytes, uint32_t * status) {
    epoch = packet.runtime != nullptr ? packet.runtime->epoch : epoch;
    if (packet.header->status != 0 || packet.header->epoch != epoch || packet.header->transfer_count > capacity) {
        atomicExch(status, 1u);
        return;
    }
    if (row >= packet.header->transfer_count) {
        return;
    }
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < bytes; i += size_t(gridDim.x) * blockDim.x) {
        destination[i] = source[i];
    }
}

static __global__ void ggml_cuda_moe_hybrid_assemble(
        ggml_cuda_moe_hybrid_packet_view packet, uint64_t epoch, uint32_t transfer_capacity, size_t row_values,
        const float * residents, const float * transfers, const float * cpu, float * output, uint32_t * status) {
    epoch = packet.runtime != nullptr ? packet.runtime->epoch : epoch;
    const auto & h = *packet.header;
    if (h.status != 0 || h.epoch != epoch || h.route_count != packet.capacity ||
            h.resident_count > packet.capacity || h.transfer_count > transfer_capacity || h.cpu_count > packet.capacity ||
            h.gpu_count > packet.capacity || h.gpu_count != uint64_t(h.resident_count) + h.transfer_count ||
            (packet.combine_gpu && h.admissions != h.transfer_count)) {
        atomicExch(status, 1u);
        return;
    }
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            i < row_values * packet.capacity; i += size_t(gridDim.x) * blockDim.x) {
        const uint32_t rank = i / row_values;
        const int32_t kind = packet.classes[rank], row = packet.rows[rank];
        const uint32_t count = kind == 0 ? h.resident_count : kind == 1 ? h.transfer_count : packet.capacity;
        if (kind < 0 || kind > 2 || row < 0 || uint32_t(row) >= count) {
            atomicExch(status, 1u);
            return;
        }
        const float * source = kind == 0 ? residents : kind == 1 ?
            (packet.combine_gpu ? residents + size_t(h.resident_count) * row_values : transfers) : cpu;
        output[i] = source[size_t(row) * row_values + i % row_values];
    }
}

static uint32_t ggml_cuda_moe_hybrid_blocks(int device, size_t values) {
    const auto & info = ggml_cuda_info().devices[device];
    return uint32_t(std::max<size_t>(1, std::min({(values + 255) / 256,
        size_t(info.nsm) * 4, size_t(info.max_grid_size[0])})));
}

static bool ggml_cuda_moe_hybrid_reserve(size_t bytes, size_t & cursor) {
    if (bytes > SIZE_MAX - (GGML_CUDA_MOE_HYBRID_ARENA_ALIGNMENT - 1)) {
        return false;
    }
    const size_t aligned = ggml_cuda_moe_hybrid_align(bytes, GGML_CUDA_MOE_HYBRID_ARENA_ALIGNMENT);
    if (cursor > SIZE_MAX - aligned) {
        return false;
    }
    cursor += aligned;
    return true;
}

static bool ggml_cuda_moe_hybrid_work_bytes(const ggml_tensor * node, size_t & bytes) {
    if (node->op != GGML_OP_MUL_MAT_ID) {
        return true;
    }
    const auto * input = node->src[1];
    if (input == nullptr || input->ne[0] <= 0 || input->ne[0] > INT64_MAX - MATRIX_ROW_PADDING) {
        return false;
    }
    size_t work = GGML_PAD(input->ne[0], MATRIX_ROW_PADDING) / QK8_1;
    if (work > SIZE_MAX / sizeof(block_q8_1)) {
        return false;
    }
    work *= sizeof(block_q8_1);
    for (int d = 1; d < GGML_MAX_DIMS; ++d) {
        if (input->ne[d] <= 0 || size_t(input->ne[d]) > SIZE_MAX / work) {
            return false;
        }
        work *= input->ne[d];
    }
    if (work > SIZE_MAX - (GGML_CUDA_MOE_HYBRID_ARENA_ALIGNMENT - 1)) {
        return false;
    }
    bytes = std::max(bytes, ggml_cuda_moe_hybrid_align(work, GGML_CUDA_MOE_HYBRID_ARENA_ALIGNMENT));
    return true;
}

static bool ggml_cuda_moe_hybrid_lane_shape(ggml_tensor & tensor, uint32_t lanes, bool ids) {
    if (lanes == 0 || lanes > INT32_MAX || tensor.ne[2] != 1 || tensor.ne[3] != 1) {
        return false;
    }
    if (ids) {
        if (tensor.type != GGML_TYPE_I32 || tensor.nb[0] != sizeof(int32_t)) { return false; }
        tensor.ne[0] = lanes;
        tensor.ne[1] = 1;
        tensor.nb[1] = size_t(lanes) * sizeof(int32_t);
    } else {
        if (tensor.type != GGML_TYPE_F32 || tensor.nb[1] > SIZE_MAX / lanes) { return false; }
        tensor.ne[1] = lanes;
    }
    tensor.nb[2] = tensor.nb[1] * tensor.ne[1];
    tensor.nb[3] = tensor.nb[2];
    return true;
}

static bool ggml_cuda_moe_hybrid_measure_layout(
        ggml_cuda_moe_hybrid_session & session, const ggml_backend_moe_hybrid_region_v1 & descriptor,
        uint32_t transfer_capacity, bool resident_batch, ggml_cuda_moe_hybrid_layout & layout) {
    const auto & query = *descriptor.cpu_queries[transfer_capacity - 1];
    const bool multi_rows = descriptor.geometry.row_capacity > 1;
    const uint32_t lanes = descriptor.geometry.route_capacity;
    auto reserve_tensor = [&](const ggml_tensor * tensor, bool source) {
        ggml_tensor copy = *tensor;
        size_t bytes;
        if (source) {
            if (copy.nb[2] > SIZE_MAX / transfer_capacity) {
                return false;
            }
            copy.ne[2] = transfer_capacity;
            copy.ne[3] = 1;
            copy.nb[3] = copy.nb[2] * transfer_capacity;
            bytes = ggml_backend_buft_get_alloc_size(ggml_backend_cuda_buffer_type(session.ctx->device), &copy);
        } else {
            if (multi_rows && !ggml_cuda_moe_hybrid_lane_shape(copy, lanes, tensor->type == GGML_TYPE_I32)) {
                return false;
            }
            bytes = ggml_nbytes(&copy);
        }
        return ggml_cuda_moe_hybrid_reserve(bytes, layout.device_bytes);
    };
    const auto reserve_dynamic = [&](const ggml_backend_moe_cpu_region_query_v1 & bucket) {
        return !multi_rows || (reserve_tensor(bucket.activation, false) && reserve_tensor(bucket.ids, false));
    };
    const auto reserve_work = [&](const ggml_tensor * node) {
        ggml_tensor copy = *node;
        ggml_tensor input;
        if (multi_rows && node->op == GGML_OP_MUL_MAT_ID) {
            input = *node->src[1];
            if (!ggml_cuda_moe_hybrid_lane_shape(input, lanes, false)) { return false; }
            copy.src[1] = &input;
        }
        return ggml_cuda_moe_hybrid_work_bytes(&copy, layout.work_bytes);
    };
    if (!reserve_dynamic(query)) { return false; }
    for (uint32_t i = 0; i < query.n_sources; ++i) {
        if (!reserve_tensor(query.sources[i].tensor, true)) {
            return false;
        }
        if (query.sources[i].expert_stride > SIZE_MAX) {
            return false;
        }
        layout.host_staging_bytes = std::max(layout.host_staging_bytes, size_t(query.sources[i].expert_stride));
    }
    for (uint32_t i = 0; i < query.n_body_nodes; ++i) {
        const auto * node = query.body_nodes[i];
        if (node->op != GGML_OP_VIEW && !reserve_tensor(node, false)) {
            return false;
        }
        if (!reserve_work(node)) {
            return false;
        }
    }
    if (resident_batch) {
        const auto & compact = *descriptor.cpu_queries[descriptor.n_cpu_queries - 1];
        if (!reserve_dynamic(compact)) { return false; }
        for (uint32_t i = 0; i < compact.n_body_nodes; ++i) {
            const auto * node = compact.body_nodes[i];
            if (node->op != GGML_OP_VIEW && !reserve_tensor(node, false)) {
                return false;
            }
            if (!reserve_work(node)) {
                return false;
            }
        }
    }
    if (!ggml_cuda_moe_hybrid_reserve(ggml_nbytes(descriptor.output), layout.device_bytes) ||
            (resident_batch && !ggml_cuda_moe_hybrid_reserve(ggml_nbytes(descriptor.output), layout.device_bytes))) {
        return false;
    }
    const size_t input_bytes = ggml_nbytes(descriptor.activation);
    const size_t output_bytes = ggml_nbytes(descriptor.output);
    const size_t routes = multi_rows ? lanes : descriptor.ids->ne[0];
    layout.host_slots_offset = ggml_cuda_moe_hybrid_align(sizeof(ggml_cuda_moe_hybrid_selection), alignof(int32_t));
    if (routes > (SIZE_MAX - layout.host_slots_offset) / sizeof(int32_t)) {
        return false;
    }
    layout.host_ids_offset = layout.host_slots_offset + routes * sizeof(int32_t);
    if (routes > (SIZE_MAX - layout.host_ids_offset) / sizeof(int32_t)) {
        return false;
    }
    if (routes > (SIZE_MAX - sizeof(ggml_cuda_moe_hybrid_selection)) /
            (GGML_CUDA_MOE_HYBRID_PACKET_ARRAYS * sizeof(int32_t))) {
        return false;
    }
    const size_t packet_bytes = sizeof(ggml_cuda_moe_hybrid_selection) + routes * GGML_CUDA_MOE_HYBRID_PACKET_ARRAYS * sizeof(int32_t);
    if (!ggml_cuda_moe_hybrid_align_checked(packet_bytes,
            GGML_CUDA_MOE_HYBRID_ARENA_ALIGNMENT, layout.host_input_offset)) {
        return false;
    }
    if (layout.host_input_offset > SIZE_MAX - input_bytes) {
        return false;
    }
    if (!ggml_cuda_moe_hybrid_align_checked(layout.host_input_offset + input_bytes,
            GGML_CUDA_MOE_HYBRID_ARENA_ALIGNMENT, layout.host_raw_offset)) {
        return false;
    }
    if (layout.host_raw_offset > SIZE_MAX - output_bytes) {
        return false;
    }
    if (!ggml_cuda_moe_hybrid_align_checked(layout.host_raw_offset + output_bytes,
            GGML_CUDA_MOE_HYBRID_ARENA_ALIGNMENT, layout.host_staging_offset)) {
        return false;
    }
    if (layout.host_staging_bytes == 0 || layout.host_staging_bytes > (SIZE_MAX - layout.host_staging_offset) / 2) {
        return false;
    }
    layout.host_staging_bytes *= 2;
    return true;
}

static void ggml_cuda_moe_hybrid_rebase_tensor(
        ggml_tensor & tensor, ggml_backend_buffer_t old_arena, uint8_t * old_base, size_t old_bytes,
        ggml_backend_buffer_t new_arena, uint8_t * new_base) {
    if (tensor.data != nullptr && old_base != nullptr) {
        const auto address = reinterpret_cast<uintptr_t>(tensor.data);
        const auto begin = reinterpret_cast<uintptr_t>(old_base);
        if (address >= begin && address - begin < old_bytes) {
            tensor.data = new_base + (address - begin);
        }
    }
    if (tensor.buffer == old_arena) {
        tensor.buffer = new_arena;
    }
}

static bool ggml_cuda_moe_hybrid_ensure_device(
        ggml_cuda_moe_hybrid_session & session, uint32_t route_capacity,
        size_t arena_required, size_t work_required) {
    ggml_cuda_moe_hybrid_control_layout control_layout;
    if (!ggml_cuda_moe_hybrid_measure_control(std::max(session.route_capacity, route_capacity), control_layout)) {
        return false;
    }
    const size_t arena_bytes = std::max(session.arena_required, arena_required);
    const size_t work_bytes = std::max(session.work_bytes, work_required);
    if (arena_bytes > SIZE_MAX - work_bytes || control_layout.bytes > SIZE_MAX - arena_bytes - work_bytes) {
        return false;
    }
    const size_t required = control_layout.bytes + arena_bytes + work_bytes;
    if (session.device_limit != 0 && required > session.device_limit) {
        return false;
    }
    if (session.control == nullptr || session.control_bytes != control_layout.bytes) {
        auto * replacement = ggml_backend_buft_alloc_buffer(
            ggml_backend_cuda_buffer_type(session.ctx->device), control_layout.bytes);
        if (replacement == nullptr) {
            return false;
        }
        ggml_backend_buffer_set_usage(replacement, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_buffer_free(session.control);
        session.control = replacement;
    }
    const size_t total = arena_bytes + work_bytes;
    const size_t old_total = session.device_bytes + session.work_bytes;
    if (session.arena == nullptr || old_total != total) {
        auto * replacement = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(session.ctx->device), total);
        if (replacement == nullptr) {
            return false;
        }
        ggml_backend_buffer_set_usage(replacement, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        auto * old_base = session.data();
        auto * new_base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(replacement));
        for (auto * region : session.prepared_regions) {
            for (auto & tensor : region->tensors) {
                ggml_cuda_moe_hybrid_rebase_tensor(tensor, session.arena, old_base, old_total, replacement, new_base);
            }
            for (auto & bucket : region->resident_buckets) {
                for (auto & tensor : bucket.tensors) {
                    ggml_cuda_moe_hybrid_rebase_tensor(tensor, session.arena, old_base, old_total, replacement, new_base);
                }
            }
        }
        ggml_backend_buffer_free(session.arena);
        session.arena = replacement;
    }
    session.control_bytes = control_layout.bytes;
    session.route_capacity = std::max(session.route_capacity, route_capacity);
    session.arena_required = arena_bytes;
    session.device_bytes = arena_bytes;
    session.work_bytes = work_bytes;
    session.compute->pools[session.ctx->device][0] =
        std::make_unique<ggml_cuda_moe_hybrid_pool>(session.data() + arena_bytes, work_bytes);
    return true;
}

static bool ggml_cuda_moe_hybrid_ensure_host(
        ggml_cuda_moe_hybrid_session & session, size_t fixed_bytes, size_t staging_bytes) {
    if (fixed_bytes > SIZE_MAX - staging_bytes) {
        return false;
    }
    const size_t required = fixed_bytes + staging_bytes;
    if (session.pinned_limit != 0 && (session.pinned_limit <= fixed_bytes || session.pinned_limit - fixed_bytes < 2)) {
        return false;
    }
    const size_t bytes = session.pinned_limit != 0 ? session.pinned_limit : std::max(session.pinned_bytes, required);
    if (bytes != session.pinned_bytes) {
        uint8_t * replacement = nullptr;
        if (cudaMallocHost(&replacement, bytes) != cudaSuccess) {
            (void) cudaGetLastError();
            return false;
        }
        uint8_t * alias = nullptr;
        const auto mapped = cudaHostGetDevicePointer(&alias, replacement, 0);
        if (mapped != cudaSuccess) {
            (void) cudaGetLastError();
            if (mapped != cudaErrorInvalidValue && mapped != cudaErrorNotSupported) {
                (void) cudaFreeHost(replacement);
                return false;
            }
        }
        if (session.host != nullptr) {
            CUDA_CHECK(cudaFreeHost(session.host));
        }
        session.host = replacement;
        session.host_alias = session.test_no_host_alias ? nullptr : alias;
        session.pinned_bytes = bytes;
    }
    return true;
}

static int32_t ggml_cuda_moe_hybrid_create(
        const ggml_backend_moe_hybrid_config_v1 * config, void ** output) try {
    if (output == nullptr) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
    }
    *output = nullptr;
    if (config == nullptr || config->struct_size != sizeof(*config) || !ggml_backend_is_cuda(config->backend) ||
            config->admission_quota > config->gpu_miss_quota ||
            config->demand_admission > 1 ||
            config->resident_batch > 1 || config->device_bytes > SIZE_MAX || config->pinned_bytes > SIZE_MAX) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
    }
    auto state = std::make_unique<ggml_cuda_moe_hybrid_session>();
    state->backend = config->backend;
    state->ctx = static_cast<ggml_backend_cuda_context *>(config->backend->context);
    state->device_limit = config->device_bytes;
    state->pinned_limit = config->pinned_bytes;
    state->fail_before_publish = getenv("GGML_MOE_HYBRID_FAIL_BEFORE_PUBLISH") != nullptr;
    state->diagnostic = getenv("GGML_MOE_HYBRID_LOG") != nullptr;
    state->gpu_miss_quota = config->gpu_miss_quota;
    state->admission_quota = config->admission_quota;
    state->demand_admission = config->demand_admission != 0;
    state->resident_batch = config->resident_batch != 0;
    state->test_no_host_alias = getenv("GGML_MOE_HYBRID_TEST_NO_HOST_ALIAS") != nullptr;
    state->window_requested = getenv("GGML_MOE_HYBRID_WINDOW") != nullptr && atoi(getenv("GGML_MOE_HYBRID_WINDOW")) != 0;
    state->window_cpu_pipeline = getenv("GGML_MOE_HYBRID_WINDOW_PIPELINE_CPU") == nullptr ||
        atoi(getenv("GGML_MOE_HYBRID_WINDOW_PIPELINE_CPU")) != 0;
    state->window_combine_gpu = getenv("GGML_MOE_HYBRID_WINDOW_COMBINE_GPU") == nullptr ||
        atoi(getenv("GGML_MOE_HYBRID_WINDOW_COMBINE_GPU")) != 0;
    state->window_direct_gather = getenv("GGML_MOE_HYBRID_WINDOW_DIRECT_GATHER") != nullptr &&
        atoi(getenv("GGML_MOE_HYBRID_WINDOW_DIRECT_GATHER")) != 0;
    state->window_fuse_experts = getenv("GGML_MOE_HYBRID_WINDOW_FUSE_EXPERTS") != nullptr &&
        atoi(getenv("GGML_MOE_HYBRID_WINDOW_FUSE_EXPERTS")) != 0;
    ggml_cuda_set_device(state->ctx->device);
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    const unsigned producer_flags = cudaEventDisableTiming;
#else
    const unsigned producer_flags = cudaEventBlockingSync | cudaEventDisableTiming;
#endif
    if (cudaEventCreateWithFlags(&state->tile_done[0], cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&state->tile_done[1], cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&state->transfer_ready, cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&state->cpu_ready, cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreate(&state->gpu_started) != cudaSuccess || cudaEventCreate(&state->gpu_done) != cudaSuccess ||
            cudaEventCreateWithFlags(&state->plan_ready, cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&state->producer_done, producer_flags) != cudaSuccess ||
            cudaStreamCreateWithFlags(&state->io_stream, cudaStreamNonBlocking) != cudaSuccess) {
        (void) cudaGetLastError();
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
    }
    state->compute = std::make_unique<ggml_backend_cuda_context>(state->ctx->device);
    state->compute->borrowed_stream = state->ctx->stream();
    state->compute->streams[state->ctx->device][0] = state->ctx->stream();
    state->coordinator = std::thread([session = state.get()] { session->run(); });
    fprintf(stderr, "moe-hybrid: eager required policy=fixed-quota gpu_misses=%u admission_misses=%u demand_admission=%u resident_batch=%u cpu_limit=%llu device_limit=%llu pinned_limit=%llu threads=%u cpu_flags=%u\n",
        config->gpu_miss_quota, config->admission_quota, config->demand_admission, config->resident_batch,
        (unsigned long long) config->cpu_bytes, (unsigned long long) config->device_bytes,
        (unsigned long long) config->pinned_bytes, config->n_threads, config->cpu_flags);
    *output = state.release();
    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
} catch (...) {
    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
}

static int32_t ggml_cuda_moe_hybrid_prepare(
        void * opaque, const ggml_backend_moe_hybrid_region_v1 * descriptor,
        const ggml_backend_moe_cpu_region_service_api_v1 * cpu_api,
        ggml_backend_moe_cpu_service_v1_t cpu_service,
        const ggml_backend_moe_cpu_prepared_region_v1_t * cpu_regions, void ** output) try {
    if (output == nullptr) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
    }
    *output = nullptr;
    auto * session = static_cast<ggml_cuda_moe_hybrid_session *>(opaque);
    if (session == nullptr || descriptor == nullptr || descriptor->query == nullptr ||
            descriptor->activation == nullptr || descriptor->ids == nullptr || descriptor->output == nullptr ||
            descriptor->activation->type != GGML_TYPE_F32 || descriptor->activation->ne[1] != 1 ||
            descriptor->activation->ne[2] <= 0 || descriptor->ids->type != GGML_TYPE_I32 ||
            descriptor->ids->ne[0] <= 0 || uint64_t(descriptor->ids->ne[0]) > UINT32_MAX ||
            descriptor->ids->ne[1] != descriptor->activation->ne[2] ||
            descriptor->output->type != GGML_TYPE_F32 || !ggml_is_contiguous(descriptor->output) ||
            cpu_api == nullptr || cpu_service == nullptr || cpu_regions == nullptr || descriptor->cpu_queries == nullptr ||
            descriptor->n_cpu_queries != descriptor->ids->ne[0]) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
    }
    if (ggml_backend_moe_hybrid_validate_buckets_v1(descriptor) != GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
    }
    const uint32_t transfer_capacity = std::max(1u, std::min(session->gpu_miss_quota, descriptor->n_cpu_queries));
    const auto & query = *descriptor->cpu_queries[transfer_capacity - 1];
    if (query.n_dynamic_inputs != 2 || query.n_live_outputs != 1 || query.n_sources == 0 ||
            query.n_body_nodes == 0 || query.activation != query.dynamic_inputs[0] ||
            query.ids != query.dynamic_inputs[1] || query.live_outputs[0] != query.body_nodes[query.n_body_nodes - 1]) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
    }
    ggml_cuda_set_device(session->ctx->device);
    auto region = std::make_unique<ggml_cuda_moe_hybrid_region>();
    region->session = session;
    region->descriptor = *descriptor;
    region->descriptor.query = nullptr;
    region->descriptor.cpu_queries = nullptr;
    region->descriptor.cpu_batch_queries = nullptr;
    region->cpu_api = cpu_api;
    region->cpu_service = cpu_service;
    region->cpu_regions.resize(descriptor->n_cpu_queries);
    region->graph_uids.resize(descriptor->n_cpu_queries);
    for (uint32_t i = 0; i < descriptor->n_cpu_queries; ++i) {
        if (cpu_regions[i] == 0 || descriptor->cpu_queries[i] == nullptr) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
        }
        region->cpu_regions[i] = cpu_regions[i];
        region->graph_uids[i] = descriptor->cpu_queries[i]->graph_uid;
    }
    region->cpu_batch_regions.resize(descriptor->n_cpu_batch_queries);
    region->batch_graph_uids.resize(descriptor->n_cpu_batch_queries);
    for (uint32_t i = 0; i < descriptor->n_cpu_batch_queries; ++i) {
        const uint32_t prepared_index = descriptor->n_cpu_queries + i;
        if (cpu_regions[prepared_index] == 0 || descriptor->cpu_batch_queries[i] == nullptr) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
        }
        region->cpu_batch_regions[i] = cpu_regions[prepared_index];
        region->batch_graph_uids[i] = descriptor->cpu_batch_queries[i]->graph_uid;
    }
    region->graph_generation = query.graph_generation;
    region->source_generation = query.source_generation;
    region->n_dynamic = query.n_dynamic_inputs;
    region->expert_count = query.sources[0].tensor->ne[2];
    region->transfer_capacity = transfer_capacity;
    region->multi_rows = descriptor->geometry.row_capacity > 1;
    if (region->multi_rows && descriptor->n_cpu_batch_queries != 1) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
    }
    if (region->multi_rows && (!session->resident_batch || !session->window_requested ||
            descriptor->certificate.domain != GGML_GRAPH_EXECUTION_DOMAIN_MAIN ||
            descriptor->certificate.row_semantics != GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SPECULATIVE)) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
    }
    if (session->resident_batch) {
        ggml_cuda_moe_candidate_group_info info;
        if (session->ctx->moe_grouped_context == nullptr ||
                !session->ctx->moe_grouped_context->find_down_group_key(descriptor->down, &region->resident_key) ||
                !session->ctx->moe_grouped_context->get_group(region->resident_key, &info) ||
                info.n_resource_banks != query.n_sources || info.n_slots == 0 || info.n_slots > INT32_MAX) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
        }
        region->resident_slots = info.n_slots;
    }
    ggml_cuda_moe_hybrid_layout layout;
    if (!ggml_cuda_moe_hybrid_measure_layout(*session, *descriptor, transfer_capacity, session->resident_batch, layout)) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
    }
    const uint32_t routes = region->multi_rows ? descriptor->geometry.route_capacity : descriptor->n_cpu_queries;
    if (region->multi_rows) {
        std::vector<uint8_t> accessible(descriptor->query->n_sources, 0);
        std::vector<ggml_cuda_moe_hybrid_rows_source> measured_sources;
        ggml_cuda_moe_hybrid_rows_query rows_query;
        rows_query.region = descriptor;
        rows_query.certificate = descriptor->certificate;
        rows_query.plan_capacity = routes;
        rows_query.slot_capacity = region->resident_slots;
        rows_query.transfer_capacity = transfer_capacity;
        rows_query.workspace_generation = session->ctx->workspace_generation + 1;
        rows_query.resource_fingerprint = descriptor->source_graph_uid;
        rows_query.device_alignment = GGML_CUDA_MOE_HYBRID_ARENA_ALIGNMENT;
        rows_query.host_alignment = GGML_CUDA_MOE_HYBRID_ARENA_ALIGNMENT;
        rows_query.source_device_accessible = accessible.data();
        rows_query.n_sources = accessible.size();
        if (routes > region->resident_slots ||
                !ggml_cuda_moe_hybrid_rows_measure(rows_query, region->rows_layout, measured_sources)) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
        }
        for (const auto & source : measured_sources) {
            if (source.role != GGML_CUDA_MOE_HYBRID_SOURCE_MMID_WEIGHT) {
                return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
            }
        }
        region->rows_offset = layout.device_bytes;
        size_t control_bytes = 0;
        if (!ggml_cuda_moe_hybrid_reserve(region->rows_layout.device_bytes, layout.device_bytes) ||
                !ggml_cuda_moe_hybrid_align_checked(region->rows_layout.control_bytes,
                    GGML_CUDA_MOE_HYBRID_ARENA_ALIGNMENT, control_bytes) ||
                control_bytes > SIZE_MAX - layout.host_staging_offset) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
        }
        region->host_rows_offset = layout.host_input_offset;
        layout.host_input_offset += control_bytes;
        layout.host_raw_offset += control_bytes;
        layout.host_staging_offset += control_bytes;
    }
    if (!ggml_cuda_moe_hybrid_ensure_device(*session, routes, layout.device_bytes, layout.work_bytes) ||
            !ggml_cuda_moe_hybrid_ensure_host(*session, layout.host_staging_offset, layout.host_staging_bytes)) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
    }
    region->host_slots_offset = layout.host_slots_offset;
    region->host_ids_offset = layout.host_ids_offset;
    region->host_input_offset = layout.host_input_offset;
    region->host_raw_offset = layout.host_raw_offset;
    region->host_staging_offset = layout.host_staging_offset;
    region->host_staging_bytes = (session->pinned_bytes - layout.host_staging_offset) / 2;
    region->distinct_ids.resize(routes);
    region->slots.resize(routes);
    region->cpu_ids.resize(routes);
    region->classes.resize(routes);
    region->first_ranks.resize(routes);
    region->fanout.resize(routes);
    region->cpu_ranks.resize(routes);
    region->cpu_rows.resize(routes);
    region->resident_rows.resize(routes);
    region->transfer_rows.resize(routes);
    region->source_banks.resize(query.n_sources);
    region->bank_indices.resize(query.n_sources);
    std::vector<const ggml_tensor *> originals(query.dynamic_inputs, query.dynamic_inputs + query.n_dynamic_inputs);
    for (uint32_t i = 0; i < query.n_sources; ++i) {
        const auto & source = query.sources[i];
        if (!ggml_is_contiguous(source.tensor) || source.tensor->ne[2] <= 0 ||
                uint64_t(source.tensor->ne[2]) > UINT32_MAX ||
                source.tensor->ne[3] != 1 || source.expert_stride != source.tensor->nb[2]) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
        }
        originals.push_back(source.tensor);
        region->sources.push_back(source);
    }
    originals.insert(originals.end(), query.body_nodes, query.body_nodes + query.n_body_nodes);
    region->tensors.resize(originals.size());
    const auto resolve = [&](const ggml_tensor * tensor) -> ggml_tensor * {
        if (tensor == nullptr) {
            return nullptr;
        }
        const auto it = std::find(originals.begin(), originals.end(), tensor);
        return it == originals.end() ? nullptr : &region->tensors[it - originals.begin()];
    };
    size_t cursor = 0;
    const auto reserve = [&](size_t bytes, size_t & offset) {
        if (cursor > session->device_bytes || bytes > session->device_bytes - cursor ||
                bytes > SIZE_MAX - (GGML_CUDA_MOE_HYBRID_ARENA_ALIGNMENT - 1)) {
            return false;
        }
        offset = cursor;
        cursor += ggml_cuda_moe_hybrid_align(bytes, GGML_CUDA_MOE_HYBRID_ARENA_ALIGNMENT);
        return cursor <= session->device_bytes;
    };
    for (size_t i = 0; i < originals.size(); ++i) {
        auto & tensor = region->tensors[i];
        tensor = *originals[i];
        tensor.buffer = session->arena;
        tensor.extra = nullptr;
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            tensor.src[s] = resolve(originals[i]->src[s]);
        }
        tensor.view_src = resolve(originals[i]->view_src);
        if (i < query.n_dynamic_inputs) {
            if (region->multi_rows) {
                size_t offset = 0;
                if (!ggml_cuda_moe_hybrid_lane_shape(tensor, routes, i == 1) || !reserve(ggml_nbytes(&tensor), offset)) {
                    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
                }
                tensor.data = session->data() + offset;
            } else {
                tensor.data = i == 0 ? descriptor->activation->data : nullptr;
            }
        } else if (i < query.n_dynamic_inputs + query.n_sources) {
            tensor.ne[2] = transfer_capacity;
            tensor.ne[3] = 1;
            tensor.nb[3] = tensor.nb[2] * transfer_capacity;
            size_t offset = 0;
            const size_t bytes = ggml_backend_buft_get_alloc_size(ggml_backend_cuda_buffer_type(session->ctx->device), &tensor);
            if (!reserve(bytes, offset)) {
                return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
            }
            tensor.data = session->data() + offset;
            region->source_offsets.push_back(offset);
            region->sources[i - query.n_dynamic_inputs].tensor = &tensor;
        } else {
            if (region->multi_rows && !ggml_cuda_moe_hybrid_lane_shape(tensor, routes, false)) {
                return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
            }
            if (tensor.op == GGML_OP_VIEW && tensor.view_src != nullptr) {
                tensor.data = static_cast<uint8_t *>(tensor.view_src->data) + tensor.view_offs;
            } else {
                if (tensor.op != GGML_OP_MUL_MAT_ID && tensor.op != GGML_OP_GLU) {
                    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
                }
                size_t offset = 0;
                if (!reserve(ggml_nbytes(&tensor), offset)) {
                    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
                }
                tensor.data = session->data() + offset;
            }
            if (!ggml_backend_supports_op(session->backend, &tensor) ||
                    (tensor.op == GGML_OP_MUL_MAT_ID &&
                     !ggml_cuda_moe_use_compact_mmvq(originals[i], transfer_capacity))) {
                return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
            }
            region->nodes.push_back(&tensor);
        }
    }
    if (session->resident_batch) {
        const auto * pool = static_cast<const ggml_cuda_moe_hybrid_pool *>(
            session->compute->pools[session->ctx->device][0].get());
        const auto & compact = *descriptor->cpu_queries[descriptor->n_cpu_queries - 1];
        const auto prepare_bucket = [&](ggml_cuda_moe_hybrid_bucket & bucket, uint32_t physical_slots) {
            originals.assign(compact.dynamic_inputs, compact.dynamic_inputs + compact.n_dynamic_inputs);
            for (uint32_t bank = 0; bank < compact.n_sources; ++bank) {
                originals.push_back(compact.sources[bank].tensor);
            }
            originals.insert(originals.end(), compact.body_nodes, compact.body_nodes + compact.n_body_nodes);
            bucket.tensors.resize(originals.size());
            const auto resolve_bucket = [&](const ggml_tensor * tensor) -> ggml_tensor * {
                const auto it = std::find(originals.begin(), originals.end(), tensor);
                return tensor == nullptr || it == originals.end() ? nullptr : &bucket.tensors[it - originals.begin()];
            };
            for (size_t i = 0; i < originals.size(); ++i) {
                auto & tensor = bucket.tensors[i];
                tensor = *originals[i];
                tensor.buffer = session->arena;
                tensor.extra = nullptr;
                for (int s = 0; s < GGML_MAX_SRC; ++s) {
                    tensor.src[s] = resolve_bucket(originals[i]->src[s]);
                }
                tensor.view_src = resolve_bucket(originals[i]->view_src);
                if (i < compact.n_dynamic_inputs) {
                    if (region->multi_rows) {
                        size_t offset = 0;
                        if (!ggml_cuda_moe_hybrid_lane_shape(tensor, routes, i == 1) || !reserve(ggml_nbytes(&tensor), offset)) {
                            return false;
                        }
                        tensor.data = session->data() + offset;
                    } else {
                        tensor.data = i == 0 ? descriptor->activation->data : nullptr;
                    }
                } else if (i < compact.n_dynamic_inputs + compact.n_sources) {
                    if (tensor.nb[2] > SIZE_MAX / physical_slots) {
                        return false;
                    }
                    tensor.ne[2] = physical_slots;
                    tensor.nb[3] = tensor.nb[2] * physical_slots;
                    tensor.data = nullptr;
                } else {
                    if (region->multi_rows && !ggml_cuda_moe_hybrid_lane_shape(tensor, routes, false)) { return false; }
                    if (tensor.op == GGML_OP_VIEW && tensor.view_src != nullptr) {
                        tensor.data = static_cast<uint8_t *>(tensor.view_src->data) + tensor.view_offs;
                    } else {
                        if (tensor.op != GGML_OP_MUL_MAT_ID && tensor.op != GGML_OP_GLU) {
                            return false;
                        }
                        size_t offset = 0;
                        if (!reserve(ggml_nbytes(&tensor), offset)) {
                            return false;
                        }
                        tensor.data = session->data() + offset;
                    }
                    if (!ggml_backend_supports_op(session->backend, &tensor) ||
                            (tensor.op == GGML_OP_MUL_MAT_ID &&
                             !ggml_cuda_moe_use_compact_mmvq(originals[i], physical_slots))) {
                        return false;
                    }
                    if (tensor.op == GGML_OP_MUL_MAT_ID) {
                        const auto * input = tensor.src[1];
                        if (input->ne[0] <= 0 || input->ne[0] > INT64_MAX - MATRIX_ROW_PADDING) {
                            return false;
                        }
                        const size_t blocks = GGML_PAD(input->ne[0], MATRIX_ROW_PADDING) / QK8_1;
                        if (blocks > pool->capacity / sizeof(block_q8_1)) {
                            return false;
                        }
                        size_t work = blocks * sizeof(block_q8_1);
                        for (int d = 1; d < GGML_MAX_DIMS; ++d) {
                            if (input->ne[d] <= 0 || uint64_t(input->ne[d]) > pool->capacity / work) {
                                return false;
                            }
                            work *= input->ne[d];
                        }
                        if (work > pool->capacity ||
                                ggml_cuda_moe_hybrid_align(work, GGML_CUDA_MOE_HYBRID_ARENA_ALIGNMENT) > pool->capacity) {
                            return false;
                        }
                    }
                    bucket.nodes.push_back(&tensor);
                }
            }
            return true;
        };
        region->resident_buckets.resize(1);
        if (!prepare_bucket(region->resident_buckets[0], region->resident_slots)) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
        }
    }
    if (!reserve(ggml_nbytes(descriptor->output), region->raw_offset) ||
            (session->resident_batch && !reserve(ggml_nbytes(descriptor->output), region->cpu_offset))) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
    }
    if (region->multi_rows) {
        size_t offset = 0;
        if (!reserve(region->rows_layout.device_bytes, offset) || offset != region->rows_offset) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
        }
        region->raw_offset = offset + region->rows_layout.publish_offset;
        region->cpu_offset = offset + region->rows_layout.cpu_output_offset;
    }
    if (session->resident_batch) {
        for (uint32_t bank = 0; bank < region->sources.size(); ++bank) {
            for (uint32_t row = 0; row < transfer_capacity; ++row) {
                for (size_t offset = 0; offset < region->sources[bank].expert_stride;) {
                    const size_t bytes = std::min(region->host_staging_bytes, size_t(region->sources[bank].expert_stride - offset));
                    region->copies.push_back({region.get(), bank, row, uint32_t(region->copies.size() % 2), offset, bytes});
                    offset += bytes;
                }
            }
        }
    }
    if (cursor != layout.device_bytes) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
    }
    {
        std::lock_guard<std::mutex> lock(session->mutex);
        if (session->dispatch_active || session->ticket.state != 0 ||
                (session->cpu_service != nullptr && session->cpu_service != cpu_service) ||
                cpu_api->set_test_hook(cpu_service, ggml_cuda_moe_hybrid_session::cpu_hook, session) !=
                    GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_ACTIVE_JOBS;
        }
        session->cpu_api = cpu_api;
        session->cpu_service = cpu_service;
        session->quiescing = false;
        const auto * pool = static_cast<const ggml_cuda_moe_hybrid_pool *>(
            session->compute->pools[session->ctx->device][0].get());
        session->prepared_device_bytes = std::max(
            session->prepared_device_bytes.load(), uint64_t(session->control_bytes + cursor + pool->capacity));
        session->prepared_regions.push_back(region.get());
    }
    *output = region.release();
    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
} catch (...) {
    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
}

static bool ggml_cuda_moe_hybrid_can_fuse_experts(const std::vector<ggml_tensor *> & nodes) {
    if (nodes.size() != 4 || nodes[0]->op != GGML_OP_MUL_MAT_ID || nodes[1]->op != GGML_OP_MUL_MAT_ID ||
            nodes[2]->op != GGML_OP_GLU || nodes[3]->op != GGML_OP_MUL_MAT_ID ||
            (nodes[0]->flags & GGML_TENSOR_FLAG_OUTPUT) || (nodes[1]->flags & GGML_TENSOR_FLAG_OUTPUT) ||
            (nodes[2]->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }
    const auto * glu = nodes[2];
    const auto * gate = glu->src[0];
    const auto * up = glu->src[1];
    return ((gate == nodes[0] && up == nodes[1]) || (gate == nodes[1] && up == nodes[0])) &&
        nodes[3]->src[1] == glu && nodes[3]->src[2] == up->src[2] &&
        ggml_cuda_should_fuse_mul_mat(up, gate, glu);
}

static __global__ void ggml_cuda_moe_hybrid_rows_begin(ggml_cuda_moe_hybrid_rows_view rows,
        const ggml_cuda_moe_hybrid_runtime * runtime, uint64_t identity, uint32_t n_rows, uint32_t n_sequences, uint32_t quota) {
    *rows.runtime = {runtime->epoch, identity, 0, n_rows, n_sequences, quota, 0};
    for (uint32_t producer = 0; producer < n_rows + 2; ++producer) { rows.producers[producer] = {}; }
}

static __global__ void ggml_cuda_moe_hybrid_rows_check(ggml_cuda_moe_hybrid_rows_view rows,
        ggml_cuda_moe_hybrid_packet_view packet) {
    auto & ticket = *rows.ticket;
    const auto header = *packet.header;
    ticket.selection = header;
    if (ticket.status != 0 || header.status != 0 || header.epoch != ticket.epoch || header.route_count != ticket.route_count ||
            header.distinct_count != ticket.weight_count || header.transfer_count != ticket.transfer_count ||
            header.admissions > header.transfer_count || header.replacements > header.admissions) {
        ticket.status = GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_BINDING;
        return;
    }
    for (uint32_t route = 0; route < ticket.route_count; ++route) {
        const uint32_t weight = rows.routes[route].weight_index;
        if (weight >= ticket.weight_count || packet.classes[route] < 0 ||
                uint32_t(packet.classes[route]) != rows.weight_classes[weight]) {
            ticket.status = GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_BINDING;
            return;
        }
        const uint32_t kind = rows.weight_classes[weight], storage = rows.weight_storage[weight];
        if ((kind == GGML_CUDA_MOE_HYBRID_RESIDENT && packet.slots[route] != int32_t(storage)) ||
                (kind == GGML_CUDA_MOE_HYBRID_TRANSFER && (storage >= header.transfer_count ||
                 packet.rows[route] != int32_t(storage) || packet.transfers[storage] != rows.weight_experts[weight]))) {
            ticket.status = GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_BINDING;
            return;
        }
    }
}

static __global__ void ggml_cuda_moe_hybrid_rows_status(ggml_cuda_moe_hybrid_rows_view rows, const uint32_t * status) {
    if (*status != 0) { rows.ticket->status = GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_COMPUTE_FAILED; }
}

static ggml_cuda_moe_graph_group_dispatch * ggml_cuda_moe_hybrid_preflight_packet(
        ggml_cuda_moe_hybrid_session & session, ggml_cuda_moe_hybrid_region & region,
        ggml_cuda_moe_graph_execution & execution, ggml_tensor * first) {
    auto & ctx = *session.ctx;
    const auto stream = ctx.stream();
    const uint32_t top_k = ggml_cuda_moe_hybrid_route_capacity(region);
    const uint32_t quota = std::min(session.gpu_miss_quota, region.transfer_capacity);
    const uint32_t admission_quota = std::min(session.admission_quota, quota);
    ggml_cuda_moe_hybrid_control_layout control;
    if (!ggml_cuda_moe_hybrid_measure_control(session.route_capacity, control) || top_k > session.route_capacity ||
            session.packet_epoch == UINT64_MAX || region.resident_buckets.size() != 1) {
        return nullptr;
    }
    auto packet = ggml_cuda_moe_hybrid_packet(session.control_data(), top_k);
    region.combined_gpu = ggml_cuda_moe_hybrid_can_combine_gpu(session.window_active && session.window_combine_gpu,
        session.demand_admission && admission_quota == quota, region.resident_slots, top_k);
    region.direct_gather = region.combined_gpu && session.window_direct_gather && quota != 0;
    region.fused_experts = session.window_active && session.window_fuse_experts &&
        ggml_cuda_moe_hybrid_can_fuse_experts(region.nodes) &&
        ggml_cuda_moe_hybrid_can_fuse_experts(region.resident_buckets[0].nodes);
    packet.combine_gpu = region.combined_gpu;
    session.packet_epoch = std::max(session.packet_epoch, session.epoch);
    if (session.packet_epoch == UINT64_MAX) { return nullptr; }
    region.packet_epoch = ++session.packet_epoch;
    region.packet_status.store(0);
    region.packet_admissions.store(0);
    session.last_submit_us.store(0);
    session.last_cpu_start_us.store(0);
    session.last_cpu_done_us.store(0);
    session.last_gpu_enqueue_done_us.store(0);
    session.last_gpu_done_observed_us.store(0);
    session.last_join_start_us.store(0);
    session.last_join_done_us.store(0);
    ggml_cuda_moe_graph_binding binding;
    auto * group = execution.find_group(first, &binding);
    region.group = group;
    if (group == nullptr || group->first_reader != first || group->last_reader != region.descriptor.output ||
            group->key.ids.tensor != region.descriptor.ids || session.canceled.load() ||
            (!session.window_active && !session.hook(GGML_BACKEND_MOE_HYBRID_TEST_BEFORE_ROUTES, stream)) ||
            !ctx.moe_grouped_context->prepare_hybrid_group(group, stream, packet.header, packet.slots, packet.residents,
                session.demand_admission ? admission_quota : 0, packet, quota, region.packet_epoch,
                session.window_active ? region.host_runtime : nullptr, region.multi_rows ? top_k : 0)) {
        return nullptr;
    }
    if (group->n_slots != region.resident_slots || group->key.candidate.generation != region.resident_key.generation ||
            group->key.candidate.group_index != region.resident_key.group_index ||
            region.sources.size() != group->key.n_banks) {
        return nullptr;
    }
    auto & residents = region.resident_buckets[0];
    if (!region.multi_rows) {
        residents.tensors[1].data = packet.residents;
        region.tensors[1].data = packet.transfer_ids;
    }
    size_t admission_stride = 0;
    for (size_t bank = 0; bank < region.sources.size(); ++bank) {
        const auto & source = region.sources[bank];
        auto & weight = region.tensors[region.n_dynamic + bank];
        auto & resident = region.source_banks[bank];
        uint32_t index = 0;
        for (; index < group->key.n_banks; ++index) {
            if (!ctx.moe_grouped_context->get_group_resource_bank(group->transaction, index, &resident)) {
                return nullptr;
            }
            if (resident.tensor == source.witness) {
                break;
            }
        }
        if (index == group->key.n_banks || std::find(region.bank_indices.begin(), region.bank_indices.begin() + bank, index) !=
                region.bank_indices.begin() + bank || resident.source_data != source.data ||
                resident.type != uint32_t(weight.type) || resident.expert_stride != source.expert_stride ||
                resident.byte_extent != source.bytes || resident.ne[0] != weight.ne[0] || resident.ne[1] != weight.ne[1] ||
                resident.ne[2] != region.expert_count || resident.ne[3] != 1 || resident.nb[0] != weight.nb[0] ||
                resident.nb[1] != weight.nb[1] || resident.nb[2] != weight.nb[2] || group->bank_data[index] == nullptr ||
                resident.source_path >= MOE_GROUPED_SOURCE_PATH_COUNT ||
                (resident.source_path != MOE_GROUPED_SOURCE_PAGEABLE_STAGED &&
                    (resident.source_device_alias == nullptr || source.expert_stride % sizeof(uint4) != 0 ||
                     reinterpret_cast<uintptr_t>(resident.source_device_alias) % alignof(uint4) != 0))) {
            return nullptr;
        }
        region.bank_indices[bank] = index;
        region.direct_gather &= resident.source_device_alias != nullptr &&
            (resident.source_path == MOE_GROUPED_SOURCE_DIRECT_REGISTERED ||
             resident.source_path == MOE_GROUPED_SOURCE_MAPPED || resident.source_path == MOE_GROUPED_SOURCE_DEVICE);
        weight.data = session.data() + region.source_offsets[bank];
        residents.tensors[region.n_dynamic + bank].data = const_cast<void *>(group->bank_data[index]);
        if (source.expert_stride > SIZE_MAX - admission_stride) {
            return nullptr;
        }
        admission_stride += source.expert_stride;
    }
    region.admission_stride = admission_stride;
    if (session.window_active) {
        region.host_runtime->epoch = region.packet_epoch;
    }
    return group;
}

static bool ggml_cuda_moe_hybrid_execute_packet(
        ggml_cuda_moe_hybrid_session & session, ggml_cuda_moe_hybrid_region & region,
        ggml_cuda_moe_graph_execution & execution, ggml_tensor * first) try {
    auto & ctx = *session.ctx;
    const auto stream = ctx.stream();
    struct drain_guard {
        ggml_cuda_moe_hybrid_session & session;
        ggml_cuda_moe_hybrid_region & region;
        cudaStream_t stream;
        ggml_cuda_moe_graph_group_dispatch * group = nullptr;
        bool drained = false;
        ~drain_guard() {
            if (drained) {
                return;
            }
            session.cancel_pending();
            if (!session.wait_producers(session.io_stream, true)) {
                (void) cudaStreamSynchronize(session.io_stream);
            }
            if (!session.wait_producers(stream, true)) {
                (void) cudaStreamSynchronize(stream);
            }
            session.join(true);
            if (group != nullptr && group->transaction.transaction_token != 0) {
                (void) session.ctx->moe_grouped_context->finish_hybrid_admission(*group, false);
                session.admission_aborted += region.packet_admissions.load();
            }
        }
    } drain{session, region, stream};
    const uint32_t top_k = ggml_cuda_moe_hybrid_route_capacity(region);
    const uint32_t quota = std::min(session.gpu_miss_quota, region.transfer_capacity);
    const uint32_t admission_quota = std::min(session.admission_quota, quota);
    const bool pipeline_cpu = session.window_active && session.window_cpu_pipeline;
    ggml_cuda_moe_hybrid_control_layout control;
    if (!ggml_cuda_moe_hybrid_measure_control(session.route_capacity, control)) {
        return false;
    }
    drain.drained = session.window_active;
    auto * group = session.window_active ? region.group :
        ggml_cuda_moe_hybrid_preflight_packet(session, region, execution, first);
    drain.group = region.group;
    if (group == nullptr) {
        return false;
    }
    auto packet = ggml_cuda_moe_hybrid_packet(session.control_data(), top_k);
    packet.combine_gpu = region.combined_gpu;
    packet.direct_gather = region.direct_gather;
    auto host_packet = ggml_cuda_moe_hybrid_packet(session.host, top_k);
    const auto rows = region.multi_rows ? ggml_cuda_moe_hybrid_device_rows(region) : ggml_cuda_moe_hybrid_rows_view{};
    auto * rows_base = region.multi_rows ? session.data() + region.rows_offset : nullptr;
    auto * rows_plan = region.multi_rows ? rows_base + region.rows_layout.plan_offset : nullptr;
    auto * packed_input = region.multi_rows ? reinterpret_cast<float *>(rows_base + region.rows_layout.packed_input_offset) : nullptr;
    auto * gpu_output = region.multi_rows ? reinterpret_cast<float *>(rows_base + region.rows_layout.gpu_output_offset) : nullptr;
    auto * producer_status = reinterpret_cast<uint32_t *>(session.control_data() + control.consumer_status_offset);
    if (session.window_active) {
        packet.runtime = region.device_runtime;
        packet.window_failed = session.window_failed;
        if (!session.hook(GGML_BACKEND_MOE_HYBRID_TEST_BEFORE_ROUTES, stream)) { return false; }
        if (region.multi_rows) {
            ggml_cuda_moe_hybrid_rows_begin<<<1, 1, 0, stream>>>(rows, region.device_runtime, region.rows_layout.identity,
                region.rows_layout.geometry.row_capacity, region.rows_layout.sequence_capacity, quota);
            if (cudaGetLastError() != cudaSuccess ||
                    cudaMemcpy2DAsync(const_cast<int32_t *>(ggml_cuda_moe_hybrid_rows_ids(rows_plan, region.rows_layout.plan_capacity)),
                        size_t(region.rows_layout.geometry.routes_per_row) * sizeof(int32_t), region.descriptor.ids->data,
                        region.descriptor.ids->nb[1], size_t(region.rows_layout.geometry.routes_per_row) * sizeof(int32_t),
                        region.rows_layout.geometry.row_capacity, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
                return false;
            }
        }
        if (!ctx.moe_grouped_context->select_hybrid_group(*group, packet.header, packet.slots, packet.residents,
                    session.demand_admission ? admission_quota : 0, packet, quota, 0, 0, true,
                    region.multi_rows ? ggml_cuda_moe_hybrid_rows_ids(rows_plan, region.rows_layout.plan_capacity) : nullptr,
                    region.multi_rows ? top_k : 0)) {
            return false;
        }
        if (region.multi_rows) {
            // Selection settles the prior transaction before the immutable row plan reads maps.
            if (!ctx.moe_grouped_context->plan_hybrid_rows(*group, region.rows_layout, rows_plan, rows, session.window_failed)) {
                return false;
            }
            ggml_cuda_moe_hybrid_rows_check<<<1, 1, 0, stream>>>(rows, packet);
            if (cudaGetLastError() != cudaSuccess || !ggml_cuda_moe_hybrid_rows_pack(region.rows_layout, rows,
                    static_cast<const float *>(region.descriptor.activation->data), packed_input, stream)) {
                return false;
            }
        }
    } else if (region.multi_rows) {
        return false;
    }
    auto & residents = region.resident_buckets[0];
    const size_t admission_stride = region.admission_stride;
    const size_t packet_bytes = sizeof(*packet.header) + size_t(top_k) * GGML_CUDA_MOE_HYBRID_PACKET_ARRAYS * sizeof(int32_t);
    region.readback_started = ggml_time_us();
    if (!session.window_active) {
        memset(session.host + region.host_raw_offset, 0, ggml_nbytes(region.descriptor.output));
    }
    if (cudaMemsetAsync(producer_status, 0, sizeof(*producer_status), stream) != cudaSuccess ||
            (region.direct_gather && !ctx.moe_grouped_context->begin_hybrid_admissions(*group, packet, producer_status))) {
        return false;
    }
    if (cudaEventRecord(session.plan_ready, stream) != cudaSuccess ||
            cudaStreamWaitEvent(session.io_stream, session.plan_ready, 0) != cudaSuccess ||
            (region.multi_rows ? cudaMemcpyAsync(session.host + region.host_rows_offset, rows.ticket,
                reinterpret_cast<const uint8_t *>(rows.producers) - reinterpret_cast<const uint8_t *>(rows.ticket),
                cudaMemcpyDeviceToHost, session.io_stream) :
                cudaMemcpyAsync(session.host, packet.header, packet_bytes, cudaMemcpyDeviceToHost, session.io_stream)) != cudaSuccess ||
            cudaMemcpyAsync(session.host + region.host_input_offset, region.descriptor.activation->data,
                ggml_nbytes(region.descriptor.activation), cudaMemcpyDeviceToHost, session.io_stream) != cudaSuccess ||
            cudaLaunchHostFunc(session.io_stream, ggml_cuda_moe_hybrid_submit_packet, &region) != cudaSuccess ||
            cudaEventRecord(session.gpu_started, stream) != cudaSuccess) {
        return false;
    }
    const int64_t enqueue_started = ggml_time_us();
    const auto submit_body = [&](const std::vector<ggml_tensor *> & nodes, const uint32_t * count) {
        if (!session.window_active) {
            ++session.gpu_body_submissions;
        }
        size_t first_node = 0;
        if (region.fused_experts) {
            auto * glu = nodes[2];
            const auto * up = glu->src[1];
            ggml_cuda_mm_fusion_args_host fusion{};
            fusion.gate = glu->src[0]->src[0];
            fusion.glu_op = ggml_get_glu_op(glu);
            fusion.glu_limit = ggml_get_op_params_f32(glu, 3);
            if (cudaMemsetAsync(glu->data, 0, ggml_nbytes(glu), stream) != cudaSuccess ||
                    !ggml_cuda_mul_mat_vec_q_bounded(*session.compute, up->src[0], up->src[1], up->src[2],
                        glu, count, producer_status, &fusion)) {
                return false;
            }
            first_node = 3;
        }
        for (size_t i = first_node; i < nodes.size(); ++i) {
            auto * node = nodes[i];
            const bool mmid = node->op == GGML_OP_MUL_MAT_ID;
            const bool cleared = !mmid || cudaMemsetAsync(node->data, 0, ggml_nbytes(node), stream) == cudaSuccess;
            if (!cleared || !(mmid ? ggml_cuda_mul_mat_vec_q_bounded(*session.compute,
                    node->src[0], node->src[1], node->src[2], node, count, producer_status) :
                    ggml_cuda_is_view_or_noop(node) || ggml_cuda_compute_forward(*session.compute, node, nullptr))) {
                return false;
            }
        }
        return true;
    };
    if (!session.window_active) {
        ++session.resident_batches;
        ++session.resident_body_submissions;
    }
    if (region.multi_rows && !region.combined_gpu && !ggml_cuda_moe_hybrid_rows_gpu_binding(region.rows_layout, rows, GGML_CUDA_MOE_HYBRID_RESIDENT,
            packed_input, static_cast<float *>(residents.tensors[0].data), static_cast<int32_t *>(residents.tensors[1].data), stream)) {
        return false;
    }
    if (!region.combined_gpu && !submit_body(residents.nodes,
            region.multi_rows ? &rows.ticket->resident_lanes : &packet.header->resident_count)) {
        return false;
    }
    if (region.multi_rows && !region.combined_gpu && !ggml_cuda_moe_hybrid_rows_gpu_complete(region.rows_layout, rows, GGML_CUDA_MOE_HYBRID_RESIDENT,
            static_cast<const float *>(residents.nodes.back()->data), gpu_output, stream)) {
        return false;
    }
    if (region.direct_gather &&
            (!ctx.moe_grouped_context->gather_hybrid_admissions(*group, packet, producer_status, session.io_stream) ||
             (!session.hook(GGML_BACKEND_MOE_HYBRID_TEST_ADMISSION_BANK_COPIED, session.io_stream) &&
              cudaMemsetAsync(producer_status, 1, sizeof(*producer_status), session.io_stream) != cudaSuccess))) {
        return false;
    }
    for (size_t bank = 0; !region.direct_gather && bank < region.sources.size(); ++bank) {
        const auto & source = region.sources[bank];
        const auto & descriptor = region.source_banks[bank];
        auto * destination = session.data() + region.source_offsets[bank];
        if (descriptor.source_path != MOE_GROUPED_SOURCE_PAGEABLE_STAGED) {
            const size_t words = source.expert_stride / sizeof(uint4);
            ggml_cuda_moe_hybrid_gather<<<ggml_cuda_moe_hybrid_blocks(ctx.device, words * region.transfer_capacity), 256, 0, session.io_stream>>>(
                packet, region.packet_epoch, region.transfer_capacity, region.expert_count,
                static_cast<const uint4 *>(descriptor.source_device_alias), reinterpret_cast<uint4 *>(destination), words, producer_status);
            if (cudaGetLastError() != cudaSuccess) {
                return false;
            }
        } else {
            for (auto & copy : region.copies) {
                if (copy.bank != bank) {
                    continue;
                }
                const size_t offset = region.host_staging_offset + copy.tile * region.host_staging_bytes;
                auto * output = destination + size_t(copy.row) * source.expert_stride + copy.offset;
                if (cudaLaunchHostFunc(session.io_stream, ggml_cuda_moe_hybrid_fill_tile, &copy) != cudaSuccess) {
                    return false;
                }
                if (session.host_alias != nullptr) {
                    ggml_cuda_moe_hybrid_copy_tile<<<ggml_cuda_moe_hybrid_blocks(ctx.device, copy.bytes), 256, 0, session.io_stream>>>(
                        packet, region.packet_epoch, region.transfer_capacity, copy.row,
                        session.host_alias + offset, output, copy.bytes, producer_status);
                    if (cudaGetLastError() != cudaSuccess) {
                        return false;
                    }
                } else if (cudaMemcpyAsync(output, session.host + offset, copy.bytes,
                        cudaMemcpyHostToDevice, session.io_stream) != cudaSuccess) {
                    return false;
                }
            }
        }
        const size_t bytes = source.expert_stride * region.transfer_capacity;
        const size_t allocated = ggml_backend_buffer_get_alloc_size(session.arena, &region.tensors[region.n_dynamic + bank]);
        if (allocated > bytes && cudaMemsetAsync(destination + bytes, 0, allocated - bytes, session.io_stream) != cudaSuccess) {
            return false;
        }
    }
    if (!session.hook(GGML_BACKEND_MOE_HYBRID_TEST_TRANSFER_ENQUEUED, session.io_stream) ||
            cudaEventRecord(session.transfer_ready, session.io_stream) != cudaSuccess) {
        return false;
    }
    if (pipeline_cpu) {
        if (cudaLaunchHostFunc(session.io_stream, ggml_cuda_moe_hybrid_join_packet, &region) != cudaSuccess ||
                (session.host_alias == nullptr && cudaMemcpyAsync(session.data() + region.cpu_offset,
                    session.host + region.host_raw_offset, ggml_nbytes(region.descriptor.output),
                    cudaMemcpyHostToDevice, session.io_stream) != cudaSuccess) ||
                cudaEventRecord(session.cpu_ready, session.io_stream) != cudaSuccess) {
            return false;
        }
    }
    if (cudaStreamWaitEvent(stream, session.transfer_ready, 0) != cudaSuccess ||
            (region.multi_rows && !region.combined_gpu && !ggml_cuda_moe_hybrid_rows_gpu_binding(region.rows_layout, rows, GGML_CUDA_MOE_HYBRID_TRANSFER,
                packed_input, static_cast<float *>(region.tensors[0].data), static_cast<int32_t *>(region.tensors[1].data), stream)) ||
            (!region.combined_gpu && !submit_body(region.nodes,
                region.multi_rows ? &rows.ticket->transfer_lanes : &packet.header->transfer_count)) ||
            (region.multi_rows && !region.combined_gpu && !ggml_cuda_moe_hybrid_rows_gpu_complete(region.rows_layout, rows, GGML_CUDA_MOE_HYBRID_TRANSFER,
                static_cast<const float *>(region.nodes.back()->data), gpu_output, stream))) {
        return false;
    }
    uint64_t auxiliary_bytes = 0;
    if (session.demand_admission && quota != 0) {
        if (!region.direct_gather && !ctx.moe_grouped_context->begin_hybrid_admissions(*group, packet, producer_status)) {
            return false;
        }
        for (size_t bank = 0; !region.direct_gather && bank < region.sources.size(); ++bank) {
            if (!ctx.moe_grouped_context->copy_hybrid_admission_bank(*group, packet, region.bank_indices[bank],
                    session.data() + region.source_offsets[bank], producer_status)) {
                return false;
            }
            if (!session.hook(GGML_BACKEND_MOE_HYBRID_TEST_ADMISSION_BANK_COPIED, stream) &&
                    (!session.window_active || cudaMemsetAsync(producer_status, 1, sizeof(*producer_status), stream) != cudaSuccess)) {
                return false;
            }
        }
        if (!ctx.moe_grouped_context->complete_hybrid_admissions(*group, packet, producer_status, auxiliary_bytes)) {
            return false;
        }
    }
    if (region.combined_gpu) {
        if (region.multi_rows && !ggml_cuda_moe_hybrid_rows_gpu_combined_binding(region.rows_layout, rows, packet,
                packed_input, static_cast<float *>(residents.tensors[0].data),
                static_cast<int32_t *>(residents.tensors[1].data), stream)) {
            return false;
        }
        if (!submit_body(residents.nodes, region.multi_rows ? &rows.ticket->gpu_lanes : &packet.header->gpu_count)) {
            return false;
        }
        if (region.multi_rows && !ggml_cuda_moe_hybrid_rows_gpu_combined_complete(region.rows_layout, rows,
                static_cast<const float *>(residents.nodes.back()->data), gpu_output, stream)) {
            return false;
        }
    }
    session.gpu_enqueue_us += ggml_time_us() - enqueue_started;
    if (cudaEventRecord(session.gpu_done, stream) != cudaSuccess) {
        return false;
    }
    session.last_gpu_enqueue_done_us.store(ggml_time_us());
    if (!session.hook(GGML_BACKEND_MOE_HYBRID_TEST_GPU_ENQUEUED, stream) ||
            (pipeline_cpu && cudaStreamWaitEvent(stream, session.cpu_ready, 0) != cudaSuccess) ||
            cudaLaunchHostFunc(stream, pipeline_cpu ? ggml_cuda_moe_hybrid_complete_packet :
                ggml_cuda_moe_hybrid_join_packet, &region) != cudaSuccess) {
        return false;
    }
    const uint8_t * cpu_output = session.host_alias == nullptr ? session.data() + region.cpu_offset :
        session.host_alias + region.host_raw_offset;
    if (!pipeline_cpu && session.host_alias == nullptr && cudaMemcpyAsync(session.data() + region.cpu_offset,
            session.host + region.host_raw_offset, ggml_nbytes(region.descriptor.output), cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        return false;
    }
    if (region.multi_rows) {
        const auto host_rows = ggml_cuda_moe_hybrid_rows_packet(session.host + region.host_rows_offset, region.rows_layout.geometry);
        if (cudaMemcpyAsync(rows.producers + 2, host_rows.producers + 2,
                size_t(region.rows_layout.geometry.row_capacity) * sizeof(*rows.producers), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
                cudaMemcpyAsync(&rows.runtime->cancel_epoch, &host_rows.runtime->cancel_epoch,
                    sizeof(rows.runtime->cancel_epoch), cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            return false;
        }
        ggml_cuda_moe_hybrid_rows_status<<<1, 1, 0, stream>>>(rows, producer_status);
        if (cudaGetLastError() != cudaSuccess || !ggml_cuda_moe_hybrid_rows_commit(region.rows_layout, rows_plan, rows,
                gpu_output, reinterpret_cast<const float *>(cpu_output),
                reinterpret_cast<float *>(session.data() + region.raw_offset), stream)) {
            return false;
        }
    } else {
        ggml_cuda_moe_hybrid_assemble<<<ggml_cuda_moe_hybrid_blocks(ctx.device, ggml_nelements(region.descriptor.output)), 256, 0, stream>>>(
            packet, region.packet_epoch, region.transfer_capacity, region.descriptor.output->nb[1] / sizeof(float),
            static_cast<const float *>(residents.nodes.back()->data), static_cast<const float *>(region.nodes.back()->data),
            reinterpret_cast<const float *>(cpu_output), reinterpret_cast<float *>(session.data() + region.raw_offset), producer_status);
    }
    if (session.window_active) {
        region.admission_auxiliary_bytes = auxiliary_bytes;
        const size_t offset = offsetof(ggml_cuda_moe_hybrid_runtime, cpu_status);
        return cudaGetLastError() == cudaSuccess &&
            cudaMemcpyAsync(reinterpret_cast<uint8_t *>(region.device_runtime) + offset,
                reinterpret_cast<const uint8_t *>(region.host_runtime) + offset,
                sizeof(ggml_cuda_moe_hybrid_runtime) - offset, cudaMemcpyHostToDevice, stream) == cudaSuccess;
    }
    if (cudaGetLastError() != cudaSuccess || cudaMemcpyAsync(&host_packet.header->status, producer_status,
            sizeof(*producer_status), cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        return false;
    }
    const int64_t publication_started = ggml_time_us();
    const bool producers_done = session.wait_producers(stream);
    session.publication_us += ggml_time_us() - publication_started;
    if (!producers_done || host_packet.header->status != 0 || region.packet_status.load() != 0 ||
            session.fail_before_publish || !session.hook(GGML_BACKEND_MOE_HYBRID_TEST_BEFORE_PUBLISH) || session.canceled.load()) {
        return false;
    }
    session.last_gpu_done_observed_us.store(ggml_time_us());
    float gpu_ms = 0;
    if (cudaEventElapsedTime(&gpu_ms, session.gpu_started, session.gpu_done) != cudaSuccess) {
        return false;
    }
    session.gpu_branch_us += uint64_t(gpu_ms * 1000.0f);
    const uint32_t admissions = host_packet.header->admissions;
    if (admissions != 0) {
        if (!ctx.moe_grouped_context->finish_hybrid_admission(*group, true, false)) {
            return false;
        }
        session.admission_committed += admissions;
        session.admission_bytes += admissions * (admission_stride + auxiliary_bytes);
    }
    if (cudaMemcpyAsync(region.descriptor.output->data, session.data() + region.raw_offset,
            ggml_nbytes(region.descriptor.output), cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
        return false;
    }
    ++session.packet_regions;
    ++session.completed;
    drain.drained = true;
    return true;
} catch (const std::bad_alloc &) {
    ++session.capacity_errors;
    return false;
} catch (...) {
    return false;
}

static bool ggml_cuda_moe_hybrid_execute(
        ggml_cuda_moe_hybrid_session & session, ggml_cuda_moe_hybrid_region & region,
        ggml_cuda_moe_graph_execution & execution, ggml_tensor * first) try {
    if (session.resident_batch) {
        return ggml_cuda_moe_hybrid_execute_packet(session, region, execution, first);
    }
    auto & ctx = *session.ctx;
    const auto stream = ctx.stream();
    struct drain_guard {
        ggml_cuda_moe_hybrid_session & session;
        cudaStream_t stream;
        bool drained = false;
        ggml_cuda_moe_graph_group_dispatch * group = nullptr;
        uint32_t admissions = 0;
        ~drain_guard() {
            if (!drained) {
                session.join(true);
                (void) cudaStreamSynchronize(stream);
                (void) cudaStreamSynchronize(session.io_stream);
                if (admissions != 0) {
                    (void) session.ctx->moe_grouped_context->finish_hybrid_admission(*group, false);
                    session.admission_aborted += admissions;
                }
            }
        }
    } drain{session, stream};
    const uint32_t top_k = region.descriptor.ids->ne[0];
    ggml_cuda_moe_hybrid_control_layout control_layout;
    if (!ggml_cuda_moe_hybrid_measure_control(session.route_capacity, control_layout) || top_k > session.route_capacity) {
        return false;
    }
    auto * device_selection = reinterpret_cast<ggml_cuda_moe_hybrid_selection *>(session.control_data());
    auto * device_slots = reinterpret_cast<int32_t *>(session.control_data() + control_layout.slots_offset);
    auto * device_resident_slots = reinterpret_cast<int32_t *>(session.control_data() + control_layout.resident_slots_offset);
    auto * device_scalar_id = reinterpret_cast<int32_t *>(session.control_data() + control_layout.scalar_ids_offset);
    auto * device_transfer_ids = reinterpret_cast<int32_t *>(session.control_data() + control_layout.transfer_ids_offset);
    auto * device_consumer_status = reinterpret_cast<uint32_t *>(session.control_data() + control_layout.consumer_status_offset);
    const uint32_t gpu_miss_quota = std::min(session.gpu_miss_quota, top_k);
    const uint32_t admission_quota = std::min(session.admission_quota, gpu_miss_quota);
    ggml_cuda_moe_graph_binding binding;
    auto * group = execution.find_group(first, &binding);
    if (group == nullptr || group->first_reader != first || group->last_reader != region.descriptor.output ||
            group->key.ids.tensor != region.descriptor.ids || session.canceled.load() ||
            !session.hook(GGML_BACKEND_MOE_HYBRID_TEST_BEFORE_ROUTES, stream) ||
            !ctx.moe_grouped_context->prepare_hybrid_group(group, stream,
                device_selection, device_slots, device_resident_slots,
                session.demand_admission ? admission_quota : 0)) {
        return false;
    }
    ggml_cuda_moe_hybrid_bucket * resident_bucket = nullptr;
    if (session.resident_batch) {
        if (group->n_slots != region.resident_slots || group->key.candidate.generation != region.resident_key.generation ||
                group->key.candidate.group_index != region.resident_key.group_index ||
                region.resident_buckets.size() != 1) {
            return false;
        }
        resident_bucket = &region.resident_buckets[0];
        auto & resident_ids = resident_bucket->tensors[1];
        resident_ids.ne[0] = top_k;
        resident_ids.nb[1] = ggml_row_size(resident_ids.type, resident_ids.ne[0]);
        resident_ids.nb[2] = resident_ids.nb[3] = resident_ids.nb[1];
        resident_ids.data = device_resident_slots;
        for (auto * node : resident_bucket->nodes) {
            if (node->nb[1] > SIZE_MAX / top_k) {
                return false;
            }
            node->ne[1] = top_k;
            node->ne[2] = node->ne[3] = 1;
            node->nb[2] = node->nb[3] = node->nb[1] * top_k;
        }
        uint32_t bank_mask = 0;
        for (size_t bank = 0; bank < region.sources.size(); ++bank) {
            const auto & source = region.sources[bank];
            auto & weight = resident_bucket->tensors[region.n_dynamic + bank];
            bool found = false;
            for (uint32_t resident_bank = 0; resident_bank < group->key.n_banks; ++resident_bank) {
                ggml_cuda_moe_grouped_bank_descriptor resident;
                if (!ctx.moe_grouped_context->get_group_resource_bank(group->transaction, resident_bank, &resident)) {
                    return false;
                }
                if (resident.tensor != source.witness) {
                    continue;
                }
                if ((bank_mask & (1u << resident_bank)) != 0 || resident.source_data != source.data ||
                        resident.type != uint32_t(weight.type) || resident.expert_stride != source.expert_stride ||
                        resident.byte_extent != source.bytes || resident.ne[0] != weight.ne[0] ||
                        resident.ne[1] != weight.ne[1] || resident.ne[2] != region.expert_count || resident.ne[3] != 1 ||
                        resident.nb[0] != weight.nb[0] || resident.nb[1] != weight.nb[1] || resident.nb[2] != weight.nb[2] ||
                        group->bank_data[resident_bank] == nullptr) {
                    return false;
                }
                bank_mask |= 1u << resident_bank;
                weight.data = const_cast<void *>(group->bank_data[resident_bank]);
                found = true;
                break;
            }
            if (!found) {
                return false;
            }
        }
        if (bank_mask != (1u << group->key.n_banks) - 1 || resident_bucket->nodes.back()->ne[1] != top_k ||
                resident_bucket->nodes.back()->nb[1] != region.descriptor.output->nb[1]) {
            return false;
        }
    }
    const size_t route_bytes = region.descriptor.output->nb[1];
    auto * selection = reinterpret_cast<ggml_cuda_moe_hybrid_selection *>(session.host);
    auto * selection_slots = reinterpret_cast<int32_t *>(session.host + region.host_slots_offset);
    auto * ids = reinterpret_cast<int32_t *>(session.host + region.host_ids_offset);
    const int64_t readback_started = ggml_time_us();
    const bool readback_submitted = cudaEventRecord(session.plan_ready, stream) == cudaSuccess &&
        cudaStreamWaitEvent(session.io_stream, session.plan_ready, 0) == cudaSuccess &&
        cudaMemcpyAsync(selection, device_selection, sizeof(*selection), cudaMemcpyDeviceToHost, session.io_stream) == cudaSuccess &&
        cudaMemcpyAsync(selection_slots, device_slots, top_k * sizeof(int32_t), cudaMemcpyDeviceToHost, session.io_stream) == cudaSuccess &&
        cudaMemcpyAsync(ids, region.descriptor.ids->data, top_k * sizeof(int32_t), cudaMemcpyDeviceToHost, session.io_stream) == cudaSuccess &&
            cudaMemcpyAsync(session.host + region.host_input_offset, region.descriptor.activation->data,
                ggml_nbytes(&region.tensors[0]), cudaMemcpyDeviceToHost, session.io_stream) == cudaSuccess;
    if (!readback_submitted || cudaMemsetAsync(device_scalar_id, 0, sizeof(int32_t), stream) != cudaSuccess ||
            cudaMemsetAsync(device_consumer_status, 0, sizeof(uint32_t), stream) != cudaSuccess ||
            cudaEventRecord(session.gpu_started, stream) != cudaSuccess) {
        return false;
    }
    if (resident_bucket != nullptr) {
        ++session.resident_batches;
        ++session.resident_body_submissions;
        ++session.gpu_body_submissions;
        for (auto * node : resident_bucket->nodes) {
            const bool mmid = node->op == GGML_OP_MUL_MAT_ID;
            const bool cleared = !mmid || cudaMemsetAsync(node->data, 0, ggml_nbytes(node), stream) == cudaSuccess;
            const bool ok = cleared && (mmid ?
                ggml_cuda_mul_mat_vec_q_bounded(*session.compute, node->src[0], node->src[1], node->src[2], node,
                    &device_selection->resident_count, device_consumer_status) :
                (ggml_cuda_is_view_or_noop(node) || ggml_cuda_compute_forward(*session.compute, node, nullptr)));
            if (!ok) {
                return false;
            }
        }
    }
    const bool readback_ok = cudaStreamSynchronize(session.io_stream) == cudaSuccess;
    session.readback_us += ggml_time_us() - readback_started;
    if (!readback_ok || selection->status != 0) {
        return false;
    }
    drain.group = group;
    drain.admissions = selection->admissions;
    if (selection->admissions > top_k || selection->admissions > gpu_miss_quota ||
            selection->replacements > selection->admissions || region.sources.size() != group->key.n_banks) {
        return false;
    }
    session.admission_reserved += selection->admissions;
    session.admission_replacements += selection->replacements;
    auto * distinct_ids = region.distinct_ids.data();
    auto * slots = region.slots.data();
    auto * cpu_ids = region.cpu_ids.data();
    auto * classes = region.classes.data();
    auto * first_ranks = region.first_ranks.data();
    auto * fanout = region.fanout.data();
    auto * cpu_ranks = region.cpu_ranks.data();
    auto * resident_rows = region.resident_rows.data();
    auto * transfer_rows = region.transfer_rows.data();
    uint32_t n_distinct = 0, n_transfer = 0, n_cpu = 0, n_resident = 0;
    for (uint32_t rank = 0; rank < top_k; ++rank) {
        if (ids[rank] < 0 || uint32_t(ids[rank]) >= region.expert_count) {
            return false;
        }
        uint32_t expert = 0;
        while (expert < n_distinct && distinct_ids[expert] != ids[rank]) {
            ++expert;
        }
        if (expert == n_distinct) {
            distinct_ids[expert] = ids[rank];
            slots[expert] = selection_slots[rank];
            first_ranks[expert] = rank;
            if (slots[expert] >= 0) {
                if (uint32_t(slots[expert]) >= group->n_slots) {
                    return false;
                }
                classes[expert] = 0;
                resident_rows[expert] = n_resident;
                ++n_resident;
                ++session.resident_experts;
            } else if (n_transfer < gpu_miss_quota) {
                classes[expert] = 1;
                transfer_rows[expert] = n_transfer;
                ++n_transfer;
                ++session.transfer_experts;
            } else {
                classes[expert] = 2;
                cpu_ids[n_cpu] = ids[rank];
                cpu_ranks[n_cpu++] = rank;
                ++session.cpu_experts;
            }
            ++n_distinct;
        } else if (slots[expert] != selection_slots[rank]) {
            return false;
        }
        fanout[rank] = expert;
        session.resident_routes += classes[expert] == 0;
        session.transferred_routes += classes[expert] == 1;
        session.cpu_routes += classes[expert] == 2;
    }
    if (selection->admissions > n_transfer) {
        return false;
    }
    if (session.demand_admission) {
        session.admission_no_slot += n_transfer - selection->admissions;
    }
    session.distinct_experts += n_distinct;
    session.last_submit_us.store(0);
    session.last_cpu_start_us.store(0);
    session.last_cpu_done_us.store(0);
    session.last_gpu_enqueue_done_us.store(0);
    session.last_gpu_done_observed_us.store(0);
    session.last_join_start_us.store(0);
    session.last_join_done_us.store(0);
    if (n_cpu != 0 && !session.submit(region, cpu_ids, cpu_ranks, n_cpu)) {
        return false;
    }
    const int64_t gpu_started = ggml_time_us();
    if (resident_bucket != nullptr) {
        if (session.canceled.load() || selection->resident_count != n_resident) {
            return false;
        }
        const auto * output = static_cast<const uint8_t *>(resident_bucket->nodes.back()->data);
        for (uint32_t rank = 0; rank < top_k; ++rank) {
            const uint32_t expert = fanout[rank];
            if (classes[expert] == 0 && cudaMemcpyAsync(session.data() + region.raw_offset + rank * route_bytes,
                    output + resident_rows[expert] * route_bytes, route_bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
                return false;
            }
        }
    }
    auto & source_banks = region.source_banks;
    auto & bank_indices = region.bank_indices;
    uint32_t bank_mask = 0;
    if (region.sources.size() > source_banks.size() || n_transfer > region.transfer_capacity) {
        return false;
    }
    for (size_t bank = 0; bank < region.sources.size(); ++bank) {
        const auto & source = region.sources[bank];
        const auto & weight = region.tensors[region.n_dynamic + bank];
        auto & resident = source_banks[bank];
        uint32_t resident_bank = 0;
        for (; resident_bank < group->key.n_banks; ++resident_bank) {
            if (!ctx.moe_grouped_context->get_group_resource_bank(group->transaction, resident_bank, &resident)) {
                return false;
            }
            if (resident.tensor == source.witness) {
                break;
            }
        }
        if (resident_bank == group->key.n_banks || (bank_mask & (1u << resident_bank)) != 0 ||
                resident.source_data != source.data || resident.type != uint32_t(weight.type) ||
                resident.expert_stride != source.expert_stride || resident.byte_extent != source.bytes ||
                resident.ne[0] != weight.ne[0] || resident.ne[1] != weight.ne[1] ||
                resident.ne[2] != region.expert_count || resident.ne[3] != 1 ||
                resident.nb[0] != weight.nb[0] || resident.nb[1] != weight.nb[1] || resident.nb[2] != weight.nb[2] ||
                group->bank_data[resident_bank] == nullptr) {
            return false;
        }
        bank_indices[bank] = resident_bank;
        bank_mask |= 1u << resident_bank;
    }
    if (bank_mask != (1u << group->key.n_banks) - 1) {
        return false;
    }
    const auto shape_body = [&](uint32_t count, int32_t * body_ids) {
        if (count == 0 || count > region.transfer_capacity) {
            return false;
        }
        auto & compact_ids = region.tensors[1];
        compact_ids.ne[0] = count;
        compact_ids.nb[1] = compact_ids.nb[2] = compact_ids.nb[3] = count * sizeof(int32_t);
        compact_ids.data = body_ids;
        for (size_t bank = 0; bank < region.sources.size(); ++bank) {
            auto & weight = region.tensors[region.n_dynamic + bank];
            weight.ne[2] = count;
            weight.nb[3] = weight.nb[2] * count;
        }
        for (auto * node : region.nodes) {
            node->ne[1] = count;
            node->ne[2] = node->ne[3] = 1;
            node->nb[2] = node->nb[3] = node->nb[1] * count;
        }
        return true;
    };
    const auto submit_body = [&]() {
        if (session.canceled.load()) {
            return false;
        }
        ++session.gpu_body_submissions;
        for (auto * node : region.nodes) {
            const bool ok = node->op == GGML_OP_MUL_MAT_ID ?
                ggml_cuda_mul_mat_id_impl(*session.compute, node, false, nullptr, GGML_CUDA_MMID_CONSUMER_MMVQ) :
                ggml_cuda_is_view_or_noop(node) || ggml_cuda_compute_forward(*session.compute, node, nullptr);
            if (!ok) {
                return false;
            }
        }
        return true;
    };
    if (!session.resident_batch) {
        if (!shape_body(1, device_scalar_id)) {
            return false;
        }
        for (uint32_t expert = 0; expert < n_distinct; ++expert) {
            if (classes[expert] != 0) {
                continue;
            }
            for (size_t bank = 0; bank < region.sources.size(); ++bank) {
                region.tensors[region.n_dynamic + bank].data = const_cast<uint8_t *>(
                    static_cast<const uint8_t *>(group->bank_data[bank_indices[bank]])) + size_t(slots[expert]) * region.sources[bank].expert_stride;
            }
            ++session.resident_body_submissions;
            if (!submit_body()) {
                return false;
            }
            for (uint32_t rank = 0; rank < top_k; ++rank) {
                if (fanout[rank] == expert && cudaMemcpyAsync(session.data() + region.raw_offset + rank * route_bytes,
                        region.nodes.back()->data, route_bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
                    return false;
                }
            }
        }
    }
    if (n_transfer != 0) {
        if (!shape_body(n_transfer, device_transfer_ids)) {
            return false;
        }
        for (uint32_t row = 0; row < n_transfer; ++row) {
            selection_slots[row] = row;
        }
        if (cudaMemcpyAsync(device_transfer_ids, selection_slots, n_transfer * sizeof(int32_t),
                cudaMemcpyHostToDevice, session.io_stream) != cudaSuccess) {
            return false;
        }
        for (size_t bank = 0; bank < region.sources.size(); ++bank) {
            auto & weight = region.tensors[region.n_dynamic + bank];
            const auto & source = region.sources[bank];
            weight.data = session.data() + region.source_offsets[bank];
            for (uint32_t expert = 0; expert < n_distinct; ++expert) {
                if (classes[expert] == 1 && (session.canceled.load() || !session.copy_source(
                        static_cast<uint8_t *>(weight.data) + size_t(transfer_rows[expert]) * source.expert_stride,
                        static_cast<const uint8_t *>(source.data) + size_t(distinct_ids[expert]) * source.expert_stride,
                        source.expert_stride, source_banks[bank].source_path, region.host_staging_offset, region.host_staging_bytes))) {
                    return false;
                }
            }
            const size_t bytes = source.expert_stride * n_transfer;
            const size_t allocated = ggml_backend_buffer_get_alloc_size(session.arena, &weight);
            if (allocated > bytes && cudaMemsetAsync(static_cast<uint8_t *>(weight.data) + bytes, 0,
                    allocated - bytes, session.io_stream) != cudaSuccess) {
                return false;
            }
        }
        if (!session.hook(GGML_BACKEND_MOE_HYBRID_TEST_TRANSFER_ENQUEUED, session.io_stream) ||
                cudaEventRecord(session.transfer_ready, session.io_stream) != cudaSuccess ||
                cudaStreamWaitEvent(stream, session.transfer_ready, 0) != cudaSuccess) {
            return false;
        }
        if (!submit_body()) {
            return false;
        }
        const auto * output = static_cast<const uint8_t *>(region.nodes.back()->data);
        for (uint32_t rank = 0; rank < top_k; ++rank) {
            const uint32_t expert = fanout[rank];
            if (classes[expert] == 1 && cudaMemcpyAsync(session.data() + region.raw_offset + rank * route_bytes,
                    output + size_t(transfer_rows[expert]) * route_bytes, route_bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
                return false;
            }
        }
        for (uint32_t expert = 0; expert < n_distinct; ++expert) {
            if (classes[expert] != 1 || slots[expert] > -2) {
                continue;
            }
            const int32_t admission_slot = -2 - slots[expert];
            if (!ctx.moe_grouped_context->begin_hybrid_admission(*group, distinct_ids[expert], admission_slot)) {
                return false;
            }
            for (size_t bank = 0; bank < region.sources.size(); ++bank) {
                const auto & source = region.sources[bank];
                auto * destination = const_cast<uint8_t *>(static_cast<const uint8_t *>(group->bank_data[bank_indices[bank]])) +
                    size_t(admission_slot) * source.expert_stride;
                const auto * data = session.data() + region.source_offsets[bank] + size_t(transfer_rows[expert]) * source.expert_stride;
                if (cudaMemcpyAsync(destination, data, source.expert_stride, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
                    return false;
                }
                session.admission_bytes += source.expert_stride;
                if (!session.hook(GGML_BACKEND_MOE_HYBRID_TEST_ADMISSION_BANK_COPIED, stream)) {
                    return false;
                }
            }
            uint64_t auxiliary_bytes = 0;
            if (!ctx.moe_grouped_context->complete_hybrid_admission(*group, distinct_ids[expert], admission_slot, auxiliary_bytes)) {
                return false;
            }
            session.admission_bytes += auxiliary_bytes;
        }
    }
    session.gpu_enqueue_us += ggml_time_us() - gpu_started;
    if (cudaEventRecord(session.gpu_done, stream) != cudaSuccess) {
        return false;
    }
    session.last_gpu_enqueue_done_us.store(ggml_time_us());
    if (!session.hook(GGML_BACKEND_MOE_HYBRID_TEST_GPU_ENQUEUED, stream) || !session.join(false) || session.canceled.load()) {
        return false;
    }
    for (uint32_t rank = 0; rank < top_k; ++rank) {
        const uint32_t expert = fanout[rank];
        if (classes[expert] != 2) {
            continue;
        }
        if (rank != first_ranks[expert]) {
            memcpy(session.host + region.host_raw_offset + rank * route_bytes,
                session.host + region.host_raw_offset + first_ranks[expert] * route_bytes, route_bytes);
        }
        if (cudaMemcpyAsync(session.data() + region.raw_offset + rank * route_bytes,
                session.host + region.host_raw_offset + rank * route_bytes, route_bytes,
                cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            return false;
        }
        session.cpu_upload_bytes += route_bytes;
    }
    if (resident_bucket != nullptr && cudaMemcpyAsync(&selection->status, device_consumer_status,
            sizeof(selection->status), cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        return false;
    }
    int64_t publication_started = ggml_time_us();
    const bool producers_done = cudaStreamSynchronize(stream) == cudaSuccess;
    session.publication_us += ggml_time_us() - publication_started;
    if (!producers_done || selection->status != 0 || session.fail_before_publish ||
            !session.hook(GGML_BACKEND_MOE_HYBRID_TEST_BEFORE_PUBLISH) || session.canceled.load()) {
        return false;
    }
    session.last_gpu_done_observed_us.store(ggml_time_us());
    float gpu_ms = 0;
    if (cudaEventElapsedTime(&gpu_ms, session.gpu_started, session.gpu_done) != cudaSuccess) {
        return false;
    }
    session.gpu_branch_us += uint64_t(gpu_ms * 1000.0f);
    if (drain.admissions != 0) {
        const int64_t admission_started = ggml_time_us();
        const bool admitted = ctx.moe_grouped_context->finish_hybrid_admission(*group, true);
        session.admission_fence_us += ggml_time_us() - admission_started;
        if (!admitted) {
            drain.admissions = 0;
            return false;
        }
        session.admission_committed += drain.admissions;
        drain.admissions = 0;
    }
    publication_started = ggml_time_us();
    const bool published = cudaMemcpyAsync(region.descriptor.output->data, session.data() + region.raw_offset,
        ggml_nbytes(region.descriptor.output), cudaMemcpyDeviceToDevice, stream) == cudaSuccess &&
        cudaStreamSynchronize(stream) == cudaSuccess;
    session.publication_us += ggml_time_us() - publication_started;
    if (!published) {
        return false;
    }
    ++session.completed;
    drain.drained = true;
    return true;
} catch (const std::bad_alloc &) {
    ++session.capacity_errors;
    return false;
} catch (...) {
    return false;
}

static void ggml_cuda_moe_hybrid_destroy_region(void * opaque, void * prepared) {
    auto * session = static_cast<ggml_cuda_moe_hybrid_session *>(opaque);
    auto * region = static_cast<ggml_cuda_moe_hybrid_region *>(prepared);
    if (session != nullptr) {
        session->window_unavailable = false;
#ifdef GGML_CUDA_MOE_HYBRID_WINDOW
        session->window_graph.reset();
#endif
        const auto found = std::find(session->prepared_regions.begin(), session->prepared_regions.end(), region);
        if (found != session->prepared_regions.end()) {
            session->prepared_regions.erase(found);
        }
    }
    delete region;
}

#ifdef GGML_CUDA_MOE_HYBRID_WINDOW
static __global__ void ggml_cuda_moe_hybrid_decide(ggml_cuda_moe_hybrid_packet_view packet,
        const uint32_t * producer_status, cudaGraphConditionalHandle handle,
        const ggml_cuda_moe_hybrid_rows_ticket * rows = nullptr) {
    auto & runtime = *packet.runtime;
    const auto & header = *packet.header;
    const bool accepted = rows != nullptr ? (*packet.window_failed == 0 && runtime.started &&
        rows->accepted && rows->published && rows->commit_decision == 2 && rows->status == 0 &&
        rows->epoch == runtime.epoch && rows->published_routes == packet.capacity && *producer_status == 0) :
        *packet.window_failed == 0 && runtime.started &&
        header.status == 0 && header.epoch == runtime.epoch && header.route_count == packet.capacity &&
        header.admissions <= header.transfer_count && header.distinct_count <= packet.capacity &&
        uint64_t(header.resident_count) + header.transfer_count + header.cpu_count == header.distinct_count &&
        header.gpu_count <= packet.capacity && header.gpu_count == uint64_t(header.resident_count) + header.transfer_count &&
        (!packet.combine_gpu || header.admissions == header.transfer_count) &&
        *producer_status == 0 && runtime.cpu_status == 0 && runtime.cpu_epoch == runtime.epoch &&
        runtime.cpu_routes == header.cpu_count;
    runtime.accepted = accepted;
    runtime.admissions = header.admissions <= packet.capacity ? header.admissions : 0;
    if (!accepted) {
        *packet.window_failed = 1;
    }
    cudaGraphSetConditional(handle, accepted);
}

static __global__ void ggml_cuda_moe_hybrid_continue(const ggml_cuda_moe_hybrid_runtime * runtime,
        cudaGraphConditionalHandle handle) {
    cudaGraphSetConditional(handle, runtime->accepted);
}

static bool ggml_cuda_moe_hybrid_emit_range(ggml_cuda_moe_hybrid_session & session, ggml_cgraph * graph,
        ggml_cuda_moe_graph_execution & execution, uint32_t first, uint32_t end) {
    GGML_ASSERT(first <= end && end <= uint32_t(graph->n_nodes));
    auto range = ggml_graph_view(graph, first, end);
    range.uid = graph->uid;
    range.execution_certificate = graph->execution_certificate;
    const bool log_fusion = getenv("GGML_MOE_HYBRID_FUSION_LOG") != nullptr;
    for (int i = 0; i < range.n_nodes; ++i) {
        auto * node = range.nodes[i];
        if (ggml_cuda_is_view_or_noop(node) || !(node->flags & GGML_TENSOR_FLAG_COMPUTE)) {
            continue;
        }
        // The view bounds lookahead and keeps use counts from the complete graph.
        const int skip = ggml_cuda_try_fuse(session.ctx, &range, i, &execution);
        if (skip < 0) {
            return false;
        }
        if (skip > 0) {
            GGML_ASSERT(skip < range.n_nodes - i);
            if (log_fusion) {
                fprintf(stderr, "moe-hybrid-fusion: range=%u:%u node=%u skip=%d type=%s src0=%s ne=%lld,%lld,%lld,%lld ops=",
                    first, end, first + i, skip, ggml_type_name(node->type),
                    node->src[0] == nullptr ? "none" : ggml_type_name(node->src[0]->type),
                    (long long) node->ne[0], (long long) node->ne[1], (long long) node->ne[2], (long long) node->ne[3]);
                for (int j = 0; j <= skip; ++j) {
                    fprintf(stderr, "%s%s", j == 0 ? "" : ",", ggml_op_desc(range.nodes[i + j]));
                }
                fprintf(stderr, "\n");
            }
            session.window_fused_nodes += skip;
            i += skip;
            continue;
        }
        if (!ggml_cuda_compute_forward(*session.ctx, node, &execution)) {
            return false;
        }
    }
    return true;
}
#endif

static int ggml_cuda_moe_hybrid_execute_window(ggml_cuda_moe_hybrid_session & session,
        ggml_cgraph * graph, ggml_cuda_moe_graph_execution & execution) try {
#ifndef GGML_CUDA_MOE_HYBRID_WINDOW
    GGML_UNUSED(session);
    GGML_UNUSED(graph);
    GGML_UNUSED(execution);
    return 0;
#else
    if (session.window_unavailable || !session.resident_batch || session.n_regions == 0 ||
            getenv("GGML_CUDA_DISABLE_GRAPHS") != nullptr) {
        return 0;
    }
    const uint32_t count = session.n_regions;
    const auto stream = session.ctx->stream();
    auto & owner = *session.ctx->moe_grouped_context;
    size_t input_bytes = 0;
    const auto first_tail = static_cast<ggml_cuda_moe_hybrid_region *>(session.regions[0])->descriptor.last_node + 1;
    for (uint32_t i = first_tail; i < uint32_t(graph->n_nodes); ++i) {
        if ((graph->nodes[i]->flags & GGML_TENSOR_FLAG_COMPUTE) && ggml_cuda_staged_input_supports(graph->nodes[i])) {
            input_bytes = std::max(input_bytes, ggml_nbytes(graph->nodes[i]));
        }
    }
    const size_t window_bytes = std::max(session.window_bytes,
        size_t(count) * sizeof(ggml_cuda_moe_hybrid_runtime) + sizeof(uint64_t));
    const size_t input_capacity = std::max(input_bytes, session.window_input_bytes);
    const size_t existing_bytes = session.control_bytes + session.device_bytes + session.work_bytes;
    if (session.device_limit != 0 && (window_bytes > session.device_limit ||
            input_capacity > session.device_limit - window_bytes ||
            existing_bytes > session.device_limit - window_bytes - input_capacity)) {
        return 0;
    }
    if (input_bytes > session.window_input_bytes) {
        session.window_graph.reset();
        if (session.window_input != nullptr) {
            CUDA_CHECK(cudaFree(session.window_input));
            session.window_input = nullptr;
            session.window_input_bytes = 0;
        }
        const auto allocated = cudaMalloc(&session.window_input, input_bytes);
        if (allocated != cudaSuccess) {
            fprintf(stderr, "moe-hybrid-window: input allocation failed, bytes=%zu: %s\n", input_bytes, cudaGetErrorString(allocated));
            (void) cudaGetLastError();
            return -1;
        }
        session.window_input_bytes = input_bytes;
    }
    if (session.window_capacity < count) {
        const size_t bytes = size_t(count) * sizeof(ggml_cuda_moe_hybrid_runtime) + sizeof(uint64_t);
        const size_t device_bytes = session.control_bytes + session.device_bytes + session.work_bytes;
        if ((session.device_limit != 0 && (bytes > session.device_limit || device_bytes > session.device_limit - bytes)) ||
                (session.pinned_limit != 0 && (bytes > session.pinned_limit || session.pinned_bytes > session.pinned_limit - bytes))) {
            return 0;
        }
        session.window_graph.reset();
        if (session.window_device != nullptr) {
            CUDA_CHECK(cudaFree(session.window_device));
            session.window_device = nullptr;
        }
        if (session.window_host != nullptr) {
            CUDA_CHECK(cudaFreeHost(session.window_host));
            session.window_host = nullptr;
        }
        if (cudaMalloc(&session.window_device, bytes) != cudaSuccess || cudaMallocHost(&session.window_host, bytes) != cudaSuccess) {
            fprintf(stderr, "moe-hybrid-window: control allocation failed, bytes=%zu\n", bytes);
            (void) cudaGetLastError();
            return -1;
        }
        session.window_capacity = count;
        session.window_bytes = bytes;
        session.window_failed = reinterpret_cast<uint32_t *>(session.window_device + count);
        session.window_host_failed = reinterpret_cast<uint32_t *>(session.window_host + count);
    }
    for (uint32_t r = 0; r < session.window_capacity; ++r) {
        session.window_host[r] = {};
    }
    *session.window_host_failed = 0;
    session.window_active = true;
    struct active_guard {
        ggml_cuda_moe_hybrid_session & session;
        ~active_guard() { session.window_active = false; }
    } active{session};
    uint64_t identity = 1469598103934665603ULL;
    const auto add_identity = [&](const auto & value) {
        const auto * bytes = reinterpret_cast<const uint8_t *>(&value);
        for (size_t i = 0; i < sizeof(value); ++i) {
            identity = (identity ^ bytes[i]) * 1099511628211ULL;
        }
    };
    add_identity(graph->execution_certificate);
    add_identity(session.ctx->workspace_generation);
    add_identity(session.host);
    add_identity(session.arena);
    add_identity(session.control);
    add_identity(session.window_input);
    std::vector<uint32_t> original_strategies;
    original_strategies.reserve(count);
    uint32_t combined_count = 0;
    uint32_t direct_count = 0;
    uint32_t fused_bodies = 0;
    for (uint32_t r = 0; r < count; ++r) {
        auto & region = *static_cast<ggml_cuda_moe_hybrid_region *>(session.regions[r]);
        if (r != 0 && region.descriptor.first_node <=
                static_cast<ggml_cuda_moe_hybrid_region *>(session.regions[r - 1])->descriptor.last_node) {
            fprintf(stderr, "moe-hybrid-window: invalid region order, region=%u first=%u\n", r, region.descriptor.first_node);
            return -1;
        }
        region.host_runtime = session.window_host + r;
        region.device_runtime = session.window_device + r;
        const auto * planned = execution.find_group(graph->nodes[region.descriptor.first_node], nullptr);
        if (planned == nullptr) {
            fprintf(stderr, "moe-hybrid-window: missing planned group, region=%u\n", r);
            return -1;
        }
        original_strategies.push_back(planned->strategy);
        if (ggml_cuda_moe_hybrid_preflight_packet(session, region, execution, graph->nodes[region.descriptor.first_node]) == nullptr) {
            fprintf(stderr, "moe-hybrid-window: grouped preflight failed, region=%u\n", r);
            return -1;
        }
        add_identity(region.descriptor.allocator_generation);
        add_identity(region.descriptor.owner_generation);
        add_identity(region.descriptor.geometry);
        add_identity(region.multi_rows);
        add_identity(region.rows_layout.identity);
        add_identity(region.graph_generation);
        add_identity(region.source_generation);
        add_identity(region.combined_gpu);
        combined_count += region.combined_gpu;
        add_identity(region.direct_gather);
        direct_count += region.direct_gather;
        add_identity(region.fused_experts);
        fused_bodies += region.fused_experts ? (region.combined_gpu ? 1 : 2) : 0;
        add_identity(session.regions[r]);
    }
    std::vector<std::shared_ptr<void>> leases;
    uint64_t resources = 0;
    if (!owner.graph_resource_fingerprint(execution, stream, &resources, &leases)) {
        fprintf(stderr, "moe-hybrid-window: resource fingerprint failed, regions=%u\n", count);
        return -1;
    }
    add_identity(resources);
    bool changed = session.window_graph == nullptr || session.window_identity != identity ||
        session.window_graph->node_props.size() != size_t(graph->n_nodes);
    for (int i = 0; !changed && i < graph->n_nodes; ++i) {
        const auto properties = ggml_cuda_graph_node_properties(graph->nodes[i]);
        changed = memcmp(&properties, &session.window_graph->node_props[i], sizeof(properties)) != 0;
    }
    if (changed) {
        session.window_graph.reset();
        auto window = std::make_unique<ggml_cuda_graph>();
        auto status = cudaGraphCreate(&window->graph, 0);
        const char * phase = "create";
        uint32_t capture_region = UINT32_MAX;
        std::vector<cudaGraphNode_t> dependencies;
        const auto first = static_cast<ggml_cuda_moe_hybrid_region *>(session.regions[0])->descriptor.first_node;
        if (status == cudaSuccess) {
            phase = "prefix";
            status = ggml_cuda_moe_hybrid_capture(window->graph, stream, dependencies, [&] {
                const auto copied = cudaMemcpyAsync(session.window_device, session.window_host,
                    session.window_bytes, cudaMemcpyHostToDevice, stream);
                return copied != cudaSuccess ? copied : ggml_cuda_moe_hybrid_emit_range(session, graph, execution, 0, first) ?
                    cudaSuccess : cudaErrorNotSupported;
            });
        }
        for (uint32_t r = 0; status == cudaSuccess && r < count; ++r) {
            capture_region = r;
            phase = "conditional_handle";
            auto & region = *static_cast<ggml_cuda_moe_hybrid_region *>(session.regions[r]);
            cudaGraphConditionalHandle handle;
            status = cudaGraphConditionalHandleCreate(&handle, window->graph, 0, cudaGraphCondAssignDefault);
            if (status != cudaSuccess) {
                break;
            }
            auto packet = ggml_cuda_moe_hybrid_packet(session.control_data(), ggml_cuda_moe_hybrid_route_capacity(region));
            packet.combine_gpu = region.combined_gpu;
            packet.runtime = region.device_runtime;
            packet.window_failed = session.window_failed;
            ggml_cuda_moe_hybrid_control_layout layout;
            if (!ggml_cuda_moe_hybrid_measure_control(session.route_capacity, layout)) {
                return -1;
            }
            auto * producer_status = reinterpret_cast<uint32_t *>(session.control_data() + layout.consumer_status_offset);
            phase = "producer";
            status = ggml_cuda_moe_hybrid_capture(window->graph, stream, dependencies, [&] {
                if (!ggml_cuda_moe_hybrid_execute_packet(session, region, execution, graph->nodes[region.descriptor.first_node])) {
                    return cudaErrorUnknown;
                }
                ggml_cuda_moe_hybrid_decide<<<1, 1, 0, stream>>>(packet, producer_status, handle,
                    region.multi_rows ? ggml_cuda_moe_hybrid_device_rows(region).ticket : nullptr);
                return cudaGetLastError();
            });
            const uint32_t end = r + 1 < count ?
                static_cast<ggml_cuda_moe_hybrid_region *>(session.regions[r + 1])->descriptor.first_node : graph->n_nodes;
            uint32_t cursor = region.descriptor.last_node + 1;
            ggml_tensor * staged = nullptr;
            bool publish_region = true;
            while (status == cudaSuccess) {
                uint32_t next = cursor;
                while (next < end && (!(graph->nodes[next]->flags & GGML_TENSOR_FLAG_COMPUTE) ||
                        !ggml_cuda_staged_input_supports(graph->nodes[next]))) {
                    ++next;
                }
                cudaGraphNode_t conditional = nullptr;
                cudaGraph_t body = nullptr;
                phase = "conditional_node";
                status = ggml_cuda_moe_hybrid_add_if(window->graph, handle, dependencies, conditional, body);
                if (status != cudaSuccess) {
                    break;
                }
                phase = "body";
                std::vector<cudaGraphNode_t> body_dependencies;
                status = ggml_cuda_moe_hybrid_capture(body, stream, body_dependencies, [&] {
                    if (publish_region) {
                        if (!owner.finish_hybrid_admission(*region.group, true, false, region.device_runtime)) {
                            return cudaErrorUnknown;
                        }
                        const auto copied = cudaMemcpyAsync(region.descriptor.output->data, session.data() + region.raw_offset,
                            ggml_nbytes(region.descriptor.output), cudaMemcpyDeviceToDevice, stream);
                        if (copied != cudaSuccess) {
                            return copied;
                        }
                    }
                    if (staged != nullptr) {
                        const auto copied = cudaMemcpyAsync(staged->data, session.window_input,
                            ggml_nbytes(staged), cudaMemcpyDeviceToDevice, stream);
                        if (copied != cudaSuccess) {
                            return copied;
                        }
                    }
                    return ggml_cuda_moe_hybrid_emit_range(session, graph, execution, cursor, next) &&
                        session.hook(GGML_BACKEND_MOE_HYBRID_TEST_WINDOW_BODY, stream) ? cudaSuccess : cudaErrorNotSupported;
                });
                if (status == cudaSuccess && !ggml_cuda_moe_hybrid_body_supported(body)) {
                    if (const char * path = getenv("GGML_MOE_HYBRID_REJECTED_GRAPH")) {
                        const auto dumped = cudaGraphDebugDotPrint(body, path, cudaGraphDebugDotFlagsVerbose);
                        fprintf(stderr, "moe-hybrid-window: rejected body graph path=%s status=%s\n", path, cudaGetErrorString(dumped));
                    }
                    status = cudaErrorNotSupported;
                }
                if (status != cudaSuccess) {
                    break;
                }
                dependencies.assign(1, conditional);
                if (next == end) {
                    break;
                }
                staged = graph->nodes[next];
                phase = "input_handle";
                status = cudaGraphConditionalHandleCreate(&handle, window->graph, 0, cudaGraphCondAssignDefault);
                if (status != cudaSuccess) {
                    break;
                }
                phase = "input_transport";
                status = ggml_cuda_moe_hybrid_capture(window->graph, stream, dependencies, [&] {
                    auto private_input = *staged;
                    private_input.data = session.window_input;
                    if (!ggml_cuda_staged_input_compute(*session.ctx, &private_input)) {
                        return cudaErrorNotSupported;
                    }
                    ggml_cuda_moe_hybrid_continue<<<1, 1, 0, stream>>>(region.device_runtime, handle);
                    return cudaGetLastError();
                });
                cursor = next + 1;
                publish_region = false;
            }
            if (status == cudaSuccess) {
                phase = "cleanup";
                status = ggml_cuda_moe_hybrid_capture(window->graph, stream, dependencies, [&] {
                    return owner.finish_hybrid_admission(*region.group, false, false, region.device_runtime) ?
                        cudaSuccess : cudaErrorUnknown;
                });
            }
        }
        if (status == cudaSuccess) {
            phase = "readback";
            status = ggml_cuda_moe_hybrid_capture(window->graph, stream, dependencies, [&] {
                return cudaMemcpyAsync(session.window_host, session.window_device,
                    session.window_bytes, cudaMemcpyDeviceToHost, stream);
            });
        }
        if (status == cudaSuccess) {
            phase = "instantiate";
            status = cudaGraphInstantiate(&window->instance, window->graph, nullptr, nullptr, 0);
        }
        if (status != cudaSuccess) {
            if (status != cudaErrorNotSupported && status != cudaErrorInvalidValue &&
                    status != cudaErrorStreamCaptureUnsupported && status != cudaErrorStreamCaptureInvalidated) {
                fprintf(stderr, "moe-hybrid-window: capture failed, phase=%s region=%u: %s\n", phase, capture_region, cudaGetErrorString(status));
                return -1;
            }
            session.window_unavailable = true;
            fprintf(stderr, "moe-hybrid-window: unavailable during capture, phase=%s region=%u: %s; using eager S5b\n", phase, capture_region, cudaGetErrorString(status));
            (void) cudaGetLastError();
            if (!owner.finish_graph_dispatch(&execution)) {
                fprintf(stderr, "moe-hybrid-window: capture dispatch cleanup failed\n");
                return -1;
            }
            for (uint32_t r = 0; r < count; ++r) {
                static_cast<ggml_cuda_moe_hybrid_region *>(session.regions[r])->group->strategy = original_strategies[r];
            }
            if (!owner.begin_graph_dispatch(&execution, GGML_CUDA_MOE_GRAPH_DISPATCH_DIRECT)) {
                fprintf(stderr, "moe-hybrid-window: eager dispatch restart failed\n");
                return -1;
            }
            return 0;
        }
        window->node_props.reserve(graph->n_nodes);
        for (int i = 0; i < graph->n_nodes; ++i) {
            window->node_props.push_back(ggml_cuda_graph_node_properties(graph->nodes[i]));
        }
        session.window_graph = std::move(window);
        session.window_identity = identity;
        ++session.window_captures;
    }
    if (!session.hook(GGML_BACKEND_MOE_HYBRID_TEST_WINDOW_SUBMIT, stream)) {
        fprintf(stderr, "moe-hybrid-window: submission hook rejected\n");
        return -1;
    }
    const auto launched = cudaGraphLaunch(session.window_graph->instance, stream);
    if (launched != cudaSuccess) {
        fprintf(stderr, "moe-hybrid-window: launch failed: %s\n", cudaGetErrorString(launched));
        const auto main_drained = cudaStreamSynchronize(stream);
        const auto io_drained = cudaStreamSynchronize(session.io_stream);
        session.join(true);
        if (main_drained != cudaSuccess || io_drained != cudaSuccess) {
            GGML_LOG_ERROR("moe-hybrid-window: terminal CUDA submission drain failed\n");
        }
        return -1;
    }
    ++session.window_launches;
    ++session.window_waits;
    session.resident_batches += count;
    session.resident_body_submissions += count;
    session.gpu_body_submissions += 2 * count - combined_count;
    session.window_combined_regions += combined_count;
    session.window_direct_regions += direct_count;
    session.window_compact_select_regions += direct_count;
    session.window_fused_expert_bodies += fused_bodies;
    const int64_t started = ggml_time_us();
    const auto recorded = cudaEventRecord(session.producer_done, stream);
    const auto observed = recorded == cudaSuccess ? cudaEventSynchronize(session.producer_done) : recorded;
    const bool done = observed == cudaSuccess;
    session.publication_us += ggml_time_us() - started;
    if (!done) {
        fprintf(stderr, "moe-hybrid-window: observation failed, record=%s wait=%s\n", cudaGetErrorString(recorded), cudaGetErrorString(observed));
        const auto drained = cudaStreamSynchronize(stream);
        session.join(true);
        if (drained != cudaSuccess) {
            GGML_LOG_ERROR("moe-hybrid-window: terminal CUDA drain failed: %s\n", cudaGetErrorString(drained));
        }
        return -1;
    }
    for (uint32_t r = 0; r < count; ++r) {
        auto & region = *static_cast<ggml_cuda_moe_hybrid_region *>(session.regions[r]);
        const auto & result = session.window_host[r];
        if (result.accepted) {
            ++session.completed;
            ++session.packet_regions;
            session.admission_committed += result.admissions;
            session.admission_bytes += result.admissions * (region.admission_stride + region.admission_auxiliary_bytes);
        } else {
            session.admission_aborted += region.packet_admissions.load();
        }
    }
    const bool rows_committed = static_cast<ggml_cuda_moe_hybrid_region *>(session.regions[0])->multi_rows;
    const bool accepted = *session.window_host_failed == 0 && (rows_committed || !session.canceled.load()) && session.completed == count;
    if (!accepted) {
        for (uint32_t r = 0; r < count; ++r) {
            const auto & result = session.window_host[r];
            if (!result.accepted) {
                fprintf(stderr, "moe-hybrid-window: rejected, region=%u started=%u accepted=%u cpu_status=%u completed=%u/%u failed=%u\n",
                    r, result.started, result.accepted, result.cpu_status, session.completed, count, *session.window_host_failed);
                break;
            }
        }
    }
    return accepted ? 1 : -1;
#endif
} catch (const std::bad_alloc &) {
    ++session.capacity_errors;
    fprintf(stderr, "moe-hybrid-window: host allocation failed\n");
    return -1;
} catch (...) {
    fprintf(stderr, "moe-hybrid-window: preparation exception\n");
    return -1;
}

static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph);

static enum ggml_status ggml_cuda_moe_hybrid_compute(
        void * opaque, ggml_cgraph * graph, void * const * regions, uint32_t count) {
    auto * session = static_cast<ggml_cuda_moe_hybrid_session *>(opaque);
    if (session == nullptr || graph == nullptr || (count != 0 && regions == nullptr) ||
            count > GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS || session->ctx->moe_hybrid_active != nullptr) {
        return GGML_STATUS_FAILED;
    }
    {
        std::lock_guard<std::mutex> lock(session->mutex);
        if (session->quiescing || session->dispatch_active) {
            return GGML_STATUS_FAILED;
        }
        session->dispatch_active = true;
        session->canceled.store(false);
    }
    struct dispatch_guard {
        ggml_cuda_moe_hybrid_session & session;
        bool success = false;
        ~dispatch_guard() {
            const auto * pool = static_cast<const ggml_cuda_moe_hybrid_pool *>(
                session.compute->pools[session.ctx->device][0].get());
            session.work_peak.store(pool == nullptr ? 0 : pool->peak);
            session.errors += !success;
            std::lock_guard<std::mutex> lock(session.mutex);
            session.dispatch_active = false;
            session.ready.notify_all();
        }
    } dispatch{*session};
    const auto & certificate = graph->execution_certificate;
    const bool speculative = certificate.row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SPECULATIVE;
    if (certificate.magic != GGML_GRAPH_EXECUTION_CERTIFICATE_MAGIC ||
            certificate.domain != GGML_GRAPH_EXECUTION_DOMAIN_MAIN ||
            (speculative ? (certificate.flags != GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED ||
                certificate.n_rows <= 1 || certificate.n_sequences == 0 || certificate.n_sequences >= certificate.n_rows) :
                (certificate.row_semantics != GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT ||
                 certificate.n_rows != 1 || certificate.n_sequences != 1)) || certificate.split_graph_uid != graph->uid) {
        return GGML_STATUS_FAILED;
    }
    for (uint32_t r = 0; r < count; ++r) {
        if (regions[r] == nullptr) {
            return GGML_STATUS_FAILED;
        }
        const auto & region = *static_cast<ggml_cuda_moe_hybrid_region *>(regions[r]);
        const auto & descriptor = region.descriptor;
        if (region.multi_rows != speculative || (speculative &&
                (memcmp(&descriptor.certificate, &certificate, sizeof(certificate)) != 0 ||
                 region.rows_layout.geometry.row_capacity != certificate.n_rows)) ||
                descriptor.split_graph_uid != graph->uid || descriptor.source_graph_uid != certificate.source_graph_uid ||
                descriptor.owner_generation != certificate.owner_generation || descriptor.last_node >= uint32_t(graph->n_nodes) ||
                graph->nodes[descriptor.last_node] != descriptor.output ||
                descriptor.last_node - descriptor.first_node + 1 != region.nodes.size()) {
            return GGML_STATUS_FAILED;
        }
        for (uint32_t i = descriptor.first_node; i <= descriptor.last_node; ++i) {
            if (graph->nodes[i]->op != region.nodes[i - descriptor.first_node]->op) {
                return GGML_STATUS_FAILED;
            }
        }
    }
    for (int i = 0; i < graph->n_nodes; ++i) {
        const auto * node = graph->nodes[i];
        if (node->op != GGML_OP_MUL_MAT_ID || node->src[0]->buffer == nullptr ||
                !ggml_backend_buft_is_cuda_moe_cached(node->src[0]->buffer->buft)) {
            continue;
        }
        uint32_t coverage = 0;
        for (uint32_t r = 0; r < count; ++r) {
            const auto & descriptor = static_cast<ggml_cuda_moe_hybrid_region *>(regions[r])->descriptor;
            coverage += uint32_t(i) >= descriptor.first_node && uint32_t(i) <= descriptor.last_node;
        }
        if (coverage != 1) {
            return GGML_STATUS_FAILED;
        }
    }
    session->regions = regions;
    session->n_regions = count;
    session->completed = 0;
    session->ctx->stream_context().reset();
    session->ctx->curr_stream_no = 0;
    session->ctx->moe_hybrid_active = session;
    auto status = GGML_STATUS_FAILED;
    try {
        status = ggml_backend_cuda_graph_compute(session->backend, graph);
    } catch (...) {
        status = GGML_STATUS_FAILED;
    }
    const auto synchronized = cudaStreamSynchronize(session->ctx->stream());
    if (status != GGML_STATUS_SUCCESS || synchronized != cudaSuccess) {
        session->cancel_pending();
        (void) cudaStreamSynchronize(session->io_stream);
        session->join(true);
    }
    session->ctx->moe_hybrid_active = nullptr;
    session->regions = nullptr;
    session->n_regions = 0;
    if (session->diagnostic) {
        fprintf(stderr, "moe-hybrid: regions=%u/%u resident_routes=%llu transfer_routes=%llu cpu_routes=%llu h2d_bytes=%llu cpu_jobs=%llu status=%d\n",
            session->completed, count, (unsigned long long) session->resident_routes.load(),
            (unsigned long long) session->transferred_routes.load(), (unsigned long long) session->cpu_routes.load(),
            (unsigned long long) session->h2d_bytes.load(), (unsigned long long) session->cpu_jobs.load(), int(status));
    }
    dispatch.success = status == GGML_STATUS_SUCCESS && synchronized == cudaSuccess && session->completed == count;
    return dispatch.success ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
}

static const ggml_backend_moe_hybrid_api_v1 * ggml_cuda_moe_hybrid_api() {
    static const ggml_backend_moe_hybrid_api_v1 api = {
        sizeof(api), 1, ggml_cuda_moe_hybrid_create, ggml_cuda_moe_hybrid_prepare, ggml_cuda_moe_hybrid_compute,
        ggml_cuda_moe_hybrid_destroy_region,
        [](void * session) { delete static_cast<ggml_cuda_moe_hybrid_session *>(session); },
        [](void * opaque, ggml_backend_moe_hybrid_state_v1 * output) {
            auto * session = static_cast<ggml_cuda_moe_hybrid_session *>(opaque);
            if (session == nullptr || output == nullptr || output->struct_size != sizeof(*output)) {
                return false;
            }
            const auto * pool = static_cast<const ggml_cuda_moe_hybrid_pool *>(
                session->compute->pools[session->ctx->device][0].get());
            std::lock_guard<std::mutex> lock(session->mutex);
            ggml_backend_moe_cpu_service_state_v1 cpu = {};
            cpu.struct_size = sizeof(cpu);
            if (session->cpu_service != nullptr && session->cpu_api->state(session->cpu_service, &cpu) !=
                    GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK) {
                return false;
            }
            *output = {sizeof(*output), session->ticket.state,
                session->resident_routes.load(), session->transferred_routes.load(), session->cpu_routes.load(),
                session->h2d_bytes.load(), session->control_bytes + session->device_bytes + session->window_bytes + session->window_input_bytes + (pool == nullptr ? 0 : pool->capacity),
                session->pinned_bytes + session->window_bytes, session->work_peak.load(),
                session->distinct_experts.load(), session->resident_experts.load(), session->transfer_experts.load(),
                session->cpu_experts.load(), session->cpu_jobs.load(), session->cpu_upload_bytes.load(),
                session->cpu_us.load(), session->gpu_enqueue_us.load(), session->join_us.load(), session->errors.load(),
                session->capacity_errors.load(), session->cancellations.load(), session->epoch, cpu.active_jobs,
                uint32_t(session->dispatch_active), uint32_t(session->quiescing),
                session->submit_to_start_us.load(), session->gpu_branch_us.load(), session->last_submit_us.load(),
                session->last_cpu_start_us.load(), session->last_cpu_done_us.load(), session->last_gpu_enqueue_done_us.load(),
                session->last_gpu_done_observed_us.load(), session->last_join_start_us.load(), session->last_join_done_us.load(),
                session->admission_reserved.load(), session->admission_committed.load(), session->admission_replacements.load(),
                session->admission_no_slot.load(), session->admission_bytes.load(), session->admission_aborted.load(),
                session->resident_batches.load(), session->resident_body_submissions.load(), session->gpu_body_submissions.load(),
                session->readback_us.load(), session->publication_us.load(), session->admission_fence_us.load(),
                session->prepared_device_bytes.load(), cpu.prepared_payload_peak,
                session->packet_regions.load(), session->producer_events.load(),
                session->producer_fences.load(), session->producer_drain_events.load(),
                session->window_launches.load(), session->window_captures.load(),
                session->window_waits.load(), session->window_fallbacks.load(), session->window_fused_nodes.load(),
                session->window_combined_regions.load(), session->window_direct_regions.load(),
                session->window_compact_select_regions.load(), session->window_fused_expert_bodies.load(),
                session->cpu_execute_calls.load(), session->cpu_row_jobs.load()};
            return true;
        },
        [](void * opaque) { static_cast<ggml_cuda_moe_hybrid_session *>(opaque)->quiesce(); },
        [](void * opaque, ggml_backend_moe_hybrid_test_hook_v1_t hook, void * data) {
            auto * session = static_cast<ggml_cuda_moe_hybrid_session *>(opaque);
            std::lock_guard<std::mutex> lock(session->mutex);
            if (session->dispatch_active || session->ticket.state != 0) {
                return false;
            }
            session->test_hook = hook;
            session->test_hook_data = data;
            return true;
        },
    };
    return &api;
}

static bool ggml_cuda_compute_graph_nodes(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph,
        ggml_cuda_moe_graph_execution * moe_execution, bool use_cuda_graph, bool cuda_graph_update_required,
        uint64_t * fused_nodes, size_t * scratch_bytes, uint64_t * image_groups, uint64_t * emitted_images) {
    auto & stream_ctx = cuda_ctx->stream_context();
    const bool integrated = ggml_cuda_info().devices[cuda_ctx->device].integrated;
    bool is_concurrent_event_active = false;
    ggml_cuda_concurrent_event * concurrent_event = nullptr;
    int prev_i = 0;
    static const bool disable_fusion = getenv("GGML_CUDA_DISABLE_FUSION") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_FUSION"));
    const bool disable_reuse = disable_fusion || cuda_ctx->moe_hybrid_active != nullptr ||
        (moe_execution != nullptr && moe_execution->requires_dispatch());
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
    ggml_cuda_reuse_plan mmvq_reuse(cgraph, stream_ctx, mmvq_keys, mmvq_sizes, [](const ggml_tensor * n, const ggml_tensor * in) { return ggml_cuda_mmvq_input_overwritten(n, in) || ggml_cuda_prepared_input_overwritten(n, in); });
    std::vector<ggml_cuda_norm_q8_match> mmvq_norm_emits;
    if (!disable_reuse && stream_ctx.concurrent_events.empty()) {
        mmvq_norm_emits = ggml_cuda_plan_norm_q8(cgraph, cuda_ctx->device, mmvq_keys, mmvq_sizes, mmvq_reuse);
    }
    std::vector<int> mmq_keys;
    std::vector<size_t> mmq_sizes;
    const int cc = ggml_cuda_info().devices[cuda_ctx->device].cc;
    const auto scale_size = [&](const ggml_tensor * node) {
        return ggml_cuda_mmq_get_prec_src1(node->src[0], node, cc) == GGML_PREC_Q4 && node->src[0]->type == GGML_TYPE_NVFP4 ?
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
                const bool fp4 = ggml_cuda_mmq_get_prec_src1(node->src[0], node, cc) == GGML_PREC_Q4;
                const size_t blocks = GGML_PAD(size_t(input->ne[0]), MATRIX_ROW_PADDING)/(fp4 ? QK_FP4_MMQ : QK8_1_MMQ);
                if (blocks > SIZE_MAX/sizeof(block_q8_1_mmq)) { continue; }
                const size_t row = blocks*sizeof(block_q8_1_mmq);
                // The kernel reads a full tile, including columns past the input.
                const size_t tail = 128*sizeof(block_q8_1_mmq);
                if (uint64_t(input->ne[1]) > (SIZE_MAX - tail - 255)/row ||
                        uint64_t(input->ne[1]) > (SIZE_MAX - 255)/sizeof(float) - 128) { continue; }
                const size_t quantized = size_t(input->ne[1])*row + tail;
                if (scale_size(node) > SIZE_MAX - GGML_PAD(quantized, 256)) { continue; }
                mmq_keys[i] = ggml_cuda_mmq_get_prec_src1(node->src[0], node, cc) == GGML_PREC_Q4 ?
                    3 + (node->src[0]->type == GGML_TYPE_NVFP4) : int(mmq_get_q8_1_ds_layout(node->src[0]->type));
                mmq_sizes[i] = GGML_PAD(quantized, 256) + scale_size(node);
            }
        }
    }
    ggml_cuda_reuse_plan mmq_reuse(cgraph, stream_ctx, mmq_keys, mmq_sizes, [](const ggml_tensor * n, const ggml_tensor * in) { return ggml_cuda_mmq_input_overwritten(n, in) || ggml_cuda_prepared_input_overwritten(n, in); });
    std::vector<ggml_cuda_norm_mmq_match> mmq_norm_emits;
    if (!disable_reuse && stream_ctx.concurrent_events.empty()) {
        mmq_norm_emits = ggml_cuda_plan_norm_mmq(cgraph, cuda_ctx->device, mmq_keys, mmq_sizes, mmq_reuse);
    }
    std::vector<int> route_keys, mmqid_keys;
    std::vector<size_t> route_sizes, mmqid_sizes, mmqid_scale_sizes;
    if (!disable_reuse) {
        route_keys.assign(cgraph->n_nodes, -1);
        route_sizes.resize(cgraph->n_nodes);
        mmqid_keys.assign(cgraph->n_nodes, -1);
        mmqid_sizes.resize(cgraph->n_nodes);
        mmqid_scale_sizes.resize(cgraph->n_nodes);
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            const ggml_tensor * node = cgraph->nodes[i];
            if (!ggml_cuda_can_share_mmq_id_input(node, cuda_ctx->device)) { continue; }
            const ggml_tensor * input = node->src[1];
            if (input->ne[0] <= 0 || input->ne[0] > INT64_MAX - MATRIX_ROW_PADDING + 1 ||
                    uint64_t(input->ne[0]) > SIZE_MAX - MATRIX_ROW_PADDING + 1 || input->ne[2] <= 0 || node->src[2]->ne[0] <= 0 ||
                    uint64_t(node->src[2]->ne[0]) > SIZE_MAX/uint64_t(input->ne[2]) || node->src[0]->ne[2] <= 0 ||
                    uint64_t(node->src[0]->ne[2]) > SIZE_MAX - 1) { continue; }
            const size_t rows = size_t(input->ne[2])*size_t(node->src[2]->ne[0]);
            if (rows > SIZE_MAX - MMQ_ID_INPUT_GUARD) { continue; }
            const size_t guarded_rows = rows + MMQ_ID_INPUT_GUARD;
            const size_t experts = size_t(node->src[0]->ne[2]) + 1;
            const bool fp4 = ggml_cuda_mmq_get_prec_src1(node->src[0], node, cc) == GGML_PREC_Q4;
            const size_t blocks = GGML_PAD(size_t(input->ne[0]), MATRIX_ROW_PADDING)/(fp4 ? QK_FP4_MMQ : QK8_1_MMQ);
            if (blocks > SIZE_MAX/sizeof(block_q8_1_mmq)) { continue; }
            const size_t row = blocks*sizeof(block_q8_1_mmq);
            const size_t tail = MMQ_ID_INPUT_GUARD*sizeof(block_q8_1_mmq);
            if (rows > (SIZE_MAX - tail - 255)/row || guarded_rows > (SIZE_MAX - 255)/sizeof(int32_t) ||
                    guarded_rows > (SIZE_MAX - 255)/sizeof(float) || experts > (SIZE_MAX - 255)/sizeof(int32_t)) { continue; }
            const size_t quantized = GGML_PAD(rows*row + tail, 256);
            const size_t scales = fp4 && node->src[0]->type == GGML_TYPE_NVFP4 ? GGML_PAD(guarded_rows*sizeof(float), 256) : 0;
            if (scales > SIZE_MAX - quantized) { continue; }
            const size_t ids_size = GGML_PAD(rows*sizeof(int32_t), 256);
            const size_t dst_size = GGML_PAD(guarded_rows*sizeof(int32_t), 256);
            const size_t bounds_size = GGML_PAD(experts*sizeof(int32_t), 256);
            if (ids_size > SIZE_MAX - dst_size || ids_size + dst_size > SIZE_MAX - bounds_size) { continue; }
            route_keys[i] = 0;
            route_sizes[i] = ids_size + dst_size + bounds_size;
            mmqid_keys[i] = fp4 ? 3 + (node->src[0]->type == GGML_TYPE_NVFP4) : int(mmq_get_q8_1_ds_layout(node->src[0]->type));
            mmqid_sizes[i] = quantized + scales;
            mmqid_scale_sizes[i] = scales;
        }
    }
    ggml_cuda_reuse_plan routes(cgraph, stream_ctx, route_keys, route_sizes, [](const ggml_tensor * n, const ggml_tensor * in) { return ggml_cuda_mmq_id_input_overwritten(n, in) || ggml_cuda_prepared_input_overwritten(n, in); });
    for (size_t i = 0; i < mmqid_keys.size(); ++i) {
        if (routes.nodes.empty() || routes.nodes[i] < 0) { mmqid_keys[i] = -1; }
    }
    ggml_cuda_reuse_plan mmqid_reuse(cgraph, stream_ctx, mmqid_keys, mmqid_sizes, [](const ggml_tensor * n, const ggml_tensor * in) { return ggml_cuda_mmq_id_input_overwritten(n, in) || ggml_cuda_prepared_input_overwritten(n, in); }, true);
    if (scratch_bytes) {
        *scratch_bytes = 0;
        for (const size_t size : {reuse.size, mmvq_reuse.size, mmq_reuse.size, routes.size, mmqid_reuse.size}) {
            if (size > SIZE_MAX - *scratch_bytes) { return false; }
            *scratch_bytes += size;
        }
        return true;
    }
    if (image_groups) {
        *image_groups += reuse.groups.size() + mmvq_reuse.groups.size() + mmq_reuse.groups.size() + mmqid_reuse.groups.size();
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
    ggml_cuda_reuse_inputs<ggml_cuda_pool_alloc<char>> mmvq_inputs(cuda_ctx->pool(), mmvq_reuse);
    const auto prepare_mmvq_group = [&](int group) {
        if (mmvq_inputs[group].get()) { return; }
        const ggml_tensor * node = cgraph->nodes[mmvq_reuse.groups[group].node];
        ggml_cuda_quantize_mmvq_input(*cuda_ctx, node->src[0], node->src[1], mmvq_inputs[group]);
    };
    const auto prepare_mmvq_shared = [&](int i, bool after) {
        if (mmvq_reuse.starts.empty()) { return; }
        for (int group = mmvq_reuse.starts[i]; group >= 0; group = mmvq_reuse.groups[group].next) {
            if (mmvq_reuse.groups[group].after == after) {
                prepare_mmvq_group(group);
            }
        }
    };
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
    ggml_cuda_reuse_inputs<ggml_cuda_mmq_id_input> route_inputs(cuda_ctx->pool(), routes);
    ggml_cuda_reuse_inputs<ggml_cuda_mmq_id_input> mmqid_inputs(cuda_ctx->pool(), mmqid_reuse);
    const auto prepare_mmqid_route = [&](int group) {
        if (route_inputs[group].ids_src1.get()) { return; }
        ggml_cuda_prepare_mmq_id_routes(*cuda_ctx, cgraph->nodes[routes.groups[group].node], route_inputs[group], MMQ_ID_INPUT_GUARD);
    };
    const auto prepare_mmqid_group = [&](int group) {
        if (mmqid_inputs[group].quantized.get()) { return; }
        const int i = mmqid_reuse.groups[group].node;
        const int route = routes.nodes[i];
        if (!route_inputs[route].ids_src1.get()) { return; }
        const ggml_tensor * node = cgraph->nodes[i];
        ggml_cuda_prepare_mmq_id_input(*cuda_ctx, node, mmqid_reuse.groups[group].size - mmqid_scale_sizes[i],
            mmqid_inputs[group], &route_inputs[route]);
    };
    const auto prepare_mmqid_shared = [&](int i, bool after) {
        if (!routes.starts.empty()) {
            for (int group = routes.starts[i]; group >= 0; group = routes.groups[group].next) {
                if (routes.groups[group].after == after) { prepare_mmqid_route(group); }
            }
        }
        if (!mmqid_reuse.starts.empty()) {
            for (int group = mmqid_reuse.starts[i]; group >= 0; group = mmqid_reuse.groups[group].next) {
                if (mmqid_reuse.groups[group].after == after) { prepare_mmqid_group(group); }
            }
        }
    };
    const auto try_launch_concurrent_event = [&](const ggml_tensor * node) {
        if (stream_ctx.concurrent_events.find(node) != stream_ctx.concurrent_events.end()) {
            const int i = reuse.indices.empty() ? -1 : reuse.indices.at(node);
            if (i >= 0) { prepare_shared(i, true); }
            const int q = mmvq_reuse.indices.empty() ? -1 : mmvq_reuse.indices.at(node);
            if (q >= 0) { prepare_mmvq_shared(q, true); }
            const int q_mmq = mmq_reuse.indices.empty() ? -1 : mmq_reuse.indices.at(node);
            if (q_mmq >= 0) { prepare_mmq_shared(q_mmq, true); }
            const int q_id = mmqid_reuse.indices.empty() ? -1 : mmqid_reuse.indices.at(node);
            if (q_id >= 0) { prepare_mmqid_shared(q_id, true); }
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

        if (cuda_ctx->moe_grouped_context != nullptr) {
            cuda_ctx->moe_grouped_context->launch_early_router(node, moe_execution, cuda_ctx->stream());
        }
        if (cuda_ctx->moe_hybrid_active != nullptr) {
            auto & session = *cuda_ctx->moe_hybrid_active;
            bool handled = false;
            for (uint32_t r = 0; r < session.n_regions; ++r) {
                auto & region = *static_cast<ggml_cuda_moe_hybrid_region *>(session.regions[r]);
                if (region.descriptor.first_node == uint32_t(i)) {
                    if (!ggml_cuda_moe_hybrid_execute(session, region, *moe_execution, node)) {
                        return false;
                    }
                    i = region.descriptor.last_node;
                    handled = true;
                    break;
                }
            }
            if (handled) {
                continue;
            }
        }
        if (!mmvq_norm_emits.empty() && mmvq_norm_emits[i].norm &&
                ((!norm_emits.empty() && norm_emits[i].norm) || mmvq_norm_emits[i].post)) {
            const auto & emit = mmvq_norm_emits[i];
            const int g = emit.image;
            const ggml_tensor * input = cgraph->nodes[mmvq_reuse.groups[g].node]->src[1];
            mmvq_inputs[g].alloc(mmvq_sizes[mmvq_reuse.groups[g].node]);
            void * f16 = nullptr;
            void * bf16 = nullptr;
            if (!norm_emits.empty() && norm_emits[i].norm) {
                const auto & typed = norm_emits[i];
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
                if (emit.scale) {
                    ggml_cuda_op_dsv4_hc_post(*cuda_ctx, emit.post);
                    ggml_cuda_op_rms_norm_emit_q8(*cuda_ctx, emit.norm, nullptr, nullptr, emit.scale, f16, bf16,
                        mmvq_inputs[g].get(), input->ne[0], GGML_PAD(input->ne[0], MATRIX_ROW_PADDING));
                } else {
                    ggml_cuda_op_hc_post_norm_emit_q8(*cuda_ctx, emit.post, emit.norm, emit.mul, f16, bf16,
                        mmvq_inputs[g].get(), input->ne[0], GGML_PAD(input->ne[0], MATRIX_ROW_PADDING));
                }
            } else {
                ggml_cuda_op_rms_norm_emit_q8(*cuda_ctx, emit.norm, emit.mul, emit.add, emit.scale, f16, bf16,
                    mmvq_inputs[g].get(), input->ne[0], GGML_PAD(input->ne[0], MATRIX_ROW_PADDING));
            }
            if (emitted_images) { ++*emitted_images; }
            if (fused_nodes) { *fused_nodes += emit.last - i; }
            i = emit.last;
            continue;
        }

        if (!mmvq_norm_emits.empty() && mmvq_norm_emits[i].norm) {
            const auto & emit = mmvq_norm_emits[i];
            const int g = emit.image;
            const ggml_tensor * input = cgraph->nodes[mmvq_reuse.groups[g].node]->src[1];
            mmvq_inputs[g].alloc(mmvq_sizes[mmvq_reuse.groups[g].node]);
            ggml_cuda_op_rms_norm_q8(*cuda_ctx, emit.norm, emit.mul, emit.add, emit.scale,
                mmvq_inputs[g].get(), input->ne[0], GGML_PAD(input->ne[0], MATRIX_ROW_PADDING));
            if (emitted_images) { ++*emitted_images; }
            if (fused_nodes) { *fused_nodes += emit.last - i; }
            i = emit.last;
            continue;
        }

        if (!mmq_norm_emits.empty() && mmq_norm_emits[i].norm) {
            const auto & emit = mmq_norm_emits[i];
            const int g = emit.image;
            const int reader = mmq_reuse.groups[g].node;
            const ggml_tensor * input = cgraph->nodes[reader]->src[1];
            mmq_inputs[g].quantized.alloc(mmq_reuse.groups[g].size);
            ggml_cuda_op_rms_norm_mmq(*cuda_ctx, emit.norm, emit.mul, emit.add, emit.scale,
                mmq_inputs[g].quantized.get(), input->ne[0], GGML_PAD(input->ne[0], MATRIX_ROW_PADDING), input->ne[1], mmq_keys[reader]);
            if (emitted_images) { ++*emitted_images; }
            if (fused_nodes) { *fused_nodes += emit.last - i; }
            i = emit.last;
            continue;
        }

        const int group = reuse.nodes.empty() ? -1 : reuse.nodes[i];
        prepare_shared(i, false);
        if (group >= 0) { prepare_group(group); }
        const void * prepared_src1 = group >= 0 ? shared_inputs[group].get() : nullptr;

        if (!norm_emits.empty() && norm_emits[i].norm) {
            const auto & emit = norm_emits[i];
            const auto image = [&](int g) -> void * {
                if (g < 0) { return nullptr; }
                shared_inputs[g].alloc(input_sizes[reuse.groups[g].node]);
                return shared_inputs[g].get();
            };
            if (emit.post) {
                if (emit.scale) {
                    ggml_cuda_op_hc_post_norm_scale(*cuda_ctx, emit.post, emit.norm, emit.scale, image(emit.f16), image(emit.bf16));
                } else {
                    ggml_cuda_op_hc_post_norm_emit(*cuda_ctx, emit.post, emit.norm, emit.mul, image(emit.f16), image(emit.bf16));
                }
            } else {
            ggml_cuda_op_rms_norm_emit(*cuda_ctx, emit.norm, emit.mul, emit.add, emit.scale, image(emit.f16), image(emit.bf16));
            }
            if (emitted_images) { ++*emitted_images; }
            if (fused_nodes) { *fused_nodes += emit.last - i; }
            i = emit.last;
            continue;
        }

        const int mmvq_group = mmvq_reuse.nodes.empty() ? -1 : mmvq_reuse.nodes[i];
        if (mmvq_group >= 0 && !mmvq_inputs[mmvq_group].get()) {
            bool ready = !is_concurrent_event_active;
            if (!ready && !mmvq_reuse.groups[mmvq_group].after) {
                const auto first = concurrent_event->stream_mapping.find(cgraph->nodes[mmvq_reuse.groups[mmvq_group].node]);
                ready = first != concurrent_event->stream_mapping.end() && first->second == cuda_ctx->curr_stream_no;
            }
            if (ready) { prepare_mmvq_group(mmvq_group); }
        }
        const ggml_tensor * shared_input = mmvq_group >= 0 ? node->src[1] : nullptr;
        const char * quantized = mmvq_group >= 0 ? mmvq_inputs[mmvq_group].get() : nullptr;
        const int mmq_group = mmq_reuse.nodes.empty() ? -1 : mmq_reuse.nodes[i];
        if (mmq_group >= 0 && !mmq_inputs[mmq_group].quantized.get()) {
            bool ready = !is_concurrent_event_active;
            if (!ready && !mmq_reuse.groups[mmq_group].after) {
                const auto first = concurrent_event->stream_mapping.find(cgraph->nodes[mmq_reuse.groups[mmq_group].node]);
                ready = first != concurrent_event->stream_mapping.end() && first->second == cuda_ctx->curr_stream_no;
            }
            if (ready) { prepare_mmq_group(mmq_group); }
        }
        const int route = routes.nodes.empty() ? -1 : routes.nodes[i];
        const int mmqid_group = mmqid_reuse.nodes.empty() ? -1 : mmqid_reuse.nodes[i];
        const auto ready = [&](const ggml_cuda_reuse_group & group) {
            if (!is_concurrent_event_active) { return true; }
            if (group.after) { return false; }
            const auto first = concurrent_event->stream_mapping.find(cgraph->nodes[group.node]);
            return first != concurrent_event->stream_mapping.end() && first->second == cuda_ctx->curr_stream_no;
        };
        if (route >= 0 && !route_inputs[route].ids_src1.get() && ready(routes.groups[route])) { prepare_mmqid_route(route); }
        if (mmqid_group >= 0 && !mmqid_inputs[mmqid_group].quantized.get() && ready(mmqid_reuse.groups[mmqid_group])) { prepare_mmqid_group(mmqid_group); }
        int nodes_to_skip = 0;
        if (fused_nodes) {
            auto cut = ggml_graph_view(cgraph, i, ggml_cuda_moe_source_fusion_end(cgraph, i));
            nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, &cut, 0, nullptr, shared_input, quantized);
        } else if (cuda_ctx->moe_hybrid_active == nullptr) {
            nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i, moe_execution, shared_input, quantized);
        }

        if (nodes_to_skip < 0) {
            GGML_ASSERT(!use_cuda_graph && !cuda_graph_update_required);
            return false;
        }
        if (nodes_to_skip > 0) {
            if (fused_nodes) { *fused_nodes += nodes_to_skip; }
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
                       ggml_backend_buft_is_cuda_moe_cached(node->src[j]->buffer->buft) ||
                       (integrated && ggml_backend_buft_is_cuda_host(node->src[j]->buffer->buft)));
            }
        }
#else
        GGML_UNUSED(integrated);
#endif  // NDEBUG

        bool ok;
        if (quantized) {
            ggml_cuda_mul_mat_vec_q(*cuda_ctx, node->src[0], node->src[1], nullptr, node, nullptr, quantized);
            ok = true;
        } else if (mmq_group >= 0 && mmq_inputs[mmq_group].quantized.get()) {
            ggml_cuda_mul_mat_q(*cuda_ctx, node->src[0], node->src[1], nullptr, node, &mmq_inputs[mmq_group]);
            ok = true;
        } else if (mmqid_group >= 0 && mmqid_inputs[mmqid_group].quantized.get()) {
            ggml_cuda_mul_mat_q(*cuda_ctx, node->src[0], node->src[1], node->src[2], node, nullptr, &mmqid_inputs[mmqid_group]);
            ok = true;
        } else {
            ok = ggml_cuda_compute_forward(*cuda_ctx, node, moe_execution, prepared_src1);
        }
        if (!ok) {
            GGML_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
            if (moe_execution != nullptr &&
                    (moe_execution->outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED ||
                        moe_execution->find_group(node, nullptr) != nullptr || moe_execution->rejects_cached_mmid(node))) {
                GGML_ASSERT(!use_cuda_graph && !cuda_graph_update_required);
                return false;
            }
        }
        GGML_ASSERT(ok);

        if (!is_concurrent_event_active) {
            try_launch_concurrent_event(node);
       }
    }
    return true;
}

static bool ggml_cuda_graph_evaluate_and_capture(
        ggml_backend_cuda_context * cuda_ctx,
        ggml_cgraph * cgraph,
        const bool use_cuda_graph,
        const bool cuda_graph_update_required,
        uint64_t graph_key,
        ggml_cuda_moe_graph_execution * moe_execution) {
    if (cuda_ctx->moe_hybrid_active != nullptr && cuda_ctx->moe_hybrid_active->window_requested) {
        auto & session = *cuda_ctx->moe_hybrid_active;
        const int result = ggml_cuda_moe_hybrid_execute_window(session, cgraph, *moe_execution);
        if (result != 0) {
            return result > 0;
        }
        if (session.window_fallbacks.load() == 0) {
            GGML_LOG_INFO("moe-hybrid-window: capability or storage unavailable; using eager S5b\n");
        }
        ++session.window_fallbacks;
    }
    bool graph_evaluated_or_captured = false;
#ifdef USE_CUDA_GRAPH
    cudaGraph_t retired_graph = nullptr;
    static const bool profile = getenv("GGML_CUDA_GRAPH_PROFILE") != nullptr;
    const bool measure = profile && use_cuda_graph && cuda_graph_update_required;
    const int64_t capture_start = measure ? ggml_time_us() : 0;
    int64_t capture_end = 0;
#endif

    ggml_cuda_stream_context & stream_ctx = cuda_ctx->stream_context();
    bool                         should_launch_concurrent_events = false;


    while (!graph_evaluated_or_captured) {
        // Only perform the graph execution if CUDA graphs are not enabled, or we are capturing the graph.
        // With the use of CUDA graphs, the execution will be performed by the graph launch.
        if (!use_cuda_graph || cuda_graph_update_required) {
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

            if (!ggml_cuda_compute_graph_nodes(cuda_ctx, cgraph, moe_execution, use_cuda_graph,
                    cuda_graph_update_required, nullptr, nullptr, nullptr, nullptr)) { return false; }
        }

#ifdef USE_CUDA_GRAPH
        ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
        if (use_cuda_graph && cuda_graph_update_required) { // End CUDA graph capture
            if (cuda_ctx->decode_boundary_overlap) {
                retired_graph = graph->graph;
            } else if (graph->graph != nullptr) {
                CUDA_CHECK(cudaGraphDestroy(graph->graph));
                graph->graph = nullptr;
            }
            CUDA_CHECK(cudaStreamEndCapture(cuda_ctx->stream(), &graph->graph));
            capture_end = measure ? ggml_time_us() : 0;
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
        const char * kind = "instantiate";
        if (graph->instance == nullptr) { // Create executable graph from captured graph.
            CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));
            if (!cuda_ctx->decode_boundary_overlap && cuda_graph_update_required) {
                kind = ggml_cuda_graph_update_executable(cuda_ctx, graph_key) ? "instantiate+update" : "reinstantiate";
            }
        } else if (cuda_graph_update_required) { // Update graph executable
            kind = ggml_cuda_graph_update_executable(cuda_ctx, graph_key) ? "update" : "reinstantiate";
        }
        const int64_t update_end = measure ? ggml_time_us() : 0;
        // Launch graph
        CUDA_CHECK(cudaGraphLaunch(graph->instance, cuda_ctx->stream()));
        const int64_t launch_end = measure ? ggml_time_us() : 0;
        if (retired_graph != nullptr) {
            // Retire the old description while the new executable runs.
            CUDA_CHECK(cudaGraphDestroy(retired_graph));
        }
        if (measure) {
            GGML_LOG_INFO("cuda-graph-profile: uid=%llu kind=%s capture_us=%lld update_us=%lld launch_us=%lld cleanup_us=%lld\n",
                (unsigned long long) cgraph->uid, kind,
                (long long) (capture_end - capture_start), (long long) (update_end - capture_end),
                (long long) (launch_end - update_end), (long long) (ggml_time_us() - launch_end));
        }
#else
        GGML_UNUSED(graph_key);
        graph_evaluated_or_captured = true;
#endif  // USE_CUDA_GRAPH
    }
    return true;
}


#ifdef USE_CUDA_GRAPH
static bool ggml_cuda_graph_set_enabled(ggml_backend_cuda_context * cuda_ctx, uint64_t graph_key) {
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

static void ggml_cuda_graph_invalidate_moe_capture(ggml_cuda_graph * graph, bool reset_properties = false) {
    if (graph == nullptr) {
        return;
    }
#ifdef USE_CUDA_GRAPH
    graph->warmup_complete = false;
    if (reset_properties) {
        graph->uid = 0;
        graph->node_props.clear();
    }
    if (graph->moe_resource_fingerprint != 0) {
        if (graph->instance != nullptr) {
            CUDA_CHECK(cudaGraphExecDestroy(graph->instance));
            graph->instance = nullptr;
        }
        if (graph->graph != nullptr) {
            CUDA_CHECK(cudaGraphDestroy(graph->graph));
            graph->graph = nullptr;
        }
        graph->moe_resource_witnesses.clear();
    }
#endif
    GGML_UNUSED(reset_properties);
    graph->moe_resource_fingerprint = 0;
    graph->moe_registry_generation = 0;
}

#ifdef USE_CUDA_GRAPH
static bool ggml_cuda_graph_has_complete_moe_capture(const ggml_cuda_graph * graph) {
    return graph != nullptr && graph->graph != nullptr && graph->instance != nullptr && graph->warmup_complete &&
        graph->execution_semantic_key != 0 && graph->moe_resource_fingerprint != 0 && !graph->node_props.empty() &&
        !graph->moe_resource_witnesses.empty() && std::all_of(
            graph->moe_resource_witnesses.begin(), graph->moe_resource_witnesses.end(),
            [](const std::weak_ptr<void> & witness) { return !witness.expired(); });
}
#endif

static bool ggml_cuda_required_grouped_certificate_valid(const ggml_cgraph * cgraph) {
    if (cgraph == nullptr) {
        return false;
    }
    const auto & certificate = cgraph->execution_certificate;
    if (certificate.magic != GGML_GRAPH_EXECUTION_CERTIFICATE_MAGIC ||
            certificate.abi_version != GGML_GRAPH_EXECUTION_CERTIFICATE_VERSION ||
            certificate.struct_size != sizeof(certificate) ||
            certificate.flags != GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED ||
            certificate.n_rows == 0 || certificate.n_sequences == 0 || certificate.n_sequences > certificate.n_rows ||
            certificate.owner_namespace == 0 || certificate.owner_generation == 0 ||
            certificate.source_graph_uid == 0 || certificate.split_graph_uid == 0 ||
            certificate.source_graph_uid == certificate.split_graph_uid || certificate.split_graph_uid != cgraph->uid) {
        return false;
    }
    for (uint64_t reserved : certificate.reserved) {
        if (reserved != 0) {
            return false;
        }
    }
    const bool draft = certificate.domain == GGML_GRAPH_EXECUTION_DOMAIN_DRAFT &&
        certificate.row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT &&
        certificate.n_rows == certificate.n_sequences;
    const bool draft_sequential = certificate.domain == GGML_GRAPH_EXECUTION_DOMAIN_DRAFT &&
        certificate.row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL &&
        certificate.n_sequences <= certificate.n_rows;
    const bool mtp = certificate.domain == GGML_GRAPH_EXECUTION_DOMAIN_MTP &&
        certificate.row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT &&
        certificate.n_rows == certificate.n_sequences;
    const bool mtp_sequential = certificate.domain == GGML_GRAPH_EXECUTION_DOMAIN_MTP &&
        certificate.row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL &&
        certificate.n_sequences <= certificate.n_rows;
    const bool target_verification = certificate.domain == GGML_GRAPH_EXECUTION_DOMAIN_MAIN &&
        certificate.row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SPECULATIVE &&
        certificate.n_sequences < certificate.n_rows;
    return draft || draft_sequential || mtp || mtp_sequential || target_verification;
}

static bool ggml_cuda_graph_has_cached_buffer_mmid(const ggml_cgraph * cgraph) {
    if (cgraph == nullptr) {
        return false;
    }
    for (int32_t node_index = 0; node_index < cgraph->n_nodes; ++node_index) {
        const ggml_tensor * node = cgraph->nodes[node_index];
        const ggml_tensor * source = node != nullptr && node->op == GGML_OP_MUL_MAT_ID ? node->src[0] : nullptr;
        if (source != nullptr && source->buffer != nullptr &&
                !ggml_is_empty(node) &&
                ggml_backend_buft_is_cuda_moe_cached(ggml_backend_buffer_get_type(source->buffer))) {
            return true;
        }
    }
    return false;
}

static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    if (cgraph != nullptr && cuda_ctx->moe_ids_cache != nullptr) {
        ++cuda_ctx->moe_ids_cache->dispatch_id;
        ggml_cuda_moe_ids_cache_clear_pending(*cuda_ctx->moe_ids_cache);
    }

    const bool required_requested = cgraph != nullptr &&
        (cgraph->execution_certificate.flags & GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED) != 0;
    const auto required_failure = [&](const char * reason) {
        GGML_LOG_ERROR("moe-cache: required grouped execution failed: %s\n", reason);
        if (cuda_ctx->moe_grouped_context != nullptr) {
            cuda_ctx->moe_grouped_context->record_required_grouped_failure();
        }
        return GGML_STATUS_FAILED;
    };
    if (required_requested && !ggml_cuda_required_grouped_certificate_valid(cgraph)) {
        return required_failure("invalid split certificate");
    }

    const bool required_here = required_requested && ggml_cuda_graph_has_cached_buffer_mmid(cgraph);
    if (required_here && cuda_ctx->moe_grouped_context == nullptr) {
        return required_failure("grouped cache context unavailable");
    }

    ggml_cuda_set_device(cuda_ctx->device);

    bool use_cuda_graph             = false;
    bool cuda_graph_update_required = false;
    bool graph_has_cached_mmid      = false;
#ifdef USE_CUDA_GRAPH
    bool graph_enabled_compatible   = false;
    bool graph_properties_changed   = false;
    bool graph_property_uid_match   = false;
#endif
    const uint64_t graph_key = ggml_cuda_graph_get_key(cgraph);
    ggml_cuda_graph * graph = nullptr;
#ifdef USE_CUDA_GRAPH
    graph = cuda_ctx->cuda_graph(graph_key);
#else
    if (cuda_ctx->moe_grouped_context != nullptr) {
        graph = cuda_ctx->cuda_graph(graph_key);
    }
#endif
    ggml_cuda_moe_graph_property_hint moe_property_hint = GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN;
    ggml_cuda_moe_graph_execution moe_execution;
    std::shared_ptr<ggml_cuda_moe_graph_plan> prepared_plan;
    uint64_t coverage_epoch = 0;
    uint64_t coverage_mmid_fingerprint = 0;
    const void * coverage_nodes = nullptr;
    uint32_t coverage_mmid_count = 0;

    if (graph != nullptr && graph->moe_resource_fingerprint != 0 && cuda_ctx->moe_grouped_context != nullptr &&
            graph->moe_registry_generation != cuda_ctx->moe_grouped_context->state().generation) {
        ggml_cuda_graph_invalidate_moe_capture(graph, true);
    }

#ifdef USE_CUDA_GRAPH
    ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);

    if (graph->is_enabled()) {
        graph_enabled_compatible = ggml_cuda_graph_check_compability(cgraph, &graph_has_cached_mmid);
        if (graph_enabled_compatible) {
            const auto property_probe = ggml_cuda_graph_probe_properties(cuda_ctx, cgraph);
            graph_properties_changed = property_probe.changed;
            graph_property_uid_match = property_probe.uid_match;
            moe_property_hint = graph_properties_changed ?
                GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED : GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED;
        }
    }
#endif // USE_CUDA_GRAPH

    if (cuda_ctx->moe_grouped_context != nullptr) {
        std::shared_ptr<ggml_cuda_moe_graph_plan> local_plan;
        std::shared_ptr<ggml_cuda_moe_graph_plan> * plan = &local_plan;
        bool recover_coverage = graph->moe_coverage_nodes != cgraph->nodes ||
            graph->moe_coverage_n_nodes != cgraph->n_nodes;
#ifdef USE_CUDA_GRAPH
        recover_coverage = recover_coverage || graph_properties_changed;
#endif
        if (recover_coverage) {
            cuda_ctx->recover_moe_graph(cgraph, graph);
        }
        if (graph->moe_coverage_epoch != 0 && graph->moe_coverage_nodes == cgraph->nodes &&
                graph->moe_coverage_n_nodes == cgraph->n_nodes) {
            plan = &graph->moe_graph_plan;
            coverage_epoch = graph->moe_coverage_epoch;
            coverage_mmid_fingerprint = graph->moe_coverage_mmid_fingerprint;
            coverage_nodes = graph->moe_coverage_nodes;
            coverage_mmid_count = graph->moe_coverage_mmid_count;
        }
        const auto prepare_result = cuda_ctx->moe_grouped_context->prepare_graph_execution(
            cgraph, cgraph->uid, moe_property_hint, plan, &moe_execution, coverage_epoch, coverage_nodes,
            coverage_mmid_count, coverage_mmid_fingerprint);
        if (prepare_result == GGML_CUDA_MOE_GRAPH_PREPARE_UNAVAILABLE) {
            ggml_cuda_graph_invalidate_moe_capture(graph, true);
            return required_here ? required_failure("plan preparation failed") : GGML_STATUS_FAILED;
        }
        prepared_plan = *plan;
    }
    if (required_here) {
        if (prepared_plan == nullptr || !ggml_cuda_moe_required_grouped_plan_ready(*prepared_plan, moe_execution)) {
            ggml_cuda_graph_invalidate_moe_capture(graph, true);
            return required_failure("grouped plan unavailable");
        }
        const auto & diagnostics = prepared_plan->coverage_diagnostics();
        if (coverage_epoch == 0 || coverage_nodes != cgraph->nodes || coverage_mmid_fingerprint == 0 ||
                diagnostics.cached_mmid == 0 || coverage_mmid_count != diagnostics.cached_mmid ||
                diagnostics.counts[GGML_CUDA_MOE_GRAPH_COVERAGE_REGISTERED] != diagnostics.cached_mmid) {
            ggml_cuda_graph_invalidate_moe_capture(graph, true);
            return required_failure("incomplete cached MMID coverage");
        }
    }
    const bool moe_dispatch = moe_execution.requires_dispatch();
    if (cuda_ctx->moe_hybrid_active != nullptr && cuda_ctx->moe_hybrid_active->n_regions != 0 &&
            (moe_execution.outcome() != GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED ||
             moe_execution.size() != cuda_ctx->moe_hybrid_active->n_regions)) {
        return GGML_STATUS_FAILED;
    }
    ggml_cuda_moe_graph_dispatch_mode moe_dispatch_mode = GGML_CUDA_MOE_GRAPH_DISPATCH_STAGED;
    uint64_t moe_resource_fingerprint = 0;
    std::vector<std::shared_ptr<void>> moe_resource_leases;
    std::vector<std::weak_ptr<void>> moe_resource_witnesses;
    const auto set_moe_direct = [&]() {
        moe_resource_leases.clear();
        moe_resource_witnesses.clear();
        use_cuda_graph = false;
        cuda_graph_update_required = false;
    };
    const auto force_moe_direct = [&](bool reset_properties = false) {
        set_moe_direct();
        ggml_cuda_graph_invalidate_moe_capture(graph, reset_properties);
    };
    bool retain_grouped_capture = false;
    bool update_grouped_capture = false;
#ifdef USE_CUDA_GRAPH
    if (graph_enabled_compatible && graph_has_cached_mmid && prepared_plan != nullptr && moe_dispatch &&
            moe_execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_PREFILL_STAGED &&
            prepared_plan->outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_PREFILL_STAGED &&
            prepared_plan->has_certified_complete_mmid_inventory() &&
            prepared_plan->size() != 0 && prepared_plan->size() == moe_execution.size() &&
            prepared_plan->graph_node_count() == cgraph->n_nodes && coverage_epoch != 0 &&
            coverage_nodes == cgraph->nodes && coverage_mmid_fingerprint != 0) {
        const auto & diagnostics = prepared_plan->coverage_diagnostics();
        retain_grouped_capture = diagnostics.cached_mmid != 0 &&
            coverage_mmid_count == diagnostics.cached_mmid &&
            diagnostics.counts[GGML_CUDA_MOE_GRAPH_COVERAGE_REGISTERED] == diagnostics.cached_mmid &&
            ggml_cuda_graph_has_complete_moe_capture(graph);
    }

    if (retain_grouped_capture) {
        set_moe_direct();
        graph->uid = 0;
    } else if (graph_enabled_compatible) {
        if (!graph_property_uid_match) {
            ggml_cuda_graph_commit_properties(cuda_ctx, cgraph);
        }
        if (graph_properties_changed && graph->moe_resource_fingerprint != 0) {
            if (cuda_ctx->decode_boundary_overlap && moe_dispatch && moe_execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED &&
                    ggml_cuda_graph_has_complete_moe_capture(graph)) {
                use_cuda_graph = true;
                cuda_graph_update_required = true;
                update_grouped_capture = true;
            } else {
                force_moe_direct();
            }
        } else if (!graph->warmup_complete) {
            // Warmup: need at least 2 calls with no property change on the 2nd call
            if (!graph_properties_changed) {
                graph->warmup_complete = true;
                GGML_LOG_DEBUG("%s: CUDA graph warmup complete\n", __func__);
                use_cuda_graph = true;
                cuda_graph_update_required = true;
            }
            // else: properties changed or first call - execute directly (use_cuda_graph stays false)
        } else if (graph_properties_changed) {
            // Properties changed - reset warmup, execute directly until stable again
            graph->warmup_complete = false;
            GGML_LOG_DEBUG("%s: CUDA graph warmup reset\n", __func__);
        } else {
            use_cuda_graph = true;
            cuda_graph_update_required = graph->instance == nullptr;
        }
    } else if (graph != nullptr && graph->moe_resource_fingerprint != 0) {
        force_moe_direct();
    }
#endif
    if (required_here && !moe_execution.allows_graph_capture()) {
        force_moe_direct();
    }
    if (cuda_ctx->moe_hybrid_active != nullptr) {
        force_moe_direct();
    }
    const bool prefill_resident_witnesses = moe_execution.has_prefill_resident_witnesses();
    if (!retain_grouped_capture && (graph_has_cached_mmid || prefill_resident_witnesses) &&
            moe_execution.outcome() != GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED) {
        force_moe_direct();
    }
    if (moe_dispatch) {
        if (!moe_execution.resolve_streams(ggml_cuda_moe_graph_stream, cuda_ctx)) {
            force_moe_direct(true);
            return required_here ? required_failure("stream resolution failed") : GGML_STATUS_FAILED;
        }
        const auto outcome = moe_execution.outcome();
        if (outcome == GGML_CUDA_MOE_GRAPH_OUTCOME_PREFILL_STAGED ||
                outcome == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_STAGED) {
            force_moe_direct();
        }
        if (outcome == GGML_CUDA_MOE_GRAPH_OUTCOME_PREFILL_GROUPED && !moe_execution.has_coherent_grouped_streams()) {
            force_moe_direct();
            return GGML_STATUS_FAILED;
        } else if (outcome == GGML_CUDA_MOE_GRAPH_OUTCOME_PREFILL_GROUPED) {
            force_moe_direct();
            moe_dispatch_mode = GGML_CUDA_MOE_GRAPH_DISPATCH_DIRECT;
        } else if (outcome == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED && !moe_execution.has_coherent_grouped_streams()) {
            force_moe_direct(required_here);
            if (required_here) {
                return required_failure("incoherent grouped streams");
            }
        } else if (outcome == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED) {
            if (use_cuda_graph && !cuda_graph_update_required) {
                if (graph == nullptr || graph->moe_resource_fingerprint == 0) {
                    force_moe_direct();
                } else {
                    moe_resource_fingerprint = graph->moe_resource_fingerprint;
                }
            } else {
                const bool validate_resources = use_cuda_graph || (graph != nullptr && graph->moe_resource_fingerprint != 0);
                if (validate_resources && !cuda_ctx->moe_grouped_context->graph_resource_fingerprint(
                        moe_execution, cuda_ctx->stream(), &moe_resource_fingerprint,
                        use_cuda_graph ? &moe_resource_leases : nullptr)) {
                    force_moe_direct(required_here);
                    if (required_here) {
                        return required_failure("grouped resource validation failed");
                    }
                } else if (graph != nullptr && graph->moe_resource_fingerprint != 0 &&
                        graph->moe_resource_fingerprint != moe_resource_fingerprint) {
                    force_moe_direct(required_here);
                    if (required_here) {
                        return required_failure("grouped resource fingerprint changed");
                    }
                } else if (use_cuda_graph) {
                    try {
                        moe_resource_witnesses.reserve(moe_resource_leases.size());
                        for (const auto & lease : moe_resource_leases) {
                            moe_resource_witnesses.emplace_back(lease);
                        }
                    } catch (const std::bad_alloc &) {
                        force_moe_direct();
                    }
                }
            }
            if (use_cuda_graph && update_grouped_capture) {
                bool same_owners = graph->moe_resource_witnesses.size() == moe_resource_leases.size();
                for (size_t i = 0; same_owners && i < moe_resource_leases.size(); ++i) {
                    const auto & witness = graph->moe_resource_witnesses[i];
                    const auto & lease = moe_resource_leases[i];
                    same_owners = !witness.owner_before(lease) && !lease.owner_before(witness);
                }
                if (!same_owners) {
                    force_moe_direct();
                }
            }
            moe_dispatch_mode = !use_cuda_graph ? GGML_CUDA_MOE_GRAPH_DISPATCH_DIRECT :
                cuda_graph_update_required ? GGML_CUDA_MOE_GRAPH_DISPATCH_CAPTURE : GGML_CUDA_MOE_GRAPH_DISPATCH_REPLAY;
        } else if (outcome == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR) {
            force_moe_direct();
            moe_dispatch_mode = GGML_CUDA_MOE_GRAPH_DISPATCH_DIRECT;
        } else if (!retain_grouped_capture && graph != nullptr && graph->moe_resource_fingerprint != 0) {
            force_moe_direct();
        }
        if (!cuda_ctx->moe_grouped_context->begin_graph_dispatch(&moe_execution, moe_dispatch_mode)) {
            force_moe_direct(true);
            return required_here ? required_failure("grouped dispatch admission failed") : GGML_STATUS_FAILED;
        }
        if ((moe_dispatch_mode == GGML_CUDA_MOE_GRAPH_DISPATCH_CAPTURE ||
                moe_dispatch_mode == GGML_CUDA_MOE_GRAPH_DISPATCH_REPLAY) &&
                !cuda_ctx->moe_grouped_context->activate_graph_resources(
                    &moe_execution, moe_dispatch_mode, moe_resource_fingerprint,
                    moe_dispatch_mode == GGML_CUDA_MOE_GRAPH_DISPATCH_REPLAY ? &graph->moe_resource_witnesses : nullptr)) {
            if (!cuda_ctx->moe_grouped_context->finish_graph_dispatch(&moe_execution)) {
                force_moe_direct(true);
                return required_here ? required_failure("capture activation cleanup failed") : GGML_STATUS_FAILED;
            }
            force_moe_direct(required_here);
            if (required_here) {
                return required_failure("capture activation failed");
            }
            moe_dispatch_mode = GGML_CUDA_MOE_GRAPH_DISPATCH_DIRECT;
            if (!cuda_ctx->moe_grouped_context->begin_graph_dispatch(&moe_execution, moe_dispatch_mode)) {
                force_moe_direct(true);
                return GGML_STATUS_FAILED;
            }
        }
    } else if (graph != nullptr && graph->moe_resource_fingerprint != 0) {
        force_moe_direct();
    }

    if (cuda_ctx->moe_grouped_context != nullptr && (!use_cuda_graph || cuda_graph_update_required)) {
        cuda_ctx->moe_grouped_context->configure_early_router(cgraph, &moe_execution, cuda_ctx->stream(), use_cuda_graph && cuda_graph_update_required, *cuda_ctx);
    }
    if (use_cuda_graph && cuda_graph_update_required) {
        // Start CUDA graph capture
        {
            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            ggml_cuda_lock_counter.fetch_add(1, std::memory_order_relaxed);
        }

        CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));
    }

    const bool graph_evaluated = ggml_cuda_graph_evaluate_and_capture(
        cuda_ctx, cgraph, use_cuda_graph, cuda_graph_update_required, graph_key, &moe_execution);

    if (graph_evaluated && moe_dispatch_mode == GGML_CUDA_MOE_GRAPH_DISPATCH_CAPTURE) {
        graph->moe_resource_fingerprint = moe_resource_fingerprint;
        graph->moe_registry_generation = prepared_plan->registry_generation();
        graph->moe_resource_witnesses = std::move(moe_resource_witnesses);
    }

    const bool dispatch_finished = !moe_dispatch || cuda_ctx->moe_grouped_context->finish_graph_dispatch(&moe_execution);
    if (!graph_evaluated || !dispatch_finished) {
        force_moe_direct(true);
        return required_here ? required_failure(
            !graph_evaluated ? "grouped evaluator failed" : "grouped finalization failed") : GGML_STATUS_FAILED;
    }

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
            ggml_cuda_rms_norm_gated_match norm_match;
            if (ggml_cuda_match_rms_norm_gated(cgraph, i, norm_match)) {
                add_alloc_deps(i, i + norm_match.node_count - 1);
                i += norm_match.node_count - 1;
                continue;
            }

            if (cgraph->nodes[i]->op == GGML_OP_SSM_CONV && ggml_cuda_match_ssm_conv_qk(cgraph, i)) {
                ggml_tensor * last = cgraph->nodes[i + 7];
                params->add_alloc_dep(params->user_data, cgraph->nodes[i]->src[0], last);
                params->add_alloc_dep(params->user_data, cgraph->nodes[i]->src[1], last);
                params->add_alloc_dep(params->user_data, cgraph->nodes[i + 1], last);
                params->add_alloc_dep(params->user_data, cgraph->nodes[i + 4], last);
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
            ggml_cuda_gdn_packed_match packed_projection;
            if (cgraph->nodes[i]->op == GGML_OP_MUL_MAT &&
                    ggml_cuda_match_gdn_packed_projection(cgraph, i, cuda_ctx->device, packed_projection)) {
                add_alloc_deps(i, i + packed_projection.count - 1);
            }
            ggml_cuda_gdn_projection_match projection;
            if (cgraph->nodes[i]->op == GGML_OP_MUL_MAT &&
                    (ggml_cuda_match_gdn_projections(cgraph, i, cuda_ctx->device, ggml_cuda_gdn_projection::VECTOR, false, &projection) ||
                     ggml_cuda_match_gdn_projections(cgraph, i, cuda_ctx->device, ggml_cuda_gdn_projection::MMF, false, &projection) ||
                     ggml_cuda_match_gdn_projections(cgraph, i, cuda_ctx->device, ggml_cuda_gdn_projection::BF16_ROUNDED, false, &projection))) {
                add_alloc_deps(i, i + projection.count - 1);
            }
            if (cgraph->nodes[i]->op == GGML_OP_MUL_MAT &&
                    ggml_cuda_match_gdn_projections(cgraph, i, cuda_ctx->device, ggml_cuda_gdn_projection::CUBLAS)) {
                add_alloc_deps(i, i + 8);
            }
            ggml_cuda_gdn_packed_match packed_post;
            if (cgraph->nodes[i]->op == GGML_OP_CONT && ggml_cuda_match_gdn_packed(cgraph, i, packed_post)) {
                add_alloc_deps(i, i + packed_post.count - 1);
                params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(packed_post.packed), cgraph->nodes[i + packed_post.count - 1]);
            }
            ggml_cuda_gdn_post_match post;
            if (cgraph->nodes[i]->op == GGML_OP_ADD &&
                    (ggml_cuda_match_gdn_post(cgraph, i, post) || ggml_cuda_match_gdn_post(cgraph, i, post, -1, false, true))) {
                add_alloc_deps(i, i + post.count - 1);
            }
            if (cgraph->nodes[i]->op == GGML_OP_MUL_MAT_ID) {
                const int count = ggml_cuda_match_mmf_id_pair(cgraph, i, cuda_ctx->device);
                if (count != 0) {
                    add_alloc_deps(i, i + count - 1);
                    i += count - 1;
                    continue;
                }
            }
            const auto hc_up = ggml_cuda_match_hc_up_shape(*cuda_ctx, cgraph, i);
            if (hc_up.count) {
                params->add_alloc_dep(params->user_data, hc_up.mm->src[0], hc_up.dst);
                params->add_alloc_dep(params->user_data, hc_up.mm->src[1], hc_up.dst);
            }

            if (cgraph->nodes[i]->op == GGML_OP_MUL_MAT_ID) {
                const int count = ggml_cuda_match_mmq_id_pair(cgraph, i, cuda_ctx->device);
                if (count != 0) {
                    add_alloc_deps(i, i + count - 1);
                    i += count - 1;
                    continue;
                }
            }
            ggml_cuda_moe_weighted_reduction_match match;
            if (ggml_cuda_match_moe_weighted_reduction(cgraph, i, match)) {
                ggml_cuda_moe_weighted_reduction_alloc_deps(match, params);
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
    const uint64_t graph_key = ggml_cuda_graph_get_key(cgraph);
    const bool use_cuda_graph = ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);
#else
    const bool use_cuda_graph = false;
#endif

    static bool enable_graph_optimization = [] {
        const char * env     = getenv("GGML_CUDA_GRAPH_OPT");
        return env != nullptr && atoi(env) == 1;
    }();

    if (!enable_graph_optimization) {
        cuda_ctx->certify_moe_graph(cgraph);
        return;
    }

    ggml_cuda_stream_context & stream_context = cuda_ctx->stream_context();
    stream_context.reset();

    if (!use_cuda_graph) {
        cuda_ctx->certify_moe_graph(cgraph);
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
    cuda_ctx->certify_moe_graph(cgraph);
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
        case GGML_OP_CUSTOM:
            return ggml_cuda_staged_input_supports(op);
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
                if (op->op == GGML_OP_MUL_MAT_ID) {
                    return (ggml_cuda_mmid_source_capability_for(a->type).flags & GGML_CUDA_MMID_SOURCE_ADVERTISED) != 0;
                }
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
            return ggml_cuda_gated_delta_net_supported(dev_ctx->device, op);
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
    if ((ggml_backend_buft_is_cuda(buft) && buft->device == dev) ||
        (integrated && ggml_backend_buft_is_cuda_host(buft))) {
        return true;
    }
    // CUDA_MoE_Cached is host-pinned but reads through the CUDA backend's
    // mul_mat_id dispatch hook (which stages slabs to a GPU slot pool on
    // miss). The buffer's device is set to the first CUDA device; let any
    // CUDA device claim it so the scheduler routes ops to CUDA instead of
    // CPU.
    if (ggml_backend_buft_is_cuda_moe_cached(buft)) {
        return true;
    }
    return false;
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

static bool ggml_backend_cuda_flash_attn_causal_prefix_supported(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
#if defined(FLASH_ATTN_AVAILABLE) && defined(GGML_CUDA_COMPACT_CAUSAL_MASK)
    return true;
#else
    return false;
#endif
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
static bool ggml_backend_cuda_required_grouped_execution_supported(ggml_backend_t backend) {
    return backend != nullptr && ggml_backend_is_cuda(backend);
}
#endif
void ggml_backend_cuda_set_decode_boundary_overlap(ggml_backend_t backend, bool enabled) {
    GGML_ASSERT(ggml_backend_is_cuda(backend));
    auto * ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    GGML_ASSERT(ctx->cuda_graphs.empty());
    ctx->decode_boundary_overlap = enabled;
}

static void * ggml_backend_cuda_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (strcmp(name, GGML_STAGED_INPUT_PROC) == 0) {
        return (void *) ggml_cuda_staged_input_api;
    }
    if (strcmp(name, GGML_BACKEND_MOE_HYBRID_STAGED_PENDING_V1_PROC_NAME) == 0) {
        return (void *) ggml_cuda_staged_input_pending_for_test;
    }
    if (strcmp(name, "ggml_backend_cuda_set_decode_boundary_overlap") == 0) {
        return (void *) ggml_backend_cuda_set_decode_boundary_overlap;
    }
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
    if (strcmp(name, GGML_BACKEND_MOE_CACHE_BUFFER_TYPE_PROC_NAME) == 0) {
        return (void *) ggml_backend_cuda_moe_cached_buffer_type;
    }
    if (strcmp(name, GGML_BACKEND_MOE_CACHE_BOUNDED_BUFFER_TYPE_PROC_NAME) == 0) {
        return (void *) ggml_backend_cuda_moe_cached_bounded_buffer_type;
    }
    if (strcmp(name, GGML_BACKEND_MOE_STAGING_SIZE_V1_PROC_NAME) == 0) {
        return (void *) ggml_backend_cuda_moe_staging_size_v1;
    }
    if (strcmp(name, GGML_BACKEND_MOE_DEVICE_SIZE_V1_PROC_NAME) == 0) {
        return (void *) ggml_backend_cuda_moe_device_size_v1;
    }
    if (strcmp(name, GGML_BACKEND_MOE_DEVICE_SIZE_V2_PROC_NAME) == 0) {
        return (void *) ggml_backend_cuda_moe_device_size_v2;
    }
    if (strcmp(name, GGML_BACKEND_MOE_CACHE_FREE_BUFFER_TYPE_PROC_NAME) == 0) {
        return (void *) ggml_backend_cuda_moe_cached_free_buffer_type;
    }
    if (strcmp(name, GGML_BACKEND_MOE_CACHE_CONFIGURE_SOURCES_PROC_NAME) == 0) {
        return (void *) ggml_backend_cuda_moe_cached_configure_sources;
    }
    if (strcmp(name, GGML_BACKEND_MOE_CACHE_IS_BUFFER_TYPE_PROC_NAME) == 0) {
        return (void *) ggml_backend_buft_is_cuda_moe_cached;
    }
    if (strcmp(name, GGML_BACKEND_MOE_CACHE_BUFFER_FROM_HOST_PTR_PROC_NAME) == 0) {
        return (void *) ggml_backend_cuda_moe_cached_buffer_from_host_ptr;
    }
    if (strcmp(name, GGML_BACKEND_MOE_CACHE_WRITABLE_LOAD_DATA_PROC_NAME) == 0) {
        return (void *) ggml_backend_cuda_moe_cached_writable_load_data;
    }
    if (strcmp(name, GGML_BACKEND_MOE_CACHE_READABLE_SOURCE_PROC_NAME) == 0) {
        return (void *) ggml_backend_cuda_moe_cached_readable_source;
    }
    if (strcmp(name, GGML_BACKEND_MOE_CACHE_SET_DEBUG_PROC_NAME) == 0) {
        return (void *) ggml_backend_cuda_moe_set_debug_mm;
    }
    if (strcmp(name, GGML_BACKEND_MOE_EARLY_ROUTER_SET_ENABLED_PROC_NAME) == 0) {
        return (void *) ggml_backend_cuda_moe_early_router_set_enabled;
    }
    if (strcmp(name, GGML_BACKEND_MOE_EARLY_ROUTER_SET_MAX_ROWS_PROC_NAME) == 0) {
        return (void *) ggml_backend_cuda_moe_early_router_set_max_rows;
    }
    if (strcmp(name, GGML_BACKEND_MOE_CACHE_LOG_AND_RESET_STATS_PROC_NAME) == 0) {
        return (void *) ggml_backend_cuda_moe_log_and_reset_stats;
    }
    if (strcmp(name, GGML_BACKEND_MOE_CANDIDATE_REPLACE_V1_PROC_NAME) == 0) {
        return (void *)ggml_backend_cuda_moe_candidate_replace_v1;
    }
    if (strcmp(name, GGML_BACKEND_MOE_STATISTICS_INITIALIZE_V2_PROC_NAME) == 0) {
        return (void *) ggml_backend_cuda_moe_statistics_initialize_v2;
    }
    if (strcmp(name, GGML_BACKEND_MOE_STATISTICS_INITIALIZE_V1_PROC_NAME) == 0) {
        return (void *) ggml_backend_cuda_moe_statistics_initialize_v1;
    }
    if (strcmp(name, GGML_BACKEND_MOE_PROFILE_INITIALIZE_V1_PROC_NAME) == 0) {
        return (void *) ggml_backend_cuda_moe_profile_initialize_v1;
    }
    if (strcmp(name, GGML_BACKEND_MOE_CANDIDATE_REPLACE_V2_PROC_NAME) == 0) {
        return (void *)ggml_backend_cuda_moe_candidate_replace_v2;
    }
    if (strcmp(name, GGML_BACKEND_MOE_CANDIDATE_REPLACE_CAPACITIES_V1_PROC_NAME) == 0) {
        return (void *) ggml_backend_cuda_moe_candidate_replace_capacities_v1;
    }
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (strcmp(name, GGML_BACKEND_REQUIRED_GROUPED_EXECUTION_SUPPORTED_PROC_NAME) == 0) {
        return (void *)ggml_backend_cuda_required_grouped_execution_supported;
    }
    if (strcmp(name, GGML_BACKEND_MOE_HYBRID_V1_PROC_NAME) == 0) {
        return (void *) ggml_cuda_moe_hybrid_api;
    }
    if (strcmp(name, GGML_BACKEND_MOE_HYBRID_FIDELITY_V1_PROC_NAME) == 0) {
        return (void *) ggml_cuda_moe_fidelity_hybrid_api;
    }
    if (strcmp(name, GGML_BACKEND_MOE_SOURCE_CORE_V1_PROC_NAME) == 0) {
        return (void *) ggml_cuda_moe_source_core_api;
    }
    if (strcmp(name, GGML_BACKEND_MOE_HYBRID_WINDOW_PROBE_V1_PROC_NAME) == 0) {
        return (void *) ggml_cuda_moe_hybrid_window_probe;
    }
    if (strcmp(name, GGML_CUDA_MOE_FIDELITY_FIXTURE_V1_PROC_NAME) == 0) {
        return (void *) ggml_cuda_moe_fidelity_fixture_api;
    }
    if (strcmp(name, GGML_CUDA_MOE_FIDELITY_WINDOW_V1_PROC_NAME) == 0) {
        return (void *) ggml_cuda_moe_fidelity_window_api;
    }
#endif
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_cuda_get_features;
    }
    if (strcmp(name, "ggml_backend_flash_attn_causal_prefix_supported") == 0) {
        return (void *)ggml_backend_cuda_flash_attn_causal_prefix_supported;
    }
    if (strcmp(name, "ggml_backend_cuda_trim_transient_pools") == 0) {
        return (void *)ggml_backend_cuda_trim_transient_pools;
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
