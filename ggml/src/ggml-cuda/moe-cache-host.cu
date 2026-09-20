#include "moe-cache-host.cuh"
#include "moe-cache.cuh"
#include "common.cuh"

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <memory>
#include <new>

#ifdef __linux__
#include <unistd.h>
#endif

size_t moe_host_page_size() {
#ifdef __linux__
    static const long page = sysconf(_SC_PAGESIZE);
    return page > 0 ? static_cast<size_t>(page) : 0;
#else
    return 4096;
#endif
}

bool moe_host_round_size(size_t size, size_t & rounded) {
    const size_t page = moe_host_page_size();
    if (page == 0 || size > SIZE_MAX - (page - 1)) {
        return false;
    }
    rounded = (size + page - 1) / page * page;
    return true;
}

moe_host_budget * moe_host_budget_for(ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_cuda_moe_cached(buft) ? static_cast<moe_host_budget *>(buft->context) : nullptr;
}

moe_host_copy_worker::~moe_host_copy_worker() { stop(); }

bool moe_host_copy_worker::start(bool fail_for_test) {
    std::call_once(start_once, [this, fail_for_test] {
        try {
            if (fail_for_test) {
                throw std::bad_alloc();
            }
            std::lock_guard<std::mutex> lock(mutex);
            thread = std::thread([this] {
                std::unique_lock<std::mutex> lock(mutex);
                for (;;) {
                    ready_cv.wait(lock, [this] { return stopping || ready; });
                    if (stopping) {
                        return;
                    }
                    ready = false;
                    auto task = function;
                    auto task_data = data;
                    lock.unlock();
                    task(task_data);
                    lock.lock();
                    done = true;
                    done_cv.notify_one();
                }
            });
        } catch (const std::exception &) {
            // Keep serial copies if the helper cannot start.
        }
    });
    return thread.joinable();
}

bool moe_host_copy_worker::try_submit(void (*task)(void *), void * task_data) {
    std::unique_lock<std::mutex> lock(mutex, std::try_to_lock);
    if (!lock.owns_lock() || !thread.joinable() || busy || stopping) {
        return false;
    }
    busy = true;
    done = false;
    function = task;
    data = task_data;
    ready = true;
    ++submitted;
    ready_cv.notify_one();
    return true;
}

void moe_host_copy_worker::wait() {
    std::unique_lock<std::mutex> lock(mutex);
    GGML_ASSERT(busy);
    done_cv.wait(lock, [this] { return done; });
    // The submitting callback owns the job until it observes completion.
    busy = false;
    function = nullptr;
    data = nullptr;
}

void moe_host_copy_worker::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        GGML_ASSERT(!busy);
        stopping = true;
    }
    ready_cv.notify_one();
    if (thread.joinable()) {
        thread.join();
    }
}

moe_host_budget::moe_host_budget(size_t limit, bool automatic) : limit(limit), automatic(automatic) {}

moe_host_budget::~moe_host_budget() {
    copy_worker.stop();
    GGML_ASSERT(staging_bytes == 0 && staging_optional_bytes == 0);
    GGML_LOG_INFO("moe-cache-host: limit=%zu source=%zu staging_peak=%zu pinned_peak=%zu materialized=%llu helper_jobs=%llu\n",
        limit, source_bytes, staging_peak, source_bytes + staging_peak, (unsigned long long) materialized_bytes.load(std::memory_order_relaxed),
        (unsigned long long) copy_worker.submitted);
    for (const auto & range : registered) {
        CUDA_CHECK(cudaHostUnregister(reinterpret_cast<void *>(range.begin)));
    }
    for (auto buffer : backing) {
        ggml_backend_buffer_free(buffer);
    }
}

void moe_host_budget::retain() { references.fetch_add(1, std::memory_order_relaxed); }

void moe_host_budget::release() {
    if (references.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        delete this;
    }
}

moe_host_allocation::moe_host_allocation(moe_host_budget * owner, size_t base_size, size_t tile_stride, uint32_t max_tiles, bool optional, bool fail_growth_for_test, size_t cached_prefix_bytes) : owner(owner), size(base_size) {
    if (owner == nullptr || size == 0 || max_tiles == 0 || !moe_host_round_size(size, pinned_size) || getenv("GGML_CUDA_NO_PINNED") != nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(owner->mutex);
    if (pinned_size > owner->limit - owner->source_bytes - owner->staging_bytes) {
        GGML_LOG_ERROR("moe-cache-host: staging requires %zu bytes, budget has %zu bytes free\n", pinned_size,
            owner->limit - owner->source_bytes - owner->staging_bytes);
        return;
    }
    const size_t base_pinned_size = pinned_size;
    if (optional) {
        const size_t committed = owner->source_bytes + std::max(owner->staging_bytes, owner->staging_reserved + owner->staging_optional_bytes);
        if (committed > owner->limit || pinned_size > owner->limit - committed) {
            return;
        }
        optional_bytes = pinned_size;
    }
    if (tile_stride != 0 && max_tiles > 1) {
        // Tile growth must leave the unallocated mandatory scratch available.
        const size_t mandatory = std::max(owner->staging_reserved, owner->staging_bytes - owner->staging_optional_bytes + base_pinned_size);
        const size_t available = owner->limit - owner->source_bytes - owner->staging_optional_bytes;
        GGML_ASSERT(mandatory <= available);
        const size_t page = moe_host_page_size();
        const size_t capacity = (base_pinned_size + (available - mandatory)) / page * page;
        const size_t additional = std::min(static_cast<size_t>(max_tiles - 1), (capacity - size) / tile_stride);
        size += additional * tile_stride;
        tiles += static_cast<uint32_t>(additional);
        GGML_ASSERT(moe_host_round_size(size, pinned_size));
        optional_bytes = pinned_size - base_pinned_size;
    }
    const auto allocate = [&]() {
        const unsigned flags = cudaHostAllocMapped | cudaHostAllocPortable;
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && (defined(__x86_64__) || defined(_M_X64))
        size_t prefix_pinned_size = 0, payload_pinned_size = 0;
        const size_t mandatory = std::max(owner->staging_reserved, owner->staging_bytes - owner->staging_optional_bytes + base_pinned_size);
        const size_t committed = owner->source_bytes + mandatory + owner->staging_optional_bytes + optional_bytes;
        if (cached_prefix_bytes != 0 && cached_prefix_bytes < size && moe_host_round_size(cached_prefix_bytes, prefix_pinned_size) &&
                moe_host_round_size(size - cached_prefix_bytes, payload_pinned_size) && payload_pinned_size <= SIZE_MAX - prefix_pinned_size &&
                committed <= owner->limit && prefix_pinned_size + payload_pinned_size - pinned_size <= owner->limit - committed) {
            // Extra page padding is optional; preserve the reserved mandatory scratch.
            cudaError_t error = cudaHostAlloc(&data, prefix_pinned_size, flags);
            if (error == cudaSuccess) {
                error = cudaHostAlloc(&payload, payload_pinned_size, flags | cudaHostAllocWriteCombined);
            }
            if (error == cudaSuccess) {
                error = cudaHostGetDevicePointer(&payload_alias, payload, 0);
            }
            if (error == cudaSuccess) {
                optional_bytes += prefix_pinned_size + payload_pinned_size - pinned_size;
                pinned_size = prefix_pinned_size + payload_pinned_size;
                split_payload = true;
                return cudaSuccess;
            }
            if (payload != nullptr) {
                CUDA_CHECK(cudaFreeHost(payload));
                payload = nullptr;
            }
            if (data != nullptr) {
                CUDA_CHECK(cudaFreeHost(data));
                data = nullptr;
            }
            payload_alias = nullptr;
            (void) cudaGetLastError();
        }
#endif
        return cudaHostAlloc(&data, pinned_size, flags);
    };
    cudaError_t error = fail_growth_for_test && optional_bytes != 0 ? cudaErrorMemoryAllocation : allocate();
    if (error != cudaSuccess && optional_bytes != 0 && !optional) {
        GGML_LOG_WARN("moe-cache-host: tile growth failed: %s; retrying mandatory staging\n", cudaGetErrorString(error));
        (void) cudaGetLastError();
        size = base_size;
        pinned_size = base_pinned_size;
        optional_bytes = 0;
        tiles = 1;
        data = nullptr;
        error = allocate();
    }
    if (error != cudaSuccess) {
        GGML_LOG_ERROR("moe-cache-host: failed to pin %zu staging bytes: %s\n", pinned_size, cudaGetErrorString(error));
        (void) cudaGetLastError();
        return;
    }
    if (cudaHostGetDevicePointer(&device_alias, data, 0) != cudaSuccess) {
        (void) cudaGetLastError();
        if (split_payload) {
            (void) cudaFreeHost(payload);
            payload = nullptr;
        }
        (void) cudaFreeHost(data);
        data = nullptr;
        return;
    }
    if (!split_payload) {
        payload = static_cast<char *>(data) + cached_prefix_bytes;
        payload_alias = static_cast<char *>(device_alias) + cached_prefix_bytes;
    }
    owner->staging_bytes += pinned_size;
    owner->staging_optional_bytes += optional_bytes;
    owner->staging_peak = std::max(owner->staging_peak, owner->staging_bytes);
    owner->retain();
}

moe_host_allocation::~moe_host_allocation() {
    if (data != nullptr) {
        if (split_payload) {
            CUDA_CHECK(cudaFreeHost(payload));
        }
        CUDA_CHECK(cudaFreeHost(data));
        {
            std::lock_guard<std::mutex> lock(owner->mutex);
            owner->staging_bytes -= pinned_size;
            owner->staging_optional_bytes -= optional_bytes;
        }
        owner->release();
    }
}

namespace {

static std::atomic<bool> g_fail_full_pinning_for_test{false};

// ---------------------------------------------------------------------------
// ggml buffer type: CUDA_MoE_Cached
//
// Allocator and free hooks mirror ggml_backend_cuda_host_buffer_type --
// pinned host memory via cudaMallocHost / cudaFreeHost. The only thing that
// changes is the type's name (so the dispatch hook can identify it) and the
// alignment carried over from the CPU buffer. The GPU slot pool is created
// lazily by the dispatch hook on first cached-tensor access; the buffer type
// itself stays cheap and stateless.
// ---------------------------------------------------------------------------

static const char * ggml_backend_cuda_moe_cached_buffer_type_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return GGML_CUDA_NAME "_MoE_Cached";
}

struct moe_host_buffer {
    moe_host_budget * owner;
    ggml_backend_buffer_t backing;
    bool read_only;
    bool auto_pin;
};

static void moe_host_buffer_free(ggml_backend_buffer_t buffer) {
    auto * context = static_cast<moe_host_buffer *>(buffer->context);
    auto * owner = context->owner;
    auto * backing = context->backing;
    delete context;
    if (owner != nullptr) {
        owner->release();
    } else {
        ggml_backend_buffer_free(backing);
    }
}

static void * moe_host_buffer_base(ggml_backend_buffer_t buffer) {
    return ggml_backend_buffer_get_base(static_cast<moe_host_buffer *>(buffer->context)->backing);
}

static void moe_host_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_buffer_clear(static_cast<moe_host_buffer *>(buffer->context)->backing, value);
}

static ggml_backend_buffer_t moe_host_buffer_wrap(
        ggml_backend_buffer_type_t buft, ggml_backend_buffer_t backing, bool read_only, bool auto_pin = false) {
    if (backing == nullptr) {
        return nullptr;
    }
    auto * owner = moe_host_budget_for(buft);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> storage(backing, ggml_backend_buffer_free);
    std::unique_ptr<moe_host_buffer> context(new moe_host_buffer{owner, backing, read_only, auto_pin});
    auto iface = backing->iface;
    iface.free_buffer = moe_host_buffer_free;
    iface.get_base = moe_host_buffer_base;
    iface.clear = moe_host_buffer_clear;
    if (owner != nullptr) {
        owner->retain();
    }
    ggml_backend_buffer_t result = nullptr;
    try {
        result = ggml_backend_buffer_init(buft, iface, context.get(), backing->size);
        if (result != nullptr) {
            if (owner != nullptr) {
                std::lock_guard<std::mutex> lock(owner->mutex);
                owner->backing.push_back(backing);
            }
            storage.release();
            context.release();
            return result;
        }
    } catch (...) {
        if (result != nullptr) {
            result->iface.free_buffer = nullptr;
            ggml_backend_buffer_free(result);
        }
    }
    if (owner != nullptr) {
        owner->release();
    }
    return nullptr;
}

static void ggml_backend_cuda_moe_cached_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    CUDA_CHECK(cudaFreeHost(buffer->context));
}

static void * ggml_cuda_moe_cached_pinned_malloc(size_t size) {
    if (getenv("GGML_CUDA_NO_PINNED") != nullptr) {
        return nullptr;
    }
    if (g_fail_full_pinning_for_test.load(std::memory_order_relaxed)) {
        return nullptr;
    }
    void * ptr = nullptr;
    cudaError_t err = cudaMallocHost((void **) &ptr, size);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        GGML_LOG_WARN("%s: failed to allocate %.2f MiB of pinned memory: %s; using pageable fallback with automatic group registration\n",
                      __func__, size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return nullptr;
    }
    return ptr;
}

static ggml_backend_buffer_t ggml_backend_cuda_moe_cached_buffer_type_alloc_buffer(
        ggml_backend_buffer_type_t buft, size_t size) {

    if (moe_host_budget_for(buft) != nullptr) {
        return moe_host_buffer_wrap(buft, ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size), false);
    }

    void * ptr = ggml_cuda_moe_cached_pinned_malloc(size);

    if (ptr == nullptr) {
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
        if (buffer == nullptr) {
            return nullptr;
        }
        if (getenv("GGML_CUDA_NO_PINNED") == nullptr) {
            auto * wrapped = moe_host_buffer_wrap(buft, buffer, false, true);
            if (wrapped != nullptr) {
                return wrapped;
            }
            buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
            if (buffer == nullptr) {
                return nullptr;
            }
        }
        // Keep CUDA cache dispatch and the CPU backing's deallocator.
        buffer->buft = buft;
        return buffer;
    }

    std::unique_ptr<void, decltype(&cudaFreeHost)> allocation(ptr, cudaFreeHost);
    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    if (buffer == nullptr) {
        return nullptr;
    }
    buffer->buft             = buft;
    buffer->iface.free_buffer = ggml_backend_cuda_moe_cached_buffer_free_buffer;
    allocation.release();
    return buffer;
}

// is_host MUST return false (or be NULL) for this buffer type, even though
// the data is technically in pinned host memory. If is_host returns true,
// ggml's scheduler treats tensors here as CPU-backend-resident and routes
// mul_mat_id ops to the CPU backend, completely bypassing our dispatch hook.
// We rely on CUDA reading the pinned mapping directly via cudaMemcpyAsync
// for the H2D copy in the dispatch hook.
static bool ggml_backend_cuda_moe_cached_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return false;
}

} // namespace

bool moe_host_buffer_is_read_only(ggml_backend_buffer_t buffer) {
    return static_cast<const moe_host_buffer *>(buffer->context)->read_only;
}

bool moe_host_buffer_auto_pin(ggml_backend_buffer_t buffer) {
    return buffer != nullptr && buffer->iface.free_buffer == moe_host_buffer_free &&
        static_cast<const moe_host_buffer *>(buffer->context)->auto_pin;
}

extern "C"
void ggml_cuda_moe_cache_fail_full_pinning_for_test(bool fail) {
    g_fail_full_pinning_for_test.store(fail, std::memory_order_relaxed);
}

extern "C"
ggml_backend_buffer_type_t ggml_backend_cuda_moe_cached_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cuda_buffer_type_moe_cached = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_cuda_moe_cached_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_cuda_moe_cached_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .is_host          = */ ggml_backend_cuda_moe_cached_buffer_type_is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), 0),
        /* .context  = */ nullptr,
    };
    return &ggml_backend_cuda_buffer_type_moe_cached;
}

extern "C"
bool ggml_backend_buft_is_cuda_moe_cached(ggml_backend_buffer_type_t buft) {
    return buft != nullptr
        && buft->iface.get_name == ggml_backend_cuda_moe_cached_buffer_type_name;
}

extern "C"
ggml_backend_buffer_t ggml_backend_cuda_moe_cached_buffer_from_host_ptr(ggml_backend_buffer_type_t buft, void * ptr, size_t size) {
    if (moe_host_budget_for(buft) != nullptr) {
        return moe_host_buffer_wrap(buft, ggml_backend_cpu_buffer_from_ptr(ptr, size), true);
    }
    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    if (buffer == nullptr) {
        return nullptr;
    }

    buffer->buft = ggml_backend_cuda_moe_cached_buffer_type();
    return buffer;
}

extern "C"
ggml_backend_buffer_type_t ggml_backend_cuda_moe_cached_bounded_buffer_type(size_t bytes) {
    if (bytes == 0) {
        return ggml_backend_cuda_moe_cached_buffer_type();
    }
    auto * owner = new (std::nothrow) moe_host_budget(bytes);
    if (owner == nullptr) {
        return nullptr;
    }
    owner->type = *ggml_backend_cuda_moe_cached_buffer_type();
    owner->type.context = owner;
    return &owner->type;
}

extern "C"
void ggml_backend_cuda_moe_cached_free_buffer_type(ggml_backend_buffer_type_t buft) {
    auto * owner = moe_host_budget_for(buft);
    if (owner != nullptr) {
        owner->release();
    }
}

bool moe_host_register(moe_host_budget & owner, const std::vector<moe_host_source *> & sources, bool require_identity, uint32_t group) {
    if (sources.empty()) {
        return true;
    }
    size_t bytes = 0;
    auto decline = [&](const char * reason, cudaError_t error = cudaSuccess) {
        GGML_LOG_WARN("moe-cache-host: admission group=%u banks=%zu bytes=%zu reason=%s cuda=%s\n",
            group, sources.size(), bytes, reason, error == cudaSuccess ? "none" : cudaGetErrorString(error));
        if (error != cudaSuccess) {
            (void) cudaGetLastError();
        }
        return false;
    };
    const size_t page = moe_host_page_size();
    if (page == 0) {
        return decline("invalid_page_size");
    }
    std::vector<moe_host_range> ranges;
    for (const auto * source : sources) {
        const uintptr_t begin = reinterpret_cast<uintptr_t>(source->data);
        if (source->size > UINTPTR_MAX - begin || begin + source->size > UINTPTR_MAX - (page - 1)) {
            return decline("source_range_overflow");
        }
        ranges.push_back({begin - begin % page, (begin + source->size + page - 1) / page * page, source->read_only});
    }
    std::sort(ranges.begin(), ranges.end(), [](const moe_host_range & a, const moe_host_range & b) { return a.begin < b.begin; });
    std::vector<moe_host_range> merged;
    for (const auto & range : ranges) {
        if (!merged.empty() && (range.begin < merged.back().end ||
                (range.begin == merged.back().end && range.read_only == merged.back().read_only))) {
            merged.back().end = std::max(merged.back().end, range.end);
            merged.back().read_only = merged.back().read_only || range.read_only;
        } else {
            merged.push_back(range);
        }
    }
    ranges.clear();
    for (auto range : merged) {
        for (const auto & existing : owner.registered) {
            if (existing.end <= range.begin) {
                continue;
            }
            if (existing.begin >= range.end) {
                break;
            }
            if (existing.begin > range.begin) {
                ranges.push_back({range.begin, existing.begin, range.read_only});
            }
            range.begin = std::max(range.begin, existing.end);
        }
        if (range.begin < range.end) {
            ranges.push_back(range);
        }
    }
    for (const auto & range : ranges) {
        if (range.end - range.begin > SIZE_MAX - bytes) {
            return decline("registration_size_overflow");
        }
        bytes += range.end - range.begin;
    }
    if (bytes > owner.limit - owner.staging_reserved - owner.source_bytes) {
        return decline("source_budget_exhausted");
    }
    const bool read_only = std::any_of(ranges.begin(), ranges.end(), [](const moe_host_range & range) { return range.read_only; });
    cudaError_t error = cudaSuccess;
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    int device = 0;
    int supported = 0;
    if ((error = cudaGetDevice(&device)) != cudaSuccess ||
            (error = cudaDeviceGetAttribute(&supported, cudaDevAttrHostRegisterSupported, device)) != cudaSuccess || !supported) {
        return decline("host_registration_unsupported", error);
    }
    if (read_only) {
#if CUDART_VERSION >= 11010
        if ((error = cudaDeviceGetAttribute(&supported, cudaDevAttrHostRegisterReadOnlySupported, device)) != cudaSuccess || !supported) {
            return decline("read_only_registration_unsupported", error);
        }
#else
        return decline("read_only_registration_requires_cuda_11_1");
#endif
    }
#else
    if (read_only) {
        return decline("read_only_registration_unsupported");
    }
#endif
    owner.registered.reserve(owner.registered.size() + ranges.size());
    std::vector<void *> aliases(sources.size(), nullptr);
    size_t pinned = 0;
    const char * failure = nullptr;
    for (const auto & range : ranges) {
        unsigned int flags = cudaHostRegisterPortable | cudaHostRegisterMapped;
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11010
        if (range.read_only) {
            flags |= cudaHostRegisterReadOnly;
        }
#endif
        error = cudaHostRegister(reinterpret_cast<void *>(range.begin), range.end - range.begin, flags);
        if (error != cudaSuccess) {
            failure = "host_register_failed";
            break;
        }
        ++pinned;
    }
    for (size_t i = 0; failure == nullptr && i < sources.size(); ++i) {
        error = cudaHostGetDevicePointer(&aliases[i], const_cast<char *>(sources[i]->data), 0);
        if (error != cudaSuccess || aliases[i] == nullptr) {
            failure = "device_alias_unavailable";
        } else if (require_identity && aliases[i] != sources[i]->data) {
            failure = "auxiliary_alias_not_identity";
        }
    }
    if (failure != nullptr) {
        (void) cudaGetLastError();
        for (size_t i = 0; i < pinned; ++i) {
            CUDA_CHECK(cudaHostUnregister(reinterpret_cast<void *>(ranges[i].begin)));
        }
        return decline(failure, error);
    }
    owner.registered.insert(owner.registered.end(), ranges.begin(), ranges.end());
    std::sort(owner.registered.begin(), owner.registered.end(), [](const moe_host_range & a, const moe_host_range & b) { return a.begin < b.begin; });
    owner.source_bytes += bytes;
    for (size_t i = 0; i < sources.size(); ++i) {
        sources[i]->device_alias = static_cast<const char *>(aliases[i]);
    }
    return true;
}
