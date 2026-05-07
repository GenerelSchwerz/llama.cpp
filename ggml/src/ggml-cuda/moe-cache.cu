// MoE expert cache — week-1 implementation + buffer type registration.
// See ../../../../DESIGN.md (project root) for the full design.
//
// This file owns:
//   - struct ggml_cuda_moe_cache : the GPU slot pool + LRU bookkeeping
//   - init / free / acquire / slot_ptr / stats : the cache C API
//   - ggml_backend_cuda_moe_cached_buffer_type : a ggml buffer type that
//     marks expert tensors (`ffn_*_exps`) as "live in CPU pinned memory and
//     route GPU access through the cache". The implementation mirrors the
//     existing CUDA host buffer type (same pinned-memory allocator) but
//     uses a distinct name so the dispatch hook in ggml_cuda_mul_mat_id can
//     distinguish them and divert to the cached path.
//
// Design choices for v1 (kept deliberately simple):
//   * Single std::mutex around the cache metadata. Single-GPU only.
//   * Linear scan for LRU eviction. n_slots is typically 16-256, so the
//     O(n) scan per miss is in the hundreds of ns.
//   * acquire() hands back a slot_id; the caller is responsible for
//     synchronizing the compute stream against the copy stream before
//     touching the slab. The cache does not own the compute stream.
//   * The cache itself is created lazily on first cached-buffer access in
//     mul_mat_id_cached (forthcoming). The buffer type only flags tensors;
//     it does not allocate GPU slot pool memory.

#include "moe-cache.cuh"
#include "common.cuh"

#include "ggml-backend-impl.h"
#include "ggml-cuda.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

struct ggml_cuda_moe_cache {
    int      device;
    size_t   slot_size_bytes;
    int      n_slots;

    void *   slot_pool_d;            // device alloc, n_slots * slot_size_bytes

    // Per-slot state.
    std::vector<const void *> slot_to_host;  // [n_slots], nullptr if empty
    std::vector<uint64_t>     last_used;     // [n_slots]

    // host_ptr -> slot_id, O(1) lookup.
    std::unordered_map<const void *, int> host_to_slot;

    uint64_t access_counter;
    std::mutex mu;

    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> misses{0};
    std::atomic<uint64_t> evictions{0};
};

extern "C"
struct ggml_cuda_moe_cache * ggml_cuda_moe_cache_init(
    int    device,
    size_t slot_size_bytes,
    int    n_slots) {

    if (slot_size_bytes == 0 || n_slots <= 0) {
        return nullptr;
    }

    int prev_device = 0;
    cudaError_t err = cudaGetDevice(&prev_device);
    if (err != cudaSuccess) {
        fprintf(stderr, "moe-cache: cudaGetDevice failed: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        fprintf(stderr, "moe-cache: cudaSetDevice(%d) failed: %s\n", device, cudaGetErrorString(err));
        return nullptr;
    }

    auto * c = new ggml_cuda_moe_cache;
    c->device          = device;
    c->slot_size_bytes = slot_size_bytes;
    c->n_slots         = n_slots;
    c->slot_pool_d     = nullptr;
    c->access_counter  = 0;

    err = cudaMalloc(&c->slot_pool_d, (size_t)n_slots * slot_size_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "moe-cache: cudaMalloc(%zu bytes) failed: %s\n",
                (size_t)n_slots * slot_size_bytes, cudaGetErrorString(err));
        delete c;
        cudaSetDevice(prev_device);
        return nullptr;
    }

    c->slot_to_host.assign(n_slots, nullptr);
    c->last_used  .assign(n_slots, 0);
    c->host_to_slot.reserve(n_slots * 2);

    cudaSetDevice(prev_device);
    return c;
}

extern "C"
void ggml_cuda_moe_cache_free(struct ggml_cuda_moe_cache * cache) {
    if (!cache) return;

    int prev_device = 0;
    cudaGetDevice(&prev_device);
    cudaSetDevice(cache->device);

    if (cache->slot_pool_d) {
        cudaFree(cache->slot_pool_d);
    }
    cudaSetDevice(prev_device);

    delete cache;
}

extern "C"
int ggml_cuda_moe_cache_acquire(
    struct ggml_cuda_moe_cache * cache,
    const void * host_src,
    size_t       byte_count,
    cudaStream_t copy_stream) {

    if (!cache || host_src == nullptr || byte_count == 0) {
        return -1;
    }

    std::lock_guard<std::mutex> lk(cache->mu);

    if (byte_count > cache->slot_size_bytes) {
        // Caller forgot to grow first. Bail rather than clobber the next slot.
        return -1;
    }

    // Hit path: O(1) hash lookup.
    auto it = cache->host_to_slot.find(host_src);
    if (it != cache->host_to_slot.end()) {
        int slot = it->second;
        cache->last_used[slot] = ++cache->access_counter;
        cache->hits.fetch_add(1, std::memory_order_relaxed);
        return slot;
    }

    // Miss: pick the LRU slot. Empty slots (last_used==0) win automatically.
    int      lru_slot = 0;
    uint64_t lru_t    = std::numeric_limits<uint64_t>::max();
    for (int i = 0; i < cache->n_slots; ++i) {
        if (cache->last_used[i] < lru_t) {
            lru_t    = cache->last_used[i];
            lru_slot = i;
        }
    }

    const void * evicted = cache->slot_to_host[lru_slot];
    if (evicted != nullptr) {
        cache->host_to_slot.erase(evicted);
        cache->evictions.fetch_add(1, std::memory_order_relaxed);
    }

    cache->slot_to_host[lru_slot] = host_src;
    cache->host_to_slot[host_src] = lru_slot;
    cache->last_used[lru_slot]    = ++cache->access_counter;
    cache->misses.fetch_add(1, std::memory_order_relaxed);

    void * dst = (char *)cache->slot_pool_d + (size_t)lru_slot * cache->slot_size_bytes;
    cudaError_t err = cudaMemcpyAsync(
        dst, host_src, byte_count,
        cudaMemcpyHostToDevice, copy_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "moe-cache: cudaMemcpyAsync failed: %s\n",
                cudaGetErrorString(err));
        // Roll back so the next acquire retries cleanly.
        cache->slot_to_host[lru_slot] = nullptr;
        cache->host_to_slot.erase(host_src);
        return -1;
    }

    return lru_slot;
}

extern "C"
bool ggml_cuda_moe_cache_grow_pool(
    struct ggml_cuda_moe_cache * cache,
    size_t min_slot_size_bytes) {

    if (!cache || min_slot_size_bytes == 0) {
        return false;
    }

    std::lock_guard<std::mutex> lk(cache->mu);

    if (min_slot_size_bytes <= cache->slot_size_bytes) {
        return true; // already big enough
    }

    int prev_device = 0;
    cudaGetDevice(&prev_device);
    cudaSetDevice(cache->device);

    void * new_pool = nullptr;
    cudaError_t err = cudaMalloc(&new_pool, (size_t)cache->n_slots * min_slot_size_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "moe-cache: grow_pool cudaMalloc(%zu) failed: %s\n",
                (size_t)cache->n_slots * min_slot_size_bytes, cudaGetErrorString(err));
        cudaSetDevice(prev_device);
        return false;
    }

    if (cache->slot_pool_d) {
        cudaFree(cache->slot_pool_d);
    }
    cache->slot_pool_d     = new_pool;
    cache->slot_size_bytes = min_slot_size_bytes;

    // Existing cached state is invalidated by the realloc -- clear bookkeeping
    // so the next acquires re-populate slots in the new (larger) pool.
    std::fill(cache->slot_to_host.begin(), cache->slot_to_host.end(), nullptr);
    std::fill(cache->last_used.begin(),    cache->last_used.end(),    0ull);
    cache->host_to_slot.clear();
    cache->access_counter = 0;

    GGML_LOG_INFO("moe-cache: device %d  grew slot_size to %.2f MiB  pool=%.2f MiB\n",
                  cache->device,
                  min_slot_size_bytes / 1024.0 / 1024.0,
                  ((double)cache->n_slots * min_slot_size_bytes) / 1024.0 / 1024.0);

    cudaSetDevice(prev_device);
    return true;
}

extern "C"
size_t ggml_cuda_moe_cache_slot_size_bytes(const struct ggml_cuda_moe_cache * cache) {
    return cache ? cache->slot_size_bytes : 0;
}

extern "C"
int ggml_cuda_moe_cache_n_slots(const struct ggml_cuda_moe_cache * cache) {
    return cache ? cache->n_slots : 0;
}

extern "C"
void * ggml_cuda_moe_cache_slot_ptr(
    struct ggml_cuda_moe_cache * cache,
    int slot_id) {
    if (!cache || slot_id < 0 || slot_id >= cache->n_slots) {
        return nullptr;
    }
    return (char *)cache->slot_pool_d + (size_t)slot_id * cache->slot_size_bytes;
}

extern "C"
void ggml_cuda_moe_cache_stats(
    const struct ggml_cuda_moe_cache * cache,
    uint64_t * out_hits,
    uint64_t * out_misses,
    uint64_t * out_evictions) {
    if (!cache) {
        if (out_hits)      *out_hits      = 0;
        if (out_misses)    *out_misses    = 0;
        if (out_evictions) *out_evictions = 0;
        return;
    }
    if (out_hits)      *out_hits      = cache->hits.load(std::memory_order_relaxed);
    if (out_misses)    *out_misses    = cache->misses.load(std::memory_order_relaxed);
    if (out_evictions) *out_evictions = cache->evictions.load(std::memory_order_relaxed);
}

extern "C"
void ggml_cuda_moe_cache_reset_stats(struct ggml_cuda_moe_cache * cache) {
    if (!cache) return;
    cache->hits.store(0, std::memory_order_relaxed);
    cache->misses.store(0, std::memory_order_relaxed);
    cache->evictions.store(0, std::memory_order_relaxed);
}

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

static void ggml_backend_cuda_moe_cached_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    CUDA_CHECK(cudaFreeHost(buffer->context));
}

static void * ggml_cuda_moe_cached_pinned_malloc(size_t size) {
    if (getenv("GGML_CUDA_NO_PINNED") != nullptr) {
        return nullptr;
    }
    void * ptr = nullptr;
    cudaError_t err = cudaMallocHost((void **) &ptr, size);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        GGML_LOG_DEBUG("%s: failed to allocate %.2f MiB of pinned memory: %s\n",
                       __func__, size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return nullptr;
    }
    return ptr;
}

static ggml_backend_buffer_t ggml_backend_cuda_moe_cached_buffer_type_alloc_buffer(
        ggml_backend_buffer_type_t buft, size_t size) {

    void * ptr = ggml_cuda_moe_cached_pinned_malloc(size);

    if (ptr == nullptr) {
        // Pinned alloc failed -- fall back to a regular CPU buffer. This costs
        // PCIe bandwidth on cache miss but keeps the model loadable.
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft             = buft;
    buffer->iface.free_buffer = ggml_backend_cuda_moe_cached_buffer_free_buffer;
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

// ---------------------------------------------------------------------------
// Per-device cache singleton
//
// The dispatch hook calls get_or_create on first cached op; subsequent ops
// reuse the same cache. Slot size is set on first creation. If a later op
// needs a different slot size, the caller is expected to detect the mismatch
// via ggml_cuda_moe_cache_slot_size_bytes() and fall back to per-op staging.
// ---------------------------------------------------------------------------

// Single-cache-per-device registry. The cache's slot_size grows on demand
// (via grow_pool) when an op presents an expert larger than the current slot.
namespace {

struct moe_cache_registry {
    std::mutex mu;
    std::map<int, ggml_cuda_moe_cache *> by_device;
};

static moe_cache_registry & get_registry() {
    // Function-local static, constructed on first use, destroyed at process exit.
    static moe_cache_registry inst;
    return inst;
}

} // namespace

extern "C"
struct ggml_cuda_moe_cache * ggml_cuda_moe_cache_get_or_create(
    int    device,
    size_t slot_size_bytes,
    int    n_slots) {

    auto & reg = get_registry();
    std::lock_guard<std::mutex> lk(reg.mu);

    auto it = reg.by_device.find(device);
    if (it != reg.by_device.end()) {
        return it->second;
    }

    ggml_cuda_moe_cache * c = ggml_cuda_moe_cache_init(device, slot_size_bytes, n_slots);
    if (!c) {
        return nullptr;
    }

    reg.by_device.emplace(device, c);
    // Note: we don't log here -- the eager preallocate path
    // (ggml_backend_cuda_moe_preallocate_pool) emits an "load_tensors:"-style
    // line with the same info, and we don't want to duplicate.
    return c;
}

extern "C"
void ggml_cuda_moe_cache_free_all(void) {
    auto & reg = get_registry();
    std::lock_guard<std::mutex> lk(reg.mu);
    for (auto & kv : reg.by_device) {
        ggml_cuda_moe_cache_free(kv.second);
    }
    reg.by_device.clear();
}

// ---------------------------------------------------------------------------
// User-facing slot count configuration (settable from llama_model_load)
// ---------------------------------------------------------------------------

static std::atomic<int> g_moe_cache_slots{0};

extern "C"
void ggml_backend_cuda_moe_set_cache_slots(int n_slots) {
    if (n_slots < 0) n_slots = 0;
    g_moe_cache_slots.store(n_slots, std::memory_order_relaxed);
}

extern "C"
int ggml_backend_cuda_moe_get_cache_slots(void) {
    return g_moe_cache_slots.load(std::memory_order_relaxed);
}

// Max observed expert byte stride. Updated by the model loader; read by the
// dispatch hook when sizing the cache for the first time.
static std::atomic<size_t> g_max_expert_size{0};

extern "C"
void ggml_backend_cuda_moe_observe_expert_size(size_t per_expert_bytes) {
    if (per_expert_bytes == 0) return;
    size_t cur = g_max_expert_size.load(std::memory_order_relaxed);
    while (per_expert_bytes > cur &&
           !g_max_expert_size.compare_exchange_weak(cur, per_expert_bytes,
                                                    std::memory_order_relaxed)) {
        // retry on contention
    }
}

extern "C"
size_t ggml_backend_cuda_moe_get_max_expert_size(void) {
    return g_max_expert_size.load(std::memory_order_relaxed);
}

extern "C"
void ggml_backend_cuda_moe_reset_expert_size_observation(void) {
    g_max_expert_size.store(0, std::memory_order_relaxed);
}

extern "C"
void ggml_backend_cuda_moe_preallocate_pool(int device) {
    const int    n_slots   = ggml_backend_cuda_moe_get_cache_slots();
    const size_t slot_size = ggml_backend_cuda_moe_get_max_expert_size();
    if (n_slots <= 0 || slot_size == 0) {
        return; // feature off or no expert tensors observed
    }
    ggml_cuda_moe_cache * c = ggml_cuda_moe_cache_get_or_create(device, slot_size, n_slots);
    if (!c) {
        return; // alloc failed; init already logged the cause
    }
    // Match the "load_tensors:" prefix used elsewhere so the line groups
    // visually with the other model buffer allocations.
    GGML_LOG_INFO("load_tensors: CUDA_MoE_Cache_Pool model buffer size = %8.2f MiB\n",
                  ((double)n_slots * slot_size) / 1024.0 / 1024.0);
}

extern "C"
void ggml_backend_cuda_moe_log_and_reset_stats(void) {
    auto & reg = get_registry();
    std::lock_guard<std::mutex> lk(reg.mu);
    for (auto & kv : reg.by_device) {
        uint64_t hits = 0, misses = 0, evictions = 0;
        ggml_cuda_moe_cache_stats(kv.second, &hits, &misses, &evictions);
        const uint64_t total = hits + misses;
        const double rate = total > 0 ? 100.0 * (double)hits / (double)total : 0.0;
        const size_t slot_size = ggml_cuda_moe_cache_slot_size_bytes(kv.second);
        GGML_LOG_INFO("moe-cache: device %d  slot_size=%.2f MiB  hits=%llu  misses=%llu  evictions=%llu  hit-rate=%.2f%%\n",
                      kv.first,
                      slot_size / 1024.0 / 1024.0,
                      (unsigned long long)hits,
                      (unsigned long long)misses,
                      (unsigned long long)evictions,
                      rate);
        ggml_cuda_moe_cache_reset_stats(kv.second);
    }
}
