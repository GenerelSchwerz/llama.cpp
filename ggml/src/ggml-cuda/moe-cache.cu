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

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <mutex>
#include <unordered_map>
#include <vector>

struct ggml_cuda_moe_cache {
    int      device;
    size_t   slot_size_bytes;
    int      n_slots;

    void *   slot_pool_d;            // device alloc, n_slots * slot_size_bytes

    // Dedicated copy stream so H2D acquires can pipeline with the compute
    // stream's kernel work. The dispatch hook records `copy_done` after all
    // acquires for an op finish, then makes the compute stream wait on it.
    cudaStream_t copy_stream;

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
    c->copy_stream     = nullptr;
    c->access_counter  = 0;

    err = cudaMalloc(&c->slot_pool_d, (size_t)n_slots * slot_size_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "moe-cache: cudaMalloc(%zu bytes) failed: %s\n",
                (size_t)n_slots * slot_size_bytes, cudaGetErrorString(err));
        delete c;
        cudaSetDevice(prev_device);
        return nullptr;
    }

    err = cudaStreamCreateWithFlags(&c->copy_stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        fprintf(stderr, "moe-cache: cudaStreamCreate failed: %s\n", cudaGetErrorString(err));
        cudaFree(c->slot_pool_d);
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

    if (cache->copy_stream) {
        cudaStreamDestroy(cache->copy_stream);
    }
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
cudaStream_t ggml_cuda_moe_cache_copy_stream(const struct ggml_cuda_moe_cache * cache) {
    return cache ? cache->copy_stream : nullptr;
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

extern "C"
ggml_backend_buffer_t ggml_backend_cuda_moe_cached_buffer_from_host_ptr(void * ptr, size_t size) {
    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    if (buffer == nullptr) {
        return nullptr;
    }

    buffer->buft = ggml_backend_cuda_moe_cached_buffer_type();
    return buffer;
}

// ---------------------------------------------------------------------------
// Per-device cache singleton
//
// The dispatch hook calls get_or_create on first cached op; subsequent ops
// reuse the same cache. Slot size is set on first creation. If a later op
// needs a different slot size, the caller is expected to detect the mismatch
// via ggml_cuda_moe_cache_slot_size_bytes() and fall back to per-op staging.
// ---------------------------------------------------------------------------

// Per-tensor cache registry: one cache per (device, tensor_name).
// "Tensor" here means a model expert weight tensor like blk.0.ffn_up_exps.weight.
// Each name is unique and identifies a (layer, matrix_kind) pair. The N slots
// in that tensor's cache are dedicated to that tensor's experts only -- no
// cross-layer competition.
//
// We key by name (not data ptr) because t_meta->data isn't reliably set at
// the point the model loader observes tensors (it's pre-allocation in the
// override-matching loop). Names are set with metadata and stay stable.
//
// User-facing N is slots-per-tensor, not slots-total. For Qwen3.6 35B-A3B
// with 40 layers × 3 matrices = 120 expert tensors and N=32:
//   total memory = 120 × 32 × ~0.82 MiB ≈ 3.1 GiB per device
//
// Slot size per cache equals that tensor's expert_stride, so slots are tightly
// packed; the synthetic src0 stays contiguous and kernels run unchanged.
namespace {

struct moe_cache_key {
    int         device;
    std::string tensor_name;
    bool operator<(const moe_cache_key & o) const {
        if (device != o.device) return device < o.device;
        return tensor_name < o.tensor_name;
    }
};

struct moe_cache_registry {
    std::mutex mu;
    std::map<moe_cache_key, ggml_cuda_moe_cache *> by_key;
};

static moe_cache_registry & get_registry() {
    static moe_cache_registry inst;
    return inst;
}

} // namespace

extern "C"
struct ggml_cuda_moe_cache * ggml_cuda_moe_cache_get_or_create_for_tensor(
    int          device,
    const void * tensor_data,
    size_t       slot_size_bytes,
    int          n_slots,
    const char * tensor_name_for_log) {
    GGML_UNUSED(tensor_data);

    if (tensor_name_for_log == nullptr || tensor_name_for_log[0] == '\0') {
        return nullptr;
    }

    auto & reg = get_registry();
    std::lock_guard<std::mutex> lk(reg.mu);

    moe_cache_key k{device, std::string(tensor_name_for_log)};
    auto it = reg.by_key.find(k);
    if (it != reg.by_key.end()) {
        return it->second;
    }

    ggml_cuda_moe_cache * c = ggml_cuda_moe_cache_init(device, slot_size_bytes, n_slots);
    if (!c) {
        return nullptr;
    }

    reg.by_key.emplace(k, c);
    GGML_LOG_INFO("load_tensors: CUDA_MoE_Cache_Pool[%-32s] = %7.2f MiB  (%d slots × %.2f MiB)\n",
                  tensor_name_for_log,
                  ((double)n_slots * slot_size_bytes) / 1024.0 / 1024.0,
                  n_slots,
                  slot_size_bytes / 1024.0 / 1024.0);
    return c;
}

// Backward-compat wrapper: keys solely by slot_size. Kept so existing callers
// don't break, but prefer the per-tensor variant.
extern "C"
struct ggml_cuda_moe_cache * ggml_cuda_moe_cache_get_or_create(
    int    device,
    size_t slot_size_bytes,
    int    n_slots) {
    GGML_UNUSED(slot_size_bytes);
    GGML_UNUSED(n_slots);
    GGML_UNUSED(device);
    return nullptr; // dispatch hook now uses the per-tensor variant
}

extern "C"
void ggml_cuda_moe_cache_free_all(void) {
    auto & reg = get_registry();
    std::lock_guard<std::mutex> lk(reg.mu);
    for (auto & kv : reg.by_key) {
        ggml_cuda_moe_cache_free(kv.second);
    }
    reg.by_key.clear();
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

// Per-tensor observation list: (tensor_data, name, expert_stride) populated
// by the model loader, drained by preallocate_pools to create one cache per
// observed tensor. Reset between models.
namespace {
struct observed_tensor {
    const void * tensor_data;
    std::string  tensor_name;
    size_t       per_expert_bytes;
    int64_t      n_experts;
};
struct observation_state {
    std::mutex mu;
    std::vector<observed_tensor> tensors;
};
static observation_state & get_observation_state() {
    static observation_state inst;
    return inst;
}
} // namespace

extern "C"
void ggml_backend_cuda_moe_observe_expert_tensor(
    const void * tensor_data,
    const char * tensor_name,
    size_t       per_expert_bytes,
    int64_t      n_experts) {
    if (tensor_data == nullptr || per_expert_bytes == 0) return;
    auto & st = get_observation_state();
    std::lock_guard<std::mutex> lk(st.mu);
    st.tensors.push_back({tensor_data,
                          tensor_name ? std::string(tensor_name) : std::string(),
                          per_expert_bytes,
                          n_experts});
}

extern "C"
void ggml_backend_cuda_moe_reset_expert_size_observation(void) {
    auto & st = get_observation_state();
    std::lock_guard<std::mutex> lk(st.mu);
    st.tensors.clear();
}

extern "C"
void ggml_backend_cuda_moe_preallocate_pools(int device) {
    const int n_slots = ggml_backend_cuda_moe_get_cache_slots();
    if (n_slots <= 0) return;
    auto & st = get_observation_state();
    std::lock_guard<std::mutex> lk(st.mu);
    for (const auto & t : st.tensors) {
        ggml_cuda_moe_cache_get_or_create_for_tensor(
            device, t.tensor_data, t.per_expert_bytes, n_slots,
            t.tensor_name.empty() ? "?" : t.tensor_name.c_str());
    }
}

extern "C"
void ggml_backend_cuda_moe_prefill_pools(int device) {
    const int n_slots = ggml_backend_cuda_moe_get_cache_slots();
    if (n_slots <= 0) return;

    auto & st = get_observation_state();
    std::lock_guard<std::mutex> lk(st.mu);
    for (const auto & t : st.tensors) {
        ggml_cuda_moe_cache * cache = ggml_cuda_moe_cache_get_or_create_for_tensor(
            device, t.tensor_data, t.per_expert_bytes, n_slots,
            t.tensor_name.empty() ? "?" : t.tensor_name.c_str());
        if (!cache) {
            continue;
        }

        cudaStream_t copy_stream = ggml_cuda_moe_cache_copy_stream(cache);
        const char * src_base = (const char *) t.tensor_data;
        const int64_t n_prefill = std::min<int64_t>(n_slots, t.n_experts);
        for (int64_t eid = 0; eid < n_prefill; ++eid) {
            const void * host_ptr = src_base + (size_t) eid * t.per_expert_bytes;
            (void) ggml_cuda_moe_cache_acquire(cache, host_ptr, t.per_expert_bytes, copy_stream);
        }

        CUDA_CHECK(cudaStreamSynchronize(copy_stream));
        ggml_cuda_moe_cache_reset_stats(cache);
    }
}

extern "C"
void ggml_backend_cuda_moe_prefetch_experts(
    int             device,
    const char *    tensor_name,
    const int32_t * eids,
    int             n_eids) {
    if (!tensor_name || !eids || n_eids <= 0) return;

    // 1. Look up the observed tensor (data ptr + stride) by name.
    const observed_tensor * found = nullptr;
    {
        auto & st = get_observation_state();
        std::lock_guard<std::mutex> lk(st.mu);
        for (const auto & t : st.tensors) {
            if (t.tensor_name == tensor_name) {
                found = &t;
                // Note: holding the reference past the lock is okay because
                // the vector is only ever appended to during model load,
                // never reallocated during inference. Still, copy locally
                // before releasing.
                break;
            }
        }
        if (!found) return;
    }

    // Re-lookup outside the observation lock since we copied what we need.
    const void * tensor_data    = found->tensor_data;
    const size_t expert_stride  = found->per_expert_bytes;

    // 2. Find the cache for this tensor.
    auto & reg = get_registry();
    ggml_cuda_moe_cache * cache = nullptr;
    {
        std::lock_guard<std::mutex> lk(reg.mu);
        moe_cache_key k{device, std::string(tensor_name)};
        auto it = reg.by_key.find(k);
        if (it == reg.by_key.end()) return; // cache not created yet
        cache = it->second;
    }

    if (!cache) return;
    cudaStream_t copy_stream = cache->copy_stream;
    const char * src_base = (const char *)tensor_data;

    // 3. Issue acquires for each expert id. Failures are silent -- this is
    //    speculative; the real op will re-acquire if needed.
    for (int i = 0; i < n_eids; ++i) {
        int32_t eid = eids[i];
        if (eid < 0) continue;
        const void * host_ptr = src_base + (size_t)eid * expert_stride;
        (void)ggml_cuda_moe_cache_acquire(cache, host_ptr, expert_stride, copy_stream);
    }
}

// Deprecated singular-pool entry point; superseded by preallocate_pools (plural).
extern "C"
void ggml_backend_cuda_moe_preallocate_pool(int device) {
    ggml_backend_cuda_moe_preallocate_pools(device);
}

extern "C"
void ggml_backend_cuda_moe_log_and_reset_stats(void) {
    auto & reg = get_registry();
    std::lock_guard<std::mutex> lk(reg.mu);

    // Aggregate across all per-tensor caches into one line per request.
    // 120+ per-tensor lines would drown the timing output.
    uint64_t total_hits = 0, total_misses = 0, total_evictions = 0;
    size_t   n_caches = reg.by_key.size();
    for (auto & kv : reg.by_key) {
        uint64_t h = 0, m = 0, e = 0;
        ggml_cuda_moe_cache_stats(kv.second, &h, &m, &e);
        total_hits      += h;
        total_misses    += m;
        total_evictions += e;
        ggml_cuda_moe_cache_reset_stats(kv.second);
    }
    const uint64_t total = total_hits + total_misses;
    const double rate = total > 0 ? 100.0 * (double)total_hits / (double)total : 0.0;
    GGML_LOG_INFO("moe-cache: %zu caches  hits=%llu  misses=%llu  evictions=%llu  hit-rate=%.2f%%\n",
                  n_caches,
                  (unsigned long long)total_hits,
                  (unsigned long long)total_misses,
                  (unsigned long long)total_evictions,
                  rate);
}
