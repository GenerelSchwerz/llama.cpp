// MoE expert cache - week-1 implementation + buffer type registration.
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
#include "ggml.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <new>
#include <sstream>
#include <string>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {

static constexpr uint64_t MOE_CACHE_MM_SAMPLE_RATE = 64;
static std::atomic<bool> g_moe_cache_mm_debug{false};
static std::atomic<size_t> g_moe_cache_l2_pinned_size{0};

static bool moe_cache_mm_debug_enabled() {
    return g_moe_cache_mm_debug.load(std::memory_order_relaxed);
}

static bool moe_cache_mm_verbose_enabled() {
    return moe_cache_mm_debug_enabled();
}

static int moe_cache_phase_index(bool is_decode) {
    return is_decode ? 1 : 0;
}

static std::atomic<uint64_t> g_moe_cache_mm_miss_counter{0};

struct moe_cache_op_phase_stats {
    std::atomic<uint64_t> ops{0};
    std::atomic<uint64_t> staged_ops{0};
    std::atomic<uint64_t> overflow_ops{0};
    std::atomic<uint64_t> unique_experts{0};
    std::atomic<uint64_t> unique_experts_max{0};
    std::atomic<uint64_t> ids_bytes{0};
    std::atomic<uint64_t> ids_d2h_time_us{0};
    std::atomic<uint64_t> ids_d2h_sync_count{0};
    std::atomic<uint64_t> ids_cache_hits{0};
    std::atomic<uint64_t> acquire_time_us{0};
    std::atomic<uint64_t> remap_time_us{0};
    std::atomic<uint64_t> copy_wait_event_count{0};
    std::atomic<uint64_t> copy_wait_event_time_us{0};
    std::atomic<uint64_t> total_time_us{0};
};

static moe_cache_op_phase_stats g_moe_cache_op_stats[2];

static void moe_cache_atomic_max(std::atomic<uint64_t> & dst, uint64_t value) {
    uint64_t cur = dst.load(std::memory_order_relaxed);
    while (cur < value && !dst.compare_exchange_weak(cur, value, std::memory_order_relaxed)) {
    }
}

struct moe_cache_mm_stats {
    uint64_t h2d_copy_count;
    uint64_t h2d_copy_bytes;
    uint64_t h2d_enqueue_time_us;
    uint64_t sampled_mincore_checks;
    uint64_t sampled_pages_total;
    uint64_t sampled_pages_resident;
    uint64_t sampled_nonresident_expert_count;
    uint64_t mincore_failures;
};

static moe_cache_mm_stats moe_cache_mm_stats_zero() {
    return {};
}

#ifdef __linux__
static size_t round_up_size(size_t value, size_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

static void moe_cache_mm_sample_mincore(
    const void * host_src,
    size_t       byte_count,
    std::atomic<uint64_t> & sampled_mincore_checks,
    std::atomic<uint64_t> & sampled_pages_total,
    std::atomic<uint64_t> & sampled_pages_resident,
    std::atomic<uint64_t> & sampled_nonresident_expert_count,
    std::atomic<uint64_t> & mincore_failures) {

    const long page_size_long = sysconf(_SC_PAGESIZE);
    if (page_size_long <= 0) {
        mincore_failures.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    const size_t page_size = (size_t) page_size_long;
    const uintptr_t src = (uintptr_t) host_src;
    const uintptr_t start_addr = src & ~(uintptr_t)(page_size - 1);
    const size_t offset = (size_t)(src - start_addr);

    if (byte_count > std::numeric_limits<size_t>::max() - offset) {
        mincore_failures.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    const size_t len = round_up_size(offset + byte_count, page_size);
    const size_t vec_len = len / page_size;
    if (vec_len == 0) {
        return;
    }

    std::vector<unsigned char> vec(vec_len);
    sampled_mincore_checks.fetch_add(1, std::memory_order_relaxed);
    sampled_pages_total.fetch_add(vec_len, std::memory_order_relaxed);

    if (mincore((void *) start_addr, len, vec.data()) != 0) {
        mincore_failures.fetch_add(1, std::memory_order_relaxed);
        if (moe_cache_mm_verbose_enabled()) {
            GGML_LOG("moe-cache-mm: mincore failed ptr=%p bytes=%zu errno=%d\n",
                     host_src, byte_count, errno);
        }
        return;
    }

    uint64_t resident = 0;
    for (unsigned char v : vec) {
        resident += (v & 1) ? 1 : 0;
    }

    sampled_pages_resident.fetch_add(resident, std::memory_order_relaxed);
    if (resident < vec_len) {
        sampled_nonresident_expert_count.fetch_add(1, std::memory_order_relaxed);
        if (moe_cache_mm_verbose_enabled()) {
            GGML_LOG("moe-cache-mm: nonresident expert ptr=%p bytes=%zu resident=%llu/%zu\n",
                     host_src,
                     byte_count,
                     (unsigned long long) resident,
                     vec_len);
        }
    }
}
#else
static void moe_cache_mm_sample_mincore(
    const void *,
    size_t,
    std::atomic<uint64_t> &,
    std::atomic<uint64_t> &,
    std::atomic<uint64_t> &,
    std::atomic<uint64_t> &,
    std::atomic<uint64_t> &) {
}
#endif

struct moe_cache_proc_snapshot {
    uint64_t proc_minflt;
    uint64_t proc_majflt;
    uint64_t vm_pswpin;
    uint64_t vm_pswpout;
    uint64_t vm_pgfault;
    uint64_t vm_pgmajfault;
    uint64_t vm_pgpgin;
    uint64_t vm_pgpgout;
    uint64_t vm_workingset_refault_file;
    uint64_t vm_workingset_refault_anon;
};

static moe_cache_proc_snapshot moe_cache_proc_snapshot_zero() {
    return {};
}

static moe_cache_proc_snapshot moe_cache_proc_delta(const moe_cache_proc_snapshot & newer, const moe_cache_proc_snapshot & older) {
    moe_cache_proc_snapshot d = {};
#define MOE_CACHE_DELTA_FIELD(field) d.field = newer.field >= older.field ? newer.field - older.field : 0
    MOE_CACHE_DELTA_FIELD(proc_minflt);
    MOE_CACHE_DELTA_FIELD(proc_majflt);
    MOE_CACHE_DELTA_FIELD(vm_pswpin);
    MOE_CACHE_DELTA_FIELD(vm_pswpout);
    MOE_CACHE_DELTA_FIELD(vm_pgfault);
    MOE_CACHE_DELTA_FIELD(vm_pgmajfault);
    MOE_CACHE_DELTA_FIELD(vm_pgpgin);
    MOE_CACHE_DELTA_FIELD(vm_pgpgout);
    MOE_CACHE_DELTA_FIELD(vm_workingset_refault_file);
    MOE_CACHE_DELTA_FIELD(vm_workingset_refault_anon);
#undef MOE_CACHE_DELTA_FIELD
    return d;
}

#ifdef __linux__
static bool moe_cache_read_proc_self_stat(uint64_t & minflt, uint64_t & majflt) {
    std::ifstream f("/proc/self/stat");
    std::string line;
    if (!f || !std::getline(f, line)) {
        return false;
    }

    const size_t rparen = line.rfind(')');
    if (rparen == std::string::npos || rparen + 2 >= line.size()) {
        return false;
    }

    std::istringstream fields(line.substr(rparen + 2));
    std::string field;
    std::vector<std::string> values;
    while (fields >> field) {
        values.push_back(field);
    }

    if (values.size() <= 9) {
        return false;
    }

    minflt = strtoull(values[7].c_str(), nullptr, 10);
    majflt = strtoull(values[9].c_str(), nullptr, 10);
    return true;
}

static moe_cache_proc_snapshot moe_cache_read_proc_snapshot() {
    moe_cache_proc_snapshot s = {};
    (void) moe_cache_read_proc_self_stat(s.proc_minflt, s.proc_majflt);

    std::ifstream f("/proc/vmstat");
    if (!f) {
        return s;
    }

    std::string key;
    uint64_t value = 0;
    while (f >> key >> value) {
        if (key == "pswpin") {
            s.vm_pswpin = value;
        } else if (key == "pswpout") {
            s.vm_pswpout = value;
        } else if (key == "pgfault") {
            s.vm_pgfault = value;
        } else if (key == "pgmajfault") {
            s.vm_pgmajfault = value;
        } else if (key == "pgpgin") {
            s.vm_pgpgin = value;
        } else if (key == "pgpgout") {
            s.vm_pgpgout = value;
        } else if (key == "workingset_refault_file") {
            s.vm_workingset_refault_file = value;
        } else if (key == "workingset_refault_anon") {
            s.vm_workingset_refault_anon = value;
        }
    }

    return s;
}
#else
static moe_cache_proc_snapshot moe_cache_read_proc_snapshot() {
    return {};
}
#endif

static moe_cache_proc_snapshot moe_cache_get_proc_delta() {
    static std::mutex mu;
    static bool have_previous = false;
    static moe_cache_proc_snapshot previous = {};

    std::lock_guard<std::mutex> lk(mu);
    const moe_cache_proc_snapshot current = moe_cache_read_proc_snapshot();
    if (!have_previous) {
        previous = current;
        have_previous = true;
        return moe_cache_proc_snapshot_zero();
    }

    const moe_cache_proc_snapshot delta = moe_cache_proc_delta(current, previous);
    previous = current;
    return delta;
}

struct moe_cache_mmap_range {
    const char * begin;
    const char * end;
};

struct moe_cache_mmap_registry {
    std::mutex mu;
    std::vector<moe_cache_mmap_range> ranges;
};

static moe_cache_mmap_registry & get_mmap_registry() {
    static moe_cache_mmap_registry inst;
    return inst;
}

static void moe_cache_register_mmap_range(void * ptr, size_t size) {
    if (ptr == nullptr || size == 0) {
        return;
    }

    auto & reg = get_mmap_registry();
    std::lock_guard<std::mutex> lk(reg.mu);
    const char * begin = (const char *) ptr;
    reg.ranges.push_back({begin, begin + size});
}

static void moe_cache_unregister_mmap_range(void * ptr, size_t size) {
    if (ptr == nullptr || size == 0) {
        return;
    }

    auto & reg = get_mmap_registry();
    std::lock_guard<std::mutex> lk(reg.mu);
    const char * begin = (const char *) ptr;
    const char * end   = begin + size;
    reg.ranges.erase(
        std::remove_if(reg.ranges.begin(), reg.ranges.end(),
            [&](const moe_cache_mmap_range & r) {
                return r.begin == begin && r.end == end;
            }),
        reg.ranges.end());
}

static bool moe_cache_is_mmap_range(const void * ptr, size_t size) {
    if (ptr == nullptr || size == 0) {
        return false;
    }

    auto & reg = get_mmap_registry();
    std::lock_guard<std::mutex> lk(reg.mu);
    const char * begin = (const char *) ptr;
    const char * end   = begin + size;
    for (const moe_cache_mmap_range & r : reg.ranges) {
        if (begin >= r.begin && end <= r.end) {
            return true;
        }
    }
    return false;
}

struct moe_cache_l2 {
    size_t slot_size_bytes = 0;
    int    n_slots = 0;

    void * slot_pool_h = nullptr;
    std::vector<const void *> slot_to_host;
    std::vector<uint64_t> last_used;
    std::vector<char> is_protected;
    std::unordered_map<const void *, int> host_to_slot;
    uint64_t access_counter = 0;

    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> misses{0};
    std::atomic<uint64_t> fills{0};
    std::atomic<uint64_t> evictions{0};
    std::atomic<uint64_t> fill_bytes{0};
    std::atomic<uint64_t> fill_time_us{0};

    std::atomic<uint64_t> phase_hits[2];
    std::atomic<uint64_t> phase_misses[2];
    std::atomic<uint64_t> phase_fills[2];
    std::atomic<uint64_t> phase_evictions[2];
    std::atomic<uint64_t> phase_fill_bytes[2];
    std::atomic<uint64_t> phase_fill_time_us[2];
};

static bool moe_cache_l2_init(moe_cache_l2 & l2, size_t slot_size_bytes, int n_slots) {
    if (slot_size_bytes == 0 || n_slots <= 0) {
        return false;
    }
    if (getenv("GGML_CUDA_NO_PINNED") != nullptr) {
        return false;
    }

    void * ptr = nullptr;
    cudaError_t err = cudaMallocHost(&ptr, (size_t) n_slots * slot_size_bytes);
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        GGML_LOG_DEBUG("moe-cache-l2: failed to allocate %.2f MiB of pinned memory: %s\n",
                       ((double) n_slots * slot_size_bytes) / 1024.0 / 1024.0,
                       cudaGetErrorString(err));
        return false;
    }

    l2.slot_size_bytes = slot_size_bytes;
    l2.n_slots = n_slots;
    l2.slot_pool_h = ptr;
    l2.slot_to_host.assign(n_slots, nullptr);
    l2.last_used.assign(n_slots, 0);
    l2.is_protected.assign(n_slots, 0);
    l2.host_to_slot.reserve(n_slots * 2);
    for (int phase = 0; phase < 2; ++phase) {
        l2.phase_hits[phase].store(0, std::memory_order_relaxed);
        l2.phase_misses[phase].store(0, std::memory_order_relaxed);
        l2.phase_fills[phase].store(0, std::memory_order_relaxed);
        l2.phase_evictions[phase].store(0, std::memory_order_relaxed);
        l2.phase_fill_bytes[phase].store(0, std::memory_order_relaxed);
        l2.phase_fill_time_us[phase].store(0, std::memory_order_relaxed);
    }
    return true;
}

static void moe_cache_l2_free(moe_cache_l2 & l2) {
    if (l2.slot_pool_h) {
        CUDA_CHECK(cudaFreeHost(l2.slot_pool_h));
    }
    l2.slot_size_bytes = 0;
    l2.n_slots = 0;
    l2.slot_pool_h = nullptr;
    l2.slot_to_host.clear();
    l2.last_used.clear();
    l2.is_protected.clear();
    l2.host_to_slot.clear();
    l2.access_counter = 0;
    l2.hits.store(0, std::memory_order_relaxed);
    l2.misses.store(0, std::memory_order_relaxed);
    l2.fills.store(0, std::memory_order_relaxed);
    l2.evictions.store(0, std::memory_order_relaxed);
    l2.fill_bytes.store(0, std::memory_order_relaxed);
    l2.fill_time_us.store(0, std::memory_order_relaxed);
    for (int phase = 0; phase < 2; ++phase) {
        l2.phase_hits[phase].store(0, std::memory_order_relaxed);
        l2.phase_misses[phase].store(0, std::memory_order_relaxed);
        l2.phase_fills[phase].store(0, std::memory_order_relaxed);
        l2.phase_evictions[phase].store(0, std::memory_order_relaxed);
        l2.phase_fill_bytes[phase].store(0, std::memory_order_relaxed);
        l2.phase_fill_time_us[phase].store(0, std::memory_order_relaxed);
    }
}

static int moe_cache_l2_select_slot(moe_cache_l2 & l2) {
    for (int i = 0; i < l2.n_slots; ++i) {
        if (l2.slot_to_host[i] == nullptr) {
            return i;
        }
    }

    int slot = -1;
    uint64_t lru_t = std::numeric_limits<uint64_t>::max();
    for (int i = 0; i < l2.n_slots; ++i) {
        if (!l2.is_protected[i] && l2.last_used[i] < lru_t) {
            lru_t = l2.last_used[i];
            slot = i;
        }
    }
    if (slot >= 0) {
        return slot;
    }

    lru_t = std::numeric_limits<uint64_t>::max();
    for (int i = 0; i < l2.n_slots; ++i) {
        if (l2.last_used[i] < lru_t) {
            lru_t = l2.last_used[i];
            slot = i;
        }
    }
    return slot;
}

static const void * moe_cache_l2_acquire(
        moe_cache_l2 & l2,
        const void * host_src,
        size_t byte_count,
        bool is_decode,
        cudaStream_t copy_stream) {
    const int phase = moe_cache_phase_index(is_decode);
    auto it = l2.host_to_slot.find(host_src);
    if (it != l2.host_to_slot.end()) {
        const int slot = it->second;
        l2.is_protected[slot] = 1;
        l2.last_used[slot] = ++l2.access_counter;
        l2.hits.fetch_add(1, std::memory_order_relaxed);
        l2.phase_hits[phase].fetch_add(1, std::memory_order_relaxed);
        return (const char *) l2.slot_pool_h + (size_t) slot * l2.slot_size_bytes;
    }

    l2.misses.fetch_add(1, std::memory_order_relaxed);
    l2.phase_misses[phase].fetch_add(1, std::memory_order_relaxed);
    const int slot = moe_cache_l2_select_slot(l2);
    if (slot < 0) {
        return nullptr;
    }

    const void * evicted = l2.slot_to_host[slot];
    if (evicted != nullptr) {
        cudaError_t err = cudaStreamSynchronize(copy_stream);
        if (err != cudaSuccess) {
            GGML_LOG_ERROR("moe-cache-l2: cudaStreamSynchronize failed: %s\n", cudaGetErrorString(err));
            return nullptr;
        }
        l2.host_to_slot.erase(evicted);
        l2.evictions.fetch_add(1, std::memory_order_relaxed);
        l2.phase_evictions[phase].fetch_add(1, std::memory_order_relaxed);
    }

    void * dst = (char *) l2.slot_pool_h + (size_t) slot * l2.slot_size_bytes;
    const int64_t start_us = ggml_time_us();
    memcpy(dst, host_src, byte_count);
    const uint64_t fill_time_us = (uint64_t) (ggml_time_us() - start_us);
    l2.fill_time_us.fetch_add(fill_time_us, std::memory_order_relaxed);
    l2.fill_bytes.fetch_add(byte_count, std::memory_order_relaxed);
    l2.fills.fetch_add(1, std::memory_order_relaxed);
    l2.phase_fill_time_us[phase].fetch_add(fill_time_us, std::memory_order_relaxed);
    l2.phase_fill_bytes[phase].fetch_add(byte_count, std::memory_order_relaxed);
    l2.phase_fills[phase].fetch_add(1, std::memory_order_relaxed);

    l2.slot_to_host[slot] = host_src;
    l2.host_to_slot[host_src] = slot;
    l2.is_protected[slot] = 0;
    l2.last_used[slot] = ++l2.access_counter;
    return dst;
}

struct moe_cache_l2_stats {
    size_t budget_bytes;
    uint64_t slots;
    uint64_t used_bytes;
    uint64_t hits;
    uint64_t misses;
    uint64_t fills;
    uint64_t evictions;
    uint64_t fill_bytes;
    uint64_t fill_time_us;
    uint64_t phase_hits[2];
    uint64_t phase_misses[2];
    uint64_t phase_fills[2];
    uint64_t phase_evictions[2];
    uint64_t phase_fill_bytes[2];
    uint64_t phase_fill_time_us[2];
};

struct moe_cache_phase_stats {
    uint64_t l1_hits;
    uint64_t l1_misses;
    uint64_t l1_evictions;
    uint64_t h2d_copy_count;
    uint64_t h2d_copy_bytes;
    uint64_t h2d_enqueue_time_us;
    uint64_t prefetch_hits;
    uint64_t prefetch_misses;
    uint64_t prefetch_used;
    uint64_t prefetch_evictions;
    uint64_t demand_evictions;
    uint64_t evicted_prefetched;
    uint64_t evicted_hit_count_le1;
    uint64_t evicted_hit_count_ge2;
    uint64_t evicted_age_le_l1;
    uint64_t evicted_age_gt_l1;
    uint64_t prefetch_h2d_copy_count;
    uint64_t prefetch_h2d_copy_bytes;
    uint64_t prefetch_h2d_enqueue_time_us;
    uint64_t l2_hits;
    uint64_t l2_misses;
    uint64_t l2_fills;
    uint64_t l2_evictions;
    uint64_t l2_fill_bytes;
    uint64_t l2_fill_time_us;
    uint64_t ops;
    uint64_t staged_ops;
    uint64_t overflow_ops;
    uint64_t unique_experts;
    uint64_t unique_experts_max;
    uint64_t ids_bytes;
    uint64_t ids_d2h_time_us;
    uint64_t ids_d2h_sync_count;
    uint64_t ids_cache_hits;
    uint64_t acquire_time_us;
    uint64_t remap_time_us;
    uint64_t copy_wait_event_count;
    uint64_t copy_wait_event_time_us;
    uint64_t total_time_us;
};

struct moe_cache_reuse_hist {
    uint64_t first_touch = 0;
    uint64_t le8 = 0;
    uint64_t le16 = 0;
    uint64_t le32 = 0;
    uint64_t le_l1 = 0;
    uint64_t le_2xl1 = 0;
    uint64_t gt_2xl1 = 0;
};

struct moe_cache_expert_stats {
    uint64_t tensors;
    uint64_t experts;
    uint64_t unique_experts;
    uint64_t accesses;
    uint64_t first_touches;
    uint64_t reuse_le_l1;
    uint64_t reuse_le_l2;
    uint64_t reuse_gt_l2;
    uint64_t touched_once;
    uint64_t touched_ge2;
    uint64_t top1_accesses;
    uint64_t top5_accesses;
    uint64_t top10_accesses;
};

struct moe_cache_hot_tensor_stats {
    std::string name;
    uint64_t experts = 0;
    uint64_t unique_experts = 0;
    uint64_t accesses = 0;
    uint64_t first_touches = 0;
    uint64_t reuse_le_l1 = 0;
    uint64_t reuse_le_l2 = 0;
    uint64_t reuse_gt_l2 = 0;
    uint64_t touched_once = 0;
    uint64_t touched_ge2 = 0;
    uint64_t top1_accesses = 0;
    uint64_t top5_accesses = 0;
    uint64_t top10_accesses = 0;
};

struct moe_cache_tensor_decode_stats {
    std::string name;
    uint64_t experts = 0;
    uint64_t unique_experts = 0;
    uint64_t accesses = 0;
    uint64_t touched_once = 0;
    uint64_t touched_ge2 = 0;
    uint64_t top1_accesses = 0;
    uint64_t top5_accesses = 0;
    uint64_t top10_accesses = 0;
    uint64_t l1_hits = 0;
    uint64_t l1_misses = 0;
    uint64_t l1_evictions = 0;
    uint64_t h2d_copy_count = 0;
    uint64_t h2d_copy_bytes = 0;
    uint64_t h2d_enqueue_time_us = 0;
    uint64_t prefetch_h2d_copy_count = 0;
    uint64_t prefetch_h2d_copy_bytes = 0;
    uint64_t prefetch_used = 0;
    uint64_t prefetch_evictions = 0;
    uint64_t demand_evictions = 0;
    uint64_t evicted_prefetched = 0;
    uint64_t evicted_hit_count_le1 = 0;
    uint64_t evicted_hit_count_ge2 = 0;
    uint64_t evicted_age_le_l1 = 0;
    uint64_t evicted_age_gt_l1 = 0;
    uint64_t reuse_le_l1 = 0;
    uint64_t reuse_gt_l1 = 0;
    uint64_t reuse_total = 0;
    moe_cache_reuse_hist reuse_hist = {};
};

} // namespace

namespace {

struct moe_candidate_bank_record {
    ggml_cuda_moe_candidate_bank_info info;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_backend_buffer_type_t buft = nullptr;
    const void * buffer_base = nullptr;
    uint64_t buffer_size = 0;
    uint64_t data_offset = 0;
    uint64_t alignment = 0;
    int64_t ne[GGML_MAX_DIMS] = {};
    size_t nb[GGML_MAX_DIMS] = {};
};

struct moe_candidate_group_record {
    uint32_t layout = GGML_BACKEND_MOE_CANDIDATE_LAYOUT_INVALID;
    const ggml_tensor * down = nullptr;
    std::vector<moe_candidate_bank_record> banks;
};

struct moe_candidate_reverse_entry {
    uint32_t group_index;
    uint32_t bank_index;
};

struct moe_candidate_table {
    uint32_t n_slots = 0;
    uint64_t logical_signature = 0;
    uint64_t slot_bound_bytes = 0;
    uint64_t permanent_candidate_bytes = 0;
    std::vector<moe_candidate_group_record> groups;
    std::unordered_map<const ggml_tensor *, uint32_t> down_map;
    std::unordered_map<const ggml_tensor *, moe_candidate_reverse_entry> reverse_map;
};

static bool moe_candidate_add(uint64_t a, uint64_t b, uint64_t & result) {
    if (a > UINT64_MAX - b) {
        return false;
    }
    result = a + b;
    return true;
}

static bool moe_candidate_mul(uint64_t a, uint64_t b, uint64_t & result) {
    if (a != 0 && b > UINT64_MAX / a) {
        return false;
    }
    result = a * b;
    return true;
}

static void moe_candidate_hash_bytes(uint64_t & hash, const void * data, size_t size) {
    const uint8_t * bytes = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
}

template<typename T>
static void moe_candidate_hash_value(uint64_t & hash, const T & value) {
    moe_candidate_hash_bytes(hash, &value, sizeof(value));
}

static bool moe_candidate_is_scale(uint32_t role) {
    return role >= GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_SCALE &&
        role <= GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_SCALE;
}

static bool moe_candidate_is_block_scale(uint32_t role) {
    return role >= GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_BLOCK_SCALE &&
        role <= GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_BLOCK_SCALE;
}

static bool moe_candidate_is_bias(uint32_t role) {
    return role >= GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_BIAS &&
        role <= GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_BIAS;
}

static uint32_t moe_candidate_base_role(uint32_t role) {
    if (moe_candidate_is_scale(role)) {
        return GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT +
            role - GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_SCALE;
    }
    if (moe_candidate_is_block_scale(role)) {
        return GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT +
            role - GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_BLOCK_SCALE;
    }
    if (moe_candidate_is_bias(role)) {
        return GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT +
            role - GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_BIAS;
    }
    return GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID;
}

static ggml_cuda_moe_candidate_rejection moe_candidate_source(
        ggml_backend_dev_t owner,
        const ggml_tensor * tensor,
        uint64_t byte_extent,
        moe_candidate_bank_record & record) {
    if (tensor == nullptr || tensor->buffer == nullptr || tensor->data == nullptr || tensor->view_src != nullptr || tensor->op != GGML_OP_NONE) {
        return GGML_CUDA_MOE_CANDIDATE_REJECT_INVALID_TENSOR;
    }

    ggml_backend_buffer_t buffer = tensor->buffer;
    ggml_backend_buffer_type_t buft = buffer->buft;
    if (owner == nullptr || buft == nullptr || owner->iface.supports_buft == nullptr || !ggml_backend_dev_supports_buft(owner, buft)) {
        return GGML_CUDA_MOE_CANDIDATE_REJECT_INACCESSIBLE_SOURCE;
    }
    if (buffer->size == 0 || buffer->iface.get_base == nullptr || buft->iface.get_alignment == nullptr) {
        return GGML_CUDA_MOE_CANDIDATE_REJECT_INVALID_BOUNDS;
    }

    const void * base_ptr = buffer->iface.get_base(buffer);
    const size_t alignment = buft->iface.get_alignment(buft);
    if (base_ptr == nullptr || alignment == 0) {
        return GGML_CUDA_MOE_CANDIDATE_REJECT_INVALID_BOUNDS;
    }

    const uintptr_t base = reinterpret_cast<uintptr_t>(base_ptr);
    const uintptr_t data = reinterpret_cast<uintptr_t>(tensor->data);
    if (data < base || data % alignment != 0) {
        return GGML_CUDA_MOE_CANDIDATE_REJECT_INVALID_BOUNDS;
    }

    const uint64_t offset = data - base;
    if (offset > buffer->size || byte_extent > buffer->size - offset) {
        return GGML_CUDA_MOE_CANDIDATE_REJECT_INVALID_BOUNDS;
    }

    record.buffer = buffer;
    record.buft = buft;
    record.buffer_base = base_ptr;
    record.buffer_size = buffer->size;
    record.data_offset = offset;
    record.alignment = alignment;
    record.info.byte_extent = byte_extent;
    record.info.tensor = tensor;
    record.info.source_data = tensor->data;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        record.ne[i] = tensor->ne[i];
        record.nb[i] = tensor->nb[i];
    }
    return GGML_CUDA_MOE_CANDIDATE_REJECT_NONE;
}

static bool moe_candidate_record_matches(const moe_candidate_bank_record & record, const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor != record.info.tensor || tensor->buffer != record.buffer || tensor->data != record.info.source_data ||
            tensor->view_src != nullptr || tensor->op != GGML_OP_NONE || tensor->type != record.info.type) {
        return false;
    }
    if (record.buffer->buft != record.buft || record.buffer->size != record.buffer_size || record.buffer->iface.get_base == nullptr ||
            record.buft->iface.get_alignment == nullptr || record.buffer->iface.get_base(record.buffer) != record.buffer_base ||
            record.buft->iface.get_alignment(record.buft) != record.alignment) {
        return false;
    }
    const uintptr_t base = reinterpret_cast<uintptr_t>(record.buffer_base);
    const uintptr_t data = reinterpret_cast<uintptr_t>(tensor->data);
    if (data < base || data - base != record.data_offset || record.info.byte_extent > record.buffer_size - record.data_offset) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] != record.ne[i] || tensor->nb[i] != record.nb[i]) {
            return false;
        }
    }
    return true;
}

static const moe_candidate_bank_record * moe_candidate_find_role(const moe_candidate_group_record & group, uint32_t role) {
    for (const auto & bank : group.banks) {
        if (bank.info.role == role) {
            return &bank;
        }
    }
    return nullptr;
}

static bool moe_candidate_ids_valid(const ggml_tensor * ids) {
    if (ids == nullptr || ids->type != GGML_TYPE_I32 || ids->buffer == nullptr || ids->data == nullptr) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (ids->ne[i] <= 0 || ids->nb[i] == 0) {
            return false;
        }
    }
    return true;
}

static ggml_cuda_moe_ids_signature moe_candidate_ids_signature(const ggml_tensor * ids) {
    ggml_cuda_moe_ids_signature result;
    result.tensor = ids;
    result.data = ids->data;
    result.buffer = ids->buffer;
    result.type = ids->type;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        result.ne[i] = ids->ne[i];
        result.nb[i] = ids->nb[i];
    }
    return result;
}

static bool moe_candidate_ids_equal(const ggml_cuda_moe_ids_signature & a, const ggml_cuda_moe_ids_signature & b) {
    if (a.tensor != b.tensor || a.data != b.data || a.buffer != b.buffer || a.type != b.type) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (a.ne[i] != b.ne[i] || a.nb[i] != b.nb[i]) {
            return false;
        }
    }
    return true;
}

static ggml_cuda_moe_candidate_rejection moe_candidate_weight(
        ggml_backend_dev_t owner,
        const ggml_tensor * tensor,
        uint32_t role,
        uint32_t group_index,
        moe_candidate_bank_record & record) {
    if (tensor == nullptr || ggml_n_dims(tensor) != 3 || tensor->ne[0] <= 0 || tensor->ne[1] <= 0 || tensor->ne[2] <= 0 || tensor->ne[3] != 1) {
        return GGML_CUDA_MOE_CANDIDATE_REJECT_INVALID_TENSOR;
    }

    uint32_t encoding = GGML_CUDA_MOE_CANDIDATE_ENCODING_PLAIN;
    uint32_t index_modes = GGML_CUDA_MOE_CANDIDATE_INDEX_GROUP_SLOT_DIRECT |
        GGML_CUDA_MOE_CANDIDATE_INDEX_ORIGINAL_SOURCE_MAP;
    switch (tensor->type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_BF16:
            break;
        case GGML_TYPE_NVFP4:
            encoding = GGML_CUDA_MOE_CANDIDATE_ENCODING_NVFP4_COMPOUND;
            index_modes = GGML_CUDA_MOE_CANDIDATE_INDEX_GROUP_SLOT_DIRECT;
            break;
        default:
            return GGML_CUDA_MOE_CANDIDATE_REJECT_UNSUPPORTED_TYPE;
    }

    const int64_t block_size = ggml_blck_size(tensor->type);
    if (block_size <= 0 || tensor->ne[0] % block_size != 0) {
        return GGML_CUDA_MOE_CANDIDATE_REJECT_INCOMPATIBLE_SHAPE;
    }
    uint64_t row_bytes = 0;
    uint64_t expert_bytes = 0;
    uint64_t byte_extent = 0;
    if (!moe_candidate_mul(ggml_type_size(tensor->type), tensor->ne[0] / block_size, row_bytes) ||
            !moe_candidate_mul(row_bytes, tensor->ne[1], expert_bytes) ||
            !moe_candidate_mul(expert_bytes, tensor->ne[2], byte_extent)) {
        return GGML_CUDA_MOE_CANDIDATE_REJECT_OVERFLOW;
    }
    if (tensor->nb[0] != ggml_type_size(tensor->type) || tensor->nb[1] != row_bytes ||
            tensor->nb[2] != expert_bytes || tensor->nb[3] != byte_extent) {
        return GGML_CUDA_MOE_CANDIDATE_REJECT_INCOMPATIBLE_SHAPE;
    }

    record.info.group_index = group_index;
    record.info.role = role;
    record.info.type = tensor->type;
    record.info.encoding = encoding;
    record.info.movement = GGML_CUDA_MOE_CANDIDATE_MOVEMENT_SLOT_BOUND;
    record.info.index_modes = index_modes;
    record.info.expert_stride = tensor->nb[2];
    return moe_candidate_source(owner, tensor, byte_extent, record);
}

static ggml_cuda_moe_candidate_rejection moe_candidate_aux(
        ggml_backend_dev_t owner,
        const ggml_tensor * tensor,
        const ggml_tensor * weight,
        uint32_t role,
        uint32_t group_index,
        moe_candidate_bank_record & record) {
    if (tensor == nullptr || weight == nullptr || tensor->type != GGML_TYPE_F32 || tensor->ne[0] <= 0 || tensor->ne[1] <= 0 || tensor->ne[2] <= 0 || tensor->ne[3] <= 0) {
        return GGML_CUDA_MOE_CANDIDATE_REJECT_UNSUPPORTED_TYPE;
    }
    if (moe_candidate_is_block_scale(role)) {
        return GGML_CUDA_MOE_CANDIDATE_REJECT_UNSUPPORTED_ROLE;
    }

    uint64_t row_bytes = 0;
    if (!moe_candidate_mul(ggml_type_size(tensor->type), tensor->ne[0], row_bytes)) {
        return GGML_CUDA_MOE_CANDIDATE_REJECT_OVERFLOW;
    }
    if (tensor->nb[0] != ggml_type_size(tensor->type) || tensor->nb[1] != row_bytes) {
        return GGML_CUDA_MOE_CANDIDATE_REJECT_INCOMPATIBLE_SHAPE;
    }
    uint64_t byte_extent = 0;
    if (moe_candidate_is_scale(role)) {
        if (ggml_n_dims(tensor) != 1 || tensor->ne[0] != weight->ne[2] || tensor->ne[1] != 1 || tensor->ne[2] != 1 || tensor->ne[3] != 1 ||
                tensor->nb[2] != row_bytes || tensor->nb[3] != row_bytes) {
            return GGML_CUDA_MOE_CANDIDATE_REJECT_INCOMPATIBLE_SHAPE;
        }
        byte_extent = row_bytes;
    } else if (moe_candidate_is_bias(role)) {
        if (!moe_candidate_mul(row_bytes, tensor->ne[1], byte_extent)) {
            return GGML_CUDA_MOE_CANDIDATE_REJECT_OVERFLOW;
        }
        if (ggml_n_dims(tensor) != 2 || tensor->ne[0] != weight->ne[1] || tensor->ne[1] != weight->ne[2] || tensor->ne[2] != 1 || tensor->ne[3] != 1 ||
                tensor->nb[2] != byte_extent || tensor->nb[3] != byte_extent) {
            return GGML_CUDA_MOE_CANDIDATE_REJECT_INCOMPATIBLE_SHAPE;
        }
    } else {
        return GGML_CUDA_MOE_CANDIDATE_REJECT_INVALID_ROLE;
    }

    record.info.group_index = group_index;
    record.info.role = role;
    record.info.type = tensor->type;
    record.info.encoding = GGML_CUDA_MOE_CANDIDATE_ENCODING_PLAIN;
    record.info.movement = GGML_CUDA_MOE_CANDIDATE_MOVEMENT_PERMANENT_CANDIDATE;
    record.info.index_modes = GGML_CUDA_MOE_CANDIDATE_INDEX_ORIGINAL_DIRECT;
    return moe_candidate_source(owner, tensor, byte_extent, record);
}

static void moe_candidate_hash_bank(uint64_t & hash, const moe_candidate_bank_record & bank) {
    const uintptr_t tensor = reinterpret_cast<uintptr_t>(bank.info.tensor);
    const uintptr_t data = reinterpret_cast<uintptr_t>(bank.info.source_data);
    const uintptr_t buffer = reinterpret_cast<uintptr_t>(bank.buffer);
    const uintptr_t buft = reinterpret_cast<uintptr_t>(bank.buft);
    const uintptr_t buffer_base = reinterpret_cast<uintptr_t>(bank.buffer_base);
    moe_candidate_hash_value(hash, tensor);
    moe_candidate_hash_value(hash, data);
    moe_candidate_hash_value(hash, buffer);
    moe_candidate_hash_value(hash, buft);
    moe_candidate_hash_value(hash, buffer_base);
    moe_candidate_hash_value(hash, bank.buffer_size);
    moe_candidate_hash_value(hash, bank.info.role);
    moe_candidate_hash_value(hash, bank.info.type);
    moe_candidate_hash_value(hash, bank.info.encoding);
    moe_candidate_hash_value(hash, bank.info.movement);
    moe_candidate_hash_value(hash, bank.info.index_modes);
    moe_candidate_hash_value(hash, bank.data_offset);
    moe_candidate_hash_value(hash, bank.info.byte_extent);
    moe_candidate_hash_value(hash, bank.info.expert_stride);
    moe_candidate_hash_value(hash, bank.alignment);
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        moe_candidate_hash_value(hash, bank.ne[i]);
        moe_candidate_hash_value(hash, bank.nb[i]);
    }
}

static ggml_cuda_moe_candidate_rejection moe_candidate_group(
        ggml_backend_dev_t owner,
        const ggml_backend_moe_candidate_group_v1 & input,
        uint32_t group_index,
        uint32_t n_slots,
        std::unordered_set<const ggml_tensor *> & tensors,
        moe_candidate_table & table) {
    if (input.flags != GGML_BACKEND_MOE_CANDIDATE_GROUP_FLAG_NONE || input.reserved != 0) {
        return GGML_CUDA_MOE_CANDIDATE_REJECT_INVALID_FLAGS;
    }
    if (input.n_banks == 0 || input.n_banks > GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS || input.banks == nullptr) {
        return GGML_CUDA_MOE_CANDIDATE_REJECT_INVALID_COUNT;
    }

    const ggml_backend_moe_candidate_bank_v1 * by_role[GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_COUNT] = {};
    for (uint32_t i = 0; i < input.n_banks; ++i) {
        const auto & bank = input.banks[i];
        if (bank.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID || bank.role >= GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_COUNT) {
            return GGML_CUDA_MOE_CANDIDATE_REJECT_INVALID_ROLE;
        }
        if (bank.reserved != 0) {
            return GGML_CUDA_MOE_CANDIDATE_REJECT_INVALID_FLAGS;
        }
        if (by_role[bank.role] != nullptr) {
            return GGML_CUDA_MOE_CANDIDATE_REJECT_DUPLICATE_ROLE;
        }
        if (bank.tensor == nullptr || !tensors.insert(bank.tensor).second) {
            return GGML_CUDA_MOE_CANDIDATE_REJECT_DUPLICATE_TENSOR;
        }
        by_role[bank.role] = &bank;
    }

    const bool has_gate = by_role[GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT] != nullptr;
    const bool has_up = by_role[GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT] != nullptr;
    const bool has_gate_up = by_role[GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT] != nullptr;
    const bool has_down = by_role[GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT] != nullptr;
    if ((input.layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE && (!has_gate || !has_up || has_gate_up || !has_down)) ||
            (input.layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP && (has_gate || has_up || !has_gate_up || !has_down)) ||
            (input.layout != GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE && input.layout != GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP)) {
        return GGML_CUDA_MOE_CANDIDATE_REJECT_INVALID_LAYOUT;
    }

    moe_candidate_group_record group;
    group.layout = input.layout;
    group.banks.reserve(input.n_banks);
    const ggml_tensor * weights[GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_COUNT] = {};

    for (uint32_t role = GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT; role <= GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT; ++role) {
        if (by_role[role] == nullptr) {
            continue;
        }
        moe_candidate_bank_record record;
        const auto rejection = moe_candidate_weight(owner, by_role[role]->tensor, role, group_index, record);
        if (rejection != GGML_CUDA_MOE_CANDIDATE_REJECT_NONE) {
            return rejection;
        }
        weights[role] = by_role[role]->tensor;
        group.banks.push_back(record);
    }

    const ggml_tensor * down = weights[GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT];
    if (input.layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE) {
        const ggml_tensor * gate = weights[GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT];
        const ggml_tensor * up = weights[GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT];
        if (gate->ne[0] != up->ne[0] || gate->ne[1] != up->ne[1] || gate->ne[2] != up->ne[2] ||
                down->ne[0] != gate->ne[1] || down->ne[1] != gate->ne[0] || down->ne[2] != gate->ne[2]) {
            return GGML_CUDA_MOE_CANDIDATE_REJECT_INCOMPATIBLE_SHAPE;
        }
    } else {
        const ggml_tensor * gate_up = weights[GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT];
        uint64_t doubled_down = 0;
        if (!moe_candidate_mul(down->ne[0], 2, doubled_down) || gate_up->ne[0] != down->ne[1] ||
                static_cast<uint64_t>(gate_up->ne[1]) != doubled_down || gate_up->ne[2] != down->ne[2]) {
            return GGML_CUDA_MOE_CANDIDATE_REJECT_INCOMPATIBLE_SHAPE;
        }
    }

    for (uint32_t role = GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_SCALE; role < GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_COUNT; ++role) {
        if (by_role[role] == nullptr) {
            continue;
        }
        const uint32_t base_role = moe_candidate_base_role(role);
        if (base_role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID || weights[base_role] == nullptr) {
            return GGML_CUDA_MOE_CANDIDATE_REJECT_INVALID_ROLE;
        }
        moe_candidate_bank_record record;
        const auto rejection = moe_candidate_aux(owner, by_role[role]->tensor, weights[base_role], role, group_index, record);
        if (rejection != GGML_CUDA_MOE_CANDIDATE_REJECT_NONE) {
            return rejection;
        }
        group.banks.push_back(record);
    }

    uint64_t group_hash = UINT64_C(1469598103934665603);
    moe_candidate_hash_value(group_hash, group.layout);
    const uint32_t n_banks = group.banks.size();
    moe_candidate_hash_value(group_hash, n_banks);
    for (uint32_t bank_index = 0; bank_index < group.banks.size(); ++bank_index) {
        auto & bank = group.banks[bank_index];
        moe_candidate_hash_bank(group_hash, bank);
        if (bank.info.movement == GGML_CUDA_MOE_CANDIDATE_MOVEMENT_SLOT_BOUND) {
            uint64_t bytes = 0;
            if (!moe_candidate_mul(bank.info.expert_stride, n_slots, bytes) ||
                    !moe_candidate_add(table.slot_bound_bytes, bytes, table.slot_bound_bytes)) {
                return GGML_CUDA_MOE_CANDIDATE_REJECT_OVERFLOW;
            }
            const auto inserted = table.reverse_map.emplace(by_role[bank.info.role]->tensor, moe_candidate_reverse_entry{group_index, bank_index});
            if (!inserted.second) {
                return GGML_CUDA_MOE_CANDIDATE_REJECT_DUPLICATE_TENSOR;
            }
        } else if (!moe_candidate_add(table.permanent_candidate_bytes, bank.info.byte_extent, table.permanent_candidate_bytes)) {
            return GGML_CUDA_MOE_CANDIDATE_REJECT_OVERFLOW;
        }
    }

    group.down = down;
    if (!table.down_map.emplace(down, group_index).second) {
        return GGML_CUDA_MOE_CANDIDATE_REJECT_DUPLICATE_TENSOR;
    }
    moe_candidate_hash_value(table.logical_signature, group_hash);
    table.groups.push_back(std::move(group));
    return GGML_CUDA_MOE_CANDIDATE_REJECT_NONE;
}

static ggml_cuda_moe_candidate_rejection moe_candidate_build(
        ggml_backend_dev_t owner,
        const ggml_backend_moe_candidate_snapshot_v1 & snapshot,
        moe_candidate_table & table) {
    if (snapshot.flags != GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_FLAG_NONE || snapshot.reserved[0] != 0 || snapshot.reserved[1] != 0) {
        return GGML_CUDA_MOE_CANDIDATE_REJECT_INVALID_FLAGS;
    }
    if (snapshot.n_groups > GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS || (snapshot.n_groups > 0 && snapshot.groups == nullptr)) {
        return GGML_CUDA_MOE_CANDIDATE_REJECT_INVALID_COUNT;
    }

    table.n_slots = snapshot.n_slots;
    table.logical_signature = UINT64_C(1469598103934665603);
    moe_candidate_hash_value(table.logical_signature, snapshot.n_groups);
    table.groups.reserve(snapshot.n_groups);
    table.down_map.reserve(snapshot.n_groups);
    table.reverse_map.reserve(snapshot.n_groups * 3);
    std::unordered_set<const ggml_tensor *> tensors;
    tensors.reserve(snapshot.n_groups * 4);
    for (uint32_t i = 0; i < snapshot.n_groups; ++i) {
        const auto rejection = moe_candidate_group(owner, snapshot.groups[i], i, snapshot.n_slots, tensors, table);
        if (rejection != GGML_CUDA_MOE_CANDIDATE_REJECT_NONE) {
            return rejection;
        }
    }
    return GGML_CUDA_MOE_CANDIDATE_REJECT_NONE;
}

} // namespace

ggml_cuda_moe_graph_plan::ggml_cuda_moe_graph_plan() :
    owner_(nullptr), registry_generation_(0), graph_uid_(0), graph_node_count_(0), n_groups_(0), n_nodes_(0), initialized_(false) {
}

void ggml_cuda_moe_graph_plan::reset() {
    for (auto & entry : nodes_) {
        entry.node = nullptr;
    }
    owner_ = nullptr;
    registry_generation_ = 0;
    graph_uid_ = 0;
    graph_node_count_ = 0;
    n_groups_ = 0;
    n_nodes_ = 0;
    initialized_ = false;
}

static uint32_t ggml_cuda_moe_graph_node_hash(const ggml_tensor * node) {
    uint64_t value = reinterpret_cast<uintptr_t>(node);
    value ^= value >> 33;
    value *= UINT64_C(0xff51afd7ed558ccd);
    value ^= value >> 33;
    return static_cast<uint32_t>(value);
}

bool ggml_cuda_moe_graph_plan::insert(
        const ggml_tensor * node,
        uint32_t group_record,
        uint32_t role,
        uint32_t bank_index) {
    if (node == nullptr || group_record >= n_groups_ || n_nodes_ == MAX_NODE_BINDINGS) {
        return false;
    }
    uint32_t index = ggml_cuda_moe_graph_node_hash(node) & (NODE_TABLE_SIZE - 1);
    for (uint32_t probe = 0; probe < NODE_TABLE_SIZE; ++probe) {
        auto & entry = nodes_[index];
        if (entry.node == nullptr) {
            entry = {node, group_record, role, bank_index};
            ++n_nodes_;
            return true;
        }
        if (entry.node == node) {
            return false;
        }
        index = (index + 1) & (NODE_TABLE_SIZE - 1);
    }
    return false;
}

const ggml_cuda_moe_graph_plan::node_entry * ggml_cuda_moe_graph_plan::find(const ggml_tensor * node) const {
    if (!initialized_ || node == nullptr) {
        return nullptr;
    }
    uint32_t index = ggml_cuda_moe_graph_node_hash(node) & (NODE_TABLE_SIZE - 1);
    for (uint32_t probe = 0; probe < NODE_TABLE_SIZE; ++probe) {
        const auto & entry = nodes_[index];
        if (entry.node == nullptr) {
            return nullptr;
        }
        if (entry.node == node) {
            return &entry;
        }
        index = (index + 1) & (NODE_TABLE_SIZE - 1);
    }
    return nullptr;
}

uint32_t ggml_cuda_moe_graph_plan::size() const {
    return initialized_ ? n_groups_ : 0;
}

uint64_t ggml_cuda_moe_graph_plan::registry_generation() const {
    return initialized_ ? registry_generation_ : 0;
}

uint64_t ggml_cuda_moe_graph_plan::graph_uid() const {
    return initialized_ ? graph_uid_ : 0;
}

int32_t ggml_cuda_moe_graph_plan::graph_node_count() const {
    return initialized_ ? graph_node_count_ : 0;
}

ggml_cuda_moe_graph_execution::ggml_cuda_moe_graph_execution() : plan_(nullptr), n_groups_(0) {
}

void ggml_cuda_moe_graph_execution::reset() {
    plan_ = nullptr;
    n_groups_ = 0;
}

bool ggml_cuda_moe_graph_execution::find(const ggml_tensor * node, ggml_cuda_moe_graph_binding * binding) const {
    if (plan_ == nullptr) {
        return false;
    }
    const auto * entry = plan_->find(node);
    if (entry == nullptr || entry->group_record >= n_groups_) {
        return false;
    }
    if (binding != nullptr) {
        binding->key = groups_[entry->group_record];
        binding->role = entry->role;
        binding->bank_index = entry->bank_index;
    }
    return true;
}

uint32_t ggml_cuda_moe_graph_execution::size() const {
    return plan_ != nullptr ? n_groups_ : 0;
}

struct ggml_cuda_moe_grouped_context::impl {
    explicit impl(ggml_backend_dev_t owner) : owner(owner) {}

    struct grouped_snapshot {
        ggml_cuda_moe_grouped_acquisition acquisition;
        const ggml_tensor * down = nullptr;
        uint32_t layout = GGML_BACKEND_MOE_CANDIDATE_LAYOUT_INVALID;
        uint32_t n_slots = 0;
        std::vector<ggml_cuda_moe_grouped_bank_descriptor> banks;
    };

    struct grouped_resource {
        explicit grouped_resource(grouped_snapshot && snapshot) : snapshot(std::move(snapshot)) {}

        const grouped_snapshot snapshot;
        uint64_t active_transaction_token = 0;
    };

    struct resource_build_input {
        ggml_cuda_moe_candidate_group_key candidate;
        uint64_t resource_generation = 0;
        const ggml_tensor * down = nullptr;
        uint32_t layout = GGML_BACKEND_MOE_CANDIDATE_LAYOUT_INVALID;
        uint32_t n_slots = 0;
        uint32_t n_groups = 0;
        uint32_t n_banks = 0;
        std::array<moe_candidate_bank_record, GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS> banks;
    };

    using resource_slots = std::vector<std::unique_ptr<grouped_resource>>;

    ggml_backend_dev_t owner;
    std::mutex resource_lifecycle_mutex;
    mutable std::mutex mutex;
    std::condition_variable resource_cv;
    ggml_cuda_moe_candidate_registry_state state;
    moe_candidate_table table;
    resource_slots resources;
    uint64_t next_resource_generation = 0;
    uint64_t next_transaction_token = 0;
    bool replacement_pending = false;
    bool draining = false;

    grouped_resource * find_resource(const ggml_cuda_moe_grouped_acquisition & acquisition) {
        if (acquisition.resource_generation == 0 || acquisition.candidate.generation != state.generation ||
                acquisition.candidate.group_index >= resources.size()) {
            return nullptr;
        }
        auto * resource = resources[acquisition.candidate.group_index].get();
        if (resource == nullptr || resource->snapshot.acquisition.resource_generation != acquisition.resource_generation ||
                resource->snapshot.acquisition.candidate.generation != acquisition.candidate.generation ||
                resource->snapshot.acquisition.candidate.group_index != acquisition.candidate.group_index) {
            return nullptr;
        }
        return resource;
    }

    const grouped_resource * find_resource(const ggml_cuda_moe_grouped_acquisition & acquisition) const {
        return const_cast<impl *>(this)->find_resource(acquisition);
    }

    grouped_resource * find_resource(const ggml_cuda_moe_grouped_transaction & transaction) {
        auto * resource = find_resource(transaction.acquisition);
        if (transaction.transaction_token == 0 || resource == nullptr ||
                resource->active_transaction_token != transaction.transaction_token) {
            return nullptr;
        }
        return resource;
    }

    const grouped_resource * find_resource(const ggml_cuda_moe_grouped_transaction & transaction) const {
        return const_cast<impl *>(this)->find_resource(transaction);
    }

    resource_slots detach_resources() {
        resource_slots result;
        result.swap(resources);
        return result;
    }

    static void retire_resources(resource_slots resources) {
        resources.clear();
    }

    bool has_active_transaction() const {
        for (const auto & resource : resources) {
            if (resource != nullptr && resource->active_transaction_token != 0) {
                return true;
            }
        }
        return false;
    }
};

ggml_cuda_moe_grouped_context::ggml_cuda_moe_grouped_context(ggml_backend_dev_t owner) :
    impl_(std::make_unique<impl>(owner)) {
}

ggml_cuda_moe_grouped_context::~ggml_cuda_moe_grouped_context() {
    shutdown();
}

int32_t ggml_cuda_moe_grouped_context::replace(const ggml_backend_moe_candidate_snapshot_v1 * snapshot) {
    auto publish_failure = [&](ggml_cuda_moe_candidate_rejection rejection, uint32_t n_slots, int32_t result) {
        std::lock_guard<std::mutex> lifecycle_lock(impl_->resource_lifecycle_mutex);
        impl::resource_slots retired;
        int32_t published = result;
        {
            std::unique_lock<std::mutex> lock(impl_->mutex);
            if (impl_->draining) {
                return static_cast<int32_t>(GGML_BACKEND_MOE_CANDIDATE_REPLACE_ERROR);
            }
            impl_->replacement_pending = true;
            impl_->resource_cv.wait(lock, [&]() { return !impl_->has_active_transaction(); });
            retired = impl_->detach_resources();
            impl_->table = {};
            if (impl_->state.generation == UINT64_MAX) {
                impl_->state = {};
                impl_->state.generation = UINT64_MAX;
                impl_->state.rejection = GGML_CUDA_MOE_CANDIDATE_REJECT_GENERATION_EXHAUSTED;
                published = GGML_BACKEND_MOE_CANDIDATE_REPLACE_ERROR;
            } else {
                const uint64_t generation = impl_->state.generation + 1;
                impl_->state = {};
                impl_->state.generation = generation;
                impl_->state.n_slots = n_slots;
                impl_->state.rejection = rejection;
            }
            impl_->replacement_pending = false;
        }
        impl::retire_resources(std::move(retired));
        return published;
    };

    if (snapshot == nullptr) {
        return publish_failure(GGML_CUDA_MOE_CANDIDATE_REJECT_INVALID_ABI, 0, GGML_BACKEND_MOE_CANDIDATE_REPLACE_INVALID_ARGUMENT);
    }
    if (snapshot->magic != GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V1_MAGIC ||
            snapshot->abi_version != GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V1_VERSION ||
            snapshot->struct_size != sizeof(*snapshot)) {
        return publish_failure(GGML_CUDA_MOE_CANDIDATE_REJECT_INVALID_ABI, 0, GGML_BACKEND_MOE_CANDIDATE_REPLACE_INVALID_ABI);
    }

    try {
        moe_candidate_table table;
        const auto rejection = moe_candidate_build(impl_->owner, *snapshot, table);
        if (rejection != GGML_CUDA_MOE_CANDIDATE_REJECT_NONE) {
            return publish_failure(rejection, snapshot->n_slots, GGML_BACKEND_MOE_CANDIDATE_REPLACE_REJECTED);
        }

        std::lock_guard<std::mutex> lifecycle_lock(impl_->resource_lifecycle_mutex);
        impl::resource_slots retired;
        int32_t published = GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED;
        {
            std::unique_lock<std::mutex> lock(impl_->mutex);
            if (impl_->draining) {
                return static_cast<int32_t>(GGML_BACKEND_MOE_CANDIDATE_REPLACE_ERROR);
            }
            impl_->replacement_pending = true;
            impl_->resource_cv.wait(lock, [&]() { return !impl_->has_active_transaction(); });
            retired = impl_->detach_resources();
            if (impl_->state.generation == UINT64_MAX) {
                impl_->table = {};
                impl_->state.accepted = 0;
                impl_->state.rejection = GGML_CUDA_MOE_CANDIDATE_REJECT_GENERATION_EXHAUSTED;
                published = GGML_BACKEND_MOE_CANDIDATE_REPLACE_ERROR;
            } else {
                const uint64_t generation = impl_->state.generation + 1;
                impl_->table = std::move(table);
                impl_->state = {};
                impl_->state.generation = generation;
                impl_->state.logical_signature = impl_->table.logical_signature;
                impl_->state.slot_bound_bytes = impl_->table.slot_bound_bytes;
                impl_->state.permanent_candidate_bytes = impl_->table.permanent_candidate_bytes;
                impl_->state.n_slots = impl_->table.n_slots;
                impl_->state.n_groups = impl_->table.groups.size();
                impl_->state.n_weights = impl_->table.reverse_map.size();
                impl_->state.accepted = 1;
            }
            impl_->replacement_pending = false;
        }
        impl::retire_resources(std::move(retired));
        return published;
    } catch (const std::bad_alloc &) {
        return publish_failure(GGML_CUDA_MOE_CANDIDATE_REJECT_ALLOCATION, snapshot->n_slots, GGML_BACKEND_MOE_CANDIDATE_REPLACE_ERROR);
    } catch (...) {
        return publish_failure(GGML_CUDA_MOE_CANDIDATE_REJECT_ALLOCATION, snapshot->n_slots, GGML_BACKEND_MOE_CANDIDATE_REPLACE_ERROR);
    }
}

ggml_cuda_moe_candidate_registry_state ggml_cuda_moe_grouped_context::state() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->state;
}

bool ggml_cuda_moe_grouped_context::find_down_group(const ggml_tensor * tensor, uint32_t * group_index) const {
    ggml_cuda_moe_candidate_group_key key;
    if (!find_down_group_key(tensor, &key)) {
        return false;
    }
    if (group_index != nullptr) {
        *group_index = key.group_index;
    }
    return true;
}

bool ggml_cuda_moe_grouped_context::find_down_group_key(
        const ggml_tensor * tensor,
        ggml_cuda_moe_candidate_group_key * key) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto it = impl_->table.down_map.find(tensor);
    if (!impl_->state.accepted || it == impl_->table.down_map.end()) {
        return false;
    }
    if (key != nullptr) {
        key->generation = impl_->state.generation;
        key->group_index = it->second;
    }
    return true;
}

bool ggml_cuda_moe_grouped_context::find_weight(const ggml_tensor * tensor, ggml_cuda_moe_candidate_bank_info * info) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto it = impl_->table.reverse_map.find(tensor);
    if (!impl_->state.accepted || it == impl_->table.reverse_map.end()) {
        return false;
    }
    if (info != nullptr) {
        *info = impl_->table.groups[it->second.group_index].banks[it->second.bank_index].info;
        info->generation = impl_->state.generation;
    }
    return true;
}

bool ggml_cuda_moe_grouped_context::get_group(
        const ggml_cuda_moe_candidate_group_key & key,
        ggml_cuda_moe_candidate_group_info * info) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->state.accepted || key.generation != impl_->state.generation || key.group_index >= impl_->table.groups.size()) {
        return false;
    }
    if (info != nullptr) {
        const auto & group = impl_->table.groups[key.group_index];
        info->key = key;
        info->down = group.down;
        info->layout = group.layout;
        info->n_banks = static_cast<uint32_t>(group.banks.size());
        info->n_slots = impl_->table.n_slots;
    }
    return true;
}

bool ggml_cuda_moe_grouped_context::get_bank(
        const ggml_cuda_moe_candidate_group_key & key,
        uint32_t role,
        ggml_cuda_moe_candidate_bank_info * info) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->state.accepted || key.generation != impl_->state.generation || key.group_index >= impl_->table.groups.size()) {
        return false;
    }
    for (const auto & bank : impl_->table.groups[key.group_index].banks) {
        if (bank.info.role == role) {
            if (info != nullptr) {
                *info = bank.info;
                info->generation = key.generation;
            }
            return true;
        }
    }
    return false;
}

bool ggml_cuda_moe_grouped_context::probe(
        const ggml_cuda_moe_candidate_probe_input & input,
        ggml_cuda_moe_candidate_probe_result * result) const {
    if (result != nullptr) {
        *result = {};
    }
    if (input.n_banks == 0 || input.n_banks > 2 || input.exact_auxiliaries > 1) {
        return false;
    }

    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->state.accepted || (input.expected_generation != 0 && input.expected_generation != impl_->state.generation)) {
        return false;
    }

    uint32_t group_index = UINT32_MAX;
    uint32_t roles[2] = {};
    for (uint32_t i = 0; i < input.n_banks; ++i) {
        const auto & observed = input.banks[i];
        if (!moe_candidate_ids_valid(observed.ids) || (i > 0 && observed.ids != input.banks[0].ids) ||
                (i > 0 && observed.weight == input.banks[0].weight)) {
            return false;
        }
        const auto reverse = impl_->table.reverse_map.find(observed.weight);
        if (reverse == impl_->table.reverse_map.end()) {
            return false;
        }
        const auto & entry = reverse->second;
        const auto & record = impl_->table.groups[entry.group_index].banks[entry.bank_index];
        if (!moe_candidate_record_matches(record, observed.weight) ||
                (observed.expected_role != GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID && observed.expected_role != record.info.role) ||
                (i > 0 && entry.group_index != group_index)) {
            return false;
        }
        group_index = entry.group_index;
        roles[i] = record.info.role;
    }
    if (input.n_banks == 2 && roles[0] == roles[1]) {
        return false;
    }

    const auto & group = impl_->table.groups[group_index];
    for (uint32_t i = 0; i < input.n_banks; ++i) {
        const auto & observed = input.banks[i];
        if (!input.exact_auxiliaries) {
            if (observed.scale != nullptr || observed.bias != nullptr) {
                return false;
            }
            continue;
        }
        const uint32_t scale_role = GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_SCALE + roles[i] - GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT;
        const uint32_t bias_role = GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_BIAS + roles[i] - GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT;
        const auto * scale = moe_candidate_find_role(group, scale_role);
        const auto * bias = moe_candidate_find_role(group, bias_role);
        if ((scale == nullptr) != (observed.scale == nullptr) || (bias == nullptr) != (observed.bias == nullptr) ||
                (scale != nullptr && !moe_candidate_record_matches(*scale, observed.scale)) ||
                (bias != nullptr && !moe_candidate_record_matches(*bias, observed.bias))) {
            return false;
        }
    }

    if (result != nullptr) {
        result->key.generation = impl_->state.generation;
        result->key.group_index = group_index;
        result->roles[0] = roles[0];
        result->roles[1] = roles[1];
    }
    return true;
}

bool ggml_cuda_moe_grouped_context::acquire_group_resources(
        const ggml_cuda_moe_candidate_group_key & key,
        ggml_cuda_moe_grouped_acquisition * acquisition) {
    if (acquisition == nullptr) {
        return false;
    }
    *acquisition = {};
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (impl_->draining || impl_->replacement_pending) {
            return false;
        }
    }
    std::lock_guard<std::mutex> lifecycle_lock(impl_->resource_lifecycle_mutex);

    impl::resource_build_input input;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (impl_->draining || impl_->replacement_pending || !impl_->state.accepted ||
                key.generation != impl_->state.generation || key.group_index >= impl_->table.groups.size()) {
            return false;
        }
        if (key.group_index < impl_->resources.size() && impl_->resources[key.group_index] != nullptr) {
            *acquisition = impl_->resources[key.group_index]->snapshot.acquisition;
            return true;
        }
        if (impl_->next_resource_generation == UINT64_MAX) {
            return false;
        }
        input.candidate = key;
        input.resource_generation = ++impl_->next_resource_generation;
        input.n_slots = impl_->table.n_slots;
        input.n_groups = static_cast<uint32_t>(impl_->table.groups.size());
        const auto & group = impl_->table.groups[key.group_index];
        input.down = group.down;
        input.layout = group.layout;
        input.n_banks = static_cast<uint32_t>(group.banks.size());
        for (uint32_t i = 0; i < input.n_banks; ++i) {
            input.banks[i] = group.banks[i];
        }
    }

    impl::resource_slots prospective;
    try {
        impl::grouped_snapshot snapshot;
        snapshot.acquisition.candidate = input.candidate;
        snapshot.acquisition.resource_generation = input.resource_generation;
        snapshot.down = input.down;
        snapshot.layout = input.layout;
        snapshot.n_slots = input.n_slots;
        snapshot.banks.reserve(input.n_banks);
        for (uint32_t i = 0; i < input.n_banks; ++i) {
            const auto & source = input.banks[i];
            ggml_cuda_moe_grouped_bank_descriptor descriptor;
            descriptor.tensor = source.info.tensor;
            descriptor.buffer = source.buffer;
            descriptor.buft = source.buft;
            descriptor.source_data = source.info.source_data;
            descriptor.buffer_base = source.buffer_base;
            descriptor.buffer_size = source.buffer_size;
            descriptor.data_offset = source.data_offset;
            descriptor.byte_extent = source.info.byte_extent;
            descriptor.expert_stride = source.info.expert_stride;
            descriptor.alignment = source.alignment;
            descriptor.role = source.info.role;
            descriptor.type = source.info.type;
            descriptor.encoding = source.info.encoding;
            descriptor.movement = source.info.movement;
            descriptor.index_modes = source.info.index_modes;
            for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
                descriptor.ne[dim] = source.ne[dim];
                descriptor.nb[dim] = source.nb[dim];
            }
            snapshot.banks.push_back(descriptor);
        }
        prospective.resize(input.n_groups);
        prospective[key.group_index] = std::make_unique<impl::grouped_resource>(std::move(snapshot));
    } catch (const std::bad_alloc &) {
        return false;
    }

    ggml_cuda_moe_grouped_acquisition result;
    bool installed = false;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (!impl_->draining && !impl_->replacement_pending && impl_->state.accepted && input.candidate.generation == impl_->state.generation &&
                input.candidate.group_index < impl_->table.groups.size() && input.n_slots == impl_->table.n_slots &&
                input.n_groups == impl_->table.groups.size()) {
            const auto & group = impl_->table.groups[input.candidate.group_index];
            if (input.down == group.down && input.layout == group.layout && input.n_banks == group.banks.size()) {
                if (impl_->resources.empty()) {
                    impl_->resources = std::move(prospective);
                } else if (impl_->resources.size() == input.n_groups && impl_->resources[input.candidate.group_index] == nullptr) {
                    impl_->resources[input.candidate.group_index] = std::move(prospective[input.candidate.group_index]);
                }
                if (input.candidate.group_index < impl_->resources.size() && impl_->resources[input.candidate.group_index] != nullptr) {
                    result = impl_->resources[input.candidate.group_index]->snapshot.acquisition;
                    installed = true;
                }
            }
        }
    }
    if (installed) {
        *acquisition = result;
    }
    return installed;
}

bool ggml_cuda_moe_grouped_context::begin_group_transaction(
        const ggml_cuda_moe_grouped_acquisition & acquisition,
        ggml_cuda_moe_grouped_transaction * transaction) {
    if (transaction == nullptr) {
        return false;
    }
    *transaction = {};
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto * resource = impl_->find_resource(acquisition);
    if (impl_->draining || impl_->replacement_pending || resource == nullptr ||
            resource->active_transaction_token != 0 || impl_->next_transaction_token == UINT64_MAX) {
        return false;
    }
    resource->active_transaction_token = ++impl_->next_transaction_token;
    transaction->acquisition = resource->snapshot.acquisition;
    transaction->transaction_token = resource->active_transaction_token;
    return true;
}

bool ggml_cuda_moe_grouped_context::end_group_transaction(const ggml_cuda_moe_grouped_transaction & transaction) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto * resource = impl_->find_resource(transaction);
    if (resource == nullptr) {
        return false;
    }
    resource->active_transaction_token = 0;
    impl_->resource_cv.notify_all();
    return true;
}

bool ggml_cuda_moe_grouped_context::get_group_resources(
        const ggml_cuda_moe_grouped_acquisition & acquisition,
        ggml_cuda_moe_grouped_resource_info * info) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto * resource = impl_->find_resource(acquisition);
    if (impl_->draining || resource == nullptr) {
        return false;
    }
    if (info != nullptr) {
        info->acquisition = resource->snapshot.acquisition;
        info->down = resource->snapshot.down;
        info->layout = resource->snapshot.layout;
        info->n_slots = resource->snapshot.n_slots;
        info->n_banks = static_cast<uint32_t>(resource->snapshot.banks.size());
        info->transaction_active = resource->active_transaction_token != 0;
    }
    return true;
}

bool ggml_cuda_moe_grouped_context::get_group_resource_bank(
        const ggml_cuda_moe_grouped_transaction & transaction,
        uint32_t bank_index,
        ggml_cuda_moe_grouped_bank_descriptor * descriptor) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto * resource = impl_->find_resource(transaction);
    if (resource == nullptr || bank_index >= resource->snapshot.banks.size()) {
        return false;
    }
    if (descriptor != nullptr) {
        *descriptor = resource->snapshot.banks[bank_index];
    }
    return true;
}

void ggml_cuda_moe_grouped_context::compile_graph_plan(
        const ggml_cgraph * cgraph,
        uint64_t graph_uid,
        ggml_cuda_moe_graph_plan * plan,
        ggml_cuda_moe_graph_execution * execution) const {
    if (plan != nullptr) {
        plan->reset();
    }
    if (execution != nullptr) {
        execution->reset();
    }
    if (cgraph == nullptr || plan == nullptr || execution == nullptr) {
        return;
    }

    struct group_observation {
        ggml_cuda_moe_ids_signature ids;
        const ggml_tensor * nodes[4] = {};
        uint32_t node_indices[4] = {};
        uint32_t bank_indices[4] = {};
        uint32_t required_roles = 0;
        uint32_t seen_roles = 0;
        uint32_t n_banks = 0;
        bool eligible = false;
        bool invalid = false;
        bool has_ids = false;
    };

    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->draining || impl_->replacement_pending || !impl_->state.accepted || impl_->state.n_slots == 0 || impl_->table.groups.empty()) {
        return;
    }
    plan->owner_ = impl_.get();
    plan->registry_generation_ = impl_->state.generation;
    plan->graph_uid_ = graph_uid;
    plan->graph_node_count_ = ggml_graph_n_nodes(const_cast<ggml_cgraph *>(cgraph));
    plan->initialized_ = true;

    std::array<group_observation, GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS> observations = {};
    const uint32_t mapped_index_modes = GGML_CUDA_MOE_CANDIDATE_INDEX_GROUP_SLOT_DIRECT |
        GGML_CUDA_MOE_CANDIDATE_INDEX_ORIGINAL_SOURCE_MAP;
    for (uint32_t group_index = 0; group_index < impl_->table.groups.size(); ++group_index) {
        const auto & group = impl_->table.groups[group_index];
        auto & observation = observations[group_index];
        if (group.layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP) {
            observation.required_roles = (1u << GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT) |
                (1u << GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT);
            observation.n_banks = 2;
        } else if (group.layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE) {
            observation.required_roles = (1u << GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT) |
                (1u << GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT) |
                (1u << GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT);
            observation.n_banks = 3;
        } else {
            continue;
        }
        if (group.banks.size() != observation.n_banks) {
            continue;
        }
        observation.eligible = true;
        uint32_t group_roles = 0;
        for (uint32_t bank_index = 0; bank_index < group.banks.size(); ++bank_index) {
            const auto & bank = group.banks[bank_index];
            const uint32_t role = bank.info.role;
            if (role < GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT || role > GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT ||
                    (observation.required_roles & (1u << role)) == 0 || (group_roles & (1u << role)) != 0 ||
                    (bank.info.type != GGML_TYPE_Q4_0 && bank.info.type != GGML_TYPE_Q4_K) ||
                    bank.info.encoding != GGML_CUDA_MOE_CANDIDATE_ENCODING_PLAIN ||
                    bank.info.movement != GGML_CUDA_MOE_CANDIDATE_MOVEMENT_SLOT_BOUND || bank.info.index_modes != mapped_index_modes) {
                observation.eligible = false;
                break;
            }
            group_roles |= 1u << role;
            observation.bank_indices[role - GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT] = bank_index;
        }
        observation.eligible = observation.eligible && group_roles == observation.required_roles;
    }

    const int n_nodes = plan->graph_node_count_;
    for (int node_index = 0; node_index < n_nodes; ++node_index) {
        const ggml_tensor * node = ggml_graph_node(const_cast<ggml_cgraph *>(cgraph), node_index);
        if (node == nullptr) {
            continue;
        }
        if (node->op == GGML_OP_ADD || node->op == GGML_OP_ADD_ID || node->op == GGML_OP_MUL) {
            for (uint32_t src_index = 0; src_index < 2; ++src_index) {
                const ggml_tensor * mmid = node->src[src_index];
                if (mmid == nullptr || mmid->op != GGML_OP_MUL_MAT_ID || mmid->src[0] == nullptr) {
                    continue;
                }
                const auto reverse = impl_->table.reverse_map.find(mmid->src[0]);
                if (reverse == impl_->table.reverse_map.end() || reverse->second.group_index >= impl_->table.groups.size()) {
                    continue;
                }
                auto & observation = observations[reverse->second.group_index];
                if (!observation.eligible) {
                    continue;
                }
                const ggml_tensor * other = node->src[1 - src_index];
                const bool is_scale = node->op == GGML_OP_MUL && other != nullptr && other->op == GGML_OP_GET_ROWS && other->src[1] == mmid->src[2];
                if (node->op == GGML_OP_ADD || node->op == GGML_OP_ADD_ID || is_scale) {
                    observation.invalid = true;
                }
            }
        }
        if (node->op != GGML_OP_MUL_MAT_ID || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || node->src[0] == nullptr || node->src[1] == nullptr) {
            continue;
        }
        const auto reverse = impl_->table.reverse_map.find(node->src[0]);
        if (reverse == impl_->table.reverse_map.end() || reverse->second.group_index >= impl_->table.groups.size()) {
            continue;
        }
        const auto & entry = reverse->second;
        const auto & group = impl_->table.groups[entry.group_index];
        const auto & bank = group.banks[entry.bank_index];
        auto & observation = observations[entry.group_index];
        const ggml_tensor * ids = node->src[2];
        if (!observation.eligible || !moe_candidate_record_matches(bank, node->src[0]) || !moe_candidate_ids_valid(ids) ||
                ids->ne[1] != 1 || ids->ne[2] != 1 || ids->ne[3] != 1 || node->ne[2] != 1 || node->ne[3] != 1) {
            observation.invalid = true;
            continue;
        }
        const uint32_t role = bank.info.role;
        const uint32_t role_bit = 1u << role;
        const uint32_t role_slot = role - GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT;
        const auto ids_signature = moe_candidate_ids_signature(ids);
        if ((observation.seen_roles & role_bit) != 0 ||
                (observation.has_ids && !moe_candidate_ids_equal(observation.ids, ids_signature))) {
            observation.invalid = true;
            continue;
        }
        if (!observation.has_ids) {
            observation.ids = ids_signature;
            observation.has_ids = true;
        }
        observation.seen_roles |= role_bit;
        observation.nodes[role_slot] = node;
        observation.node_indices[role_slot] = node_index;
    }

    for (uint32_t group_index = 0; group_index < impl_->table.groups.size(); ++group_index) {
        const auto & group = impl_->table.groups[group_index];
        const auto & observation = observations[group_index];
        if (!observation.eligible || observation.invalid || !observation.has_ids || observation.seen_roles != observation.required_roles) {
            continue;
        }
        if (plan->n_groups_ == GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS) {
            plan->reset();
            execution->reset();
            return;
        }
        const uint32_t record_index = plan->n_groups_++;
        auto & record = plan->groups_[record_index];
        record = {};
        record.candidate.generation = impl_->state.generation;
        record.candidate.group_index = group_index;
        record.layout = group.layout;
        record.n_banks = observation.n_banks;
        auto & key = execution->groups_[record_index];
        key.candidate = record.candidate;
        key.ids = observation.ids;
        key.layout = record.layout;
        key.n_banks = record.n_banks;
        for (uint32_t role = GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT;
                role <= GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT; ++role) {
            if ((observation.required_roles & (1u << role)) == 0) {
                continue;
            }
            const uint32_t role_slot = role - GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT;
            record.nodes[role_slot] = observation.nodes[role_slot];
            record.node_indices[role_slot] = observation.node_indices[role_slot];
            record.bank_indices[role_slot] = observation.bank_indices[role_slot];
            if (!plan->insert(record.nodes[role_slot], record_index, role, record.bank_indices[role_slot])) {
                plan->reset();
                execution->reset();
                return;
            }
        }
    }
    execution->plan_ = plan;
    execution->n_groups_ = plan->n_groups_;
}

bool ggml_cuda_moe_grouped_context::bind_graph_plan(
        const ggml_cgraph * cgraph,
        uint64_t graph_uid,
        bool node_properties_unchanged,
        const ggml_cuda_moe_graph_plan & plan,
        ggml_cuda_moe_graph_execution * execution) const {
    if (execution != nullptr) {
        execution->reset();
    }
    if (cgraph == nullptr || execution == nullptr || !node_properties_unchanged || graph_uid == 0) {
        return false;
    }

    std::lock_guard<std::mutex> lock(impl_->mutex);
    const int n_nodes = ggml_graph_n_nodes(const_cast<ggml_cgraph *>(cgraph));
    if (!plan.initialized_ || plan.owner_ != impl_.get() || plan.graph_uid_ != graph_uid || plan.graph_node_count_ != n_nodes ||
            impl_->draining || impl_->replacement_pending || !impl_->state.accepted || impl_->state.n_slots == 0 ||
            plan.registry_generation_ != impl_->state.generation || plan.n_groups_ > impl_->table.groups.size()) {
        return false;
    }

    for (uint32_t record_index = 0; record_index < plan.n_groups_; ++record_index) {
        const auto & record = plan.groups_[record_index];
        if (record.candidate.generation != impl_->state.generation || record.candidate.group_index >= impl_->table.groups.size()) {
            return false;
        }
        const auto & group = impl_->table.groups[record.candidate.group_index];
        if (group.layout != record.layout || group.banks.size() != record.n_banks) {
            return false;
        }
        ggml_cuda_moe_ids_signature ids_signature;
        bool has_ids = false;
        uint32_t seen_roles = 0;
        for (uint32_t role = GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT;
                role <= GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT; ++role) {
            const bool required = record.layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP ?
                (role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT || role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT) :
                (role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT || role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT ||
                    role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT);
            if (!required) {
                continue;
            }
            const uint32_t role_slot = role - GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT;
            const uint32_t node_index = record.node_indices[role_slot];
            const uint32_t bank_index = record.bank_indices[role_slot];
            if (node_index >= static_cast<uint32_t>(n_nodes) || bank_index >= group.banks.size()) {
                return false;
            }
            const ggml_tensor * node = ggml_graph_node(const_cast<ggml_cgraph *>(cgraph), node_index);
            const auto & bank = group.banks[bank_index];
            const ggml_tensor * ids = node != nullptr ? node->src[2] : nullptr;
            if (node == nullptr || node != record.nodes[role_slot] || node->op != GGML_OP_MUL_MAT_ID ||
                    (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
                    bank.info.role != role || !moe_candidate_record_matches(bank, node->src[0]) || !moe_candidate_ids_valid(ids) ||
                    ids->ne[1] != 1 || ids->ne[2] != 1 || ids->ne[3] != 1 || node->ne[2] != 1 || node->ne[3] != 1) {
                return false;
            }
            const auto current_ids = moe_candidate_ids_signature(ids);
            if (has_ids && !moe_candidate_ids_equal(ids_signature, current_ids)) {
                return false;
            }
            if (!has_ids) {
                ids_signature = current_ids;
                has_ids = true;
            }
            seen_roles |= 1u << role;
        }
        const uint32_t expected_roles = record.layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP ?
            (1u << GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT) | (1u << GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT) :
            (1u << GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT) | (1u << GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT) |
                (1u << GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT);
        if (!has_ids || seen_roles != expected_roles) {
            return false;
        }
        auto & key = execution->groups_[record_index];
        key.candidate = record.candidate;
        key.ids = ids_signature;
        key.layout = record.layout;
        key.n_banks = record.n_banks;
    }
    execution->plan_ = &plan;
    execution->n_groups_ = plan.n_groups_;
    return true;
}

void ggml_cuda_moe_grouped_context::shutdown() {
    std::lock_guard<std::mutex> lifecycle_lock(impl_->resource_lifecycle_mutex);
    impl::resource_slots retired;
    {
        std::unique_lock<std::mutex> lock(impl_->mutex);
        if (impl_->draining) {
            return;
        }
        impl_->draining = true;
        impl_->resource_cv.wait(lock, [&]() { return !impl_->has_active_transaction(); });
        retired = impl_->detach_resources();
        impl_->table = {};
        const uint64_t generation = impl_->state.generation;
        impl_->state = {};
        impl_->state.generation = generation;
    }
    impl::retire_resources(std::move(retired));
}

extern "C"
int32_t ggml_backend_cuda_moe_candidate_replace_v1(
        ggml_backend_t backend,
        const ggml_backend_moe_candidate_snapshot_v1 * snapshot) {
    if (!ggml_backend_is_cuda(backend) || backend->context == nullptr || backend->device == nullptr) {
        return GGML_BACKEND_MOE_CANDIDATE_REPLACE_INVALID_ARGUMENT;
    }

    auto * ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    try {
        std::call_once(ctx->moe_grouped_context_once, [&]() {
            ctx->moe_grouped_context = new ggml_cuda_moe_grouped_context(backend->device);
        });
    } catch (...) {
        return GGML_BACKEND_MOE_CANDIDATE_REPLACE_ERROR;
    }
    return ctx->moe_grouped_context->replace(snapshot);
}

struct ggml_cuda_moe_cache {
    int      device;
    size_t   slot_size_bytes;
    int      n_slots;

    void *   slot_pool_d;            // device alloc, n_slots * slot_size_bytes

    // Dedicated copy stream for cache fills, prefetches, and staging copies.
    cudaStream_t copy_stream;
    cudaEvent_t  compute_done;
    cudaEvent_t  stage_done;
    bool         stream_mem_ops_supported;

    // Per-slot state.
    std::vector<const void *> slot_to_host;  // [n_slots], nullptr if empty
    std::vector<uint64_t>     last_used;     // [n_slots]
    std::vector<char>         slot_prefetched;
    std::vector<uint64_t>     slot_hit_count;
    std::vector<uint64_t>     slot_fill_access;
    std::vector<uint32_t>     slot_pin_count;

    // host_ptr -> slot_id, O(1) lookup.
    std::unordered_map<const void *, int> host_to_slot;

    uint64_t access_counter;
    std::mutex mu;

    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> misses{0};
    std::atomic<uint64_t> evictions{0};

    std::atomic<uint64_t> h2d_copy_count{0};
    std::atomic<uint64_t> h2d_copy_bytes{0};
    std::atomic<uint64_t> h2d_enqueue_time_us{0};
    std::atomic<uint64_t> phase_hits[2];
    std::atomic<uint64_t> phase_misses[2];
    std::atomic<uint64_t> phase_evictions[2];
    std::atomic<uint64_t> phase_h2d_copy_count[2];
    std::atomic<uint64_t> phase_h2d_copy_bytes[2];
    std::atomic<uint64_t> phase_h2d_enqueue_time_us[2];
    std::atomic<uint64_t> phase_prefetch_hits[2];
    std::atomic<uint64_t> phase_prefetch_misses[2];
    std::atomic<uint64_t> phase_prefetch_used[2];
    std::atomic<uint64_t> phase_prefetch_evictions[2];
    std::atomic<uint64_t> phase_demand_evictions[2];
    std::atomic<uint64_t> phase_evicted_prefetched[2];
    std::atomic<uint64_t> phase_evicted_hit_count_le1[2];
    std::atomic<uint64_t> phase_evicted_hit_count_ge2[2];
    std::atomic<uint64_t> phase_evicted_age_le_l1[2];
    std::atomic<uint64_t> phase_evicted_age_gt_l1[2];
    std::atomic<uint64_t> phase_prefetch_h2d_copy_count[2];
    std::atomic<uint64_t> phase_prefetch_h2d_copy_bytes[2];
    std::atomic<uint64_t> phase_prefetch_h2d_enqueue_time_us[2];
    std::atomic<uint64_t> sampled_mincore_checks{0};
    std::atomic<uint64_t> sampled_pages_total{0};
    std::atomic<uint64_t> sampled_pages_resident{0};
    std::atomic<uint64_t> sampled_nonresident_expert_count{0};
    std::atomic<uint64_t> mincore_failures{0};

    bool source_is_mmap = false;
    bool l2_alloc_failed = false;
    int  l2_target_slots = 0;
    size_t l2_budget_bytes = 0;
    moe_cache_l2 l2;

    std::string tensor_name;
    const void * tensor_data = nullptr;
    int64_t n_experts = 0;
    uint64_t expert_access_counter = 0;
    std::vector<uint64_t> expert_access_counts;
    std::vector<uint64_t> expert_last_access;
    uint64_t expert_first_touches = 0;
    uint64_t expert_reuse_le_l1 = 0;
    uint64_t expert_reuse_le_l2 = 0;
    uint64_t expert_reuse_gt_l2 = 0;
    uint64_t phase_expert_access_counter[2] = {};
    std::vector<uint64_t> phase_expert_access_counts[2];
    std::vector<uint64_t> phase_expert_last_access[2];
    uint64_t phase_expert_first_touches[2] = {};
    uint64_t phase_expert_reuse_le_l1[2] = {};
    uint64_t phase_expert_reuse_gt_l1[2] = {};
    moe_cache_reuse_hist phase_expert_reuse_hist[2];
};

static void ggml_cuda_moe_cache_append_expert_counts(
        const struct ggml_cuda_moe_cache * cache,
        std::vector<uint64_t> & counts) {
    if (!cache || cache->expert_access_counts.empty()) {
        return;
    }

    counts.insert(counts.end(), cache->expert_access_counts.begin(), cache->expert_access_counts.end());
}

extern "C"
struct ggml_cuda_moe_cache * ggml_cuda_moe_cache_init(
    int    device,
    size_t slot_size_bytes,
    int    n_slots,
    bool   source_is_mmap,
    size_t l2_budget_bytes,
    int    l2_target_slots) {

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
    c->compute_done    = nullptr;
    c->stage_done      = nullptr;
    c->stream_mem_ops_supported = false;
    c->access_counter  = 0;
    c->source_is_mmap  = source_is_mmap;
    c->l2_budget_bytes = l2_budget_bytes;
    c->l2_target_slots = l2_target_slots;

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

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && !defined(GGML_CUDA_NO_VMM)
    bool can_use_stream_mem_ops = true;
#if CUDA_VERSION < 13000
    CUdevice cu_device;
    int stream_mem_ops_attribute = 0;
    can_use_stream_mem_ops =
        cuDeviceGet(&cu_device, device) == CUDA_SUCCESS &&
        cuDeviceGetAttribute(
            &stream_mem_ops_attribute, CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS, cu_device) == CUDA_SUCCESS &&
        stream_mem_ops_attribute != 0;
#endif
    if (can_use_stream_mem_ops &&
        cuStreamWriteValue32(
            c->copy_stream, (CUdeviceptr)c->slot_pool_d, 0, CU_STREAM_WRITE_VALUE_DEFAULT) == CUDA_SUCCESS) {
        err = cudaStreamSynchronize(c->copy_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "moe-cache: stream memory operation probe failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            (void)cudaStreamDestroy(c->copy_stream);
            (void)cudaFree(c->slot_pool_d);
            delete c;
            (void)cudaSetDevice(prev_device);
            return nullptr;
        }
        c->stream_mem_ops_supported = true;
    }
#endif

    err = cudaEventCreateWithFlags(&c->compute_done, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        fprintf(stderr, "moe-cache: cudaEventCreate failed: %s\n", cudaGetErrorString(err));
        cudaStreamDestroy(c->copy_stream);
        cudaFree(c->slot_pool_d);
        delete c;
        cudaSetDevice(prev_device);
        return nullptr;
    }

    err = cudaEventCreateWithFlags(&c->stage_done, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        fprintf(stderr, "moe-cache: cudaEventCreate failed: %s\n", cudaGetErrorString(err));
        cudaEventDestroy(c->compute_done);
        cudaStreamDestroy(c->copy_stream);
        cudaFree(c->slot_pool_d);
        delete c;
        cudaSetDevice(prev_device);
        return nullptr;
    }

    c->slot_to_host.assign(n_slots, nullptr);
    c->last_used  .assign(n_slots, 0);
    c->slot_prefetched.assign(n_slots, 0);
    c->slot_hit_count.assign(n_slots, 0);
    c->slot_fill_access.assign(n_slots, 0);
    c->slot_pin_count.assign(n_slots, 0);
    c->host_to_slot.reserve(n_slots * 2);
    for (int phase = 0; phase < 2; ++phase) {
        c->phase_hits[phase].store(0, std::memory_order_relaxed);
        c->phase_misses[phase].store(0, std::memory_order_relaxed);
        c->phase_evictions[phase].store(0, std::memory_order_relaxed);
        c->phase_h2d_copy_count[phase].store(0, std::memory_order_relaxed);
        c->phase_h2d_copy_bytes[phase].store(0, std::memory_order_relaxed);
        c->phase_h2d_enqueue_time_us[phase].store(0, std::memory_order_relaxed);
        c->phase_prefetch_hits[phase].store(0, std::memory_order_relaxed);
        c->phase_prefetch_misses[phase].store(0, std::memory_order_relaxed);
        c->phase_prefetch_used[phase].store(0, std::memory_order_relaxed);
        c->phase_prefetch_evictions[phase].store(0, std::memory_order_relaxed);
        c->phase_demand_evictions[phase].store(0, std::memory_order_relaxed);
        c->phase_evicted_prefetched[phase].store(0, std::memory_order_relaxed);
        c->phase_evicted_hit_count_le1[phase].store(0, std::memory_order_relaxed);
        c->phase_evicted_hit_count_ge2[phase].store(0, std::memory_order_relaxed);
        c->phase_evicted_age_le_l1[phase].store(0, std::memory_order_relaxed);
        c->phase_evicted_age_gt_l1[phase].store(0, std::memory_order_relaxed);
        c->phase_prefetch_h2d_copy_count[phase].store(0, std::memory_order_relaxed);
        c->phase_prefetch_h2d_copy_bytes[phase].store(0, std::memory_order_relaxed);
        c->phase_prefetch_h2d_enqueue_time_us[phase].store(0, std::memory_order_relaxed);
    }

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
        cudaStreamSynchronize(cache->copy_stream);
        cudaStreamDestroy(cache->copy_stream);
    }
    if (cache->compute_done) {
        cudaEventDestroy(cache->compute_done);
    }
    if (cache->stage_done) {
        cudaEventDestroy(cache->stage_done);
    }
    if (cache->slot_pool_d) {
        cudaFree(cache->slot_pool_d);
    }
    moe_cache_l2_free(cache->l2);
    cudaSetDevice(prev_device);

    delete cache;
}

static const void * ggml_cuda_moe_cache_l2_source(
        struct ggml_cuda_moe_cache * cache,
        const void * host_src,
        size_t byte_count,
        bool is_decode,
        cudaStream_t copy_stream) {
    if (!cache->source_is_mmap || cache->l2_target_slots <= 0 || cache->l2_budget_bytes == 0 || cache->l2_alloc_failed) {
        return host_src;
    }

    if (cache->l2.slot_pool_h == nullptr) {
        if (!moe_cache_l2_init(cache->l2, cache->slot_size_bytes, cache->l2_target_slots)) {
            cache->l2_alloc_failed = true;
            return host_src;
        }
    }

    const void * l2_src = moe_cache_l2_acquire(cache->l2, host_src, byte_count, is_decode, copy_stream);
    return l2_src ? l2_src : host_src;
}

static int64_t ggml_cuda_moe_cache_expert_id(const ggml_cuda_moe_cache * cache, const void * host_src) {
    if (!cache || !cache->tensor_data || !host_src || cache->slot_size_bytes == 0 || cache->n_experts <= 0) {
        return -1;
    }

    const char * base = (const char *) cache->tensor_data;
    const char * ptr  = (const char *) host_src;
    if (ptr < base) {
        return -1;
    }

    const size_t offset = (size_t) (ptr - base);
    if ((offset % cache->slot_size_bytes) != 0) {
        return -1;
    }

    const int64_t eid = (int64_t) (offset / cache->slot_size_bytes);
    return eid >= 0 && eid < cache->n_experts ? eid : -1;
}

static void ggml_cuda_moe_cache_record_expert_access(ggml_cuda_moe_cache * cache, const void * host_src, int phase) {
    if (!moe_cache_mm_debug_enabled()) {
        return;
    }

    const int64_t eid = ggml_cuda_moe_cache_expert_id(cache, host_src);
    if (eid < 0) {
        return;
    }

    if (cache->expert_access_counts.empty()) {
        cache->expert_access_counts.assign((size_t) cache->n_experts, 0);
        cache->expert_last_access.assign((size_t) cache->n_experts, 0);
    }
    if (phase >= 0 && phase < 2 && cache->phase_expert_access_counts[phase].empty()) {
        cache->phase_expert_access_counts[phase].assign((size_t) cache->n_experts, 0);
        cache->phase_expert_last_access[phase].assign((size_t) cache->n_experts, 0);
    }

    const uint64_t access = ++cache->expert_access_counter;
    const size_t idx = (size_t) eid;
    cache->expert_access_counts[idx]++;
    const uint64_t last_access = cache->expert_last_access[idx];
    cache->expert_last_access[idx] = access;

    if (last_access == 0) {
        cache->expert_first_touches++;
    } else {
        const uint64_t distance = access - last_access;
        if (distance <= (uint64_t) cache->n_slots) {
            cache->expert_reuse_le_l1++;
        } else if (cache->l2_target_slots > cache->n_slots && distance <= (uint64_t) cache->l2_target_slots) {
            cache->expert_reuse_le_l2++;
        } else {
            cache->expert_reuse_gt_l2++;
        }
    }

    if (phase < 0 || phase >= 2) {
        return;
    }

    const uint64_t phase_access = ++cache->phase_expert_access_counter[phase];
    cache->phase_expert_access_counts[phase][idx]++;
    const uint64_t phase_last_access = cache->phase_expert_last_access[phase][idx];
    cache->phase_expert_last_access[phase][idx] = phase_access;

    moe_cache_reuse_hist & hist = cache->phase_expert_reuse_hist[phase];
    if (phase_last_access == 0) {
        cache->phase_expert_first_touches[phase]++;
        hist.first_touch++;
        return;
    }

    const uint64_t phase_distance = phase_access - phase_last_access;
    if (phase_distance <= (uint64_t) cache->n_slots) {
        cache->phase_expert_reuse_le_l1[phase]++;
    } else {
        cache->phase_expert_reuse_gt_l1[phase]++;
    }

    if (phase_distance <= 8) {
        hist.le8++;
    } else if (phase_distance <= 16) {
        hist.le16++;
    } else if (phase_distance <= 32) {
        hist.le32++;
    } else if (phase_distance <= (uint64_t) cache->n_slots) {
        hist.le_l1++;
    } else if (phase_distance <= 2ull * (uint64_t) cache->n_slots) {
        hist.le_2xl1++;
    } else {
        hist.gt_2xl1++;
    }
}

static int ggml_cuda_moe_cache_acquire_locked(
    struct ggml_cuda_moe_cache * cache,
    const void * host_src,
    size_t       byte_count,
    cudaStream_t copy_stream,
    bool         use_l2,
    bool         is_decode,
    bool         is_prefetch,
    bool         wait_for_compute) {

    if (byte_count > cache->slot_size_bytes) {
        // Caller forgot to grow first. Bail rather than clobber the next slot.
        return -1;
    }

    const int phase = moe_cache_phase_index(is_decode);
    ggml_cuda_moe_cache_record_expert_access(cache, host_src, phase);

    // Hit path: O(1) hash lookup.
    auto it = cache->host_to_slot.find(host_src);
    if (it != cache->host_to_slot.end()) {
        int slot = it->second;
        cache->last_used[slot] = ++cache->access_counter;
        cache->slot_hit_count[slot]++;
        cache->hits.fetch_add(1, std::memory_order_relaxed);
        cache->phase_hits[phase].fetch_add(1, std::memory_order_relaxed);
        if (is_prefetch) {
            cache->phase_prefetch_hits[phase].fetch_add(1, std::memory_order_relaxed);
        } else if (cache->slot_prefetched[slot]) {
            cache->phase_prefetch_used[phase].fetch_add(1, std::memory_order_relaxed);
            cache->slot_prefetched[slot] = 0;
        }
        return slot;
    }

    // Miss: pick the LRU slot. Empty slots (last_used==0) win automatically.
    int      lru_slot = -1;
    uint64_t lru_t    = std::numeric_limits<uint64_t>::max();
    for (int i = 0; i < cache->n_slots; ++i) {
        if (cache->slot_pin_count[i] != 0) {
            continue;
        }
        if (cache->last_used[i] < lru_t) {
            lru_t    = cache->last_used[i];
            lru_slot = i;
        }
    }
    if (lru_slot < 0) {
        return -1;
    }

    const void * evicted = cache->slot_to_host[lru_slot];
    if (evicted != nullptr) {
        cache->host_to_slot.erase(evicted);
        cache->evictions.fetch_add(1, std::memory_order_relaxed);
        cache->phase_evictions[phase].fetch_add(1, std::memory_order_relaxed);
        if (is_prefetch) {
            cache->phase_prefetch_evictions[phase].fetch_add(1, std::memory_order_relaxed);
        } else {
            cache->phase_demand_evictions[phase].fetch_add(1, std::memory_order_relaxed);
        }
        if (cache->slot_prefetched[lru_slot]) {
            cache->phase_evicted_prefetched[phase].fetch_add(1, std::memory_order_relaxed);
        }
        const uint64_t hit_count = cache->slot_hit_count[lru_slot];
        if (hit_count <= 1) {
            cache->phase_evicted_hit_count_le1[phase].fetch_add(1, std::memory_order_relaxed);
        } else {
            cache->phase_evicted_hit_count_ge2[phase].fetch_add(1, std::memory_order_relaxed);
        }
        const uint64_t age = cache->access_counter >= cache->slot_fill_access[lru_slot] ?
            cache->access_counter - cache->slot_fill_access[lru_slot] : 0;
        if (age <= (uint64_t) cache->n_slots) {
            cache->phase_evicted_age_le_l1[phase].fetch_add(1, std::memory_order_relaxed);
        } else {
            cache->phase_evicted_age_gt_l1[phase].fetch_add(1, std::memory_order_relaxed);
        }
    }

    cache->slot_to_host[lru_slot] = host_src;
    cache->host_to_slot[host_src] = lru_slot;
    cache->last_used[lru_slot]    = ++cache->access_counter;
    cache->slot_prefetched[lru_slot] = is_prefetch ? 1 : 0;
    cache->slot_hit_count[lru_slot] = 0;
    cache->slot_fill_access[lru_slot] = cache->access_counter;
    cache->misses.fetch_add(1, std::memory_order_relaxed);
    cache->phase_misses[phase].fetch_add(1, std::memory_order_relaxed);
    if (is_prefetch) {
        cache->phase_prefetch_misses[phase].fetch_add(1, std::memory_order_relaxed);
    }
    auto rollback_miss = [&]() {
        cache->slot_to_host[lru_slot] = nullptr;
        cache->host_to_slot.erase(host_src);
        cache->slot_prefetched[lru_slot] = 0;
    };

    const bool debug_mm = moe_cache_mm_debug_enabled();
    int64_t enqueue_start_us = 0;
    void * dst = (char *)cache->slot_pool_d + (size_t)lru_slot * cache->slot_size_bytes;
    const void * copy_src = use_l2 ? ggml_cuda_moe_cache_l2_source(cache, host_src, byte_count, is_decode, copy_stream) : host_src;
    if (debug_mm && copy_src == host_src) {
        const uint64_t miss_index = g_moe_cache_mm_miss_counter.fetch_add(1, std::memory_order_relaxed) + 1;
        if ((miss_index % MOE_CACHE_MM_SAMPLE_RATE) == 0) {
            moe_cache_mm_sample_mincore(
                host_src,
                byte_count,
                cache->sampled_mincore_checks,
                cache->sampled_pages_total,
                cache->sampled_pages_resident,
                cache->sampled_nonresident_expert_count,
                cache->mincore_failures);
        }
    }
    if (debug_mm) {
        enqueue_start_us = ggml_time_us();
    }
    cudaError_t err = cudaSuccess;
    if (wait_for_compute) {
        err = cudaStreamWaitEvent(copy_stream, cache->compute_done, 0);
        if (err != cudaSuccess) {
            fprintf(stderr, "moe-cache: cudaStreamWaitEvent failed: %s\n", cudaGetErrorString(err));
            rollback_miss();
            return -1;
        }
    }
    err = cudaMemcpyAsync(
        dst, copy_src, byte_count,
        cudaMemcpyHostToDevice, copy_stream);
    if (debug_mm) {
        cache->h2d_copy_count.fetch_add(1, std::memory_order_relaxed);
        cache->h2d_copy_bytes.fetch_add(byte_count, std::memory_order_relaxed);
        const uint64_t h2d_enqueue_time_us = (uint64_t) (ggml_time_us() - enqueue_start_us);
        cache->h2d_enqueue_time_us.fetch_add(h2d_enqueue_time_us, std::memory_order_relaxed);
        cache->phase_h2d_copy_count[phase].fetch_add(1, std::memory_order_relaxed);
        cache->phase_h2d_copy_bytes[phase].fetch_add(byte_count, std::memory_order_relaxed);
        cache->phase_h2d_enqueue_time_us[phase].fetch_add(h2d_enqueue_time_us, std::memory_order_relaxed);
        if (is_prefetch) {
            cache->phase_prefetch_h2d_copy_count[phase].fetch_add(1, std::memory_order_relaxed);
            cache->phase_prefetch_h2d_copy_bytes[phase].fetch_add(byte_count, std::memory_order_relaxed);
            cache->phase_prefetch_h2d_enqueue_time_us[phase].fetch_add(h2d_enqueue_time_us, std::memory_order_relaxed);
        }
    }
    if (err != cudaSuccess) {
        fprintf(stderr, "moe-cache: cudaMemcpyAsync failed: %s\n",
                cudaGetErrorString(err));
        // Roll back so the next acquire retries cleanly.
        rollback_miss();
        return -1;
    }

    return lru_slot;
}

extern "C"
int ggml_cuda_moe_cache_acquire(
    struct ggml_cuda_moe_cache * cache,
    const void * host_src,
    size_t       byte_count,
    cudaStream_t copy_stream,
    bool         use_l2,
    bool         is_decode,
    bool         is_prefetch,
    bool         pin) {

    if (!cache || host_src == nullptr || byte_count == 0) {
        return -1;
    }

    std::lock_guard<std::mutex> lk(cache->mu);
    const int slot = ggml_cuda_moe_cache_acquire_locked(
        cache, host_src, byte_count, copy_stream, use_l2, is_decode, is_prefetch, true);
    if (slot >= 0 && pin) {
        cache->slot_pin_count[slot]++;
    }
    return slot;
}

extern "C"
void ggml_cuda_moe_cache_release_slots(
    struct ggml_cuda_moe_cache * cache,
    const int * slot_ids,
    int n_slot_ids) {
    if (!cache || !slot_ids || n_slot_ids <= 0) {
        return;
    }

    std::lock_guard<std::mutex> lk(cache->mu);
    for (int i = 0; i < n_slot_ids; ++i) {
        const int slot = slot_ids[i];
        GGML_ASSERT(slot >= 0 && slot < cache->n_slots);
        GGML_ASSERT(cache->slot_pin_count[slot] > 0);
        cache->slot_pin_count[slot]--;
    }
}

extern "C"
bool ggml_cuda_moe_cache_copy_to_staging(
    struct ggml_cuda_moe_cache * cache,
    const void * const * host_srcs,
    int                  n_host_srcs,
    size_t               byte_count,
    void *               dst,
    cudaStream_t         compute_stream) {

    if (!cache || !host_srcs || n_host_srcs <= 0 || byte_count == 0 || !dst || !compute_stream) {
        return false;
    }
    for (int i = 0; i < n_host_srcs; ++i) {
        if (!host_srcs[i]) {
            return false;
        }
    }

    std::lock_guard<std::mutex> lk(cache->mu);
    if (byte_count > cache->slot_size_bytes) {
        return false;
    }

    CUDA_CHECK(cudaEventRecord(cache->compute_done, compute_stream));
    CUDA_CHECK(cudaStreamWaitEvent(cache->copy_stream, cache->compute_done, 0));

    for (int i = 0; i < n_host_srcs;) {
        const auto it = cache->host_to_slot.find(host_srcs[i]);
        const bool resident = it != cache->host_to_slot.end();
        const void * src = resident ?
            (const char *)cache->slot_pool_d + (size_t)it->second * cache->slot_size_bytes : host_srcs[i];
        int run = 1;
        while (i + run < n_host_srcs) {
            const auto next_it = cache->host_to_slot.find(host_srcs[i + run]);
            const bool next_resident = next_it != cache->host_to_slot.end();
            const void * next_src = next_resident ?
                (const char *)cache->slot_pool_d + (size_t)next_it->second * cache->slot_size_bytes : host_srcs[i + run];
            if (next_resident != resident || (uintptr_t)next_src != (uintptr_t)src + (size_t)run * byte_count) {
                break;
            }
            ++run;
        }
        CUDA_CHECK(cudaMemcpyAsync(
            (char *)dst + (size_t)i * byte_count,
            src,
            (size_t)run * byte_count,
            resident ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice,
            cache->copy_stream));
        i += run;
    }

    CUDA_CHECK(cudaEventRecord(cache->stage_done, cache->copy_stream));
    CUDA_CHECK(cudaStreamWaitEvent(compute_stream, cache->stage_done, 0));
    return true;
}

extern "C"
bool ggml_cuda_moe_cache_prepare_split_staging(
    struct ggml_cuda_moe_cache * cache,
    const void * const * host_srcs,
    int                  n_host_srcs,
    size_t               byte_count,
    int                  min_resident,
    int *                slot_ids,
    int32_t *            source_wait_class,
    int *                out_n_resident,
    void *               miss_dst,
    uint32_t *           stage_ready,
    int                  stage_ready_capacity,
    int *                out_n_wait_classes,
    cudaStream_t         compute_stream) {

    if (!cache || !host_srcs || n_host_srcs <= 0 || byte_count == 0 || min_resident <= 0 ||
        !slot_ids || !out_n_resident || !miss_dst || !out_n_wait_classes || !compute_stream ||
        ((source_wait_class == nullptr) != (stage_ready == nullptr)) ||
        (stage_ready == nullptr ? stage_ready_capacity != 0 : stage_ready_capacity < 2)) {
        return false;
    }
    for (int i = 0; i < n_host_srcs; ++i) {
        if (!host_srcs[i]) {
            return false;
        }
    }

    std::lock_guard<std::mutex> lk(cache->mu);
    if (byte_count > cache->slot_size_bytes || n_host_srcs <= cache->n_slots) {
        return false;
    }

    const bool overlap = stage_ready != nullptr && cache->stream_mem_ops_supported;
    if (stage_ready != nullptr && !overlap) {
        return false;
    }

    int n_resident = 0;
    for (int i = 0; i < n_host_srcs; ++i) {
        const auto it = cache->host_to_slot.find(host_srcs[i]);
        slot_ids[i] = it == cache->host_to_slot.end() ? -1 : it->second;
        n_resident += slot_ids[i] >= 0;
        if (source_wait_class != nullptr) {
            source_wait_class[i] = slot_ids[i] >= 0 ? 0 : -1;
        }
    }
    if (n_resident == n_host_srcs) {
        return false;
    }

    if (overlap) {
        CUDA_CHECK(cudaMemsetAsync(
            stage_ready, 0, (size_t)stage_ready_capacity * sizeof(uint32_t), compute_stream));
    }
    CUDA_CHECK(cudaEventRecord(cache->compute_done, compute_stream));
    CUDA_CHECK(cudaStreamWaitEvent(cache->copy_stream, cache->compute_done, 0));

    for (int i = 0; i < n_host_srcs; ++i) {
        if (slot_ids[i] >= 0) {
            cache->slot_pin_count[slot_ids[i]]++;
        }
    }

    for (int i = 0; i < n_host_srcs && n_resident < cache->n_slots; ++i) {
        if (slot_ids[i] >= 0) {
            continue;
        }

        const int slot = ggml_cuda_moe_cache_acquire_locked(
            cache, host_srcs[i], byte_count, cache->copy_stream, false, false, false, false);
        if (slot < 0) {
            break;
        }
        slot_ids[i] = slot;
        if (source_wait_class != nullptr) {
            source_wait_class[i] = 1;
        }
        cache->slot_pin_count[slot]++;
        n_resident++;
    }

    if (n_resident < min_resident) {
        for (int i = 0; i < n_host_srcs; ++i) {
            if (slot_ids[i] >= 0) {
                cache->slot_pin_count[slot_ids[i]]--;
            }
        }
        return false;
    }

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && !defined(GGML_CUDA_NO_VMM)
    if (overlap) {
        CU_CHECK(cuStreamWriteValue32(
            cache->copy_stream, (CUdeviceptr)(stage_ready + 0), 1, CU_STREAM_WRITE_VALUE_DEFAULT));
    }
#endif

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 12080
    std::vector<void *> batch_dsts;
    std::vector<const void *> batch_srcs;
    std::vector<size_t> batch_sizes;
    cudaMemcpyAttributes attributes = {};
    attributes.srcAccessOrder = cudaMemcpySrcAccessOrderAny;
    attributes.flags = overlap ? cudaMemcpyFlagPreferOverlapWithCompute : cudaMemcpyFlagDefault;
    size_t attributes_index = 0;
#endif
    int n_misses = n_host_srcs - n_resident;
    int wave_size = n_misses;
    if (overlap) {
        const int max_staging_waves = stage_ready_capacity - 1;
        wave_size = std::max(cache->n_slots, (n_misses + max_staging_waves - 1) / max_staging_waves);
    }

    int miss = 0;
    int wave_count = 0;
    int wait_class = 2;
    for (int i = 0; i < n_host_srcs;) {
        if (slot_ids[i] >= 0) {
            ++i;
            continue;
        }

        const void * src = host_srcs[i];
        int run = 1;
        while (i + run < n_host_srcs && slot_ids[i + run] < 0 &&
               (uintptr_t)host_srcs[i + run] == (uintptr_t)src + (size_t)run * byte_count) {
            ++run;
        }
        run = std::min(run, wave_size - wave_count);
        if (source_wait_class != nullptr) {
            std::fill_n(source_wait_class + i, run, wait_class);
        }
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 12080
        batch_dsts.push_back((char *)miss_dst + (size_t)miss * byte_count);
        batch_srcs.push_back(src);
        batch_sizes.push_back((size_t)run * byte_count);
#else
        CUDA_CHECK(cudaMemcpyAsync(
            (char *)miss_dst + (size_t)miss * byte_count,
            src,
            (size_t)run * byte_count,
            cudaMemcpyHostToDevice,
            cache->copy_stream));
#endif
        miss += run;
        wave_count += run;
        i += run;

        if (wave_count < wave_size && miss < n_misses) {
            continue;
        }

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 12080
        CUDA_CHECK(cudaMemcpyBatchAsync(
            batch_dsts.data(), batch_srcs.data(), batch_sizes.data(), batch_srcs.size(),
            &attributes, &attributes_index, 1, cache->copy_stream));
        batch_dsts.clear();
        batch_srcs.clear();
        batch_sizes.clear();
#endif
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && !defined(GGML_CUDA_NO_VMM)
        if (overlap) {
            CU_CHECK(cuStreamWriteValue32(
                cache->copy_stream, (CUdeviceptr)(stage_ready + wait_class - 1), 1, CU_STREAM_WRITE_VALUE_DEFAULT));
        }
#endif
        wave_count = 0;
        ++wait_class;
    }

    CUDA_CHECK(cudaEventRecord(cache->stage_done, cache->copy_stream));
    if (!overlap) {
        CUDA_CHECK(cudaStreamWaitEvent(compute_stream, cache->stage_done, 0));
    }
    *out_n_resident = n_resident;
    *out_n_wait_classes = overlap ? wait_class : 1;
    return true;
}

extern "C"
bool ggml_cuda_moe_cache_can_overlap_staging(
    const struct ggml_cuda_moe_cache * cache) {

    return cache != nullptr && cache->stream_mem_ops_supported;
}

extern "C"
bool ggml_cuda_moe_cache_finish_split_staging(
    struct ggml_cuda_moe_cache * cache,
    cudaStream_t         compute_stream) {

    if (!cache || !compute_stream) {
        return false;
    }

    std::lock_guard<std::mutex> lk(cache->mu);
    CUDA_CHECK(cudaStreamWaitEvent(compute_stream, cache->stage_done, 0));
    return true;
}

extern "C"
bool ggml_cuda_moe_cache_release_split_slots(
    struct ggml_cuda_moe_cache * cache,
    const int *          slot_ids,
    int                  n_slot_ids,
    cudaStream_t         compute_stream) {

    if (!cache || !slot_ids || n_slot_ids <= 0 || !compute_stream) {
        return false;
    }

    std::lock_guard<std::mutex> lk(cache->mu);
    cudaError_t err = cudaEventRecord(cache->compute_done, compute_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "moe-cache: cudaEventRecord failed: %s\n", cudaGetErrorString(err));
        return false;
    }

    for (int i = 0; i < n_slot_ids; ++i) {
        const int slot = slot_ids[i];
        if (slot < 0) {
            continue;
        }
        GGML_ASSERT(slot < cache->n_slots);
        GGML_ASSERT(cache->slot_pin_count[slot] > 0);
        cache->slot_pin_count[slot]--;
    }
    return true;
}

extern "C"
bool ggml_cuda_moe_cache_grow_pool(
    struct ggml_cuda_moe_cache * cache,
    size_t min_slot_size_bytes) {

    if (!cache || min_slot_size_bytes == 0) {
        return false;
    }

    std::lock_guard<std::mutex> lk(cache->mu);

    for (uint32_t pin_count : cache->slot_pin_count) {
        if (pin_count != 0) {
            return false;
        }
    }

    if (min_slot_size_bytes <= cache->slot_size_bytes) {
        return true; // already big enough
    }

    int prev_device = 0;
    cudaGetDevice(&prev_device);
    cudaSetDevice(cache->device);

    cudaError_t err = cudaEventSynchronize(cache->compute_done);
    if (err != cudaSuccess) {
        fprintf(stderr, "moe-cache: grow_pool cudaEventSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaSetDevice(prev_device);
        return false;
    }
    err = cudaStreamSynchronize(cache->copy_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "moe-cache: grow_pool cudaStreamSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaSetDevice(prev_device);
        return false;
    }

    void * new_pool = nullptr;
    err = cudaMalloc(&new_pool, (size_t)cache->n_slots * min_slot_size_bytes);
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
    std::fill(cache->slot_prefetched.begin(), cache->slot_prefetched.end(), 0);
    std::fill(cache->slot_hit_count.begin(), cache->slot_hit_count.end(), 0ull);
    std::fill(cache->slot_pin_count.begin(), cache->slot_pin_count.end(), 0u);
    std::fill(cache->slot_fill_access.begin(), cache->slot_fill_access.end(), 0ull);
    cache->host_to_slot.clear();
    cache->access_counter = 0;
    if (cache->l2.slot_pool_h) {
        moe_cache_l2_free(cache->l2);
    }
    const int old_l2_target_slots = cache->l2_target_slots;
    cache->l2_target_slots = 0;
    if (cache->source_is_mmap && cache->l2_budget_bytes >= min_slot_size_bytes && old_l2_target_slots > 0) {
        cache->l2_target_slots = (int) std::min<size_t>(
            cache->l2_budget_bytes / min_slot_size_bytes, (size_t) old_l2_target_slots);
    }

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
bool ggml_cuda_moe_cache_mark_used(
    struct ggml_cuda_moe_cache * cache,
    cudaStream_t compute_stream) {
    if (!cache || !compute_stream) {
        return false;
    }

    std::lock_guard<std::mutex> lk(cache->mu);
    cudaError_t err = cudaEventRecord(cache->compute_done, compute_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "moe-cache: cudaEventRecord failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
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

static moe_cache_mm_stats ggml_cuda_moe_cache_mm_stats(const struct ggml_cuda_moe_cache * cache) {
    if (!cache) {
        return moe_cache_mm_stats_zero();
    }

    moe_cache_mm_stats s = {};
    s.h2d_copy_count                    = cache->h2d_copy_count.load(std::memory_order_relaxed);
    s.h2d_copy_bytes                    = cache->h2d_copy_bytes.load(std::memory_order_relaxed);
    s.h2d_enqueue_time_us               = cache->h2d_enqueue_time_us.load(std::memory_order_relaxed);
    s.sampled_mincore_checks            = cache->sampled_mincore_checks.load(std::memory_order_relaxed);
    s.sampled_pages_total               = cache->sampled_pages_total.load(std::memory_order_relaxed);
    s.sampled_pages_resident            = cache->sampled_pages_resident.load(std::memory_order_relaxed);
    s.sampled_nonresident_expert_count  = cache->sampled_nonresident_expert_count.load(std::memory_order_relaxed);
    s.mincore_failures                  = cache->mincore_failures.load(std::memory_order_relaxed);
    return s;
}

static moe_cache_l2_stats ggml_cuda_moe_cache_l2_stats(const struct ggml_cuda_moe_cache * cache) {
    if (!cache || !cache->source_is_mmap || cache->l2_target_slots <= 0) {
        return {};
    }

    moe_cache_l2_stats s = {};
    s.budget_bytes = cache->l2_budget_bytes;
    s.slots = (uint64_t) cache->l2_target_slots;
    if (cache->l2.slot_pool_h) {
        s.used_bytes = (uint64_t) cache->l2.n_slots * cache->l2.slot_size_bytes;
        s.hits       = cache->l2.hits.load(std::memory_order_relaxed);
        s.misses     = cache->l2.misses.load(std::memory_order_relaxed);
        s.fills      = cache->l2.fills.load(std::memory_order_relaxed);
        s.evictions  = cache->l2.evictions.load(std::memory_order_relaxed);
        s.fill_bytes = cache->l2.fill_bytes.load(std::memory_order_relaxed);
        s.fill_time_us = cache->l2.fill_time_us.load(std::memory_order_relaxed);
        for (int phase = 0; phase < 2; ++phase) {
            s.phase_hits[phase] = cache->l2.phase_hits[phase].load(std::memory_order_relaxed);
            s.phase_misses[phase] = cache->l2.phase_misses[phase].load(std::memory_order_relaxed);
            s.phase_fills[phase] = cache->l2.phase_fills[phase].load(std::memory_order_relaxed);
            s.phase_evictions[phase] = cache->l2.phase_evictions[phase].load(std::memory_order_relaxed);
            s.phase_fill_bytes[phase] = cache->l2.phase_fill_bytes[phase].load(std::memory_order_relaxed);
            s.phase_fill_time_us[phase] = cache->l2.phase_fill_time_us[phase].load(std::memory_order_relaxed);
        }
    }
    return s;
}

static moe_cache_phase_stats ggml_cuda_moe_cache_phase_stats(const struct ggml_cuda_moe_cache * cache, int phase) {
    moe_cache_phase_stats s = {};
    if (!cache || phase < 0 || phase >= 2) {
        return s;
    }

    s.l1_hits = cache->phase_hits[phase].load(std::memory_order_relaxed);
    s.l1_misses = cache->phase_misses[phase].load(std::memory_order_relaxed);
    s.l1_evictions = cache->phase_evictions[phase].load(std::memory_order_relaxed);
    s.h2d_copy_count = cache->phase_h2d_copy_count[phase].load(std::memory_order_relaxed);
    s.h2d_copy_bytes = cache->phase_h2d_copy_bytes[phase].load(std::memory_order_relaxed);
    s.h2d_enqueue_time_us = cache->phase_h2d_enqueue_time_us[phase].load(std::memory_order_relaxed);
    s.prefetch_hits = cache->phase_prefetch_hits[phase].load(std::memory_order_relaxed);
    s.prefetch_misses = cache->phase_prefetch_misses[phase].load(std::memory_order_relaxed);
    s.prefetch_used = cache->phase_prefetch_used[phase].load(std::memory_order_relaxed);
    s.prefetch_evictions = cache->phase_prefetch_evictions[phase].load(std::memory_order_relaxed);
    s.demand_evictions = cache->phase_demand_evictions[phase].load(std::memory_order_relaxed);
    s.evicted_prefetched = cache->phase_evicted_prefetched[phase].load(std::memory_order_relaxed);
    s.evicted_hit_count_le1 = cache->phase_evicted_hit_count_le1[phase].load(std::memory_order_relaxed);
    s.evicted_hit_count_ge2 = cache->phase_evicted_hit_count_ge2[phase].load(std::memory_order_relaxed);
    s.evicted_age_le_l1 = cache->phase_evicted_age_le_l1[phase].load(std::memory_order_relaxed);
    s.evicted_age_gt_l1 = cache->phase_evicted_age_gt_l1[phase].load(std::memory_order_relaxed);
    s.prefetch_h2d_copy_count = cache->phase_prefetch_h2d_copy_count[phase].load(std::memory_order_relaxed);
    s.prefetch_h2d_copy_bytes = cache->phase_prefetch_h2d_copy_bytes[phase].load(std::memory_order_relaxed);
    s.prefetch_h2d_enqueue_time_us = cache->phase_prefetch_h2d_enqueue_time_us[phase].load(std::memory_order_relaxed);
    if (cache->l2.slot_pool_h) {
        s.l2_hits = cache->l2.phase_hits[phase].load(std::memory_order_relaxed);
        s.l2_misses = cache->l2.phase_misses[phase].load(std::memory_order_relaxed);
        s.l2_fills = cache->l2.phase_fills[phase].load(std::memory_order_relaxed);
        s.l2_evictions = cache->l2.phase_evictions[phase].load(std::memory_order_relaxed);
        s.l2_fill_bytes = cache->l2.phase_fill_bytes[phase].load(std::memory_order_relaxed);
        s.l2_fill_time_us = cache->l2.phase_fill_time_us[phase].load(std::memory_order_relaxed);
    }
    return s;
}

static moe_cache_phase_stats ggml_cuda_moe_op_phase_stats(int phase) {
    moe_cache_phase_stats s = {};
    if (phase < 0 || phase >= 2) {
        return s;
    }

    const moe_cache_op_phase_stats & op = g_moe_cache_op_stats[phase];
    s.ops = op.ops.load(std::memory_order_relaxed);
    s.staged_ops = op.staged_ops.load(std::memory_order_relaxed);
    s.overflow_ops = op.overflow_ops.load(std::memory_order_relaxed);
    s.unique_experts = op.unique_experts.load(std::memory_order_relaxed);
    s.unique_experts_max = op.unique_experts_max.load(std::memory_order_relaxed);
    s.ids_bytes = op.ids_bytes.load(std::memory_order_relaxed);
    s.ids_d2h_time_us = op.ids_d2h_time_us.load(std::memory_order_relaxed);
    s.ids_d2h_sync_count = op.ids_d2h_sync_count.load(std::memory_order_relaxed);
    s.ids_cache_hits = op.ids_cache_hits.load(std::memory_order_relaxed);
    s.acquire_time_us = op.acquire_time_us.load(std::memory_order_relaxed);
    s.remap_time_us = op.remap_time_us.load(std::memory_order_relaxed);
    s.copy_wait_event_count = op.copy_wait_event_count.load(std::memory_order_relaxed);
    s.copy_wait_event_time_us = op.copy_wait_event_time_us.load(std::memory_order_relaxed);
    s.total_time_us = op.total_time_us.load(std::memory_order_relaxed);
    return s;
}

static void ggml_cuda_moe_reset_op_phase_stats(void) {
    for (int phase = 0; phase < 2; ++phase) {
        moe_cache_op_phase_stats & s = g_moe_cache_op_stats[phase];
        s.ops.store(0, std::memory_order_relaxed);
        s.staged_ops.store(0, std::memory_order_relaxed);
        s.overflow_ops.store(0, std::memory_order_relaxed);
        s.unique_experts.store(0, std::memory_order_relaxed);
        s.unique_experts_max.store(0, std::memory_order_relaxed);
        s.ids_bytes.store(0, std::memory_order_relaxed);
        s.ids_d2h_time_us.store(0, std::memory_order_relaxed);
        s.ids_d2h_sync_count.store(0, std::memory_order_relaxed);
        s.ids_cache_hits.store(0, std::memory_order_relaxed);
        s.acquire_time_us.store(0, std::memory_order_relaxed);
        s.remap_time_us.store(0, std::memory_order_relaxed);
        s.copy_wait_event_count.store(0, std::memory_order_relaxed);
        s.copy_wait_event_time_us.store(0, std::memory_order_relaxed);
        s.total_time_us.store(0, std::memory_order_relaxed);
    }
}

static void ggml_cuda_moe_add_phase_stats(moe_cache_phase_stats & dst, const moe_cache_phase_stats & src) {
    dst.l1_hits += src.l1_hits;
    dst.l1_misses += src.l1_misses;
    dst.l1_evictions += src.l1_evictions;
    dst.h2d_copy_count += src.h2d_copy_count;
    dst.h2d_copy_bytes += src.h2d_copy_bytes;
    dst.h2d_enqueue_time_us += src.h2d_enqueue_time_us;
    dst.prefetch_hits += src.prefetch_hits;
    dst.prefetch_misses += src.prefetch_misses;
    dst.prefetch_used += src.prefetch_used;
    dst.prefetch_evictions += src.prefetch_evictions;
    dst.demand_evictions += src.demand_evictions;
    dst.evicted_prefetched += src.evicted_prefetched;
    dst.evicted_hit_count_le1 += src.evicted_hit_count_le1;
    dst.evicted_hit_count_ge2 += src.evicted_hit_count_ge2;
    dst.evicted_age_le_l1 += src.evicted_age_le_l1;
    dst.evicted_age_gt_l1 += src.evicted_age_gt_l1;
    dst.prefetch_h2d_copy_count += src.prefetch_h2d_copy_count;
    dst.prefetch_h2d_copy_bytes += src.prefetch_h2d_copy_bytes;
    dst.prefetch_h2d_enqueue_time_us += src.prefetch_h2d_enqueue_time_us;
    dst.l2_hits += src.l2_hits;
    dst.l2_misses += src.l2_misses;
    dst.l2_fills += src.l2_fills;
    dst.l2_evictions += src.l2_evictions;
    dst.l2_fill_bytes += src.l2_fill_bytes;
    dst.l2_fill_time_us += src.l2_fill_time_us;
    dst.ops += src.ops;
    dst.staged_ops += src.staged_ops;
    dst.overflow_ops += src.overflow_ops;
    dst.unique_experts += src.unique_experts;
    dst.unique_experts_max = std::max(dst.unique_experts_max, src.unique_experts_max);
    dst.ids_bytes += src.ids_bytes;
    dst.ids_d2h_time_us += src.ids_d2h_time_us;
    dst.ids_d2h_sync_count += src.ids_d2h_sync_count;
    dst.ids_cache_hits += src.ids_cache_hits;
    dst.acquire_time_us += src.acquire_time_us;
    dst.remap_time_us += src.remap_time_us;
    dst.copy_wait_event_count += src.copy_wait_event_count;
    dst.copy_wait_event_time_us += src.copy_wait_event_time_us;
    dst.total_time_us += src.total_time_us;
}

static void ggml_cuda_moe_log_phase_stats(const char * name, const moe_cache_phase_stats & s) {
    const uint64_t l1_total = s.l1_hits + s.l1_misses;
    const uint64_t l2_total = s.l2_hits + s.l2_misses;
    const double l1_hit_rate = l1_total > 0 ? 100.0 * (double) s.l1_hits / (double) l1_total : 0.0;
    const double l2_hit_rate = l2_total > 0 ? 100.0 * (double) s.l2_hits / (double) l2_total : 0.0;
    const double avg_unique = s.ops > 0 ? (double) s.unique_experts / (double) s.ops : 0.0;
    GGML_LOG(
        "moe-cache-phase: phase=%s ops=%llu staged_ops=%llu overflow_ops=%llu unique_avg=%.2f unique_max=%llu ids_mib=%.2f ids_d2h_mib=%.2f ids_d2h_ms=%.3f ids_d2h_syncs=%llu ids_cache_hits=%llu acquire_ms=%.3f remap_ms=%.3f copy_wait_events=%llu copy_wait_event_ms=%.3f op_cpu_ms=%.3f l1_hits=%llu l1_misses=%llu l1_evictions=%llu l1_hit_rate=%.2f%% l2_hits=%llu l2_misses=%llu l2_fills=%llu l2_evictions=%llu l2_fill_mib=%.2f l2_fill_ms=%.3f l2_hit_rate=%.2f%% h2d_copies=%llu h2d_mib=%.2f h2d_enqueue_ms=%.3f prefetch_hits=%llu prefetch_misses=%llu prefetch_used=%llu prefetch_h2d_copies=%llu prefetch_h2d_mib=%.2f prefetch_h2d_enqueue_ms=%.3f\n",
        name,
        (unsigned long long) s.ops,
        (unsigned long long) s.staged_ops,
        (unsigned long long) s.overflow_ops,
        avg_unique,
        (unsigned long long) s.unique_experts_max,
        (double) s.ids_bytes / 1024.0 / 1024.0,
        (double) s.ids_bytes / 1024.0 / 1024.0,
        (double) s.ids_d2h_time_us / 1000.0,
        (unsigned long long) s.ids_d2h_sync_count,
        (unsigned long long) s.ids_cache_hits,
        (double) s.acquire_time_us / 1000.0,
        (double) s.remap_time_us / 1000.0,
        (unsigned long long) s.copy_wait_event_count,
        (double) s.copy_wait_event_time_us / 1000.0,
        (double) s.total_time_us / 1000.0,
        (unsigned long long) s.l1_hits,
        (unsigned long long) s.l1_misses,
        (unsigned long long) s.l1_evictions,
        l1_hit_rate,
        (unsigned long long) s.l2_hits,
        (unsigned long long) s.l2_misses,
        (unsigned long long) s.l2_fills,
        (unsigned long long) s.l2_evictions,
        (double) s.l2_fill_bytes / 1024.0 / 1024.0,
        (double) s.l2_fill_time_us / 1000.0,
        l2_hit_rate,
        (unsigned long long) s.h2d_copy_count,
        (double) s.h2d_copy_bytes / 1024.0 / 1024.0,
        (double) s.h2d_enqueue_time_us / 1000.0,
        (unsigned long long) s.prefetch_hits,
        (unsigned long long) s.prefetch_misses,
        (unsigned long long) s.prefetch_used,
        (unsigned long long) s.prefetch_h2d_copy_count,
        (double) s.prefetch_h2d_copy_bytes / 1024.0 / 1024.0,
        (double) s.prefetch_h2d_enqueue_time_us / 1000.0);
}

static uint64_t ggml_cuda_moe_cache_top_accesses(std::vector<uint64_t> counts, size_t n_top) {
    if (counts.empty() || n_top == 0) {
        return 0;
    }

    if (n_top > counts.size()) {
        n_top = counts.size();
    }

    std::sort(counts.begin(), counts.end(), [](uint64_t a, uint64_t b) {
        return a > b;
    });
    uint64_t total = 0;
    for (size_t i = 0; i < n_top; ++i) {
        total += counts[i];
    }
    return total;
}

static double moe_cache_pct(uint64_t numerator, uint64_t denominator) {
    return denominator > 0 ? 100.0 * (double) numerator / (double) denominator : 0.0;
}

static moe_cache_hot_tensor_stats ggml_cuda_moe_cache_expert_stats(const struct ggml_cuda_moe_cache * cache) {
    moe_cache_hot_tensor_stats s = {};
    if (!cache || cache->n_experts <= 0) {
        return s;
    }

    s.name = cache->tensor_name;
    s.experts = (uint64_t) cache->n_experts;
    if (cache->expert_access_counts.empty()) {
        return s;
    }

    s.first_touches = cache->expert_first_touches;
    s.reuse_le_l1 = cache->expert_reuse_le_l1;
    s.reuse_le_l2 = cache->expert_reuse_le_l2;
    s.reuse_gt_l2 = cache->expert_reuse_gt_l2;

    for (uint64_t count : cache->expert_access_counts) {
        s.accesses += count;
        if (count > 0) {
            s.unique_experts++;
            if (count == 1) {
                s.touched_once++;
            } else {
                s.touched_ge2++;
            }
        }
    }

    const size_t n_experts = (size_t) cache->n_experts;
    const size_t n_top1 = std::max<size_t>(1, (n_experts + 99) / 100);
    const size_t n_top5 = std::max<size_t>(1, (n_experts * 5 + 99) / 100);
    const size_t n_top10 = std::max<size_t>(1, (n_experts * 10 + 99) / 100);
    s.top1_accesses = ggml_cuda_moe_cache_top_accesses(cache->expert_access_counts, n_top1);
    s.top5_accesses = ggml_cuda_moe_cache_top_accesses(cache->expert_access_counts, n_top5);
    s.top10_accesses = ggml_cuda_moe_cache_top_accesses(cache->expert_access_counts, n_top10);
    return s;
}

static moe_cache_tensor_decode_stats ggml_cuda_moe_cache_decode_tensor_stats(const struct ggml_cuda_moe_cache * cache) {
    moe_cache_tensor_decode_stats s = {};
    if (!cache || cache->n_experts <= 0) {
        return s;
    }

    s.name = cache->tensor_name;
    s.experts = (uint64_t) cache->n_experts;
    const moe_cache_phase_stats ds = ggml_cuda_moe_cache_phase_stats(cache, 1);
    s.l1_hits = ds.l1_hits;
    s.l1_misses = ds.l1_misses;
    s.l1_evictions = ds.l1_evictions;
    s.h2d_copy_count = ds.h2d_copy_count;
    s.h2d_copy_bytes = ds.h2d_copy_bytes;
    s.h2d_enqueue_time_us = ds.h2d_enqueue_time_us;
    s.prefetch_h2d_copy_count = ds.prefetch_h2d_copy_count;
    s.prefetch_h2d_copy_bytes = ds.prefetch_h2d_copy_bytes;
    s.prefetch_used = ds.prefetch_used;
    s.prefetch_evictions = ds.prefetch_evictions;
    s.demand_evictions = ds.demand_evictions;
    s.evicted_prefetched = ds.evicted_prefetched;
    s.evicted_hit_count_le1 = ds.evicted_hit_count_le1;
    s.evicted_hit_count_ge2 = ds.evicted_hit_count_ge2;
    s.evicted_age_le_l1 = ds.evicted_age_le_l1;
    s.evicted_age_gt_l1 = ds.evicted_age_gt_l1;

    const std::vector<uint64_t> & counts = cache->phase_expert_access_counts[1];
    if (!counts.empty()) {
        for (uint64_t count : counts) {
            s.accesses += count;
            if (count > 0) {
                s.unique_experts++;
                if (count == 1) {
                    s.touched_once++;
                } else {
                    s.touched_ge2++;
                }
            }
        }

        const size_t n_experts = (size_t) cache->n_experts;
        const size_t n_top1 = std::max<size_t>(1, (n_experts + 99) / 100);
        const size_t n_top5 = std::max<size_t>(1, (n_experts * 5 + 99) / 100);
        const size_t n_top10 = std::max<size_t>(1, (n_experts * 10 + 99) / 100);
        s.top1_accesses = ggml_cuda_moe_cache_top_accesses(counts, n_top1);
        s.top5_accesses = ggml_cuda_moe_cache_top_accesses(counts, n_top5);
        s.top10_accesses = ggml_cuda_moe_cache_top_accesses(counts, n_top10);
    }

    s.reuse_le_l1 = cache->phase_expert_reuse_le_l1[1];
    s.reuse_gt_l1 = cache->phase_expert_reuse_gt_l1[1];
    s.reuse_total = s.reuse_le_l1 + s.reuse_gt_l1;
    s.reuse_hist = cache->phase_expert_reuse_hist[1];
    return s;
}

static double moe_cache_donor_score(const moe_cache_tensor_decode_stats & s) {
    const uint64_t l1_total = s.l1_hits + s.l1_misses;
    const double l1_hit_rate = moe_cache_pct(s.l1_hits, l1_total);
    const double h2d_mib = (double) s.h2d_copy_bytes / 1024.0 / 1024.0;
    const double reuse_gt_l1_pct = moe_cache_pct(s.reuse_gt_l1, s.reuse_total);
    return l1_hit_rate - 0.10 * h2d_mib - 0.001 * (double) s.l1_misses - reuse_gt_l1_pct;
}

static void moe_cache_log_decode_tensor_line(
        const char * tag,
        int rank,
        const moe_cache_tensor_decode_stats & s,
        bool include_eviction_detail) {
    const uint64_t l1_total = s.l1_hits + s.l1_misses;
    const double l1_hit_rate = moe_cache_pct(s.l1_hits, l1_total);
    const double unique_pct = moe_cache_pct(s.unique_experts, s.experts);
    const double top1_pct = moe_cache_pct(s.top1_accesses, s.accesses);
    const double top5_pct = moe_cache_pct(s.top5_accesses, s.accesses);
    const double top10_pct = moe_cache_pct(s.top10_accesses, s.accesses);
    const double reuse_le_l1_pct = moe_cache_pct(s.reuse_le_l1, s.reuse_total);
    const double reuse_gt_l1_pct = moe_cache_pct(s.reuse_gt_l1, s.reuse_total);
    const double prefetch_use_rate = moe_cache_pct(s.prefetch_used, s.prefetch_h2d_copy_count);
    const uint64_t prefetch_unused = s.prefetch_h2d_copy_count > s.prefetch_used ?
        s.prefetch_h2d_copy_count - s.prefetch_used : 0;
    const uint64_t demand_h2d_copies = s.h2d_copy_count > s.prefetch_h2d_copy_count ?
        s.h2d_copy_count - s.prefetch_h2d_copy_count : 0;

    if (include_eviction_detail) {
        GGML_LOG(
            "%s: rank=%d tensor=%s phase=decode l1_hits=%llu l1_misses=%llu l1_hit_rate=%.2f%% l1_evictions=%llu h2d_copies=%llu h2d_mib=%.2f reuse_gt_l1_pct=%.2f reuse_le_l1_pct=%.2f unique_experts=%llu unique_pct=%.2f touched_once=%llu touched_ge2=%llu top1_pct=%.2f top5_pct=%.2f top10_pct=%.2f demand_h2d_copies=%llu prefetch_h2d_copies=%llu prefetch_used=%llu prefetch_unused=%llu prefetch_use_rate=%.2f%% demand_evictions=%llu prefetch_evictions=%llu evicted_prefetched=%llu evicted_hit_count_le1=%llu evicted_hit_count_ge2=%llu evicted_age_le_l1=%llu evicted_age_gt_l1=%llu rd_first_touch=%llu rd_le8=%llu rd_le16=%llu rd_le32=%llu rd_le_l1=%llu rd_le_2xl1=%llu rd_gt_2xl1=%llu\n",
            tag,
            rank,
            s.name.empty() ? "?" : s.name.c_str(),
            (unsigned long long) s.l1_hits,
            (unsigned long long) s.l1_misses,
            l1_hit_rate,
            (unsigned long long) s.l1_evictions,
            (unsigned long long) s.h2d_copy_count,
            (double) s.h2d_copy_bytes / 1024.0 / 1024.0,
            reuse_gt_l1_pct,
            reuse_le_l1_pct,
            (unsigned long long) s.unique_experts,
            unique_pct,
            (unsigned long long) s.touched_once,
            (unsigned long long) s.touched_ge2,
            top1_pct,
            top5_pct,
            top10_pct,
            (unsigned long long) demand_h2d_copies,
            (unsigned long long) s.prefetch_h2d_copy_count,
            (unsigned long long) s.prefetch_used,
            (unsigned long long) prefetch_unused,
            prefetch_use_rate,
            (unsigned long long) s.demand_evictions,
            (unsigned long long) s.prefetch_evictions,
            (unsigned long long) s.evicted_prefetched,
            (unsigned long long) s.evicted_hit_count_le1,
            (unsigned long long) s.evicted_hit_count_ge2,
            (unsigned long long) s.evicted_age_le_l1,
            (unsigned long long) s.evicted_age_gt_l1,
            (unsigned long long) s.reuse_hist.first_touch,
            (unsigned long long) s.reuse_hist.le8,
            (unsigned long long) s.reuse_hist.le16,
            (unsigned long long) s.reuse_hist.le32,
            (unsigned long long) s.reuse_hist.le_l1,
            (unsigned long long) s.reuse_hist.le_2xl1,
            (unsigned long long) s.reuse_hist.gt_2xl1);
    } else {
        GGML_LOG(
            "%s: rank=%d tensor=%s l1_hit_rate=%.2f%% h2d_mib=%.2f l1_misses=%llu reuse_gt_l1_pct=%.2f unique_pct=%.2f top10_pct=%.2f\n",
            tag,
            rank,
            s.name.empty() ? "?" : s.name.c_str(),
            l1_hit_rate,
            (double) s.h2d_copy_bytes / 1024.0 / 1024.0,
            (unsigned long long) s.l1_misses,
            reuse_gt_l1_pct,
            unique_pct,
            top10_pct);
    }
}

extern "C"
void ggml_cuda_moe_cache_reset_stats(struct ggml_cuda_moe_cache * cache) {
    if (!cache) return;
    cache->hits.store(0, std::memory_order_relaxed);
    cache->misses.store(0, std::memory_order_relaxed);
    cache->evictions.store(0, std::memory_order_relaxed);
    cache->h2d_copy_count.store(0, std::memory_order_relaxed);
    cache->h2d_copy_bytes.store(0, std::memory_order_relaxed);
    cache->h2d_enqueue_time_us.store(0, std::memory_order_relaxed);
    std::fill(cache->slot_hit_count.begin(), cache->slot_hit_count.end(), 0ull);
    std::fill(cache->slot_fill_access.begin(), cache->slot_fill_access.end(), cache->access_counter);
    for (int phase = 0; phase < 2; ++phase) {
        cache->phase_hits[phase].store(0, std::memory_order_relaxed);
        cache->phase_misses[phase].store(0, std::memory_order_relaxed);
        cache->phase_evictions[phase].store(0, std::memory_order_relaxed);
        cache->phase_h2d_copy_count[phase].store(0, std::memory_order_relaxed);
        cache->phase_h2d_copy_bytes[phase].store(0, std::memory_order_relaxed);
        cache->phase_h2d_enqueue_time_us[phase].store(0, std::memory_order_relaxed);
        cache->phase_prefetch_hits[phase].store(0, std::memory_order_relaxed);
        cache->phase_prefetch_misses[phase].store(0, std::memory_order_relaxed);
        cache->phase_prefetch_used[phase].store(0, std::memory_order_relaxed);
        cache->phase_prefetch_evictions[phase].store(0, std::memory_order_relaxed);
        cache->phase_demand_evictions[phase].store(0, std::memory_order_relaxed);
        cache->phase_evicted_prefetched[phase].store(0, std::memory_order_relaxed);
        cache->phase_evicted_hit_count_le1[phase].store(0, std::memory_order_relaxed);
        cache->phase_evicted_hit_count_ge2[phase].store(0, std::memory_order_relaxed);
        cache->phase_evicted_age_le_l1[phase].store(0, std::memory_order_relaxed);
        cache->phase_evicted_age_gt_l1[phase].store(0, std::memory_order_relaxed);
        cache->phase_prefetch_h2d_copy_count[phase].store(0, std::memory_order_relaxed);
        cache->phase_prefetch_h2d_copy_bytes[phase].store(0, std::memory_order_relaxed);
        cache->phase_prefetch_h2d_enqueue_time_us[phase].store(0, std::memory_order_relaxed);
    }
    cache->sampled_mincore_checks.store(0, std::memory_order_relaxed);
    cache->sampled_pages_total.store(0, std::memory_order_relaxed);
    cache->sampled_pages_resident.store(0, std::memory_order_relaxed);
    cache->sampled_nonresident_expert_count.store(0, std::memory_order_relaxed);
    cache->mincore_failures.store(0, std::memory_order_relaxed);
    if (cache->l2.slot_pool_h) {
        cache->l2.hits.store(0, std::memory_order_relaxed);
        cache->l2.misses.store(0, std::memory_order_relaxed);
        cache->l2.fills.store(0, std::memory_order_relaxed);
        cache->l2.evictions.store(0, std::memory_order_relaxed);
        cache->l2.fill_bytes.store(0, std::memory_order_relaxed);
        cache->l2.fill_time_us.store(0, std::memory_order_relaxed);
        for (int phase = 0; phase < 2; ++phase) {
            cache->l2.phase_hits[phase].store(0, std::memory_order_relaxed);
            cache->l2.phase_misses[phase].store(0, std::memory_order_relaxed);
            cache->l2.phase_fills[phase].store(0, std::memory_order_relaxed);
            cache->l2.phase_evictions[phase].store(0, std::memory_order_relaxed);
            cache->l2.phase_fill_bytes[phase].store(0, std::memory_order_relaxed);
            cache->l2.phase_fill_time_us[phase].store(0, std::memory_order_relaxed);
        }
    }
    std::fill(cache->expert_access_counts.begin(), cache->expert_access_counts.end(), 0);
    std::fill(cache->expert_last_access.begin(), cache->expert_last_access.end(), 0);
    cache->expert_access_counter = 0;
    cache->expert_first_touches = 0;
    cache->expert_reuse_le_l1 = 0;
    cache->expert_reuse_le_l2 = 0;
    cache->expert_reuse_gt_l2 = 0;
    for (int phase = 0; phase < 2; ++phase) {
        std::fill(cache->phase_expert_access_counts[phase].begin(), cache->phase_expert_access_counts[phase].end(), 0);
        std::fill(cache->phase_expert_last_access[phase].begin(), cache->phase_expert_last_access[phase].end(), 0);
        cache->phase_expert_access_counter[phase] = 0;
        cache->phase_expert_first_touches[phase] = 0;
        cache->phase_expert_reuse_le_l1[phase] = 0;
        cache->phase_expert_reuse_gt_l1[phase] = 0;
        cache->phase_expert_reuse_hist[phase] = {};
    }
}

extern "C"
void ggml_cuda_moe_record_op_stats(
    bool     is_decode,
    bool     staged,
    bool     overflow,
    uint64_t unique_experts,
    uint64_t ids_bytes,
    uint64_t ids_d2h_time_us,
    uint64_t ids_d2h_sync_count,
    uint64_t acquire_time_us,
    uint64_t remap_time_us,
    uint64_t copy_wait_event_count,
    uint64_t copy_wait_event_time_us,
    uint64_t total_time_us,
    bool     ids_cache_hit) {

    if (!moe_cache_mm_debug_enabled()) {
        return;
    }

    const int phase = moe_cache_phase_index(is_decode);
    moe_cache_op_phase_stats & s = g_moe_cache_op_stats[phase];
    s.ops.fetch_add(1, std::memory_order_relaxed);
    s.staged_ops.fetch_add(staged ? 1 : 0, std::memory_order_relaxed);
    s.overflow_ops.fetch_add(overflow ? 1 : 0, std::memory_order_relaxed);
    s.unique_experts.fetch_add(unique_experts, std::memory_order_relaxed);
    moe_cache_atomic_max(s.unique_experts_max, unique_experts);
    s.ids_bytes.fetch_add(ids_bytes, std::memory_order_relaxed);
    s.ids_d2h_time_us.fetch_add(ids_d2h_time_us, std::memory_order_relaxed);
    s.ids_d2h_sync_count.fetch_add(ids_d2h_sync_count, std::memory_order_relaxed);
    s.ids_cache_hits.fetch_add(ids_cache_hit ? 1 : 0, std::memory_order_relaxed);
    s.acquire_time_us.fetch_add(acquire_time_us, std::memory_order_relaxed);
    s.remap_time_us.fetch_add(remap_time_us, std::memory_order_relaxed);
    s.copy_wait_event_count.fetch_add(copy_wait_event_count, std::memory_order_relaxed);
    s.copy_wait_event_time_us.fetch_add(copy_wait_event_time_us, std::memory_order_relaxed);
    s.total_time_us.fetch_add(total_time_us, std::memory_order_relaxed);
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

static void ggml_backend_cuda_moe_cached_mmap_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    moe_cache_unregister_mmap_range(buffer->context, buffer->size);
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
    buffer->iface.free_buffer = ggml_backend_cuda_moe_cached_mmap_buffer_free_buffer;
    moe_cache_register_mmap_range(ptr, size);
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

// Per-tensor cache registry: one cache per (device, tensor_data).
// "Tensor" here means a model expert weight tensor like blk.0.ffn_up_exps.weight.
// Each name is unique and identifies a (layer, matrix_kind) pair. The N slots
// in that tensor's cache are dedicated to that tensor's experts only -- no
// cross-layer competition.
//
// User-facing N is slots-per-tensor, not slots-total. For Qwen3.6 35B-A3B
// with 40 layers x 3 matrices = 120 expert tensors and N=32:
//   total memory = 120 x 32 x ~0.82 MiB ~= 3.1 GiB per device
//
// Slot size per cache equals that tensor's expert_stride, so slots are tightly
// packed; the synthetic src0 stays contiguous and kernels run unchanged.
namespace {

struct moe_cache_key {
    int          device;
    const void * tensor_data;
    bool operator<(const moe_cache_key & o) const {
        if (device != o.device) return device < o.device;
        return std::less<const void *>{}(tensor_data, o.tensor_data);
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

static std::atomic<int> g_moe_cache_l2_mmap_tensor_count{0};

} // namespace

extern "C"
struct ggml_cuda_moe_cache * ggml_cuda_moe_cache_get_or_create_for_tensor(
    int          device,
    const void * tensor_data,
    size_t       slot_size_bytes,
    int          n_slots,
    int64_t      n_experts,
    const char * tensor_name_for_log) {

    if (tensor_data == nullptr || tensor_name_for_log == nullptr || tensor_name_for_log[0] == '\0') {
        return nullptr;
    }

    auto & reg = get_registry();
    std::lock_guard<std::mutex> lk(reg.mu);

    moe_cache_key k{device, tensor_data};
    auto it = reg.by_key.find(k);
    if (it != reg.by_key.end()) {
        return it->second;
    }

    const bool source_is_mmap = moe_cache_is_mmap_range(tensor_data, slot_size_bytes);
    size_t l2_budget_bytes = 0;
    int l2_target_slots = 0;
    if (source_is_mmap) {
        const size_t total_l2_budget = g_moe_cache_l2_pinned_size.load(std::memory_order_relaxed);
        const int n_mmap_tensors = g_moe_cache_l2_mmap_tensor_count.load(std::memory_order_relaxed);
        if (total_l2_budget > 0 && n_mmap_tensors > 0) {
            l2_budget_bytes = total_l2_budget / (size_t) n_mmap_tensors;
            const size_t budget_slots = l2_budget_bytes / slot_size_bytes;
            const size_t expert_slots = n_experts > 0 ? (size_t) n_experts : budget_slots;
            l2_target_slots = (int) std::min(budget_slots, expert_slots);
        }
    }

    ggml_cuda_moe_cache * c = ggml_cuda_moe_cache_init(
        device, slot_size_bytes, n_slots, source_is_mmap, l2_budget_bytes, l2_target_slots);
    if (!c) {
        return nullptr;
    }

    c->tensor_name = tensor_name_for_log;
    c->tensor_data = tensor_data;
    c->n_experts = n_experts;
    reg.by_key.emplace(k, c);
    GGML_LOG_INFO("load_tensors: CUDA_MoE_Cache_Pool[%-32s] = %7.2f MiB  (%d slots x %.2f MiB)\n",
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

extern "C"
void ggml_backend_cuda_moe_set_l2_pinned_cache_size(size_t bytes) {
    g_moe_cache_l2_pinned_size.store(bytes, std::memory_order_relaxed);
}

extern "C"
size_t ggml_backend_cuda_moe_get_l2_pinned_cache_size(void) {
    return g_moe_cache_l2_pinned_size.load(std::memory_order_relaxed);
}

extern "C"
void ggml_backend_cuda_moe_set_debug_mm(bool enabled) {
    g_moe_cache_mm_debug.store(enabled, std::memory_order_relaxed);
}

extern "C"
bool ggml_backend_cuda_moe_get_debug_mm(void) {
    return g_moe_cache_mm_debug.load(std::memory_order_relaxed);
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
    int n_mmap_tensors = 0;
    for (const auto & t : st.tensors) {
        if (moe_cache_is_mmap_range(t.tensor_data, t.per_expert_bytes)) {
            ++n_mmap_tensors;
        }
    }
    g_moe_cache_l2_mmap_tensor_count.store(n_mmap_tensors, std::memory_order_relaxed);
    for (const auto & t : st.tensors) {
        ggml_cuda_moe_cache_get_or_create_for_tensor(
            device, t.tensor_data, t.per_expert_bytes, n_slots,
            t.n_experts,
            t.tensor_name.empty() ? "?" : t.tensor_name.c_str());
    }
}

extern "C"
void ggml_backend_cuda_moe_prefetch_experts(
    int             device,
    const char *    tensor_name,
    const int32_t * eids,
    int             n_eids,
    bool            use_l2,
    bool            is_decode) {
    if (!tensor_name || !eids || n_eids <= 0) return;

    // 1. Look up the observed tensor (data ptr + stride) by name.
    observed_tensor found;
    bool has_found = false;
    {
        auto & st = get_observation_state();
        std::lock_guard<std::mutex> lk(st.mu);
        for (const auto & t : st.tensors) {
            if (t.tensor_name == tensor_name) {
                found = t;
                has_found = true;
                break;
            }
        }
        if (!has_found) return;
    }

    const void * tensor_data    = found.tensor_data;
    const size_t expert_stride  = found.per_expert_bytes;

    // 2. Find the cache for this tensor.
    auto & reg = get_registry();
    ggml_cuda_moe_cache * cache = nullptr;
    {
        std::lock_guard<std::mutex> lk(reg.mu);
        moe_cache_key k{device, tensor_data};
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
        (void)ggml_cuda_moe_cache_acquire(cache, host_ptr, expert_stride, copy_stream, use_l2, is_decode, true, false);
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
    moe_cache_mm_stats mm = {};
    moe_cache_l2_stats l2 = {};
    moe_cache_expert_stats experts = {};
    moe_cache_hot_tensor_stats hot_tensor = {};
    std::vector<uint64_t> all_expert_access_counts;
    moe_cache_tensor_decode_stats hot_decode_miss_tensor = {};
    std::vector<moe_cache_tensor_decode_stats> decode_tensor_stats;
    moe_cache_phase_stats phase_stats[2] = {};
    if (moe_cache_mm_debug_enabled()) {
        for (int phase = 0; phase < 2; ++phase) {
            ggml_cuda_moe_add_phase_stats(phase_stats[phase], ggml_cuda_moe_op_phase_stats(phase));
        }
    }
    size_t   n_caches = reg.by_key.size();
    for (auto & kv : reg.by_key) {
        uint64_t h = 0, m = 0, e = 0;
        ggml_cuda_moe_cache_stats(kv.second, &h, &m, &e);
        total_hits      += h;
        total_misses    += m;
        total_evictions += e;
        if (moe_cache_mm_debug_enabled()) {
            moe_cache_mm_stats s = ggml_cuda_moe_cache_mm_stats(kv.second);
            mm.h2d_copy_count                   += s.h2d_copy_count;
            mm.h2d_copy_bytes                   += s.h2d_copy_bytes;
            mm.h2d_enqueue_time_us              += s.h2d_enqueue_time_us;
            mm.sampled_mincore_checks           += s.sampled_mincore_checks;
            mm.sampled_pages_total              += s.sampled_pages_total;
            mm.sampled_pages_resident           += s.sampled_pages_resident;
            mm.sampled_nonresident_expert_count += s.sampled_nonresident_expert_count;
            mm.mincore_failures                 += s.mincore_failures;

            moe_cache_l2_stats ls = ggml_cuda_moe_cache_l2_stats(kv.second);
            l2.budget_bytes += ls.budget_bytes;
            l2.slots        += ls.slots;
            l2.used_bytes   += ls.used_bytes;
            l2.hits         += ls.hits;
            l2.misses       += ls.misses;
            l2.fills        += ls.fills;
            l2.evictions    += ls.evictions;
            l2.fill_bytes   += ls.fill_bytes;
            l2.fill_time_us += ls.fill_time_us;

            for (int phase = 0; phase < 2; ++phase) {
                ggml_cuda_moe_add_phase_stats(phase_stats[phase], ggml_cuda_moe_cache_phase_stats(kv.second, phase));
            }

            moe_cache_hot_tensor_stats es = ggml_cuda_moe_cache_expert_stats(kv.second);
            if (es.experts > 0) {
                experts.tensors++;
                experts.experts        += es.experts;
                experts.unique_experts += es.unique_experts;
                experts.accesses       += es.accesses;
                experts.first_touches  += es.first_touches;
                experts.reuse_le_l1    += es.reuse_le_l1;
                experts.reuse_le_l2    += es.reuse_le_l2;
                experts.reuse_gt_l2    += es.reuse_gt_l2;
                experts.touched_once   += es.touched_once;
                experts.touched_ge2    += es.touched_ge2;
                ggml_cuda_moe_cache_append_expert_counts(kv.second, all_expert_access_counts);
                if (es.accesses > hot_tensor.accesses) {
                    hot_tensor = es;
                }
            }

            moe_cache_tensor_decode_stats ds = ggml_cuda_moe_cache_decode_tensor_stats(kv.second);
            if (ds.l1_hits + ds.l1_misses + ds.h2d_copy_count > 0) {
                decode_tensor_stats.push_back(ds);
            }
            if (ds.h2d_copy_bytes > hot_decode_miss_tensor.h2d_copy_bytes) {
                hot_decode_miss_tensor = ds;
            }
        }
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

    if (moe_cache_mm_debug_enabled()) {
        const moe_cache_proc_snapshot proc = moe_cache_get_proc_delta();
        const double h2d_mib = (double) mm.h2d_copy_bytes / 1024.0 / 1024.0;
        const double h2d_enqueue_ms = (double) mm.h2d_enqueue_time_us / 1000.0;
        const double resident_pct = mm.sampled_pages_total > 0 ?
            100.0 * (double) mm.sampled_pages_resident / (double) mm.sampled_pages_total : 0.0;
        const uint64_t l2_total = l2.hits + l2.misses;
        const double l2_hit_rate = l2_total > 0 ? 100.0 * (double) l2.hits / (double) l2_total : 0.0;
        const uint64_t reuse_total = experts.reuse_le_l1 + experts.reuse_le_l2 + experts.reuse_gt_l2;
        const double unique_pct = experts.experts > 0 ?
            100.0 * (double) experts.unique_experts / (double) experts.experts : 0.0;
        const double first_touch_pct = experts.accesses > 0 ?
            100.0 * (double) experts.first_touches / (double) experts.accesses : 0.0;
        const double reuse_l1_pct = reuse_total > 0 ?
            100.0 * (double) experts.reuse_le_l1 / (double) reuse_total : 0.0;
        const double reuse_l2_pct = reuse_total > 0 ?
            100.0 * (double) experts.reuse_le_l2 / (double) reuse_total : 0.0;
        const double reuse_gt_l2_pct = reuse_total > 0 ?
            100.0 * (double) experts.reuse_gt_l2 / (double) reuse_total : 0.0;
        if (!all_expert_access_counts.empty()) {
            const size_t n_experts = all_expert_access_counts.size();
            const size_t n_top1 = std::max<size_t>(1, (n_experts + 99) / 100);
            const size_t n_top5 = std::max<size_t>(1, (n_experts * 5 + 99) / 100);
            const size_t n_top10 = std::max<size_t>(1, (n_experts * 10 + 99) / 100);
            experts.top1_accesses = ggml_cuda_moe_cache_top_accesses(all_expert_access_counts, n_top1);
            experts.top5_accesses = ggml_cuda_moe_cache_top_accesses(all_expert_access_counts, n_top5);
            experts.top10_accesses = ggml_cuda_moe_cache_top_accesses(all_expert_access_counts, n_top10);
        }
        const double top1_pct = experts.accesses > 0 ?
            100.0 * (double) experts.top1_accesses / (double) experts.accesses : 0.0;
        const double top5_pct = experts.accesses > 0 ?
            100.0 * (double) experts.top5_accesses / (double) experts.accesses : 0.0;
        const double top10_pct = experts.accesses > 0 ?
            100.0 * (double) experts.top10_accesses / (double) experts.accesses : 0.0;
        const double hot_unique_pct = hot_tensor.experts > 0 ?
            100.0 * (double) hot_tensor.unique_experts / (double) hot_tensor.experts : 0.0;
        const double hot_top10_pct = hot_tensor.accesses > 0 ?
            100.0 * (double) hot_tensor.top10_accesses / (double) hot_tensor.accesses : 0.0;
        const uint64_t hot_reuse_total = hot_tensor.reuse_le_l1 + hot_tensor.reuse_le_l2 + hot_tensor.reuse_gt_l2;
        const double hot_reuse_gt_l2_pct = hot_reuse_total > 0 ?
            100.0 * (double) hot_tensor.reuse_gt_l2 / (double) hot_reuse_total : 0.0;
        const uint64_t hot_decode_total = hot_decode_miss_tensor.l1_hits + hot_decode_miss_tensor.l1_misses;
        const double hot_decode_hit_rate = hot_decode_total > 0 ?
            100.0 * (double) hot_decode_miss_tensor.l1_hits / (double) hot_decode_total : 0.0;
        const double hot_decode_reuse_gt_l1_pct = hot_decode_miss_tensor.reuse_total > 0 ?
            100.0 * (double) hot_decode_miss_tensor.reuse_gt_l1 / (double) hot_decode_miss_tensor.reuse_total : 0.0;

        GGML_LOG(
            "moe-cache-l2: l2_budget_mib=%.2f l2_slots=%llu l2_used_mib=%.2f l2_hits=%llu l2_misses=%llu l2_fills=%llu l2_evictions=%llu l2_fill_mib=%.2f l2_fill_ms=%.3f l2_hit_rate=%.2f%%\n",
            (double) l2.budget_bytes / 1024.0 / 1024.0,
            (unsigned long long) l2.slots,
            (double) l2.used_bytes / 1024.0 / 1024.0,
            (unsigned long long) l2.hits,
            (unsigned long long) l2.misses,
            (unsigned long long) l2.fills,
            (unsigned long long) l2.evictions,
            (double) l2.fill_bytes / 1024.0 / 1024.0,
            (double) l2.fill_time_us / 1000.0,
            l2_hit_rate);
        GGML_LOG(
            "moe-cache-experts: tensors=%llu experts=%llu unique=%llu unique_pct=%.2f accesses=%llu first_touches=%llu first_touch_pct=%.2f touched_once=%llu touched_ge2=%llu reuse_le_l1=%llu reuse_le_l1_pct=%.2f reuse_le_l2=%llu reuse_le_l2_pct=%.2f reuse_gt_l2=%llu reuse_gt_l2_pct=%.2f top1_pct=%.2f top5_pct=%.2f top10_pct=%.2f\n",
            (unsigned long long) experts.tensors,
            (unsigned long long) experts.experts,
            (unsigned long long) experts.unique_experts,
            unique_pct,
            (unsigned long long) experts.accesses,
            (unsigned long long) experts.first_touches,
            first_touch_pct,
            (unsigned long long) experts.touched_once,
            (unsigned long long) experts.touched_ge2,
            (unsigned long long) experts.reuse_le_l1,
            reuse_l1_pct,
            (unsigned long long) experts.reuse_le_l2,
            reuse_l2_pct,
            (unsigned long long) experts.reuse_gt_l2,
            reuse_gt_l2_pct,
            top1_pct,
            top5_pct,
            top10_pct);
        GGML_LOG(
            "moe-cache-experts-hot: tensor=%s experts=%llu unique=%llu unique_pct=%.2f accesses=%llu touched_once=%llu touched_ge2=%llu top10_pct=%.2f reuse_gt_l2_pct=%.2f\n",
            hot_tensor.name.empty() ? "?" : hot_tensor.name.c_str(),
            (unsigned long long) hot_tensor.experts,
            (unsigned long long) hot_tensor.unique_experts,
            hot_unique_pct,
            (unsigned long long) hot_tensor.accesses,
            (unsigned long long) hot_tensor.touched_once,
            (unsigned long long) hot_tensor.touched_ge2,
            hot_top10_pct,
            hot_reuse_gt_l2_pct);
        GGML_LOG(
            "moe-cache-tensor-hot-decode-miss: tensor=%s l1_hits=%llu l1_misses=%llu l1_evictions=%llu l1_hit_rate=%.2f%% h2d_copies=%llu h2d_mib=%.2f h2d_enqueue_ms=%.3f prefetch_used=%llu prefetch_h2d_copies=%llu prefetch_h2d_mib=%.2f reuse_gt_l1_pct=%.2f\n",
            hot_decode_miss_tensor.name.empty() ? "?" : hot_decode_miss_tensor.name.c_str(),
            (unsigned long long) hot_decode_miss_tensor.l1_hits,
            (unsigned long long) hot_decode_miss_tensor.l1_misses,
            (unsigned long long) hot_decode_miss_tensor.l1_evictions,
            hot_decode_hit_rate,
            (unsigned long long) hot_decode_miss_tensor.h2d_copy_count,
            (double) hot_decode_miss_tensor.h2d_copy_bytes / 1024.0 / 1024.0,
            (double) hot_decode_miss_tensor.h2d_enqueue_time_us / 1000.0,
            (unsigned long long) hot_decode_miss_tensor.prefetch_used,
            (unsigned long long) hot_decode_miss_tensor.prefetch_h2d_copy_count,
            (double) hot_decode_miss_tensor.prefetch_h2d_copy_bytes / 1024.0 / 1024.0,
            hot_decode_reuse_gt_l1_pct);

        std::vector<moe_cache_tensor_decode_stats> top_miss = decode_tensor_stats;
        std::sort(top_miss.begin(), top_miss.end(),
            [](const moe_cache_tensor_decode_stats & a, const moe_cache_tensor_decode_stats & b) {
                return a.h2d_copy_bytes > b.h2d_copy_bytes;
            });
        const size_t top_miss_n = std::min<size_t>(10, top_miss.size());
        for (size_t i = 0; i < top_miss_n; ++i) {
            moe_cache_log_decode_tensor_line("moe-cache-tensor-top-decode-miss", (int) i + 1, top_miss[i], true);
        }

        std::vector<moe_cache_tensor_decode_stats> donors = decode_tensor_stats;
        std::sort(donors.begin(), donors.end(),
            [](const moe_cache_tensor_decode_stats & a, const moe_cache_tensor_decode_stats & b) {
                return moe_cache_donor_score(a) > moe_cache_donor_score(b);
            });
        const size_t donor_n = std::min<size_t>(10, donors.size());
        for (size_t i = 0; i < donor_n; ++i) {
            moe_cache_log_decode_tensor_line("moe-cache-tensor-donor-candidate", (int) i + 1, donors[i], false);
        }

        ggml_cuda_moe_log_phase_stats("prefill", phase_stats[0]);
        ggml_cuda_moe_log_phase_stats("decode", phase_stats[1]);
        GGML_LOG(
            "moe-cache-mm: h2d_copies=%llu h2d_mib=%.2f h2d_enqueue_ms=%.3f mincore_sampled=%llu resident_pages=%llu/%llu resident_pct=%.2f nonresident_experts=%llu mincore_fail=%llu proc_minflt=%llu proc_majflt=%llu vm_pswpin=%llu vm_pswpout=%llu vm_pgfault=%llu vm_pgmajfault=%llu vm_pgpgin=%llu vm_pgpgout=%llu vm_refault_file=%llu vm_refault_anon=%llu\n",
            (unsigned long long) mm.h2d_copy_count,
            h2d_mib,
            h2d_enqueue_ms,
            (unsigned long long) mm.sampled_mincore_checks,
            (unsigned long long) mm.sampled_pages_resident,
            (unsigned long long) mm.sampled_pages_total,
            resident_pct,
            (unsigned long long) mm.sampled_nonresident_expert_count,
            (unsigned long long) mm.mincore_failures,
            (unsigned long long) proc.proc_minflt,
            (unsigned long long) proc.proc_majflt,
            (unsigned long long) proc.vm_pswpin,
            (unsigned long long) proc.vm_pswpout,
            (unsigned long long) proc.vm_pgfault,
            (unsigned long long) proc.vm_pgmajfault,
            (unsigned long long) proc.vm_pgpgin,
            (unsigned long long) proc.vm_pgpgout,
            (unsigned long long) proc.vm_workingset_refault_file,
            (unsigned long long) proc.vm_workingset_refault_anon);
    }
    ggml_cuda_moe_reset_op_phase_stats();
}
