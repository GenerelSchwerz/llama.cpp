// MoE expert buffers, backend-owned GPU caches, and grouped execution.

#include "moe-cache.cuh"
#include "moe-cache-host.cuh"
#include "common.cuh"
#include "mmid.cuh"

#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <new>
#include <sstream>
#include <string>
#include <thread>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <xmmintrin.h>
#endif

#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {

static constexpr uint64_t MOE_CACHE_MM_SAMPLE_RATE = 64;
static constexpr uint64_t MOE_ORIGINAL_DIRECT_AUX_MAX_BYTES = 4 * 1024;
static constexpr size_t MOE_PREFILL_RESIDENT_AUX_BUDGET = 32 * 1024 * 1024;
static constexpr size_t MOE_PAGEABLE_STAGING_HEADROOM = 64 * 1024 * 1024;
static constexpr size_t MOE_ORIGINAL_AUX_BUDGET = 32 * 1024 * 1024;
static std::atomic<bool> g_moe_cache_mm_debug{false};

static bool moe_host_reserve_size(size_t limit, size_t bytes, size_t & reserved) {
    size_t rounded = 0;
    if (reserved > limit || !moe_host_round_size(bytes, rounded) || rounded > limit - reserved) {
        return false;
    }
    reserved += rounded;
    return true;
}

static const moe_host_source * moe_host_source_for(const ggml_tensor * tensor) {
    auto * owner = tensor != nullptr && tensor->buffer != nullptr ? moe_host_budget_for(tensor->buffer->buft) : nullptr;
    if (owner == nullptr) {
        return nullptr;
    }
    const auto found = owner->sources.find(tensor);
    return found != owner->sources.end() && found->second.data == tensor->data && found->second.size == ggml_nbytes(tensor) ? &found->second : nullptr;
}

static bool moe_host_contains(const moe_host_source & source, const void * input, size_t size) {
    const uintptr_t begin = reinterpret_cast<uintptr_t>(source.data);
    const uintptr_t address = reinterpret_cast<uintptr_t>(input);
    return address >= begin && address - begin <= source.size && size <= source.size - (address - begin);
}

static bool moe_host_materialize(const moe_host_source & source, const void * input, size_t size, void * output) {
    if (output == nullptr || !moe_host_contains(source, input, size)) {
        return false;
    }
    const size_t offset = reinterpret_cast<uintptr_t>(input) - reinterpret_cast<uintptr_t>(source.data);
    memcpy(output, source.data + offset, size);
    source.owner->materialized_bytes.fetch_add(size, std::memory_order_relaxed);
    return true;
}

static size_t moe_cache_quantized_source_padding(uint32_t type, int64_t ne0) {
    if (type >= GGML_TYPE_COUNT || ne0 <= 0 || !ggml_is_quantized((ggml_type) type)) {
        return 0;
    }
    const int64_t remainder = ne0 % MATRIX_ROW_PADDING;
    return remainder == 0 ? 0 : ggml_row_size((ggml_type) type, MATRIX_ROW_PADDING - remainder);
}

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
    std::atomic<uint64_t> split_staged_ops{0};
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
    uint64_t upload_calls;
    uint64_t upload_bytes;
    uint64_t staging_tile_count;
    uint64_t pipeline_tiles;
    uint64_t tile_wait_calls;
    uint64_t tile_wait_time_us;
    uint64_t cpu_pack_bytes;
    uint64_t cpu_pack_time_us;
    uint64_t pre_sync_calls;
    uint64_t pre_sync_time_us;
    uint64_t h2d_submit_calls;
    uint64_t h2d_submit_bytes;
    uint64_t h2d_submit_time_us;
    uint64_t post_sync_calls;
    uint64_t post_sync_time_us;
    uint64_t upload_errors;
    uint64_t ops;
    uint64_t staged_ops;
    uint64_t split_staged_ops;
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
    uint64_t reuse_gt_l1;
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
    uint64_t reuse_gt_l1 = 0;
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

struct moe_grouped_decode_debug_stats {
    std::atomic<uint64_t> decode_grouped{ 0 };
    std::atomic<uint64_t> decode_legacy{ 0 };
    std::atomic<uint64_t> submitted{ 0 };
    std::atomic<uint64_t> direct{ 0 };
    std::atomic<uint64_t> captures{ 0 };
    std::atomic<uint64_t> replays{ 0 };
    moe_grouped_decode_debug_stats() {
        for (auto & value : covered) {
            value.store(0, std::memory_order_relaxed);
        }
        for (auto & value : ready_by_group) {
            value.store(0, std::memory_order_relaxed);
        }
        for (auto & value : completed_by_group) {
            value.store(0, std::memory_order_relaxed);
        }
        for (auto & value : submitted_by_group) {
            value.store(0, std::memory_order_relaxed);
        }
    }

    std::atomic<uint64_t> plan_calls{0};
    std::atomic<uint64_t> plan_compiles{0};
    std::atomic<uint64_t> plan_reuses{0};
    std::atomic<uint64_t> calls{0};
    std::atomic<uint64_t> ready{0};
    std::atomic<uint64_t> completed{0};
    std::atomic<uint64_t> admitted_banks{0};
    std::atomic<uint64_t> fallback{0};
    std::atomic<uint64_t> rollback{0};
    std::atomic<uint64_t> host_staged_calls{0};
    std::atomic<uint64_t> host_staged_ops{0};
    std::atomic<uint64_t> host_staged_split_ops{0};
    std::atomic<uint64_t> strategy_switches{0};
    std::atomic<uint64_t> required_unsupported{0};
    std::atomic<uint64_t> prepare_error{0};
    std::atomic<uint64_t> finish_error{0};
    std::array<std::atomic<uint64_t>, (GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS + 63) / 64> covered;
    std::array<std::atomic<uint64_t>, GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS> ready_by_group;
    std::array<std::atomic<uint64_t>, GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS> completed_by_group;
    std::array<std::atomic<uint64_t>, GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS>             submitted_by_group;
    std::atomic<uint64_t *> device_transfers{nullptr};
    std::atomic<bool> device_transfers_failed{false};
};

struct moe_cache_telemetry {
    size_t n_caches = 0;
    uint64_t total_hits = 0;
    uint64_t total_misses = 0;
    uint64_t total_evictions = 0;
    moe_cache_mm_stats mm = {};
    moe_cache_expert_stats experts = {};
    moe_cache_hot_tensor_stats hot_tensor = {};
    std::vector<uint64_t> all_expert_access_counts;
    bool expert_details_omitted = false;
    moe_cache_tensor_decode_stats hot_decode_miss_tensor = {};
    std::vector<moe_cache_tensor_decode_stats> decode_tensor_stats;
    bool decode_details_omitted = false;
    moe_cache_phase_stats phase_stats[2] = {};
    ggml_cuda_moe_grouped_debug_telemetry grouped = {};
};

struct moe_cache_owner_telemetry {
    std::mutex mutex;
    std::unordered_set<ggml_cuda_moe_grouped_context *> active;
    moe_cache_telemetry retired;
};

static moe_cache_owner_telemetry & moe_cache_owner_telemetry_state() {
    static moe_cache_owner_telemetry state;
    return state;
}

} // namespace

static void moe_cache_add_telemetry(moe_cache_telemetry & dst, moe_cache_telemetry && src);
static bool moe_grouped_has_activity(const ggml_cuda_moe_grouped_debug_telemetry & telemetry) {
    return telemetry.covered != 0 || telemetry.plan_calls != 0 || telemetry.plan_compiles != 0 ||
           telemetry.plan_reuses != 0 || telemetry.calls != 0 || telemetry.ready != 0 || telemetry.completed != 0 ||
           telemetry.admitted_banks != 0 || telemetry.fallback != 0 || telemetry.rollback != 0 ||
           telemetry.host_staged_calls != 0 || telemetry.host_staged_ops != 0 || telemetry.host_staged_split_ops != 0 ||
           telemetry.strategy_switches != 0 || telemetry.required_unsupported != 0 || telemetry.prepare_error != 0 ||
           telemetry.finish_error != 0 || telemetry.h2d_banks != 0 || telemetry.h2d_bytes != 0 ||
           telemetry.decode_grouped != 0 || telemetry.decode_legacy != 0 || telemetry.submitted != 0;
}

static void moe_grouped_add_telemetry(
        ggml_cuda_moe_grouped_debug_telemetry & dst,
        const ggml_cuda_moe_grouped_debug_telemetry & src);
static void moe_cache_capture_telemetry(moe_cache_telemetry & dst, ggml_cuda_moe_cache * cache, bool reset);
static moe_cache_phase_stats moe_cache_take_op_stats(moe_cache_op_phase_stats & stats, bool reset);
static void ggml_cuda_moe_add_phase_stats(moe_cache_phase_stats & dst, const moe_cache_phase_stats & src);
static void moe_cache_log_telemetry(moe_cache_telemetry telemetry);

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
    uint32_t slot_index = UINT32_MAX;
};

struct moe_candidate_group_record {
    uint32_t layout = GGML_BACKEND_MOE_CANDIDATE_LAYOUT_INVALID;
    uint32_t domain = GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_INVALID;
    uint32_t semantic_group_index = UINT32_MAX;
    uint32_t flags = 0;
    const ggml_tensor * down = nullptr;
    uint64_t prefill_resident_auxiliary_bytes = 0;
    bool prefill_resident_auxiliary = false;
    std::vector<moe_candidate_bank_record> banks;
};

struct moe_candidate_reverse_entry {
    uint32_t group_index = UINT32_MAX;
    uint32_t bank_index = UINT32_MAX;
    uint32_t semantic_group_index = UINT32_MAX;
    uint32_t role = GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID;
    uint32_t status = GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_INVALID;
    uint32_t flags = 0;
    uint32_t group_flags = 0;
    uint32_t layout = GGML_BACKEND_MOE_CANDIDATE_LAYOUT_INVALID;
    uint32_t domain = GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_INVALID;
    uint32_t rejection = GGML_CUDA_MOE_CANDIDATE_REJECT_NONE;
    moe_candidate_bank_record descriptor;
    bool active = false;
    bool descriptor_valid = false;
};

struct moe_candidate_table {
    uint32_t n_slots = 0;
    uint32_t manifest_version = GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V1_VERSION;
    uint32_t submitted_flags = 0;
    uint32_t submitted_groups = 0;
    uint32_t submitted_tensors = 0;
    uint32_t active_weights = 0;
    uint64_t logical_signature = 0;
    uint64_t slot_bound_bytes = 0;
    uint64_t permanent_candidate_bytes = 0;
    bool prefill_resident_auxiliary = false;
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

static uint32_t moe_candidate_required_base_roles(uint32_t layout) {
    switch (layout) {
        case GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE:
            return (1u << GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT) |
                (1u << GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT) |
                (1u << GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT);
        case GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP:
            return (1u << GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT) |
                (1u << GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT);
        case GGML_BACKEND_MOE_CANDIDATE_LAYOUT_UNGATED:
            return (1u << GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT) |
                (1u << GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT);
        default:
            return 0;
    }
}

static uint32_t moe_candidate_required_base_bank_count(uint32_t layout) {
    switch (layout) {
        case GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE:      return 3;
        case GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP: return 2;
        case GGML_BACKEND_MOE_CANDIDATE_LAYOUT_UNGATED:       return 2;
        default:                                               return 0;
    }
}

static bool moe_candidate_routed_base(const ggml_backend_moe_candidate_tensor_v2 & record) {
    return record.status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE &&
        record.role >= GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT &&
        record.role <= GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT;
}

static bool moe_candidate_is_base_role(uint32_t role) {
    return role >= GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT &&
        role <= GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT;
}

static bool moe_candidate_slot_resource(const moe_candidate_bank_record & record) {
    return record.info.movement == GGML_CUDA_MOE_CANDIDATE_MOVEMENT_SLOT_BOUND &&
        (record.info.index_modes & GGML_CUDA_MOE_CANDIDATE_INDEX_GROUP_SLOT_DIRECT) != 0;
}

static bool moe_candidate_base_slot_resource(const moe_candidate_bank_record & record) {
    return moe_candidate_is_base_role(record.info.role) && moe_candidate_slot_resource(record);
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

static const moe_candidate_reverse_entry * moe_candidate_active_reverse(
        const moe_candidate_table & table,
        const ggml_tensor * tensor) {
    const auto it = table.reverse_map.find(tensor);
    if (it == table.reverse_map.end() || !it->second.active || it->second.group_index >= table.groups.size() ||
            it->second.bank_index >= table.groups[it->second.group_index].banks.size()) {
        return nullptr;
    }
    return &it->second;
}

static bool moe_candidate_reverse_matches(const moe_candidate_table & table, const moe_candidate_reverse_entry & entry, const ggml_tensor * tensor) {
    if (entry.active && entry.group_index < table.groups.size() && entry.bank_index < table.groups[entry.group_index].banks.size()) {
        return moe_candidate_record_matches(table.groups[entry.group_index].banks[entry.bank_index], tensor);
    }
    return entry.descriptor_valid && moe_candidate_record_matches(entry.descriptor, tensor);
}

static uint32_t moe_candidate_coverage_reason(const moe_candidate_reverse_entry & entry) {
    if (entry.active) {
        return GGML_CUDA_MOE_GRAPH_COVERAGE_REGISTERED;
    }
    if ((entry.flags & GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_TENSOR_OVERRIDES) != 0 ||
            (entry.group_flags & GGML_BACKEND_MOE_CANDIDATE_GROUP_V2_FLAG_TENSOR_OVERRIDES) != 0) {
        return GGML_CUDA_MOE_GRAPH_COVERAGE_TENSOR_OVERRIDE;
    }
    if ((entry.flags & GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_ACTIVE_LORA) != 0 ||
            (entry.group_flags & GGML_BACKEND_MOE_CANDIDATE_GROUP_V2_FLAG_ACTIVE_LORA) != 0) {
        return GGML_CUDA_MOE_GRAPH_COVERAGE_ACTIVE_LORA;
    }
    if ((entry.group_flags & GGML_BACKEND_MOE_CANDIDATE_GROUP_V2_FLAG_INCOMPLETE) != 0) {
        return GGML_CUDA_MOE_GRAPH_COVERAGE_INCOMPLETE;
    }
    if (entry.status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_UNCLASSIFIED) {
        return GGML_CUDA_MOE_GRAPH_COVERAGE_UNCLASSIFIED;
    }
    if (entry.status != GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE) {
        if (entry.status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_EXCLUDED_SHARED ||
                entry.status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_EXCLUDED_DENSE) {
            return GGML_CUDA_MOE_GRAPH_COVERAGE_EXCLUDED;
        }
        return GGML_CUDA_MOE_GRAPH_COVERAGE_NON_ROUTED_BASE;
    }
    if (entry.rejection != GGML_CUDA_MOE_CANDIDATE_REJECT_NONE) {
        return GGML_CUDA_MOE_GRAPH_COVERAGE_UNSUPPORTED_DESCRIPTOR;
    }
    return GGML_CUDA_MOE_GRAPH_COVERAGE_DORMANT_LAYOUT;
}

static const char * moe_candidate_coverage_reason_name(uint32_t reason) {
    switch (reason) {
        case GGML_CUDA_MOE_GRAPH_COVERAGE_REGISTERED:             return "registered";
        case GGML_CUDA_MOE_GRAPH_COVERAGE_REVERSE_MAP_MISS:       return "reverse_map_miss";
        case GGML_CUDA_MOE_GRAPH_COVERAGE_SOURCE_CHANGED:         return "source_changed";
        case GGML_CUDA_MOE_GRAPH_COVERAGE_INVALID_REVERSE_MAP:    return "invalid_reverse_map";
        case GGML_CUDA_MOE_GRAPH_COVERAGE_DORMANT_LAYOUT:         return "dormant_layout";
        case GGML_CUDA_MOE_GRAPH_COVERAGE_ACTIVE_LORA:            return "active_lora";
        case GGML_CUDA_MOE_GRAPH_COVERAGE_TENSOR_OVERRIDE:        return "tensor_override";
        case GGML_CUDA_MOE_GRAPH_COVERAGE_NON_ROUTED_BASE:        return "non_routed_base";
        case GGML_CUDA_MOE_GRAPH_COVERAGE_EXCLUDED:               return "excluded";
        case GGML_CUDA_MOE_GRAPH_COVERAGE_UNCLASSIFIED:           return "unclassified";
        case GGML_CUDA_MOE_GRAPH_COVERAGE_UNSUPPORTED_DESCRIPTOR: return "unsupported_descriptor";
        case GGML_CUDA_MOE_GRAPH_COVERAGE_INCOMPLETE:              return "incomplete";
        default:                                                    return "unknown";
    }
}

static const moe_candidate_bank_record * moe_candidate_find_role(const moe_candidate_group_record & group, uint32_t role) {
    for (const auto & bank : group.banks) {
        if (bank.info.role == role) {
            return &bank;
        }
    }
    return nullptr;
}

static uint32_t moe_candidate_base_slot_bank_count(const moe_candidate_group_record & group) {
    uint32_t result = 0;
    for (const auto & bank : group.banks) {
        result += moe_candidate_base_slot_resource(bank);
    }
    return result;
}

static const moe_candidate_bank_record * moe_candidate_base_slot_bank(
        const moe_candidate_group_record & group,
        uint32_t slot_bank_index) {
    uint32_t current = 0;
    for (const auto & bank : group.banks) {
        if (moe_candidate_base_slot_resource(bank)) {
            if (current == slot_bank_index) {
                return &bank;
            }
            ++current;
        }
    }
    return nullptr;
}

static uint32_t moe_candidate_expected_base_role(uint32_t layout, uint32_t bank_index) {
    if (layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP) {
        return bank_index == 0 ? GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT :
            bank_index == 1 ? GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT :
            GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID;
    }
    if (layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE) {
        return bank_index == 0 ? GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT :
            bank_index == 1 ? GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT :
            bank_index == 2 ? GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT :
            GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID;
    }
    if (layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_UNGATED) {
        return bank_index == 0 ? GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT :
            bank_index == 1 ? GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT :
            GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID;
    }
    return GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID;
}

static bool moe_candidate_original_direct_scale(
        const moe_candidate_bank_record & bank,
        uint32_t role,
        uint32_t n_experts) {
    return bank.info.role == role && bank.info.type == GGML_TYPE_F32 &&
        bank.info.encoding == GGML_CUDA_MOE_CANDIDATE_ENCODING_PLAIN &&
        bank.info.movement == GGML_CUDA_MOE_CANDIDATE_MOVEMENT_PERMANENT_CANDIDATE &&
        bank.info.index_modes == GGML_CUDA_MOE_CANDIDATE_INDEX_ORIGINAL_DIRECT &&
        ggml_n_dims(bank.info.tensor) == 1 && bank.slot_index == UINT32_MAX &&
        bank.ne[0] == n_experts && bank.ne[1] == 1 && bank.ne[2] == 1 && bank.ne[3] == 1 &&
        bank.info.byte_extent == (uint64_t) n_experts * sizeof(float) &&
        bank.info.byte_extent <= MOE_ORIGINAL_DIRECT_AUX_MAX_BYTES;
}

static bool moe_candidate_original_direct_bias(
        const moe_candidate_bank_record & bank,
        uint32_t role,
        const moe_candidate_bank_record & weight) {
    uint64_t row_bytes = 0;
    uint64_t byte_extent = 0;
    return bank.info.role == role && bank.info.type == GGML_TYPE_F32 &&
        bank.info.source_flags == 0 && bank.info.encoding == GGML_CUDA_MOE_CANDIDATE_ENCODING_PLAIN &&
        bank.info.movement == GGML_CUDA_MOE_CANDIDATE_MOVEMENT_PERMANENT_CANDIDATE &&
        bank.info.index_modes == GGML_CUDA_MOE_CANDIDATE_INDEX_ORIGINAL_DIRECT &&
        bank.info.expert_stride == 0 && ggml_n_dims(bank.info.tensor) == 2 && bank.slot_index == UINT32_MAX &&
        bank.ne[0] == weight.ne[1] && bank.ne[1] == weight.ne[2] && bank.ne[2] == 1 && bank.ne[3] == 1 &&
        moe_candidate_mul(sizeof(float), bank.ne[0], row_bytes) &&
        moe_candidate_mul(row_bytes, bank.ne[1], byte_extent) &&
        bank.nb[0] == sizeof(float) && bank.nb[1] == row_bytes && bank.nb[2] == byte_extent && bank.nb[3] == byte_extent &&
        bank.info.byte_extent == byte_extent;
}

static bool moe_candidate_structural_group(
        const moe_candidate_group_record & group,
        uint32_t * n_slot_banks,
        uint32_t * n_original_direct_aux = nullptr,
        uint32_t * n_original_direct_bias = nullptr) {
    const uint32_t required_roles = moe_candidate_required_base_roles(group.layout);
    const uint32_t required_banks = moe_candidate_required_base_bank_count(group.layout);
    if (required_banks == 0 || group.domain != GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY || group.flags != 0) {
        return false;
    }

    const auto * down = moe_candidate_find_role(group, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT);
    if (down == nullptr || down->ne[2] <= 0 || (uint64_t) down->ne[2] > UINT32_MAX) {
        return false;
    }
    const uint32_t n_experts = down->ne[2];
    uint32_t roles = 0;
    uint32_t slot_banks = 0;
    for (const auto & bank : group.banks) {
        const uint32_t role = bank.info.role;
        if (bank.info.movement == GGML_CUDA_MOE_CANDIDATE_MOVEMENT_SLOT_BOUND &&
                !moe_candidate_base_slot_resource(bank)) {
            return false;
        }
        if (moe_candidate_base_slot_resource(bank)) {
            if (role != moe_candidate_expected_base_role(group.layout, slot_banks) ||
                    (required_roles & (1u << role)) == 0 || (roles & (1u << role)) != 0 ||
                    bank.ne[2] != n_experts || bank.slot_index != slot_banks ||
                    (bank.info.source_flags & GGML_CUDA_MMID_SOURCE_ADVERTISED) == 0) {
                return false;
            }
            roles |= 1u << role;
            ++slot_banks;
        }
    }
    if (roles != required_roles || slot_banks != required_banks) {
        return false;
    }

    uint32_t scale_banks = 0;
    uint32_t scale_roles = 0;
    uint32_t bias_banks = 0;
    uint32_t bias_roles = 0;
    for (const auto & bank : group.banks) {
        if (bank.info.movement == GGML_CUDA_MOE_CANDIDATE_MOVEMENT_SLOT_BOUND) {
            continue;
        }
        const uint32_t role = bank.info.role;
        if (moe_candidate_is_scale(role)) {
            if ((scale_roles & (1u << role)) != 0 || !moe_candidate_original_direct_scale(bank, role, n_experts)) {
                return false;
            }
            scale_roles |= 1u << role;
            ++scale_banks;
            continue;
        }
        if (!moe_candidate_is_bias(role) || (bias_roles & (1u << role)) != 0) {
            return false;
        }
        const uint32_t base_role = moe_candidate_base_role(role);
        const auto * weight = moe_candidate_find_role(group, base_role);
        if (weight == nullptr || (required_roles & (1u << base_role)) == 0 ||
                !moe_candidate_original_direct_bias(bank, role, *weight)) {
            return false;
        }
        bias_roles |= 1u << role;
        ++bias_banks;
    }
    if (scale_banks != 0 && bias_banks != 0) {
        return false;
    }

    uint32_t original_direct_aux = 0;
    if (scale_banks != 0) {
        const uint32_t all_scale_roles = (1u << GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_SCALE) |
            (1u << GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_SCALE) |
            (1u << GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_SCALE);
        const bool fused_down_scale = group.layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP &&
            scale_banks == 1 && scale_roles == (1u << GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_SCALE);
        bool separate_nvfp4_scales = group.layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE &&
            scale_banks == 3 && scale_roles == all_scale_roles;
        for (uint32_t bank_index = 0; separate_nvfp4_scales && bank_index < slot_banks; ++bank_index) {
            const auto * bank = moe_candidate_base_slot_bank(group, bank_index);
            separate_nvfp4_scales = bank != nullptr && bank->info.type == GGML_TYPE_NVFP4 &&
                bank->info.encoding == GGML_CUDA_MOE_CANDIDATE_ENCODING_NVFP4_COMPOUND &&
                bank->info.index_modes == GGML_CUDA_MOE_CANDIDATE_INDEX_GROUP_SLOT_DIRECT;
        }
        if (!fused_down_scale && !separate_nvfp4_scales) {
            return false;
        }
        original_direct_aux = separate_nvfp4_scales ? 3 : 0;
    }
    if (n_slot_banks != nullptr) {
        *n_slot_banks = slot_banks;
    }
    if (n_original_direct_aux != nullptr) {
        *n_original_direct_aux = original_direct_aux;
    }
    if (n_original_direct_bias != nullptr) {
        *n_original_direct_bias = bias_banks;
    }
    return true;
}

static const moe_candidate_bank_record * moe_candidate_original_direct_aux_bank(
        const moe_candidate_group_record & group,
        uint32_t auxiliary_index) {
    const uint32_t role = auxiliary_index == 0 ? GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_SCALE :
        auxiliary_index == 1 ? GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_SCALE :
        auxiliary_index == 2 ? GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_SCALE :
        GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID;
    return role != GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID ? moe_candidate_find_role(group, role) : nullptr;
}

static uint32_t moe_candidate_output_bias_role(uint32_t reader_role) {
    switch (reader_role) {
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

static const moe_candidate_bank_record * moe_candidate_slot_auxiliary_bank(
        const moe_candidate_group_record & group,
        uint32_t auxiliary_index) {
    uint32_t n_slot_banks = 0;
    uint32_t n_scales = 0;
    uint32_t n_biases = 0;
    if (!moe_candidate_structural_group(group, &n_slot_banks, &n_scales, &n_biases)) {
        return nullptr;
    }
    if (auxiliary_index >= n_scales + n_biases) {
        return nullptr;
    }
    if (n_scales != 0) {
        return moe_candidate_original_direct_aux_bank(group, auxiliary_index);
    }
    uint32_t current = 0;
    for (uint32_t bank_index = 0; bank_index < n_slot_banks; ++bank_index) {
        const auto * base = moe_candidate_base_slot_bank(group, bank_index);
        const auto * bias = base != nullptr ? moe_candidate_find_role(group, moe_candidate_output_bias_role(base->info.role)) : nullptr;
        if (bias != nullptr && current++ == auxiliary_index) {
            return bias;
        }
    }
    return nullptr;
}

static void moe_candidate_set_prefill_resident_policy(moe_candidate_table & table, size_t byte_budget) {
    uint64_t total = 0;
    bool valid = true;
    for (auto & group : table.groups) {
        group.prefill_resident_auxiliary = false;
        group.prefill_resident_auxiliary_bytes = 0;
        uint32_t n_banks = 0;
        uint32_t n_scales = 0;
        uint32_t n_biases = 0;
        if (!moe_candidate_structural_group(group, &n_banks, &n_scales, &n_biases) || n_biases == 0 || n_scales != 0) {
            continue;
        }
        uint64_t group_bytes = 0;
        bool eligible = true;
        for (uint32_t auxiliary = 0; auxiliary < n_biases; ++auxiliary) {
            const auto * bank = moe_candidate_slot_auxiliary_bank(group, auxiliary);
            if (bank == nullptr || !moe_candidate_is_bias(bank->info.role) || bank->info.type != GGML_TYPE_F32 ||
                    !ggml_backend_buft_is_cuda_moe_cached(bank->buft)) {
                eligible = false;
                break;
            }
            if (!moe_candidate_add(group_bytes, bank->info.byte_extent, group_bytes)) {
                valid = false;
                break;
            }
        }
        if (!valid) {
            break;
        }
        if (!eligible) {
            continue;
        }
        if (group_bytes == 0 || !moe_candidate_add(total, group_bytes, total)) {
            valid = false;
            break;
        }
        group.prefill_resident_auxiliary_bytes = group_bytes;
    }
    valid = valid && total != 0 && total <= byte_budget;
    table.prefill_resident_auxiliary = valid;
    for (auto & group : table.groups) {
        group.prefill_resident_auxiliary = valid && group.prefill_resident_auxiliary_bytes != 0;
        if (!valid) {
            group.prefill_resident_auxiliary_bytes = 0;
        }
    }
}

static uint32_t moe_candidate_resource_bank_count(const moe_candidate_group_record & group) {
    return moe_candidate_base_slot_bank_count(group);
}

static const moe_candidate_bank_record * moe_candidate_resource_bank(
        const moe_candidate_group_record & group,
        uint32_t bank_index) {
    return moe_candidate_base_slot_bank(group, bank_index);
}

static ggml_cuda_moe_graph_capability_witness moe_candidate_capability(
        const ggml_tensor * tensor,
        const void * source_data,
        uint64_t byte_extent,
        uint64_t expert_stride,
        const int64_t * ne,
        const size_t * nb,
        uint32_t role,
        uint32_t source_type,
        uint32_t input_type,
        uint32_t output_type,
        int64_t n_tokens,
        uint32_t top_k,
        uint32_t n_rows,
        uint32_t n_routes,
        uint32_t row_stride,
        uint32_t n_slots,
        uint32_t row_semantics,
        uint32_t strategy,
        int device) {
    ggml_cuda_mmid_capability_query legacy_query;
    legacy_query.source_type = static_cast<ggml_type>(source_type);
    legacy_query.input_type = static_cast<ggml_type>(input_type);
    legacy_query.output_type = static_cast<ggml_type>(output_type);
    memcpy(legacy_query.source_ne, ne, sizeof(legacy_query.source_ne));
    memcpy(legacy_query.source_nb, nb, sizeof(legacy_query.source_nb));
    legacy_query.n_tokens = n_tokens;
    legacy_query.n_experts = ne[2];
    legacy_query.phase = n_rows == 1 ? GGML_CUDA_MMID_PHASE_DECODE : GGML_CUDA_MMID_PHASE_PREFILL;
    legacy_query.mapping = GGML_CUDA_MMID_MAPPING_DIRECT;
    legacy_query.use_mmq = ggml_cuda_moe_use_mmq(tensor, n_tokens);
    const auto & info = ggml_cuda_info();
    if (device >= 0 && device < info.device_count) {
        legacy_query.cc = info.devices[device].cc;
        legacy_query.warp_size = info.devices[device].warp_size;
        legacy_query.smpbo = info.devices[device].smpbo;
    }
    if (n_rows > 1 && !ggml_cuda_mmid_can_use_compact_mmvq(legacy_query, n_slots)) {
        legacy_query.mapping = GGML_CUDA_MMID_MAPPING_SOURCE_MAP;
    }
    const auto legacy = ggml_cuda_mmid_get_capability(legacy_query);

    ggml_cuda_mmid_capability_query grouped_query = legacy_query;
    grouped_query.phase = row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT ?
        GGML_CUDA_MMID_PHASE_DECODE : legacy_query.phase;
    grouped_query.preferred_consumer = legacy.selection;
    grouped_query.independent_rows = row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT;
    const bool device_direct = strategy == GGML_CUDA_MOE_EXECUTION_STRATEGY_DEVICE_DIRECT;
    const bool host_staged = strategy == GGML_CUDA_MOE_EXECUTION_STRATEGY_HOST_STAGED;
    grouped_query.mapping = GGML_CUDA_MMID_MAPPING_DIRECT;
    if (device_direct) {
        grouped_query.n_experts = n_slots;
        grouped_query.source_ne[2] = n_slots;
    }
    bool grouped_geometry_valid = device_direct && n_slots != 0 && grouped_query.source_nb[2] <= SIZE_MAX / n_slots;
    if (grouped_geometry_valid) {
        grouped_query.source_nb[3] = grouped_query.source_nb[2] * n_slots;
    }
    ggml_cuda_mmid_capability grouped = grouped_geometry_valid ?
        ggml_cuda_mmid_get_capability(grouped_query) : ggml_cuda_mmid_capability{};
    if (host_staged && n_slots != 0) {
        grouped_query = legacy_query;
        grouped_query.phase = row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT ?
            GGML_CUDA_MMID_PHASE_DECODE : legacy_query.phase;
        grouped_query.mapping = GGML_CUDA_MMID_MAPPING_SOURCE_MAP;
        grouped_query.preferred_consumer = legacy.selection;
        grouped_query.independent_rows = row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT;
        grouped_geometry_valid = true;
        grouped = ggml_cuda_mmid_get_capability(grouped_query);
        if (grouped.reason != GGML_CUDA_MMID_CAPABILITY_OK) {
            const uint32_t compact_experts = std::min<uint32_t>(n_routes, static_cast<uint32_t>(ne[2]));
            grouped_query = legacy_query;
            grouped_query.phase = row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT ?
                GGML_CUDA_MMID_PHASE_DECODE : legacy_query.phase;
            grouped_query.mapping = GGML_CUDA_MMID_MAPPING_DIRECT;
            grouped_query.preferred_consumer = legacy.selection;
            grouped_query.independent_rows = row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT;
            grouped_query.n_experts = compact_experts;
            grouped_query.source_ne[2] = compact_experts;
            grouped_geometry_valid = compact_experts != 0 && grouped_query.source_nb[2] <= SIZE_MAX / compact_experts;
            if (grouped_geometry_valid) {
                grouped_query.source_nb[3] = grouped_query.source_nb[2] * compact_experts;
                grouped = ggml_cuda_mmid_get_capability(grouped_query);
            }
        }
    }

    ggml_cuda_moe_graph_capability_witness result;
    result.tensor = tensor;
    result.source_data = source_data;
    result.byte_extent = byte_extent;
    result.expert_stride = expert_stride;
    memcpy(result.source_ne, legacy_query.source_ne, sizeof(result.source_ne));
    memcpy(result.source_nb, legacy_query.source_nb, sizeof(result.source_nb));
    memcpy(result.grouped_ne, grouped_query.source_ne, sizeof(result.grouped_ne));
    memcpy(result.grouped_nb, grouped_query.source_nb, sizeof(result.grouped_nb));
    result.n_tokens = legacy_query.n_tokens;
    result.n_experts = legacy_query.n_experts;
    result.smpbo = legacy_query.smpbo;
    result.device = device;
    result.cc = legacy_query.cc;
    result.warp_size = legacy_query.warp_size;
    result.role = role;
    result.source_type = source_type;
    result.source_flags = legacy.source.flags;
    result.input_type = input_type;
    result.output_type = output_type;
    result.phase = legacy_query.phase;
    result.mapping = legacy_query.mapping;
    result.materialized_phase = grouped_query.phase;
    result.materialized_mapping = grouped_query.mapping;
    result.row_semantics = row_semantics;
    result.strategy = strategy;
    result.consumer = legacy.selection;
    result.reason = legacy.reason;
    result.equivalence_reason = !grouped_geometry_valid ? GGML_CUDA_MMID_CAPABILITY_INVALID_GEOMETRY : grouped.reason;
    result.top_k = top_k;
    result.n_rows = n_rows;
    result.n_routes = n_routes;
    result.row_stride = row_stride;
    result.n_slots = n_slots;
    result.use_mmq = legacy_query.use_mmq;
    return result;
}

static bool moe_candidate_capability_supported(const ggml_cuda_moe_graph_capability_witness & capability) {
    return capability.reason == GGML_CUDA_MMID_CAPABILITY_OK &&
        capability.equivalence_reason == GGML_CUDA_MMID_CAPABILITY_OK &&
        capability.consumer != GGML_CUDA_MMID_CONSUMER_UNSUPPORTED &&
        capability.consumer != GGML_CUDA_MMID_CONSUMER_GENERIC;
}

static bool moe_candidate_capability_equivalence_unavailable(
        const ggml_cuda_moe_graph_capability_witness & capability) {
    return capability.reason == GGML_CUDA_MMID_CAPABILITY_OK &&
        (capability.consumer == GGML_CUDA_MMID_CONSUMER_UNSUPPORTED ||
            capability.consumer == GGML_CUDA_MMID_CONSUMER_GENERIC ||
            capability.equivalence_reason == GGML_CUDA_MMID_CAPABILITY_UNSUPPORTED_CONSUMER);
}

static bool moe_candidate_capability_invariant_valid(
        const ggml_cuda_moe_graph_capability_witness & capability) {
    return capability.reason == GGML_CUDA_MMID_CAPABILITY_OK &&
        (capability.equivalence_reason == GGML_CUDA_MMID_CAPABILITY_OK ||
            moe_candidate_capability_equivalence_unavailable(capability));
}

static const char * moe_candidate_capability_reason_name(uint32_t reason) {
    switch (reason) {
        case GGML_CUDA_MMID_CAPABILITY_OK:                   return "OK";
        case GGML_CUDA_MMID_CAPABILITY_UNADVERTISED_SOURCE:  return "UNADVERTISED_SOURCE";
        case GGML_CUDA_MMID_CAPABILITY_UNSUPPORTED_CONSUMER: return "UNSUPPORTED_CONSUMER";
        case GGML_CUDA_MMID_CAPABILITY_UNSUPPORTED_MAPPING:  return "UNSUPPORTED_MAPPING";
        case GGML_CUDA_MMID_CAPABILITY_INVALID_IO:           return "INVALID_IO";
        case GGML_CUDA_MMID_CAPABILITY_INVALID_GEOMETRY:     return "INVALID_GEOMETRY";
        case GGML_CUDA_MMID_CAPABILITY_INVALID_PHASE:        return "INVALID_PHASE";
        case GGML_CUDA_MMID_CAPABILITY_INVALID_MAPPING:      return "INVALID_MAPPING";
        case GGML_CUDA_MMID_CAPABILITY_INVALID_DEVICE:       return "INVALID_DEVICE";
    }
    return "UNKNOWN";
}

static bool moe_candidate_capability_matches(
        const ggml_cuda_moe_graph_capability_witness & capability,
        const moe_candidate_bank_record & bank,
        const ggml_tensor * node,
        uint32_t top_k,
        uint32_t n_rows,
        uint32_t n_routes,
        uint32_t row_stride,
        uint32_t n_slots,
        int device) {
    if (node == nullptr || node->src[1] == nullptr) {
        return false;
    }
    const auto current = moe_candidate_capability(
        bank.info.tensor, bank.info.source_data, bank.info.byte_extent, bank.info.expert_stride, bank.ne, bank.nb,
        bank.info.role, bank.info.type, node->src[1]->type, node->type, node->src[1]->ne[2],
        top_k, n_rows, n_routes, row_stride, n_slots,
        capability.row_semantics, capability.strategy, device);
    return capability.tensor == current.tensor && capability.source_data == current.source_data &&
        capability.byte_extent == current.byte_extent && capability.expert_stride == current.expert_stride &&
        memcmp(capability.source_ne, current.source_ne, sizeof(capability.source_ne)) == 0 &&
        memcmp(capability.source_nb, current.source_nb, sizeof(capability.source_nb)) == 0 &&
        memcmp(capability.grouped_ne, current.grouped_ne, sizeof(capability.grouped_ne)) == 0 &&
        memcmp(capability.grouped_nb, current.grouped_nb, sizeof(capability.grouped_nb)) == 0 &&
        capability.n_tokens == current.n_tokens && capability.n_experts == current.n_experts &&
        capability.smpbo == current.smpbo && capability.device == current.device && capability.cc == current.cc &&
        capability.warp_size == current.warp_size && capability.role == current.role &&
        capability.source_type == current.source_type && capability.source_flags == current.source_flags &&
        capability.input_type == current.input_type && capability.output_type == current.output_type &&
        capability.phase == current.phase && capability.mapping == current.mapping &&
        capability.materialized_phase == current.materialized_phase &&
        capability.materialized_mapping == current.materialized_mapping &&
        capability.row_semantics == current.row_semantics && capability.strategy == current.strategy &&
        capability.consumer == current.consumer &&
        capability.reason == current.reason && capability.equivalence_reason == current.equivalence_reason &&
        capability.top_k == current.top_k && capability.n_rows == current.n_rows &&
        capability.n_routes == current.n_routes && capability.row_stride == current.row_stride &&
        capability.n_slots == current.n_slots &&
        capability.use_mmq == current.use_mmq;
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

static bool moe_candidate_execution_certificate_present(const ggml_cgraph * cgraph) {
    static const ggml_graph_execution_certificate empty = {};
    return cgraph != nullptr && memcmp(&cgraph->execution_certificate, &empty, sizeof(empty)) != 0;
}

static bool moe_candidate_execution_certificate(
        const ggml_cgraph * cgraph,
        ggml_graph_execution_certificate * certificate,
        uint64_t * semantic_key) {
    if (certificate != nullptr) {
        *certificate = {};
    }
    if (semantic_key != nullptr) {
        *semantic_key = 0;
    }
    if (cgraph == nullptr) {
        return false;
    }

    const auto & current = cgraph->execution_certificate;
    if (current.magic != GGML_GRAPH_EXECUTION_CERTIFICATE_MAGIC ||
            current.abi_version != GGML_GRAPH_EXECUTION_CERTIFICATE_VERSION ||
            current.struct_size != sizeof(current) ||
            (current.flags & ~GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED) != 0 ||
            current.domain < GGML_GRAPH_EXECUTION_DOMAIN_MAIN || current.domain > GGML_GRAPH_EXECUTION_DOMAIN_MTP ||
            current.row_semantics < GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT ||
            current.row_semantics > GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SPECULATIVE ||
            current.n_rows == 0 || current.n_sequences == 0 || current.owner_namespace == 0 || current.owner_generation == 0 ||
            current.source_graph_uid == 0 || current.split_graph_uid == 0 || current.source_graph_uid == current.split_graph_uid ||
            cgraph->uid == 0 || current.split_graph_uid != cgraph->uid) {
        return false;
    }
    for (uint64_t value : current.reserved) {
        if (value != 0) {
            return false;
        }
    }

    uint64_t key = UINT64_C(1469598103934665603);
    moe_candidate_hash_value(key, current.magic);
    moe_candidate_hash_value(key, current.abi_version);
    moe_candidate_hash_value(key, current.struct_size);
    moe_candidate_hash_value(key, current.flags);
    moe_candidate_hash_value(key, current.domain);
    moe_candidate_hash_value(key, current.row_semantics);
    moe_candidate_hash_value(key, current.n_rows);
    moe_candidate_hash_value(key, current.n_sequences);
    moe_candidate_hash_value(key, current.owner_namespace);
    moe_candidate_hash_value(key, current.owner_generation);
    if (certificate != nullptr) {
        *certificate = current;
    }
    if (semantic_key != nullptr) {
        *semantic_key = key != 0 ? key : 1;
    }
    return true;
}

static bool moe_candidate_execution_semantics_equal(
        ggml_graph_execution_certificate first,
        ggml_graph_execution_certificate second) {
    first.source_graph_uid = 0;
    first.split_graph_uid = 0;
    second.source_graph_uid = 0;
    second.split_graph_uid = 0;
    return memcmp(&first, &second, sizeof(first)) == 0;
}

enum moe_candidate_execution_phase : uint32_t {
    MOE_CANDIDATE_EXECUTION_PHASE_UNKNOWN = 0,
    MOE_CANDIDATE_EXECUTION_PHASE_DECODE,
    MOE_CANDIDATE_EXECUTION_PHASE_PREFILL,
    MOE_CANDIDATE_EXECUTION_PHASE_SPECULATIVE,
};

static moe_candidate_execution_phase moe_candidate_execution_phase_for(
        const ggml_cgraph * cgraph,
        const ggml_tensor * node) {
    ggml_graph_execution_certificate certificate;
    if (moe_candidate_execution_certificate(cgraph, &certificate, nullptr)) {
        switch (certificate.row_semantics) {
            case GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT: return MOE_CANDIDATE_EXECUTION_PHASE_DECODE;
            case GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL:  return MOE_CANDIDATE_EXECUTION_PHASE_PREFILL;
            case GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SPECULATIVE: return MOE_CANDIDATE_EXECUTION_PHASE_SPECULATIVE;
        }
    }
    if (moe_candidate_execution_certificate_present(cgraph)) {
        return MOE_CANDIDATE_EXECUTION_PHASE_UNKNOWN;
    }
    const ggml_tensor * ids = node != nullptr && node->op == GGML_OP_MUL_MAT_ID ? node->src[2] : nullptr;
    return ids != nullptr && (ids->ne[1] != 1 || ids->ne[2] != 1) ?
        MOE_CANDIDATE_EXECUTION_PHASE_PREFILL : MOE_CANDIDATE_EXECUTION_PHASE_DECODE;
}

struct moe_candidate_execution_geometry {
    uint64_t semantic_key = 0;
    uint32_t top_k = 0;
    uint32_t n_rows = 0;
    uint32_t n_routes = 0;
    uint32_t row_stride = 0;
    uint32_t row_semantics = GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INVALID;
    uint32_t strategy = GGML_CUDA_MOE_EXECUTION_STRATEGY_INVALID;
    moe_candidate_execution_phase phase = MOE_CANDIDATE_EXECUTION_PHASE_UNKNOWN;
    bool certificate_valid = false;
    bool required_grouped = false;
    bool tensor_valid = false;
    bool grouped_eligible = false;
};

static moe_candidate_execution_geometry moe_candidate_execution_geometry_for(
        const ggml_cgraph * cgraph,
        const ggml_tensor * node,
        int64_t n_experts,
        uint32_t n_slots) {
    moe_candidate_execution_geometry result;
    ggml_graph_execution_certificate certificate;
    result.certificate_valid = moe_candidate_execution_certificate(cgraph, &certificate, &result.semantic_key);
    result.required_grouped = result.certificate_valid &&
        certificate.flags == GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED;
    result.phase = moe_candidate_execution_phase_for(cgraph, node);
    result.row_semantics = result.certificate_valid ? certificate.row_semantics :
        static_cast<uint32_t>(GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INVALID);

    const ggml_tensor * activation = node != nullptr && node->op == GGML_OP_MUL_MAT_ID ? node->src[1] : nullptr;
    const ggml_tensor * ids = node != nullptr && node->op == GGML_OP_MUL_MAT_ID ? node->src[2] : nullptr;
    if (!moe_candidate_ids_valid(ids) || activation == nullptr || n_experts <= 0 || n_experts > INT32_MAX ||
            ids->ne[0] <= 0 || ids->ne[0] > n_experts || ids->ne[0] > UINT32_MAX ||
            ids->ne[1] <= 0 || ids->ne[1] > UINT32_MAX || ids->ne[2] != 1 || ids->ne[3] != 1 ||
            ids->nb[0] != sizeof(int32_t) || ids->nb[1] % sizeof(int32_t) != 0 ||
            static_cast<uint64_t>(ids->ne[0]) > SIZE_MAX / sizeof(int32_t)) {
        return result;
    }

    const size_t row_bytes = static_cast<size_t>(ids->ne[0]) * sizeof(int32_t);
    if (ids->nb[1] < row_bytes || ids->nb[1] / sizeof(int32_t) > UINT32_MAX ||
            static_cast<uint64_t>(ids->ne[1]) > SIZE_MAX / ids->nb[1] ||
            ids->nb[2] < static_cast<size_t>(ids->ne[1]) * ids->nb[1] || ids->nb[3] < ids->nb[2] ||
            activation->ne[2] != ids->ne[1] || activation->ne[3] != 1 ||
            node->ne[2] != ids->ne[1] || node->ne[3] != 1 ||
            static_cast<uint64_t>(ids->ne[0]) > UINT32_MAX / static_cast<uint64_t>(ids->ne[1])) {
        return result;
    }

    result.top_k = static_cast<uint32_t>(ids->ne[0]);
    result.n_rows = static_cast<uint32_t>(ids->ne[1]);
    result.n_routes = result.top_k * result.n_rows;
    result.row_stride = static_cast<uint32_t>(ids->nb[1] / sizeof(int32_t));
    result.tensor_valid = true;
    const bool required = result.required_grouped;
    const bool main_independent = certificate.flags == GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_NONE &&
        certificate.domain == GGML_GRAPH_EXECUTION_DOMAIN_MAIN &&
        certificate.row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT &&
        certificate.n_sequences == result.n_rows;
    const bool draft_independent = required && certificate.domain == GGML_GRAPH_EXECUTION_DOMAIN_DRAFT &&
        certificate.row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT &&
        certificate.n_sequences == result.n_rows;
    const bool draft_sequential = required && certificate.domain == GGML_GRAPH_EXECUTION_DOMAIN_DRAFT &&
        certificate.row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL &&
        certificate.n_sequences <= result.n_rows;
    const bool mtp_independent = required && certificate.domain == GGML_GRAPH_EXECUTION_DOMAIN_MTP &&
        certificate.row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT &&
        certificate.n_sequences == result.n_rows;
    const bool mtp_sequential = required && certificate.domain == GGML_GRAPH_EXECUTION_DOMAIN_MTP &&
        certificate.row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL &&
        certificate.n_sequences <= result.n_rows;
    const bool target_speculative = required && certificate.domain == GGML_GRAPH_EXECUTION_DOMAIN_MAIN &&
        certificate.row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SPECULATIVE &&
        certificate.n_sequences < result.n_rows;
    result.grouped_eligible = result.certificate_valid && result.n_rows <= certificate.n_rows &&
        result.n_rows >= 1 && n_slots > 0 && n_slots <= INT32_MAX &&
        ((main_independent && result.n_routes <= n_slots) || draft_independent || draft_sequential ||
            mtp_independent || mtp_sequential || target_speculative);
    if (result.grouped_eligible) {
        result.strategy = result.n_routes <= n_slots ?
            GGML_CUDA_MOE_EXECUTION_STRATEGY_DEVICE_DIRECT : GGML_CUDA_MOE_EXECUTION_STRATEGY_HOST_STAGED;
    }
    return result;
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

static bool moe_candidate_signature_matches(const ggml_cuda_moe_ids_signature & signature, const ggml_tensor * tensor) {
    return tensor != nullptr ? moe_candidate_ids_equal(signature, moe_candidate_ids_signature(tensor)) : signature.tensor == nullptr;
}

static bool moe_candidate_graph_use(
        const ggml_cgraph * cgraph,
        const ggml_tensor * tensor,
        bool & present,
        int32_t & use_count) {
    present = false;
    use_count = 0;
    if (cgraph == nullptr || tensor == nullptr || cgraph->visited_hash_set.size == 0) {
        return false;
    }
    const size_t hash_pos = ggml_hash_find(&cgraph->visited_hash_set, tensor);
    if (hash_pos == GGML_HASHSET_FULL || !ggml_bitset_get(cgraph->visited_hash_set.used, hash_pos)) {
        return true;
    }
    present = true;
    use_count = cgraph->use_counts[hash_pos];
    return use_count >= 0;
}

static bool moe_candidate_auxiliary_consumer(
        const ggml_tensor * consumer,
        const ggml_tensor * producer,
        const ggml_tensor * ids) {
    if (consumer == nullptr || producer == nullptr) {
        return false;
    }
    for (uint32_t src_index = 0; src_index < 2; ++src_index) {
        if (consumer->src[src_index] != producer) {
            continue;
        }
        if (consumer->op == GGML_OP_ADD || consumer->op == GGML_OP_ADD_ID) {
            return true;
        }
        const ggml_tensor * other = consumer->src[1 - src_index];
        const ggml_tensor * repeat = other != nullptr && other->op == GGML_OP_GET_ROWS ? other->src[0] : nullptr;
        if (consumer->op == GGML_OP_MUL && other != nullptr && other->op == GGML_OP_GET_ROWS && other->src[1] == ids &&
                repeat != nullptr && repeat->op == GGML_OP_REPEAT && repeat->src[0] != nullptr && repeat->src[0]->op == GGML_OP_RESHAPE) {
            return true;
        }
    }
    return false;
}

enum moe_candidate_auxiliary_kind : uint32_t {
    MOE_CANDIDATE_AUXILIARY_NONE = 0,
    MOE_CANDIDATE_AUXILIARY_SCALE,
    MOE_CANDIDATE_AUXILIARY_BIAS,
};

struct moe_candidate_route_proof {
    ggml_cuda_moe_ids_signature ids;
    ggml_cuda_moe_ids_signature root;
    ggml_cuda_moe_ids_signature source;
    ggml_cuda_moe_ids_signature indices;
    uint32_t root_node_index = 0;
    uint32_t ids_node_index = 0;
};

static bool moe_candidate_graph_node_before(
        const ggml_cgraph * cgraph,
        const ggml_tensor * tensor,
        uint32_t before,
        uint32_t * node_index) {
    for (uint32_t i = 0; i < before; ++i) {
        if (ggml_graph_node(const_cast<ggml_cgraph *>(cgraph), i) == tensor) {
            *node_index = i;
            return true;
        }
    }
    return false;
}

static uint32_t moe_candidate_output_scale_role(
        const moe_candidate_group_record & group,
        uint32_t reader_role) {
    if (group.layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP) {
        return reader_role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT ?
            GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_SCALE : GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID;
    }
    uint32_t n_slot_banks = 0;
    uint32_t n_original_direct_aux = 0;
    if (group.layout != GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE ||
            !moe_candidate_structural_group(group, &n_slot_banks, &n_original_direct_aux) ||
            n_slot_banks != 3 || n_original_direct_aux != 3 ||
            reader_role < GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT ||
            reader_role > GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT) {
        return GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID;
    }
    return GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_SCALE +
        reader_role - GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT;
}

static bool moe_candidate_single_graph_use(const ggml_cgraph * cgraph, const ggml_tensor * tensor) {
    bool present = false;
    int32_t use_count = 0;
    return moe_candidate_graph_use(cgraph, tensor, present, use_count) && present && use_count == 1;
}

static bool moe_candidate_f32_metadata_matches(const ggml_tensor * tensor, const int64_t * ne) {
    if (tensor == nullptr || tensor->type != GGML_TYPE_F32) {
        return false;
    }
    size_t nb = sizeof(float);
    for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
        if (ne[dim] <= 0 || tensor->ne[dim] != ne[dim] || tensor->nb[dim] != nb ||
                static_cast<uint64_t>(ne[dim]) > SIZE_MAX / nb) {
            return false;
        }
        nb *= ne[dim];
    }
    return true;
}

static bool moe_candidate_original_direct_scale_consumer(
        const ggml_cgraph * cgraph,
        uint32_t consumer_node_index,
        const ggml_tensor * consumer,
        const ggml_tensor * producer,
        const ggml_tensor * ids,
        uint32_t reader_role,
        const moe_candidate_group_record & group,
        uint32_t * bank_index,
        const ggml_tensor ** auxiliary_nodes,
        uint32_t * auxiliary_node_indices) {
    const uint32_t scale_role = moe_candidate_output_scale_role(group, reader_role);
    if (consumer == nullptr || producer == nullptr || ids == nullptr || consumer->op != GGML_OP_MUL ||
            consumer->src[0] != producer || scale_role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID) {
        return false;
    }
    const ggml_tensor * rows = consumer->src[1];
    const ggml_tensor * repeat = rows != nullptr ? rows->src[0] : nullptr;
    const ggml_tensor * reshape = repeat != nullptr ? repeat->src[0] : nullptr;
    if (rows == nullptr || rows->op != GGML_OP_GET_ROWS || rows->src[1] != ids || repeat == nullptr || repeat->op != GGML_OP_REPEAT ||
            reshape == nullptr || reshape->op != GGML_OP_RESHAPE || reshape->src[0] == nullptr || rows->type != GGML_TYPE_F32 ||
            reshape->ne[0] != 1 || reshape->ne[1] <= 0 || reshape->ne[2] != 1 || reshape->ne[3] != 1 ||
            repeat->ne[0] != 1 || repeat->ne[1] != reshape->ne[1] || repeat->ne[2] != producer->ne[2] || repeat->ne[3] != 1) {
        return false;
    }

    uint32_t reshape_node_index = 0;
    uint32_t repeat_node_index = 0;
    uint32_t rows_node_index = 0;
    if (!moe_candidate_graph_node_before(cgraph, reshape, consumer_node_index, &reshape_node_index) ||
            !moe_candidate_graph_node_before(cgraph, repeat, consumer_node_index, &repeat_node_index) ||
            !moe_candidate_graph_node_before(cgraph, rows, consumer_node_index, &rows_node_index) ||
            reshape_node_index >= repeat_node_index || repeat_node_index >= rows_node_index ||
            !moe_candidate_single_graph_use(cgraph, reshape) || !moe_candidate_single_graph_use(cgraph, repeat) ||
            !moe_candidate_single_graph_use(cgraph, rows)) {
        return false;
    }

    for (uint32_t i = 0; i < group.banks.size(); ++i) {
        const auto & bank = group.banks[i];
        if (bank.info.role == scale_role &&
                bank.info.movement == GGML_CUDA_MOE_CANDIDATE_MOVEMENT_PERMANENT_CANDIDATE &&
                bank.info.index_modes == GGML_CUDA_MOE_CANDIDATE_INDEX_ORIGINAL_DIRECT &&
                bank.info.type == GGML_TYPE_F32 && bank.info.encoding == GGML_CUDA_MOE_CANDIDATE_ENCODING_PLAIN &&
                reshape->ne[1] == bank.ne[0] && moe_candidate_record_matches(bank, reshape->src[0])) {
            const int64_t reshape_ne[GGML_MAX_DIMS] = {1, bank.ne[0], 1, 1};
            const int64_t repeat_ne[GGML_MAX_DIMS] = {1, bank.ne[0], producer->ne[2], 1};
            const int64_t rows_ne[GGML_MAX_DIMS] = {1, ids->ne[0], ids->ne[1], ids->ne[2]};
            if (group.layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE &&
                    (!moe_candidate_f32_metadata_matches(reshape, reshape_ne) ||
                        !moe_candidate_f32_metadata_matches(repeat, repeat_ne) ||
                        !moe_candidate_f32_metadata_matches(rows, rows_ne))) {
                return false;
            }
            *bank_index = i;
            auxiliary_nodes[0] = reshape;
            auxiliary_nodes[1] = repeat;
            auxiliary_nodes[2] = rows;
            auxiliary_node_indices[0] = reshape_node_index;
            auxiliary_node_indices[1] = repeat_node_index;
            auxiliary_node_indices[2] = rows_node_index;
            return true;
        }
    }
    return false;
}

static bool moe_candidate_original_direct_scale_witness_matches(
        const ggml_cgraph * cgraph,
        uint32_t consumer_node_index,
        const ggml_tensor * consumer,
        const ggml_tensor * producer,
        const ggml_tensor * ids,
        uint32_t reader_role,
        const moe_candidate_group_record & group,
        uint32_t bank_index,
        const ggml_tensor * const * auxiliary_nodes,
        const uint32_t * auxiliary_node_indices) {
    const uint32_t scale_role = moe_candidate_output_scale_role(group, reader_role);
    if (cgraph == nullptr || consumer == nullptr || producer == nullptr || ids == nullptr ||
            consumer->op != GGML_OP_MUL || consumer->src[0] != producer ||
            scale_role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID ||
            bank_index >= group.banks.size() || auxiliary_node_indices[0] >= auxiliary_node_indices[1] ||
            auxiliary_node_indices[1] >= auxiliary_node_indices[2] || auxiliary_node_indices[2] >= consumer_node_index ||
            consumer_node_index >= static_cast<uint32_t>(cgraph->n_nodes)) {
        return false;
    }
    for (uint32_t i = 0; i < 3; ++i) {
        if (auxiliary_node_indices[i] >= static_cast<uint32_t>(cgraph->n_nodes) ||
                ggml_graph_node(const_cast<ggml_cgraph *>(cgraph), auxiliary_node_indices[i]) != auxiliary_nodes[i]) {
            return false;
        }
    }

    const ggml_tensor * reshape = auxiliary_nodes[0];
    const ggml_tensor * repeat = auxiliary_nodes[1];
    const ggml_tensor * rows = auxiliary_nodes[2];
    const auto & bank = group.banks[bank_index];
    const int64_t reshape_ne[GGML_MAX_DIMS] = {1, bank.ne[0], 1, 1};
    const int64_t repeat_ne[GGML_MAX_DIMS] = {1, bank.ne[0], producer->ne[2], 1};
    const int64_t rows_ne[GGML_MAX_DIMS] = {1, ids->ne[0], ids->ne[1], ids->ne[2]};
    const bool metadata_matches = group.layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE ?
        moe_candidate_f32_metadata_matches(reshape, reshape_ne) &&
            moe_candidate_f32_metadata_matches(repeat, repeat_ne) && moe_candidate_f32_metadata_matches(rows, rows_ne) :
        rows != nullptr && rows->type == GGML_TYPE_F32 &&
            reshape != nullptr && reshape->ne[0] == 1 && reshape->ne[1] > 0 && reshape->ne[2] == 1 && reshape->ne[3] == 1 &&
            repeat != nullptr && repeat->ne[0] == 1 && repeat->ne[1] == reshape->ne[1] &&
            repeat->ne[2] == producer->ne[2] && repeat->ne[3] == 1;
    return reshape != nullptr && reshape->op == GGML_OP_RESHAPE && reshape->src[0] != nullptr &&
        repeat != nullptr && repeat->op == GGML_OP_REPEAT && repeat->src[0] == reshape &&
        rows != nullptr && rows->op == GGML_OP_GET_ROWS && rows->src[0] == repeat && rows->src[1] == ids && consumer->src[1] == rows &&
        moe_candidate_single_graph_use(cgraph, reshape) && moe_candidate_single_graph_use(cgraph, repeat) &&
        moe_candidate_single_graph_use(cgraph, rows) &&
        metadata_matches &&
        bank.info.role == scale_role &&
        bank.info.movement == GGML_CUDA_MOE_CANDIDATE_MOVEMENT_PERMANENT_CANDIDATE &&
        bank.info.index_modes == GGML_CUDA_MOE_CANDIDATE_INDEX_ORIGINAL_DIRECT && bank.info.type == GGML_TYPE_F32 &&
        bank.info.encoding == GGML_CUDA_MOE_CANDIDATE_ENCODING_PLAIN && reshape->ne[1] == bank.ne[0] &&
        moe_candidate_record_matches(bank, reshape->src[0]);
}

static bool moe_candidate_original_direct_bias_consumer(
        const ggml_tensor * consumer,
        const ggml_tensor * producer,
        const ggml_tensor * ids,
        uint32_t reader_role,
        const moe_candidate_group_record & group,
        uint32_t * bank_index) {
    const uint32_t bias_role = moe_candidate_output_bias_role(reader_role);
    if (consumer == nullptr || producer == nullptr || ids == nullptr || consumer->op != GGML_OP_ADD_ID ||
            consumer->src[0] != producer || consumer->src[1] == nullptr || consumer->src[2] != ids ||
            bias_role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID ||
            !moe_candidate_f32_metadata_matches(producer, producer->ne) ||
            !moe_candidate_f32_metadata_matches(consumer, producer->ne)) {
        return false;
    }
    for (uint32_t src_index = 3; src_index < GGML_MAX_SRC; ++src_index) {
        if (consumer->src[src_index] != nullptr) {
            return false;
        }
    }
    for (uint32_t i = 0; i < group.banks.size(); ++i) {
        const auto & bank = group.banks[i];
        if (bank.info.role == bias_role &&
                bank.info.movement == GGML_CUDA_MOE_CANDIDATE_MOVEMENT_PERMANENT_CANDIDATE &&
                bank.info.index_modes == GGML_CUDA_MOE_CANDIDATE_INDEX_ORIGINAL_DIRECT &&
                bank.info.type == GGML_TYPE_F32 && bank.info.encoding == GGML_CUDA_MOE_CANDIDATE_ENCODING_PLAIN &&
                moe_candidate_record_matches(bank, consumer->src[1])) {
            *bank_index = i;
            return true;
        }
    }
    return false;
}

static bool moe_candidate_original_direct_bias_witness_matches(
        const ggml_tensor * consumer,
        const ggml_tensor * producer,
        const ggml_tensor * ids,
        uint32_t reader_role,
        const moe_candidate_group_record & group,
        uint32_t bank_index) {
    uint32_t current_bank_index = 0;
    return moe_candidate_original_direct_bias_consumer(
        consumer, producer, ids, reader_role, group, &current_bank_index) && current_bank_index == bank_index;
}

static bool moe_candidate_prefill_add_id_witness(
        const ggml_tensor * consumer,
        const moe_candidate_group_record & group,
        uint32_t * bank_index,
        uint32_t * reader_role) {
    if (consumer == nullptr || consumer->op != GGML_OP_ADD_ID || consumer->src[0] == nullptr ||
            consumer->src[1] == nullptr || consumer->src[2] == nullptr ||
            consumer->src[0]->op != GGML_OP_MUL_MAT_ID || consumer->src[0]->src[0] == nullptr ||
            consumer->src[0]->src[2] != consumer->src[2]) {
        return false;
    }
    for (const auto & bank : group.banks) {
        if (!moe_candidate_is_bias(bank.info.role) || bank.info.tensor != consumer->src[1]) {
            continue;
        }
        const uint32_t role = moe_candidate_base_role(bank.info.role);
        const auto * base = moe_candidate_find_role(group, role);
        if (base == nullptr || !moe_candidate_record_matches(*base, consumer->src[0]->src[0]) ||
                !moe_candidate_original_direct_bias_witness_matches(
                    consumer, consumer->src[0], consumer->src[2], role, group,
                    static_cast<uint32_t>(&bank - group.banks.data()))) {
            return false;
        }
        *bank_index = static_cast<uint32_t>(&bank - group.banks.data());
        *reader_role = role;
        return true;
    }
    return false;
}

static bool moe_candidate_validate_route(
        const ggml_tensor * ids,
        int64_t n_experts,
        moe_candidate_route_proof & proof) {
    if (moe_candidate_ids_valid(ids) && ids->op == GGML_OP_GET_ROWS) {
        const ggml_tensor * table = ids->src[0];
        const ggml_tensor * indices = ids->src[1];
        if (n_experts <= 0 || ids->ne[0] > n_experts || ids->ne[2] != 1 || ids->ne[3] != 1 ||
                (ids->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || !ggml_is_contiguous(ids) ||
                ids->view_src != nullptr || ids->view_offs != 0 ||
                !moe_candidate_ids_valid(table) || table->op != GGML_OP_NONE || table->view_src != nullptr ||
                table->view_offs != 0 || !ggml_is_contiguous(table) ||
                ggml_backend_buffer_get_usage(table->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
                table->ne[0] != ids->ne[0] || table->ne[2] != 1 || table->ne[3] != 1 ||
                !moe_candidate_ids_valid(indices) || !ggml_is_contiguous(indices) ||
                indices->ne[0] != ids->ne[1] || indices->ne[1] != 1 || indices->ne[2] != 1 || indices->ne[3] != 1) {
            return false;
        }
        for (int i = 2; i < GGML_MAX_SRC; ++i) {
            if (ids->src[i] != nullptr) {
                return false;
            }
        }
        proof.ids = moe_candidate_ids_signature(ids);
        proof.root = proof.ids;
        proof.source = moe_candidate_ids_signature(table);
        proof.indices = moe_candidate_ids_signature(indices);
        return true;
    }
    if (!moe_candidate_ids_valid(ids) || ids->op != GGML_OP_VIEW || (ids->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
            ids->ne[0] <= 0 || ids->ne[1] <= 0 || ids->ne[2] != 1 || ids->ne[3] != 1 || ids->nb[0] != sizeof(int32_t) ||
            ids->src[0] == nullptr || ids->view_src != ids->src[0] || ids->view_offs != 0) {
        return false;
    }
    size_t op_view_offs = 0;
    memcpy(&op_view_offs, ids->op_params, sizeof(op_view_offs));
    if (op_view_offs != 0) {
        return false;
    }

    const ggml_tensor * root = ids->src[0];
    const ggml_tensor * source = root->src[0];
    if (n_experts <= 0 || root->op != GGML_OP_ARGSORT || (root->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
            root->type != GGML_TYPE_I32 || root->buffer == nullptr || root->data == nullptr || root->view_src != nullptr || root->view_offs != 0 ||
            source == nullptr || source->type != GGML_TYPE_F32 || source->buffer == nullptr || source->data == nullptr ||
            ggml_get_op_params_i32(root, 0) != GGML_SORT_ORDER_DESC || root->ne[0] != n_experts || ids->ne[0] > root->ne[0] ||
            root->nb[0] != sizeof(int32_t) || !ggml_is_contiguous(root) || ids->buffer != root->buffer || ids->data != root->data) {
        return false;
    }
    for (int i = 1; i < GGML_MAX_SRC; ++i) {
        if (ids->src[i] != nullptr || root->src[i] != nullptr) {
            return false;
        }
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (ids->nb[i] != root->nb[i] || root->ne[i] != source->ne[i] || (i > 0 && ids->ne[i] != root->ne[i])) {
            return false;
        }
    }

    proof.ids = moe_candidate_ids_signature(ids);
    proof.root = moe_candidate_ids_signature(root);
    proof.source = moe_candidate_ids_signature(source);
    return true;
}

static bool moe_candidate_discover_route(
        const ggml_cgraph * cgraph,
        uint32_t consumer_node_index,
        moe_candidate_route_proof & proof) {
    if (!moe_candidate_graph_node_before(cgraph, proof.root.tensor, consumer_node_index, &proof.root_node_index) ||
            !moe_candidate_graph_node_before(cgraph, proof.ids.tensor, consumer_node_index, &proof.ids_node_index) ||
            proof.root_node_index > proof.ids_node_index) {
        return false;
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

    const auto source = ggml_cuda_mmid_source_capability_for(tensor->type);
    if ((source.flags & GGML_CUDA_MMID_SOURCE_ADVERTISED) == 0) {
        return GGML_CUDA_MOE_CANDIDATE_REJECT_UNSUPPORTED_TYPE;
    }

    uint32_t encoding = GGML_CUDA_MOE_CANDIDATE_ENCODING_PLAIN;
    uint32_t index_modes = GGML_CUDA_MOE_CANDIDATE_INDEX_GROUP_SLOT_DIRECT |
        GGML_CUDA_MOE_CANDIDATE_INDEX_ORIGINAL_SOURCE_MAP;
    if (tensor->type == GGML_TYPE_NVFP4) {
        encoding = GGML_CUDA_MOE_CANDIDATE_ENCODING_NVFP4_COMPOUND;
        index_modes = GGML_CUDA_MOE_CANDIDATE_INDEX_GROUP_SLOT_DIRECT;
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
    record.info.source_flags = source.flags;
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
    moe_candidate_hash_value(hash, bank.info.source_flags);
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

    uint32_t base_roles = 0;
    for (uint32_t role = GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT;
            role <= GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT; ++role) {
        if (by_role[role] != nullptr) {
            base_roles |= 1u << role;
        }
    }
    const uint32_t required_base_roles = moe_candidate_required_base_roles(input.layout);
    if (required_base_roles == 0 || base_roles != required_base_roles) {
        return GGML_CUDA_MOE_CANDIDATE_REJECT_INVALID_LAYOUT;
    }

    moe_candidate_group_record group;
    group.layout = input.layout;
    group.domain = GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY;
    group.semantic_group_index = group_index;
    group.banks.reserve(input.n_banks);
    const ggml_tensor * weights[GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_COUNT] = {};
    std::array<moe_candidate_bank_record, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_COUNT> records = {};

    for (uint32_t role = GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT; role <= GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT; ++role) {
        if (by_role[role] == nullptr) {
            continue;
        }
        const auto rejection = moe_candidate_weight(owner, by_role[role]->tensor, role, group_index, records[role]);
        if (rejection != GGML_CUDA_MOE_CANDIDATE_REJECT_NONE) {
            return rejection;
        }
        weights[role] = by_role[role]->tensor;
    }

    const ggml_tensor * down = weights[GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT];
    if (input.layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE) {
        const ggml_tensor * gate = weights[GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT];
        const ggml_tensor * up = weights[GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT];
        if (gate->ne[0] != up->ne[0] || gate->ne[1] != up->ne[1] || gate->ne[2] != up->ne[2] ||
                down->ne[0] != gate->ne[1] || down->ne[1] != gate->ne[0] || down->ne[2] != gate->ne[2]) {
            return GGML_CUDA_MOE_CANDIDATE_REJECT_INCOMPATIBLE_SHAPE;
        }
    } else if (input.layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP) {
        const ggml_tensor * gate_up = weights[GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT];
        uint64_t doubled_down = 0;
        if (!moe_candidate_mul(down->ne[0], 2, doubled_down) || gate_up->ne[0] != down->ne[1] ||
                static_cast<uint64_t>(gate_up->ne[1]) != doubled_down || gate_up->ne[2] != down->ne[2]) {
            return GGML_CUDA_MOE_CANDIDATE_REJECT_INCOMPATIBLE_SHAPE;
        }
    } else {
        const ggml_tensor * up = weights[GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT];
        if (down->ne[0] != up->ne[1] || down->ne[1] != up->ne[0] || down->ne[2] != up->ne[2]) {
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
        const auto rejection = moe_candidate_aux(owner, by_role[role]->tensor, weights[base_role], role, group_index, records[role]);
        if (rejection != GGML_CUDA_MOE_CANDIDATE_REJECT_NONE) {
            return rejection;
        }
    }

    uint32_t slot_index = 0;
    for (uint32_t input_index = 0; input_index < input.n_banks; ++input_index) {
        auto record = records[input.banks[input_index].role];
        if (moe_candidate_slot_resource(record)) {
            record.slot_index = slot_index++;
        }
        group.banks.push_back(record);
    }

    uint64_t group_hash = UINT64_C(1469598103934665603);
    moe_candidate_hash_value(group_hash, group.layout);
    const uint32_t n_banks = group.banks.size();
    moe_candidate_hash_value(group_hash, n_banks);
    for (uint32_t role = GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT; role < GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_COUNT; ++role) {
        const auto * bank = moe_candidate_find_role(group, role);
        if (bank != nullptr) {
            moe_candidate_hash_bank(group_hash, *bank);
        }
    }
    for (uint32_t bank_index = 0; bank_index < group.banks.size(); ++bank_index) {
        auto & bank = group.banks[bank_index];
        if (moe_candidate_slot_resource(bank)) {
            uint64_t bytes = 0;
            if (!moe_candidate_mul(bank.info.expert_stride, n_slots, bytes) ||
                    !moe_candidate_add(table.slot_bound_bytes, bytes, table.slot_bound_bytes)) {
                return GGML_CUDA_MOE_CANDIDATE_REJECT_OVERFLOW;
            }
            moe_candidate_reverse_entry entry;
            entry.group_index = group_index;
            entry.bank_index = bank_index;
            entry.semantic_group_index = group_index;
            entry.role = bank.info.role;
            entry.status = GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE;
            entry.layout = group.layout;
            entry.domain = GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY;
            entry.descriptor = bank;
            entry.active = true;
            entry.descriptor_valid = true;
            const auto inserted = table.reverse_map.emplace(bank.info.tensor, std::move(entry));
            if (!inserted.second) {
                return GGML_CUDA_MOE_CANDIDATE_REJECT_DUPLICATE_TENSOR;
            }
            ++table.active_weights;
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
    table.submitted_groups = snapshot.n_groups;
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

static bool moe_candidate_v2_role_status(uint32_t role, uint32_t status) {
    if (status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE) {
        return role >= GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT && role <= GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT;
    }
    if (status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_OUTPUT_SCALE) {
        return moe_candidate_is_scale(role);
    }
    if (status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_OUTPUT_BIAS) {
        return moe_candidate_is_bias(role);
    }
    if (status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_INPUT_SCALE) {
        return role >= GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_INPUT_SCALE &&
            role <= GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_INPUT_SCALE;
    }
    return (status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_UNCLASSIFIED ||
            status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_EXCLUDED_SHARED ||
            status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_EXCLUDED_DENSE) &&
        role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID;
}

static ggml_cuda_moe_candidate_rejection moe_candidate_build_v2(
        ggml_backend_dev_t owner,
        const ggml_backend_moe_candidate_snapshot_v2 & snapshot,
        moe_candidate_table & table) {
    const uint32_t snapshot_flags = GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_FLAG_TENSOR_OVERRIDES |
        GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_FLAG_INCOMPLETE;
    const uint32_t group_flags = GGML_BACKEND_MOE_CANDIDATE_GROUP_V2_FLAG_ACTIVE_LORA |
        GGML_BACKEND_MOE_CANDIDATE_GROUP_V2_FLAG_TENSOR_OVERRIDES |
        GGML_BACKEND_MOE_CANDIDATE_GROUP_V2_FLAG_INCOMPLETE;
    const uint32_t tensor_flags = GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_CACHED_BUFFER |
        GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_ACTIVE_LORA |
        GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_TENSOR_OVERRIDES;
    if ((snapshot.flags & ~snapshot_flags) != 0 || snapshot.reserved32 != 0 || snapshot.reserved[0] != 0 || snapshot.reserved[1] != 0 ||
            snapshot.n_groups > GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS || snapshot.n_tensors > GGML_BACKEND_MOE_CANDIDATE_MAX_TENSORS_V2 ||
            (snapshot.n_groups > 0 && snapshot.groups == nullptr) || (snapshot.n_tensors > 0 && snapshot.tensors == nullptr)) {
        return GGML_CUDA_MOE_CANDIDATE_REJECT_INVALID_COUNT;
    }

    std::vector<std::vector<const ggml_backend_moe_candidate_tensor_v2 *>> records(snapshot.n_groups);
    std::unordered_set<const ggml_tensor *> tensors;
    tensors.reserve(snapshot.n_tensors);
    for (uint32_t group_index = 0; group_index < snapshot.n_groups; ++group_index) {
        const auto & group = snapshot.groups[group_index];
        if (group.reserved != 0 || (group.flags & ~group_flags) != 0 ||
                (group.layout != GGML_BACKEND_MOE_CANDIDATE_LAYOUT_INVALID && group.layout != GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE &&
                 group.layout != GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP && group.layout != GGML_BACKEND_MOE_CANDIDATE_LAYOUT_UNGATED) ||
                (group.domain != GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY && group.domain != GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_CHUNK)) {
            return GGML_CUDA_MOE_CANDIDATE_REJECT_INVALID_FLAGS;
        }
    }
    for (uint32_t tensor_index = 0; tensor_index < snapshot.n_tensors; ++tensor_index) {
        const auto & record = snapshot.tensors[tensor_index];
        if (record.tensor == nullptr || record.reserved != 0 || (record.flags & ~tensor_flags) != 0 ||
                !moe_candidate_v2_role_status(record.role, record.status) || !tensors.insert(record.tensor).second) {
            return GGML_CUDA_MOE_CANDIDATE_REJECT_INVALID_TENSOR;
        }
        const bool ungrouped = record.status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_UNCLASSIFIED ||
            record.status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_EXCLUDED_SHARED ||
            record.status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_EXCLUDED_DENSE;
        if ((ungrouped && record.group_index != UINT32_MAX) || (!ungrouped && record.group_index >= snapshot.n_groups)) {
            return GGML_CUDA_MOE_CANDIDATE_REJECT_INVALID_ROLE;
        }
        const bool cached = record.tensor->buffer != nullptr &&
            ggml_backend_buft_is_cuda_moe_cached(ggml_backend_buffer_get_type(record.tensor->buffer));
        if (cached != ((record.flags & GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_CACHED_BUFFER) != 0)) {
            return GGML_CUDA_MOE_CANDIDATE_REJECT_INVALID_TENSOR;
        }
        if (!ungrouped) {
            records[record.group_index].push_back(&record);
        }
    }

    struct active_group_storage {
        uint32_t semantic_group_index;
        std::vector<ggml_backend_moe_candidate_bank_v1> banks;
    };
    std::vector<active_group_storage> active_storage;
    std::vector<uint32_t> group_rejections(snapshot.n_groups, GGML_CUDA_MOE_CANDIDATE_REJECT_NONE);
    std::vector<uint32_t> active_group_indices(snapshot.n_groups, UINT32_MAX);
    active_storage.reserve(snapshot.n_groups);
    uint32_t inherited_group_flags = 0;
    if ((snapshot.flags & GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_FLAG_TENSOR_OVERRIDES) != 0) {
        inherited_group_flags |= GGML_BACKEND_MOE_CANDIDATE_GROUP_V2_FLAG_TENSOR_OVERRIDES;
    }
    if ((snapshot.flags & GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_FLAG_INCOMPLETE) != 0) {
        inherited_group_flags |= GGML_BACKEND_MOE_CANDIDATE_GROUP_V2_FLAG_INCOMPLETE;
    }
    for (uint32_t group_index = 0; group_index < snapshot.n_groups; ++group_index) {
        uint32_t routed_roles = 0;
        for (const auto * record : records[group_index]) {
            if (!moe_candidate_routed_base(*record)) {
                continue;
            }
            const uint32_t role = 1u << record->role;
            if ((routed_roles & role) != 0) {
                group_rejections[group_index] = GGML_CUDA_MOE_CANDIDATE_REJECT_DUPLICATE_ROLE;
                break;
            }
            routed_roles |= role;
        }
        const uint32_t required_roles = moe_candidate_required_base_roles(snapshot.groups[group_index].layout);
        if (group_rejections[group_index] == GGML_CUDA_MOE_CANDIDATE_REJECT_NONE && routed_roles != 0 && routed_roles != required_roles) {
            group_rejections[group_index] = GGML_CUDA_MOE_CANDIDATE_REJECT_INVALID_LAYOUT;
        }
    }
    const bool snapshot_dormant = snapshot.flags != GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_FLAG_NONE;
    for (uint32_t group_index = 0; group_index < snapshot.n_groups; ++group_index) {
        const auto & group = snapshot.groups[group_index];
        if (group_rejections[group_index] != GGML_CUDA_MOE_CANDIDATE_REJECT_NONE || snapshot_dormant ||
                group.flags != GGML_BACKEND_MOE_CANDIDATE_GROUP_V2_FLAG_NONE ||
                group.domain != GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY ||
                moe_candidate_required_base_bank_count(group.layout) == 0) {
            continue;
        }

        active_group_storage prospective;
        prospective.semantic_group_index = group_index;
        bool bases_cached = true;
        bool association_valid = true;
        bool current_policy_supported = true;
        for (const auto * record : records[group_index]) {
            if (record->status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_INPUT_SCALE) {
                continue;
            }
            if (record->status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE) {
                bases_cached = bases_cached &&
                    (record->flags & GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_CACHED_BUFFER) != 0;
            } else {
                const uint32_t base_role = moe_candidate_base_role(record->role);
                const bool has_base = std::any_of(records[group_index].begin(), records[group_index].end(), [&](const auto * base) {
                    return base->status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE && base->role == base_role;
                });
                association_valid = association_valid && has_base;
            }
            current_policy_supported = current_policy_supported &&
                record->role != GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_SCALE;
            prospective.banks.push_back({record->tensor, record->role, 0});
        }
        if (!current_policy_supported) {
            group_rejections[group_index] = GGML_CUDA_MOE_CANDIDATE_REJECT_UNSUPPORTED_ROLE;
            continue;
        }
        if (!bases_cached || !association_valid) {
            group_rejections[group_index] = GGML_CUDA_MOE_CANDIDATE_REJECT_INVALID_ROLE;
            continue;
        }

        ggml_backend_moe_candidate_group_v1 input = {
            prospective.banks.data(), static_cast<uint32_t>(prospective.banks.size()), group.layout, 0, 0,
        };
        ggml_backend_moe_candidate_snapshot_v1 trial_snapshot = {};
        trial_snapshot.magic = GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V1_MAGIC;
        trial_snapshot.abi_version = GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V1_VERSION;
        trial_snapshot.struct_size = sizeof(trial_snapshot);
        trial_snapshot.n_slots = snapshot.n_slots;
        trial_snapshot.n_groups = 1;
        trial_snapshot.groups = &input;
        moe_candidate_table trial;
        const auto rejection = moe_candidate_build(owner, trial_snapshot, trial);
        group_rejections[group_index] = rejection;
        if (rejection == GGML_CUDA_MOE_CANDIDATE_REJECT_NONE) {
            active_group_indices[group_index] = active_storage.size();
            active_storage.push_back(std::move(prospective));
        }
    }

    std::vector<ggml_backend_moe_candidate_group_v1> active_groups;
    active_groups.reserve(active_storage.size());
    for (const auto & group : active_storage) {
        const auto & semantic = snapshot.groups[group.semantic_group_index];
        active_groups.push_back({group.banks.data(), static_cast<uint32_t>(group.banks.size()), semantic.layout, 0, 0});
    }
    ggml_backend_moe_candidate_snapshot_v1 active_snapshot = {};
    active_snapshot.magic = GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V1_MAGIC;
    active_snapshot.abi_version = GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V1_VERSION;
    active_snapshot.struct_size = sizeof(active_snapshot);
    active_snapshot.n_slots = snapshot.n_slots;
    active_snapshot.n_groups = active_groups.size();
    active_snapshot.groups = active_groups.empty() ? nullptr : active_groups.data();
    const auto active_rejection = moe_candidate_build(owner, active_snapshot, table);
    if (active_rejection != GGML_CUDA_MOE_CANDIDATE_REJECT_NONE) {
        return active_rejection;
    }
    table.manifest_version = GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_VERSION;
    table.submitted_flags = snapshot.flags;
    table.submitted_groups = snapshot.n_groups;
    table.submitted_tensors = snapshot.n_tensors;
    for (uint32_t active_group = 0; active_group < active_storage.size(); ++active_group) {
        const uint32_t semantic_group = active_storage[active_group].semantic_group_index;
        table.groups[active_group].domain = snapshot.groups[semantic_group].domain;
        table.groups[active_group].semantic_group_index = semantic_group;
        table.groups[active_group].flags = inherited_group_flags | snapshot.groups[semantic_group].flags;
    }

    for (uint32_t tensor_index = 0; tensor_index < snapshot.n_tensors; ++tensor_index) {
        const auto & record = snapshot.tensors[tensor_index];
        if ((record.flags & GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_CACHED_BUFFER) == 0) {
            continue;
        }
        const uint32_t active_group = record.group_index < active_group_indices.size() ? active_group_indices[record.group_index] : UINT32_MAX;
        auto reverse = table.reverse_map.find(record.tensor);
        if (reverse == table.reverse_map.end()) {
            moe_candidate_reverse_entry entry;
            entry.group_index = active_group;
            entry.semantic_group_index = record.group_index;
            entry.role = record.role;
            entry.status = record.status;
            entry.flags = record.flags;
            entry.group_flags = inherited_group_flags;
            if (record.group_index < snapshot.n_groups) {
                entry.layout = snapshot.groups[record.group_index].layout;
                entry.domain = snapshot.groups[record.group_index].domain;
                entry.group_flags |= snapshot.groups[record.group_index].flags;
                entry.rejection = group_rejections[record.group_index];
            }
            if (active_group < table.groups.size()) {
                for (uint32_t bank_index = 0; bank_index < table.groups[active_group].banks.size(); ++bank_index) {
                    if (table.groups[active_group].banks[bank_index].info.tensor == record.tensor) {
                        entry.bank_index = bank_index;
                        break;
                    }
                }
            }
            ggml_cuda_moe_candidate_rejection descriptor_rejection;
            if (moe_candidate_routed_base(record)) {
                descriptor_rejection = moe_candidate_weight(owner, record.tensor, record.role, active_group, entry.descriptor);
            } else {
                entry.descriptor.info.type = record.tensor->type;
                entry.descriptor.info.role = record.role;
                entry.descriptor.info.group_index = active_group;
                descriptor_rejection = moe_candidate_source(owner, record.tensor, ggml_nbytes(record.tensor), entry.descriptor);
            }
            entry.descriptor_valid = descriptor_rejection == GGML_CUDA_MOE_CANDIDATE_REJECT_NONE;
            if (entry.rejection == GGML_CUDA_MOE_CANDIDATE_REJECT_NONE && !entry.descriptor_valid) {
                entry.rejection = descriptor_rejection;
            }
            reverse = table.reverse_map.emplace(record.tensor, std::move(entry)).first;
        } else {
            reverse->second.semantic_group_index = record.group_index;
            reverse->second.role = record.role;
            reverse->second.status = record.status;
            reverse->second.flags = record.flags;
            reverse->second.layout = snapshot.groups[record.group_index].layout;
            reverse->second.domain = snapshot.groups[record.group_index].domain;
            reverse->second.group_flags = inherited_group_flags | snapshot.groups[record.group_index].flags;
            reverse->second.rejection = group_rejections[record.group_index];
        }
    }
    return GGML_CUDA_MOE_CANDIDATE_REJECT_NONE;
}

} // namespace

uint64_t ggml_cuda_moe_execution_semantic_key(const ggml_cgraph * cgraph) {
    uint64_t result = 0;
    (void) moe_candidate_execution_certificate(cgraph, nullptr, &result);
    return result;
}

namespace {

static constexpr uint32_t MOE_GROUPED_PLAN_THREADS = 128;
static constexpr uint32_t MOE_GROUPED_FREQUENCY_EPOCH_SHIFT = 4;
static constexpr uint32_t MOE_GROUPED_TRANSFER_THREADS = 256;
static constexpr uint32_t MOE_GROUPED_TRANSFER_BLOCKS_PER_SM = 4;

static uint32_t moe_grouped_plan_threads(uint32_t n_slots) {
    return n_slots <= WARP_SIZE ? WARP_SIZE : n_slots <= 2 * WARP_SIZE ? 2 * WARP_SIZE : MOE_GROUPED_PLAN_THREADS;
}

static uint32_t moe_grouped_transfer_blocks(
        int device,
        size_t words_per_miss,
        size_t auxiliary_values_per_miss,
        uint32_t n_routes) {
    const size_t values_per_miss = std::max(words_per_miss, auxiliary_values_per_miss);
    const size_t blocks_per_miss = values_per_miss / MOE_GROUPED_TRANSFER_THREADS +
        (values_per_miss % MOE_GROUPED_TRANSFER_THREADS != 0);
    const size_t blocks_needed = blocks_per_miss > SIZE_MAX / n_routes ? SIZE_MAX : blocks_per_miss * n_routes;
    const auto & info = ggml_cuda_info();
    const size_t device_blocks = info.devices[device].nsm > 0 ?
        static_cast<size_t>(info.devices[device].nsm) * MOE_GROUPED_TRANSFER_BLOCKS_PER_SM : blocks_needed;
    const size_t grid_limit = info.devices[device].max_grid_size[0] > 0 ?
        static_cast<size_t>(info.devices[device].max_grid_size[0]) : static_cast<size_t>(UINT32_MAX);
    return static_cast<uint32_t>(std::max<size_t>(
        1, std::min({blocks_needed, device_blocks, grid_limit, static_cast<size_t>(UINT32_MAX)})));
}

enum moe_grouped_plan_status : uint32_t {
    MOE_GROUPED_PLAN_BUILDING = 0,
    MOE_GROUPED_PLAN_READY,
    MOE_GROUPED_PLAN_INVALID_ROUTE,
    MOE_GROUPED_PLAN_INVALID_STATE,
};

struct moe_grouped_decode_plan {
    uint32_t status;
    uint32_t n_routes;
    uint32_t n_unique;
    uint32_t n_misses;
    uint64_t next_clock;
    uint64_t frequency_epoch;
};

enum moe_grouped_plan_array_index : uint32_t {
    MOE_GROUPED_PLAN_ROUTE_STORAGE = 0,
    MOE_GROUPED_PLAN_ROUTE_UNIQUE,
    MOE_GROUPED_PLAN_UNIQUE_EXPERTS,
    MOE_GROUPED_PLAN_UNIQUE_SLOTS,
    MOE_GROUPED_PLAN_MISS_UNIQUE,
    MOE_GROUPED_PLAN_MISS_EXPERTS,
    MOE_GROUPED_PLAN_MISS_SLOTS,
    MOE_GROUPED_PLAN_REMAPPED_IDS,
    MOE_GROUPED_PLAN_ROUTE_ARRAY_COUNT,
};

static __host__ __device__ int32_t * moe_grouped_plan_array_ptr(
        moe_grouped_decode_plan * plan, uint32_t capacity, moe_grouped_plan_array_index array) {
    return reinterpret_cast<int32_t *>(plan + 1) + static_cast<size_t>(array) * capacity;
}

static __host__ __device__ const int32_t * moe_grouped_plan_array_ptr(
        const moe_grouped_decode_plan * plan, uint32_t capacity, moe_grouped_plan_array_index array) {
    return reinterpret_cast<const int32_t *>(plan + 1) + static_cast<size_t>(array) * capacity;
}

static __host__ __device__ uint32_t * moe_grouped_plan_expert_routes(
        moe_grouped_decode_plan * plan, uint32_t capacity) {
    return reinterpret_cast<uint32_t *>(moe_grouped_plan_array_ptr(plan, capacity, MOE_GROUPED_PLAN_ROUTE_ARRAY_COUNT));
}

static bool moe_grouped_plan_size(uint32_t capacity, uint32_t n_experts, size_t * size) {
    if (size == nullptr || capacity == 0 || n_experts == 0 ||
            capacity > (SIZE_MAX - n_experts) / MOE_GROUPED_PLAN_ROUTE_ARRAY_COUNT) {
        return false;
    }
    const size_t values = static_cast<size_t>(capacity) * MOE_GROUPED_PLAN_ROUTE_ARRAY_COUNT + n_experts;
    if (values > (SIZE_MAX - sizeof(moe_grouped_decode_plan)) / sizeof(int32_t)) {
        return false;
    }
    *size = sizeof(moe_grouped_decode_plan) + values * sizeof(int32_t);
    return true;
}

struct moe_grouped_device_bank {
    const char * source;
    char * data;
    size_t expert_stride;
    bool staged = false;
};

static constexpr uint32_t MOE_PREPACK_ROUTES = 32;
static bool moe_early_router_enabled();

struct moe_prepack_control {
    int32_t predicted[MOE_PREPACK_ROUTES];
    const char * adopted[MOE_PREPACK_ROUTES];
};

static bool moe_grouped_staging_layout(size_t plan_bytes, size_t payload_bytes, bool prediction,
        size_t & control_offset, size_t & staging_offset, size_t & allocation_bytes) {
    if (plan_bytes > SIZE_MAX - (alignof(uint4) - 1)) {
        return false;
    }
    control_offset = GGML_PAD(plan_bytes, alignof(uint4));
    const size_t control_bytes = prediction ? sizeof(moe_prepack_control) : 0;
    if (control_offset > SIZE_MAX - control_bytes) {
        return false;
    }
    staging_offset = control_offset + control_bytes;
    if (payload_bytes > SIZE_MAX - staging_offset) {
        return false;
    }
    allocation_bytes = staging_offset + payload_bytes;
    return true;
}

struct moe_prepack_request {
    uint64_t resource = 0;
    uint32_t top_k = 0;
    uint32_t n_experts = 0;
    size_t expert_bytes = 0;
    std::array<const moe_host_source *, GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS> sources = {};
};

struct moe_host_prepack {
    enum state { FREE, QUEUED, COPYING, READY, ADOPTED };
    enum rank_counter { COPIED_BYTES, ADOPTED_BYTES, WRONG_BYTES, LATE_BYTES, N_RANK_COUNTERS };
    struct tile {
        state status = FREE;
        moe_prepack_request request;
        uint64_t invocation = 0;
        std::array<int32_t, MOE_PREPACK_ROUTES> experts;
        std::atomic<bool> expired{false};
        char * data = nullptr;
        const char * alias = nullptr;
        size_t copied = 0;
        size_t proposed = 0;
        uint32_t copied_ranks = 0;
    };

    moe_host_prepack(void * data, const void * alias, size_t tile_bytes) : tile_bytes(tile_bytes) {
        for (size_t i = 0; i < tiles.size(); ++i) {
            tiles[i].data = static_cast<char *>(data) + i * tile_bytes;
            tiles[i].alias = static_cast<const char *>(alias) + i * tile_bytes;
        }
    }

    ~moe_host_prepack() {
        close_admission();
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop = true;
            for (auto & tile : tiles) {
                tile.expired.store(true, std::memory_order_relaxed);
                if (tile.status == READY) {
                    late_bytes += tile.copied;
                    record_ranks(tile, tile.copied_ranks, LATE_BYTES);
                    tile.status = FREE;
                } else if (tile.status == QUEUED) {
                    dropped_bytes.fetch_add(tile.proposed, std::memory_order_relaxed);
                    tile.status = FREE;
                }
            }
        }
        wake.notify_one();
        if (worker.joinable()) {
            worker.join();
        }
        fprintf(stderr, "moe-prepack: published=%llu dropped=%llu dropped_bytes=%llu callbacks=%llu adopted_bytes=%llu wrong_bytes=%llu late_bytes=%llu copied_bytes=%llu demand_bytes=%llu peak_tiles=%u\n",
            (unsigned long long) published, (unsigned long long) dropped.load(), (unsigned long long) dropped_bytes.load(), (unsigned long long) callbacks.load(),
            (unsigned long long) adopted_bytes, (unsigned long long) wrong_bytes, (unsigned long long) late_bytes,
            (unsigned long long) copied_bytes, (unsigned long long) demand_bytes.load(), peak_tiles);
        for (uint32_t rank = 0; rank < MOE_PREPACK_ROUTES; ++rank) {
            if (proposed_by_rank[rank].load() != 0) {
                fprintf(stderr, "moe-prepack-rank: rank=%u proposed_bytes=%llu copied_bytes=%llu adopted_bytes=%llu wrong_bytes=%llu late_bytes=%llu\n", rank,
                    (unsigned long long) proposed_by_rank[rank].load(), (unsigned long long) rank_bytes[rank][COPIED_BYTES],
                    (unsigned long long) rank_bytes[rank][ADOPTED_BYTES], (unsigned long long) rank_bytes[rank][WRONG_BYTES],
                    (unsigned long long) rank_bytes[rank][LATE_BYTES]);
            }
        }
    }

    void record_ranks(const tile & item, uint32_t mask, rank_counter counter) {
        for (uint32_t rank = 0; rank < item.request.top_k; ++rank) {
            if ((mask & (uint32_t(1) << rank)) != 0) {
                rank_bytes[rank][counter] += item.request.expert_bytes;
            }
        }
    }

    void close_admission() {
        admission_closed.store(true, std::memory_order_release);
    }

    void start() {
        worker = std::thread([this] {
            std::unique_lock<std::mutex> lock(mutex);
            for (;;) {
                wake.wait(lock, [&] { return stop || queued() != nullptr; });
                if (stop) {
                    return;
                }
                auto & tile = *queued();
                tile.status = COPYING;
                lock.unlock();
                pack(tile);
                lock.lock();
                copied_bytes += tile.copied;
                record_ranks(tile, tile.copied_ranks, COPIED_BYTES);
                if (tile.expired.load(std::memory_order_relaxed)) {
                    late_bytes += tile.copied;
                    record_ranks(tile, tile.copied_ranks, LATE_BYTES);
                    dropped_bytes.fetch_add(tile.proposed - tile.copied, std::memory_order_relaxed);
                    tile.status = FREE;
                } else {
                    tile.status = READY;
                }
                wake.notify_all();
            }
        });
    }

    tile * queued() {
        for (auto & tile : tiles) {
            if (tile.status == QUEUED) {
                return &tile;
            }
        }
        return nullptr;
    }

    static void pack(tile & tile) {
        tile.copied = 0;
        tile.copied_ranks = 0;
        for (uint32_t i = 0; i < tile.request.top_k && !tile.expired.load(std::memory_order_relaxed); ++i) {
            const int32_t expert = tile.experts[i];
            if (expert < 0) {
                continue;
            }
            size_t offset = i * tile.request.expert_bytes;
            for (const auto * source : tile.request.sources) {
                if (source != nullptr) {
                    GGML_ASSERT(static_cast<uint32_t>(expert) < tile.request.n_experts);
                    GGML_ASSERT(moe_host_materialize(*source, source->data + size_t(expert) * source->expert_stride, source->expert_stride, tile.data + offset));
                    tile.copied += source->expert_stride;
                    offset += source->expert_stride;
                }
            }
            tile.copied_ranks |= uint32_t(1) << i;
        }
    }

    void visit(uint64_t resource, uint64_t invocation, const int32_t * misses, uint32_t count,
            const moe_prepack_request & next, moe_prepack_control & control) {
        callbacks.fetch_add(1, std::memory_order_relaxed);
        std::fill_n(control.adopted, MOE_PREPACK_ROUTES, nullptr);
        GGML_ASSERT(count <= MOE_PREPACK_ROUTES && next.top_k <= MOE_PREPACK_ROUTES);
        size_t proposed = 0;
        for (uint32_t p = 0; p < next.top_k; ++p) {
            if (control.predicted[p] >= 0) {
                if (uint32_t(control.predicted[p]) >= next.n_experts) {
                    dropped.fetch_add(1, std::memory_order_relaxed);
                    return;
                }
                proposed += next.expert_bytes;
                proposed_by_rank[p].fetch_add(next.expert_bytes, std::memory_order_relaxed);
            }
        }
        std::unique_lock<std::mutex> lock(mutex, std::try_to_lock);
        if (!lock.owns_lock() || admission_closed.load(std::memory_order_acquire)) {
            dropped.fetch_add(1, std::memory_order_relaxed);
            dropped_bytes.fetch_add(proposed, std::memory_order_relaxed);
            return;
        }
        for (auto & tile : tiles) {
            // This callback follows every prior adopted tile's gather on the same stream.
            if (tile.status == ADOPTED) {
                tile.status = FREE;
            } else if (tile.status == READY) {
                size_t useful = 0;
                uint32_t adopted_ranks = 0;
                const bool current = tile.request.resource == resource && tile.invocation == invocation;
                if (current) {
                    for (uint32_t m = 0; m < count; ++m) {
                        for (uint32_t p = 0; p < tile.request.top_k; ++p) {
                            if (misses[m] == tile.experts[p]) {
                                control.adopted[m] = tile.alias + p * tile.request.expert_bytes;
                                useful += tile.request.expert_bytes;
                                adopted_ranks |= uint32_t(1) << p;
                                break;
                            }
                        }
                    }
                }
                adopted_bytes += useful;
                record_ranks(tile, adopted_ranks, ADOPTED_BYTES);
                if (current) {
                    wrong_bytes += tile.copied - useful;
                    record_ranks(tile, tile.copied_ranks & ~adopted_ranks, WRONG_BYTES);
                } else {
                    late_bytes += tile.copied;
                    record_ranks(tile, tile.copied_ranks, LATE_BYTES);
                }
                tile.status = useful != 0 ? ADOPTED : FREE;
            } else if (tile.status != FREE) {
                tile.expired.store(true, std::memory_order_relaxed);
                if (tile.status == QUEUED) {
                    dropped_bytes.fetch_add(tile.proposed, std::memory_order_relaxed);
                    tile.status = FREE;
                }
            }
        }
        if (next.resource == 0 || proposed == 0) {
            return;
        }
        for (auto & tile : tiles) {
            if (tile.status != FREE) {
                continue;
            }
            GGML_ASSERT(next.top_k <= MOE_PREPACK_ROUTES && next.expert_bytes <= tile_bytes / next.top_k);
            tile.request = next;
            tile.invocation = invocation;
            tile.proposed = proposed;
            std::copy_n(control.predicted, next.top_k, tile.experts.begin());
            tile.expired.store(false, std::memory_order_relaxed);
            tile.status = QUEUED;
            ++published;
            peak_tiles = std::max(peak_tiles, uint32_t(std::count_if(tiles.begin(), tiles.end(), [](const auto & item) { return item.status != FREE; })));
            lock.unlock();
            wake.notify_one();
            return;
        }
        dropped.fetch_add(1, std::memory_order_relaxed);
        dropped_bytes.fetch_add(proposed, std::memory_order_relaxed);
    }

    const size_t tile_bytes;
    std::array<tile, 2> tiles;
    std::mutex mutex;
    std::condition_variable wake;
    std::thread worker;
    bool stop = false;
    std::atomic<bool> admission_closed{false};
    uint64_t published = 0, copied_bytes = 0, adopted_bytes = 0, wrong_bytes = 0, late_bytes = 0;
    uint32_t peak_tiles = 0;
    std::atomic<uint64_t> dropped{0}, dropped_bytes{0}, callbacks{0}, demand_bytes{0};
    std::array<std::atomic<uint64_t>, MOE_PREPACK_ROUTES> proposed_by_rank{};
    uint64_t rank_bytes[MOE_PREPACK_ROUTES][N_RANK_COUNTERS] = {};
};

struct moe_host_copy_span {
    const moe_host_source * source;
    const char * input;
    char * output;
    size_t bytes;
    bool write_combined = false;
};

struct moe_host_copy_range {
    const moe_host_copy_span * spans;
    size_t count;
    size_t first;
    size_t last;
    bool success = true;
};

static void moe_host_copy_range_run(void * opaque) {
    auto & range = *static_cast<moe_host_copy_range *>(opaque);
    size_t offset = 0;
    bool write_combined = false;
    for (size_t i = 0; i < range.count && offset < range.last; ++i) {
        const auto & span = range.spans[i];
        const size_t end = offset + span.bytes;
        if (end > range.first) {
            write_combined |= span.write_combined;
            const size_t first = range.first > offset ? range.first - offset : 0;
            const size_t last = std::min(span.bytes, range.last - offset);
            if (!moe_host_materialize(*span.source, span.input + first, last - first, span.output + first)) {
                range.success = false;
                return;
            }
        }
        offset = end;
    }
#if defined(__x86_64__) || defined(_M_X64)
    if (write_combined) {
        _mm_sfence();
    }
#else
    GGML_UNUSED(write_combined);
#endif
}

static bool moe_host_copy_spans(moe_host_copy_worker & worker, const moe_host_copy_span * spans, size_t count, size_t bytes) {
    const size_t split = bytes / 2 / 64 * 64;
    moe_host_copy_range main{spans, count, 0, bytes};
    moe_host_copy_range helper{spans, count, split, bytes};
    if (split >= 512 * 1024 && worker.try_submit(moe_host_copy_range_run, &helper)) {
        main.last = split;
        moe_host_copy_range_run(&main);
        worker.wait();
        return main.success && helper.success;
    }
    moe_host_copy_range_run(&main);
    return main.success;
}

struct moe_grouped_materialization {
    struct copy {
        moe_grouped_materialization * owner;
        uint32_t miss;
        moe_prepack_request next;
        bool predict = false;
    };

    moe_grouped_materialization(moe_host_budget * owner, size_t control_bytes, size_t tile_stride, uint32_t capacity, uint32_t n_experts, uint32_t top_k) :
        storage(owner, control_bytes + tile_stride, tile_stride, std::min(top_k, capacity), false, false, control_bytes), capacity(capacity), n_experts(n_experts), copies(capacity),
        copy_spans(static_cast<size_t>(storage.tiles) * GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS) {
        for (uint32_t i = 0; i < capacity; ++i) {
            copies[i] = {this, i, {}, false};
        }
    }

    bool publish_prepack(moe_host_prepack * executor) {
        moe_host_prepack * expected = nullptr;
        const bool published = prepack.compare_exchange_strong(expected, executor, std::memory_order_release, std::memory_order_relaxed);
        GGML_ASSERT(published || expected == executor);
        return published;
    }

    size_t prepare_copy(const int32_t * experts, uint32_t count, size_t & bytes) {
        size_t n_spans = 0;
        bytes = 0;
        for (uint32_t i = 0; i < count; ++i) {
            const int32_t expert = experts[i];
            GGML_ASSERT(expert >= 0 && static_cast<uint32_t>(expert) < n_experts);
            if (control != nullptr && control->adopted[i] != nullptr) {
                continue;
            }
            for (size_t bank = 0; bank < sources.size(); ++bank) {
                const auto * source = sources[bank];
                if (source != nullptr) {
                    GGML_ASSERT(n_spans < copy_spans.size() && source->expert_stride <= SIZE_MAX - bytes);
                    copy_spans[n_spans++] = {source, source->data + static_cast<size_t>(expert) * source->expert_stride,
                        static_cast<char *>(destinations[bank]) + static_cast<size_t>(i) * source->expert_stride, source->expert_stride, storage.split_payload};
                    bytes += source->expert_stride;
                }
            }
        }
        return n_spans;
    }

    moe_host_allocation storage;
    uint32_t capacity;
    uint32_t n_experts;
    std::array<const moe_host_source *, GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS> sources = {};
    std::array<void *, GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS> destinations = {};
    std::vector<copy> copies;
    std::vector<moe_host_copy_span> copy_spans;
    std::vector<std::unique_ptr<copy>> prediction_copies;
    std::atomic<moe_host_prepack *> prepack{nullptr};
    moe_prepack_control * control = nullptr;
    const moe_prepack_control * device_control = nullptr;
    uint64_t serial = 0;
    uint64_t invocation = 0;
};

static void CUDART_CB moe_grouped_materialize(void * opaque) {
    const auto & copy = *static_cast<const moe_grouped_materialization::copy *>(opaque);
    auto & materialization = *copy.owner;
    auto * prepack = materialization.prepack.load(std::memory_order_acquire);
    const auto * plan = static_cast<const moe_grouped_decode_plan *>(materialization.storage.data);
    GGML_ASSERT(plan->status == MOE_GROUPED_PLAN_READY && plan->n_misses <= materialization.capacity);
    if (copy.miss == 0) {
        ++materialization.invocation;
        if (materialization.control != nullptr) {
            std::fill_n(materialization.control->adopted, MOE_PREPACK_ROUTES, nullptr);
        }
        if (prepack != nullptr && copy.predict) {
            const auto * experts = moe_grouped_plan_array_ptr(plan, materialization.capacity, MOE_GROUPED_PLAN_MISS_EXPERTS);
            prepack->visit(materialization.serial, materialization.invocation, experts, plan->n_misses, copy.next, *materialization.control);
        }
    }
    if (copy.miss >= plan->n_misses) {
        return;
    }
    const uint32_t count = std::min(materialization.storage.tiles, plan->n_misses - copy.miss);
    const int32_t * experts = moe_grouped_plan_array_ptr(plan, materialization.capacity, MOE_GROUPED_PLAN_MISS_EXPERTS);
    size_t copy_bytes = 0;
    const size_t n_spans = materialization.prepare_copy(experts + copy.miss, count, copy_bytes);
    GGML_ASSERT(moe_host_copy_spans(materialization.storage.owner->copy_worker, materialization.copy_spans.data(), n_spans, copy_bytes));
    if (prepack != nullptr) {
        prepack->demand_bytes.fetch_add(copy_bytes, std::memory_order_relaxed);
    }
}

struct moe_grouped_device_auxiliary {
    const float * source;
    float * data;
    size_t n_values;
};

static __device__ uint32_t moe_grouped_effective_frequency(
        uint32_t frequency,
        uint64_t stored_epoch,
        uint64_t current_epoch) {
    if (stored_epoch > current_epoch) {
        return UINT32_MAX;
    }
    const uint64_t elapsed = current_epoch - stored_epoch;
    return elapsed < 32 ? frequency >> elapsed : 0;
}

static __device__ void moe_grouped_warp_min(uint32_t & frequency, unsigned long long & age, uint32_t & slot) {
#pragma unroll
    for (uint32_t offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        const uint32_t other_frequency = __shfl_xor_sync(0xffffffff, frequency, offset, WARP_SIZE);
        const unsigned long long other_age = __shfl_xor_sync(0xffffffff, age, offset, WARP_SIZE);
        const uint32_t other_slot = __shfl_xor_sync(0xffffffff, slot, offset, WARP_SIZE);
        if (other_frequency < frequency ||
                (other_frequency == frequency && (other_age < age || (other_age == age && other_slot < slot)))) {
            frequency = other_frequency;
            age = other_age;
            slot = other_slot;
        }
    }
}

static __device__ uint32_t moe_grouped_warp_min_slot(uint32_t frequency, unsigned long long age) {
    uint32_t min_frequency = frequency;
#pragma unroll
    for (uint32_t offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        min_frequency = min(min_frequency, __shfl_xor_sync(0xffffffff, min_frequency, offset, WARP_SIZE));
    }
    unsigned long long min_age = frequency == min_frequency ? age : UINT64_MAX;
#pragma unroll
    for (uint32_t offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        min_age = min(min_age, __shfl_xor_sync(0xffffffff, min_age, offset, WARP_SIZE));
    }
    const uint32_t candidates = __ballot_sync(0xffffffff, frequency == min_frequency && age == min_age);
    return candidates != 0 ? __ffs(candidates) - 1 : UINT32_MAX;
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
static __device__ uint32_t moe_grouped_lane_mask_lt(uint32_t lane) {
    return lane == 0 ? 0 : UINT32_MAX >> (WARP_SIZE - lane);
}
#endif

struct moe_grouped_device_scope {
    explicit moe_grouped_device_scope(int device) : previous(ggml_cuda_get_device()) {
        ggml_cuda_set_device(device);
    }

    ~moe_grouped_device_scope() {
        (void) cudaSetDevice(previous);
    }

    int previous;
};

static __device__ void moe_grouped_plan_fail(moe_grouped_decode_plan * plan, moe_grouped_plan_status status) {
    atomicCAS(&plan->status, MOE_GROUPED_PLAN_BUILDING, status);
    __trap();
}

static __global__ void moe_grouped_set_clock(uint64_t * clock, uint64_t value) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *clock = value;
    }
}

static __global__ void moe_grouped_plan_decode(
        const int32_t * ids,
        uint32_t n_routes,
        uint32_t top_k,
        uint32_t row_stride,
        uint32_t n_experts,
        uint32_t n_slots,
        uint32_t plan_capacity,
        int32_t * slot_for_expert,
        int32_t * expert_for_slot,
        uint64_t * last_used,
        uint32_t * expert_frequency,
        uint64_t * expert_frequency_epoch,
        uint64_t * device_step,
        bool frequency_aware,
        uint64_t host_clock_begin,
        uint64_t host_clock_end,
        uint64_t * device_clock,
        moe_grouped_decode_plan * plan) {
    if (blockIdx.x != 0) {
        return;
    }

    __shared__ int32_t warp_route_experts[WARP_SIZE];
    __shared__ uint32_t warp_frequencies[MOE_GROUPED_PLAN_THREADS / WARP_SIZE];
    __shared__ unsigned long long warp_ages[MOE_GROUPED_PLAN_THREADS / WARP_SIZE];
    __shared__ uint32_t warp_slots[MOE_GROUPED_PLAN_THREADS / WARP_SIZE];
    __shared__ uint32_t selected_slot;
    __shared__ uint64_t clock_begin;
    __shared__ uint64_t clock_end;
    const uint32_t thread = threadIdx.x;
    int32_t * route_storage = moe_grouped_plan_array_ptr(plan, plan_capacity, MOE_GROUPED_PLAN_ROUTE_STORAGE);
    int32_t * route_unique = moe_grouped_plan_array_ptr(plan, plan_capacity, MOE_GROUPED_PLAN_ROUTE_UNIQUE);
    int32_t * unique_experts = moe_grouped_plan_array_ptr(plan, plan_capacity, MOE_GROUPED_PLAN_UNIQUE_EXPERTS);
    int32_t * unique_slots = moe_grouped_plan_array_ptr(plan, plan_capacity, MOE_GROUPED_PLAN_UNIQUE_SLOTS);
    int32_t * miss_unique = moe_grouped_plan_array_ptr(plan, plan_capacity, MOE_GROUPED_PLAN_MISS_UNIQUE);
    int32_t * miss_experts = moe_grouped_plan_array_ptr(plan, plan_capacity, MOE_GROUPED_PLAN_MISS_EXPERTS);
    int32_t * miss_slots = moe_grouped_plan_array_ptr(plan, plan_capacity, MOE_GROUPED_PLAN_MISS_SLOTS);
    int32_t * remapped_ids = moe_grouped_plan_array_ptr(plan, plan_capacity, MOE_GROUPED_PLAN_REMAPPED_IDS);
    uint32_t * expert_routes = moe_grouped_plan_expert_routes(plan, plan_capacity);

    // Commit the prior admission before this call overwrites the shared plan.
    if (plan->status == MOE_GROUPED_PLAN_READY) {
        if (plan->n_unique > plan_capacity || plan->n_misses > plan->n_unique) {
            moe_grouped_plan_fail(plan, MOE_GROUPED_PLAN_INVALID_STATE);
        }
        for (uint32_t miss = thread; miss < plan->n_misses; miss += blockDim.x) {
            const int32_t expert = miss_experts[miss];
            const int32_t slot = miss_slots[miss];
            if (expert < 0 || (uint32_t) expert >= n_experts || slot < 0 || (uint32_t) slot >= n_slots) {
                moe_grouped_plan_fail(plan, MOE_GROUPED_PLAN_INVALID_STATE);
            }
            const int32_t old_expert = expert_for_slot[slot];
            if (old_expert >= 0) {
                slot_for_expert[old_expert] = -1;
            }
            expert_for_slot[slot] = expert;
            slot_for_expert[expert] = slot;
        }
        __syncthreads();

        for (uint32_t unique = thread; unique < plan->n_unique; unique += blockDim.x) {
            const int32_t expert = unique_experts[unique];
            const int32_t slot = unique_slots[unique];
            if (expert < 0 || (uint32_t) expert >= n_experts || slot < 0 || (uint32_t) slot >= n_slots) {
                moe_grouped_plan_fail(plan, MOE_GROUPED_PLAN_INVALID_STATE);
            }
            const uint64_t stored_epoch = expert_frequency_epoch[expert];
            if (stored_epoch > plan->frequency_epoch) {
                moe_grouped_plan_fail(plan, MOE_GROUPED_PLAN_INVALID_STATE);
            }
            last_used[slot] = plan->next_clock - plan->n_unique + unique + 1;
            const uint32_t frequency = moe_grouped_effective_frequency(
                expert_frequency[expert], stored_epoch, plan->frequency_epoch);
            const uint32_t updated_frequency = frequency != UINT32_MAX ? frequency + 1 : frequency;
            expert_frequency[expert] = updated_frequency;
            expert_frequency_epoch[expert] = plan->frequency_epoch;
        }
    }
    __syncthreads();

    if (thread == 0) {
        plan->status = MOE_GROUPED_PLAN_BUILDING;
        plan->n_routes = n_routes;
        plan->n_unique = 0;
        plan->n_misses = 0;
        const unsigned long long step = atomicAdd(reinterpret_cast<unsigned long long *>(device_step), 1ULL);
        if (step == UINT64_MAX) {
            moe_grouped_plan_fail(plan, MOE_GROUPED_PLAN_INVALID_STATE);
        }
        plan->frequency_epoch = step >> MOE_GROUPED_FREQUENCY_EPOCH_SHIFT;
        clock_begin = host_clock_begin;
        clock_end = host_clock_end;
        if (device_clock != nullptr) {
            unsigned long long observed = atomicAdd(reinterpret_cast<unsigned long long *>(device_clock), 0ULL);
            while (true) {
                if (observed > UINT64_MAX - n_routes) {
                    moe_grouped_plan_fail(plan, MOE_GROUPED_PLAN_INVALID_STATE);
                }
                const unsigned long long next = observed + n_routes;
                const unsigned long long previous = atomicCAS(
                    reinterpret_cast<unsigned long long *>(device_clock), observed, next);
                if (previous == observed) {
                    clock_begin = observed;
                    clock_end = next;
                    break;
                }
                observed = previous;
            }
        }
    }
    __syncthreads();

    const uint64_t frequency_epoch = plan->frequency_epoch;

    if (n_routes == 0 || n_routes > plan_capacity || top_k == 0 || row_stride < top_k ||
            n_routes % top_k != 0 || n_experts == 0 || n_experts > INT32_MAX ||
            n_slots == 0 || n_slots > INT32_MAX || plan_capacity != n_slots || n_routes > n_slots) {
        moe_grouped_plan_fail(plan, MOE_GROUPED_PLAN_INVALID_STATE);
    }

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    const bool warp_fast = n_routes == top_k && n_routes <= WARP_SIZE;
#else
    const bool warp_fast = false;
#endif
    if (thread < WARP_SIZE) {
        warp_route_experts[thread] = -1;
    }
    for (uint32_t route = thread; route < n_routes; route += blockDim.x) {
        route_storage[route] = -1;
        route_unique[route] = -1;
        unique_experts[route] = -1;
        unique_slots[route] = -1;
        miss_unique[route] = -1;
        miss_experts[route] = -1;
        miss_slots[route] = -1;
        remapped_ids[route] = -1;
    }
    if (!warp_fast) {
        for (uint32_t expert = thread; expert < n_experts; expert += blockDim.x) {
            expert_routes[expert] = UINT32_MAX;
        }
    }
    __syncthreads();

    for (uint32_t route = thread; route < n_routes; route += blockDim.x) {
        const uint32_t input_index = warp_fast ? route : (route / top_k) * row_stride + route % top_k;
        const int32_t expert = ids[input_index];
        if (expert < 0 || (uint32_t) expert >= n_experts) {
            moe_grouped_plan_fail(plan, MOE_GROUPED_PLAN_INVALID_ROUTE);
        }
        route_storage[route] = expert;
        if (warp_fast) {
            warp_route_experts[route] = expert;
        } else {
            atomicMin(&expert_routes[expert], route);
        }
    }
    __syncthreads();

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (warp_fast) {
        if (thread < WARP_SIZE) {
            const uint32_t lane = thread;
            const bool active = lane < n_routes;
            const int32_t expert = active ? warp_route_experts[lane] : -1;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
            const uint32_t first_lane = __ffs(__match_any_sync(0xffffffff, expert)) - 1;
#else
            uint32_t first_lane = lane;
            for (uint32_t source = 0; source < n_routes; ++source) {
                const int32_t other = __shfl_sync(0xffffffff, expert, source, WARP_SIZE);
                if (active && source < first_lane && other == expert) {
                    first_lane = source;
                }
            }
#endif
            const bool first = active && first_lane == lane;
            const uint32_t first_mask = __ballot_sync(0xffffffff, first);
            const uint32_t unique = __popc(first_mask & moe_grouped_lane_mask_lt(first_lane));
            if (active) {
                route_unique[lane] = unique;
            }

            int32_t slot = -1;
            bool invalid_slot = false;
            if (first) {
                slot = slot_for_expert[expert];
                invalid_slot = slot < -1 ||
                    (slot >= 0 && ((uint32_t) slot >= n_slots || expert_for_slot[slot] != expert));
                unique_experts[unique] = expert;
                unique_slots[unique] = slot;
            }
            const uint32_t invalid_mask = __ballot_sync(0xffffffff, invalid_slot);
            const bool miss = first && slot < 0;
            const uint32_t miss_mask = __ballot_sync(0xffffffff, miss);
            if (miss) {
                const uint32_t miss_index = __popc(miss_mask & moe_grouped_lane_mask_lt(lane));
                miss_unique[miss_index] = unique;
                miss_experts[miss_index] = expert;
            }
            if (lane == 0) {
                if (invalid_mask != 0) {
                    moe_grouped_plan_fail(plan, MOE_GROUPED_PLAN_INVALID_STATE);
                }
                plan->n_unique = __popc(first_mask);
                plan->n_misses = __popc(miss_mask);
            }
            __syncwarp(0xffffffff);
        }
        __syncthreads();
    } else
#endif
    {
        if (thread == 0) {
            uint32_t n_unique = 0;
            for (uint32_t route = 0; route < n_routes; ++route) {
                const uint32_t expert = route_storage[route];
                if (expert_routes[expert] == route) {
                    expert_routes[expert] = n_unique;
                    unique_experts[n_unique++] = expert;
                }
                route_unique[route] = expert_routes[expert];
            }
            plan->n_unique = n_unique;
        }
        __syncthreads();

        for (uint32_t unique = thread; unique < plan->n_unique; unique += blockDim.x) {
            const int32_t expert = unique_experts[unique];
            const int32_t slot = slot_for_expert[expert];
            if (slot < -1 || (slot >= 0 && ((uint32_t) slot >= n_slots || expert_for_slot[slot] != expert))) {
                moe_grouped_plan_fail(plan, MOE_GROUPED_PLAN_INVALID_STATE);
            }
            unique_slots[unique] = slot;
        }
        __syncthreads();

        if (thread == 0) {
            uint32_t n_misses = 0;
            for (uint32_t unique = 0; unique < plan->n_unique; ++unique) {
                if (unique_slots[unique] < 0) {
                    miss_unique[n_misses] = unique;
                    miss_experts[n_misses++] = unique_experts[unique];
                }
            }
            plan->n_misses = n_misses;
        }
        __syncthreads();
    }

    for (uint32_t slot = thread; slot < n_slots; slot += blockDim.x) {
        const int32_t resident = expert_for_slot[slot];
        if (resident < -1 || (resident >= 0 && ((uint32_t) resident >= n_experts || slot_for_expert[resident] != (int32_t) slot)) ||
                (resident >= 0 && last_used[slot] == 0)) {
            moe_grouped_plan_fail(plan, MOE_GROUPED_PLAN_INVALID_STATE);
        }
        route_storage[slot] = 0;
    }
    __syncthreads();
    for (uint32_t unique = thread; unique < plan->n_unique; unique += blockDim.x) {
        if (unique_slots[unique] >= 0) {
            route_storage[unique_slots[unique]] = 1;
        }
    }
    __syncthreads();

    const uint32_t lane = thread % WARP_SIZE;
    const uint32_t warp = thread / WARP_SIZE;
    for (uint32_t miss = 0; miss < plan->n_misses; ++miss) {
        uint32_t frequency = UINT32_MAX;
        unsigned long long age = UINT64_MAX;
        uint32_t slot = UINT32_MAX;
        for (uint32_t candidate = thread; candidate < n_slots; candidate += blockDim.x) {
            if (route_storage[candidate] == 0) {
                const int32_t resident = expert_for_slot[candidate];
                const unsigned long long candidate_age = last_used[candidate];
                const uint64_t stored_epoch = resident < 0 ? 0 : expert_frequency_epoch[resident];
                if (stored_epoch > frequency_epoch) {
                    moe_grouped_plan_fail(plan, MOE_GROUPED_PLAN_INVALID_STATE);
                }
                const uint32_t candidate_frequency = !frequency_aware || resident < 0 ? 0 : moe_grouped_effective_frequency(
                    expert_frequency[resident], stored_epoch, frequency_epoch);
                if (candidate_frequency < frequency ||
                        (candidate_frequency == frequency && (candidate_age < age ||
                            (candidate_age == age && candidate < slot)))) {
                    frequency = candidate_frequency;
                    age = candidate_age;
                    slot = candidate;
                }
            }
        }
        if (blockDim.x == WARP_SIZE) {
            const uint32_t candidate_slot = moe_grouped_warp_min_slot(frequency, age);
            if (lane == 0) {
                selected_slot = candidate_slot;
            }
        } else {
            moe_grouped_warp_min(frequency, age, slot);
            if (lane == 0) {
                warp_frequencies[warp] = frequency;
                warp_ages[warp] = age;
                warp_slots[warp] = slot;
            }
            __syncthreads();
            if (warp == 0) {
                frequency = lane < blockDim.x / WARP_SIZE ? warp_frequencies[lane] : UINT32_MAX;
                age = lane < blockDim.x / WARP_SIZE ? warp_ages[lane] : UINT64_MAX;
                slot = lane < blockDim.x / WARP_SIZE ? warp_slots[lane] : UINT32_MAX;
                moe_grouped_warp_min(frequency, age, slot);
                if (lane == 0) {
                    selected_slot = slot;
                }
            }
        }
        __syncthreads();
        if (thread == 0) {
            const int32_t unique = miss_unique[miss];
            if (selected_slot >= n_slots || unique < 0 || (uint32_t) unique >= plan->n_unique) {
                moe_grouped_plan_fail(plan, MOE_GROUPED_PLAN_INVALID_STATE);
            }
            unique_slots[unique] = selected_slot;
            miss_slots[miss] = selected_slot;
            route_storage[selected_slot] = 1;
        }
        __syncthreads();
    }

    for (uint32_t unique = thread; unique < plan->n_unique; unique += blockDim.x) {
        if (unique_slots[unique] < 0) {
            moe_grouped_plan_fail(plan, MOE_GROUPED_PLAN_INVALID_STATE);
        }
    }
    for (uint32_t route = thread; route < n_routes; route += blockDim.x) {
        const int32_t unique = route_unique[route];
        if (unique < 0 || (uint32_t) unique >= plan->n_unique) {
            moe_grouped_plan_fail(plan, MOE_GROUPED_PLAN_INVALID_STATE);
        }
        remapped_ids[route] = unique_slots[unique];
    }
    if (thread == 0) {
        if (plan->n_unique == 0 || plan->n_unique > n_routes || clock_begin > clock_end || plan->n_unique > clock_end - clock_begin) {
            moe_grouped_plan_fail(plan, MOE_GROUPED_PLAN_INVALID_STATE);
        }
        plan->next_clock = clock_begin + plan->n_unique;
    }
    __syncthreads();
    if (thread == 0) {
        plan->status = MOE_GROUPED_PLAN_READY;
    }
}

static bool moe_router_constant(const ggml_tensor * t) {
    return t != nullptr && t->op == GGML_OP_NONE && t->view_src == nullptr && t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

static bool moe_router_pure_op(const ggml_tensor * t) {
    switch (t->op) {
        case GGML_OP_NONE:
            return t->type == GGML_TYPE_I32 && (t->flags & GGML_TENSOR_FLAG_INPUT) != 0;
        case GGML_OP_CPY:
            return t->src[1] == t && t->view_src == nullptr;
        case GGML_OP_MUL_MAT:
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_SCALE:
        case GGML_OP_RMS_NORM:
        case GGML_OP_NORM:
        case GGML_OP_UNARY:
        case GGML_OP_SQRT:
        case GGML_OP_CLAMP:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_ARGSORT:
        case GGML_OP_GET_ROWS:
        case GGML_OP_SET_ROWS:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_FILL:
        case GGML_OP_CONT:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        default:
            return false;
    }
}

static bool moe_router_scale(const ggml_tensor * t, float & scale) {
    if (t == nullptr || t->op != GGML_OP_SCALE) {
        return false;
    }
    float parameters[2];
    memcpy(parameters, t->op_params, sizeof(parameters));
    scale = parameters[0];
    return std::isfinite(scale) && scale > 0.0f && parameters[1] == 0.0f;
}

static bool moe_router_normalized(const ggml_tensor * t, float & multiplier) {
    multiplier = 1.0f;
    if (t == nullptr || t->op != GGML_OP_MUL || !moe_router_constant(t->src[1]) ||
            ggml_nelements(t->src[1]) != t->ne[0]) {
        return false;
    }
    t = t->src[0];
    while (t != nullptr && (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_SCALE)) {
        if (t->op == GGML_OP_SCALE) {
            float scale;
            if (!moe_router_scale(t, scale)) {
                return false;
            }
            multiplier *= scale;
        }
        t = t->src[0];
    }
    return t != nullptr && t->op == GGML_OP_RMS_NORM && std::isfinite(multiplier);
}

static const ggml_tensor * moe_router_scores_node(const ggml_tensor * ids) {
    std::unordered_set<const ggml_tensor *> visited;
    const ggml_tensor * result = nullptr;
    bool ambiguous = false;
    std::function<void(const ggml_tensor *)> visit = [&](const ggml_tensor * t) {
        if (t == nullptr || !visited.insert(t).second || moe_router_constant(t)) {
            return;
        }
        if (t->flags & GGML_TENSOR_FLAG_MOE_ROUTER) {
            ambiguous |= result != nullptr && result != t;
            result = t;
            return;
        }
        if (moe_router_pure_op(t)) {
            for (const auto * src : t->src) {
                visit(src);
            }
        }
    };
    visit(ids);
    return ambiguous ? nullptr : result;
}

static const ggml_tensor * moe_router_projection(const ggml_tensor * ids) {
    std::unordered_set<const ggml_tensor *> visited[2];
    const ggml_tensor * result = nullptr;
    std::function<void(const ggml_tensor *, bool)> visit = [&](const ggml_tensor * t, bool marked) {
        if (t == nullptr || result != nullptr || moe_router_constant(t)) {
            return;
        }
        marked |= (t->flags & GGML_TENSOR_FLAG_MOE_ROUTER) != 0;
        if (!visited[marked].insert(t).second) {
            return;
        }
        if (marked && t->op == GGML_OP_MUL_MAT) {
            result = t;
            return;
        }
        if (!moe_router_pure_op(t)) {
            return;
        }
        for (const auto * src : t->src) {
            visit(src, marked);
        }
    };
    visit(ids, false);
    return result;
}

static const ggml_tensor * moe_router_boundary(const ggml_tensor * input) {
    std::unordered_set<const ggml_tensor *> visited, boundaries;
    bool closed = true;
    std::function<void(const ggml_tensor *)> visit = [&](const ggml_tensor * t) {
        if (t == nullptr || !visited.insert(t).second || moe_router_constant(t)) {
            return;
        }
        float multiplier;
        if (moe_router_normalized(t, multiplier)) {
            boundaries.insert(t);
            return;
        }
        if (!moe_router_pure_op(t)) {
            closed = false;
            return;
        }
        for (const auto * src : t->src) {
            visit(src);
        }
    };
    visit(input);
    return closed && boundaries.size() == 1 ? *boundaries.begin() : input;
}

static bool moe_router_plain_selection(const ggml_tensor * ids, const ggml_tensor * projection,
        const ggml_tensor ** bias) {
    *bias = nullptr;
    while (ids != nullptr && (ids->op == GGML_OP_VIEW || ids->op == GGML_OP_RESHAPE)) {
        ids = ids->src[0];
    }
    if (ids == nullptr || ids->op != GGML_OP_ARGSORT ||
            ggml_get_op_params_i32(ids, 0) != GGML_SORT_ORDER_DESC) {
        return false;
    }
    const auto * scores = ids->src[0];
    if (scores != nullptr && scores->op == GGML_OP_SOFT_MAX && scores->src[1] == nullptr) {
        scores = scores->src[0];
    } else if (scores != nullptr && scores->op == GGML_OP_UNARY &&
            ggml_get_unary_op(scores) == GGML_UNARY_OP_SIGMOID) {
        scores = scores->src[0];
    }
    if (scores != nullptr && scores->op == GGML_OP_ADD && scores->src[0] == projection &&
            moe_router_constant(scores->src[1])) {
        *bias = scores->src[1];
        scores = scores->src[0];
    }
    return scores == projection;
}

static bool moe_router_hc_pattern(const ggml_tensor * mixed, const ggml_tensor * input,
        const ggml_tensor * projection, const ggml_tensor ** down, const ggml_tensor ** up, uint32_t & streams) {
    const int64_t width = projection->src[0]->ne[0];
    if (width <= 0 || input->ne[0] <= width || input->ne[0] % width != 0) {
        return false;
    }
    streams = input->ne[0] / width;
    float scale;
    if (!moe_router_scale(mixed, scale) || scale != 1.0f / streams) {
        return false;
    }
    const auto * sum = mixed->src[0];
    const ggml_tensor * gated = nullptr;
    for (uint32_t index = streams; index-- > 0;) {
        const ggml_tensor * view = nullptr;
        if (index == 0) {
            if (sum == nullptr || sum->op != GGML_OP_CONT) {
                return false;
            }
            view = sum->src[0];
        } else {
            if (sum == nullptr || sum->op != GGML_OP_ADD) {
                return false;
            }
            view = sum->src[1];
            sum = sum->src[0];
        }
        if (view == nullptr || view->op != GGML_OP_VIEW || view->ne[0] != width ||
                view->nb[1] != input->ne[0] * sizeof(float) || view->view_offs != index * width * sizeof(float) ||
                view->src[0] == nullptr || view->src[0]->op != GGML_OP_RESHAPE) {
            return false;
        }
        const auto * source = view->src[0]->src[0];
        if (gated != nullptr && gated != source) {
            return false;
        }
        gated = source;
    }
    if (gated == nullptr || gated->op != GGML_OP_MUL || gated->src[0] != input) {
        return false;
    }
    const auto * gate = gated->src[1];
    if (gate == nullptr || gate->op != GGML_OP_UNARY || ggml_get_unary_op(gate) != GGML_UNARY_OP_SIGMOID) {
        return false;
    }
    const auto * project_up = gate->src[0];
    if (project_up == nullptr || project_up->op != GGML_OP_MUL_MAT || !moe_router_constant(project_up->src[0])) {
        return false;
    }
    const auto * activation = project_up->src[1];
    if (activation == nullptr || activation->op != GGML_OP_UNARY ||
            ggml_get_unary_op(activation) != GGML_UNARY_OP_SILU ||
            !moe_router_scale(activation->src[0], scale) || scale != 1.0f / streams) {
        return false;
    }
    const auto * project_down = activation->src[0]->src[0];
    if (project_down == nullptr || project_down->op != GGML_OP_MUL_MAT ||
            project_down->src[1] != input || !moe_router_constant(project_down->src[0])) {
        return false;
    }
    *down = project_down->src[0];
    *up = project_up->src[0];
    return true;
}

struct moe_router_program {
    using input_binding = std::pair<const ggml_tensor *, ggml_tensor *>;
    uint32_t group = UINT32_MAX;
    uint32_t lane = UINT32_MAX;
    std::vector<uint64_t> signature;
    std::vector<std::unique_ptr<ggml_tensor>> tensors;
    std::vector<ggml_tensor *> operations;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * ids = nullptr;
    size_t bytes = 0;

    static std::vector<uint64_t> describe(const ggml_tensor * boundary, const ggml_tensor * selected) {
        std::vector<uint64_t> result;
        std::unordered_map<const ggml_tensor *, uint64_t> visited;
        std::function<void(const ggml_tensor *)> visit = [&](const ggml_tensor * t) {
            if (t == nullptr) {
                result.push_back(0);
                return;
            }
            const auto found = visited.find(t);
            if (found != visited.end()) {
                result.push_back(found->second);
                result.push_back(0);
                return;
            }
            const uint64_t id = visited.size() + 1;
            visited.emplace(t, id);
            result.push_back(id);
            result.push_back(1);
            const bool input = t == boundary;
            const bool constant = !input && moe_router_constant(t);
            result.push_back(input ? 1 : constant ? 2 : 3);
            result.push_back(input ? GGML_OP_NONE : t->op);
            result.push_back(input ? 0 : (t->flags & GGML_TENSOR_FLAG_INPUT));
            result.push_back(t->type);
            result.push_back(input ? 0 : t->view_offs);
            for (int i = 0; i < GGML_MAX_DIMS; ++i) {
                result.push_back(t->ne[i]);
                result.push_back(t->nb[i]);
            }
            for (const auto value : t->op_params) {
                result.push_back(input ? 0 : (uint32_t) value);
            }
            if (constant) {
                result.push_back((uintptr_t) t);
                result.push_back((uintptr_t) t->data);
                result.push_back((uintptr_t) t->buffer);
            } else if (!input) {
                for (const auto * src : t->src) {
                    visit(src);
                }
                visit(t->view_src);
            }
        };
        visit(selected);
        return result;
    }

    ~moe_router_program() {
        ggml_backend_buffer_free(buffer);
    }

    bool bind_inputs(const ggml_tensor * selected, int device, std::vector<input_binding> & bindings) const {
        bool valid = true;
        std::unordered_set<const ggml_tensor *> visited;
        std::function<void(const ggml_tensor *, ggml_tensor *)> visit = [&](const ggml_tensor * src, ggml_tensor * dst) {
            if (src == nullptr || dst == input || !visited.insert(dst).second || moe_router_constant(src)) {
                return;
            }
            if (src->op == GGML_OP_NONE) {
                valid &= moe_router_pure_op(src) && src->data != nullptr && src->buffer != nullptr &&
                    src->buffer->buft == ggml_backend_cuda_buffer_type(device) && ggml_is_contiguous(src);
                bindings.emplace_back(src, dst);
                return;
            }
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                visit(src->src[s], dst->src[s]);
            }
            visit(src->view_src, dst->view_src);
        };
        visit(selected, ids);
        return valid;
    }

    static std::unique_ptr<moe_router_program> create(
            const ggml_tensor * boundary, const ggml_tensor * selected, int device, std::string & reason) {
        auto result = std::make_unique<moe_router_program>();
        result->signature = describe(boundary, selected);
        std::unordered_map<const ggml_tensor *, ggml_tensor *> copies;
        std::unordered_set<const ggml_tensor *> visiting;
        const auto buft = ggml_backend_cuda_buffer_type(device);
        const auto dev = ggml_backend_buft_get_device(buft);
        std::function<ggml_tensor *(const ggml_tensor *)> copy = [&](const ggml_tensor * t) -> ggml_tensor * {
            if (t == nullptr) {
                return nullptr;
            }
            if (t != boundary && moe_router_constant(t)) {
                if (t->buffer->buft != buft || t->data == nullptr) {
                    reason = "router constant is not on the owning CUDA device";
                    return nullptr;
                }
                return const_cast<ggml_tensor *>(t);
            }
            const auto found = copies.find(t);
            if (found != copies.end()) {
                return found->second;
            }
            if (!visiting.insert(t).second || (t != boundary && !moe_router_pure_op(t))) {
                reason = "router has an unsupported or stateful dependency";
                return nullptr;
            }
            auto tensor = std::make_unique<ggml_tensor>(*t);
            auto * dst = tensor.get();
            dst->extra = nullptr;
            dst->buffer = nullptr;
            dst->data = nullptr;
            dst->flags = t->op == GGML_OP_NONE ? t->flags & GGML_TENSOR_FLAG_INPUT : 0;
            if (t == boundary) {
                dst->op = GGML_OP_NONE;
                std::fill(std::begin(dst->src), std::end(dst->src), nullptr);
                dst->view_src = nullptr;
                dst->view_offs = 0;
            } else {
                for (int s = 0; s < GGML_MAX_SRC; ++s) {
                    if (t->op == GGML_OP_CPY && s == 1 && t->src[s] == t) {
                        dst->src[s] = dst;
                        continue;
                    }
                    dst->src[s] = copy(t->src[s]);
                    if (t->src[s] != nullptr && dst->src[s] == nullptr) {
                        return nullptr;
                    }
                }
                dst->view_src = copy(t->view_src);
                if (t->view_src != nullptr && dst->view_src == nullptr) {
                    return nullptr;
                }
                const ggml_tensor * view_root = dst->view_src;
                while (view_root != nullptr && view_root->view_src != nullptr) {
                    view_root = view_root->view_src;
                }
                if (view_root != nullptr && moe_router_constant(view_root) &&
                        dst->op != GGML_OP_VIEW && dst->op != GGML_OP_RESHAPE &&
                        dst->op != GGML_OP_PERMUTE && dst->op != GGML_OP_TRANSPOSE) {
                    reason = "router would write through a constant view";
                    return nullptr;
                }
                if (!ggml_backend_dev_supports_op(dev, dst)) {
                    reason = "native CUDA does not support a router operation";
                    return nullptr;
                }
            }
            visiting.erase(t);
            copies.emplace(t, dst);
            result->tensors.push_back(std::move(tensor));
            if (t != boundary) {
                result->operations.push_back(dst);
            }
            return dst;
        };
        result->ids = copy(selected);
        const auto input = copies.find(boundary);
        if (result->ids == nullptr || input == copies.end() || result->ids->type != GGML_TYPE_I32) {
            if (reason.empty()) {
                reason = "router does not close over the prediction input";
            }
            return nullptr;
        }
        result->input = input->second;
        std::vector<input_binding> bindings;
        if (!result->bind_inputs(selected, device, bindings)) {
            reason = "router index input is not on the owning CUDA device";
            return nullptr;
        }
        const size_t alignment = ggml_backend_buft_get_alignment(buft);
        for (const auto & t : result->tensors) {
            if (t->view_src == nullptr) {
                const size_t size = ggml_backend_buft_get_alloc_size(buft, t.get());
                if (result->bytes > SIZE_MAX - alignment || size > SIZE_MAX - result->bytes - alignment) {
                    reason = "router scratch size overflow";
                    return nullptr;
                }
                result->bytes += GGML_PAD(size, alignment);
            }
        }
        result->buffer = ggml_backend_buft_alloc_buffer(buft, result->bytes);
        if (result->buffer == nullptr) {
            reason = "router scratch allocation failed";
            return nullptr;
        }
        auto * data = static_cast<char *>(ggml_backend_buffer_get_base(result->buffer));
        for (const auto & t : result->tensors) {
            if (t->view_src != nullptr) {
                t->buffer = t->view_src->buffer;
                t->data = static_cast<char *>(t->view_src->data) + t->view_offs;
            } else {
                const size_t size = ggml_backend_buft_get_alloc_size(buft, t.get());
                if (ggml_backend_tensor_alloc(result->buffer, t.get(), data) != GGML_STATUS_SUCCESS) {
                    reason = "router scratch tensor allocation failed";
                    return nullptr;
                }
                data += GGML_PAD(size, alignment);
            }
        }
        return result;
    }

    void compute(ggml_backend_cuda_context & context) {
        for (auto * node : operations) {
            GGML_ASSERT(ggml_cuda_moe_router_compute(context, node));
        }
    }
};

static bool moe_early_router_enabled() {
    static const bool enabled = [] {
        const char * value = getenv("GGML_CUDA_MOE_EARLY_ROUTER");
        if (value == nullptr || strcmp(value, "1") != 0) {
            return false;
        }
        const char * distance = getenv("GGML_CUDA_MOE_EARLY_ROUTER_LOOKAHEAD");
        if (distance != nullptr && strcmp(distance, "1") != 0) {
            fprintf(stderr, "moe-prepack: disabled reason=only lookahead 1 is supported\n");
            return false;
        }
        for (const char * name : {"NATIVE", "COPY_ENGINE", "COPY_MAILBOX", "COPY_POLL", "COPY_BATCH", "COPY_SPLIT", "COPY_READY_ONLY", "COPY_BANKS", "COPY_DEBUG", "STAGE_BLOCKS"}) {
            const std::string setting = std::string("GGML_CUDA_MOE_EARLY_ROUTER_") + name;
            const char * detail = getenv(setting.c_str());
            if (detail != nullptr && strcmp(detail, "0") != 0) {
                fprintf(stderr, "moe-prepack: disabled reason=retired setting %s\n", setting.c_str());
                return false;
            }
        }
        return true;
    }();
    return enabled;
}

static __global__ void moe_early_router_scores(
        const float * input, const float * attention_scale, const float * ffn_scale,
        const float * weights, int width, float * scores, float input_scale = 1.0f, const float * bias = nullptr) {
    input += (size_t) blockIdx.y * width;
    scores += (size_t) blockIdx.y * gridDim.x;
    __shared__ float partial[128];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const float scale = attention_scale[i];
        const float value = scale != 0.0f ? input[i] * (ffn_scale[i] / scale) : 0.0f;
        sum += value * weights[(size_t) blockIdx.x * width + i];
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int step = blockDim.x / 2; step > 0; step /= 2) {
        if (threadIdx.x < step) {
            partial[threadIdx.x] += partial[threadIdx.x + step];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        scores[blockIdx.x] = partial[0] * input_scale + (bias != nullptr ? bias[blockIdx.x] : 0.0f);
    }
}

static __global__ void moe_early_hc_translate(float * input, const float * attention_scale, const float * ffn_scale, int width, float multiplier = 1.0f) {
    input += (size_t) blockIdx.y * width;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width) {
        const float scale = attention_scale != nullptr ? attention_scale[i] : 1.0f;
        const float target = ffn_scale != nullptr ? ffn_scale[i] : 1.0f;
        input[i] = scale != 0.0f ? input[i] * (target / scale) * multiplier : 0.0f;
    }
}

template<ggml_type type, int threads>
static __global__ void moe_early_hc_mv(const void * weights, const float * input, float * output, int width, int rows, int streams, int activation) {
    input += (size_t) blockIdx.y * width;
    output += (size_t) blockIdx.y * rows;
    const int lane = threadIdx.x % threads;
    const int row = blockIdx.x * (128 / threads) + threadIdx.x / threads;
    float sum = 0.0f;
    if (row < rows) {
        for (int i = lane; i < width; i += threads) {
            float weight;
            if constexpr (type == GGML_TYPE_Q8_0) {
                const auto & block = ((const block_q8_0 *) weights)[(size_t) row * (width / QK8_0) + i / QK8_0];
                weight = __half2float(block.d) * block.qs[i % QK8_0];
            } else {
                weight = ((const float *) weights)[(size_t) row * width + i];
            }
            sum += input[i] * weight;
        }
    }
    if constexpr (threads == 32) {
        sum = warp_reduce_sum<32>(sum);
    } else {
        __shared__ float partial[128];
        partial[lane] = sum;
        __syncthreads();
        for (int step = threads / 2; step > 0; step /= 2) {
            if (lane < step) {
                partial[lane] += partial[lane + step];
            }
            __syncthreads();
        }
        sum = partial[0];
    }
    if (lane == 0 && row < rows) {
        if (activation == 1) {
            sum /= streams;
            sum /= 1.0f + expf(-sum);
        } else if (activation == 2) {
            sum = 1.0f / (1.0f + expf(-sum));
        }
        output[row] = sum;
    }
}

static __global__ void moe_early_hc_collapse(const float * input, const float * gate, float * mixed, int width, int streams) {
    input += (size_t) blockIdx.y * width * streams;
    gate += (size_t) blockIdx.y * width * streams;
    mixed += (size_t) blockIdx.y * width;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width) {
        float sum = 0.0f;
        for (int s = 0; s < streams; ++s) {
            sum += input[s * width + i] * gate[s * width + i];
        }
        mixed[i] = sum / streams;
    }
}

template<int threads>
static void moe_early_hc_project(const ggml_tensor * weights, const float * input, float * output, int streams, int activation, cudaStream_t stream, uint32_t n_rows = 1) {
    const int blocks = (weights->ne[1] + 128 / threads - 1) / (128 / threads);
    const dim3 grid(blocks, n_rows);
    if (weights->type == GGML_TYPE_Q8_0) {
        moe_early_hc_mv<GGML_TYPE_Q8_0, threads><<<grid, 128, 0, stream>>>(weights->data, input, output, weights->ne[0], weights->ne[1], streams, activation);
    } else {
        GGML_ASSERT(weights->type == GGML_TYPE_F32);
        moe_early_hc_mv<GGML_TYPE_F32, threads><<<grid, 128, 0, stream>>>(weights->data, input, output, weights->ne[0], weights->ne[1], streams, activation);
    }
}

template<int items, bool batched = false, bool ordered_ids = false>
static __global__ void moe_early_router_select(
        float * scores, int experts, int top_k, const int32_t * residents,
        const moe_grouped_decode_plan * prior, uint32_t capacity,
        int32_t * predicted, int32_t * positions, uint64_t * counters, size_t entry_bytes, uint32_t n_rows = 1,
        const int32_t * selected_ids = nullptr, size_t ids_row_stride = 0) {
    extern __shared__ float values[];
    __shared__ int32_t incoming_head[32], slots_head[32];
    const uint32_t n_pending = prior->status == MOE_GROUPED_PLAN_READY && prior->n_misses <= capacity ? prior->n_misses : 0;
    const int32_t * incoming = moe_grouped_plan_array_ptr(prior, capacity, MOE_GROUPED_PLAN_MISS_EXPERTS);
    const int32_t * slots = moe_grouped_plan_array_ptr(prior, capacity, MOE_GROUPED_PLAN_MISS_SLOTS);
    if (threadIdx.x < n_pending) {
        incoming_head[threadIdx.x] = incoming[threadIdx.x];
        slots_head[threadIdx.x] = slots[threadIdx.x];
    }
    for (int p = threadIdx.x; p < top_k * (batched ? n_rows : 1); p += blockDim.x) {
        predicted[p] = -1;
    }
    for (int e = threadIdx.x; e < experts; e += blockDim.x) {
        positions[e] = -1;
    }
    int count = 0;
    for (uint32_t row = 0; row < (batched ? n_rows : 1); ++row) {
        for (int e = threadIdx.x; e < experts; e += blockDim.x) {
            if constexpr (items == 0) {
                values[e] = isfinite(scores[e]) ? scores[e] : -INFINITY;
            }
        }
        __syncwarp();
        if constexpr (ordered_ids) {
            for (int rank = threadIdx.x; rank < top_k; rank += blockDim.x) {
                int32_t expert = selected_ids[rank];
                for (int previous = 0; previous < rank; ++previous) {
                    if (selected_ids[previous] == expert) {
                        expert = -1;
                        break;
                    }
                }
                predicted[rank] = expert >= 0 && expert < experts ? expert : -1;
            }
        } else if (threadIdx.x < 32) {
            float local[items > 0 ? items : 1];
#pragma unroll
            for (int i = 0; i < items; ++i) {
                const int e = threadIdx.x + 32 * i;
                const float value = e < experts ? scores[e] : -INFINITY;
                local[i] = isfinite(value) ? value : -INFINITY;
            }
            for (int rank = 0; rank < top_k; ++rank) {
                float best = -INFINITY;
                int expert = INT_MAX;
                if constexpr (items > 0) {
#pragma unroll
                    for (int i = 0; i < items; ++i) {
                        const int e = threadIdx.x + 32 * i;
                        if (local[i] > best) {
                            best = local[i];
                            expert = e;
                        }
                    }
                } else {
                    for (int e = threadIdx.x; e < experts; e += 32) {
                        const float value = values[e];
                        if (value > best) {
                            best = value;
                            expert = e;
                        }
                    }
                }
#pragma unroll
                for (int mask = 16; mask > 0; mask /= 2) {
                    const float value = __shfl_xor_sync(0xffffffff, best, mask, 32);
                    const int other = __shfl_xor_sync(0xffffffff, expert, mask, 32);
                    if (value > best || (value == best && other < expert)) {
                        best = value;
                        expert = other;
                    }
                }
                if (!isfinite(best)) {
                    break;
                }
                if constexpr (items > 0) {
#pragma unroll
                    for (int i = 0; i < items; ++i) {
                        if (threadIdx.x + 32 * i == expert) {
                            local[i] = -INFINITY;
                        }
                    }
                } else {
                    if (threadIdx.x == 0) {
                        values[expert] = -INFINITY;
                    }
                    __syncwarp();
                }
                if (threadIdx.x == rank % 32) {
                    predicted[rank] = expert;
                }
            }
        }
        __syncwarp();
        for (int rank = threadIdx.x; rank < top_k; rank += blockDim.x) {
            if (rank >= (top_k + 1) / 2) {
                predicted[rank] = -1;
                continue;
            }
            const int expert = predicted[rank];
            if (expert < 0) {
                continue;
            }
            const int32_t old_slot = residents[expert];
            bool present = old_slot >= 0;
            // The prior gather completed, but its slot updates are not published yet.
            for (uint32_t m = 0; m < n_pending; ++m) {
                const int32_t incoming_expert = m < 32 ? incoming_head[m] : incoming[m];
                const int32_t incoming_slot = m < 32 ? slots_head[m] : slots[m];
                if (incoming_expert == expert) {
                    present = true;
                    break;
                }
                if (incoming_slot == old_slot) {
                    present = false;
                }
            }
            if (present || (batched && positions[expert] >= 0)) {
                predicted[rank] = -1;
            } else {
                positions[expert] = row * top_k + rank;
                ++count;
            }
        }
        if constexpr (batched) {
            __syncwarp();
            if constexpr (ordered_ids) {
                selected_ids = (const int32_t *) ((const char *) selected_ids + ids_row_stride);
            } else {
                scores += experts;
            }
            predicted += top_k;
        }
    }
    count = warp_reduce_sum(count);
    if (threadIdx.x == 0) {
        counters[0] += count * entry_bytes;
        counters[2] += count;
        counters[4] += 1;
    }
}

template<int items = 1, bool batched = false>
static void moe_early_router_select_launch(
        float * scores, int experts, int top_k, const int32_t * residents,
        const moe_grouped_decode_plan * prior, uint32_t capacity,
        int32_t * predicted, int32_t * positions, uint64_t * counters, size_t entry_bytes, cudaStream_t stream, uint32_t n_rows = 1) {
    if (experts <= 32 * items) {
        moe_early_router_select<items, batched><<<1, 32, 0, stream>>>(scores, experts, top_k, residents, prior, capacity, predicted, positions, counters, entry_bytes, n_rows);
    } else if constexpr (items < 64) {
        moe_early_router_select_launch<2 * items, batched>(scores, experts, top_k, residents, prior, capacity, predicted, positions, counters, entry_bytes, stream, n_rows);
    } else {
        // Larger expert counts use the same selection rules without a register-size limit.
        moe_early_router_select<0, batched><<<1, 32, experts * sizeof(float), stream>>>(scores, experts, top_k, residents, prior, capacity, predicted, positions, counters, entry_bytes, n_rows);
    }
}

template<bool debug_transfers>
static __global__ void moe_grouped_gather_decode(
        const moe_grouped_device_bank * banks,
        uint32_t n_banks,
        size_t words_per_miss,
        const moe_grouped_device_auxiliary * auxiliaries,
        uint32_t n_auxiliaries,
        size_t auxiliary_values_per_miss,
        uint32_t plan_capacity,
        const moe_grouped_decode_plan * plan,
        uint64_t * transfer_counters,
        const moe_prepack_control * control = nullptr,
        uint32_t first_miss = 0,
        uint32_t miss_count = UINT32_MAX) {
    if (plan->status != MOE_GROUPED_PLAN_READY || first_miss >= plan->n_misses) {
        return;
    }
    const uint32_t n_misses = min(miss_count, plan->n_misses - first_miss);
    if constexpr (debug_transfers) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            atomicAdd(reinterpret_cast<unsigned long long *>(&transfer_counters[0]),
                static_cast<unsigned long long>(n_misses) * n_banks);
            atomicAdd(reinterpret_cast<unsigned long long *>(&transfer_counters[1]),
                static_cast<unsigned long long>(n_misses) * words_per_miss * sizeof(uint4));
        }
    }
    const int32_t * miss_experts = moe_grouped_plan_array_ptr(plan, plan_capacity, MOE_GROUPED_PLAN_MISS_EXPERTS);
    const int32_t * miss_slots = moe_grouped_plan_array_ptr(plan, plan_capacity, MOE_GROUPED_PLAN_MISS_SLOTS);
    const size_t total_words = words_per_miss * n_misses;
    const size_t first = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    for (size_t word = first; word < total_words; word += stride) {
        const uint32_t miss = first_miss + word / words_per_miss;
        size_t bank_word = word % words_per_miss;
        for (uint32_t bank = 0; bank < n_banks; ++bank) {
            const auto descriptor = banks[bank];
            const size_t bank_words = descriptor.expert_stride / sizeof(uint4);
            if (bank_word >= bank_words) {
                bank_word -= bank_words;
                continue;
            }
            const int32_t expert = miss_experts[miss];
            const int32_t slot = miss_slots[miss];
            const uint4 * source = reinterpret_cast<const uint4 *>(descriptor.source + (descriptor.staged ? miss - first_miss : (size_t) expert) * descriptor.expert_stride);
            if (control != nullptr && control->adopted[miss - first_miss] != nullptr) {
                source = reinterpret_cast<const uint4 *>(control->adopted[miss - first_miss]) + word % words_per_miss - bank_word;
            }
            uint4 * destination = reinterpret_cast<uint4 *>(descriptor.data + (size_t) slot * descriptor.expert_stride);
            destination[bank_word] = source[bank_word];
            break;
        }
    }
    if (n_auxiliaries != 0) {
        const size_t total_values = auxiliary_values_per_miss * n_misses;
        for (size_t index = first; index < total_values; index += stride) {
            const uint32_t miss = first_miss + index / auxiliary_values_per_miss;
            size_t auxiliary_value = index % auxiliary_values_per_miss;
            uint32_t auxiliary = 0;
            while (auxiliary < n_auxiliaries && auxiliary_value >= auxiliaries[auxiliary].n_values) {
                auxiliary_value -= auxiliaries[auxiliary++].n_values;
            }
            if (auxiliary == n_auxiliaries) {
                continue;
            }
            const auto descriptor = auxiliaries[auxiliary];
            const int32_t expert = miss_experts[miss];
            const int32_t slot = miss_slots[miss];
            descriptor.data[(size_t) slot * descriptor.n_values + auxiliary_value] =
                descriptor.source[(size_t) expert * descriptor.n_values + auxiliary_value];
        }
    }
}

static bool moe_grouped_cuda_success(cudaError_t error) {
    if (error == cudaSuccess) {
        return true;
    }
    (void) cudaGetLastError();
    return false;
}

} // namespace

static bool ggml_cuda_moe_cache_set_metadata(ggml_cuda_moe_cache * cache, const ggml_tensor * tensor);

static ggml_cuda_moe_cache * ggml_cuda_moe_cache_init_with_pool(
        int device,
        size_t slot_size_bytes,
        size_t trailing_padding_bytes,
        int n_slots,
        void * slot_pool_d,
        cudaEvent_t wait_event,
        bool fail_after_stream_probe);

static bool ggml_cuda_moe_cache_prepare_legacy(ggml_cuda_moe_cache * cache, cudaEvent_t grouped_done);
static bool ggml_cuda_moe_cache_prepare_host_staged(ggml_cuda_moe_cache * cache, cudaEvent_t grouped_done);
static bool ggml_cuda_moe_cache_handoff_grouped(ggml_cuda_moe_cache * cache, cudaStream_t grouped_stream);
static bool ggml_cuda_moe_cache_abort_host_staged(ggml_cuda_moe_cache * cache, cudaStream_t compute_stream);

struct ggml_cuda_moe_grouped_context::impl {
    explicit impl(ggml_backend_dev_t owner, int device) : owner(owner), device(device) {
        const char * value = getenv("GGML_CUDA_MOE_FREQUENCY");
        frequency_aware = value == nullptr || strcmp(value, "0") != 0;
    }

    bool frequency_aware = true;

    ~impl() {
        auto * stats = grouped_debug.load(std::memory_order_acquire);
        uint64_t * transfers = stats != nullptr ? stats->device_transfers.load(std::memory_order_acquire) : nullptr;
        if (transfers != nullptr) {
            moe_grouped_device_scope device_scope(device);
            (void) cudaFree(transfers);
        }
    }

    struct grouped_resource;

    struct legacy_record {
        ggml_cuda_moe_legacy_acquisition acquisition;
        const void * source_data = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        ggml_backend_buffer_type_t buft = nullptr;
        size_t byte_extent = 0;
        int64_t ne[GGML_MAX_DIMS] = {};
        size_t nb[GGML_MAX_DIMS] = {};
        uint32_t type = GGML_TYPE_COUNT;
        uint32_t active_leases = 0;
        ggml_cuda_moe_cache * cache = nullptr;
        std::shared_ptr<grouped_resource> backing;
        bool building_cache = false;
    };

    struct grouped_snapshot {
        ggml_cuda_moe_grouped_acquisition acquisition;
        const ggml_tensor * down = nullptr;
        uint32_t layout = GGML_BACKEND_MOE_CANDIDATE_LAYOUT_INVALID;
        uint32_t n_slots = 0;
        uint32_t n_slot_auxiliaries = 0;
        uint64_t prefill_resident_auxiliary_bytes = 0;
        bool prefill_resident_auxiliary = false;
        std::array<ggml_cuda_moe_grouped_bank_descriptor, 3> slot_auxiliaries;
        uint32_t n_original_auxiliaries = 0;
        std::array<ggml_cuda_moe_grouped_bank_descriptor, 3> original_auxiliaries;
        std::vector<ggml_cuda_moe_grouped_bank_descriptor> banks;
    };

    struct early_binding {
        uint32_t group;
        uint32_t top_k;
        const ggml_tensor * input;
        const ggml_tensor * attention_scale;
        const ggml_tensor * ffn_scale;
        const ggml_tensor * weights;
        const ggml_tensor * hc_down = nullptr;
        const ggml_tensor * hc_up = nullptr;
        uint32_t hc_streams = 0;
        const ggml_tensor * ffn_input = nullptr;
        const ggml_tensor * ffn_trigger = nullptr;
        float input_multiplier = 1.0f;
        float ffn_multiplier = 1.0f;
        const ggml_tensor * router_bias = nullptr;
        uint32_t order = 0;
        uint32_t source_group = 0;

    };

    struct early_workspace {
        ~early_workspace() {
            uint64_t counts[5] = {};
            if (initialized && cudaMemcpy(counts, counters, sizeof(counts), cudaMemcpyDeviceToHost) == cudaSuccess) {
                fprintf(stderr, "moe-prediction: calls=%llu nonresident_experts=%llu proposed_bytes=%llu\n",
                    (unsigned long long) counts[4], (unsigned long long) counts[2], (unsigned long long) counts[0]);
            }
            (void) cudaFree(input);
            (void) cudaFree(hc_low);
            (void) cudaFree(hc_gate);
            (void) cudaFree(hc_mixed);
            (void) cudaFree(scores);
            (void) cudaFree(predicted);
            (void) cudaFree(positions);
            (void) cudaFree(counters);
        }
        cudaStream_t stream = nullptr;
        float * input = nullptr;
        float * hc_low = nullptr;
        float * hc_gate = nullptr;
        float * hc_mixed = nullptr;
        float * scores = nullptr;
        int32_t * predicted = nullptr;
        int32_t * positions = nullptr;
        uint64_t * counters = nullptr;
        uint32_t width = 0, hc_rank = 0, experts = 0, top_k = 0;
        bool initialized = false;
    };

    std::vector<std::unique_ptr<early_workspace>> early;
    std::unordered_map<const ggml_tensor *, early_binding> early_bindings;
    std::unordered_map<uint32_t, moe_grouped_materialization::copy *> prepack_callbacks;
    std::unique_ptr<moe_host_allocation> prepack_storage;
    std::unique_ptr<moe_host_prepack> host_prepack;

    void clear_early() {
        if (host_prepack != nullptr) {
            host_prepack->close_admission();
        }
        // Keep replaced resources until their callbacks and the executor have stopped.
        for (const auto & resource : prepack_resources) {
            const auto & device = *resource->device;
            if (device.has_completion) {
                CUDA_CHECK(cudaEventSynchronize(device.completion));
            }
        }
        if (!early.empty()) {
            CUDA_CHECK(cudaStreamSynchronize(early[0]->stream));
        }
        host_prepack.reset();
        prepack_storage.reset();
        early.clear();
        early_bindings.clear();
        prepack_callbacks.clear();
        prepack_resources.clear();
    }

    struct auxiliary_budget {
        std::mutex mutex;
        size_t limit = MOE_ORIGINAL_AUX_BUDGET;
        size_t bytes = 0;
    };

    struct grouped_device_resource {
        explicit grouped_device_resource(int device) : device(device) {}

        void set_serial(uint64_t value) {
            GGML_ASSERT(serial == 0 && value != 0);
            serial = value;
            if (materialization != nullptr) {
                materialization->serial = value;
            }
        }

        ~grouped_device_resource() {
            moe_grouped_device_scope device_scope(device);
            if (completion != nullptr) {
                if (has_completion) {
                    (void) cudaEventSynchronize(completion);
                }
                (void) cudaEventDestroy(completion);
            }
            for (void * data : bank_data) {
                if (data != nullptr) {
                    (void) cudaFree(data);
                }
            }
            for (float * data : auxiliary_data) {
                if (data != nullptr) {
                    (void) cudaFree(data);
                }
            }
            for (float * data : prefill_auxiliary_data) {
                if (data != nullptr) {
                    (void) cudaFree(data);
                }
            }
            for (float * data : original_auxiliary_data) {
                if (data != nullptr) {
                    (void) cudaFree(data);
                }
            }
            if (original_auxiliary_owner != nullptr) {
                std::lock_guard<std::mutex> lock(original_auxiliary_owner->mutex);
                GGML_ASSERT(original_auxiliary_owner->bytes >= original_auxiliary_bytes);
                original_auxiliary_owner->bytes -= original_auxiliary_bytes;
            }
            if (slot_for_expert != nullptr) {
                (void) cudaFree(slot_for_expert);
            }
            if (expert_for_slot != nullptr) {
                (void) cudaFree(expert_for_slot);
            }
            if (last_used != nullptr) {
                (void) cudaFree(last_used);
            }
            if (expert_frequency != nullptr) {
                (void) cudaFree(expert_frequency);
            }
            if (expert_frequency_epoch != nullptr) {
                (void) cudaFree(expert_frequency_epoch);
            }
            if (device_step != nullptr) {
                (void) cudaFree(device_step);
            }
            if (device_clock != nullptr) {
                (void) cudaFree(device_clock);
            }
            if (plan != nullptr) {
                (void) cudaFree(plan);
            }
            if (device_banks != nullptr) {
                (void) cudaFree(device_banks);
            }
            if (device_auxiliaries != nullptr) {
                (void) cudaFree(device_auxiliaries);
            }
            if (source_owner != nullptr) {
                source_owner->release();
            }
        }

        int device;
        moe_host_budget * source_owner = nullptr;
        std::unique_ptr<moe_grouped_materialization> materialization;
        uint32_t n_experts = 0;
        size_t words_per_miss = 0;
        size_t auxiliary_values_per_miss = 0;
        std::vector<void *> bank_data;
        std::array<float *, 3> auxiliary_data = {};
        std::array<size_t, 3> auxiliary_n_values = {};
        std::array<float *, 3> prefill_auxiliary_data = {};
        std::array<float *, 3> original_auxiliary_data = {};
        std::shared_ptr<auxiliary_budget> original_auxiliary_owner;
        size_t original_auxiliary_bytes = 0;
        const ggml_tensor * prefill_auxiliary_pending_node = nullptr;
        const float * prefill_auxiliary_pending_source = nullptr;
        cudaStream_t prefill_auxiliary_pending_stream = nullptr;
        uint64_t prefill_auxiliary_cross_stream_waits = 0;
        uint64_t prefill_auxiliary_pending_declines = 0;
        int32_t * slot_for_expert = nullptr;
        int32_t * expert_for_slot = nullptr;
        uint64_t * last_used = nullptr;
        uint32_t * expert_frequency = nullptr;
        uint64_t * expert_frequency_epoch = nullptr;
        uint64_t * device_step = nullptr;
        uint64_t * device_clock = nullptr;
        moe_grouped_decode_plan * plan = nullptr;
        size_t plan_bytes = 0;
        moe_grouped_device_bank * device_banks = nullptr;
        moe_grouped_device_auxiliary * device_auxiliaries = nullptr;
        cudaEvent_t completion = nullptr;
        cudaStream_t completion_stream = nullptr;
        uint64_t serial = 0;
        uint64_t clock_bound = 0;
        bool has_completion = false;
        bool graph_clock_active = false;
    };

    struct grouped_resource {
        explicit grouped_resource(grouped_snapshot && snapshot) : snapshot(std::move(snapshot)) {}

        ~grouped_resource() {
            for (ggml_cuda_moe_cache * cache : host_staging_caches) {
                ggml_cuda_moe_cache_free(cache);
            }
        }

        const grouped_snapshot snapshot;
        std::unique_ptr<grouped_device_resource> device;
        std::array<ggml_cuda_moe_cache *, GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS> host_staging_caches = {};
        uint64_t active_transaction_token = 0;
        cudaStream_t active_decode_stream = nullptr;
        uint32_t active_host_staging_leases = 0;
        bool retire_host_staging_on_last_release = false;
        bool legacy_dirty = false;
        bool building_device = false;
    };

    struct resource_build_input {
        ggml_cuda_moe_candidate_group_key candidate;
        uint64_t resource_generation = 0;
        const ggml_tensor * down = nullptr;
        uint32_t layout = GGML_BACKEND_MOE_CANDIDATE_LAYOUT_INVALID;
        uint32_t n_slots = 0;
        uint32_t n_groups = 0;
        uint32_t n_banks = 0;
        uint32_t n_slot_auxiliaries = 0;
        uint64_t prefill_resident_auxiliary_bytes = 0;
        bool prefill_resident_auxiliary = false;
        std::array<moe_candidate_bank_record, GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS> banks;
        std::array<moe_candidate_bank_record, 3> slot_auxiliaries;
        uint32_t n_original_auxiliaries = 0;
        std::array<moe_candidate_bank_record, 3> original_auxiliaries;
    };

    struct group_authority_record {
        uint64_t epoch = 0;
        uint32_t active_calls = 0;
        ggml_cuda_moe_group_authority authority = GGML_CUDA_MOE_GROUP_AUTHORITY_LEGACY;
        bool admission_closed = false;
    };

    struct graph_coverage_record {
        uint64_t epoch = 0;
    };

    struct graph_coverage_key {
        const void * nodes = nullptr;
        uint64_t mmid_fingerprint = 0;
        int32_t n_nodes = 0;
        uint32_t mmid_count = 0;

        bool operator==(const graph_coverage_key & other) const {
            return nodes == other.nodes && mmid_fingerprint == other.mmid_fingerprint && n_nodes == other.n_nodes &&
                mmid_count == other.mmid_count;
        }
    };

    struct graph_coverage_key_hash {
        size_t operator()(const graph_coverage_key & key) const {
            uint64_t result = UINT64_C(1469598103934665603);
            moe_candidate_hash_value(result, reinterpret_cast<uintptr_t>(key.nodes));
            moe_candidate_hash_value(result, key.mmid_fingerprint);
            moe_candidate_hash_value(result, key.n_nodes);
            moe_candidate_hash_value(result, key.mmid_count);
            return static_cast<size_t>(result);
        }
    };

    using legacy_record_map = std::unordered_map<const ggml_tensor *, std::unique_ptr<legacy_record>>;
    using legacy_record_list = std::vector<std::unique_ptr<legacy_record>>;
    using terminal_legacy_records = std::array<std::unique_ptr<legacy_record>, GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS>;
    using resource_slots = std::vector<std::shared_ptr<grouped_resource>>;
    using graph_coverage_map = std::unordered_map<graph_coverage_key, graph_coverage_record, graph_coverage_key_hash>;

    ggml_backend_dev_t owner;
    int device;
    std::mutex resource_lifecycle_mutex;
    mutable std::mutex mutex;
    std::mutex telemetry_mutex;
    std::condition_variable resource_cv;
    ggml_cuda_moe_candidate_registry_state state;
    moe_candidate_table table;
    std::shared_ptr<moe_host_budget> pageable_sources;
    std::shared_ptr<auxiliary_budget> original_auxiliary_budget = std::make_shared<auxiliary_budget>();
    mutable std::atomic<uint32_t> fail_device_resource_for_test{0};
    legacy_record_map legacy_records;
    graph_coverage_map graph_coverages;
    resource_slots resources;
    std::vector<std::shared_ptr<grouped_resource>> prepack_resources;
    std::array<uint8_t, GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS> refreshing = {};
    std::array<group_authority_record, GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS> group_authorities = {};
    uint64_t next_resource_generation = 0;
    uint64_t next_device_resource_serial = 0;
    uint64_t next_transaction_token = 0;
    uint64_t next_group_authority_epoch = 0;
    uint64_t next_graph_coverage_epoch = 0;
    uint32_t active_maintenance = 0;
    uint32_t active_legacy_operations = 0;
    uint64_t next_legacy_authority_epoch = 0;
    size_t prefill_resident_auxiliary_budget = MOE_PREFILL_RESIDENT_AUX_BUDGET;
    std::atomic<bool> legacy_debug_mm{false};
    std::atomic<uint64_t> fallback_notice_generation{0};
    bool legacy_policy_initialized = false;
    bool fail_borrowed_cache_init_after_probe_for_test = false;
    std::atomic<uint32_t> split_staging_poison_calls_for_test{0};
    std::atomic<bool> fail_host_staged_evaluator_for_test{false};
    moe_cache_op_phase_stats legacy_op_stats[2];
    moe_cache_telemetry retired_telemetry;
    std::atomic<moe_grouped_decode_debug_stats *> grouped_debug{nullptr};
    std::atomic<bool> grouped_debug_failed{false};
    std::unique_ptr<moe_grouped_decode_debug_stats> grouped_debug_storage;
    bool replacement_pending = false;
    bool authority_transition_pending = false;
    bool draining = false;
    bool shutdown_complete = false;
    uint32_t draining_registered_groups = 0;

    moe_grouped_decode_debug_stats * debug_stats() {
        if (!moe_cache_mm_debug_enabled() || grouped_debug_failed.load(std::memory_order_relaxed)) {
            return nullptr;
        }
        auto * result = grouped_debug.load(std::memory_order_acquire);
        if (result != nullptr) {
            return result;
        }
        std::lock_guard<std::mutex> lock(telemetry_mutex);
        result = grouped_debug.load(std::memory_order_relaxed);
        if (result == nullptr && grouped_debug_failed.load(std::memory_order_relaxed)) {
            return nullptr;
        }
        if (result == nullptr) {
            try {
                grouped_debug_storage = std::make_unique<moe_grouped_decode_debug_stats>();
            } catch (...) {
                grouped_debug_failed.store(true, std::memory_order_relaxed);
                return nullptr;
            }
            result = grouped_debug_storage.get();
            grouped_debug.store(result, std::memory_order_release);
        }
        return result;
    }

    uint64_t * debug_device_transfers() {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
        return nullptr;
#else
        auto * stats = debug_stats();
        if (stats == nullptr || device < 0 || stats->device_transfers_failed.load(std::memory_order_relaxed)) {
            return nullptr;
        }
        auto * result = stats->device_transfers.load(std::memory_order_acquire);
        if (result != nullptr) {
            return result;
        }
        std::lock_guard<std::mutex> lock(telemetry_mutex);
        result = stats->device_transfers.load(std::memory_order_relaxed);
        if (result == nullptr && stats->device_transfers_failed.load(std::memory_order_relaxed)) {
            return nullptr;
        }
        if (result == nullptr) {
            moe_grouped_device_scope device_scope(device);
            if (!moe_grouped_cuda_success(cudaMalloc(&result, 2 * sizeof(uint64_t))) ||
                    !moe_grouped_cuda_success(cudaMemset(result, 0, 2 * sizeof(uint64_t)))) {
                if (result != nullptr) {
                    (void) cudaFree(result);
                }
                stats->device_transfers_failed.store(true, std::memory_order_relaxed);
                return nullptr;
            }
            stats->device_transfers.store(result, std::memory_order_release);
        }
        return result;
#endif
    }

    ggml_cuda_moe_grouped_debug_telemetry take_grouped_debug_telemetry_locked(uint32_t registered) {
        ggml_cuda_moe_grouped_debug_telemetry result;
        result.registered = registered;
        auto * stats = grouped_debug.load(std::memory_order_acquire);
        if (stats == nullptr) {
            return result;
        }
        const auto take = [](std::atomic<uint64_t> & value) {
            return value.exchange(0, std::memory_order_relaxed);
        };
        result.plan_calls = take(stats->plan_calls);
        result.plan_compiles = take(stats->plan_compiles);
        result.plan_reuses = take(stats->plan_reuses);
        result.calls = take(stats->calls);
        result.ready = take(stats->ready);
        result.completed = take(stats->completed);
        result.admitted_banks = take(stats->admitted_banks);
        result.fallback = take(stats->fallback);
        result.rollback = take(stats->rollback);
        result.host_staged_calls = take(stats->host_staged_calls);
        result.host_staged_ops = take(stats->host_staged_ops);
        result.host_staged_split_ops = take(stats->host_staged_split_ops);
        result.strategy_switches = take(stats->strategy_switches);
        result.required_unsupported = take(stats->required_unsupported);
        result.prepare_error = take(stats->prepare_error);
        result.finish_error = take(stats->finish_error);
        result.decode_grouped        = take(stats->decode_grouped);
        result.decode_legacy         = take(stats->decode_legacy);
        result.submitted             = take(stats->submitted);
        result.direct                = take(stats->direct);
        result.captures              = take(stats->captures);
        result.replays               = take(stats->replays);
        uint64_t ready_min = UINT64_MAX;
        uint64_t completed_min = UINT64_MAX;
        for (uint32_t group_index = 0; group_index < GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS; ++group_index) {
            const uint64_t ready = take(stats->ready_by_group[group_index]);
            const uint64_t completed = take(stats->completed_by_group[group_index]);
            const uint64_t submitted = take(stats->submitted_by_group[group_index]);
            if ((ready || submitted || completed) && group_index < table.groups.size()) {
                const auto & group = table.groups[group_index];
                GGML_LOG_INFO(
                    "moe-grouped-owner-group: owner=%p device=%d physical=%d generation=%llu semantic_group=%u "
                    "domain=%u ready=%llu submitted=%llu completed=%llu\n",
                    (void *) this, device, device >= 0 ? ggml_cuda_info().devices[device].physical_device : -1,
                    (unsigned long long) state.generation, group.semantic_group_index, group.domain,
                    (unsigned long long) ready, (unsigned long long) submitted, (unsigned long long) completed);
            }
            if (group_index < registered) {
                ready_min = std::min(ready_min, ready);
                completed_min = std::min(completed_min, completed);
                result.ready_max = std::max(result.ready_max, ready);
                result.completed_max = std::max(result.completed_max, completed);
            }
        }
        if (registered != 0) {
            result.ready_min = ready_min;
            result.completed_min = completed_min;
        }
        for (uint32_t word_index = 0; word_index < stats->covered.size(); ++word_index) {
            uint64_t word = take(stats->covered[word_index]);
            const uint32_t first_group = word_index * 64;
            if (first_group >= registered) {
                continue;
            }
            if (registered - first_group < 64) {
                word &= (1ULL << (registered - first_group)) - 1;
            }
            while (word != 0) {
                word &= word - 1;
                ++result.covered;
            }
        }
        uint64_t * transfers = stats->device_transfers.load(std::memory_order_acquire);
        if (transfers != nullptr) {
            uint64_t counters[2] = {};
            moe_grouped_device_scope device_scope(device);
            if (moe_grouped_cuda_success(cudaMemcpy(counters, transfers, sizeof(counters), cudaMemcpyDeviceToHost))) {
                result.h2d_banks = counters[0];
                result.h2d_bytes = counters[1];
                if (!moe_grouped_cuda_success(cudaMemset(transfers, 0, sizeof(counters)))) {
                    result.h2d_banks = 0;
                    result.h2d_bytes = 0;
                }
            }
        }
        if (moe_grouped_has_activity(result)) {
            ggml_backend_dev_props props{};
            if (owner && owner->iface.get_props) {
                ggml_backend_dev_get_props(owner, &props);
            }
            GGML_LOG_INFO(
                "moe-grouped-owner: owner=%p device=%d physical=%d pci=%s generation=%llu decode_grouped=%llu "
                "decode_legacy=%llu direct=%llu captures=%llu replays=%llu ready=%llu submitted=%llu completed=%llu "
                "prepare_error=%llu finish_error=%llu\n",
                (void *) this, device, device >= 0 ? ggml_cuda_info().devices[device].physical_device : -1,
                props.device_id ? props.device_id : "unknown", (unsigned long long) state.generation,
                (unsigned long long) result.decode_grouped, (unsigned long long) result.decode_legacy,
                (unsigned long long) result.direct, (unsigned long long) result.captures,
                (unsigned long long) result.replays, (unsigned long long) result.ready,
                (unsigned long long) result.submitted, (unsigned long long) result.completed,
                (unsigned long long) result.prepare_error, (unsigned long long) result.finish_error);
        }
        return result;
    }

    ggml_cuda_moe_grouped_debug_telemetry take_grouped_debug_telemetry(uint32_t registered) {
        std::lock_guard<std::mutex> lock(telemetry_mutex);
        auto result = retired_telemetry.grouped;
        retired_telemetry.grouped = {};
        moe_grouped_add_telemetry(result, take_grouped_debug_telemetry_locked(registered));
        return result;
    }

    void fold_grouped_debug_telemetry(uint32_t registered) {
        std::lock_guard<std::mutex> lock(telemetry_mutex);
        if (grouped_debug.load(std::memory_order_acquire) == nullptr) {
            return;
        }
        const auto sample = take_grouped_debug_telemetry_locked(registered);
        if (moe_grouped_has_activity(sample)) {
            moe_grouped_add_telemetry(retired_telemetry.grouped, sample);
        }
    }

    moe_cache_telemetry take_owner_telemetry(uint32_t registered) {
        std::lock_guard<std::mutex> lock(telemetry_mutex);
        moe_cache_telemetry result;
        moe_cache_telemetry retired = std::move(retired_telemetry);
        retired_telemetry = {};
        try {
            moe_cache_add_telemetry(result, std::move(retired));
        } catch (...) {
        }
        moe_grouped_add_telemetry(result.grouped, take_grouped_debug_telemetry_locked(registered));
        for (int phase = 0; phase < 2; ++phase) {
            ggml_cuda_moe_add_phase_stats(result.phase_stats[phase], moe_cache_take_op_stats(legacy_op_stats[phase], true));
        }
        return result;
    }

    void release_maintenance() {
        std::lock_guard<std::mutex> lock(mutex);
        GGML_ASSERT(active_maintenance > 0);
        --active_maintenance;
        resource_cv.notify_all();
    }

    struct maintenance_lease {
        maintenance_lease() = default;
        maintenance_lease(const maintenance_lease &) = delete;
        maintenance_lease & operator=(const maintenance_lease &) = delete;

        ~maintenance_lease() {
            if (owner != nullptr) {
                owner->release_maintenance();
            }
        }

        impl * owner = nullptr;
    };

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

    grouped_resource * begin_decode(
            const ggml_cuda_moe_complete_group_key & key,
            cudaStream_t compute_stream,
            ggml_cuda_moe_grouped_transaction & transaction,
            bool & resource_missing) {
        resource_missing = false;
        if (draining || replacement_pending || !state.accepted || key.candidate.generation != state.generation ||
                key.candidate.group_index >= table.groups.size() || refreshing[key.candidate.group_index]) {
            return nullptr;
        }
        if (key.candidate.group_index >= resources.size() || resources[key.candidate.group_index] == nullptr) {
            resource_missing = true;
            return nullptr;
        }
        auto * resource = resources[key.candidate.group_index].get();
        if (resource->snapshot.acquisition.candidate.generation != key.candidate.generation ||
                resource->snapshot.acquisition.candidate.group_index != key.candidate.group_index ||
                resource->snapshot.layout != key.layout || resource->snapshot.banks.size() != key.n_banks ||
                (resource->legacy_dirty && group_authorities[key.candidate.group_index].authority != GGML_CUDA_MOE_GROUP_AUTHORITY_GROUPED) ||
                resource->active_transaction_token != 0 || next_transaction_token == UINT64_MAX) {
            return nullptr;
        }
        resource->active_transaction_token = ++next_transaction_token;
        resource->active_decode_stream = compute_stream;
        transaction.acquisition = resource->snapshot.acquisition;
        transaction.transaction_token = resource->active_transaction_token;
        return resource;
    }

    grouped_resource * begin_host_staged(
            const ggml_cuda_moe_complete_group_key & key,
            cudaStream_t compute_stream,
            ggml_cuda_moe_grouped_transaction & transaction,
            bool & resource_missing) {
        resource_missing = false;
        if (draining || replacement_pending || !state.accepted || key.candidate.generation != state.generation ||
                key.candidate.group_index >= table.groups.size() || refreshing[key.candidate.group_index] ||
                group_authorities[key.candidate.group_index].authority != GGML_CUDA_MOE_GROUP_AUTHORITY_GROUPED_HOST_STAGED) {
            return nullptr;
        }
        if (key.candidate.group_index >= resources.size() || resources[key.candidate.group_index] == nullptr) {
            resource_missing = true;
            return nullptr;
        }
        auto * resource = resources[key.candidate.group_index].get();
        if (resource->snapshot.acquisition.candidate.generation != key.candidate.generation ||
                resource->snapshot.acquisition.candidate.group_index != key.candidate.group_index ||
                resource->snapshot.layout != key.layout || resource->snapshot.banks.size() != key.n_banks ||
                resource->legacy_dirty || resource->active_transaction_token != 0 ||
                resource->active_host_staging_leases != 0 || next_transaction_token == UINT64_MAX) {
            return nullptr;
        }
        resource->active_transaction_token = ++next_transaction_token;
        resource->active_decode_stream = compute_stream;
        transaction.acquisition = resource->snapshot.acquisition;
        transaction.transaction_token = resource->active_transaction_token;
        return resource;
    }

    resource_slots detach_resources() {
        resource_slots result;
        result.swap(resources);
        return result;
    }

    legacy_record_map detach_legacy_records() {
        legacy_record_map result;
        result.swap(legacy_records);
        return result;
    }

    static void retire_resources(resource_slots resources) {
        resources.clear();
    }

    void retire_legacy_records(legacy_record_map records);
    void retire_legacy_records(legacy_record_list records);
    void retire_legacy_records(terminal_legacy_records records);

    void detach_legacy_records_for_resource(
            const std::shared_ptr<grouped_resource> & resource,
            terminal_legacy_records & retired) {
        uint32_t count = 0;
        for (auto it = legacy_records.begin(); it != legacy_records.end();) {
            if (it->second->backing != resource) {
                ++it;
                continue;
            }
            GGML_ASSERT(it->second->active_leases == 0 && !it->second->building_cache);
            GGML_ASSERT(count < retired.size());
            retired[count++] = std::move(it->second);
            it = legacy_records.erase(it);
        }
    }

    bool retire_failed_group_resource(const ggml_cuda_moe_grouped_transaction & transaction) {
        std::shared_ptr<grouped_resource> retired;
        terminal_legacy_records retired_legacy;
        {
            std::lock_guard<std::mutex> lock(mutex);
            auto * resource = find_resource(transaction);
            if (resource == nullptr) {
                return false;
            }
            if (resource->active_host_staging_leases != 0) {
                return false;
            }
            if (resource->device != nullptr && resource->active_decode_stream != nullptr) {
                if (resource->device->completion != nullptr &&
                        moe_grouped_cuda_success(cudaEventRecord(resource->device->completion, resource->active_decode_stream))) {
                    resource->device->has_completion = true;
                    resource->device->completion_stream = resource->active_decode_stream;
                } else {
                    (void) cudaStreamSynchronize(resource->active_decode_stream);
                    resource->device->has_completion = false;
                    resource->device->completion_stream = nullptr;
                }
            }
            resource->active_transaction_token = 0;
            resource->active_decode_stream = nullptr;
            const uint32_t group_index = transaction.acquisition.candidate.group_index;
            retired = std::move(resources[group_index]);
            detach_legacy_records_for_resource(retired, retired_legacy);
            refreshing[group_index] = 0;
            resource_cv.notify_all();
        }
        retire_legacy_records(std::move(retired_legacy));
        retired.reset();
        return true;
    }

    bool has_active_transaction() const {
        for (const auto & resource : resources) {
            if (resource != nullptr && resource->active_transaction_token != 0) {
                return true;
            }
        }
        return false;
    }

    bool has_active_group_call() const {
        for (const auto & authority : group_authorities) {
            if (authority.active_calls != 0) {
                return true;
            }
        }
        return false;
    }

    bool group_has_active_transaction(uint32_t group_index) const {
        return group_index < resources.size() && resources[group_index] != nullptr &&
            resources[group_index]->active_transaction_token != 0;
    }

    bool group_has_active_legacy_lease(uint32_t group_index) const {
        for (const auto & entry : legacy_records) {
            if (entry.second->acquisition.registered_source && entry.second->acquisition.group_index == group_index &&
                    entry.second->active_leases != 0) {
                return true;
            }
        }
        return false;
    }

    bool reset_group_authorities(uint32_t n_groups) {
        if (n_groups > group_authorities.size() || n_groups > UINT64_MAX - next_group_authority_epoch) {
            return false;
        }
        group_authorities = {};
        for (uint32_t group_index = 0; group_index < n_groups; ++group_index) {
            group_authorities[group_index].epoch = ++next_group_authority_epoch;
        }
        return true;
    }

    bool has_active_legacy_lease() const {
        for (const auto & entry : legacy_records) {
            if (entry.second->active_leases != 0) {
                return true;
            }
        }
        return false;
    }

    static bool legacy_record_matches(const legacy_record & record, const ggml_tensor * tensor) {
        if (tensor == nullptr || tensor->buffer == nullptr || record.acquisition.tensor != tensor || record.source_data != tensor->data ||
                record.buffer != tensor->buffer || record.buft != tensor->buffer->buft || record.byte_extent != ggml_nbytes(tensor) ||
                record.type != tensor->type || tensor->view_src != nullptr || tensor->op != GGML_OP_NONE) {
            return false;
        }
        for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
            if (record.ne[dim] != tensor->ne[dim] || record.nb[dim] != tensor->nb[dim]) {
                return false;
            }
        }
        return true;
    }


    bool capture_resource_input(
            const ggml_cuda_moe_candidate_group_key & key,
            uint64_t resource_generation,
            resource_build_input & input) const {
        if (!state.accepted || key.generation != state.generation || key.group_index >= table.groups.size()) {
            return false;
        }
        input = {};
        input.candidate = key;
        input.resource_generation = resource_generation;
        input.n_slots = table.n_slots;
        input.n_groups = static_cast<uint32_t>(table.groups.size());
        const auto & group = table.groups[key.group_index];
        input.down = group.down;
        input.layout = group.layout;
        input.prefill_resident_auxiliary = group.prefill_resident_auxiliary;
        input.prefill_resident_auxiliary_bytes = group.prefill_resident_auxiliary_bytes;
        uint32_t n_scales = 0;
        uint32_t n_biases = 0;
        if (!moe_candidate_structural_group(group, &input.n_banks, &n_scales, &n_biases) ||
                input.n_banks != moe_candidate_resource_bank_count(group)) {
            return false;
        }
        input.n_slot_auxiliaries = n_scales + n_biases;
        if (input.n_slot_auxiliaries > input.slot_auxiliaries.size()) {
            return false;
        }
        for (uint32_t bank_index = 0; bank_index < input.n_banks; ++bank_index) {
            input.banks[bank_index] = *moe_candidate_resource_bank(group, bank_index);
            if ((input.banks[bank_index].info.source_flags & GGML_CUDA_MMID_SOURCE_GENERIC) == 0) {
                return false;
            }
        }
        for (uint32_t auxiliary_index = 0; auxiliary_index < input.n_slot_auxiliaries; ++auxiliary_index) {
            const auto * auxiliary = moe_candidate_slot_auxiliary_bank(group, auxiliary_index);
            if (auxiliary == nullptr) {
                return false;
            }
            input.slot_auxiliaries[auxiliary_index] = *auxiliary;
        }
        for (const auto & bank : group.banks) {
            if (bank.info.movement == GGML_CUDA_MOE_CANDIDATE_MOVEMENT_PERMANENT_CANDIDATE) {
                if (input.n_original_auxiliaries == input.original_auxiliaries.size()) {
                    return false;
                }
                input.original_auxiliaries[input.n_original_auxiliaries++] = bank;
            }
        }
        return input.n_banks != 0;
    }

    static ggml_cuda_moe_grouped_bank_descriptor make_descriptor(const moe_candidate_bank_record & source) {
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
        return descriptor;
    }

    static std::shared_ptr<grouped_resource> make_grouped_resource(const resource_build_input & input) {
        try {
            grouped_snapshot snapshot;
            snapshot.acquisition.candidate = input.candidate;
            snapshot.acquisition.resource_generation = input.resource_generation;
            snapshot.down = input.down;
            snapshot.layout = input.layout;
            snapshot.n_slots = input.n_slots;
            snapshot.n_slot_auxiliaries = input.n_slot_auxiliaries;
            snapshot.prefill_resident_auxiliary = input.prefill_resident_auxiliary;
            snapshot.prefill_resident_auxiliary_bytes = input.prefill_resident_auxiliary_bytes;
            for (uint32_t i = 0; i < input.n_slot_auxiliaries; ++i) {
                snapshot.slot_auxiliaries[i] = make_descriptor(input.slot_auxiliaries[i]);
            }
            snapshot.n_original_auxiliaries = input.n_original_auxiliaries;
            for (uint32_t i = 0; i < input.n_original_auxiliaries; ++i) {
                snapshot.original_auxiliaries[i] = make_descriptor(input.original_auxiliaries[i]);
            }
            snapshot.banks.reserve(input.n_banks);
            for (uint32_t i = 0; i < input.n_banks; ++i) {
                snapshot.banks.push_back(make_descriptor(input.banks[i]));
            }
            return std::make_shared<grouped_resource>(std::move(snapshot));
        } catch (const std::bad_alloc &) {
            return nullptr;
        }
    }

    static bool descriptor_matches(const ggml_cuda_moe_grouped_bank_descriptor & bank) {
        if (bank.tensor == nullptr || bank.buffer == nullptr || bank.buft == nullptr ||
                bank.tensor->buffer != bank.buffer || bank.tensor->data != bank.source_data ||
                bank.tensor->view_src != nullptr || bank.tensor->op != GGML_OP_NONE || bank.tensor->type != bank.type || bank.buffer->buft != bank.buft ||
                bank.buffer->size != bank.buffer_size || bank.buffer->iface.get_base == nullptr ||
                bank.buft->iface.get_alignment == nullptr || bank.buffer->iface.get_base(bank.buffer) != bank.buffer_base ||
                bank.buft->iface.get_alignment(bank.buft) != bank.alignment) {
            return false;
        }
        for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
            if (bank.tensor->ne[dim] != bank.ne[dim] || bank.tensor->nb[dim] != bank.nb[dim]) {
                return false;
            }
        }
        return true;
    }

    static bool descriptor_matches_record(
            const ggml_cuda_moe_grouped_bank_descriptor & descriptor,
            const moe_candidate_bank_record & record) {
        if (descriptor.tensor != record.info.tensor || descriptor.buffer != record.buffer || descriptor.buft != record.buft ||
                descriptor.source_data != record.info.source_data || descriptor.buffer_base != record.buffer_base ||
                descriptor.buffer_size != record.buffer_size || descriptor.data_offset != record.data_offset ||
                descriptor.byte_extent != record.info.byte_extent || descriptor.expert_stride != record.info.expert_stride ||
                descriptor.alignment != record.alignment || descriptor.role != record.info.role || descriptor.type != record.info.type ||
                descriptor.encoding != record.info.encoding || descriptor.movement != record.info.movement ||
                descriptor.index_modes != record.info.index_modes || !descriptor_matches(descriptor) ||
                !moe_candidate_record_matches(record, descriptor.tensor)) {
            return false;
        }
        for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
            if (descriptor.ne[dim] != record.ne[dim] || descriptor.nb[dim] != record.nb[dim]) {
                return false;
            }
        }
        return true;
    }

    bool resource_matches_table(const grouped_resource & resource) const {
        const auto & snapshot = resource.snapshot;
        const auto & candidate = snapshot.acquisition.candidate;
        if (!state.accepted || candidate.generation != state.generation || candidate.group_index >= table.groups.size() ||
                snapshot.n_slots != table.n_slots) {
            return false;
        }
        const auto & group = table.groups[candidate.group_index];
        size_t original_index = 0;
        for (const auto & bank : group.banks) {
            if (bank.info.movement == GGML_CUDA_MOE_CANDIDATE_MOVEMENT_PERMANENT_CANDIDATE) {
                if (original_index >= snapshot.n_original_auxiliaries ||
                        !descriptor_matches_record(snapshot.original_auxiliaries[original_index++], bank)) {
                    return false;
                }
            }
        }
        if (original_index != snapshot.n_original_auxiliaries) {
            return false;
        }
        uint32_t n_slot_banks = 0;
        uint32_t n_scales = 0;
        uint32_t n_biases = 0;
        if (snapshot.down != group.down || snapshot.layout != group.layout ||
                snapshot.prefill_resident_auxiliary != group.prefill_resident_auxiliary ||
                snapshot.prefill_resident_auxiliary_bytes != group.prefill_resident_auxiliary_bytes ||
                !moe_candidate_structural_group(group, &n_slot_banks, &n_scales, &n_biases) ||
                snapshot.banks.size() != n_slot_banks || snapshot.banks.size() != moe_candidate_resource_bank_count(group) ||
                snapshot.n_slot_auxiliaries != n_scales + n_biases) {
            return false;
        }
        for (uint32_t i = 0; i < snapshot.banks.size(); ++i) {
            const auto * bank = moe_candidate_resource_bank(group, i);
            if (bank == nullptr || !descriptor_matches_record(snapshot.banks[i], *bank)) {
                return false;
            }
        }
        for (uint32_t i = 0; i < snapshot.n_slot_auxiliaries; ++i) {
            const auto * auxiliary = moe_candidate_slot_auxiliary_bank(group, i);
            if (auxiliary == nullptr || !descriptor_matches_record(snapshot.slot_auxiliaries[i], *auxiliary)) {
                return false;
            }
        }
        return true;
    }

    static bool device_alias(
            int device,
            const void * buffer_base,
            uint64_t data_offset,
            const void * source_data,
            bool allow_device,
            bool require_identity,
            const void ** alias_data = nullptr,
            bool * host_alias = nullptr,
            const ggml_tensor * tensor = nullptr) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
        GGML_UNUSED(buffer_base);
        GGML_UNUSED(data_offset);
        GGML_UNUSED(source_data);
        GGML_UNUSED(allow_device);
        GGML_UNUSED(require_identity);
        GGML_UNUSED(alias_data);
        GGML_UNUSED(host_alias);
        GGML_UNUSED(device);
        GGML_UNUSED(tensor);
        return false;
#else
        cudaPointerAttributes attributes = {};
        void * alias_base = nullptr;
        if (alias_data != nullptr) {
            *alias_data = nullptr;
        }
        if (host_alias != nullptr) {
            *host_alias = false;
        }
        if (const auto * source = moe_host_source_for(tensor)) {
            if (source->data != source_data || source->device_alias == nullptr ||
                    (require_identity && source->device_alias != source_data)) {
                return false;
            }
            if (alias_data != nullptr) {
                *alias_data = source->device_alias;
            }
            if (host_alias != nullptr) {
                *host_alias = true;
            }
            return true;
        }
        if (tensor != nullptr && tensor->buffer != nullptr && moe_host_budget_for(tensor->buffer->buft) != nullptr) {
            return false;
        }
        if (buffer_base == nullptr || source_data == nullptr || device < 0 ||
                !moe_grouped_cuda_success(cudaPointerGetAttributes(&attributes, buffer_base))) {
            return false;
        }
        if (attributes.type == cudaMemoryTypeHost) {
            if (!moe_grouped_cuda_success(cudaHostGetDevicePointer(&alias_base, const_cast<void *>(buffer_base), 0))) {
                return false;
            }
            if (host_alias != nullptr) {
                *host_alias = true;
            }
        } else if (allow_device && attributes.type == cudaMemoryTypeDevice &&
                attributes.device == ggml_cuda_info().devices[device].physical_device) {
            alias_base = const_cast<void *>(buffer_base);
        } else {
            return false;
        }
        if (alias_base == nullptr || data_offset > UINTPTR_MAX - reinterpret_cast<uintptr_t>(alias_base)) {
            return false;
        }
        const void * result = reinterpret_cast<const void *>(reinterpret_cast<uintptr_t>(alias_base) + data_offset);
        if (require_identity && result != source_data) {
            return false;
        }
        if (alias_data != nullptr) {
            *alias_data = result;
        }
        return true;
#endif
    }

    std::shared_ptr<moe_host_budget> make_pageable_sources(const moe_candidate_table & candidates) const {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
        GGML_UNUSED(candidates);
        return nullptr;
#else
        if (device < 0 || device >= ggml_cuda_info().device_count || getenv("GGML_CUDA_NO_PINNED") != nullptr) {
            return nullptr;
        }
        moe_grouped_device_scope device_scope(device);
        std::unordered_map<const ggml_tensor *, moe_host_source> sources;
        size_t reserved = 0;
        for (const auto & group : candidates.groups) {
            uint32_t n_banks = 0;
            if (!moe_candidate_structural_group(group, &n_banks, nullptr, nullptr)) {
                continue;
            }
            size_t tile_bytes = 0;
            uint32_t n_experts = 0;
            for (uint32_t i = 0; i < n_banks; ++i) {
                const auto & bank = *moe_candidate_base_slot_bank(group, i);
                if (moe_host_budget_for(bank.buft) != nullptr ||
                        device_alias(device, bank.buffer_base, bank.data_offset, bank.info.source_data, false, false, nullptr, nullptr, bank.info.tensor)) {
                    continue;
                }
                cudaPointerAttributes attributes = {};
                const auto error = cudaPointerGetAttributes(&attributes, bank.info.source_data);
                if (error != cudaSuccess) {
                    (void) cudaGetLastError();
                }
                if ((error != cudaSuccess && error != cudaErrorInvalidValue) ||
                        (error == cudaSuccess && attributes.type != cudaMemoryTypeUnregistered && attributes.type != cudaMemoryTypeHost)) {
                    continue;
                }
                if (bank.info.expert_stride > SIZE_MAX - tile_bytes) {
                    throw std::bad_alloc();
                }
                tile_bytes += bank.info.expert_stride;
                n_experts = bank.ne[2];
                sources.emplace(bank.info.tensor, moe_host_source{nullptr, static_cast<const char *>(bank.info.source_data),
                    static_cast<size_t>(bank.info.byte_extent), static_cast<size_t>(bank.info.expert_stride), true});
            }
            if (tile_bytes == 0) {
                continue;
            }
            size_t plan_bytes = 0, control_offset = 0, staging_offset = 0, allocation_bytes = 0;
            if (!moe_grouped_plan_size(candidates.n_slots, n_experts, &plan_bytes) ||
                    !moe_grouped_staging_layout(plan_bytes, tile_bytes, moe_early_router_enabled(), control_offset, staging_offset, allocation_bytes) ||
                    !moe_host_reserve_size(SIZE_MAX - MOE_PAGEABLE_STAGING_HEADROOM, allocation_bytes, reserved)) {
                throw std::bad_alloc();
            }
        }
        if (sources.empty()) {
            return nullptr;
        }
        auto result = std::shared_ptr<moe_host_budget>(new moe_host_budget(reserved + MOE_PAGEABLE_STAGING_HEADROOM),
            [](moe_host_budget * budget) { budget->release(); });
        result->staging_reserved = reserved;
        result->sources = std::move(sources);
        for (auto & entry : result->sources) {
            entry.second.owner = result.get();
        }
        result->configured = true;
        result->copy_worker.start();
        GGML_LOG_INFO("moe-cache-host: device=%d pageable_sources=%zu staging_reserved=%zu limit=%zu\n",
            device, result->sources.size(), reserved, result->limit);
        return result;
#endif
    }

    const moe_host_source * source_for(const ggml_tensor * tensor) const {
        if (const auto * source = moe_host_source_for(tensor)) {
            return source;
        }
        if (pageable_sources == nullptr || tensor == nullptr) {
            return nullptr;
        }
        const auto found = pageable_sources->sources.find(tensor);
        return found != pageable_sources->sources.end() && found->second.data == tensor->data &&
            found->second.size == ggml_nbytes(tensor) && found->second.expert_stride == tensor->nb[2] ? &found->second : nullptr;
    }

    static bool host_readable(const void * data) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
        GGML_UNUSED(data);
        return false;
#else
        if (data == nullptr) {
            return false;
        }
        cudaPointerAttributes attributes = {};
        const auto error = cudaPointerGetAttributes(&attributes, data);
        if (error != cudaSuccess) {
            (void) cudaGetLastError();
        }
        return error == cudaErrorInvalidValue || (error == cudaSuccess &&
            (attributes.type == cudaMemoryTypeUnregistered || attributes.type == cudaMemoryTypeHost));
#endif
    }

    bool group_source_mapped(const moe_candidate_group_record & group) const {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
        GGML_UNUSED(group);
        return false;
#else
        uint32_t n_banks = 0;
        uint32_t n_scales = 0;
        uint32_t n_biases = 0;
        if (device < 0 || device >= ggml_cuda_info().device_count ||
                !moe_candidate_structural_group(group, &n_banks, &n_scales, &n_biases)) {
            return false;
        }
        moe_grouped_device_scope device_scope(device);
        for (uint32_t bank_index = 0; bank_index < n_banks; ++bank_index) {
            const auto * bank = moe_candidate_base_slot_bank(group, bank_index);
            if (bank == nullptr || !moe_candidate_record_matches(*bank, bank->info.tensor) ||
                    bank->info.expert_stride % sizeof(uint4) != 0 ||
                    reinterpret_cast<uintptr_t>(bank->info.source_data) % alignof(uint4) != 0 ||
                    (source_for(bank->info.tensor) == nullptr &&
                        !device_alias(device, bank->buffer_base, bank->data_offset, bank->info.source_data, false, false, nullptr, nullptr, bank->info.tensor))) {
                return false;
            }
        }
        for (const auto & auxiliary : group.banks) {
            if (auxiliary.info.movement != GGML_CUDA_MOE_CANDIDATE_MOVEMENT_PERMANENT_CANDIDATE) {
                continue;
            }
            if (!moe_candidate_record_matches(auxiliary, auxiliary.info.tensor) ||
                    (!device_alias(device, auxiliary.buffer_base, auxiliary.data_offset, auxiliary.info.source_data, true, true, nullptr, nullptr, auxiliary.info.tensor) &&
                        !host_readable(auxiliary.info.source_data))) {
                return false;
            }
        }
        return true;
#endif
    }

    static bool decode_eligible(const grouped_snapshot & snapshot, int device, uint32_t * n_experts) {
        const uint32_t expected_roles = moe_candidate_required_base_roles(snapshot.layout);
        const uint32_t expected_banks = moe_candidate_required_base_bank_count(snapshot.layout);
        if (device < 0 || device >= ggml_cuda_info().device_count || expected_banks == 0 ||
                snapshot.n_slots == 0 || snapshot.n_slots > INT32_MAX ||
                snapshot.banks.size() < 2 || snapshot.banks.size() > 3) {
            return false;
        }

        uint32_t roles = 0;
        uint32_t experts = 0;
        for (uint32_t bank_index = 0; bank_index < snapshot.banks.size(); ++bank_index) {
            const auto & bank = snapshot.banks[bank_index];
            if (bank.role != moe_candidate_expected_base_role(snapshot.layout, bank_index) ||
                    (roles & (1u << bank.role)) != 0 ||
                    bank.movement != GGML_CUDA_MOE_CANDIDATE_MOVEMENT_SLOT_BOUND ||
                    (bank.index_modes & GGML_CUDA_MOE_CANDIDATE_INDEX_GROUP_SLOT_DIRECT) == 0 ||
                    !ggml_backend_buft_is_cuda_moe_cached(bank.buft) || bank.ne[2] <= 0 || bank.expert_stride == 0 ||
                    bank.ne[2] > INT32_MAX || bank.expert_stride > SIZE_MAX / snapshot.n_slots ||
                    bank.byte_extent != bank.expert_stride * (uint64_t) bank.ne[2] || !descriptor_matches(bank)) {
                return false;
            }
            if (experts == 0) {
                experts = (uint32_t) bank.ne[2];
            } else if (experts != (uint32_t) bank.ne[2]) {
                return false;
            }
            roles |= 1u << bank.role;
        }
        if (roles != expected_roles || snapshot.banks.size() != expected_banks) {
            return false;
        }
        *n_experts = experts;
        return true;
    }

    bool acquire_group_resources_impl(
            const ggml_cuda_moe_candidate_group_key & key,
            ggml_cuda_moe_grouped_acquisition * acquisition) {
        resource_build_input input;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (draining || replacement_pending || !state.accepted || key.generation != state.generation ||
                    key.group_index >= table.groups.size() || refreshing[key.group_index]) {
                return false;
            }
            if (key.group_index < resources.size() && resources[key.group_index] != nullptr) {
                *acquisition = resources[key.group_index]->snapshot.acquisition;
                return true;
            }
            if (next_resource_generation == UINT64_MAX ||
                    !capture_resource_input(key, ++next_resource_generation, input)) {
                return false;
            }
        }

        resource_slots prospective;
        auto resource = make_grouped_resource(input);
        if (resource == nullptr) {
            return false;
        }
        try {
            prospective.resize(input.n_groups);
            prospective[key.group_index] = std::move(resource);
        } catch (const std::bad_alloc &) {
            return false;
        }

        ggml_cuda_moe_grouped_acquisition result;
        bool installed = false;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!draining && !replacement_pending && state.accepted && input.candidate.generation == state.generation &&
                    input.candidate.group_index < table.groups.size() && input.n_slots == table.n_slots &&
                    input.n_groups == table.groups.size()) {
                const auto & group = table.groups[input.candidate.group_index];
                if (input.down == group.down && input.layout == group.layout && input.n_banks == moe_candidate_resource_bank_count(group)) {
                    auto * prospective_resource = prospective[input.candidate.group_index].get();
                    const bool valid = prospective_resource != nullptr && resource_matches_table(*prospective_resource);
                    if (valid && resources.empty()) {
                        resources = std::move(prospective);
                    } else if (valid && resources.size() == input.n_groups && resources[input.candidate.group_index] == nullptr) {
                        resources[input.candidate.group_index] = std::move(prospective[input.candidate.group_index]);
                    }
                    if (input.candidate.group_index < resources.size() && resources[input.candidate.group_index] != nullptr &&
                            !refreshing[input.candidate.group_index] && resource_matches_table(*resources[input.candidate.group_index])) {
                        result = resources[input.candidate.group_index]->snapshot.acquisition;
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

    std::unique_ptr<grouped_device_resource> make_device_resource(
            const grouped_snapshot & snapshot,
            cudaStream_t compute_stream,
            bool prefill_resident_certified = false,
            uint32_t top_k = 1) const try {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
        GGML_UNUSED(snapshot);
        GGML_UNUSED(compute_stream);
        GGML_UNUSED(prefill_resident_certified);
        GGML_UNUSED(top_k);
        return nullptr;
#else
        uint32_t n_experts = 0;
        if (device < 0 || compute_stream == nullptr || snapshot.n_slots == 0 || top_k == 0 ||
                snapshot.n_slot_auxiliaries > snapshot.slot_auxiliaries.size() ||
                !decode_eligible(snapshot, device, &n_experts)) {
            return nullptr;
        }

        moe_grouped_device_scope device_scope(device);
        const uint32_t failure = fail_device_resource_for_test.exchange(0, std::memory_order_relaxed);
        if (failure == 1) {
            throw std::bad_alloc();
        }
        auto result = std::make_unique<grouped_device_resource>(device);
        result->n_experts = n_experts;
        result->bank_data.resize(snapshot.banks.size(), nullptr);
        if (snapshot.n_original_auxiliaries > result->original_auxiliary_data.size()) {
            return nullptr;
        }
        std::array<bool, 3> shadow = {};
        size_t original_bytes = 0;
        for (uint32_t i = 0; i < snapshot.n_original_auxiliaries; ++i) {
            const auto & auxiliary = snapshot.original_auxiliaries[i];
            if (!descriptor_matches(auxiliary) || auxiliary.type != GGML_TYPE_F32 ||
                    !ggml_is_contiguous(auxiliary.tensor) || auxiliary.byte_extent == 0 || auxiliary.byte_extent > SIZE_MAX) {
                return nullptr;
            }
            if (device_alias(device, auxiliary.buffer_base, auxiliary.data_offset, auxiliary.source_data,
                    true, true, nullptr, nullptr, auxiliary.tensor)) {
                continue;
            }
            if (!host_readable(auxiliary.source_data) || auxiliary.byte_extent > SIZE_MAX - original_bytes) {
                return nullptr;
            }
            shadow[i] = true;
            original_bytes += auxiliary.byte_extent;
        }
        if (original_bytes != 0) {
            // Retired graph leases count against the same owner-wide limit.
            std::lock_guard<std::mutex> lock(original_auxiliary_budget->mutex);
            if (original_auxiliary_budget->bytes > original_auxiliary_budget->limit ||
                    original_bytes > original_auxiliary_budget->limit - original_auxiliary_budget->bytes) {
                GGML_LOG_ERROR("moe-cache: original-ID auxiliary shadows need %zu bytes, %zu/%zu owner bytes live\n",
                    original_bytes, original_auxiliary_budget->bytes, original_auxiliary_budget->limit);
                return nullptr;
            }
            result->original_auxiliary_owner = original_auxiliary_budget;
            result->original_auxiliary_bytes = original_bytes;
            original_auxiliary_budget->bytes += original_bytes;
        }
        for (uint32_t i = 0; i < snapshot.n_original_auxiliaries; ++i) {
            if (!shadow[i]) {
                continue;
            }
            const auto & auxiliary = snapshot.original_auxiliaries[i];
            if ((failure == 2 && i == 1) ||
                    !moe_grouped_cuda_success(cudaMalloc(&result->original_auxiliary_data[i], auxiliary.byte_extent))) {
                return nullptr;
            }
            // Published model sources are immutable until synchronized replacement.
            const bool copied = moe_grouped_cuda_success(cudaMemcpyAsync(result->original_auxiliary_data[i],
                auxiliary.source_data, auxiliary.byte_extent, cudaMemcpyHostToDevice, compute_stream));
            const bool ready = moe_grouped_cuda_success(cudaStreamSynchronize(compute_stream));
            if (!copied || !ready) {
                return nullptr;
            }
        }
        std::array<moe_grouped_device_auxiliary, 3> device_auxiliaries = {};
        std::array<bool, 3> auxiliary_host_alias = {};
        for (uint32_t i = 0; i < snapshot.n_slot_auxiliaries; ++i) {
            const auto & auxiliary = snapshot.slot_auxiliaries[i];
            const void * alias_data = nullptr;
            for (uint32_t j = 0; j < snapshot.n_original_auxiliaries; ++j) {
                if (snapshot.original_auxiliaries[j].tensor == auxiliary.tensor &&
                        snapshot.original_auxiliaries[j].role == auxiliary.role) {
                    alias_data = result->original_auxiliary_data[j];
                }
            }
            if (!descriptor_matches(auxiliary) ||
                    (alias_data == nullptr && !device_alias(device, auxiliary.buffer_base, auxiliary.data_offset, auxiliary.source_data,
                        true, true, &alias_data, &auxiliary_host_alias[i], auxiliary.tensor)) ||
                    auxiliary.type != GGML_TYPE_F32 || !ggml_is_contiguous(auxiliary.tensor)) {
                return nullptr;
            }
            const bool scalar = ggml_n_dims(auxiliary.tensor) == 1 && auxiliary.ne[0] == n_experts;
            const bool vector = ggml_n_dims(auxiliary.tensor) == 2 && auxiliary.ne[1] == n_experts;
            const size_t n_values = scalar ? 1 : vector && auxiliary.ne[0] > 0 ? static_cast<size_t>(auxiliary.ne[0]) : 0;
            if (n_values == 0 || n_values > SIZE_MAX / sizeof(float) / n_experts ||
                    auxiliary.byte_extent != n_values * sizeof(float) * n_experts) {
                return nullptr;
            }
            device_auxiliaries[i].source = static_cast<const float *>(alias_data);
            device_auxiliaries[i].n_values = n_values;
        }

        if (!moe_grouped_plan_size(snapshot.n_slots, n_experts, &result->plan_bytes)) {
            return nullptr;
        }

        size_t staging_bytes = 0;
        for (const auto & bank : snapshot.banks) {
            const auto * source = source_for(bank.tensor);
            if (source == nullptr) {
                continue;
            }
            if (result->source_owner != nullptr && result->source_owner != source->owner) {
                return nullptr;
            }
            if (result->source_owner == nullptr) {
                result->source_owner = source->owner;
                result->source_owner->retain();
            }
            if (source->device_alias == nullptr) {
                if (source->expert_stride > SIZE_MAX - staging_bytes) {
                    return nullptr;
                }
                staging_bytes += source->expert_stride;
            }
        }
        const bool prediction_control = moe_early_router_enabled() && top_k <= MOE_PREPACK_ROUTES;
        size_t control_offset = 0, staging_offset = 0, allocation_bytes = 0;
        if (!moe_grouped_staging_layout(result->plan_bytes, staging_bytes, prediction_control, control_offset, staging_offset, allocation_bytes)) {
            return nullptr;
        }
        if (staging_bytes != 0) {
            if (failure == 3) {
                throw std::bad_alloc();
            }
            result->materialization = std::make_unique<moe_grouped_materialization>(result->source_owner,
                staging_offset, staging_bytes, snapshot.n_slots, n_experts, top_k);
            if (result->materialization->storage.data == nullptr) {
                return nullptr;
            }
            if (prediction_control) {
                auto & materialization = *result->materialization;
                materialization.control = reinterpret_cast<moe_prepack_control *>(static_cast<char *>(materialization.storage.data) + control_offset);
                materialization.device_control = reinterpret_cast<const moe_prepack_control *>(static_cast<char *>(materialization.storage.device_alias) + control_offset);
                memset(materialization.control, 0, sizeof(*materialization.control));
                std::fill_n(materialization.control->predicted, MOE_PREPACK_ROUTES, -1);
            }
        }
        size_t next_staging = 0;

        std::array<moe_grouped_device_bank, GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS> device_banks = {};
        for (uint32_t i = 0; i < snapshot.banks.size(); ++i) {
            const auto & bank = snapshot.banks[i];
            const void * alias_data = nullptr;
            if (bank.expert_stride % sizeof(uint4) != 0 ||
                    (uintptr_t) bank.source_data % alignof(uint4) != 0) {
                return nullptr;
            }
            const auto * source = source_for(bank.tensor);
            if (source != nullptr && source->device_alias == nullptr) {
                auto & materialization = *result->materialization;
                materialization.sources[i] = source;
                materialization.destinations[i] = static_cast<char *>(materialization.storage.payload) + next_staging;
                alias_data = static_cast<const char *>(materialization.storage.payload_alias) + next_staging;
                next_staging += source->expert_stride * materialization.storage.tiles;
                device_banks[i].staged = true;
            } else if (!device_alias(device, bank.buffer_base, bank.data_offset, bank.source_data, false, false, &alias_data, nullptr, bank.tensor)) {
                return nullptr;
            }
            if (bank.expert_stride > SIZE_MAX / snapshot.n_slots) {
                return nullptr;
            }
            const size_t size = bank.expert_stride * snapshot.n_slots;
            const size_t trailing_padding = moe_cache_quantized_source_padding(bank.type, bank.ne[0]);
            const size_t bank_words = bank.expert_stride / sizeof(uint4);
            if (bank_words > SIZE_MAX - result->words_per_miss || size > SIZE_MAX - trailing_padding) {
                return nullptr;
            }
            result->words_per_miss += bank_words;
            if (!moe_grouped_cuda_success(cudaMalloc(&result->bank_data[i], size + trailing_padding)) ||
                    (trailing_padding > 0 && !moe_grouped_cuda_success(cudaMemsetAsync(
                        static_cast<char *>(result->bank_data[i]) + size, 0, trailing_padding, compute_stream)))) {
                return nullptr;
            }
            device_banks[i].source = static_cast<const char *>(alias_data);
            device_banks[i].data = static_cast<char *>(result->bank_data[i]);
            device_banks[i].expert_stride = bank.expert_stride;
        }
        for (uint32_t i = 0; i < snapshot.n_slot_auxiliaries; ++i) {
            const size_t n_values = device_auxiliaries[i].n_values;
            if (n_values > SIZE_MAX / sizeof(float) / snapshot.n_slots ||
                    n_values > SIZE_MAX - result->auxiliary_values_per_miss ||
                    !moe_grouped_cuda_success(cudaMalloc(&result->auxiliary_data[i], snapshot.n_slots * n_values * sizeof(float)))) {
                return nullptr;
            }
            result->auxiliary_values_per_miss += n_values;
            device_auxiliaries[i].data = result->auxiliary_data[i];
            result->auxiliary_n_values[i] = n_values;
        }
        if (result->words_per_miss > SIZE_MAX / snapshot.n_slots ||
                result->auxiliary_values_per_miss > SIZE_MAX / snapshot.n_slots) {
            return nullptr;
        }
        if (!moe_grouped_cuda_success(cudaMalloc(&result->slot_for_expert, n_experts * sizeof(int32_t))) ||
                !moe_grouped_cuda_success(cudaMalloc(&result->expert_for_slot, snapshot.n_slots * sizeof(int32_t))) ||
                !moe_grouped_cuda_success(cudaMalloc(&result->last_used, snapshot.n_slots * sizeof(uint64_t))) ||
                !moe_grouped_cuda_success(cudaMalloc(&result->expert_frequency, n_experts * sizeof(uint32_t))) ||
                !moe_grouped_cuda_success(cudaMalloc(&result->expert_frequency_epoch, n_experts * sizeof(uint64_t))) ||
                !moe_grouped_cuda_success(cudaMalloc(&result->device_step, sizeof(uint64_t))) ||
                !moe_grouped_cuda_success(cudaMalloc(&result->device_clock, sizeof(uint64_t))) ||
                !moe_grouped_cuda_success(cudaMalloc(&result->plan, result->plan_bytes)) ||
                !moe_grouped_cuda_success(cudaMalloc(&result->device_banks, snapshot.banks.size() * sizeof(moe_grouped_device_bank))) ||
                (snapshot.n_slot_auxiliaries != 0 &&
                    !moe_grouped_cuda_success(cudaMalloc(&result->device_auxiliaries,
                        snapshot.n_slot_auxiliaries * sizeof(moe_grouped_device_auxiliary)))) ||
                !moe_grouped_cuda_success(cudaEventCreateWithFlags(&result->completion, cudaEventDisableTiming))) {
            return nullptr;
        }
        bool prefill_resident = prefill_resident_certified && snapshot.prefill_resident_auxiliary &&
            snapshot.prefill_resident_auxiliary_bytes != 0;
        uint64_t resident_bytes = 0;
        for (uint32_t i = 0; prefill_resident && i < snapshot.n_slot_auxiliaries; ++i) {
            const auto & auxiliary = snapshot.slot_auxiliaries[i];
            prefill_resident = auxiliary_host_alias[i] && moe_candidate_is_bias(auxiliary.role) &&
                auxiliary.byte_extent != 0 && auxiliary.byte_extent <= SIZE_MAX &&
                moe_candidate_add(resident_bytes, auxiliary.byte_extent, resident_bytes) &&
                moe_grouped_cuda_success(cudaMalloc(
                    &result->prefill_auxiliary_data[i], static_cast<size_t>(auxiliary.byte_extent)));
        }
        prefill_resident = prefill_resident && snapshot.n_slot_auxiliaries != 0 &&
            resident_bytes == snapshot.prefill_resident_auxiliary_bytes;
        if (!prefill_resident) {
            for (uint32_t i = 0; i < result->prefill_auxiliary_data.size(); ++i) {
                if (result->prefill_auxiliary_data[i] != nullptr) {
                    (void) cudaFree(result->prefill_auxiliary_data[i]);
                    result->prefill_auxiliary_data[i] = nullptr;
                }
            }
        }
        for (uint32_t i = 0; i < snapshot.n_slot_auxiliaries; ++i) {
            if (!moe_grouped_cuda_success(cudaMemsetAsync(
                    result->auxiliary_data[i], 0,
                    snapshot.n_slots * device_auxiliaries[i].n_values * sizeof(float), compute_stream))) {
                (void) cudaStreamSynchronize(compute_stream);
                return nullptr;
            }
        }
        if (!moe_grouped_cuda_success(cudaMemsetAsync(
                result->slot_for_expert, 0xff, n_experts * sizeof(int32_t), compute_stream)) ||
                !moe_grouped_cuda_success(cudaMemsetAsync(
                    result->expert_for_slot, 0xff, snapshot.n_slots * sizeof(int32_t), compute_stream)) ||
                !moe_grouped_cuda_success(cudaMemsetAsync(
                    result->last_used, 0, snapshot.n_slots * sizeof(uint64_t), compute_stream)) ||
                !moe_grouped_cuda_success(cudaMemsetAsync(
                    result->expert_frequency, 0, n_experts * sizeof(uint32_t), compute_stream)) ||
                !moe_grouped_cuda_success(cudaMemsetAsync(
                    result->expert_frequency_epoch, 0, n_experts * sizeof(uint64_t), compute_stream)) ||
                !moe_grouped_cuda_success(cudaMemsetAsync(result->device_step, 0, sizeof(uint64_t), compute_stream)) ||
                !moe_grouped_cuda_success(cudaMemsetAsync(result->device_clock, 0, sizeof(uint64_t), compute_stream)) ||
                !moe_grouped_cuda_success(cudaMemsetAsync(result->plan, 0, result->plan_bytes, compute_stream)) ||
                !moe_grouped_cuda_success(cudaMemcpyAsync(
                    result->device_banks, device_banks.data(), snapshot.banks.size() * sizeof(moe_grouped_device_bank),
                    cudaMemcpyHostToDevice, compute_stream)) ||
                (snapshot.n_slot_auxiliaries != 0 &&
                    !moe_grouped_cuda_success(cudaMemcpyAsync(
                        result->device_auxiliaries, device_auxiliaries.data(),
                        snapshot.n_slot_auxiliaries * sizeof(moe_grouped_device_auxiliary),
                        cudaMemcpyHostToDevice, compute_stream)))) {
            (void) cudaStreamSynchronize(compute_stream);
            return nullptr;
        }
        bool resident_copied = prefill_resident;
        for (uint32_t i = 0; resident_copied && i < snapshot.n_slot_auxiliaries; ++i) {
            resident_copied = moe_grouped_cuda_success(cudaMemcpyAsync(
                result->prefill_auxiliary_data[i], snapshot.slot_auxiliaries[i].source_data,
                static_cast<size_t>(snapshot.slot_auxiliaries[i].byte_extent), cudaMemcpyHostToDevice, compute_stream));
        }
        if (prefill_resident && !resident_copied) {
            (void) cudaStreamSynchronize(compute_stream);
            for (uint32_t i = 0; i < result->prefill_auxiliary_data.size(); ++i) {
                if (result->prefill_auxiliary_data[i] != nullptr) {
                    (void) cudaFree(result->prefill_auxiliary_data[i]);
                    result->prefill_auxiliary_data[i] = nullptr;
                }
            }
        }
        if (!moe_grouped_cuda_success(cudaEventRecord(result->completion, compute_stream))) {
            (void) cudaStreamSynchronize(compute_stream);
            return nullptr;
        }
        result->has_completion = true;
        result->completion_stream = compute_stream;
        return result;
#endif
    } catch (const std::bad_alloc &) {
        return nullptr;
    }

    bool cold_reset_grouped_resource(grouped_resource & resource, cudaStream_t compute_stream) const {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
        GGML_UNUSED(resource);
        GGML_UNUSED(compute_stream);
        return false;
#else
        if (resource.device == nullptr || compute_stream == nullptr) {
            return false;
        }
        auto & device_resource = *resource.device;
        moe_grouped_device_scope device_scope(device);
        if (device_resource.has_completion && device_resource.completion_stream != compute_stream &&
                !moe_grouped_cuda_success(cudaStreamWaitEvent(compute_stream, device_resource.completion, 0))) {
            return false;
        }
        for (uint32_t i = 0; i < resource.snapshot.n_slot_auxiliaries; ++i) {
            if (!moe_grouped_cuda_success(cudaMemsetAsync(
                    device_resource.auxiliary_data[i], 0,
                    resource.snapshot.n_slots * device_resource.auxiliary_n_values[i] * sizeof(float), compute_stream))) {
                return false;
            }
        }
        if (!moe_grouped_cuda_success(cudaMemsetAsync(
                    device_resource.slot_for_expert, 0xff, device_resource.n_experts * sizeof(int32_t), compute_stream)) ||
                !moe_grouped_cuda_success(cudaMemsetAsync(
                    device_resource.expert_for_slot, 0xff, resource.snapshot.n_slots * sizeof(int32_t), compute_stream)) ||
                !moe_grouped_cuda_success(cudaMemsetAsync(
                    device_resource.last_used, 0, resource.snapshot.n_slots * sizeof(uint64_t), compute_stream)) ||
                !moe_grouped_cuda_success(cudaMemsetAsync(
                    device_resource.expert_frequency, 0, device_resource.n_experts * sizeof(uint32_t), compute_stream)) ||
                !moe_grouped_cuda_success(cudaMemsetAsync(
                    device_resource.expert_frequency_epoch, 0, device_resource.n_experts * sizeof(uint64_t), compute_stream)) ||
                !moe_grouped_cuda_success(cudaMemsetAsync(device_resource.device_step, 0, sizeof(uint64_t), compute_stream)) ||
                !moe_grouped_cuda_success(cudaMemsetAsync(device_resource.device_clock, 0, sizeof(uint64_t), compute_stream)) ||
                !moe_grouped_cuda_success(cudaMemsetAsync(
                    device_resource.plan, 0, device_resource.plan_bytes, compute_stream)) ||
                !moe_grouped_cuda_success(cudaEventRecord(device_resource.completion, compute_stream))) {
            return false;
        }
        device_resource.clock_bound = 0;
        // Captured planners keep using the same device counters after a logical reset.
        device_resource.has_completion = true;
        device_resource.completion_stream = compute_stream;
        resource.legacy_dirty = false;
        return true;
#endif
    }

    enum clock_reservation : uint32_t {
        CLOCK_RESERVATION_FAILED = 0,
        CLOCK_RESERVATION_READY,
        CLOCK_RESERVATION_DEVICE,
        CLOCK_RESERVATION_REFRESH,
    };

    void clear_refresh(const ggml_cuda_moe_candidate_group_key & key) {
        std::lock_guard<std::mutex> lock(mutex);
        if (key.generation == state.generation && key.group_index < refreshing.size()) {
            refreshing[key.group_index] = 0;
        }
    }

    clock_reservation reserve_clock(
            const ggml_cuda_moe_grouped_transaction & transaction,
            uint32_t n_routes,
            uint64_t & clock_begin,
            uint64_t & clock_end) {
        std::lock_guard<std::mutex> lock(mutex);
        auto * resource = find_resource(transaction);
        if (resource == nullptr || resource->device == nullptr || n_routes == 0) {
            return CLOCK_RESERVATION_FAILED;
        }
        auto & clock_bound = resource->device->clock_bound;
        if (clock_bound > UINT64_MAX - n_routes) {
            refreshing[transaction.acquisition.candidate.group_index] = 1;
            return CLOCK_RESERVATION_REFRESH;
        }
        clock_begin = clock_bound;
        clock_end = clock_bound + n_routes;
        clock_bound = clock_end;
        return resource->device->graph_clock_active ? CLOCK_RESERVATION_DEVICE : CLOCK_RESERVATION_READY;
    }

    bool begin_refresh(
            const ggml_cuda_moe_grouped_transaction & transaction,
            maintenance_lease & lease) {
        std::lock_guard<std::mutex> lock(mutex);
        auto * resource = find_resource(transaction);
        if (resource == nullptr || !refreshing[transaction.acquisition.candidate.group_index] || lease.owner != nullptr) {
            return false;
        }
        ++active_maintenance;
        lease.owner = this;
        resource->active_transaction_token = 0;
        resource->active_decode_stream = nullptr;
        resource_cv.notify_all();
        return !draining && !replacement_pending;
    }

    bool refresh_group_resource(
            const ggml_cuda_moe_grouped_acquisition & acquisition,
            cudaStream_t compute_stream,
            uint32_t top_k) {
        resource_build_input input;
        std::shared_ptr<grouped_resource> retired;
        std::shared_ptr<grouped_resource> stable;
        bool captured = false;
        {
            std::lock_guard<std::mutex> lock(mutex);
            auto * resource = find_resource(acquisition);
            if (draining || replacement_pending || resource == nullptr ||
                    !refreshing[acquisition.candidate.group_index] || resource->active_transaction_token != 0) {
                return false;
            }
            const auto & current = resources[acquisition.candidate.group_index];
            for (const auto & entry : legacy_records) {
                if (entry.second->backing == current) {
                    stable = current;
                    break;
                }
            }
            if (stable == nullptr) {
                if (current.use_count() != 1) {
                    refreshing[acquisition.candidate.group_index] = 0;
                    return false;
                }
                if (next_resource_generation != UINT64_MAX) {
                    captured = capture_resource_input(acquisition.candidate, ++next_resource_generation, input);
                }
                retired = std::move(resources[acquisition.candidate.group_index]);
            }
        }
        if (stable != nullptr) {
            const bool reset = cold_reset_grouped_resource(*stable, compute_stream);
            std::lock_guard<std::mutex> lock(mutex);
            const uint32_t group_index = acquisition.candidate.group_index;
            if (group_index < resources.size() && resources[group_index] == stable &&
                    acquisition.candidate.generation == state.generation) {
                refreshing[group_index] = 0;
            }
            return reset;
        }
        retired.reset();
        if (!captured) {
            clear_refresh(acquisition.candidate);
            return false;
        }
        bool current = false;
        {
            std::lock_guard<std::mutex> lock(mutex);
            const uint32_t group_index = input.candidate.group_index;
            current = !draining && !replacement_pending && input.candidate.generation == state.generation &&
                group_index < resources.size() && resources[group_index] == nullptr && refreshing[group_index];
        }
        if (!current) {
            clear_refresh(input.candidate);
            return false;
        }

        auto prospective = make_grouped_resource(input);
        if (prospective == nullptr) {
            clear_refresh(input.candidate);
            return false;
        }
        prospective->device = make_device_resource(prospective->snapshot, compute_stream, false, top_k);
        if (prospective->device == nullptr) {
            clear_refresh(input.candidate);
            return false;
        }

        std::lock_guard<std::mutex> lock(mutex);
        const uint32_t group_index = input.candidate.group_index;
        if (draining || replacement_pending || group_index >= resources.size() || resources[group_index] != nullptr ||
                prospective->snapshot.acquisition.resource_generation != input.resource_generation ||
                !resource_matches_table(*prospective) || next_device_resource_serial == UINT64_MAX) {
            if (input.candidate.generation == state.generation) {
                refreshing[group_index] = 0;
            }
            return false;
        }
        prospective->device->set_serial(++next_device_resource_serial);
        resources[group_index] = std::move(prospective);
        refreshing[group_index] = 0;
        return true;
    }

    int32_t publish_failure(ggml_cuda_moe_candidate_rejection rejection, uint32_t n_slots, int32_t result) {
        std::lock_guard<std::mutex> lifecycle_lock(resource_lifecycle_mutex);
        legacy_record_map retired_legacy;
        resource_slots retired;
        uint32_t old_registered_groups = 0;
        int32_t published = result;
        {
            std::unique_lock<std::mutex> lock(mutex);
            if (draining) {
                return GGML_BACKEND_MOE_CANDIDATE_REPLACE_ERROR;
            }
            replacement_pending = true;
            resource_cv.wait(lock, [&]() {
                return !has_active_transaction() && active_maintenance == 0 && active_legacy_operations == 0 &&
                    !has_active_legacy_lease() && !has_active_group_call();
            });
            retired = detach_resources();
            retired_legacy = detach_legacy_records();
            old_registered_groups = state.accepted ? static_cast<uint32_t>(table.groups.size()) : 0;
            refreshing.fill(0);
            authority_transition_pending = false;
        }
        retire_resources(std::move(retired));
        retire_legacy_records(std::move(retired_legacy));
        fold_grouped_debug_telemetry(old_registered_groups);
        {
            std::lock_guard<std::mutex> lock(mutex);
            (void) reset_group_authorities(0);
            table = {};
            pageable_sources.reset();
            if (state.generation == UINT64_MAX) {
                state = {};
                state.generation = UINT64_MAX;
                state.rejection = GGML_CUDA_MOE_CANDIDATE_REJECT_GENERATION_EXHAUSTED;
                published = GGML_BACKEND_MOE_CANDIDATE_REPLACE_ERROR;
            } else {
                const uint64_t generation = state.generation + 1;
                state = {};
                state.generation = generation;
                state.n_slots = n_slots;
                state.rejection = rejection;
            }
            replacement_pending = false;
        }
        resource_cv.notify_all();
        return published;
    }

    int32_t publish(moe_candidate_table && replacement) {
        std::lock_guard<std::mutex> lifecycle_lock(resource_lifecycle_mutex);
        moe_candidate_set_prefill_resident_policy(replacement, prefill_resident_auxiliary_budget);
        auto sources = make_pageable_sources(replacement);
        legacy_record_map retired_legacy;
        resource_slots retired;
        uint32_t old_registered_groups = 0;
        int32_t published = GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED;
        {
            std::unique_lock<std::mutex> lock(mutex);
            if (draining) {
                return GGML_BACKEND_MOE_CANDIDATE_REPLACE_ERROR;
            }
            replacement_pending = true;
            resource_cv.wait(lock, [&]() {
                return !has_active_transaction() && active_maintenance == 0 && active_legacy_operations == 0 &&
                    !has_active_legacy_lease() && !has_active_group_call();
            });
            retired = detach_resources();
            retired_legacy = detach_legacy_records();
            old_registered_groups = state.accepted ? static_cast<uint32_t>(table.groups.size()) : 0;
            refreshing.fill(0);
            authority_transition_pending = false;
        }
        retire_resources(std::move(retired));
        retire_legacy_records(std::move(retired_legacy));
        fold_grouped_debug_telemetry(old_registered_groups);
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (state.generation == UINT64_MAX || replacement.groups.size() > UINT64_MAX - next_group_authority_epoch) {
                const uint64_t generation = state.generation == UINT64_MAX ? UINT64_MAX : state.generation + 1;
                group_authorities = {};
                table = {};
                pageable_sources.reset();
                state = {};
                state.generation = generation;
                state.rejection = GGML_CUDA_MOE_CANDIDATE_REJECT_GENERATION_EXHAUSTED;
                published = GGML_BACKEND_MOE_CANDIDATE_REPLACE_ERROR;
            } else {
                const uint64_t generation = state.generation + 1;
                const bool reset = reset_group_authorities(static_cast<uint32_t>(replacement.groups.size()));
                GGML_ASSERT(reset);
                table = std::move(replacement);
                pageable_sources = std::move(sources);
                state = {};
                state.generation = generation;
                state.logical_signature = table.logical_signature;
                state.slot_bound_bytes = table.slot_bound_bytes;
                state.permanent_candidate_bytes = table.permanent_candidate_bytes;
                state.n_slots = table.n_slots;
                state.n_groups = table.groups.size();
                state.n_weights = table.active_weights;
                state.accepted = 1;
            }
            replacement_pending = false;
        }
        resource_cv.notify_all();
        return published;
    }
};

ggml_cuda_moe_grouped_context::ggml_cuda_moe_grouped_context(ggml_backend_dev_t owner, int device) :
    impl_(std::make_unique<impl>(owner, device)) {
    auto & telemetry = moe_cache_owner_telemetry_state();
    std::lock_guard<std::mutex> lock(telemetry.mutex);
    telemetry.active.insert(this);
}

ggml_cuda_moe_grouped_context::~ggml_cuda_moe_grouped_context() {
    shutdown();
}

size_t ggml_cuda_moe_grouped_context::early_program_count_for_test() const {
    return 0;
}

uint64_t ggml_cuda_moe_grouped_context::early_bytes_for_test(uint64_t * calls) const {
    uint64_t bytes = 0;
    if (calls != nullptr) {
        *calls = 0;
    }
    for (const auto & lane : impl_->early) {
        uint64_t counts[5] = {};
        CUDA_CHECK(cudaStreamSynchronize(lane->stream));
        CUDA_CHECK(cudaMemcpy(counts, lane->counters, sizeof(counts), cudaMemcpyDeviceToHost));
        bytes += counts[0];
        if (calls != nullptr) {
            *calls += counts[4];
        }
    }
    return bytes;
}

bool ggml_cuda_moe_grouped_context::early_graph_for_test() {
    CUDA_CHECK(cudaSetDevice(impl_->device));
    bool valid = true;
    const auto buft = ggml_backend_cuda_buffer_type(impl_->device);
    for (int rows : {1, 4}) {
        for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0}) {
            for (int variant = 0; variant < 11; ++variant) {
                const int width = variant == 10 ? 96 : 64;
                const int experts = variant == 4 ? 17 : variant == 9 ? 257 : variant == 10 ? 33 : 16;
                const int top_k = variant == 5 ? 1 : variant == 6 ? 8 : variant == 9 ? 16 : 4;
                ggml_init_params params{1024 * 1024, nullptr, true};
                auto * weights_ctx = ggml_init(params);
                auto * ctx = ggml_init(params);
                auto * weights = ggml_new_tensor_2d(weights_ctx, type, width, experts);
                auto * bias = ggml_new_tensor_1d(weights_ctx, GGML_TYPE_F32, experts);
                auto * scale = ggml_new_tensor_1d(weights_ctx, GGML_TYPE_F32, width);
                auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, rows);
                auto * norm = ggml_rms_norm(ctx, input, 1e-6f);
                if (variant == 7) {
                    norm = ggml_scale(ctx, norm, 0.125f);
                }
                auto * boundary = variant == 8 ? input : ggml_mul(ctx, norm, scale);
                auto * router_weights = variant == 10 ? ggml_view_2d(ctx, weights, width, experts, weights->nb[1], 0) : weights;
                auto * projection = ggml_mul_mat(ctx, router_weights, boundary);
                projection->flags |= GGML_TENSOR_FLAG_MOE_ROUTER;
                ggml_set_name(projection, "arbitrary-router-name");
                ggml_tensor * scores = variant == 0 ? ggml_soft_max(ctx, ggml_add(ctx, projection, bias)) :
                    ggml_add(ctx, ggml_sigmoid(ctx, projection), bias);
                if (variant == 3) {
                    projection->flags &= ~GGML_TENSOR_FLAG_MOE_ROUTER;
                    scores->flags |= GGML_TENSOR_FLAG_MOE_ROUTER;
                }
                if (variant == 4) {
                    scores = ggml_add(ctx, ggml_sqrt(ctx, ggml_softplus(ctx, projection)), bias);
                }
                if (variant == 5) {
                    scores = projection;
                }
                if (variant == 2) {
                    auto * groups = ggml_reshape_3d(ctx, scores, 4, 4, rows);
                    auto * ranks = ggml_argsort_top_k(ctx, groups, 2);
                    auto * best = ggml_get_rows(ctx, ggml_reshape_4d(ctx, groups, 1, 4, 4, rows), ranks);
                    auto * totals = ggml_sum_rows(ctx, ggml_reshape_3d(ctx, best, 2, 4, rows));
                    auto * selected_groups = ggml_argsort_top_k(ctx, ggml_reshape_2d(ctx, totals, 4, rows), 2);
                    scores = ggml_reshape_2d(ctx, ggml_set_rows(ctx, ggml_fill(ctx, groups, -INFINITY),
                        ggml_get_rows(ctx, groups, selected_groups), selected_groups), experts, rows);
                }
                auto * ids = ggml_argsort_top_k(ctx, scores, top_k);
                if (variant == 6) {
                    ids = ggml_cast(ctx, ggml_scale(ctx, ggml_cast(ctx, ids, GGML_TYPE_F32), 0.5f), GGML_TYPE_I32);
                }
                auto * graph = ggml_new_graph(ctx);
                ggml_build_forward_expand(graph, ids);
                auto weights_buffer = ggml_backend_alloc_ctx_tensors_from_buft(weights_ctx, buft);
                auto buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
                GGML_ASSERT(weights_buffer != nullptr && buffer != nullptr);
                ggml_backend_buffer_set_usage(weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                std::vector<float> values(width * experts), inputs(width * rows), biases(experts), scales(width, 1.0f);
                for (size_t i = 0; i < values.size(); ++i) {
                    values[i] = std::sin(float(i * 13 + 7)) / 8.0f;
                }
                for (int i = 0; i < experts; ++i) {
                    biases[i] = float((i * 7) % 11) / 17.0f;
                }
                std::vector<uint8_t> quant(ggml_nbytes(weights));
                ggml_quantize_chunk(type, values.data(), quant.data(), 0, experts, width, nullptr);
                ggml_backend_tensor_set(weights, quant.data(), 0, quant.size());
                ggml_backend_tensor_set(bias, biases.data(), 0, biases.size() * sizeof(float));
                ggml_backend_tensor_set(scale, scales.data(), 0, scales.size() * sizeof(float));
                std::string reason;
                auto program = moe_router_program::create(boundary, ids, impl_->device, reason);
                if (program == nullptr) {
                    fprintf(stderr, "early-graph-test: rejected variant=%d type=%s rows=%d reason=%s\n", variant, ggml_type_name(type), rows, reason.c_str());
                    valid = false;
                } else {
                    valid &= moe_router_projection(ids) == projection && moe_router_boundary(boundary) == boundary;
                    valid &= program->signature == moe_router_program::describe(program->input, program->ids);
                    const ggml_tensor * found_bias = nullptr;
                    valid &= moe_router_plain_selection(ids, projection, &found_bias) == (variant == 0 || variant == 5);
                    ggml_backend_cuda_context reference(impl_->device), predictor(impl_->device);
                    cudaGraph_t captured = nullptr;
                    cudaGraphExec_t replay = nullptr;
                    std::vector<int32_t> expected(rows * top_k), actual(rows * top_k);
                    int32_t * selection;
                    uint64_t * counters;
                    moe_grouped_decode_plan * prior;
                    size_t plan_bytes;
                    GGML_ASSERT(moe_grouped_plan_size(experts, experts, &plan_bytes));
                    CUDA_CHECK(cudaMalloc(&selection, (2 * experts + rows * top_k) * sizeof(int32_t)));
                    CUDA_CHECK(cudaMalloc(&counters, 5 * sizeof(uint64_t)));
                    CUDA_CHECK(cudaMalloc(&prior, plan_bytes));
                    CUDA_CHECK(cudaMemset(selection, 0xff, (2 * experts + rows * top_k) * sizeof(int32_t)));
                    CUDA_CHECK(cudaMemset(counters, 0, 5 * sizeof(uint64_t)));
                    CUDA_CHECK(cudaMemset(prior, 0, plan_bytes));
                    const auto predict = [&]() {
                        program->compute(predictor);
                        moe_early_router_select<1, true, true><<<1, 32, 0, predictor.stream()>>>(
                            nullptr, experts, top_k, selection, prior, experts, selection + 2 * experts,
                            selection + experts, counters, 4096, rows, (const int32_t *) program->ids->data, program->ids->nb[1]);
                        CUDA_CHECK(cudaGetLastError());
                    };
                    for (int step = 0; step < 3; ++step) {
                        for (size_t i = 0; i < inputs.size(); ++i) {
                            inputs[i] = std::cos(float(i * 3 + step * 11));
                        }
                        ggml_backend_tensor_set(input, inputs.data(), 0, inputs.size() * sizeof(float));
                        for (int i = 0; i < graph->n_nodes; ++i) {
                            GGML_ASSERT(ggml_cuda_moe_router_compute(reference, graph->nodes[i]));
                        }
                        CUDA_CHECK(cudaStreamSynchronize(reference.stream()));
                        CUDA_CHECK(cudaMemcpyAsync(program->input->data, boundary->data, ggml_nbytes(boundary), cudaMemcpyDeviceToDevice, predictor.stream()));
                        if (step == 0) {
                            predict();
                        } else if (step == 1) {
                            CUDA_CHECK(cudaStreamBeginCapture(predictor.stream(), cudaStreamCaptureModeThreadLocal));
                            predict();
                            CUDA_CHECK(cudaStreamEndCapture(predictor.stream(), &captured));
                            CUDA_CHECK(cudaGraphInstantiate(&replay, captured, nullptr, nullptr, 0));
                            CUDA_CHECK(cudaGraphLaunch(replay, predictor.stream()));
                        } else {
                            CUDA_CHECK(cudaGraphLaunch(replay, predictor.stream()));
                        }
                        CUDA_CHECK(cudaStreamSynchronize(predictor.stream()));
                        for (int row = 0; row < rows; ++row) {
                            CUDA_CHECK(cudaMemcpy(expected.data() + row * top_k, (char *) ids->data + row * ids->nb[1], top_k * sizeof(int32_t), cudaMemcpyDeviceToHost));
                            CUDA_CHECK(cudaMemcpy(actual.data() + row * top_k, (char *) program->ids->data + row * program->ids->nb[1], top_k * sizeof(int32_t), cudaMemcpyDeviceToHost));
                        }
                        valid &= expected == actual;
                        std::unordered_set<int32_t> seen;
                        for (size_t i = 0; i < expected.size(); ++i) {
                            auto & expert = expected[i];
                            if (i % top_k >= size_t((top_k + 1) / 2) || !seen.insert(expert).second) {
                                expert = -1;
                            }
                        }
                        CUDA_CHECK(cudaMemcpy(actual.data(), selection + 2 * experts, actual.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
                        valid &= expected == actual;
                    }
                    CUDA_CHECK(cudaGraphExecDestroy(replay));
                    CUDA_CHECK(cudaGraphDestroy(captured));
                    CUDA_CHECK(cudaFree(prior));
                    CUDA_CHECK(cudaFree(counters));
                    CUDA_CHECK(cudaFree(selection));
                    auto invalid = *ids;
                    invalid.op = GGML_OP_CPY;
                    valid &= moe_router_program::create(boundary, &invalid, impl_->device, reason) == nullptr;
                    auto write = *weights;
                    write.op = GGML_OP_SCALE;
                    write.src[0] = weights;
                    write.view_src = weights;
                    auto unsafe_projection = *projection;
                    unsafe_projection.src[0] = &write;
                    auto * unsafe_ids = ggml_argsort_top_k(ctx, &unsafe_projection, top_k);
                    const bool rejected_write = moe_router_program::create(boundary, unsafe_ids, impl_->device, reason) == nullptr;
                    valid &= rejected_write;
                    if (!rejected_write) {
                        fprintf(stderr, "early-graph-test: accepted a write through a weight alias\n");
                    }
                    auto weight_view = *weights;
                    weight_view.op = GGML_OP_VIEW;
                    weight_view.src[0] = weights;
                    weight_view.view_src = weights;
                    write.src[0] = &weight_view;
                    write.view_src = &weight_view;
                    valid &= moe_router_program::create(boundary, unsafe_ids, impl_->device, reason) == nullptr;
                    auto mutable_input = *input;
                    mutable_input.flags = 0;
                    unsafe_projection.src[0] = router_weights;
                    unsafe_projection.src[1] = ggml_add(ctx, boundary, &mutable_input);
                    valid &= moe_router_program::create(boundary, unsafe_ids, impl_->device, reason) == nullptr;
                    fprintf(stderr, "early-graph-test: variant=%d type=%s rows=%d scratch=%zu exact=%d\n", variant, ggml_type_name(type), rows, program->bytes, valid);
                }
                program.reset();
                ggml_backend_buffer_free(buffer);
                ggml_backend_buffer_free(weights_buffer);
                ggml_free(ctx);
                ggml_free(weights_ctx);
            }
        }
    }
    return valid;
}

bool ggml_cuda_moe_grouped_context::early_select_for_test() {
    CUDA_CHECK(cudaSetDevice(impl_->device));
    bool valid = true;
    int cases = 0;
    for (int experts : {17, 33, 65, 129, 256, 257, 512, 513, 1025, 2049}) {
        const bool long_plan = experts == 513 || experts == 2049;
        const uint32_t capacity = std::min(experts / 2, long_plan ? 60 : 30);
        size_t plan_bytes;
        GGML_ASSERT(moe_grouped_plan_size(capacity, experts, &plan_bytes));
        std::vector<uint64_t> storage((plan_bytes + 7) / 8);
        auto * prior = reinterpret_cast<moe_grouped_decode_plan *>(storage.data());
        prior->n_misses = long_plan ? 40 : 3;
        for (uint32_t i = 0; i < prior->n_misses; ++i) {
            moe_grouped_plan_array_ptr(prior, capacity, MOE_GROUPED_PLAN_MISS_EXPERTS)[i] = experts - 1 - i;
            moe_grouped_plan_array_ptr(prior, capacity, MOE_GROUPED_PLAN_MISS_SLOTS)[i] = i;
        }
        std::vector<float> scores(experts);
        std::vector<int32_t> residents(experts, -1), actual(experts), positions(experts);
        for (uint32_t i = 0; i < capacity; ++i) {
            residents[i] = i;
        }
        float * d_scores;
        int32_t * d_residents, * d_predicted, * d_positions;
        uint64_t * d_counters;
        moe_grouped_decode_plan * d_prior;
        CUDA_CHECK(cudaMalloc(&d_scores, 4 * experts * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_residents, experts * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_predicted, 4 * experts * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_positions, experts * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_counters, 5 * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_prior, plan_bytes));
        CUDA_CHECK(cudaMemcpy(d_residents, residents.data(), experts * sizeof(int32_t), cudaMemcpyHostToDevice));
        for (int pattern = 0; pattern < 4; ++pattern) {
            uint32_t random = 12345;
            for (int e = 0; e < experts; ++e) {
                random = random * 1664525u + 1013904223u;
                scores[e] = pattern == 0 ? (int(random % 101) - 50) / 8.0f : pattern == 1 ? 0.0f : float(e);
                if (pattern == 1 && e % 2 == 0) {
                    scores[e] = -0.0f;
                }
                if (pattern == 3 || e % 37 == 0) {
                    scores[e] = e % 3 == 0 ? NAN : e % 3 == 1 ? INFINITY : -INFINITY;
                }
            }
            std::vector<int> order;
            for (int e = 0; e < experts; ++e) {
                if (std::isfinite(scores[e])) {
                    order.push_back(e);
                }
            }
            std::sort(order.begin(), order.end(), [&](int a, int b) {
                return scores[a] > scores[b] || (scores[a] == scores[b] && a < b);
            });
            CUDA_CHECK(cudaMemcpy(d_scores, scores.data(), experts * sizeof(float), cudaMemcpyHostToDevice));
            for (bool pending : {false, true}) {
                prior->status = pending ? MOE_GROUPED_PLAN_READY : MOE_GROUPED_PLAN_BUILDING;
                CUDA_CHECK(cudaMemcpy(d_prior, prior, plan_bytes, cudaMemcpyHostToDevice));
                auto after = residents;
                if (pending) {
                    for (uint32_t i = 0; i < prior->n_misses; ++i) {
                        after[i] = -1;
                        after[experts - 1 - i] = i;
                    }
                }
                for (int top_k : {1, 10, experts}) {
                    std::vector<int32_t> expected(top_k, -1), expected_positions(experts, -1);
                    uint64_t expected_count = 0;
                    for (int r = 0; r < std::min((top_k + 1) / 2, (int) order.size()); ++r) {
                        if (after[order[r]] < 0) {
                            expected[r] = order[r];
                            expected_positions[order[r]] = r;
                            ++expected_count;
                        }
                    }
                    CUDA_CHECK(cudaMemset(d_counters, 0, 5 * sizeof(uint64_t)));
                    moe_early_router_select_launch(d_scores, experts, top_k, d_residents,
                        d_prior, capacity, d_predicted, d_positions, d_counters, 4096, nullptr);
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaMemcpy(actual.data(), d_predicted, top_k * sizeof(int32_t), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(positions.data(), d_positions, experts * sizeof(int32_t), cudaMemcpyDeviceToHost));
                    uint64_t counters[5];
                    CUDA_CHECK(cudaMemcpy(counters, d_counters, sizeof(counters), cudaMemcpyDeviceToHost));
                    const bool match = std::equal(expected.begin(), expected.end(), actual.begin()) && positions == expected_positions &&
                        counters[0] == expected_count * 4096 && counters[1] == 0 && counters[2] == expected_count && counters[3] == 0 && counters[4] == 1;
                    if (!match) {
                        fprintf(stderr, "early-select-test: mismatch experts=%d top_k=%d pattern=%d pending=%d\n", experts, top_k, pattern, pending);
                    }
                    valid = valid && match;
                    ++cases;
                }
            }
        }
        for (int n_rows : {2, 3, 4}) {
            for (bool shared : {false, true}) {
                const int top_k = std::min(experts, 10);
                std::vector<float> batch_scores(n_rows * experts);
                std::vector<int32_t> expected(n_rows * top_k, -1), expected_positions(experts, -1);
                std::vector<int32_t> batch_actual(n_rows * top_k);
                uint64_t expected_count = 0;
                prior->status = MOE_GROUPED_PLAN_READY;
                auto after = residents;
                for (uint32_t i = 0; i < prior->n_misses; ++i) {
                    after[i] = -1;
                    after[experts - 1 - i] = i;
                }
                for (int row = 0; row < n_rows; ++row) {
                    std::vector<int> order;
                    for (int e = 0; e < experts; ++e) {
                        const float value = e % 19 == 0 ? NAN : float((e + (shared ? 0 : row * 7)) % experts);
                        batch_scores[row * experts + e] = value;
                        if (std::isfinite(value)) {
                            order.push_back(e);
                        }
                    }
                    std::sort(order.begin(), order.end(), [&](int a, int b) {
                        return batch_scores[row * experts + a] > batch_scores[row * experts + b];
                    });
                    for (int rank = 0; rank < std::min((top_k + 1) / 2, (int) order.size()); ++rank) {
                        const int e = order[rank];
                        if (after[e] < 0 && expected_positions[e] < 0) {
                            const int position = row * top_k + rank;
                            expected[position] = e;
                            expected_positions[e] = position;
                            ++expected_count;
                        }
                    }
                }
                CUDA_CHECK(cudaMemcpy(d_scores, batch_scores.data(), batch_scores.size() * sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_prior, prior, plan_bytes, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemset(d_counters, 0, 5 * sizeof(uint64_t)));
                moe_early_router_select_launch<1, true>(d_scores, experts, top_k, d_residents,
                    d_prior, capacity, d_predicted, d_positions, d_counters, 4096, nullptr, n_rows);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpy(batch_actual.data(), d_predicted, batch_actual.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(positions.data(), d_positions, experts * sizeof(int32_t), cudaMemcpyDeviceToHost));
                uint64_t counters[5];
                CUDA_CHECK(cudaMemcpy(counters, d_counters, sizeof(counters), cudaMemcpyDeviceToHost));
                const bool match = batch_actual == expected && positions == expected_positions &&
                    counters[0] == expected_count * 4096 && counters[2] == expected_count && counters[4] == 1;
                if (!match) {
                    fprintf(stderr, "early-select-test: batch mismatch experts=%d rows=%d shared=%d\n", experts, n_rows, shared);
                }
                valid = valid && match;
                ++cases;
            }
        }
        CUDA_CHECK(cudaFree(d_scores));
        CUDA_CHECK(cudaFree(d_residents));
        CUDA_CHECK(cudaFree(d_predicted));
        CUDA_CHECK(cudaFree(d_positions));
        CUDA_CHECK(cudaFree(d_counters));
        CUDA_CHECK(cudaFree(d_prior));
    }
    fprintf(stderr, "early-select-test: cases=%d valid=%d\n", cases, valid);
    return valid;
}

bool ggml_cuda_moe_grouped_context::early_hc_for_test() {
    CUDA_CHECK(cudaSetDevice(impl_->device));
    bool valid = true;
    for (int streams : {3, 4}) {
        constexpr int width = 64, rank = 32;
        const int wide = width * streams;
        std::vector<float> input(wide), attention(wide), ffn(wide), translated(wide), low(rank), gate(wide), expected(width), actual(width);
        for (int i = 0; i < wide; ++i) {
            input[i] = ((i * 7) % 31 - 15) / 16.0f;
            attention[i] = i % 23 == 0 ? 0.0f : (i % 7 + 1) / 8.0f;
            ffn[i] = (i % 11 - 5) / 8.0f;
            translated[i] = attention[i] == 0.0f ? 0.0f : input[i] * ffn[i] / attention[i];
        }
        float * scratch;
        CUDA_CHECK(cudaMalloc(&scratch, (5 * wide + rank + width) * sizeof(float)));
        float * d_input = scratch, * d_attention = scratch + wide, * d_ffn = scratch + 2 * wide;
        float * d_low = scratch + 3 * wide, * d_gate = d_low + rank, * d_mixed = d_gate + wide;
        CUDA_CHECK(cudaMemcpy(d_attention, attention.data(), wide * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ffn, ffn.data(), wide * sizeof(float), cudaMemcpyHostToDevice));
        for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_Q8_0}) {
            std::vector<float> weights(wide * rank);
            std::vector<block_q8_0> quant(weights.size() / QK8_0);
            for (size_t b = 0; b < quant.size(); ++b) {
                quant[b].d = __float2half(1.0f / 256.0f);
                for (int i = 0; i < QK8_0; ++i) {
                    const int v = ((b * QK8_0 + i) * 13) % 31 - 15;
                    quant[b].qs[i] = v;
                    weights[b * QK8_0 + i] = v / 256.0f;
                }
            }
            ggml_tensor down = {}, up = {};
            down.type = up.type = type;
            down.ne[0] = up.ne[1] = wide;
            down.ne[1] = up.ne[0] = rank;
            const size_t bytes = type == GGML_TYPE_F32 ? weights.size() * sizeof(float) : quant.size() * sizeof(block_q8_0);
            CUDA_CHECK(cudaMalloc(&down.data, bytes));
            up.data = down.data;
            CUDA_CHECK(cudaMemcpy(down.data, type == GGML_TYPE_F32 ? (void *) weights.data() : (void *) quant.data(), bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_input, input.data(), wide * sizeof(float), cudaMemcpyHostToDevice));
            moe_early_hc_translate<<<(wide + 255) / 256, 256>>>(d_input, d_attention, d_ffn, wide);
            moe_early_hc_project<128>(&down, d_input, d_low, streams, 1, nullptr);
            moe_early_hc_project<32>(&up, d_low, d_gate, streams, 2, nullptr);
            moe_early_hc_collapse<<<1, 128>>>(d_input, d_gate, d_mixed, width, streams);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(actual.data(), d_mixed, width * sizeof(float), cudaMemcpyDeviceToHost));
            for (int r = 0; r < rank; ++r) {
                double sum = 0;
                for (int i = 0; i < wide; ++i) {
                    sum += (double) weights[r * wide + i] * translated[i];
                }
                sum /= streams;
                low[r] = sum / (1.0 + exp(-sum));
            }
            for (int r = 0; r < wide; ++r) {
                double sum = 0;
                for (int i = 0; i < rank; ++i) {
                    sum += (double) weights[r * rank + i] * low[i];
                }
                gate[r] = 1.0 / (1.0 + exp(-sum));
            }
            float max_error = 0;
            for (int i = 0; i < width; ++i) {
                double sum = 0;
                for (int s = 0; s < streams; ++s) {
                    sum += (double) translated[s * width + i] * gate[s * width + i];
                }
                expected[i] = sum / streams;
                const float error = fabsf(actual[i] - expected[i]);
                valid = valid && std::isfinite(actual[i]) && error < 2e-5f;
                max_error = std::max(max_error, error);
            }
            fprintf(stderr, "early-hc-test: streams=%d type=%s max_abs_error=%g\n", streams, ggml_type_name(type), max_error);
            constexpr int n_rows = 4;
            std::vector<float> batch_input(n_rows * wide), batch_actual(n_rows * width), serial(width);
            for (int row = 0; row < n_rows; ++row) {
                for (int i = 0; i < wide; ++i) {
                    batch_input[row * wide + i] = input[i] + row / 8.0f;
                }
            }
            float * batch_scratch;
            CUDA_CHECK(cudaMalloc(&batch_scratch, n_rows * (2 * wide + rank + width) * sizeof(float)));
            float * batch_low = batch_scratch + n_rows * wide;
            float * batch_gate = batch_low + n_rows * rank;
            float * batch_mixed = batch_gate + n_rows * wide;
            CUDA_CHECK(cudaMemcpy(batch_scratch, batch_input.data(), batch_input.size() * sizeof(float), cudaMemcpyHostToDevice));
            moe_early_hc_translate<<<dim3((wide + 255) / 256, n_rows), 256>>>(batch_scratch, d_attention, d_ffn, wide);
            moe_early_hc_project<128>(&down, batch_scratch, batch_low, streams, 1, nullptr, n_rows);
            moe_early_hc_project<32>(&up, batch_low, batch_gate, streams, 2, nullptr, n_rows);
            moe_early_hc_collapse<<<dim3(1, n_rows), 128>>>(batch_scratch, batch_gate, batch_mixed, width, streams);
            CUDA_CHECK(cudaMemcpy(batch_actual.data(), batch_mixed, batch_actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
            bool batch_valid = true;
            for (int row = 0; row < n_rows; ++row) {
                CUDA_CHECK(cudaMemcpy(d_input, batch_input.data() + row * wide, wide * sizeof(float), cudaMemcpyHostToDevice));
                moe_early_hc_translate<<<(wide + 255) / 256, 256>>>(d_input, d_attention, d_ffn, wide);
                moe_early_hc_project<128>(&down, d_input, d_low, streams, 1, nullptr);
                moe_early_hc_project<32>(&up, d_low, d_gate, streams, 2, nullptr);
                moe_early_hc_collapse<<<1, 128>>>(d_input, d_gate, d_mixed, width, streams);
                CUDA_CHECK(cudaMemcpy(serial.data(), d_mixed, width * sizeof(float), cudaMemcpyDeviceToHost));
                batch_valid = batch_valid && std::equal(serial.begin(), serial.end(), batch_actual.begin() + row * width);
            }
            if (type == GGML_TYPE_F32) {
                std::vector<float> batch_scores(n_rows * rank), serial_scores(rank);
                CUDA_CHECK(cudaMemcpy(batch_scratch, batch_input.data(), batch_input.size() * sizeof(float), cudaMemcpyHostToDevice));
                moe_early_router_scores<<<dim3(rank, n_rows), 128>>>(batch_scratch, d_attention, d_ffn, (const float *) down.data, wide, batch_low);
                CUDA_CHECK(cudaMemcpy(batch_scores.data(), batch_low, batch_scores.size() * sizeof(float), cudaMemcpyDeviceToHost));
                for (int row = 0; row < n_rows; ++row) {
                    moe_early_router_scores<<<rank, 128>>>(batch_scratch + row * wide, d_attention, d_ffn, (const float *) down.data, wide, d_low);
                    CUDA_CHECK(cudaMemcpy(serial_scores.data(), d_low, rank * sizeof(float), cudaMemcpyDeviceToHost));
                    batch_valid = batch_valid && std::equal(serial_scores.begin(), serial_scores.end(), batch_scores.begin() + row * rank);
                }
                std::vector<float> bias(rank);
                for (int r = 0; r < rank; ++r) {
                    bias[r] = (r % 7 - 3) / 16.0f;
                }
                CUDA_CHECK(cudaMemcpy(d_gate, bias.data(), rank * sizeof(float), cudaMemcpyHostToDevice));
                for (float multiplier : {1.0f, 0.125f}) {
                    moe_early_router_scores<<<dim3(rank, n_rows), 128>>>(
                        batch_scratch, d_attention, d_ffn, (const float *) down.data, wide, batch_low, multiplier, d_gate);
                    CUDA_CHECK(cudaMemcpy(batch_scores.data(), batch_low, batch_scores.size() * sizeof(float), cudaMemcpyDeviceToHost));
                    for (int row = 0; row < n_rows; ++row) {
                        for (int r = 0; r < rank; ++r) {
                            double sum = 0;
                            for (int i = 0; i < wide; ++i) {
                                const float value = attention[i] != 0.0f ? batch_input[row * wide + i] * (ffn[i] / attention[i]) : 0.0f;
                                sum += (double) value * weights[r * wide + i];
                            }
                            const float expected_score = sum * multiplier + bias[r];
                            batch_valid &= std::isfinite(batch_scores[row * rank + r]) &&
                                fabsf(batch_scores[row * rank + r] - expected_score) < 2e-5f;
                        }
                    }
                }
            }
            CUDA_CHECK(cudaGetLastError());
            fprintf(stderr, "early-hc-test: streams=%d type=%s rows=%d serial_exact=%d\n", streams, ggml_type_name(type), n_rows, batch_valid);
            valid = valid && batch_valid;
            CUDA_CHECK(cudaFree(batch_scratch));
            CUDA_CHECK(cudaFree(down.data));
        }
        CUDA_CHECK(cudaFree(scratch));
    }
    return valid;
}

bool ggml_cuda_moe_grouped_context::prepack_for_test() {
    {
        moe_host_budget owner(0);
        moe_grouped_materialization materialization(nullptr, 0, 0, 3, 5, 3);
        materialization.copy_spans.resize(3 * GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS);
        moe_prepack_control control = {};
        materialization.control = &control;
        const int32_t experts[] = {4, 1, 3};
        std::vector<char> input(5 * (700003 + 400001)), output(3 * (700003 + 400001)), expected(output.size());
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = static_cast<char>(i % 127);
        }
        moe_host_source a{&owner, input.data(), 5 * 700003, 700003, false};
        moe_host_source b{&owner, input.data() + a.size, 5 * 400001, 400001, false};
        materialization.sources[0] = &a;
        materialization.sources[3] = &b;
        materialization.destinations[0] = output.data();
        materialization.destinations[3] = output.data() + 3 * a.expert_stride;
        auto check_copy = [&](moe_host_copy_worker & worker, uint32_t count, uint32_t adopted) {
            std::fill(output.begin(), output.end(), -1);
            std::fill(expected.begin(), expected.end(), -1);
            for (uint32_t i = 0; i < 3; ++i) {
                control.adopted[i] = (adopted & (1u << i)) != 0 ? input.data() : nullptr;
                if (i >= count || control.adopted[i] != nullptr) {
                    continue;
                }
                memcpy(expected.data() + i * a.expert_stride, a.data + experts[i] * a.expert_stride, a.expert_stride);
                memcpy(expected.data() + 3 * a.expert_stride + i * b.expert_stride, b.data + experts[i] * b.expert_stride, b.expert_stride);
            }
            size_t bytes = 0;
            const size_t spans = materialization.prepare_copy(experts, count, bytes);
            const uint64_t before = owner.materialized_bytes.load();
            return moe_host_copy_spans(worker, materialization.copy_spans.data(), spans, bytes) && output == expected &&
                owner.materialized_bytes.load() - before == bytes;
        };
        moe_host_copy_worker unavailable, failed;
        if (failed.start(true) || failed.start() || !check_copy(unavailable, 3, 0) || !check_copy(failed, 3, 2) || !owner.copy_worker.start()) {
            return false;
        }
        bool valid = true;
        for (uint32_t mask = 0; mask < 8; ++mask) {
            valid &= check_copy(owner.copy_worker, 3, mask);
        }
        valid &= check_copy(owner.copy_worker, 0, 0);
        const moe_host_copy_span small{&a, a.data + 7, output.data() + 1, 3};
        valid &= moe_host_copy_spans(owner.copy_worker, &small, 1, 3) && memcmp(output.data() + 1, a.data + 7, 3) == 0;
        struct barrier {
            std::atomic<bool> entered{false}, released{false};
        } blocked;
        auto wait_for_release = [](void * opaque) {
            auto & value = *static_cast<barrier *>(opaque);
            value.entered.store(true, std::memory_order_release);
            while (!value.released.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
        };
        while (!owner.copy_worker.try_submit(wait_for_release, &blocked)) {
            std::this_thread::yield();
        }
        while (!blocked.entered.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        const uint64_t submitted = owner.copy_worker.submitted;
        valid &= check_copy(owner.copy_worker, 3, 2) && owner.copy_worker.submitted == submitted;
        blocked.released.store(true, std::memory_order_release);
        {
            std::unique_lock<std::mutex> lock(owner.copy_worker.mutex);
            owner.copy_worker.done_cv.wait(lock, [&] { return owner.copy_worker.done; });
        }
        valid &= check_copy(owner.copy_worker, 3, 0) && owner.copy_worker.submitted == submitted;
        owner.copy_worker.wait();
        while (owner.copy_worker.submitted == submitted) {
            valid &= check_copy(owner.copy_worker, 1, 0);
        }
        owner.copy_worker.stop();
        valid &= check_copy(owner.copy_worker, 3, 0);
        if (!valid) {
            return false;
        }
        GGML_LOG_INFO("test-moe-cache: demand helper odd/small/adopted bytes, busy/startup fallback and teardown OK\n");
    }
    constexpr size_t stride = 64;
    constexpr uint32_t experts = 5, top_k = 3;
    moe_host_budget owner(0);
    std::array<char, 2 * experts * stride> source;
    std::array<char, 2 * top_k * 2 * stride> storage;
    for (size_t i = 0; i < source.size(); ++i) {
        source[i] = char(i % 127);
    }
    moe_host_source a{&owner, source.data(), experts * stride, stride, false};
    moe_host_source b{&owner, source.data() + experts * stride, experts * stride, stride, false};
    moe_prepack_request request{12, top_k, experts, 2 * stride, {&a, &b}};
    moe_host_prepack executor(storage.data(), storage.data(), top_k * 2 * stride);
    moe_prepack_control control = {};
    const int32_t misses[] = {3, 4};
    for (uint64_t invocation = 1; invocation <= 1024; ++invocation) {
        control.predicted[0] = 1;
        control.predicted[1] = 3;
        control.predicted[2] = -1;
        executor.visit(11, invocation, nullptr, 0, request, control);
        executor.visit(12, invocation, misses, 2, {}, control);
        if (control.adopted[0] != nullptr || control.adopted[1] != nullptr) {
            return false;
        }
    }
    executor.visit(11, 1025, nullptr, 0, request, control);
    auto * copying = executor.queued();
    GGML_ASSERT(copying != nullptr);
    copying->status = moe_host_prepack::COPYING;
    executor.visit(12, 1025, misses, 2, request, control);
    if (control.adopted[0] != nullptr || copying->status != moe_host_prepack::COPYING || !copying->expired.load()) {
        return false;
    }
    moe_host_prepack::pack(*copying);
    if (copying->copied != 0 || copying->copied_ranks != 0) {
        return false;
    }
    copying->status = moe_host_prepack::FREE;
    executor.visit(12, 1025, misses, 2, {}, control);
    executor.start();
    executor.visit(11, 1025, nullptr, 0, request, control);
    {
        std::unique_lock<std::mutex> lock(executor.mutex);
        if (!executor.wake.wait_for(lock, std::chrono::seconds(2), [&] {
                return std::any_of(executor.tiles.begin(), executor.tiles.end(), [](const auto & tile) { return tile.status == moe_host_prepack::READY; });
            })) {
            return false;
        }
    }
    {
        std::lock_guard<std::mutex> lock(executor.mutex);
        executor.stop = true;
    }
    executor.wake.notify_one();
    executor.worker.join();
    executor.visit(12, 1025, misses, 2, {}, control);
    bool valid = control.adopted[0] != nullptr && control.adopted[1] == nullptr;
    if (valid) {
        valid = memcmp(control.adopted[0], a.data + 3 * stride, stride) == 0 &&
            memcmp(control.adopted[0] + stride, b.data + 3 * stride, stride) == 0;
    }
    executor.visit(11, 1026, nullptr, 0, {}, control);
    std::atomic<bool> locked{false}, released{false};
    std::thread contender([&] {
        std::lock_guard<std::mutex> lock(executor.mutex);
        locked.store(true, std::memory_order_release);
        while (!released.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
    });
    while (!locked.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    executor.visit(11, 1027, nullptr, 0, request, control);
    released.store(true, std::memory_order_release);
    contender.join();
    {
        std::lock_guard<std::mutex> lock(executor.mutex);
        valid &= executor.adopted_bytes == 2 * stride && executor.wrong_bytes == 2 * stride &&
            executor.copied_bytes == 4 * stride;
        valid &= executor.rank_bytes[0][moe_host_prepack::COPIED_BYTES] == 2 * stride &&
            executor.rank_bytes[0][moe_host_prepack::WRONG_BYTES] == 2 * stride &&
            executor.rank_bytes[1][moe_host_prepack::COPIED_BYTES] == 2 * stride &&
            executor.rank_bytes[1][moe_host_prepack::ADOPTED_BYTES] == 2 * stride &&
            executor.proposed_by_rank[0].load() == executor.proposed_by_rank[1].load() &&
            executor.proposed_by_rank[2].load() == 0;
        valid &= std::all_of(executor.tiles.begin(), executor.tiles.end(), [](const auto & tile) { return tile.status == moe_host_prepack::FREE; });
    }
    valid &= executor.dropped.load() != 0 && executor.peak_tiles == 2;
    const uint64_t published = executor.published;
    executor.close_admission();
    executor.visit(11, 1028, nullptr, 0, request, control);
    valid &= executor.published == published && executor.queued() == nullptr && control.adopted[0] == nullptr;

    moe_grouped_materialization materialization(nullptr, 0, 0, 1, experts, top_k);
    materialization.serial = 29;
    std::atomic<bool> finished{false};
    bool publication_valid = true;
    std::thread callback([&] {
        do {
            auto * observed = materialization.prepack.load(std::memory_order_acquire);
            publication_valid &= observed == nullptr || (observed == &executor && materialization.serial == 29);
        } while (!finished.load(std::memory_order_acquire));
    });
    for (int recapture = 0; recapture < 1024; ++recapture) {
        valid &= materialization.publish_prepack(&executor) == (recapture == 0);
    }
    finished.store(true, std::memory_order_release);
    callback.join();
    valid &= publication_valid && materialization.prepack.load() == &executor;

    size_t plan_bytes = 0, control_offset = 0, staging_offset = 0, bytes = 0;
    const size_t page = moe_host_page_size();
    valid &= moe_grouped_plan_size(12, 16, &plan_bytes) && page > GGML_PAD(plan_bytes, alignof(uint4));
    const size_t payload = page - GGML_PAD(plan_bytes, alignof(uint4));
    valid &= moe_grouped_staging_layout(plan_bytes, payload, false, control_offset, staging_offset, bytes) && bytes == page;
    size_t reserved = page;
    valid &= moe_host_reserve_size(2 * page, bytes, reserved) && reserved == 2 * page;
    valid &= moe_grouped_staging_layout(plan_bytes, payload, true, control_offset, staging_offset, bytes) && bytes == page + sizeof(moe_prepack_control);
    reserved = page;
    valid &= !moe_host_reserve_size(3 * page - 1, bytes, reserved) && reserved == page;
    valid &= moe_host_reserve_size(3 * page, bytes, reserved) && reserved == 3 * page;
    valid &= !moe_grouped_staging_layout(SIZE_MAX, 0, false, control_offset, staging_offset, bytes);
    valid &= !moe_grouped_staging_layout(SIZE_MAX - (alignof(uint4) - 1), 0, true, control_offset, staging_offset, bytes);
    valid &= !moe_grouped_staging_layout(0, SIZE_MAX, true, control_offset, staging_offset, bytes);
    valid &= !moe_host_reserve_size(SIZE_MAX, SIZE_MAX, reserved);
    fprintf(stderr, "prepack-test: stalled-worker=1024 copying-cancellation=1 contended-admission=1 closed-admission=1 recapture-publication=1024 budget-boundary=1 byte-identity=%d\n", valid);
    return valid;
}

static void moe_candidate_graph_mmid_fingerprint_add(
        uint64_t & fingerprint,
        int32_t node_index,
        const ggml_tensor * node,
        const ggml_tensor * source) {
    moe_candidate_hash_value(fingerprint, node_index);
    const ggml_tensor * tensors[] = {
        node, source, node != nullptr ? node->src[1] : nullptr, node != nullptr ? node->src[2] : nullptr,
    };
    for (const ggml_tensor * tensor : tensors) {
        moe_candidate_hash_value(fingerprint, reinterpret_cast<uintptr_t>(tensor));
        if (tensor == nullptr) {
            continue;
        }
        moe_candidate_hash_value(fingerprint, tensor->type);
        for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
            moe_candidate_hash_value(fingerprint, tensor->ne[dim]);
            moe_candidate_hash_value(fingerprint, tensor->nb[dim]);
        }
    }
}

static uint64_t moe_candidate_graph_mmid_fingerprint_finish(uint64_t fingerprint, uint32_t count) {
    moe_candidate_hash_value(fingerprint, count);
    return fingerprint != 0 ? fingerprint : UINT64_C(1);
}

static bool moe_candidate_graph_mmid_inventory(
        const ggml_cgraph * cgraph,
        uint32_t * mmid_count,
        uint64_t * mmid_fingerprint) {
    if (cgraph == nullptr || mmid_count == nullptr || mmid_fingerprint == nullptr || cgraph->n_nodes < 0) {
        return false;
    }
    uint32_t count = 0;
    uint64_t fingerprint = UINT64_C(1469598103934665603);
    for (int32_t node_index = 0; node_index < cgraph->n_nodes; ++node_index) {
        const ggml_tensor * node = cgraph->nodes[node_index];
        const ggml_tensor * source = node != nullptr && node->op == GGML_OP_MUL_MAT_ID ? node->src[0] : nullptr;
        if (source == nullptr || source->buffer == nullptr ||
                ggml_is_empty(node) ||
                !ggml_backend_buft_is_cuda_moe_cached(ggml_backend_buffer_get_type(source->buffer))) {
            continue;
        }
        if (count == UINT32_MAX) {
            return false;
        }
        moe_candidate_graph_mmid_fingerprint_add(fingerprint, node_index, node, source);
        ++count;
    }
    *mmid_count = count;
    *mmid_fingerprint = moe_candidate_graph_mmid_fingerprint_finish(fingerprint, count);
    return true;
}

uint64_t ggml_cuda_moe_grouped_context::certify_graph_coverage(
        const ggml_cgraph * cgraph,
        uint32_t * coverage_mmid_count,
        uint64_t * coverage_mmid_fingerprint) {
    if (coverage_mmid_count != nullptr) {
        *coverage_mmid_count = 0;
    }
    if (coverage_mmid_fingerprint != nullptr) {
        *coverage_mmid_fingerprint = 0;
    }
    ggml_cuda_moe_graph_span span;
    if (cgraph == nullptr || !ggml_cuda_moe_graph_span_bounds(cgraph->nodes, cgraph->n_nodes, &span)) {
        return 0;
    }
    uint32_t mmid_count = 0;
    uint64_t mmid_fingerprint = 0;
    if (!moe_candidate_graph_mmid_inventory(cgraph, &mmid_count, &mmid_fingerprint)) {
        return 0;
    }
    const impl::graph_coverage_key key = {
        cgraph->nodes, mmid_fingerprint, cgraph->n_nodes, mmid_count,
    };

    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->draining) {
        return 0;
    }
    if (impl_->next_graph_coverage_epoch == UINT64_MAX) {
        return 0;
    }
    const auto existing = impl_->graph_coverages.find(key);
    if (existing == impl_->graph_coverages.end() &&
            impl_->graph_coverages.size() >= GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS) {
        const auto oldest = std::min_element(
            impl_->graph_coverages.begin(), impl_->graph_coverages.end(),
            [](const auto & first, const auto & second) { return first.second.epoch < second.second.epoch; });
        impl_->graph_coverages.erase(oldest);
    }

    const uint64_t epoch = impl_->next_graph_coverage_epoch + 1;
    try {
        impl_->graph_coverages.insert_or_assign(key, impl::graph_coverage_record{epoch});
    } catch (...) {
        return 0;
    }
    impl_->next_graph_coverage_epoch = epoch;
    if (coverage_mmid_count != nullptr) {
        *coverage_mmid_count = mmid_count;
    }
    if (coverage_mmid_fingerprint != nullptr) {
        *coverage_mmid_fingerprint = mmid_fingerprint;
    }
    return epoch;
}

bool ggml_cuda_moe_grouped_context::recover_graph_coverage(
        const ggml_cgraph * cgraph,
        uint64_t * coverage_epoch,
        uint32_t * coverage_mmid_count,
        uint64_t * coverage_mmid_fingerprint) const {
    if (coverage_epoch != nullptr) {
        *coverage_epoch = 0;
    }
    if (coverage_mmid_count != nullptr) {
        *coverage_mmid_count = 0;
    }
    if (coverage_mmid_fingerprint != nullptr) {
        *coverage_mmid_fingerprint = 0;
    }
    ggml_cuda_moe_graph_span span;
    if (cgraph == nullptr || coverage_epoch == nullptr ||
            !ggml_cuda_moe_graph_span_bounds(cgraph->nodes, cgraph->n_nodes, &span)) {
        return false;
    }

    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->draining) {
        return false;
    }
    uint32_t mmid_count = 0;
    uint64_t mmid_fingerprint = 0;
    if (!moe_candidate_graph_mmid_inventory(cgraph, &mmid_count, &mmid_fingerprint)) {
        return false;
    }
    const impl::graph_coverage_key key = {
        cgraph->nodes, mmid_fingerprint, cgraph->n_nodes, mmid_count,
    };
    const auto it = impl_->graph_coverages.find(key);
    if (it == impl_->graph_coverages.end() || it->second.epoch == 0) {
        return false;
    }
    *coverage_epoch = it->second.epoch;
    if (coverage_mmid_count != nullptr) {
        *coverage_mmid_count = mmid_count;
    }
    if (coverage_mmid_fingerprint != nullptr) {
        *coverage_mmid_fingerprint = mmid_fingerprint;
    }
    return true;
}

bool ggml_cuda_moe_grouped_context::set_clock_bound_for_test(
        const ggml_cuda_moe_grouped_acquisition & acquisition,
        uint64_t clock_bound) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto * resource = impl_->find_resource(acquisition);
    if (impl_->draining || impl_->replacement_pending || resource == nullptr || resource->device == nullptr ||
            impl_->refreshing[acquisition.candidate.group_index] || resource->active_transaction_token != 0) {
        return false;
    }
    resource->device->clock_bound = clock_bound;
    return true;
}

bool ggml_cuda_moe_grouped_context::admission_closed_for_test() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->draining || impl_->replacement_pending;
}

uint64_t ggml_cuda_moe_grouped_context::host_copy_jobs_for_test(const ggml_cuda_moe_candidate_group_key & key) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (key.generation != impl_->state.generation || key.group_index >= impl_->resources.size()) {
        return 0;
    }
    const auto & resource = impl_->resources[key.group_index];
    if (resource == nullptr || resource->device == nullptr || resource->device->source_owner == nullptr) {
        return 0;
    }
    auto & worker = resource->device->source_owner->copy_worker;
    std::lock_guard<std::mutex> worker_lock(worker.mutex);
    return worker.submitted;
}

bool ggml_cuda_moe_grouped_context::attach_prepack_for_test(const ggml_cuda_moe_candidate_group_key & key, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (key.generation != impl_->state.generation || key.group_index >= impl_->resources.size()) {
        return false;
    }
    const auto & resource = impl_->resources[key.group_index];
    if (resource == nullptr || resource->device == nullptr || resource->device->materialization == nullptr) {
        return false;
    }
    auto & materialization = *resource->device->materialization;
    if (impl_->host_prepack == nullptr) {
        // No jobs are published; the ordinary callback only records demand bytes.
        impl_->host_prepack = std::make_unique<moe_host_prepack>(materialization.storage.data, materialization.storage.device_alias, 1);
        auto early = std::make_unique<impl::early_workspace>();
        early->stream = stream;
        impl_->early.push_back(std::move(early));
    }
    if (materialization.prepack.load(std::memory_order_relaxed) == nullptr) {
        impl_->prepack_resources.emplace_back(resource);
        materialization.publish_prepack(impl_->host_prepack.get());
    }
    return impl_->prepack_resources.size() == 1;
}

bool ggml_cuda_moe_grouped_context::has_device_resource_for_test(const ggml_cuda_moe_candidate_group_key & key) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (key.generation != impl_->state.generation || key.group_index >= impl_->resources.size()) {
        return false;
    }
    const auto * resource = impl_->resources[key.group_index].get();
    return resource != nullptr && resource->snapshot.acquisition.candidate.generation == key.generation && resource->device != nullptr;
}

bool ggml_cuda_moe_grouped_context::get_clock_bound_for_test(
        const ggml_cuda_moe_candidate_group_key & key,
        uint64_t * clock_bound) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (clock_bound == nullptr || key.generation != impl_->state.generation || key.group_index >= impl_->resources.size()) {
        return false;
    }
    const auto * resource = impl_->resources[key.group_index].get();
    if (resource == nullptr || resource->snapshot.acquisition.candidate.generation != key.generation || resource->device == nullptr) {
        return false;
    }
    *clock_bound = resource->device->clock_bound;
    return true;
}

bool ggml_cuda_moe_grouped_context::device_slot_for_expert_for_test(
        const ggml_cuda_moe_candidate_group_key & key,
        uint32_t expert,
        int32_t * slot) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (slot == nullptr || key.generation != impl_->state.generation || key.group_index >= impl_->resources.size()) {
        return false;
    }
    const auto * resource = impl_->resources[key.group_index].get();
    if (resource == nullptr || resource->device == nullptr || expert >= resource->device->n_experts ||
            resource->snapshot.acquisition.candidate.generation != key.generation) {
        return false;
    }
    moe_grouped_device_scope device_scope(impl_->device);
    return moe_grouped_cuda_success(cudaMemcpy(
        slot, resource->device->slot_for_expert + expert, sizeof(*slot), cudaMemcpyDeviceToHost));
}

void * ggml_cuda_moe_grouped_context::device_bank_data_for_test(
        const ggml_cuda_moe_candidate_group_key & key,
        const ggml_tensor * tensor) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (tensor == nullptr || key.generation != impl_->state.generation || key.group_index >= impl_->resources.size()) {
        return nullptr;
    }
    const auto * resource = impl_->resources[key.group_index].get();
    if (resource == nullptr || resource->device == nullptr ||
            resource->snapshot.acquisition.candidate.generation != key.generation) {
        return nullptr;
    }
    for (uint32_t bank_index = 0; bank_index < resource->snapshot.banks.size(); ++bank_index) {
        if (resource->snapshot.banks[bank_index].tensor == tensor && bank_index < resource->device->bank_data.size()) {
            return resource->device->bank_data[bank_index];
        }
    }
    return nullptr;
}

const float * ggml_cuda_moe_grouped_context::device_auxiliary_data_for_test(
        const ggml_cuda_moe_candidate_group_key & key,
        const ggml_tensor * tensor) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (tensor == nullptr || key.generation != impl_->state.generation || key.group_index >= impl_->resources.size()) {
        return nullptr;
    }
    const auto * resource = impl_->resources[key.group_index].get();
    if (resource == nullptr || resource->device == nullptr ||
            resource->snapshot.acquisition.candidate.generation != key.generation) {
        return nullptr;
    }
    for (uint32_t auxiliary = 0; auxiliary < resource->snapshot.n_slot_auxiliaries; ++auxiliary) {
        if (resource->snapshot.slot_auxiliaries[auxiliary].tensor == tensor) {
            return resource->device->auxiliary_data[auxiliary];
        }
    }
    return nullptr;
}

const float * ggml_cuda_moe_grouped_context::device_prefill_auxiliary_data_for_test(
        const ggml_cuda_moe_candidate_group_key & key,
        const ggml_tensor * tensor,
        size_t * byte_extent,
        uint64_t * resource_generation) const {
    if (byte_extent != nullptr) {
        *byte_extent = 0;
    }
    if (resource_generation != nullptr) {
        *resource_generation = 0;
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (tensor == nullptr || key.generation != impl_->state.generation || key.group_index >= impl_->resources.size()) {
        return nullptr;
    }
    const auto * resource = impl_->resources[key.group_index].get();
    if (resource == nullptr || resource->device == nullptr ||
            resource->snapshot.acquisition.candidate.generation != key.generation) {
        return nullptr;
    }
    for (uint32_t auxiliary = 0; auxiliary < resource->snapshot.n_slot_auxiliaries; ++auxiliary) {
        if (resource->snapshot.slot_auxiliaries[auxiliary].tensor != tensor) {
            continue;
        }
        const float * data = resource->device->prefill_auxiliary_data[auxiliary];
        if (data == nullptr) {
            return nullptr;
        }
        if (byte_extent != nullptr) {
            *byte_extent = static_cast<size_t>(resource->snapshot.slot_auxiliaries[auxiliary].byte_extent);
        }
        if (resource_generation != nullptr) {
            *resource_generation = resource->snapshot.acquisition.resource_generation;
        }
        return data;
    }
    return nullptr;
}

bool ggml_cuda_moe_grouped_context::prefill_auxiliary_ordering_for_test(
        const ggml_cuda_moe_candidate_group_key & key,
        uint64_t * cross_stream_waits,
        uint64_t * pending_declines) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (cross_stream_waits == nullptr || pending_declines == nullptr || key.generation != impl_->state.generation ||
            key.group_index >= impl_->resources.size() || impl_->resources[key.group_index] == nullptr ||
            impl_->resources[key.group_index]->device == nullptr) {
        return false;
    }
    const auto & device = *impl_->resources[key.group_index]->device;
    *cross_stream_waits = device.prefill_auxiliary_cross_stream_waits;
    *pending_declines = device.prefill_auxiliary_pending_declines;
    return true;
}

bool ggml_cuda_moe_grouped_context::set_prefill_resident_budget_for_test(size_t byte_budget) {
    std::lock_guard<std::mutex> lifecycle_lock(impl_->resource_lifecycle_mutex);
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->draining || impl_->replacement_pending || !impl_->resources.empty() ||
            (impl_->state.accepted && !impl_->table.groups.empty())) {
        return false;
    }
    impl_->prefill_resident_auxiliary_budget = byte_budget;
    return true;
}

bool ggml_cuda_moe_grouped_context::set_original_auxiliary_budget_for_test(size_t byte_budget) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto & budget = *impl_->original_auxiliary_budget;
    std::lock_guard<std::mutex> budget_lock(budget.mutex);
    if (impl_->draining || impl_->replacement_pending || !impl_->resources.empty() || budget.bytes != 0) {
        return false;
    }
    budget.limit = byte_budget;
    return true;
}

size_t ggml_cuda_moe_grouped_context::original_auxiliary_bytes_for_test() const {
    auto & budget = *impl_->original_auxiliary_budget;
    std::lock_guard<std::mutex> lock(budget.mutex);
    return budget.bytes;
}

void ggml_cuda_moe_grouped_context::fail_device_resource_allocation_for_test(uint32_t stage) {
    impl_->fail_device_resource_for_test.store(stage, std::memory_order_relaxed);
}

bool ggml_cuda_moe_grouped_context::device_resource_complete_for_test(
        const ggml_cuda_moe_candidate_group_key & key) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (key.generation != impl_->state.generation || key.group_index >= impl_->resources.size()) {
        return false;
    }
    const auto * resource = impl_->resources[key.group_index].get();
    return resource != nullptr && resource->device != nullptr && resource->device->completion != nullptr &&
        resource->device->has_completion && resource->device->completion_stream != nullptr;
}

bool ggml_cuda_moe_grouped_context::graph_clock_active_for_test(const ggml_cuda_moe_candidate_group_key & key) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (key.generation != impl_->state.generation || key.group_index >= impl_->resources.size()) {
        return false;
    }
    const auto * resource = impl_->resources[key.group_index].get();
    return resource != nullptr && resource->device != nullptr && resource->device->graph_clock_active;
}

size_t ggml_cuda_moe_grouped_context::legacy_backing_count_for_test(
        const ggml_cuda_moe_candidate_group_key & key) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    size_t result = 0;
    for (const auto & entry : impl_->legacy_records) {
        const auto & record = *entry.second;
        if (record.backing != nullptr && record.acquisition.registered_source &&
                record.acquisition.candidate_generation == key.generation && record.acquisition.group_index == key.group_index) {
            ++result;
        }
    }
    return result;
}

void ggml_cuda_moe_grouped_context::fail_borrowed_cache_init_after_probe_for_test() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->fail_borrowed_cache_init_after_probe_for_test = true;
}

void ggml_cuda_moe_grouped_context::poison_split_staging_for_test(uint32_t calls) {
    impl_->split_staging_poison_calls_for_test.store(calls, std::memory_order_relaxed);
}

void ggml_cuda_moe_grouped_context::fail_host_staged_evaluator_for_test() {
    impl_->fail_host_staged_evaluator_for_test.store(true, std::memory_order_relaxed);
}

uint32_t ggml_cuda_moe_grouped_context::split_staging_poison_calls_for_test() const {
    return impl_->split_staging_poison_calls_for_test.load(std::memory_order_relaxed);
}

bool ggml_cuda_moe_take_split_staging_poison_for_test(ggml_cuda_moe_grouped_context * context) {
    if (context == nullptr) {
        return false;
    }
    uint32_t calls = context->impl_->split_staging_poison_calls_for_test.load(std::memory_order_relaxed);
    while (calls > 0) {
        if (context->impl_->split_staging_poison_calls_for_test.compare_exchange_weak(
                calls, calls - 1, std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

bool ggml_cuda_moe_take_host_staged_evaluator_failure_for_test(ggml_cuda_moe_grouped_context * context) {
    return context != nullptr && context->impl_->fail_host_staged_evaluator_for_test.exchange(false, std::memory_order_relaxed);
}

uint64_t ggml_cuda_moe_grouped_context::legacy_op_count_for_test(bool is_decode) const {
    return impl_->legacy_op_stats[moe_cache_phase_index(is_decode)].ops.load(std::memory_order_relaxed);
}

ggml_cuda_moe_legacy_debug_telemetry ggml_cuda_moe_grouped_context::legacy_debug_telemetry_for_test(bool is_decode) const {
    const auto & stats = impl_->legacy_op_stats[moe_cache_phase_index(is_decode)];
    ggml_cuda_moe_legacy_debug_telemetry result;
    result.ops = stats.ops.load(std::memory_order_relaxed);
    result.staged_ops = stats.staged_ops.load(std::memory_order_relaxed);
    result.split_staged_ops = stats.split_staged_ops.load(std::memory_order_relaxed);
    result.overflow_ops = stats.overflow_ops.load(std::memory_order_relaxed);
    result.unique_experts_max = stats.unique_experts_max.load(std::memory_order_relaxed);
    result.ids_cache_hits = stats.ids_cache_hits.load(std::memory_order_relaxed);
    return result;
}

ggml_cuda_moe_grouped_debug_telemetry ggml_cuda_moe_grouped_context::take_grouped_debug_telemetry_for_test() {
    std::lock_guard<std::mutex> lifecycle_lock(impl_->resource_lifecycle_mutex);
    uint32_t registered_groups = 0;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        registered_groups = impl_->state.accepted ? static_cast<uint32_t>(impl_->table.groups.size()) : 0;
    }
    return impl_->take_grouped_debug_telemetry(registered_groups);
}

int32_t ggml_cuda_moe_grouped_context::replace(const ggml_backend_moe_candidate_snapshot_v1 * snapshot) {
    if (snapshot == nullptr) {
        return impl_->publish_failure(GGML_CUDA_MOE_CANDIDATE_REJECT_INVALID_ABI, 0, GGML_BACKEND_MOE_CANDIDATE_REPLACE_INVALID_ARGUMENT);
    }
    if (snapshot->magic != GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V1_MAGIC ||
            snapshot->abi_version != GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V1_VERSION ||
            snapshot->struct_size != sizeof(*snapshot)) {
        return impl_->publish_failure(GGML_CUDA_MOE_CANDIDATE_REJECT_INVALID_ABI, 0, GGML_BACKEND_MOE_CANDIDATE_REPLACE_INVALID_ABI);
    }

    try {
        moe_candidate_table table;
        const auto rejection = moe_candidate_build(impl_->owner, *snapshot, table);
        if (rejection != GGML_CUDA_MOE_CANDIDATE_REJECT_NONE) {
            return impl_->publish_failure(rejection, snapshot->n_slots, GGML_BACKEND_MOE_CANDIDATE_REPLACE_REJECTED);
        }
        return impl_->publish(std::move(table));
    } catch (const std::bad_alloc &) {
        return impl_->publish_failure(GGML_CUDA_MOE_CANDIDATE_REJECT_ALLOCATION, snapshot->n_slots, GGML_BACKEND_MOE_CANDIDATE_REPLACE_ERROR);
    } catch (...) {
        return impl_->publish_failure(GGML_CUDA_MOE_CANDIDATE_REJECT_ALLOCATION, snapshot->n_slots, GGML_BACKEND_MOE_CANDIDATE_REPLACE_ERROR);
    }
}

int32_t ggml_cuda_moe_grouped_context::replace(const ggml_backend_moe_candidate_snapshot_v2 * snapshot) {
    if (snapshot == nullptr) {
        return impl_->publish_failure(GGML_CUDA_MOE_CANDIDATE_REJECT_INVALID_ABI, 0, GGML_BACKEND_MOE_CANDIDATE_REPLACE_INVALID_ARGUMENT);
    }
    if (snapshot->magic != GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_MAGIC ||
            snapshot->abi_version != GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_VERSION || snapshot->struct_size != sizeof(*snapshot)) {
        return impl_->publish_failure(GGML_CUDA_MOE_CANDIDATE_REJECT_INVALID_ABI, 0, GGML_BACKEND_MOE_CANDIDATE_REPLACE_INVALID_ABI);
    }
    try {
        moe_candidate_table table;
        const auto rejection = moe_candidate_build_v2(impl_->owner, *snapshot, table);
        if (rejection != GGML_CUDA_MOE_CANDIDATE_REJECT_NONE) {
            return impl_->publish_failure(rejection, snapshot->n_slots, GGML_BACKEND_MOE_CANDIDATE_REPLACE_REJECTED);
        }
        return impl_->publish(std::move(table));
    } catch (...) {
        return impl_->publish_failure(GGML_CUDA_MOE_CANDIDATE_REJECT_ALLOCATION, snapshot->n_slots, GGML_BACKEND_MOE_CANDIDATE_REPLACE_ERROR);
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
    const auto * entry = moe_candidate_active_reverse(impl_->table, tensor);
    if (!impl_->state.accepted || entry == nullptr) {
        return false;
    }
    if (info != nullptr) {
        *info = impl_->table.groups[entry->group_index].banks[entry->bank_index].info;
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
        info->domain = group.domain;
        info->semantic_group_index = group.semantic_group_index;
        info->flags = group.flags;
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

ggml_cuda_moe_legacy_operation_lease ggml_cuda_moe_grouped_context::begin_legacy_operation() {
    ggml_cuda_moe_legacy_operation_lease result;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->draining || impl_->replacement_pending || impl_->authority_transition_pending ||
            impl_->active_legacy_operations == UINT32_MAX) {
        return result;
    }
    ++impl_->active_legacy_operations;
    result.owner_ = this;
    return result;
}

void ggml_cuda_moe_grouped_context::end_legacy_operation(ggml_cuda_moe_legacy_operation_lease & lease) noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    GGML_ASSERT(lease.owner_ == this && impl_->active_legacy_operations > 0);
    --impl_->active_legacy_operations;
    lease.owner_ = nullptr;
    impl_->resource_cv.notify_all();
}

ggml_cuda_moe_legacy_cache_lease ggml_cuda_moe_grouped_context::acquire_legacy_cache(
        const ggml_tensor * tensor,
        const ggml_cuda_moe_legacy_acquisition * expected,
        const ggml_cuda_moe_group_call_lease * authority,
        cudaStream_t compute_stream,
        uint32_t top_k) {
    ggml_cuda_moe_legacy_cache_lease result;
    if (tensor == nullptr) {
        return result;
    }
    auto authority_matches = [&](const ggml_cuda_moe_legacy_acquisition & acquisition) {
        if (!acquisition.registered_source) {
            return authority == nullptr;
        }
        if (acquisition.candidate_generation != impl_->state.generation ||
                acquisition.group_index >= impl_->table.groups.size()) {
            return false;
        }
        const auto & current = impl_->group_authorities[acquisition.group_index];
        if (current.admission_closed || current.authority != GGML_CUDA_MOE_GROUP_AUTHORITY_LEGACY ||
                acquisition.group_authority_epoch != current.epoch) {
            return false;
        }
        return authority == nullptr ||
            (authority->owner_ == this && authority->candidate_generation_ == impl_->state.generation &&
                authority->authority_epoch_ == current.epoch && authority->group_index_ == acquisition.group_index &&
                authority->authority_ == GGML_CUDA_MOE_GROUP_AUTHORITY_LEGACY && current.active_calls != 0);
    };

    auto acquire_record = [&](bool allow_missing_cache) {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (impl_->draining || impl_->replacement_pending || impl_->authority_transition_pending ||
                impl_->state.generation == 0 || impl_->state.n_slots == 0) {
            return false;
        }
        auto existing = impl_->legacy_records.find(tensor);
        if (existing == impl_->legacy_records.end() || !impl::legacy_record_matches(*existing->second, tensor)) {
            return false;
        }
        auto * record = existing->second.get();
        if ((!allow_missing_cache && record->cache == nullptr) || record->active_leases == UINT32_MAX ||
                !authority_matches(record->acquisition)) {
            return false;
        }
        if (expected != nullptr) {
            const auto & acquisition = record->acquisition;
            if (expected->owner != this || expected->tensor != tensor ||
                    expected->candidate_generation != acquisition.candidate_generation ||
                    expected->authority_epoch != acquisition.authority_epoch ||
                    expected->group_authority_epoch != acquisition.group_authority_epoch ||
                    expected->group_index != acquisition.group_index || expected->role != acquisition.role ||
                    expected->n_slots != acquisition.n_slots || expected->registered_source != acquisition.registered_source) {
                return false;
            }
        }
        record->active_leases++;
        if (record->backing != nullptr) {
            record->backing->legacy_dirty = true;
        }
        result.owner_ = this;
        result.record_ = record;
        result.cache_ = record->cache;
        result.acquisition_ = record->acquisition;
        return true;
    };

    if (acquire_record(impl_->device < 0)) {
        return result;
    }

    impl::legacy_record_map retired;
    impl::legacy_record * record = nullptr;
    bool installed = false;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (impl_->draining || impl_->replacement_pending || impl_->authority_transition_pending ||
                impl_->state.generation == 0 || impl_->state.n_slots == 0) {
            return result;
        }

        auto existing = impl_->legacy_records.find(tensor);
        if (existing != impl_->legacy_records.end() &&
                !impl::legacy_record_matches(*existing->second, tensor)) {
            if (existing->second->active_leases != 0 || existing->second->building_cache || expected != nullptr) {
                return result;
            }
            retired.emplace(existing->first, std::move(existing->second));
            impl_->legacy_records.erase(existing);
            existing = impl_->legacy_records.end();
        }
        if (expected != nullptr && existing == impl_->legacy_records.end()) {
            return result;
        }
        if (existing == impl_->legacy_records.end()) {
            if (tensor->data == nullptr || tensor->buffer == nullptr || tensor->view_src != nullptr || tensor->op != GGML_OP_NONE ||
                    ggml_n_dims(tensor) < 3 || tensor->ne[2] <= 0 || tensor->nb[2] == 0 || impl_->next_legacy_authority_epoch == UINT64_MAX) {
                return result;
            }

            std::unique_ptr<impl::legacy_record> prospective;
            try {
                prospective = std::make_unique<impl::legacy_record>();
            } catch (const std::bad_alloc &) {
                return result;
            }
            prospective->acquisition.owner = this;
            prospective->acquisition.tensor = tensor;
            prospective->acquisition.candidate_generation = impl_->state.generation;
            prospective->acquisition.authority_epoch = ++impl_->next_legacy_authority_epoch;
            prospective->acquisition.n_slots = impl_->state.n_slots;
            prospective->source_data = tensor->data;
            prospective->buffer = tensor->buffer;
            prospective->buft = tensor->buffer->buft;
            prospective->byte_extent = ggml_nbytes(tensor);
            prospective->type = tensor->type;
            for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
                prospective->ne[dim] = tensor->ne[dim];
                prospective->nb[dim] = tensor->nb[dim];
            }
            if (impl_->state.accepted) {
                const auto * reverse = moe_candidate_active_reverse(impl_->table, tensor);
                if (reverse != nullptr) {
                    const auto & bank = impl_->table.groups[reverse->group_index].banks[reverse->bank_index];
                    if (!moe_candidate_record_matches(bank, tensor) ||
                            bank.info.movement != GGML_CUDA_MOE_CANDIDATE_MOVEMENT_SLOT_BOUND) {
                        return result;
                    }
                    const auto & group_authority = impl_->group_authorities[reverse->group_index];
                    if (group_authority.admission_closed ||
                            group_authority.authority != GGML_CUDA_MOE_GROUP_AUTHORITY_LEGACY) {
                        return result;
                    }
                    prospective->acquisition.group_index = reverse->group_index;
                    prospective->acquisition.role = bank.info.role;
                    prospective->acquisition.group_authority_epoch = group_authority.epoch;
                    prospective->acquisition.registered_source = 1;
                }
            }
            if (!authority_matches(prospective->acquisition)) {
                return result;
            }
            try {
                existing = impl_->legacy_records.emplace(tensor, std::move(prospective)).first;
            } catch (const std::bad_alloc &) {
                return result;
            }
        }

        record = existing->second.get();
        if (expected != nullptr) {
            const auto & acquisition = record->acquisition;
            if (expected->owner != this || expected->tensor != tensor ||
                    expected->candidate_generation != acquisition.candidate_generation ||
                    expected->authority_epoch != acquisition.authority_epoch ||
                    expected->group_authority_epoch != acquisition.group_authority_epoch ||
                    expected->group_index != acquisition.group_index || expected->role != acquisition.role ||
                    expected->n_slots != acquisition.n_slots || expected->registered_source != acquisition.registered_source) {
                return result;
            }
        }
        if (!authority_matches(record->acquisition)) {
            return result;
        }
        if (record->cache != nullptr) {
            if (record->active_leases == UINT32_MAX) {
                return result;
            }
            record->active_leases++;
            if (record->backing != nullptr) {
                record->backing->legacy_dirty = true;
            }
            result.owner_ = this;
            result.record_ = record;
            result.cache_ = record->cache;
            result.acquisition_ = record->acquisition;
            return result;
        }
        if (impl_->device < 0) {
            record->active_leases++;
            result.owner_ = this;
            result.record_ = record;
            result.acquisition_ = record->acquisition;
            return result;
        }
        if (record->building_cache || impl_->active_maintenance == UINT32_MAX) {
            return result;
        }
        if (!impl_->legacy_policy_initialized) {
            impl_->legacy_debug_mm.store(g_moe_cache_mm_debug.load(std::memory_order_relaxed), std::memory_order_relaxed);
            impl_->legacy_policy_initialized = true;
        }
        record->building_cache = true;
        ++impl_->active_maintenance;
    }
    impl_->retire_legacy_records(std::move(retired));

    ggml_cuda_moe_cache * prospective = nullptr;
    std::shared_ptr<impl::grouped_resource> backing;
    void * slot_pool_d = nullptr;
    cudaEvent_t grouped_done = nullptr;
    bool stable_backing_required = false;
    if (record->acquisition.registered_source) {
        const ggml_cuda_moe_candidate_group_key key = {
            record->acquisition.candidate_generation,
            record->acquisition.group_index,
        };
        ggml_cuda_moe_grouped_acquisition acquisition;
        if (impl_->acquire_group_resources_impl(key, &acquisition)) {
            bool needs_device = false;
            {
                std::lock_guard<std::mutex> lock(impl_->mutex);
                if (record->building_cache && authority_matches(record->acquisition) &&
                        key.group_index < impl_->resources.size()) {
                    backing = impl_->resources[key.group_index];
                    if (backing != nullptr && backing->device == nullptr && !backing->building_device && compute_stream != nullptr) {
                        backing->building_device = true;
                        needs_device = true;
                    } else if (backing != nullptr && backing->device == nullptr && backing->building_device) {
                        stable_backing_required = true;
                    }
                }
            }
            if (needs_device) {
                auto device = impl_->make_device_resource(
                    backing->snapshot, compute_stream,
                    authority != nullptr && authority->prefill_resident_certified_, top_k);
                std::lock_guard<std::mutex> lock(impl_->mutex);
                if (record->building_cache && authority_matches(record->acquisition) &&
                        key.group_index < impl_->resources.size() && impl_->resources[key.group_index] == backing) {
                    if (device != nullptr && backing->device == nullptr && impl_->next_device_resource_serial != UINT64_MAX) {
                        device->set_serial(++impl_->next_device_resource_serial);
                        backing->device = std::move(device);
                    }
                    backing->building_device = false;
                }
            }
            {
                std::lock_guard<std::mutex> lock(impl_->mutex);
                if (record->building_cache && authority_matches(record->acquisition) &&
                        key.group_index < impl_->resources.size() && impl_->resources[key.group_index] == backing &&
                        backing != nullptr && backing->device != nullptr) {
                    stable_backing_required = true;
                    for (uint32_t bank_index = 0; bank_index < backing->snapshot.banks.size(); ++bank_index) {
                        const auto & bank = backing->snapshot.banks[bank_index];
                        if (bank.tensor == tensor && bank.source_data == tensor->data && bank.expert_stride == tensor->nb[2] &&
                                bank_index < backing->device->bank_data.size()) {
                            slot_pool_d = backing->device->bank_data[bank_index];
                            break;
                        }
                    }
                    if (backing->device->has_completion) {
                        grouped_done = backing->device->completion;
                    }
                }
            }
        }
    }
    bool fail_after_stream_probe = false;
    const size_t trailing_padding = moe_cache_quantized_source_padding(tensor->type, tensor->ne[0]);
    if (slot_pool_d != nullptr) {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        fail_after_stream_probe = impl_->fail_borrowed_cache_init_after_probe_for_test;
        impl_->fail_borrowed_cache_init_after_probe_for_test = false;
    }
    try {
        if (slot_pool_d != nullptr) {
            prospective = ggml_cuda_moe_cache_init_with_pool(
                impl_->device, tensor->nb[2], trailing_padding, record->acquisition.n_slots, slot_pool_d, grouped_done, fail_after_stream_probe);
        } else if (!stable_backing_required) {
            prospective = ggml_cuda_moe_cache_init_with_pool(
                impl_->device, tensor->nb[2], trailing_padding, record->acquisition.n_slots, nullptr, nullptr, false);
        }
        if (prospective != nullptr && !ggml_cuda_moe_cache_set_metadata(prospective, tensor)) {
            ggml_cuda_moe_cache_free(prospective);
            prospective = nullptr;
        }
    } catch (...) {
        ggml_cuda_moe_cache_free(prospective);
        prospective = nullptr;
    }

    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        auto existing = impl_->legacy_records.find(tensor);
        if (!impl_->draining && !impl_->replacement_pending && !impl_->authority_transition_pending && existing != impl_->legacy_records.end() &&
                existing->second.get() == record && record->building_cache && impl::legacy_record_matches(*record, tensor) && record->cache == nullptr &&
                record->active_leases != UINT32_MAX && prospective != nullptr && authority_matches(record->acquisition)) {
            record->cache = prospective;
            record->backing = slot_pool_d != nullptr ? backing : nullptr;
            if (record->backing != nullptr) {
                record->backing->legacy_dirty = true;
            }
            prospective = nullptr;
            record->active_leases++;
            result.owner_ = this;
            result.record_ = record;
            result.cache_ = record->cache;
            result.acquisition_ = record->acquisition;
            installed = true;
        }
        if (existing != impl_->legacy_records.end() && existing->second.get() == record) {
            record->building_cache = false;
        }
        GGML_ASSERT(impl_->active_maintenance > 0);
        --impl_->active_maintenance;
        impl_->resource_cv.notify_all();
    }
    ggml_cuda_moe_cache_free(prospective);
    if (installed) {
        GGML_LOG_INFO("moe-cache: CUDA_MoE_Cache_Pool[%-32s] = %7.2f MiB  (%u slots x %.2f MiB)\n",
            tensor->name[0] ? tensor->name : "?",
            ((double) record->acquisition.n_slots * tensor->nb[2]) / 1024.0 / 1024.0,
            record->acquisition.n_slots,
            tensor->nb[2] / 1024.0 / 1024.0);
    }
    return result;
}

void ggml_cuda_moe_grouped_context::release_legacy_cache(ggml_cuda_moe_legacy_cache_lease & lease) noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto * record = static_cast<impl::legacy_record *>(lease.record_);
    GGML_ASSERT(lease.owner_ == this && record != nullptr && record->active_leases > 0);
    record->active_leases--;
    lease.owner_ = nullptr;
    lease.record_ = nullptr;
    lease.cache_ = nullptr;
    lease.acquisition_ = {};
    impl_->resource_cv.notify_all();
}

void ggml_cuda_moe_grouped_context::prefetch_legacy_siblings(
        const ggml_cuda_moe_legacy_cache_lease & source,
        const int32_t * expert_ids,
        int n_expert_ids,
        bool is_decode) {
    if (source.owner_ != this || source.record_ == nullptr || expert_ids == nullptr || n_expert_ids <= 0 ||
            !source.acquisition_.registered_source || source.acquisition_.group_index == UINT32_MAX) {
        return;
    }

    std::vector<const ggml_tensor *> siblings;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        auto * record = static_cast<impl::legacy_record *>(source.record_);
        if (impl_->draining || impl_->replacement_pending || record->active_leases == 0 ||
                source.acquisition_.candidate_generation != impl_->state.generation ||
                source.acquisition_.group_index >= impl_->table.groups.size() ||
                source.acquisition_.group_authority_epoch != impl_->group_authorities[source.acquisition_.group_index].epoch ||
                impl_->group_authorities[source.acquisition_.group_index].authority != GGML_CUDA_MOE_GROUP_AUTHORITY_LEGACY ||
                impl_->group_authorities[source.acquisition_.group_index].admission_closed) {
            return;
        }
        const auto & group = impl_->table.groups[source.acquisition_.group_index];
        for (const auto & bank : group.banks) {
            const uint32_t role = bank.info.role;
            const bool base_role = role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT ||
                role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT || role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT ||
                role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT;
            if (base_role && bank.info.movement == GGML_CUDA_MOE_CANDIDATE_MOVEMENT_SLOT_BOUND &&
                    bank.info.tensor != source.acquisition_.tensor) {
                siblings.push_back(bank.info.tensor);
            }
        }
    }

    for (const ggml_tensor * tensor : siblings) {
        auto lease = acquire_legacy_cache(tensor);
        ggml_cuda_moe_cache * cache = lease.get();
        if (cache == nullptr) {
            continue;
        }
        cudaStream_t copy_stream = ggml_cuda_moe_cache_copy_stream(cache);
        const char * source_data = static_cast<const char *>(tensor->data);
        for (int i = 0; i < n_expert_ids; ++i) {
            const int32_t expert = expert_ids[i];
            if (expert < 0 || expert >= tensor->ne[2]) {
                continue;
            }
            const void * expert_data = source_data + (size_t) expert * tensor->nb[2];
            (void) ggml_cuda_moe_cache_acquire(
                cache, expert_data, tensor->nb[2], copy_stream, is_decode, true, false);
        }
    }
}

void ggml_cuda_moe_grouped_context::record_legacy_op(
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
    if (!impl_->legacy_debug_mm.load(std::memory_order_relaxed)) {
        return;
    }
    moe_cache_op_phase_stats & stats = impl_->legacy_op_stats[moe_cache_phase_index(is_decode)];
    stats.ops.fetch_add(1, std::memory_order_relaxed);
    stats.staged_ops.fetch_add(staged ? 1 : 0, std::memory_order_relaxed);
    stats.split_staged_ops.fetch_add(split_staged ? 1 : 0, std::memory_order_relaxed);
    stats.overflow_ops.fetch_add(overflow ? 1 : 0, std::memory_order_relaxed);
    stats.unique_experts.fetch_add(unique_experts, std::memory_order_relaxed);
    moe_cache_atomic_max(stats.unique_experts_max, unique_experts);
    stats.ids_bytes.fetch_add(ids_bytes, std::memory_order_relaxed);
    stats.ids_d2h_time_us.fetch_add(ids_d2h_time_us, std::memory_order_relaxed);
    stats.ids_d2h_sync_count.fetch_add(ids_d2h_sync_count, std::memory_order_relaxed);
    stats.ids_cache_hits.fetch_add(ids_cache_hit ? 1 : 0, std::memory_order_relaxed);
    stats.acquire_time_us.fetch_add(acquire_time_us, std::memory_order_relaxed);
    stats.remap_time_us.fetch_add(remap_time_us, std::memory_order_relaxed);
    stats.copy_wait_event_count.fetch_add(copy_wait_event_count, std::memory_order_relaxed);
    stats.copy_wait_event_time_us.fetch_add(copy_wait_event_time_us, std::memory_order_relaxed);
    stats.total_time_us.fetch_add(total_time_us, std::memory_order_relaxed);
}

void ggml_cuda_moe_grouped_context::record_host_staged_op(bool split_staged) {
    auto * debug = impl_->debug_stats();
    if (debug == nullptr) {
        return;
    }
    debug->host_staged_ops.fetch_add(1, std::memory_order_relaxed);
    debug->host_staged_split_ops.fetch_add(split_staged ? 1 : 0, std::memory_order_relaxed);
}

void ggml_cuda_moe_grouped_context::record_required_grouped_failure() {
    auto * debug = impl_->debug_stats();
    if (debug != nullptr) {
        debug->required_unsupported.fetch_add(1, std::memory_order_relaxed);
    }
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
        const auto * entry = moe_candidate_active_reverse(impl_->table, observed.weight);
        if (entry == nullptr) {
            return false;
        }
        const auto & record = impl_->table.groups[entry->group_index].banks[entry->bank_index];
        if (!moe_candidate_record_matches(record, observed.weight) ||
                (observed.expected_role != GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID && observed.expected_role != record.info.role) ||
                (i > 0 && entry->group_index != group_index)) {
            return false;
        }
        group_index = entry->group_index;
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
    return impl_->acquire_group_resources_impl(key, acquisition);
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
            impl_->refreshing[acquisition.candidate.group_index] || resource->active_transaction_token != 0 ||
            (resource->legacy_dirty && impl_->group_authorities[acquisition.candidate.group_index].authority !=
                GGML_CUDA_MOE_GROUP_AUTHORITY_GROUPED) ||
            impl_->next_transaction_token == UINT64_MAX) {
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
    resource->active_decode_stream = nullptr;
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

void ggml_cuda_moe_grouped_context::configure_early_router(
        const ggml_cgraph * graph, ggml_cuda_moe_graph_execution * execution, cudaStream_t stream, bool capture,
        ggml_backend_cuda_context & parent) try {
    GGML_UNUSED(parent);
    impl_->early_bindings.clear();
    impl_->prepack_callbacks.clear();
    if (!moe_early_router_enabled() || !capture || execution == nullptr || execution->plan_ == nullptr ||
            execution->outcome() != GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED) {
        return;
    }
    const auto & certificate = execution->plan_->execution_certificate_;
    if (certificate.domain != GGML_GRAPH_EXECUTION_DOMAIN_MAIN ||
            certificate.row_semantics != GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT) {
        fprintf(stderr, "moe-prepack: disabled reason=requires independent MAIN decode\n");
        return;
    }
    const bool native_only = false;
    size_t max_bytes = 0;
    uint32_t max_width = 0, max_experts = 0, max_top_k = 0, max_hc_rank = 0, hc_groups = 0, max_rows = 1;
    std::unordered_map<const ggml_tensor *, int> node_order;
    for (int i = 0; i < graph->n_nodes; ++i) {
        node_order.emplace(graph->nodes[i], i);
    }
    const auto gpu_f32 = [&](const ggml_tensor * t) {
        return t != nullptr && t->data != nullptr && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) &&
            t->buffer != nullptr && t->buffer->buft == ggml_backend_cuda_buffer_type(impl_->device);
    };
    for (uint32_t index = 0; index < execution->n_groups_; ++index) {
        const auto & group = execution->groups_[index];
        if (group.strategy != GGML_CUDA_MOE_EXECUTION_STRATEGY_DEVICE_DIRECT ||
                group.key.ids.ne[1] != 1 || group.key.ids.ne[0] > MOE_PREPACK_ROUTES || group.key.ids.ne[2] != 1 || group.key.ids.ne[3] != 1 ||
                group.stream != stream || group.first_reader == nullptr || group.key.candidate.group_index >= impl_->resources.size()) {
            continue;
        }
        const auto & resource = impl_->resources[group.key.candidate.group_index];
        if (resource == nullptr || resource->device == nullptr || resource->device->materialization == nullptr ||
                resource->device->materialization->storage.tiles < uint32_t(group.key.ids.ne[0])) {
            continue;
        }
        const auto decline = [&](const char * reason) {
            fprintf(stderr, "moe-early-router: group=%u prediction=disabled reason=%s; grouped demand unchanged\n", index, reason);
        };
        const auto * ids = group.first_reader->src[2];
        const auto * scores = moe_router_scores_node(ids);
        const auto * router = moe_router_projection(ids);
        if (scores == nullptr || (router != nullptr && (router->src[0] == nullptr || router->src[1] == nullptr))) {
            decline("no unambiguous graph-described router scores");
            continue;
        }
        const auto * late = moe_router_boundary(router != nullptr ? router->src[1] : scores);
        float ffn_multiplier = 1.0f;
        const bool normalized = moe_router_normalized(late, ffn_multiplier);
        if (!gpu_f32(late) || ggml_nelements(late) != late->ne[0] * group.key.ids.ne[1] ||
                (normalized && !gpu_f32(late->src[1])) ||
                (router == nullptr && late == scores) ||
                (router != nullptr && (router->src[0]->ne[1] != resource->device->n_experts || ggml_n_dims(router->src[0]) != 2))) {
            decline("unsupported router boundary shape or device");
            continue;
        }
        const ggml_tensor * input = late;
        const ggml_tensor * trigger = router != nullptr ? router : scores;
        const ggml_tensor * ffn_trigger = trigger;
        float input_multiplier = ffn_multiplier;
        const auto previous = index == 0 ? node_order.end() : node_order.find(execution->groups_[index - 1].last_reader);
        const int begin = previous == node_order.end() ? 0 : previous->second + 1;
        const auto late_pos = node_order.find(late);
        if (late_pos == node_order.end()) {
            decline("router boundary is outside this backend graph");
            continue;
        }
        for (int i = late_pos->second + 1; i < graph->n_nodes; ++i) {
            const auto * node = graph->nodes[i];
            if (node->op == GGML_OP_MUL_MAT && node->src[1] == late) {
                ffn_trigger = node;
                trigger = node;
                break;
            }
        }
        if (normalized) {
            for (int i = begin; i < late_pos->second; ++i) {
                const auto * node = graph->nodes[i];
                float multiplier;
                if (node->op == GGML_OP_MUL_MAT && moe_router_normalized(node->src[1], multiplier) &&
                        gpu_f32(node->src[1]) && gpu_f32(node->src[1]->src[1]) &&
                        ggml_are_same_shape(node->src[1], late)) {
                    input = node->src[1];
                    input_multiplier = multiplier;
                    trigger = node;
                    break;
                }
            }
        }
        const ggml_tensor * router_bias = nullptr;
        const ggml_tensor * hc_down = nullptr;
        const ggml_tensor * hc_up = nullptr;
        uint32_t hc_streams = 0;
        const bool plain_selection = router != nullptr && moe_router_plain_selection(ids, router, &router_bias);
        const bool router_f32 = router != nullptr && gpu_f32(router->src[0]) &&
            (router_bias == nullptr || (gpu_f32(router_bias) && ggml_nelements(router_bias) == resource->device->n_experts));
        bool hc = !native_only && normalized && plain_selection && router_f32 && router_bias == nullptr &&
            input_multiplier == 1.0f && ffn_multiplier == 1.0f &&
            moe_router_hc_pattern(router->src[1], late, router, &hc_down, &hc_up, hc_streams);
        if (hc) {
            const auto gpu_hc_weight = [&](const ggml_tensor * t) {
                return t != nullptr && t->data != nullptr && ggml_n_dims(t) == 2 && ggml_is_contiguous(t) &&
                    (t->type == GGML_TYPE_F32 || (t->type == GGML_TYPE_Q8_0 && t->ne[0] % QK8_0 == 0)) &&
                    t->buffer != nullptr && t->buffer->buft == ggml_backend_cuda_buffer_type(impl_->device);
            };
            hc = gpu_hc_weight(hc_down) && gpu_hc_weight(hc_up) &&
                hc_down->ne[0] == late->ne[0] && hc_up->ne[1] == late->ne[0] && hc_up->ne[0] == hc_down->ne[1];
        }
        const bool plain = !native_only && normalized && plain_selection && router_f32 &&
            router->src[1] == late && router->src[0]->ne[0] == late->ne[0];
        if (hc) {
            max_hc_rank = std::max(max_hc_rank, (uint32_t) hc_up->ne[0]);
            ++hc_groups;
        } else {
            hc_down = hc_up = nullptr;
            hc_streams = 0;
        }
        if (!plain && !hc) {
            decline("unsupported prediction geometry");
            continue;
        }
        const uint32_t top_k = group.key.ids.ne[0];
        const uint32_t n_rows = group.key.ids.ne[1];
        max_rows = std::max(max_rows, n_rows);
        max_bytes = std::max(max_bytes, resource->device->words_per_miss * sizeof(uint4) * top_k);
        max_width = std::max(max_width, (uint32_t) late->ne[0]);
        max_experts = std::max(max_experts, resource->device->n_experts);
        max_top_k = std::max(max_top_k, top_k);
        impl::early_binding binding{
            group.key.candidate.group_index, top_k, input, normalized ? input->src[1] : nullptr,
            normalized ? late->src[1] : nullptr, router != nullptr ? router->src[0] : nullptr, hc_down, hc_up, hc_streams,
            late, ffn_trigger, input_multiplier, ffn_multiplier, router_bias};
        binding.order = index;
        if (!impl_->early_bindings.emplace(trigger, binding).second) {
            decline("prediction trigger is already owned");
        }
    }

    std::vector<impl::early_binding> ordered;
    for (const auto & entry : impl_->early_bindings) {
        ordered.push_back(entry.second);
    }
    std::sort(ordered.begin(), ordered.end(), [](const auto & a, const auto & b) { return a.order < b.order; });
    impl_->early_bindings.clear();
    if (ordered.size() < 2) {
        fprintf(stderr, "moe-prepack: disabled reason=no adjacent staged groups\n");
        return;
    }
    moe_host_budget * owner = nullptr;
    for (uint32_t i = 0; i < execution->n_groups_; ++i) {
        const auto & group = execution->groups_[i];
        if (group.key.ids.ne[1] != 1 || group.stream != stream ||
                group.key.candidate.group_index >= impl_->resources.size()) {
            return;
        }
        const auto & resource = impl_->resources[group.key.candidate.group_index];
        if (resource == nullptr || resource->device == nullptr ||
                (resource->device->has_completion && resource->device->completion_stream != stream)) {
            return;
        }
        if (resource->device->materialization != nullptr) {
            auto * current = resource->device->source_owner;
            if (owner != nullptr && owner != current) {
                return;
            }
            owner = current;
        }
    }
    if (owner == nullptr || max_bytes > SIZE_MAX / 2) {
        return;
    }
    if (impl_->early.empty()) {
        size_t rounded = 0;
        if (!moe_host_round_size(2 * max_bytes, rounded)) {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(owner->mutex);
            const size_t committed = owner->source_bytes + std::max(owner->staging_bytes, owner->staging_reserved + owner->staging_optional_bytes);
            if (committed > owner->limit || rounded > owner->limit - committed) {
                fprintf(stderr, "moe-prepack: disabled reason=optional host budget\n");
                return;
            }
        }
        auto storage = std::make_unique<moe_host_allocation>(owner, 2 * max_bytes, 0, 1, true);
        if (storage->data == nullptr) {
            return;
        }
        auto early = std::make_unique<impl::early_workspace>();
        early->stream = stream;
        early->width = max_width;
        early->experts = max_experts;
        early->top_k = max_top_k;
        early->hc_rank = max_hc_rank;
        if (!moe_grouped_cuda_success(cudaMalloc(&early->input, max_width * sizeof(float)))) {
            return;
        }
        if (max_hc_rank != 0) {
            if (!moe_grouped_cuda_success(cudaMalloc(&early->hc_low, max_hc_rank * sizeof(float))) ||
                    !moe_grouped_cuda_success(cudaMalloc(&early->hc_gate, max_width * sizeof(float))) ||
                    !moe_grouped_cuda_success(cudaMalloc(&early->hc_mixed, max_width * sizeof(float)))) {
                return;
            }
        }
        if (!moe_grouped_cuda_success(cudaMalloc(&early->scores, max_experts * sizeof(float))) ||
                !moe_grouped_cuda_success(cudaMalloc(&early->predicted, max_top_k * sizeof(int32_t))) ||
                !moe_grouped_cuda_success(cudaMalloc(&early->positions, max_experts * sizeof(int32_t))) ||
                !moe_grouped_cuda_success(cudaMalloc(&early->counters, 5 * sizeof(uint64_t))) ||
                !moe_grouped_cuda_success(cudaMemsetAsync(early->counters, 0, 5 * sizeof(uint64_t), stream))) {
            return;
        }
        auto executor = std::make_unique<moe_host_prepack>(storage->data, storage->device_alias, max_bytes);
        early->initialized = true;
        executor->start();
        impl_->prepack_storage = std::move(storage);
        impl_->host_prepack = std::move(executor);
        impl_->early.push_back(std::move(early));
    }
    const auto & early = *impl_->early[0];
    if (early.stream != stream || impl_->prepack_storage->owner != owner || early.width < max_width ||
            early.experts < max_experts || early.top_k < max_top_k || early.hc_rank < max_hc_rank ||
            impl_->host_prepack->tile_bytes < max_bytes) {
        fprintf(stderr, "moe-prepack: disabled reason=changed stream or capacity\n");
        return;
    }
    for (size_t i = 1; i < ordered.size(); ++i) {
        auto binding = ordered[i];
        const auto & source = ordered[i - 1];
        if (binding.order != source.order + 1 || source.ffn_input->ne[0] != binding.input->ne[0] ||
                !gpu_f32(source.ffn_input) || !gpu_f32(source.ffn_scale) || impl_->early_bindings.count(source.ffn_trigger) != 0) {
            continue;
        }
        auto & from_device = *impl_->resources[source.group]->device;
        auto & to_device = *impl_->resources[binding.group]->device;
        auto & from = *from_device.materialization;
        auto & to = *to_device.materialization;
        if (from.control == nullptr || to.control == nullptr) {
            continue;
        }
        if (std::any_of(to.sources.begin(), to.sources.begin() + impl_->resources[binding.group]->snapshot.banks.size(),
                [](const auto * source) { return source == nullptr; })) {
            continue;
        }
        const moe_prepack_request next{to_device.serial, binding.top_k, to_device.n_experts,
            to_device.words_per_miss * sizeof(uint4), to.sources};
        const auto callback = [&](const std::shared_ptr<impl::grouped_resource> & resource, const moe_prepack_request & next) {
            auto & materialization = *resource->device->materialization;
            if (materialization.prepack.load(std::memory_order_relaxed) == nullptr) {
                impl_->prepack_resources.emplace_back(resource);
                materialization.publish_prepack(impl_->host_prepack.get());
            }
            for (const auto & saved : materialization.prediction_copies) {
                if (saved->next.resource == next.resource) {
                    return saved.get();
                }
            }
            auto copy = std::make_unique<moe_grouped_materialization::copy>(moe_grouped_materialization::copy{&materialization, 0, next, true});
            auto * result = copy.get();
            materialization.prediction_copies.push_back(std::move(copy));
            return result;
        };
        impl_->prepack_callbacks[source.group] = callback(impl_->resources[source.group], next);
        impl_->prepack_callbacks.emplace(binding.group, callback(impl_->resources[binding.group], {}));
        binding.input = source.ffn_input;
        binding.attention_scale = source.ffn_scale;
        binding.input_multiplier = source.ffn_multiplier;
        binding.source_group = source.group;
        impl_->early_bindings.emplace(source.ffn_trigger, binding);
    }
    fprintf(stderr, "moe-prepack: pairs=%zu groups=%u tiles=2 pinned_bytes=%zu lookahead=1 rows=1\n",
        impl_->early_bindings.size(), execution->n_groups_, impl_->prepack_storage->pinned_size);
} catch (const std::exception & error) {
    impl_->early_bindings.clear();
    impl_->prepack_callbacks.clear();
    fprintf(stderr, "moe-prepack: disabled reason=%s\n", error.what());
}

void ggml_cuda_moe_grouped_context::launch_early_router(
        const ggml_tensor * node, ggml_cuda_moe_graph_execution * execution, cudaStream_t stream) {
    if (impl_->early.empty() || execution == nullptr || execution->outcome() != GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED) {
        return;
    }
    const auto found = impl_->early_bindings.find(node);
    if (found == impl_->early_bindings.end()) {
        return;
    }
    const auto & binding = found->second;
    auto & early = *impl_->early[0];
    const auto & resource = *impl_->resources[binding.group];
    const auto & device = *resource.device;
    GGML_ASSERT(stream == early.stream);
    CUDA_CHECK(cudaMemcpyAsync(early.input, binding.input->data, ggml_nbytes(binding.input), cudaMemcpyDeviceToDevice, stream));
    if (binding.hc_streams != 0) {
        const int wide = binding.input->ne[0];
        const int width = binding.weights->ne[0];
        moe_early_hc_translate<<<(wide + 255) / 256, 256, 0, stream>>>(
            early.input, (const float *) binding.attention_scale->data, (const float *) binding.ffn_scale->data, wide);
        moe_early_hc_project<128>(binding.hc_down, early.input, early.hc_low, binding.hc_streams, 1, stream);
        moe_early_hc_project<32>(binding.hc_up, early.hc_low, early.hc_gate, binding.hc_streams, 2, stream);
        moe_early_hc_collapse<<<(width + 255) / 256, 256, 0, stream>>>(early.input, early.hc_gate, early.hc_mixed, width, binding.hc_streams);
        moe_early_hc_project<128>(binding.weights, early.hc_mixed, early.scores, 1, 0, stream);
    } else {
        moe_early_router_scores<<<device.n_experts, 128, 0, stream>>>(
            early.input, (const float *) binding.attention_scale->data, (const float *) binding.ffn_scale->data,
            (const float *) binding.weights->data, binding.input->ne[0], early.scores,
            binding.ffn_multiplier / binding.input_multiplier, binding.router_bias != nullptr ? (const float *) binding.router_bias->data : nullptr);
    }
    moe_early_router_select_launch(early.scores, device.n_experts, binding.top_k,
        device.slot_for_expert, device.plan, resource.snapshot.n_slots, early.predicted, early.positions,
        early.counters, device.words_per_miss * sizeof(uint4), stream);
    CUDA_CHECK(cudaGetLastError());
    auto & materialization = *impl_->resources[binding.source_group]->device->materialization;
    CUDA_CHECK(cudaMemcpyAsync(materialization.control->predicted, early.predicted, binding.top_k * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
}

ggml_cuda_moe_grouped_decode_result ggml_cuda_moe_grouped_context::prepare_decode(
        const ggml_cuda_moe_complete_group_key & key,
        cudaStream_t compute_stream,
        ggml_cuda_moe_grouped_decode_acquisition * decode,
        const ggml_cuda_moe_group_call_lease * authority) {
    if (decode == nullptr) {
        return GGML_CUDA_MOE_GROUPED_DECODE_FALLBACK;
    }
    *decode = ggml_cuda_moe_grouped_decode_acquisition{};
    if (compute_stream == nullptr || impl_->device < 0 || key.n_banks == 0 ||
            key.n_banks > GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS) {
        return GGML_CUDA_MOE_GROUPED_DECODE_FALLBACK;
    }
    if (authority != nullptr) {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (authority->owner_ != this || authority->candidate_generation_ != impl_->state.generation ||
                authority->group_index_ != key.candidate.group_index || authority->authority_ != GGML_CUDA_MOE_GROUP_AUTHORITY_GROUPED ||
                authority->group_index_ >= impl_->table.groups.size()) {
            return GGML_CUDA_MOE_GROUPED_DECODE_FALLBACK;
        }
        const auto & current = impl_->group_authorities[authority->group_index_];
        if (authority->authority_epoch_ != current.epoch || current.authority != GGML_CUDA_MOE_GROUP_AUTHORITY_GROUPED ||
                current.active_calls == 0) {
            return GGML_CUDA_MOE_GROUPED_DECODE_FALLBACK;
        }
    }
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    return GGML_CUDA_MOE_GROUPED_DECODE_FALLBACK;
#else
    const ggml_tensor * ids = key.ids.tensor;
    const uint32_t top_k = ids != nullptr && ids->ne[0] > 0 && ids->ne[0] <= UINT32_MAX ?
        static_cast<uint32_t>(ids->ne[0]) : 0;
    const uint32_t n_rows = ids != nullptr && ids->ne[1] > 0 && ids->ne[1] <= UINT32_MAX ?
        static_cast<uint32_t>(ids->ne[1]) : 0;
    const uint32_t n_routes = top_k != 0 && n_rows != 0 && top_k <= UINT32_MAX / n_rows ? top_k * n_rows : 0;
    const uint32_t row_stride = ids != nullptr && ids->nb[1] % sizeof(int32_t) == 0 &&
        ids->nb[1] / sizeof(int32_t) <= UINT32_MAX ? static_cast<uint32_t>(ids->nb[1] / sizeof(int32_t)) : 0;
    if (ids == nullptr || ids->buffer == nullptr || ids->buffer->buft != ggml_backend_cuda_buffer_type(impl_->device) ||
            ids->data != key.ids.data || ids->buffer != key.ids.buffer || ids->type != key.ids.type ||
            ids->type != GGML_TYPE_I32 || (authority != nullptr && key.execution_semantic_key == 0) ||
            top_k == 0 || n_rows == 0 || n_routes == 0 || n_routes > INT32_MAX ||
            ids->ne[2] != 1 || ids->ne[3] != 1 ||
            ids->nb[0] != sizeof(int32_t) || row_stride < top_k ||
            ids->nb[2] < static_cast<size_t>(n_rows) * ids->nb[1] || ids->nb[3] < ids->nb[2]) {
        return GGML_CUDA_MOE_GROUPED_DECODE_FALLBACK;
    }
    for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
        if (ids->ne[dim] != key.ids.ne[dim] || ids->nb[dim] != key.ids.nb[dim]) {
            return GGML_CUDA_MOE_GROUPED_DECODE_FALLBACK;
        }
    }

    impl::maintenance_lease maintenance;
    for (uint32_t attempt = 0; attempt < 2; ++attempt) {
        ggml_cuda_moe_grouped_transaction transaction = {};
        impl::grouped_resource * resource = nullptr;
        bool resource_missing = false;
        {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            resource = impl_->begin_decode(key, compute_stream, transaction, resource_missing);
        }
        if (resource == nullptr && resource_missing) {
            ggml_cuda_moe_grouped_acquisition resource_acquisition;
            if (!acquire_group_resources(key.candidate, &resource_acquisition)) {
                return GGML_CUDA_MOE_GROUPED_DECODE_FALLBACK;
            }
            std::lock_guard<std::mutex> lock(impl_->mutex);
            resource = impl_->begin_decode(key, compute_stream, transaction, resource_missing);
        }
        if (resource == nullptr || n_routes > resource->snapshot.n_slots) {
            if (resource != nullptr) {
                (void) end_group_transaction(transaction);
            }
            return GGML_CUDA_MOE_GROUPED_DECODE_FALLBACK;
        }

        if (resource->device == nullptr) {
            auto prospective = impl_->make_device_resource(resource->snapshot, compute_stream, false, top_k);
            if (prospective == nullptr) {
                (void) end_group_transaction(transaction);
                return GGML_CUDA_MOE_GROUPED_DECODE_FALLBACK;
            }
            bool installed = false;
            {
                std::lock_guard<std::mutex> lock(impl_->mutex);
                auto * current = impl_->find_resource(transaction);
                if (current == resource && !impl_->refreshing[transaction.acquisition.candidate.group_index] &&
                        impl_->resource_matches_table(*resource) && impl_->next_device_resource_serial != UINT64_MAX) {
                    if (resource->device == nullptr) {
                        prospective->set_serial(++impl_->next_device_resource_serial);
                        resource->device = std::move(prospective);
                    }
                    installed = true;
                }
            }
            if (!installed) {
                (void) end_group_transaction(transaction);
                return GGML_CUDA_MOE_GROUPED_DECODE_FALLBACK;
            }
        }

        uint64_t clock_begin = 0;
        uint64_t clock_end = 0;
        const auto reservation = impl_->reserve_clock(transaction, n_routes, clock_begin, clock_end);
        if (reservation == impl::CLOCK_RESERVATION_REFRESH) {
            if (attempt != 0) {
                (void) end_group_transaction(transaction);
                impl_->clear_refresh(transaction.acquisition.candidate);
                return GGML_CUDA_MOE_GROUPED_DECODE_FALLBACK;
            }
            if (!impl_->begin_refresh(transaction, maintenance)) {
                if (maintenance.owner == nullptr) {
                    (void) end_group_transaction(transaction);
                }
                return GGML_CUDA_MOE_GROUPED_DECODE_FALLBACK;
            }
            if (!impl_->refresh_group_resource(transaction.acquisition, compute_stream, top_k)) {
                return GGML_CUDA_MOE_GROUPED_DECODE_FALLBACK;
            }
            continue;
        }
        if (reservation != impl::CLOCK_RESERVATION_READY && reservation != impl::CLOCK_RESERVATION_DEVICE) {
            (void) end_group_transaction(transaction);
            return GGML_CUDA_MOE_GROUPED_DECODE_FALLBACK;
        }
        auto & device = *resource->device;
        if (device.has_completion && device.completion_stream != compute_stream) {
            if (!moe_grouped_cuda_success(cudaStreamWaitEvent(compute_stream, device.completion, 0))) {
                (void) end_group_transaction(transaction);
                return GGML_CUDA_MOE_GROUPED_DECODE_FALLBACK;
            }
        }

        const uint32_t plan_threads = moe_grouped_plan_threads(resource->snapshot.n_slots);
        moe_grouped_plan_decode<<<1, plan_threads, 0, compute_stream>>>(
            static_cast<const int32_t *>(ids->data), n_routes, top_k, row_stride,
            device.n_experts, resource->snapshot.n_slots, resource->snapshot.n_slots,
            device.slot_for_expert, device.expert_for_slot, device.last_used,
            device.expert_frequency, device.expert_frequency_epoch, device.device_step, impl_->frequency_aware, clock_begin, clock_end,
            reservation == impl::CLOCK_RESERVATION_DEVICE ? device.device_clock : nullptr, device.plan);
        CUDA_CHECK(cudaGetLastError());
        auto * debug = impl_->grouped_debug.load(std::memory_order_acquire);
        uint64_t * transfer_counters = debug != nullptr ? debug->device_transfers.load(std::memory_order_acquire) : nullptr;
        const uint32_t transfer_blocks = moe_grouped_transfer_blocks(
            impl_->device, device.words_per_miss, device.auxiliary_values_per_miss, n_routes);
        if (device.materialization != nullptr) {
            auto & materialization = *device.materialization;
            const uint32_t tile_capacity = materialization.storage.tiles;
            const uint32_t tile_blocks = moe_grouped_transfer_blocks(
                impl_->device, device.words_per_miss, device.auxiliary_values_per_miss, std::min(tile_capacity, n_routes));
            CUDA_CHECK(cudaMemcpyAsync(materialization.storage.data, device.plan, device.plan_bytes, cudaMemcpyDeviceToHost, compute_stream));
            for (uint32_t miss = 0; miss < n_routes; miss += tile_capacity) {
                // The gather consumes the tile before the next callback reuses it.
                const auto predicted = impl_->prepack_callbacks.find(key.candidate.group_index);
                auto * callback = miss == 0 && n_routes == top_k && n_routes <= tile_capacity && predicted != impl_->prepack_callbacks.end() ?
                    predicted->second : &materialization.copies[miss];
                CUDA_CHECK(cudaLaunchHostFunc(compute_stream, moe_grouped_materialize, callback));
                if (transfer_counters != nullptr) {
                    moe_grouped_gather_decode<true><<<tile_blocks, MOE_GROUPED_TRANSFER_THREADS, 0, compute_stream>>>(
                        device.device_banks, resource->snapshot.banks.size(), device.words_per_miss,
                        device.device_auxiliaries, resource->snapshot.n_slot_auxiliaries,
                        device.auxiliary_values_per_miss, resource->snapshot.n_slots, device.plan, transfer_counters,
                        materialization.device_control, miss, tile_capacity);
                } else {
                    moe_grouped_gather_decode<false><<<tile_blocks, MOE_GROUPED_TRANSFER_THREADS, 0, compute_stream>>>(
                        device.device_banks, resource->snapshot.banks.size(), device.words_per_miss,
                        device.device_auxiliaries, resource->snapshot.n_slot_auxiliaries,
                        device.auxiliary_values_per_miss, resource->snapshot.n_slots, device.plan, nullptr,
                        materialization.device_control, miss, tile_capacity);
                }
                CUDA_CHECK(cudaGetLastError());
            }
        } else if (transfer_counters != nullptr) {
            moe_grouped_gather_decode<true><<<transfer_blocks, MOE_GROUPED_TRANSFER_THREADS, 0, compute_stream>>>(
                device.device_banks, resource->snapshot.banks.size(), device.words_per_miss,
                device.device_auxiliaries, resource->snapshot.n_slot_auxiliaries,
                device.auxiliary_values_per_miss, resource->snapshot.n_slots, device.plan, transfer_counters,
                nullptr);
        } else {
            moe_grouped_gather_decode<false><<<transfer_blocks, MOE_GROUPED_TRANSFER_THREADS, 0, compute_stream>>>(
                device.device_banks, resource->snapshot.banks.size(), device.words_per_miss,
                device.device_auxiliaries, resource->snapshot.n_slot_auxiliaries,
                device.auxiliary_values_per_miss, resource->snapshot.n_slots, device.plan, nullptr,
                nullptr);
        }
        CUDA_CHECK(cudaGetLastError());
        decode->transaction = transaction;
        decode->remapped_ids = moe_grouped_plan_array_ptr(
            device.plan, resource->snapshot.n_slots, MOE_GROUPED_PLAN_REMAPPED_IDS);
        decode->layout = resource->snapshot.layout;
        decode->n_banks = resource->snapshot.banks.size();
        decode->n_slots = resource->snapshot.n_slots;
        decode->n_auxiliary_shadows = resource->snapshot.n_slot_auxiliaries;
        for (uint32_t bank = 0; bank < resource->snapshot.banks.size(); ++bank) {
            decode->banks[bank].tensor = resource->snapshot.banks[bank].tensor;
            decode->banks[bank].data = device.bank_data[bank];
            decode->banks[bank].bank_index = bank;
            decode->banks[bank].role = resource->snapshot.banks[bank].role;
            decode->banks[bank].type = resource->snapshot.banks[bank].type;
        }
        for (uint32_t auxiliary = 0; auxiliary < resource->snapshot.n_slot_auxiliaries; ++auxiliary) {
            decode->auxiliary_tensors[auxiliary] = resource->snapshot.slot_auxiliaries[auxiliary].tensor;
            decode->auxiliary_data[auxiliary] = device.auxiliary_data[auxiliary];
            decode->auxiliary_roles[auxiliary] = resource->snapshot.slot_auxiliaries[auxiliary].role;
        }
        return GGML_CUDA_MOE_GROUPED_DECODE_READY;
    }
    return GGML_CUDA_MOE_GROUPED_DECODE_FALLBACK;
#endif
}

bool ggml_cuda_moe_grouped_context::finish_decode(
        const ggml_cuda_moe_grouped_decode_acquisition & decode,
        cudaStream_t compute_stream) {
    if (compute_stream == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto * resource = impl_->find_resource(decode.transaction);
    if (resource == nullptr || resource->device == nullptr || resource->active_decode_stream != compute_stream) {
        return false;
    }
    CUDA_CHECK(cudaEventRecord(resource->device->completion, compute_stream));
    resource->device->has_completion = true;
    resource->device->completion_stream = compute_stream;
    resource->active_transaction_token = 0;
    resource->active_decode_stream = nullptr;
    impl_->resource_cv.notify_all();
    return true;
}

void ggml_cuda_moe_grouped_context::compile_graph_plan(
        const ggml_cgraph * cgraph,
        uint64_t graph_uid,
        ggml_cuda_moe_graph_plan * plan,
        ggml_cuda_moe_graph_execution * execution,
        uint64_t coverage_epoch,
        const void * coverage_nodes,
        uint32_t coverage_mmid_count,
        uint64_t coverage_mmid_fingerprint) const {
    if (plan != nullptr) {
        plan->reset();
    }
    if (execution != nullptr) {
        execution->reset();
    }
    if (cgraph == nullptr || plan == nullptr || execution == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->draining || impl_->replacement_pending) {
        return;
    }
    plan->owner_ = impl_.get();
    plan->registry_generation_ = impl_->state.generation;
    plan->graph_uid_ = graph_uid;
    const bool execution_certificate_valid = moe_candidate_execution_certificate(
        cgraph, &plan->execution_certificate_, &plan->execution_semantic_key_);
    const bool required_grouped_execution = execution_certificate_valid &&
        plan->execution_certificate_.flags == GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED;
    plan->graph_node_count_ = ggml_graph_n_nodes(const_cast<ggml_cgraph *>(cgraph));
    plan->graph_key_ = plan->graph_node_count_ > 0 ? ggml_graph_node(const_cast<ggml_cgraph *>(cgraph), 0) : nullptr;
    const bool coverage_valid = coverage_epoch != 0 && coverage_nodes == cgraph->nodes && coverage_mmid_fingerprint != 0;
    plan->coverage_nodes_ = coverage_valid ? coverage_nodes : nullptr;
    plan->coverage_epoch_ = coverage_valid ? coverage_epoch : 0;
    plan->coverage_mmid_count_ = coverage_valid ? coverage_mmid_count : 0;
    plan->coverage_mmid_fingerprint_ = coverage_valid ? coverage_mmid_fingerprint : 0;
    plan->initialized_ = true;
    auto & diagnostics = plan->coverage_diagnostics_;
    diagnostics.manifest_version = impl_->table.manifest_version;
    std::array<const ggml_tensor *, GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS> prefill_authorities = {};
    std::array<uint32_t, GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS> prefill_authority_indices;
    std::array<uint32_t, ggml_cuda_moe_graph_plan::MAX_MMID_INVENTORY> mmid_indices;
    prefill_authority_indices.fill(UINT32_MAX);
    uint32_t n_mmid_indices = 0;
    uint64_t compiled_mmid_fingerprint = UINT64_C(1469598103934665603);
    bool inventory_complete = true;
    bool cached_prefill = false;
    bool cached_decode = false;
    for (int node_index = 0; node_index < plan->graph_node_count_; ++node_index) {
        const ggml_tensor * node = ggml_graph_node(const_cast<ggml_cgraph *>(cgraph), node_index);
        const ggml_tensor * source = node != nullptr && node->op == GGML_OP_MUL_MAT_ID ? node->src[0] : nullptr;
        if (source == nullptr || source->buffer == nullptr ||
                ggml_is_empty(node) ||
                !ggml_backend_buft_is_cuda_moe_cached(ggml_backend_buffer_get_type(source->buffer))) {
            continue;
        }

        ++diagnostics.cached_mmid;
        moe_candidate_graph_mmid_fingerprint_add(compiled_mmid_fingerprint, node_index, node, source);
        if (n_mmid_indices == mmid_indices.size()) {
            inventory_complete = false;
        } else {
            mmid_indices[n_mmid_indices++] = node_index;
        }
        const bool prefill = moe_candidate_execution_phase_for(cgraph, node) == MOE_CANDIDATE_EXECUTION_PHASE_PREFILL &&
            !required_grouped_execution;
        cached_prefill = cached_prefill || prefill;
        cached_decode = cached_decode || !prefill;
        uint32_t reason = GGML_CUDA_MOE_GRAPH_COVERAGE_REVERSE_MAP_MISS;
        uint32_t group_index = UINT32_MAX;
        uint32_t bank_index = UINT32_MAX;
        const auto reverse = impl_->table.reverse_map.find(source);
        if (reverse != impl_->table.reverse_map.end()) {
            group_index = reverse->second.group_index;
            bank_index = reverse->second.bank_index;
            if (!reverse->second.descriptor_valid) {
                reason = reverse->second.rejection == GGML_CUDA_MOE_CANDIDATE_REJECT_NONE ?
                    GGML_CUDA_MOE_GRAPH_COVERAGE_INVALID_REVERSE_MAP : moe_candidate_coverage_reason(reverse->second);
            } else if (!moe_candidate_reverse_matches(impl_->table, reverse->second, source)) {
                reason = GGML_CUDA_MOE_GRAPH_COVERAGE_SOURCE_CHANGED;
            } else {
                reason = moe_candidate_coverage_reason(reverse->second);
            }
            if (reverse->second.active && reverse->second.group_index < impl_->table.groups.size() &&
                    prefill_authorities[reverse->second.group_index] == nullptr) {
                prefill_authorities[reverse->second.group_index] = node;
                prefill_authority_indices[reverse->second.group_index] = node_index;
            }
        }
        ++diagnostics.counts[reason];
        if (diagnostics.first_node_index[reason] == UINT32_MAX) {
            diagnostics.first_source[reason] = source;
            diagnostics.first_node_index[reason] = node_index;
            diagnostics.first_group_index[reason] = group_index;
            diagnostics.first_bank_index[reason] = bank_index;
            if (reverse != impl_->table.reverse_map.end()) {
                diagnostics.first_group_index[reason] = reverse->second.semantic_group_index;
                diagnostics.first_role[reason] = reverse->second.role;
                diagnostics.first_status[reason] = reverse->second.status;
                diagnostics.first_layout[reason] = reverse->second.layout;
                diagnostics.first_domain[reason] = reverse->second.domain;
                diagnostics.first_flags[reason] = reverse->second.flags;
                diagnostics.first_group_flags[reason] = reverse->second.group_flags;
                diagnostics.first_rejection[reason] = reverse->second.rejection;
            }
        }
    }
    compiled_mmid_fingerprint = moe_candidate_graph_mmid_fingerprint_finish(
        compiled_mmid_fingerprint, diagnostics.cached_mmid);
    try {
        plan->mmid_inventory_.reserve(n_mmid_indices);
    } catch (const std::bad_alloc &) {
        plan->reset();
        execution->reset();
        return;
    }
    plan->inventory_complete_ = inventory_complete;
    for (uint32_t inventory_index = 0; inventory_index < n_mmid_indices; ++inventory_index) {
        const uint32_t node_index = mmid_indices[inventory_index];
        const ggml_tensor * node = ggml_graph_node(const_cast<ggml_cgraph *>(cgraph), node_index);
        ggml_cuda_moe_graph_plan::mmid_witness witness;
        witness.output = moe_candidate_ids_signature(node);
        witness.source = moe_candidate_ids_signature(node->src[0]);
        witness.activation = node->src[1] != nullptr ? moe_candidate_ids_signature(node->src[1]) : ggml_cuda_moe_ids_signature{};
        witness.ids = node->src[2] != nullptr ? moe_candidate_ids_signature(node->src[2]) : ggml_cuda_moe_ids_signature{};
        witness.node_index = node_index;
        witness.flags = node->flags;
        plan->mmid_inventory_.push_back(witness);
    }
    const bool certified_inventory = coverage_valid && plan->coverage_mmid_count_ == diagnostics.cached_mmid &&
        plan->coverage_mmid_fingerprint_ == compiled_mmid_fingerprint &&
        plan->mmid_inventory_.size() == diagnostics.cached_mmid && plan->inventory_complete_;
    const bool pure_prefill = cached_prefill && !cached_decode;
    if (pure_prefill) {
        plan->outcome_ = GGML_CUDA_MOE_GRAPH_OUTCOME_PREFILL_LEGACY;
        plan->unknown_reusable_ = certified_inventory;
        if (impl_->state.accepted && impl_->state.n_slots != 0 && !impl_->table.groups.empty()) {
            uint32_t n_prefill_groups = 0;
            for (uint32_t group_index = 0; group_index < impl_->table.groups.size(); ++group_index) {
                n_prefill_groups += prefill_authorities[group_index] != nullptr;
            }
            try {
                plan->groups_.reserve(n_prefill_groups);
            } catch (const std::bad_alloc &) {
                plan->reset();
                execution->reset();
                return;
            }
            for (uint32_t group_index = 0; group_index < impl_->table.groups.size(); ++group_index) {
                if (prefill_authorities[group_index] == nullptr) {
                    continue;
                }
                const auto & group = impl_->table.groups[group_index];
                const uint32_t record_index = plan->groups_.size();
                plan->groups_.emplace_back();
                auto & record = plan->groups_.back();
                record.candidate.generation = impl_->state.generation;
                record.candidate.group_index = group_index;
                record.layout = group.layout;
                record.n_banks = moe_candidate_base_slot_bank_count(group);
                record.prefill = 1;
                record.authority_node = prefill_authorities[group_index];
                record.authority_node_index = prefill_authority_indices[group_index];
                record.reason = ggml_cuda_moe_graph_plan::GROUP_REASON_PREFILL;
                auto & dispatch = execution->groups_[record_index];
                dispatch.key.candidate = record.candidate;
                dispatch.key.layout = record.layout;
                dispatch.key.n_banks = record.n_banks;
                dispatch.first_reader = record.authority_node;
            }
            plan->n_groups_ = plan->groups_.size();
            if (certified_inventory && impl_->table.prefill_resident_auxiliary) {
                try {
                    bool resident_bindings = true;
                    for (uint32_t inventory_index = 0; resident_bindings && inventory_index < n_mmid_indices; ++inventory_index) {
                        const ggml_tensor * node = ggml_graph_node(
                            const_cast<ggml_cgraph *>(cgraph), mmid_indices[inventory_index]);
                        const auto * reverse = moe_candidate_active_reverse(impl_->table, node->src[0]);
                        if (reverse == nullptr || reverse->group_index >= impl_->table.groups.size() ||
                                !impl_->table.groups[reverse->group_index].prefill_resident_auxiliary) {
                            continue;
                        }
                        uint32_t record_index = 0;
                        while (record_index < plan->n_groups_ &&
                                plan->groups_[record_index].candidate.group_index != reverse->group_index) {
                            ++record_index;
                        }
                        const auto & bank = impl_->table.groups[reverse->group_index].banks[reverse->bank_index];
                        resident_bindings = record_index < plan->n_groups_ &&
                            plan->insert(node, record_index, bank.info.role, reverse->bank_index, bank.slot_index);
                    }
                    if (!resident_bindings) {
                        throw std::bad_alloc();
                    }
                    plan->prefill_add_id_witnesses_.reserve(plan->n_groups_ * 3);
                    for (uint32_t node_index = 0; node_index < static_cast<uint32_t>(plan->graph_node_count_); ++node_index) {
                        const ggml_tensor * node = ggml_graph_node(const_cast<ggml_cgraph *>(cgraph), node_index);
                        if (node == nullptr || node->src[0] == nullptr) {
                            continue;
                        }
                        uint32_t producer_index = 0;
                        while (producer_index < node_index &&
                                ggml_graph_node(const_cast<ggml_cgraph *>(cgraph), producer_index) != node->src[0]) {
                            ++producer_index;
                        }
                        if (producer_index == node_index) {
                            continue;
                        }
                        for (uint32_t record_index = 0; record_index < plan->n_groups_; ++record_index) {
                            const auto & group = impl_->table.groups[plan->groups_[record_index].candidate.group_index];
                            uint32_t bank_index = UINT32_MAX;
                            uint32_t reader_role = GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID;
                            if (!group.prefill_resident_auxiliary ||
                                    !moe_candidate_prefill_add_id_witness(node, group, &bank_index, &reader_role)) {
                                continue;
                            }
                            ggml_cuda_moe_graph_plan::prefill_add_id_witness witness = {};
                            witness.consumer.node = node;
                            memcpy(witness.consumer.src, node->src, sizeof(witness.consumer.src));
                            witness.consumer.node_index = node_index;
                            witness.consumer.src_index = 1;
                            witness.consumer.op = node->op;
                            witness.group_record = record_index;
                            witness.bank_index = bank_index;
                            witness.reader_role = reader_role;
                            plan->prefill_add_id_witnesses_.push_back(witness);
                            break;
                        }
                    }
                } catch (const std::bad_alloc &) {
                    plan->prefill_add_id_witnesses_.clear();
                }
            }
        }
        execution->plan_ = plan;
        execution->owner_ = const_cast<ggml_cuda_moe_grouped_context *>(this);
        execution->n_groups_ = plan->n_groups_;
        return;
    }
    const bool empty_manifest = impl_->table.manifest_version == GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V1_VERSION ?
        impl_->table.submitted_groups == 0 :
        impl_->table.submitted_flags == GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_FLAG_NONE &&
            impl_->table.submitted_groups == 0 && impl_->table.submitted_tensors == 0;
    const bool explicitly_disabled = impl_->state.accepted &&
        (impl_->state.n_slots == 0 || empty_manifest);
    if (!impl_->state.accepted || impl_->state.n_slots == 0 || impl_->table.groups.empty()) {
        plan->outcome_ = cached_decode && !explicitly_disabled ?
            GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR : GGML_CUDA_MOE_GRAPH_OUTCOME_PREFILL_LEGACY;
        plan->unknown_reusable_ = explicitly_disabled && certified_inventory;
        execution->plan_ = plan;
        execution->owner_ = const_cast<ggml_cuda_moe_grouped_context *>(this);
        return;
    }

    std::vector<ggml_cuda_moe_graph_plan::group_observation> observations;
    try {
        observations.resize(impl_->table.groups.size());
    } catch (const std::bad_alloc &) {
        plan->reset();
        execution->reset();
        return;
    }
    for (uint32_t group_index = 0; group_index < impl_->table.groups.size(); ++group_index) {
        const auto & group = impl_->table.groups[group_index];
        auto & observation = observations[group_index];
        observation.required_roles = moe_candidate_required_base_roles(group.layout);
        observation.n_banks = moe_candidate_required_base_bank_count(group.layout);
        if (observation.n_banks == 0) {
            continue;
        }
        uint32_t n_slot_banks = 0;
        observation.descriptor_supported = moe_candidate_structural_group(group, &n_slot_banks) && n_slot_banks == observation.n_banks;
        for (uint32_t bank_index = 0; bank_index < group.banks.size(); ++bank_index) {
            const auto & bank = group.banks[bank_index];
            const uint32_t role = bank.info.role;
            if (bank.info.movement == GGML_CUDA_MOE_CANDIDATE_MOVEMENT_SLOT_BOUND &&
                    role >= GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT &&
                    role <= GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT) {
                observation.bank_indices[role - GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT] = bank_index;
            }
        }
    }

    const int n_nodes = plan->graph_node_count_;
    uint32_t observed_cached_readers = 0;
    for (int node_index = 0; node_index < n_nodes; ++node_index) {
        const ggml_tensor * node = ggml_graph_node(const_cast<ggml_cgraph *>(cgraph), node_index);
        if (node == nullptr) {
            continue;
        }

        for (uint32_t src_index = 0; src_index < GGML_MAX_SRC; ++src_index) {
            const ggml_tensor * reader = node->src[src_index];
            if (reader == nullptr || reader->op != GGML_OP_MUL_MAT_ID || reader->src[0] == nullptr) {
                continue;
            }
            const auto * reverse = moe_candidate_active_reverse(impl_->table, reader->src[0]);
            if (reverse == nullptr) {
                continue;
            }
            auto & observation = observations[reverse->group_index];
            auto found = observation.n_readers;
            for (uint32_t i = 0; i < observation.n_readers; ++i) {
                if (observation.readers[i].output.tensor == reader) {
                    found = i;
                    break;
                }
            }
            if (found == observation.n_readers) {
                observation.unproven = true;
                continue;
            }
            auto & witness = observation.readers[found];
            if (witness.n_consumers == ggml_cuda_moe_graph_plan::MAX_READER_CONSUMERS) {
                observation.unproven = true;
                continue;
            }
            auto & consumer = witness.consumers[witness.n_consumers++];
            consumer.node = node;
            memcpy(consumer.src, node->src, sizeof(consumer.src));
            consumer.node_index = node_index;
            consumer.src_index = src_index;
            consumer.op = node->op;
        }

        if (node->op != GGML_OP_MUL_MAT_ID || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || node->src[0] == nullptr) {
            continue;
        }
        const auto * reverse = moe_candidate_active_reverse(impl_->table, node->src[0]);
        if (reverse == nullptr || ggml_is_empty(node)) {
            continue;
        }
        const auto & entry = *reverse;
        const auto & group = impl_->table.groups[entry.group_index];
        const auto & bank = group.banks[entry.bank_index];
        auto & observation = observations[entry.group_index];
        if (node->src[0]->buffer != nullptr &&
                ggml_backend_buft_is_cuda_moe_cached(ggml_backend_buffer_get_type(node->src[0]->buffer))) {
            ++observed_cached_readers;
        }
        if (!observation.observed) {
            observation.authority_node = node;
            observation.authority_node_index = node_index;
            observation.observed = true;
        }
        if (observation.n_readers == ggml_cuda_moe_graph_plan::MAX_GROUP_READERS) {
            observation.unproven = true;
            continue;
        }
        auto & reader = observation.readers[observation.n_readers++];
        reader.output = moe_candidate_ids_signature(node);
        reader.activation = node->src[1] != nullptr ? moe_candidate_ids_signature(node->src[1]) : ggml_cuda_moe_ids_signature{};
        reader.ids = node->src[2] != nullptr ? moe_candidate_ids_signature(node->src[2]) : ggml_cuda_moe_ids_signature{};
        reader.node_index = node_index;
        reader.role = bank.info.role;
        reader.bank_index = entry.bank_index;
        if (entry.bank_index < GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS) {
            observation.bank_readers[entry.bank_index]++;
        } else {
            observation.unproven = true;
        }
        const ggml_tensor * ids = node->src[2];
        const auto geometry = moe_candidate_execution_geometry_for(cgraph, node, bank.ne[2], impl_->state.n_slots);
        const bool prefill = geometry.phase == MOE_CANDIDATE_EXECUTION_PHASE_PREFILL && !geometry.required_grouped;
        observation.prefill = observation.prefill || prefill;
        observation.decode = observation.decode || !prefill;
        if (observation.n_readers == 1) {
            observation.execution_semantic_key = geometry.semantic_key;
            observation.top_k = geometry.top_k;
            observation.n_rows = geometry.n_rows;
            observation.n_routes = geometry.n_routes;
            observation.row_stride = geometry.row_stride;
            observation.strategy = geometry.strategy;
        } else if (observation.execution_semantic_key != geometry.semantic_key ||
                observation.top_k != geometry.top_k || observation.n_rows != geometry.n_rows ||
                observation.n_routes != geometry.n_routes || observation.row_stride != geometry.row_stride ||
                observation.strategy != geometry.strategy) {
            observation.execution_ineligible = true;
        }
        if (!moe_candidate_record_matches(bank, node->src[0])) {
            observation.source_invalid = true;
        }
        const uint32_t role = bank.info.role;
        if (role < GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT || role > GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT) {
            observation.source_invalid = true;
            continue;
        }
        const uint32_t role_bit = 1u << role;
        const uint32_t role_slot = role - GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT;
        if ((observation.seen_roles & role_bit) != 0) {
            observation.duplicate_role = true;
        }
        observation.seen_roles |= role_bit;
        if (observation.nodes[role_slot] == nullptr) {
            observation.nodes[role_slot] = node;
            observation.node_indices[role_slot] = node_index;
            observation.bank_indices[role_slot] = entry.bank_index;
        }
        if (!geometry.tensor_valid) {
            observation.geometry_invalid = true;
            continue;
        }
        if (!geometry.grouped_eligible) {
            observation.execution_ineligible = true;
            continue;
        }
        const uint32_t input_type = node->src[1] != nullptr ? node->src[1]->type : GGML_TYPE_COUNT;
        const int64_t n_tokens = node->src[1] != nullptr ? node->src[1]->ne[2] : 0;
        const auto capability = moe_candidate_capability(
            bank.info.tensor, bank.info.source_data, bank.info.byte_extent, bank.info.expert_stride, bank.ne, bank.nb,
            bank.info.role, bank.info.type, input_type, node->type, n_tokens,
            geometry.top_k, geometry.n_rows, geometry.n_routes, geometry.row_stride, impl_->state.n_slots,
            geometry.row_semantics, geometry.strategy, impl_->device);
        if (bank.slot_index >= observation.n_banks) {
            observation.unproven = true;
        } else if (observation.capabilities[bank.slot_index].tensor == nullptr) {
            observation.capabilities[bank.slot_index] = capability;
        } else if (!moe_candidate_capability_matches(
                observation.capabilities[bank.slot_index], bank, node, geometry.top_k, geometry.n_rows,
                geometry.n_routes, geometry.row_stride, impl_->state.n_slots, impl_->device) ||
                observation.capabilities[bank.slot_index].consumer != capability.consumer ||
                observation.capabilities[bank.slot_index].reason != capability.reason) {
            observation.unproven = true;
        }
        if (capability.source_flags != bank.info.source_flags || !moe_candidate_capability_invariant_valid(capability)) {
            observation.capability_invalid = true;
        } else if (moe_candidate_capability_equivalence_unavailable(capability)) {
            observation.consumer_incompatible = true;
        }
        const auto ids_signature = moe_candidate_ids_signature(ids);
        if (observation.has_ids && !moe_candidate_ids_equal(observation.ids, ids_signature)) {
            observation.mixed_ids = true;
        }
        if (!observation.has_ids) {
            moe_candidate_route_proof route;
            if (!moe_candidate_validate_route(ids, bank.ne[2], route)) {
                observation.route_invalid = true;
            } else {
                observation.ids = route.ids;
                observation.route_root = route.root;
                observation.route_source = route.source;
                observation.route_indices = route.indices;
                if (!moe_candidate_discover_route(cgraph, node_index, route)) {
                    observation.route_invalid = true;
                    if (route.root_node_index >= static_cast<uint32_t>(n_nodes) || route.ids_node_index >= static_cast<uint32_t>(n_nodes) ||
                            ggml_graph_node(const_cast<ggml_cgraph *>(cgraph), route.root_node_index) != route.root.tensor ||
                            ggml_graph_node(const_cast<ggml_cgraph *>(cgraph), route.ids_node_index) != route.ids.tensor) {
                        observation.unproven = true;
                    }
                } else {
                    observation.has_ids = true;
                }
                observation.route_root_node_index = route.root_node_index;
                observation.route_ids_node_index = route.ids_node_index;
            }
        } else if (static_cast<uint32_t>(node_index) <= observation.route_ids_node_index) {
            observation.route_invalid = true;
        }
    }

    uint32_t n_observed = 0;
    for (const auto & observation : observations) {
        n_observed += observation.observed;
    }
    plan->unknown_reusable_ = certified_inventory &&
        diagnostics.counts[GGML_CUDA_MOE_GRAPH_COVERAGE_REGISTERED] == diagnostics.cached_mmid;
    try {
        plan->groups_.reserve(n_observed);
    } catch (const std::bad_alloc &) {
        plan->reset();
        execution->reset();
        return;
    }

    for (uint32_t group_index = 0; group_index < impl_->table.groups.size(); ++group_index) {
        const auto & group = impl_->table.groups[group_index];
        auto & observation = observations[group_index];
        if (!observation.observed) {
            continue;
        }
        if (plan->groups_.size() == GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS) {
            plan->reset();
            execution->reset();
            return;
        }
        const uint32_t record_index = plan->groups_.size();
        plan->groups_.emplace_back();
        plan->n_groups_ = plan->groups_.size();
        auto & record = plan->groups_.back();
        record.candidate.generation = impl_->state.generation;
        record.candidate.group_index = group_index;
        record.layout = group.layout;
        record.n_banks = observation.n_banks;
        record.prefill = observation.prefill && !observation.decode;
        record.execution_semantic_key = observation.execution_semantic_key;
        record.top_k = observation.top_k;
        record.n_rows = observation.n_rows;
        record.n_routes = observation.n_routes;
        record.row_stride = observation.row_stride;
        record.strategy = observation.strategy;
        record.authority_node = observation.authority_node;
        record.authority_node_index = observation.authority_node_index;
        record.n_readers = observation.n_readers;
        memcpy(record.capabilities, observation.capabilities, sizeof(record.capabilities));

        uint32_t n_slot_banks = 0;
        uint32_t n_original_direct_aux = 0;
        uint32_t n_original_direct_bias = 0;
        const bool original_direct_nvfp4 =
            moe_candidate_structural_group(group, &n_slot_banks, &n_original_direct_aux, &n_original_direct_bias) &&
            n_slot_banks == 3 && n_original_direct_aux == 3;
        const bool geometry_only = original_direct_nvfp4 && observation.geometry_invalid;
        const bool has_original_direct_witness = original_direct_nvfp4 || n_original_direct_bias != 0;
        const bool discover_grouped_original_direct_witness = has_original_direct_witness && !observation.geometry_invalid &&
            !observation.execution_ineligible &&
            observation.has_ids && !observation.route_invalid && !observation.mixed_ids &&
            !observation.duplicate_role && observation.seen_roles == observation.required_roles &&
            observation.n_readers == n_slot_banks;
        const bool discover_legacy_bias_witness = n_original_direct_bias != 0 && !observation.geometry_invalid &&
            observation.execution_ineligible && observation.decode && !observation.prefill;
        const bool discover_original_direct_witness =
            discover_grouped_original_direct_witness || discover_legacy_bias_witness;
        for (uint32_t reader_index = 0; reader_index < observation.n_readers; ++reader_index) {
            auto & reader = observation.readers[reader_index];
            if (geometry_only) {
                record.readers[reader_index] = reader;
                continue;
            }
            bool present = false;
            if (!moe_candidate_graph_use(cgraph, reader.output.tensor, present, reader.use_count) || !present ||
                    reader.use_count < static_cast<int32_t>(reader.n_consumers)) {
                observation.unproven = true;
            } else if (reader.use_count > static_cast<int32_t>(reader.n_consumers)) {
                observation.external_consumer = true;
            }
            for (uint32_t consumer_index = 0; !observation.geometry_invalid && !observation.prefill &&
                    (!observation.execution_ineligible || discover_legacy_bias_witness) &&
                    consumer_index < reader.n_consumers; ++consumer_index) {
                const auto & consumer = reader.consumers[consumer_index];
                if (moe_candidate_auxiliary_consumer(consumer.node, reader.output.tensor, reader.ids.tensor)) {
                    uint32_t bank_index = 0;
                    const bool discover = !has_original_direct_witness || discover_original_direct_witness;
                    if (discover && reader.auxiliary_kind == MOE_CANDIDATE_AUXILIARY_NONE &&
                            moe_candidate_original_direct_scale_consumer(
                            cgraph, consumer.node_index, consumer.node, reader.output.tensor, reader.ids.tensor, reader.role, group, &bank_index,
                            reader.auxiliary_nodes, reader.auxiliary_node_indices)) {
                        reader.auxiliary_bank_index = bank_index;
                        reader.auxiliary_consumer_index = consumer_index;
                        reader.n_auxiliary_nodes = 3;
                        reader.auxiliary_kind = MOE_CANDIDATE_AUXILIARY_SCALE;
                        observation.bank_readers[bank_index]++;
                    } else if (discover && reader.auxiliary_kind == MOE_CANDIDATE_AUXILIARY_NONE &&
                            moe_candidate_original_direct_bias_consumer(
                                consumer.node, reader.output.tensor, reader.ids.tensor, reader.role, group, &bank_index)) {
                        reader.auxiliary_bank_index = bank_index;
                        reader.auxiliary_consumer_index = consumer_index;
                        reader.auxiliary_kind = MOE_CANDIDATE_AUXILIARY_BIAS;
                        observation.bank_readers[bank_index]++;
                    } else if (discover) {
                        observation.auxiliary = true;
                    }
                }
            }
            const auto * bias = moe_candidate_find_role(group, moe_candidate_output_bias_role(reader.role));
            if (discover_original_direct_witness && bias != nullptr &&
                    (reader.n_consumers != 1 || reader.auxiliary_kind != MOE_CANDIDATE_AUXILIARY_BIAS)) {
                observation.auxiliary = true;
            }
            if (discover_original_direct_witness && original_direct_nvfp4 && bias == nullptr &&
                    (reader.n_consumers != 1 || reader.auxiliary_kind != MOE_CANDIDATE_AUXILIARY_SCALE)) {
                observation.auxiliary = true;
            }
            record.readers[reader_index] = reader;
        }
        if (discover_original_direct_witness && original_direct_nvfp4) {
            for (uint32_t auxiliary_index = 0; auxiliary_index < n_original_direct_aux; ++auxiliary_index) {
                const auto * auxiliary = moe_candidate_original_direct_aux_bank(group, auxiliary_index);
                if (auxiliary == nullptr || observation.bank_readers[auxiliary - group.banks.data()] != 1) {
                    observation.auxiliary = true;
                }
            }
        }
        if (discover_original_direct_witness) {
            for (uint32_t bank_index = 0; bank_index < group.banks.size(); ++bank_index) {
                if (moe_candidate_is_bias(group.banks[bank_index].info.role) && observation.bank_readers[bank_index] != 1) {
                    observation.auxiliary = true;
                }
            }
        }
        if (!geometry_only) {
            for (uint32_t bank_index = 0; bank_index < group.banks.size(); ++bank_index) {
                if (has_original_direct_witness && !discover_original_direct_witness &&
                        group.banks[bank_index].info.movement != GGML_CUDA_MOE_CANDIDATE_MOVEMENT_SLOT_BOUND) {
                    continue;
                }
                bool present = false;
                int32_t use_count = 0;
                if (!moe_candidate_graph_use(cgraph, group.banks[bank_index].info.tensor, present, use_count)) {
                    observation.unproven = true;
                }
                record.bank_uses[bank_index] = {group.banks[bank_index].info.tensor, use_count, present ? 1u : 0u};
                if (use_count != static_cast<int32_t>(observation.bank_readers[bank_index]) ||
                        (observation.bank_readers[bank_index] != 0 && !present)) {
                    observation.unproven = true;
                }
            }
        }
        if (observation.decode && !observation.prefill) {
            const uint32_t down_slot = GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT -
                GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT;
            for (uint32_t role_slot = 0; role_slot < 4 && observation.nodes[down_slot] != nullptr; ++role_slot) {
                if (observation.nodes[role_slot] != nullptr && observation.node_indices[role_slot] > observation.node_indices[down_slot]) {
                    observation.unproven = true;
                }
            }
        }

        if (!observation.descriptor_supported) {
            record.reason = ggml_cuda_moe_graph_plan::GROUP_REASON_DESCRIPTOR;
        } else if (observation.source_invalid) {
            record.reason = ggml_cuda_moe_graph_plan::GROUP_REASON_SOURCE;
        } else if (observation.geometry_invalid) {
            record.reason = ggml_cuda_moe_graph_plan::GROUP_REASON_GEOMETRY;
        } else if (observation.execution_ineligible && !record.prefill) {
            record.reason = ggml_cuda_moe_graph_plan::GROUP_REASON_EXECUTION;
        } else if (observation.capability_invalid) {
            record.reason = ggml_cuda_moe_graph_plan::GROUP_REASON_CAPABILITY;
        } else if (observation.consumer_incompatible) {
            record.reason = ggml_cuda_moe_graph_plan::GROUP_REASON_CONSUMER_EQUIVALENCE;
        } else if (observation.decode && (observation.route_invalid || !observation.has_ids)) {
            record.reason = ggml_cuda_moe_graph_plan::GROUP_REASON_ROUTE;
        } else if (observation.duplicate_role) {
            record.reason = ggml_cuda_moe_graph_plan::GROUP_REASON_DUPLICATE_ROLE;
        } else if (observation.mixed_ids) {
            record.reason = ggml_cuda_moe_graph_plan::GROUP_REASON_MIXED_IDS;
        } else if (observation.auxiliary) {
            record.reason = ggml_cuda_moe_graph_plan::GROUP_REASON_AUXILIARY;
        } else if (observation.external_consumer) {
            record.reason = ggml_cuda_moe_graph_plan::GROUP_REASON_EXTERNAL_CONSUMER;
        } else if (observation.seen_roles != observation.required_roles) {
            record.reason = ggml_cuda_moe_graph_plan::GROUP_REASON_MISSING_ROLE;
        } else if (observation.unproven) {
            record.reason = ggml_cuda_moe_graph_plan::GROUP_REASON_UNPROVEN;
        } else if (record.prefill) {
            record.reason = ggml_cuda_moe_graph_plan::GROUP_REASON_PREFILL;
        } else if (!impl_->group_source_mapped(group)) {
            record.reason = ggml_cuda_moe_graph_plan::GROUP_REASON_MATERIALIZATION;
        } else {
            record.reason = ggml_cuda_moe_graph_plan::GROUP_REASON_ELIGIBLE;
        }
        record.witness_reusable = !observation.unproven &&
            record.reason != ggml_cuda_moe_graph_plan::GROUP_REASON_UNPROVEN &&
            (record.reason != ggml_cuda_moe_graph_plan::GROUP_REASON_ROUTE || observation.ids.tensor != nullptr);
        auto & key = execution->groups_[record_index];
        key.key.candidate = record.candidate;
        key.key.layout = record.layout;
        key.key.n_banks = record.n_banks;
        key.key.execution_semantic_key = record.execution_semantic_key;
        key.strategy = record.strategy;
        key.capabilities = record.capabilities;
        key.first_reader = record.authority_node;
        if (observation.ids.tensor != nullptr) {
            record.ids = observation.ids;
            record.ids_root = observation.route_root;
            record.ids_source = observation.route_source;
            record.ids_indices = observation.route_indices;
            record.ids_root_node_index = observation.route_root_node_index;
            record.ids_node_index = observation.route_ids_node_index;
        }
        for (uint32_t role = GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT;
                role <= GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT; ++role) {
            if ((observation.required_roles & (1u << role)) == 0) {
                continue;
            }
            const uint32_t role_slot = role - GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT;
            record.nodes[role_slot] = observation.nodes[role_slot];
            record.node_indices[role_slot] = observation.node_indices[role_slot];
            record.bank_indices[role_slot] = observation.bank_indices[role_slot];
            if (role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT) {
                key.last_reader = record.nodes[role_slot];
            }
        }
    }
    bool observed_prefill = false;
    bool observed_decode = false;
    for (const auto & record : plan->groups_) {
        const auto & observation = observations[record.candidate.group_index];
        observed_prefill = observed_prefill || observation.prefill;
        observed_decode = observed_decode || observation.decode;
    }
    const bool call_prefill = cached_prefill || observed_prefill;
    const bool call_decode = cached_decode || observed_decode;
    const bool complete_coverage = diagnostics.counts[GGML_CUDA_MOE_GRAPH_COVERAGE_REGISTERED] == diagnostics.cached_mmid;
    const bool complete_slice = complete_coverage &&
        observed_cached_readers == diagnostics.counts[GGML_CUDA_MOE_GRAPH_COVERAGE_REGISTERED] && plan->n_groups_ != 0;
    bool mixed_certificate = call_prefill && call_decode && complete_slice;
    bool decode_certificate = call_decode && !call_prefill && complete_slice;
    bool decode_legacy_certificate = call_decode && !call_prefill && complete_slice;
    uint32_t legacy_groups = 0;
    for (const auto & record : plan->groups_) {
        const auto & observation = observations[record.candidate.group_index];
        const bool prefill_group = observation.prefill && !observation.decode;
        const bool decode_group = observation.decode && !observation.prefill;
        const bool materialization_legacy = decode_group &&
            record.reason == ggml_cuda_moe_graph_plan::GROUP_REASON_MATERIALIZATION;
        const bool execution_legacy = decode_group &&
            record.reason == ggml_cuda_moe_graph_plan::GROUP_REASON_EXECUTION;
        const bool consumer_legacy = decode_group &&
            record.reason == ggml_cuda_moe_graph_plan::GROUP_REASON_CONSUMER_EQUIVALENCE;
        // layer-split decode graphs often keep argsort on one device; fail closed to cached mmid
        const bool route_legacy = decode_group &&
            record.reason == ggml_cuda_moe_graph_plan::GROUP_REASON_ROUTE;
        mixed_certificate = mixed_certificate &&
            ((prefill_group && record.reason == ggml_cuda_moe_graph_plan::GROUP_REASON_PREFILL) ||
                (decode_group && (record.reason == ggml_cuda_moe_graph_plan::GROUP_REASON_ELIGIBLE ||
                    materialization_legacy || execution_legacy || consumer_legacy || route_legacy)));
        decode_certificate = decode_certificate && decode_group &&
            record.reason == ggml_cuda_moe_graph_plan::GROUP_REASON_ELIGIBLE;
        decode_legacy_certificate = decode_legacy_certificate && decode_group &&
            (record.reason == ggml_cuda_moe_graph_plan::GROUP_REASON_ELIGIBLE ||
                materialization_legacy || execution_legacy || consumer_legacy || route_legacy);
        legacy_groups += materialization_legacy || execution_legacy || consumer_legacy || route_legacy;
    }
    decode_legacy_certificate = decode_legacy_certificate && legacy_groups != 0;
    plan->outcome_ = call_prefill && !call_decode ? GGML_CUDA_MOE_GRAPH_OUTCOME_PREFILL_LEGACY :
        mixed_certificate ? GGML_CUDA_MOE_GRAPH_OUTCOME_PREFILL_LEGACY :
        decode_certificate ? GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED :
        decode_legacy_certificate ? GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_LEGACY : GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR;
    if (impl_->debug_stats() != nullptr) {
        for (const auto & record : plan->groups_) {
            const auto & candidate = impl_->table.groups[record.candidate.group_index];
            GGML_LOG_INFO(
                "moe-grouped-plan: owner=%p device=%d physical=%d generation=%llu semantic_group=%u domain=%u "
                "outcome=%u reason=%u\n",
                (void *) impl_.get(), impl_->device,
                impl_->device >= 0 ? ggml_cuda_info().devices[impl_->device].physical_device : -1,
                (unsigned long long) plan->registry_generation_, candidate.semantic_group_index, candidate.domain,
                (unsigned) plan->outcome_, record.reason);
        }
    }
    if (decode_legacy_certificate) {
        GGML_LOG_DEBUG("moe-cache: grouped decode selected legacy: groups=%u\n", legacy_groups);
        if (impl_->fallback_notice_generation.exchange(
                plan->registry_generation_, std::memory_order_relaxed) != plan->registry_generation_) {
            GGML_LOG_INFO("moe-cache: device %d grouped decode unavailable for %u group(s); "
                          "using cached mul_mat_id fallback\n", impl_->device, legacy_groups);
        }
    }
    if (mixed_certificate || decode_certificate || decode_legacy_certificate) {
        for (uint32_t record_index = 0; record_index < plan->n_groups_; ++record_index) {
            const auto & record = plan->groups_[record_index];
            if (record.reason == ggml_cuda_moe_graph_plan::GROUP_REASON_ROUTE) {
                continue;
            }
            const auto & group = impl_->table.groups[record.candidate.group_index];
            auto & dispatch = execution->groups_[record_index];
            dispatch.key.ids = record.ids;
            for (uint32_t role = GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT;
                    role <= GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT; ++role) {
                const uint32_t role_slot = role - GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT;
                if (record.nodes[role_slot] == nullptr) {
                    continue;
                }
                const auto & bank = group.banks[record.bank_indices[role_slot]];
                if (bank.slot_index >= record.n_banks || !plan->insert(
                        record.nodes[role_slot], record_index, role, record.bank_indices[role_slot], bank.slot_index)) {
                    plan->reset();
                    execution->reset();
                    return;
                }
            }
        }
    }
    execution->plan_ = plan;
    execution->owner_ = const_cast<ggml_cuda_moe_grouped_context *>(this);
    execution->n_groups_ = plan->n_groups_;
}

bool ggml_cuda_moe_grouped_context::graph_mmid_inventory_matches(
        const ggml_cgraph * cgraph,
        const ggml_cuda_moe_graph_plan & plan) const {
    uint32_t current_mmid_count = 0;
    uint64_t current_mmid_fingerprint = 0;
    if (!moe_candidate_graph_mmid_inventory(cgraph, &current_mmid_count, &current_mmid_fingerprint) ||
            current_mmid_count != plan.coverage_mmid_count_ ||
            current_mmid_fingerprint != plan.coverage_mmid_fingerprint_ ||
            !plan.inventory_complete_ || plan.mmid_inventory_.size() != plan.coverage_diagnostics_.cached_mmid ||
            plan.coverage_mmid_count_ != plan.mmid_inventory_.size()) {
        return false;
    }
    for (const auto & witness : plan.mmid_inventory_) {
        if (witness.node_index >= static_cast<uint32_t>(cgraph->n_nodes)) {
            return false;
        }
        const ggml_tensor * node = cgraph->nodes[witness.node_index];
        const ggml_tensor * source = node != nullptr && node->op == GGML_OP_MUL_MAT_ID ? node->src[0] : nullptr;
        if (source == nullptr || source->buffer == nullptr ||
                !ggml_backend_buft_is_cuda_moe_cached(ggml_backend_buffer_get_type(source->buffer)) ||
                node->flags != witness.flags ||
                !moe_candidate_signature_matches(witness.output, node) ||
                !moe_candidate_signature_matches(witness.source, source) ||
                !moe_candidate_signature_matches(witness.activation, node->src[1]) ||
                !moe_candidate_signature_matches(witness.ids, node->src[2])) {
            return false;
        }
    }
    return true;
}

bool ggml_cuda_moe_grouped_context::graph_group_witness_matches(
        const ggml_cgraph * cgraph,
        const ggml_cuda_moe_graph_plan::group_record & record) const {
    if (!record.witness_reusable || record.candidate.group_index >= impl_->table.groups.size()) {
        return false;
    }
    const auto & group = impl_->table.groups[record.candidate.group_index];
    if (record.reason == ggml_cuda_moe_graph_plan::GROUP_REASON_DESCRIPTOR) {
        return true;
    }
    if (group.banks.size() > GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS) {
        return false;
    }

    uint32_t structural_slot_banks = 0;
    uint32_t structural_original_direct_aux = 0;
    uint32_t structural_original_direct_bias = 0;
    const bool original_direct_nvfp4 =
        moe_candidate_structural_group(
            group, &structural_slot_banks, &structural_original_direct_aux, &structural_original_direct_bias) &&
        structural_slot_banks == 3 && structural_original_direct_aux == 3;
    if (record.reason == ggml_cuda_moe_graph_plan::GROUP_REASON_GEOMETRY && original_direct_nvfp4) {
        if (record.prefill != 0 || record.n_readers == 0) {
            return false;
        }
        bool geometry_invalid = false;
        for (uint32_t reader_index = 0; reader_index < record.n_readers; ++reader_index) {
            const auto & reader = record.readers[reader_index];
            if (reader.node_index >= static_cast<uint32_t>(cgraph->n_nodes) || reader.bank_index >= group.banks.size()) {
                return false;
            }
            const ggml_tensor * node = ggml_graph_node(const_cast<ggml_cgraph *>(cgraph), reader.node_index);
            const auto & bank = group.banks[reader.bank_index];
            if (node != reader.output.tensor || node == nullptr || node->op != GGML_OP_MUL_MAT_ID ||
                    (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || node->src[0] == nullptr ||
                    !moe_candidate_signature_matches(reader.output, node) ||
                    !moe_candidate_signature_matches(reader.activation, node->src[1]) ||
                    !moe_candidate_signature_matches(reader.ids, node->src[2]) ||
                    bank.info.role != reader.role || !moe_candidate_record_matches(bank, node->src[0])) {
                return false;
            }
            const auto geometry = moe_candidate_execution_geometry_for(cgraph, node, bank.ne[2], impl_->state.n_slots);
            if (!geometry.tensor_valid) {
                geometry_invalid = true;
            }
        }
        return geometry_invalid;
    }

    const uint32_t required_roles = moe_candidate_required_base_roles(record.layout);
    std::array<uint32_t, GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS> bank_readers = {};
    ggml_cuda_moe_ids_signature ids_signature;
    bool source_invalid = false;
    bool geometry_invalid = false;
    bool capability_invalid = false;
    bool consumer_incompatible = false;
    bool route_invalid = false;
    bool duplicate_role = false;
    bool mixed_ids = false;
    bool auxiliary = false;
    bool external_consumer = false;
    bool unproven = false;
    bool has_ids = false;
    bool prefill = false;
    bool decode = false;
    bool execution_ineligible = false;
    uint32_t seen_roles = 0;

    for (uint32_t reader_index = 0; reader_index < record.n_readers; ++reader_index) {
        const auto & reader = record.readers[reader_index];
        if (reader.node_index >= static_cast<uint32_t>(cgraph->n_nodes)) {
            return false;
        }
        const ggml_tensor * node = ggml_graph_node(const_cast<ggml_cgraph *>(cgraph), reader.node_index);
        if (node != reader.output.tensor || node == nullptr || node->op != GGML_OP_MUL_MAT_ID ||
                (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || node->src[0] == nullptr ||
                !moe_candidate_signature_matches(reader.output, node) ||
                !moe_candidate_signature_matches(reader.activation, node->src[1]) ||
                !moe_candidate_signature_matches(reader.ids, node->src[2]) ||
                reader.bank_index >= group.banks.size()) {
            return false;
        }
        const auto & bank = group.banks[reader.bank_index];
        if (bank.info.role != reader.role || !moe_candidate_record_matches(bank, node->src[0])) {
            source_invalid = true;
        }
        const auto geometry = moe_candidate_execution_geometry_for(cgraph, node, bank.ne[2], impl_->state.n_slots);
        const bool reader_prefill = geometry.phase == MOE_CANDIDATE_EXECUTION_PHASE_PREFILL && !geometry.required_grouped;
        prefill = prefill || reader_prefill;
        decode = decode || !reader_prefill;
        if (record.execution_semantic_key != geometry.semantic_key || record.top_k != geometry.top_k ||
                record.n_rows != geometry.n_rows || record.n_routes != geometry.n_routes ||
                record.row_stride != geometry.row_stride || record.strategy != geometry.strategy || !geometry.grouped_eligible) {
            execution_ineligible = true;
        }
        if (geometry.grouped_eligible) {
            if (bank.slot_index >= record.n_banks || !moe_candidate_capability_matches(
                    record.capabilities[bank.slot_index], bank, node, geometry.top_k, geometry.n_rows,
                    geometry.n_routes, geometry.row_stride, impl_->state.n_slots, impl_->device)) {
                capability_invalid = true;
            } else if (!moe_candidate_capability_invariant_valid(record.capabilities[bank.slot_index])) {
                capability_invalid = true;
            } else if (moe_candidate_capability_equivalence_unavailable(record.capabilities[bank.slot_index])) {
                consumer_incompatible = true;
            }
        }
        if (reader.bank_index < GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS) {
            bank_readers[reader.bank_index]++;
        } else {
            unproven = true;
        }

        const ggml_tensor * ids = node->src[2];
        const bool geometry_valid = geometry.tensor_valid;
        if (!geometry_valid) {
            geometry_invalid = true;
        }
        if (reader.role < GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT ||
                reader.role > GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT) {
            source_invalid = true;
        } else {
            const uint32_t role_bit = 1u << reader.role;
            duplicate_role = duplicate_role || (seen_roles & role_bit) != 0;
            seen_roles |= role_bit;
        }
        if (geometry.grouped_eligible) {
            const auto current_ids = moe_candidate_ids_signature(ids);
            if (has_ids) {
                mixed_ids = mixed_ids || !moe_candidate_ids_equal(ids_signature, current_ids);
                route_invalid = route_invalid || reader.node_index <= record.ids_node_index;
            } else {
                moe_candidate_route_proof current_route;
                const bool route_valid = moe_candidate_validate_route(ids, bank.ne[2], current_route);
                if (!route_valid || record.ids.tensor == nullptr ||
                        !moe_candidate_ids_equal(current_route.ids, record.ids) ||
                        !moe_candidate_ids_equal(current_route.root, record.ids_root) ||
                        !moe_candidate_ids_equal(current_route.source, record.ids_source) ||
                        !moe_candidate_ids_equal(current_route.indices, record.ids_indices) ||
                        record.ids_root_node_index >= static_cast<uint32_t>(cgraph->n_nodes) ||
                        record.ids_node_index >= static_cast<uint32_t>(cgraph->n_nodes) ||
                        ggml_graph_node(const_cast<ggml_cgraph *>(cgraph), record.ids_root_node_index) != record.ids_root.tensor ||
                        ggml_graph_node(const_cast<ggml_cgraph *>(cgraph), record.ids_node_index) != record.ids.tensor ||
                        record.ids_root_node_index > record.ids_node_index || record.ids_node_index >= reader.node_index) {
                    route_invalid = true;
                }
                ids_signature = current_ids;
                has_ids = true;
            }
        }

        bool reader_present = false;
        int32_t reader_use_count = 0;
        if (!moe_candidate_graph_use(cgraph, node, reader_present, reader_use_count) || !reader_present ||
                reader_use_count != reader.use_count || reader_use_count < static_cast<int32_t>(reader.n_consumers)) {
            unproven = true;
        } else if (reader_use_count > static_cast<int32_t>(reader.n_consumers)) {
            external_consumer = true;
        }
        for (uint32_t consumer_index = 0; consumer_index < reader.n_consumers; ++consumer_index) {
            const auto & consumer = reader.consumers[consumer_index];
            if (consumer.node_index >= static_cast<uint32_t>(cgraph->n_nodes) || consumer.src_index >= GGML_MAX_SRC ||
                    consumer.node_index <= reader.node_index) {
                return false;
            }
            const ggml_tensor * current = ggml_graph_node(const_cast<ggml_cgraph *>(cgraph), consumer.node_index);
            if (current != consumer.node || current == nullptr || current->op != consumer.op ||
                    memcmp(current->src, consumer.src, sizeof(consumer.src)) != 0 || current->src[consumer.src_index] != node) {
                return false;
            }
            if (moe_candidate_auxiliary_consumer(current, node, ids)) {
                if (reader.auxiliary_kind == MOE_CANDIDATE_AUXILIARY_SCALE && reader.n_auxiliary_nodes == 3 &&
                        reader.auxiliary_consumer_index == consumer_index &&
                        moe_candidate_original_direct_scale_witness_matches(
                            cgraph, consumer.node_index, current, node, ids, reader.role, group, reader.auxiliary_bank_index,
                            reader.auxiliary_nodes, reader.auxiliary_node_indices)) {
                    bank_readers[reader.auxiliary_bank_index]++;
                } else if (reader.auxiliary_kind == MOE_CANDIDATE_AUXILIARY_BIAS && reader.n_auxiliary_nodes == 0 &&
                        reader.auxiliary_consumer_index == consumer_index &&
                        moe_candidate_original_direct_bias_witness_matches(
                            current, node, ids, reader.role, group, reader.auxiliary_bank_index)) {
                    bank_readers[reader.auxiliary_bank_index]++;
                } else {
                    auxiliary = true;
                }
            }
        }
        const auto * bias = moe_candidate_find_role(group, moe_candidate_output_bias_role(reader.role));
        if (geometry.tensor_valid && !reader_prefill && bias != nullptr &&
                (reader.n_consumers != 1 || reader.auxiliary_kind != MOE_CANDIDATE_AUXILIARY_BIAS)) {
            auxiliary = true;
        }
        if (original_direct_nvfp4 && geometry.grouped_eligible && bias == nullptr &&
                (reader.n_consumers != 1 || reader.auxiliary_kind != MOE_CANDIDATE_AUXILIARY_SCALE)) {
            auxiliary = true;
        }
    }

    uint32_t n_slot_banks = 0;
    uint32_t n_original_direct_aux = 0;
    if (moe_candidate_structural_group(group, &n_slot_banks, &n_original_direct_aux) && n_original_direct_aux == 3) {
        for (uint32_t auxiliary_index = 0; auxiliary_index < n_original_direct_aux; ++auxiliary_index) {
            const auto * auxiliary_bank = moe_candidate_original_direct_aux_bank(group, auxiliary_index);
            if (auxiliary_bank == nullptr || bank_readers[auxiliary_bank - group.banks.data()] != 1) {
                auxiliary = true;
            }
        }
    }
    if (structural_original_direct_bias != 0) {
        for (uint32_t bank_index = 0; bank_index < group.banks.size(); ++bank_index) {
            if (moe_candidate_is_bias(group.banks[bank_index].info.role) && bank_readers[bank_index] != 1) {
                auxiliary = true;
            }
        }
    }

    for (uint32_t bank_index = 0; bank_index < group.banks.size(); ++bank_index) {
        const auto & saved = record.bank_uses[bank_index];
        bool present = false;
        int32_t use_count = 0;
        if (saved.tensor != group.banks[bank_index].info.tensor ||
                !moe_candidate_graph_use(cgraph, saved.tensor, present, use_count) || saved.present != (present ? 1u : 0u) ||
                saved.use_count != use_count || use_count != static_cast<int32_t>(bank_readers[bank_index])) {
            unproven = true;
        }
    }

    uint32_t reason = ggml_cuda_moe_graph_plan::GROUP_REASON_ELIGIBLE;
    if (source_invalid) {
        reason = ggml_cuda_moe_graph_plan::GROUP_REASON_SOURCE;
    } else if (geometry_invalid) {
        reason = ggml_cuda_moe_graph_plan::GROUP_REASON_GEOMETRY;
    } else if (execution_ineligible && !(prefill && !decode)) {
        reason = ggml_cuda_moe_graph_plan::GROUP_REASON_EXECUTION;
    } else if (capability_invalid) {
        reason = ggml_cuda_moe_graph_plan::GROUP_REASON_CAPABILITY;
    } else if (consumer_incompatible) {
        reason = ggml_cuda_moe_graph_plan::GROUP_REASON_CONSUMER_EQUIVALENCE;
    } else if (decode && (route_invalid || !has_ids)) {
        reason = ggml_cuda_moe_graph_plan::GROUP_REASON_ROUTE;
    } else if (duplicate_role) {
        reason = ggml_cuda_moe_graph_plan::GROUP_REASON_DUPLICATE_ROLE;
    } else if (mixed_ids) {
        reason = ggml_cuda_moe_graph_plan::GROUP_REASON_MIXED_IDS;
    } else if (auxiliary) {
        reason = ggml_cuda_moe_graph_plan::GROUP_REASON_AUXILIARY;
    } else if (external_consumer) {
        reason = ggml_cuda_moe_graph_plan::GROUP_REASON_EXTERNAL_CONSUMER;
    } else if (seen_roles != required_roles) {
        reason = ggml_cuda_moe_graph_plan::GROUP_REASON_MISSING_ROLE;
    } else if (unproven) {
        reason = ggml_cuda_moe_graph_plan::GROUP_REASON_UNPROVEN;
    } else if (prefill && !decode) {
        reason = ggml_cuda_moe_graph_plan::GROUP_REASON_PREFILL;
    } else if (record.reason == ggml_cuda_moe_graph_plan::GROUP_REASON_MATERIALIZATION) {
        reason = ggml_cuda_moe_graph_plan::GROUP_REASON_MATERIALIZATION;
    }
    return reason == record.reason && record.prefill == (prefill && !decode) && !unproven;
}

bool ggml_cuda_moe_grouped_context::bind_graph_plan(
        const ggml_cgraph * cgraph,
        uint64_t graph_uid,
        ggml_cuda_moe_graph_property_hint property_hint,
        const ggml_cuda_moe_graph_plan & plan,
        ggml_cuda_moe_graph_execution * execution,
        uint64_t coverage_epoch,
        const void * coverage_nodes,
        uint32_t coverage_mmid_count,
        uint64_t coverage_mmid_fingerprint) const {
    if (execution != nullptr) {
        execution->reset();
    }
    if (cgraph == nullptr || execution == nullptr || property_hint == GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED) {
        return false;
    }

    ggml_graph_execution_certificate execution_certificate = {};
    uint64_t execution_semantic_key = 0;
    (void) moe_candidate_execution_certificate(cgraph, &execution_certificate, &execution_semantic_key);
    if (execution_semantic_key != plan.execution_semantic_key_ ||
            !moe_candidate_execution_semantics_equal(execution_certificate, plan.execution_certificate_)) {
        return false;
    }

    std::lock_guard<std::mutex> lock(impl_->mutex);
    const int n_nodes = ggml_graph_n_nodes(const_cast<ggml_cgraph *>(cgraph));
    const void * graph_key = n_nodes > 0 ? ggml_graph_node(const_cast<ggml_cgraph *>(cgraph), 0) : nullptr;
    const bool coverage_matches = plan.coverage_epoch_ == 0 ||
        (plan.coverage_epoch_ == coverage_epoch && plan.coverage_nodes_ == coverage_nodes && coverage_nodes == cgraph->nodes &&
            plan.coverage_mmid_count_ == coverage_mmid_count &&
            plan.coverage_mmid_fingerprint_ == coverage_mmid_fingerprint);
    if (!plan.initialized_ || plan.owner_ != impl_.get() || plan.graph_key_ != graph_key || plan.graph_node_count_ != n_nodes ||
            impl_->draining || impl_->replacement_pending || plan.registry_generation_ != impl_->state.generation ||
            plan.n_groups_ > impl_->table.groups.size() || !coverage_matches || (graph_uid == 0 && plan.coverage_epoch_ == 0) ||
            (property_hint == GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN && !plan.unknown_reusable_)) {
        return false;
    }
    if (property_hint == GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN && !graph_mmid_inventory_matches(cgraph, plan)) {
        return false;
    }
    for (const auto & witness : plan.prefill_add_id_witnesses_) {
        if (witness.consumer.node_index >= static_cast<uint32_t>(n_nodes) ||
                witness.group_record >= plan.groups_.size()) {
            return false;
        }
        const auto & record = plan.groups_[witness.group_record];
        if (record.candidate.group_index >= impl_->table.groups.size()) {
            return false;
        }
        const ggml_tensor * node = ggml_graph_node(
            const_cast<ggml_cgraph *>(cgraph), witness.consumer.node_index);
        if (node == nullptr) {
            return false;
        }
        uint32_t producer_index = 0;
        while (producer_index < witness.consumer.node_index &&
                ggml_graph_node(const_cast<ggml_cgraph *>(cgraph), producer_index) != node->src[0]) {
            ++producer_index;
        }
        uint32_t bank_index = UINT32_MAX;
        uint32_t reader_role = GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID;
        if (node != witness.consumer.node || node->op != witness.consumer.op ||
                producer_index == witness.consumer.node_index ||
                memcmp(node->src, witness.consumer.src, sizeof(witness.consumer.src)) != 0 ||
                !moe_candidate_prefill_add_id_witness(
                    node, impl_->table.groups[record.candidate.group_index],
                    &bank_index, &reader_role) || bank_index != witness.bank_index || reader_role != witness.reader_role) {
            return false;
        }
    }
    if (!impl_->state.accepted || impl_->state.n_slots == 0 || impl_->table.groups.empty()) {
        if (plan.n_groups_ != 0) {
            return false;
        }
        execution->plan_ = &plan;
        execution->owner_ = const_cast<ggml_cuda_moe_grouped_context *>(this);
        return true;
    }

    for (uint32_t record_index = 0; record_index < plan.n_groups_; ++record_index) {
        const auto & record = plan.groups_[record_index];
        if (record.candidate.generation != impl_->state.generation || record.candidate.group_index >= impl_->table.groups.size()) {
            return false;
        }
        const auto & group = impl_->table.groups[record.candidate.group_index];
        if (group.layout != record.layout || moe_candidate_base_slot_bank_count(group) != record.n_banks) {
            return false;
        }
        const bool minimal_prefill = plan.outcome_ == GGML_CUDA_MOE_GRAPH_OUTCOME_PREFILL_LEGACY &&
            record.reason == ggml_cuda_moe_graph_plan::GROUP_REASON_PREFILL && record.n_readers == 0 && record.witness_reusable == 0;
        if (property_hint == GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN && !minimal_prefill &&
                !graph_group_witness_matches(cgraph, record)) {
            return false;
        }
        auto & dispatch = execution->groups_[record_index];
        dispatch.key.candidate = record.candidate;
        dispatch.key.layout = record.layout;
        dispatch.key.n_banks = record.n_banks;
        dispatch.key.execution_semantic_key = record.execution_semantic_key;
        dispatch.strategy = record.strategy;
        dispatch.capabilities = record.capabilities;
        dispatch.first_reader = record.authority_node;
        const uint32_t down_slot = GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT -
            GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT;
        dispatch.last_reader = record.nodes[down_slot];
        if (plan.outcome_ != GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED) {
            if (record.authority_node_index >= static_cast<uint32_t>(n_nodes)) {
                return false;
            }
            const ggml_tensor * node = ggml_graph_node(const_cast<ggml_cgraph *>(cgraph), record.authority_node_index);
            if (node == nullptr || node != record.authority_node || node->op != GGML_OP_MUL_MAT_ID ||
                    (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || node->src[0] == nullptr) {
                return false;
            }
            const auto * reverse = moe_candidate_active_reverse(impl_->table, node->src[0]);
            if (reverse == nullptr || reverse->group_index != record.candidate.group_index ||
                    !moe_candidate_record_matches(group.banks[reverse->bank_index], node->src[0])) {
                return false;
            }
            continue;
        }
        if (record.reason != ggml_cuda_moe_graph_plan::GROUP_REASON_ELIGIBLE) {
            return false;
        }
        uint32_t first_node_index = static_cast<uint32_t>(n_nodes);
        for (uint32_t role = GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT;
                role <= GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT; ++role) {
            const bool required = (moe_candidate_required_base_roles(record.layout) & (1u << role)) != 0;
            if (required) {
                const uint32_t role_slot = role - GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT;
                first_node_index = std::min(first_node_index, record.node_indices[role_slot]);
            }
        }
        moe_candidate_route_proof current_route;
        const auto * route_bank = moe_candidate_base_slot_bank(group, 0);
        if (route_bank == nullptr || record.ids_root_node_index >= static_cast<uint32_t>(n_nodes) ||
                record.ids_node_index >= static_cast<uint32_t>(n_nodes) || record.ids_root_node_index > record.ids_node_index ||
                record.ids_node_index >= first_node_index ||
                ggml_graph_node(const_cast<ggml_cgraph *>(cgraph), record.ids_root_node_index) != record.ids_root.tensor ||
                ggml_graph_node(const_cast<ggml_cgraph *>(cgraph), record.ids_node_index) != record.ids.tensor ||
                !moe_candidate_validate_route(record.ids.tensor, route_bank->ne[2], current_route) ||
                !moe_candidate_ids_equal(current_route.ids, record.ids) || !moe_candidate_ids_equal(current_route.root, record.ids_root) ||
                !moe_candidate_ids_equal(current_route.source, record.ids_source) ||
                !moe_candidate_ids_equal(current_route.indices, record.ids_indices)) {
            return false;
        }
        ggml_cuda_moe_ids_signature ids_signature;
        bool has_ids = false;
        uint32_t seen_roles = 0;
        for (uint32_t role = GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT;
                role <= GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT; ++role) {
            const bool required = (moe_candidate_required_base_roles(record.layout) & (1u << role)) != 0;
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
            const auto geometry = moe_candidate_execution_geometry_for(cgraph, node, bank.ne[2], impl_->state.n_slots);
            if (node == nullptr || node != record.nodes[role_slot] || node->op != GGML_OP_MUL_MAT_ID ||
                    (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
                    bank.info.role != role || !moe_candidate_record_matches(bank, node->src[0]) || !moe_candidate_ids_valid(ids) ||
                    node_index <= record.ids_node_index || !geometry.grouped_eligible ||
                    geometry.semantic_key != record.execution_semantic_key || geometry.top_k != record.top_k ||
                    geometry.n_rows != record.n_rows || geometry.n_routes != record.n_routes ||
                    geometry.row_stride != record.row_stride) {
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
        const uint32_t expected_roles = moe_candidate_required_base_roles(record.layout);
        if (!has_ids || seen_roles != expected_roles) {
            return false;
        }
        if (!moe_candidate_ids_equal(ids_signature, record.ids)) {
            return false;
        }
        dispatch.key.ids = ids_signature;
        dispatch.first_reader = ggml_graph_node(const_cast<ggml_cgraph *>(cgraph), first_node_index);
        dispatch.last_reader = record.nodes[down_slot];
    }
    execution->plan_ = &plan;
    execution->owner_ = const_cast<ggml_cuda_moe_grouped_context *>(this);
    execution->n_groups_ = plan.n_groups_;
    return true;
}

ggml_cuda_moe_graph_prepare_result ggml_cuda_moe_grouped_context::prepare_graph_execution(
        const ggml_cgraph * cgraph,
        uint64_t graph_uid,
        ggml_cuda_moe_graph_property_hint property_hint,
        std::shared_ptr<ggml_cuda_moe_graph_plan> * plan,
        ggml_cuda_moe_graph_execution * execution,
        uint64_t coverage_epoch,
        const void * coverage_nodes,
        uint32_t coverage_mmid_count,
        uint64_t coverage_mmid_fingerprint) const {
    if (execution != nullptr) {
        execution->reset();
    }
    if (cgraph == nullptr || plan == nullptr || execution == nullptr) {
        return GGML_CUDA_MOE_GRAPH_PREPARE_UNAVAILABLE;
    }
    const auto record_result = [&](ggml_cuda_moe_graph_prepare_result result) {
        auto * stats = impl_->debug_stats();
        bool eligible = false;
        if (stats != nullptr) {
            std::lock_guard<std::mutex> lifecycle_lock(impl_->resource_lifecycle_mutex);
            {
                std::lock_guard<std::mutex> lock(impl_->mutex);
                if (impl_->draining || impl_->replacement_pending || execution->plan_ == nullptr ||
                        execution->plan_->registry_generation_ != impl_->state.generation) {
                    return result;
                }
            }
            for (uint32_t record_index = 0; record_index < execution->n_groups_; ++record_index) {
                const auto & group = execution->groups_[record_index];
                if (execution->plan_->outcome_ == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED &&
                        group.key.candidate.group_index < GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS) {
                    eligible = true;
                    const uint32_t word = group.key.candidate.group_index / 64;
                    const uint32_t bit = group.key.candidate.group_index % 64;
                    stats->covered[word].fetch_or(1ULL << bit, std::memory_order_relaxed);
                }
            }
            if (eligible) {
                (void) impl_->debug_device_transfers();
                stats->plan_calls.fetch_add(1, std::memory_order_relaxed);
                if (result == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED) {
                    stats->plan_compiles.fetch_add(1, std::memory_order_relaxed);
                } else {
                    stats->plan_reuses.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
        return result;
    };
    if (*plan != nullptr && bind_graph_plan(
            cgraph, graph_uid, property_hint, **plan, execution, coverage_epoch, coverage_nodes,
            coverage_mmid_count, coverage_mmid_fingerprint)) {
        execution->retain(*plan);
        return record_result(GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
    }

    std::shared_ptr<ggml_cuda_moe_graph_plan> replacement;
    try {
        replacement = std::make_shared<ggml_cuda_moe_graph_plan>();
    } catch (...) {
        plan->reset();
        execution->reset();
        return GGML_CUDA_MOE_GRAPH_PREPARE_UNAVAILABLE;
    }
    compile_graph_plan(cgraph, graph_uid, replacement.get(), execution, coverage_epoch, coverage_nodes,
        coverage_mmid_count, coverage_mmid_fingerprint);
    if (!replacement->initialized_) {
        plan->reset();
        execution->reset();
        return GGML_CUDA_MOE_GRAPH_PREPARE_UNAVAILABLE;
    }
    *plan = replacement;
    execution->retain(replacement);
    return record_result(GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
}

static void moe_grouped_resource_fingerprint_add(uint64_t & fingerprint, uint64_t value) {
    fingerprint ^= value;
    fingerprint *= 1099511628211ULL;
}

static void moe_grouped_resource_fingerprint_add(uint64_t & fingerprint, const void * value) {
    moe_grouped_resource_fingerprint_add(fingerprint, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(value)));
}

bool ggml_cuda_moe_grouped_context::graph_resource_fingerprint_locked(
        const ggml_cuda_moe_graph_execution & execution,
        cudaStream_t stream,
        uint64_t * fingerprint,
        std::vector<std::shared_ptr<void>> * leases) const {
    if (fingerprint == nullptr || stream == nullptr || execution.owner_ != this || execution.plan_ == nullptr ||
            execution.plan_->owner_ != impl_.get() || execution.plan_->outcome_ != GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED ||
            execution.n_groups_ == 0 || execution.n_groups_ > impl_->table.groups.size() || impl_->draining ||
            impl_->replacement_pending || !impl_->state.accepted || execution.plan_->registry_generation_ != impl_->state.generation) {
        return false;
    }

    std::vector<std::shared_ptr<void>> result_leases;
    if (leases != nullptr) {
        try {
            result_leases.reserve(execution.n_groups_);
        } catch (const std::bad_alloc &) {
            return false;
        }
    }
    uint64_t result = 1469598103934665603ULL;
    moe_grouped_resource_fingerprint_add(result, impl_->state.generation);
    moe_grouped_resource_fingerprint_add(result, execution.plan_->execution_semantic_key_);
    moe_grouped_resource_fingerprint_add(result, execution.n_groups_);
    auto * debug = impl_->grouped_debug.load(std::memory_order_acquire);
    moe_grouped_resource_fingerprint_add(
        result, debug != nullptr ? debug->device_transfers.load(std::memory_order_acquire) : nullptr);
    for (uint32_t record_index = 0; record_index < execution.n_groups_; ++record_index) {
        const auto & group = execution.groups_[record_index];
        const uint32_t group_index = group.key.candidate.group_index;
        const uint32_t top_k = group.key.ids.ne[0] > 0 && group.key.ids.ne[0] <= UINT32_MAX ?
            static_cast<uint32_t>(group.key.ids.ne[0]) : 0;
        const uint32_t n_rows = group.key.ids.ne[1] > 0 && group.key.ids.ne[1] <= UINT32_MAX ?
            static_cast<uint32_t>(group.key.ids.ne[1]) : 0;
        const uint32_t n_routes = top_k != 0 && n_rows != 0 && top_k <= UINT32_MAX / n_rows ? top_k * n_rows : 0;
        const uint32_t row_stride = group.key.ids.nb[1] % sizeof(int32_t) == 0 &&
            group.key.ids.nb[1] / sizeof(int32_t) <= UINT32_MAX ?
                static_cast<uint32_t>(group.key.ids.nb[1] / sizeof(int32_t)) : 0;
        if (group.stream != stream || group.key.candidate.generation != impl_->state.generation ||
                group.key.execution_semantic_key == 0 ||
                group.key.execution_semantic_key != execution.plan_->execution_semantic_key_ ||
                top_k == 0 || n_rows == 0 || n_routes == 0 || n_routes > INT32_MAX || row_stride < top_k ||
                group.capabilities == nullptr || group_index >= impl_->resources.size()) {
            return false;
        }
        const auto * resource = impl_->resources[group_index].get();
        if (resource == nullptr || resource->device == nullptr || resource->device->serial == 0 ||
                n_routes > resource->snapshot.n_slots || resource->snapshot.n_slots > INT32_MAX ||
                resource->active_transaction_token != 0 || resource->active_decode_stream != nullptr ||
                impl_->refreshing[group_index] || !impl_->resource_matches_table(*resource) ||
                resource->snapshot.acquisition.candidate.generation != group.key.candidate.generation ||
                resource->snapshot.acquisition.candidate.group_index != group_index ||
                resource->snapshot.layout != group.key.layout || resource->snapshot.banks.size() != group.key.n_banks) {
            return false;
        }
        if (leases != nullptr) {
            result_leases.emplace_back(impl_->resources[group_index]);
        }
        const auto & device = *resource->device;
        size_t plan_bytes = 0;
        if (device.slot_for_expert == nullptr || device.expert_for_slot == nullptr || device.last_used == nullptr ||
                device.expert_frequency == nullptr || device.expert_frequency_epoch == nullptr || device.device_step == nullptr ||
                device.device_clock == nullptr || device.plan == nullptr || device.device_banks == nullptr ||
                device.completion == nullptr || device.bank_data.size() != resource->snapshot.banks.size() ||
                !moe_grouped_plan_size(resource->snapshot.n_slots, device.n_experts, &plan_bytes) ||
                device.plan_bytes != plan_bytes) {
            return false;
        }
        if (resource->snapshot.n_slot_auxiliaries > device.auxiliary_data.size() ||
                (resource->snapshot.n_slot_auxiliaries == 0) != (device.device_auxiliaries == nullptr)) {
            return false;
        }
        size_t auxiliary_values_per_miss = 0;
        for (uint32_t auxiliary = 0; auxiliary < device.auxiliary_data.size(); ++auxiliary) {
            if ((auxiliary < resource->snapshot.n_slot_auxiliaries) != (device.auxiliary_data[auxiliary] != nullptr) ||
                    (auxiliary < resource->snapshot.n_slot_auxiliaries) != (device.auxiliary_n_values[auxiliary] != 0)) {
                return false;
            }
            if (auxiliary < resource->snapshot.n_slot_auxiliaries) {
                if (device.auxiliary_n_values[auxiliary] > SIZE_MAX - auxiliary_values_per_miss) {
                    return false;
                }
                auxiliary_values_per_miss += device.auxiliary_n_values[auxiliary];
            }
        }
        if (auxiliary_values_per_miss != device.auxiliary_values_per_miss) {
            return false;
        }
        bool any_prefill_resident = false;
        bool all_prefill_resident = resource->snapshot.prefill_resident_auxiliary &&
            resource->snapshot.n_slot_auxiliaries != 0;
        uint64_t prefill_resident_bytes = 0;
        for (uint32_t auxiliary = 0; auxiliary < device.prefill_auxiliary_data.size(); ++auxiliary) {
            const bool present = device.prefill_auxiliary_data[auxiliary] != nullptr;
            any_prefill_resident = any_prefill_resident || present;
            if (auxiliary < resource->snapshot.n_slot_auxiliaries) {
                const auto & descriptor = resource->snapshot.slot_auxiliaries[auxiliary];
                const bool valid = present && moe_candidate_is_bias(descriptor.role) &&
                    moe_candidate_add(prefill_resident_bytes, descriptor.byte_extent, prefill_resident_bytes);
                all_prefill_resident = all_prefill_resident && valid;
            } else if (present) {
                return false;
            }
        }
        if (any_prefill_resident != all_prefill_resident ||
                (any_prefill_resident && prefill_resident_bytes != resource->snapshot.prefill_resident_auxiliary_bytes)) {
            return false;
        }
        moe_grouped_resource_fingerprint_add(result, group_index);
        moe_grouped_resource_fingerprint_add(result, group.key.execution_semantic_key);
        moe_grouped_resource_fingerprint_add(result, top_k);
        moe_grouped_resource_fingerprint_add(result, n_rows);
        moe_grouped_resource_fingerprint_add(result, n_routes);
        moe_grouped_resource_fingerprint_add(result, row_stride);
        moe_grouped_resource_fingerprint_add(result, resource->snapshot.acquisition.resource_generation);
        moe_grouped_resource_fingerprint_add(result, device.serial);
        moe_grouped_resource_fingerprint_add(result, group.stream);
        moe_grouped_resource_fingerprint_add(result, device.slot_for_expert);
        moe_grouped_resource_fingerprint_add(result, device.expert_for_slot);
        moe_grouped_resource_fingerprint_add(result, device.last_used);
        moe_grouped_resource_fingerprint_add(result, device.expert_frequency);
        moe_grouped_resource_fingerprint_add(result, device.expert_frequency_epoch);
        moe_grouped_resource_fingerprint_add(result, device.device_step);
        moe_grouped_resource_fingerprint_add(result, device.device_clock);
        moe_grouped_resource_fingerprint_add(result, device.plan);
        moe_grouped_resource_fingerprint_add(result, device.plan_bytes);
        moe_grouped_resource_fingerprint_add(result, device.device_banks);
        moe_grouped_resource_fingerprint_add(result, device.device_auxiliaries);
        moe_grouped_resource_fingerprint_add(result, device.original_auxiliary_bytes);
        moe_grouped_resource_fingerprint_add(result, resource->snapshot.n_original_auxiliaries);
        for (uint32_t i = 0; i < resource->snapshot.n_original_auxiliaries; ++i) {
            const auto & auxiliary = resource->snapshot.original_auxiliaries[i];
            moe_grouped_resource_fingerprint_add(result, device.original_auxiliary_data[i]);
            moe_grouped_resource_fingerprint_add(result, auxiliary.tensor);
            moe_grouped_resource_fingerprint_add(result, auxiliary.source_data);
            moe_grouped_resource_fingerprint_add(result, auxiliary.role);
            moe_grouped_resource_fingerprint_add(result, auxiliary.byte_extent);
        }
        moe_grouped_resource_fingerprint_add(result, resource->snapshot.n_slot_auxiliaries);
        moe_grouped_resource_fingerprint_add(result, device.auxiliary_values_per_miss);
        for (uint32_t auxiliary = 0; auxiliary < resource->snapshot.n_slot_auxiliaries; ++auxiliary) {
            moe_grouped_resource_fingerprint_add(result, device.auxiliary_data[auxiliary]);
            moe_grouped_resource_fingerprint_add(result, device.auxiliary_n_values[auxiliary]);
            moe_grouped_resource_fingerprint_add(result, resource->snapshot.slot_auxiliaries[auxiliary].tensor);
            moe_grouped_resource_fingerprint_add(result, resource->snapshot.slot_auxiliaries[auxiliary].source_data);
            moe_grouped_resource_fingerprint_add(result, resource->snapshot.slot_auxiliaries[auxiliary].role);
            moe_grouped_resource_fingerprint_add(result, device.prefill_auxiliary_data[auxiliary]);
            moe_grouped_resource_fingerprint_add(result, resource->snapshot.slot_auxiliaries[auxiliary].byte_extent);
            moe_grouped_resource_fingerprint_add(result, resource->snapshot.acquisition.resource_generation);
        }
        for (uint32_t bank_index = 0; bank_index < resource->snapshot.banks.size(); ++bank_index) {
            const auto & bank = resource->snapshot.banks[bank_index];
            const auto & capability = group.capabilities[bank_index];
            int64_t grouped_ne[GGML_MAX_DIMS];
            size_t grouped_nb[GGML_MAX_DIMS];
            memcpy(grouped_ne, bank.ne, sizeof(grouped_ne));
            memcpy(grouped_nb, bank.nb, sizeof(grouped_nb));
            grouped_ne[2] = impl_->state.n_slots;
            const bool grouped_stride_valid = grouped_nb[2] <= SIZE_MAX / impl_->state.n_slots;
            if (grouped_stride_valid) {
                grouped_nb[3] = grouped_nb[2] * impl_->state.n_slots;
            }
            if (device.bank_data[bank_index] == nullptr || !moe_candidate_capability_supported(capability) ||
                    capability.tensor != bank.tensor || capability.source_data != bank.source_data ||
                    capability.byte_extent != bank.byte_extent || capability.expert_stride != bank.expert_stride ||
                    memcmp(capability.source_ne, bank.ne, sizeof(capability.source_ne)) != 0 ||
                    memcmp(capability.source_nb, bank.nb, sizeof(capability.source_nb)) != 0 ||
                    !grouped_stride_valid || memcmp(capability.grouped_ne, grouped_ne, sizeof(grouped_ne)) != 0 ||
                    memcmp(capability.grouped_nb, grouped_nb, sizeof(grouped_nb)) != 0 ||
                    capability.top_k != top_k || capability.n_rows != n_rows || capability.n_routes != n_routes ||
                    capability.row_stride != row_stride || capability.n_slots != impl_->state.n_slots) {
                return false;
            }
            moe_grouped_resource_fingerprint_add(result, device.bank_data[bank_index]);
            moe_grouped_resource_fingerprint_add(result, bank.source_data);
            moe_grouped_resource_fingerprint_add(result, capability.consumer);
            moe_grouped_resource_fingerprint_add(result, capability.reason);
            moe_grouped_resource_fingerprint_add(result, capability.equivalence_reason);
            moe_grouped_resource_fingerprint_add(result, capability.phase);
            moe_grouped_resource_fingerprint_add(result, capability.mapping);
            moe_grouped_resource_fingerprint_add(result, capability.materialized_phase);
            moe_grouped_resource_fingerprint_add(result, capability.materialized_mapping);
            moe_grouped_resource_fingerprint_add(result, capability.row_semantics);
            moe_grouped_resource_fingerprint_add(result, capability.strategy);
            moe_grouped_resource_fingerprint_add(result, capability.use_mmq);
            moe_grouped_resource_fingerprint_add(result, capability.n_tokens);
            moe_grouped_resource_fingerprint_add(result, capability.n_experts);
            moe_grouped_resource_fingerprint_add(result, capability.smpbo);
            moe_grouped_resource_fingerprint_add(result, capability.device);
            moe_grouped_resource_fingerprint_add(result, capability.cc);
            moe_grouped_resource_fingerprint_add(result, capability.warp_size);
            moe_grouped_resource_fingerprint_add(result, capability.top_k);
            moe_grouped_resource_fingerprint_add(result, capability.n_rows);
            moe_grouped_resource_fingerprint_add(result, capability.n_routes);
            moe_grouped_resource_fingerprint_add(result, capability.row_stride);
            moe_grouped_resource_fingerprint_add(result, capability.n_slots);
            moe_grouped_resource_fingerprint_add(result, capability.source_type);
            moe_grouped_resource_fingerprint_add(result, capability.source_flags);
            moe_grouped_resource_fingerprint_add(result, capability.input_type);
            moe_grouped_resource_fingerprint_add(result, capability.output_type);
            for (uint32_t dim = 0; dim < GGML_MAX_DIMS; ++dim) {
                moe_grouped_resource_fingerprint_add(result, capability.source_ne[dim]);
                moe_grouped_resource_fingerprint_add(result, capability.source_nb[dim]);
                moe_grouped_resource_fingerprint_add(result, capability.grouped_ne[dim]);
                moe_grouped_resource_fingerprint_add(result, capability.grouped_nb[dim]);
            }
        }
    }
    *fingerprint = result != 0 ? result : 1;
    if (leases != nullptr) {
        *leases = std::move(result_leases);
    }
    return true;
}

bool ggml_cuda_moe_grouped_context::graph_resource_fingerprint(
        const ggml_cuda_moe_graph_execution & execution,
        cudaStream_t stream,
        uint64_t * fingerprint,
        std::vector<std::shared_ptr<void>> * leases) const {
    if (fingerprint != nullptr) {
        *fingerprint = 0;
    }
    if (leases != nullptr) {
        leases->clear();
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return graph_resource_fingerprint_locked(execution, stream, fingerprint, leases);
}

bool ggml_cuda_moe_grouped_context::activate_graph_resources(
        ggml_cuda_moe_graph_execution * execution,
        ggml_cuda_moe_graph_dispatch_mode mode,
        uint64_t expected_fingerprint,
        const std::vector<std::weak_ptr<void>> * resource_witnesses) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(execution);
    GGML_UNUSED(mode);
    GGML_UNUSED(expected_fingerprint);
    GGML_UNUSED(resource_witnesses);
    return false;
#else
    if (execution == nullptr || expected_fingerprint == 0 ||
            (mode != GGML_CUDA_MOE_GRAPH_DISPATCH_CAPTURE && mode != GGML_CUDA_MOE_GRAPH_DISPATCH_REPLAY) ||
            (mode == GGML_CUDA_MOE_GRAPH_DISPATCH_REPLAY && resource_witnesses == nullptr)) {
        return false;
    }
    auto * debug = impl_->debug_stats();
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto decline = [&]() {
        for (uint32_t record_index = 0; record_index < execution->n_groups_; ++record_index) {
            if (execution->groups_[record_index].state == GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_ARMED) {
                execution->groups_[record_index].state = GGML_CUDA_MOE_GRAPH_GROUP_FINISHED;
            }
        }
        return false;
    };
    if (!execution->dispatch_active_ || execution->dispatch_mode_ != mode || execution->owner_ != this ||
            execution->n_groups_ == 0 || execution->groups_[0].stream == nullptr) {
        return decline();
    }
    uint64_t fingerprint = 0;
    if (!graph_resource_fingerprint_locked(*execution, execution->groups_[0].stream, &fingerprint, nullptr) ||
            fingerprint != expected_fingerprint) {
        return decline();
    }
    if (mode == GGML_CUDA_MOE_GRAPH_DISPATCH_REPLAY) {
        if (resource_witnesses->size() != execution->n_groups_) {
            return decline();
        }
        for (uint32_t record_index = 0; record_index < execution->n_groups_; ++record_index) {
            const uint32_t group_index = execution->groups_[record_index].key.candidate.group_index;
            const auto & resource = impl_->resources[group_index];
            const auto & witness = (*resource_witnesses)[record_index];
            if (witness.owner_before(resource) || resource.owner_before(witness)) {
                return decline();
            }
        }
    }
    if (impl_->next_transaction_token > UINT64_MAX - execution->n_groups_) {
        return decline();
    }
    for (uint32_t record_index = 0; record_index < execution->n_groups_; ++record_index) {
        auto & group = execution->groups_[record_index];
        auto * resource = impl_->resources[group.key.candidate.group_index].get();
        const uint32_t n_routes = static_cast<uint32_t>(group.key.ids.ne[0] * group.key.ids.ne[1]);
        if (group.state != GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_ARMED || resource == nullptr || resource->device == nullptr ||
                n_routes == 0 || n_routes > resource->snapshot.n_slots || resource->device->clock_bound > UINT64_MAX - n_routes ||
                (mode == GGML_CUDA_MOE_GRAPH_DISPATCH_REPLAY && !resource->device->graph_clock_active)) {
            return decline();
        }
    }
    for (uint32_t record_index = 0; record_index < execution->n_groups_; ++record_index) {
        auto & group = execution->groups_[record_index];
        auto & device = *impl_->resources[group.key.candidate.group_index]->device;
        if (device.has_completion && device.completion_stream != group.stream) {
            if (!moe_grouped_cuda_success(cudaStreamWaitEvent(group.stream, device.completion, 0))) {
                return decline();
            }
            device.completion_stream = group.stream;
        }
    }
    if (mode == GGML_CUDA_MOE_GRAPH_DISPATCH_CAPTURE) {
        for (uint32_t record_index = 0; record_index < execution->n_groups_; ++record_index) {
            auto & group = execution->groups_[record_index];
            auto & device = *impl_->resources[group.key.candidate.group_index]->device;
            if (!device.graph_clock_active) {
                moe_grouped_set_clock<<<1, 1, 0, group.stream>>>(device.device_clock, device.clock_bound);
                if (!moe_grouped_cuda_success(cudaGetLastError())) {
                    return decline();
                }
                device.graph_clock_active = true;
            }
        }
        for (uint32_t record_index = 0; record_index < execution->n_groups_; ++record_index) {
            execution->groups_[record_index].defer_completion = true;
        }
        return true;
    }

    for (uint32_t record_index = 0; record_index < execution->n_groups_; ++record_index) {
        auto & group = execution->groups_[record_index];
        auto & resource = *impl_->resources[group.key.candidate.group_index];
        auto & device = *resource.device;
        const uint32_t n_routes = static_cast<uint32_t>(group.key.ids.ne[0] * group.key.ids.ne[1]);
        resource.active_transaction_token = ++impl_->next_transaction_token;
        resource.active_decode_stream = group.stream;
        group.transaction.acquisition = resource.snapshot.acquisition;
        group.transaction.transaction_token = resource.active_transaction_token;
        group.remapped_ids = moe_grouped_plan_array_ptr(
            device.plan, resource.snapshot.n_slots, MOE_GROUPED_PLAN_REMAPPED_IDS);
        std::fill_n(group.bank_data, GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS, nullptr);
        for (uint32_t bank_index = 0; bank_index < resource.snapshot.banks.size(); ++bank_index) {
            group.bank_data[bank_index] = device.bank_data[bank_index];
        }
        std::fill_n(group.auxiliary_tensors, 3, nullptr);
        std::fill_n(group.auxiliary_data, 3, nullptr);
        std::fill_n(group.auxiliary_roles, 3, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID);
        for (uint32_t auxiliary = 0; auxiliary < resource.snapshot.n_slot_auxiliaries; ++auxiliary) {
            group.auxiliary_tensors[auxiliary] = resource.snapshot.slot_auxiliaries[auxiliary].tensor;
            group.auxiliary_data[auxiliary] = device.auxiliary_data[auxiliary];
            group.auxiliary_roles[auxiliary] = resource.snapshot.slot_auxiliaries[auxiliary].role;
        }
        group.n_slots = resource.snapshot.n_slots;
        group.n_auxiliary_shadows = resource.snapshot.n_slot_auxiliaries;
        group.state = GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_REPLAY;
        device.clock_bound += n_routes;
        if (debug != nullptr) {
            debug->calls.fetch_add(1, std::memory_order_relaxed);
            debug->ready.fetch_add(1, std::memory_order_relaxed);
            debug->admitted_banks.fetch_add(resource.snapshot.banks.size(), std::memory_order_relaxed);
            const uint32_t group_index = group.key.candidate.group_index;
            if (group_index < debug->ready_by_group.size()) {
                debug->ready_by_group[group_index].fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    return true;
#endif
}

bool ggml_cuda_moe_grouped_context::begin_graph_dispatch(
        ggml_cuda_moe_graph_execution * execution,
        ggml_cuda_moe_graph_dispatch_mode mode) {
    if (execution == nullptr) {
        return false;
    }

    auto * debug = impl_->debug_stats();
    std::lock_guard<std::mutex> lifecycle_lock(impl_->resource_lifecycle_mutex);
    const uint32_t n_groups = execution->n_groups_;
    const uint64_t generation = execution->plan_ != nullptr ? execution->plan_->registry_generation_ : 0;
    std::array<ggml_cuda_moe_group_authority, GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS> desired = {};
    std::array<uint8_t, GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS> barrier = {};
    size_t legacy_capacity = 0;
    size_t resource_capacity = 0;
    bool has_barrier = false;
    const auto                                                                       record_dispatch   = [&]() {
        if (debug == nullptr) {
            return;
        }
        const auto outcome = execution->plan_->outcome_;
        if (outcome == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED && mode != GGML_CUDA_MOE_GRAPH_DISPATCH_LEGACY) {
            debug->decode_grouped.fetch_add(n_groups, std::memory_order_relaxed);
            if (mode == GGML_CUDA_MOE_GRAPH_DISPATCH_REPLAY) {
                debug->replays.fetch_add(n_groups, std::memory_order_relaxed);
            } else if (mode == GGML_CUDA_MOE_GRAPH_DISPATCH_CAPTURE) {
                debug->captures.fetch_add(n_groups, std::memory_order_relaxed);
            } else {
                debug->direct.fetch_add(n_groups, std::memory_order_relaxed);
            }
        } else if (outcome == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_LEGACY ||
                   (outcome == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED &&
                    mode == GGML_CUDA_MOE_GRAPH_DISPATCH_LEGACY)) {
            debug->decode_legacy.fetch_add(n_groups, std::memory_order_relaxed);
            debug->fallback.fetch_add(n_groups, std::memory_order_relaxed);
        }
    };
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (execution->dispatch_active_ || execution->owner_ != this || execution->plan_ == nullptr ||
                execution->plan_->owner_ != impl_.get() || impl_->draining || impl_->replacement_pending ||
                !impl_->state.accepted || generation != impl_->state.generation || n_groups > impl_->table.groups.size()) {
            return false;
        }
        const auto outcome = execution->plan_->outcome_;
        const bool grouped_enabled = mode != GGML_CUDA_MOE_GRAPH_DISPATCH_LEGACY;
        if (mode > GGML_CUDA_MOE_GRAPH_DISPATCH_REPLAY ||
                ((mode == GGML_CUDA_MOE_GRAPH_DISPATCH_CAPTURE || mode == GGML_CUDA_MOE_GRAPH_DISPATCH_REPLAY) &&
                    (outcome != GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED || !execution->allows_graph_capture()))) {
            return false;
        }
        if (grouped_enabled && outcome == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR) {
            const auto graph_reason_name = [](uint32_t reason) {
                switch (reason) {
                    case ggml_cuda_moe_graph_plan::GROUP_REASON_ELIGIBLE:             return "eligible";
                    case ggml_cuda_moe_graph_plan::GROUP_REASON_PREFILL:              return "prefill";
                    case ggml_cuda_moe_graph_plan::GROUP_REASON_EXECUTION:            return "execution";
                    case ggml_cuda_moe_graph_plan::GROUP_REASON_DESCRIPTOR:           return "descriptor";
                    case ggml_cuda_moe_graph_plan::GROUP_REASON_SOURCE:               return "source";
                    case ggml_cuda_moe_graph_plan::GROUP_REASON_GEOMETRY:             return "geometry";
                    case ggml_cuda_moe_graph_plan::GROUP_REASON_CAPABILITY:           return "capability";
                    case ggml_cuda_moe_graph_plan::GROUP_REASON_CONSUMER_EQUIVALENCE: return "consumer_equivalence";
                    case ggml_cuda_moe_graph_plan::GROUP_REASON_ROUTE:                return "route";
                    case ggml_cuda_moe_graph_plan::GROUP_REASON_DUPLICATE_ROLE:       return "duplicate_role";
                    case ggml_cuda_moe_graph_plan::GROUP_REASON_MIXED_IDS:            return "mixed_ids";
                    case ggml_cuda_moe_graph_plan::GROUP_REASON_AUXILIARY:            return "unsupported_auxiliary";
                    case ggml_cuda_moe_graph_plan::GROUP_REASON_EXTERNAL_CONSUMER:    return "external_consumer";
                    case ggml_cuda_moe_graph_plan::GROUP_REASON_MISSING_ROLE:         return "missing_role";
                    case ggml_cuda_moe_graph_plan::GROUP_REASON_UNPROVEN:             return "unproven";
                    case ggml_cuda_moe_graph_plan::GROUP_REASON_MATERIALIZATION:      return "materialization";
                    default:                                                          return "unknown";
                }
            };
            for (const auto & record : execution->plan_->groups_) {
                for (uint32_t bank_index = 0; bank_index < record.n_banks; ++bank_index) {
                    const auto & capability = record.capabilities[bank_index];
                    if (capability.tensor != nullptr && !moe_candidate_capability_supported(capability)) {
                        const char * type = capability.source_type < GGML_TYPE_COUNT ?
                            ggml_type_name(static_cast<ggml_type>(capability.source_type)) : "unknown";
                        const uint32_t semantic_group = impl_->table.groups[record.candidate.group_index].semantic_group_index;
                        GGML_LOG_ERROR("moe-cache: grouped decode certificate failed: semantic_group=%u role=%u tensor=%s type=%s capability=%s graph=%u\n",
                            semantic_group, capability.role, capability.tensor != nullptr ? capability.tensor->name : "unknown",
                            type, moe_candidate_capability_reason_name(capability.reason), record.reason);
                        return false;
                    }
                }
            }
            const auto failed_group = std::find_if(execution->plan_->groups_.begin(), execution->plan_->groups_.end(),
                [](const ggml_cuda_moe_graph_plan::group_record & record) {
                    return record.reason != ggml_cuda_moe_graph_plan::GROUP_REASON_ELIGIBLE &&
                        record.reason != ggml_cuda_moe_graph_plan::GROUP_REASON_PREFILL;
                });
            if (failed_group != execution->plan_->groups_.end()) {
                const auto & record = *failed_group;
                const auto & candidate = impl_->table.groups[record.candidate.group_index];
                const char * tensor = record.authority_node != nullptr && record.authority_node->src[0] != nullptr ?
                    record.authority_node->src[0]->name : "unknown";
                GGML_LOG_ERROR("moe-cache: grouped decode certificate failed: semantic_group=%u layout=%u domain=%u tensor=%s graph=%s(%u)\n",
                    candidate.semantic_group_index, candidate.layout, candidate.domain, tensor,
                    graph_reason_name(record.reason), record.reason);
                return false;
            }
            const auto & diagnostics = execution->plan_->coverage_diagnostics_;
            uint32_t reason = GGML_CUDA_MOE_GRAPH_COVERAGE_REVERSE_MAP_MISS;
            while (reason < GGML_CUDA_MOE_GRAPH_COVERAGE_REASON_COUNT && diagnostics.counts[reason] == 0) {
                ++reason;
            }
            const char * tensor = reason < GGML_CUDA_MOE_GRAPH_COVERAGE_REASON_COUNT && diagnostics.first_source[reason] != nullptr ?
                diagnostics.first_source[reason]->name : "unknown";
            GGML_LOG_ERROR("moe-cache: grouped decode certificate failed: cached_mmid=%u coverage=%s(%u) semantic_group=%u layout=%u domain=%u role=%u tensor=%s\n",
                diagnostics.cached_mmid, moe_candidate_coverage_reason_name(reason), reason,
                reason < GGML_CUDA_MOE_GRAPH_COVERAGE_REASON_COUNT ? diagnostics.first_group_index[reason] : UINT32_MAX,
                reason < GGML_CUDA_MOE_GRAPH_COVERAGE_REASON_COUNT ? diagnostics.first_layout[reason] : 0,
                reason < GGML_CUDA_MOE_GRAPH_COVERAGE_REASON_COUNT ? diagnostics.first_domain[reason] : 0,
                reason < GGML_CUDA_MOE_GRAPH_COVERAGE_REASON_COUNT ? diagnostics.first_role[reason] : 0, tensor);
            return false;
        }
        const bool grouped_call = grouped_enabled && outcome == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED;
        uint64_t epoch_count = 0;
        for (uint32_t record_index = 0; record_index < n_groups; ++record_index) {
            const auto & dispatch = execution->groups_[record_index];
            const uint32_t group_index = dispatch.key.candidate.group_index;
            if (dispatch.authority || dispatch.key.candidate.generation != generation ||
                    group_index >= impl_->table.groups.size()) {
                return false;
            }
            if (grouped_call && impl_->device >= 0 && dispatch.stream == nullptr) {
                GGML_LOG_ERROR("moe-cache: grouped decode certificate failed: semantic_group=%u has no single CUDA stream\n",
                    impl_->table.groups[group_index].semantic_group_index);
                return false;
            }
            if (!grouped_call) {
                desired[record_index] = GGML_CUDA_MOE_GROUP_AUTHORITY_LEGACY;
            } else if (dispatch.strategy == GGML_CUDA_MOE_EXECUTION_STRATEGY_DEVICE_DIRECT) {
                desired[record_index] = GGML_CUDA_MOE_GROUP_AUTHORITY_GROUPED;
            } else if (dispatch.strategy == GGML_CUDA_MOE_EXECUTION_STRATEGY_HOST_STAGED) {
                desired[record_index] = GGML_CUDA_MOE_GROUP_AUTHORITY_GROUPED_HOST_STAGED;
            } else {
                return false;
            }
            const auto & current = impl_->group_authorities[group_index];
            barrier[record_index] = current.authority != desired[record_index] ||
                (desired[record_index] != GGML_CUDA_MOE_GROUP_AUTHORITY_LEGACY && current.active_calls != 0);
            has_barrier = has_barrier || barrier[record_index];
            epoch_count += current.authority != desired[record_index];
            if (current.active_calls == UINT32_MAX || epoch_count > UINT64_MAX - impl_->next_group_authority_epoch) {
                return false;
            }
        }
        legacy_capacity = impl_->table.active_weights;
        resource_capacity = impl_->resources.size();
        if (!has_barrier) {
            for (uint32_t record_index = 0; record_index < n_groups; ++record_index) {
                const uint32_t group_index = execution->groups_[record_index].key.candidate.group_index;
                auto & current = impl_->group_authorities[group_index];
                if (current.admission_closed || current.authority != desired[record_index] || current.active_calls == UINT32_MAX) {
                    return false;
                }
                ++current.active_calls;
                auto & lease = execution->groups_[record_index].authority;
                lease.owner_ = this;
                lease.candidate_generation_ = generation;
                lease.authority_epoch_ = current.epoch;
                lease.group_index_ = group_index;
                lease.authority_ = current.authority;
                lease.execution_domain_ = execution->plan_->execution_certificate_.domain;
                lease.row_semantics_ = execution->plan_->execution_certificate_.row_semantics;
                lease.prefill_resident_certified_ = std::any_of(
                    execution->plan_->prefill_add_id_witnesses_.begin(),
                    execution->plan_->prefill_add_id_witnesses_.end(),
                    [record_index](const ggml_cuda_moe_graph_plan::prefill_add_id_witness & witness) {
                        return witness.group_record == record_index;
                    });
                auto & dispatch = execution->groups_[record_index];
                dispatch.transaction = {};
                dispatch.remapped_ids = nullptr;
                std::fill_n(dispatch.bank_data, GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS, nullptr);
                std::fill_n(dispatch.auxiliary_tensors, 3, nullptr);
                std::fill_n(dispatch.auxiliary_data, 3, nullptr);
                std::fill_n(dispatch.auxiliary_roles, 3, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID);
                dispatch.n_slots = 0;
                dispatch.n_auxiliary_shadows = 0;
                dispatch.defer_completion = false;
                dispatch.state = current.authority == GGML_CUDA_MOE_GROUP_AUTHORITY_GROUPED ?
                    GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_ARMED :
                    current.authority == GGML_CUDA_MOE_GROUP_AUTHORITY_GROUPED_HOST_STAGED ?
                        GGML_CUDA_MOE_GRAPH_GROUP_HOST_STAGED_ARMED : GGML_CUDA_MOE_GRAPH_GROUP_WHOLE_LEGACY;
            }
            execution->dispatch_mode_ = mode;
            execution->dispatch_active_ = true;
            record_dispatch();
            return true;
        }
    }

    impl::legacy_record_list retired_legacy;
    impl::resource_slots retired_resources;
    struct cache_handoff {
        ggml_cuda_moe_cache * cache;
        cudaEvent_t grouped_done;
        cudaStream_t grouped_stream;
        bool to_legacy;
    };
    struct grouped_reset {
        std::shared_ptr<impl::grouped_resource> resource;
        cudaStream_t stream;
    };
    struct host_staging_transition {
        ggml_cuda_moe_cache * cache;
        cudaEvent_t grouped_done;
        cudaStream_t grouped_stream;
        bool to_host_staged;
    };
    std::vector<cache_handoff> cache_handoffs;
    std::vector<grouped_reset> grouped_resets;
    std::vector<host_staging_transition> host_staging_transitions;
    try {
        retired_legacy.reserve(legacy_capacity);
        retired_resources.resize(resource_capacity);
        cache_handoffs.reserve(legacy_capacity);
        grouped_resets.reserve(n_groups);
        host_staging_transitions.reserve(n_groups * GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS);
    } catch (const std::bad_alloc &) {
        return false;
    }

    {
        std::unique_lock<std::mutex> lock(impl_->mutex);
        if (impl_->draining || impl_->replacement_pending || !impl_->state.accepted || generation != impl_->state.generation) {
            return false;
        }
        for (uint32_t record_index = 0; record_index < n_groups; ++record_index) {
            if (barrier[record_index]) {
                impl_->group_authorities[execution->groups_[record_index].key.candidate.group_index].admission_closed = true;
            }
        }
        impl_->authority_transition_pending = true;
        impl_->resource_cv.wait(lock, [&]() {
            if (impl_->active_maintenance != 0 || impl_->active_legacy_operations != 0) {
                return false;
            }
            for (uint32_t record_index = 0; record_index < n_groups; ++record_index) {
                if (!barrier[record_index]) {
                    continue;
                }
                const uint32_t group_index = execution->groups_[record_index].key.candidate.group_index;
                if (impl_->group_authorities[group_index].active_calls != 0 || impl_->group_has_active_transaction(group_index) ||
                        impl_->group_has_active_legacy_lease(group_index)) {
                    return false;
                }
            }
            return true;
        });
        for (uint32_t record_index = 0; record_index < n_groups; ++record_index) {
            if (!barrier[record_index]) {
                continue;
            }
            const uint32_t group_index = execution->groups_[record_index].key.candidate.group_index;
            const auto resource = group_index < impl_->resources.size() ? impl_->resources[group_index] : nullptr;
            const auto current_authority = impl_->group_authorities[group_index].authority;
            if (desired[record_index] == GGML_CUDA_MOE_GROUP_AUTHORITY_LEGACY) {
                impl_->refreshing[group_index] = 0;
                if (resource != nullptr && resource->device == nullptr) {
                    retired_resources[group_index] = std::move(impl_->resources[group_index]);
                } else if (resource != nullptr) {
                    const cudaEvent_t completion = resource->device->has_completion ? resource->device->completion : nullptr;
                    for (const auto & entry : impl_->legacy_records) {
                        const auto & legacy = *entry.second;
                        if (legacy.acquisition.registered_source && legacy.acquisition.group_index == group_index &&
                                legacy.backing == resource && legacy.cache != nullptr) {
                            cache_handoffs.push_back({legacy.cache, completion, nullptr, true});
                        }
                    }
                }
            } else {
                for (auto it = impl_->legacy_records.begin(); it != impl_->legacy_records.end();) {
                    if (it->second->acquisition.registered_source && it->second->acquisition.group_index == group_index) {
                        if (resource != nullptr && resource->device != nullptr && it->second->backing == resource &&
                                it->second->cache != nullptr) {
                            cache_handoffs.push_back({it->second->cache, nullptr, execution->groups_[record_index].stream, false});
                            ++it;
                        } else {
                            retired_legacy.push_back(std::move(it->second));
                            it = impl_->legacy_records.erase(it);
                        }
                    } else {
                        ++it;
                    }
                }
                if (resource != nullptr && resource->device != nullptr &&
                        (resource->legacy_dirty || current_authority == GGML_CUDA_MOE_GROUP_AUTHORITY_GROUPED_HOST_STAGED)) {
                    grouped_resets.push_back({resource, execution->groups_[record_index].stream});
                }
                if (resource != nullptr && resource->device != nullptr &&
                        current_authority != desired[record_index]) {
                    const bool to_host_staged = desired[record_index] == GGML_CUDA_MOE_GROUP_AUTHORITY_GROUPED_HOST_STAGED;
                    const cudaEvent_t completion = resource->device->has_completion ? resource->device->completion : nullptr;
                    for (ggml_cuda_moe_cache * cache : resource->host_staging_caches) {
                        if (cache != nullptr) {
                            host_staging_transitions.push_back({
                                cache, completion, execution->groups_[record_index].stream, to_host_staged});
                        }
                    }
                }
            }
        }
    }

    bool transition_ok = true;
    for (const auto & handoff : cache_handoffs) {
        transition_ok = handoff.to_legacy ?
            ggml_cuda_moe_cache_prepare_legacy(handoff.cache, handoff.grouped_done) :
            ggml_cuda_moe_cache_handoff_grouped(handoff.cache, handoff.grouped_stream);
        if (!transition_ok) {
            break;
        }
    }
    for (const auto & transition : host_staging_transitions) {
        if (!transition_ok || transition.to_host_staged) {
            continue;
        }
        transition_ok = ggml_cuda_moe_cache_handoff_grouped(transition.cache, transition.grouped_stream);
    }
    for (const auto & reset : grouped_resets) {
        if (transition_ok) {
            transition_ok = impl_->cold_reset_grouped_resource(*reset.resource, reset.stream);
        }
    }
    for (const auto & transition : host_staging_transitions) {
        if (!transition_ok) {
            break;
        }
        if (transition.to_host_staged) {
            transition_ok = ggml_cuda_moe_cache_prepare_host_staged(transition.cache, transition.grouped_done);
        }
    }
    impl::retire_resources(std::move(retired_resources));
    impl_->retire_legacy_records(std::move(retired_legacy));

    if (!transition_ok) {
        {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            for (uint32_t record_index = 0; record_index < n_groups; ++record_index) {
                if (barrier[record_index]) {
                    impl_->group_authorities[execution->groups_[record_index].key.candidate.group_index].admission_closed = false;
                }
            }
            impl_->authority_transition_pending = false;
        }
        impl_->resource_cv.notify_all();
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        GGML_ASSERT(!impl_->draining && !impl_->replacement_pending && impl_->state.accepted && generation == impl_->state.generation);
        for (uint32_t record_index = 0; record_index < n_groups; ++record_index) {
            const uint32_t group_index = execution->groups_[record_index].key.candidate.group_index;
            auto & current = impl_->group_authorities[group_index];
            if (current.authority != desired[record_index]) {
                if (debug != nullptr && current.authority != GGML_CUDA_MOE_GROUP_AUTHORITY_LEGACY &&
                        desired[record_index] != GGML_CUDA_MOE_GROUP_AUTHORITY_LEGACY) {
                    debug->strategy_switches.fetch_add(1, std::memory_order_relaxed);
                }
                current.authority = desired[record_index];
                current.epoch = ++impl_->next_group_authority_epoch;
            }
            if (current.authority == GGML_CUDA_MOE_GROUP_AUTHORITY_LEGACY) {
                for (auto & entry : impl_->legacy_records) {
                    auto & legacy = *entry.second;
                    if (legacy.acquisition.registered_source && legacy.acquisition.group_index == group_index) {
                        legacy.acquisition.group_authority_epoch = current.epoch;
                    }
                }
            }
            current.admission_closed = false;
            ++current.active_calls;
            auto & lease = execution->groups_[record_index].authority;
            lease.owner_ = this;
            lease.candidate_generation_ = generation;
            lease.authority_epoch_ = current.epoch;
            lease.group_index_ = group_index;
            lease.authority_ = current.authority;
            lease.execution_domain_ = execution->plan_->execution_certificate_.domain;
            lease.row_semantics_ = execution->plan_->execution_certificate_.row_semantics;
            lease.prefill_resident_certified_ = std::any_of(
                execution->plan_->prefill_add_id_witnesses_.begin(),
                execution->plan_->prefill_add_id_witnesses_.end(),
                [record_index](const ggml_cuda_moe_graph_plan::prefill_add_id_witness & witness) {
                    return witness.group_record == record_index;
                });
            auto & dispatch = execution->groups_[record_index];
            dispatch.transaction = {};
            dispatch.remapped_ids = nullptr;
            std::fill_n(dispatch.bank_data, GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS, nullptr);
            std::fill_n(dispatch.auxiliary_tensors, 3, nullptr);
            std::fill_n(dispatch.auxiliary_data, 3, nullptr);
            std::fill_n(dispatch.auxiliary_roles, 3, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID);
            dispatch.n_slots = 0;
            dispatch.n_auxiliary_shadows = 0;
            dispatch.defer_completion = false;
            dispatch.state = current.authority == GGML_CUDA_MOE_GROUP_AUTHORITY_GROUPED ?
                GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_ARMED :
                current.authority == GGML_CUDA_MOE_GROUP_AUTHORITY_GROUPED_HOST_STAGED ?
                    GGML_CUDA_MOE_GRAPH_GROUP_HOST_STAGED_ARMED : GGML_CUDA_MOE_GRAPH_GROUP_WHOLE_LEGACY;
        }
        impl_->authority_transition_pending = false;
        execution->dispatch_mode_ = mode;
        execution->dispatch_active_ = true;
        record_dispatch();
    }
    impl_->resource_cv.notify_all();
    return true;
}

bool ggml_cuda_moe_grouped_context::prefill_add_id_source(
        const ggml_cuda_moe_graph_execution & execution,
        const ggml_tensor * node,
        ggml_cuda_moe_stream_t stream,
        const float ** source) const {
    if (source != nullptr) {
        *source = nullptr;
    }
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(execution);
    GGML_UNUSED(node);
    GGML_UNUSED(stream);
    return false;
#else
    if (source == nullptr || node == nullptr || stream == nullptr || execution.owner_ != this ||
            execution.plan_ == nullptr || execution.plan_->owner_ != impl_.get() || !execution.dispatch_active_ ||
            (execution.dispatch_mode_ != GGML_CUDA_MOE_GRAPH_DISPATCH_LEGACY &&
                execution.dispatch_mode_ != GGML_CUDA_MOE_GRAPH_DISPATCH_DIRECT) ||
            execution.plan_->outcome_ != GGML_CUDA_MOE_GRAPH_OUTCOME_PREFILL_LEGACY ||
            !execution.plan_->has_certified_complete_mmid_inventory()) {
        return false;
    }

    const ggml_cuda_moe_graph_plan::prefill_add_id_witness * matched = nullptr;
    for (const auto & witness : execution.plan_->prefill_add_id_witnesses_) {
        if (witness.consumer.node == node) {
            matched = &witness;
            break;
        }
    }
    if (matched == nullptr || matched->group_record >= execution.n_groups_) {
        return false;
    }

    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto & record = execution.plan_->groups_[matched->group_record];
    const auto & dispatch = execution.groups_[matched->group_record];
    const uint32_t group_index = record.candidate.group_index;
    if (record.reason != ggml_cuda_moe_graph_plan::GROUP_REASON_PREFILL ||
            record.candidate.generation != impl_->state.generation || group_index >= impl_->table.groups.size() ||
            group_index >= impl_->resources.size() || dispatch.state != GGML_CUDA_MOE_GRAPH_GROUP_WHOLE_LEGACY ||
            !dispatch.authority || dispatch.authority.owner_ != this ||
            dispatch.authority.candidate_generation_ != impl_->state.generation ||
            dispatch.authority.group_index_ != group_index ||
            dispatch.authority.authority_ != GGML_CUDA_MOE_GROUP_AUTHORITY_LEGACY) {
        return false;
    }
    const auto & authority = impl_->group_authorities[group_index];
    if (authority.authority != GGML_CUDA_MOE_GROUP_AUTHORITY_LEGACY || authority.active_calls == 0 ||
            authority.epoch != dispatch.authority.authority_epoch_) {
        return false;
    }
    const auto & group = impl_->table.groups[group_index];
    uint32_t bank_index = UINT32_MAX;
    uint32_t reader_role = GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID;
    if (!group.prefill_resident_auxiliary ||
            !moe_candidate_prefill_add_id_witness(node, group, &bank_index, &reader_role) ||
            bank_index != matched->bank_index || reader_role != matched->reader_role) {
        return false;
    }
    const auto * resource = impl_->resources[group_index].get();
    if (resource == nullptr || resource->device == nullptr || !impl_->resource_matches_table(*resource) ||
            resource->snapshot.acquisition.candidate.generation != impl_->state.generation) {
        return false;
    }
    uint32_t auxiliary_index = 0;
    while (auxiliary_index < resource->snapshot.n_slot_auxiliaries &&
            resource->snapshot.slot_auxiliaries[auxiliary_index].tensor != node->src[1]) {
        ++auxiliary_index;
    }
    if (auxiliary_index == resource->snapshot.n_slot_auxiliaries) {
        return false;
    }
    const auto & descriptor = resource->snapshot.slot_auxiliaries[auxiliary_index];
    auto & device_resource = *resource->device;
    if (!impl_->descriptor_matches(descriptor) || descriptor.role != group.banks[bank_index].info.role ||
            device_resource.prefill_auxiliary_data[auxiliary_index] == nullptr ||
            device_resource.completion == nullptr || !device_resource.has_completion) {
        return false;
    }
    if (device_resource.prefill_auxiliary_pending_node != nullptr) {
        ++device_resource.prefill_auxiliary_pending_declines;
        return false;
    }
    moe_grouped_device_scope device_scope(impl_->device);
    if (device_resource.completion_stream != stream) {
        if (!moe_grouped_cuda_success(cudaStreamWaitEvent(stream, device_resource.completion, 0))) {
            return false;
        }
        ++device_resource.prefill_auxiliary_cross_stream_waits;
    }
    *source = device_resource.prefill_auxiliary_data[auxiliary_index];
    device_resource.prefill_auxiliary_pending_node = node;
    device_resource.prefill_auxiliary_pending_source = *source;
    device_resource.prefill_auxiliary_pending_stream = stream;
    return true;
#endif
}

bool ggml_cuda_moe_grouped_context::original_auxiliary_source(
        const ggml_cuda_moe_graph_execution & execution,
        const ggml_tensor * node,
        ggml_cuda_moe_stream_t stream,
        const float ** source) const {
    if (source == nullptr || node == nullptr || (node->op != GGML_OP_REPEAT && node->op != GGML_OP_ADD_ID)) {
        return false;
    }
    const auto * input = node->src[node->op == GGML_OP_REPEAT ? 0 : 1];
    *source = input != nullptr ? static_cast<const float *>(input->data) : nullptr;
    if (execution.outcome() != GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED) {
        return true;
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!execution.dispatch_active_ || execution.owner_ != this || execution.plan_ == nullptr ||
            execution.plan_->owner_ != impl_.get() || execution.plan_->registry_generation_ != impl_->state.generation) {
        return false;
    }
    for (uint32_t g = 0; g < execution.n_groups_; ++g) {
        const auto & record = execution.plan_->groups_[g];
        for (uint32_t r = 0; r < record.n_readers; ++r) {
            const auto & reader = record.readers[r];
            const bool scale = reader.auxiliary_kind == MOE_CANDIDATE_AUXILIARY_SCALE &&
                reader.n_auxiliary_nodes == 3 && reader.auxiliary_nodes[1] == node;
            const bool bias = reader.auxiliary_kind == MOE_CANDIDATE_AUXILIARY_BIAS &&
                reader.auxiliary_consumer_index < reader.n_consumers &&
                reader.consumers[reader.auxiliary_consumer_index].node == node;
            if (!scale && !bias) {
                continue;
            }
            const auto & dispatch = execution.groups_[g];
            const uint32_t index = record.candidate.group_index;
            if (!dispatch.authority || dispatch.authority.owner_ != this || dispatch.stream != stream ||
                    index >= impl_->resources.size() || impl_->resources[index] == nullptr) {
                return false;
            }
            const auto & resource = *impl_->resources[index];
            if (resource.device == nullptr || !impl_->resource_matches_table(resource) ||
                    reader.auxiliary_bank_index >= impl_->table.groups[index].banks.size()) {
                return false;
            }
            const auto & bank = impl_->table.groups[index].banks[reader.auxiliary_bank_index];
            for (uint32_t i = 0; i < resource.snapshot.n_original_auxiliaries; ++i) {
                const auto & auxiliary = resource.snapshot.original_auxiliaries[i];
                if (auxiliary.tensor != bank.info.tensor || auxiliary.role != bank.info.role) {
                    continue;
                }
                if (input == nullptr || input->type != GGML_TYPE_F32 || input->data != auxiliary.source_data ||
                        ggml_nbytes(input) != auxiliary.byte_extent ||
                        (scale ? input->op != GGML_OP_RESHAPE || input->src[0] != auxiliary.tensor : input != auxiliary.tensor)) {
                    return false;
                }
                if (resource.device->original_auxiliary_data[i] != nullptr) {
                    *source = resource.device->original_auxiliary_data[i];
                }
                return true;
            }
            return false;
        }
    }
    return true;
}

bool ggml_cuda_moe_grouped_context::finish_prefill_add_id(
        const ggml_cuda_moe_graph_execution & execution,
        const ggml_tensor * node,
        ggml_cuda_moe_stream_t stream,
        const float * source) const {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(execution);
    GGML_UNUSED(node);
    GGML_UNUSED(stream);
    GGML_UNUSED(source);
    return false;
#else
    if (execution.owner_ != this || execution.plan_ == nullptr || execution.plan_->owner_ != impl_.get() ||
            node == nullptr || stream == nullptr || source == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    for (uint32_t record_index = 0; record_index < execution.n_groups_; ++record_index) {
        const uint32_t group_index = execution.plan_->groups_[record_index].candidate.group_index;
        if (group_index >= impl_->resources.size() || impl_->resources[group_index] == nullptr ||
                impl_->resources[group_index]->device == nullptr) {
            continue;
        }
        auto & device_resource = *impl_->resources[group_index]->device;
        if (device_resource.prefill_auxiliary_pending_node != node ||
                device_resource.prefill_auxiliary_pending_source != source ||
                device_resource.prefill_auxiliary_pending_stream != stream) {
            continue;
        }
        moe_grouped_device_scope device_scope(impl_->device);
        bool completed = moe_grouped_cuda_success(cudaEventRecord(device_resource.completion, stream));
        if (!completed) {
            completed = moe_grouped_cuda_success(cudaStreamSynchronize(stream));
        } else {
            device_resource.completion_stream = stream;
        }
        device_resource.prefill_auxiliary_pending_node = nullptr;
        device_resource.prefill_auxiliary_pending_source = nullptr;
        device_resource.prefill_auxiliary_pending_stream = nullptr;
        return completed;
    }
    return false;
#endif
}

ggml_cuda_moe_grouped_decode_result ggml_cuda_moe_grouped_context::prepare_host_staged_group(
        ggml_cuda_moe_graph_group_dispatch * group,
        const ggml_cuda_moe_graph_binding & binding,
        const ggml_tensor * node,
        cudaStream_t stream) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(group);
    GGML_UNUSED(binding);
    GGML_UNUSED(node);
    GGML_UNUSED(stream);
    return GGML_CUDA_MOE_GROUPED_DECODE_ERROR;
#else
    auto * debug = impl_->debug_stats();
    const auto fail = [&]() {
        if (debug != nullptr) {
            debug->prepare_error.fetch_add(1, std::memory_order_relaxed);
        }
        return GGML_CUDA_MOE_GROUPED_DECODE_ERROR;
    };
    if (group == nullptr || node == nullptr || stream == nullptr ||
            group->state != GGML_CUDA_MOE_GRAPH_GROUP_HOST_STAGED_ARMED ||
            group->strategy != GGML_CUDA_MOE_EXECUTION_STRATEGY_HOST_STAGED || group->stream != stream ||
            node != group->first_reader || binding.key.candidate.generation != group->key.candidate.generation ||
            binding.key.candidate.group_index != group->key.candidate.group_index ||
            binding.key.execution_semantic_key != group->key.execution_semantic_key ||
            binding.slot_index >= group->key.n_banks || group->capabilities == nullptr) {
        return fail();
    }
    if (group->key.ids.ne[0] <= 0 || group->key.ids.ne[1] <= 0 ||
            group->key.ids.ne[0] > UINT32_MAX || group->key.ids.ne[1] > UINT32_MAX ||
            group->key.ids.ne[0] > UINT32_MAX / group->key.ids.ne[1] ||
            group->key.ids.nb[1] % sizeof(int32_t) != 0 ||
            group->key.ids.nb[1] / sizeof(int32_t) > UINT32_MAX) {
        return fail();
    }
    const uint32_t top_k = static_cast<uint32_t>(group->key.ids.ne[0]);
    const uint32_t n_rows = static_cast<uint32_t>(group->key.ids.ne[1]);
    const uint32_t n_routes = top_k * n_rows;
    const uint32_t row_stride = static_cast<uint32_t>(group->key.ids.nb[1] / sizeof(int32_t));

    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        const auto & lease = group->authority;
        if (lease.owner_ != this || lease.candidate_generation_ != impl_->state.generation ||
                lease.group_index_ != group->key.candidate.group_index ||
                lease.authority_ != GGML_CUDA_MOE_GROUP_AUTHORITY_GROUPED_HOST_STAGED ||
                lease.group_index_ >= impl_->table.groups.size()) {
            return fail();
        }
        const auto & current = impl_->group_authorities[lease.group_index_];
        const auto & candidate = impl_->table.groups[lease.group_index_];
        uint32_t n_slot_banks = 0;
        uint32_t n_scales = 0;
        uint32_t n_biases = 0;
        if (lease.authority_epoch_ != current.epoch ||
                current.authority != GGML_CUDA_MOE_GROUP_AUTHORITY_GROUPED_HOST_STAGED || current.active_calls != 1 ||
                candidate.layout != group->key.layout ||
                !moe_candidate_structural_group(candidate, &n_slot_banks, &n_scales, &n_biases) ||
                n_slot_banks != group->key.n_banks || binding.bank_index >= candidate.banks.size() ||
                candidate.banks[binding.bank_index].info.role != binding.role ||
                candidate.banks[binding.bank_index].slot_index != binding.slot_index ||
                node->op != GGML_OP_MUL_MAT_ID || node->src[0] == nullptr || node->src[2] != group->key.ids.tensor ||
                !moe_candidate_record_matches(candidate.banks[binding.bank_index], node->src[0]) ||
                !moe_candidate_ids_equal(moe_candidate_ids_signature(node->src[2]), group->key.ids)) {
            return fail();
        }
        for (uint32_t bank_index = 0; bank_index < group->key.n_banks; ++bank_index) {
            const auto * bank = moe_candidate_base_slot_bank(candidate, bank_index);
            const auto & capability = group->capabilities[bank_index];
            if (bank == nullptr || capability.strategy != GGML_CUDA_MOE_EXECUTION_STRATEGY_HOST_STAGED ||
                    !moe_candidate_capability_matches(capability, *bank, node,
                        top_k, n_rows, n_routes, row_stride, impl_->state.n_slots, impl_->device) ||
                    !moe_candidate_capability_supported(capability)) {
                return fail();
            }
        }
    }

    if (debug != nullptr) {
        debug->calls.fetch_add(1, std::memory_order_relaxed);
        debug->host_staged_calls.fetch_add(1, std::memory_order_relaxed);
    }

    ggml_cuda_moe_grouped_transaction transaction;
    impl::grouped_resource * resource = nullptr;
    bool resource_missing = false;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        resource = impl_->begin_host_staged(group->key, stream, transaction, resource_missing);
    }
    if (resource == nullptr && resource_missing) {
        ggml_cuda_moe_grouped_acquisition acquisition;
        if (!acquire_group_resources(group->key.candidate, &acquisition)) {
            return fail();
        }
        std::lock_guard<std::mutex> lock(impl_->mutex);
        resource = impl_->begin_host_staged(group->key, stream, transaction, resource_missing);
    }
    if (resource == nullptr) {
        return fail();
    }

    if (resource->device == nullptr) {
        auto prospective = impl_->make_device_resource(resource->snapshot, stream, false, top_k);
        if (prospective == nullptr) {
            (void) end_group_transaction(transaction);
            return fail();
        }
        bool installed = false;
        {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            auto * current = impl_->find_resource(transaction);
            if (current == resource && !impl_->refreshing[transaction.acquisition.candidate.group_index] &&
                    impl_->resource_matches_table(*resource) && impl_->next_device_resource_serial != UINT64_MAX) {
                if (resource->device == nullptr) {
                    prospective->set_serial(++impl_->next_device_resource_serial);
                    resource->device = std::move(prospective);
                }
                installed = true;
            }
        }
        if (!installed) {
            (void) end_group_transaction(transaction);
            return fail();
        }
    }

    std::array<ggml_cuda_moe_cache *, GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS> prospective = {};
    bool caches_ready = true;
    for (uint32_t bank_index = 0; caches_ready && bank_index < resource->snapshot.banks.size(); ++bank_index) {
        if (resource->host_staging_caches[bank_index] != nullptr) {
            continue;
        }
        const auto & bank = resource->snapshot.banks[bank_index];
        const size_t trailing_padding = moe_cache_quantized_source_padding(bank.type, bank.ne[0]);
        cudaEvent_t wait_event = resource->device->has_completion ? resource->device->completion : nullptr;
        prospective[bank_index] = ggml_cuda_moe_cache_init_with_pool(
            impl_->device, bank.expert_stride, trailing_padding, resource->snapshot.n_slots, resource->device->bank_data[bank_index], wait_event, false);
        caches_ready = prospective[bank_index] != nullptr;
        if (caches_ready) {
            caches_ready = ggml_cuda_moe_cache_set_metadata(prospective[bank_index], bank.tensor);
        }
    }
    if (caches_ready) {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        auto * current = impl_->find_resource(transaction);
        caches_ready = current == resource && resource->active_host_staging_leases == 0;
        for (uint32_t bank_index = 0; caches_ready && bank_index < resource->snapshot.banks.size(); ++bank_index) {
            if (prospective[bank_index] != nullptr) {
                caches_ready = resource->host_staging_caches[bank_index] == nullptr;
            }
        }
        if (caches_ready) {
            for (uint32_t bank_index = 0; bank_index < resource->snapshot.banks.size(); ++bank_index) {
                if (prospective[bank_index] != nullptr) {
                    resource->host_staging_caches[bank_index] = prospective[bank_index];
                    prospective[bank_index] = nullptr;
                }
            }
        }
    }
    for (ggml_cuda_moe_cache * cache : prospective) {
        ggml_cuda_moe_cache_free(cache);
    }
    if (!caches_ready) {
        (void) end_group_transaction(transaction);
        return fail();
    }

    group->transaction = transaction;
    group->n_slots = resource->snapshot.n_slots;
    group->state = GGML_CUDA_MOE_GRAPH_GROUP_HOST_STAGED_ACTIVE;
    if (debug != nullptr) {
        const uint32_t group_index = group->key.candidate.group_index;
        debug->ready.fetch_add(1, std::memory_order_relaxed);
        debug->admitted_banks.fetch_add(resource->snapshot.banks.size(), std::memory_order_relaxed);
        if (group_index < debug->ready_by_group.size()) {
            debug->ready_by_group[group_index].fetch_add(1, std::memory_order_relaxed);
        }
    }
    return GGML_CUDA_MOE_GROUPED_DECODE_READY;
#endif
}

ggml_cuda_moe_grouped_host_staging_lease ggml_cuda_moe_grouped_context::acquire_grouped_host_staging(
        ggml_cuda_moe_graph_group_dispatch * group,
        const ggml_cuda_moe_graph_binding & binding,
        const ggml_tensor * node,
        cudaStream_t stream) {
    ggml_cuda_moe_grouped_host_staging_lease result;
    if (group == nullptr || node == nullptr || stream == nullptr ||
            group->state != GGML_CUDA_MOE_GRAPH_GROUP_HOST_STAGED_ACTIVE || group->stream != stream ||
            group->strategy != GGML_CUDA_MOE_EXECUTION_STRATEGY_HOST_STAGED ||
            binding.key.candidate.generation != group->key.candidate.generation ||
            binding.key.candidate.group_index != group->key.candidate.group_index ||
            binding.key.execution_semantic_key != group->key.execution_semantic_key ||
            binding.slot_index >= group->key.n_banks || group->capabilities == nullptr) {
        return result;
    }

    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto * resource = impl_->find_resource(group->transaction);
    const auto & authority = group->authority;
    if (resource == nullptr || resource->active_decode_stream != stream ||
            authority.owner_ != this || authority.authority_ != GGML_CUDA_MOE_GROUP_AUTHORITY_GROUPED_HOST_STAGED ||
            authority.group_index_ != group->key.candidate.group_index ||
            authority.group_index_ >= impl_->table.groups.size() ||
            authority.authority_epoch_ != impl_->group_authorities[authority.group_index_].epoch ||
            impl_->group_authorities[authority.group_index_].authority != GGML_CUDA_MOE_GROUP_AUTHORITY_GROUPED_HOST_STAGED ||
            impl_->group_authorities[authority.group_index_].active_calls != 1 ||
            binding.slot_index >= resource->snapshot.banks.size() ||
            resource->host_staging_caches[binding.slot_index] == nullptr ||
            resource->active_host_staging_leases == UINT32_MAX) {
        return result;
    }
    const auto & bank = resource->snapshot.banks[binding.slot_index];
    const auto & capability = group->capabilities[binding.slot_index];
    if (node->op != GGML_OP_MUL_MAT_ID || node->src[0] != bank.tensor || node->src[2] != group->key.ids.tensor ||
            capability.tensor != bank.tensor || capability.strategy != GGML_CUDA_MOE_EXECUTION_STRATEGY_HOST_STAGED ||
            !moe_candidate_capability_supported(capability)) {
        return result;
    }
    ++resource->active_host_staging_leases;
    result.owner_ = this;
    result.transaction_ = group->transaction;
    result.cache_ = resource->host_staging_caches[binding.slot_index];
    result.capability_ = &capability;
    result.bank_index_ = binding.slot_index;
    return result;
}

void ggml_cuda_moe_grouped_context::release_grouped_host_staging(
        ggml_cuda_moe_grouped_host_staging_lease & lease) noexcept {
    std::array<ggml_cuda_moe_cache *, GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS> caches = {};
    const auto transaction = lease.transaction_;
    cudaStream_t stream = nullptr;
    uint32_t n_caches = 0;
    bool retire = false;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        auto * resource = impl_->find_resource(transaction);
        GGML_ASSERT(lease.owner_ == this);
        if (resource != nullptr && resource->active_host_staging_leases > 0 &&
                lease.bank_index_ < resource->snapshot.banks.size() &&
                resource->host_staging_caches[lease.bank_index_] == lease.cache_) {
            --resource->active_host_staging_leases;
            if (resource->active_host_staging_leases == 0 && resource->retire_host_staging_on_last_release) {
                n_caches = static_cast<uint32_t>(resource->snapshot.banks.size());
                for (uint32_t bank_index = 0; bank_index < n_caches; ++bank_index) {
                    caches[bank_index] = resource->host_staging_caches[bank_index];
                }
                stream = resource->active_decode_stream;
                resource->retire_host_staging_on_last_release = false;
                retire = true;
            }
        }
        lease.owner_ = nullptr;
        lease.transaction_ = {};
        lease.cache_ = nullptr;
        lease.capability_ = nullptr;
        lease.bank_index_ = UINT32_MAX;
        impl_->resource_cv.notify_all();
    }
    bool staging_clean = retire;
    for (uint32_t bank_index = 0; staging_clean && bank_index < n_caches; ++bank_index) {
        staging_clean = ggml_cuda_moe_cache_abort_host_staged(caches[bank_index], stream);
    }
    if (retire) {
        const bool retired = impl_->retire_failed_group_resource(transaction);
        if ((!staging_clean || !retired) && impl_->debug_stats() != nullptr) {
            impl_->debug_stats()->finish_error.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

ggml_cuda_moe_grouped_decode_result ggml_cuda_moe_grouped_context::prepare_graph_group(
        ggml_cuda_moe_graph_group_dispatch * group,
        const ggml_cuda_moe_graph_binding & binding,
        const ggml_tensor * node,
        cudaStream_t stream) {
    auto * debug = impl_->debug_stats();
    const bool armed = group != nullptr && group->state == GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_ARMED;
    const auto record_prepare_error = [&]() {
        if (debug != nullptr && armed) {
            debug->prepare_error.fetch_add(1, std::memory_order_relaxed);
        }
        return GGML_CUDA_MOE_GROUPED_DECODE_ERROR;
    };
    if (group == nullptr || node == nullptr || stream == nullptr || !armed ||
            group->stream != stream || node != group->first_reader || binding.key.candidate.generation != group->key.candidate.generation ||
            binding.key.candidate.group_index != group->key.candidate.group_index || binding.slot_index >= group->key.n_banks ||
            binding.key.execution_semantic_key != group->key.execution_semantic_key ||
            group->capabilities == nullptr) {
        return record_prepare_error();
    }
    std::array<const ggml_tensor *, 4> expected_tensors = {};
    std::array<uint32_t, 4> expected_roles = {};
    std::array<uint32_t, 4> expected_types = {};
    std::array<const ggml_tensor *, 3> expected_auxiliary_tensors = {};
    std::array<uint32_t, 3> expected_auxiliary_roles = {};
    uint32_t expected_n_auxiliary_shadows = 0;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        const auto & lease = group->authority;
        if (lease.owner_ != this || lease.candidate_generation_ != impl_->state.generation ||
                lease.group_index_ != group->key.candidate.group_index || lease.authority_ != GGML_CUDA_MOE_GROUP_AUTHORITY_GROUPED ||
                lease.group_index_ >= impl_->table.groups.size() || group->key.n_banks > 4) {
            return record_prepare_error();
        }
        const auto & current = impl_->group_authorities[lease.group_index_];
        const auto & candidate = impl_->table.groups[lease.group_index_];
        uint32_t n_slot_banks = 0;
        uint32_t n_scales = 0;
        uint32_t n_biases = 0;
        if (lease.authority_epoch_ != current.epoch || current.authority != GGML_CUDA_MOE_GROUP_AUTHORITY_GROUPED ||
                current.active_calls != 1 || candidate.layout != group->key.layout ||
                !moe_candidate_structural_group(candidate, &n_slot_banks, &n_scales, &n_biases) ||
                n_slot_banks != group->key.n_banks ||
                binding.bank_index >= candidate.banks.size() || candidate.banks[binding.bank_index].info.role != binding.role ||
                candidate.banks[binding.bank_index].slot_index != binding.slot_index ||
                node->op != GGML_OP_MUL_MAT_ID || node->src[0] == nullptr || node->src[2] != group->key.ids.tensor ||
                !moe_candidate_record_matches(candidate.banks[binding.bank_index], node->src[0]) ||
                !moe_candidate_ids_equal(moe_candidate_ids_signature(node->src[2]), group->key.ids)) {
            return record_prepare_error();
        }
        for (uint32_t bank_index = 0; bank_index < group->key.n_banks; ++bank_index) {
            const auto * bank_ptr = moe_candidate_base_slot_bank(candidate, bank_index);
            if (bank_ptr == nullptr) {
                return record_prepare_error();
            }
            const auto & bank = *bank_ptr;
            if (bank.info.role != moe_candidate_expected_base_role(candidate.layout, bank_index) ||
                    !moe_candidate_capability_matches(group->capabilities[bank_index], bank, node,
                        static_cast<uint32_t>(group->key.ids.ne[0]), static_cast<uint32_t>(group->key.ids.ne[1]),
                        static_cast<uint32_t>(group->key.ids.ne[0] * group->key.ids.ne[1]),
                        static_cast<uint32_t>(group->key.ids.nb[1] / sizeof(int32_t)), impl_->state.n_slots, impl_->device) ||
                    !moe_candidate_capability_supported(group->capabilities[bank_index])) {
                return record_prepare_error();
            }
            expected_tensors[bank_index] = bank.info.tensor;
            expected_roles[bank_index] = bank.info.role;
            expected_types[bank_index] = bank.info.type;
        }
        expected_n_auxiliary_shadows = n_scales + n_biases;
        for (uint32_t auxiliary_index = 0; auxiliary_index < expected_n_auxiliary_shadows; ++auxiliary_index) {
            const auto * auxiliary = moe_candidate_slot_auxiliary_bank(candidate, auxiliary_index);
            if (auxiliary == nullptr) {
                return record_prepare_error();
            }
            expected_auxiliary_tensors[auxiliary_index] = auxiliary->info.tensor;
            expected_auxiliary_roles[auxiliary_index] = auxiliary->info.role;
        }
    }
    if (debug != nullptr) {
        debug->calls.fetch_add(1, std::memory_order_relaxed);
    }

    ggml_cuda_moe_grouped_decode_acquisition decode;
    const auto result = prepare_decode(group->key, stream, &decode, &group->authority);
    if (result != GGML_CUDA_MOE_GROUPED_DECODE_READY) {
        return record_prepare_error();
    }

    bool valid = decode.layout == group->key.layout && decode.n_banks == group->key.n_banks && decode.n_banks <= 4 &&
        decode.n_slots != 0 && decode.remapped_ids != nullptr &&
        decode.n_auxiliary_shadows == expected_n_auxiliary_shadows;
    std::array<uint8_t, 4> seen_roles = {};
    for (uint32_t bank_index = 0; valid && bank_index < decode.n_banks; ++bank_index) {
        const auto & bank = decode.banks[bank_index];
        valid = bank.bank_index == bank_index && bank.tensor != nullptr && bank.data != nullptr &&
            bank.tensor == expected_tensors[bank_index] && bank.role == expected_roles[bank_index] && bank.type == expected_types[bank_index] &&
            bank.role >= GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT &&
            bank.role <= GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT && bank.type == static_cast<uint32_t>(bank.tensor->type);
        if (valid) {
            const uint32_t role_index = bank.role - GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT;
            valid = !seen_roles[role_index];
            seen_roles[role_index] = 1;
        }
    }
    for (uint32_t auxiliary = 0; valid && auxiliary < decode.n_auxiliary_shadows; ++auxiliary) {
        valid = decode.auxiliary_tensors[auxiliary] == expected_auxiliary_tensors[auxiliary] &&
            decode.auxiliary_data[auxiliary] != nullptr && decode.auxiliary_roles[auxiliary] == expected_auxiliary_roles[auxiliary];
    }
    if (!valid) {
        if (debug != nullptr) {
            debug->prepare_error.fetch_add(1, std::memory_order_relaxed);
        }
        (void) finish_decode(decode, stream);
        group->state = GGML_CUDA_MOE_GRAPH_GROUP_FINISHED;
        return GGML_CUDA_MOE_GROUPED_DECODE_ERROR;
    }

    group->transaction = decode.transaction;
    group->remapped_ids = decode.remapped_ids;
    std::fill_n(group->bank_data, GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS, nullptr);
    for (uint32_t bank_index = 0; bank_index < decode.n_banks; ++bank_index) {
        const auto & bank = decode.banks[bank_index];
        group->bank_data[bank.bank_index] = bank.data;
    }
    std::fill_n(group->auxiliary_tensors, 3, nullptr);
    std::fill_n(group->auxiliary_data, 3, nullptr);
    std::fill_n(group->auxiliary_roles, 3, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID);
    for (uint32_t auxiliary = 0; auxiliary < decode.n_auxiliary_shadows; ++auxiliary) {
        group->auxiliary_tensors[auxiliary] = decode.auxiliary_tensors[auxiliary];
        group->auxiliary_data[auxiliary] = decode.auxiliary_data[auxiliary];
        group->auxiliary_roles[auxiliary] = decode.auxiliary_roles[auxiliary];
    }
    group->n_slots = decode.n_slots;
    group->n_auxiliary_shadows = decode.n_auxiliary_shadows;
    group->state = GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_ACTIVE;
    if (debug != nullptr) {
        const uint32_t group_index = group->key.candidate.group_index;
        debug->ready.fetch_add(1, std::memory_order_relaxed);
        debug->admitted_banks.fetch_add(decode.n_banks, std::memory_order_relaxed);
        if (group_index < debug->ready_by_group.size()) {
            debug->ready_by_group[group_index].fetch_add(1, std::memory_order_relaxed);
        }
    }
    return GGML_CUDA_MOE_GROUPED_DECODE_READY;
}

bool ggml_cuda_moe_grouped_context::finish_graph_group(
        ggml_cuda_moe_graph_group_dispatch * group,
        const ggml_cuda_moe_graph_binding & binding,
        const ggml_tensor * node,
        cudaStream_t stream) {
    auto * debug = impl_->debug_stats();
    const bool active = group != nullptr && group->state == GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_ACTIVE;
    if (group == nullptr || node == nullptr || stream == nullptr || !active ||
            group->stream != stream || node != group->last_reader || binding.role != GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT ||
            binding.key.candidate.generation != group->key.candidate.generation ||
            binding.key.candidate.group_index != group->key.candidate.group_index || node->src[2] != group->key.ids.tensor) {
        if (debug != nullptr) {
            debug->finish_error.fetch_add(1, std::memory_order_relaxed);
        }
        return false;
    }
    if (group->defer_completion) {
        return true;
    }
    if (debug != nullptr) {
        debug->submitted.fetch_add(1, std::memory_order_relaxed);
        if (group->key.candidate.group_index < debug->submitted_by_group.size()) {
            debug->submitted_by_group[group->key.candidate.group_index].fetch_add(1, std::memory_order_relaxed);
        }
    }
    ggml_cuda_moe_grouped_decode_acquisition decode;
    decode.transaction = group->transaction;
    if (!finish_decode(decode, stream)) {
        if (debug != nullptr) {
            debug->finish_error.fetch_add(1, std::memory_order_relaxed);
        }
        return false;
    }
    const uint32_t group_index = group->key.candidate.group_index;
    group->transaction = {};
    group->remapped_ids = nullptr;
    std::fill_n(group->bank_data, GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS, nullptr);
    std::fill_n(group->auxiliary_tensors, 3, nullptr);
    std::fill_n(group->auxiliary_data, 3, nullptr);
    std::fill_n(group->auxiliary_roles, 3, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID);
    group->n_slots = 0;
    group->n_auxiliary_shadows = 0;
    group->state = GGML_CUDA_MOE_GRAPH_GROUP_FINISHED;
    if (debug != nullptr) {
        debug->completed.fetch_add(1, std::memory_order_relaxed);
        if (group_index < debug->completed_by_group.size()) {
            debug->completed_by_group[group_index].fetch_add(1, std::memory_order_relaxed);
        }
    }
    return true;
}

bool ggml_cuda_moe_grouped_context::finish_host_staged_group(
        ggml_cuda_moe_graph_group_dispatch * group,
        const ggml_cuda_moe_graph_binding & binding,
        const ggml_tensor * node,
        cudaStream_t stream) {
    auto * debug = impl_->debug_stats();
    if (group == nullptr || node == nullptr || stream == nullptr ||
            group->state != GGML_CUDA_MOE_GRAPH_GROUP_HOST_STAGED_ACTIVE || group->stream != stream ||
            node != group->last_reader || binding.role != GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT ||
            binding.key.candidate.generation != group->key.candidate.generation ||
            binding.key.candidate.group_index != group->key.candidate.group_index || node->src[2] != group->key.ids.tensor) {
        if (debug != nullptr) {
            debug->finish_error.fetch_add(1, std::memory_order_relaxed);
        }
        return false;
    }

    if (debug != nullptr) {
        debug->submitted.fetch_add(1, std::memory_order_relaxed);
        if (group->key.candidate.group_index < debug->submitted_by_group.size()) {
            debug->submitted_by_group[group->key.candidate.group_index].fetch_add(1, std::memory_order_relaxed);
        }
    }
    bool finished = false;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        auto * resource = impl_->find_resource(group->transaction);
        if (resource != nullptr && resource->device != nullptr && resource->active_decode_stream == stream &&
                resource->active_host_staging_leases == 0 &&
                moe_grouped_cuda_success(cudaEventRecord(resource->device->completion, stream))) {
            resource->device->has_completion = true;
            resource->device->completion_stream = stream;
            resource->active_transaction_token = 0;
            resource->active_decode_stream = nullptr;
            finished = true;
            impl_->resource_cv.notify_all();
        }
    }
    if (!finished) {
        if (debug != nullptr) {
            debug->finish_error.fetch_add(1, std::memory_order_relaxed);
        }
        return false;
    }

    const uint32_t group_index = group->key.candidate.group_index;
    group->transaction = {};
    group->n_slots = 0;
    group->state = GGML_CUDA_MOE_GRAPH_GROUP_FINISHED;
    if (debug != nullptr) {
        debug->completed.fetch_add(1, std::memory_order_relaxed);
        if (group_index < debug->completed_by_group.size()) {
            debug->completed_by_group[group_index].fetch_add(1, std::memory_order_relaxed);
        }
    }
    return true;
}

bool ggml_cuda_moe_grouped_context::finish_graph_dispatch(ggml_cuda_moe_graph_execution * execution) {
    if (execution == nullptr || execution->owner_ != this || !execution->dispatch_active_) {
        return false;
    }
    auto * debug = impl_->debug_stats();
    bool success = true;
    for (uint32_t record_index = 0; record_index < execution->n_groups_; ++record_index) {
        auto & group = execution->groups_[record_index];
        const bool grouped = group.authority.authority() == GGML_CUDA_MOE_GROUP_AUTHORITY_GROUPED;
        const bool host_staged = group.authority.authority() == GGML_CUDA_MOE_GROUP_AUTHORITY_GROUPED_HOST_STAGED;
        if (grouped && ((group.state == GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_ACTIVE && group.defer_completion) ||
                group.state == GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_REPLAY)) {
            if (debug != nullptr) {
                debug->submitted.fetch_add(1, std::memory_order_relaxed);
                if (group.key.candidate.group_index < debug->submitted_by_group.size()) {
                    debug->submitted_by_group[group.key.candidate.group_index].fetch_add(1, std::memory_order_relaxed);
                }
            }
            ggml_cuda_moe_grouped_decode_acquisition decode;
            decode.transaction = group.transaction;
            if (!finish_decode(decode, group.stream)) {
                success = false;
                if (debug != nullptr) {
                    debug->finish_error.fetch_add(1, std::memory_order_relaxed);
                }
                (void) impl_->retire_failed_group_resource(group.transaction);
            } else if (debug != nullptr) {
                const uint32_t group_index = group.key.candidate.group_index;
                debug->completed.fetch_add(1, std::memory_order_relaxed);
                if (group_index < debug->completed_by_group.size()) {
                    debug->completed_by_group[group_index].fetch_add(1, std::memory_order_relaxed);
                }
            }
        } else if (grouped && group.state == GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_ACTIVE) {
            success = false;
            if (debug != nullptr) {
                debug->finish_error.fetch_add(1, std::memory_order_relaxed);
            }
            ggml_cuda_moe_grouped_decode_acquisition decode;
            decode.transaction = group.transaction;
            if (!finish_decode(decode, group.stream)) {
                (void) impl_->retire_failed_group_resource(group.transaction);
            }
        } else if (grouped && group.state != GGML_CUDA_MOE_GRAPH_GROUP_FINISHED) {
            success = false;
            if (debug != nullptr) {
                debug->finish_error.fetch_add(1, std::memory_order_relaxed);
            }
        } else if (host_staged && group.state != GGML_CUDA_MOE_GRAPH_GROUP_FINISHED) {
            success = false;
            if (debug != nullptr) {
                debug->finish_error.fetch_add(1, std::memory_order_relaxed);
                debug->required_unsupported.fetch_add(1, std::memory_order_relaxed);
            }
            if (group.state == GGML_CUDA_MOE_GRAPH_GROUP_HOST_STAGED_ACTIVE && group.transaction.transaction_token != 0) {
                std::array<ggml_cuda_moe_cache *, GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS> caches = {};
                uint32_t n_caches = 0;
                bool cleanup_ready = false;
                {
                    std::lock_guard<std::mutex> lock(impl_->mutex);
                    auto * resource = impl_->find_resource(group.transaction);
                    if (resource != nullptr) {
                        if (resource->active_host_staging_leases == 0) {
                            n_caches = static_cast<uint32_t>(resource->snapshot.banks.size());
                            for (uint32_t bank_index = 0; bank_index < n_caches; ++bank_index) {
                                caches[bank_index] = resource->host_staging_caches[bank_index];
                            }
                            cleanup_ready = true;
                        } else {
                            resource->retire_host_staging_on_last_release = true;
                        }
                    }
                }
                bool staging_clean = cleanup_ready;
                for (uint32_t bank_index = 0; staging_clean && bank_index < n_caches; ++bank_index) {
                    staging_clean = ggml_cuda_moe_cache_abort_host_staged(caches[bank_index], group.stream);
                }
                const bool retired = cleanup_ready && impl_->retire_failed_group_resource(group.transaction);
                if ((cleanup_ready && (!staging_clean || !retired)) && debug != nullptr) {
                    debug->finish_error.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
        // Original-ID consumers can follow the final MMID on this stream.
        if ((grouped || host_staged) && group.state == GGML_CUDA_MOE_GRAPH_GROUP_FINISHED) {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            const uint32_t index = group.key.candidate.group_index;
            if (index < impl_->resources.size() && impl_->resources[index] != nullptr &&
                    impl_->resources[index]->device != nullptr) {
                auto & device = *impl_->resources[index]->device;
                if (device.original_auxiliary_bytes != 0) {
                    moe_grouped_device_scope device_scope(impl_->device);
                    if (moe_grouped_cuda_success(cudaEventRecord(device.completion, group.stream))) {
                        device.has_completion = true;
                        device.completion_stream = group.stream;
                    } else {
                        (void) cudaStreamSynchronize(group.stream);
                        success = false;
                    }
                }
            }
        }
        group.transaction = {};
        group.remapped_ids = nullptr;
        std::fill_n(group.bank_data, GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS, nullptr);
        std::fill_n(group.auxiliary_tensors, 3, nullptr);
        std::fill_n(group.auxiliary_data, 3, nullptr);
        std::fill_n(group.auxiliary_roles, 3, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID);
        group.n_slots = 0;
        group.n_auxiliary_shadows = 0;
        group.defer_completion = false;
        group.state = GGML_CUDA_MOE_GRAPH_GROUP_FINISHED;
    }
    execution->dispatch_active_ = false;
    execution->dispatch_mode_ = GGML_CUDA_MOE_GRAPH_DISPATCH_LEGACY;
    for (uint32_t record_index = 0; record_index < execution->n_groups_; ++record_index) {
        execution->groups_[record_index].authority = {};
    }
    return success;
}

void ggml_cuda_moe_grouped_context::end_group_call(ggml_cuda_moe_group_call_lease & lease) noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    GGML_ASSERT(lease.owner_ == this && lease.group_index_ < impl_->table.groups.size());
    auto & current = impl_->group_authorities[lease.group_index_];
    GGML_ASSERT(lease.candidate_generation_ == impl_->state.generation && lease.authority_epoch_ == current.epoch &&
        lease.authority_ == current.authority && current.active_calls > 0);
    --current.active_calls;
    lease.owner_ = nullptr;
    impl_->resource_cv.notify_all();
}

void ggml_cuda_moe_grouped_context::shutdown() {
    bool terminal_owner = false;
    uint32_t registered_groups = 0;
    std::unique_lock<std::mutex> lifecycle_lock(impl_->resource_lifecycle_mutex);
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (!impl_->draining) {
            impl_->draining = true;
            terminal_owner = true;
        }
    }
    if (!terminal_owner) {
        lifecycle_lock.unlock();
        std::unique_lock<std::mutex> lock(impl_->mutex);
        impl_->resource_cv.wait(lock, [&]() { return impl_->shutdown_complete; });
        return;
    }

    impl::legacy_record_map retired_legacy;
    impl::resource_slots retired;
    {
        std::unique_lock<std::mutex> lock(impl_->mutex);
        impl_->resource_cv.wait(lock, [&]() {
            return !impl_->has_active_transaction() && impl_->active_maintenance == 0 &&
                impl_->active_legacy_operations == 0 && !impl_->has_active_legacy_lease() && !impl_->has_active_group_call();
        });
        if (impl_->host_prepack != nullptr) {
            impl_->host_prepack->close_admission();
        }
        retired = impl_->detach_resources();
        retired_legacy = impl_->detach_legacy_records();
        impl_->refreshing.fill(0);
        impl_->group_authorities = {};
        impl_->authority_transition_pending = false;
        registered_groups = impl_->state.accepted ? static_cast<uint32_t>(impl_->table.groups.size()) : 0;
        impl_->draining_registered_groups = registered_groups;
        impl_->table = {};
        impl_->graph_coverages.clear();
        const uint64_t generation = impl_->state.generation;
        impl_->state = {};
        impl_->state.generation = generation;
    }
    if (!impl_->early.empty() || impl_->host_prepack != nullptr) {
        moe_grouped_device_scope device_scope(impl_->device);
        for (const auto & resource : retired) {
            if (resource != nullptr && resource->device != nullptr && resource->device->has_completion) {
                CUDA_CHECK(cudaEventSynchronize(resource->device->completion));
            }
        }
        impl_->clear_early();
    }
    impl::retire_resources(std::move(retired));
    impl_->retire_legacy_records(std::move(retired_legacy));
    impl_->pageable_sources.reset();
    lifecycle_lock.unlock();
    {
        auto & telemetry = moe_cache_owner_telemetry_state();
        std::lock_guard<std::mutex> lock(telemetry.mutex);
        moe_cache_telemetry final_telemetry = impl_->take_owner_telemetry(registered_groups);
        telemetry.active.erase(this);
        try {
            moe_cache_add_telemetry(telemetry.retired, std::move(final_telemetry));
        } catch (...) {
        }
    }
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->shutdown_complete = true;
    }
    impl_->resource_cv.notify_all();
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
            ctx->moe_grouped_context = new ggml_cuda_moe_grouped_context(backend->device, ctx->device);
        });
    } catch (...) {
        return GGML_BACKEND_MOE_CANDIDATE_REPLACE_ERROR;
    }
    return ctx->moe_grouped_context->replace(snapshot);
}

extern "C"
int32_t ggml_backend_cuda_moe_candidate_replace_v2(
        ggml_backend_t backend,
        const ggml_backend_moe_candidate_snapshot_v2 * snapshot) {
    if (!ggml_backend_is_cuda(backend) || backend->context == nullptr || backend->device == nullptr) {
        return GGML_BACKEND_MOE_CANDIDATE_REPLACE_INVALID_ARGUMENT;
    }

    auto * ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    try {
        std::call_once(ctx->moe_grouped_context_once, [&]() {
            ctx->moe_grouped_context = new ggml_cuda_moe_grouped_context(backend->device, ctx->device);
        });
    } catch (...) {
        return GGML_BACKEND_MOE_CANDIDATE_REPLACE_ERROR;
    }
    return ctx->moe_grouped_context->replace(snapshot);
}

ggml_cuda_moe_grouped_context * ggml_cuda_moe_grouped_context_for_test(ggml_backend_t backend) {
    if (!ggml_backend_is_cuda(backend) || backend->context == nullptr) {
        return nullptr;
    }
    return static_cast<ggml_backend_cuda_context *>(backend->context)->moe_grouped_context;
}

struct ggml_cuda_moe_cache {
    int      device;
    size_t   slot_size_bytes;
    size_t   trailing_padding_bytes;
    int      n_slots;

    void *   slot_pool_d;            // device alloc, slots followed by trailing padding
    bool     owns_slot_pool;

    // Dedicated copy stream for cache fills, prefetches, and staging copies.
    cudaStream_t copy_stream;
    cudaEvent_t  compute_done;
    cudaEvent_t  stage_done;
    cudaEvent_t  handoff_done;
    cudaStream_t last_compute_stream;
    bool         has_compute_done;
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
    std::atomic<uint64_t> phase_upload_calls[2];
    std::atomic<uint64_t> phase_upload_bytes[2];
    std::atomic<uint64_t> phase_staging_tile_count[2];
    std::atomic<uint64_t> phase_pipeline_tiles[2];
    std::atomic<uint64_t> phase_tile_wait_calls[2];
    std::atomic<uint64_t> phase_tile_wait_time_us[2];
    std::atomic<uint64_t> phase_cpu_pack_bytes[2];
    std::atomic<uint64_t> phase_cpu_pack_time_us[2];
    std::atomic<uint64_t> phase_pre_sync_calls[2];
    std::atomic<uint64_t> phase_pre_sync_time_us[2];
    std::atomic<uint64_t> phase_h2d_submit_calls[2];
    std::atomic<uint64_t> phase_h2d_submit_bytes[2];
    std::atomic<uint64_t> phase_h2d_submit_time_us[2];
    std::atomic<uint64_t> phase_post_sync_calls[2];
    std::atomic<uint64_t> phase_post_sync_time_us[2];
    std::atomic<uint64_t> phase_upload_errors[2];
    std::atomic<uint64_t> sampled_mincore_checks{0};
    std::atomic<uint64_t> sampled_pages_total{0};
    std::atomic<uint64_t> sampled_pages_resident{0};
    std::atomic<uint64_t> sampled_nonresident_expert_count{0};
    std::atomic<uint64_t> mincore_failures{0};

    bool debug_mm = false;

    const moe_host_source * source = nullptr;
    std::unique_ptr<moe_host_allocation> host_staging;
    size_t host_tile_stride = 0;
    cudaEvent_t host_tile_done[2] = {};
    bool host_tile_pending[2] = {};
    uint32_t next_host_tile = 0;
    bool host_staging_failed = false;
    bool staging_pipeline_enabled = true;
    ggml_cuda_moe_staging_failure_for_test staging_failure_for_test = GGML_CUDA_MOE_STAGING_FAIL_NONE;

    std::string tensor_name;
    const void * tensor_data = nullptr;
    int64_t n_experts = 0;
    uint64_t expert_access_counter = 0;
    std::vector<uint64_t> expert_access_counts;
    std::vector<uint64_t> expert_last_access;
    uint64_t expert_first_touches = 0;
    uint64_t expert_reuse_le_l1 = 0;
    uint64_t expert_reuse_gt_l1 = 0;
    uint64_t phase_expert_access_counter[2] = {};
    std::vector<uint64_t> phase_expert_access_counts[2];
    std::vector<uint64_t> phase_expert_last_access[2];
    uint64_t phase_expert_first_touches[2] = {};
    uint64_t phase_expert_reuse_le_l1[2] = {};
    uint64_t phase_expert_reuse_gt_l1[2] = {};
    moe_cache_reuse_hist phase_expert_reuse_hist[2];
};

static bool ggml_cuda_moe_cache_set_metadata(ggml_cuda_moe_cache * cache, const ggml_tensor * tensor) {
    cache->tensor_name = tensor->name[0] ? tensor->name : "?";
    cache->tensor_data = tensor->data;
    cache->n_experts = tensor->ne[2];
    cache->source = moe_host_source_for(tensor);
    if (cache->source != nullptr) {
        cache->source->owner->retain();
        if (cache->source->device_alias == nullptr) {
            cache->host_tile_stride = cache->slot_size_bytes;
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
            constexpr uint32_t max_host_tiles = 2;
#else
            constexpr uint32_t max_host_tiles = 1;
#endif
            cache->host_staging = std::make_unique<moe_host_allocation>(cache->source->owner, cache->host_tile_stride, cache->host_tile_stride, max_host_tiles, false,
                cache->staging_failure_for_test == GGML_CUDA_MOE_STAGING_FAIL_GROWTH);
            if (cache->host_staging->data != nullptr && cache->host_staging->tiles == 2) {
                moe_grouped_device_scope device_scope(cache->device);
                for (uint32_t tile = 0; tile < 2; ++tile) {
                    const cudaError_t error = tile == 1 && cache->staging_failure_for_test == GGML_CUDA_MOE_STAGING_FAIL_EVENT_CREATE ? cudaErrorMemoryAllocation :
                        cudaEventCreateWithFlags(&cache->host_tile_done[tile], cudaEventDisableTiming);
                    if (error != cudaSuccess) {
                        (void) cudaGetLastError();
                        for (auto & event : cache->host_tile_done) {
                            if (event != nullptr) {
                                CUDA_CHECK(cudaEventDestroy(event));
                                event = nullptr;
                            }
                        }
                        cache->host_staging.reset();
                        cache->host_staging = std::make_unique<moe_host_allocation>(cache->source->owner, cache->host_tile_stride);
                        break;
                    }
                }
            }
            cache->staging_failure_for_test = GGML_CUDA_MOE_STAGING_FAIL_NONE;
            return cache->host_staging->data != nullptr;
        }
    } else if (moe_host_budget_for(tensor->buffer->buft) != nullptr) {
        return false;
    }
    return true;
}

static cudaError_t moe_cache_wait_host_tile(ggml_cuda_moe_cache * cache, uint32_t tile, int phase) {
    if (!cache->host_tile_pending[tile]) {
        return cudaSuccess;
    }
    const int64_t start_us = cache->debug_mm ? ggml_time_us() : 0;
    const cudaError_t error = cache->staging_failure_for_test == GGML_CUDA_MOE_STAGING_FAIL_WAIT ? cudaErrorUnknown :
        cudaEventSynchronize(cache->host_tile_done[tile]);
    if (cache->debug_mm) {
        cache->phase_tile_wait_calls[phase].fetch_add(1, std::memory_order_relaxed);
        cache->phase_tile_wait_time_us[phase].fetch_add((uint64_t) (ggml_time_us() - start_us), std::memory_order_relaxed);
    }
    if (error == cudaSuccess) {
        cache->host_tile_pending[tile] = false;
    }
    return error;
}

static cudaError_t moe_cache_copy_source(
        ggml_cuda_moe_cache * cache,
        void * destination,
        const void * source,
        size_t bytes,
        cudaStream_t stream,
        bool is_decode) {
    const int phase = moe_cache_phase_index(is_decode);
    const bool debug_mm = cache->debug_mm;
    if (debug_mm) {
        cache->phase_upload_calls[phase].fetch_add(1, std::memory_order_relaxed);
        cache->phase_upload_bytes[phase].fetch_add(bytes, std::memory_order_relaxed);
    }
    auto failed = [&](const char * reason, cudaError_t error, size_t offset = 0, size_t count = 0) {
        if (debug_mm) {
            cache->phase_upload_errors[phase].fetch_add(1, std::memory_order_relaxed);
        }
        GGML_LOG_ERROR("moe-cache-copy: tensor=%s reason=%s residence=%s bytes=%zu copy_offset=%zu copy_bytes=%zu cuda=%s\n",
            cache->tensor_name.c_str(), reason, cache->source == nullptr ? "unbounded" : cache->source->device_alias != nullptr ? "direct_registered" : "pageable_staged",
            bytes, offset, count, cudaGetErrorString(error));
        return error;
    };
    if (cache->source != nullptr && !moe_host_contains(*cache->source, source, bytes)) {
        return failed("source_range", cudaErrorInvalidValue);
    }
    if (cache->source == nullptr) {
        const int64_t submit_start_us = debug_mm ? ggml_time_us() : 0;
        const cudaError_t error = cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, stream);
        if (debug_mm) {
            cache->phase_h2d_submit_calls[phase].fetch_add(1, std::memory_order_relaxed);
            cache->phase_h2d_submit_bytes[phase].fetch_add(bytes, std::memory_order_relaxed);
            cache->phase_h2d_submit_time_us[phase].fetch_add(
                (uint64_t) (ggml_time_us() - submit_start_us), std::memory_order_relaxed);
        }
        return error == cudaSuccess ? error : failed("direct_memcpy_async", error, 0, bytes);
    }
    if (cache->source->device_alias != nullptr) {
        const auto & ranges = cache->source->owner->registered;
        const uintptr_t address = reinterpret_cast<uintptr_t>(source);
        const auto first = std::upper_bound(ranges.begin(), ranges.end(), address,
            [](uintptr_t value, const moe_host_range & range) { return value < range.end; });
        size_t covered = 0;
        for (auto it = first; covered < bytes; ++it) {
            const uintptr_t current = address + covered;
            if (it == ranges.end() || current < it->begin || current >= it->end) {
                return failed("registration_gap", cudaErrorInvalidValue, covered, bytes - covered);
            }
            covered += std::min(bytes - covered, static_cast<size_t>(it->end - current));
        }
        // Each CUDA copy must stay inside one registration.
        size_t offset = 0;
        for (auto it = first; offset < bytes; ++it) {
            const size_t count = std::min(bytes - offset, static_cast<size_t>(it->end - (address + offset)));
            const int64_t submit_start_us = debug_mm ? ggml_time_us() : 0;
            const cudaError_t error = cudaMemcpyAsync(static_cast<char *>(destination) + offset,
                static_cast<const char *>(source) + offset, count, cudaMemcpyHostToDevice, stream);
            if (debug_mm) {
                cache->phase_h2d_submit_calls[phase].fetch_add(1, std::memory_order_relaxed);
                cache->phase_h2d_submit_bytes[phase].fetch_add(count, std::memory_order_relaxed);
                cache->phase_h2d_submit_time_us[phase].fetch_add(
                    (uint64_t) (ggml_time_us() - submit_start_us), std::memory_order_relaxed);
            }
            if (error != cudaSuccess) {
                return failed("direct_memcpy_async", error, offset, count);
            }
            offset += count;
        }
        return cudaSuccess;
    }
    if (cache->host_staging == nullptr) {
        return failed("staging_absent", cudaErrorInvalidValue);
    }
    if (cache->host_staging_failed) {
        return failed("staging_retired", cudaErrorUnknown);
    }
    auto & staging = *cache->host_staging;
    cudaError_t error = cudaSuccess;
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    error = cudaStreamIsCapturing(stream, &capture);
    if (error == cudaSuccess && capture == cudaStreamCaptureStatusNone && stream != cache->copy_stream) {
        error = cudaStreamIsCapturing(cache->copy_stream, &capture);
    }
    if (error != cudaSuccess || capture != cudaStreamCaptureStatusNone) {
        return failed("staging_capture", error != cudaSuccess ? error : cudaErrorStreamCaptureUnsupported);
    }
#endif
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    const bool pipeline = cache->staging_pipeline_enabled && !is_decode && stream == cache->copy_stream && cache->host_tile_done[1] != nullptr;
#else
    const bool pipeline = false;
#endif
    auto staging_failed = [&](const char * reason, cudaError_t failure, size_t offset = 0, size_t count = 0) {
        // A failed reuse wait can report an error from any earlier fill.
        cudaError_t drain_error = cudaStreamSynchronize(cache->copy_stream);
        if (stream != cache->copy_stream) {
            const cudaError_t alternate_error = cudaStreamSynchronize(stream);
            if (drain_error == cudaSuccess) {
                drain_error = alternate_error;
            }
        }
        cache->host_staging_failed = true;
        cache->host_to_slot.clear();
        std::fill(cache->slot_to_host.begin(), cache->slot_to_host.end(), nullptr);
        if (drain_error == cudaSuccess) {
            std::fill(std::begin(cache->host_tile_pending), std::end(cache->host_tile_pending), false);
        }
        return failed(reason, drain_error == cudaSuccess ? failure : drain_error, offset, count);
    };
    if (!pipeline) {
        for (uint32_t tile = 0; tile < 2; ++tile) {
            error = moe_cache_wait_host_tile(cache, tile, phase);
            if (error != cudaSuccess) {
                return staging_failed("staging_tile_drain", error);
            }
        }
        const int64_t pre_sync_start_us = debug_mm ? ggml_time_us() : 0;
        error = cudaStreamSynchronize(stream);
        if (debug_mm) {
            cache->phase_pre_sync_calls[phase].fetch_add(1, std::memory_order_relaxed);
            cache->phase_pre_sync_time_us[phase].fetch_add(
                (uint64_t) (ggml_time_us() - pre_sync_start_us), std::memory_order_relaxed);
        }
        if (error != cudaSuccess) {
            return staging_failed("staging_pre_sync", error);
        }
    }
    for (size_t offset = 0; error == cudaSuccess && offset < bytes;) {
        const uint32_t tile = pipeline ? cache->next_host_tile : 0;
        if (pipeline) {
            error = moe_cache_wait_host_tile(cache, tile, phase);
            if (error != cudaSuccess) {
                return staging_failed("staging_tile_reuse", error, offset);
            }
        }
        const size_t count = std::min(cache->host_tile_stride, bytes - offset);
        void * tile_data = static_cast<char *>(staging.data) + tile * cache->host_tile_stride;
        const int64_t pack_start_us = debug_mm ? ggml_time_us() : 0;
        const bool materialized = moe_host_materialize(
            *cache->source, static_cast<const char *>(source) + offset, count, tile_data);
        if (debug_mm) {
            cache->phase_staging_tile_count[phase].fetch_add(1, std::memory_order_relaxed);
            cache->phase_cpu_pack_bytes[phase].fetch_add(count, std::memory_order_relaxed);
            cache->phase_cpu_pack_time_us[phase].fetch_add(
                (uint64_t) (ggml_time_us() - pack_start_us), std::memory_order_relaxed);
        }
        if (!materialized) {
            return failed("staging_materialize", cudaErrorInvalidValue, offset, count);
        }
        const int64_t submit_start_us = debug_mm ? ggml_time_us() : 0;
        error = cache->staging_failure_for_test == GGML_CUDA_MOE_STAGING_FAIL_SUBMIT ? cudaErrorUnknown :
            cudaMemcpyAsync(static_cast<char *>(destination) + offset, tile_data, count, cudaMemcpyHostToDevice, stream);
        if (debug_mm) {
            cache->phase_h2d_submit_calls[phase].fetch_add(1, std::memory_order_relaxed);
            cache->phase_h2d_submit_bytes[phase].fetch_add(count, std::memory_order_relaxed);
            cache->phase_h2d_submit_time_us[phase].fetch_add(
                (uint64_t) (ggml_time_us() - submit_start_us), std::memory_order_relaxed);
        }
        if (error != cudaSuccess) {
            return staging_failed("staged_memcpy_async", error, offset, count);
        }
        if (pipeline) {
            error = cache->staging_failure_for_test == GGML_CUDA_MOE_STAGING_FAIL_RECORD ? cudaErrorUnknown :
                cudaEventRecord(cache->host_tile_done[tile], stream);
            if (error != cudaSuccess) {
                return staging_failed("staging_tile_record", error, offset, count);
            }
            cache->host_tile_pending[tile] = true;
            cache->next_host_tile = 1 - tile;
            if (debug_mm) {
                cache->phase_pipeline_tiles[phase].fetch_add(1, std::memory_order_relaxed);
            }
            offset += count;
            continue;
        }
        const int64_t post_sync_start_us = debug_mm ? ggml_time_us() : 0;
        error = cudaStreamSynchronize(stream);
        if (debug_mm) {
            cache->phase_post_sync_calls[phase].fetch_add(1, std::memory_order_relaxed);
            cache->phase_post_sync_time_us[phase].fetch_add(
                (uint64_t) (ggml_time_us() - post_sync_start_us), std::memory_order_relaxed);
        }
        if (error != cudaSuccess) {
            return staging_failed("staging_post_sync", error, offset, count);
        }
        offset += count;
    }
    return error;
}

bool ggml_cuda_moe_cache_set_source_for_test(ggml_cuda_moe_cache * cache, const ggml_tensor * tensor) {
    std::lock_guard<std::mutex> lock(cache->mu);
    return cache->source == nullptr && ggml_cuda_moe_cache_set_metadata(cache, tensor);
}

void ggml_cuda_moe_cache_fail_staging_for_test(ggml_cuda_moe_cache * cache, ggml_cuda_moe_staging_failure_for_test failure) {
    std::lock_guard<std::mutex> lock(cache->mu);
    cache->staging_failure_for_test = failure;
}

ggml_cuda_moe_staging_state_for_test ggml_cuda_moe_cache_staging_state_for_test(ggml_cuda_moe_cache * cache, bool is_decode) {
    std::lock_guard<std::mutex> lock(cache->mu);
    ggml_cuda_moe_staging_state_for_test result = {};
    const int phase = moe_cache_phase_index(is_decode);
    result.tiles = cache->host_staging == nullptr ? 0 : cache->host_staging->tiles;
    result.pending = cache->host_tile_pending[0] + cache->host_tile_pending[1];
    result.failed = cache->host_staging_failed;
    result.pipeline_tiles = cache->phase_pipeline_tiles[phase].load(std::memory_order_relaxed);
    result.tile_wait_calls = cache->phase_tile_wait_calls[phase].load(std::memory_order_relaxed);
    result.pre_sync_calls = cache->phase_pre_sync_calls[phase].load(std::memory_order_relaxed);
    result.post_sync_calls = cache->phase_post_sync_calls[phase].load(std::memory_order_relaxed);
    result.upload_errors = cache->phase_upload_errors[phase].load(std::memory_order_relaxed);
    if (cache->source != nullptr) {
        auto & owner = *cache->source->owner;
        std::lock_guard<std::mutex> budget_lock(owner.mutex);
        result.host_limit = owner.limit;
        result.host_source_bytes = owner.source_bytes;
        result.host_staging_bytes = owner.staging_bytes;
        result.host_optional_bytes = owner.staging_optional_bytes;
    }
    return result;
}

static void ggml_cuda_moe_cache_append_expert_counts(
        const struct ggml_cuda_moe_cache * cache,
        std::vector<uint64_t> & counts) {
    if (!cache || cache->expert_access_counts.empty()) {
        return;
    }

    counts.insert(counts.end(), cache->expert_access_counts.begin(), cache->expert_access_counts.end());
}

static ggml_cuda_moe_cache * ggml_cuda_moe_cache_init_with_pool(
        int device,
        size_t slot_size_bytes,
        size_t trailing_padding_bytes,
        int n_slots,
        void * slot_pool_d,
        cudaEvent_t wait_event,
        bool fail_after_stream_probe) {

    if (slot_size_bytes == 0 || n_slots <= 0 ||
            slot_size_bytes > (SIZE_MAX - trailing_padding_bytes) / (size_t) n_slots) {
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

    struct device_restore {
        int previous;
        ~device_restore() {
            (void) cudaSetDevice(previous);
        }
    } restore{prev_device};
    std::unique_ptr<ggml_cuda_moe_cache, decltype(&ggml_cuda_moe_cache_free)> owner(
        new ggml_cuda_moe_cache{}, ggml_cuda_moe_cache_free);
    auto * c = owner.get();
    c->device          = device;
    c->slot_size_bytes = slot_size_bytes;
    c->trailing_padding_bytes = trailing_padding_bytes;
    c->n_slots         = n_slots;
    c->slot_pool_d     = slot_pool_d;
    c->owns_slot_pool  = slot_pool_d == nullptr;
    c->copy_stream     = nullptr;
    c->compute_done    = nullptr;
    c->stage_done      = nullptr;
    c->handoff_done    = nullptr;
    c->last_compute_stream = nullptr;
    c->has_compute_done = false;
    c->stream_mem_ops_supported = false;
    c->access_counter  = 0;
    c->debug_mm        = moe_cache_mm_debug_enabled();
    c->staging_pipeline_enabled = true;

    if (c->owns_slot_pool) {
        const size_t allocation_size = (size_t) n_slots * slot_size_bytes + trailing_padding_bytes;
        err = cudaMalloc(&c->slot_pool_d, allocation_size);
        if (err != cudaSuccess) {
            fprintf(stderr, "moe-cache: cudaMalloc(%zu bytes) failed: %s\n",
                    allocation_size, cudaGetErrorString(err));
            return nullptr;
        }
        if (trailing_padding_bytes > 0) {
            err = cudaMemset(
                (char *) c->slot_pool_d + (size_t) n_slots * slot_size_bytes, 0, trailing_padding_bytes);
            if (err != cudaSuccess) {
                return nullptr;
            }
        }
    }

    err = cudaStreamCreateWithFlags(&c->copy_stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        fprintf(stderr, "moe-cache: cudaStreamCreate failed: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    if (wait_event != nullptr && !moe_grouped_cuda_success(cudaStreamWaitEvent(c->copy_stream, wait_event, 0))) {
        return nullptr;
    }

// Use event staging on Windows; a failed host-mapped memop can poison the CUDA context.
#if !defined(_WIN32) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && !defined(GGML_CUDA_NO_VMM) && CUDART_VERSION >= 12080
    bool can_use_stream_mem_ops = true;
#if CUDA_VERSION < 13000
    CUdevice cu_device;
    int stream_mem_ops_attribute = 0;
    can_use_stream_mem_ops =
        cuDeviceGet(&cu_device, device) == CUDA_SUCCESS &&
        cuDeviceGetAttribute(
            &stream_mem_ops_attribute, CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1, cu_device) == CUDA_SUCCESS &&
        stream_mem_ops_attribute != 0;
#endif
    uint32_t * stream_mem_probe = nullptr;
    cudaStream_t probe_stream = nullptr;
    cudaGraph_t probe_graph = nullptr;
    cudaGraphExec_t probe_exec = nullptr;
    bool probe_ok = can_use_stream_mem_ops &&
        cudaStreamCreateWithFlags(&probe_stream, cudaStreamNonBlocking) == cudaSuccess &&
        cudaMalloc(&stream_mem_probe, sizeof(*stream_mem_probe)) == cudaSuccess;
    const auto probe_enqueue = [&] {
        return cudaMemsetAsync(stream_mem_probe, 0, sizeof(*stream_mem_probe), probe_stream) == cudaSuccess &&
            cuStreamWriteValue32(probe_stream, (CUdeviceptr) stream_mem_probe, 1, CU_STREAM_WRITE_VALUE_DEFAULT) == CUDA_SUCCESS &&
            cuStreamWaitValue32(probe_stream, (CUdeviceptr) stream_mem_probe, 1, CU_STREAM_WAIT_VALUE_EQ) == CUDA_SUCCESS &&
            cuStreamWriteValue32(probe_stream, (CUdeviceptr) stream_mem_probe, 2, CU_STREAM_WRITE_VALUE_DEFAULT) == CUDA_SUCCESS;
    };
    const auto probe_complete = [&] {
        uint32_t value = 0;
        return cudaStreamSynchronize(probe_stream) == cudaSuccess &&
            cudaMemcpy(&value, stream_mem_probe, sizeof(value), cudaMemcpyDeviceToHost) == cudaSuccess && value == 2;
    };
    if (probe_ok) {
        probe_ok = probe_enqueue() && probe_complete();
    }
    if (probe_ok) {
        probe_ok = cudaStreamBeginCapture(probe_stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
        if (probe_ok) {
            probe_ok = probe_enqueue();
            const cudaError_t end = cudaStreamEndCapture(probe_stream, &probe_graph);
            probe_ok = probe_ok && end == cudaSuccess && probe_graph != nullptr;
        }
    }
    if (probe_ok) {
        probe_ok = cudaGraphInstantiate(&probe_exec, probe_graph, nullptr, nullptr, 0) == cudaSuccess;
    }
    for (int replay = 0; probe_ok && replay < 2; ++replay) {
        probe_ok = cudaGraphLaunch(probe_exec, probe_stream) == cudaSuccess && probe_complete();
    }
    if (probe_exec != nullptr) { (void) cudaGraphExecDestroy(probe_exec); }
    if (probe_graph != nullptr) { (void) cudaGraphDestroy(probe_graph); }
    if (probe_stream != nullptr) { (void) cudaStreamSynchronize(probe_stream); }
    if (stream_mem_probe != nullptr) { (void) cudaFree(stream_mem_probe); }
    if (probe_stream != nullptr) { (void) cudaStreamDestroy(probe_stream); }
    c->stream_mem_ops_supported = probe_ok;
    if (!probe_ok) {
        (void) cudaGetLastError();
        GGML_LOG_DEBUG("moe-cache: stream memory operations unavailable; using event staging\n");
    }
#endif

    err = cudaEventCreateWithFlags(&c->compute_done, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        fprintf(stderr, "moe-cache: cudaEventCreate failed: %s\n", cudaGetErrorString(err));
        return nullptr;
    }

    err = cudaEventCreateWithFlags(&c->stage_done, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        fprintf(stderr, "moe-cache: cudaEventCreate failed: %s\n", cudaGetErrorString(err));
        return nullptr;
    }

    err = cudaEventCreateWithFlags(&c->handoff_done, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        fprintf(stderr, "moe-cache: cudaEventCreate failed: %s\n", cudaGetErrorString(err));
        return nullptr;
    }

    if (fail_after_stream_probe) {
        throw std::bad_alloc();
    }

    c->slot_to_host.assign(n_slots, nullptr);
    c->last_used  .assign(n_slots, 0);
    c->slot_prefetched.assign(n_slots, 0);
    c->slot_hit_count.assign(n_slots, 0);
    c->slot_fill_access.assign(n_slots, 0);
    c->slot_pin_count.assign(n_slots, 0);
    c->host_to_slot.reserve((size_t) n_slots * 2);
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

    return owner.release();
}

extern "C"
struct ggml_cuda_moe_cache * ggml_cuda_moe_cache_init(
        int device,
        size_t slot_size_bytes,
        int n_slots) {
    return ggml_cuda_moe_cache_init_with_pool(
        device, slot_size_bytes, 0, n_slots, nullptr, nullptr, false);
}

extern "C"
void ggml_cuda_moe_cache_free(struct ggml_cuda_moe_cache * cache) {
    if (!cache) return;

    {
        std::lock_guard<std::mutex> lock(cache->mu);
        for (uint32_t pins : cache->slot_pin_count) {
            GGML_ASSERT(pins == 0);
        }
    }

    int prev_device = 0;
    cudaGetDevice(&prev_device);
    cudaSetDevice(cache->device);

    if (cache->compute_done) {
        cudaEventSynchronize(cache->compute_done);
    }
    if (cache->copy_stream) {
        cudaStreamSynchronize(cache->copy_stream);
        cudaStreamDestroy(cache->copy_stream);
    }
    for (auto event : cache->host_tile_done) {
        if (event != nullptr) {
            cudaEventDestroy(event);
        }
    }
    if (cache->compute_done) {
        cudaEventDestroy(cache->compute_done);
    }
    if (cache->stage_done) {
        cudaEventDestroy(cache->stage_done);
    }
    if (cache->handoff_done) {
        cudaEventDestroy(cache->handoff_done);
    }
    if (cache->owns_slot_pool && cache->slot_pool_d) {
        cudaFree(cache->slot_pool_d);
    }
    cache->host_staging.reset();
    cudaSetDevice(prev_device);

    if (cache->source != nullptr) {
        cache->source->owner->release();
    }
    delete cache;
}

static void ggml_cuda_moe_cache_clear_slots_locked(ggml_cuda_moe_cache * cache) {
    for (uint32_t pins : cache->slot_pin_count) {
        GGML_ASSERT(pins == 0);
    }
    std::fill(cache->slot_to_host.begin(), cache->slot_to_host.end(), nullptr);
    std::fill(cache->last_used.begin(), cache->last_used.end(), 0);
    std::fill(cache->slot_prefetched.begin(), cache->slot_prefetched.end(), 0);
    std::fill(cache->slot_hit_count.begin(), cache->slot_hit_count.end(), 0);
    std::fill(cache->slot_fill_access.begin(), cache->slot_fill_access.end(), 0);
    cache->host_to_slot.clear();
    cache->access_counter = 0;
}

static bool ggml_cuda_moe_cache_prepare_legacy(ggml_cuda_moe_cache * cache, cudaEvent_t grouped_done) {
    if (cache == nullptr || cache->owns_slot_pool) {
        return false;
    }
    std::lock_guard<std::mutex> lock(cache->mu);
    if (cache->host_staging_failed) {
        return false;
    }
    if (grouped_done != nullptr && !moe_grouped_cuda_success(cudaStreamWaitEvent(cache->copy_stream, grouped_done, 0))) {
        return false;
    }
    ggml_cuda_moe_cache_clear_slots_locked(cache);
    cache->last_compute_stream = nullptr;
    cache->has_compute_done = false;
    return true;
}

static bool ggml_cuda_moe_cache_prepare_host_staged(ggml_cuda_moe_cache * cache, cudaEvent_t grouped_done) {
    if (cache == nullptr || cache->owns_slot_pool) {
        return false;
    }
    std::lock_guard<std::mutex> lock(cache->mu);
    if (cache->host_staging_failed) {
        return false;
    }
    if (grouped_done != nullptr && !moe_grouped_cuda_success(cudaStreamWaitEvent(cache->copy_stream, grouped_done, 0))) {
        return false;
    }
    ggml_cuda_moe_cache_clear_slots_locked(cache);
    cache->last_compute_stream = nullptr;
    cache->has_compute_done = false;
    return true;
}

static bool ggml_cuda_moe_cache_handoff_grouped(ggml_cuda_moe_cache * cache, cudaStream_t grouped_stream) {
    if (cache == nullptr || cache->owns_slot_pool || grouped_stream == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(cache->mu);
    if (cache->host_staging_failed) {
        return false;
    }
    if (!moe_grouped_cuda_success(cudaEventRecord(cache->handoff_done, cache->copy_stream)) ||
            !moe_grouped_cuda_success(cudaStreamWaitEvent(grouped_stream, cache->handoff_done, 0)) ||
            (cache->has_compute_done && !moe_grouped_cuda_success(cudaStreamWaitEvent(grouped_stream, cache->compute_done, 0)))) {
        return false;
    }
    ggml_cuda_moe_cache_clear_slots_locked(cache);
    return true;
}

static bool ggml_cuda_moe_cache_abort_host_staged(ggml_cuda_moe_cache * cache, cudaStream_t compute_stream) {
    if (cache == nullptr || cache->owns_slot_pool || compute_stream == nullptr) {
        return false;
    }
    if (!moe_grouped_cuda_success(cudaStreamSynchronize(compute_stream)) ||
            !moe_grouped_cuda_success(cudaStreamSynchronize(cache->copy_stream))) {
        return false;
    }
    std::lock_guard<std::mutex> lock(cache->mu);
    if (std::any_of(cache->slot_pin_count.begin(), cache->slot_pin_count.end(), [](uint32_t pins) { return pins != 0; })) {
        return false;
    }
    ggml_cuda_moe_cache_clear_slots_locked(cache);
    cache->last_compute_stream = nullptr;
    cache->has_compute_done = false;
    return true;
}

static bool ggml_cuda_moe_cache_record_compute_locked(ggml_cuda_moe_cache * cache, cudaStream_t compute_stream) {
    if (cache->has_compute_done && cache->last_compute_stream != compute_stream &&
            !moe_grouped_cuda_success(cudaStreamWaitEvent(cache->copy_stream, cache->compute_done, 0))) {
        return false;
    }
    if (!moe_grouped_cuda_success(cudaEventRecord(cache->compute_done, compute_stream))) {
        return false;
    }
    cache->last_compute_stream = compute_stream;
    cache->has_compute_done = true;
    return true;
}

void ggml_cuda_moe_grouped_context::impl::retire_legacy_records(legacy_record_map records) {
    moe_cache_telemetry retired;
    for (auto & entry : records) {
        try {
            moe_cache_capture_telemetry(retired, entry.second->cache, false);
        } catch (...) {
        }
        ggml_cuda_moe_cache_free(entry.second->cache);
        entry.second->cache = nullptr;
    }
    records.clear();
    try {
        std::lock_guard<std::mutex> lock(telemetry_mutex);
        moe_cache_add_telemetry(retired_telemetry, std::move(retired));
    } catch (...) {
    }
}

void ggml_cuda_moe_grouped_context::impl::retire_legacy_records(legacy_record_list records) {
    moe_cache_telemetry retired;
    for (auto & record : records) {
        try {
            moe_cache_capture_telemetry(retired, record->cache, false);
        } catch (...) {
        }
        ggml_cuda_moe_cache_free(record->cache);
        record->cache = nullptr;
    }
    records.clear();
    try {
        std::lock_guard<std::mutex> lock(telemetry_mutex);
        moe_cache_add_telemetry(retired_telemetry, std::move(retired));
    } catch (...) {
    }
}

void ggml_cuda_moe_grouped_context::impl::retire_legacy_records(terminal_legacy_records records) {
    moe_cache_telemetry retired;
    for (auto & record : records) {
        if (record == nullptr) {
            continue;
        }
        try {
            moe_cache_capture_telemetry(retired, record->cache, false);
        } catch (...) {
        }
        ggml_cuda_moe_cache_free(record->cache);
        record->cache = nullptr;
        record.reset();
    }
    try {
        std::lock_guard<std::mutex> lock(telemetry_mutex);
        moe_cache_add_telemetry(retired_telemetry, std::move(retired));
    } catch (...) {
    }
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
    if (!cache->debug_mm) {
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
        } else {
            cache->expert_reuse_gt_l1++;
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
    bool         is_decode,
    bool         is_prefetch,
    bool         wait_for_compute) {

    if (cache->host_staging_failed || byte_count > cache->slot_size_bytes) {
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

    const bool debug_mm = cache->debug_mm;
    int64_t enqueue_start_us = 0;
    void * dst = (char *)cache->slot_pool_d + (size_t)lru_slot * cache->slot_size_bytes;
    if (debug_mm) {
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
    err = moe_cache_copy_source(cache, dst, host_src, byte_count, copy_stream, is_decode);
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
    bool         is_decode,
    bool         is_prefetch,
    bool         pin) {

    if (!cache || host_src == nullptr || byte_count == 0) {
        return -1;
    }

    std::lock_guard<std::mutex> lk(cache->mu);
    const int slot = ggml_cuda_moe_cache_acquire_locked(
        cache, host_src, byte_count, copy_stream, is_decode, is_prefetch, true);
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
    cudaStream_t         compute_stream,
    bool                 is_decode) {

    if (!cache || !host_srcs || n_host_srcs <= 0 || byte_count == 0 || !dst || !compute_stream) {
        return false;
    }
    for (int i = 0; i < n_host_srcs; ++i) {
        if (!host_srcs[i]) {
            return false;
        }
    }

    std::lock_guard<std::mutex> lk(cache->mu);
    if (cache->host_staging_failed || byte_count > cache->slot_size_bytes) {
        return false;
    }

    if (!ggml_cuda_moe_cache_record_compute_locked(cache, compute_stream)) {
        return false;
    }
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
        if (resident) {
            CUDA_CHECK(cudaMemcpyAsync((char *) dst + (size_t) i * byte_count, src, (size_t) run * byte_count, cudaMemcpyDeviceToDevice, cache->copy_stream));
        } else {
            if (moe_cache_copy_source(
                    cache, (char *) dst + (size_t) i * byte_count, src, (size_t) run * byte_count,
                    cache->copy_stream, is_decode) != cudaSuccess) {
                return false;
            }
        }
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
    size_t               trailing_padding,
    int                  min_resident,
    int *                slot_ids,
    int32_t *            source_wait_class,
    int *                out_n_resident,
    void *               miss_dst,
    uint32_t *           stage_ready,
    int                  stage_ready_capacity,
    int *                out_n_wait_classes,
    cudaStream_t         compute_stream,
    bool                 is_decode) {

    if (!cache || !host_srcs || n_host_srcs <= 0 || byte_count == 0 || trailing_padding > byte_count ||
        n_host_srcs > std::numeric_limits<int>::max() - 2 ||
        (size_t) n_host_srcs > (SIZE_MAX - trailing_padding) / byte_count || min_resident <= 0 ||
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
    if (cache->host_staging_failed || byte_count > cache->slot_size_bytes || n_host_srcs <= cache->n_slots) {
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
            source_wait_class[i] = slot_ids[i] >= 0 ? 1 : -1;
        }
    }
    if (n_resident == n_host_srcs) {
        return false;
    }

    if (overlap) {
        CUDA_CHECK(cudaMemsetAsync(
            stage_ready, 0, (size_t)stage_ready_capacity * sizeof(uint32_t), compute_stream));
    }
    if (!ggml_cuda_moe_cache_record_compute_locked(cache, compute_stream)) {
        return false;
    }
    CUDA_CHECK(cudaStreamWaitEvent(cache->copy_stream, cache->compute_done, 0));

    for (int i = 0; i < n_host_srcs; ++i) {
        if (slot_ids[i] >= 0) {
            cache->slot_pin_count[slot_ids[i]]++;
        }
    }

    auto release_pins = [&]() {
        for (int i = 0; i < n_host_srcs; ++i) {
            if (slot_ids[i] >= 0) {
                cache->slot_pin_count[slot_ids[i]]--;
            }
        }
    };

    for (int i = 0; i < n_host_srcs && n_resident < cache->n_slots; ++i) {
        if (slot_ids[i] >= 0) {
            continue;
        }

        const int slot = ggml_cuda_moe_cache_acquire_locked(
            cache, host_srcs[i], byte_count, cache->copy_stream, is_decode, false, false);
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

    if (cache->host_staging_failed || n_resident < min_resident) {
        release_pins();
        return false;
    }

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && !defined(GGML_CUDA_NO_VMM)
    if (overlap) {
        CU_CHECK(cuStreamWriteValue32(
            cache->copy_stream, (CUdeviceptr)(stage_ready + 0), 1, CU_STREAM_WRITE_VALUE_DEFAULT));
    }
#endif

    std::vector<const void *> miss_sources;
    std::vector<int> miss_source_indices;
    miss_sources.reserve((size_t) n_host_srcs - n_resident);
    miss_source_indices.reserve((size_t) n_host_srcs - n_resident);
    for (int i = 0; i < n_host_srcs; ++i) {
        if (slot_ids[i] < 0) {
            miss_sources.push_back(host_srcs[i]);
            miss_source_indices.push_back(i);
        }
    }
    const size_t n_misses = miss_sources.size();
    GGML_ASSERT(n_misses > 0);
    size_t wave_size = n_misses;
    if (overlap) {
        const size_t max_staging_waves = (size_t) stage_ready_capacity - 1;
        const size_t min_wave_size = 1 + (n_misses - 1) / max_staging_waves;
        wave_size = std::max((size_t) cache->n_slots, min_wave_size);
    }
    const size_t n_waves = 1 + (n_misses - 1) / wave_size;
    GGML_ASSERT(n_waves <= (size_t) std::numeric_limits<int>::max() - 2);
    if (source_wait_class != nullptr) {
        for (size_t miss = 0; miss < n_misses; ++miss) {
            source_wait_class[miss_source_indices[miss]] = (int32_t) (2 + miss / wave_size);
        }
    }

    size_t copied = 0;
    size_t wave_end = 0;
    for (size_t wave = 0; wave < n_waves; ++wave) {
        wave_end += std::min(wave_size, n_misses - wave_end);
        const size_t copy_end = wave_end + (trailing_padding > 0 && wave + 1 < n_waves ? 1 : 0);
        while (copied < copy_end) {
            const void * src = miss_sources[copied];
            size_t run = 1;
            while (copied + run < copy_end &&
                   (uintptr_t) miss_sources[copied + run] == (uintptr_t) src + (size_t) run * byte_count) {
                ++run;
            }
            if (moe_cache_copy_source(
                    cache, (char *) miss_dst + copied * byte_count, src, run * byte_count,
                    cache->copy_stream, is_decode) != cudaSuccess) {
                release_pins();
                return false;
            }
            copied += run;
        }
        if (wave + 1 == n_waves && trailing_padding > 0) {
            CUDA_CHECK(cudaMemsetAsync(
                (char *) miss_dst + (size_t) n_misses * byte_count, 0, trailing_padding, cache->copy_stream));
        }
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && !defined(GGML_CUDA_NO_VMM)
        if (overlap) {
            CU_CHECK(cuStreamWriteValue32(
                cache->copy_stream, (CUdeviceptr)(stage_ready + wave + 1), 1, CU_STREAM_WRITE_VALUE_DEFAULT));
        }
#endif
    }

    CUDA_CHECK(cudaEventRecord(cache->stage_done, cache->copy_stream));
    if (!overlap) {
        CUDA_CHECK(cudaStreamWaitEvent(compute_stream, cache->stage_done, 0));
    }
    *out_n_resident = n_resident;
    *out_n_wait_classes = overlap ? (int) (2 + n_waves) : 1;
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
    if (cache->host_staging_failed) {
        return false;
    }
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
    if (!ggml_cuda_moe_cache_record_compute_locked(cache, compute_stream)) {
        fprintf(stderr, "moe-cache: cudaEventRecord failed\n");
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
    if (!cache->owns_slot_pool) {
        return false;
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

    if (min_slot_size_bytes > (SIZE_MAX - cache->trailing_padding_bytes) / (size_t) cache->n_slots) {
        cudaSetDevice(prev_device);
        return false;
    }
    const size_t allocation_size =
        (size_t) cache->n_slots * min_slot_size_bytes + cache->trailing_padding_bytes;
    void * new_pool = nullptr;
    err = cudaMalloc(&new_pool, allocation_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "moe-cache: grow_pool cudaMalloc(%zu) failed: %s\n",
                allocation_size, cudaGetErrorString(err));
        cudaSetDevice(prev_device);
        return false;
    }
    if (cache->trailing_padding_bytes > 0) {
        err = cudaMemset(
            (char *) new_pool + (size_t) cache->n_slots * min_slot_size_bytes,
            0, cache->trailing_padding_bytes);
        if (err != cudaSuccess) {
            cudaFree(new_pool);
            cudaSetDevice(prev_device);
            return false;
        }
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

size_t ggml_cuda_moe_cache_trailing_padding_bytes_for_test(const struct ggml_cuda_moe_cache * cache) {
    return cache != nullptr ? cache->trailing_padding_bytes : 0;
}

bool ggml_cuda_moe_cache_trailing_padding_zero_for_test(struct ggml_cuda_moe_cache * cache) {
    if (cache == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(cache->mu);
    if (cache->trailing_padding_bytes == 0) {
        return true;
    }
    int previous_device = 0;
    if (cudaGetDevice(&previous_device) != cudaSuccess || cudaSetDevice(cache->device) != cudaSuccess) {
        return false;
    }
    std::vector<uint8_t> padding(cache->trailing_padding_bytes);
    const cudaError_t error = cudaMemcpy(
        padding.data(),
        static_cast<const char *>(cache->slot_pool_d) + (size_t) cache->n_slots * cache->slot_size_bytes,
        padding.size(), cudaMemcpyDeviceToHost);
    (void) cudaSetDevice(previous_device);
    return error == cudaSuccess && std::all_of(padding.begin(), padding.end(), [](uint8_t value) { return value == 0; });
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
    if (!ggml_cuda_moe_cache_record_compute_locked(cache, compute_stream)) {
        fprintf(stderr, "moe-cache: cudaEventRecord failed\n");
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
    s.upload_calls = cache->phase_upload_calls[phase].load(std::memory_order_relaxed);
    s.upload_bytes = cache->phase_upload_bytes[phase].load(std::memory_order_relaxed);
    s.staging_tile_count = cache->phase_staging_tile_count[phase].load(std::memory_order_relaxed);
    s.pipeline_tiles = cache->phase_pipeline_tiles[phase].load(std::memory_order_relaxed);
    s.tile_wait_calls = cache->phase_tile_wait_calls[phase].load(std::memory_order_relaxed);
    s.tile_wait_time_us = cache->phase_tile_wait_time_us[phase].load(std::memory_order_relaxed);
    s.cpu_pack_bytes = cache->phase_cpu_pack_bytes[phase].load(std::memory_order_relaxed);
    s.cpu_pack_time_us = cache->phase_cpu_pack_time_us[phase].load(std::memory_order_relaxed);
    s.pre_sync_calls = cache->phase_pre_sync_calls[phase].load(std::memory_order_relaxed);
    s.pre_sync_time_us = cache->phase_pre_sync_time_us[phase].load(std::memory_order_relaxed);
    s.h2d_submit_calls = cache->phase_h2d_submit_calls[phase].load(std::memory_order_relaxed);
    s.h2d_submit_bytes = cache->phase_h2d_submit_bytes[phase].load(std::memory_order_relaxed);
    s.h2d_submit_time_us = cache->phase_h2d_submit_time_us[phase].load(std::memory_order_relaxed);
    s.post_sync_calls = cache->phase_post_sync_calls[phase].load(std::memory_order_relaxed);
    s.post_sync_time_us = cache->phase_post_sync_time_us[phase].load(std::memory_order_relaxed);
    s.upload_errors = cache->phase_upload_errors[phase].load(std::memory_order_relaxed);
    return s;
}

static moe_cache_phase_stats moe_cache_take_op_stats(moe_cache_op_phase_stats & op, bool reset) {
    moe_cache_phase_stats s = {};
    auto take = [reset](std::atomic<uint64_t> & value) {
        return reset ? value.exchange(0, std::memory_order_relaxed) : value.load(std::memory_order_relaxed);
    };
    s.ops = take(op.ops);
    s.staged_ops = take(op.staged_ops);
    s.split_staged_ops = take(op.split_staged_ops);
    s.overflow_ops = take(op.overflow_ops);
    s.unique_experts = take(op.unique_experts);
    s.unique_experts_max = take(op.unique_experts_max);
    s.ids_bytes = take(op.ids_bytes);
    s.ids_d2h_time_us = take(op.ids_d2h_time_us);
    s.ids_d2h_sync_count = take(op.ids_d2h_sync_count);
    s.ids_cache_hits = take(op.ids_cache_hits);
    s.acquire_time_us = take(op.acquire_time_us);
    s.remap_time_us = take(op.remap_time_us);
    s.copy_wait_event_count = take(op.copy_wait_event_count);
    s.copy_wait_event_time_us = take(op.copy_wait_event_time_us);
    s.total_time_us = take(op.total_time_us);
    return s;
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
    dst.upload_calls += src.upload_calls;
    dst.upload_bytes += src.upload_bytes;
    dst.staging_tile_count += src.staging_tile_count;
    dst.pipeline_tiles += src.pipeline_tiles;
    dst.tile_wait_calls += src.tile_wait_calls;
    dst.tile_wait_time_us += src.tile_wait_time_us;
    dst.cpu_pack_bytes += src.cpu_pack_bytes;
    dst.cpu_pack_time_us += src.cpu_pack_time_us;
    dst.pre_sync_calls += src.pre_sync_calls;
    dst.pre_sync_time_us += src.pre_sync_time_us;
    dst.h2d_submit_calls += src.h2d_submit_calls;
    dst.h2d_submit_bytes += src.h2d_submit_bytes;
    dst.h2d_submit_time_us += src.h2d_submit_time_us;
    dst.post_sync_calls += src.post_sync_calls;
    dst.post_sync_time_us += src.post_sync_time_us;
    dst.upload_errors += src.upload_errors;
    dst.ops += src.ops;
    dst.staged_ops += src.staged_ops;
    dst.split_staged_ops += src.split_staged_ops;
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
    const double l1_hit_rate = l1_total > 0 ? 100.0 * (double) s.l1_hits / (double) l1_total : 0.0;
    const double avg_unique = s.ops > 0 ? (double) s.unique_experts / (double) s.ops : 0.0;
    GGML_LOG(
        "moe-cache-phase: phase=%s ops=%llu staged_ops=%llu split_staged_ops=%llu overflow_ops=%llu unique_avg=%.2f unique_max=%llu ids_mib=%.2f ids_d2h_mib=%.2f ids_d2h_ms=%.3f ids_d2h_syncs=%llu ids_cache_hits=%llu acquire_ms=%.3f remap_ms=%.3f copy_wait_events=%llu copy_wait_event_ms=%.3f op_cpu_ms=%.3f l1_hits=%llu l1_misses=%llu l1_evictions=%llu l1_hit_rate=%.2f%% h2d_copies=%llu h2d_mib=%.2f h2d_enqueue_ms=%.3f prefetch_hits=%llu prefetch_misses=%llu prefetch_used=%llu prefetch_h2d_copies=%llu prefetch_h2d_mib=%.2f prefetch_h2d_enqueue_ms=%.3f upload_calls=%llu upload_mib=%.2f staging_tiles=%llu pipeline_tiles=%llu tile_wait_calls=%llu tile_wait_ms=%.3f cpu_pack_mib=%.2f cpu_pack_ms=%.3f pre_sync_calls=%llu pre_sync_ms=%.3f h2d_submit_calls=%llu h2d_submit_mib=%.2f h2d_submit_ms=%.3f post_sync_calls=%llu post_sync_ms=%.3f upload_errors=%llu\n",
        name,
        (unsigned long long) s.ops,
        (unsigned long long) s.staged_ops,
        (unsigned long long) s.split_staged_ops,
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
        (unsigned long long) s.h2d_copy_count,
        (double) s.h2d_copy_bytes / 1024.0 / 1024.0,
        (double) s.h2d_enqueue_time_us / 1000.0,
        (unsigned long long) s.prefetch_hits,
        (unsigned long long) s.prefetch_misses,
        (unsigned long long) s.prefetch_used,
        (unsigned long long) s.prefetch_h2d_copy_count,
        (double) s.prefetch_h2d_copy_bytes / 1024.0 / 1024.0,
        (double) s.prefetch_h2d_enqueue_time_us / 1000.0,
        (unsigned long long) s.upload_calls,
        (double) s.upload_bytes / 1024.0 / 1024.0,
        (unsigned long long) s.staging_tile_count,
        (unsigned long long) s.pipeline_tiles,
        (unsigned long long) s.tile_wait_calls,
        (double) s.tile_wait_time_us / 1000.0,
        (double) s.cpu_pack_bytes / 1024.0 / 1024.0,
        (double) s.cpu_pack_time_us / 1000.0,
        (unsigned long long) s.pre_sync_calls,
        (double) s.pre_sync_time_us / 1000.0,
        (unsigned long long) s.h2d_submit_calls,
        (double) s.h2d_submit_bytes / 1024.0 / 1024.0,
        (double) s.h2d_submit_time_us / 1000.0,
        (unsigned long long) s.post_sync_calls,
        (double) s.post_sync_time_us / 1000.0,
        (unsigned long long) s.upload_errors);
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
    s.reuse_gt_l1 = cache->expert_reuse_gt_l1;

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
        cache->phase_upload_calls[phase].store(0, std::memory_order_relaxed);
        cache->phase_upload_bytes[phase].store(0, std::memory_order_relaxed);
        cache->phase_staging_tile_count[phase].store(0, std::memory_order_relaxed);
        cache->phase_pipeline_tiles[phase].store(0, std::memory_order_relaxed);
        cache->phase_tile_wait_calls[phase].store(0, std::memory_order_relaxed);
        cache->phase_tile_wait_time_us[phase].store(0, std::memory_order_relaxed);
        cache->phase_cpu_pack_bytes[phase].store(0, std::memory_order_relaxed);
        cache->phase_cpu_pack_time_us[phase].store(0, std::memory_order_relaxed);
        cache->phase_pre_sync_calls[phase].store(0, std::memory_order_relaxed);
        cache->phase_pre_sync_time_us[phase].store(0, std::memory_order_relaxed);
        cache->phase_h2d_submit_calls[phase].store(0, std::memory_order_relaxed);
        cache->phase_h2d_submit_bytes[phase].store(0, std::memory_order_relaxed);
        cache->phase_h2d_submit_time_us[phase].store(0, std::memory_order_relaxed);
        cache->phase_post_sync_calls[phase].store(0, std::memory_order_relaxed);
        cache->phase_post_sync_time_us[phase].store(0, std::memory_order_relaxed);
        cache->phase_upload_errors[phase].store(0, std::memory_order_relaxed);
    }
    cache->sampled_mincore_checks.store(0, std::memory_order_relaxed);
    cache->sampled_pages_total.store(0, std::memory_order_relaxed);
    cache->sampled_pages_resident.store(0, std::memory_order_relaxed);
    cache->sampled_nonresident_expert_count.store(0, std::memory_order_relaxed);
    cache->mincore_failures.store(0, std::memory_order_relaxed);
    std::fill(cache->expert_access_counts.begin(), cache->expert_access_counts.end(), 0);
    std::fill(cache->expert_last_access.begin(), cache->expert_last_access.end(), 0);
    cache->expert_access_counter = 0;
    cache->expert_first_touches = 0;
    cache->expert_reuse_le_l1 = 0;
    cache->expert_reuse_gt_l1 = 0;
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

static void moe_grouped_add_telemetry(
        ggml_cuda_moe_grouped_debug_telemetry & dst,
        const ggml_cuda_moe_grouped_debug_telemetry & src) {
    if (src.registered != 0) {
        if (dst.registered == 0) {
            dst.ready_min = src.ready_min;
            dst.completed_min = src.completed_min;
        } else {
            dst.ready_min = std::min(dst.ready_min, src.ready_min);
            dst.completed_min = std::min(dst.completed_min, src.completed_min);
        }
        dst.ready_max = std::max(dst.ready_max, src.ready_max);
        dst.completed_max = std::max(dst.completed_max, src.completed_max);
    }
    dst.registered += src.registered;
    dst.covered += src.covered;
    dst.plan_calls += src.plan_calls;
    dst.plan_compiles += src.plan_compiles;
    dst.plan_reuses += src.plan_reuses;
    dst.calls += src.calls;
    dst.ready += src.ready;
    dst.completed += src.completed;
    dst.admitted_banks += src.admitted_banks;
    dst.fallback += src.fallback;
    dst.rollback += src.rollback;
    dst.host_staged_calls += src.host_staged_calls;
    dst.host_staged_ops += src.host_staged_ops;
    dst.host_staged_split_ops += src.host_staged_split_ops;
    dst.strategy_switches += src.strategy_switches;
    dst.required_unsupported += src.required_unsupported;
    dst.prepare_error += src.prepare_error;
    dst.finish_error += src.finish_error;
    dst.h2d_banks += src.h2d_banks;
    dst.h2d_bytes += src.h2d_bytes;
    dst.decode_grouped += src.decode_grouped;
    dst.decode_legacy += src.decode_legacy;
    dst.submitted += src.submitted;
    dst.direct += src.direct;
    dst.captures += src.captures;
    dst.replays += src.replays;
}

static void moe_cache_add_telemetry(moe_cache_telemetry & dst, moe_cache_telemetry && src) {
    dst.n_caches += src.n_caches;
    dst.total_hits += src.total_hits;
    dst.total_misses += src.total_misses;
    dst.total_evictions += src.total_evictions;
    dst.mm.h2d_copy_count += src.mm.h2d_copy_count;
    dst.mm.h2d_copy_bytes += src.mm.h2d_copy_bytes;
    dst.mm.h2d_enqueue_time_us += src.mm.h2d_enqueue_time_us;
    dst.mm.sampled_mincore_checks += src.mm.sampled_mincore_checks;
    dst.mm.sampled_pages_total += src.mm.sampled_pages_total;
    dst.mm.sampled_pages_resident += src.mm.sampled_pages_resident;
    dst.mm.sampled_nonresident_expert_count += src.mm.sampled_nonresident_expert_count;
    dst.mm.mincore_failures += src.mm.mincore_failures;
    dst.experts.tensors += src.experts.tensors;
    dst.experts.experts += src.experts.experts;
    dst.experts.unique_experts += src.experts.unique_experts;
    dst.experts.accesses += src.experts.accesses;
    dst.experts.first_touches += src.experts.first_touches;
    dst.experts.reuse_le_l1 += src.experts.reuse_le_l1;
    dst.experts.reuse_gt_l1 += src.experts.reuse_gt_l1;
    dst.experts.touched_once += src.experts.touched_once;
    dst.experts.touched_ge2 += src.experts.touched_ge2;
    if (src.hot_tensor.accesses > dst.hot_tensor.accesses) {
        dst.hot_tensor = std::move(src.hot_tensor);
    }
    // Retired owners can accumulate without a stats reset. Keep totals when detail exceeds the limit.
    constexpr size_t max_expert_counts = 1024 * 1024;
    constexpr size_t max_decode_tensors = 8192;
    dst.expert_details_omitted |= src.expert_details_omitted ||
        src.all_expert_access_counts.size() > max_expert_counts - dst.all_expert_access_counts.size();
    if (dst.expert_details_omitted) {
        std::vector<uint64_t>().swap(dst.all_expert_access_counts);
    } else {
        dst.all_expert_access_counts.insert(
            dst.all_expert_access_counts.end(), src.all_expert_access_counts.begin(), src.all_expert_access_counts.end());
    }
    if (src.hot_decode_miss_tensor.h2d_copy_bytes > dst.hot_decode_miss_tensor.h2d_copy_bytes) {
        dst.hot_decode_miss_tensor = std::move(src.hot_decode_miss_tensor);
    }
    dst.decode_details_omitted |= src.decode_details_omitted ||
        src.decode_tensor_stats.size() > max_decode_tensors - dst.decode_tensor_stats.size();
    if (dst.decode_details_omitted) {
        std::vector<moe_cache_tensor_decode_stats>().swap(dst.decode_tensor_stats);
    } else {
        dst.decode_tensor_stats.insert(
            dst.decode_tensor_stats.end(),
            std::make_move_iterator(src.decode_tensor_stats.begin()),
            std::make_move_iterator(src.decode_tensor_stats.end()));
    }
    for (int phase = 0; phase < 2; ++phase) {
        ggml_cuda_moe_add_phase_stats(dst.phase_stats[phase], src.phase_stats[phase]);
    }
    moe_grouped_add_telemetry(dst.grouped, src.grouped);
}

static void moe_cache_capture_telemetry(moe_cache_telemetry & dst, ggml_cuda_moe_cache * cache, bool reset) {
    if (cache == nullptr) {
        return;
    }
    moe_cache_telemetry sample;
    std::lock_guard<std::mutex> lock(cache->mu);
    sample.n_caches = 1;
    ggml_cuda_moe_cache_stats(cache, &sample.total_hits, &sample.total_misses, &sample.total_evictions);
    if (cache->debug_mm) {
        sample.mm = ggml_cuda_moe_cache_mm_stats(cache);
        for (int phase = 0; phase < 2; ++phase) {
            ggml_cuda_moe_add_phase_stats(sample.phase_stats[phase], ggml_cuda_moe_cache_phase_stats(cache, phase));
        }
        const moe_cache_hot_tensor_stats expert = ggml_cuda_moe_cache_expert_stats(cache);
        if (expert.experts > 0) {
            sample.experts.tensors = 1;
            sample.experts.experts = expert.experts;
            sample.experts.unique_experts = expert.unique_experts;
            sample.experts.accesses = expert.accesses;
            sample.experts.first_touches = expert.first_touches;
            sample.experts.reuse_le_l1 = expert.reuse_le_l1;
            sample.experts.reuse_gt_l1 = expert.reuse_gt_l1;
            sample.experts.touched_once = expert.touched_once;
            sample.experts.touched_ge2 = expert.touched_ge2;
            sample.hot_tensor = expert;
            ggml_cuda_moe_cache_append_expert_counts(cache, sample.all_expert_access_counts);
        }
        moe_cache_tensor_decode_stats decode = ggml_cuda_moe_cache_decode_tensor_stats(cache);
        if (decode.l1_hits + decode.l1_misses + decode.h2d_copy_count > 0) {
            sample.decode_tensor_stats.push_back(decode);
        }
        sample.hot_decode_miss_tensor = std::move(decode);
    }
    if (reset) {
        ggml_cuda_moe_cache_reset_stats(cache);
    }
    moe_cache_add_telemetry(dst, std::move(sample));
}

extern "C"
void ggml_cuda_moe_record_op_stats(
    bool     is_decode,
    bool     staged,
    bool     split_staged,
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
    s.split_staged_ops.fetch_add(split_staged ? 1 : 0, std::memory_order_relaxed);
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

extern "C"
bool ggml_backend_cuda_moe_cached_configure_sources(ggml_backend_buffer_type_t buft, const ggml_backend_moe_candidate_snapshot_v2 * snapshot) {
    auto * owner = moe_host_budget_for(buft);
    if (owner == nullptr || snapshot == nullptr || snapshot->magic != GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_MAGIC ||
            snapshot->abi_version != GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_VERSION || snapshot->struct_size != sizeof(*snapshot) ||
            snapshot->n_slots == 0 || snapshot->n_groups > GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS ||
            (snapshot->n_groups != 0 && snapshot->groups == nullptr) || (snapshot->n_tensors != 0 && snapshot->tensors == nullptr)) {
        return false;
    }
    std::lock_guard<std::mutex> lock(owner->mutex);
    if (owner->configured || getenv("GGML_CUDA_NO_PINNED") != nullptr) {
        return false;
    }
    std::vector<std::vector<moe_host_source *>> groups(snapshot->n_groups);
    std::vector<size_t> group_staging(snapshot->n_groups, 0);
    std::vector<uint32_t> roles(snapshot->n_groups, 0);
    std::vector<bool> complete(snapshot->n_groups, true);
    std::vector<moe_host_source *> auxiliaries;
    size_t staging = 0;
    auto reserve = [&](size_t bytes) {
        if (!moe_host_reserve_size(owner->limit, bytes, staging)) {
            GGML_LOG_ERROR("moe-cache-host: budget cannot reserve mandatory staging\n");
            return false;
        }
        return true;
    };
    for (uint32_t i = 0; i < snapshot->n_tensors; ++i) {
        const auto & record = snapshot->tensors[i];
        const auto * tensor = record.tensor;
        const bool base = moe_candidate_routed_base(record);
        if (tensor == nullptr || tensor->buffer == nullptr || tensor->buffer->buft != buft) {
            if (base && record.group_index < complete.size()) {
                complete[record.group_index] = false;
            }
            continue;
        }
        const size_t size = ggml_nbytes(tensor);
        const uintptr_t begin = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(tensor->buffer));
        const uintptr_t address = reinterpret_cast<uintptr_t>(tensor->data);
        if (tensor->data == nullptr || size == 0 || address < begin || address - begin > tensor->buffer->size ||
                size > tensor->buffer->size - (address - begin) || owner->sources.count(tensor) != 0) {
            return false;
        }
        const bool read_only = moe_host_buffer_is_read_only(tensor->buffer);
        auto inserted = owner->sources.emplace(tensor, moe_host_source{owner, static_cast<const char *>(tensor->data), size, base ? tensor->nb[2] : size, read_only});
        auto * source = &inserted.first->second;
        if (!base) {
            auxiliaries.push_back(source);
            continue;
        }
        if (record.group_index >= groups.size() || tensor->ne[2] <= 0 || tensor->ne[2] > INT32_MAX ||
                tensor->nb[2] == 0 || size / tensor->nb[2] != static_cast<uint64_t>(tensor->ne[2]) || size % tensor->nb[2] != 0 ||
                tensor->nb[2] > SIZE_MAX - group_staging[record.group_index] || !reserve(tensor->nb[2])) {
            return false;
        }
        group_staging[record.group_index] += tensor->nb[2];
        groups[record.group_index].push_back(source);
        roles[record.group_index] |= 1u << record.role;
    }
    for (uint32_t i = 0; i < groups.size(); ++i) {
        if (groups[i].empty()) {
            continue;
        }
        size_t plan_bytes = 0;
        const uint32_t n_experts = groups[i][0]->size / groups[i][0]->expert_stride;
        size_t control_offset = 0, staging_offset = 0, allocation_bytes = 0;
        if (!moe_grouped_plan_size(snapshot->n_slots, n_experts, &plan_bytes) ||
                !moe_grouped_staging_layout(plan_bytes, group_staging[i], moe_early_router_enabled(), control_offset, staging_offset, allocation_bytes)) {
            return false;
        }
        if (!reserve(allocation_bytes)) {
            return false;
        }
        complete[i] = complete[i] && snapshot->groups[i].flags == 0 &&
            roles[i] == moe_candidate_required_base_roles(snapshot->groups[i].layout) &&
            groups[i].size() == moe_candidate_required_base_bank_count(snapshot->groups[i].layout);
    }
    owner->staging_reserved = staging;
    if (!moe_host_register(*owner, auxiliaries, true, UINT32_MAX)) {
        GGML_LOG_ERROR("moe-cache-host: budget cannot pin mandatory auxiliary sources\n");
        return false;
    }
    size_t direct = 0;
    bool staged = false;
    for (uint32_t i = 0; i < groups.size(); ++i) {
        if (groups[i].empty()) {
            continue;
        }
        if (!complete[i]) {
            GGML_LOG_WARN("moe-cache-host: admission group=%u reason=incomplete_group\n", i);
            staged = true;
        } else if (moe_host_register(*owner, groups[i], false, i)) {
            ++direct;
        } else {
            staged = true;
        }
    }
    if (staged) {
        owner->copy_worker.start();
    }
    owner->configured = true;
    GGML_LOG_INFO("moe-cache-host: limit=%zu source=%zu staging_reserved=%zu direct_groups=%zu total_groups=%zu\n",
        owner->limit, owner->source_bytes, owner->staging_reserved, direct, groups.size());
    return true;
}

extern "C"
void ggml_backend_cuda_moe_set_debug_mm(bool enabled) {
    g_moe_cache_mm_debug.store(enabled, std::memory_order_relaxed);
}

extern "C"
bool ggml_backend_cuda_moe_get_debug_mm(void) {
    return g_moe_cache_mm_debug.load(std::memory_order_relaxed);
}

static void moe_cache_log_telemetry(moe_cache_telemetry telemetry) {
    const size_t n_caches = telemetry.n_caches;
    const uint64_t total_hits = telemetry.total_hits;
    const uint64_t total_misses = telemetry.total_misses;
    const uint64_t total_evictions = telemetry.total_evictions;
    const moe_cache_mm_stats & mm = telemetry.mm;
    moe_cache_expert_stats & experts = telemetry.experts;
    const moe_cache_hot_tensor_stats & hot_tensor = telemetry.hot_tensor;
    std::vector<uint64_t> & all_expert_access_counts = telemetry.all_expert_access_counts;
    const moe_cache_tensor_decode_stats & hot_decode_miss_tensor = telemetry.hot_decode_miss_tensor;
    std::vector<moe_cache_tensor_decode_stats> & decode_tensor_stats = telemetry.decode_tensor_stats;
    moe_cache_phase_stats * phase_stats = telemetry.phase_stats;
    const auto & grouped = telemetry.grouped;
    const uint64_t total = total_hits + total_misses;
    const double rate = total > 0 ? 100.0 * (double)total_hits / (double)total : 0.0;
    GGML_LOG_INFO("moe-cache: %zu caches  hits=%llu  misses=%llu  evictions=%llu  hit-rate=%.2f%%\n",
                  n_caches,
                  (unsigned long long)total_hits,
                  (unsigned long long)total_misses,
                  (unsigned long long)total_evictions,
                  rate);

    if (moe_cache_mm_debug_enabled()) {
        GGML_LOG(
            "moe-grouped-decode: registered=%llu covered=%llu plan_calls=%llu plan_compiles=%llu plan_reuses=%llu calls=%llu ready=%llu ready_min=%llu ready_max=%llu completed=%llu completed_min=%llu completed_max=%llu admitted_banks=%llu fallback=%llu rollback=%llu host_calls=%llu host_ops=%llu host_split_ops=%llu strategy_switches=%llu required_unsupported=%llu prepare_error=%llu finish_error=%llu h2d_banks=%llu h2d_bytes=%llu\n",
            (unsigned long long) grouped.registered,
            (unsigned long long) grouped.covered,
            (unsigned long long) grouped.plan_calls,
            (unsigned long long) grouped.plan_compiles,
            (unsigned long long) grouped.plan_reuses,
            (unsigned long long) grouped.calls,
            (unsigned long long) grouped.ready,
            (unsigned long long) grouped.ready_min,
            (unsigned long long) grouped.ready_max,
            (unsigned long long) grouped.completed,
            (unsigned long long) grouped.completed_min,
            (unsigned long long) grouped.completed_max,
            (unsigned long long) grouped.admitted_banks,
            (unsigned long long) grouped.fallback,
            (unsigned long long) grouped.rollback,
            (unsigned long long) grouped.host_staged_calls,
            (unsigned long long) grouped.host_staged_ops,
            (unsigned long long) grouped.host_staged_split_ops,
            (unsigned long long) grouped.strategy_switches,
            (unsigned long long) grouped.required_unsupported,
            (unsigned long long) grouped.prepare_error,
            (unsigned long long) grouped.finish_error,
            (unsigned long long) grouped.h2d_banks,
            (unsigned long long) grouped.h2d_bytes);
        const moe_cache_proc_snapshot proc = moe_cache_get_proc_delta();
        const double h2d_mib = (double) mm.h2d_copy_bytes / 1024.0 / 1024.0;
        const double h2d_enqueue_ms = (double) mm.h2d_enqueue_time_us / 1000.0;
        const double resident_pct = mm.sampled_pages_total > 0 ?
            100.0 * (double) mm.sampled_pages_resident / (double) mm.sampled_pages_total : 0.0;
        const uint64_t reuse_total = experts.reuse_le_l1 + experts.reuse_gt_l1;
        const double unique_pct = experts.experts > 0 ?
            100.0 * (double) experts.unique_experts / (double) experts.experts : 0.0;
        const double first_touch_pct = experts.accesses > 0 ?
            100.0 * (double) experts.first_touches / (double) experts.accesses : 0.0;
        const double reuse_l1_pct = reuse_total > 0 ?
            100.0 * (double) experts.reuse_le_l1 / (double) reuse_total : 0.0;
        const double reuse_gt_l1_pct = reuse_total > 0 ?
            100.0 * (double) experts.reuse_gt_l1 / (double) reuse_total : 0.0;
        if (!all_expert_access_counts.empty()) {
            const size_t n_experts = all_expert_access_counts.size();
            const size_t n_top1 = std::max<size_t>(1, (n_experts + 99) / 100);
            const size_t n_top5 = std::max<size_t>(1, (n_experts * 5 + 99) / 100);
            const size_t n_top10 = std::max<size_t>(1, (n_experts * 10 + 99) / 100);
            experts.top1_accesses = ggml_cuda_moe_cache_top_accesses(all_expert_access_counts, n_top1);
            experts.top5_accesses = ggml_cuda_moe_cache_top_accesses(all_expert_access_counts, n_top5);
            experts.top10_accesses = ggml_cuda_moe_cache_top_accesses(all_expert_access_counts, n_top10);
        }
        const double top1_pct = telemetry.expert_details_omitted ? -1.0 : experts.accesses > 0 ?
            100.0 * (double) experts.top1_accesses / (double) experts.accesses : 0.0;
        const double top5_pct = telemetry.expert_details_omitted ? -1.0 : experts.accesses > 0 ?
            100.0 * (double) experts.top5_accesses / (double) experts.accesses : 0.0;
        const double top10_pct = telemetry.expert_details_omitted ? -1.0 : experts.accesses > 0 ?
            100.0 * (double) experts.top10_accesses / (double) experts.accesses : 0.0;
        const double hot_unique_pct = hot_tensor.experts > 0 ?
            100.0 * (double) hot_tensor.unique_experts / (double) hot_tensor.experts : 0.0;
        const double hot_top10_pct = hot_tensor.accesses > 0 ?
            100.0 * (double) hot_tensor.top10_accesses / (double) hot_tensor.accesses : 0.0;
        const uint64_t hot_reuse_total = hot_tensor.reuse_le_l1 + hot_tensor.reuse_gt_l1;
        const double hot_reuse_gt_l1_pct = hot_reuse_total > 0 ?
            100.0 * (double) hot_tensor.reuse_gt_l1 / (double) hot_reuse_total : 0.0;
        const uint64_t hot_decode_total = hot_decode_miss_tensor.l1_hits + hot_decode_miss_tensor.l1_misses;
        const double hot_decode_hit_rate = hot_decode_total > 0 ?
            100.0 * (double) hot_decode_miss_tensor.l1_hits / (double) hot_decode_total : 0.0;
        const double hot_decode_reuse_gt_l1_pct = hot_decode_miss_tensor.reuse_total > 0 ?
            100.0 * (double) hot_decode_miss_tensor.reuse_gt_l1 / (double) hot_decode_miss_tensor.reuse_total : 0.0;

        GGML_LOG(
            "moe-cache-experts: tensors=%llu experts=%llu unique=%llu unique_pct=%.2f accesses=%llu first_touches=%llu first_touch_pct=%.2f touched_once=%llu touched_ge2=%llu reuse_le_l1=%llu reuse_le_l1_pct=%.2f reuse_gt_l1=%llu reuse_gt_l1_pct=%.2f top1_pct=%.2f top5_pct=%.2f top10_pct=%.2f\n",
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
            (unsigned long long) experts.reuse_gt_l1,
            reuse_gt_l1_pct,
            top1_pct,
            top5_pct,
            top10_pct);
        GGML_LOG(
            "moe-cache-experts-hot: tensor=%s experts=%llu unique=%llu unique_pct=%.2f accesses=%llu touched_once=%llu touched_ge2=%llu top10_pct=%.2f reuse_gt_l1_pct=%.2f\n",
            hot_tensor.name.empty() ? "?" : hot_tensor.name.c_str(),
            (unsigned long long) hot_tensor.experts,
            (unsigned long long) hot_tensor.unique_experts,
            hot_unique_pct,
            (unsigned long long) hot_tensor.accesses,
            (unsigned long long) hot_tensor.touched_once,
            (unsigned long long) hot_tensor.touched_ge2,
            hot_top10_pct,
            hot_reuse_gt_l1_pct);
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

        if (telemetry.expert_details_omitted || telemetry.decode_details_omitted) {
            GGML_LOG("moe-cache-telemetry: detail limit reached; expert_percentiles_omitted=%d (-1 means unavailable) decode_rankings_omitted=%d; reset stats more often for full detail\n",
                telemetry.expert_details_omitted, telemetry.decode_details_omitted);
        }

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
}

ggml_cuda_moe_grouped_debug_telemetry ggml_cuda_moe_grouped_context::log_and_reset_legacy_stats() {
    moe_cache_telemetry aggregate;
    auto & owners = moe_cache_owner_telemetry_state();
    {
        std::lock_guard<std::mutex> owner_registry_lock(owners.mutex);
        moe_cache_telemetry retired = std::move(owners.retired);
        owners.retired = {};
        moe_cache_add_telemetry(aggregate, std::move(retired));
        for (ggml_cuda_moe_grouped_context * owner : owners.active) {
            std::lock_guard<std::mutex> lifecycle_lock(owner->impl_->resource_lifecycle_mutex);
            uint32_t registered_groups = 0;
            {
                std::lock_guard<std::mutex> owner_lock(owner->impl_->mutex);
                registered_groups = owner->impl_->state.accepted ? static_cast<uint32_t>(owner->impl_->table.groups.size()) :
                    owner->impl_->draining ? owner->impl_->draining_registered_groups : 0;
                for (auto & entry : owner->impl_->legacy_records) {
                    moe_cache_capture_telemetry(aggregate, entry.second->cache, true);
                }
            }
            moe_cache_add_telemetry(aggregate, owner->impl_->take_owner_telemetry(registered_groups));
        }
    }
    for (int phase = 0; phase < 2; ++phase) {
        ggml_cuda_moe_add_phase_stats(aggregate.phase_stats[phase], moe_cache_take_op_stats(g_moe_cache_op_stats[phase], true));
    }
    const ggml_cuda_moe_grouped_debug_telemetry grouped = aggregate.grouped;
    moe_cache_log_telemetry(std::move(aggregate));
    return grouped;
}

extern "C"
void ggml_backend_cuda_moe_log_and_reset_stats(void) {
    ggml_cuda_moe_grouped_context::log_and_reset_legacy_stats();
}
