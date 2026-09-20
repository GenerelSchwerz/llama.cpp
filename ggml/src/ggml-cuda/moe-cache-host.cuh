#pragma once

#include "ggml-backend-impl.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__GNUC__) && !defined(_WIN32)
#define GGML_CUDA_MOE_HOST_HIDDEN __attribute__((visibility("hidden")))
#else
#define GGML_CUDA_MOE_HOST_HIDDEN
#endif

struct GGML_CUDA_MOE_HOST_HIDDEN moe_host_budget;

struct GGML_CUDA_MOE_HOST_HIDDEN moe_host_source {
    moe_host_budget * owner;
    const char * data;
    size_t size;
    size_t expert_stride;
    bool read_only;
    const char * device_alias = nullptr;
};

struct GGML_CUDA_MOE_HOST_HIDDEN moe_host_range {
    uintptr_t begin;
    uintptr_t end;
    bool read_only = false;
};

struct GGML_CUDA_MOE_HOST_HIDDEN moe_host_copy_worker {
    ~moe_host_copy_worker();

    bool start(bool fail_for_test = false);

    bool try_submit(void (*task)(void *), void * task_data);

    void wait();

    void stop();

    std::mutex mutex;
    std::condition_variable ready_cv;
    std::condition_variable done_cv;
    std::once_flag start_once;
    std::thread thread;
    void (*function)(void *) = nullptr;
    void * data = nullptr;
    bool stopping = false;
    bool busy = false;
    bool ready = false;
    bool done = false;
    uint64_t submitted = 0;
};

struct GGML_CUDA_MOE_HOST_HIDDEN moe_host_budget {
    explicit moe_host_budget(size_t limit, bool automatic = false);

    ~moe_host_budget();

    void retain();
    void release();

    ggml_backend_buffer_type type = {};
    std::atomic<size_t> references{1};
    size_t limit;
    const bool automatic;
    size_t source_bytes = 0;
    size_t staging_reserved = 0;
    size_t staging_bytes = 0;
    size_t staging_optional_bytes = 0;
    size_t staging_peak = 0;
    bool configured = false;
    std::mutex mutex;
    std::vector<ggml_backend_buffer_t> backing;
    std::vector<moe_host_range> registered;
    std::unordered_map<const ggml_tensor *, moe_host_source> sources;
    std::atomic<uint64_t> materialized_bytes{0};
    moe_host_copy_worker copy_worker;
};

struct GGML_CUDA_MOE_HOST_HIDDEN moe_host_allocation {
    moe_host_allocation(moe_host_budget * owner, size_t base_size, size_t tile_stride = 0, uint32_t max_tiles = 1, bool optional = false, bool fail_growth_for_test = false, size_t cached_prefix_bytes = 0);

    ~moe_host_allocation();

    moe_host_allocation(const moe_host_allocation &) = delete;
    moe_host_allocation & operator=(const moe_host_allocation &) = delete;
    moe_host_budget * owner;
    size_t size;
    size_t pinned_size = 0;
    size_t optional_bytes = 0;
    uint32_t tiles = 1;
    void * data = nullptr;
    void * device_alias = nullptr;
    void * payload = nullptr;
    void * payload_alias = nullptr;
    bool split_payload = false;
};

GGML_CUDA_MOE_HOST_HIDDEN size_t moe_host_page_size();
GGML_CUDA_MOE_HOST_HIDDEN bool moe_host_round_size(size_t size, size_t & rounded);
GGML_CUDA_MOE_HOST_HIDDEN moe_host_budget * moe_host_budget_for(ggml_backend_buffer_type_t buft);
GGML_CUDA_MOE_HOST_HIDDEN bool moe_host_register(moe_host_budget & owner, const std::vector<moe_host_source *> & sources, bool require_identity, uint32_t group);
GGML_CUDA_MOE_HOST_HIDDEN bool moe_host_buffer_is_read_only(ggml_backend_buffer_t buffer);
GGML_CUDA_MOE_HOST_HIDDEN bool moe_host_buffer_auto_pin(ggml_backend_buffer_t buffer);

#undef GGML_CUDA_MOE_HOST_HIDDEN
