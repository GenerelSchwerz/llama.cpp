#pragma once

#include <atomic>
#include <cstdint>

struct alignas(64) ggml_cuda_moe_source_flag {
    std::atomic<uint32_t> value{0};
    void store(uint32_t v) { value.store(v, std::memory_order_release); }
    uint32_t load() const { return value.load(std::memory_order_acquire); }
};
static_assert(sizeof(std::atomic<uint32_t>) == sizeof(uint32_t), "source flag size");
static_assert(alignof(std::atomic<uint32_t>) == alignof(uint32_t), "source flag alignment");
static_assert(std::atomic<uint32_t>::is_always_lock_free, "source flag lock freedom");

struct ggml_cuda_moe_source_control {
    ggml_cuda_moe_source_flag ready, plan, copied, cpu, stop;
    uint64_t epoch = 0;
    int32_t status = 0;
};

struct ggml_cuda_moe_source_runtime {
    uint32_t failed = 0;
    uint32_t joined = 0;
    uint64_t epoch = 0;
    uint32_t resident = 0, selected = 0;
    uint32_t kernel_layers = 0;
    uint64_t copy_jobs = 0, copy_bytes = 0;
};
