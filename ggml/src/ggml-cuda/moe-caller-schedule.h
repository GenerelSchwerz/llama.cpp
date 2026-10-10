#pragma once

#include <cstdint>
#include <thread>

#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
#include <immintrin.h>
#endif

enum class ggml_moe_caller_progress { pending, complete, failed };
enum class ggml_moe_caller_status { complete, canceled, deadline, launch_failed, progress_failed, missing_ready, service_failed };

struct ggml_moe_caller_schedule {
    void * context;
    uint32_t layers;
    bool segmented;
    uint64_t deadline_ns;
    bool (*launch)(void *, uint32_t);
    bool (*ready)(void *, uint32_t);
    bool (*canceled)(void *);
    ggml_moe_caller_progress (*progress)(void *);
    bool (*service)(void *, uint32_t);
};

struct ggml_moe_caller_sample {
    uint64_t launch_ns = 0;
    uint64_t observation_ns = 0;
    uint64_t jobs = 0;
    uint64_t progress_probes = 0;
};

inline void ggml_moe_caller_pause() {
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
    _mm_pause();
#elif defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
    __asm__ __volatile__("yield");
#else
    std::this_thread::yield();
#endif
}

ggml_moe_caller_status ggml_moe_caller_run(const ggml_moe_caller_schedule & schedule, ggml_moe_caller_sample & sample);
