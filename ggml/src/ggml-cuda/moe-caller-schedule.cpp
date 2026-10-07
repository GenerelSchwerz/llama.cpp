#include "moe-caller-schedule.h"

#include <chrono>
#include <thread>

#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
#include <immintrin.h>
#endif

static uint64_t caller_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void caller_pause() {
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
    _mm_pause();
#elif defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
    __asm__ __volatile__("yield");
#else
    std::this_thread::yield();
#endif
}

ggml_moe_caller_status ggml_moe_caller_run(const ggml_moe_caller_schedule & schedule, ggml_moe_caller_sample & sample) {
    sample = {};
    for (uint32_t layer = 0; layer <= schedule.layers; ++layer) {
        if (schedule.canceled(schedule.context)) { return ggml_moe_caller_status::canceled; }
        if (caller_now_ns() >= schedule.deadline_ns) { return ggml_moe_caller_status::deadline; }
        if (schedule.segmented || layer == 0) {
            const uint64_t started = caller_now_ns();
            const bool launched = schedule.launch(schedule.context, layer);
            sample.launch_ns += caller_now_ns() - started;
            if (!launched) { return ggml_moe_caller_status::launch_failed; }
            ++sample.progress_probes;
            if (schedule.progress(schedule.context) == ggml_moe_caller_progress::failed) { return ggml_moe_caller_status::progress_failed; }
        }
        if (layer == schedule.layers) { break; }
        const uint64_t started = caller_now_ns();
        uint64_t last_probe = started;
        uint32_t spins = 0;
        while (!schedule.ready(schedule.context, layer)) {
            if (schedule.canceled(schedule.context)) { return ggml_moe_caller_status::canceled; }
            caller_pause();
            if ((++spins & 1023u) != 0) { continue; }
            const uint64_t now = caller_now_ns();
            if (now >= schedule.deadline_ns) { return ggml_moe_caller_status::deadline; }
            if (now - last_probe >= 2'000'000ull) {
                last_probe = now;
                ++sample.progress_probes;
                const auto progress = schedule.progress(schedule.context);
                if (progress == ggml_moe_caller_progress::failed) { return ggml_moe_caller_status::progress_failed; }
                if (progress == ggml_moe_caller_progress::complete && !schedule.ready(schedule.context, layer)) {
                    return ggml_moe_caller_status::missing_ready;
                }
            }
        }
        const bool served = schedule.service(schedule.context, layer);
        sample.observation_ns += caller_now_ns() - started;
        if (!served) { return ggml_moe_caller_status::service_failed; }
        ++sample.jobs;
    }
    return ggml_moe_caller_status::complete;
}
