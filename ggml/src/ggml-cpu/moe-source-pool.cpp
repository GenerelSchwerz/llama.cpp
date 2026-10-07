// Schedule adapted from MIT-licensed expert primitives; see moe-reference-LICENSE.
// See moe-reference-LICENSE. Geometry and arithmetic belong to prepared region lanes.
#include "moe-source-pool.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <pthread.h>
#endif
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#    include <immintrin.h>
#endif

namespace {
uint64_t profile_now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
void relax() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    _mm_pause();
#elif defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
    __asm__ volatile("yield" ::: "memory");
#else
    std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}
}

struct ggml_moe_source_pool {
    struct worker {
        ggml_moe_source_pool * pool = nullptr;
#if defined(_WIN32)
        HANDLE thread = nullptr;
#else
        pthread_t thread = {};
#endif
        bool started = false;
        bool sleeping = false;
        uint32_t sleep_epoch = 0;
    };
    uint32_t n_threads, n_workers;
    std::unique_ptr<worker[]> workers;
    alignas(64) std::atomic<uint64_t> head{0};
    alignas(64) std::atomic<uint32_t> done{0};
    alignas(64) std::atomic<uint32_t> parked{0};
    alignas(64) std::atomic<uint32_t> epoch{0};
    alignas(64) std::atomic<bool> stopped{false};
    alignas(64) std::atomic<uint32_t> sleepers{0};
    std::atomic<bool> suspended{true}, failed{false};
    std::mutex sleep_mutex, stop_mutex;
    std::condition_variable sleep_condition;
    ggml_moe_cpu_fidelity * body = nullptr;
    const uint32_t tasks;
    bool down = false;
    bool cpu_profile = getenv("GGML_MOE_CPU_PROFILE") && !strcmp(getenv("GGML_MOE_CPU_PROFILE"), "1");
    uint64_t profile_jobs = 0, profile_begin_ns = 0, profile_hidden_ns = 0;
    uint64_t profile_phase_ns[2][4] = {};

    explicit ggml_moe_source_pool(uint32_t n) : n_threads(n), n_workers(n - 1), workers(n_workers ? new worker[n_workers] : nullptr), tasks(3 * n) {}
    ~ggml_moe_source_pool() {
        stop();
        if (cpu_profile) {
            fprintf(stderr, "moe-cpu-pool-profile: jobs=%llu begin_ns=%llu hidden_ns=%llu gu_setup_ns=%llu gu_caller_ns=%llu gu_join_ns=%llu gu_park_ns=%llu down_setup_ns=%llu down_caller_ns=%llu down_join_ns=%llu down_park_ns=%llu\n",
                (unsigned long long) profile_jobs, (unsigned long long) profile_begin_ns, (unsigned long long) profile_hidden_ns,
                (unsigned long long) profile_phase_ns[0][0], (unsigned long long) profile_phase_ns[0][1],
                (unsigned long long) profile_phase_ns[0][2], (unsigned long long) profile_phase_ns[0][3],
                (unsigned long long) profile_phase_ns[1][0], (unsigned long long) profile_phase_ns[1][1],
                (unsigned long long) profile_phase_ns[1][2], (unsigned long long) profile_phase_ns[1][3]);
        }
    }

#if defined(_WIN32)
    static DWORD WINAPI entry(void * data) {
#else
    static void * entry(void * data) {
#endif
        auto & self = *static_cast<worker *>(data);
        self.pool->work(self);
#if defined(_WIN32)
        return 0;
#else
        return nullptr;
#endif
    }

    bool start() {
        for (uint32_t i = 0; i < n_workers; ++i) {
            workers[i].pool = this;
#if defined(_WIN32)
            workers[i].thread = CreateThread(nullptr, 0, entry, &workers[i], 0, nullptr);
            if (!workers[i].thread) { stop(); return false; }
#else
            if (pthread_create(&workers[i].thread, nullptr, entry, &workers[i]) != 0) { stop(); return false; }
#endif
            workers[i].started = true;
        }
        wait_parked();
        return true;
    }

    void publish() {
        epoch.fetch_add(1, std::memory_order_seq_cst);
        if (sleepers.load(std::memory_order_seq_cst)) {
            std::lock_guard<std::mutex> lock(sleep_mutex);
            sleep_condition.notify_all();
        }
    }

    void stop() noexcept {
        std::lock_guard<std::mutex> stop_lock(stop_mutex);
        if (stopped.exchange(true, std::memory_order_acq_rel)) { return; }
        publish();
        for (uint32_t i = 0; i < n_workers; ++i) {
            if (!workers[i].started) { continue; }
#if defined(_WIN32)
            if (WaitForSingleObject(workers[i].thread, INFINITE) != WAIT_OBJECT_0) { std::terminate(); }
            if (!CloseHandle(workers[i].thread)) { std::terminate(); }
#else
            if (pthread_join(workers[i].thread, nullptr) != 0) { std::terminate(); }
#endif
            workers[i].started = false;
        }
    }

    void wait_parked() {
        while (parked.load(std::memory_order_acquire) != n_workers) { relax(); }
    }

    void suspend() {
        if (stopped.load(std::memory_order_acquire)) { return; }
        wait_parked();
        suspended.store(true, std::memory_order_seq_cst);
        std::unique_lock<std::mutex> lock(sleep_mutex);
        sleep_condition.wait(lock, [&] {
            if (sleepers.load(std::memory_order_acquire) != n_workers) { return false; }
            const uint32_t e = epoch.load(std::memory_order_acquire);
            for (uint32_t i = 0; i < n_workers; ++i) {
                if (!workers[i].sleeping || workers[i].sleep_epoch != e) { return false; }
            }
            return true;
        });
    }

    int32_t claim(uint32_t seen) {
        uint64_t h = head.load(std::memory_order_acquire);
        for (;;) {
            if (uint32_t(h >> 32) != seen) { return -1; }
            const uint32_t n = uint32_t(h >> 16) & 0xffffu, i = uint32_t(h) & 0xffffu;
            if (i >= n) { return -1; }
            if (head.compare_exchange_weak(h, h + 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
                return int32_t(i);
            }
        }
    }

    void drain(uint32_t seen) {
        for (;;) {
            const int32_t task = claim(seen);
            if (task < 0) { return; }
            try { ggml_moe_cpu_fidelity_task(body, down, uint32_t(task), tasks); }
            catch (...) { failed.store(true, std::memory_order_relaxed); }
            done.fetch_add(1, std::memory_order_release);
        }
    }

    void work(worker & self) {
        uint32_t seen = 0;
        parked.fetch_add(1, std::memory_order_acq_rel);
        for (;;) {
            const auto parked_at = std::chrono::steady_clock::now();
            uint32_t spins = 0;
            while (epoch.load(std::memory_order_acquire) == seen) {
                if (stopped.load(std::memory_order_relaxed)) { return; }
                relax();
                if ((++spins & 1023u) != 0) { continue; }
                if (!suspended.load(std::memory_order_acquire) &&
                        std::chrono::steady_clock::now() - parked_at < std::chrono::milliseconds(20)) { continue; }
                std::unique_lock<std::mutex> lock(sleep_mutex);
                self.sleeping = true;
                self.sleep_epoch = seen;
                sleepers.fetch_add(1, std::memory_order_seq_cst);
                if (suspended.load(std::memory_order_acquire)) { sleep_condition.notify_all(); }
                sleep_condition.wait(lock, [&] {
                    return epoch.load(std::memory_order_seq_cst) != seen || stopped.load(std::memory_order_relaxed);
                });
                sleepers.fetch_sub(1, std::memory_order_relaxed);
                self.sleeping = false;
            }
            if (stopped.load(std::memory_order_acquire)) { return; }
            seen = epoch.load(std::memory_order_acquire);
            parked.fetch_sub(1, std::memory_order_acq_rel);
            drain(seen);
            parked.fetch_add(1, std::memory_order_acq_rel);
        }
    }

    bool phase(ggml_moe_cpu_fidelity * value, bool is_down) {
        auto at = cpu_profile ? profile_now() : 0;
        const auto record = [&](uint32_t stage) {
            if (cpu_profile) {
                const auto next = profile_now();
                profile_phase_ns[is_down][stage] += next - at; at = next;
            }
        };
        wait_parked();
        body = value;
        down = is_down;
        done.store(0, std::memory_order_relaxed);
        failed.store(false, std::memory_order_relaxed);
        suspended.store(false, std::memory_order_release);
        const uint32_t e = epoch.load(std::memory_order_relaxed) + 1;
        head.store((uint64_t(e) << 32) | (uint64_t(tasks) << 16), std::memory_order_release);
        publish();
        record(0);
        drain(e);
        record(1);
        while (done.load(std::memory_order_acquire) < tasks) { relax(); }
        record(2);
        wait_parked();
        record(3);
        body = nullptr;
        return !failed.load(std::memory_order_relaxed);
    }
};

uint64_t ggml_moe_source_pool_bytes(uint32_t n_threads) {
    if (!n_threads || n_threads > GGML_MAX_N_THREADS) { return 0; }
    return sizeof(ggml_moe_source_pool) + uint64_t(n_threads - 1) * sizeof(ggml_moe_source_pool::worker);
}

ggml_moe_source_pool * ggml_moe_source_pool_create(uint32_t n_threads) {
    if (!ggml_moe_source_pool_bytes(n_threads)) { return nullptr; }
    try {
        auto pool = std::unique_ptr<ggml_moe_source_pool>(new ggml_moe_source_pool(n_threads));
        if (!pool->start()) { return nullptr; }
        return pool.release();
    } catch (...) { return nullptr; }
}
void ggml_moe_source_pool_suspend(ggml_moe_source_pool * pool) { if (pool) { pool->suspend(); } }
void ggml_moe_source_pool_stop(ggml_moe_source_pool * pool) { if (pool) { pool->stop(); } }
void ggml_moe_source_pool_free(ggml_moe_source_pool * pool) { delete pool; }

ggml_status ggml_moe_source_pool_run(ggml_moe_source_pool * pool, ggml_moe_cpu_fidelity * body,
        const ggml_backend_moe_cpu_region_binding_v1 * binding, const ggml_backend_moe_cpu_dynamic_input_v1 * input,
        ggml_tensor * const * outputs, ggml_abort_callback abort, void * abort_data,
        ggml_backend_moe_cpu_test_hook_v1_t hook, void * hook_data,
        const ggml_backend_moe_cpu_output_v1 * private_outputs, const uint32_t * destinations) {
    if (!pool || pool->stopped.load(std::memory_order_acquire)) { return GGML_STATUS_FAILED; }
    auto at = pool->cpu_profile ? profile_now() : 0;
    auto status = ggml_moe_cpu_fidelity_begin(body, binding, input, outputs, abort, abort_data, hook, hook_data,
        private_outputs, destinations);
    if (pool->cpu_profile) { pool->profile_begin_ns += profile_now() - at; ++pool->profile_jobs; }
    if (status != GGML_STATUS_SUCCESS) { return status; }
    if (!pool->phase(body, false)) { return GGML_STATUS_FAILED; }
    ggml_moe_cpu_fidelity_after_gu(body);
    status = ggml_moe_cpu_fidelity_result(body);
    if (status != GGML_STATUS_SUCCESS) { return status; }
    at = pool->cpu_profile ? profile_now() : 0;
    ggml_moe_cpu_fidelity_quantize_hidden(body);
    if (pool->cpu_profile) { pool->profile_hidden_ns += profile_now() - at; }
    status = ggml_moe_cpu_fidelity_result(body);
    if (status != GGML_STATUS_SUCCESS) { return status; }
    if (!pool->phase(body, true)) { return GGML_STATUS_FAILED; }
    return ggml_moe_cpu_fidelity_result(body);
}
