#pragma once

#include <cstdint>

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

ggml_moe_caller_status ggml_moe_caller_run(const ggml_moe_caller_schedule & schedule, ggml_moe_caller_sample & sample);
