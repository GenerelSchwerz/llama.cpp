#pragma once
#include "ggml.h"
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>

struct ggml_moe_fidelity_config {
    bool valid = true;
    bool reference = false;
    bool source_pool = false;
    unsigned pcie_num = 0;
    bool tune_misses = false;
    uint32_t keep_ranks = 0;
};

// One immutable selection is shared by the CPU and CUDA backends.
GGML_API const ggml_moe_fidelity_config & ggml_moe_fidelity_selection();
