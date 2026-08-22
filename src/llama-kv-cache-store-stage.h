#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>

enum class llama_kv_store_stage_side {
    K,
    V,
};

struct llama_kv_store_stage_key {
    llama_kv_store_stage_side side;
    ggml_backend_buffer_type_t buft;
    ggml_type type;
    int64_t n_embd;

    bool operator==(const llama_kv_store_stage_key & other) const {
        return side == other.side &&
                buft == other.buft &&
                type == other.type &&
                n_embd == other.n_embd;
    }
};
