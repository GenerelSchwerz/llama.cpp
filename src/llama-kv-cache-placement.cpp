#include "llama-kv-cache-placement.h"

#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace {

bool parse_cache_component_suffix(
        const std::string & name,
        const char * prefix,
        bool allow_stream,
        uint32_t & layer_id) {
    const size_t prefix_size = std::strlen(prefix);
    if (name.compare(0, prefix_size, prefix) != 0 || prefix_size == name.size()) {
        return false;
    }
    size_t pos = prefix_size;
    uint64_t parsed = 0;
    const size_t digits_begin = pos;
    while (pos < name.size() && std::isdigit(static_cast<unsigned char>(name[pos]))) {
        parsed = parsed*10 + uint64_t(name[pos] - '0');
        if (parsed > UINT32_MAX) {
            return false;
        }
        ++pos;
    }
    if (pos == digits_begin) {
        return false;
    }
    if (pos != name.size()) {
        if (!allow_stream || name.compare(pos, 2, "_s") != 0) {
            return false;
        }
        pos += 2;
        const size_t stream_begin = pos;
        while (pos < name.size() && std::isdigit(static_cast<unsigned char>(name[pos]))) {
            ++pos;
        }
        if (pos == stream_begin || pos != name.size()) {
            return false;
        }
    }
    layer_id = uint32_t(parsed);
    return true;
}

} // namespace

llama_kv_cache_component llama_kv_cache_component_from_name(const std::string & name) {
    struct component_pattern {
        const char * prefix;
        llama_kv_cache_component_role role;
        int split_axis;
        bool allow_stream;
    };
    static const component_pattern patterns[] = {
        { "cache_kvarn_k_records_l", LLAMA_KV_CACHE_COMPONENT_KVARN_K_RECORDS, 1, true  },
        { "cache_kvarn_v_records_l", LLAMA_KV_CACHE_COMPONENT_KVARN_V_RECORDS, 1, true  },
        { "cache_kvarn_k_stage_l",   LLAMA_KV_CACHE_COMPONENT_KVARN_K_STAGE,   1, true  },
        { "cache_kvarn_v_stage_l",   LLAMA_KV_CACHE_COMPONENT_KVARN_V_STAGE,   1, true  },
        { "cache_kvarn_k_tail_l",    LLAMA_KV_CACHE_COMPONENT_KVARN_K_TAIL,    0, false },
        { "cache_kvarn_v_tail_l",    LLAMA_KV_CACHE_COMPONENT_KVARN_V_TAIL,    0, false },
        { "cache_k_tail_l",          LLAMA_KV_CACHE_COMPONENT_STANDARD_K_TAIL, 0, false },
        { "cache_v_tail_l",          LLAMA_KV_CACHE_COMPONENT_STANDARD_V_TAIL, 0, false },
        { "cache_k_l",               LLAMA_KV_CACHE_COMPONENT_STANDARD_K,      0, false },
        { "cache_v_l",               LLAMA_KV_CACHE_COMPONENT_STANDARD_V,      0, false },
    };
    for (const auto & pattern : patterns) {
        uint32_t layer_id = 0;
        if (parse_cache_component_suffix(name, pattern.prefix, pattern.allow_stream, layer_id)) {
            return { true, pattern.role, layer_id, pattern.split_axis };
        }
    }
    return { false, LLAMA_KV_CACHE_COMPONENT_UNKNOWN, 0, -1 };
}

std::vector<int64_t> llama_tensor_split_counts(
        int64_t n_elements,
        const std::vector<float> & weights,
        int64_t granularity) {
    if (n_elements < 0) {
        throw std::invalid_argument("tensor split element count must be non-negative");
    }
    if (granularity <= 0) {
        throw std::invalid_argument("tensor split granularity must be positive");
    }
    if (weights.empty()) {
        throw std::invalid_argument("tensor split requires at least one device");
    }
    double total = 0.0;
    for (float weight : weights) {
        if (!std::isfinite(weight) || weight < 0.0f) {
            throw std::invalid_argument("tensor split weights must be finite and non-negative");
        }
        total += weight;
    }
    std::vector<int64_t> result(weights.size(), 0);
    int64_t low = 0;
    double cumulative = 0.0;
    for (size_t i = 0; i < weights.size(); ++i) {
        cumulative += weights[i];
        const bool is_last = i + 1 == weights.size();
        int64_t high = is_last ? n_elements :
                total == 0.0 ?
                    int64_t(static_cast<long double>(n_elements)*(i + 1)/weights.size()) :
                    int64_t(static_cast<long double>(n_elements)*cumulative/total);
        if (!is_last) {
            high -= high % granularity;
        }
        result[i] = high - low;
        low = high;
    }
    return result;
}

llama_kv_gpu_window_config_error llama_kv_gpu_window_validate_config(
        const llama_kv_gpu_window_requirements & requirements) {
    if (requirements.requested_tokens == 0) {
        return LLAMA_KV_GPU_WINDOW_CONFIG_OK;
    }
    if (requirements.requested_tokens < 256) {
        return LLAMA_KV_GPU_WINDOW_CONFIG_TOO_SMALL;
    }
    if (requirements.requested_tokens >= requirements.n_ctx) {
        return LLAMA_KV_GPU_WINDOW_CONFIG_NOT_SMALLER_THAN_CONTEXT;
    }
    if (!requirements.default_context) {
        return LLAMA_KV_GPU_WINDOW_CONFIG_CONTEXT_TYPE;
    }
    if (!requirements.standard_attention) {
        return LLAMA_KV_GPU_WINDOW_CONFIG_ATTENTION_LAYOUT;
    }
    if (requirements.kv_offload) {
        return LLAMA_KV_GPU_WINDOW_CONFIG_KV_OFFLOAD;
    }
    if (!requirements.pinned_host) {
        return LLAMA_KV_GPU_WINDOW_CONFIG_PINNED_HOST;
    }
    if (!requirements.op_offload) {
        return LLAMA_KV_GPU_WINDOW_CONFIG_OP_OFFLOAD;
    }
    if (requirements.kv_gpu_layers != 0) {
        return LLAMA_KV_GPU_WINDOW_CONFIG_GPU_LAYERS;
    }
    if (requirements.n_seq_max != 1) {
        return LLAMA_KV_GPU_WINDOW_CONFIG_SEQUENCE_COUNT;
    }
    if (!requirements.flash_attn) {
        return LLAMA_KV_GPU_WINDOW_CONFIG_FLASH_ATTN;
    }
    if (!requirements.q8_0_kv) {
        return LLAMA_KV_GPU_WINDOW_CONFIG_CACHE_TYPES;
    }
    if (!requirements.kvarn_disabled) {
        return LLAMA_KV_GPU_WINDOW_CONFIG_KVARN;
    }
    if (!requirements.precision_tail_disabled) {
        return LLAMA_KV_GPU_WINDOW_CONFIG_PRECISION_TAIL;
    }
    if (requirements.tensor_split) {
        return LLAMA_KV_GPU_WINDOW_CONFIG_TENSOR_SPLIT;
    }
    return LLAMA_KV_GPU_WINDOW_CONFIG_OK;
}

const char * llama_kv_gpu_window_config_error_string(
        llama_kv_gpu_window_config_error error) {
    switch (error) {
        case LLAMA_KV_GPU_WINDOW_CONFIG_OK:
            return nullptr;
        case LLAMA_KV_GPU_WINDOW_CONFIG_TOO_SMALL:
            return "the requested window must contain at least 256 rows";
        case LLAMA_KV_GPU_WINDOW_CONFIG_NOT_SMALLER_THAN_CONTEXT:
            return "the requested window must be smaller than the resolved context size";
        case LLAMA_KV_GPU_WINDOW_CONFIG_CONTEXT_TYPE:
            return "only the default target context is supported";
        case LLAMA_KV_GPU_WINDOW_CONFIG_ATTENTION_LAYOUT:
            return "only standard non-SWA attention is supported";
        case LLAMA_KV_GPU_WINDOW_CONFIG_KV_OFFLOAD:
            return "--no-kv-offload is required";
        case LLAMA_KV_GPU_WINDOW_CONFIG_PINNED_HOST:
            return "--kv-cpu-pinned is required";
        case LLAMA_KV_GPU_WINDOW_CONFIG_OP_OFFLOAD:
            return "operation offload must remain enabled";
        case LLAMA_KV_GPU_WINDOW_CONFIG_GPU_LAYERS:
            return "--kv-gpu-layers must be 0";
        case LLAMA_KV_GPU_WINDOW_CONFIG_SEQUENCE_COUNT:
            return "exactly one sequence/stream is required";
        case LLAMA_KV_GPU_WINDOW_CONFIG_FLASH_ATTN:
            return "--flash-attn on is required";
        case LLAMA_KV_GPU_WINDOW_CONFIG_CACHE_TYPES:
            return "both standard cache types must be Q8_0";
        case LLAMA_KV_GPU_WINDOW_CONFIG_KVARN:
            return "KVarN must be disabled";
        case LLAMA_KV_GPU_WINDOW_CONFIG_PRECISION_TAIL:
            return "a separate precision-tail request cannot be combined with the GPU window";
        case LLAMA_KV_GPU_WINDOW_CONFIG_TENSOR_SPLIT:
            return "tensor-split model placement is not supported";
    }
    return "unknown GPU KV window configuration error";
}

llama_kv_gpu_window_device_error llama_kv_gpu_window_validate_devices(
        const std::vector<llama_kv_gpu_window_device_placement> & devices) {
    if (devices.empty()) {
        return LLAMA_KV_GPU_WINDOW_DEVICE_NO_ATTENTION_LAYER;
    }
    const uintptr_t expected = devices.front().identity;
    for (const auto & device : devices) {
        if (!device.is_cuda) {
            return LLAMA_KV_GPU_WINDOW_DEVICE_NOT_CUDA;
        }
        if (device.registry_index != 0) {
            return LLAMA_KV_GPU_WINDOW_DEVICE_NOT_PRIMARY;
        }
        if (device.identity != expected) {
            return LLAMA_KV_GPU_WINDOW_DEVICE_MULTIPLE;
        }
    }
    return LLAMA_KV_GPU_WINDOW_DEVICE_OK;
}

const char * llama_kv_gpu_window_device_error_string(
        llama_kv_gpu_window_device_error error) {
    switch (error) {
        case LLAMA_KV_GPU_WINDOW_DEVICE_OK:
            return nullptr;
        case LLAMA_KV_GPU_WINDOW_DEVICE_NO_ATTENTION_LAYER:
            return "no attention-layer device was found";
        case LLAMA_KV_GPU_WINDOW_DEVICE_NOT_CUDA:
            return "every attention layer must execute on CUDA";
        case LLAMA_KV_GPU_WINDOW_DEVICE_NOT_PRIMARY:
            return "the first prototype requires CUDA registry device 0";
        case LLAMA_KV_GPU_WINDOW_DEVICE_MULTIPLE:
            return "all attention layers must use the same CUDA device";
    }
    return "unknown GPU KV window device-placement error";
}
