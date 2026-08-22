#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

enum llama_kv_cache_component_role {
    LLAMA_KV_CACHE_COMPONENT_UNKNOWN,
    LLAMA_KV_CACHE_COMPONENT_STANDARD_K,
    LLAMA_KV_CACHE_COMPONENT_STANDARD_V,
    LLAMA_KV_CACHE_COMPONENT_STANDARD_K_TAIL,
    LLAMA_KV_CACHE_COMPONENT_STANDARD_V_TAIL,
    LLAMA_KV_CACHE_COMPONENT_KVARN_K_RECORDS,
    LLAMA_KV_CACHE_COMPONENT_KVARN_V_RECORDS,
    LLAMA_KV_CACHE_COMPONENT_KVARN_K_STAGE,
    LLAMA_KV_CACHE_COMPONENT_KVARN_V_STAGE,
    LLAMA_KV_CACHE_COMPONENT_KVARN_K_TAIL,
    LLAMA_KV_CACHE_COMPONENT_KVARN_V_TAIL,
};

// Typed adapter between cache-owned component roles and upstream's tensor-name
// split callback. Persistent payload is split along complete KV heads: standard
// rows and exact tails use axis 0, while KVarN records/stages use their explicit
// sliced-head axis 1.
struct llama_kv_cache_component {
    bool valid;
    llama_kv_cache_component_role role;
    uint32_t layer_id;
    int split_axis;
};

llama_kv_cache_component llama_kv_cache_component_from_name(const std::string & name);

// Return per-device element counts using the cumulative-ratio convention used
// by tensor-parallel placement. Non-final boundaries are rounded down to the
// requested granularity; empty shards and a shorter final remainder are valid.
// Throws std::invalid_argument for invalid dimensions, granularity, or weights.
std::vector<int64_t> llama_tensor_split_counts(
        int64_t n_elements,
        const std::vector<float> & weights,
        int64_t granularity);

enum llama_kv_gpu_window_config_error {
    LLAMA_KV_GPU_WINDOW_CONFIG_OK,
    LLAMA_KV_GPU_WINDOW_CONFIG_TOO_SMALL,
    LLAMA_KV_GPU_WINDOW_CONFIG_NOT_SMALLER_THAN_CONTEXT,
    LLAMA_KV_GPU_WINDOW_CONFIG_CONTEXT_TYPE,
    LLAMA_KV_GPU_WINDOW_CONFIG_ATTENTION_LAYOUT,
    LLAMA_KV_GPU_WINDOW_CONFIG_KV_OFFLOAD,
    LLAMA_KV_GPU_WINDOW_CONFIG_PINNED_HOST,
    LLAMA_KV_GPU_WINDOW_CONFIG_OP_OFFLOAD,
    LLAMA_KV_GPU_WINDOW_CONFIG_GPU_LAYERS,
    LLAMA_KV_GPU_WINDOW_CONFIG_SEQUENCE_COUNT,
    LLAMA_KV_GPU_WINDOW_CONFIG_FLASH_ATTN,
    LLAMA_KV_GPU_WINDOW_CONFIG_CACHE_TYPES,
    LLAMA_KV_GPU_WINDOW_CONFIG_KVARN,
    LLAMA_KV_GPU_WINDOW_CONFIG_PRECISION_TAIL,
    LLAMA_KV_GPU_WINDOW_CONFIG_TENSOR_SPLIT,
};

struct llama_kv_gpu_window_requirements {
    uint32_t requested_tokens;
    uint32_t n_ctx;
    uint32_t n_seq_max;
    uint32_t kv_gpu_layers;
    bool default_context;
    bool standard_attention;
    bool kv_offload;
    bool pinned_host;
    bool op_offload;
    bool flash_attn;
    bool q8_0_kv;
    bool kvarn_disabled;
    bool precision_tail_disabled;
    bool tensor_split;
};

llama_kv_gpu_window_config_error llama_kv_gpu_window_validate_config(
        const llama_kv_gpu_window_requirements & requirements);

const char * llama_kv_gpu_window_config_error_string(
        llama_kv_gpu_window_config_error error);

enum llama_kv_gpu_window_device_error {
    LLAMA_KV_GPU_WINDOW_DEVICE_OK,
    LLAMA_KV_GPU_WINDOW_DEVICE_NO_ATTENTION_LAYER,
    LLAMA_KV_GPU_WINDOW_DEVICE_NOT_CUDA,
    LLAMA_KV_GPU_WINDOW_DEVICE_NOT_PRIMARY,
    LLAMA_KV_GPU_WINDOW_DEVICE_MULTIPLE,
};

struct llama_kv_gpu_window_device_placement {
    uintptr_t identity;
    size_t registry_index;
    bool is_cuda;
};

llama_kv_gpu_window_device_error llama_kv_gpu_window_validate_devices(
        const std::vector<llama_kv_gpu_window_device_placement> & devices);

const char * llama_kv_gpu_window_device_error_string(
        llama_kv_gpu_window_device_error error);
