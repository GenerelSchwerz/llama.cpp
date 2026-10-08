#pragma once

#include "ggml-backend-moe.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>
#include <vector>

class ggml_moe_source_body;

struct ggml_moe_profile_bank_statistics {
    const uint64_t * counts = nullptr;
    uint64_t observations = 0;
    uint64_t payload_bytes = 0;
    uint32_t n_experts = 0;
    const double * scores = nullptr;
};

GGML_API bool ggml_moe_source_statistics_valid(const ggml_backend_moe_source_statistics_v1 * statistics, uint32_t count);
GGML_API bool ggml_moe_source_scores_valid(const ggml_backend_moe_source_statistics_v1 * statistics, const double * const * scores, uint32_t count);
GGML_API bool ggml_moe_source_rank_statistics(const std::vector<ggml_moe_profile_bank_statistics> & banks, std::vector<int32_t> & ranks);

constexpr bool ggml_moe_source_tensor_byte_comparable =
    std::has_unique_object_representations<decltype(ggml_tensor::buffer)>::value &&
    std::has_unique_object_representations<decltype(ggml_tensor::op)>::value &&
    std::has_unique_object_representations<decltype(ggml_tensor::flags)>::value &&
    std::has_unique_object_representations<decltype(ggml_tensor::view_src)>::value &&
    std::has_unique_object_representations<decltype(ggml_tensor::view_offs)>::value &&
    std::has_unique_object_representations<decltype(ggml_tensor::data)>::value &&
    offsetof(ggml_tensor, ne) == offsetof(ggml_tensor, buffer) + sizeof(ggml_tensor::buffer) &&
    offsetof(ggml_tensor, nb) == offsetof(ggml_tensor, ne) + sizeof(ggml_tensor::ne) &&
    offsetof(ggml_tensor, op) == offsetof(ggml_tensor, nb) + sizeof(ggml_tensor::nb) &&
    offsetof(ggml_tensor, op_params) == offsetof(ggml_tensor, op) + sizeof(ggml_tensor::op) &&
    offsetof(ggml_tensor, flags) == offsetof(ggml_tensor, op_params) + sizeof(ggml_tensor::op_params) &&
    offsetof(ggml_tensor, src) == offsetof(ggml_tensor, flags) + sizeof(ggml_tensor::flags) &&
    offsetof(ggml_tensor, view_src) == offsetof(ggml_tensor, src) + sizeof(ggml_tensor::src) &&
    offsetof(ggml_tensor, view_offs) == offsetof(ggml_tensor, view_src) + sizeof(ggml_tensor::view_src) &&
    offsetof(ggml_tensor, data) == offsetof(ggml_tensor, view_offs) + sizeof(ggml_tensor::view_offs);

constexpr size_t ggml_moe_source_tensor_metadata_bytes = offsetof(ggml_tensor, data) + sizeof(ggml_tensor::data) - offsetof(ggml_tensor, buffer);

inline bool ggml_moe_source_tensor_matches(const ggml_tensor & current, const ggml_tensor & previous) {
    if constexpr (ggml_moe_source_tensor_byte_comparable) {
        // Compare the same fields in one range when the layout has no padding.
        if (current.type != previous.type || memcmp(reinterpret_cast<const char *>(&current) + offsetof(ggml_tensor, buffer),
                reinterpret_cast<const char *>(&previous) + offsetof(ggml_tensor, buffer), ggml_moe_source_tensor_metadata_bytes)) { return false; }
    } else {
        if (current.type != previous.type || current.op != previous.op || current.flags != previous.flags ||
                current.data != previous.data || current.buffer != previous.buffer || current.view_src != previous.view_src ||
                current.view_offs != previous.view_offs || memcmp(current.ne, previous.ne, sizeof(current.ne)) ||
                memcmp(current.nb, previous.nb, sizeof(current.nb)) || memcmp(current.src, previous.src, sizeof(current.src)) ||
                memcmp(current.op_params, previous.op_params, sizeof(current.op_params))) { return false; }
    }
    return true;
}

struct ggml_moe_source_role {
    uint32_t bank = 0;
    size_t offset = 0;
    size_t row_stride = 0;
    ggml_type type = GGML_TYPE_COUNT;
};

struct ggml_moe_source_bank {
    ggml_backend_moe_cpu_region_source_v1 source = {};
    ggml_tensor metadata = {};
    size_t expert_bytes = 0;
};

struct ggml_moe_source_expert {
    ggml_backend_moe_hybrid_region_v1 region = {};
    std::vector<ggml_moe_source_bank> banks;
    std::unique_ptr<ggml_moe_source_body> body;
    ggml_moe_source_role roles[3];
    uint32_t input_width = 0;
    uint32_t hidden_width = 0;
    uint32_t output_width = 0;
    bool routed_operation = false, generic_body = false;
    uint64_t cpu_graph_uid = 0;
    uint64_t cpu_graph_generation = 0;
    uint64_t cpu_source_generation = 0;
    ggml_backend_moe_cpu_prepared_region_v1_t cpu_region = 0;
};

enum class ggml_moe_source_effect : uint32_t {
    metadata,
    private_value,
    owner_write,
};

struct ggml_moe_source_operation {
    ggml_tensor * tensor = nullptr;
    const ggml_tensor * original = nullptr;
    ggml_moe_source_effect effect = ggml_moe_source_effect::private_value;
    size_t original_index = SIZE_MAX;
};

struct ggml_moe_source_layer {
    const ggml_moe_source_expert * expert = nullptr;
    ggml_tensor * activation = nullptr;
    ggml_tensor * ids = nullptr;
    ggml_tensor * output = nullptr;
    std::vector<ggml_moe_source_operation> prelude;
    std::vector<ggml_moe_source_operation> overlap;
};

struct ggml_moe_source_allocation_dependency {
    const ggml_tensor * tensor = nullptr;
    const ggml_tensor * until = nullptr;
};

struct ggml_moe_source_schedule_options {
    bool overlap_independent_ordinary = false;
    std::vector<ggml_moe_source_allocation_dependency> allocation_dependencies;
    bool (*bind_leaf)(void * context, const ggml_tensor * original, ggml_backend_buffer_t * buffer, void ** data) = nullptr;
    void * leaf_context = nullptr;
};

GGML_API bool ggml_moe_source_expert_prepare(
    const ggml_backend_moe_hybrid_region_v1 & region,
    const ggml_backend_moe_cpu_prepared_region_v1_t * cpu_regions,
    ggml_moe_source_expert & expert);

#ifdef GGML_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef GGML_BUILD
#            define GGML_MOE_SOURCE_PROGRAM_API __declspec(dllexport)
#        else
#            define GGML_MOE_SOURCE_PROGRAM_API __declspec(dllimport)
#        endif
#    else
#        define GGML_MOE_SOURCE_PROGRAM_API __attribute__((visibility("default")))
#    endif
#else
#    define GGML_MOE_SOURCE_PROGRAM_API
#endif

class GGML_MOE_SOURCE_PROGRAM_API ggml_moe_source_body {
public:
    static std::unique_ptr<ggml_moe_source_body> prepare(const ggml_backend_moe_cpu_region_query_v1 & query);
    std::unique_ptr<ggml_moe_source_body> compact(uint32_t route_capacity) const;
    bool bind_routes(uint32_t count);
    bool gather_input(size_t input, const void * source, size_t source_bytes,
        const uint32_t * rows, const uint32_t * routes, uint32_t count, void * destination, size_t destination_bytes) const;
    ggml_moe_source_body(const ggml_moe_source_body &) = delete;
    ggml_moe_source_body & operator=(const ggml_moe_source_body &) = delete;

    const std::vector<const ggml_tensor *> & nodes() const { return nodes_; }
    const std::vector<const ggml_tensor *> & dynamic_inputs() const { return dynamic_inputs_; }
    const std::vector<const ggml_tensor *> & live_outputs() const { return live_outputs_; }
    const std::vector<ggml_backend_moe_cpu_region_source_v1> & sources() const { return sources_; }

private:
    struct input_layout {
        int64_t rows = 0, routes = 0;
        size_t row_stride = 0, route_stride = 0, route_bytes = 0;
    };
    ggml_moe_source_body() = default;
    std::vector<ggml_tensor> tensors_;
    std::vector<const ggml_tensor *> nodes_, dynamic_inputs_, live_outputs_;
    std::vector<ggml_backend_moe_cpu_region_source_v1> sources_;
    std::vector<input_layout> input_layouts_;
    std::vector<uint8_t> route_dimensions_;
    size_t ids_input_ = SIZE_MAX;
    uint32_t row_capacity_ = 0, routes_per_row_ = 0, route_capacity_ = 0;
    bool compact_ = false;
};

GGML_MOE_SOURCE_PROGRAM_API size_t ggml_moe_source_operation_group_size(
    const std::vector<ggml_moe_source_operation> & operations, size_t index);

class GGML_MOE_SOURCE_PROGRAM_API ggml_moe_source_program {
public:
    ggml_moe_source_program();
    ~ggml_moe_source_program();
    ggml_moe_source_program(const ggml_moe_source_program &) = delete;
    ggml_moe_source_program & operator=(const ggml_moe_source_program &) = delete;

    bool prepare(const ggml_cgraph * graph, const std::vector<const ggml_moe_source_expert *> & experts,
        ggml_backend_buffer_type_t buft);
    bool prepare(const ggml_cgraph * graph, const std::vector<const ggml_moe_source_expert *> & experts,
        ggml_backend_buffer_type_t buft, const ggml_moe_source_schedule_options & options);
    bool allocate();
    bool matches(const ggml_cgraph * graph) const;
    bool matches(const ggml_cgraph * graph, const ggml_graph_execution_certificate & certificate) const;
    ggml_tensor * find(const ggml_tensor * original) const;
    bool closed_cut(const ggml_tensor * const * cut, size_t count,
        const ggml_tensor * const * retained, size_t retained_count) const;
    const std::vector<ggml_moe_source_layer> & layers() const;
    const std::vector<ggml_moe_source_operation> & epilogue() const;
    const std::vector<const ggml_tensor *> & public_outputs() const;
    const ggml_graph_execution_certificate & certificate() const;
    size_t storage_bytes() const;
    size_t operation_count() const;
    size_t owner_write_count() const;
    size_t overlap_operation_count() const;
    size_t overlap_layer_count() const;

private:
    struct impl;
    std::unique_ptr<impl> state;
};

#undef GGML_MOE_SOURCE_PROGRAM_API
