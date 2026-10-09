#include "../src/llama-context.h"
#include "../src/llama-graph.h"
#include "../src/llama-model.h"
#include "ggml-cuda/moe-cache-host.cuh"
#include "ggml-cuda/moe-fidelity.cuh"
#include "ggml-cuda/moe-fidelity-executor.cuh"
#include "ggml-cuda/fattn.cuh"
#include "ggml-cuda/top-k.cuh"
#include "ggml-cuda/softmax.cuh"
#include "ggml-cuda/staged-input.cuh"
#include "ggml-cpu/moe-fidelity.h"
#include "ggml-cuda/moe-source-core.cuh"
#include "ggml-moe-source-program.h"
#include "ggml-impl.h"
#include "moe-fidelity-config.h"
#include "ggml-staged-input.h"
#include "test-moe-cache.h"

#include <cuda.h>
#include <cudaTypedefs.h>

#include <condition_variable>
#include <limits>
#include <mutex>
#include <unordered_set>

struct body_route_partition {
    int pattern;
    bool complement = false;
    const ggml_tensor * node = nullptr;
    size_t calls = 0;
    const std::vector<uint8_t> * expert_mask = nullptr;
};

static bool body_route_selected(const body_route_partition & partition, int64_t row, int64_t column, int32_t expert) {
    if (partition.expert_mask) {
        CHECK(expert >= 0 && size_t(expert) < partition.expert_mask->size());
        return (*partition.expert_mask)[expert] != 0;
    }
    switch (partition.pattern) {
        case 0: return false;
        case 1: return true;
        case 2: return column % 2 == 0;
        case 3: return expert % 2 == 0;
        case 4: return row % 2 == 0;
        default: return (row + column + expert) % 3 == 0;
    }
}

static bool filter_body_route(const ggml_tensor * node, int64_t row, int64_t column, int32_t expert, void * data) {
    auto & partition = *static_cast<body_route_partition *>(data);
    CHECK(node == partition.node);
    ++partition.calls;
    return body_route_selected(partition, row, column, expert) != partition.complement;
}

static void complete_body_projection_gpu(ggml_backend_t backend, ggml_tensor * node, const body_route_partition & partition, bool per_expert = false, bool mapped_quant = false) {
    const auto * input = node->src[1];
    const auto * ids = node->src[2];
    const auto * weight = node->src[0];
    CHECK(input->type == GGML_TYPE_F32 && node->type == GGML_TYPE_F32 && input->ne[3] == 1);
    std::vector<int32_t> experts;
    std::vector<int32_t> original_experts;
    std::vector<uint8_t *> destinations;
    std::vector<size_t> logical_routes;
    std::vector<float> activation;
    for (int64_t row = 0; row < ids->ne[1]; ++row) {
        for (int64_t column = 0; column < ids->ne[0]; ++column) {
            int32_t expert;
            memcpy(&expert, static_cast<const uint8_t *>(ids->data) + row * ids->nb[1] + column * ids->nb[0], sizeof(expert));
            original_experts.push_back(expert);
            if (body_route_selected(partition, row, column, expert)) { continue; }
            experts.push_back(expert);
            destinations.push_back(static_cast<uint8_t *>(node->data) + row * node->nb[2] + column * node->nb[1]);
            logical_routes.push_back(row * ids->ne[0] + column);
            const auto * values = static_cast<const uint8_t *>(input->data) + row * input->nb[2] + (column % input->ne[1]) * input->nb[1];
            for (int64_t feature = 0; feature < input->ne[0]; ++feature) {
                float value; memcpy(&value, values + feature * input->nb[0], sizeof(value)); activation.push_back(value);
            }
        }
    }
    if (experts.empty()) { return; }
    if (per_expert) {
        std::vector<std::vector<size_t>> cohorts(weight->ne[2]);
        for (size_t route = 0; route < experts.size(); ++route) { cohorts[experts[route]].push_back(route); }
        const size_t tensors = 1 + 3 * cohorts.size();
        ggml_context_ptr context(ggml_init({tensors * ggml_tensor_overhead() + ggml_graph_overhead_custom(tensors, false), nullptr, true}));
        auto * bank = ggml_new_tensor_3d(context.get(), weight->type, weight->ne[0], weight->ne[1], weight->ne[2]);
        auto * graph = ggml_new_graph_custom(context.get(), tensors, false);
        std::vector<ggml_tensor *> inputs(cohorts.size()), outputs(cohorts.size());
        for (size_t expert = 0; expert < cohorts.size(); ++expert) {
            if (cohorts[expert].empty()) { continue; }
            auto * slice = ggml_view_2d(context.get(), bank, bank->ne[0], bank->ne[1], bank->nb[1], expert * bank->nb[2]);
            inputs[expert] = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, input->ne[0], cohorts[expert].size());
            outputs[expert] = ggml_mul_mat(context.get(), slice, inputs[expert]);
            ggml_set_op_params_i32(outputs[expert], 0, ggml_get_op_params_i32(node, 3));
            ggml_set_output(outputs[expert]); ggml_build_forward_expand(graph, outputs[expert]);
        }
        auto * allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        CHECK(allocator && ggml_gallocr_alloc_graph(allocator, graph));
        ggml_backend_tensor_set(bank, weight->data, 0, ggml_nbytes(weight));
        for (size_t expert = 0; expert < cohorts.size(); ++expert) {
            if (cohorts[expert].empty()) { continue; }
            std::vector<float> values;
            for (const auto route : cohorts[expert]) {
                values.insert(values.end(), activation.begin() + route * input->ne[0], activation.begin() + (route + 1) * input->ne[0]);
            }
            ggml_backend_tensor_set(inputs[expert], values.data(), 0, ggml_nbytes(inputs[expert]));
        }
        CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        for (size_t expert = 0; expert < cohorts.size(); ++expert) {
            if (cohorts[expert].empty()) { continue; }
            std::vector<float> values(ggml_nelements(outputs[expert]));
            ggml_backend_tensor_get(outputs[expert], values.data(), 0, ggml_nbytes(outputs[expert]));
            for (size_t row = 0; row < cohorts[expert].size(); ++row) {
                std::copy_n(values.data() + row * node->ne[0], node->ne[0], reinterpret_cast<float *>(destinations[cohorts[expert][row]]));
            }
        }
        ggml_gallocr_free(allocator);
        return;
    }
    // Pad the independent matrix reference; discard the extra output rows.
    const size_t minimum_rows = mapped_quant && input->ne[2] < 16 ? 16 : 0;
    const size_t complement_rows = std::max(experts.size(), minimum_rows);
    experts.resize(complement_rows, 0);
    activation.resize(complement_rows * input->ne[0], 0.0f);
    ggml_context_ptr context(ggml_init({7 * ggml_tensor_overhead() + ggml_graph_overhead_custom(8, false), nullptr, true}));
    CHECK(context);
    auto * bank = ggml_new_tensor_3d(context.get(), weight->type, weight->ne[0], weight->ne[1], weight->ne[2]);
    CHECK(ggml_are_same_shape(bank, weight) && ggml_is_contiguous(weight));
    auto * routes = ggml_new_tensor_2d(context.get(), GGML_TYPE_I32, 1, experts.size());
    auto * rows = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, input->ne[0], 1, experts.size());
    auto * result = ggml_mul_mat_id(context.get(), bank, rows, routes);
    memcpy(result->op_params, node->op_params, sizeof(node->op_params));
    const int64_t reference_rows = std::max(size_t(input->ne[2]), minimum_rows);
    auto * original_rows = ggml_new_tensor_3d(context.get(), input->type, input->ne[0], input->ne[1], reference_rows);
    auto * original_ids = ggml_new_tensor_2d(context.get(), ids->type, ids->ne[0], reference_rows);
    auto * original_result = ggml_mul_mat_id(context.get(), bank, original_rows, original_ids);
    memcpy(original_result->op_params, node->op_params, sizeof(node->op_params));
    ggml_set_output(result); ggml_set_output(original_result);
    auto * graph = ggml_new_graph_custom(context.get(), 8, false);
    ggml_build_forward_expand(graph, result);
    ggml_build_forward_expand(graph, original_result);
    auto * allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    CHECK(allocator && ggml_gallocr_alloc_graph(allocator, graph));
    ggml_backend_tensor_set(bank, weight->data, 0, ggml_nbytes(weight));
    ggml_backend_tensor_set(rows, activation.data(), 0, ggml_nbytes(rows));
    ggml_backend_tensor_set(routes, experts.data(), 0, ggml_nbytes(routes));
    std::vector<float> original_activation;
    for (int64_t row = 0; row < input->ne[2]; ++row) {
        for (int64_t column = 0; column < input->ne[1]; ++column) {
            for (int64_t feature = 0; feature < input->ne[0]; ++feature) {
                float value;
                memcpy(&value, static_cast<const uint8_t *>(input->data) + row * input->nb[2] +
                    column * input->nb[1] + feature * input->nb[0], sizeof(value));
                original_activation.push_back(value);
            }
        }
    }
    CHECK(original_activation.size() == size_t(ggml_nelements(input)) && original_experts.size() == size_t(ggml_nelements(ids)));
    original_activation.resize(ggml_nelements(original_rows), 0.0f);
    original_experts.resize(ggml_nelements(original_ids), 0);
    ggml_backend_tensor_set(original_rows, original_activation.data(), 0, ggml_nbytes(original_rows));
    ggml_backend_tensor_set(original_ids, original_experts.data(), 0, ggml_nbytes(original_ids));
    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> output(ggml_nelements(result));
    ggml_backend_tensor_get(result, output.data(), 0, ggml_nbytes(result));
    std::vector<float> reference(ggml_nelements(original_result));
    ggml_backend_tensor_get(original_result, reference.data(), 0, ggml_nbytes(original_result));
    for (size_t route = 0; route < destinations.size(); ++route) {
        for (int64_t feature = 0; feature < result->ne[0]; ++feature) {
            const float expected = reference[logical_routes[route] * result->ne[0] + feature];
            CHECK(std::isfinite(output[route * result->ne[0] + feature]) &&
                std::fabs(output[route * result->ne[0] + feature] - expected) <= 5e-6f * (1.0f + std::fabs(expected)));
        }
        memcpy(destinations[route], output.data() + route * result->ne[0], result->ne[0] * sizeof(float));
    }
    ggml_gallocr_free(allocator);
}

static std::vector<float> evaluate_body(const std::vector<const ggml_tensor *> & nodes,
        const std::vector<const ggml_tensor *> & dynamic, const std::vector<const ggml_tensor *> & outputs,
        const std::vector<ggml_backend_moe_cpu_region_source_v1> & sources, const std::vector<const void *> & inputs,
        int partition_pattern = -1, ggml_backend_buffer_type_t source_buft = nullptr, ggml_backend_t complement_backend = nullptr,
        const std::vector<std::vector<uint8_t>> * projection_masks = nullptr, const std::vector<std::vector<int32_t>> * projection_ids = nullptr) {
    const size_t count = nodes.size() + dynamic.size() + sources.size();
    ggml_context_ptr ctx(ggml_init({ggml_tensor_overhead() * count + ggml_graph_overhead_custom(count, false), nullptr, true}));
    CHECK(ctx && inputs.size() == dynamic.size());
    std::vector<const ggml_tensor *> originals = dynamic;
    for (const auto & source : sources) { originals.push_back(source.tensor); }
    originals.insert(originals.end(), nodes.begin(), nodes.end());
    std::unordered_map<const ggml_tensor *, ggml_tensor *> clones;
    for (const auto * original : originals) {
        auto * clone = ggml_new_tensor_4d(ctx.get(), original->type, original->ne[0], original->ne[1], original->ne[2], original->ne[3]);
        *clone = *original;
        clone->data = nullptr; clone->buffer = nullptr; clone->extra = nullptr;
        clones.emplace(original, clone);
    }
    for (const auto * original : originals) {
        auto * clone = clones.at(original);
        for (int s = 0; s < GGML_MAX_SRC; ++s) { clone->src[s] = original->src[s] ? clones.at(original->src[s]) : nullptr; }
        clone->view_src = original->view_src ? clones.at(original->view_src) : nullptr;
    }
    auto * graph = ggml_new_graph_custom(ctx.get(), count, false);
    for (const auto * output : outputs) { ggml_build_forward_expand(graph, clones.at(output)); }
    std::vector<ggml_backend_buffer_t> source_buffers;
    if (source_buft) {
        auto * device = ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0);
        for (const auto & source : sources) {
            auto * tensor = clones.at(source.tensor);
            const auto reader = std::find_if(graph->nodes, graph->nodes + graph->n_nodes, [&](const ggml_tensor * node) {
                return node->op == GGML_OP_MUL_MAT_ID && node->src[0] == tensor;
            });
            if (reader == graph->nodes + graph->n_nodes) { continue; }
            auto * buffer = ggml_backend_buft_alloc_buffer(source_buft, ggml_backend_buft_get_alloc_size(source_buft, tensor));
            CHECK(buffer && ggml_backend_tensor_alloc(buffer, tensor, ggml_backend_buffer_get_base(buffer)) == GGML_STATUS_SUCCESS);
            if (!ggml_backend_dev_supports_op(device, *reader)) {
                tensor->buffer = nullptr; tensor->data = nullptr; tensor->extra = nullptr;
                ggml_backend_buffer_free(buffer);
                continue;
            }
            ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            source_buffers.push_back(buffer);
        }
        CHECK(!source_buffers.empty());
    }
    auto * allocator = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    CHECK(allocator && ggml_gallocr_alloc_graph(allocator, graph));
    for (size_t i = 0; i < dynamic.size(); ++i) {
        auto * clone = clones.at(dynamic[i]);
        if (!clone->data) {
            CHECK(std::find(graph->leafs, graph->leafs + graph->n_leafs, clone) == graph->leafs + graph->n_leafs);
            CHECK(std::find(graph->nodes, graph->nodes + graph->n_nodes, clone) == graph->nodes + graph->n_nodes);
            continue;
        }
        memcpy(clone->data, inputs[i], ggml_nbytes(dynamic[i]));
    }
    for (const auto & source : sources) { ggml_backend_tensor_set(clones.at(source.tensor), source.data, 0, ggml_nbytes(source.tensor)); }
    auto parameters = ggml_threadpool_params_default(2);
    auto * pool = ggml_threadpool_new(&parameters);
    CHECK(pool);
    auto plan = ggml_graph_plan(graph, 2, pool);
    std::vector<uint8_t> work(plan.work_size);
    plan.work_data = work.data();
    if (partition_pattern < 0) {
        CHECK(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS);
    } else {
        ggml_backend_ptr backend(ggml_backend_cpu_init());
        ggml_backend_ptr other_backend(ggml_backend_cpu_init());
        CHECK(backend && other_backend);
        ggml_backend_cpu_set_n_threads(backend.get(), 2);
        ggml_backend_cpu_set_threadpool(backend.get(), pool);
        CHECK(ggml_backend_reg_get_proc_address(ggml_backend_cpu_reg(), "ggml_backend_cpu_graph_plan_set_mmid_route_filter") ==
            reinterpret_cast<void *>(ggml_backend_cpu_graph_plan_set_mmid_route_filter));
        CHECK(!ggml_backend_cpu_graph_plan_set_mmid_route_filter(backend.get(), nullptr, nullptr, nullptr));
        size_t projection = 0;
        for (int i = 0; i < graph->n_nodes; ++i) {
            auto view = ggml_graph_view(graph, i, i + 1);
            auto * node = graph->nodes[i];
            auto prepared = ggml_backend_graph_plan_create(backend.get(), &view);
            CHECK(prepared);
            if (node->op != GGML_OP_MUL_MAT_ID) {
                CHECK(ggml_backend_graph_plan_compute(backend.get(), prepared) == GGML_STATUS_SUCCESS);
                ggml_backend_graph_plan_free(backend.get(), prepared);
                continue;
            }
            CHECK(node->type == GGML_TYPE_F32 && ggml_is_contiguous(node));
            constexpr float sentinel = -271828.0f;
            body_route_partition partition{partition_pattern, false, node, 0};
            if (projection_masks) {
                CHECK(projection_ids && projection < projection_masks->size() && projection < projection_ids->size());
                partition.expert_mask = &(*projection_masks)[projection];
                const auto * ids = node->src[2];
                CHECK((*projection_ids)[projection].size() == size_t(ggml_nelements(ids)));
                for (int64_t row = 0; row < ids->ne[1]; ++row) {
                    for (int64_t column = 0; column < ids->ne[0]; ++column) {
                        int32_t expert;
                        memcpy(&expert, static_cast<const uint8_t *>(ids->data) + row * ids->nb[1] + column * ids->nb[0], sizeof(expert));
                        CHECK(expert == (*projection_ids)[projection][size_t(row * ids->ne[0] + column)]);
                    }
                }
                ++projection;
            }
            CHECK(ggml_backend_cpu_graph_plan_set_mmid_route_filter(backend.get(), prepared, filter_body_route, &partition));
            auto unfiltered = ggml_backend_graph_plan_create(backend.get(), &view);
            CHECK(unfiltered && ggml_backend_graph_plan_compute(backend.get(), unfiltered) == GGML_STATUS_SUCCESS && partition.calls == 0);
            ggml_backend_graph_plan_free(backend.get(), unfiltered);
            CHECK(ggml_backend_cpu_graph_plan_set_mmid_route_filter(backend.get(), prepared, nullptr, &partition));
            CHECK(ggml_backend_graph_plan_compute(backend.get(), prepared) == GGML_STATUS_SUCCESS && partition.calls == 0);
            CHECK(ggml_backend_cpu_graph_plan_set_mmid_route_filter(backend.get(), prepared, filter_body_route, &partition));
            CHECK(!ggml_backend_cpu_graph_plan_set_mmid_route_filter(other_backend.get(), prepared, nullptr, nullptr));
            CHECK(!ggml_backend_cpu_graph_plan_set_mmid_route_filter(nullptr, prepared, nullptr, nullptr));
            std::fill_n(static_cast<float *>(node->data), ggml_nelements(node), sentinel);
            CHECK(ggml_backend_graph_plan_compute(backend.get(), prepared) == GGML_STATUS_SUCCESS);
            const auto * ids = node->src[2];
            const size_t routes = ids->ne[0] * ids->ne[1];
            CHECK(partition.calls == routes);
            for (int64_t row = 0; row < ids->ne[1]; ++row) {
                for (int64_t column = 0; column < ids->ne[0]; ++column) {
                    int32_t expert;
                    memcpy(&expert, static_cast<const uint8_t *>(ids->data) + row * ids->nb[1] + column * ids->nb[0], sizeof(expert));
                    const auto * values = reinterpret_cast<const float *>(static_cast<const uint8_t *>(node->data) + row * node->nb[2] + column * node->nb[1]);
                    for (int64_t feature = 0; feature < node->ne[0]; ++feature) {
                        CHECK((values[feature] != sentinel) == body_route_selected(partition, row, column, expert));
                    }
                }
            }
            const std::vector<float> first(static_cast<float *>(node->data), static_cast<float *>(node->data) + ggml_nelements(node));
            if (complement_backend) {
                complete_body_projection_gpu(complement_backend, node, partition,
                    projection_masks != nullptr && !ggml_is_quantized(node->src[0]->type),
                    projection_masks != nullptr && ggml_is_quantized(node->src[0]->type));
                CHECK(partition.calls == routes);
            } else {
                partition.complement = true;
                CHECK(ggml_backend_graph_plan_compute(backend.get(), prepared) == GGML_STATUS_SUCCESS);
                CHECK(partition.calls == 2 * routes);
            }
            for (size_t element = 0; element < first.size(); ++element) {
                const float value = static_cast<const float *>(node->data)[element];
                CHECK(value != sentinel && (first[element] == sentinel || first[element] == value));
            }
            CHECK(ggml_backend_cpu_graph_plan_set_mmid_route_filter(backend.get(), prepared, nullptr, &partition));
            ggml_backend_graph_plan_free(backend.get(), prepared);
        }
        if (projection_masks) { CHECK(projection == projection_masks->size()); }
    }
    ggml_threadpool_free(pool);
    std::vector<float> result;
    for (const auto * original : outputs) {
        const auto * output = clones.at(original);
        CHECK(output->type == GGML_TYPE_F32);
        for (int64_t i3 = 0; i3 < output->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < output->ne[2]; ++i2) {
                for (int64_t i1 = 0; i1 < output->ne[1]; ++i1) {
                    for (int64_t i0 = 0; i0 < output->ne[0]; ++i0) {
                        const auto * ptr = static_cast<const uint8_t *>(output->data) + i0 * output->nb[0] + i1 * output->nb[1] + i2 * output->nb[2] + i3 * output->nb[3];
                        float value; memcpy(&value, ptr, sizeof(value)); result.push_back(value);
                    }
                }
            }
        }
    }
    ggml_gallocr_free(allocator);
    for (auto * buffer : source_buffers) { ggml_backend_buffer_free(buffer); }
    return result;
}

static void test_source_body_program() {
    for (uint32_t variant = 0; variant < 5; ++variant) {
        const int64_t width = variant == 0 ? 2688 : variant == 1 ? 1024 : variant == 4 ? 128 : 32;
        const int64_t hidden_width = variant == 0 ? 1856 : variant == 1 ? 3584 : variant == 4 ? 96 : 48;
        const ggml_type up_type = variant == 0 ? GGML_TYPE_Q5_0 : variant == 1 ? GGML_TYPE_Q5_K : variant == 4 ? GGML_TYPE_Q4_0 : GGML_TYPE_F32;
        const ggml_type down_type = variant == 0 ? GGML_TYPE_Q8_0 : variant == 1 ? GGML_TYPE_Q5_K : variant == 4 ? GGML_TYPE_Q4_0 : GGML_TYPE_F32;
        const int64_t routes_per_row = variant == 2 || variant == 3 ? 4 : 2;
        ggml_context_ptr ctx(ggml_init({64 * ggml_tensor_overhead() + ggml_graph_overhead_custom(64, false), nullptr, true}));
        CHECK(ctx);
        auto * input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, width, variant == 2 || variant == 3 ? 2 : 1, 2);
        auto * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, routes_per_row, 2);
        auto * up_bank = ggml_new_tensor_3d(ctx.get(), up_type, width, variant == 2 ? 2 * hidden_width : hidden_width, 3);
        auto * down_bank = ggml_new_tensor_3d(ctx.get(), down_type, hidden_width, width, 3);
        auto * up = ggml_mul_mat_id(ctx.get(), up_bank, input, ids);
        auto * value = up;
        std::vector<ggml_tensor *> banks{up_bank, down_bank};
        if (variant == 0) { value = ggml_sqr(ctx.get(), ggml_relu(ctx.get(), up)); }
        if (variant == 1) {
            auto * gate_bank = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_Q3_K, width, hidden_width, 3);
            banks.push_back(gate_bank);
            value = ggml_swiglu_split(ctx.get(), ggml_mul_mat_id(ctx.get(), gate_bank, input, ids), up);
        }
        if (variant == 2) {
            auto * expert_scale = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 3);
            banks.push_back(expert_scale);
            auto * repeated_scale = ggml_repeat_4d(ctx.get(), ggml_reshape_3d(ctx.get(), expert_scale, 1, 3, 1), 1, 3, 2, 1);
            up = ggml_mul(ctx.get(), up, ggml_get_rows(ctx.get(), repeated_scale, ids));
            auto * gate = ggml_view_3d(ctx.get(), up, hidden_width, routes_per_row, 2, up->nb[1], up->nb[2], 0);
            auto * linear = ggml_view_3d(ctx.get(), up, hidden_width, routes_per_row, 2, up->nb[1], up->nb[2], hidden_width * sizeof(float));
            value = ggml_swiglu_oai(ctx.get(), gate, linear, 1.702f, 7.0f);
        }
        if (variant == 3) {
            auto * route_first = ggml_cont(ctx.get(), ggml_permute(ctx.get(), up, 1, 0, 2, 3));
            auto * coupled = ggml_cumsum(ctx.get(), ggml_soft_max(ctx.get(), route_first));
            value = ggml_norm(ctx.get(), ggml_cont(ctx.get(), ggml_permute(ctx.get(), coupled, 1, 0, 2, 3)), 1e-5f);
        }
        if (variant == 4) { value = ggml_gelu(ctx.get(), up); }
        auto * down = ggml_mul_mat_id(ctx.get(), down_bank, value, ids);
        ggml_prec_set_acc(down, GGML_PREC_F32);
        auto * adapted = down;
        if (variant == 2) {
            auto * adapter_a = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, hidden_width, 8, 3);
            auto * adapter_b = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 8, width, 3);
            banks.push_back(adapter_a); banks.push_back(adapter_b);
            auto * delta = ggml_mul_mat_id(ctx.get(), adapter_b, ggml_mul_mat_id(ctx.get(), adapter_a, value, ids), ids);
            adapted = ggml_add(ctx.get(), down, ggml_scale(ctx.get(), delta, 0.125f));
        }
        auto * bias = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, 3);
        banks.push_back(bias);
        auto * output = ggml_add_id(ctx.get(), adapted, bias, ids);
        auto * scale = variant == 2 ? ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, width, 2, 2) : nullptr;
        if (scale) { output = ggml_mul(ctx.get(), output, scale); }
        auto * graph = ggml_new_graph_custom(ctx.get(), 64, false);
        ggml_build_forward_expand(graph, output);
        std::vector<const ggml_tensor *> nodes(graph->nodes, graph->nodes + graph->n_nodes);
        std::vector<const ggml_tensor *> dynamic{input, ids}, outputs{output, value};
        if (scale) { dynamic.push_back(scale); }
        std::vector<std::vector<uint8_t>> backing(banks.size());
        std::vector<ggml_backend_moe_cpu_region_source_v1> sources;
        for (size_t i = 0; i < banks.size(); ++i) {
            auto * bank = banks[i];
            backing[i].resize(ggml_nbytes(bank));
            std::vector<float> values(ggml_nelements(bank));
            for (size_t j = 0; j < values.size(); ++j) { values[j] = 0.025f * std::sin(float(j + i) * 0.013f); }
            if (bank->type == GGML_TYPE_F32) { memcpy(backing[i].data(), values.data(), backing[i].size()); }
            else { CHECK(ggml_quantize_chunk(bank->type, values.data(), backing[i].data(), 0, bank->ne[1] * bank->ne[2], bank->ne[0], nullptr) == backing[i].size()); }
            sources.push_back({bank, bank, backing[i].data(), backing[i].size(), bank->nb[2], 1});
        }
        std::vector<float> activation(ggml_nelements(input));
        for (size_t i = 0; i < activation.size(); ++i) { activation[i] = std::sin(float(i) * 0.07f); }
        const int32_t routes[] = {2, 0, 1, 2, 0, 1, 2, 0};
        std::vector<float> scales(size_t(width) * 4);
        for (size_t i = 0; i < scales.size(); ++i) { scales[i] = 0.5f + 0.1f * float(i / width); }
        std::vector<const void *> inputs{activation.data(), routes};
        if (scale) { inputs.push_back(scales.data()); }
        ggml_backend_moe_cpu_region_query_v1 query = {};
        query.struct_size = sizeof(query); query.graph = graph; query.source_generation = 1;
        query.activation = input; query.ids = ids;
        query.body_nodes = nodes.data(); query.n_body_nodes = nodes.size();
        query.dynamic_inputs = dynamic.data(); query.n_dynamic_inputs = dynamic.size();
        query.live_outputs = outputs.data(); query.n_live_outputs = outputs.size();
        query.sources = sources.data(); query.n_sources = sources.size();
        auto body = ggml_moe_source_body::prepare(query);
        CHECK(body && body->nodes().size() == nodes.size() && body->live_outputs().size() == 2);
        const auto expected = evaluate_body(nodes, dynamic, outputs, sources, inputs);
        for (int pattern = 0; pattern < 6; ++pattern) {
            const auto partitioned = evaluate_body(nodes, dynamic, outputs, sources, inputs, pattern);
            CHECK(partitioned.size() == expected.size());
            for (size_t i = 0; i < expected.size(); ++i) {
                CHECK(std::isfinite(partitioned[i]) && std::fabs(partitioned[i] - expected[i]) <= 1e-5f * (1.0f + std::fabs(expected[i])));
            }
        }
        if (variant == 4) {
            auto * device = ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0);
            auto get_bufts = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(ggml_backend_reg_get_proc_address(
                ggml_backend_cpu_reg(), "ggml_backend_dev_get_extra_bufts"));
            auto ** bufts = get_bufts ? get_bufts(device) : nullptr;
            for (size_t i = 0; bufts && bufts[i]; ++i) {
                if (strcmp(ggml_backend_buft_name(bufts[i]), "CPU_REPACK")) { continue; }
                const auto repacked = evaluate_body(nodes, dynamic, outputs, sources, inputs, -1, bufts[i]);
                CHECK(repacked.size() == expected.size());
                for (size_t k = 0; k < expected.size(); ++k) {
                    CHECK(std::fabs(repacked[k] - expected[k]) <= 1e-4f * (1.0f + std::fabs(expected[k])));
                }
                for (int pattern = 0; pattern < 6; ++pattern) {
                    const auto selected = evaluate_body(nodes, dynamic, outputs, sources, inputs, pattern, bufts[i]);
                    CHECK(selected == repacked);
                }
                fprintf(stderr, "test-moe-cache: CPU_REPACK Q4_0 complementary route partitions preserve exact same-backend outputs OK\n");
            }
        }
        if (variant == 3) {
            auto * backend = ggml_backend_cuda_init(0);
            CHECK(backend);
            for (int pattern = 0; pattern < 6; ++pattern) {
                const auto joined = evaluate_body(nodes, dynamic, outputs, sources, inputs, pattern, nullptr, backend);
                CHECK(joined.size() == expected.size());
                double squared_error = 0, squared_reference = 0;
                float maximum_error = 0;
                size_t worst = 0;
                for (size_t i = 0; i < expected.size(); ++i) {
                    const float error = std::fabs(joined[i] - expected[i]);
                    squared_error += double(error) * error;
                    squared_reference += double(expected[i]) * expected[i];
                    if (error > maximum_error) { maximum_error = error; worst = i; }
                }
                fprintf(stderr, "test-moe-cache: CPU/CUDA join pattern=%d nmse=%.9g max_error=%.9g worst=%zu cpu=%.9g joined=%.9g\n",
                    pattern, squared_error / std::max(squared_reference, 1e-30), maximum_error, worst, expected[worst], joined[worst]);
                CHECK(squared_error <= 1e-7 * std::max(squared_reference, 1e-30));
                for (size_t i = 0; i < expected.size(); ++i) {
                    CHECK(std::isfinite(joined[i]) && std::fabs(joined[i] - expected[i]) <= 1e-4f * (1.0f + std::fabs(expected[i])));
                }
            }
            ggml_backend_free(backend);
            fprintf(stderr, "test-moe-cache: six CPU/CUDA complementary projection joins preserve cross-route graph outputs OK\n");
        }
        for (size_t i = 0; i < nodes.size(); ++i) {
            CHECK(!memcmp(nodes[i]->op_params, body->nodes()[i]->op_params, sizeof(nodes[i]->op_params)));
            CHECK(!memcmp(nodes[i]->ne, body->nodes()[i]->ne, sizeof(nodes[i]->ne)));
            CHECK(!memcmp(nodes[i]->nb, body->nodes()[i]->nb, sizeof(nodes[i]->nb)));
        }
        const auto * saved = down->src[1]; down->src[1] = output;
        CHECK(!ggml_moe_source_body::prepare(query)); down->src[1] = const_cast<ggml_tensor *>(saved);
        sources[0].bytes = 1; CHECK(!ggml_moe_source_body::prepare(query)); sources[0].bytes = backing[0].size();
        auto duplicate = outputs; duplicate[1] = duplicate[0]; query.live_outputs = duplicate.data();
        CHECK(!ggml_moe_source_body::prepare(query)); query.live_outputs = outputs.data();
        const auto graph_size = graph->size; graph->size = graph->n_nodes - 1;
        CHECK(!ggml_moe_source_body::prepare(query)); graph->size = graph_size;
        if (variant == 2) {
            const auto offset = value->src[0]->view_offs; value->src[0]->view_offs = SIZE_MAX;
            CHECK(!ggml_moe_source_body::prepare(query)); value->src[0]->view_offs = offset;
        }
        ctx.reset();
        const auto actual = evaluate_body(body->nodes(), body->dynamic_inputs(), body->live_outputs(), body->sources(), inputs);
        CHECK(actual == expected);
        CHECK(std::any_of(actual.begin(), actual.end(), [](float value) { return value != 0; }));
        if (variant >= 3) {
            if (variant == 3) { CHECK(!body->compact(4)); }
            continue;
        }
        CHECK(!body->compact(0) && !body->compact(uint32_t(INT_MAX) + 1));
        const uint32_t selected[] = {uint32_t(2 * routes_per_row - 1), 0, uint32_t(routes_per_row), uint32_t(routes_per_row - 1)};
        auto compact = body->compact(4);
        CHECK(compact && !compact->compact(4) && !compact->bind_routes(0) && !compact->bind_routes(5));
        for (uint32_t count : {1u, 3u, 4u}) {
            CHECK(compact->bind_routes(count));
            std::vector<uint32_t> rows(count), columns(count);
            std::vector<int32_t> compact_ids(count);
            for (uint32_t i = 0; i < count; ++i) {
                rows[i] = selected[i] / routes_per_row; columns[i] = selected[i] % routes_per_row;
                compact_ids[i] = routes[selected[i]];
            }
            std::vector<std::vector<uint8_t>> gathered(dynamic.size());
            std::vector<const void *> compact_inputs;
            for (size_t i = 0; i < dynamic.size(); ++i) {
                gathered[i].resize(ggml_nbytes(compact->dynamic_inputs()[i]));
                if (i == 1) { memcpy(gathered[i].data(), compact_ids.data(), gathered[i].size()); }
                else {
                    const size_t bytes = i == 0 ? activation.size() * sizeof(float) : scales.size() * sizeof(float);
                    CHECK(compact->gather_input(i, inputs[i], bytes, rows.data(), columns.data(), count, gathered[i].data(), gathered[i].size()));
                    const auto valid = gathered[i];
                    CHECK(!compact->gather_input(i, inputs[i], 1, rows.data(), columns.data(), count, gathered[i].data(), gathered[i].size()));
                    CHECK(gathered[i] == valid);
                    rows[0] = 2;
                    CHECK(!compact->gather_input(i, inputs[i], bytes, rows.data(), columns.data(), count, gathered[i].data(), gathered[i].size()));
                    CHECK(gathered[i] == valid); rows[0] = selected[0] / routes_per_row;
                    columns[0] = routes_per_row;
                    CHECK(!compact->gather_input(i, inputs[i], bytes, rows.data(), columns.data(), count, gathered[i].data(), gathered[i].size()));
                    CHECK(gathered[i] == valid); columns[0] = selected[0] % routes_per_row;
                    CHECK(!compact->gather_input(i, inputs[i], bytes, rows.data(), columns.data(), count, const_cast<void *>(inputs[i]), bytes));
                    CHECK(!compact->gather_input(i, inputs[i], bytes, rows.data(), columns.data(), count, rows.data(), gathered[i].size()));
                }
                compact_inputs.push_back(gathered[i].data());
            }
            const auto compact_values = evaluate_body(compact->nodes(), compact->dynamic_inputs(), compact->live_outputs(), compact->sources(), compact_inputs);
            size_t at = 0, original_at = 0;
            for (const auto * live : body->live_outputs()) {
                const size_t features = live->ne[0];
                for (uint32_t i = 0; i < count; ++i) {
                    for (size_t j = 0; j < features; ++j) {
                        const float oracle = expected[original_at + selected[i] * features + j];
                        CHECK(std::abs(compact_values[at++] - oracle) <= 1e-5f * (1.0f + std::abs(oracle)));
                    }
                }
                original_at += 2 * routes_per_row * features;
            }
            CHECK(at == compact_values.size());
        }
    }
    fprintf(stderr, "test-moe-cache: body program stock shapes/formats, OAI, bias/scales/adapters, released graph, compact subsets and 30 shared route-partition oracles including cross-route softmax/cumsum/norm OK\n");
}

static void check_original_body(const llm_graph_moe_region & region, const ggml_backend_moe_hybrid_region_v1 & descriptor) {
    const auto * query = descriptor.body_query;
    CHECK(query && query->n_body_nodes == region.body_operations.size());
    CHECK(query->n_dynamic_inputs == region.dynamic_inputs.size() && query->n_live_outputs == region.live_outs.size());
    auto body = ggml_moe_source_body::prepare(*query);
    CHECK(body && body->nodes().size() == region.body_operations.size());
    for (size_t i = 0; i < region.body_operations.size(); ++i) {
        const auto * actual = body->nodes()[i];
        const auto * original = region.body_operations[i];
        CHECK(actual->op == original->op && actual->type == original->type);
        CHECK(!memcmp(actual->ne, original->ne, sizeof(actual->ne)));
        CHECK(!memcmp(actual->nb, original->nb, sizeof(actual->nb)));
        CHECK(!memcmp(actual->op_params, original->op_params, sizeof(actual->op_params)));
    }
    for (size_t i = 0; i < region.dynamic_inputs.size(); ++i) {
        CHECK(!memcmp(query->dynamic_inputs[i]->ne, region.dynamic_inputs[i]->ne, sizeof(ggml_tensor::ne)));
        CHECK(!memcmp(query->dynamic_inputs[i]->nb, region.dynamic_inputs[i]->nb, sizeof(ggml_tensor::nb)));
    }
}

void test_moe_static_profile() {
    const std::vector<uint32_t> counts{3, 0, 5};
    std::vector<uint8_t> binary{'S', 'T', 'R', 'P'};
    const auto append = [](std::vector<uint8_t> & output, uint32_t value, uint32_t width = 4) {
        for (uint32_t i = 0; i < width; ++i) { output.push_back(uint8_t(value >> (8 * i))); }
    };
    for (uint32_t value : {1u, 3u, 5u, 2u, 8u}) { append(binary, value); }
    const std::vector<std::pair<uint32_t, uint32_t>> pairs{{2, 4}, {0, 2}, {2, 0}, {0, 0}, {2, 1}, {2, 3}, {0, 1}, {2, 2}};
    for (const auto & pair : pairs) { append(binary, pair.first, 2); append(binary, pair.second, 2); }
    const std::vector<std::vector<int32_t>> expected{{2, 0, 1}, {}, {4, 0, 1, 3, 2}};
    CHECK(llama_moe_profile_parse(binary.data(), binary.size(), counts) == expected);
    auto historical = binary;
    for (uint32_t layer = 0; layer < counts.size(); ++layer) {
        for (uint32_t expert = 0; expert < 5; ++expert) {
            const auto found = std::find(pairs.begin(), pairs.end(), std::make_pair(layer, expert));
            append(historical, found == pairs.end() ? UINT32_MAX : uint32_t(found - pairs.begin()));
        }
    }
    CHECK(llama_moe_profile_parse(historical.data(), historical.size(), counts) == expected);
    auto partial = binary;
    partial.resize(24 + 3 * 4); partial[20] = 3;
    CHECK(llama_moe_profile_parse(partial.data(), partial.size(), counts) ==
        (std::vector<std::vector<int32_t>>{{2}, {}, {4, 0}}));
    uint32_t rejected = 0;
    const auto invalid = [&](const std::vector<uint8_t> & bytes, const std::vector<uint32_t> & geometry = std::vector<uint32_t>{}) {
        bool failed = false;
        try { (void) llama_moe_profile_parse(bytes.data(), bytes.size(), geometry.empty() ? counts : geometry); }
        catch (const std::runtime_error &) { failed = true; }
        CHECK(failed); ++rejected;
    };
    for (size_t size = 0; size < binary.size(); ++size) { invalid(std::vector<uint8_t>(binary.begin(), binary.begin() + size)); }
    for (size_t offset : {size_t(0), size_t(4), size_t(8), size_t(12), size_t(16), size_t(20), size_t(24), size_t(26)}) {
        auto corrupt = binary; corrupt[offset] = 255; invalid(corrupt);
    }
    auto duplicate = binary; std::copy_n(duplicate.begin() + 24, 4, duplicate.begin() + 28); invalid(duplicate);
    auto tail = binary; tail.push_back(0); invalid(tail);
    historical.back() ^= 1; invalid(historical);
    invalid(binary, {5, 0, 3}); invalid(binary, {3, 5}); invalid(binary, {3, 0, 6});
    fprintf(stderr, "test-moe-cache: STRP full/partial/historical sparse heterogeneous ranks and %u malformed cases OK\n", rejected);
}

namespace {

constexpr int n_dim     = 256;
constexpr int n_experts = 8;
constexpr int n_used    = 2;
constexpr int n_slots   = 4;
constexpr size_t bounded_host_budget = 2 * 1024 * 1024;
constexpr uint32_t all_cached_layers = (1u << 2) - 1;

ggml_graph_execution_certificate layer_certificate(bool required = false) {
    ggml_graph_execution_certificate certificate{};
    certificate.magic            = GGML_GRAPH_EXECUTION_CERTIFICATE_MAGIC;
    certificate.abi_version      = GGML_GRAPH_EXECUTION_CERTIFICATE_VERSION;
    certificate.struct_size      = sizeof(certificate);
    certificate.flags            = required ? GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED : 0;
    certificate.domain           = required ? GGML_GRAPH_EXECUTION_DOMAIN_DRAFT : GGML_GRAPH_EXECUTION_DOMAIN_MAIN;
    certificate.row_semantics    = GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT;
    certificate.owner_namespace  = 723;
    certificate.owner_generation = 1;
    certificate.n_rows           = 1;
    certificate.n_sequences      = 1;
    return certificate;
}

struct layer_fixture {
    bool routed_backend_stage = false;
    bool independent_overlap = false;
    bool ordinary_scratch = false;
    bool ordinary_matmul = false;
    bool ordinary_empty_view = false;
    ggml_tensor * ordinary_rotation = nullptr;
    bool ordinary_library = false;
    bool ordinary_cpu_prefix = false;
    ggml_tensor * overlap_value[2] = {};
    ggml_context_ptr                                  weights;
    ggml_backend_buffer_ptr                           weight_buffer;
    std::vector<ggml_backend_buffer_ptr>              ordinary_buffers;
    ggml_backend_buffer_type_t                        buft = nullptr;
    uint32_t                                           cached_layers = all_cached_layers;
    uint32_t                                           cache_slots   = n_slots;
    llm_graph_result                                  result{ 256 };
    ggml_tensor *                                     input      = nullptr;
    ggml_tensor *                                     logits[2]  = {};
    ggml_tensor *                                     ids[2]     = {};
    ggml_tensor *                                     up[2]      = {};
    ggml_tensor *                                     gate[2]    = {};
    ggml_tensor *                                     gate_up[2] = {};
    ggml_tensor *                                     down[2]    = {};
    ggml_tensor *                                     down_bias[2] = {};
    ggml_tensor *                                     up_bias[2] = {};
    ggml_tensor *                                     gate_bias[2] = {};
    ggml_tensor *                                     up_scale[2] = {};
    ggml_tensor *                                     gate_scale[2] = {};
    ggml_tensor *                                     down_scale[2] = {};
    ggml_tensor *                                     output[2]  = {};
    ggml_tensor *                                     prefix = nullptr;
    ggml_tensor *                                     staged = nullptr;
    bool                                              activation_images = false;
    ggml_tensor *                                     bf16_projection_weight = nullptr;
    ggml_tensor *                                     bf16_projection = nullptr;
    ggml_tensor *                                     bf16_prefix = nullptr;
    ggml_tensor *                                     attention_key = nullptr;
    ggml_tensor *                                     attention_value = nullptr;
    ggml_tensor *                                     attention_mask = nullptr;
    ggml_tensor *                                     attention_output = nullptr;
    ggml_tensor *                                     attention_prefix = nullptr;
    ggml_tensor *                                     route_probe = nullptr;
    ggml_tensor *                                     ordinary_gamma = nullptr;
    ggml_tensor *                                     ordinary_norm_output = nullptr;
    std::vector<ggml_backend_moe_candidate_group_v2>  groups;
    std::vector<ggml_backend_moe_candidate_tensor_v2> tensors;
    llm_arch                                           arch;
    llm_ffn_op_type                                    type_op;
    int                                                expert_used;
    int                                                n_embd;
    int                                                n_expert;

    explicit layer_fixture(size_t host_budget, uint32_t layout, ggml_type type, bool pageable = false,
                           const std::array<int, 2> & ordinary_devices = { 0, 0 },
                           uint32_t cache_layer_mask = all_cached_layers, uint32_t slots = n_slots, bool named_weights = false,
                           llm_arch arch = LLM_ARCH_QWEN3MOE, llm_ffn_op_type type_op = LLM_FFN_SILU,
                           int expert_used = n_used, ggml_type down_type = GGML_TYPE_COUNT, int n_embd = n_dim, int n_ff = n_dim,
                           int n_expert = n_experts, bool legacy_down = false, bool output_auxiliaries = false) :
        cached_layers(cache_layer_mask), cache_slots(slots), arch(arch), type_op(type_op), expert_used(expert_used), n_embd(n_embd), n_expert(n_expert) {
        if (down_type == GGML_TYPE_COUNT) {
            down_type = type;
        }
        weights.reset(ggml_init({ 32 * ggml_tensor_overhead(), nullptr, true }));
        CHECK(weights != nullptr);
        buft = pageable ? pageable_cached_buffer_type() : ggml_backend_cuda_moe_cached_bounded_buffer_type(host_budget);
        CHECK(buft != nullptr);
        for (int layer = 0; layer < 2; ++layer) {
            const bool cached = (cached_layers & (1u << layer)) != 0;
            const auto ordinary = [&](ggml_tensor * tensor) {
                if (cached) {
                    return;
                }
                auto * ordinary_buft = ggml_backend_cuda_buffer_type(ordinary_devices[layer]);
                auto buffer = ggml_backend_buft_alloc_buffer(ordinary_buft, ggml_backend_buft_get_alloc_size(ordinary_buft, tensor));
                CHECK(buffer != nullptr);
                ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                CHECK(ggml_backend_tensor_alloc(buffer, tensor, ggml_backend_buffer_get_base(buffer)) == GGML_STATUS_SUCCESS);
                ordinary_buffers.emplace_back(buffer);
            };
            groups.push_back({ layout, GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY, 0, 0 });
            const auto bank = [&](uint32_t role, int rows) {
                const bool is_down = role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT;
                auto * tensor = ggml_new_tensor_3d(weights.get(), is_down ? down_type : type, is_down ? n_ff : n_embd, rows, n_expert);
                if (named_weights) {
                    const char * name = role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT ? "gate_up" :
                        role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT ? "gate" :
                        role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT ? "up" : "down";
                    ggml_format_name(tensor, "blk.%d.ffn_%s_exps.weight", layer, name);
                }
                tensors.push_back({ tensor, (uint32_t) layer, role, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE,
                                    cached ? static_cast<uint32_t>(GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_CACHED_BUFFER) : 0u, 0 });
                if (cached && legacy_down && is_down) {
                    auto * direct_buft = ggml_backend_cuda_moe_cached_bounded_buffer_type(0);
                    auto buffer = ggml_backend_buft_alloc_buffer(direct_buft, ggml_backend_buft_get_alloc_size(direct_buft, tensor));
                    CHECK(buffer && ggml_backend_tensor_alloc(buffer, tensor, ggml_backend_buffer_get_base(buffer)) == GGML_STATUS_SUCCESS);
                    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                    ordinary_buffers.emplace_back(buffer);
                } else { ordinary(tensor); }
                return tensor;
            };
            if (layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP) {
                gate_up[layer] = bank(GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, 2 * n_ff);
            } else {
                if (layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE) {
                    gate[layer] = bank(GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT, n_ff);
                }
                up[layer] = bank(GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT, n_ff);
            }
            down[layer] = bank(GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, n_embd);
            if (pageable || output_auxiliaries) {
                down_bias[layer] = ggml_new_tensor_2d(weights.get(), GGML_TYPE_F32, n_embd, n_expert);
                tensors.push_back({down_bias[layer], (uint32_t) layer, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_BIAS,
                    GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_OUTPUT_BIAS,
                    cached ? static_cast<uint32_t>(GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_CACHED_BUFFER) : 0u, 0});
                ordinary(down_bias[layer]);
            }
            if (output_auxiliaries) {
                CHECK(layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE);
                const auto auxiliary = [&](uint32_t role, bool scale, int rows, const char * name) {
                    auto * tensor = scale ? ggml_new_tensor_1d(weights.get(), GGML_TYPE_F32, n_expert) :
                        ggml_new_tensor_2d(weights.get(), GGML_TYPE_F32, rows, n_expert);
                    ggml_format_name(tensor, "layer.%d.%s", layer, name);
                    tensors.push_back({tensor, (uint32_t) layer, role,
                        scale ? GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_OUTPUT_SCALE : GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_OUTPUT_BIAS,
                        cached ? static_cast<uint32_t>(GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_CACHED_BUFFER) : 0u, 0});
                    ordinary(tensor);
                    return tensor;
                };
                ggml_format_name(down_bias[layer], "layer.%d.down.bias", layer);
                up_bias[layer] = auxiliary(GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_BIAS, false, n_ff, "up.bias");
                gate_bias[layer] = auxiliary(GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_BIAS, false, n_ff, "gate.bias");
                up_scale[layer] = auxiliary(GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_SCALE, true, 1, "up.scale");
                gate_scale[layer] = auxiliary(GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_SCALE, true, 1, "gate.scale");
                down_scale[layer] = auxiliary(GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_SCALE, true, 1, "down.scale");
            }
        }
        weight_buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(weights.get(), buft));
        if (weight_buffer != nullptr) {
            ggml_backend_buffer_set_usage(weight_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }
        CHECK(weight_buffer != nullptr || !ordinary_buffers.empty());
        size_t salt = 101;
        for (auto * tensor = ggml_get_first_tensor(weights.get()); tensor;
             tensor        = ggml_get_next_tensor(weights.get(), tensor)) {
            const auto bytes = cached_fusion_test_data(tensor, salt++);
            ggml_backend_tensor_set(tensor, bytes.data(), 0, bytes.size());
        }
        if (host_budget != 0) {
            auto snapshot = manifest();
            CHECK(ggml_backend_cuda_moe_cached_configure_sources(buft, &snapshot));
        }
        build_graph();
    }

    void build_graph(bool add_prefix = false, const ggml_staged_input_api * stage_api = nullptr, void * stage = nullptr,
                     bool check_fusion = false, uint32_t rows = 1, bool empty_prefix = false, uint32_t bf16_columns = 0,
                     bool flash_attention = false, bool cpu_oracle = false, bool weighted_norm = false) {
        result.reset();
        auto * ctx = result.get_ctx();
        input      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, rows);
        ggml_set_input(input);
        if (empty_prefix) {
            auto * empty = ggml_view_1d(ctx, input, 0, 0);
            auto * scale = ggml_scale_inplace(ctx, empty, 0.0f);
            ggml_set_name(scale, "fidelity_empty_scale");
            ggml_build_forward_expand(result.get_gf(), scale);
            auto * copy = ggml_cpy(ctx, empty, empty);
            ggml_set_name(copy, "fidelity_empty_copy");
            ggml_build_forward_expand(result.get_gf(), copy);
        }
        prefix = nullptr;
        staged = nullptr;
        if (add_prefix) {
            auto * prefix_input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
            ggml_set_input(prefix_input);
            prefix = ggml_scale(ctx, prefix_input, 0.75f);
            ggml_set_output(prefix);
            ggml_build_forward_expand(result.get_gf(), prefix);
        }
        auto *              cur = input;
        ordinary_rotation = nullptr;
        if (ordinary_empty_view) {
            auto * empty = ggml_view_2d(ctx, cur, n_embd, 0, cur->nb[1], ggml_nbytes(cur) + GGML_MEM_ALIGN);
            ggml_set_name(empty, "fidelity_empty_history");
            cur = ggml_concat(ctx, empty, cur, 1);
        }
        if (ordinary_matmul) {
            ordinary_rotation = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);
            ggml_set_name(ordinary_rotation, "fidelity_ordinary_rotation");
            cur = ggml_mul_mat(ctx, ordinary_rotation, cur);
            ggml_mul_mat_set_hint(cur, GGML_HINT_SRC0_IS_HADAMARD);
            ggml_set_name(cur, "fidelity_ordinary_matmul_input");
        }
        if (ordinary_library) {
            const int64_t head_dim = ordinary_cpu_prefix ? 8 : 16, heads = 256 / head_dim;
            const int64_t state_dim = ordinary_cpu_prefix ? 32 : 128, tokens = 256;
            CHECK(n_embd == head_dim * heads);
            auto * zero = ggml_reshape_4d(ctx, ggml_scale(ctx, ggml_sum_rows(ctx, cur), 0.0f), 1, 1, 1, rows);
            auto * state = ggml_repeat_4d(ctx, zero, state_dim, head_dim, heads, rows);
            auto * x = ggml_repeat_4d(ctx, ggml_reshape_4d(ctx, cur, head_dim, heads, 1, rows), head_dim, heads, tokens, rows);
            auto * dt = ggml_fill(ctx, ggml_reshape_3d(ctx, ggml_repeat_4d(ctx, zero, heads, tokens, 1, rows), heads, tokens, rows), -2.0f);
            auto * decay = ggml_fill(ctx, ggml_repeat_4d(ctx, ggml_scale(ctx, ggml_sum(ctx, cur), 0.0f), 1, heads, 1, 1), -1.0f);
            auto * b = ggml_fill(ctx, ggml_repeat_4d(ctx, zero, state_dim, 1, tokens, rows), 0.05f);
            auto * c = ggml_fill(ctx, ggml_repeat_4d(ctx, zero, state_dim, 1, tokens, rows), 0.025f);
            auto * indices = ggml_cast(ctx, ggml_arange(ctx, 0.0f, float(rows), 1.0f), GGML_TYPE_I32);
            auto * scan = ggml_ssm_scan(ctx, state, x, dt, decay, b, c, indices, 1);
            ggml_set_name(scan, "fidelity_ordinary_library_scan");
            auto * y = ggml_view_4d(ctx, scan, head_dim, heads, tokens, rows,
                head_dim * sizeof(float), head_dim * heads * sizeof(float), head_dim * heads * tokens * sizeof(float), 0);
            auto * mean = ggml_sum_rows(ctx, ggml_cont(ctx, ggml_permute(ctx, y, 1, 2, 0, 3)));
            cur = ggml_add(ctx, cur, ggml_scale(ctx, ggml_reshape_2d(ctx, mean, n_embd, rows), 1.0f / tokens));
            ggml_set_name(cur, "fidelity_ordinary_library_input");
        }
        if (ordinary_scratch) {
            constexpr int64_t columns = 1 << 20;
            auto * shape = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, columns, rows);
            auto * expanded = ggml_repeat(ctx, ggml_sgn(ctx, cur), shape);
            auto * cumulative = ggml_cumsum(ctx, expanded);
            auto * total = ggml_scale(ctx, ggml_sum(ctx, cumulative), 1.0f / (float(columns) * float(columns) * rows));
            cur = ggml_add(ctx, cur, ggml_repeat(ctx, total, cur));
            ggml_set_name(cur, "fidelity_ordinary_scratch_input");
        }
        bf16_projection_weight = nullptr;
        bf16_projection = nullptr;
        bf16_prefix = nullptr;
        if (bf16_columns) {
            bf16_projection_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, n_embd, bf16_columns);
            ggml_set_name(bf16_projection_weight, "fidelity_bf16_weight");
            cur = ggml_scale(ctx, input, 1.0f);
            ggml_set_name(cur, "fidelity_bf16_activation");
            if (activation_images) { cur = ggml_rms_norm(ctx, cur, 1e-6f); }
            bf16_projection = ggml_mul_mat(ctx, bf16_projection_weight, cur);
            if (activation_images) {
                auto * second = ggml_mul_mat(ctx, bf16_projection_weight, cur);
                bf16_projection = ggml_add(ctx, bf16_projection, second);
            }
            ggml_set_name(bf16_projection, "fidelity_bf16_projection");
            auto * mean = ggml_scale(ctx, ggml_sum_rows(ctx, bf16_projection), 1.0f / bf16_columns);
            cur = bf16_prefix = ggml_add(ctx, cur, ggml_repeat(ctx, mean, cur));
            ggml_set_name(bf16_prefix, "fidelity_bf16_moe_input");
        }
        attention_key = attention_value = attention_mask = attention_output = attention_prefix = nullptr;
        route_probe = nullptr;
        if (flash_attention) {
            CHECK(n_embd == 256);
            auto * q = ggml_repeat(ctx, ggml_reshape_3d(ctx, cur, n_embd, 1, rows),
                ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 24, rows));
            q = ggml_permute(ctx, q, 0, 2, 1, 3);
            attention_key = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, n_embd, 2, 16384);
            attention_value = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, n_embd, 2, 16384);
            attention_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 8448, rows);
            ggml_set_name(attention_key, "fidelity_attention_key");
            ggml_set_name(attention_value, "fidelity_attention_value");
            ggml_set_name(attention_mask, "fidelity_attention_mask");
            const auto cache_view = [&](ggml_tensor * cache) {
                auto * view = ggml_view_4d(ctx, cache, n_embd, 2, 8448, 1, cache->nb[1], cache->nb[2], cache->nb[3], 0);
                return ggml_permute(ctx, view, 0, 2, 1, 3);
            };
            attention_output = ggml_flash_attn_ext(ctx, q, cache_view(attention_key), cache_view(attention_value),
                attention_mask, 1.0f / sqrtf(float(n_embd)), 0.0f, 0.0f);
            ggml_flash_attn_ext_set_n_kv_max(attention_output, 2051);
            CHECK(ggml_prec_set_acc(attention_output, GGML_PREC_F32));
            ggml_set_name(attention_output, "fidelity_flash_attention");
            auto * first_head = ggml_view_2d(ctx, attention_output, n_embd, rows, attention_output->nb[2], 0);
            cur = attention_prefix = ggml_add(ctx, cur, first_head);
            ggml_set_name(attention_prefix, "fidelity_attention_moe_input");
        }
        ggml_tensor * retained = nullptr;
        ordinary_gamma = nullptr;
        ordinary_norm_output = nullptr;
        if (weighted_norm) {
            CHECK(n_embd % 2 == 0);
            ordinary_gamma = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd / 2, 2);
            ggml_set_input(ordinary_gamma);
            ggml_set_name(ordinary_gamma, "fidelity_ordinary_gamma");
            auto * norm_input = ggml_reshape_3d(ctx, cur, n_embd / 2, 2, rows);
            auto * norm = ggml_rms_norm(ctx, norm_input, 1e-6f);
            ggml_set_name(norm, "fidelity_ordinary_rms");
            ordinary_norm_output = ggml_mul(ctx, norm, ordinary_gamma);
            ggml_set_output(ordinary_norm_output);
            ggml_build_forward_expand(result.get_gf(), ordinary_norm_output);
        }
        if (check_fusion) {
            retained = ggml_add(ctx, input, input);
            cur = ggml_add(ctx, retained, input);
        }
        llama_adapter_loras loras;
        llm_graph_params    params{};
        params.arch                = arch;
        params.hparams.n_embd      = n_embd;
        params.hparams.n_layer_all = 2;
        params.hparams.n_expert    = n_expert;
        params.hparams.n_expert_used_arr.fill(expert_used);
        params.hparams.n_head_arr.fill(1);
        params.hparams.n_head_kv_arr.fill(1);
        params.ubatch.n_tokens = rows;
        params.loras           = &loras;
        params.res             = &result;
        llm_graph_context builder(params);
        for (int layer = 0; layer < 2; ++layer) {
            auto * branch_input = cur;
            logits[layer] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_expert, rows);
            ggml_set_input(logits[layer]);
            cur = output[layer] =
                builder.build_moe_ffn(cur, nullptr, nullptr, up[layer], up_bias[layer], gate[layer], gate_bias[layer], down[layer],
                                      down_bias[layer], nullptr, n_expert, expert_used, type_op, true, 1.0f,
                                      LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, layer, logits[layer], gate_up[layer], nullptr,
                                      up_scale[layer], gate_scale[layer], down_scale[layer]);
            auto & region         = result.get_moe_regions().back();
            if (routed_backend_stage) {
                const auto found = std::find_if(region.body_operations.begin(), region.body_operations.end(), [&](const ggml_tensor * node) {
                    return node->op == GGML_OP_MUL_MAT_ID && node->src[0] == down[layer];
                });
                CHECK(found != region.body_operations.end());
                const size_t index = found - region.body_operations.begin();
                auto * projection = const_cast<ggml_tensor *>(*found);
                auto * scaled = ggml_scale(ctx, projection->src[1], 4.0f);
                auto * activated = ggml_tanh(ctx, scaled);
                ggml_set_name(scaled, "routed_backend_scale");
                ggml_set_name(activated, "routed_backend_tanh");
                projection->src[1] = activated;
                region.body_operations.insert(region.body_operations.begin() + index, scaled);
                region.body_operations.insert(region.body_operations.begin() + index + 1, activated);
                const auto operation = std::find(region.operations.begin(), region.operations.end(), projection);
                CHECK(operation != region.operations.end());
                region.operations.insert(operation, {scaled, activated});
                // The builder expanded the old dependencies before this fixture added the stages.
                auto * graph = result.get_gf();
                const std::vector<ggml_tensor *> roots(graph->nodes, graph->nodes + graph->n_nodes);
                ggml_graph_clear(graph);
                for (auto * root : roots) { ggml_build_forward_expand(graph, root); }
            }
            region.semantic_group = layer;
            region.domain         = GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY;
            ids[layer]            = region.route;
            if (independent_overlap) {
                auto * scale = ggml_scale(ctx, branch_input, 0.125f);
                ggml_set_name(scale, "source_overlap_scale");
                overlap_value[layer] = ggml_silu(ctx, scale);
                ggml_set_name(overlap_value[layer], "source_overlap_silu");
                cur = output[layer] = ggml_add(ctx, cur, overlap_value[layer]);
            }
            if (layer == 0 && cpu_oracle) {
                route_probe = ggml_cont(ctx, ids[layer]);
                ggml_set_name(route_probe, "fidelity_cpu_oracle_ids");
                ggml_build_forward_expand(result.get_gf(), route_probe);
            }
            if (layer == 0 && stage != nullptr) {
                staged = stage_api->build(stage, ctx, cur, n_embd);
                ggml_set_output(staged);
                cur = output[layer] = ggml_add(ctx, cur, staged);
            }
            if (check_fusion) {
                cur = output[layer] = ggml_add(ctx, ggml_add(ctx, cur, input), input);
            }
            ggml_set_output(cur);
            ggml_build_forward_expand(result.get_gf(), cur);
            if (layer == 0 && retained != nullptr) {
                auto * probe = ggml_scale(ctx, retained, 1.0f);
                ggml_set_name(probe, "hybrid_fusion_retained");
                ggml_set_output(probe);
                ggml_build_forward_expand(result.get_gf(), probe);
            }
        }
    }

    ~layer_fixture() {
        weight_buffer.reset();
        ggml_backend_cuda_moe_cached_free_buffer_type(buft);
    }

    ggml_backend_moe_candidate_snapshot_v2 manifest() const {
        return candidate_snapshot_v2(cache_slots, groups.data(), groups.size(), tensors.data(), tensors.size());
    }

    void allocate_ordinary_rotation(ggml_backend_buffer_type_t buft) {
        if (!ordinary_rotation) { return; }
        auto buffer = ggml_backend_buft_alloc_buffer(buft, ggml_backend_buft_get_alloc_size(buft, ordinary_rotation));
        CHECK(buffer && ggml_backend_tensor_alloc(buffer, ordinary_rotation, ggml_backend_buffer_get_base(buffer)) == GGML_STATUS_SUCCESS);
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::vector<float> values(size_t(n_embd) * n_embd);
        const float scale = 1.0f / std::sqrt(float(n_embd));
        for (int row = 0; row < n_embd; ++row) {
            for (int col = 0; col < n_embd; ++col) {
                int parity = 0;
                for (int bits = row & col; bits; bits >>= 1) { parity ^= bits & 1; }
                values[size_t(row) * n_embd + col] = parity ? -scale : scale;
            }
        }
        ggml_backend_tensor_set(ordinary_rotation, values.data(), 0, values.size() * sizeof(float));
        ordinary_buffers.emplace_back(buffer);
    }

    void set_inputs(int iteration) {
        if (prefix != nullptr) {
            const float value = 1;
            ggml_backend_tensor_set(prefix->src[0], &value, 0, sizeof(value));
        }
        std::vector<float> values(n_embd);
        for (int i = 0; i < n_embd; ++i) {
            values[i] = 0.001f * ((i + iteration) % 17 - 8);
        }
        ggml_backend_tensor_set(input, values.data(), 0, ggml_nbytes(input));
        for (int layer = 0; layer < 2; ++layer) {
            std::vector<float> scores(n_expert);
            for (int i = 0; i < n_expert; ++i) {
                scores[i] = (i + iteration + 3 * layer) % n_expert;
            }
            ggml_backend_tensor_set(logits[layer], scores.data(), 0, scores.size() * sizeof(float));
        }
    }
};

void check_region_descriptor(const llm_graph_moe_region & region,
                             ggml_cgraph *                graph,
                             ggml_tensor *                down,
                             ggml_tensor *                route) {
    CHECK(region.down == down && region.route == route);
    CHECK(region.builder_cut_closed && !region.has_lora);
    CHECK(!region.body_operations.empty() && region.first_body == region.body_operations.front() &&
          region.body_output == region.body_operations.back());
    CHECK(region.first_body->op == GGML_OP_MUL_MAT_ID && region.body_output->op == GGML_OP_MUL_MAT_ID &&
          region.body_output->src[0] == down);
    CHECK(region.tail_resume != nullptr && std::find(region.body_operations.begin(), region.body_operations.end(),
                                                     region.tail_resume) == region.body_operations.end());
    CHECK(region.dynamic_inputs.size() == 2 && region.first_body->src[1] == region.dynamic_inputs[0] &&
          region.dynamic_inputs[1] == region.route);
    const auto body_live_out =
        std::find_if(region.live_outs.begin(), region.live_outs.end(),
                     [&](const llm_graph_moe_live_out & live_out) { return live_out.tensor == region.body_output; });
    CHECK(body_live_out != region.live_outs.end() && !body_live_out->consumers.empty());
    const auto body_output_it = std::find(region.operations.begin(), region.operations.end(), region.body_output);
    CHECK(body_output_it != region.operations.end() && body_output_it + 1 != region.operations.end() &&
          *(body_output_it + 1) == region.tail_resume);
    size_t body_index = 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph) && body_index < region.body_operations.size(); ++i) {
        if (ggml_graph_node(graph, i) == region.body_operations[body_index]) {
            ++body_index;
        }
    }
    CHECK(body_index == region.body_operations.size());
}

void check_finalized_region(const llm_graph_moe_region & region) {
    const auto & snapshot = region.finalized_metadata;
    CHECK(snapshot != nullptr && snapshot->source_graph_uid != 0 && snapshot->split_graph_uid != 0);
    CHECK(snapshot->owner_generation == 17 && snapshot->allocator_generation == 23);
    CHECK(snapshot->first_node_index <= snapshot->last_node_index &&
          snapshot->tail_node_index == snapshot->last_node_index + 1);
    CHECK(snapshot->graph() != nullptr && ggml_graph_n_nodes(const_cast<ggml_cgraph *>(snapshot->graph())) ==
          static_cast<int>(snapshot->nodes().size()));
    CHECK(snapshot->graph()->uid == 0);
    CHECK(snapshot->nodes().size() == region.body_operations.size());
    CHECK(snapshot->input_origins().size() == snapshot->inputs().size());
    CHECK(snapshot->dynamic().size() == region.dynamic_inputs.size());
    CHECK(snapshot->outputs().size() == region.live_outs.size());
    for (size_t i = 0; i < snapshot->nodes().size(); ++i) {
        const auto * original = region.body_operations[i];
        const auto * clone    = snapshot->nodes()[i];
        CHECK(clone != original && clone->type == original->type && clone->op == original->op &&
              clone->flags == original->flags && clone->buffer == nullptr && clone->data == nullptr &&
              clone->extra == nullptr && memcmp(clone->ne, original->ne, sizeof(clone->ne)) == 0 &&
              memcmp(clone->nb, original->nb, sizeof(clone->nb)) == 0 &&
              memcmp(clone->op_params, original->op_params, sizeof(clone->op_params)) == 0);
        for (const auto * src : clone->src) {
            CHECK(src == nullptr || std::find(snapshot->nodes().begin(), snapshot->nodes().end(), src) != snapshot->nodes().end() ||
                  std::find(snapshot->inputs().begin(), snapshot->inputs().end(), src) != snapshot->inputs().end());
        }
        CHECK(clone->view_src == nullptr ||
              std::find(snapshot->nodes().begin(), snapshot->nodes().end(), clone->view_src) != snapshot->nodes().end() ||
              std::find(snapshot->inputs().begin(), snapshot->inputs().end(), clone->view_src) != snapshot->inputs().end());
    }
    for (size_t i = 0; i < snapshot->inputs().size(); ++i) {
        const auto * input = snapshot->inputs()[i];
        const auto & origin = snapshot->input_origins()[i];
        CHECK(origin.witness != nullptr && origin.witness != input && input->type == origin.type &&
              memcmp(input->ne, origin.ne, sizeof(input->ne)) == 0 &&
              memcmp(input->nb, origin.nb, sizeof(input->nb)) == 0);
        CHECK(input->op == GGML_OP_NONE && input->buffer == nullptr && input->data == nullptr && input->extra == nullptr);
        for (const auto * src : input->src) {
            CHECK(src == nullptr || std::find(snapshot->inputs().begin(), snapshot->inputs().end(), src) != snapshot->inputs().end());
        }
        CHECK(input->view_src == nullptr ||
              std::find(snapshot->inputs().begin(), snapshot->inputs().end(), input->view_src) != snapshot->inputs().end());
    }
    for (const auto * input : snapshot->dynamic()) {
        CHECK(input->op == GGML_OP_NONE && input->view_src == nullptr &&
              std::all_of(std::begin(input->src), std::end(input->src),
                          [](const ggml_tensor * src) { return src == nullptr; }));
    }
    for (size_t i = 0; i < snapshot->outputs().size(); ++i) {
        CHECK(snapshot->outputs()[i].body_node < snapshot->nodes().size());
        CHECK(snapshot->nodes()[snapshot->outputs()[i].body_node]->op == region.live_outs[i].tensor->op);
        CHECK(snapshot->outputs()[i].consumer_nodes.size() == region.live_outs[i].consumers.size());
    }
    for (const auto * tensor : snapshot->inputs()) {
        const size_t position = ggml_hash_find(&snapshot->graph()->visited_hash_set,
                                               const_cast<ggml_tensor *>(tensor));
        CHECK(position != GGML_HASHSET_FULL && ggml_bitset_get(snapshot->graph()->visited_hash_set.used, position));
        int32_t expected_uses = 0;
        for (const auto * consumer : snapshot->nodes()) {
            expected_uses += std::count(std::begin(consumer->src), std::end(consumer->src), tensor);
        }
        CHECK(snapshot->graph()->use_counts[position] == expected_uses);
    }
    for (const auto * tensor : snapshot->nodes()) {
        const size_t position = ggml_hash_find(&snapshot->graph()->visited_hash_set,
                                               const_cast<ggml_tensor *>(tensor));
        CHECK(position != GGML_HASHSET_FULL && ggml_bitset_get(snapshot->graph()->visited_hash_set.used, position));
        int32_t expected_uses = 0;
        for (const auto * consumer : snapshot->nodes()) {
            expected_uses += std::count(std::begin(consumer->src), std::end(consumer->src), tensor);
        }
        CHECK(snapshot->graph()->use_counts[position] == expected_uses);
    }
    ggml_cgraph * graph = const_cast<ggml_cgraph *>(snapshot->graph());
    const int n_nodes = graph->n_nodes;
    const int n_leafs = graph->n_leafs;
    ggml_build_forward_expand(graph, const_cast<ggml_tensor *>(snapshot->nodes().back()));
    CHECK(graph->n_nodes == n_nodes && graph->n_leafs == n_leafs);
    const char cloned_name = snapshot->nodes()[0]->name[0];
    const char source_name = region.body_operations[0]->name[0];
    region.body_operations[0]->name[0] ^= 1;
    CHECK(snapshot->nodes()[0]->name[0] == cloned_name);
    region.body_operations[0]->name[0] = source_name;
}

static void test_sigmoid_moe_routing() {
    {
        ggml_context_ptr context(ggml_init({10 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false), nullptr, true}));
        CHECK(context);
        auto * input = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, 6144, 1, 3);
        auto * ids = ggml_new_tensor_2d(context.get(), GGML_TYPE_I32, 8, 3);
        auto * gate = ggml_new_tensor_3d(context.get(), GGML_TYPE_BF16, 6144, 2048, 256);
        auto * up = ggml_new_tensor_3d(context.get(), GGML_TYPE_BF16, 6144, 2048, 256);
        auto * down = ggml_new_tensor_3d(context.get(), GGML_TYPE_BF16, 2048, 6144, 256);
        auto * gate_output = ggml_mul_mat_id(context.get(), gate, input, ids);
        auto * up_output = ggml_mul_mat_id(context.get(), up, input, ids);
        auto * activated = ggml_swiglu_split(context.get(), gate_output, up_output);
        auto * output = ggml_mul_mat_id(context.get(), down, activated, ids);
        auto * graph = ggml_new_graph_custom(context.get(), 16, false);
        ggml_build_forward_expand(graph, output);
        ggml_cpu_init();
        const auto plan = ggml_graph_plan(graph, 2, nullptr);
        CHECK(plan.work_size > 0 && plan.work_size < 1024 * 1024);
        auto * device = ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0);
        for (const auto * node : {gate_output, up_output, output}) { CHECK(ggml_backend_dev_supports_op(device, node)); }
        for (const auto * bank : {gate, up, down}) {
            CHECK(bank->data == nullptr && bank->buffer == nullptr && ggml_nbytes(bank) == uint64_t(6144) * 2048 * 256 * 2);
            CHECK(bank->nb[2] == uint64_t(6144) * 2048 * 2);
        }
        CHECK(ggml_used_mem(context.get()) < 1024 * 1024);
        fprintf(stderr, "test-moe-cache: GLM-config actual-size BF16 descriptors6144/2048/E256/K8 bank_bytes=%zu expert_bytes=%zu routed_weight_bytes=%llu cpu_work_bytes=%zu metadata_bytes=%zu payload_unallocated OK\n",
            ggml_nbytes(gate), gate->nb[2], (unsigned long long) (uint64_t(ggml_nbytes(gate)) * 3), plan.work_size, ggml_used_mem(context.get()));
    }
    constexpr int64_t width = 128, hidden = 96, experts = 256, selected = 8, rows = 3;
    ggml_context_ptr weights(ggml_init({10 * ggml_tensor_overhead(), nullptr, true}));
    CHECK(weights);
    auto * up = ggml_new_tensor_3d(weights.get(), GGML_TYPE_F32, width, hidden, experts);
    auto * gate = ggml_new_tensor_3d(weights.get(), GGML_TYPE_F32, width, hidden, experts);
    auto * down = ggml_new_tensor_3d(weights.get(), GGML_TYPE_F32, hidden, width, experts);
    auto * bias = ggml_new_tensor_1d(weights.get(), GGML_TYPE_F32, experts);
    auto * dense_up = ggml_new_tensor_2d(weights.get(), GGML_TYPE_F32, width, 2 * width);
    auto * dense_gate = ggml_new_tensor_2d(weights.get(), GGML_TYPE_F32, width, 2 * width);
    auto * dense_down = ggml_new_tensor_2d(weights.get(), GGML_TYPE_F32, 2 * width, width);
    auto * shared_up = ggml_new_tensor_2d(weights.get(), GGML_TYPE_F32, width, hidden);
    auto * shared_gate = ggml_new_tensor_2d(weights.get(), GGML_TYPE_F32, width, hidden);
    auto * shared_down = ggml_new_tensor_2d(weights.get(), GGML_TYPE_F32, hidden, width);
    std::vector<ggml_tensor *> banks{up, gate, down, bias, dense_up, dense_gate, dense_down, shared_up, shared_gate, shared_down};
    std::vector<std::vector<float>> backing(banks.size());
    std::vector<ggml_backend_moe_cpu_region_source_v1> sources;
    for (size_t b = 0; b < banks.size(); ++b) {
        auto * tensor = banks[b];
        auto & values = backing[b];
        values.resize(ggml_nelements(tensor));
        for (size_t i = 0; i < values.size(); ++i) { values[i] = std::sin(float((i + b * 19) % 997) * 0.17f) * 0.025f; }
        if (tensor == bias) {
            for (int64_t e = 0; e < experts; ++e) { values[e] = e < 16 ? 2.5f + float(e) * 0.017f : 0; }
        }
        sources.push_back({tensor, tensor, values.data(), values.size() * sizeof(float), tensor->nb[2], 1});
    }

    llm_graph_result result{256};
    auto * ctx = result.get_ctx();
    auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, rows);
    auto * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, experts, rows);
    ggml_set_input(input); ggml_set_input(logits);
    ggml_tensor * mixture = nullptr;
    llama_adapter_loras loras;
    llm_graph_params params{};
    params.arch = LLM_ARCH_GLM_DSA;
    params.hparams.n_embd = width;
    params.hparams.n_layer_all = 4;
    params.hparams.n_expert = experts;
    params.hparams.n_expert_used_arr.fill(selected);
    params.hparams.n_head_arr.fill(1);
    params.hparams.n_head_kv_arr.fill(1);
    params.hparams.n_expert_groups = 1;
    params.hparams.n_group_used = 1;
    params.ubatch.n_tokens = rows;
    params.loras = &loras;
    params.res = &result;
    params.cb = [&](const llama_ubatch &, ggml_tensor * tensor, const char * name, int) {
        if (!strcmp(name, "ffn_moe_weights_scaled")) { mixture = tensor; }
    };
    llm_graph_context builder(params);
    auto * value = input;
    for (int layer = 0; layer < 3; ++layer) {
        auto * dense = builder.build_ffn(value, dense_up, nullptr, nullptr, dense_gate, nullptr, nullptr,
            dense_down, nullptr, nullptr, nullptr, LLM_FFN_SILU, LLM_FFN_PAR, layer);
        value = ggml_add(ctx, value, dense);
    }
    auto * routed = builder.build_moe_ffn(value, nullptr, up, gate, down, bias, experts, selected,
        LLM_FFN_SILU, true, 2.5f, LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID, 3, logits);
    auto * shared = builder.build_ffn(value, shared_up, nullptr, nullptr, shared_gate, nullptr, nullptr,
        shared_down, nullptr, nullptr, nullptr, LLM_FFN_SILU, LLM_FFN_PAR, 3);
    auto * output = ggml_add(ctx, routed, shared);
    CHECK(mixture && result.get_moe_regions().size() == 1);
    auto * ids = result.get_moe_regions().back().route;
    CHECK(ids && ids->type == GGML_TYPE_I32 && ids->ne[0] == selected && ids->ne[1] == rows);
    auto * float_ids = ggml_cast(ctx, ids, GGML_TYPE_F32);
    const std::vector<const ggml_tensor *> outputs{output, mixture, float_ids, shared, routed};
    for (const auto * tensor : outputs) {
        ggml_set_output(const_cast<ggml_tensor *>(tensor));
        ggml_build_forward_expand(result.get_gf(), const_cast<ggml_tensor *>(tensor));
    }
    const std::vector<const ggml_tensor *> nodes(result.get_gf()->nodes, result.get_gf()->nodes + result.get_gf()->n_nodes);
    const std::vector<const ggml_tensor *> dynamic{input, logits};
    std::vector<float> activation(width * rows), scores(experts * rows);
    const std::vector<const void *> inputs{activation.data(), scores.data()};
    ggml_backend_ptr gpu(ggml_backend_cuda_init(0));
    CHECK(gpu);
    std::vector<int32_t> previous;
    bool numerical_ok = true;
    constexpr size_t output_elements = width * rows, route_elements = selected * rows;
    for (int replay = 0; replay < 2; ++replay) {
        for (size_t i = 0; i < activation.size(); ++i) { activation[i] = std::sin(float(i + replay * 7) * 0.07f); }
        for (int64_t row = 0; row < rows; ++row) {
            for (int64_t e = 0; e < experts; ++e) {
                float score = -1.8f + float(e) * 0.012f + float((e + row * 13) % 11) * 0.03f;
                if (e < 16) { score += (replay ? e < 8 : e >= 8) ? 0.9f : -0.9f; }
                scores[row * experts + e] = score;
            }
        }
        const auto reference = evaluate_body(nodes, dynamic, outputs, sources, inputs);
        CHECK(reference.size() == 3 * output_elements + 2 * route_elements);
        std::vector<int32_t> expected_ids;
        for (int64_t row = 0; row < rows; ++row) {
            std::vector<int32_t> order(experts);
            std::vector<double> probabilities(experts);
            for (int32_t e = 0; e < experts; ++e) {
                order[e] = e;
                probabilities[e] = 1.0 / (1.0 + std::exp(-double(scores[row * experts + e])));
            }
            std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
                return probabilities[a] + backing[3][a] > probabilities[b] + backing[3][b];
            });
            double total = 0;
            for (int64_t column = 0; column < selected; ++column) { total += probabilities[order[column]]; }
            double weight_sum = 0;
            for (int64_t column = 0; column < selected; ++column) {
                const size_t route = row * selected + column;
                const int32_t expert = order[column];
                const float actual_id = reference[output_elements + route_elements + route];
                const float weight = reference[output_elements + route];
                CHECK(actual_id == float(expert) && expert < 16);
                CHECK(std::fabs(weight - float(2.5 * probabilities[expert] / total)) <= 2e-6f);
                expected_ids.push_back(expert);
                weight_sum += weight;
            }
            CHECK(std::fabs(weight_sum - 2.5) <= 2e-6);
        }
        if (replay) { CHECK(previous != expected_ids); }
        previous = expected_ids;
        for (size_t i = 0; i < output_elements; ++i) {
            CHECK(reference[i] == reference[output_elements + 2 * route_elements + i] +
                reference[2 * output_elements + 2 * route_elements + i]);
        }
        for (int pattern = 0; pattern < 6; ++pattern) {
            for (const auto complement : {static_cast<ggml_backend_t>(nullptr), gpu.get()}) {
                const auto actual = evaluate_body(nodes, dynamic, outputs, sources, inputs, pattern, nullptr, complement);
                CHECK(actual.size() == reference.size());
                double squared_error = 0, squared_reference = 0;
                float maximum_error = 0;
                size_t worst = 0;
                for (size_t i = 0; i < actual.size(); ++i) {
                    CHECK(std::isfinite(actual[i]) && std::fabs(actual[i] - reference[i]) <= 1e-5f * (1 + std::fabs(reference[i])));
                    if (i < output_elements) {
                        squared_error += double(actual[i] - reference[i]) * (actual[i] - reference[i]);
                        squared_reference += double(reference[i]) * reference[i];
                        const float error = std::fabs(actual[i] - reference[i]);
                        if (error > maximum_error) { maximum_error = error; worst = i; }
                    }
                }
                fprintf(stderr, "test-moe-cache: sigmoid MoE replay=%d partition=%d complement=%s nmse=%.9g max_error=%.9g worst=%zu cpu=%.9g actual=%.9g\n",
                    replay, pattern, complement ? "CUDA" : "CPU", squared_error / std::max(squared_reference, 1e-30), maximum_error, worst, reference[worst], actual[worst]);
                // Three routed projections use ordinary TF32; the shared MMID test permits NMSE5e-4.
                numerical_ok &= squared_error <= (complement ? 5e-6 : 1e-7) * std::max(squared_reference, 1e-30);
                for (size_t i = output_elements; i < output_elements + 2 * route_elements; ++i) { CHECK(actual[i] == reference[i]); }
            }
        }
    }
    CHECK(numerical_ok);
    fprintf(stderr, "test-moe-cache: GLM-config synthetic E256/K8 sigmoid/bias/scale2.5 dense/shared joins replays=2 partitions=24 OK; features128/96, no attention/state/serving qualification\n");
}

static void test_region_route_producers() {
    for (uint32_t producer = 0; producer < 4; ++producer) {
        ggml_backend_ptr cpu(ggml_backend_cpu_init());
        CHECK(cpu);
        ggml_backend_cpu_set_n_threads(cpu.get(), 2);
        ggml_backend_t backends[] = {cpu.get()};
        ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, 1, 128, false, false));
        CHECK(sched);
        ggml_context_ptr weights(ggml_init({4 * ggml_tensor_overhead(), nullptr, true}));
        auto * up = ggml_new_tensor_3d(weights.get(), GGML_TYPE_F32, 32, 48, 3);
        auto * down = ggml_new_tensor_3d(weights.get(), GGML_TYPE_F32, 48, 32, 3);
        auto * table = producer == 2 ? ggml_new_tensor_2d(weights.get(), GGML_TYPE_I32, 2, 3) : nullptr;
        ggml_backend_buffer_ptr weight_buffer(ggml_backend_alloc_ctx_tensors(weights.get(), cpu.get()));
        CHECK(weight_buffer);
        ggml_backend_buffer_set_usage(weight_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (auto * weight : {up, down}) {
            std::vector<float> values(ggml_nelements(weight));
            for (size_t i = 0; i < values.size(); ++i) { values[i] = float(int(i % 19) - 9) * 0.003f; }
            ggml_backend_tensor_set(weight, values.data(), 0, values.size() * sizeof(float));
        }
        llm_graph_result result{128};
        auto * ctx = result.get_ctx();
        auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 2);
        auto * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2);
        ggml_set_input(input);
        ggml_set_input(logits);
        ggml_tensor * route_input = nullptr;
        ggml_tensor * selected = nullptr;
        if (producer == 1) { selected = route_input = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 2); }
        if (producer == 2) {
            route_input = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
            selected = ggml_get_rows(ctx, table, route_input);
        }
        if (producer == 3) {
            route_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
            selected = ggml_cast(ctx, ggml_scale(ctx, route_input, 0.5f), GGML_TYPE_I32);
        }
        if (route_input) { ggml_set_input(route_input); }
        llama_adapter_loras loras;
        llm_graph_params params{};
        params.arch = LLM_ARCH_QWEN3MOE;
        params.hparams.n_embd = 32;
        params.hparams.n_layer_all = 1;
        params.hparams.n_expert = 3;
        params.hparams.n_expert_used_arr.fill(2);
        params.hparams.n_head_arr.fill(1);
        params.hparams.n_head_kv_arr.fill(1);
        params.ubatch.n_tokens = 2;
        params.loras = &loras;
        params.res = &result;
        llm_graph_context builder(params);
        auto * output = builder.build_moe_ffn(input, nullptr, up, nullptr, down, nullptr, 3, 2,
            LLM_FFN_RELU_SQR, true, 1.0f, LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, 0,
            logits, nullptr, nullptr, nullptr, nullptr, selected);
        ggml_set_output(output);
        ggml_build_forward_expand(result.get_gf(), output);
        const auto original_body = result.get_moe_regions().back().body_operations;
        const auto original_operations = result.get_moe_regions().back().operations;
        llama_moe_source_group source = {};
        source.layout = GGML_BACKEND_MOE_CANDIDATE_LAYOUT_ROUTED_MATRIX;
        source.domain = GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY;
        source.layer = 0;
        for (auto * weight : {up, down}) {
            source.banks.push_back({weight, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_ROUTED_WEIGHT,
                GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE});
        }
        CHECK(result.discover_moe_regions({source}, +[](const ggml_tensor * tensor) {
            return tensor && tensor->op == GGML_OP_NONE && tensor->ne[2] == 3;
        }));
        CHECK(result.get_moe_regions().size() == 1);
        CHECK(result.get_moe_regions()[0].body_operations == original_body &&
            result.get_moe_regions()[0].operations == original_operations);
        auto & region = result.get_moe_regions().back();
        CHECK(region.builder_cut_closed && region.external_route == (producer != 0));
        ggml_set_output(region.route);
        if (producer) {
            CHECK(region.route == selected && std::find(region.operations.begin(), region.operations.end(), selected) == region.operations.end());
            const auto dynamic = region.dynamic_inputs;
            region.dynamic_inputs.erase(std::remove(region.dynamic_inputs.begin(), region.dynamic_inputs.end(), selected), region.dynamic_inputs.end());
            CHECK(!region.place(sched.get(), cpu.get()));
            for (auto * node : region.operations) { CHECK(!ggml_backend_sched_get_tensor_backend(sched.get(), node)); }
            region.dynamic_inputs = dynamic;
        }
        CHECK(region.place(sched.get(), cpu.get()));
        CHECK(region.finalize_metadata(sched.get(), result.get_gf(), 17, 23) == GGML_BACKEND_SCHED_REGION_STATUS_V1_NOT_FINALIZED);
        CHECK(ggml_backend_sched_alloc_graph(sched.get(), result.get_gf()));
        CHECK(region.finalize_metadata(sched.get(), result.get_gf(), 17, 23) == GGML_BACKEND_SCHED_REGION_STATUS_V1_OK);
        check_finalized_region(region);
        const auto snapshot = region.finalized_metadata;
        std::vector<float> activation(64, 0.125f);
        const float probabilities[] = {3, 2, 1, 1, 2, 3};
        ggml_backend_tensor_set(input, activation.data(), 0, activation.size() * sizeof(float));
        ggml_backend_tensor_set(logits, probabilities, 0, sizeof(probabilities));
        if (table) {
            const int32_t ids[] = {0, 1, 2, 0, 1, 1};
            ggml_backend_tensor_set(table, ids, 0, sizeof(ids));
        }
        for (uint32_t replay = 0; replay < 2; ++replay) {
            std::vector<int32_t> expected = replay ? std::vector<int32_t>{2, 2, 1, 0} : std::vector<int32_t>{0, 1, 2, 0};
            if (producer == 1) { ggml_backend_tensor_set(route_input, expected.data(), 0, expected.size() * sizeof(int32_t)); }
            if (producer == 2) {
                const int32_t tokens[] = {replay ? 2 : 0, replay ? 0 : 1};
                expected = replay ? std::vector<int32_t>{1, 1, 0, 1} : std::vector<int32_t>{0, 1, 2, 0};
                ggml_backend_tensor_set(route_input, tokens, 0, sizeof(tokens));
            }
            if (producer == 3) {
                std::vector<float> values;
                for (const auto id : expected) { values.push_back(float(id) * 2); }
                ggml_backend_tensor_set(route_input, values.data(), 0, values.size() * sizeof(float));
            }
            CHECK(ggml_backend_sched_graph_compute(sched.get(), result.get_gf()) == GGML_STATUS_SUCCESS);
            std::vector<int32_t> actual(4);
            ggml_backend_tensor_get(region.route, actual.data(), 0, actual.size() * sizeof(int32_t));
            if (producer && actual != expected) {
                fprintf(stderr, "frontend route producer=%u replay=%u actual=%d/%d/%d/%d expected=%d/%d/%d/%d\n",
                    producer, replay, actual[0], actual[1], actual[2], actual[3], expected[0], expected[1], expected[2], expected[3]);
            }
            if (producer) { CHECK(actual == expected); }
            std::vector<float> values(64);
            ggml_backend_tensor_get(output, values.data(), 0, values.size() * sizeof(float));
            CHECK(std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); }));
            CHECK(std::any_of(values.begin(), values.end(), [](float value) { return value != 0; }));
            CHECK(region.finalized_metadata == snapshot);
        }
        ggml_backend_sched_reset(sched.get());
        fprintf(stderr, "test-moe-cache: frontend route producer=%u boundary/placement/snapshot/replay OK\n", producer);
    }
    fprintf(stderr, "test-moe-cache: frontend route producers=4 replays=8 boundary/placement/snapshot/freshness OK\n");
}

struct region_finalize_tally {
    std::array<uint32_t, GGML_BACKEND_SCHED_REGION_STATUS_V1_CAPACITY + 1> statuses = {};

    int32_t record(int32_t status) {
        CHECK(status >= GGML_BACKEND_SCHED_REGION_STATUS_V1_OK &&
              status <= GGML_BACKEND_SCHED_REGION_STATUS_V1_CAPACITY);
        statuses[status]++;
        return status;
    }
};

using finalized_snapshot_ids = std::vector<const llm_graph_moe_finalized_snapshot *>;

bool required_structural_regions_present(const llm_graph_result & result, uint64_t owner_generation,
                                         uint64_t allocator_generation, const finalized_snapshot_ids * ids = nullptr) {
    const auto & regions = result.get_moe_regions();
    if (regions.empty() || (ids != nullptr && ids->size() != regions.size())) {
        return false;
    }
    for (size_t i = 0; i < regions.size(); ++i) {
        const auto & region = regions[i];
        const auto & snapshot = region.finalized_metadata;
        if (!region.builder_cut_closed || region.has_lora || region.external_route || region.backend == nullptr ||
                snapshot == nullptr || snapshot->owner_generation != owner_generation ||
                snapshot->allocator_generation != allocator_generation ||
                (ids != nullptr && (*ids)[i] != snapshot.get())) {
            return false;
        }
    }
    return true;
}

finalized_snapshot_ids require_structural_regions(const llm_graph_result & result, uint64_t owner_generation,
                                                  uint64_t allocator_generation) {
    CHECK(required_structural_regions_present(result, owner_generation, allocator_generation));
    finalized_snapshot_ids ids;
    for (const auto & region : result.get_moe_regions()) {
        ids.push_back(region.finalized_metadata.get());
    }
    return ids;
}

bool reject_region_callback(ggml_tensor *, bool, void *) {
    return true;
}

void test_region_descriptor_variants() {
    layer_fixture biased(0, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_TYPE_Q4_K, true, { 0, 0 },
                         all_cached_layers, n_slots, false, LLM_ARCH_LLAMA4, LLM_FFN_GELU);
    CHECK(biased.result.get_moe_regions().size() == 2);
    for (int layer = 0; layer < 2; ++layer) {
        check_region_descriptor(biased.result.get_moe_regions()[layer], biased.result.get_gf(), biased.down[layer],
                                biased.ids[layer]);
        CHECK(biased.result.get_moe_regions()[layer].first_body->src[0] == biased.gate[layer]);
        CHECK(biased.down_bias[layer] != nullptr &&
              biased.result.get_moe_regions()[layer].tail_resume->op == GGML_OP_ADD_ID);
    }

    layer_fixture single(0, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_TYPE_Q4_K, false, { 0, 0 },
                         all_cached_layers, n_slots, false, LLM_ARCH_LLAMA4, LLM_FFN_GELU, 1);
    for (int layer = 0; layer < 2; ++layer) {
        check_region_descriptor(single.result.get_moe_regions()[layer], single.result.get_gf(), single.down[layer],
                                single.ids[layer]);
        CHECK(single.result.get_moe_regions()[layer].tail_resume->op == GGML_OP_VIEW);
    }
}

struct split_probe {
    layer_fixture *                                             fixture;
    std::vector<ggml_backend_t>                                 backends;
    std::vector<ggml_status (*)(ggml_backend_t, ggml_cgraph *)> delegates;
    int                                                         grouped_splits[2]  = {};
    int                                                         imported_splits[2] = {};
    int                                                         fail_layer         = -1;
    bool                                                        failed             = false;
};

split_probe * active_probe = nullptr;

ggml_status probe_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    CHECK(active_probe != nullptr);
    auto & probe = *active_probe;
    CHECK(!probe.failed);
    const auto found = std::find(probe.backends.begin(), probe.backends.end(), backend);
    CHECK(found != probe.backends.end());
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto * node = ggml_graph_node(graph, i);
        for (int layer = 0; layer < 2; ++layer) {
            if (node->op != GGML_OP_MUL_MAT_ID || node->src[0] != probe.fixture->down[layer]) {
                continue;
            }
            bool local = false;
            for (int j = 0; j < i; ++j) {
                local = local || (ggml_graph_node(graph, j) == probe.fixture->ids[layer]->src[0] &&
                                  node->src[2] == probe.fixture->ids[layer]);
            }
            if (local) {
                ++probe.grouped_splits[layer];
            } else {
                ++probe.imported_splits[layer];
            }
            if (probe.fail_layer == layer) {
                probe.failed = true;
                return GGML_STATUS_FAILED;
            }
        }
    }
    return probe.delegates[found - probe.backends.begin()](backend, graph);
}

void test_boundary_copy(ggml_backend_t source, ggml_backend_t destination, int source_device, int destination_device) {
    ggml_context_ptr source_ctx(ggml_init({ ggml_tensor_overhead(), nullptr, true }));
    ggml_context_ptr destination_ctx(ggml_init({ ggml_tensor_overhead(), nullptr, true }));
    CHECK(source_ctx && destination_ctx);
    auto *                  src = ggml_new_tensor_1d(source_ctx.get(), GGML_TYPE_F32, n_dim);
    auto *                  dst = ggml_new_tensor_1d(destination_ctx.get(), GGML_TYPE_F32, n_dim);
    ggml_backend_buffer_ptr source_buffer(ggml_backend_alloc_ctx_tensors(source_ctx.get(), source));
    ggml_backend_buffer_ptr destination_buffer(ggml_backend_alloc_ctx_tensors(destination_ctx.get(), destination));
    CHECK(source_buffer && destination_buffer);
    int peer = 0;
    CUDA_OK(cudaDeviceCanAccessPeer(&peer, source_device, destination_device));
    if (!peer) {
        CHECK(!destination->iface.cpy_tensor_async(source, destination, src, dst));
        CHECK(!destination_buffer->iface.cpy_tensor(destination_buffer.get(), src, dst));
    }
    std::vector<float> expected(n_dim);
    for (int i = 0; i < n_dim; ++i) {
        expected[i] = i - 123.25f;
    }
    ggml_backend_tensor_set_async(source, src, expected.data(), 0, ggml_nbytes(src));
    ggml_backend_tensor_copy_async(source, destination, src, dst);
    ggml_backend_synchronize(destination);
    CHECK(active_grouped_tensor_values(dst) == expected);
    expected[0] += 1;
    ggml_backend_tensor_set(src, expected.data(), 0, ggml_nbytes(src));
    ggml_backend_tensor_copy(src, dst);
    CHECK(active_grouped_tensor_values(dst) == expected);
    fprintf(stderr, "test-moe-cache: boundary copy %d -> %d exact, peer=%d\n", source_device, destination_device, peer);
}

void run_layers(const std::vector<int> & devices,
                size_t                   host_budget,
                bool                     split_route = false,
                uint32_t                 layout      = GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP,
                ggml_type                type        = GGML_TYPE_Q4_0,
                bool                     pageable    = false,
                uint32_t                 cached_layers = all_cached_layers,
                uint32_t                 cache_slots   = n_slots,
                bool                     named_weights = false) {
    int physical_devices = 0;
    CUDA_OK(cudaGetDeviceCount(&physical_devices));
    std::array<int, 2> ordinary_devices = {};
    for (int layer = 0; layer < 2; ++layer) {
        const int device = devices[layer % devices.size()];
        ordinary_devices[layer] = device < physical_devices ? device : 0;
    }
    layer_fixture                 fixture(host_budget, layout, type, pageable, ordinary_devices, cached_layers, cache_slots, named_weights);
    std::vector<ggml_backend_ptr> owners;
    std::vector<ggml_backend_t>   backends;
    for (int device : devices) {
        owners.emplace_back(ggml_backend_cuda_init(device));
        CHECK(owners.back() != nullptr);
        backends.push_back(owners.back().get());
    }
    const bool virtual_devices = std::any_of(devices.begin(), devices.end(), [&](int device) { return device >= physical_devices; });
    if (devices.size() == 2 && devices[0] != devices[1] && devices[0] < physical_devices && devices[1] < physical_devices) {
        test_boundary_copy(backends[0], backends[1], devices[0], devices[1]);
        test_boundary_copy(backends[1], backends[0], devices[1], devices[0]);
    }
    owners.emplace_back(ggml_backend_cpu_init());
    CHECK(owners.back() != nullptr);
    backends.push_back(owners.back().get());
    split_probe probe{ &fixture, backends, {} };
    for (auto * backend : backends) {
        probe.delegates.push_back(backend->iface.graph_compute);
        backend->iface.graph_compute = probe_compute;
    }
    active_probe = &probe;
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends.data(), nullptr, backends.size(), 256, true, true));
    CHECK(sched != nullptr);
    constexpr uint64_t snapshot_owner_generation = 17;
    constexpr uint64_t snapshot_allocator_generation = 23;
    finalized_snapshot_ids required_snapshot_ids;
    const auto allocate = [&](bool affinity = true, bool measure = false, bool reference = false) {
        required_snapshot_ids.clear();
        CHECK(fixture.result.get_moe_regions().size() == 2);
        for (int layer = 0; layer < 2; ++layer) {
            auto * owner  = backends[reference ? 0 : layer % devices.size()];
            auto & region = fixture.result.get_moe_regions()[layer];
            if (!affinity) {
                continue;
            }
            CHECK(region.layer == layer && region.down == fixture.down[layer] && region.route == fixture.ids[layer]);
            CHECK(std::find(region.operations.begin(), region.operations.end(), fixture.input) ==
                  region.operations.end());
            CHECK(std::find(region.inputs.begin(), region.inputs.end(), fixture.logits[layer]) != region.inputs.end());
            check_region_descriptor(region, fixture.result.get_gf(), fixture.down[layer], fixture.ids[layer]);
            const auto route_type = region.route->type;
            region.route->type = GGML_TYPE_F32;
            CHECK(!region.place(sched.get(), owner));
            CHECK(ggml_backend_sched_get_tensor_backend(sched.get(), region.output) == nullptr);
            region.route->type = route_type;
            auto unsupported                     = *owner;
            auto unsupported_device              = *ggml_backend_get_device(owner);
            unsupported.device                   = &unsupported_device;
            unsupported_device.iface.supports_op = [](ggml_backend_dev_t, const ggml_tensor * op) {
                return op->op != GGML_OP_ADD;
            };
            std::vector<ggml_tensor *> inplace;
            for (auto * node : region.operations) {
                if (node->op == GGML_OP_ADD) {
                    CHECK(node->view_src == nullptr);
                    node->view_src = node->src[0];
                    inplace.push_back(node);
                }
            }
            CHECK(!inplace.empty());
            CHECK(!region.place(sched.get(), &unsupported));
            for (auto * node : inplace) {
                node->view_src = nullptr;
            }
            for (auto * node : region.operations) {
                CHECK(ggml_backend_sched_get_tensor_backend(sched.get(), node) == nullptr);
            }
            CHECK(!region.place(sched.get(), &unsupported));
            for (auto * node : region.operations) {
                CHECK(ggml_backend_sched_get_tensor_backend(sched.get(), node) == nullptr);
            }
            region.output->buffer = fixture.down[layer]->buffer;
            CHECK(!region.place(sched.get(), owner));
            region.output->buffer = nullptr;
            for (auto * node : region.operations) {
                CHECK(ggml_backend_sched_get_tensor_backend(sched.get(), node) == nullptr);
            }
            CHECK(region.place(sched.get(), owner));
        }
        if (affinity) {
            CHECK(llama_speculative_grouped_intent_test_access::graph_supported(sched.get(), fixture.result.get_gf()));
            for (auto * node : fixture.result.get_moe_regions()[1].operations) {
                if (node->op == GGML_OP_MUL_MAT_ID && node->src[0] == fixture.down[1]) {
                    ggml_backend_sched_set_tensor_backend(sched.get(), node, backends.back());
                    CHECK(!llama_speculative_grouped_intent_test_access::graph_supported(sched.get(),
                                                                                         fixture.result.get_gf()));
                    ggml_backend_sched_set_tensor_backend(sched.get(), node, backends[reference ? 0 : 1 % devices.size()]);
                }
            }
        }
        if (affinity && split_route) {
            ggml_backend_sched_set_tensor_backend(sched.get(), fixture.ids[1]->src[0], backends.back());
            ggml_backend_sched_set_tensor_backend(sched.get(), fixture.ids[1], backends.back());
        }
        if (measure) {
            std::vector<size_t> sizes(backends.size());
            ggml_backend_sched_reserve_size(sched.get(), fixture.result.get_gf(), sizes.data());
        } else {
            region_finalize_tally finalize_tally;
            const auto finalize = [&](llm_graph_moe_region & region) {
                return finalize_tally.record(region.finalize_metadata(
                    sched.get(), fixture.result.get_gf(), snapshot_owner_generation, snapshot_allocator_generation));
            };
            if (affinity) {
                CHECK(fixture.result.get_moe_regions()[0].finalized_metadata == nullptr);
                CHECK(finalize(fixture.result.get_moe_regions()[0]) ==
                      GGML_BACKEND_SCHED_REGION_STATUS_V1_NOT_FINALIZED);
            }
            CHECK(ggml_backend_sched_alloc_graph(sched.get(), fixture.result.get_gf()));
            if (affinity) {
                ggml_backend_sched_set_eval_callback(sched.get(), reject_region_callback, nullptr);
                CHECK(finalize(fixture.result.get_moe_regions()[0]) == GGML_BACKEND_SCHED_REGION_STATUS_V1_CALLBACK);
                CHECK(fixture.result.get_moe_regions()[0].finalized_metadata == nullptr);
                ggml_backend_sched_set_eval_callback(sched.get(), nullptr, nullptr);
                for (auto & region : fixture.result.get_moe_regions()) {
                    CHECK(finalize(region) == GGML_BACKEND_SCHED_REGION_STATUS_V1_OK);
                    check_finalized_region(region);
                }
                required_snapshot_ids = require_structural_regions(
                    fixture.result, snapshot_owner_generation, snapshot_allocator_generation);
                std::vector<std::shared_ptr<const llm_graph_moe_finalized_snapshot>> retained;
                for (const auto & region : fixture.result.get_moe_regions()) {
                    retained.push_back(region.finalized_metadata);
                }
                fixture.result.get_moe_regions()[0].finalized_metadata.reset();
                CHECK(!required_structural_regions_present(
                    fixture.result, snapshot_owner_generation, snapshot_allocator_generation, &required_snapshot_ids));
                for (auto & region : fixture.result.get_moe_regions()) {
                    region.finalized_metadata.reset();
                }
                CHECK(std::all_of(fixture.result.get_moe_regions().begin(), fixture.result.get_moe_regions().end(),
                                  [](const llm_graph_moe_region & region) {
                                      return region.finalized_metadata == nullptr;
                                  }));
                for (size_t i = 0; i < retained.size(); ++i) {
                    fixture.result.get_moe_regions()[i].finalized_metadata = std::move(retained[i]);
                }
                CHECK(required_structural_regions_present(
                    fixture.result, snapshot_owner_generation, snapshot_allocator_generation, &required_snapshot_ids));
                auto consumers = fixture.result.get_moe_regions()[0].live_outs[0].consumers;
                fixture.result.get_moe_regions()[0].live_outs[0].consumers.clear();
                CHECK(finalize(fixture.result.get_moe_regions()[0]) ==
                      GGML_BACKEND_SCHED_REGION_STATUS_V1_INVALID_ARGUMENT);
                CHECK(fixture.result.get_moe_regions()[0].finalized_metadata == nullptr);
                fixture.result.get_moe_regions()[0].live_outs[0].consumers = std::move(consumers);
                auto & region = fixture.result.get_moe_regions()[0];
                ggml_tensor * forward = region.body_operations.front();
                CHECK(forward->src[GGML_MAX_SRC - 1] == nullptr);
                forward->src[GGML_MAX_SRC - 1] = region.body_operations.back();
                CHECK(finalize(region) == GGML_BACKEND_SCHED_REGION_STATUS_V1_INCOMPLETE_CUT);
                CHECK(region.finalized_metadata == nullptr);
                forward->src[GGML_MAX_SRC - 1] = nullptr;

                CHECK(forward->view_src == nullptr);
                forward->view_src = region.dynamic_inputs[0];
                CHECK(finalize(region) == GGML_BACKEND_SCHED_REGION_STATUS_V1_INCOMPLETE_CUT);
                CHECK(region.finalized_metadata == nullptr);
                forward->view_src = nullptr;

                ggml_tensor * dynamic = region.dynamic_inputs[0];
                region.dynamic_inputs[0] = region.body_operations[0];
                CHECK(finalize(region) == GGML_BACKEND_SCHED_REGION_STATUS_V1_INVALID_ARGUMENT);
                CHECK(region.finalized_metadata == nullptr);
                region.dynamic_inputs[0] = dynamic;

                auto output = std::find_if(region.body_operations.begin(), region.body_operations.end(),
                    [&](const ggml_tensor * tensor) {
                        return std::none_of(region.live_outs.begin(), region.live_outs.end(),
                            [&](const llm_graph_moe_live_out & live) { return live.tensor == tensor; });
                    });
                CHECK(output != region.body_operations.end());
                const int32_t flags = (*output)->flags;
                (*output)->flags |= GGML_TENSOR_FLAG_OUTPUT;
                CHECK(finalize(region) == GGML_BACKEND_SCHED_REGION_STATUS_V1_INCOMPLETE_CUT);
                CHECK(region.finalized_metadata == nullptr);
                (*output)->flags = flags;

                region.live_outs.push_back(region.live_outs[0]);
                CHECK(finalize(region) == GGML_BACKEND_SCHED_REGION_STATUS_V1_INVALID_ARGUMENT);
                CHECK(region.finalized_metadata == nullptr);
                region.live_outs.pop_back();

                region.live_outs[0].consumers.push_back(region.live_outs[0].consumers[0]);
                CHECK(finalize(region) == GGML_BACKEND_SCHED_REGION_STATUS_V1_INVALID_ARGUMENT);
                CHECK(region.finalized_metadata == nullptr);
                region.live_outs[0].consumers.pop_back();

                ggml_cgraph * graph = fixture.result.get_gf();
                ggml_tensor * final_node = graph->nodes[graph->n_nodes - 1];
                graph->nodes[graph->n_nodes - 1] = nullptr;
                CHECK(finalize(region) == GGML_BACKEND_SCHED_REGION_STATUS_V1_NOT_FINALIZED);
                CHECK(region.finalized_metadata == nullptr);
                graph->nodes[graph->n_nodes - 1] = final_node;

                ggml_tensor * weight = fixture.down[0];
                const ggml_type weight_type = weight->type;
                weight->type = static_cast<ggml_type>(31);
                CHECK(finalize(region) == GGML_BACKEND_SCHED_REGION_STATUS_V1_INVALID_ARGUMENT);
                CHECK(region.finalized_metadata == nullptr);
                weight->type = static_cast<ggml_type>(999);
                CHECK(finalize(region) == GGML_BACKEND_SCHED_REGION_STATUS_V1_INVALID_ARGUMENT);
                CHECK(region.finalized_metadata == nullptr);
                weight->type = weight_type;
                const int64_t weight_width = weight->ne[0];
                weight->ne[0] = 1;
                CHECK(finalize(region) == GGML_BACKEND_SCHED_REGION_STATUS_V1_INVALID_ARGUMENT);
                CHECK(region.finalized_metadata == nullptr);
                weight->ne[0] = weight_width;
                const int64_t weight_rows = weight->ne[1];
                weight->ne[1] = 0;
                CHECK(finalize(region) == GGML_BACKEND_SCHED_REGION_STATUS_V1_INVALID_ARGUMENT);
                CHECK(region.finalized_metadata == nullptr);
                weight->ne[1] = weight_rows;
                const size_t weight_stride = weight->nb[2];
                weight->nb[2] = 1;
                CHECK(finalize(region) == GGML_BACKEND_SCHED_REGION_STATUS_V1_INVALID_ARGUMENT);
                CHECK(region.finalized_metadata == nullptr);
                weight->nb[2] = weight_stride;
                const int64_t weight_dim3 = weight->ne[3];
                const size_t weight_stride3 = weight->nb[3];
                weight->ne[3] = 2;
                weight->nb[3] = SIZE_MAX;
                CHECK(finalize(region) == GGML_BACKEND_SCHED_REGION_STATUS_V1_INVALID_ARGUMENT);
                CHECK(region.finalized_metadata == nullptr);
                weight->ne[3] = weight_dim3;
                weight->nb[3] = weight_stride3;

                const ggml_op body_op = forward->op;
                forward->op = static_cast<ggml_op>(999);
                CHECK(finalize(region) == GGML_BACKEND_SCHED_REGION_STATUS_V1_INVALID_ARGUMENT);
                CHECK(region.finalized_metadata == nullptr);
                forward->op = GGML_OP_MAP_CUSTOM1;
                CHECK(finalize(region) == GGML_BACKEND_SCHED_REGION_STATUS_V1_INVALID_ARGUMENT);
                CHECK(region.finalized_metadata == nullptr);
                forward->op = body_op;

                CHECK(weight->src[0] == nullptr && weight->view_src == nullptr);
                weight->src[0] = region.body_operations.back();
                CHECK(finalize(region) == GGML_BACKEND_SCHED_REGION_STATUS_V1_INCOMPLETE_CUT);
                CHECK(region.finalized_metadata == nullptr);
                weight->src[0] = weight;
                CHECK(finalize(region) == GGML_BACKEND_SCHED_REGION_STATUS_V1_INCOMPLETE_CUT);
                CHECK(region.finalized_metadata == nullptr);
                weight->src[0] = nullptr;
                CHECK(finalize(fixture.result.get_moe_regions()[0]) == GGML_BACKEND_SCHED_REGION_STATUS_V1_OK);
                check_finalized_region(fixture.result.get_moe_regions()[0]);
                required_snapshot_ids = require_structural_regions(
                    fixture.result, snapshot_owner_generation, snapshot_allocator_generation);
                CHECK(finalize_tally.statuses[GGML_BACKEND_SCHED_REGION_STATUS_V1_OK] == 3);
                CHECK(finalize_tally.statuses[GGML_BACKEND_SCHED_REGION_STATUS_V1_NOT_FINALIZED] == 2);
                CHECK(finalize_tally.statuses[GGML_BACKEND_SCHED_REGION_STATUS_V1_CALLBACK] == 1);
                CHECK(finalize_tally.statuses[GGML_BACKEND_SCHED_REGION_STATUS_V1_INCOMPLETE_CUT] == 5);
                CHECK(finalize_tally.statuses[GGML_BACKEND_SCHED_REGION_STATUS_V1_INVALID_ARGUMENT] == 12);
                CHECK(finalize_tally.statuses[GGML_BACKEND_SCHED_REGION_STATUS_V1_CROSS_SPLIT] == 0);
                CHECK(finalize_tally.statuses[GGML_BACKEND_SCHED_REGION_STATUS_V1_CAPACITY] == 0);
                uint32_t finalize_attempts = 0;
                for (uint32_t count : finalize_tally.statuses) {
                    finalize_attempts += count;
                }
                CHECK(finalize_attempts == 23 &&
                      finalize_attempts - finalize_tally.statuses[GGML_BACKEND_SCHED_REGION_STATUS_V1_OK] == 20);
            }
        }
        fprintf(stderr, "test-moe-cache: layer fixture devices=%zu splits=%d host_budget=%zu cached_layers=0x%x slots=%u\n",
                devices.size(), ggml_backend_sched_get_n_splits(sched.get()), host_budget, cached_layers, cache_slots);
        for (int layer = 0; layer < 2; ++layer) {
            if (affinity) {
                CHECK(ggml_backend_sched_get_tensor_backend(sched.get(), fixture.output[layer]) ==
                      backends[reference ? 0 : layer % devices.size()]);
                for (auto * node : fixture.result.get_moe_regions()[layer].operations) {
                    auto * actual = ggml_backend_sched_get_tensor_backend(sched.get(), node);
                    if (ggml_is_view(node)) {
                        CHECK(actual == ggml_backend_sched_get_tensor_backend(sched.get(), node->view_src));
                    } else {
                        auto * expected = split_route && layer == 1 && node == fixture.ids[layer]->src[0] ?
                                              backends.back() :
                                              backends[reference ? 0 : layer % devices.size()];
                        CHECK(actual == expected);
                    }
                }
            } else {
                auto * actual = ggml_backend_sched_get_tensor_backend(sched.get(), fixture.output[layer]);
                fprintf(stderr, "test-moe-cache: default-placement layer=%d expected=%s actual=%s\n", layer,
                        ggml_backend_name(backends[layer % devices.size()]), ggml_backend_name(actual));
            }
        }
    };
    const auto enabled  = fixture.manifest();
    const auto disabled = candidate_snapshot_v2(cache_slots, nullptr, 0, nullptr, 0);
    const auto publish  = [&](const ggml_backend_moe_candidate_snapshot_v2 & snapshot) {
        ggml_backend_sched_synchronize(sched.get());
        for (size_t i = 0; i < devices.size(); ++i) {
            CHECK(ggml_backend_cuda_moe_candidate_replace_v2(backends[i], &snapshot) ==
                  GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        }
    };
    using layer_outputs = std::array<std::vector<float>, 2>;
    const auto compute = [&](int iteration, layer_outputs * outputs = nullptr) {
        fixture.set_inputs(iteration);
        const auto certificate = layer_certificate();
        CHECK(ggml_backend_sched_graph_compute_async_ext(sched.get(), fixture.result.get_gf(), &certificate) ==
              GGML_STATUS_SUCCESS);
        ggml_backend_sched_synchronize(sched.get());
        if (!required_snapshot_ids.empty()) {
            CHECK(required_structural_regions_present(fixture.result, snapshot_owner_generation,
                                                      snapshot_allocator_generation, &required_snapshot_ids));
        }
        if (outputs != nullptr) {
            for (int layer = 0; layer < 2; ++layer) {
                (*outputs)[layer] = active_grouped_tensor_values(fixture.output[layer]);
            }
        }
        return active_grouped_tensor_values(fixture.output[1]);
    };
    std::vector<layer_outputs> ordinary_expected;
    if (cached_layers != 0 && cached_layers != all_cached_layers) {
        // Use ordinary device weights to check both legacy and grouped cache execution.
        layer_fixture ordinary(0, layout, type, pageable, ordinary_devices, 0, cache_slots, named_weights);
        ggml_backend_sched_ptr ordinary_sched(ggml_backend_sched_new(backends.data(), nullptr, backends.size(), 256, true, true));
        CHECK(ordinary_sched != nullptr);
        for (int layer = 0; layer < 2; ++layer) {
            CHECK(ordinary.result.get_moe_regions()[layer].place(ordinary_sched.get(), backends[layer % devices.size()]));
        }
        CHECK(ggml_backend_sched_alloc_graph(ordinary_sched.get(), ordinary.result.get_gf()));
        for (int iteration = 0; iteration < 8; ++iteration) {
            ordinary.set_inputs(iteration);
            CHECK(ggml_backend_sched_graph_compute(ordinary_sched.get(), ordinary.result.get_gf()) == GGML_STATUS_SUCCESS);
            ordinary_expected.emplace_back();
            for (int layer = 0; layer < 2; ++layer) {
                ordinary_expected.back()[layer] = active_grouped_tensor_values(ordinary.output[layer]);
            }
        }
    }
    if (devices.size() == 2) {
        allocate(true, true);
        for (int layer = 0; layer < 2; ++layer) {
            CHECK(ggml_backend_sched_get_tensor_backend(sched.get(), fixture.output[layer]) == backends[layer]);
        }
        ggml_backend_sched_reset(sched.get());
        fixture.build_graph();
    }
    ggml_backend_buffer_ptr reference_auxiliaries;
    std::array<void *, 2> pageable_bias_data = {};
    std::array<ggml_backend_buffer_t, 2> pageable_bias_buffers = {};
    if (pageable) {
        const size_t size = ggml_nbytes(fixture.down_bias[0]);
        reference_auxiliaries.reset(ggml_backend_buft_alloc_buffer(ggml_backend_cuda_moe_cached_buffer_type(), 2 * size));
        CHECK(reference_auxiliaries != nullptr);
        for (int layer = 0; layer < 2; ++layer) {
            auto * bias = fixture.down_bias[layer];
            std::vector<char> bytes(size);
            ggml_backend_tensor_get(bias, bytes.data(), 0, size);
            pageable_bias_data[layer] = bias->data;
            pageable_bias_buffers[layer] = bias->buffer;
            bias->data = static_cast<char *>(ggml_backend_buffer_get_base(reference_auxiliaries.get())) + layer * size;
            bias->buffer = reference_auxiliaries.get();
            memcpy(bias->data, bytes.data(), size);
        }
    }
    publish(disabled);
    allocate(false);
    ggml_backend_sched_reset(sched.get());
    fixture.build_graph();
    // The legacy reference uses physical GPU 0; its raw CUDA ordinal API does not emulate devices.
    allocate(true, false, virtual_devices);
    std::vector<std::vector<float>> expected;
    std::vector<layer_outputs> expected_layers;
    for (int iteration = 0; iteration < 8; ++iteration) {
        expected_layers.emplace_back();
        expected.push_back(compute(iteration, &expected_layers.back()));
        if (!ordinary_expected.empty()) {
            for (int layer = 0; layer < 2; ++layer) {
                check_active_grouped_exact_output(ordinary_expected[iteration][layer], expected_layers.back()[layer]);
            }
        }
    }
    for (size_t i = 0; i < devices.size(); ++i) {
        const auto telemetry = ggml_cuda_moe_grouped_context_test_access::legacy_debug_telemetry(
            *ggml_cuda_moe_grouped_context_for_test(backends[i]), true);
        fprintf(stderr, "test-moe-cache: route publication cached_layers=0x%x owner=%zu slots=%u named=%d ops=%llu reused_ids=%llu d2h_syncs=%llu publications=%zu\n",
            cached_layers, i, cache_slots, named_weights, (unsigned long long) telemetry.ops,
            (unsigned long long) telemetry.ids_cache_hits, (unsigned long long) telemetry.ids_d2h_sync_count,
            ggml_cuda_moe_ids_cache_count_for_test(backends[i]));
        if (cached_layers == 0) {
            CHECK(ggml_cuda_moe_ids_cache_count_for_test(backends[i]) == 0);
        }
        const char * disable_fusion = getenv("GGML_CUDA_DISABLE_FUSION");
        if (named_weights && cached_layers != 0 && !split_route && !virtual_devices &&
                (!disable_fusion || std::atoi(disable_fusion) == 0)) {
            CHECK(telemetry.ids_d2h_sync_count == 0);
        }
        if (!split_route && !virtual_devices && (!disable_fusion || std::atoi(disable_fusion) == 0)) {
            size_t expected_publications = 0;
            for (int layer = 0; layer < 2; ++layer) {
                expected_publications += (cached_layers & (1u << layer)) != 0 && layer % devices.size() == i;
            }
            CHECK(ggml_cuda_moe_ids_cache_count_for_test(backends[i]) == expected_publications);
        }
    }
    if (pageable) {
        for (int layer = 0; layer < 2; ++layer) {
            fixture.down_bias[layer]->data = pageable_bias_data[layer];
            fixture.down_bias[layer]->buffer = pageable_bias_buffers[layer];
        }
    }
    publish(enabled);
    ggml_backend_sched_reset(sched.get());
    fixture.build_graph();
    allocate();
    std::vector<uint64_t> legacy_before;
    for (size_t i = 0; i < devices.size(); ++i) {
        legacy_before.push_back(active_grouped_legacy_op_count(backends[i]));
    }
    for (int iteration = 0; iteration < 8; ++iteration) {
        layer_outputs actual;
        const auto final = compute(iteration, &actual);
        check_active_grouped_exact_output(expected[iteration], final);
        for (int layer = 0; layer < 2; ++layer) {
            check_active_grouped_exact_output(expected_layers[iteration][layer], actual[layer]);
        }
    }
    for (size_t i = 0; i < devices.size(); ++i) {
        auto * context = ggml_cuda_moe_grouped_context_for_test(backends[i]);
        CHECK(context != nullptr);
        const auto telemetry = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
        fprintf(stderr, "test-moe-cache: device=%d grouped_completed=%llu legacy_ops=%llu\n", devices[i],
                (unsigned long long) telemetry.completed,
                (unsigned long long) active_grouped_legacy_op_count(backends[i]));
        const bool fallback_owner = split_route && (devices.size() == 1 || i == 1);
        CHECK(telemetry.decode_staged == (fallback_owner ? 8 : 0));
        CHECK(telemetry.fallback == telemetry.decode_staged);
        uint64_t expected_completed = 0;
        for (int layer = 0; layer < 2; ++layer) {
            if ((cached_layers & (1u << layer)) && layer % devices.size() == i &&
                !(split_route && layer == 1)) {
                expected_completed += 8;
            }
        }
        CHECK(telemetry.completed == expected_completed && telemetry.decode_grouped == expected_completed);
        CHECK(telemetry.ready == expected_completed);
        if (!fallback_owner && expected_completed > 0) {
            CHECK(active_grouped_legacy_op_count(backends[i]) == legacy_before[i]);
            if (getenv("GGML_CUDA_DISABLE_GRAPHS") == nullptr) {
                CHECK(telemetry.captures > 0 && telemetry.replays > 0);
            } else {
                CHECK(telemetry.direct > 0 && telemetry.captures == 0 && telemetry.replays == 0);
            }
        }
        CHECK(telemetry.prepare_error == 0 && telemetry.finish_error == 0);
        CHECK(telemetry.submitted == telemetry.completed);
        if (cached_layers == 0) {
            CHECK(telemetry.admitted_banks == 0 && telemetry.h2d_banks == 0 && telemetry.h2d_bytes == 0);
        }
        for (int layer = 0; layer < 2; ++layer) {
            ggml_cuda_moe_candidate_group_key key;
            if (cached_layers & (1u << layer)) {
                CHECK(context->find_down_group_key(fixture.down[layer], &key));
                ggml_cuda_moe_candidate_group_info info;
                CHECK(context->get_group(key, &info) && info.semantic_group_index == (uint32_t) layer);
                if (layer % devices.size() != i) {
                    CHECK(!ggml_cuda_moe_grouped_context_test_access::has_device_resource(*context, key));
                }
            } else {
                CHECK(!context->find_down_group_key(fixture.down[layer], &key));
            }
        }
    }
    CHECK(probe.grouped_splits[0] > 0 && probe.imported_splits[0] == 0);
    CHECK(split_route ? probe.imported_splits[1] > 0 : probe.imported_splits[1] == 0);
    CHECK(split_route || probe.grouped_splits[1] > 0);

    if (split_route || devices.size() == 2) {
        probe.fail_layer       = 1;
        const auto certificate = layer_certificate();
        CHECK(ggml_backend_sched_graph_compute_async_ext(sched.get(), fixture.result.get_gf(), &certificate) ==
              GGML_STATUS_FAILED);
        ggml_backend_sched_synchronize(sched.get());
        CHECK(probe.failed);  // No downstream split may run after the injected failure.
        if (devices.size() == 2) {
            for (size_t i = 0; i < devices.size(); ++i) {
                const auto telemetry = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(
                    *ggml_cuda_moe_grouped_context_for_test(backends[i]));
                CHECK(telemetry.submitted == (i == 0 && (cached_layers & 1u) ? 1 : 0) && telemetry.completed == telemetry.submitted);
                CHECK(telemetry.fallback == 0);
            }
        }
        // Failed graph outputs can alias scratch. Read them only after successful recomputation.
        probe.failed     = false;
        probe.fail_layer = -1;
        check_active_grouped_exact_output(expected[0], compute(0));
    }
    if (split_route) {
        const auto required        = layer_certificate(true);
        const int  imported_before = probe.imported_splits[1];
        CHECK(ggml_backend_sched_graph_compute_async_ext(sched.get(), fixture.result.get_gf(), &required) !=
              GGML_STATUS_SUCCESS);
        ggml_backend_sched_synchronize(sched.get());
        CHECK(probe.imported_splits[1] > imported_before);
    } else if (devices.size() == 2) {
        auto *     first             = ggml_cuda_moe_grouped_context_for_test(backends[0]);
        auto *     second            = ggml_cuda_moe_grouped_context_for_test(backends[1]);
        const auto first_generation  = first->state().generation;
        const auto second_generation = second->state().generation;
        CHECK(ggml_backend_cuda_moe_candidate_replace_v2(backends[1], &enabled) ==
              GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        CHECK(first->state().generation == first_generation && second->state().generation > second_generation);
        ggml_backend_sched_reset(sched.get());
        fixture.build_graph();
        allocate();
        check_active_grouped_exact_output(expected[1], compute(1));
    }
    if (!split_route) {
        fixture.set_inputs(0);
        const auto required = layer_certificate(true);
        CHECK(ggml_backend_sched_graph_compute_ext(sched.get(), fixture.result.get_gf(), &required) ==
              GGML_STATUS_SUCCESS);
        check_active_grouped_exact_output(expected[0], active_grouped_tensor_values(fixture.output[1]));
    }
    ggml_backend_sched_synchronize(sched.get());
    sched.reset();
    for (size_t i = 0; i < backends.size(); ++i) {
        backends[i]->iface.graph_compute = probe.delegates[i];
    }
    active_probe = nullptr;
    int survivor_layer = -1;
    for (int layer = 0; layer < 2; ++layer) {
        if ((cached_layers & (1u << layer)) &&
            (survivor_layer < 0 || layer % devices.size() == devices.size() - 1)) {
            survivor_layer = layer;
        }
    }
    if (!split_route && !virtual_devices && survivor_layer >= 0) {
        const size_t survivor_owner = survivor_layer % devices.size();
        if (devices.size() == 2) {
            owners[1 - survivor_owner].reset();
        }
        auto *                        survivor = backends[survivor_owner];
        active_grouped_dispatch_graph shared;
        shared.n_experts = n_experts;
        shared.n_used    = n_used;
        for (const auto & bank : fixture.tensors) {
            if (bank.group_index == (uint32_t) survivor_layer) {
                if (bank.status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE) {
                    shared.banks.push_back(const_cast<ggml_tensor *>(bank.tensor));
                    shared.roles.push_back(bank.role);
                } else {
                    CHECK(bank.status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_OUTPUT_BIAS);
                    shared.biases.push_back(const_cast<ggml_tensor *>(bank.tensor));
                    shared.bias_roles.push_back(bank.role);
                }
            }
        }
        auto graph = build_active_grouped_dispatch_graph(survivor, fixture.buft, type, layout, false, 1, n_experts,
                                                         n_used, n_dim, &shared);
        // Fill only graph inputs; the shared immutable weights remain unchanged.
        const auto input = cached_fusion_test_data(graph.input, 71);
        ggml_backend_tensor_set(graph.input, input.data(), 0, input.size());
        set_active_grouped_dispatch_logits({ &graph }, 0);
        if (pageable) {
            fixture.down_bias[survivor_layer]->data = static_cast<char *>(ggml_backend_buffer_get_base(reference_auxiliaries.get())) +
                survivor_layer * ggml_nbytes(fixture.down_bias[survivor_layer]);
            fixture.down_bias[survivor_layer]->buffer = reference_auxiliaries.get();
        }
        CHECK(ggml_backend_cuda_moe_candidate_replace_v2(survivor, &disabled) ==
              GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        const auto reference = run_active_grouped_dispatch(survivor, graph, 0);
        if (pageable) {
            fixture.down_bias[survivor_layer]->data = pageable_bias_data[survivor_layer];
            fixture.down_bias[survivor_layer]->buffer = pageable_bias_buffers[survivor_layer];
        }
        CHECK(ggml_backend_cuda_moe_candidate_replace_v2(survivor, &enabled) ==
              GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        auto * context = ggml_cuda_moe_grouped_context_for_test(survivor);
        CHECK(context != nullptr);
        (void) ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
        const auto legacy_before = active_grouped_legacy_op_count(survivor);
        const auto capability =
            native_mmid_capability(devices[survivor_owner], graph.banks[0], graph.n_rows, GGML_CUDA_MMID_MAPPING_DIRECT);
        const char * disable_fusion = getenv("GGML_CUDA_DISABLE_FUSION");
        // F3 writes the GLU result directly and leaves the gate/up sentinels intact.
        const bool   f3_skipped =
            (!disable_fusion || std::atoi(disable_fusion) == 0) &&
            layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE && capability.reason == GGML_CUDA_MMID_CAPABILITY_OK &&
            capability.selection == GGML_CUDA_MMID_CONSUMER_MMVQ && graph.banks[0]->type == graph.banks[1]->type &&
            ggml_are_same_shape(graph.banks[0], graph.banks[1]) && ggml_are_same_stride(graph.banks[0], graph.banks[1]);
        fprintf(stderr, "test-moe-cache: survivor device=%d layout=%u type=%s generation=%llu f3_skipped=%d\n",
                devices[survivor_owner], layout, ggml_type_name(type), (unsigned long long) context->state().generation,
                f3_skipped);
        for (int pass = 0; pass < 3; ++pass) {
            check_active_grouped_exact_output(reference, run_active_grouped_dispatch(survivor, graph, 0, f3_skipped));
        }
        ggml_backend_synchronize(survivor);
        const auto telemetry = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
        CHECK(telemetry.decode_grouped == 3 && telemetry.ready == 3 && telemetry.submitted == 3 &&
              telemetry.completed == 3);
        CHECK(telemetry.decode_staged == 0 && telemetry.fallback == 0 &&
              active_grouped_legacy_op_count(survivor) == legacy_before);
        CHECK(telemetry.prepare_error == 0 && telemetry.finish_error == 0);
        if (getenv("GGML_CUDA_DISABLE_GRAPHS") == nullptr) {
            CHECK(telemetry.direct == 1 && telemetry.captures == 1 && telemetry.replays == 1);
        } else {
            CHECK(telemetry.direct == 3 && telemetry.captures == 0 && telemetry.replays == 0);
        }
        {
        auto * backend = backends[survivor_layer % devices.size()];
        auto * context = ggml_cuda_moe_grouped_context_for_test(backend);
        ggml_cuda_moe_candidate_group_key key;
        ggml_cuda_moe_candidate_group_info info;
        CHECK(context->find_down_group_key(fixture.down[survivor_layer], &key) && context->get_group(key, &info));
        std::vector<int32_t> rank(info.n_slots);
        for (uint32_t i = 0; i < info.n_slots; ++i) { rank[i] = n_experts - 1 - i; }
        ggml_backend_moe_static_profile_v1 profile{fixture.down[survivor_layer], rank.data(), uint32_t(rank.size())};
        uint64_t bytes = 0;
        ggml_cuda_moe_grouped_acquisition acquisition;
        ggml_cuda_moe_grouped_transaction transaction;
        CHECK(context->acquire_group_resources(key, &acquisition));
        CHECK(context->begin_group_transaction(acquisition, &transaction));
        CHECK(!ggml_backend_cuda_moe_profile_initialize_v1(backend, &profile, 1, 0, &bytes));
        CHECK(context->end_group_transaction(transaction));
        auto invalid = rank;
        if (invalid.size() > 1) {
            invalid[1] = invalid[0];
            auto bad = profile;
            bad.experts = invalid.data();
            CHECK(!ggml_backend_cuda_moe_profile_initialize_v1(backend, &bad, 1, 0, &bytes));
        }
        CHECK(ggml_backend_cuda_moe_profile_initialize_v1(backend, &profile, 1, 0, &bytes));
        for (const auto expert : rank) {
            const int32_t slot = ggml_cuda_moe_grouped_context_test_access::device_slot_for_expert(*context, key, expert);
            CHECK(slot >= 0 && uint32_t(slot) < info.n_slots);
            for (const auto & bank : fixture.tensors) {
                if (bank.group_index != uint32_t(survivor_layer) || bank.status != GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE) { continue; }
                const auto * tensor = bank.tensor;
                const auto * data = static_cast<const uint8_t *>(ggml_cuda_moe_grouped_context_test_access::device_bank_data(*context, key, tensor));
                CHECK(data);
                const size_t stride = tensor->nb[2];
                std::vector<uint8_t> actual(stride), expected(stride);
                CUDA_OK(cudaMemcpy(actual.data(), data + size_t(slot) * stride, stride, cudaMemcpyDeviceToHost));
                ggml_backend_tensor_get(tensor, expected.data(), size_t(expert) * stride, stride);
                CHECK(actual == expected);
            }
        }
        CHECK(ggml_backend_cuda_moe_profile_initialize_v1(backend, &profile, 1, 0, &bytes) && bytes == 0);
        CHECK(!ggml_backend_cuda_moe_profile_initialize_v1(backend, &profile, 1, 1, &bytes));
        CHECK(!ggml_backend_cuda_moe_profile_initialize_v1(backend, &profile, 1, 3, &bytes));
        check_active_grouped_exact_output(reference, run_active_grouped_dispatch(survivor, graph, 0, f3_skipped));
        fprintf(stderr, "test-moe-cache: GPU profile all-bank payload, busy rejection, zero-copy repeat and post-profile output passed\n");
        }
    }
    owners.clear();
    if (host_budget != 0) {
        auto * budget = static_cast<moe_host_budget *>(fixture.buft->context);
        CHECK(budget != nullptr);
        std::lock_guard<std::mutex> lock(budget->mutex);
        CHECK(budget->source_bytes + budget->staging_peak <= budget->limit);
        CHECK(budget->materialized_bytes > 0);
        CHECK(budget->staging_bytes == 0 && budget->staging_optional_bytes == 0);
    }
    fixture.result.reset();
    CHECK(fixture.result.get_moe_regions().empty());
}

void test_shared_source_owners(int device, size_t host_budget, bool pageable = false) {
    ggml_backend_ptr first(ggml_backend_cuda_init(device));
    ggml_backend_ptr second(ggml_backend_cuda_init(device));
    ggml_backend_ptr third(ggml_backend_cuda_init(device));
    CHECK(first && second && third);
    auto * buft = pageable ? pageable_cached_buffer_type() : ggml_backend_cuda_moe_cached_bounded_buffer_type(host_budget);
    auto   a    = build_active_grouped_dispatch_graph(first.get(), buft, GGML_TYPE_Q4_0,
                                                      GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, pageable);
    auto   b = build_active_grouped_dispatch_graph(second.get(), buft, GGML_TYPE_Q4_0,
                                                   GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, pageable, 1, n_experts,
                                                   n_used, n_dim, &a);
    auto c = build_active_grouped_dispatch_graph(third.get(), buft, GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, pageable, 1, n_experts, n_used, n_dim, &a);
    initialize_active_grouped_dispatch_graphs({ &a, &b, &c });
    candidate_stamp_execution(b.graph, GGML_GRAPH_EXECUTION_DOMAIN_DRAFT, GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 1, 1, 0,
        GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED);
    candidate_stamp_execution(c.graph, GGML_GRAPH_EXECUTION_DOMAIN_MTP, GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL, 1, 1, 0,
        GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED);
    c.graph->execution_certificate.owner_namespace++;
    ggml_backend_moe_candidate_group_v2               group{ GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP,
                                                             GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY, 0, 0 };
    std::vector<ggml_backend_moe_candidate_tensor_v2> banks;
    for (size_t i = 0; i < a.banks.size(); ++i) {
        banks.push_back({ a.banks[i], 0, a.roles[i], GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE,
                          GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_CACHED_BUFFER, 0 });
    }
    if (pageable) {
        banks.push_back({a.down_scale, 0, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_SCALE,
            GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_OUTPUT_SCALE, GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_CACHED_BUFFER, 0});
    }
    const auto snapshot = candidate_snapshot_v2(n_slots, &group, 1, banks.data(), banks.size());
    if (host_budget) {
        CHECK(ggml_backend_cuda_moe_cached_configure_sources(buft, &snapshot));
    }
    CHECK(ggml_backend_cuda_moe_candidate_replace_v2(first.get(), &snapshot) ==
          GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v2(second.get(), &snapshot) ==
          GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v2(third.get(), &snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    (void) candidate_certify_graph(*ggml_cuda_moe_grouped_context_for_test(second.get()), b.graph);
    (void) candidate_certify_graph(*ggml_cuda_moe_grouped_context_for_test(third.get()), c.graph);
    const auto expected = run_active_grouped_dispatch(first.get(), a, 0);
    check_active_grouped_exact_output(expected, run_active_grouped_dispatch(second.get(), b, 0));
    check_active_grouped_exact_output(expected, run_active_grouped_dispatch(third.get(), c, 0));
    std::vector<float> draft_output, mtp_output;
    std::thread draft([&] { draft_output = run_active_grouped_dispatch(second.get(), b, 0); });
    std::thread mtp([&] { mtp_output = run_active_grouped_dispatch(third.get(), c, 0); });
    draft.join();
    mtp.join();
    check_active_grouped_exact_output(expected, draft_output);
    check_active_grouped_exact_output(expected, mtp_output);
    const auto generation = ggml_cuda_moe_grouped_context_for_test(second.get())->state().generation;
    CHECK(ggml_backend_cuda_moe_candidate_replace_v2(first.get(), &snapshot) ==
          GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(ggml_cuda_moe_grouped_context_for_test(second.get())->state().generation == generation);
    first.reset();
    check_active_grouped_exact_output(expected, run_active_grouped_dispatch(second.get(), b, 0));
    second.reset();
    check_active_grouped_exact_output(expected, run_active_grouped_dispatch(third.get(), c, 0));
    CHECK(active_grouped_legacy_op_count(third.get()) == 0);
    third.reset();
    if (host_budget) {
        auto * budget = static_cast<moe_host_budget *>(buft->context);
        CHECK(budget->staging_bytes == 0 && budget->staging_peak + budget->source_bytes <= budget->limit);
    }
    ggml_backend_cuda_moe_cached_free_buffer_type(buft);
    fprintf(stderr, "test-moe-cache: three shared-source owners replacement/teardown pageable=%d OK (one physical device)\n", pageable);
}

void test_hybrid_row_producers(const ggml_backend_moe_hybrid_region_v1 & region,
                             ggml_graph_execution_certificate certificate, bool pageable) {
    const auto & geometry = region.geometry;
    certificate.source_graph_uid = region.source_graph_uid;
    certificate.split_graph_uid = region.split_graph_uid;
    certificate.owner_generation = region.owner_generation;
    std::vector<uint8_t> accessible(region.query->n_sources, pageable ? 0 : 1);
    ggml_cuda_moe_hybrid_rows_query query;
    query.region = &region;
    query.certificate = certificate;
    query.plan_capacity = geometry.route_capacity;
    query.slot_capacity = 1;
    query.transfer_capacity = 1;
    query.workspace_generation = region.allocator_generation;
    query.resource_fingerprint = reinterpret_cast<uintptr_t>(region.query);
    query.device_alignment = 256;
    query.host_alignment = 64;
    query.producer_workspace_bytes = region.query->n_body_nodes * sizeof(void *) + 1;
    query.source_device_accessible = accessible.data();
    query.n_sources = accessible.size();
    ggml_cuda_moe_hybrid_rows_layout layout;
    std::vector<ggml_cuda_moe_hybrid_rows_source> sources;
    CHECK(ggml_cuda_moe_hybrid_rows_measure(query, layout, sources));
    CHECK(layout.slot_capacity < geometry.route_capacity && sources.size() == accessible.size());
    const auto check_layout = [&] {
        const auto check = [](size_t offset, size_t bytes, size_t alignment, size_t capacity) {
            CHECK(offset % alignment == 0 && offset <= capacity && bytes <= capacity - offset);
        };
        check(layout.plan_offset, layout.plan_bytes, query.device_alignment, layout.device_bytes);
        check(layout.packed_input_offset, layout.packed_input_bytes, query.device_alignment, layout.device_bytes);
        check(layout.gpu_output_offset, layout.private_output_bytes, query.device_alignment, layout.device_bytes);
        check(layout.publish_offset, layout.private_output_bytes, query.device_alignment, layout.device_bytes);
        check(layout.workspace_offset, query.producer_workspace_bytes, query.device_alignment, layout.device_bytes);
        check(layout.host_control_offset, layout.control_bytes, query.host_alignment, layout.pinned_bytes);
        check(layout.host_input_offset, layout.host_input_bytes, query.host_alignment, layout.pinned_bytes);
        check(layout.host_output_offset, layout.private_output_bytes, query.host_alignment, layout.pinned_bytes);
        for (size_t i = 0; i < sources.size(); ++i) {
            CHECK(sources[i].role == GGML_CUDA_MOE_HYBRID_SOURCE_MMID_WEIGHT);
            CHECK(sources[i].expert_bytes == region.query->sources[i].tensor->nb[2]);
            CHECK(sources[i].bytes == sources[i].expert_bytes * query.transfer_capacity);
            check(sources[i].device_offset, sources[i].bytes, query.device_alignment, layout.device_bytes);
        }
        if (query.host_control_alias) {
            CHECK(layout.control_offset == SIZE_MAX && layout.cpu_output_offset == SIZE_MAX);
        } else {
            check(layout.control_offset, layout.control_bytes, query.device_alignment, layout.device_bytes);
            check(layout.cpu_output_offset, layout.private_output_bytes, query.device_alignment, layout.device_bytes);
        }
        if (layout.staging_tile_bytes != 0) {
            for (size_t offset : layout.host_staging_offsets) {
                check(offset, layout.staging_tile_bytes, query.host_alignment, layout.pinned_bytes);
            }
        } else { CHECK(layout.host_staging_offsets[0] == SIZE_MAX && layout.host_staging_offsets[1] == SIZE_MAX); }
    };
    check_layout();
    const auto internal_layout = layout;
    const auto internal_sources = sources;
    auto external = query;
    external.external_publish_storage = true;
    external.external_transfer_storage = true;
    CHECK(ggml_cuda_moe_hybrid_rows_measure(external, layout, sources));
    CHECK(layout.identity != internal_layout.identity);
    CHECK(layout.device_bytes < internal_layout.device_bytes);
    CHECK(layout.pinned_bytes == internal_layout.pinned_bytes);
    CHECK(layout.publish_offset == SIZE_MAX);
    CHECK(layout.external_publish_storage && layout.external_transfer_storage);
    CHECK(layout.transfer_bytes == internal_layout.transfer_bytes);
    CHECK(sources.size() == internal_sources.size());
    for (size_t i = 0; i < sources.size(); ++i) {
        CHECK(sources[i].device_offset == SIZE_MAX);
        CHECK(sources[i].bytes == internal_sources[i].bytes);
        CHECK(sources[i].expert_bytes == internal_sources[i].expert_bytes);
    }
    auto external_limited = external;
    external_limited.device_capacity_bytes = layout.device_bytes;
    external_limited.pinned_capacity_bytes = layout.pinned_bytes;
    CHECK(ggml_cuda_moe_hybrid_rows_measure(external_limited, layout, sources));
    --external_limited.device_capacity_bytes;
    {
        ggml_cuda_moe_hybrid_rows_layout rejected;
        std::vector<ggml_cuda_moe_hybrid_rows_source> unused;
        CHECK(!ggml_cuda_moe_hybrid_rows_measure(external_limited, rejected, unused));
    }
    CHECK(ggml_cuda_moe_hybrid_rows_measure(query, layout, sources));
    const auto rejected_query = [&](ggml_cuda_moe_hybrid_rows_query invalid) {
        ggml_cuda_moe_hybrid_rows_layout rejected;
        std::vector<ggml_cuda_moe_hybrid_rows_source> unused;
        CHECK(!ggml_cuda_moe_hybrid_rows_measure(invalid, rejected, unused));
        CHECK(rejected.identity == 0 && unused.empty());
    };
    auto limited = query;
    limited.device_capacity_bytes = layout.device_bytes;
    limited.pinned_capacity_bytes = layout.pinned_bytes;
    CHECK(ggml_cuda_moe_hybrid_rows_measure(limited, layout, sources));
    --limited.device_capacity_bytes; rejected_query(limited); ++limited.device_capacity_bytes;
    --limited.pinned_capacity_bytes; rejected_query(limited);
    auto invalid = query;
    --invalid.plan_capacity; rejected_query(invalid);
    invalid = query; invalid.transfer_capacity = geometry.weight_capacity + 1; rejected_query(invalid);
    invalid = query; invalid.certificate.n_rows = 0; rejected_query(invalid);
    invalid = query; invalid.certificate.n_sequences = 0; rejected_query(invalid);
    invalid = query; invalid.producer_workspace_bytes = SIZE_MAX; rejected_query(invalid);
    const uint64_t identity = layout.identity;
    ++query.workspace_generation;
    CHECK(ggml_cuda_moe_hybrid_rows_measure(query, layout, sources) && layout.identity != identity);
    --query.workspace_generation;
    CHECK(ggml_cuda_moe_hybrid_rows_measure(query, layout, sources) && layout.identity == identity);

    const auto make_state = [&](uint32_t active_rows, bool all_cpu = false) {
        ggml_cuda_moe_hybrid_rows_test_state state;
        state.layout = layout;
        state.execution_identity = layout.identity;
        state.epoch = 17;
        state.active_rows = active_rows;
        state.n_sequences = 1;
        state.gpu_miss_quota = all_cpu ? 0 : 1;
        state.ids.assign(size_t(geometry.row_capacity) * layout.ids_row_stride, INT32_MAX);
        state.slot_for_expert.assign(geometry.expert_count, -1);
        state.expert_for_slot.assign(layout.slot_capacity, -1);
        if (!all_cpu) { state.slot_for_expert[0] = 0; state.expert_for_slot[0] = 0; }
        state.input.assign(size_t(geometry.row_capacity) * layout.input_width, NAN);
        state.gpu_output.resize(size_t(geometry.route_capacity) * layout.output_width);
        state.cpu_output.resize(state.gpu_output.size());
        state.producers.resize(size_t(geometry.row_capacity) + 2);
        for (uint32_t row = 0; row < active_rows; ++row) {
            for (uint32_t rank = 0; rank < geometry.routes_per_row; ++rank) {
                state.ids[size_t(row) * layout.ids_row_stride + rank] = rank == 0 ? 0 : 1 + row % 2;
            }
            for (uint32_t col = 0; col < layout.input_width; ++col) {
                state.input[size_t(row) * layout.input_width + col] = float(row * layout.input_width + col + 1);
            }
        }
        for (size_t i = 0; i < state.gpu_output.size(); ++i) {
            state.gpu_output[i] = float(i + 1000);
            state.cpu_output[i] = float(i + 5000);
        }
        return state;
    };
    const auto expected_producers = [&](ggml_cuda_moe_hybrid_rows_test_state & state) {
        std::vector<int32_t> distinct, resident, transfer;
        for (uint32_t row = 0; row < state.active_rows; ++row) {
            for (uint32_t rank = 0; rank < geometry.routes_per_row; ++rank) {
                const int32_t expert = state.ids[size_t(row) * layout.ids_row_stride + rank];
                if (std::find(distinct.begin(), distinct.end(), expert) == distinct.end()) {
                    distinct.push_back(expert);
                    if (state.slot_for_expert[expert] >= 0) { resident.push_back(expert); }
                    else if (transfer.size() < state.gpu_miss_quota) { transfer.push_back(expert); }
                }
                const size_t producer = std::find(resident.begin(), resident.end(), expert) != resident.end() ? 0 :
                    std::find(transfer.begin(), transfer.end(), expert) != transfer.end() ? 1 : 2 + row;
                auto & result = state.producers[producer];
                result.epoch = state.epoch;
                result.complete = 1;
                ++result.completed_routes;
            }
        }
    };
    const auto check_success = [&](ggml_cuda_moe_hybrid_rows_test_state state) {
        expected_producers(state);
        CHECK(ggml_cuda_moe_hybrid_rows_for_test(state));
        const auto & ticket = state.ticket;
        CHECK(state.plan_status == 0 && ticket.status == 0 && ticket.producers_ready == 1 && ticket.commit_decision == 2);
        CHECK(ticket.epoch == state.epoch + state.replays - 1 && ticket.accepted == 1 && ticket.published == 1);
        CHECK(ticket.route_count == state.active_rows * geometry.routes_per_row && ticket.published_routes == ticket.route_count);
        CHECK(ticket.completed_routes == ticket.route_count && ticket.completed_producers == ticket.expected_producers);
        CHECK(ticket.resident_lanes == state.producers[0].completed_routes && ticket.transfer_lanes == state.producers[1].completed_routes);
        CHECK(ticket.gpu_lanes == ticket.resident_lanes + ticket.transfer_lanes && ticket.weight_count < ticket.route_count);
        for (uint32_t route = 0; route < geometry.route_capacity; ++route) {
            if (route < ticket.route_count) {
                const auto & record = state.routes[route];
                CHECK(record.source_row == route / geometry.routes_per_row && record.source_route == route % geometry.routes_per_row);
                CHECK(record.scatter_destination == route && record.weight_index < ticket.weight_count);
                CHECK(state.weight_experts[record.weight_index] == state.ids[size_t(record.source_row) * layout.ids_row_stride + record.source_route]);
                const uint32_t kind = state.weight_classes[record.weight_index];
                const auto found = std::find(state.gpu_routes.begin(), state.gpu_routes.begin() + ticket.gpu_lanes, route);
                const size_t lane = found - state.gpu_routes.begin();
                if (kind != GGML_CUDA_MOE_HYBRID_CPU) {
                    CHECK(lane < ticket.gpu_lanes && (kind == GGML_CUDA_MOE_HYBRID_RESIDENT) == (lane < ticket.resident_lanes));
                }
                for (uint32_t col = 0; col < layout.output_width; ++col) {
                    const float expected = kind == GGML_CUDA_MOE_HYBRID_CPU ? state.cpu_output[size_t(route) * layout.output_width + col] :
                        state.gpu_output[lane * layout.output_width + col];
                    CHECK(state.output[size_t(route) * layout.output_width + col] == expected);
                }
            } else {
                CHECK(state.routes[route].source_row == UINT32_MAX);
                for (uint32_t col = 0; col < layout.output_width; ++col) { CHECK(state.output[size_t(route) * layout.output_width + col] == -123.0f); }
            }
            for (uint32_t col = 0; col < layout.input_width; ++col) {
                const float expected = route < ticket.gpu_lanes ?
                    state.input[size_t(state.routes[state.gpu_routes[route]].source_row) * layout.input_width + col] : 0.0f;
                CHECK(state.packed_input[size_t(route) * layout.input_width + col] == expected);
            }
        }
    };
    const auto check_rejected = [&](ggml_cuda_moe_hybrid_rows_test_state state) {
        CHECK(ggml_cuda_moe_hybrid_rows_for_test(state));
        if (state.import_plan) { CHECK(state.plan_status != 0); }
        CHECK(state.ticket.status != 0 && state.ticket.accepted == 0 && state.ticket.published == 0 &&
            state.ticket.published_routes == 0 && state.ticket.commit_decision == 1);
        CHECK(std::all_of(state.output.begin(), state.output.end(), [](float value) { return value == -123.0f; }));
    };
    for (bool alias : {false, true}) {
        query.host_control_alias = alias;
        CHECK(ggml_cuda_moe_hybrid_rows_measure(query, layout, sources));
        check_layout();
        auto base = make_state(geometry.row_capacity);
        check_success(base);
        base.capture = true; base.replays = 3; check_success(base);
        base = make_state(geometry.row_capacity); base.cancel_phase = 2; check_success(base);
        check_success(make_state(geometry.row_capacity, true));
        if (geometry.row_capacity > 2) { check_success(make_state(geometry.row_capacity - 1)); }
        for (bool capture : {false, true}) {
            auto imported = make_state(geometry.row_capacity);
            imported.import_plan = true;
            imported.capture = capture;
            imported.replays = capture ? 3 : 1;
            check_success(imported);
            expected_producers(imported);
            CHECK(imported.producers[0].completed_routes != 0 && imported.producers[1].completed_routes != 0);
            CHECK(std::any_of(imported.producers.begin() + 2, imported.producers.end(),
                [](const ggml_cuda_moe_hybrid_producer_result & producer) { return producer.completed_routes != 0; }));
            for (auto fault : {GGML_CUDA_MOE_HYBRID_IMPORT_TEST_STALE_EPOCH,
                    GGML_CUDA_MOE_HYBRID_IMPORT_TEST_WRONG_EXPERT, GGML_CUDA_MOE_HYBRID_IMPORT_TEST_DUPLICATE_EXPERT,
                    GGML_CUDA_MOE_HYBRID_IMPORT_TEST_MISSING_ROUTE, GGML_CUDA_MOE_HYBRID_IMPORT_TEST_WEIGHT_INDEX,
                    GGML_CUDA_MOE_HYBRID_IMPORT_TEST_TRANSFER_INDEX, GGML_CUDA_MOE_HYBRID_IMPORT_TEST_RECIPROCAL_MAP,
                    GGML_CUDA_MOE_HYBRID_IMPORT_TEST_CPU_ROUTE}) {
                auto rejected = imported;
                rejected.import_fault = fault;
                check_rejected(rejected);
            }
        }
        base = make_state(geometry.row_capacity);
        expected_producers(base);
        auto invalid_state = base;
        invalid_state.active_rows = 0; check_rejected(invalid_state);
        invalid_state = base; ++invalid_state.active_rows; check_rejected(invalid_state);
        invalid_state = base; invalid_state.n_sequences = 0; check_rejected(invalid_state);
        invalid_state = base; ++invalid_state.gpu_miss_quota; check_rejected(invalid_state);
        invalid_state = base; invalid_state.slot_for_expert[0] = layout.slot_capacity; check_rejected(invalid_state);
        invalid_state = base; invalid_state.ids[0] = -1; check_rejected(invalid_state);
        invalid_state = base; ++invalid_state.execution_identity; check_rejected(invalid_state);
        invalid_state = base; invalid_state.corrupt_route = true; check_rejected(invalid_state);
        invalid_state = base; --invalid_state.producers[0].epoch; check_rejected(invalid_state);
        invalid_state = base; invalid_state.producers[0].complete = 0; check_rejected(invalid_state);
        invalid_state = base; ++invalid_state.producers[0].completed_routes; check_rejected(invalid_state);
        invalid_state = base; invalid_state.producers[3].status = GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CANCELED; check_rejected(invalid_state);
        invalid_state = base; invalid_state.cancel_epoch = base.epoch; check_rejected(invalid_state);
        invalid_state = base; invalid_state.cancel_phase = 1; check_rejected(invalid_state);
    }
    accessible[0] = !accessible[0];
    const uint64_t previous_identity = layout.identity;
    CHECK(ggml_cuda_moe_hybrid_rows_measure(query, layout, sources) && layout.identity != previous_identity);
    CHECK(layout.staging_tile_bytes != 0);
    check_layout();
    fprintf(stderr, "test-moe-cache: hybrid producers rows=%u routes=%u slots=%u pageable=%d mapped/mirrored/capture/fresh-epoch/scatter/cancel/import-rejection OK\n",
        geometry.row_capacity, geometry.route_capacity, query.slot_capacity, pageable);
}

static void check_finalized_routed_metadata(const llm_graph_moe_region & region,
        const ggml_backend_moe_source_owner_v1 & owner) {
    std::vector<std::unique_ptr<llm_graph_moe_hybrid_prepared>> metadata;
    CHECK(region.prepare_routed_metadata(owner, 2, metadata) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    const size_t count = std::count_if(region.body_operations.begin(), region.body_operations.end(),
        [](const auto * tensor) { return tensor->op == GGML_OP_MUL_MAT_ID; });
    CHECK(metadata.size() == count && count > 0);
    auto * reg = ggml_backend_cpu_reg();
    const auto get_api = reinterpret_cast<ggml_backend_moe_cpu_region_service_v1_t>(
        ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_CPU_REGION_SERVICE_V1_PROC_NAME));
    const auto execute = reinterpret_cast<ggml_backend_moe_cpu_routed_execute_v1_t>(
        ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_CPU_ROUTED_EXECUTE_V1_PROC_NAME));
    CHECK(get_api && execute);
    const auto * api = get_api();
    ggml_backend_moe_cpu_service_config_v1 config = {};
    config.struct_size = sizeof(config); config.abi_version = 1; config.source_owner = &owner;
    config.n_threads = 2; config.n_lanes = 1; config.max_regions = count;
    config.flags = GGML_BACKEND_MOE_CPU_SERVICE_FLAG_V1_ALLOW_UNKNOWN_THREAD_STACK_BYTES |
        GGML_BACKEND_MOE_CPU_SERVICE_FLAG_V1_ALLOW_UNPROVEN_RUNTIME_ALLOCATIONS;
    ggml_backend_moe_cpu_service_v1_t service = nullptr;
    CHECK(api->create(&config, &service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    uint64_t epoch = 0;
    uint32_t executions = 0;
    for (const auto & prepared_metadata : metadata) {
        const auto & descriptor = prepared_metadata->descriptor();
        const auto & query = *descriptor.query;
        CHECK(query.flags == GGML_BACKEND_MOE_CPU_REGION_FLAG_V1_ROUTED_OPERATION && query.n_body_nodes == 1);
        CHECK(descriptor.first_node == descriptor.last_node && descriptor.output->op == GGML_OP_MUL_MAT_ID);
        CHECK(descriptor.activation == descriptor.output->src[1] && descriptor.ids == descriptor.output->src[2]);
        CHECK(!memcmp(query.body_nodes[0]->op_params, descriptor.output->op_params, sizeof(ggml_tensor::op_params)));
        auto * node = const_cast<ggml_tensor *>(query.body_nodes[0]);
        const auto saved_parameter = node->op_params[0];
        node->op_params[0] ^= 1;
        CHECK(ggml_backend_moe_hybrid_validate_buckets_v1(&descriptor) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
        node->op_params[0] = saved_parameter;
        auto * source = const_cast<ggml_backend_moe_cpu_region_source_v1 *>(query.sources);
        ++source->generation;
        CHECK(ggml_backend_moe_hybrid_validate_buckets_v1(&descriptor) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
        --source->generation;
        CHECK(ggml_backend_moe_hybrid_validate_buckets_v1(&descriptor) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        ggml_backend_moe_cpu_prepared_requirements_v1 requirements = {};
        requirements.struct_size = sizeof(requirements); requirements.abi_version = 1;
        ggml_backend_moe_cpu_prepared_region_v1_t prepared = 0;
        const int32_t prepare_status = api->prepare(service, &query, &requirements, &prepared);
        if (prepare_status) {
            fprintf(stderr, "test-moe-cache: routed prepare status=%d rows=%u routes=%u ids_pitch=%zu activation_period=%lld\n",
                prepare_status, query.bucket_rows, query.routes_per_row, query.ids->nb[1], (long long) query.activation->ne[1]);
        }
        CHECK(prepare_status == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        ggml_moe_source_expert operation;
        CHECK(ggml_moe_source_expert_prepare(descriptor, &prepared, operation));
        CHECK(operation.routed_operation && operation.banks.size() == 1 && operation.body->nodes().size() == 1);
        CHECK(operation.cpu_region == prepared && operation.input_width == descriptor.activation->ne[0] &&
            operation.output_width == descriptor.output->ne[0]);
        const auto rows = descriptor.geometry.row_capacity, routes = descriptor.geometry.routes_per_row;
        const auto capacity = descriptor.geometry.route_capacity;
        std::vector<uint8_t> activation(ggml_nbytes(descriptor.activation)), ids(ggml_nbytes(descriptor.ids));
        ggml_backend_tensor_get(descriptor.activation, activation.data(), 0, activation.size());
        ggml_backend_tensor_get(descriptor.ids, ids.data(), 0, ids.size());
        const auto expected = evaluate_body({node}, {query.activation, query.ids}, {node}, {query.sources[0]},
            {activation.data(), ids.data()});
        std::vector<int32_t> experts(capacity);
        std::vector<uint32_t> source_rows(capacity), scatter(capacity);
        for (uint32_t route = 0; route < capacity; ++route) {
            memcpy(&experts[route], ids.data() + (route / routes) * descriptor.ids->nb[1] +
                (route % routes) * descriptor.ids->nb[0], sizeof(int32_t));
            source_rows[route] = route / routes; scatter[route] = route;
        }
        ggml_backend_moe_cpu_region_binding_v1 binding = {sizeof(binding), rows, capacity,
            experts.data(), source_rows.data(), scatter.data()};
        ggml_backend_moe_cpu_dynamic_input_v1 inputs[] = {{activation.data(), activation.size(), descriptor.activation->nb[2]}, {}};
        std::vector<float> actual(expected.size());
        ggml_backend_moe_cpu_output_v1 output = {actual.data(), actual.size() * sizeof(float), descriptor.output->nb[1]};
        ggml_backend_moe_cpu_execute_v1 work = {};
        work.struct_size = sizeof(work); work.graph_uid = query.graph_uid; work.graph_generation = query.graph_generation;
        work.source_generation = query.source_generation; work.binding = &binding;
        work.dynamic_inputs = inputs; work.n_dynamic_inputs = 2; work.outputs = &output; work.n_outputs = 1;
        std::vector<uint8_t> ownership(capacity);
        for (uint32_t pass = 0; pass < 4; ++pass) {
            std::fill(actual.begin(), actual.end(), -123.0f);
            for (uint32_t route = 0; route < capacity; ++route) {
                ownership[route] = pass == 0 || (pass > 1 && route % 2 == pass % 2);
            }
            work.epoch = ++epoch;
            ggml_backend_moe_cpu_execute_result_v1 result = {}; result.struct_size = sizeof(result);
            CHECK(execute(service, prepared, &work, ownership.data(), capacity, &result) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
            CHECK(result.published_routes == uint32_t(std::count(ownership.begin(), ownership.end(), uint8_t(1))));
            for (uint32_t route = 0; route < capacity; ++route) {
                for (int64_t feature = 0; feature < descriptor.output->ne[0]; ++feature) {
                    const auto index = route * descriptor.output->ne[0] + feature;
                    CHECK(actual[index] == (ownership[route] ? expected[index] : -123.0f));
                }
            }
            ++executions;
        }
        CHECK(api->destroy_region(service, &prepared) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    }
    CHECK(api->close(service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    CHECK(api->drain(service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    CHECK(api->destroy(&service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    fprintf(stderr, "test-moe-cache: finalized routed metadata %zu original projections, %u exact CPU publications, independent layouts/precision/source witnesses OK\n", count, executions);
}

static void test_projection_discovery() {
    for (bool hash_route : {false, true}) {
        for (bool terminal : {false, true}) {
            ggml_backend_ptr cpu(ggml_backend_cpu_init());
            ggml_backend_t backends[] = {cpu.get()};
            ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, 1, 128, false, true));
            ggml_context_ptr weights(ggml_init({4 * ggml_tensor_overhead(), nullptr, true}));
            auto * up = ggml_new_tensor_3d(weights.get(), GGML_TYPE_F32, 32, 48, 3);
            auto * down = ggml_new_tensor_3d(weights.get(), GGML_TYPE_F32, 48, 32, 3);
            auto * adapter = ggml_new_tensor_3d(weights.get(), GGML_TYPE_F32, 32, 48, 3);
            auto * table = ggml_new_tensor_2d(weights.get(), GGML_TYPE_I32, 2, 3);
            ggml_set_name(up, "cached_projection_up");
            ggml_set_name(down, "cached_projection_down");
            ggml_backend_buffer_ptr weight_buffer(ggml_backend_alloc_ctx_tensors(weights.get(), cpu.get()));
            CHECK(weight_buffer);
            ggml_backend_buffer_set_usage(weight_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            std::unique_ptr<llama_model> source_model(llama_model_create(LLM_ARCH_QWEN3MOE, llama_model_default_params()));
            for (auto * weight : {up, down, adapter}) {
                std::vector<float> values(ggml_nelements(weight));
                for (size_t i = 0; i < values.size(); ++i) { values[i] = float(int(i % 19) - 9) * 0.003f; }
                ggml_backend_tensor_set(weight, values.data(), 0, values.size() * sizeof(float));
                CHECK(source_model->record_moe_readable_source(weight, weight->data, ggml_nbytes(weight)));
            }
            ggml_backend_moe_source_owner_v1 owner = {};
            CHECK(source_model->moe_source_owner_v1(&owner));
            llm_graph_result result{128};
            auto * ctx = result.get_ctx();
            auto * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 1, 2);
            auto * route_input = hash_route ? ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2) :
                ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 2);
            auto * ids = hash_route ? ggml_get_rows(ctx, table, route_input) : route_input;
            ggml_set_input(input); ggml_set_input(route_input); ggml_set_output(ids);
            auto * first = ggml_mul_mat_id(ctx, up, input, ids);
            auto * ordinary = ggml_mul_mat_id(ctx, adapter, input, ids);
            auto * view = ggml_view_tensor(ctx, first);
            auto * activation = ggml_silu(ctx, ggml_add(ctx, ggml_add(ctx, first, ordinary), ggml_scale(ctx, view, 0.25f)));
            auto * second = ggml_mul_mat_id(ctx, down, activation, ids);
            auto * output = terminal ? second : ggml_rms_norm(ctx, second, 1e-6f);
            ggml_set_output(first); ggml_set_output(activation); ggml_set_output(output);
            ggml_build_forward_expand(result.get_gf(), output);
            CHECK(result.get_moe_regions().empty());
            std::vector<llama_moe_source_group> sources;
            for (auto * tensor : {up, down}) {
                llama_moe_source_group source = {};
                source.layout = GGML_BACKEND_MOE_CANDIDATE_LAYOUT_ROUTED_MATRIX;
                source.domain = GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY;
                source.layer = sources.size() + 4;
                source.banks.push_back({tensor, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_ROUTED_WEIGHT,
                    GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE});
                sources.push_back(std::move(source));
            }
            const auto cached = +[](const ggml_tensor * tensor) {
                return tensor && strncmp(tensor->name, "cached_projection_", 18) == 0;
            };
            CHECK(result.discover_moe_regions(sources, cached));
            auto & regions = result.get_moe_regions();
            CHECK(regions.size() == 2 && regions[0].layer == 4 && regions[1].layer == 5);
            CHECK(regions[0].body_operations == std::vector<ggml_tensor *>{first});
            CHECK(regions[1].body_operations == std::vector<ggml_tensor *>{second});
            CHECK(regions[0].live_outs.size() == 1 && regions[0].live_outs[0].consumers.size() == 2);
            CHECK(regions[1].tail_resume == (terminal ? nullptr : output));
            const auto saved_first = regions[0].first_body;
            sources.push_back(sources.front());
            CHECK(!result.discover_moe_regions(sources, cached));
            CHECK(regions.size() == 2 && regions[0].first_body == saved_first);
            sources.pop_back();
            for (auto & region : regions) { CHECK(region.place(sched.get(), cpu.get())); }
            CHECK(ggml_backend_sched_alloc_graph(sched.get(), result.get_gf()));
            for (auto & region : regions) {
                CHECK(region.finalize_metadata(sched.get(), result.get_gf(), 17, 23) == GGML_BACKEND_SCHED_REGION_STATUS_V1_OK);
            }
            auto consumers = regions[0].live_outs[0].consumers;
            regions[0].live_outs[0].consumers.pop_back();
            CHECK(regions[0].finalize_metadata(sched.get(), result.get_gf(), 17, 23) == GGML_BACKEND_SCHED_REGION_STATUS_V1_INCOMPLETE_CUT);
            regions[0].live_outs[0].consumers = consumers;
            auto * tail = regions[0].tail_resume;
            regions[0].tail_resume = nullptr;
            CHECK(regions[0].finalize_metadata(sched.get(), result.get_gf(), 17, 23) == GGML_BACKEND_SCHED_REGION_STATUS_V1_INCOMPLETE_CUT);
            regions[0].tail_resume = tail;
            if (terminal) {
                CHECK(regions[1].finalized_metadata->tail_node_index == uint32_t(ggml_graph_n_nodes(result.get_gf())));
                CHECK(regions[1].live_outs[0].consumers.empty());
                second->flags &= ~GGML_TENSOR_FLAG_OUTPUT;
                CHECK(regions[1].finalize_metadata(sched.get(), result.get_gf(), 17, 23) == GGML_BACKEND_SCHED_REGION_STATUS_V1_INVALID_ARGUMENT);
                ggml_set_output(second);
            }
            std::vector<float> values(64, 0.125f);
            ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
            const int32_t routes[] = {0, 1, 2, 0};
            const int32_t tokens[] = {0, 1};
            const int32_t lookup[] = {0, 1, 2, 0, 1, 1};
            ggml_backend_tensor_set(table, lookup, 0, sizeof(lookup));
            ggml_backend_tensor_set(route_input, hash_route ? tokens : routes, 0, ggml_nbytes(route_input));
            CHECK(ggml_backend_sched_graph_compute(sched.get(), result.get_gf()) == GGML_STATUS_SUCCESS);
            const auto get_service = reinterpret_cast<ggml_backend_moe_cpu_region_service_v1_t>(
                ggml_backend_reg_get_proc_address(ggml_backend_cpu_reg(), GGML_BACKEND_MOE_CPU_REGION_SERVICE_V1_PROC_NAME));
            CHECK(get_service);
            const auto * service_api = get_service();
            ggml_backend_moe_cpu_service_config_v1 service_config = {};
            service_config.struct_size = sizeof(service_config); service_config.abi_version = 1;
            service_config.source_owner = &owner; service_config.n_threads = 2; service_config.n_lanes = 1;
            service_config.max_regions = 2;
            service_config.flags = GGML_BACKEND_MOE_CPU_SERVICE_FLAG_V1_ALLOW_UNKNOWN_THREAD_STACK_BYTES |
                GGML_BACKEND_MOE_CPU_SERVICE_FLAG_V1_ALLOW_UNPROVEN_RUNTIME_ALLOCATIONS;
            ggml_backend_moe_cpu_service_v1_t service = nullptr;
            CHECK(service_api->create(&service_config, &service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
            ggml_backend_moe_cpu_prepared_region_v1_t handles[2] = {};
            std::vector<ggml_moe_source_expert> experts(2);
            std::vector<const ggml_moe_source_expert *> pointers;
            for (size_t i = 0; i < regions.size(); ++i) {
                auto & region = regions[i];
                CHECK(region.finalize_metadata(sched.get(), result.get_gf(), 17, 23) == GGML_BACKEND_SCHED_REGION_STATUS_V1_OK);
                check_finalized_routed_metadata(region, owner);
                std::vector<std::unique_ptr<llm_graph_moe_hybrid_prepared>> metadata;
                CHECK(region.prepare_routed_metadata(owner, 2, metadata) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
                CHECK(metadata.size() == 1);
                ggml_backend_moe_cpu_prepared_requirements_v1 requirements = {};
                requirements.struct_size = sizeof(requirements); requirements.abi_version = 1;
                CHECK(service_api->prepare(service, metadata[0]->descriptor().query, &requirements, &handles[i]) ==
                    GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
                CHECK(ggml_moe_source_expert_prepare(metadata[0]->descriptor(), &handles[i], experts[i]));
                pointers.push_back(&experts[i]);
            }
            auto graph = ggml_graph_view(result.get_gf(), 0, ggml_graph_n_nodes(result.get_gf()));
            graph.uid = regions[0].finalized_metadata->split_graph_uid;
            graph.execution_certificate = layer_certificate();
            graph.execution_certificate.owner_generation = 17;
            graph.execution_certificate.n_rows = 2;
            graph.execution_certificate.source_graph_uid = regions[0].finalized_metadata->source_graph_uid;
            graph.execution_certificate.split_graph_uid = graph.uid;
            ggml_moe_source_program program;
            CHECK(program.prepare(&graph, pointers, ggml_backend_get_default_buffer_type(cpu.get())));
            CHECK(program.layers().size() == 2 && (program.epilogue().empty() == terminal));
            CHECK(program.find(ordinary) && program.find(view) && program.find(activation));
            CHECK(std::find(program.public_outputs().begin(), program.public_outputs().end(), output) != program.public_outputs().end());
            for (auto & handle : handles) {
                CHECK(service_api->destroy_region(service, &handle) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
            }
            CHECK(service_api->close(service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
            CHECK(service_api->drain(service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
            CHECK(service_api->destroy(&service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
            fprintf(stderr, "test-moe-cache: actual projection discovery hash=%d terminal=%d sources=2 adapter/fanout/public-output/complete-cut/ambiguous-source/8-exact-CPU-publications OK\n", hash_route, terminal);
        }
    }
}

void test_hybrid_row_metadata(int device, uint32_t rows, bool pageable) {
    layer_fixture fixture(0, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_TYPE_Q4_0, pageable,
        {device, device}, all_cached_layers, rows * n_used);
    fixture.build_graph(false, nullptr, nullptr, false, rows);
    std::unique_ptr<llama_model> source_owner(llama_model_create(LLM_ARCH_QWEN3MOE, llama_model_default_params()));
    for (const auto & source : fixture.tensors) {
        CHECK(source_owner->record_moe_readable_source(source.tensor, source.tensor->data, ggml_nbytes(source.tensor)));
    }
    ggml_backend_moe_source_owner_v1 owner = {};
    CHECK(source_owner->moe_source_owner_v1(&owner));
    ggml_backend_ptr gpu(ggml_backend_cuda_init(device));
    ggml_backend_ptr cpu(ggml_backend_cpu_init());
    ggml_backend_t backends[] = {gpu.get(), cpu.get()};
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, 2, 256, false, true));
    CHECK(ggml_backend_sched_set_resizable(sched.get(), nullptr));
    const auto snapshot = fixture.manifest();
    CHECK(ggml_backend_cuda_moe_candidate_replace_v2(gpu.get(), &snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    for (auto & region : fixture.result.get_moe_regions()) {
        for (auto * tensor : {region.body_output, region.dynamic_inputs[0], region.route}) {
            while (tensor->view_src != nullptr) { tensor = tensor->view_src; }
            if (tensor->op == GGML_OP_NONE) { ggml_set_input(tensor); }
            ggml_set_output(tensor);
        }
        CHECK(region.place(sched.get(), gpu.get()));
    }
    CHECK(ggml_backend_sched_alloc_graph(sched.get(), fixture.result.get_gf()));
    std::vector<float> activation(size_t(rows) * n_dim);
    for (size_t i = 0; i < activation.size(); ++i) {
        activation[i] = 0.01f * (1 + (i * 7 + i / n_dim * 3) % 17);
    }
    ggml_backend_tensor_set(fixture.input, activation.data(), 0, activation.size() * sizeof(float));
    for (auto * logits : fixture.logits) {
        std::vector<float> scores(size_t(rows) * n_experts, -10.0f);
        for (uint32_t row = 0; row < rows; ++row) {
            scores[row * n_experts] = 10.0f;
            scores[row * n_experts + 1 + row % (n_experts - 1)] = 5.0f;
        }
        ggml_backend_tensor_set(logits, scores.data(), 0, scores.size() * sizeof(float));
    }
    auto certificate = layer_certificate();
    certificate.flags = GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED;
    certificate.row_semantics = GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SPECULATIVE;
    certificate.n_rows = rows;
    CHECK(ggml_backend_sched_graph_compute_ext(sched.get(), fixture.result.get_gf(), &certificate) == GGML_STATUS_SUCCESS);
    uint64_t generation = 0, shrink = 0;
    ggml_backend_sched_get_buffer_state(sched.get(), &generation, &shrink);
    auto & region = fixture.result.get_moe_regions()[0];
    CHECK(region.finalize_metadata(sched.get(), fixture.result.get_gf(), 1, generation) == GGML_BACKEND_SCHED_REGION_STATUS_V1_OK);
    std::unique_ptr<llm_graph_moe_hybrid_prepared> metadata;
    CHECK(region.prepare_hybrid_metadata(owner, 2, metadata) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    const auto & descriptor = metadata->descriptor();
    check_original_body(region, descriptor);
    check_finalized_routed_metadata(region, owner);
    CHECK(descriptor.geometry.row_capacity == rows && descriptor.geometry.routes_per_row == n_used &&
        descriptor.geometry.route_capacity == rows * n_used && descriptor.n_cpu_queries == n_used);
    test_hybrid_row_producers(descriptor, certificate, pageable);
    const auto expected = active_grouped_tensor_values(descriptor.output);
    CHECK(region.dynamic_inputs[0]->view_src == fixture.input);
    std::vector<uint8_t> ids_data(ggml_nbytes(descriptor.ids));
    ggml_backend_tensor_get(descriptor.ids, ids_data.data(), 0, ids_data.size());
    std::vector<int32_t> weights;
    std::vector<ggml_backend_moe_hybrid_route_v1> records;
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t rank = 0; rank < n_used; ++rank) {
            int32_t expert;
            memcpy(&expert, ids_data.data() + row * descriptor.ids->nb[1] + rank * sizeof(expert), sizeof(expert));
            auto found = std::find(weights.begin(), weights.end(), expert);
            const uint32_t index = found - weights.begin();
            if (found == weights.end()) { weights.push_back(expert); }
            records.push_back({row, rank, index, row * n_used + rank});
        }
    }
    CHECK(weights.size() < records.size());
    ggml_backend_moe_hybrid_binding_v1 routing = {};
    routing.struct_size = sizeof(routing);
    routing.active_rows = rows;
    routing.n_weights = weights.size();
    routing.n_routes = records.size();
    routing.epoch = 1;
    routing.source_graph_uid = descriptor.source_graph_uid;
    routing.split_graph_uid = descriptor.split_graph_uid;
    routing.owner_generation = descriptor.owner_generation;
    routing.allocator_generation = descriptor.allocator_generation;
    routing.source_generation = descriptor.query->source_generation;
    routing.weight_experts = weights.data();
    routing.routes = records.data();
    auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(cpu.get()));
    const auto proc = reinterpret_cast<ggml_backend_moe_cpu_region_service_v1_t>(
        ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_CPU_REGION_SERVICE_V1_PROC_NAME));
    CHECK(proc != nullptr);
    const auto * api = proc();
    ggml_backend_moe_cpu_service_config_v1 config = {};
    config.struct_size = sizeof(config);
    config.abi_version = 1;
    config.source_owner = &owner;
    config.n_threads = 2;
    config.n_lanes = 1;
    config.max_regions = descriptor.n_cpu_queries;
    config.flags = GGML_BACKEND_MOE_CPU_SERVICE_FLAG_V1_ALLOW_UNKNOWN_THREAD_STACK_BYTES |
        GGML_BACKEND_MOE_CPU_SERVICE_FLAG_V1_ALLOW_UNPROVEN_RUNTIME_ALLOCATIONS;
    ggml_backend_moe_cpu_service_v1_t service = nullptr;
    CHECK(api->create(&config, &service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    std::vector<ggml_backend_moe_cpu_prepared_region_v1_t> prepared(descriptor.n_cpu_queries);
    for (uint32_t count = 1; count <= descriptor.n_cpu_queries; ++count) {
        ggml_backend_moe_cpu_prepared_requirements_v1 requirements = {};
        requirements.struct_size = sizeof(requirements);
        requirements.abi_version = 1;
        CHECK(api->prepare(service, descriptor.cpu_queries[count - 1], &requirements, &prepared[count - 1]) ==
            GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    }
    std::vector<uint8_t> marks(size_t(descriptor.geometry.expert_count) + descriptor.geometry.route_capacity);
    std::vector<int32_t> bound_ids(n_used);
    std::vector<uint32_t> bound_rows(n_used), bound_scatter(n_used), selected(n_used);
    std::vector<float> actual(expected.size(), -123.0f);
    std::vector<float> cpu_reference;
    ggml_backend_moe_cpu_region_binding_v1 binding = {};
    const auto bind = [&](uint32_t row, uint32_t count, size_t mark_bytes) {
        return ggml_backend_moe_hybrid_bind_cpu_row_v1(&descriptor, &routing, routing.epoch, row,
            selected.data(), count, selected.size(), bound_ids.data(), bound_rows.data(), bound_scatter.data(),
            &binding, marks.data(), mark_bytes);
    };
    const auto execute = [&](uint32_t row, uint32_t count) {
        for (uint32_t i = 0; i < count; ++i) { selected[i] = row * n_used + count - 1 - i; }
        CHECK(bind(row, count, marks.size()) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        for (uint32_t i = 0; i < count; ++i) {
            CHECK(bound_rows[i] == row && bound_scatter[i] == selected[i]);
        }
        ggml_backend_moe_cpu_dynamic_input_v1 inputs[2] = {{activation.data(), activation.size() * sizeof(float),
            descriptor.activation->nb[2]}, {}};
        ggml_backend_moe_cpu_output_v1 output = {actual.data(), actual.size() * sizeof(float), descriptor.output->nb[1]};
        const auto & query = *descriptor.cpu_queries[count - 1];
        ggml_backend_moe_cpu_execute_v1 work = {};
        work.struct_size = sizeof(work);
        work.epoch = routing.epoch;
        work.graph_uid = query.graph_uid;
        work.graph_generation = query.graph_generation;
        work.source_generation = query.source_generation;
        work.binding = &binding;
        work.dynamic_inputs = inputs;
        work.n_dynamic_inputs = 2;
        work.outputs = &output;
        work.n_outputs = 1;
        ggml_backend_moe_cpu_execute_result_v1 result = {};
        result.struct_size = sizeof(result);
        const auto status = api->execute(service, prepared[count - 1], &work, &result);
        if (status == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK) {
            CHECK(result.flags == GGML_BACKEND_MOE_CPU_EXECUTE_RESULT_FLAG_V1_PUBLISHED && result.published_routes == count);
        }
        return status;
    };
    for (uint32_t pass = 0; pass < 3; ++pass) {
        routing.epoch = pass + 1;
        routing.active_rows = pass == 2 ? rows - 1 : rows;
        routing.n_routes = routing.active_rows * n_used;
        if (pass != 0) {
            routing.n_weights = 1;
            for (auto & record : records) { record.weight_index = 0; }
        }
        std::fill(actual.begin(), actual.end(), -123.0f);
        for (uint32_t row = 0; row < routing.active_rows; ++row) {
            CHECK(execute(row, n_used) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        }
        double error = 0, norm = 0;
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t rank = 0; rank < n_used; ++rank) {
                for (int column = 0; column < n_dim; ++column) {
                    const size_t index = (row * n_used + rank) * n_dim + column;
                    if (row >= routing.active_rows) { CHECK(actual[index] == -123.0f); continue; }
                    const float reference = pass == 0 ? expected[index] : cpu_reference[row * n_used * n_dim + column];
                    CHECK(std::isfinite(actual[index]) && std::isfinite(reference));
                    if (pass != 0) { CHECK(actual[index] == reference); }
                    error += std::pow(double(actual[index]) - reference, 2);
                    norm += double(reference) * reference;
                }
            }
        }
        fprintf(stderr, "test-moe-cache: hybrid metadata rows=%u pageable=%d pass=%u oracle=%s NMSE=%g\n",
            rows, pageable, pass, pass == 0 ? "GPU" : "CPU-exact", error / std::max(norm, 1e-30));
        if (pass == 0) {
            CHECK(error / std::max(norm, 1e-30) <= 1e-3); // CPU/GPU Q4_0 sanity screen; route replay is exact below.
            cpu_reference = actual;
        }
    }
    CHECK(!std::equal(expected.begin(), expected.begin() + n_dim, expected.begin() + n_used * n_dim));
    selected[0] = 0;
    const auto rejected = [&] {
        std::fill(bound_ids.begin(), bound_ids.end(), -7);
        binding = {};
        CHECK(bind(0, 1, marks.size()) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_BINDING);
        CHECK(binding.n_routes == 0 && std::all_of(bound_ids.begin(), bound_ids.end(), [](int32_t id) { return id == -7; }));
    };
    for (auto * identity : {&routing.source_graph_uid, &routing.split_graph_uid, &routing.owner_generation,
            &routing.allocator_generation, &routing.source_generation}) {
        ++*identity; rejected(); --*identity;
    }
    routing.epoch = 0; rejected(); routing.epoch = 4;
    ++records[0].scatter_destination; rejected(); --records[0].scatter_destination;
    records[0].weight_index = routing.n_weights; rejected(); records[0].weight_index = 0;
    --routing.n_routes; rejected(); ++routing.n_routes;
    CHECK(bind(0, 1, marks.size() - 1) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_BINDING);
    CHECK(bind(routing.active_rows, 1, marks.size()) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_BINDING);
    selected[1] = selected[0];
    CHECK(bind(0, 2, marks.size()) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_BINDING);
    auto invalid = descriptor;
    --invalid.geometry.route_capacity;
    CHECK(ggml_backend_moe_hybrid_validate_buckets_v1(&invalid) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
    ggml_tensor huge_ids = *descriptor.ids;
    huge_ids.ne[1] = UINT32_MAX;
    ggml_backend_moe_hybrid_geometry_v1 geometry = {};
    CHECK(ggml_backend_moe_hybrid_get_geometry_v1(descriptor.activation, &huge_ids, descriptor.output,
        n_experts, &geometry) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
    const auto before_cancel = actual;
    CHECK(api->cancel(service, routing.epoch) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    CHECK(execute(0, 1) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CANCELED && actual == before_cancel);
    ++routing.epoch;
    CHECK(execute(0, 1) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    CHECK(api->close(service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    CHECK(api->drain(service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    for (auto & handle : prepared) { CHECK(api->destroy_region(service, &handle) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK); }
    CHECK(api->destroy(&service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    fprintf(stderr, "test-moe-cache: hybrid row metadata rows=%u routes=%u pageable=%d CPU/scatter/identity/cancel OK; speculative runtime disabled\n",
        rows, descriptor.geometry.route_capacity, pageable);
}

struct hybrid_layer_signature {
    uint32_t layout;
    ggml_type gate_up_type;
    ggml_type down_type;
    llm_ffn_op_type activation;
    int n_embd;
    int n_ff;
};

struct hybrid_test_policy {
    uint32_t quota = 1;
    uint32_t routes = 3;
    uint32_t hits = 1;
    bool duplicates = false;
    bool overlap = false;
    uint32_t block_phase = 0;
    uint32_t fail_phase = 0;
    bool quiesce = false;
    bool close_source = false;
    bool reset = false;
    bool gpu_pending = false;
    bool io_pending = false;
    bool admission = false;
    bool resident_batch = false;
    bool sparse_hits = false;
    bool auto_memory = false;
    bool no_host_alias = false;
    bool window = false;
    bool cpu_pipeline = true;
    bool combine_gpu = true;
    bool direct_gather = true;
    bool window_fallback = false;
    bool window_host_node = false;
    bool staged_input = false;
    bool boundary_overlap = false;
    uint32_t window_replays = 0;
    uint32_t packet_failure = 0;
    uint64_t device_bytes = 16 * 1024 * 1024;
};

struct hybrid_test_barrier {
    hybrid_test_policy policy;
    std::mutex mutex;
    std::condition_variable changed;
    bool cpu_waiting = false;
    bool gpu_enqueued = false;
    bool released = false;
    host_barrier pending;
    cudaStream_t stream = nullptr;
    cudaEvent_t gpu_produced = nullptr;
    bool gpu_probe_recorded = false;
    ggml_tensor * ids[2] = {};
    const std::array<std::vector<int32_t>, 2> * routes = nullptr;
    uint32_t next_layer = 0;

    static bool hook(void * opaque, uint32_t phase, uint64_t, void * stream) {
        auto & barrier = *static_cast<hybrid_test_barrier *>(opaque);
        std::unique_lock<std::mutex> lock(barrier.mutex);
        if (phase == GGML_BACKEND_MOE_HYBRID_TEST_WINDOW_BODY && barrier.policy.window_host_node) {
            return cudaLaunchHostFunc(static_cast<cudaStream_t>(stream), [](void *) {}, nullptr) == cudaSuccess;
        }
        if (phase == GGML_BACKEND_MOE_HYBRID_TEST_GPU_ENQUEUED && barrier.gpu_produced != nullptr && !barrier.gpu_probe_recorded) {
            CUDA_OK(cudaEventRecordWithFlags(barrier.gpu_produced, static_cast<cudaStream_t>(stream), cudaEventRecordExternal));
            barrier.gpu_probe_recorded = true;
        }
        if (phase == GGML_BACKEND_MOE_HYBRID_TEST_BEFORE_ROUTES && barrier.policy.duplicates) {
            const uint32_t layer = barrier.next_layer++ % 2;
            const auto & ids = (*barrier.routes)[layer];
            CUDA_OK(cudaMemcpyAsync(barrier.ids[layer]->data, ids.data(), ids.size() * sizeof(int32_t),
                cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream)));
        }
        const uint32_t block = barrier.policy.overlap ? uint32_t(GGML_BACKEND_MOE_HYBRID_TEST_CPU_ADMITTED) : barrier.policy.block_phase;
        if (phase == block) {
            barrier.cpu_waiting = true;
            barrier.changed.notify_all();
            CHECK(barrier.changed.wait_for(lock, std::chrono::seconds(10), [&] { return barrier.released; }));
        } else if (phase == GGML_BACKEND_MOE_HYBRID_TEST_TRANSFER_ENQUEUED && barrier.policy.io_pending) {
            CHECK(barrier.changed.wait_for(lock, std::chrono::seconds(10), [&] { return barrier.cpu_waiting; }));
            barrier.stream = static_cast<cudaStream_t>(stream);
            CUDA_OK(cudaLaunchHostFunc(barrier.stream, wait_on_host_barrier, &barrier.pending));
        } else if (phase == GGML_BACKEND_MOE_HYBRID_TEST_GPU_ENQUEUED) {
            if (barrier.policy.overlap || barrier.policy.gpu_pending) {
                CHECK(barrier.changed.wait_for(lock, std::chrono::seconds(10), [&] { return barrier.cpu_waiting; }));
                barrier.released = barrier.policy.overlap;
            }
            if (barrier.policy.gpu_pending) {
                barrier.stream = static_cast<cudaStream_t>(stream);
                CUDA_OK(cudaLaunchHostFunc(barrier.stream, wait_on_host_barrier, &barrier.pending));
            }
            barrier.gpu_enqueued = true;
            barrier.changed.notify_all();
        }
        return phase != barrier.policy.fail_phase;
    }
};

void test_hybrid_layers(int device, bool pageable, bool fail_before_publish, const hybrid_layer_signature & signature,
                        int32_t expected_prepare = GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK,
                        const hybrid_test_policy & policy = {}, uint64_t * required_device_bytes = nullptr) {
    const bool captured_window = policy.window && !policy.window_fallback;
    layer_fixture fixture(0, signature.layout, signature.gate_up_type,
                          pageable, {device, device}, all_cached_layers, n_slots, false,
                          LLM_ARCH_QWEN3MOE, signature.activation, policy.routes, signature.down_type, signature.n_embd, signature.n_ff);
    layer_fixture reference(0, signature.layout, signature.gate_up_type,
                            pageable, {device, device}, 0, n_slots, false,
                            LLM_ARCH_QWEN3MOE, signature.activation, policy.routes, signature.down_type, signature.n_embd, signature.n_ff);
    fixture.build_graph(true, nullptr, nullptr, policy.window);
    reference.build_graph(true, nullptr, nullptr, policy.window);
    std::unique_ptr<llama_model> source_owner(llama_model_create(LLM_ARCH_QWEN3MOE, llama_model_default_params()));
    for (const auto & source : fixture.tensors) {
        CHECK(source_owner->record_moe_readable_source(source.tensor, source.tensor->data, ggml_nbytes(source.tensor)));
    }
    ggml_backend_moe_source_owner_v1 owner = {};
    CHECK(source_owner->moe_source_owner_v1(&owner));
    auto pool_params = ggml_threadpool_params_default(2);
    std::unique_ptr<ggml_threadpool, decltype(&ggml_threadpool_free)> prefix_pool(
        ggml_threadpool_new(&pool_params), ggml_threadpool_free);
    CHECK(prefix_pool != nullptr);
    ggml_backend_ptr gpu(ggml_backend_cuda_init(device));
    const auto backend_reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(gpu.get()));
    auto get_stage_api = reinterpret_cast<ggml_staged_input_get_api_t>(
        ggml_backend_reg_get_proc_address(backend_reg, GGML_STAGED_INPUT_PROC));
    const auto * stage_api = get_stage_api == nullptr ? nullptr : get_stage_api();
    auto stage_pending = reinterpret_cast<ggml_backend_moe_hybrid_staged_pending_v1_t>(
        ggml_backend_reg_get_proc_address(backend_reg, GGML_BACKEND_MOE_HYBRID_STAGED_PENDING_V1_PROC_NAME));
    std::unique_ptr<void, void (*)(void *)> stage(nullptr, [](void *) {});
    if (policy.staged_input) {
        CHECK(stage_api != nullptr && stage_pending != nullptr);
        stage = {stage_api->create(gpu.get(), signature.n_embd * sizeof(float)), stage_api->destroy};
        CHECK(stage != nullptr);
        std::memset(stage_api->data(stage.get()), 0, signature.n_embd * sizeof(float));
        fixture.build_graph(true, stage_api, stage.get(), policy.window);
    }
    if (policy.boundary_overlap) {
        auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(gpu.get()));
        auto enable = reinterpret_cast<void (*)(ggml_backend_t, bool)>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_set_decode_boundary_overlap"));
        CHECK(enable != nullptr);
        enable(gpu.get(), true);
    }
    if (policy.window) {
        auto * ids_storage = fixture.ids[1];
        while (ids_storage->view_src != nullptr) {
            ids_storage = ids_storage->view_src;
        }
        // Keep the sentinel separate from earlier temporary tensors.
        ggml_set_input(ids_storage);
        ggml_set_output(ids_storage);
        auto * raw_storage = fixture.result.get_moe_regions()[0].body_output;
        while (raw_storage->view_src != nullptr) {
            raw_storage = raw_storage->view_src;
        }
        ggml_set_input(raw_storage);
        ggml_set_output(raw_storage);
    }
    ggml_backend_ptr cpu(ggml_backend_cpu_init());
    ggml_backend_cpu_set_n_threads(cpu.get(), 2);
    ggml_backend_cpu_set_threadpool(cpu.get(), prefix_pool.get());
    ggml_backend_t backends[] = {gpu.get(), cpu.get()};
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, 2, 256, false, true));
    ggml_backend_sched_ptr reference_sched(ggml_backend_sched_new(backends, nullptr, 2, 256, false, true));
    CHECK(ggml_backend_sched_set_resizable(sched.get(), nullptr));
    const auto snapshot = fixture.manifest();
    CHECK(ggml_backend_cuda_moe_candidate_replace_v2(gpu.get(), &snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    for (auto & region : fixture.result.get_moe_regions()) {
        ggml_set_output(region.body_output);
        CHECK(region.place(sched.get(), gpu.get()));
    }
    for (auto & region : reference.result.get_moe_regions()) {
        ggml_set_output(region.body_output);
        CHECK(region.place(reference_sched.get(), gpu.get()));
    }
    // Keep tail sentinels outside reused graph scratch.
    for (auto * target : {&fixture, &reference}) {
        for (auto * output : target->output) {
            auto * buft = ggml_backend_cuda_buffer_type(device);
            auto * buffer = ggml_backend_buft_alloc_buffer(buft, ggml_backend_buft_get_alloc_size(buft, output));
            CHECK(buffer != nullptr && ggml_backend_tensor_alloc(buffer, output, ggml_backend_buffer_get_base(buffer)) == GGML_STATUS_SUCCESS);
            target->ordinary_buffers.emplace_back(buffer);
        }
    }
    ggml_backend_sched_set_tensor_backend(sched.get(), fixture.prefix, cpu.get());
    ggml_backend_sched_set_tensor_backend(reference_sched.get(), reference.prefix, cpu.get());
    if (fixture.staged != nullptr) {
        ggml_backend_sched_set_tensor_backend(sched.get(), fixture.staged, gpu.get());
    }
    CHECK(ggml_backend_sched_alloc_graph(sched.get(), fixture.result.get_gf()));
    CHECK(ggml_backend_sched_alloc_graph(reference_sched.get(), reference.result.get_gf()));
    auto certificate = layer_certificate();
    const auto compute_fixture = [&](const ggml_graph_execution_certificate * cert) {
        if (stage != nullptr) {
            stage_api->publish(stage.get());
        }
        const auto status = ggml_backend_sched_graph_compute_ext(sched.get(), fixture.result.get_gf(), cert);
        if (status == GGML_STATUS_SUCCESS && policy.window) {
            const auto input = active_grouped_tensor_values(fixture.input);
            const auto * probe = ggml_get_tensor(fixture.result.get_ctx(), "hybrid_fusion_retained");
            CHECK(probe != nullptr);
            const auto retained = active_grouped_tensor_values(probe);
            CHECK(input.size() == retained.size());
            for (size_t i = 0; i < input.size(); ++i) {
                CHECK(retained[i] == input[i] + input[i]);
            }
        }
        if (status == GGML_STATUS_SUCCESS && stage != nullptr) {
            ggml_backend_sched_synchronize(sched.get());
            CHECK(!stage_pending(stage.get()));
            const auto values = active_grouped_tensor_values(fixture.staged);
            CHECK(std::all_of(values.begin(), values.end(), [](float value) { return value == 0; }));
        }
        return status;
    };
    for (int iteration : {0, 1}) {
        fixture.set_inputs(iteration);
        CHECK(compute_fixture(&certificate) == GGML_STATUS_SUCCESS);
        CHECK(compute_fixture(&certificate) == GGML_STATUS_SUCCESS);
    }
    auto * grouped = ggml_cuda_moe_grouped_context_for_test(gpu.get());
    std::array<std::vector<int32_t>, 2> routes;
    uint64_t expected_routes[3] = {}, expected_experts[3] = {}, expected_jobs = 0, transferred_bytes = 0;
    std::array<std::array<int32_t, n_experts>, 2> initial_slots;
    std::array<std::vector<int32_t>, 2> first_admissions;
    uint64_t expected_admissions = 0, expected_admission_bytes = 0, expected_no_slot = 0, expected_resident_batches = 0;
    uint64_t expected_transfer_batches = 0;
    for (int layer = 0; layer < 2; ++layer) {
        ggml_cuda_moe_candidate_group_key key;
        CHECK(grouped->find_down_group_key(fixture.down[layer], &key));
        ggml_cuda_moe_candidate_group_info info;
        CHECK(grouped->get_group(key, &info));
        const uint32_t n_weights = (fixture.gate_up[layer] != nullptr) + (fixture.gate[layer] != nullptr) +
            (fixture.up[layer] != nullptr) + (fixture.down[layer] != nullptr);
        CHECK(info.n_resource_banks == n_weights);
        CHECK(info.n_banks == n_weights + (fixture.down_bias[layer] != nullptr));
        for (int i = 0; i < n_slots && routes[layer].size() < policy.hits; ++i) {
            const int slot = policy.sparse_hits ? (i % 2 == 0 ? n_slots - 1 - i / 2 : i / 2) : n_slots - 1 - i;
            for (int expert = 0; expert < n_experts; ++expert) {
                if (ggml_cuda_moe_grouped_context_test_access::device_slot_for_expert(*grouped, key, expert) == slot) {
                    routes[layer].push_back(expert);
                }
            }
        }
        CHECK(routes[layer].size() == policy.hits);
        for (int expert = 0; expert < n_experts && routes[layer].size() < policy.routes; ++expert) {
            if (ggml_cuda_moe_grouped_context_test_access::device_slot_for_expert(*grouped, key, expert) < 0) {
                routes[layer].push_back(expert);
            }
        }
        CHECK(routes[layer].size() == policy.routes);
        if (policy.duplicates) {
            CHECK(policy.routes == 4);
            routes[layer][2] = routes[layer][1];
        }
        std::array<bool, n_experts> resident = {};
        for (int expert = 0; expert < n_experts; ++expert) {
            initial_slots[layer][expert] = ggml_cuda_moe_grouped_context_test_access::device_slot_for_expert(*grouped, key, expert);
            resident[expert] = initial_slots[layer][expert] >= 0;
        }
        size_t expert_bytes = 0;
        for (const auto * weight : {fixture.gate_up[layer], fixture.gate[layer], fixture.up[layer], fixture.down[layer]}) {
            if (weight != nullptr) {
                expert_bytes += weight->nb[2];
            }
        }
        for (uint32_t iteration = 0; iteration < 2; ++iteration) {
            std::unordered_map<int32_t, uint32_t> assignments;
            uint32_t transfers = 0, cpu_count = 0, reservations = 0, residents = 0;
            uint32_t available_slots = n_slots;
            std::array<bool, n_experts> selected = {};
            for (int32_t expert : routes[layer]) {
                if (!selected[expert] && resident[expert]) {
                    --available_slots;
                }
                selected[expert] = true;
            }
            for (int32_t expert : routes[layer]) {
                auto found = assignments.find(expert);
                if (found == assignments.end()) {
                    const uint32_t route_class = resident[expert] ? 0 : transfers < policy.quota ? 1 : 2;
                    found = assignments.emplace(expert, route_class).first;
                    ++expected_experts[route_class];
                    residents += route_class == 0;
                    transfers += route_class == 1;
                    cpu_count += route_class == 2;
                    if (policy.admission && route_class == 1 && reservations < available_slots) {
                        ++reservations;
                        resident[expert] = true;
                        ++expected_admissions;
                        expected_admission_bytes += expert_bytes + (fixture.down_bias[layer] != nullptr ? fixture.down_bias[layer]->nb[1] : 0);
                        if (iteration == 0) {
                            first_admissions[layer].push_back(expert);
                        }
                    }
                }
                ++expected_routes[found->second];
            }
            expected_jobs += cpu_count != 0;
            expected_transfer_batches += transfers != 0;
            expected_resident_batches += policy.resident_batch || residents != 0;
            expected_no_slot += policy.admission ? transfers - reservations : 0;
            transferred_bytes += transfers * expert_bytes;
        }
    }

    const auto set_inputs = [&](layer_fixture & target) {
        target.set_inputs(2);
        for (int layer = 0; layer < 2; ++layer) {
            float scores[n_experts] = {};
            for (size_t rank = 0; rank < routes[layer].size(); ++rank) {
                scores[routes[layer][rank]] = float(policy.routes - rank);
            }
            ggml_backend_tensor_set(target.logits[layer], scores, 0, sizeof(scores));
        }
    };
    const auto residency = [&] {
        std::vector<uint8_t> bytes;
        for (int layer = 0; layer < 2; ++layer) {
            ggml_cuda_moe_candidate_group_key key;
            CHECK(grouped->find_down_group_key(fixture.down[layer], &key));
            for (uint32_t expert = 0; expert < n_experts; ++expert) {
                const int32_t slot = ggml_cuda_moe_grouped_context_test_access::device_slot_for_expert(*grouped, key, expert);
                const auto * data = reinterpret_cast<const uint8_t *>(&slot);
                bytes.insert(bytes.end(), data, data + sizeof(slot));
            }
            for (const auto * weight : {fixture.gate_up[layer], fixture.gate[layer], fixture.up[layer], fixture.down[layer]}) {
                if (weight == nullptr) {
                    continue;
                }
                const void * data = ggml_cuda_moe_grouped_context_test_access::device_bank_data(*grouped, key, weight);
                CHECK(data != nullptr);
                const size_t offset = bytes.size();
                bytes.resize(offset + n_slots * weight->nb[2]);
                CUDA_OK(cudaMemcpy(bytes.data() + offset, data, bytes.size() - offset, cudaMemcpyDeviceToHost));
            }
        }
        return bytes;
    };
    const auto original_residency = residency();
    const auto verify_payloads = [&] {
        uint64_t populated = 0;
        for (int layer = 0; layer < 2; ++layer) {
            ggml_cuda_moe_candidate_group_key key;
            CHECK(grouped->find_down_group_key(fixture.down[layer], &key));
            std::array<bool, n_slots> occupied = {};
            for (int expert = 0; expert < n_experts; ++expert) {
                const int32_t slot = ggml_cuda_moe_grouped_context_test_access::device_slot_for_expert(*grouped, key, expert);
                if (std::find(routes[layer].begin(), routes[layer].end(), expert) != routes[layer].end() && initial_slots[layer][expert] >= 0) {
                    CHECK(slot == initial_slots[layer][expert]);
                }
                if (slot < 0) {
                    continue;
                }
                CHECK(slot < n_slots && !occupied[slot]);
                occupied[slot] = true;
                ++populated;
                for (const auto * tensor : {fixture.gate_up[layer], fixture.gate[layer], fixture.up[layer], fixture.down[layer], fixture.down_bias[layer]}) {
                    if (tensor == nullptr) {
                        continue;
                    }
                    const bool auxiliary = tensor == fixture.down_bias[layer];
                    const size_t stride = auxiliary ? tensor->nb[1] : tensor->nb[2];
                    const void * data = auxiliary ? static_cast<const void *>(ggml_cuda_moe_grouped_context_test_access::device_auxiliary_data(*grouped, key, tensor)) :
                        ggml_cuda_moe_grouped_context_test_access::device_bank_data(*grouped, key, tensor);
                    CHECK(data != nullptr);
                    std::vector<uint8_t> actual(stride), expected(stride);
                    CUDA_OK(cudaMemcpy(actual.data(), static_cast<const uint8_t *>(data) + size_t(slot) * stride, stride, cudaMemcpyDeviceToHost));
                    ggml_backend_tensor_get(tensor, expected.data(), size_t(expert) * stride, stride);
                    CHECK(actual == expected);
                }
            }
        }
        return populated;
    };
    const uint64_t seeded_slots = verify_payloads();
    const auto seeded = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*grouped);
    CHECK(seeded.occupancy_unavailable == 0 && seeded.populated_slots == seeded_slots);
    uint64_t generation = 0;
    uint64_t shrink_generation = 0;
    ggml_backend_sched_get_buffer_state(sched.get(), &generation, &shrink_generation);
    for (auto & region : fixture.result.get_moe_regions()) {
        CHECK(region.finalize_metadata(sched.get(), fixture.result.get_gf(), 1, generation) ==
              GGML_BACKEND_SCHED_REGION_STATUS_V1_OK);
        CHECK(region.finalized_metadata->split_index != 0);
    }
    ggml_backend_moe_hybrid_config_v1 config = {};
    config.struct_size = sizeof(config);
    config.backend = gpu.get();
    config.source_owner = &owner;
    config.cpu_module_acquire = ggml_backend_moe_cpu_module_acquire_v1;
    config.module_retain = ggml_backend_moe_module_retain_v1;
    config.module_release = ggml_backend_moe_module_release_v1;
    config.n_threads = 2;
    config.max_regions = 2;
    config.max_prepared_regions = config.max_regions * (policy.routes + 1);
    config.gpu_miss_quota = policy.quota;
    config.admission_quota = policy.admission ? policy.quota : 0;
    config.demand_admission = policy.admission;
    config.resident_batch = policy.resident_batch;
    config.cpu_flags = GGML_BACKEND_MOE_CPU_SERVICE_FLAG_V1_ALLOW_UNKNOWN_THREAD_STACK_BYTES |
                       GGML_BACKEND_MOE_CPU_SERVICE_FLAG_V1_ALLOW_UNPROVEN_RUNTIME_ALLOCATIONS;
    config.cpu_bytes = policy.auto_memory ? 0 : 16 * 1024 * 1024;
    config.device_bytes = policy.auto_memory ? 0 : policy.device_bytes;
    config.pinned_bytes = policy.auto_memory ? 0 : 1024 * 1024 + 64 * 1024;
    const char * variables[] = {"GGML_MOE_HYBRID_FAIL_BEFORE_PUBLISH", "GGML_MOE_HYBRID_TEST_NO_HOST_ALIAS",
        "GGML_MOE_HYBRID_WINDOW", "GGML_MOE_HYBRID_WINDOW_PIPELINE_CPU", "GGML_MOE_HYBRID_WINDOW_COMBINE_GPU",
        "GGML_MOE_HYBRID_WINDOW_DIRECT_GATHER"};
    const bool enabled[] = {fail_before_publish, policy.no_host_alias, policy.window, policy.cpu_pipeline, policy.combine_gpu, policy.direct_gather};
    std::string saved[6];
    bool present[6];
    const auto set_variable = [](const char * name, const char * value) {
#ifdef _WIN32
        CHECK(_putenv_s(name, value == nullptr ? "" : value) == 0);
#else
        CHECK((value == nullptr ? unsetenv(name) : setenv(name, value, 1)) == 0);
#endif
    };
    for (size_t i = 0; i < 6; ++i) {
        const char * value = getenv(variables[i]);
        present[i] = value != nullptr;
        saved[i] = value == nullptr ? "" : value;
        set_variable(variables[i], enabled[i] ? "1" : i >= 3 ? "0" : nullptr);
    }
    CHECK(ggml_backend_sched_moe_hybrid_configure_v1(sched.get(), &config) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    for (size_t i = 0; i < 6; ++i) {
        set_variable(variables[i], present[i] ? saved[i].c_str() : nullptr);
    }
    auto changed_config = config;
    ++changed_config.gpu_miss_quota;
    CHECK(ggml_backend_sched_moe_hybrid_configure_v1(sched.get(), &changed_config) ==
          GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
    changed_config = config;
    changed_config.admission_quota = config.admission_quota == 0 ? 1 : config.admission_quota - 1;
    CHECK(ggml_backend_sched_moe_hybrid_configure_v1(sched.get(), &changed_config) ==
          GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
    changed_config = config;
    changed_config.admission_quota = config.gpu_miss_quota + 1;
    CHECK(ggml_backend_sched_moe_hybrid_configure_v1(sched.get(), &changed_config) ==
          GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
    changed_config = config;
    changed_config.demand_admission ^= 1;
    CHECK(ggml_backend_sched_moe_hybrid_configure_v1(sched.get(), &changed_config) ==
          GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
    changed_config.demand_admission = 2;
    CHECK(ggml_backend_sched_moe_hybrid_configure_v1(sched.get(), &changed_config) ==
          GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
    changed_config = config;
    changed_config.resident_batch ^= 1;
    CHECK(ggml_backend_sched_moe_hybrid_configure_v1(sched.get(), &changed_config) ==
          GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
    changed_config.resident_batch = 2;
    CHECK(ggml_backend_sched_moe_hybrid_configure_v1(sched.get(), &changed_config) ==
          GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
    changed_config = config;
    ++changed_config.device_bytes;
    CHECK(ggml_backend_sched_moe_hybrid_configure_v1(sched.get(), &changed_config) ==
          GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
    auto changed_owner = owner;
    ++changed_owner.generation;
    changed_config = config;
    changed_config.source_owner = &changed_owner;
    CHECK(ggml_backend_sched_moe_hybrid_configure_v1(sched.get(), &changed_config) ==
          GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
    for (const auto & region : fixture.result.get_moe_regions()) {
        const int32_t prepare_status = region.prepare_hybrid(sched.get(), owner, 2);
        if (prepare_status != expected_prepare) {
            fprintf(stderr, "test-moe-cache: hybrid region prepare failed status=%d\n", prepare_status);
        }
        CHECK(prepare_status == expected_prepare);
    }
    ggml_backend_moe_hybrid_state_v1 prepared = {};
    prepared.struct_size = sizeof(prepared);
    CHECK(ggml_backend_sched_moe_hybrid_state_v1(sched.get(), &prepared));
    if (expected_prepare == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK) {
        CHECK(prepared.prepared_device_bytes > 0 && prepared.prepared_device_bytes == prepared.device_bytes &&
              (config.device_bytes == 0 || prepared.device_bytes <= config.device_bytes));
        CHECK(prepared.prepared_cpu_bytes > 0 &&
              (config.cpu_bytes == 0 || prepared.prepared_cpu_bytes <= config.cpu_bytes));
        CHECK(prepared.pinned_bytes > 0 && (config.pinned_bytes == 0 || prepared.pinned_bytes == config.pinned_bytes));
    } else {
        CHECK(prepared.prepared_device_bytes == 0);
    }
    if (required_device_bytes != nullptr) {
        *required_device_bytes = prepared.prepared_device_bytes;
    }
    set_inputs(fixture);
    auto * raw = fixture.result.get_moe_regions()[0].body_output;
    std::vector<uint8_t> sentinel(ggml_nbytes(raw), 0xa5);
    ggml_backend_tensor_set(raw, sentinel.data(), 0, sentinel.size());
    CHECK(compute_fixture(nullptr) == GGML_STATUS_FAILED);
    for (const uint32_t domain : {GGML_GRAPH_EXECUTION_DOMAIN_DRAFT, GGML_GRAPH_EXECUTION_DOMAIN_MTP}) {
        auto unsupported = certificate;
        unsupported.domain = domain;
        CHECK(compute_fixture(&unsupported) == GGML_STATUS_FAILED);
    }
    auto parallel = certificate;
    parallel.n_rows = parallel.n_sequences = 2;
    CHECK(compute_fixture(&parallel) == GGML_STATUS_FAILED);
    ggml_backend_sched_set_eval_callback(sched.get(), [](ggml_tensor *, bool, void *) { return false; }, nullptr);
    CHECK(compute_fixture(&certificate) == GGML_STATUS_FAILED);
    ggml_backend_sched_set_eval_callback(sched.get(), nullptr, nullptr);
    auto * first_body = fixture.result.get_moe_regions()[0].first_body;
    first_body->op_params[0] ^= 1;
    CHECK(compute_fixture(&certificate) == GGML_STATUS_FAILED);
    first_body->op_params[0] ^= 1;
    auto stale = certificate;
    ++stale.owner_generation;
    CHECK(compute_fixture(&stale) == GGML_STATUS_FAILED);
    std::vector<uint8_t> unchanged(sentinel.size());
    ggml_backend_tensor_get(raw, unchanged.data(), 0, unchanged.size());
    CHECK(unchanged == sentinel);
    std::vector<uint8_t> tail_sentinel(ggml_nbytes(fixture.output[0]), 0x5a);
    ggml_backend_tensor_set(fixture.output[0], tail_sentinel.data(), 0, tail_sentinel.size());
    std::vector<uint8_t> later_ids(ggml_nbytes(fixture.ids[1]), 0x6b);
    std::vector<uint8_t> later_tail(ggml_nbytes(fixture.output[1]), 0x7c);
    std::vector<uint8_t> staged_sentinel(fixture.staged == nullptr ? 0 : ggml_nbytes(fixture.staged), 0x39);
    if (fixture.staged != nullptr) {
        ggml_backend_tensor_set(fixture.staged, staged_sentinel.data(), 0, staged_sentinel.size());
    }
    const auto check_staged_failure = [&] {
        if (fixture.staged != nullptr) {
            std::vector<uint8_t> bytes(staged_sentinel.size());
            ggml_backend_tensor_get(fixture.staged, bytes.data(), 0, bytes.size());
            CHECK(bytes == staged_sentinel && !stage_pending(stage.get()));
        }
    };
    if (policy.window) {
        ggml_backend_tensor_set(fixture.ids[1], later_ids.data(), 0, later_ids.size());
        ggml_backend_tensor_set(fixture.output[1], later_tail.data(), 0, later_tail.size());
    }
    hybrid_test_barrier barrier;
    if (policy.block_phase == GGML_BACKEND_MOE_HYBRID_TEST_CPU_JOINED) {
        CUDA_OK(cudaEventCreateWithFlags(&barrier.gpu_produced, cudaEventDisableTiming));
    }
    barrier.policy = policy;
    barrier.ids[0] = fixture.ids[0];
    barrier.ids[1] = fixture.ids[1];
    barrier.routes = &routes;
    CHECK(ggml_backend_sched_moe_hybrid_set_test_hook_v1(sched.get(), hybrid_test_barrier::hook, &barrier));
    if (policy.packet_failure != 0) {
        ggml_cuda_moe_grouped_context_test_access::fail_hybrid_packet(*grouped, policy.packet_failure);
    }
    enum ggml_status status = GGML_STATUS_FAILED;
    std::thread source_destroyer;
    std::atomic<bool> source_destroyed{false};
    const uint64_t allocations = ggml_allocation_count();
    if (policy.gpu_pending) {
        std::atomic<bool> returned{false};
        std::thread caller([&] {
            CUDA_OK(cudaSetDevice(device));
            status = compute_fixture(&certificate);
            returned.store(true);
        });
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!barrier.pending.entered.load()) {
            CHECK(std::chrono::steady_clock::now() < deadline);
            std::this_thread::yield();
        }
        CHECK(!returned.load() && cudaStreamQuery(barrier.stream) == cudaErrorNotReady);
        {
            std::lock_guard<std::mutex> lock(barrier.mutex);
            barrier.released = true;
            barrier.changed.notify_all();
        }
        barrier.pending.released.store(true);
        caller.join();
        CUDA_OK(cudaStreamQuery(barrier.stream));
    } else if (policy.quiesce) {
        std::thread caller([&] {
            CUDA_OK(cudaSetDevice(device));
            status = compute_fixture(&certificate);
        });
        {
            std::unique_lock<std::mutex> lock(barrier.mutex);
            CHECK(barrier.changed.wait_for(lock, std::chrono::seconds(10), [&] { return barrier.cpu_waiting; }));
        }
        ggml_backend_moe_hybrid_state_v1 blocked = {};
        blocked.struct_size = sizeof(blocked);
        CHECK(ggml_backend_sched_moe_hybrid_state_v1(sched.get(), &blocked));
        const bool queued = policy.block_phase == GGML_BACKEND_MOE_HYBRID_TEST_CPU_QUEUED;
        const bool early_joined = policy.block_phase == GGML_BACKEND_MOE_HYBRID_TEST_CPU_JOINED;
        const bool joined = early_joined || policy.block_phase == GGML_BACKEND_MOE_HYBRID_TEST_BEFORE_PUBLISH;
        CHECK(blocked.dispatch_active && blocked.ticket_state == (queued ? 1u : joined ? 0u : 2u));
        CHECK(blocked.cpu_active_jobs == (queued || joined ? 0u : 1u));
        if (early_joined) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            cudaError_t ready;
            while ((ready = cudaEventQuery(barrier.gpu_produced)) == cudaErrorNotReady) {
                CHECK(std::chrono::steady_clock::now() < deadline);
                std::this_thread::yield();
            }
            CUDA_OK(ready);
        }
        const auto active_telemetry = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*grouped);
        CHECK(active_telemetry.occupancy_unavailable >= 1);
        CHECK(!ggml_backend_sched_moe_hybrid_set_test_hook_v1(sched.get(), nullptr, nullptr));
        for (auto backend : backends) {
            const auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
            ggml_backend_unload(reg);
            CHECK(ggml_backend_moe_module_retain_v1(reg));
            CHECK(ggml_backend_moe_module_release_v1(reg));
        }
        if (policy.close_source) {
            source_owner->close_moe_source_owner();
            auto * retiring = source_owner.release();
            source_destroyer = std::thread([&, retiring] {
                llama_model_free(retiring);
                source_destroyed.store(true);
            });
        }
        std::atomic<bool> quiesced{false};
        std::thread quiescer([&] {
            ggml_backend_sched_moe_hybrid_quiesce_v1(sched.get());
            quiesced.store(true);
        });
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        do {
            CHECK(std::chrono::steady_clock::now() < deadline);
            CHECK(ggml_backend_sched_moe_hybrid_state_v1(sched.get(), &blocked));
            std::this_thread::yield();
        } while (!blocked.quiescing);
        CHECK(!quiesced.load() && !source_destroyed.load());
        {
            std::lock_guard<std::mutex> lock(barrier.mutex);
            barrier.released = true;
            barrier.changed.notify_all();
        }
        caller.join();
        quiescer.join();
        CHECK(quiesced.load());
    } else {
        status = compute_fixture(&certificate);
    }
    const bool expected_failure = fail_before_publish || policy.fail_phase != 0 || policy.packet_failure != 0 || policy.quiesce ||
        expected_prepare != GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
    if (expected_failure) {
        CHECK(status == GGML_STATUS_FAILED);
        check_staged_failure();
        ggml_backend_tensor_get(raw, unchanged.data(), 0, unchanged.size());
        CHECK(unchanged == sentinel);
        std::vector<uint8_t> tail(tail_sentinel.size());
        ggml_backend_tensor_get(fixture.output[0], tail.data(), 0, tail.size());
        CHECK(tail == tail_sentinel);
        if (policy.window) {
            std::vector<uint8_t> ids(later_ids.size()), output(later_tail.size());
            ggml_backend_tensor_get(fixture.ids[1], ids.data(), 0, ids.size());
            ggml_backend_tensor_get(fixture.output[1], output.data(), 0, output.size());
            CHECK(ids == later_ids && output == later_tail);
        }
    } else {
        CHECK(status == GGML_STATUS_SUCCESS);
        const auto first = active_grouped_tensor_values(fixture.output[1]);
        std::array<std::vector<float>, 2> raw_outputs, tail_outputs;
        for (int layer = 0; layer < 2; ++layer) {
            raw_outputs[layer] = active_grouped_tensor_values(fixture.result.get_moe_regions()[layer].body_output);
            tail_outputs[layer] = active_grouped_tensor_values(fixture.output[layer]);
            ggml_cuda_moe_candidate_group_key key;
            CHECK(grouped->find_down_group_key(fixture.down[layer], &key));
            for (int expert : first_admissions[layer]) {
                CHECK(ggml_cuda_moe_grouped_context_test_access::device_slot_for_expert(*grouped, key, expert) >= 0);
            }
        }
        CHECK(compute_fixture(&certificate) == GGML_STATUS_SUCCESS);
        CHECK(ggml_allocation_count() == allocations);
        if (policy.overlap) {
            CHECK(barrier.cpu_waiting && barrier.gpu_enqueued && barrier.released);
        }
        if (!policy.admission) {
            check_active_grouped_exact_output(first, active_grouped_tensor_values(fixture.output[1]));
        }
        set_inputs(reference);
        CHECK(ggml_backend_sched_graph_compute_ext(reference_sched.get(), reference.result.get_gf(), &certificate) == GGML_STATUS_SUCCESS);
        if (policy.duplicates) {
            auto * graph = reference.result.get_gf();
            const auto index = [&](const ggml_tensor * tensor) {
                for (int i = 0; i < graph->n_nodes; ++i) {
                    if (graph->nodes[i] == tensor) { return i; }
                }
                CHECK(false);
                return -1;
            };
            int start = index(reference.prefix) + 1;
            for (int layer = 0; layer < 2; ++layer) {
                const int body = index(reference.result.get_moe_regions()[layer].first_body);
                const int end = index(reference.output[layer]) + 1;
                auto router = ggml_graph_view(graph, start, body);
                CHECK(ggml_backend_graph_compute(gpu.get(), &router) == GGML_STATUS_SUCCESS);
                ggml_backend_tensor_set(reference.ids[layer], routes[layer].data(), 0, routes[layer].size() * sizeof(int32_t));
                auto remainder = ggml_graph_view(graph, body, end);
                CHECK(ggml_backend_graph_compute(gpu.get(), &remainder) == GGML_STATUS_SUCCESS);
                start = end;
            }
        }
        const auto compare = [&](const std::vector<float> & actual, const std::vector<float> & expected, const char * part, int layer) {
            CHECK(actual.size() == expected.size());
            double squared_error = 0, squared_reference = 0;
            for (size_t i = 0; i < actual.size(); ++i) {
                CHECK(std::isfinite(actual[i]) && std::isfinite(expected[i]));
                const double delta = double(actual[i]) - expected[i];
                squared_error += delta * delta;
                squared_reference += double(expected[i]) * expected[i];
            }
            const double nmse = squared_error / std::max(squared_reference, 1e-30);
            fprintf(stderr, "test-moe-cache: hybrid %s layer=%d NMSE=%g gate_up=%s down=%s embd=%d ff=%d pageable=%d\n",
                part, layer, nmse, ggml_type_name(signature.gate_up_type), ggml_type_name(signature.down_type),
                signature.n_embd, signature.n_ff, pageable);
            CHECK(nmse <= 5e-4);
        };
        for (int layer = 0; layer < 2; ++layer) {
            auto * body = fixture.result.get_moe_regions()[layer].body_output;
            if (!policy.admission) {
                check_active_grouped_exact_output(raw_outputs[layer], active_grouped_tensor_values(body));
            }
            const auto reference_raw = active_grouped_tensor_values(reference.result.get_moe_regions()[layer].body_output);
            const auto reference_tail = active_grouped_tensor_values(reference.output[layer]);
            compare(raw_outputs[layer], reference_raw, "raw", layer);
            compare(active_grouped_tensor_values(body), reference_raw, "raw-repeat", layer);
            compare(tail_outputs[layer], reference_tail, "tail-first", layer);
            compare(active_grouped_tensor_values(fixture.output[layer]), reference_tail, "tail", layer);
        }
        ggml_backend_moe_hybrid_state_v1 state = {};
        state.struct_size = sizeof(state);
        CHECK(ggml_backend_sched_moe_hybrid_state_v1(sched.get(), &state));
        CHECK(state.resident_routes == expected_routes[0] && state.transfer_routes == expected_routes[1] &&
              state.cpu_routes == expected_routes[2]);
        CHECK(state.resident_experts == expected_experts[0] && state.transfer_experts == expected_experts[1] &&
              state.cpu_experts == expected_experts[2]);
        CHECK(state.distinct_experts == expected_experts[0] + expected_experts[1] + expected_experts[2]);
        CHECK(state.cpu_jobs == expected_jobs && state.last_epoch == expected_jobs);
        const uint64_t resident_bodies = policy.resident_batch ? expected_resident_batches : expected_experts[0];
        const bool combined_gpu = captured_window && policy.combine_gpu && policy.admission && n_slots >= policy.routes;
        CHECK(state.resident_batches == (policy.resident_batch ? expected_resident_batches : 0));
        CHECK(state.resident_body_submissions == resident_bodies);
        CHECK(state.gpu_body_submissions == resident_bodies +
            (combined_gpu ? 0 : policy.resident_batch ? expected_resident_batches : expected_transfer_batches));
        CHECK(state.window_combined_regions == (combined_gpu ? expected_resident_batches : 0));
        const bool direct_gather = combined_gpu && policy.direct_gather && policy.quota != 0 && !pageable;
        CHECK(state.window_direct_regions == (direct_gather ? expected_resident_batches : 0) &&
            state.window_compact_select_regions == state.window_direct_regions);
        CHECK(state.packet_regions == (policy.resident_batch ? expected_resident_batches : 0));
        CHECK(state.producer_events == (captured_window ? 0 : state.packet_regions) && state.producer_fences == state.producer_events &&
            state.producer_drain_events == 0);
        if (captured_window) {
            CHECK(state.window_launches > 0 && state.window_waits == state.window_launches &&
                state.window_captures > 0 && state.window_captures <= state.window_launches && state.window_fallbacks == 0);
            const char * disable_fusion = getenv("GGML_CUDA_DISABLE_FUSION");
            CHECK((state.window_fused_nodes == 0) == (disable_fusion != nullptr && atoi(disable_fusion) != 0));
        }
        if (policy.window_fallback) {
            CHECK(state.window_launches == 0 && state.window_waits == 0 && state.window_fallbacks > 0);
        }
        CHECK(state.cpu_upload_bytes == expected_routes[2] * raw->nb[1]);
        CHECK(state.h2d_bytes == transferred_bytes);
        CHECK(state.admission_reserved == expected_admissions && state.admission_committed == expected_admissions &&
              state.admission_replacements == expected_admissions && state.admission_no_slot == expected_no_slot &&
              state.admission_bytes == expected_admission_bytes && state.admission_aborted == 0);
        CHECK((policy.window && policy.auto_memory ? state.device_bytes > prepared.device_bytes && state.pinned_bytes > prepared.pinned_bytes :
            state.device_bytes == prepared.device_bytes && state.pinned_bytes == prepared.pinned_bytes) &&
              state.work_peak <= state.device_bytes);
    }
    ggml_backend_moe_hybrid_state_v1 drained = {};
    drained.struct_size = sizeof(drained);
    CHECK(ggml_backend_sched_moe_hybrid_state_v1(sched.get(), &drained));
    CHECK(drained.ticket_state == 0 && drained.cpu_active_jobs == 0 && drained.dispatch_active == 0);
    if (policy.resident_batch && expected_failure && expected_prepare == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK) {
        CHECK(drained.packet_regions == 0 && drained.producer_events == drained.producer_fences &&
            drained.producer_drain_events == (captured_window ? 0 : 2));
        if (captured_window) {
            const uint64_t submitted = policy.fail_phase == GGML_BACKEND_MOE_HYBRID_TEST_WINDOW_SUBMIT ? 0 : 1;
            CHECK(drained.window_launches == submitted && drained.window_waits == submitted && drained.window_fallbacks == 0);
        }
    }
    CHECK(ggml_backend_sched_moe_hybrid_set_test_hook_v1(sched.get(), nullptr, nullptr));
    if (!policy.admission || policy.quota == 0) {
        CHECK(residency() == original_residency);
    }
    if (policy.admission && expected_failure) {
        if (drained.admission_committed != 0 || drained.admission_aborted != drained.admission_reserved) {
            fprintf(stderr, "test-moe-cache: hybrid admission failure phase=%u reserved=%llu committed=%llu aborted=%llu\n",
                policy.fail_phase, (unsigned long long) drained.admission_reserved,
                (unsigned long long) drained.admission_committed, (unsigned long long) drained.admission_aborted);
        }
        CHECK(drained.admission_committed == 0 && drained.admission_aborted == drained.admission_reserved);
    }
    const uint64_t populated = verify_payloads();
    const auto telemetry = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*grouped);
    CHECK(telemetry.captures == 0 && telemetry.replays == 0);
    CHECK(telemetry.occupancy_unavailable == 0 && telemetry.populated_slots == populated &&
          telemetry.slot_capacity == 2 * n_slots);
    if (!expected_failure) {
        CHECK(telemetry.fallback == 0 && telemetry.rollback == 0 && telemetry.prepare_error == 0 && telemetry.finish_error == 0);
    }
    if (!expected_failure && policy.reset) {
        ggml_backend_sched_reset(sched.get());
        fixture.build_graph(true, nullptr, nullptr, policy.window);
        for (auto & region : fixture.result.get_moe_regions()) {
            ggml_set_output(region.body_output);
            CHECK(region.place(sched.get(), gpu.get()));
        }
        ggml_backend_sched_set_tensor_backend(sched.get(), fixture.prefix, cpu.get());
        CHECK(ggml_backend_sched_alloc_graph(sched.get(), fixture.result.get_gf()));
        ggml_backend_sched_get_buffer_state(sched.get(), &generation, &shrink_generation);
        for (auto & region : fixture.result.get_moe_regions()) {
            CHECK(region.finalize_metadata(sched.get(), fixture.result.get_gf(), 1, generation) == GGML_BACKEND_SCHED_REGION_STATUS_V1_OK);
            CHECK(region.prepare_hybrid(sched.get(), owner, 2) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        }
        set_inputs(fixture);
        CHECK(compute_fixture(&certificate) == GGML_STATUS_SUCCESS);
        ggml_backend_moe_hybrid_state_v1 resumed = {};
        resumed.struct_size = sizeof(resumed);
        CHECK(ggml_backend_sched_moe_hybrid_state_v1(sched.get(), &resumed));
        CHECK(resumed.last_epoch >= drained.last_epoch);
        CHECK(resumed.cpu_jobs == drained.cpu_jobs || resumed.last_epoch > drained.last_epoch);
    } else if (!expected_failure && !policy.duplicates) {
        auto sequential = certificate;
        sequential.row_semantics = GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL;
        CHECK(compute_fixture(&sequential) == GGML_STATUS_SUCCESS);
        const auto ordinary = active_grouped_tensor_values(fixture.output[1]);
        CHECK(compute_fixture(&certificate) == GGML_STATUS_SUCCESS);
        const auto resumed = active_grouped_tensor_values(fixture.output[1]);
        double error = 0, norm = 0;
        for (size_t i = 0; i < ordinary.size(); ++i) {
            CHECK(std::isfinite(ordinary[i]) && std::isfinite(resumed[i]));
            error += (double(ordinary[i]) - resumed[i]) * (double(ordinary[i]) - resumed[i]);
            norm += double(ordinary[i]) * ordinary[i];
        }
        CHECK(error / std::max(norm, 1e-30) <= 5e-4);
    }
    if (policy.window_replays != 0) {
        CHECK(policy.window && !expected_failure && !policy.duplicates);
        ggml_backend_moe_hybrid_state_v1 before = {};
        before.struct_size = sizeof(before);
        CHECK(ggml_backend_sched_moe_hybrid_state_v1(sched.get(), &before));
        const auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(gpu.get()));
        auto trim = reinterpret_cast<uint64_t (*)(ggml_backend_t)>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_trim_transient_pools"));
        CHECK(trim != nullptr);
        for (uint32_t iteration = 0; iteration < policy.window_replays; ++iteration) {
            if (iteration == policy.window_replays / 2) {
                trim(gpu.get());
            }
            for (uint32_t layer = 0; layer < 2; ++layer) {
                for (uint32_t rank = 0; rank < policy.routes; ++rank) {
                    routes[layer][rank] = (iteration * 3 + rank * 2 + layer) % n_experts;
                }
            }
            set_inputs(fixture);
            set_inputs(reference);
            CHECK(compute_fixture(&certificate) == GGML_STATUS_SUCCESS);
            CHECK(ggml_backend_sched_graph_compute(reference_sched.get(), reference.result.get_gf()) == GGML_STATUS_SUCCESS);
            for (uint32_t layer = 0; layer < 2; ++layer) {
                const auto actual = active_grouped_tensor_values(fixture.output[layer]);
                const auto expected = active_grouped_tensor_values(reference.output[layer]);
                double error = 0, norm = 0;
                for (size_t i = 0; i < actual.size(); ++i) {
                    CHECK(std::isfinite(actual[i]) && std::isfinite(expected[i]));
                    error += (double(actual[i]) - expected[i]) * (double(actual[i]) - expected[i]);
                    norm += double(expected[i]) * expected[i];
                }
                CHECK(error / std::max(norm, 1e-30) <= 5e-4);
            }
        }
        ggml_backend_moe_hybrid_state_v1 after = {};
        after.struct_size = sizeof(after);
        CHECK(ggml_backend_sched_moe_hybrid_state_v1(sched.get(), &after));
        CHECK(after.window_launches == before.window_launches + policy.window_replays &&
            after.window_waits == after.window_launches && after.window_captures == before.window_captures + 1 &&
            after.window_fallbacks == 0 && after.producer_events == 0 && after.ticket_state == 0 && after.cpu_active_jobs == 0);
        for (uint32_t layer = 0; layer < 2; ++layer) {
            ggml_cuda_moe_candidate_group_key key;
            CHECK(grouped->find_down_group_key(fixture.down[layer], &key));
            routes[layer].clear();
            for (int expert = 0; expert < n_experts && routes[layer].size() < policy.routes; ++expert) {
                if (ggml_cuda_moe_grouped_context_test_access::device_slot_for_expert(*grouped, key, expert) < 0) {
                    routes[layer].push_back(expert);
                }
            }
            CHECK(routes[layer].size() == policy.routes);
        }
        set_inputs(fixture);
        ggml_backend_tensor_set(raw, sentinel.data(), 0, sentinel.size());
        ggml_backend_tensor_set(fixture.output[0], tail_sentinel.data(), 0, tail_sentinel.size());
        ggml_backend_tensor_set(fixture.ids[1], later_ids.data(), 0, later_ids.size());
        ggml_backend_tensor_set(fixture.output[1], later_tail.data(), 0, later_tail.size());
        if (fixture.staged != nullptr) {
            ggml_backend_tensor_set(fixture.staged, staged_sentinel.data(), 0, staged_sentinel.size());
        }
        barrier.policy.fail_phase = GGML_BACKEND_MOE_HYBRID_TEST_CPU_BEFORE_COMMIT;
        CHECK(ggml_backend_sched_moe_hybrid_set_test_hook_v1(sched.get(), hybrid_test_barrier::hook, &barrier));
        CHECK(compute_fixture(&certificate) == GGML_STATUS_FAILED);
        check_staged_failure();
        ggml_backend_tensor_get(raw, unchanged.data(), 0, unchanged.size());
        std::vector<uint8_t> tail(tail_sentinel.size()), ids(later_ids.size()), output(later_tail.size());
        ggml_backend_tensor_get(fixture.output[0], tail.data(), 0, tail.size());
        ggml_backend_tensor_get(fixture.ids[1], ids.data(), 0, ids.size());
        ggml_backend_tensor_get(fixture.output[1], output.data(), 0, output.size());
        CHECK(unchanged == sentinel && tail == tail_sentinel && ids == later_ids && output == later_tail);
        ggml_backend_moe_hybrid_state_v1 failed = {};
        failed.struct_size = sizeof(failed);
        CHECK(ggml_backend_sched_moe_hybrid_state_v1(sched.get(), &failed));
        CHECK(failed.window_launches == after.window_launches + 1 && failed.window_captures == after.window_captures &&
            failed.cpu_jobs == after.cpu_jobs + 1 && failed.admission_committed == after.admission_committed &&
            failed.ticket_state == 0 && failed.cpu_active_jobs == 0);
        CHECK(ggml_backend_sched_moe_hybrid_set_test_hook_v1(sched.get(), nullptr, nullptr));
        CHECK(compute_fixture(&certificate) == GGML_STATUS_SUCCESS);
        set_inputs(reference);
        CHECK(ggml_backend_sched_graph_compute(reference_sched.get(), reference.result.get_gf()) == GGML_STATUS_SUCCESS);
        for (uint32_t layer = 0; layer < 2; ++layer) {
            const auto actual = active_grouped_tensor_values(fixture.output[layer]);
            const auto expected = active_grouped_tensor_values(reference.output[layer]);
            double error = 0, norm = 0;
            for (size_t i = 0; i < actual.size(); ++i) {
                CHECK(std::isfinite(actual[i]) && std::isfinite(expected[i]));
                error += (double(actual[i]) - expected[i]) * (double(actual[i]) - expected[i]);
                norm += double(expected[i]) * expected[i];
            }
            CHECK(error / std::max(norm, 1e-30) <= 5e-4);
        }
        CHECK(ggml_backend_sched_moe_hybrid_state_v1(sched.get(), &failed));
        CHECK(failed.window_launches == after.window_launches + 2 && failed.window_captures == after.window_captures &&
            failed.window_waits == failed.window_launches && failed.window_fallbacks == 0);
        fprintf(stderr, "test-moe-cache: conditional window arithmetic replays=%u changed_ids=1 workspace_rebuild=1 OK\n", policy.window_replays);
        fprintf(stderr, "test-moe-cache: conditional window CPU failure and recovery on replay OK\n");
    }
    sched.reset();
    if (barrier.gpu_produced != nullptr) {
        CUDA_OK(cudaEventDestroy(barrier.gpu_produced));
    }
    if (source_destroyer.joinable()) {
        source_destroyer.join();
        CHECK(source_destroyed.load());
    } else {
        source_owner->close_moe_source_owner();
    }
    fprintf(stderr, "test-moe-cache: hybrid scheduler join and drain gate_up=%s down=%s embd=%d ff=%d pageable=%d failure=%d prepare=%d quota=%u routes=%u hits=%u duplicate=%d overlap=%d fail_phase=%u quiesce=%d reset=%d admission=%d resident_batch=%d sparse_hits=%d no_host_alias=%d staged_input=%d boundary_overlap=%d packet_failure=%u prepared_bytes=%llu OK\n",
        ggml_type_name(signature.gate_up_type), ggml_type_name(signature.down_type), signature.n_embd, signature.n_ff,
        pageable, fail_before_publish, expected_prepare, policy.quota, policy.routes, policy.hits, policy.duplicates,
        policy.overlap, policy.fail_phase, policy.quiesce, policy.reset, policy.admission, policy.resident_batch,
        policy.sparse_hits, policy.no_host_alias, policy.staged_input, policy.boundary_overlap, policy.packet_failure,
        (unsigned long long) prepared.prepared_device_bytes);
}

void test_hybrid_rows_runtime(int device, uint32_t rows, bool pageable) {
    layer_fixture fixture(0, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_TYPE_Q4_0, pageable,
        {device, device}, all_cached_layers, rows * n_used);
    layer_fixture reference(0, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_TYPE_Q4_0, pageable,
        {device, device}, 0, rows * n_used);
    for (auto * target : {&fixture, &reference}) {
        target->build_graph(false, nullptr, nullptr, false, rows);
        for (auto & region : target->result.get_moe_regions()) {
            for (auto * tensor : {region.body_output, region.route, target->output[region.layer]}) {
                while (tensor->view_src != nullptr) { tensor = tensor->view_src; }
                ggml_set_input(tensor);
                ggml_set_output(tensor);
            }
        }
    }
    std::unique_ptr<llama_model> source_owner(llama_model_create(LLM_ARCH_QWEN3MOE, llama_model_default_params()));
    for (const auto & source : fixture.tensors) {
        CHECK(source_owner->record_moe_readable_source(source.tensor, source.tensor->data, ggml_nbytes(source.tensor)));
    }
    ggml_backend_moe_source_owner_v1 owner = {};
    CHECK(source_owner->moe_source_owner_v1(&owner));
    ggml_backend_ptr gpu(ggml_backend_cuda_init(device));
    ggml_backend_ptr cpu(ggml_backend_cpu_init());
    ggml_backend_t backends[] = {gpu.get(), cpu.get()};
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, 2, 256, false, true));
    ggml_backend_sched_ptr oracle(ggml_backend_sched_new(backends, nullptr, 2, 256, false, true));
    CHECK(ggml_backend_sched_set_resizable(sched.get(), nullptr));
    const auto snapshot = fixture.manifest();
    CHECK(ggml_backend_cuda_moe_candidate_replace_v2(gpu.get(), &snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    for (auto & region : fixture.result.get_moe_regions()) { CHECK(region.place(sched.get(), gpu.get())); }
    for (auto & region : reference.result.get_moe_regions()) { CHECK(region.place(oracle.get(), gpu.get())); }
    CHECK(ggml_backend_sched_alloc_graph(sched.get(), fixture.result.get_gf()));
    CHECK(ggml_backend_sched_alloc_graph(oracle.get(), reference.result.get_gf()));
    auto certificate = layer_certificate();
    certificate.flags = GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED;
    certificate.row_semantics = GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SPECULATIVE;
    certificate.n_rows = rows;
    const auto set_inputs = [&](layer_fixture & target, bool warm, uint32_t step) {
        std::vector<float> values(size_t(rows) * n_dim);
        for (size_t i = 0; i < values.size(); ++i) { values[i] = 0.003f * (1 + (i * 7 + i / n_dim * 11 + step) % 31); }
        ggml_backend_tensor_set(target.input, values.data(), 0, values.size() * sizeof(float));
        for (auto * logits : target.logits) {
            std::vector<float> scores(size_t(rows) * n_experts, -10.0f);
            for (uint32_t row = 0; row < rows; ++row) {
                scores[row * n_experts] = 10.0f;
                scores[row * n_experts + (warm ? 1 : 2 + row)] = 9.0f;
            }
            ggml_backend_tensor_set(logits, scores.data(), 0, scores.size() * sizeof(float));
        }
    };
    set_inputs(fixture, true, 0);
    CHECK(ggml_backend_sched_graph_compute_ext(sched.get(), fixture.result.get_gf(), &certificate) == GGML_STATUS_SUCCESS);
    uint64_t generation = 0, shrink = 0;
    ggml_backend_sched_get_buffer_state(sched.get(), &generation, &shrink);
    for (auto & region : fixture.result.get_moe_regions()) {
        CHECK(region.finalize_metadata(sched.get(), fixture.result.get_gf(), 1, generation) == GGML_BACKEND_SCHED_REGION_STATUS_V1_OK);
    }
    ggml_backend_moe_hybrid_config_v1 config = {};
    config.struct_size = sizeof(config);
    config.backend = gpu.get();
    config.source_owner = &owner;
    config.cpu_module_acquire = ggml_backend_moe_cpu_module_acquire_v1;
    config.module_retain = ggml_backend_moe_module_retain_v1;
    config.module_release = ggml_backend_moe_module_release_v1;
    config.n_threads = 2;
    config.max_regions = 2;
    config.max_prepared_regions = 2 * (n_used + 1);
    config.gpu_miss_quota = rows == 4 ? 2 : 1;
    config.admission_quota = 1;
    config.demand_admission = 1;
    config.resident_batch = 1;
    config.cpu_flags = GGML_BACKEND_MOE_CPU_SERVICE_FLAG_V1_ALLOW_UNKNOWN_THREAD_STACK_BYTES |
        GGML_BACKEND_MOE_CPU_SERVICE_FLAG_V1_ALLOW_UNPROVEN_RUNTIME_ALLOCATIONS;
    const auto set_env = [](const char * name, const char * value) {
#ifdef _WIN32
        CHECK(_putenv_s(name, value == nullptr ? "" : value) == 0);
#else
        CHECK((value == nullptr ? unsetenv(name) : setenv(name, value, 1)) == 0);
#endif
    };
    const char * names[] = {"GGML_MOE_HYBRID_WINDOW", "GGML_MOE_HYBRID_TEST_NO_HOST_ALIAS"};
    std::string saved[2];
    bool present[2];
    for (size_t i = 0; i < 2; ++i) {
        const auto * value = getenv(names[i]);
        present[i] = value != nullptr;
        saved[i] = value == nullptr ? "" : value;
        set_env(names[i], i == 0 || pageable ? "1" : nullptr);
    }
    CHECK(ggml_backend_sched_moe_hybrid_configure_v1(sched.get(), &config) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    for (size_t i = 0; i < 2; ++i) { set_env(names[i], present[i] ? saved[i].c_str() : nullptr); }
    for (const auto & region : fixture.result.get_moe_regions()) {
        const int32_t prepared = region.prepare_hybrid(sched.get(), owner, 2, &certificate);
        fprintf(stderr, "test-moe-cache: hybrid runtime prepare rows=%u layer=%d status=%d\n", rows, region.layer, prepared);
        CHECK(prepared == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    }
    set_inputs(fixture, false, 0);
    uint32_t fail_phase = GGML_BACKEND_MOE_HYBRID_TEST_CPU_QUEUED;
    const auto hook = [](void * data, uint32_t phase, uint64_t, void *) { return phase != *static_cast<uint32_t *>(data); };
    CHECK(ggml_backend_sched_moe_hybrid_set_test_hook_v1(sched.get(), hook, &fail_phase));
    const auto compute = [&] { return ggml_backend_sched_graph_compute_ext(sched.get(), fixture.result.get_gf(), &certificate); };
    const auto failure = [&] {
        std::vector<std::pair<ggml_tensor *, std::vector<uint8_t>>> sentinels;
        for (auto * tensor : {fixture.result.get_moe_regions()[0].body_output, fixture.output[0], fixture.ids[1], fixture.output[1]}) {
            sentinels.push_back({tensor, std::vector<uint8_t>(ggml_nbytes(tensor), 0x6b)});
            ggml_backend_tensor_set(tensor, sentinels.back().second.data(), 0, sentinels.back().second.size());
        }
        CHECK(compute() == GGML_STATUS_FAILED);
        for (const auto & sentinel : sentinels) {
            std::vector<uint8_t> actual(sentinel.second.size());
            ggml_backend_tensor_get(sentinel.first, actual.data(), 0, actual.size());
            CHECK(actual == sentinel.second);
        }
        ggml_backend_moe_hybrid_state_v1 state = {};
        state.struct_size = sizeof(state);
        CHECK(ggml_backend_sched_moe_hybrid_state_v1(sched.get(), &state));
        CHECK(state.ticket_state == 0 && state.cpu_active_jobs == 0 && state.dispatch_active == 0 && state.window_fallbacks == 0);
    };
    failure();
    fail_phase = 0;
    const auto success = [&](uint32_t step) {
        set_inputs(fixture, false, step);
        set_inputs(reference, false, step);
        CHECK(ggml_backend_sched_graph_compute_ext(oracle.get(), reference.result.get_gf(), &certificate) == GGML_STATUS_SUCCESS);
        CHECK(compute() == GGML_STATUS_SUCCESS);
        for (int layer = 0; layer < 2; ++layer) {
            const auto actual = active_grouped_tensor_values(fixture.output[layer]);
            const auto expected = active_grouped_tensor_values(reference.output[layer]);
            double error = 0, norm = 0;
            for (size_t i = 0; i < actual.size(); ++i) {
                CHECK(std::isfinite(actual[i]) && std::isfinite(expected[i]));
                error += (double(actual[i]) - expected[i]) * (double(actual[i]) - expected[i]);
                norm += double(expected[i]) * expected[i];
            }
            const double nmse = error / std::max(norm, 1e-30);
            fprintf(stderr, "test-moe-cache: hybrid runtime rows=%u layer=%d step=%u cross-backend NMSE=%g\n", rows, layer, step, nmse);
            CHECK(nmse <= 1e-3);
        }
    };
    success(0);
    success(1);
    fail_phase = GGML_BACKEND_MOE_HYBRID_TEST_BEFORE_PUBLISH;
    failure();
    fail_phase = 0;
    success(2);
    auto stale = certificate;
    ++stale.owner_generation;
    CHECK(ggml_backend_sched_graph_compute_ext(sched.get(), fixture.result.get_gf(), &stale) == GGML_STATUS_FAILED);
    ggml_backend_moe_hybrid_state_v1 state = {};
    state.struct_size = sizeof(state);
    CHECK(ggml_backend_sched_moe_hybrid_state_v1(sched.get(), &state));
    CHECK(state.cpu_jobs > 0 && state.resident_routes > 0 && state.transfer_routes > 0 && state.cpu_routes > 0 &&
        state.cpu_execute_calls > 0 && state.cpu_batch_rows > state.cpu_execute_calls &&
        state.window_launches == 5 && state.window_captures == 1 && state.window_waits == state.window_launches &&
        state.window_fallbacks == 0 && state.ticket_state == 0 && state.cpu_active_jobs == 0 && state.dispatch_active == 0);
    if (rows == 4) { CHECK(state.transfer_experts > state.admission_reserved); }
    CHECK(ggml_backend_sched_moe_hybrid_set_test_hook_v1(sched.get(), nullptr, nullptr));
    sched.reset();
    source_owner->close_moe_source_owner();
    fprintf(stderr, "test-moe-cache: hybrid runtime rows=%u pageable=%d duplicate/scatter/failure/drain/replay/recovery OK\n", rows, pageable);
}

void test_fidelity_allocation_graph() {
    auto * buft = ggml_backend_cpu_buffer_type();
    for (bool output_view : {false, true}) {
        ggml_context_ptr ctx(ggml_init({64 * ggml_tensor_overhead() + ggml_graph_overhead_custom(64, false), nullptr, true}));
        CHECK(ctx);
        auto * graph = ggml_new_graph_custom(ctx.get(), 64, false);
        auto * input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n_dim);
        ggml_set_input(input);
        auto * activation = ggml_scale(ctx.get(), input, 1.0f);
        auto * ids = ggml_argsort(ctx.get(), activation, GGML_SORT_ORDER_DESC);
        auto * route = ggml_view_1d(ctx.get(), ids, n_used, sizeof(int32_t));
        ggml_build_forward_expand(graph, route);
        ggml_tensor * retained = nullptr;
        if (output_view) {
            retained = ggml_view_1d(ctx.get(), activation, n_dim / 2, 8 * sizeof(float));
            ggml_set_output(retained);
            ggml_build_forward_expand(graph, retained);
        }
        ggml_backend_moe_hybrid_region_v1 region = {};
        region.first_node = graph->n_nodes;
        auto * hidden = ggml_sqr(ctx.get(), activation);
        auto * body_output = ggml_sqrt(ctx.get(), hidden);
        ggml_build_forward_expand(graph, body_output);
        region.last_node = graph->n_nodes - 1;
        region.activation = activation;
        region.ids = route;
        region.output = body_output;
        if (retained) { ggml_build_forward_expand(graph, ggml_scale(ctx.get(), retained, 3.0f)); }
        auto * later = ggml_scale(ctx.get(), input, 2.0f);
        auto * output = ggml_add(ctx.get(), body_output, later);
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
        ggml_gallocr_ptr original(ggml_gallocr_new(buft));
        CHECK(original && ggml_gallocr_alloc_graph(original.get(), graph));
        std::vector<ggml_tensor> headers;
        for (int i = 0; i < graph->n_nodes; ++i) { headers.push_back(*graph->nodes[i]); }
        ggml_tensor * outputs[] = {retained, output};
        const ggml_backend_moe_hybrid_region_v1 * regions[] = {&region};
        ggml_cuda_moe_fidelity_window_query_v1 query = {};
        query.graph = graph;
        query.n_regions = 1;
        query.regions = regions;
        query.public_output = output;
        if (retained) { query.n_public_outputs = 2; query.public_outputs = outputs; }

        ggml_cuda_moe_fidelity_graph_allocator plan;
        CHECK(plan.measure(buft, query) && plan.buffer_bytes() > 0 && plan.metadata_bytes() > 0);
        CHECK(plan.find(hidden) == nullptr && plan.find(body_output)->data == nullptr);
        CHECK(plan.find(input)->data == input->data);
        const size_t bytes = plan.buffer_bytes(), metadata = plan.metadata_bytes();
        CHECK(plan.allocate() && plan.buffer_bytes() == bytes && plan.metadata_bytes() == metadata);
        CHECK(plan.find(input)->data == input->data && plan.find(input)->buffer == input->buffer);
        CHECK(plan.find(activation)->buffer != activation->buffer);
        CHECK(ggml_backend_buffer_get_usage(plan.find(activation)->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        const auto overlaps = [&](const ggml_tensor * a, const ggml_tensor * b) {
            const auto * left = plan.find(a);
            const auto * right = plan.find(b);
            CHECK(left && right && left->data && right->data);
            const uintptr_t l = reinterpret_cast<uintptr_t>(left->data), r = reinterpret_cast<uintptr_t>(right->data);
            return l < r + ggml_nbytes(right) && r < l + ggml_nbytes(left);
        };
        CHECK(!overlaps(activation, body_output) && !overlaps(ids, body_output));
        CHECK(plan.find(route)->data == static_cast<uint8_t *>(plan.find(ids)->data) + route->view_offs);
        CHECK(overlaps(activation, later) == !output_view);
        if (retained) {
            CHECK(plan.find(retained)->data == static_cast<uint8_t *>(plan.find(activation)->data) + retained->view_offs);
            CHECK(plan.find(activation)->flags & GGML_TENSOR_FLAG_OUTPUT);
            CHECK(!overlaps(retained, output));
        }
        CHECK(!plan.measure(buft, query) && !plan.allocate());
        for (int i = 0; i < graph->n_nodes; ++i) { CHECK(memcmp(&headers[i], graph->nodes[i], sizeof(ggml_tensor)) == 0); }

        const auto invalid = [&]() {
            ggml_cuda_moe_fidelity_graph_allocator rejected;
            CHECK(!rejected.measure(buft, query) && rejected.buffer_bytes() == 0 && rejected.metadata_bytes() == 0);
        };
        auto * saved = output->src[1];
        output->src[1] = hidden;
        invalid();
        output->src[1] = saved;
        ggml_set_output(hidden);
        invalid();
        hidden->flags &= ~GGML_TENSOR_FLAG_OUTPUT;
        saved = route->view_src;
        route->view_src = route;
        invalid();
        route->view_src = saved;
        const size_t offset = route->view_offs;
        route->view_offs = SIZE_MAX;
        invalid();
        route->view_offs = offset;
        const uint32_t first = region.first_node;
        region.first_node = region.last_node + 1;
        invalid();
        region.first_node = first;
        region.ids = body_output;
        invalid();
        region.ids = route;
        plan.reset();
        CHECK(plan.buffer_bytes() == 0 && plan.metadata_bytes() == 0 && !plan.find(output));
        fprintf(stderr, "test-moe-cache: fidelity allocation-only output_view=%d bytes=%zu metadata=%zu cut/lifetime/root/output OK\n",
            output_view, bytes, metadata);
    }
}

struct fidelity_graph_owner : std::enable_shared_from_this<fidelity_graph_owner> {
    std::unique_ptr<layer_fixture> fixture, reference;
    std::unique_ptr<llama_model> model;
    ggml_backend_ptr gpu, cpu;
    std::unique_ptr<void, void (*)(void *)> staged_input{nullptr, [](void *) {}}, staged_reference{nullptr, [](void *) {}};
    std::atomic<uint64_t> staged_submissions{0};
    std::vector<ggml_backend_buffer_ptr> input_buffers;
    ggml_backend_sched_ptr sched, oracle;
    std::vector<std::unique_ptr<llm_graph_moe_hybrid_prepared>> metadata;
    std::vector<ggml_backend_moe_hybrid_region_v1> descriptors;
    std::vector<const ggml_backend_moe_hybrid_region_v1 *> regions;
    std::vector<ggml_tensor *> graph_nodes;
    ggml_cgraph graph = {};
    std::shared_ptr<fidelity_graph_owner> lease;
    uint32_t references = 0;
    bool capture_only = false;
    ggml_status (*delegate)(ggml_backend_t, ggml_cgraph *) = nullptr;
    void (*sync_delegate)(ggml_backend_t) = nullptr;
    uint32_t sync_calls = 0, source_entered = 0, sync_before_failure = 0;

    static int32_t retain(void * opaque) {
        auto & owner = *static_cast<fidelity_graph_owner *>(opaque);
        if (owner.references == UINT32_MAX) { return 1; }
        if (!owner.references++) { owner.lease = owner.shared_from_this(); }
        return 0;
    }
    static int32_t release(void * opaque) {
        auto & owner = *static_cast<fidelity_graph_owner *>(opaque);
        if (!owner.references) { return 1; }
        if (!--owner.references) { owner.lease.reset(); }
        return 0;
    }
};

fidelity_graph_owner * fidelity_probe = nullptr;
static ggml_status fidelity_capture_split(ggml_backend_t backend, ggml_cgraph * graph) {
    auto & owner = *fidelity_probe;
    bool body = false;
    for (int i = 0; i < graph->n_nodes; ++i) { body |= graph->nodes[i]->op == GGML_OP_MUL_MAT_ID; }
    if (body) {
        CHECK(owner.graph_nodes.empty());
        owner.graph = *graph;
        owner.graph_nodes.assign(graph->nodes, graph->nodes + graph->n_nodes);
        owner.graph.nodes = owner.graph_nodes.data();
        if (owner.capture_only) { return GGML_STATUS_SUCCESS; }
    }
    return owner.delegate(backend, graph);
}

static void fidelity_check_cpu_oracle(fidelity_graph_owner & owner, ggml_backend_reg_t module,
        const ggml_backend_moe_source_owner_v1 & source_owner, uint32_t rows, uint32_t routes) {
    auto & fixture = *owner.fixture;
    auto & reference = *owner.reference;
    const auto & query = *owner.descriptors[0].cpu_batch_queries[0];
    const uint32_t count = rows * routes;
    CHECK(query.routes_per_row == 1 && query.bucket_rows >= count && query.n_dynamic_inputs == 2 && query.n_live_outputs == 1);
    CHECK(fixture.route_probe && reference.route_probe);
    std::vector<int32_t> ids(count), expected_ids(count);
    ggml_backend_tensor_get(fixture.route_probe, ids.data(), 0, ids.size() * sizeof(int32_t));
    ggml_backend_tensor_get(reference.route_probe, expected_ids.data(), 0, expected_ids.size() * sizeof(int32_t));
    CHECK(ids == expected_ids);
    const auto input = active_grouped_tensor_values(fixture.bf16_prefix ? fixture.bf16_prefix : fixture.input);
    const auto expected_input = active_grouped_tensor_values(reference.bf16_prefix ? reference.bf16_prefix : reference.input);
    CHECK(input.size() == expected_input.size() && memcmp(input.data(), expected_input.data(), input.size() * sizeof(float)) == 0);
    std::vector<uint32_t> source_rows(count), scatter(count);
    for (uint32_t i = 0; i < count; ++i) { source_rows[i] = i / routes; scatter[i] = i; }
    const auto get = reinterpret_cast<ggml_backend_moe_cpu_region_service_v1_t>(
        ggml_backend_reg_get_proc_address(module, GGML_BACKEND_MOE_CPU_REGION_SERVICE_V1_PROC_NAME));
    CHECK(get);
    const auto * api = get();
    CHECK(api && api->abi_version == 1 && api->struct_size == sizeof(*api));
    ggml_backend_moe_cpu_service_config_v1 config = {};
    config.struct_size = sizeof(config);
    config.abi_version = 1;
    config.source_owner = &source_owner;
    config.n_threads = query.n_threads;
    config.n_lanes = query.n_lanes;
    config.max_regions = 1;
    config.flags = GGML_BACKEND_MOE_CPU_SERVICE_FLAG_V1_ALLOW_UNKNOWN_THREAD_STACK_BYTES |
        GGML_BACKEND_MOE_CPU_SERVICE_FLAG_V1_ALLOW_UNPROVEN_RUNTIME_ALLOCATIONS;
    ggml_backend_moe_cpu_service_v1_t service = nullptr;
    CHECK(api->create(&config, &service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    ggml_backend_moe_cpu_prepared_requirements_v1 prepared = {};
    prepared.struct_size = sizeof(prepared);
    prepared.abi_version = 1;
    ggml_backend_moe_cpu_prepared_region_v1_t region = 0;
    CHECK(api->prepare(service, &query, &prepared, &region) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    const size_t width = query.live_outputs[0]->ne[0];
    CHECK(query.live_outputs[0]->nb[1] == width * sizeof(float));
    std::vector<float> cpu_output(size_t(query.scatter_capacity) * width);
    ggml_backend_moe_cpu_region_binding_v1 binding{sizeof(binding), count, count, ids.data(), source_rows.data(), scatter.data()};
    ggml_backend_moe_cpu_dynamic_input_v1 inputs[2] = {{input.data(), input.size() * sizeof(float), query.activation->nb[2]}, {}};
    ggml_backend_moe_cpu_output_v1 output{cpu_output.data(), cpu_output.size() * sizeof(float), width * sizeof(float)};
    ggml_backend_moe_cpu_execute_v1 execution = {};
    execution.struct_size = sizeof(execution);
    execution.epoch = 1;
    execution.graph_uid = query.graph_uid;
    execution.graph_generation = query.graph_generation;
    execution.source_generation = query.source_generation;
    execution.binding = &binding;
    execution.dynamic_inputs = inputs;
    execution.n_dynamic_inputs = 2;
    execution.outputs = &output;
    execution.n_outputs = 1;
    ggml_backend_moe_cpu_execute_result_v1 result = {};
    result.struct_size = sizeof(result);
    CHECK(api->execute(service, region, &execution, &result) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    CHECK(result.flags == GGML_BACKEND_MOE_CPU_EXECUTE_RESULT_FLAG_V1_PUBLISHED && result.published_routes == count && result.published_outputs == 1);
    CHECK(api->close(service) == 0 && api->drain(service) == 0 && api->destroy_region(service, &region) == 0 && api->destroy(&service) == 0);
    const auto actual = active_grouped_tensor_values(fixture.result.get_moe_regions()[0].body_output);
    const auto gpu = active_grouped_tensor_values(reference.result.get_moe_regions()[0].body_output);
    CHECK(actual.size() == size_t(count) * width && gpu.size() == actual.size());
    uint32_t cpu_routes = 0, gpu_routes = 0, unmatched = 0;
    for (uint32_t route = 0; route < count; ++route) {
        const auto offset = size_t(route) * width;
        const bool cpu_exact = memcmp(actual.data() + offset, cpu_output.data() + offset, width * sizeof(float)) == 0;
        const bool gpu_exact = memcmp(actual.data() + offset, gpu.data() + offset, width * sizeof(float)) == 0;
        double error = 0, norm = 0;
        for (size_t i = offset; i < offset + width; ++i) {
            CHECK(std::isfinite(cpu_output[i]));
            const double delta = double(gpu[i]) - cpu_output[i];
            error += delta * delta;
            norm += double(gpu[i]) * gpu[i];
        }
        cpu_routes += cpu_exact && !gpu_exact;
        gpu_routes += gpu_exact;
        unmatched += !cpu_exact && !gpu_exact;
        fprintf(stderr, "test-moe-cache: fidelity CPU oracle row=%u route=%u expert=%d cpu_exact=%d gpu_exact=%d CPU_GPU_NMSE=%g\n",
            route / routes, route % routes, ids[route], cpu_exact, gpu_exact, error / std::max(norm, 1e-30));
    }
    fprintf(stderr, "test-moe-cache: fidelity CPU oracle first-layer cpu_routes=%u gpu_routes=%u unmatched=%u prepared=%llu\n",
        cpu_routes, gpu_routes, unmatched, (unsigned long long) prepared.prepared_payload_bytes);
    CHECK(cpu_routes > 0 && gpu_routes > 0 && unmatched == 0);
}

static void check_source_body_arithmetic(layer_fixture & fixture, layer_fixture & reference, const std::vector<int32_t> & owners) {
    const auto & region = fixture.result.get_moe_regions()[0];
    const auto * activation = region.first_body->src[1];
    const size_t count = region.body_operations.size() + 8;
    ggml_context_ptr ctx(ggml_init({count * ggml_tensor_overhead() + ggml_graph_overhead_custom(count, false), nullptr, true}));
    ggml_backend_ptr cpu(ggml_backend_cpu_init());
    CHECK(ctx && cpu && fixture.route_probe && owners.size() == size_t(ggml_nelements(region.route)));
    std::unordered_map<const ggml_tensor *, ggml_tensor *> copies;
    const auto copy = [&](const ggml_tensor * tensor) {
        auto * value = ggml_dup_tensor(ctx.get(), tensor);
        memcpy(value->nb, tensor->nb, sizeof(value->nb));
        copies.emplace(tensor, value);
        return value;
    };
    auto * input = copy(activation);
    auto * ids = copy(region.route);
    std::vector<const ggml_tensor *> weights;
    for (const auto * node : region.body_operations) {
        auto * value = copy(node);
        value->op = node->op;
        memcpy(value->op_params, node->op_params, sizeof(value->op_params));
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (!node->src[s]) { continue; }
            auto it = copies.find(node->src[s]);
            if (it == copies.end()) {
                CHECK(node->op == GGML_OP_MUL_MAT_ID && s == 0 && node->src[s]->op == GGML_OP_NONE);
                weights.push_back(node->src[s]);
                value->src[s] = copy(node->src[s]);
            } else { value->src[s] = it->second; }
        }
        if (node->view_src) { value->view_src = copies.at(node->view_src); value->view_offs = node->view_offs; }
    }
    auto * output = copies.at(region.body_output);
    auto * graph = ggml_new_graph_custom(ctx.get(), count, false);
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), cpu.get()));
    CHECK(buffer);
    auto values = active_grouped_tensor_values(fixture.input);
    // The fixture feeds its first body through two adds of the public input.
    for (auto & value : values) { value = (value + value) + value; }
    CHECK(values.size() * sizeof(float) == ggml_nbytes(input));
    ggml_backend_tensor_set(input, values.data(), 0, ggml_nbytes(input));
    std::vector<int32_t> route_ids(owners.size());
    ggml_backend_tensor_get(fixture.route_probe, route_ids.data(), 0, route_ids.size() * sizeof(int32_t));
    std::vector<uint8_t> ids_bytes(ggml_nbytes(ids), 0);
    for (int64_t row = 0; row < ids->ne[1]; ++row) {
        memcpy(ids_bytes.data() + size_t(row) * ids->nb[1], route_ids.data() + size_t(row) * ids->ne[0], size_t(ids->ne[0]) * sizeof(int32_t));
    }
    ggml_backend_tensor_set(ids, ids_bytes.data(), 0, ids_bytes.size());
    for (const auto * weight : weights) {
        std::vector<uint8_t> bytes(ggml_nbytes(weight));
        ggml_backend_tensor_get(weight, bytes.data(), 0, bytes.size());
        ggml_backend_tensor_set(copies.at(weight), bytes.data(), 0, bytes.size());
    }
    ggml_backend_cpu_set_n_threads(cpu.get(), 2);
    CHECK(ggml_backend_graph_compute(cpu.get(), graph) == GGML_STATUS_SUCCESS);
    const auto cpu_values = active_grouped_tensor_values(output);
    const auto gpu_values = active_grouped_tensor_values(reference.result.get_moe_regions()[0].body_output);
    const auto actual = active_grouped_tensor_values(region.body_output);
    CHECK(actual.size() == cpu_values.size() && actual.size() == gpu_values.size());
    double error = 0, energy = 0;
    size_t cpu_routes = 0, gpu_routes = 0;
    for (size_t route = 0; route < owners.size(); ++route) {
        CHECK(owners[route] >= 0 && owners[route] <= 2);
        cpu_routes += owners[route] == 2; gpu_routes += owners[route] != 2;
        for (size_t column = 0; column < size_t(output->ne[0]); ++column) {
            const size_t i = route * size_t(output->ne[0]) + column;
            const double expected = owners[route] == 2 ? cpu_values[i] : gpu_values[i];
            CHECK(std::isfinite(actual[i]) && std::isfinite(expected));
            error += (actual[i] - expected) * (actual[i] - expected); energy += expected * expected;
        }
    }
    fprintf(stderr, "test-moe-cache: source whole-body mixed ordinary arithmetic cpu=%zu gpu=%zu nmse=%.9g\n", cpu_routes, gpu_routes, error / std::max(energy, 1e-30));
    CHECK(cpu_routes && gpu_routes && error / std::max(energy, 1e-30) <= 1e-6);
}

void test_source_sort_resources(int device) {
    const char * env_name = "GGML_CUDA_TOPK_SEGMENTED";
    const char * old_env = getenv(env_name);
    const bool had_env = old_env != nullptr;
    const std::string saved_env = old_env ? old_env : "";
    const auto set_env = [env_name](const char * value) {
#ifdef _WIN32
        CHECK(_putenv_s(env_name, value ? value : "") == 0);
#else
        CHECK((value ? setenv(env_name, value, 1) : unsetenv(env_name)) == 0);
#endif
    };
    ggml_backend_ptr gpu(ggml_backend_cuda_init(device));
    CHECK(gpu);
    for (bool top_k : {false, true}) {
        for (int columns : {512, 2053}) {
            for (int rows : {1, 5}) {
                for (const char * env : {"0", "1"}) {
                    if (env[0] == '1' && (!top_k || rows == 1)) { continue; }
                    set_env(env);
                    ggml_tensor src = {}, dst = {};
                    src.type = GGML_TYPE_F32; dst.type = GGML_TYPE_I32;
                    src.ne[0] = columns; dst.ne[0] = top_k ? 17 : columns;
                    src.ne[1] = dst.ne[1] = rows;
                    src.ne[2] = src.ne[3] = dst.ne[2] = dst.ne[3] = 1;
                    size_t in_stride = sizeof(float), out_stride = sizeof(int);
                    for (int i = 0; i < 4; ++i) {
                        src.nb[i] = in_stride; dst.nb[i] = out_stride;
                        in_stride *= size_t(src.ne[i]); out_stride *= size_t(dst.ne[i]);
                    }
                    dst.op = top_k ? GGML_OP_TOP_K : GGML_OP_ARGSORT; dst.src[0] = &src;
                    dst.op_params[0] = GGML_SORT_ORDER_DESC;
                    const auto query = [&](ggml_cuda_source_sort_resources & out) {
                        return top_k ? ggml_cuda_top_k_prepare_resources(device, &dst, out) :
                            ggml_cuda_argsort_prepare_resources(device, &dst, out);
                    };
                    if (!ggml_backend_supports_op(gpu.get(), &dst)) { continue; }
                    ggml_cuda_source_sort_resources resources, again;
                    CHECK(query(resources)); CHECK(query(again));
                    CHECK(resources.identity == again.identity && resources.pool_bytes == again.pool_bytes);
                    const auto output_type = dst.type;
                    dst.type = GGML_TYPE_F32; CHECK(!query(again)); dst.type = output_type;
                    const auto dim = src.ne[1]; src.ne[1] = INT64_MAX; CHECK(!query(again)); src.ne[1] = dim;
                    CHECK(!ggml_cuda_argsort_prepare_resources(-1, &dst, again));
                    if (!top_k) {
                        dst.op_params[0] = -1; CHECK(!query(again)); dst.op_params[0] = GGML_SORT_ORDER_ASC;
                        CHECK(query(again)); CHECK(again.identity != resources.identity);
                        dst.op_params[0] = GGML_SORT_ORDER_DESC;
                    }
                    set_env(env[0] == '0' ? "1" : "0");
                    CHECK(query(again));
                    if (top_k && rows > 1 && resources.pool_bytes != again.pool_bytes) { CHECK(resources.identity != again.identity); }
                    set_env(env);
                    ggml_backend_buffer_ptr input(ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(device), in_stride));
                    ggml_backend_buffer_ptr output(ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(device), out_stride));
                    ggml_backend_buffer_ptr scratch(ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(device), std::max(size_t(256), resources.pool_bytes)));
                    CHECK(input && output && scratch);
                    src.data = ggml_backend_buffer_get_base(input.get()); src.buffer = input.get();
                    dst.data = ggml_backend_buffer_get_base(output.get()); dst.buffer = output.get();
                    std::vector<float> values(size_t(columns) * rows);
                    for (int r = 0; r < rows; ++r) {
                        for (int c = 0; c < columns; ++c) { values[size_t(r) * columns + c] = float((c * 73) % columns); }
                    }
                    CUDA_OK(cudaMemcpy(src.data, values.data(), in_stride, cudaMemcpyHostToDevice));
                    ggml_cuda_source_sort_test_result captured;
                    CHECK(ggml_cuda_source_sort_capture_for_test(device, &dst, scratch.get(), resources.pool_bytes, captured));
                    CHECK(captured.eager_peak <= resources.pool_bytes && captured.capture_peak <= resources.pool_bytes);
                    std::vector<int> indices(size_t(dst.ne[0]) * rows);
                    CUDA_OK(cudaMemcpy(indices.data(), dst.data, out_stride, cudaMemcpyDeviceToHost));
                    for (int r = 0; r < rows; ++r) {
                        std::vector<int> expected(columns);
                        std::iota(expected.begin(), expected.end(), 0);
                        std::sort(expected.begin(), expected.end(), [&](int a, int b) { return values[size_t(r)*columns+a] > values[size_t(r)*columns+b]; });
                        expected.resize(size_t(dst.ne[0]));
                        std::vector<int> actual(indices.begin() + size_t(r)*dst.ne[0], indices.begin() + size_t(r+1)*dst.ne[0]);
                        if (top_k) { std::sort(actual.begin(), actual.end()); std::sort(expected.begin(), expected.end()); }
                        CHECK(actual == expected);
                    }
                    if (resources.pool_bytes) {
                        ggml_cuda_source_sort_test_result rejected;
                        CHECK(!ggml_cuda_source_sort_capture_for_test(device, &dst, scratch.get(), resources.pool_bytes - 1, rejected));
                    }
                    fprintf(stderr, "test-moe-cache: source sort op=%s cols=%d rows=%d segmented=%s query=%zu eager=%zu capture=%zu replays=%u bounds OK\n",
                        top_k ? "TOP_K" : "ARGSORT", columns, rows, env, resources.pool_bytes, captured.eager_peak, captured.capture_peak, captured.replays);
                }
            }
        }
    }
    // Query a full chunk and a distinct remainder without allocating its operands.
    ggml_tensor src = {}, dst = {};
    src.type = GGML_TYPE_F32; dst.type = GGML_TYPE_I32;
    src.ne[0] = dst.ne[0] = 16385; src.ne[1] = dst.ne[1] = 1030;
    src.ne[2] = src.ne[3] = dst.ne[2] = dst.ne[3] = 1;
    size_t stride = sizeof(float);
    for (int i = 0; i < 4; ++i) { src.nb[i] = dst.nb[i] = stride; stride *= size_t(src.ne[i]); }
    dst.op = GGML_OP_ARGSORT; dst.op_params[0] = GGML_SORT_ORDER_DESC; dst.src[0] = &src;
    if (ggml_backend_supports_op(gpu.get(), &dst)) {
        ggml_cuda_source_sort_resources resources;
        CHECK(ggml_cuda_argsort_prepare_resources(device, &dst, resources));
        CHECK(resources.pool_bytes > 3 * size_t(16385) * 1023 * sizeof(int));
        fprintf(stderr, "test-moe-cache: source sort full-chunk/remainder query=%zu OK\n", resources.pool_bytes);
    }
    set_env(had_env ? saved_env.c_str() : nullptr);
}

void test_source_softmax_resources(int device) {
    cudaDeviceProp prop;
    CUDA_OK(cudaGetDeviceProperties(&prop, device));
    size_t shared_limit = prop.sharedMemPerBlock;
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    shared_limit = prop.sharedMemPerBlockOptin;
#endif
    CHECK(shared_limit / sizeof(float) + 64 < size_t(INT32_MAX));
    const int wide = int(std::max(size_t(32769), shared_limit / sizeof(float) + 64));
    const auto shape = [](ggml_tensor & tensor, ggml_type type, int cols, int rows, int heads, int batches) {
        tensor = {}; tensor.type = type;
        tensor.ne[0] = cols; tensor.ne[1] = rows; tensor.ne[2] = heads; tensor.ne[3] = batches;
        size_t stride = ggml_type_size(type);
        for (int i = 0; i < 4; ++i) { tensor.nb[i] = stride; stride *= size_t(tensor.ne[i]); }
    };
    struct softmax_case { int cols, rows, heads, batches; ggml_type mask; bool sinks; float scale, bias; };
    const softmax_case cases[] = {
        {512, 1, 1, 1, GGML_TYPE_COUNT, false, 1.0f, 0.0f},
        {513, 2, 3, 2, GGML_TYPE_F32, true, 0.7f, 4.0f},
        {513, 2, 3, 2, GGML_TYPE_F16, true, 0.7f, 4.0f},
        {wide, 1, 1, 1, GGML_TYPE_COUNT, false, 1.0f, 0.0f},
        {wide, 2, 1, 1, GGML_TYPE_COUNT, false, 1.0f, 0.0f},
        {wide, 1, 1, 1, GGML_TYPE_COUNT, false, 0.7f, 0.0f},
        {wide, 2, 3, 2, GGML_TYPE_F16, true, 0.7f, 4.0f},
    };
    for (const auto & c : cases) {
        ggml_tensor src, dst, mask, sinks;
        shape(src, GGML_TYPE_F32, c.cols, c.rows, c.heads, c.batches);
        shape(dst, GGML_TYPE_F32, c.cols, c.rows, c.heads, c.batches);
        dst.op = GGML_OP_SOFT_MAX; dst.src[0] = &src;
        memcpy(dst.op_params, &c.scale, sizeof(float));
        memcpy(reinterpret_cast<char *>(dst.op_params) + sizeof(float), &c.bias, sizeof(float));
        const size_t count = size_t(c.cols) * c.rows * c.heads * c.batches;
        ggml_backend_buffer_ptr input(ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(device), count * sizeof(float)));
        ggml_backend_buffer_ptr output(ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(device), count * sizeof(float)));
        CHECK(input && output);
        src.data = ggml_backend_buffer_get_base(input.get()); src.buffer = input.get();
        dst.data = ggml_backend_buffer_get_base(output.get()); dst.buffer = output.get();
        std::vector<float> values(count);
        for (size_t i = 0; i < count; ++i) { values[i] = float(int((i * 19) % 107) - 53) / 11.0f; }
        CUDA_OK(cudaMemcpy(src.data, values.data(), count * sizeof(float), cudaMemcpyHostToDevice));
        std::vector<float> mask_values;
        ggml_backend_buffer_ptr mask_buffer, sink_buffer;
        if (c.mask != GGML_TYPE_COUNT) {
            shape(mask, c.mask, c.cols, c.rows + 1, 1, 1);
            mask_values.resize(size_t(c.cols) * (c.rows + 1));
            for (size_t i = 0; i < mask_values.size(); ++i) { mask_values[i] = -float(i % 13) / 9.0f; }
            mask_buffer.reset(ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(device), mask_values.size() * ggml_type_size(c.mask)));
            CHECK(mask_buffer); mask.data = ggml_backend_buffer_get_base(mask_buffer.get()); mask.buffer = mask_buffer.get();
            if (c.mask == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> halves(mask_values.size());
                for (size_t i = 0; i < halves.size(); ++i) { halves[i] = ggml_fp32_to_fp16(mask_values[i]); mask_values[i] = ggml_fp16_to_fp32(halves[i]); }
                CUDA_OK(cudaMemcpy(mask.data, halves.data(), halves.size() * sizeof(ggml_fp16_t), cudaMemcpyHostToDevice));
            } else { CUDA_OK(cudaMemcpy(mask.data, mask_values.data(), mask_values.size() * sizeof(float), cudaMemcpyHostToDevice)); }
            dst.src[1] = &mask;
        }
        std::vector<float> sink_values(c.heads, -0.5f);
        if (c.sinks) {
            shape(sinks, GGML_TYPE_F32, c.heads, 1, 1, 1);
            sink_buffer.reset(ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(device), size_t(c.heads) * sizeof(float)));
            CHECK(sink_buffer); sinks.data = ggml_backend_buffer_get_base(sink_buffer.get()); sinks.buffer = sink_buffer.get();
            CUDA_OK(cudaMemcpy(sinks.data, sink_values.data(), size_t(c.heads) * sizeof(float), cudaMemcpyHostToDevice));
            dst.src[2] = &sinks;
        }
        const auto query = [&](ggml_cuda_source_softmax_resources & out) { return ggml_cuda_softmax_prepare_resources(device, &dst, out); };
        ggml_cuda_source_softmax_resources resources, again;
        const bool wide_plain = c.cols == wide && c.mask == GGML_TYPE_COUNT && !c.sinks && c.scale == 1.0f;
        if (wide_plain && prop.cooperativeLaunch && prop.multiProcessorCount >= prop.warpSize * 8) {
            CHECK(!query(resources));
            fprintf(stderr, "test-moe-cache: source softmax cooperative kernel SM/thread limit unsupported cols=%d\n", c.cols);
            continue;
        }
        CHECK(query(resources)); CHECK(query(again));
        CHECK(resources.identity && resources.identity == again.identity && resources.pool_bytes == again.pool_bytes);
        const auto saved_src = src;
        src.type = GGML_TYPE_F16; CHECK(!query(again)); src = saved_src;
        src.ne[1] = INT64_MAX; CHECK(!query(again)); src = saved_src;
        src.ne[0] = 0; CHECK(!query(again)); src = saved_src;
        src.nb[0] += sizeof(float); CHECK(!query(again)); src = saved_src;
        CHECK(!ggml_cuda_softmax_prepare_resources(-1, &dst, again));
        CHECK(!ggml_cuda_softmax_prepare_resources(prop.multiProcessorCount + 9999, &dst, again));
        const auto saved_dst = dst;
        dst.op = GGML_OP_SOFT_MAX_BACK; CHECK(!query(again)); dst = saved_dst;
        dst.ne[0] -= 1; CHECK(!query(again)); dst = saved_dst;
        const float invalid = std::numeric_limits<float>::quiet_NaN();
        memcpy(dst.op_params, &invalid, sizeof(float)); CHECK(!query(again)); dst = saved_dst;
        const float different = c.scale + 0.25f;
        memcpy(dst.op_params, &different, sizeof(float)); CHECK(query(again)); CHECK(again.identity != resources.identity); dst = saved_dst;
        if (dst.src[1]) {
            const auto saved = mask; mask.data = nullptr; CHECK(!query(again)); mask = saved;
            mask.ne[0] -= 1; CHECK(!query(again)); mask = saved;
        } else {
            const float positive = 1.0f;
            memcpy(reinterpret_cast<char *>(dst.op_params) + sizeof(float), &positive, sizeof(float)); CHECK(!query(again)); dst = saved_dst;
        }
        if (dst.src[2]) {
            const auto saved = sinks; sinks.type = GGML_TYPE_F16; CHECK(!query(again)); sinks = saved;
        }
        if (wide_plain && prop.cooperativeLaunch) {
            CHECK(prop.multiProcessorCount < prop.warpSize * 8);
            const size_t one = (size_t(prop.multiProcessorCount) * sizeof(float) * sizeof(float) + 255) & ~size_t(255);
            CHECK(resources.pool_bytes == 2 * one);
        } else { CHECK(resources.pool_bytes == 0); }
        ggml_backend_buffer_ptr scratch(ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(device), std::max(size_t(256), resources.pool_bytes)));
        CHECK(scratch);
        ggml_cuda_source_softmax_test_result measured, rejected;
        const auto data = dst.data; const auto buffer = dst.buffer;
        dst.data = src.data; dst.buffer = src.buffer;
        CHECK(!ggml_cuda_source_softmax_capture_for_test(device, &dst, scratch.get(), resources.pool_bytes, rejected));
        dst.data = data; dst.buffer = buffer;
        dst.data = static_cast<char *>(data) + ggml_backend_buffer_get_size(buffer);
        CHECK(!ggml_cuda_source_softmax_capture_for_test(device, &dst, scratch.get(), resources.pool_bytes, rejected)); dst.data = data;
        if (resources.pool_bytes) {
            CHECK(!ggml_cuda_source_softmax_capture_for_test(device, &dst, scratch.get(), resources.pool_bytes - 1, rejected));
        }
        CHECK(ggml_cuda_source_softmax_capture_for_test(device, &dst, scratch.get(), resources.pool_bytes, measured));
        CHECK(measured.eager_peak == resources.pool_bytes && measured.capture_peak <= resources.pool_bytes);
        std::vector<float> actual(count);
        CUDA_OK(cudaMemcpy(actual.data(), dst.data, count * sizeof(float), cudaMemcpyDeviceToHost));
        double error = 0, energy = 0, max_relative = 0;
        const int head_power = 1 << int(std::floor(std::log2(double(c.heads))));
        for (int b = 0; b < c.batches; ++b) {
            for (int h = 0; h < c.heads; ++h) {
                const double slope = c.bias <= 0 ? 1.0 : std::pow(2.0, h < head_power ? -double(c.bias)*(h+1)/head_power : -double(c.bias)*(2*(h-head_power)+1)/(2*head_power));
                for (int r = 0; r < c.rows; ++r) {
                    const size_t start = ((size_t(b)*c.heads+h)*c.rows+r)*c.cols;
                    std::vector<double> row(c.cols);
                    double maximum = c.sinks ? sink_values[h] : -std::numeric_limits<double>::infinity();
                    for (int col = 0; col < c.cols; ++col) {
                        row[col] = double(values[start+col])*c.scale + (mask_values.empty() ? 0 : slope*mask_values[size_t(r)*c.cols+col]);
                        maximum = std::max(maximum, row[col]);
                    }
                    double denominator = c.sinks ? std::exp(double(sink_values[h])-maximum) : 0;
                    for (double value : row) { denominator += std::exp(value-maximum); }
                    for (int col = 0; col < c.cols; ++col) {
                        const double expected = std::exp(row[col]-maximum)/denominator;
                        CHECK(std::isfinite(actual[start+col]));
                        const double diff = actual[start+col]-expected;
                        error += diff*diff; energy += expected*expected;
                        max_relative = std::max(max_relative, std::abs(diff)/expected);
                    }
                }
            }
        }
        CHECK(error / energy < 1e-10 && max_relative < 2e-5);
        fprintf(stderr, "test-moe-cache: source softmax cols=%d rows=%d heads=%d batches=%d mask=%d sinks=%d scale=%.2f bias=%.2f query=%zu eager=%zu capture=%zu replays=%u nmse=%.3g maxrel=%.3g bounds/oracle OK\n",
            c.cols, c.rows, c.heads, c.batches, int(c.mask), c.sinks, c.scale, c.bias, resources.pool_bytes, measured.eager_peak, measured.capture_peak, measured.replays, error/energy, max_relative);
    }
}

void staged_source_view_checks(int device, const ggml_tensor * node, const ggml_staged_input_api * api, void * input) {
    ggml_cuda_source_staged_input_view view, again;
    CHECK(ggml_cuda_staged_input_prepare_source_view(device, node, view));
    CHECK(view.host == api->data(input) && view.device == device && view.bytes == ggml_nbytes(node));
    CHECK(view.host && view.device_alias && view.host_flag && view.device_flag && view.identity);
    const auto * flag = static_cast<const std::atomic<uint32_t> *>(view.host_flag);
    CHECK(flag->load(std::memory_order_acquire) == 0);
    CHECK(ggml_cuda_staged_input_prepare_source_view(device, node, again));
    CHECK(view.identity == again.identity && view.host == again.host && view.device_alias == again.device_alias &&
        view.host_flag == again.host_flag && view.device_flag == again.device_flag && view.bytes == again.bytes);
    CHECK(!ggml_cuda_staged_input_prepare_source_view(-1, node, again));
    CHECK(!again.host && !again.identity);
    CHECK(!ggml_cuda_staged_input_prepare_source_view(device, nullptr, again));
    ggml_tensor invalid = *node;
    invalid.type = GGML_TYPE_F16; CHECK(!ggml_cuda_staged_input_prepare_source_view(device, &invalid, again));
    invalid = *node; invalid.ne[0] = INT64_MAX; CHECK(!ggml_cuda_staged_input_prepare_source_view(device, &invalid, again));
    invalid = *node; invalid.ne[0] -= 1;
    for (int i = 1; i < 4; ++i) { invalid.nb[i] = size_t(invalid.ne[0]) * sizeof(float); }
    CHECK(!ggml_cuda_staged_input_prepare_source_view(device, &invalid, again));
    invalid = *node; invalid.ne[1] = 2; CHECK(!ggml_cuda_staged_input_prepare_source_view(device, &invalid, again));
    invalid = *node; invalid.nb[1] += sizeof(float); CHECK(!ggml_cuda_staged_input_prepare_source_view(device, &invalid, again));
    invalid = *node; invalid.op = GGML_OP_NONE; CHECK(!ggml_cuda_staged_input_prepare_source_view(device, &invalid, again));
    invalid = *node; memset(invalid.op_params, 0, sizeof(invalid.op_params)); CHECK(!ggml_cuda_staged_input_prepare_source_view(device, &invalid, again));
    invalid = *node; invalid.flags ^= GGML_TENSOR_FLAG_OUTPUT;
    CHECK(ggml_cuda_staged_input_prepare_source_view(device, &invalid, again)); CHECK(view.identity != again.identity);
    std::vector<unsigned char> saved(view.bytes), actual(view.bytes);
    memcpy(saved.data(), view.host, view.bytes);
    auto * values = static_cast<float *>(api->data(input));
    for (size_t i = 0; i < view.bytes / sizeof(float); ++i) { values[i] = float(int(i % 31) - 15) / 7.0f; }
    ggml_backend_buffer_ptr copy(ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(device), view.bytes));
    CHECK(copy);
    CUDA_OK(cudaMemcpy(ggml_backend_buffer_get_base(copy.get()), view.device_alias, view.bytes, cudaMemcpyDeviceToDevice));
    CUDA_OK(cudaMemcpy(actual.data(), ggml_backend_buffer_get_base(copy.get()), view.bytes, cudaMemcpyDeviceToHost));
    CHECK(memcmp(actual.data(), view.host, view.bytes) == 0);
    api->publish(input);
    CHECK(ggml_cuda_staged_input_prepare_source_view(device, node, again)); CHECK(view.identity == again.identity);
    uint32_t observed = 0;
    CUDA_OK(cudaMemcpy(&observed, view.device_flag, sizeof(observed), cudaMemcpyDeviceToHost));
    CHECK(observed == 1 && flag->load(std::memory_order_acquire) == 1);
    const uint32_t waiting = 0;
    CUDA_OK(cudaMemcpy(view.device_flag, &waiting, sizeof(waiting), cudaMemcpyHostToDevice));
    CHECK(flag->load(std::memory_order_acquire) == 0);
    CHECK(ggml_cuda_staged_input_prepare_source_view(device, node, again)); CHECK(view.identity == again.identity);
    memcpy(api->data(input), saved.data(), view.bytes);
    fprintf(stderr, "test-moe-cache: staged source borrowed bytes=%zu device=%d pending0/published1/requery/mapped-payload/flag/malformed OK\n", view.bytes, view.device);
}

void test_source_overlap_schedule(fidelity_graph_owner & owner, layer_fixture & fixture) {
    std::vector<ggml_moe_source_expert> experts(owner.descriptors.size());
    std::vector<const ggml_moe_source_expert *> pointers;
    for (size_t i = 0; i < experts.size(); ++i) {
        const auto & descriptor = owner.descriptors[i];
        std::vector<ggml_backend_moe_cpu_prepared_region_v1_t> handles(descriptor.n_cpu_queries + descriptor.n_cpu_batch_queries, 1);
        CHECK(ggml_moe_source_expert_prepare(descriptor, handles.data(), experts[i]));
        pointers.push_back(&experts[i]);
    }
    const auto buft = ggml_backend_get_default_buffer_type(owner.gpu.get());
    ggml_moe_source_program original;
    CHECK(original.prepare(&owner.graph, pointers, buft) && original.overlap_operation_count() == 0);
    CHECK(original.find(fixture.input) && !original.find(nullptr));
    {
        std::vector<int> flags;
        for (int i = 0; i < owner.graph.n_nodes; ++i) {
            flags.push_back(owner.graph.nodes[i]->flags);
            owner.graph.nodes[i]->flags &= ~GGML_TENSOR_FLAG_OUTPUT;
        }
        auto * output = owner.graph.nodes[owner.graph.n_nodes - 1];
        const size_t slot = ggml_hash_find(&owner.graph.visited_hash_set, output);
        CHECK(slot != GGML_HASHSET_FULL && ggml_bitset_get(owner.graph.visited_hash_set.used, slot));
        const int32_t uses = owner.graph.use_counts[slot];
        CHECK(uses < INT32_MAX);
        ++owner.graph.use_counts[slot];
        ggml_moe_source_program exported;
        CHECK(exported.prepare(&owner.graph, pointers, buft) && exported.allocate());
        const auto * root = output;
        while (root->view_src) { root = root->view_src; }
        CHECK(std::find(exported.public_outputs().begin(), exported.public_outputs().end(), root) != exported.public_outputs().end());
        CHECK(exported.find(output)->flags & GGML_TENSOR_FLAG_OUTPUT);
        CHECK(!(output->flags & GGML_TENSOR_FLAG_OUTPUT));
        const auto * cut = exported.find(output);
        const auto * retained = exported.find(fixture.input);
        CHECK(!exported.closed_cut(&cut, 1, &retained, 1));
        auto malformed = owner.graph;
        malformed.use_counts = nullptr;
        ggml_moe_source_program rejected;
        CHECK(!rejected.prepare(&malformed, pointers, buft));
        const auto * internal = owner.graph.nodes[experts.front().region.first_node];
        const size_t inner_slot = ggml_hash_find(&owner.graph.visited_hash_set, internal);
        CHECK(inner_slot != GGML_HASHSET_FULL && ggml_bitset_get(owner.graph.visited_hash_set.used, inner_slot));
        const int32_t inner_uses = owner.graph.use_counts[inner_slot];
        CHECK(inner_uses < INT32_MAX);
        ++owner.graph.use_counts[inner_slot];
        CHECK(!rejected.prepare(&owner.graph, pointers, buft));
        owner.graph.use_counts[inner_slot] = inner_uses;
        owner.graph.use_counts[slot] = uses;
        for (int i = 0; i < owner.graph.n_nodes; ++i) { owner.graph.nodes[i]->flags = flags[i]; }
        CHECK(original.matches(&owner.graph));
        fprintf(stderr, "test-moe-cache: split exports parent-use-count/private-output/caller-unchanged/closed-cut/omitted-output/missing-proof OK\n");
    }
    {
        ggml_tensor immutable = *fixture.input;
        immutable.flags &= ~GGML_TENSOR_FLAG_INPUT;
        const auto * leaf = &immutable;
        const auto saved = immutable;
        auto * chain = fixture.overlap_value[0]->src[0];
        const auto * chain_input = chain->src[0];
        chain->src[0] = &immutable;
        ggml_backend_buffer_ptr replacement(ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(leaf)));
        CHECK(replacement);
        struct binding_state {
            const ggml_tensor * leaf;
            ggml_backend_buffer_t buffer;
            int invalid = 0;
            uint32_t matches = 0;
        } binding{leaf, replacement.get()};
        ggml_moe_source_schedule_options bound;
        bound.leaf_context = &binding;
        bound.bind_leaf = [](void * context, const ggml_tensor * tensor, ggml_backend_buffer_t * buffer, void ** data) {
            auto & state = *static_cast<binding_state *>(context);
            CHECK(tensor->op == GGML_OP_NONE && !tensor->view_src && !(tensor->flags & GGML_TENSOR_FLAG_INPUT));
            if (tensor != state.leaf) { return true; }
            ++state.matches;
            if (state.invalid == 1) { return false; }
            *buffer = state.buffer;
            *data = ggml_backend_buffer_get_base(state.buffer);
            if (state.invalid == 2) { *data = nullptr; }
            if (state.invalid == 3) { *data = static_cast<char *>(*data) + ggml_backend_buffer_get_size(state.buffer); }
            if (state.invalid == 4) { *data = reinterpret_cast<void *>(uintptr_t(*data) - 1); }
            return true;
        };
        ggml_moe_source_program borrowed;
        CHECK(borrowed.prepare(&owner.graph, pointers, buft, bound) && borrowed.allocate());
        CHECK(binding.matches == 1 && borrowed.matches(&owner.graph));
        const auto * clone = borrowed.find(leaf);
        CHECK(clone && clone != leaf && clone->buffer == replacement.get() &&
            clone->data == ggml_backend_buffer_get_base(replacement.get()));
        CHECK(ggml_moe_source_tensor_matches(*leaf, saved));
        for (int invalid : {1, 2, 3, 4}) {
            binding.invalid = invalid;
            ggml_moe_source_program rejected;
            CHECK(!rejected.prepare(&owner.graph, pointers, buft, bound));
            CHECK(ggml_moe_source_tensor_matches(*leaf, saved));
        }
        chain->src[0] = const_cast<ggml_tensor *>(chain_input);
        CHECK(original.matches(&owner.graph));
        fprintf(stderr, "test-moe-cache: live immutable leaf clone/bounds/null/rejection/dynamic-input/caller-metadata OK\n");
    }
    ggml_tensor unrelated = *fixture.input;
    CHECK(!original.find(&unrelated));
    const auto * first_node = owner.graph.nodes[0];
    CHECK(original.original_node(0) == first_node);
    CHECK(!original.original_node(owner.graph.n_nodes) && !original.original_node(SIZE_MAX));
    owner.graph.nodes[0] = owner.graph.nodes[1];
    CHECK(original.original_node(0) == first_node);
    CHECK(!original.matches(&owner.graph));
    ggml_moe_source_program duplicate;
    CHECK(!duplicate.prepare(&owner.graph, pointers, buft));
    owner.graph.nodes[0] = const_cast<ggml_tensor *>(first_node);
    CHECK(original.matches(&owner.graph));
    CHECK(original.public_bindings().size() == original.public_outputs().size());
    for (size_t i = 0; i < original.public_bindings().size(); ++i) {
        const auto & binding = original.public_bindings()[i];
        auto * output = const_cast<ggml_tensor *>(original.public_outputs()[i]);
        CHECK(binding.original == output && binding.buffer == output->buffer &&
            binding.data == output->data && binding.bytes == ggml_nbytes(output));
        const auto saved = *output;
        output->data = reinterpret_cast<void *>(uintptr_t(output->data) ^ 1);
        CHECK(!original.matches(&owner.graph));
        CHECK(binding.data == saved.data && binding.buffer == saved.buffer && binding.bytes == ggml_nbytes(&saved));
        *output = saved;
        CHECK(original.matches(&owner.graph));
    }
    const auto input_saved = *fixture.input;
    const std::function<void(ggml_tensor &)> mutations[] = {
        [](ggml_tensor & t) { t.type = GGML_TYPE_I32; },
        [](ggml_tensor & t) { t.buffer = reinterpret_cast<ggml_backend_buffer_t>(uintptr_t(t.buffer) ^ 1); },
        [](ggml_tensor & t) { ++t.ne[0]; },
        [](ggml_tensor & t) { ++t.nb[0]; },
        [](ggml_tensor & t) { t.op = GGML_OP_ADD; },
        [](ggml_tensor & t) { t.op_params[0] ^= 1; },
        [](ggml_tensor & t) { t.flags ^= GGML_TENSOR_FLAG_OUTPUT; },
        [&](ggml_tensor & t) { t.src[0] = owner.graph.nodes[0]; },
        [&](ggml_tensor & t) { t.view_src = owner.graph.nodes[0]; },
        [](ggml_tensor & t) { t.view_offs ^= 1; },
        [](ggml_tensor & t) { t.data = reinterpret_cast<void *>(uintptr_t(t.data) ^ 1); },
    };
    for (const auto & mutate : mutations) {
        mutate(*fixture.input);
        CHECK(!original.matches(&owner.graph));
        *fixture.input = input_saved;
        CHECK(original.matches(&owner.graph));
    }
    fixture.input->name[0] ^= 1;
    fixture.input->extra = fixture.input;
    fixture.input->padding[0] ^= 1;
    CHECK(original.matches(&owner.graph));
    *fixture.input = input_saved;
    CHECK(experts.size() > 1);
    const auto * first_output = experts.front().region.output;
    const auto * last_output = experts.back().region.output;
    for (const bool overlap : {false, true}) {
        ggml_moe_source_schedule_options dependencies;
        dependencies.overlap_independent_ordinary = overlap;
        dependencies.allocation_dependencies = {{first_output, last_output}, {first_output, last_output}};
        ggml_moe_source_program retained;
        CHECK(retained.prepare(&owner.graph, pointers, buft, dependencies) && retained.allocate());
        const auto * first = retained.find(first_output);
        const auto * last = retained.find(last_output);
        const auto begin = uintptr_t(first->data), end = uintptr_t(last->data);
        CHECK(begin + ggml_nbytes(first) <= end || end + ggml_nbytes(last) <= begin);
        CHECK(retained.matches(&owner.graph) && retained.operation_count() == original.operation_count());
        for (const auto invalid : {ggml_moe_source_allocation_dependency{last_output, first_output},
                ggml_moe_source_allocation_dependency{nullptr, last_output},
                ggml_moe_source_allocation_dependency{first_output, fixture.input},
                ggml_moe_source_allocation_dependency{owner.graph.nodes[experts.front().region.first_node], last_output}}) {
            dependencies.allocation_dependencies = {invalid};
            ggml_moe_source_program rejected;
            CHECK(!rejected.prepare(&owner.graph, pointers, buft, dependencies));
        }
    }
    fprintf(stderr, "test-moe-cache: source keep-until clone/disjoint/overlap/duplicate/invalid lifetime proof OK\n");
    ggml_moe_source_schedule_options option;
    option.overlap_independent_ordinary = true;
    const auto prepare = [&](size_t first_count, bool effect_barrier = false) {
        ggml_moe_source_program moved, unchanged;
        CHECK(unchanged.prepare(&owner.graph, pointers, buft));
        CHECK(moved.prepare(&owner.graph, pointers, buft, option));
        const auto * chain = fixture.overlap_value[0]->src[0];
        const auto found = std::find(owner.graph.nodes, owner.graph.nodes + owner.graph.n_nodes, chain);
        CHECK(found != owner.graph.nodes + owner.graph.n_nodes);
        const size_t chain_index = size_t(found - owner.graph.nodes);
        size_t compute = 0, chain_compute = 0;
        for (const auto & op : moved.layers()[0].overlap) {
            std::vector<const ggml_tensor *> pending = {op.original};
            std::unordered_set<const ggml_tensor *> visited;
            while (!pending.empty()) {
                const auto * value = pending.back();
                pending.pop_back();
                if (!visited.insert(value).second) { continue; }
                CHECK(value != experts[0].region.output);
                for (const auto * input : value->src) { if (input) { pending.push_back(input); } }
                if (value->view_src) { pending.push_back(value->view_src); }
            }
            if (op.effect == ggml_moe_source_effect::metadata) { continue; }
            CHECK(ggml_op_is_pure(op.tensor->op) && !op.tensor->view_src);
            ++compute;
            chain_compute += op.original == chain || op.original == fixture.overlap_value[0];
            if (effect_barrier) { CHECK(op.original_index < chain_index); }
        }
        CHECK(chain_compute == first_count && moved.operation_count() == unchanged.operation_count());
        if (first_count == 2) {
            fprintf(stderr, "test-moe-cache: shared purity overlap compute=%zu required_chain=%zu nodes=", compute, chain_compute);
            for (const auto & op : moved.layers()[0].overlap) {
                if (op.effect != ggml_moe_source_effect::metadata) { fprintf(stderr, "%zu:%s,", op.original_index, ggml_op_name(op.tensor->op)); }
            }
            fprintf(stderr, "\n");
        }
        CHECK(moved.matches(&owner.graph));
        return moved.storage_bytes();
    };
    const auto bytes = prepare(2);
    auto * scale = fixture.overlap_value[0]->src[0];
    const auto saved = *scale;
    scale->src[0] = const_cast<ggml_tensor *>(experts[0].region.output);
    const size_t old_slot = ggml_hash_find(&owner.graph.visited_hash_set, saved.src[0]);
    const size_t new_slot = ggml_hash_find(&owner.graph.visited_hash_set, scale->src[0]);
    CHECK(old_slot != GGML_HASHSET_FULL && new_slot != GGML_HASHSET_FULL);
    --owner.graph.use_counts[old_slot];
    ++owner.graph.use_counts[new_slot];
    prepare(0);
    ++owner.graph.use_counts[old_slot];
    --owner.graph.use_counts[new_slot];
    CHECK(!original.matches(&owner.graph));
    *scale = saved;
    scale->view_src = saved.src[0];
    scale->view_offs = 0;
    prepare(0, true);
    *scale = saved;
    for (const auto op : {GGML_OP_CPY, GGML_OP_CUSTOM}) {
        scale->op = op;
        prepare(0, true);
        *scale = saved;
    }
    scale->flags |= GGML_TENSOR_FLAG_OUTPUT;
    prepare(0, true);
    *scale = saved;
    scale->op = GGML_OP_VIEW;
    scale->view_src = saved.src[0];
    scale->view_offs = 0;
    prepare(1);
    const auto input_flags = scale->view_src->flags;
    scale->view_src->flags |= GGML_TENSOR_FLAG_OUTPUT;
    prepare(0, true);
    scale->view_src->flags = input_flags;
    *scale = saved;
    CHECK(original.matches(&owner.graph) && prepare(2) == bytes);
    std::vector<ggml_moe_source_operation> adjacent(2);
    ggml_tensor norm = {}, multiply = {};
    norm.op = GGML_OP_RMS_NORM; multiply.op = GGML_OP_MUL; multiply.src[0] = &norm;
    adjacent[0].tensor = &norm; adjacent[0].original_index = 1;
    adjacent[1].tensor = &multiply; adjacent[1].original_index = 3;
    CHECK(ggml_moe_source_operation_group_size(adjacent, 0) == 1);
    adjacent[1].original_index = 2;
    CHECK(ggml_moe_source_operation_group_size(adjacent, 0) == 2);
    fprintf(stderr, "test-moe-cache: source overlap dependency/in-place/effect/public/fusion/storage proof OK\n");
}

struct source_state_probe;
static source_state_probe * active_source_state_probe = nullptr;

struct source_state_probe {
    ggml_backend_reg_t reg;
    ggml_backend_reg_t cpu_reg;
    void * (*get_proc)(ggml_backend_reg_t, const char *);
    void * (*get_cpu_proc)(ggml_backend_reg_t, const char *);
    const ggml_backend_moe_source_core_api_v1 * original;
    const ggml_backend_moe_cpu_region_service_api_v1 * original_cpu;
    ggml_backend_moe_source_core_api_v1 api;
    ggml_backend_moe_cpu_region_service_api_v1 cpu_api;
    void * sessions[16] = {};
    ggml_backend_moe_hybrid_state_v1 samples[16] = {};
    std::atomic<uint64_t> measured_cpu_bytes{0};
    uint32_t n_sessions = 0;
    uint32_t queries = 0;
    int fail_index = -1;
    bool fail_cpu = false;
    std::atomic<bool> inject{false};

    explicit source_state_probe(ggml_backend_t backend) {
        CHECK(!active_source_state_probe);
        reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
        get_proc = reg->iface.get_proc_address;
        const auto getter = reinterpret_cast<ggml_backend_moe_source_core_v1_t>(get_proc(reg, GGML_BACKEND_MOE_SOURCE_CORE_V1_PROC_NAME));
        CHECK(getter);
        original = getter(); api = *original;
        cpu_reg = ggml_backend_cpu_reg(); get_cpu_proc = cpu_reg->iface.get_proc_address;
        const auto cpu_getter = reinterpret_cast<ggml_backend_moe_cpu_region_service_v1_t>(
            get_cpu_proc(cpu_reg, GGML_BACKEND_MOE_CPU_FIDELITY_SERVICE_V1_PROC_NAME));
        CHECK(cpu_getter && cpu_reg != reg);
        original_cpu = cpu_getter(); cpu_api = *original_cpu;
        cpu_api.create = +[](const ggml_backend_moe_cpu_service_config_v1 * config, ggml_backend_moe_cpu_service_v1_t * service) {
            auto & self = *active_source_state_probe;
            const auto status = self.original_cpu->create(config, service);
            if (!status) {
                ggml_backend_moe_cpu_service_state_v1 state = {}; state.struct_size = sizeof(state);
                CHECK(!self.original_cpu->state(*service, &state));
                self.measured_cpu_bytes.store(state.prepared_payload_bytes);
            }
            return status;
        };
        cpu_api.state = +[](ggml_backend_moe_cpu_service_v1_t service, ggml_backend_moe_cpu_service_state_v1 * state) -> int32_t {
            auto & self = *active_source_state_probe;
            if (self.inject) {
                *state = {}; state->struct_size = sizeof(*state);
                state->active_jobs = 7; state->prepared_payload_bytes = 31;
                return self.fail_cpu ? GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY : GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
            }
            const auto status = self.original_cpu->state(service, state);
            if (!status) { self.measured_cpu_bytes.store(state->prepared_payload_bytes); }
            return status;
        };
        api.state = +[](void * session, ggml_backend_moe_hybrid_state_v1 * state) {
            auto & self = *active_source_state_probe;
            uint32_t index = 0;
            for (; index < self.n_sessions && self.sessions[index] != session; ++index) {}
            if (index == self.n_sessions) {
                CHECK(index < 16);
                self.sessions[self.n_sessions++] = session;
            }
            if (!self.inject) { return self.original->state(session, state); }
            ++self.queries;
            *state = self.samples[index]; state->struct_size = sizeof(*state);
            return int(index) != self.fail_index;
        };
        active_source_state_probe = this;
        reg->iface.get_proc_address = +[](ggml_backend_reg_t module, const char * name) -> void * {
            auto & self = *active_source_state_probe;
            if (!strcmp(name, GGML_BACKEND_MOE_SOURCE_CORE_V1_PROC_NAME)) {
                return reinterpret_cast<void *>(+[]() -> const ggml_backend_moe_source_core_api_v1 * {
                    return &active_source_state_probe->api;
                });
            }
            return self.get_proc(module, name);
        };
        cpu_reg->iface.get_proc_address = +[](ggml_backend_reg_t module, const char * name) -> void * {
            auto & self = *active_source_state_probe;
            if (!strcmp(name, GGML_BACKEND_MOE_CPU_FIDELITY_SERVICE_V1_PROC_NAME)) {
                return reinterpret_cast<void *>(+[]() -> const ggml_backend_moe_cpu_region_service_api_v1 * {
                    return &active_source_state_probe->cpu_api;
                });
            }
            return self.get_cpu_proc(module, name);
        };
    }

    ~source_state_probe() {
        cpu_reg->iface.get_proc_address = get_cpu_proc;
        reg->iface.get_proc_address = get_proc;
        active_source_state_probe = nullptr;
    }

    void check_real(ggml_backend_sched_t sched, const char * label) {
        if (getenv("GGML_TEST_MOE_BACKEND_CPU_AGGREGATE_ONLY")) { return; }
        ggml_backend_moe_hybrid_state_v1 state = {}; state.struct_size = sizeof(state);
        CHECK(ggml_backend_sched_moe_hybrid_state_v1(sched, &state));
        const auto measured = measured_cpu_bytes.load();
        fprintf(stderr, "test-moe-cache: shared CPU scope=%s public=%llu service=%llu\n", label,
            (unsigned long long) state.prepared_cpu_bytes, (unsigned long long) measured);
        CHECK(measured > 0 && state.prepared_cpu_bytes == measured && state.cpu_active_jobs == 0);
    }

    void check_single_failure(ggml_backend_sched_t sched) {
        if (getenv("GGML_TEST_MOE_BACKEND_CPU_AGGREGATE_ONLY")) { return; }
        inject = true; fail_cpu = true;
        ggml_backend_moe_hybrid_state_v1 state;
        memset(&state, 0x5a, sizeof(state)); state.struct_size = sizeof(state);
        const auto before = state;
        CHECK(!ggml_backend_sched_moe_hybrid_state_v1(sched, &state));
        CHECK(!memcmp(&state, &before, sizeof(state)));
        fail_cpu = false; inject = false;
        fprintf(stderr, "test-moe-cache: single source shared CPU query failure preserves caller OK\n");
    }

    void check(ggml_backend_sched_t sched) {
        CHECK(n_sessions >= 2 && n_sessions <= 16);
        inject = true;
        const auto reset_samples = [&]() {
            for (uint32_t i = 0; i < n_sessions; ++i) {
                auto & sample = samples[i]; sample = {};
                sample.resident_routes = 3 * (i + 1);
                sample.device_bytes = 11 * (i + 1);
                sample.pinned_bytes = 7 * (i + 1);
                sample.errors = i + 1;
                sample.cpu_active_jobs = i + 1;
                sample.prepared_cpu_bytes = 17 * (i + 1);
                sample.work_peak = 13 * (i + 1);
                sample.last_epoch = 100 + i;
                sample.last_submit_us = 1000 + i;
                sample.ticket_state = GGML_BACKEND_MOE_SOURCE_CORE_TICKET_V1_REJECTED_BEFORE_EFFECTS;
            }
            samples[n_sessions - 1].dispatch_active = 1;
            samples[n_sessions - 1].quiescing = 1;
            fail_index = -1; fail_cpu = false;
        };
        const auto unchanged_on_failure = [&]() {
            ggml_backend_moe_hybrid_state_v1 state;
            memset(&state, 0x5a, sizeof(state)); state.struct_size = sizeof(state);
            const auto before = state;
            CHECK(!ggml_backend_sched_moe_hybrid_state_v1(sched, &state));
            CHECK(!memcmp(&state, &before, sizeof(state)));
        };
        reset_samples();
        ggml_backend_moe_hybrid_state_v1 state = {}; state.struct_size = sizeof(state);
        CHECK(ggml_backend_sched_moe_hybrid_state_v1(sched, &state));
        const uint64_t sum = uint64_t(n_sessions) * (n_sessions + 1) / 2;
        CHECK(state.resident_routes == 3 * sum && state.device_bytes == 11 * sum && state.pinned_bytes == 7 * sum && state.errors == sum);
        CHECK(state.prepared_cpu_bytes == 31 && state.cpu_active_jobs == 7 && state.work_peak == 13 * n_sessions);
        CHECK(state.dispatch_active == 1 && state.quiescing == 1);
        CHECK(state.ticket_state == 0 && state.last_epoch == 0 && state.last_submit_us == 0);
        fail_index = n_sessions - 1;
        unchanged_on_failure();
        reset_samples(); fail_cpu = true;
        unchanged_on_failure();
        uint64_t ggml_backend_moe_hybrid_state_v1::* overflow_fields[] = {
            &ggml_backend_moe_hybrid_state_v1::resident_routes,
            &ggml_backend_moe_hybrid_state_v1::device_bytes,
            &ggml_backend_moe_hybrid_state_v1::pinned_bytes,
            &ggml_backend_moe_hybrid_state_v1::errors,
        };
        for (auto field : overflow_fields) {
            reset_samples(); samples[0].*field = UINT64_MAX; samples[n_sessions - 1].*field = 1;
            unchanged_on_failure();
        }
        reset_samples();
        state = {}; state.struct_size = sizeof(state) - 1;
        const auto before = state;
        const auto old_queries = queries;
        CHECK(!ggml_backend_sched_moe_hybrid_state_v1(sched, &state) && queries == old_queries);
        CHECK(!memcmp(&state, &before, sizeof(state)));
        state.struct_size = sizeof(state);
        CHECK(ggml_backend_sched_moe_hybrid_state_v1(sched, &state) && state.resident_routes == 3 * sum && state.prepared_cpu_bytes == 31);
        inject = false;
        fprintf(stderr, "test-moe-cache: source public-state sessions=%u shared CPU once/sums/max/flags/ticket omission/later program+CPU query failure/four overflow classes/invalid size/recovery OK\n", n_sessions);
    }
};

static void test_source_backend_sessions(int device) {
    CHECK(ggml_moe_fidelity_selection().source_pool && ggml_moe_fidelity_selection().reference);
    const auto * auxiliary_mode = getenv("GGML_TEST_MOE_BACKEND_AUXILIARIES");
    const bool output_auxiliaries = auxiliary_mode != nullptr;
    const bool pageable_auxiliaries = auxiliary_mode && !strcmp(auxiliary_mode, "pageable");
    CHECK(!auxiliary_mode || pageable_auxiliaries || !strcmp(auxiliary_mode, "pinned"));
    auto owner = std::make_shared<fidelity_graph_owner>();
    owner->gpu.reset(ggml_backend_cuda_init(device));
    ggml_backend_ptr second(ggml_backend_cuda_init(device));
    owner->cpu.reset(ggml_backend_cpu_init());
    CHECK(owner->gpu && second && owner->cpu && owner->gpu.get() != second.get());
    std::unique_ptr<source_state_probe> state_probe;
    if (getenv("GGML_TEST_MOE_BACKEND_STATE")) { state_probe = std::make_unique<source_state_probe>(owner->gpu.get()); }
    owner->fixture = std::make_unique<layer_fixture>(0, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_TYPE_Q5_K,
        pageable_auxiliaries, std::array<int, 2>{device, device}, all_cached_layers, n_slots, false, LLM_ARCH_QWEN3MOE, LLM_FFN_SILU,
        n_used, GGML_TYPE_Q5_K, 256, 256, 16, false, output_auxiliaries);
    auto & fixture = *owner->fixture;
    owner->model.reset(llama_model_create(LLM_ARCH_QWEN3MOE, llama_model_default_params()));
    for (const auto & source : fixture.tensors) {
        CHECK(owner->model->record_moe_readable_source(source.tensor, source.tensor->data, ggml_nbytes(source.tensor)));
    }
    ggml_backend_moe_source_owner_v1 source_owner = {};
    CHECK(owner->model->moe_source_owner_v1(&source_owner));
    const auto manifest = fixture.manifest();
    for (auto * backend : {owner->gpu.get(), second.get()}) {
        CHECK(ggml_backend_cuda_moe_candidate_replace_v2(backend, &manifest) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    }
    ggml_backend_t backends[]{owner->gpu.get(), second.get(), owner->cpu.get()};
    auto secondary_buft = *ggml_backend_get_default_buffer_type(second.get());
    secondary_buft.iface.alloc_buffer = +[](ggml_backend_buffer_type_t buft, size_t bytes) {
        auto * actual = ggml_backend_dev_buffer_type(buft->device);
        return actual->iface.alloc_buffer(actual, bytes);
    };
    ggml_backend_buffer_type_t bufts[]{ggml_backend_get_default_buffer_type(owner->gpu.get()), &secondary_buft,
        ggml_backend_get_default_buffer_type(owner->cpu.get())};
    const char * collapse = getenv("GGML_TEST_MOE_BACKEND_COLLAPSE");
    const bool collapsed = collapse && !strcmp(collapse, "1");
    if (collapsed) { bufts[1] = bufts[0]; }
    owner->sched.reset(ggml_backend_sched_new(backends, bufts, 3, 256, false, true));
    CHECK(ggml_backend_sched_set_resizable(owner->sched.get(), nullptr));
    auto certificate = layer_certificate();
    const char * domain = getenv("GGML_TEST_MOE_BACKEND_DOMAIN");
    const bool auxiliary = domain != nullptr;
    bool sequential = false;
    if (auxiliary) {
        CHECK(!strcmp(domain, "draft") || !strcmp(domain, "mtp"));
        certificate.domain = !strcmp(domain, "draft") ? GGML_GRAPH_EXECUTION_DOMAIN_DRAFT : GGML_GRAPH_EXECUTION_DOMAIN_MTP;
        certificate.flags = GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED;
        const char * semantics = getenv("GGML_TEST_MOE_BACKEND_SEQUENTIAL");
        sequential = semantics && !strcmp(semantics, "1");
    }
    const auto place_allocate = [&]() {
        uint32_t index = 0;
        for (auto & region : fixture.result.get_moe_regions()) { CHECK(region.place(owner->sched.get(), backends[index++ % 2])); }
        CHECK(index == 2);
        CHECK(ggml_backend_sched_alloc_graph(owner->sched.get(), fixture.result.get_gf()));
    };
    const auto inputs = [&](uint32_t rows, uint32_t used) {
        std::vector<float> values(size_t(rows) * fixture.n_embd);
        for (size_t i = 0; i < values.size(); ++i) { values[i] = 0.003f * (1 + i % 31); }
        ggml_backend_tensor_set(fixture.input, values.data(), 0, values.size() * sizeof(float));
        if (fixture.ordinary_gamma) {
            for (size_t i = 0; i < size_t(fixture.n_embd); ++i) { values[i] = 0.75f + 0.01f * float(i % 19); }
            ggml_backend_tensor_set(fixture.ordinary_gamma, values.data(), 0, size_t(fixture.n_embd) * sizeof(float));
        }
        for (auto * tensor : fixture.logits) {
            std::vector<float> scores(size_t(rows) * fixture.n_expert, -10.0f);
            for (uint32_t row = 0; row < rows; ++row) {
                for (uint32_t expert = 0; expert < used; ++expert) { scores[size_t(row) * fixture.n_expert + expert] = 10.0f - expert; }
            }
            ggml_backend_tensor_set(tensor, scores.data(), 0, scores.size() * sizeof(float));
        }
    };
    if (!output_auxiliaries) {
        place_allocate(); inputs(1, n_used);
        for (uint32_t repeat = 0; repeat < 2; ++repeat) {
            CHECK(ggml_backend_sched_graph_compute_ext(owner->sched.get(), fixture.result.get_gf(), &certificate) == GGML_STATUS_SUCCESS);
        }
        ggml_backend_sched_reset(owner->sched.get());
    }
    fixture.expert_used = 10;
    fixture.build_graph(false, nullptr, nullptr, false, 2, false, 0, false, false, true);
    for (auto * tensor : fixture.output) { ggml_set_output(tensor); }
    place_allocate(); inputs(2, 10);
    certificate.n_rows = 2; certificate.n_sequences = auxiliary && sequential ? 1 : 2;
    certificate.row_semantics = auxiliary && sequential ? GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL : GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT;
    certificate.flags = auxiliary ? GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED : GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_NONE;
    ggml_backend_moe_hybrid_config_v1 config = {};
    config.struct_size = sizeof(config); config.n_threads = 2;
    config.max_regions = 6; config.max_prepared_regions = 66;
    config.executor = GGML_BACKEND_MOE_HYBRID_EXECUTOR_V1_FIDELITY;
    config.backend = owner->gpu.get(); config.source_owner = &source_owner;
    config.cpu_module_acquire = ggml_backend_moe_cpu_module_acquire_v1;
    config.module_retain = ggml_backend_moe_module_retain_v1; config.module_release = ggml_backend_moe_module_release_v1;
    config.cpu_flags = GGML_BACKEND_MOE_CPU_SERVICE_FLAG_V1_ALLOW_UNKNOWN_THREAD_STACK_BYTES |
        GGML_BACKEND_MOE_CPU_SERVICE_FLAG_V1_ALLOW_UNPROVEN_RUNTIME_ALLOCATIONS;
    std::vector<int32_t> ranking(fixture.n_expert);
    std::vector<uint64_t> counts(fixture.n_expert);
    for (uint32_t i = 0; i < ranking.size(); ++i) { ranking[i] = (i + 3) % ranking.size(); counts[ranking[i]] = ranking.size() - i; }
    std::vector<ggml_backend_moe_static_profile_v1> profiles;
    std::vector<ggml_backend_moe_source_statistics_v1> statistics;
    if (const char * profile = getenv("GGML_TEST_MOE_BACKEND_PROFILE")) {
        if (!strcmp(profile, "ranks")) {
            for (const auto * down : fixture.down) { profiles.push_back({down, ranking.data(), uint32_t(ranking.size())}); }
            config.profiles = profiles.data(); config.n_profiles = profiles.size();
        } else {
            CHECK(!strcmp(profile, "statistics"));
            for (const auto & tensor : fixture.tensors) {
                if (tensor.status != GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE) { continue; }
                statistics.push_back({tensor.tensor, counts.data(), uint64_t(ranking.size()) * (ranking.size() + 1) / 2,
                    uint32_t(ranking.size()), GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY});
            }
            config.statistics = statistics.data(); config.n_statistics = statistics.size();
        }
    }
    CHECK(ggml_backend_sched_moe_hybrid_configure_v1(owner->sched.get(), &config) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    if (state_probe) { state_probe->check_real(owner->sched.get(), "configured-single"); state_probe->check_single_failure(owner->sched.get()); }
    config.backend = second.get();
    auto incompatible = config; ++incompatible.n_threads;
    CHECK(ggml_backend_sched_moe_hybrid_configure_v1(owner->sched.get(), &incompatible) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
    const auto configured = ggml_backend_sched_moe_hybrid_configure_v1(owner->sched.get(), &config);
    if (const char * negative = getenv("GGML_TEST_MOE_EXPECT_SINGLE_BACKEND")) {
        CHECK(!strcmp(negative, "1") && configured == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
        auto * released = owner->sched.release();
        CHECK(ggml_backend_sched_moe_source_free_v1(&released) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK && !released);
        owner->oracle.reset();
        owner->model->close_moe_source_owner();
        fprintf(stderr, "test-moe-cache: original provider rejects second backend before source prepare OK\n");
        return;
    }
    CHECK(configured == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    if (state_probe) { state_probe->check_real(owner->sched.get(), "configured-two"); }
    uint64_t generation = 0, shrink = 0;
    ggml_backend_sched_get_buffer_state(owner->sched.get(), &generation, &shrink);
    std::vector<std::unique_ptr<llm_graph_moe_hybrid_prepared>> metadata;
    std::vector<ggml_backend_moe_hybrid_region_v1> descriptors;
    const auto prepare = [&]() {
        ggml_backend_sched_get_buffer_state(owner->sched.get(), &generation, &shrink);
        metadata.clear(); descriptors.clear();
        for (auto * backend : {owner->gpu.get(), second.get()}) {
            config.backend = backend;
            CHECK(ggml_backend_sched_moe_hybrid_configure_v1(owner->sched.get(), &config) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        }
        uint32_t layer = 0;
        for (auto & region : fixture.result.get_moe_regions()) {
            CHECK(ggml_backend_sched_get_tensor_backend(owner->sched.get(), region.body_output) == backends[collapsed ? 0 : layer % 2]);
            ++layer;
            CHECK(region.finalize_metadata(owner->sched.get(), fixture.result.get_gf(), certificate.owner_generation, generation) == GGML_BACKEND_SCHED_REGION_STATUS_V1_OK);
            std::vector<std::unique_ptr<llm_graph_moe_hybrid_prepared>> projections;
            CHECK(region.prepare_routed_metadata(source_owner, 2, projections) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
            for (auto & projection : projections) {
                auto descriptor = projection->descriptor(); descriptor.certificate = certificate;
                descriptor.certificate.source_graph_uid = descriptor.source_graph_uid;
                descriptor.certificate.split_graph_uid = descriptor.split_graph_uid;
                CHECK(ggml_backend_sched_moe_hybrid_prepare_v1(owner->sched.get(), &descriptor) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
                descriptors.push_back(descriptor); metadata.push_back(std::move(projection));
            }
        }
        CHECK(descriptors.size() == 6);
    };
    if (getenv("GGML_TEST_MOE_BACKEND_OWNER_PLAN")) {
        const auto saved_certificate = certificate;
        certificate.domain = GGML_GRAPH_EXECUTION_DOMAIN_MAIN;
        certificate.flags = GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_NONE;
        certificate.row_semantics = GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL;
        certificate.n_sequences = 1;
        prepare();
        CHECK(ggml_backend_sched_graph_compute_ext(owner->sched.get(), fixture.result.get_gf(), &certificate) == GGML_STATUS_FAILED);
        ggml_backend_moe_hybrid_state_v1 rejected = {}; rejected.struct_size = sizeof(rejected);
        CHECK(ggml_backend_sched_moe_hybrid_state_v1(owner->sched.get(), &rejected));
        CHECK(!rejected.dispatch_active && !rejected.cpu_active_jobs && !rejected.window_launches &&
            !rejected.cpu_routes && !rejected.resident_routes && !rejected.transfer_routes);
        CHECK(ggml_backend_sched_moe_source_drain_v1(owner->sched.get()) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK);
        CHECK(ggml_backend_sched_moe_source_reset_graph_v1(owner->sched.get()) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK);
        certificate = saved_certificate;
        fixture.build_graph(false, nullptr, nullptr, false, 2, false, 0, false, false, true);
        for (auto * tensor : fixture.output) { ggml_set_output(tensor); }
        place_allocate(); inputs(2, 10);
        fprintf(stderr, "test-moe-cache: unarmable source owner plan rejects without dispatch/routes/launches; drain/reset OK\n");
    }
    prepare();
    CHECK(descriptors[0].split_index != descriptors[3].split_index);
    owner->reference = std::make_unique<layer_fixture>(0, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_TYPE_Q5_K,
        false, std::array<int, 2>{device, device}, 0, n_slots, false, LLM_ARCH_QWEN3MOE, LLM_FFN_SILU,
        10, GGML_TYPE_Q5_K, 256, 256, 16, false, output_auxiliaries);
    auto & reference = *owner->reference;
    reference.build_graph(false, nullptr, nullptr, false, 2, false, 0, false, false, true);
    owner->oracle.reset(ggml_backend_sched_new(backends, bufts, 3, 256, false, true));
    uint32_t reference_index = 0;
    for (auto & region : reference.result.get_moe_regions()) { CHECK(region.place(owner->oracle.get(), backends[reference_index++ % 2])); }
    for (auto * tensor : reference.output) { ggml_set_output(tensor); }
    CHECK(ggml_backend_sched_alloc_graph(owner->oracle.get(), reference.result.get_gf()));
    const auto reference_inputs = [&]() {
        std::vector<float> values(size_t(2) * reference.n_embd);
        for (size_t i = 0; i < values.size(); ++i) { values[i] = 0.003f * (1 + i % 31); }
        ggml_backend_tensor_set(reference.input, values.data(), 0, values.size() * sizeof(float));
        for (size_t i = 0; i < size_t(reference.n_embd); ++i) { values[i] = 0.75f + 0.01f * float(i % 19); }
        ggml_backend_tensor_set(reference.ordinary_gamma, values.data(), 0, size_t(reference.n_embd) * sizeof(float));
        for (auto * tensor : reference.logits) {
            std::vector<float> scores(size_t(2) * reference.n_expert, -10.0f);
            for (uint32_t row = 0; row < 2; ++row) {
                for (uint32_t expert = 0; expert < 10; ++expert) { scores[size_t(row) * reference.n_expert + expert] = 10.0f - expert; }
            }
            ggml_backend_tensor_set(tensor, scores.data(), 0, scores.size() * sizeof(float));
        }
    };
    std::array<std::vector<float>, 2> reference_outputs;
    for (uint32_t warm = 0; warm < 2; ++warm) {
        reference_inputs();
        std::vector<std::vector<float>> input_values;
        if (output_auxiliaries) {
            for (auto * tensor : {reference.input, reference.logits[0], reference.logits[1], reference.ordinary_gamma}) {
                input_values.push_back(active_grouped_tensor_values(tensor));
            }
        }
        CHECK(ggml_backend_sched_graph_compute_ext(owner->oracle.get(), reference.result.get_gf(), &certificate) == GGML_STATUS_SUCCESS);
        if (output_auxiliaries) {
            size_t index = 0, changed = 0;
            for (auto * tensor : {reference.input, reference.logits[0], reference.logits[1], reference.ordinary_gamma}) {
                const auto observed = active_grouped_tensor_values(tensor);
                const auto & expected = input_values[index++];
                for (size_t i = 0; i < observed.size(); ++i) { changed += observed[i] != expected[i]; }
            }
            fprintf(stderr, "test-moe-cache: ordinary graph input storage changed warm=%u values=%zu; each invocation receives fresh inputs\n", warm, changed);
        }
        for (uint32_t layer = 0; layer < 2; ++layer) {
            const auto expected = active_grouped_tensor_values(reference.output[layer]);
            double norm = 0;
            for (float value : expected) { CHECK(std::isfinite(value)); norm += double(value) * value; }
            fprintf(stderr, "test-moe-cache: backend-session ordinary control warm=%u layer=%u norm=%.9g\n", warm, layer, norm);
            if (!warm) { reference_outputs[layer] = expected; }
            else { CHECK(norm > 0 && reference_outputs[layer] == expected); }
        }
    }
    for (uint32_t repeat = 0; repeat < 2; ++repeat) {
        inputs(2, 10);
        CHECK(ggml_backend_sched_graph_compute_ext(owner->sched.get(), fixture.result.get_gf(), &certificate) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_sched_moe_source_drain_v1(owner->sched.get()) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK);
        for (uint32_t layer = 0; layer < 2; ++layer) {
            const auto observed = active_grouped_tensor_values(fixture.output[layer]);
            const auto expected = active_grouped_tensor_values(reference.output[layer]);
            CHECK(observed.size() == expected.size());
            double error = 0, norm = 0;
            for (size_t i = 0; i < observed.size(); ++i) {
                CHECK(std::isfinite(observed[i]) && std::isfinite(expected[i]));
                error += (double(observed[i]) - expected[i]) * (double(observed[i]) - expected[i]);
                norm += double(expected[i]) * expected[i];
            }
            const double nmse = error / std::max(norm, 1e-30);
            fprintf(stderr, "test-moe-cache: backend-session original Q5_K layer=%u repeat=%u nmse=%.9g norm=%.9g observed0=%.9g expected0=%.9g\n", layer, repeat, nmse, norm, observed[0], expected[0]);
            CHECK(nmse <= 5e-3);
        }
    }
    if (output_auxiliaries) {
        for (uint32_t layer = 0; layer < 2; ++layer) {
            auto * grouped = ggml_cuda_moe_grouped_context_for_test(backends[collapsed ? 0 : layer]);
            const uint64_t expected_bytes = ggml_nbytes(fixture.up_scale[layer]) + ggml_nbytes(fixture.gate_scale[layer]) +
                ggml_nbytes(fixture.down_scale[layer]) + ggml_nbytes(fixture.up_bias[layer]) +
                ggml_nbytes(fixture.gate_bias[layer]) + ggml_nbytes(fixture.down_bias[layer]);
            const uint64_t shadow_bytes = pageable_auxiliaries ? expected_bytes * (collapsed ? 2 : 1) : 0;
            CHECK(ggml_cuda_moe_grouped_context_test_access::original_auxiliary_bytes(*grouped) == shadow_bytes);
            cudaStream_t stream = nullptr;
            CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            for (auto * leaf : {fixture.up_scale[layer], fixture.gate_scale[layer], fixture.down_scale[layer],
                    fixture.up_bias[layer], fixture.gate_bias[layer], fixture.down_bias[layer]}) {
                const auto saved = *leaf;
                auto * buffer = leaf->buffer;
                void * data = leaf->data;
                std::shared_ptr<void> lease;
                auto stale = source_owner; ++stale.generation;
                CHECK(!grouped->prepare_source_leaf(leaf, stream, stale, &buffer, &data, &lease));
                CHECK(buffer == saved.buffer && data == saved.data && !lease);
                ++leaf->nb[0];
                CHECK(!grouped->prepare_source_leaf(leaf, stream, source_owner, &buffer, &data, &lease));
                *leaf = saved;
                leaf->flags |= GGML_TENSOR_FLAG_INPUT;
                CHECK(!grouped->prepare_source_leaf(leaf, stream, source_owner, &buffer, &data, &lease));
                *leaf = saved;
                CHECK(grouped->prepare_source_leaf(leaf, stream, source_owner, &buffer, &data, &lease));
                CHECK(buffer && data && lease && (data != saved.data) == pageable_auxiliaries);
                CHECK(ggml_moe_source_tensor_matches(*leaf, saved));
                const auto expected = active_grouped_tensor_values(leaf);
                std::vector<float> observed(expected.size());
                CUDA_OK(cudaMemcpy(observed.data(), data, ggml_nbytes(leaf), cudaMemcpyDefault));
                CHECK(memcmp(expected.data(), observed.data(), ggml_nbytes(leaf)) == 0);
            }
            CUDA_OK(cudaStreamDestroy(stream));
            CHECK(ggml_cuda_moe_grouped_context_test_access::original_auxiliary_bytes(*grouped) == shadow_bytes);
            fprintf(stderr, "test-moe-cache: immutable canonical leaf leases/byte-reuse/stale-owner/layout/dynamic-input rejection OK\n");
            fprintf(stderr, "test-moe-cache: six original auxiliary sources layer=%u bytes=%llu shadow_bytes=%llu actual GGML graph OK\n", layer,
                (unsigned long long) expected_bytes, (unsigned long long) shadow_bytes);
        }
    }
    if (!profiles.empty() || !statistics.empty()) {
        for (uint32_t layer = 0; layer < 2; ++layer) {
            auto * grouped = ggml_cuda_moe_grouped_context_for_test(backends[collapsed ? 0 : layer]);
            ggml_cuda_moe_candidate_group_key key;
            CHECK(grouped->find_down_group_key(fixture.down[layer], &key));
            for (uint32_t expert = 0; expert < ranking.size(); ++expert) {
                const auto slot = ggml_cuda_moe_grouped_context_test_access::device_slot_for_expert(*grouped, key, expert);
                const auto rank = std::find(ranking.begin(), ranking.end(), int32_t(expert)) - ranking.begin();
                CHECK(slot == (rank < fixture.cache_slots ? int32_t(rank) : -1));
            }
        }
    }
    ggml_backend_moe_hybrid_state_v1 state = {}; state.struct_size = sizeof(state);
    CHECK(ggml_backend_sched_moe_hybrid_state_v1(owner->sched.get(), &state));
    CHECK(state.cpu_routes > 0 && state.resident_routes > 0 && state.cpu_active_jobs == 0 && state.dispatch_active == 0);
    CHECK(state.ticket_state == 0 && state.last_epoch == 0);
    if (state_probe) { state_probe->check_real(owner->sched.get(), "prepared"); state_probe->check(owner->sched.get()); }
    if (getenv("GGML_TEST_MOE_BACKEND_NAMESPACE")) {
        const std::array<ggml_tensor *, 2> outputs = {fixture.output[0], fixture.output[1]};
        std::vector<std::vector<float>> expected;
        for (auto * output : outputs) { expected.push_back(active_grouped_tensor_values(output)); }
        const auto reject = [&](const ggml_graph_execution_certificate & changed) {
            ggml_backend_moe_hybrid_state_v1 before = {}; before.struct_size = sizeof(before);
            CHECK(ggml_backend_sched_moe_hybrid_state_v1(owner->sched.get(), &before));
            CHECK(ggml_backend_sched_graph_compute_ext(owner->sched.get(), fixture.result.get_gf(), &changed) == GGML_STATUS_FAILED);
            for (size_t i = 0; i < outputs.size(); ++i) { CHECK(active_grouped_tensor_values(outputs[i]) == expected[i]); }
            ggml_backend_moe_hybrid_state_v1 after = {}; after.struct_size = sizeof(after);
            CHECK(ggml_backend_sched_moe_hybrid_state_v1(owner->sched.get(), &after));
            CHECK(after.window_launches == before.window_launches && after.cpu_routes == before.cpu_routes &&
                after.resident_routes == before.resident_routes && after.transfer_routes == before.transfer_routes &&
                !after.cpu_active_jobs && !after.dispatch_active);
        };
        auto changed = certificate; ++changed.owner_namespace; reject(changed);
        changed = certificate; ++changed.owner_generation; reject(changed);
        changed = certificate;
        changed.domain = certificate.domain == GGML_GRAPH_EXECUTION_DOMAIN_MTP ? GGML_GRAPH_EXECUTION_DOMAIN_DRAFT : GGML_GRAPH_EXECUTION_DOMAIN_MTP;
        changed.flags = GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED;
        if (!auxiliary) { changed.row_semantics = GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL; }
        reject(changed);
        changed = certificate;
        changed.domain = GGML_GRAPH_EXECUTION_DOMAIN_MAIN;
        changed.row_semantics = GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL;
        changed.flags = GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_NONE;
        reject(changed);
        inputs(2, 10);
        CHECK(ggml_backend_sched_graph_compute_ext(owner->sched.get(), fixture.result.get_gf(), &certificate) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_sched_moe_source_drain_v1(owner->sched.get()) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK);
        for (size_t i = 0; i < outputs.size(); ++i) { CHECK(active_grouped_tensor_values(outputs[i]) == expected[i]); }
        fprintf(stderr, "test-moe-cache: source certificate domain=%u semantics=%u namespace/generation/domain rejection before effects and recovery OK\n",
            certificate.domain, certificate.row_semantics);
        CHECK(ggml_backend_sched_moe_hybrid_state_v1(owner->sched.get(), &state));
    }
    ggml_backend_sched_t clone = nullptr;
    CHECK(ggml_backend_sched_moe_source_clone_v1(owner->sched.get(), &clone) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK);
    if (state_probe) { state_probe->check_real(clone, "unprepared-clone"); }
    CHECK(ggml_backend_sched_moe_source_free_v1(&clone) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK && !clone);
    const auto retirement_epoch = ggml_backend_sched_moe_source_retirement_epoch_v1(owner->sched.get());
    const auto original_uid = fixture.result.get_gf()->uid;
    uint64_t backing_before = 0, shrink_before = 0;
    ggml_backend_sched_get_buffer_state(owner->sched.get(), &backing_before, &shrink_before);
    std::array<std::vector<float>, 2> published_before = {
        active_grouped_tensor_values(fixture.output[0]), active_grouped_tensor_values(fixture.output[1])};
    CHECK(ggml_backend_sched_moe_source_retire_v1(owner->sched.get()) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK);
    CHECK(ggml_backend_sched_moe_source_retirement_epoch_v1(owner->sched.get()) == retirement_epoch + 1);
    uint64_t backing_after = 0, shrink_after = 0;
    ggml_backend_sched_get_buffer_state(owner->sched.get(), &backing_after, &shrink_after);
    CHECK(backing_before == backing_after && shrink_before == shrink_after && fixture.result.get_gf()->uid == original_uid);
    CHECK(ggml_backend_sched_get_tensor_backend(owner->sched.get(), fixture.result.get_moe_regions()[0].body_output) == backends[0]);
    CHECK(ggml_backend_sched_graph_compute_ext(owner->sched.get(), fixture.result.get_gf(), &certificate) == GGML_STATUS_FAILED);
    CHECK(active_grouped_tensor_values(fixture.output[0]) == published_before[0] && active_grouped_tensor_values(fixture.output[1]) == published_before[1]);
    CHECK(ggml_backend_sched_moe_source_reset_graph_v1(owner->sched.get()) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK);
    fixture.build_graph(false, nullptr, nullptr, false, 2, false, 0, false, false, true);
    for (auto * tensor : fixture.output) { ggml_set_output(tensor); }
    place_allocate(); inputs(2, 10); prepare();
    CHECK(ggml_backend_sched_graph_compute_ext(owner->sched.get(), fixture.result.get_gf(), &certificate) == GGML_STATUS_SUCCESS);
    for (size_t i = 0; i < published_before.size(); ++i) {
        CHECK(active_grouped_tensor_values(fixture.output[i]) == published_before[i]);
    }
    CHECK(ggml_backend_sched_moe_hybrid_state_v1(owner->sched.get(), &state));
    fprintf(stderr, "test-moe-cache: capture-only retirement preserves graph/planner/backing, rejects stale compute and recovers with preparation OK\n");
    const auto recover_backing = [&]() {
        CHECK(ggml_backend_sched_moe_source_reset_graph_v1(owner->sched.get()) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK);
        fixture.build_graph(false, nullptr, nullptr, false, 2, false, 0, false, false, true);
        for (auto * tensor : fixture.output) { ggml_set_output(tensor); }
        place_allocate(); inputs(2, 10); prepare();
        CHECK(ggml_backend_sched_graph_compute_ext(owner->sched.get(), fixture.result.get_gf(), &certificate) == GGML_STATUS_SUCCESS);
        for (size_t i = 0; i < published_before.size(); ++i) {
            CHECK(active_grouped_tensor_values(fixture.output[i]) == published_before[i]);
        }
    };
    const size_t original_bytes = ggml_backend_sched_get_buffer_size(owner->sched.get(), owner->gpu.get());
    CHECK(original_bytes > 0 && original_bytes < SIZE_MAX - 65536);
    for (bool growing : {true, false}) {
        uint64_t old_generation = 0, old_shrink = 0;
        ggml_backend_sched_get_buffer_state(owner->sched.get(), &old_generation, &old_shrink);
        const auto old_epoch = ggml_backend_sched_moe_source_retirement_epoch_v1(owner->sched.get());
        const auto old_graph_uid = fixture.result.get_gf()->uid;
        CHECK(ggml_backend_sched_moe_source_clone_v1(owner->sched.get(), &clone) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK);
        ggml_context_ptr resize_context(ggml_init({4 * ggml_tensor_overhead() + ggml_graph_overhead_custom(8, false), nullptr, true}));
        CHECK(resize_context);
        const int64_t elements = growing ? int64_t(original_bytes / sizeof(float) + 16384) : 8;
        auto * left = ggml_new_tensor_1d(resize_context.get(), GGML_TYPE_F32, elements);
        auto * right = ggml_new_tensor_1d(resize_context.get(), GGML_TYPE_F32, elements);
        ggml_set_input(left); ggml_set_input(right);
        auto * resized_output = ggml_add(resize_context.get(), left, right);
        ggml_set_output(resized_output);
        auto * resized_graph = ggml_new_graph_custom(resize_context.get(), 8, false);
        ggml_build_forward_expand(resized_graph, resized_output);
        for (auto * tensor : {left, right, resized_output}) {
            ggml_backend_sched_set_tensor_backend(clone, tensor, owner->gpu.get());
        }
        if (!growing) { ggml_backend_sched_request_buffer_shrink(clone); }
        CHECK(ggml_backend_sched_alloc_graph(clone, resized_graph));
        uint64_t new_generation = 0, new_shrink = 0, clone_generation = 0, clone_shrink = 0;
        ggml_backend_sched_get_buffer_state(owner->sched.get(), &new_generation, &new_shrink);
        ggml_backend_sched_get_buffer_state(clone, &clone_generation, &clone_shrink);
        CHECK(new_generation > old_generation && clone_generation == new_generation && clone_shrink == new_shrink);
        CHECK(ggml_backend_sched_moe_source_retirement_epoch_v1(owner->sched.get()) == old_epoch + 1);
        CHECK(fixture.result.get_gf()->uid == old_graph_uid);
        CHECK(ggml_backend_sched_get_tensor_backend(owner->sched.get(), fixture.result.get_moe_regions()[0].body_output) == backends[0]);
        CHECK(ggml_backend_sched_graph_compute_ext(owner->sched.get(), fixture.result.get_gf(), &certificate) == GGML_STATUS_FAILED);
        const size_t shared_bytes = ggml_backend_sched_get_buffer_size(owner->sched.get(), owner->gpu.get());
        CHECK(shared_bytes == ggml_backend_sched_get_buffer_size(clone, owner->gpu.get()));
        CHECK(growing ? shared_bytes > original_bytes : shared_bytes == original_bytes);
        recover_backing();
        CHECK(ggml_backend_sched_moe_source_retirement_epoch_v1(owner->sched.get()) == old_epoch + 1);
        CHECK(ggml_backend_sched_moe_source_free_v1(&clone) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK && !clone);
    }
    CHECK(ggml_backend_sched_moe_hybrid_state_v1(owner->sched.get(), &state));
    fprintf(stderr, "test-moe-cache: actual shared-plan grow/shrink retires captures before replacement, preserves graph/planner ownership and re-prepares exact output OK\n");
    auto * second_output = fixture.result.get_moe_regions()[1].body_output;
    const auto saved_op = second_output->op;
    second_output->op = GGML_OP_NONE;
    CHECK(ggml_backend_sched_graph_compute_ext(owner->sched.get(), fixture.result.get_gf(), &certificate) != GGML_STATUS_SUCCESS);
    second_output->op = saved_op;
    ggml_backend_moe_hybrid_state_v1 after = {}; after.struct_size = sizeof(after);
    CHECK(ggml_backend_sched_moe_hybrid_state_v1(owner->sched.get(), &after));
    CHECK(after.cpu_routes == state.cpu_routes && after.resident_routes == state.resident_routes && !after.dispatch_active && !after.cpu_active_jobs);
    CHECK(ggml_backend_sched_moe_source_reset_graph_v1(owner->sched.get()) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK);
    fixture.build_graph(false, nullptr, nullptr, false, 2, false, 0, false, false, true);
    for (auto * tensor : fixture.output) { ggml_set_output(tensor); }
    place_allocate(); inputs(2, 10); prepare();
    fidelity_probe = owner.get();
    owner->sync_delegate = owner->cpu->iface.synchronize;
    owner->cpu->iface.synchronize = +[](ggml_backend_t backend) {
        ++fidelity_probe->sync_calls;
        if (fidelity_probe->sync_delegate) { fidelity_probe->sync_delegate(backend); }
    };
    CHECK(ggml_backend_sched_moe_hybrid_set_test_hook_v1(owner->sched.get(), +[](void * data, uint32_t phase, uint64_t, void *) {
        auto & probe = *static_cast<fidelity_graph_owner *>(data);
        if (phase == GGML_BACKEND_MOE_HYBRID_TEST_SOURCE_COMPUTE_ENTERED && ++probe.source_entered == 2) {
            probe.sync_before_failure = probe.sync_calls;
            return false;
        }
        return true;
    }, owner.get()));
    CHECK(ggml_backend_sched_graph_compute_ext(owner->sched.get(), fixture.result.get_gf(), &certificate) != GGML_STATUS_SUCCESS && owner->source_entered == 2);
    const auto normal_syncs = owner->sync_calls - owner->sync_before_failure;
    owner->cpu->iface.synchronize = owner->sync_delegate;
    fidelity_probe = nullptr;
    fprintf(stderr, "test-moe-cache: backend-session failure after earlier split effects normal_backend_syncs=%u\n", normal_syncs);
    CHECK(getenv("GGML_TEST_MOE_EXPECT_MISSING_NORMAL_DRAIN") ? normal_syncs == 0 : normal_syncs > 0);
    state = {}; state.struct_size = sizeof(state);
    CHECK(ggml_backend_sched_moe_hybrid_state_v1(owner->sched.get(), &state));
    CHECK(state.quiescing && state.cpu_routes > 0 && !state.dispatch_active && !state.cpu_active_jobs);
    auto * released = owner->sched.release();
    CHECK(ggml_backend_sched_moe_source_free_v1(&released) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK && !released);
    owner->oracle.reset();
    owner->model->close_moe_source_owner();
    fprintf(stderr, "test-moe-cache: backend-session fixture handles=2 actual_source_backends=%u physical_devices=1 profiles=%zu statistics=%zu; routed splits/numerics/shared CPU/aggregate/clone/preflight/reset/partial-effects close OK\n",
        collapsed ? 1u : 2u, profiles.size(), statistics.size());
}

void test_fidelity_real_window(int device, uint32_t capacity, uint32_t arm, bool no_alias, bool scheduler = false,
        const hybrid_layer_signature & signature = {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, LLM_FFN_SILU, n_dim, n_dim},
        uint32_t routes = n_used, uint32_t quota = UINT32_MAX, uint32_t experts = n_experts, bool empty_prefix = false,
        uint32_t bf16_columns = 0, bool numeric_diagnostics = false, bool flash_attention = false, bool cpu_oracle = false,
        bool delayed_cancel = false, bool staged_inputs = false, bool pending_handoff = false, bool early_fallback = false,
        uint32_t transport_case = 0, bool overlap_fixture = false, bool static_profile = false, uint32_t profile_adaptation = 0, bool source_statistics = false) {
    CHECK(routes > 0 && routes <= experts && experts > 2);
    const bool native_reference = ggml_moe_fidelity_selection().reference;
    const bool source_core = scheduler && ggml_moe_fidelity_selection().source_pool;
    const char * fidelity_mode = getenv("GGML_TEST_MOE_FIDELITY");
    const bool generic_body = fidelity_mode && !strcmp(fidelity_mode, "source-core-generic");
    const bool main_prefill = fidelity_mode && !strcmp(fidelity_mode, "source-core-prefill");
    struct source_environment {
        bool enabled;
        std::string saved[2];
        bool present[2] = {};
        const char * names[2] = {"GGML_MOE_FIDELITY_ARM", "GGML_MOE_FIDELITY_NO_HOST_ALIAS"};
        static void set(const char * name, const char * value) {
#ifdef _WIN32
            CHECK(_putenv_s(name, value ? value : "") == 0);
#else
            CHECK((value ? setenv(name, value, 1) : unsetenv(name)) == 0);
#endif
        }
        source_environment(bool enabled, uint32_t arm, bool no_alias) : enabled(enabled) {
            if (!enabled) { return; }
            for (size_t i = 0; i < 2; ++i) {
                const char * value = getenv(names[i]);
                present[i] = value != nullptr;
                saved[i] = value ? value : "";
            }
            set(names[0], arm == GGML_CUDA_MOE_FIDELITY_SEGMENTED ? "segmented" : "poll");
            set(names[1], no_alias ? "1" : "0");
        }
        ~source_environment() { if (enabled) { for (size_t i = 0; i < 2; ++i) { set(names[i], present[i] ? saved[i].c_str() : nullptr); } } }
    } environment(source_core, arm, no_alias);
    CHECK(!pending_handoff || (native_reference && capacity == 1 && !staged_inputs && !delayed_cancel && experts > routes));
    CHECK(!staged_inputs || ((!scheduler || source_core) && !delayed_cancel));
    if (quota == UINT32_MAX) { quota = capacity == 1 ? 0 : 1; }
    fprintf(stderr, "test-moe-cache: fidelity signature R%u routes=%u experts=%u quota=%u gate_up=%s down=%s embd=%d ff=%d\n",
        capacity, routes, experts, quota, ggml_type_name(signature.gate_up_type), ggml_type_name(signature.down_type), signature.n_embd, signature.n_ff);
    auto owner = std::make_shared<fidelity_graph_owner>();
    owner->gpu.reset(ggml_backend_cuda_init(device));
    const ggml_staged_input_api * stage_api = nullptr;
    ggml_backend_moe_hybrid_staged_pending_v1_t stage_pending = nullptr;
    if (staged_inputs) {
        const auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(owner->gpu.get()));
        const auto get_stage_api = reinterpret_cast<ggml_staged_input_get_api_t>(ggml_backend_reg_get_proc_address(reg, GGML_STAGED_INPUT_PROC));
        stage_api = get_stage_api ? get_stage_api() : nullptr;
        stage_pending = reinterpret_cast<ggml_backend_moe_hybrid_staged_pending_v1_t>(
            ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_HYBRID_STAGED_PENDING_V1_PROC_NAME));
        CHECK(stage_api && stage_pending);
        owner->staged_input = {stage_api->create(owner->gpu.get(), signature.n_embd * sizeof(float)), stage_api->destroy};
        owner->staged_reference = {stage_api->create(owner->gpu.get(), signature.n_embd * sizeof(float)), stage_api->destroy};
        CHECK(owner->staged_input && owner->staged_reference);
        const auto set_submit = reinterpret_cast<ggml_staged_input_set_submit_t>(
            ggml_backend_reg_get_proc_address(reg, GGML_STAGED_INPUT_SET_SUBMIT_PROC));
        CHECK(set_submit);
        const auto submit = +[](void * context) {
            static_cast<fidelity_graph_owner *>(context)->staged_submissions.fetch_add(1, std::memory_order_relaxed);
        };
        CHECK(!set_submit(owner->staged_input.get(), nullptr, owner.get()));
        CHECK(set_submit(owner->staged_input.get(), submit, owner.get()));
        CHECK(!set_submit(owner->staged_input.get(), submit, owner.get()));
    }
    size_t transport_budget = 0;
    if (transport_case >= 2) {
        ggml_backend_moe_staging_query_v1 query = {};
        query.struct_size = sizeof(query);
        query.n_slots = routes; query.n_experts = experts; query.top_k = 1;
        query.n_banks = signature.layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE ? 3 : 2;
        query.staged_bank_mask = (1u << query.n_banks) - 1;
        query.family_mask = GGML_BACKEND_MOE_STAGING_FAMILY_V1_GROUPED | GGML_BACKEND_MOE_STAGING_FAMILY_V1_LEGACY |
            GGML_BACKEND_MOE_STAGING_FAMILY_V1_HOST_STAGED;
        query.flags = GGML_BACKEND_MOE_STAGING_FLAG_V1_PREDICTION_CONTROL;
        const size_t projection = ggml_row_size(signature.gate_up_type, signature.n_embd) * signature.n_ff;
        const size_t down = ggml_row_size(signature.down_type, signature.n_ff) * signature.n_embd;
        query.bank_expert_strides[0] = projection * (query.n_banks == 2 ? 2 : 1);
        query.bank_expert_strides[1] = query.n_banks == 3 ? projection : down;
        if (query.n_banks == 3) { query.bank_expert_strides[2] = down; }
        ggml_backend_moe_staging_size_v1 sizing = {};
        sizing.struct_size = sizeof(sizing);
        CHECK(ggml_backend_cuda_moe_staging_size_v1(&query, &sizing));
        size_t largest = 0;
        for (uint32_t b = 0; b < query.n_banks; ++b) { largest = std::max(largest, size_t(query.bank_expert_strides[b])); }
        transport_budget = 2 * (sizing.grouped_min_bytes + sizing.legacy_min_bytes + sizing.host_staged_min_bytes) + 8 * largest;
    }
    owner->fixture = std::make_unique<layer_fixture>(transport_budget, signature.layout, signature.gate_up_type, false,
        std::array<int, 2>{device, device}, all_cached_layers, native_reference ? routes : n_used, native_reference, LLM_ARCH_QWEN3MOE, signature.activation,
        routes, signature.down_type, signature.n_embd, signature.n_ff, experts, transport_case == 3);
    owner->reference = std::make_unique<layer_fixture>(0, signature.layout, signature.gate_up_type, false,
        std::array<int, 2>{device, device}, 0, n_used, native_reference, LLM_ARCH_QWEN3MOE, signature.activation,
        routes, signature.down_type, signature.n_embd, signature.n_ff, experts);
    auto & fixture = *owner->fixture;
    auto & reference = *owner->reference;
    if (native_reference) {
        CHECK(routes > n_used && (uint64_t(routes - n_used) * ggml_moe_fidelity_selection().pcie_num) / 256 > 0);
        const auto snapshot = fixture.manifest();
        CHECK(ggml_backend_cuda_moe_candidate_replace_v2(owner->gpu.get(), &snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        fixture.expert_used = n_used;
        fixture.build_graph();
        ggml_backend_ptr seed_cpu(ggml_backend_cpu_init());
        ggml_backend_t seed_backends[]{owner->gpu.get(), seed_cpu.get()};
        ggml_backend_sched_ptr seed_sched(ggml_backend_sched_new(seed_backends, nullptr, 2, 256, false, true));
        CHECK(seed_cpu && seed_sched);
        for (auto & region : fixture.result.get_moe_regions()) { CHECK(region.place(seed_sched.get(), owner->gpu.get())); }
        CHECK(ggml_backend_sched_alloc_graph(seed_sched.get(), fixture.result.get_gf()));
        std::vector<float> values(signature.n_embd, 0.003f);
        ggml_backend_tensor_set(fixture.input, values.data(), 0, values.size() * sizeof(float));
        const auto seed_certificate = layer_certificate();
        const uint32_t seeded_experts = pending_handoff ? routes : n_used;
        CHECK(seeded_experts % n_used == 0);
        for (uint32_t first = 0; first < seeded_experts; first += n_used) {
            for (auto * logits : fixture.logits) {
                std::vector<float> scores(experts, -10.0f);
                scores[first] = 10.0f;
                scores[first + 1] = 9.0f;
                ggml_backend_tensor_set(logits, scores.data(), 0, scores.size() * sizeof(float));
            }
            CHECK(ggml_backend_sched_graph_compute_ext(seed_sched.get(), fixture.result.get_gf(), &seed_certificate) == GGML_STATUS_SUCCESS);
            // The next grouped planner commits completed bank admissions into the reciprocal maps.
            CHECK(ggml_backend_sched_graph_compute_ext(seed_sched.get(), fixture.result.get_gf(), &seed_certificate) == GGML_STATUS_SUCCESS);
        }
        ggml_backend_sched_synchronize(seed_sched.get());
        auto * seed_grouped = ggml_cuda_moe_grouped_context_for_test(owner->gpu.get());
        for (const auto * down : fixture.down) {
            ggml_cuda_moe_candidate_group_key key;
            CHECK(seed_grouped->find_down_group_key(down, &key));
            uint32_t occupied = 0;
            for (uint32_t expert = 0; expert < experts; ++expert) {
                const int32_t slot = ggml_cuda_moe_grouped_context_test_access::device_slot_for_expert(*seed_grouped, key, expert);
                CHECK((slot >= 0) == (expert < seeded_experts));
                occupied += slot >= 0;
            }
            CHECK(occupied == seeded_experts);
        }
        fprintf(stderr, "test-moe-cache: reference seed admission committed before scheduler reset\n");
        ggml_backend_sched_reset(seed_sched.get());
        fixture.expert_used = routes;
    }
    for (auto * target : {&fixture, &reference}) {
        target->independent_overlap = overlap_fixture;
        const char * mode = getenv("GGML_TEST_MOE_FIDELITY");
        target->routed_backend_stage = mode && (!strcmp(mode, "routed-new-op") || !strcmp(mode, "routed-new-op-fallback"));
        target->activation_images = mode && !strcmp(mode, "source-core-images");
        target->ordinary_scratch = mode && !strcmp(mode, "source-core-resources");
        target->ordinary_matmul = mode && !strcmp(mode, "source-core-matmul");
        target->ordinary_empty_view = mode && !strcmp(mode, "source-core-empty-view");
        target->ordinary_cpu_prefix = mode && !strcmp(mode, "source-core-split-preflight");
        target->ordinary_library = target->ordinary_cpu_prefix || (mode && !strcmp(mode, "source-core-libraries"));
        void * stage = target == &fixture ? owner->staged_input.get() : owner->staged_reference.get();
        target->build_graph(false, stage_api, stage, source_core && !overlap_fixture, capacity, empty_prefix, bf16_columns, flash_attention, cpu_oracle,
            ggml_moe_fidelity_selection().source_pool && !cpu_oracle);
        auto * graph = target->result.get_gf();
        if (target->ordinary_library) {
            const auto * scan = ggml_get_tensor(target->result.get_ctx(), "fidelity_ordinary_library_scan");
            CHECK(ggml_backend_dev_supports_op(ggml_backend_get_device(owner->gpu.get()), scan) != target->ordinary_cpu_prefix);
            CHECK(ggml_backend_dev_supports_op(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), scan));
        }
        for (int i = 0; i < graph->n_nodes; ++i) {
            graph->nodes[i]->flags &= ~GGML_TENSOR_FLAG_OUTPUT;
        }
        ggml_set_output(target->output[1]);
        if (source_core && !overlap_fixture) {
            ggml_set_output(ggml_get_tensor(target->result.get_ctx(), "hybrid_fusion_retained"));
        }
        if (early_fallback) {
            auto * tail = ggml_view_2d(target->result.get_ctx(), target->output[1], target->output[1]->ne[0] - 1,
                target->output[1]->ne[1], target->output[1]->nb[1], 0);
            ggml_set_output(tail);
            ggml_build_forward_expand(graph, tail);
        }
        if (target->ordinary_norm_output) { ggml_set_output(target->ordinary_norm_output); }
        if (target->staged) { ggml_set_output(target->staged); }
        if (target->route_probe) { ggml_set_output(target->route_probe); }
        if (generic_body) { ggml_set_output(target->result.get_moe_regions()[0].body_output); }
        if (numeric_diagnostics) {
            for (auto * tensor : {target->bf16_projection, target->bf16_prefix, target->attention_output, target->attention_prefix}) {
                if (tensor) { ggml_set_output(tensor); }
            }
            for (uint32_t layer = 0; layer < 2; ++layer) {
                ggml_set_output(target->result.get_moe_regions()[layer].body_output);
                ggml_set_output(target->output[layer]);
            }
        }
    }
    ggml_tensor * external_effect = nullptr;
    if (delayed_cancel || staged_inputs) {
        external_effect = ggml_dup_tensor(fixture.result.get_ctx(), fixture.output[1]);
        ggml_set_input(external_effect);
        ggml_set_name(external_effect, "fidelity_external_effect");
        ggml_build_forward_expand(fixture.result.get_gf(), ggml_cpy(fixture.result.get_ctx(), fixture.output[1], external_effect));
    }
    owner->model.reset(llama_model_create(LLM_ARCH_QWEN3MOE, llama_model_default_params()));
    for (const auto & source : fixture.tensors) {
        CHECK(owner->model->record_moe_readable_source(source.tensor, source.tensor->data, ggml_nbytes(source.tensor)));
    }
    ggml_backend_moe_source_owner_v1 source_owner = {};
    CHECK(owner->model->moe_source_owner_v1(&source_owner));
    owner->cpu.reset(ggml_backend_cpu_init());
    auto * input_buft = ggml_backend_get_default_buffer_type(owner->gpu.get());
    for (auto * tensor : {fixture.input, fixture.logits[0], fixture.logits[1], external_effect}) {
        if (!tensor) { continue; }
        auto buffer = ggml_backend_buft_alloc_buffer(input_buft, ggml_backend_buft_get_alloc_size(input_buft, tensor));
        CHECK(buffer != nullptr && ggml_backend_tensor_alloc(buffer, tensor, ggml_backend_buffer_get_base(buffer)) == GGML_STATUS_SUCCESS);
        owner->input_buffers.emplace_back(buffer);
    }
    for (auto * target : {&fixture, &reference}) {
        if (auto * gamma = target->ordinary_gamma) {
            auto buffer = ggml_backend_buft_alloc_buffer(input_buft, ggml_backend_buft_get_alloc_size(input_buft, gamma));
            CHECK(buffer != nullptr);
            ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            CHECK(ggml_backend_tensor_alloc(buffer, gamma, ggml_backend_buffer_get_base(buffer)) == GGML_STATUS_SUCCESS);
            std::vector<float> values(ggml_nelements(gamma));
            for (size_t i = 0; i < values.size(); ++i) { values[i] = 0.75f + 0.01f * float(i % 19); }
            ggml_backend_tensor_set(gamma, values.data(), 0, values.size() * sizeof(float));
            target->ordinary_buffers.emplace_back(buffer);
        }
        target->allocate_ordinary_rotation(input_buft);
        if (auto * weight = target->bf16_projection_weight) {
            auto buffer = ggml_backend_buft_alloc_buffer(input_buft, ggml_backend_buft_get_alloc_size(input_buft, weight));
            CHECK(buffer != nullptr);
            ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            CHECK(ggml_backend_tensor_alloc(buffer, weight, ggml_backend_buffer_get_base(buffer)) == GGML_STATUS_SUCCESS);
            const auto bytes = cached_fusion_test_data(weight, 197);
            ggml_backend_tensor_set(weight, bytes.data(), 0, bytes.size());
            target->ordinary_buffers.emplace_back(buffer);
        }
        for (auto * tensor : {target->attention_key, target->attention_value, target->attention_mask}) {
            if (!tensor) { continue; }
            auto buffer = ggml_backend_buft_alloc_buffer(input_buft, ggml_backend_buft_get_alloc_size(input_buft, tensor));
            CHECK(buffer != nullptr);
            ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            CHECK(ggml_backend_tensor_alloc(buffer, tensor, ggml_backend_buffer_get_base(buffer)) == GGML_STATUS_SUCCESS);
            std::vector<ggml_fp16_t> values(ggml_nelements(tensor));
            for (size_t i = 0; i < values.size(); ++i) {
                const float value = tensor == target->attention_mask ? (i % tensor->ne[0] < 2051 ? 0.0f : -INFINITY) :
                    0.003f * (int((i + (tensor == target->attention_key ? 3 : 7)) % 23) - 11);
                values[i] = ggml_fp32_to_fp16(value);
            }
            ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(ggml_fp16_t));
            target->ordinary_buffers.emplace_back(buffer);
        }
    }
    ggml_backend_t backends[]{owner->gpu.get(), owner->cpu.get()};
    owner->sched.reset(ggml_backend_sched_new(backends, nullptr, 2, 256, false, true));
    owner->oracle.reset(ggml_backend_sched_new(backends, nullptr, 2, 256, false, true));
    CHECK(ggml_backend_sched_set_resizable(owner->sched.get(), nullptr));
    const auto snapshot = fixture.manifest();
    if (!native_reference) {
        CHECK(ggml_backend_cuda_moe_candidate_replace_v2(owner->gpu.get(), &snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    }
    for (auto & region : fixture.result.get_moe_regions()) { CHECK(region.place(owner->sched.get(), owner->gpu.get())); }
    for (auto & region : reference.result.get_moe_regions()) { CHECK(region.place(owner->oracle.get(), owner->gpu.get())); }
    if (staged_inputs) {
        ggml_backend_sched_set_tensor_backend(owner->sched.get(), fixture.staged, owner->gpu.get());
        ggml_backend_sched_set_tensor_backend(owner->oracle.get(), reference.staged, owner->gpu.get());
    }
    CHECK(ggml_backend_sched_alloc_graph(owner->sched.get(), fixture.result.get_gf()));
    CHECK(ggml_backend_sched_alloc_graph(owner->oracle.get(), reference.result.get_gf()));
    if (staged_inputs) {
        staged_source_view_checks(device, fixture.staged, stage_api, owner->staged_input.get());
        staged_source_view_checks(device, reference.staged, stage_api, owner->staged_reference.get());
        CHECK(ggml_backend_sched_get_tensor_backend(owner->sched.get(), fixture.staged) == owner->gpu.get());
        CHECK(ggml_backend_sched_get_tensor_backend(owner->oracle.get(), reference.staged) == owner->gpu.get());
    }
    auto certificate = layer_certificate();
    const char * independent = getenv("GGML_TEST_MOE_INDEPENDENT_ROWS");
    const bool independent_rows = source_core && independent && !strcmp(independent, "1");
    if (capacity > 1) {
        certificate.flags = independent_rows ? GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_NONE : GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED;
        certificate.row_semantics = independent_rows ? GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT : GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SPECULATIVE;
        certificate.n_rows = capacity;
        if (independent_rows) { certificate.n_sequences = capacity; }
        if (main_prefill) { certificate.row_semantics = GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL; }
    }
    const auto set_inputs = [&](layer_fixture & target, bool warm, uint32_t step, bool publish_stage = true) {
        std::vector<float> values(size_t(capacity) * signature.n_embd);
        for (size_t i = 0; i < values.size(); ++i) { values[i] = 0.003f * (1 + (i * 7 + i / signature.n_embd * 11 + step) % 31); }
        ggml_backend_tensor_set(target.input, values.data(), 0, values.size() * sizeof(float));
        for (auto * logits : target.logits) {
            std::vector<float> scores(size_t(capacity) * experts, -10.0f);
            for (uint32_t row = 0; row < capacity; ++row) {
                if (profile_adaptation) {
                    for (uint32_t rank = 0; rank < routes; ++rank) { scores[row * experts + rank] = 10.0f - float(rank); }
                } else if (routes == n_used) {
                    scores[row * experts + (warm || row == 0 ? 0 : 2 + (row + step) % (experts - 2))] = 10.0f;
                    scores[row * experts + (warm ? 1 : 2 + (row + step + 1) % (experts - 2))] = 9.0f;
                } else {
                    for (uint32_t rank = 0; rank < routes; ++rank) {
                        scores[row * experts + (rank + (warm ? 0 : (native_reference && capacity == 9 ? step : row + step))) % experts] = 10.0f - float(rank);
                    }
                }
            }
            ggml_backend_tensor_set(logits, scores.data(), 0, scores.size() * sizeof(float));
        }
        if (stage_api) {
            void * stage = &target == &fixture ? owner->staged_input.get() : owner->staged_reference.get();
            auto * staged = static_cast<float *>(stage_api->data(stage));
            for (int i = 0; i < signature.n_embd; ++i) { staged[i] = 0.001f * (int((i + step) % 19) - 9); }
            if (publish_stage) { stage_api->publish(stage); }
        }
    };
    set_inputs(fixture, true, 0);
    fidelity_probe = owner.get();
    owner->delegate = owner->gpu->iface.graph_compute;
    owner->capture_only = native_reference;
    owner->gpu->iface.graph_compute = fidelity_capture_split;
    CHECK(ggml_backend_sched_graph_compute_ext(owner->sched.get(), fixture.result.get_gf(), &certificate) == GGML_STATUS_SUCCESS);
    owner->gpu->iface.graph_compute = owner->delegate;
    fidelity_probe = nullptr;
    CHECK(!owner->graph_nodes.empty());
    if (!native_reference) {
        if (stage_api) { stage_api->publish(owner->staged_input.get()); }
        CHECK(ggml_backend_sched_graph_compute_ext(owner->sched.get(), fixture.result.get_gf(), &certificate) == GGML_STATUS_SUCCESS);
    }
    if (pending_handoff) {
        set_inputs(fixture, false, experts - routes);
        CHECK(ggml_backend_sched_graph_compute_ext(owner->sched.get(), fixture.result.get_gf(), &certificate) == GGML_STATUS_SUCCESS);
    }
    auto * grouped = ggml_cuda_moe_grouped_context_for_test(owner->gpu.get());
    const auto read_residency = [&]() {
        std::vector<int32_t> slots;
        for (const auto * down : fixture.down) {
            ggml_cuda_moe_candidate_group_key key;
            CHECK(grouped->find_down_group_key(down, &key));
            for (int64_t expert = 0; expert < down->ne[2]; ++expert) {
                slots.push_back(ggml_cuda_moe_grouped_context_test_access::device_slot_for_expert(*grouped, key, expert));
            }
        }
        return slots;
    };
    auto initial_residency = read_residency();
    if (independent_rows && capacity > 1) {
        ggml_cuda_moe_graph_plan source_plan, ordinary_plan;
        ggml_cuda_moe_graph_execution execution;
        grouped->compile_graph_plan(&owner->graph, owner->graph.uid, &source_plan, &execution, 0, nullptr, 0, 0, true);
        CHECK(source_plan.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED);
        CHECK(!grouped->bind_graph_plan(&owner->graph, owner->graph.uid, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, source_plan, &execution));
        grouped->compile_graph_plan(&owner->graph, owner->graph.uid, &ordinary_plan, &execution);
        CHECK(ordinary_plan.outcome() != GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED);
        CHECK(read_residency() == initial_residency);
        fprintf(stderr, "test-moe-cache: independent source residency plan rejects ordinary rebind; ordinary capacity and canonical maps unchanged OK\n");
    }
    CHECK(std::any_of(initial_residency.begin(), initial_residency.end(), [](int32_t slot) { return slot >= 0; }));
    if (native_reference) {
        for (uint32_t layer = 0; layer < 2; ++layer) {
            const auto first = initial_residency.begin() + size_t(layer) * experts;
            const uint32_t seeded_experts = pending_handoff ? routes : n_used;
            CHECK(std::count_if(first, first + experts, [](int32_t slot) { return slot >= 0; }) == seeded_experts);
            for (uint32_t expert = 0; expert < experts; ++expert) { CHECK((first[expert] >= 0) == (expert < seeded_experts)); }
        }
        const uint32_t seeded_experts = pending_handoff ? routes : n_used;
        fprintf(stderr, "test-moe-cache: reference seed resident=%u slots=%u routes=%u misses=%u selected=%llu\n",
            seeded_experts, fixture.cache_slots, routes, routes - seeded_experts,
            (unsigned long long) ((uint64_t(routes - seeded_experts) * ggml_moe_fidelity_selection().pcie_num) / 256));
    }
    const auto payload_mismatches = [&]() {
        uint32_t mismatches = 0;
        const auto residency = read_residency();
        for (uint32_t layer = 0; layer < 2; ++layer) {
            ggml_cuda_moe_candidate_group_key key;
            CHECK(grouped->find_down_group_key(fixture.down[layer], &key));
            for (uint32_t expert = 0; expert < experts; ++expert) {
                const int32_t slot = residency[size_t(layer) * experts + expert];
                if (slot < 0) { continue; }
                CHECK(uint32_t(slot) < fixture.cache_slots);
                for (const auto * weight : {fixture.gate_up[layer], fixture.gate[layer], fixture.up[layer], fixture.down[layer]}) {
                    if (!weight) { continue; }
                    const auto * bank = static_cast<const uint8_t *>(ggml_cuda_moe_grouped_context_test_access::device_bank_data(*grouped, key, weight));
                    CHECK(bank);
                    const size_t bytes = weight->nb[2];
                    std::vector<uint8_t> actual(bytes), expected(bytes);
                    CUDA_OK(cudaMemcpy(actual.data(), bank + size_t(slot) * bytes, bytes, cudaMemcpyDeviceToHost));
                    ggml_backend_tensor_get(weight, expected.data(), size_t(expert) * bytes, bytes);
                    mismatches += actual != expected;
                }
            }
        }
        return mismatches;
    };
    if (pending_handoff) {
        CHECK(payload_mismatches() >= 2 * (experts - routes));
        fprintf(stderr, "test-moe-cache: reference handoff has genuine pending replacements with stale reciprocal payloads\n");
    }
    uint64_t generation = 0, shrink = 0;
    ggml_backend_sched_get_buffer_state(owner->sched.get(), &generation, &shrink);
    for (auto & region : fixture.result.get_moe_regions()) {
        CHECK(region.finalize_metadata(owner->sched.get(), fixture.result.get_gf(), 1, generation) == GGML_BACKEND_SCHED_REGION_STATUS_V1_OK);
        std::unique_ptr<llm_graph_moe_hybrid_prepared> metadata;
        CHECK(region.prepare_hybrid_metadata(source_owner, 2, metadata) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        check_original_body(region, metadata->descriptor());
        owner->descriptors.push_back(metadata->descriptor());
        owner->descriptors.back().certificate = owner->graph.execution_certificate;
        owner->metadata.push_back(std::move(metadata));
    }
    for (const auto & descriptor : owner->descriptors) { owner->regions.push_back(&descriptor); }
    if (overlap_fixture) { test_source_overlap_schedule(*owner, fixture); }
    if (scheduler) {
        std::vector<int32_t> ranking(experts);
        for (uint32_t i = 0; i < experts; ++i) { ranking[i] = (i + 3) % experts; }
        std::array<std::vector<uint64_t>, 2> learned_counts;
        for (auto & values : learned_counts) { values.assign(experts, 0); }
        const auto check_learning = [&](uint32_t windows) {
            struct learned_bank {
                ggml_backend_moe_source_identity_v1 identity;
                std::vector<uint64_t> counts;
                std::vector<double> heat;
                std::vector<float> usage;
                std::vector<int32_t> prior;
                uint64_t observations, windows;
            };
            std::vector<ggml_backend_moe_source_identity_v1> identities;
            for (uint32_t layer = 0; layer < 2; ++layer) {
                for (const auto * bank : {fixture.up[layer], fixture.gate[layer], fixture.gate_up[layer], fixture.down[layer]}) {
                    if (bank) { identities.push_back({bank, GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY}); }
                }
            }
            std::vector<learned_bank> saved;
            const auto collect = +[](const ggml_backend_moe_source_learning_v1 * records, uint32_t count, void * data) {
                auto & output = *static_cast<std::vector<learned_bank> *>(data);
                for (uint32_t i = 0; i < count; ++i) {
                    const auto & record = records[i];
                    const auto experts = record.source.n_experts;
                    output.push_back({{record.source.tensor, record.source.domain},
                        {record.source.counts, record.source.counts + experts}, {record.heat, record.heat + experts},
                        {record.usage, record.usage + experts}, {record.prior, record.prior + experts},
                        record.source.observations, record.windows});
                }
                return true;
            };
            const auto deadline = [] { return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count()) + 1000000000; };
            const auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(owner->gpu.get()));
            const auto snapshot = reinterpret_cast<ggml_backend_moe_learning_snapshot_v1_t>(
                ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_LEARNING_SNAPSHOT_V1_PROC_NAME));
            const auto restore = reinterpret_cast<ggml_backend_moe_learning_restore_v1_t>(
                ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_LEARNING_RESTORE_V1_PROC_NAME));
            CHECK(snapshot && restore);
            CHECK(snapshot(owner->gpu.get(), identities.data(), identities.size(), 0, deadline(), collect, &saved));
            CHECK(saved.size() == identities.size());
            for (const auto & record : saved) {
                uint32_t layer = 0;
                while (layer < 2 && record.identity.tensor != fixture.up[layer] && record.identity.tensor != fixture.gate[layer] &&
                        record.identity.tensor != fixture.gate_up[layer] && record.identity.tensor != fixture.down[layer]) { ++layer; }
                CHECK(layer < 2);
                if (record.counts != learned_counts[layer] || record.windows != windows) {
                    fprintf(stderr, "test-moe-cache: learning mismatch source=%s windows=%llu expected=%u observations=%llu counts=",
                        ggml_get_name(record.identity.tensor), (unsigned long long) record.windows, windows, (unsigned long long) record.observations);
                    for (const auto value : record.counts) { fprintf(stderr, "%llu,", (unsigned long long) value); }
                    fprintf(stderr, " observed=");
                    for (const auto value : learned_counts[layer]) { fprintf(stderr, "%llu,", (unsigned long long) value); }
                    fprintf(stderr, "\n");
                }
                CHECK(record.counts == learned_counts[layer] && record.windows == windows);
                CHECK(record.observations == uint64_t(windows) * capacity * routes);
                for (uint32_t rank = 0; rank < experts; ++rank) {
                    CHECK(record.prior[ranking[rank]] == int32_t(rank));
                    CHECK(record.heat[rank] == (windows == 4 ? double(record.counts[rank]) : 0.0));
                    CHECK(record.usage[rank] == float(record.counts[rank]) * (windows == 4 ? 0.7f : 1.0f));
                }
            }
            if (windows < 3) { return; }
            std::vector<ggml_backend_moe_source_learning_v1> records;
            for (const auto & record : saved) {
                records.push_back({sizeof(ggml_backend_moe_source_learning_v1), 1,
                    {record.identity.tensor, record.counts.data(), record.observations, experts, record.identity.domain},
                    record.heat.data(), record.usage.data(), record.prior.data(), record.windows});
            }
            for (const uint32_t slots : {3u, fixture.cache_slots}) {
                ggml_backend_ptr cold(ggml_backend_cuda_init(device));
                auto manifest = fixture.manifest();
                manifest.n_slots = slots;
                CHECK(cold && ggml_backend_cuda_moe_candidate_replace_v2(cold.get(), &manifest) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
                auto * restored = ggml_cuda_moe_grouped_context_for_test(cold.get());
                CHECK(restored && records.size() > 1);
                CHECK(!restore(cold.get(), records.data(), records.size() - 1, 0, deadline()));
                CHECK(restore(cold.get(), records.data(), records.size(), 0, deadline()));
                CHECK(!restore(cold.get(), records.data(), records.size(), 0, deadline()));
                auto expected_ranks = ranking;
                if (windows == 4) { std::stable_partition(expected_ranks.begin(), expected_ranks.end(), [&](int32_t id) { return uint32_t(id) < routes; }); }
                std::vector<int32_t> resumed_seed;
                for (uint32_t layer = 0; layer < 2; ++layer) {
                    ggml_cuda_moe_candidate_group_key key;
                    ggml_cuda_moe_grouped_acquisition acquisition;
                    ggml_cuda_moe_grouped_transaction transaction;
                    CHECK(restored->find_down_group_key(fixture.down[layer], &key));
                    CHECK(restored->acquire_group_resources(key, &acquisition) && restored->begin_group_transaction(acquisition, &transaction));
                    std::vector<uint64_t> counts;
                    std::vector<double> heat;
                    std::vector<int32_t> ranks;
                    uint64_t observations = 0;
                    CHECK(restored->snapshot_source_profile(transaction, counts, heat, ranks, observations));
                    CHECK(counts == learned_counts[layer] && observations == uint64_t(windows) * capacity * routes);
                    CHECK(heat == saved.front().heat && ranks == expected_ranks);
                    resumed_seed = ranks;
                    CHECK(!ggml_cuda_moe_grouped_context_test_access::device_bank_data(*restored, key, fixture.down[layer]));
                    std::vector<learned_bank> rejected;
                    CHECK(!snapshot(cold.get(), identities.data(), identities.size(), 0, deadline(), collect, &rejected) && rejected.empty());
                    CHECK(restored->end_group_transaction(transaction));
                }
                std::vector<learned_bank> cold_snapshot;
                CHECK(snapshot(cold.get(), identities.data(), identities.size(), 0, deadline(), collect, &cold_snapshot) && cold_snapshot.empty());
                fprintf(stderr, "test-moe-cache: learned owner cold restore slots=%u experts=%u windows=%u complete banks/repeat/incomplete/busy/no-arena OK\n", slots, experts, windows);
                cudaStream_t stream;
                CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
                std::shared_ptr<ggml_cuda_moe_graph_plan> plan;
                ggml_cuda_moe_graph_execution execution;
                uint32_t mmids = 0;
                uint64_t fingerprint = 0;
                const auto coverage = restored->certify_graph_coverage(&owner->graph, &mmids, &fingerprint);
                CHECK(coverage && mmids && fingerprint);
                CHECK(restored->prepare_graph_execution(&owner->graph, owner->graph.uid, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED,
                    &plan, &execution, coverage, owner->graph.nodes, mmids, fingerprint, true) != GGML_CUDA_MOE_GRAPH_PREPARE_UNAVAILABLE);
                CHECK(execution.resolve_streams([](void * opaque, const ggml_tensor *) { return *static_cast<cudaStream_t *>(opaque); }, &stream));
                CHECK(restored->begin_graph_dispatch(&execution, GGML_CUDA_MOE_GRAPH_DISPATCH_DIRECT));
                struct resumed_group {
                    ggml_cuda_moe_graph_group_dispatch * group;
                    std::vector<int32_t> map, owners;
                    std::vector<uint32_t> bindings;
                };
                std::vector<resumed_group> resumed;
                resumed.reserve(2);
                ggml_cuda_moe_source_transport * transport = nullptr;
                size_t tile_bytes = 0;
                for (const auto & bank : fixture.tensors) { tile_bytes = std::max(tile_bytes, bank.tensor->nb[2]); }
                for (int i = 0; i < owner->graph.n_nodes; ++i) {
                    auto * group = execution.find_group(owner->graph.nodes[i], nullptr);
                    if (!group || std::any_of(resumed.begin(), resumed.end(), [&](const resumed_group & item) { return item.group == group; })) { continue; }
                    resumed.push_back({group, std::vector<int32_t>(experts), std::vector<int32_t>(slots), std::vector<uint32_t>(group->key.n_banks)});
                    auto & item = resumed.back();
                    CHECK(restored->prepare_source_group(group, stream, item.map.data(), experts, item.owners.data(), slots, true));
                    for (uint32_t bank = 0; bank < item.bindings.size(); ++bank) {
                        CHECK(restored->prepare_source_transport(group->transaction, bank, tile_bytes, &transport, &item.bindings[bank], true));
                    }
                }
                CHECK(resumed.size() == 2 && transport);
                CUDA_OK(cudaStreamSynchronize(stream));
                std::vector<int32_t> next_routes(size_t(capacity) * routes);
                for (size_t i = 0; i < next_routes.size(); ++i) { next_routes[i] = i % routes; }
                std::vector<ggml_cuda_moe_source_profile_update> updates;
                for (auto & item : resumed) {
                    updates.push_back({item.group, resumed_seed.data(), experts, item.map.data(), item.owners.data(),
                        item.bindings.data(), uint32_t(item.bindings.size()), next_routes.data(), uint32_t(next_routes.size()), 0});
                }
                CHECK(restored->update_source_profiles(updates.data(), updates.size(), transport, stream, deadline(), false));
                CHECK(restored->update_source_profiles(updates.data(), updates.size(), transport, stream, deadline(), true));
                CUDA_OK(cudaStreamSynchronize(stream));
                CHECK(restored->finish_source_dispatch(&execution));
                CHECK(restored->release_source_transport(&transport) && !transport);
                std::vector<learned_bank> continued;
                CHECK(snapshot(cold.get(), identities.data(), identities.size(), 0, deadline(), collect, &continued));
                CHECK(continued.size() == saved.size());
                for (const auto & record : continued) {
                    CHECK(record.windows == windows + 1 && record.observations == uint64_t(windows + 1) * capacity * routes);
                    for (uint32_t id = 0; id < experts; ++id) {
                        const uint64_t count = id < routes ? uint64_t(windows + 1) * capacity : 0;
                        const uint64_t heat = id < routes ? uint64_t(4) * capacity : 0;
                        const float usage = id < routes ? float(4 * capacity) * 0.7f + (windows == 4 ? float(capacity) : 0.0f) : 0.0f;
                        CHECK(record.counts[id] == count && record.heat[id] == double(heat) && record.usage[id] == usage);
                        CHECK(record.prior[ranking[id]] == int32_t(id));
                    }
                }
                CUDA_OK(cudaStreamDestroy(stream));
                fprintf(stderr, "test-moe-cache: learned owner resumed slots=%u windows=%u->%u actual preparation/copy/controller/decay/snapshot OK\n", slots, windows, windows + 1);
            }
        };
        std::vector<ggml_backend_moe_static_profile_v1> profiles;
        std::vector<uint64_t> counts(experts);
        for (uint32_t i = 0; i < experts; ++i) { counts[ranking[i]] = experts - i; }
        const uint64_t observations = uint64_t(experts) * (experts + 1) / 2;
        auto borrowed_counts = counts;
        std::vector<ggml_backend_moe_source_statistics_v1> statistics;
        if (static_profile) {
            CHECK(source_core);
            if (source_statistics) {
                for (uint32_t layer = 0; layer < 2; ++layer) {
                    for (const auto * bank : {fixture.up[layer], fixture.gate[layer], fixture.gate_up[layer], fixture.down[layer]}) {
                        if (bank) { statistics.push_back({bank, borrowed_counts.data(), observations, experts, GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY}); }
                    }
                }
            } else { for (const auto * down : fixture.down) { profiles.push_back({down, ranking.data(), experts}); } }
        }
        ggml_backend_moe_hybrid_config_v1 config = {};
        config.struct_size = sizeof(config);
        config.n_threads = 2;
        const char * scheduler_mode = getenv("GGML_TEST_MOE_FIDELITY");
        const bool routed_discovery = source_core && scheduler_mode &&
            (!strcmp(scheduler_mode, "routed-discovery") || !strcmp(scheduler_mode, "routed-new-op") || !strcmp(scheduler_mode, "routed-profile") ||
             !strcmp(scheduler_mode, "routed-adapt") || !strcmp(scheduler_mode, "routed-async") ||
             !strcmp(scheduler_mode, "routed-statistics") || !strcmp(scheduler_mode, "routed-statistics-adapt") || !strcmp(scheduler_mode, "routed-statistics-async"));
        const bool routed_scheduler = routed_discovery ||
            main_prefill || (source_core && scheduler_mode && (!strcmp(scheduler_mode, "routed-scheduler") || !strcmp(scheduler_mode, "routed-new-op-fallback")));
        config.max_regions = owner->descriptors.size() * (routed_scheduler ? 3 : 1);
        config.max_prepared_regions = config.max_regions * (routes + 1);
        config.gpu_miss_quota = quota;
        config.resident_batch = 1;
        config.executor = GGML_BACKEND_MOE_HYBRID_EXECUTOR_V1_FIDELITY;
        config.backend = owner->gpu.get();
        config.source_owner = &source_owner;
        config.profiles = profiles.empty() ? nullptr : profiles.data();
        config.n_profiles = profiles.size();
        config.statistics = statistics.empty() ? nullptr : statistics.data();
        config.n_statistics = statistics.size();
        config.profile_adaptation = profile_adaptation;
        config.cpu_module_acquire = ggml_backend_moe_cpu_module_acquire_v1;
        config.module_retain = ggml_backend_moe_module_retain_v1;
        config.module_release = ggml_backend_moe_module_release_v1;
        config.cpu_flags = GGML_BACKEND_MOE_CPU_SERVICE_FLAG_V1_ALLOW_UNKNOWN_THREAD_STACK_BYTES |
            GGML_BACKEND_MOE_CPU_SERVICE_FLAG_V1_ALLOW_UNPROVEN_RUNTIME_ALLOCATIONS;
        if (profile_adaptation) {
            auto invalid = config;
            invalid.profile_adaptation = 3;
            CHECK(ggml_backend_sched_moe_hybrid_configure_v1(owner->sched.get(), &invalid) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
            invalid = config;
            invalid.n_profiles = 0; invalid.n_statistics = 0;
            CHECK(ggml_backend_sched_moe_hybrid_configure_v1(owner->sched.get(), &invalid) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
        }
        CHECK(ggml_backend_sched_moe_hybrid_configure_v1(owner->sched.get(), &config) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        CHECK(ggml_backend_sched_moe_source_selected_v1(owner->sched.get()) == source_core);
        if (source_core) {
            CHECK(config.max_regions > 1 && config.max_prepared_regions > config.max_regions);
            auto smaller = config;
            --smaller.max_regions;
            --smaller.max_prepared_regions;
            CHECK(ggml_backend_sched_moe_hybrid_configure_v1(owner->sched.get(), &smaller) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
            auto larger = config;
            ++larger.max_regions;
            CHECK(ggml_backend_sched_moe_hybrid_configure_v1(owner->sched.get(), &larger) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
            larger = config;
            ++larger.max_prepared_regions;
            CHECK(ggml_backend_sched_moe_hybrid_configure_v1(owner->sched.get(), &larger) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
            CHECK(ggml_backend_sched_moe_hybrid_configure_v1(owner->sched.get(), &config) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        }
        if (source_statistics) {
            --borrowed_counts[ranking[0]]; ++borrowed_counts[ranking[1]];
            CHECK(ggml_backend_sched_moe_hybrid_configure_v1(owner->sched.get(), &config) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
            for (auto & source : statistics) { source.counts = counts.data(); }
            CHECK(ggml_backend_sched_moe_hybrid_configure_v1(owner->sched.get(), &config) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
            std::fill(borrowed_counts.begin(), borrowed_counts.end(), 0);
            fprintf(stderr, "test-moe-cache: source statistics copied before caller histogram mutation; canonical seed/policy checks follow\n");
        }
        const uint32_t initial_capacity = capacity;
        const uint32_t phases = capacity == 5 && !staged_inputs && !delayed_cancel && !early_fallback ? 3 : 1;
        const auto rebuild = [&](layer_fixture & target) {
            target.build_graph(false, nullptr, nullptr, source_core && !overlap_fixture, capacity, empty_prefix, bf16_columns, flash_attention, cpu_oracle,
                ggml_moe_fidelity_selection().source_pool && !cpu_oracle);
            if (auto * gamma = target.ordinary_gamma) {
                auto buffer = ggml_backend_buft_alloc_buffer(input_buft, ggml_backend_buft_get_alloc_size(input_buft, gamma));
                CHECK(buffer);
                ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                CHECK(ggml_backend_tensor_alloc(buffer, gamma, ggml_backend_buffer_get_base(buffer)) == GGML_STATUS_SUCCESS);
                std::vector<float> values(ggml_nelements(gamma));
                for (size_t i = 0; i < values.size(); ++i) { values[i] = 0.75f + 0.01f * float(i % 19); }
                ggml_backend_tensor_set(gamma, values.data(), 0, values.size() * sizeof(float));
                target.ordinary_buffers.emplace_back(buffer);
            }
            target.allocate_ordinary_rotation(input_buft);
            auto * graph = target.result.get_gf();
            for (int i = 0; i < graph->n_nodes; ++i) { graph->nodes[i]->flags &= ~GGML_TENSOR_FLAG_OUTPUT; }
            ggml_set_output(target.output[1]);
            if (target.ordinary_norm_output) { ggml_set_output(target.ordinary_norm_output); }
            if (source_core && !overlap_fixture) {
                ggml_set_output(ggml_get_tensor(target.result.get_ctx(), "hybrid_fusion_retained"));
            }
        };
        const auto allocate_inputs = [&]() {
            for (auto * tensor : {fixture.input, fixture.logits[0], fixture.logits[1]}) {
                auto buffer = ggml_backend_buft_alloc_buffer(input_buft, ggml_backend_buft_get_alloc_size(input_buft, tensor));
                CHECK(buffer && ggml_backend_tensor_alloc(buffer, tensor, ggml_backend_buffer_get_base(buffer)) == GGML_STATUS_SUCCESS);
                owner->input_buffers.emplace_back(buffer);
            }
        };
        const auto allocate_regions = [&]() {
            if (routed_discovery) {
                std::vector<llama_moe_source_group> sources;
                for (const auto & tensor : fixture.tensors) {
                    if (tensor.status != GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE) { continue; }
                    llama_moe_source_group source = {};
                    source.layout = GGML_BACKEND_MOE_CANDIDATE_LAYOUT_ROUTED_MATRIX;
                    source.domain = GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY;
                    source.layer = tensor.group_index;
                    source.banks.push_back({const_cast<ggml_tensor *>(tensor.tensor),
                        GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_ROUTED_WEIGHT, tensor.status});
                    sources.push_back(std::move(source));
                }
                fixture.result.get_moe_regions().clear();
                CHECK(fixture.result.discover_moe_regions(sources, +[](const ggml_tensor * tensor) {
                    return tensor && tensor->op == GGML_OP_NONE && ggml_is_quantized(tensor->type);
                }));
                CHECK(fixture.result.get_moe_regions().size() == config.max_regions);
                fprintf(stderr, "test-moe-cache: %zu actual original projections discovered without builder bodies\n", fixture.result.get_moe_regions().size());
            }
            for (auto & region : fixture.result.get_moe_regions()) { CHECK(region.place(owner->sched.get(), owner->gpu.get())); }
            CHECK(ggml_backend_sched_alloc_graph(owner->sched.get(), fixture.result.get_gf()));
            ggml_backend_sched_get_buffer_state(owner->sched.get(), &generation, &shrink);
            for (auto & region : fixture.result.get_moe_regions()) {
                CHECK(region.finalize_metadata(owner->sched.get(), fixture.result.get_gf(), certificate.owner_generation, generation) == GGML_BACKEND_SCHED_REGION_STATUS_V1_OK);
            }
        };
        const auto prepare_regions = [&](const ggml_graph_execution_certificate * supplied) {
            if (fixture.routed_backend_stage) {
                uint32_t stages = 0;
                for (int i = 0; i < fixture.result.get_gf()->n_nodes; ++i) {
                    const auto * node = fixture.result.get_gf()->nodes[i];
                    if (strcmp(node->name, "routed_backend_scale") && strcmp(node->name, "routed_backend_tanh")) { continue; }
                    CHECK((node->op == GGML_OP_SCALE || (node->op == GGML_OP_UNARY && ggml_get_unary_op(node) == GGML_UNARY_OP_TANH)) &&
                        ggml_backend_dev_supports_op(ggml_backend_get_device(owner->gpu.get()), node) &&
                        ggml_backend_dev_supports_op(ggml_backend_get_device(owner->cpu.get()), node));
                    ++stages;
                }
                CHECK(stages == 4);
                fprintf(stderr, "test-moe-cache: four original SCALE/TANH stages outside combined-body matcher retained for shared backend execution\n");
            }
            if (source_core && empty_prefix) {
                uint32_t empty_nodes = 0;
                const auto * graph = fixture.result.get_gf();
                for (int i = 0; i < graph->n_nodes; ++i) {
                    const auto * node = graph->nodes[i];
                    if (strcmp(node->name, "fidelity_empty_scale") == 0 || strcmp(node->name, "fidelity_empty_copy") == 0) {
                        CHECK(ggml_is_empty(node) && (node->flags & GGML_TENSOR_FLAG_COMPUTE));
                        CHECK(node->op == GGML_OP_SCALE || node->op == GGML_OP_CPY);
                        ++empty_nodes;
                    }
                }
                CHECK(empty_nodes == 2);
                fprintf(stderr, "test-moe-cache: source-core empty SCALE/CPY compute nodes retained for normal no-op parity\n");
            }
            if (source_core) { CHECK(ggml_backend_sched_moe_hybrid_configure_v1(owner->sched.get(), &config) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK); }
            if (routed_scheduler) {
                uint32_t count = 0;
                for (const auto & region : fixture.result.get_moe_regions()) {
                    if (fixture.routed_backend_stage && !routed_discovery) {
                        const auto residency = read_residency();
                        CHECK(region.prepare_hybrid(owner->sched.get(), source_owner, 2, supplied) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION);
                        CHECK(read_residency() == residency);
                        fprintf(stderr, "test-moe-cache: optional combined body declined; original SCALE/TANH and canonical residency retained for routed preparation\n");
                    }
                    std::vector<std::unique_ptr<llm_graph_moe_hybrid_prepared>> projections;
                    CHECK(region.prepare_routed_metadata(source_owner, 2, projections) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
                    for (const auto & projection : projections) {
                        auto descriptor = projection->descriptor();
                        if (supplied) {
                            descriptor.certificate = *supplied;
                            descriptor.certificate.source_graph_uid = descriptor.source_graph_uid;
                            descriptor.certificate.split_graph_uid = descriptor.split_graph_uid;
                        }
                        auto invalid = descriptor;
                        ++invalid.allocator_generation;
                        CHECK(ggml_backend_sched_moe_hybrid_prepare_v1(owner->sched.get(), &invalid) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
                        invalid = descriptor;
                        ++invalid.split_graph_uid;
                        CHECK(ggml_backend_sched_moe_hybrid_prepare_v1(owner->sched.get(), &invalid) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
                        if (fixture.routed_backend_stage && !routed_discovery) { ++count; continue; }
                        const int32_t status = ggml_backend_sched_moe_hybrid_prepare_v1(owner->sched.get(), &descriptor);
                        if (status != GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK) {
                            fprintf(stderr, "test-moe-cache: original routed scheduler status=%d rows=%u routes=%u input_period=%lld\n",
                                status, descriptor.geometry.row_capacity, descriptor.geometry.routes_per_row,
                                (long long) descriptor.activation->ne[1]);
                        }
                        CHECK(status == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
                        ++count;
                    }
                    if (fixture.routed_backend_stage && !routed_discovery) {
                        const auto residency = read_residency();
                        CHECK(region.prepare_hybrid(owner->sched.get(), source_owner, 2, supplied, true) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
                        CHECK(read_residency() == residency);
                        fprintf(stderr, "test-moe-cache: production region preparation retains original operations after optional-body decline OK\n");
                    }
                }
                fprintf(stderr, "test-moe-cache: %u original routed descriptors prepared for complementary CPU/GPU execution; stale allocator/split rejected\n", count);
                return;
            }
            for (const auto & region : fixture.result.get_moe_regions()) {
                std::unique_ptr<llm_graph_moe_hybrid_prepared> metadata;
                CHECK(region.prepare_hybrid_metadata(source_owner, 2, metadata) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
                auto descriptor = metadata->descriptor();
                if (supplied) {
                    descriptor.certificate = *supplied;
                    descriptor.certificate.source_graph_uid = descriptor.source_graph_uid;
                    descriptor.certificate.split_graph_uid = descriptor.split_graph_uid;
                }
                auto * node = const_cast<ggml_tensor *>(descriptor.body_query->body_nodes[0]);
                const auto original = *node;
                const auto reject_body = [&]() {
                    CHECK(ggml_backend_sched_moe_hybrid_prepare_v1(owner->sched.get(), &descriptor) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
                    *node = original;
                };
                node->op_params[0] ^= 1; reject_body();
                ++node->ne[1]; reject_body();
                node->src[0] = node; reject_body();
                auto * input = const_cast<ggml_tensor *>(descriptor.body_query->dynamic_inputs[0]);
                const size_t stride = input->nb[2];
                ++input->nb[2]; reject_body(); input->nb[2] = stride;
                auto * graph = const_cast<ggml_cgraph *>(descriptor.body_query->graph);
                const auto graph_uid = graph->uid;
                ++graph->uid; reject_body(); graph->uid = graph_uid;
                auto changed_query = *descriptor.body_query;
                const ggml_tensor * changed_output = changed_query.body_nodes[0];
                changed_query.live_outputs = &changed_output;
                descriptor.body_query = &changed_query; reject_body(); descriptor.body_query = metadata->descriptor().body_query;
                const int32_t status = region.prepare_hybrid(owner->sched.get(), source_owner, 2, supplied);
                fprintf(stderr, "test-moe-cache: hybrid prepare source_core=%d layer=%d status=%d route_rows=%lld route_width=%lld route_pitch=%zu\n",
                    int(source_core), region.layer, status, (long long) region.route->ne[1],
                    (long long) region.route->ne[0], region.route->nb[1]);
                CHECK(status == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
            }
        };
        const auto reject = [&](const ggml_graph_execution_certificate & dispatched) {
            std::vector<uint8_t> sentinel(ggml_nbytes(fixture.output[1]), 0x6b), observed(sentinel.size());
            ggml_backend_tensor_set(fixture.output[1], sentinel.data(), 0, sentinel.size());
            CHECK(ggml_backend_sched_graph_compute_ext(owner->sched.get(), fixture.result.get_gf(), &dispatched) == GGML_STATUS_FAILED);
            ggml_backend_tensor_get(fixture.output[1], observed.data(), 0, observed.size());
            CHECK(observed == sentinel);
        };
        if (routed_discovery) {
            ggml_backend_sched_reset(owner->sched.get());
            owner->regions.clear();
            owner->descriptors.clear();
            owner->metadata.clear();
            owner->graph_nodes.clear();
            owner->input_buffers.clear();
            rebuild(fixture);
            allocate_inputs();
            allocate_regions();
        }
        const bool retained_hook_test = source_core && !profile_adaptation && !staged_inputs && !delayed_cancel && !early_fallback && !overlap_fixture;
        std::atomic<uint32_t> hook_calls{0};
        struct body_observation {
            std::atomic<uint32_t> * calls;
            std::vector<int32_t> owners;
            bool enabled;
            bool prefill;
            std::vector<std::vector<uint8_t>> masks;
            std::vector<std::vector<int32_t>> ids;
            std::mutex overlap_mutex{};
            std::condition_variable overlap_condition{};
            bool check_overlap = false, cpu_started = false, gpu_issued = false, overlap_timeout = false;
        } observed_body{&hook_calls, {}, generic_body, main_prefill, {}, {}};
        observed_body.check_overlap = main_prefill && capacity == 2;
        uint32_t previous_hook_calls = 0;
        std::vector<int32_t> learned_residency;
        struct device_barrier {
            uint32_t * value = nullptr;
            cudaStream_t release_stream = nullptr;
            PFN_cuStreamWaitValue32_v11070 wait = nullptr;
            std::atomic<bool> entered{false};
        } held_adaptation;
        if (profile_adaptation == 2) {
            cudaDriverEntryPointQueryResult query;
#if CUDART_VERSION >= 12050
            CUDA_OK(cudaGetDriverEntryPointByVersion("cuStreamWaitValue32", reinterpret_cast<void **>(&held_adaptation.wait), 11070, cudaEnableDefault, &query));
#else
            CUDA_OK(cudaGetDriverEntryPoint("cuStreamWaitValue32", reinterpret_cast<void **>(&held_adaptation.wait), cudaEnableDefault, &query));
#endif
            CHECK(query == cudaDriverEntryPointSuccess && held_adaptation.wait);
            CUDA_OK(cudaMalloc(reinterpret_cast<void **>(&held_adaptation.value), sizeof(uint32_t)));
            CUDA_OK(cudaMemset(held_adaptation.value, 0, sizeof(uint32_t)));
            CUDA_OK(cudaStreamCreateWithFlags(&held_adaptation.release_stream, cudaStreamNonBlocking));
        }
        uint64_t pending_cpu_routes = 0;
        for (uint32_t phase = 0; phase < phases; ++phase) {
            if (phase) {
                ggml_backend_sched_reset(owner->sched.get());
                ggml_backend_sched_reset(owner->oracle.get());
                owner->regions.clear();
                owner->descriptors.clear();
                owner->metadata.clear();
                owner->graph_nodes.clear();
                owner->input_buffers.clear();
                capacity = phase == 1 ? 1 : initial_capacity;
                certificate = layer_certificate();
                if (capacity > 1) {
                    certificate.flags = GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED;
                    certificate.row_semantics = GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SPECULATIVE;
                    certificate.n_rows = capacity;
                }
                rebuild(fixture);
                rebuild(reference);
                allocate_inputs();
                allocate_regions();
                for (auto & region : reference.result.get_moe_regions()) { CHECK(region.place(owner->oracle.get(), owner->gpu.get())); }
                CHECK(ggml_backend_sched_alloc_graph(owner->oracle.get(), reference.result.get_gf()));
            }
            if (phase == 1) {
                prepare_regions(nullptr);
                set_inputs(fixture, false, 0);
                reject(certificate);
                ggml_backend_sched_reset(owner->sched.get());
                owner->input_buffers.clear();
                rebuild(fixture);
                allocate_inputs();
                allocate_regions();
            }
            prepare_regions(&certificate);
            if (retained_hook_test && phase == 0) {
                CHECK(ggml_backend_sched_moe_hybrid_set_test_hook_v1(owner->sched.get(),
                    +[](void * opaque, uint32_t event, uint64_t, void * view) {
                        auto & observed = *static_cast<body_observation *>(opaque);
                        if (observed.check_overlap && (event == GGML_BACKEND_MOE_HYBRID_TEST_CPU_ADMITTED ||
                                event == GGML_BACKEND_MOE_HYBRID_TEST_GPU_ENQUEUED)) {
                            std::unique_lock<std::mutex> lock(observed.overlap_mutex);
                            if (event == GGML_BACKEND_MOE_HYBRID_TEST_CPU_ADMITTED && !observed.cpu_started) {
                                observed.cpu_started = true; observed.overlap_condition.notify_all();
                                if (!observed.overlap_condition.wait_for(lock, std::chrono::seconds(5), [&] { return observed.gpu_issued; })) {
                                    observed.overlap_timeout = true; return false;
                                }
                            } else if (event == GGML_BACKEND_MOE_HYBRID_TEST_GPU_ENQUEUED && !observed.gpu_issued) {
                                const bool started = observed.overlap_condition.wait_for(lock, std::chrono::seconds(5), [&] { return observed.cpu_started; });
                                observed.gpu_issued = true; observed.overlap_condition.notify_all();
                                if (!started) { observed.overlap_timeout = true; return false; }
                            }
                        }
                        if (event == GGML_BACKEND_MOE_HYBRID_TEST_CPU_ADMITTED || event == GGML_BACKEND_MOE_HYBRID_TEST_CPU_BEFORE_COMMIT ||
                                event == GGML_BACKEND_MOE_HYBRID_TEST_CPU_AFTER_COMMIT) {
                            observed.calls->fetch_add(1, std::memory_order_relaxed);
                        }
                        if (observed.enabled && event == GGML_BACKEND_MOE_HYBRID_TEST_SOURCE_ACCESS) {
                            const auto & access = *static_cast<const ggml_backend_moe_source_access_v1 *>(view);
                            if (access.layer == 0) {
                                observed.owners.resize(access.n_routes);
                                for (uint32_t i = 0; i < access.n_routes; ++i) {
                                    CHECK(access.route_indices[i] < access.n_distinct);
                                    observed.owners[i] = access.classes[access.route_indices[i]];
                                }
                            }
                        }
                        if (observed.prefill && event == GGML_BACKEND_MOE_HYBRID_TEST_SOURCE_ACCESS) {
                            const auto & access = *static_cast<const ggml_backend_moe_source_access_v1 *>(view);
                            CHECK(access.layer < access.n_layers);
                            observed.masks.resize(access.n_layers); observed.ids.resize(access.n_layers);
                            auto & mask = observed.masks[access.layer];
                            auto & ids = observed.ids[access.layer];
                            mask.assign(access.source_witness->ne[2], 0); ids.resize(access.n_routes);
                            for (uint32_t i = 0; i < access.n_distinct; ++i) {
                                CHECK(access.expert_ids[i] >= 0 && size_t(access.expert_ids[i]) < mask.size());
                                mask[access.expert_ids[i]] = access.classes[i] == 2;
                            }
                            for (uint32_t i = 0; i < access.n_routes; ++i) {
                                CHECK(access.route_indices[i] < access.n_distinct);
                                ids[i] = access.expert_ids[access.route_indices[i]];
                            }
                        }
                        return true;
                    }, &observed_body));
            }
            if (profile_adaptation == 2 && phase == 0) {
                CHECK(ggml_backend_sched_moe_hybrid_set_test_hook_v1(owner->sched.get(),
                    +[](void * data, uint32_t event, uint64_t rounds, void * stream) {
                        if (event != GGML_BACKEND_MOE_HYBRID_TEST_SOURCE_ADAPT_BEFORE_COMPLETE || rounds != 4) { return true; }
                        auto & barrier = *static_cast<device_barrier *>(data);
                        const bool queued = barrier.wait(reinterpret_cast<CUstream>(stream), reinterpret_cast<CUdeviceptr>(barrier.value), 1, CU_STREAM_WAIT_VALUE_GEQ) == CUDA_SUCCESS;
                        barrier.entered.store(queued, std::memory_order_release);
                        return queued;
                    }, &held_adaptation));
            }
            uint64_t captures = 0;
            uint64_t owned_program = 0;
            uint32_t unused_copy_calls = 0;
            if (source_core && staged_inputs) {
                ggml_backend_sched_set_copy_callback(owner->sched.get(),
                    +[](ggml_backend_t, const ggml_tensor *, ggml_tensor *, ggml_cgraph *, void * data) {
                        ++*static_cast<uint32_t *>(data);
                        return false;
                    }, &unused_copy_calls);
            }
            for (uint32_t step = 0; step < (profile_adaptation == 2 && phase == 0 ? 10u : static_profile ? 4u : 2u); ++step) {
                set_inputs(fixture, false, step);
                set_inputs(reference, false, step);
                auto oracle_certificate = certificate;
                if (main_prefill) { oracle_certificate.flags = GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_NONE; }
                CHECK(ggml_backend_sched_graph_compute_ext(owner->oracle.get(), reference.result.get_gf(), &oracle_certificate) == GGML_STATUS_SUCCESS);
                if (overlap_fixture && !delayed_cancel && step == 1 && getenv("GGML_MOE_SOURCE_TEST_OVERLAP_PROBE") &&
                        !strcmp(getenv("GGML_MOE_SOURCE_TEST_OVERLAP_PROBE"), "1")) {
                    struct held_overlap {
                        std::mutex mutex;
                        std::condition_variable condition;
                        const std::atomic<uint32_t> * completed = nullptr;
                        bool entered = false, released = false, timed_out = false;
                        static bool hook(void * opaque, uint32_t event, uint64_t, void * view) {
                            auto & self = *static_cast<held_overlap *>(opaque);
                            if (event == GGML_BACKEND_MOE_HYBRID_TEST_SOURCE_OVERLAP_PROBE) {
                                const auto & probe = *static_cast<const ggml_cuda_moe_source_overlap_probe *>(view);
                                if (probe.layer == 0) { self.completed = probe.completed; }
                            }
                            if (event != GGML_BACKEND_MOE_HYBRID_TEST_CPU_AFTER_COMMIT || self.entered) { return true; }
                            std::unique_lock<std::mutex> lock(self.mutex);
                            self.entered = true; self.condition.notify_all();
                            self.timed_out = !self.condition.wait_for(lock, std::chrono::seconds(1), [&] { return self.released; });
                            return !self.timed_out;
                        }
                    } held;
                    std::vector<uint8_t> sentinel(ggml_nbytes(fixture.output[1]), 0x6b), observed(sentinel.size());
                    ggml_backend_tensor_set(fixture.output[1], sentinel.data(), 0, sentinel.size());
                    CHECK(ggml_backend_sched_moe_hybrid_set_test_hook_v1(owner->sched.get(), held_overlap::hook, &held));
                    ggml_status status = GGML_STATUS_FAILED;
                    std::thread caller([&] {
                        CUDA_OK(cudaSetDevice(device));
                        status = ggml_backend_sched_graph_compute_ext(owner->sched.get(), fixture.result.get_gf(), &certificate);
                    });
                    {
                        std::unique_lock<std::mutex> lock(held.mutex);
                        const bool entered = held.condition.wait_for(lock, std::chrono::seconds(10), [&] { return held.entered; });
                        if (!entered) { held.released = true; held.condition.notify_all(); lock.unlock(); caller.join(); CHECK(entered); }
                        bool progressed = false;
                        if (held.completed) {
                            const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
                            while (!(progressed = held.completed->load(std::memory_order_acquire) == 1) &&
                                    std::chrono::steady_clock::now() < deadline) { std::this_thread::yield(); }
                        }
                        cudaStream_t observation;
                        cudaEvent_t copied;
                        CUDA_OK(cudaStreamCreateWithFlags(&observation, cudaStreamNonBlocking));
                        CUDA_OK(cudaEventCreateWithFlags(&copied, cudaEventDisableTiming));
                        CUDA_OK(cudaMemcpyAsync(observed.data(), fixture.output[1]->data, observed.size(), cudaMemcpyDeviceToHost, observation));
                        CUDA_OK(cudaEventRecord(copied, observation));
                        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
                        cudaError_t copied_status;
                        do { copied_status = cudaEventQuery(copied); } while (copied_status == cudaErrorNotReady && std::chrono::steady_clock::now() < deadline);
                        held.released = true; held.condition.notify_all(); lock.unlock(); caller.join();
                        CUDA_OK(cudaStreamSynchronize(observation));
                        CUDA_OK(cudaEventDestroy(copied)); CUDA_OK(cudaStreamDestroy(observation));
                        CHECK(copied_status == cudaSuccess && observed == sentinel);
                        CHECK(!held.completed || progressed);
                        fprintf(stderr, "test-moe-cache: source overlap held CPU public sentinel/progress=%d mapping_available=%d arm=%u OK\n",
                            int(progressed), int(held.completed != nullptr), arm);
                    }
                    CHECK(!held.timed_out && status == GGML_STATUS_SUCCESS);
                    CHECK(ggml_backend_sched_moe_hybrid_set_test_hook_v1(owner->sched.get(), nullptr, nullptr));
                } else if (source_core && staged_inputs && step == 0) {
                    struct held_stage {
                        std::mutex mutex;
                        std::condition_variable condition;
                        bool launched = false, released = false, timed_out = false;
                        static bool hook(void * opaque, uint32_t phase, uint64_t, void *) {
                            auto & self = *static_cast<held_stage *>(opaque);
                            if (phase != GGML_BACKEND_MOE_HYBRID_TEST_SOURCE_GRAPH_LAUNCHED) { return true; }
                            std::unique_lock<std::mutex> lock(self.mutex);
                            if (self.launched) { return true; }
                            self.launched = true; self.condition.notify_all();
                            self.timed_out = !self.condition.wait_for(lock, std::chrono::seconds(5), [&] { return self.released; });
                            return !self.timed_out;
                        }
                    } held;
                    ggml_cuda_source_staged_input_view view;
                    CHECK(ggml_cuda_staged_input_prepare_source_view(device, fixture.staged, view));
                    const_cast<std::atomic<uint32_t> *>(static_cast<const std::atomic<uint32_t> *>(view.host_flag))->store(0, std::memory_order_release);
                    CHECK(!stage_pending(owner->staged_input.get()));
                    CHECK(ggml_backend_sched_moe_hybrid_set_test_hook_v1(owner->sched.get(), held_stage::hook, &held));
                    ggml_status status = GGML_STATUS_FAILED;
                    std::thread caller([&] {
                        CUDA_OK(cudaSetDevice(device));
                        status = ggml_backend_sched_graph_compute_ext(owner->sched.get(), fixture.result.get_gf(), &certificate);
                    });
                    {
                        std::unique_lock<std::mutex> lock(held.mutex);
                        const bool launched = held.condition.wait_for(lock, std::chrono::seconds(10), [&] { return held.launched; });
                        if (!launched) { held.released = true; held.condition.notify_all(); lock.unlock(); caller.join(); CHECK(launched); }
                        ggml_backend_moe_hybrid_state_v1 state = {};
                        state.struct_size = sizeof(state);
                        CHECK(ggml_backend_sched_moe_hybrid_state_v1(owner->sched.get(), &state) && state.dispatch_active == 1 && state.cpu_active_jobs == 0);
                        CHECK(owner->staged_submissions.load(std::memory_order_relaxed) > 0);
                        CHECK(!stage_pending(owner->staged_input.get()));
                        stage_api->publish(owner->staged_input.get());
                        held.released = true; held.condition.notify_all();
                    }
                    caller.join();
                    CHECK(!held.timed_out && status == GGML_STATUS_SUCCESS && !stage_pending(owner->staged_input.get()));
                    CHECK(ggml_backend_sched_moe_hybrid_set_test_hook_v1(owner->sched.get(), nullptr, nullptr));
                    const auto staged = active_grouped_tensor_values(fixture.staged);
                    CHECK(memcmp(staged.data(), stage_api->data(owner->staged_input.get()), view.bytes) == 0);
                } else if (source_core && early_fallback && step == 0) {
                    host_barrier pending;
                    CHECK(ggml_backend_sched_moe_hybrid_set_test_hook_v1(owner->sched.get(),
                        [](void * data, uint32_t phase, uint64_t, void * stream) {
                            return phase != GGML_BACKEND_MOE_HYBRID_TEST_SOURCE_COMPUTE_RETURNING ||
                                cudaLaunchHostFunc(static_cast<cudaStream_t>(stream), wait_on_host_barrier, data) == cudaSuccess;
                        }, &pending));
                    CHECK(ggml_backend_sched_graph_compute_async_ext(owner->sched.get(), fixture.result.get_gf(), &certificate) == GGML_STATUS_SUCCESS);
                    const auto started = std::chrono::steady_clock::now();
                    CHECK(ggml_backend_sched_moe_source_drain_v1(owner->sched.get()) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_TIMEOUT);
                    CHECK(std::chrono::steady_clock::now() - started < std::chrono::seconds(7));
                    CHECK(pending.entered.load(std::memory_order_acquire));
                    pending.released.store(true, std::memory_order_release);
                    CHECK(ggml_backend_sched_moe_source_drain_v1(owner->sched.get()) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK);
                    CHECK(ggml_backend_sched_moe_hybrid_set_test_hook_v1(owner->sched.get(), nullptr, nullptr));
                    fprintf(stderr, "test-moe-cache: source-core early unsupported fallback actual-stream retained/finite drain OK\n");
                } else if (source_core && staged_inputs && step == 1) {
                    CHECK(owned_program);
                    auto wrong = certificate;
                    ++wrong.owner_generation;
                    ggml_backend_moe_hybrid_state_v1 before = {}, after = {};
                    before.struct_size = sizeof(before); after.struct_size = sizeof(after);
                    CHECK(ggml_backend_sched_moe_hybrid_state_v1(owner->sched.get(), &before));
                    CHECK(ggml_backend_sched_moe_source_program_compute_v1(owner->sched.get(), UINT64_MAX, &certificate) == GGML_STATUS_FAILED);
                    CHECK(ggml_backend_sched_moe_source_program_compute_v1(owner->sched.get(), owned_program, &wrong) == GGML_STATUS_FAILED);
                    CHECK(ggml_backend_sched_moe_source_program_compute_v1(owner->oracle.get(), owned_program, &certificate) == GGML_STATUS_FAILED);
                    CHECK(ggml_backend_sched_moe_hybrid_state_v1(owner->sched.get(), &after));
                    CHECK(before.window_launches == after.window_launches && before.window_waits == after.window_waits);
                    auto * retained = ggml_get_tensor(fixture.result.get_ctx(), "hybrid_fusion_retained");
                    CHECK(retained);
                    retained->op_params[0] ^= 1;
                    reject(certificate);
                    auto * destination = fixture.output[1]->data;
                    fixture.output[1]->data = reinterpret_cast<void *>(uintptr_t(destination) ^ 1);
                    const auto status = ggml_backend_sched_moe_source_program_compute_v1(owner->sched.get(), owned_program, &certificate);
                    fixture.output[1]->data = destination;
                    retained->op_params[0] ^= 1;
                    CHECK(status == GGML_STATUS_SUCCESS);
                } else { CHECK(ggml_backend_sched_graph_compute_ext(owner->sched.get(), fixture.result.get_gf(), &certificate) == GGML_STATUS_SUCCESS); }
                if (source_core && staged_inputs && step == 0) {
                    CHECK(ggml_backend_sched_moe_source_program_bind_v1(owner->sched.get(), fixture.result.get_gf(), &certificate,
                        &owned_program) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK && owned_program);
                }
                if (profile_adaptation == 2 && phase == 0 && step == 3) {
                    const auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(1);
                    while (!held_adaptation.entered.load() && std::chrono::steady_clock::now() < limit) { std::this_thread::yield(); }
                    CHECK(held_adaptation.entered.load());
                    const auto expiry = [] { return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()) + 5'000'000ull; };
                    CHECK(grouped->complete_source_adaptation(false, expiry()));
                    CHECK(!grouped->complete_source_adaptation(true, expiry()));
                    const auto pending_map = read_residency();
                    CHECK(std::count(pending_map.begin(), pending_map.end(), -1) == std::count(initial_residency.begin(), initial_residency.end(), -1) + 6);
                }
                if (profile_adaptation == 2 && phase == 0 && step == 8) {
                    ggml_backend_sched_t clone = nullptr;
                    CHECK(ggml_backend_sched_moe_source_clone_v1(owner->sched.get(), &clone) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK);
                    std::atomic<bool> closed{false};
                    int32_t close_status = -1;
                    std::thread closer([&] {
                        CUDA_OK(cudaSetDevice(device));
                        close_status = ggml_backend_sched_moe_source_free_v1(&clone);
                        closed.store(true);
                    });
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                    CHECK(!closed.load());
                    CUDA_OK(cudaMemsetAsync(held_adaptation.value, 1, sizeof(uint32_t), held_adaptation.release_stream));
                    CUDA_OK(cudaStreamSynchronize(held_adaptation.release_stream));
                    closer.join();
                    CHECK(close_status == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK && !clone);
                    CHECK(ggml_backend_sched_moe_hybrid_set_test_hook_v1(owner->sched.get(), nullptr, nullptr));
                    fprintf(stderr, "test-moe-cache: asynchronous held copies stay misses, next window executes, bounded wait retains, close waits for publication OK\n");
                }
                const auto actual = active_grouped_tensor_values(fixture.output[1]);
                if (generic_body && phase == 0) { check_source_body_arithmetic(fixture, reference, observed_body.owners); }
                if (fixture.ordinary_norm_output) {
                    CHECK(reference.ordinary_norm_output);
                    const auto norm_actual = active_grouped_tensor_values(fixture.ordinary_norm_output);
                    const auto norm_expected = active_grouped_tensor_values(reference.ordinary_norm_output);
                    if (fixture.ordinary_scratch && norm_actual != norm_expected) {
                        double maximum = 0, error = 0, energy = 0;
                        CHECK(norm_actual.size() == norm_expected.size());
                        for (size_t i = 0; i < norm_actual.size(); ++i) {
                            const double difference = double(norm_actual[i]) - norm_expected[i];
                            maximum = std::max(maximum, std::abs(difference));
                            error += difference * difference;
                            energy += double(norm_expected[i]) * norm_expected[i];
                        }
                        fprintf(stderr, "test-moe-cache: scratch norm comparison max_abs=%.9g nmse=%.9g\n", maximum, error / std::max(energy, 1e-30));
                    }
                    CHECK(norm_actual == norm_expected);
                }
                if (static_profile) {
                    if (phase == 0 && step == 0) { initial_residency = read_residency(); }
                    const auto residency = read_residency();
                    if (profile_adaptation && phase && step == 0) { CHECK(residency == learned_residency); }
                    if (profile_adaptation && phase == 0 && step == 3) { CHECK(residency != initial_residency); }
                    learned_residency = residency;
                    for (uint32_t layer = 0; layer < 2; ++layer) {
                        ggml_cuda_moe_candidate_group_key key;
                        ggml_cuda_moe_candidate_group_info info;
                        CHECK(grouped->find_down_group_key(fixture.down[layer], &key) && grouped->get_group(key, &info));
                        std::vector<int32_t> owners(info.n_slots, -1);
                        for (uint32_t expert = 0; expert < experts; ++expert) {
                            const int32_t slot = residency[size_t(layer) * experts + expert];
                            if (slot < 0) { CHECK(slot == -1); continue; }
                            CHECK(uint32_t(slot) < info.n_slots && owners[slot] == -1);
                            owners[slot] = expert;
                        }
                        for (uint32_t slot = 0; slot < info.n_slots; ++slot) {
                            CHECK(owners[slot] >= 0 || (profile_adaptation == 2 && phase == 0 && step >= 3 && step < 8));
                            if (!profile_adaptation || (phase == 0 && step < 3)) { CHECK(owners[slot] == ranking[slot]); }
                        }
                        for (uint32_t bank = 0; bank < info.n_resource_banks; ++bank) {
                            ggml_cuda_moe_candidate_bank_info descriptor;
                            // These fixtures have no auxiliary banks.
                            const uint32_t roles[] = {GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT,
                                GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT};
                            const uint32_t role = signature.layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE ? roles[bank] :
                                uint32_t(bank == 0 ? GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT : GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT);
                            CHECK(grouped->get_bank(key, role, &descriptor));
                            const auto * payload = static_cast<const uint8_t *>(ggml_cuda_moe_grouped_context_test_access::device_bank_data(*grouped, key, descriptor.tensor));
                            std::vector<uint8_t> observed(descriptor.expert_stride);
                            for (uint32_t slot = 0; slot < info.n_slots; ++slot) {
                                if (owners[slot] < 0) { continue; }
                                CUDA_OK(cudaMemcpy(observed.data(), payload + size_t(slot) * descriptor.expert_stride,
                                    observed.size(), cudaMemcpyDeviceToHost));
                                CHECK(!memcmp(observed.data(), static_cast<const uint8_t *>(descriptor.source_data) +
                                    size_t(owners[slot]) * descriptor.expert_stride, observed.size()));
                            }
                        }
                    }
                }
                auto expected = active_grouped_tensor_values(reference.output[1]);
                if (main_prefill) {
                    double gpu_error = 0, gpu_energy = 0;
                    for (size_t i = 0; i < actual.size(); ++i) {
                        gpu_error += (double(actual[i]) - expected[i]) * (double(actual[i]) - expected[i]);
                        gpu_energy += double(expected[i]) * expected[i];
                    }
                    fprintf(stderr, "test-moe-cache: main prefill all-GPU comparison relative_mse=%.9g\n", gpu_error / std::max(gpu_energy, 1e-30));
                    const bool mixed_cpu = std::any_of(observed_body.masks.begin(), observed_body.masks.end(),
                        [](const std::vector<uint8_t> & mask) { return std::any_of(mask.begin(), mask.end(), [](uint8_t cpu) { return cpu != 0; }); });
                    if (!mixed_cpu) { CHECK(gpu_error / std::max(gpu_energy, 1e-30) <= 5e-3); }
                    const auto * graph = reference.result.get_gf();
                    std::vector<const ggml_tensor *> nodes(graph->nodes, graph->nodes + graph->n_nodes);
                    std::vector<const ggml_tensor *> dynamic(graph->leafs, graph->leafs + graph->n_leafs);
                    std::unordered_set<const ggml_tensor *> known(nodes.begin(), nodes.end());
                    known.insert(dynamic.begin(), dynamic.end());
                    std::vector<const ggml_tensor *> pending(nodes);
                    pending.insert(pending.end(), dynamic.begin(), dynamic.end());
                    for (size_t i = 0; i < pending.size(); ++i) {
                        const auto append = [&](const ggml_tensor * tensor) {
                            if (!tensor || !known.insert(tensor).second) { return; }
                            (tensor->op == GGML_OP_NONE ? dynamic : nodes).push_back(tensor);
                            pending.push_back(tensor);
                        };
                        const auto * tensor = pending[i];
                        for (const auto * input : tensor->src) { append(input); }
                        append(tensor->view_src);
                    }
                    std::vector<std::vector<uint8_t>> input_bytes;
                    std::vector<const void *> inputs;
                    for (const auto * tensor : dynamic) {
                        input_bytes.emplace_back(ggml_nbytes(tensor));
                        ggml_backend_tensor_get(tensor, input_bytes.back().data(), 0, input_bytes.back().size());
                        inputs.push_back(input_bytes.back().data());
                    }
                    expected = evaluate_body(nodes, dynamic, {reference.output[1]}, {}, inputs, 0, nullptr, owner->gpu.get(),
                        &observed_body.masks, &observed_body.ids);
                    CHECK(actual.size() == expected.size());
                }
                double error = 0, norm = 0;
                for (size_t i = 0; i < actual.size(); ++i) {
                    CHECK(std::isfinite(actual[i]) && std::isfinite(expected[i]));
                    error += (double(actual[i]) - expected[i]) * (double(actual[i]) - expected[i]);
                    norm += double(expected[i]) * expected[i];
                }
                CHECK(error / std::max(norm, 1e-30) <= (main_prefill ? 2e-5 : 5e-3));
                if (source_core && !overlap_fixture) {
                    const auto * probe = ggml_get_tensor(fixture.result.get_ctx(), "hybrid_fusion_retained");
                    const auto * oracle = ggml_get_tensor(reference.result.get_ctx(), "hybrid_fusion_retained");
                    CHECK(probe && oracle);
                    CHECK(active_grouped_tensor_values(probe) == active_grouped_tensor_values(oracle));
                }
                ggml_backend_moe_hybrid_state_v1 state = {};
                state.struct_size = sizeof(state);
                CHECK(ggml_backend_sched_moe_hybrid_state_v1(owner->sched.get(), &state));
                if (profile_adaptation == 2 && phase == 0 && step == 3) { pending_cpu_routes = state.cpu_routes; }
                if (profile_adaptation == 2 && phase == 0 && step >= 4 && step <= 8) {
                    CHECK(state.cpu_routes == pending_cpu_routes + uint64_t(step - 3) * capacity * 6 * (routed_scheduler ? 3 : 1));
                }
                if (early_fallback) {
                    CHECK(state.window_captures == 0 && state.window_launches == 0 && state.cpu_execute_calls == 0 && state.cpu_active_jobs == 0);
                    continue;
                }
                if (step == 0) { captures = state.window_captures; }
                const char * tail_option = getenv("GGML_MOE_SOURCE_DEVICE_TAIL");
                const bool tail_requested = source_core && arm != GGML_CUDA_MOE_FIDELITY_SEGMENTED && !no_alias && (!tail_option || atoi(tail_option));
                const bool tail_dispatch = tail_requested && captures == config.max_regions + 3;
                const uint64_t expected_captures = source_core ?
                    (arm == GGML_CUDA_MOE_FIDELITY_SEGMENTED || no_alias ? config.max_regions * 4 + 1 :
                        tail_dispatch ? config.max_regions + 3 : 1) :
                    fixture.result.get_moe_regions().size() + 1;
                if (main_prefill) {
                    CHECK(captures == 0 && state.window_launches == uint64_t(step + 1) * config.max_regions &&
                        state.window_waits == state.window_launches && state.window_fallbacks == 0 && state.errors == 0 &&
                        state.resident_routes > 0 && state.transfer_routes > 0);
                    CHECK(state.cpu_routes <= uint64_t(step + 1) * config.max_regions * config.n_threads);
                    // Fixed routing repeats each cohort for the whole prompt; stream heavy cohorts.
                    if (capacity > config.n_threads) { CHECK(state.cpu_routes == 0 && state.cpu_execute_calls == 0); }
                    else { CHECK(state.cpu_routes > 0 && state.cpu_execute_calls > 0); }
                    if (observed_body.check_overlap) {
                        CHECK(observed_body.cpu_started && observed_body.gpu_issued && !observed_body.overlap_timeout);
                    }
                    CHECK(state.cpu_active_jobs == 0 && state.dispatch_active == 0 && state.producer_events == state.producer_fences);
                    fprintf(stderr, "test-moe-cache: main source prefill type=%s rows=%u step=%u cpu_routes=%llu gpu_miss_routes=%llu relative_mse=%.9g OK\n",
                        ggml_type_name(signature.gate_up_type), capacity, step, (unsigned long long) state.cpu_routes,
                        (unsigned long long) state.transfer_routes, error / std::max(norm, 1e-30));
                } else {
                CHECK(captures == expected_captures && state.window_captures == captures &&
                    state.window_launches == uint64_t(step + 1) * (tail_dispatch ? 1 : captures) && state.window_fallbacks == 0 &&
                    state.resident_routes > 0 && (static_profile || state.transfer_routes > 0) && (capacity == 1 || profile_adaptation || state.cpu_routes > 0));
                CHECK(state.window_waits == step + 1 && state.cpu_active_jobs == 0 && state.dispatch_active == 0 &&
                    state.producer_events == state.producer_fences);
                }
                if (source_core && static_profile && profile_adaptation == 1 && phase == 0) {
                    // Fixed, strictly ordered router logits select IDs [0, routes) in each row.
                    for (auto & counts : learned_counts) { for (uint32_t id = 0; id < routes; ++id) { counts[id] += capacity; } }
                    check_learning(step + 1);
                }
                if (source_core && !overlap_fixture && !main_prefill) {
                    const char * disabled = getenv("GGML_CUDA_DISABLE_FUSION");
                    const char * shared = getenv("GGML_MOE_SOURCE_GRAPH_FUSION");
                    const bool enabled = (!disabled || !atoi(disabled)) && (!shared || atoi(shared));
                    CHECK((state.window_fused_nodes > 0) == enabled);
                }
                if (retained_hook_test) {
                    if (phase == 0) {
                        const auto calls = hook_calls.load(std::memory_order_relaxed);
                        if (main_prefill && !state.cpu_execute_calls) { CHECK(calls == 0); }
                        else if (capacity != 1) { CHECK(calls > previous_hook_calls); }
                        previous_hook_calls = calls;
                    } else { CHECK(hook_calls.load(std::memory_order_relaxed) == previous_hook_calls); }
                }
                if (source_core && step == 0 && !delayed_cancel) {
                    ggml_backend_sched_t clone = nullptr;
                    CHECK(ggml_backend_sched_moe_source_clone_v1(owner->oracle.get(), &clone) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_INVALID);
                    CHECK(!clone);
                    const auto saved_ranking = ranking;
                    if (static_profile) { std::fill(ranking.begin(), ranking.end(), -1); }
                    CHECK(ggml_backend_sched_moe_source_clone_v1(owner->sched.get(), &clone) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK);
                    std::copy(saved_ranking.begin(), saved_ranking.end(), ranking.begin());
                    CHECK(clone && ggml_backend_sched_moe_source_selected_v1(clone));
                    CHECK(ggml_backend_sched_moe_source_clone_v1(owner->sched.get(), &clone) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_INVALID);
                    CHECK(ggml_backend_sched_moe_source_reset_graph_v1(clone) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK);
                    CHECK(ggml_backend_sched_moe_source_free_v1(&clone) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK);
                    CHECK(!clone);
                    if (owned_program) {
                        CHECK(ggml_backend_sched_moe_source_program_compute_v1(owner->sched.get(), owned_program, &certificate) == GGML_STATUS_FAILED);
                        CHECK(ggml_backend_sched_moe_source_program_bind_v1(owner->sched.get(), fixture.result.get_gf(), &certificate,
                            &owned_program) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK && owned_program);
                    }
                }
                if (source_core && step == 0 && !main_prefill && !early_fallback && !delayed_cancel && !overlap_fixture) {
                    auto * retained = ggml_get_tensor(fixture.result.get_ctx(), "hybrid_fusion_retained");
                    CHECK(retained);
                    retained->op_params[0] ^= 1;
                    reject(certificate);
                    retained->op_params[0] ^= 1;
                    ggml_backend_moe_hybrid_state_v1 rejected = {};
                    rejected.struct_size = sizeof(rejected);
                    CHECK(ggml_backend_sched_moe_hybrid_state_v1(owner->sched.get(), &rejected));
                    CHECK(!rejected.quiescing && !rejected.dispatch_active && !rejected.cpu_active_jobs &&
                        rejected.window_launches == state.window_launches && rejected.window_waits == state.window_waits);
                }
                if (phase == 1 && step == 0) {
                    auto stale = certificate;
                    ++stale.owner_generation;
                    reject(stale);
                    ggml_backend_moe_hybrid_state_v1 rejected = {};
                    rejected.struct_size = sizeof(rejected);
                    CHECK(ggml_backend_sched_moe_hybrid_state_v1(owner->sched.get(), &rejected));
                    CHECK(rejected.window_launches == state.window_launches && rejected.window_waits == state.window_waits);
                }
                if (static_profile && !profile_adaptation && phase == 0 && step == 1) {
                    const auto previous = read_residency();
                    auto replacement = ranking;
                    std::rotate(replacement.begin(), replacement.begin() + 1, replacement.end());
                    std::vector<ggml_backend_moe_static_profile_v1> changed;
                    for (const auto * down : fixture.down) { changed.push_back({down, replacement.data(), experts}); }
                    cudaStream_t stream = nullptr;
                    CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
                    uint64_t copied = 0;
                    CHECK(grouped->initialize_profile(changed.data(), changed.size(), stream, &copied));
                    CUDA_OK(cudaStreamSynchronize(stream));
                    CUDA_OK(cudaStreamDestroy(stream));
                    CHECK(read_residency() != previous);
                }
            }
            fprintf(stderr, "test-moe-cache: fidelity scheduler phase=%u rows=%u certificate=%u replay/publication OK\n",
                phase, capacity, certificate.row_semantics);
            if (owned_program) {
                CHECK(unused_copy_calls == 0);
                ggml_backend_sched_set_copy_callback(owner->sched.get(), nullptr, nullptr);
                CHECK(ggml_backend_sched_moe_source_program_compute_v1(owner->sched.get(), owned_program, &certificate) == GGML_STATUS_FAILED);
                CHECK(ggml_backend_sched_moe_source_program_bind_v1(owner->sched.get(), fixture.result.get_gf(), &certificate,
                    &owned_program) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK && owned_program);
                ggml_backend_sched_set_eval_callback(owner->sched.get(), nullptr, nullptr);
                CHECK(ggml_backend_sched_moe_source_program_compute_v1(owner->sched.get(), owned_program, &certificate) == GGML_STATUS_FAILED);
                fprintf(stderr, "test-moe-cache: owned program immutable metadata/publication, stale/cross-owner/certificate rejection OK\n");
            }
        }
        if (profile_adaptation == 2) {
            CUDA_OK(cudaStreamDestroy(held_adaptation.release_stream));
            CUDA_OK(cudaFree(held_adaptation.value));
        }
        if (retained_hook_test) { CHECK(ggml_backend_sched_moe_hybrid_set_test_hook_v1(owner->sched.get(), nullptr, nullptr)); }
        if (source_core && delayed_cancel) {
            struct barrier {
                std::mutex mutex;
                std::condition_variable condition;
                uint32_t phase = 0;
                bool entered = false, released = false, timed_out = false;
                cudaStream_t stream = nullptr;
                static bool hook(void * opaque, uint32_t phase, uint64_t, void * value) {
                    auto & self = *static_cast<barrier *>(opaque);
                    if (phase == GGML_BACKEND_MOE_HYBRID_TEST_GPU_ENQUEUED) {
                        std::lock_guard<std::mutex> lock(self.mutex);
                        self.stream = static_cast<cudaStream_t>(value);
                    }
                    if (phase != self.phase) { return true; }
                    std::unique_lock<std::mutex> lock(self.mutex);
                    if (self.entered) { return true; }
                    self.entered = true;
                    self.condition.notify_all();
                    self.timed_out = !self.condition.wait_for(lock, std::chrono::seconds(30), [&] { return self.released; });
                    return !self.timed_out;
                }
                void release() { std::lock_guard<std::mutex> lock(mutex); released = true; condition.notify_all(); }
            };
            uint32_t scenario = 0;
            for (const uint32_t event : {uint32_t(GGML_BACKEND_MOE_HYBRID_TEST_CPU_BEFORE_COMMIT),
                    uint32_t(GGML_BACKEND_MOE_HYBRID_TEST_CPU_AFTER_COMMIT),
                    uint32_t(GGML_BACKEND_MOE_HYBRID_TEST_BEFORE_PUBLISH),
                    uint32_t(GGML_BACKEND_MOE_HYBRID_TEST_SOURCE_COMPUTE_ENTERED),
                    uint32_t(GGML_BACKEND_MOE_HYBRID_TEST_SOURCE_COMPUTE_RETURNING)}) {
                const char * mode = getenv("GGML_TEST_MOE_FIDELITY");
                const bool deadline_test = mode && !strcmp(mode, "source-core-deadline");
                if (deadline_test && event != GGML_BACKEND_MOE_HYBRID_TEST_CPU_BEFORE_COMMIT) { continue; }
                if (mode && !strcmp(mode, "source-core-copy-cancel") &&
                        event != GGML_BACKEND_MOE_HYBRID_TEST_CPU_BEFORE_COMMIT && event != GGML_BACKEND_MOE_HYBRID_TEST_BEFORE_PUBLISH) { continue; }
                if (scenario++) {
                    CHECK(ggml_backend_sched_moe_source_reset_v1(owner->sched.get()) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK);
                    owner->input_buffers.clear();
                    rebuild(fixture);
                    allocate_inputs();
                    allocate_regions();
                    prepare_regions(&certificate);
                }
                set_inputs(fixture, false, 0);
                CHECK(ggml_backend_sched_graph_compute_ext(owner->sched.get(), fixture.result.get_gf(), &certificate) == GGML_STATUS_SUCCESS);
                uint64_t canceled_program = 0;
                CHECK(ggml_backend_sched_moe_source_program_bind_v1(owner->sched.get(), fixture.result.get_gf(), &certificate,
                    &canceled_program) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK && canceled_program);
                std::vector<uint8_t> sentinel(ggml_nbytes(fixture.output[1]), 0x6b), observed(sentinel.size());
                ggml_backend_tensor_set(fixture.output[1], sentinel.data(), 0, sentinel.size());
                std::vector<uint8_t> effect_sentinel;
                if (scenario == 1 && external_effect) {
                    effect_sentinel.assign(ggml_nbytes(external_effect), 0x53);
                    ggml_backend_tensor_set(external_effect, effect_sentinel.data(), 0, effect_sentinel.size());
                }
                std::vector<std::vector<uint8_t>> backing;
                for (const auto & bank : fixture.tensors) {
                    const auto * first = static_cast<const uint8_t *>(bank.tensor->data);
                    backing.emplace_back(first, first + ggml_nbytes(bank.tensor));
                }
                barrier paused;
                paused.phase = event;
                CHECK(ggml_backend_sched_moe_hybrid_set_test_hook_v1(owner->sched.get(), barrier::hook, &paused));
                ggml_status result = GGML_STATUS_FAILED;
                std::thread caller([&] {
                    CUDA_OK(cudaSetDevice(device));
                    result = ggml_backend_sched_moe_source_program_compute_v1(owner->sched.get(), canceled_program, &certificate);
                });
                {
                    std::unique_lock<std::mutex> lock(paused.mutex);
                    const bool entered = paused.condition.wait_for(lock, std::chrono::seconds(10), [&] { return paused.entered; });
                    if (!entered) { paused.released = true; paused.condition.notify_all(); lock.unlock(); caller.join(); CHECK(entered); }
                }
                if (deadline_test) {
                    CHECK(paused.stream);
                    const auto expiry = std::chrono::steady_clock::now() + std::chrono::seconds(10);
                    cudaError_t status;
                    do {
                        status = cudaStreamQuery(paused.stream);
                        CHECK(std::chrono::steady_clock::now() < expiry);
                        std::this_thread::yield();
                    } while (status == cudaErrorNotReady);
                    CUDA_OK(status);
                }
                CHECK(ggml_backend_sched_moe_source_close_v1(owner->sched.get()) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK);
                if (!deadline_test) {
                    const auto started = std::chrono::steady_clock::now();
                    CHECK(ggml_backend_sched_moe_source_drain_v1(owner->sched.get()) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_TIMEOUT);
                    CHECK(std::chrono::steady_clock::now() - started < std::chrono::seconds(7));
                }
                ggml_backend_moe_hybrid_state_v1 held = {};
                held.struct_size = sizeof(held);
                CHECK(ggml_backend_sched_moe_hybrid_state_v1(owner->sched.get(), &held) && held.dispatch_active == 1);
                const bool cpu_claimed = event == GGML_BACKEND_MOE_HYBRID_TEST_CPU_BEFORE_COMMIT ||
                    event == GGML_BACKEND_MOE_HYBRID_TEST_CPU_AFTER_COMMIT;
                CHECK(held.cpu_active_jobs == uint32_t(cpu_claimed));
                if (event == GGML_BACKEND_MOE_HYBRID_TEST_CPU_AFTER_COMMIT) {
                    auto * retained = owner->sched.get();
                    CHECK(ggml_backend_sched_moe_source_free_v1(&retained) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_TIMEOUT && retained == owner->sched.get());
                }
                size_t bank_index = 0;
                for (const auto & bank : fixture.tensors) {
                    CHECK(memcmp(bank.tensor->data, backing[bank_index].data(), backing[bank_index].size()) == 0);
                    ++bank_index;
                }
                paused.release();
                caller.join();
                CHECK(!paused.timed_out);
                CHECK(result == (event == GGML_BACKEND_MOE_HYBRID_TEST_SOURCE_COMPUTE_RETURNING ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED));
                CHECK(ggml_backend_sched_moe_source_drain_v1(owner->sched.get()) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK);
                CHECK(ggml_backend_sched_moe_source_program_compute_v1(owner->sched.get(), canceled_program, &certificate) == GGML_STATUS_FAILED);
                ggml_backend_tensor_get(fixture.output[1], observed.data(), 0, observed.size());
                if (event != GGML_BACKEND_MOE_HYBRID_TEST_SOURCE_COMPUTE_RETURNING) { CHECK(observed == sentinel); }
                if (!effect_sentinel.empty()) {
                    std::vector<uint8_t> actual(effect_sentinel.size());
                    ggml_backend_tensor_get(external_effect, actual.data(), 0, actual.size());
                    CHECK(actual == effect_sentinel);
                }
                reject(certificate);
                CHECK(read_residency() == initial_residency);
                fprintf(stderr, "test-moe-cache: source-core held phase=%u wrapper/CPU/backing/finite drain/publication OK\n", event);
            }
            CHECK(ggml_backend_sched_moe_source_reset_v1(owner->sched.get()) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK);
            owner->input_buffers.clear(); rebuild(fixture); allocate_inputs(); allocate_regions(); prepare_regions(&certificate);
            set_inputs(fixture, false, 0);
            std::vector<uint8_t> sentinel(ggml_nbytes(fixture.output[1]), 0x6b), observed(sentinel.size());
            ggml_backend_tensor_set(fixture.output[1], sentinel.data(), 0, sentinel.size());
            CHECK(ggml_backend_sched_moe_hybrid_set_test_hook_v1(owner->sched.get(),
                [](void *, uint32_t phase, uint64_t, void *) { return phase != GGML_BACKEND_MOE_HYBRID_TEST_BEFORE_PUBLISH; }, nullptr));
            CHECK(ggml_backend_sched_graph_compute_ext(owner->sched.get(), fixture.result.get_gf(), &certificate) == GGML_STATUS_FAILED);
            CHECK(ggml_backend_sched_moe_source_drain_v1(owner->sched.get()) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK);
            ggml_backend_tensor_get(fixture.output[1], observed.data(), 0, observed.size());
            CHECK(observed == sentinel);
            CHECK(ggml_backend_sched_moe_source_reset_v1(owner->sched.get()) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK);
            owner->input_buffers.clear(); rebuild(fixture); allocate_inputs(); allocate_regions(); prepare_regions(&certificate);
            set_inputs(fixture, false, 0);
            CHECK(ggml_backend_sched_graph_compute_ext(owner->sched.get(), fixture.result.get_gf(), &certificate) == GGML_STATUS_SUCCESS);
        }
        if (source_core) {
            CHECK(ggml_backend_sched_moe_source_close_v1(owner->sched.get()) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK);
            CHECK(ggml_backend_sched_moe_source_drain_v1(owner->sched.get()) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK);
            if (transport_case) {
                moe_host_budget * source_budget = nullptr;
                if (transport_case >= 2) {
                    CHECK(transport_budget > 0 && fixture.buft->context);
                    source_budget = static_cast<moe_host_budget *>(fixture.buft->context);
                }
                CHECK(ggml_backend_sched_moe_source_reset_v1(owner->sched.get()) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK);
                owner->input_buffers.clear(); rebuild(fixture); allocate_inputs();
                const auto transport_totals = [&]() {
                    ggml_backend_sched_ptr capture_sched(ggml_backend_sched_new(backends, nullptr, 2, 256, false, true));
                    for (auto & region : fixture.result.get_moe_regions()) { CHECK(region.place(capture_sched.get(), owner->gpu.get())); }
                    CHECK(ggml_backend_sched_alloc_graph(capture_sched.get(), fixture.result.get_gf()));
                    set_inputs(fixture, false, 0);
                    std::vector<uint8_t> sentinel(ggml_nbytes(fixture.output[1]), 0x6b), observed(sentinel.size());
                    ggml_backend_tensor_set(fixture.output[1], sentinel.data(), 0, sentinel.size());
                    owner->graph_nodes.clear();
                    owner->graph = {};
                    fidelity_probe = owner.get();
                    owner->delegate = owner->gpu->iface.graph_compute;
                    owner->capture_only = true;
                    owner->gpu->iface.graph_compute = fidelity_capture_split;
                    const auto captured = ggml_backend_sched_graph_compute_ext(capture_sched.get(), fixture.result.get_gf(), &certificate);
                    owner->gpu->iface.graph_compute = owner->delegate;
                    fidelity_probe = nullptr;
                    CHECK(captured == GGML_STATUS_SUCCESS);
                    ggml_backend_sched_synchronize(capture_sched.get());
                    ggml_backend_tensor_get(fixture.output[1], observed.data(), 0, observed.size());
                    CHECK(observed == sentinel && !owner->graph_nodes.empty());
                    CHECK(owner->graph.execution_certificate.flags == GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED &&
                        owner->graph.execution_certificate.row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SPECULATIVE &&
                        owner->graph.execution_certificate.n_rows == capacity &&
                        owner->graph.execution_certificate.source_graph_uid == fixture.result.get_gf()->uid &&
                        owner->graph.execution_certificate.split_graph_uid == owner->graph.uid &&
                        owner->graph.uid != fixture.result.get_gf()->uid);
                    uint64_t direct_bytes = 0, staged_bytes = 0;
                    cudaStream_t copy_stream = nullptr;
                    CUDA_OK(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
                    std::shared_ptr<ggml_cuda_moe_graph_plan> transport_plan;
                    ggml_cuda_moe_graph_execution transport_execution;
                    auto * transport_graph = &owner->graph;
                    uint32_t mmids = 0;
                    uint64_t fingerprint = 0;
                    const auto coverage = grouped->certify_graph_coverage(transport_graph, &mmids, &fingerprint);
                    CHECK(coverage && mmids && fingerprint);
                    CHECK(grouped->prepare_graph_execution(transport_graph, transport_graph->uid, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED,
                        &transport_plan, &transport_execution, coverage, transport_graph->nodes, mmids, fingerprint) != GGML_CUDA_MOE_GRAPH_PREPARE_UNAVAILABLE);
                    CHECK(transport_execution.resolve_streams([](void * opaque, const ggml_tensor *) {
                        return *static_cast<cudaStream_t *>(opaque);
                    }, &copy_stream));
                    CHECK(grouped->begin_graph_dispatch(&transport_execution, GGML_CUDA_MOE_GRAPH_DISPATCH_DIRECT));
                    ggml_cuda_moe_source_transport * transport = nullptr;
                    struct bank_copy { ggml_cuda_moe_graph_group_dispatch * group; uint32_t bank, binding; ggml_cuda_moe_grouped_bank_descriptor descriptor; };
                    std::vector<bank_copy> copies;
                    size_t tile_bytes = 0;
                    for (const auto & bank : fixture.tensors) { tile_bytes = std::max(tile_bytes, bank.tensor->nb[2]); }
                    std::vector<ggml_backend_buffer_ptr> maps;
                    std::vector<ggml_cuda_moe_graph_group_dispatch *> groups;
                    std::vector<uint64_t> residency_tokens;
                    std::vector<ggml_cuda_moe_grouped_acquisition> acquisitions;
                    for (int i = 0; i < transport_graph->n_nodes; ++i) {
                        auto * group = transport_execution.find_group(transport_graph->nodes[i], nullptr);
                        if (!group || std::find(groups.begin(), groups.end(), group) != groups.end()) { continue; }
                        groups.push_back(group);
                        auto buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_host_buffer_type(), 2 * experts * sizeof(int32_t));
                        CHECK(buffer);
                        auto * slots = static_cast<int32_t *>(ggml_backend_buffer_get_base(buffer));
                        residency_tokens.push_back(0);
                        CHECK(grouped->prepare_source_group(group, copy_stream, slots, experts, slots + experts, experts, true, &residency_tokens.back()));
                        CHECK(residency_tokens.back() != 0);
                        acquisitions.push_back(group->transaction.acquisition);
                        maps.emplace_back(buffer);
                        for (uint32_t b = 0; b < group->key.n_banks; ++b) {
                            bank_copy copy = {group, b, 0, {}};
                            CHECK(grouped->get_group_resource_bank(group->transaction, b, &copy.descriptor));
                            const bool direct = transport_case == 1 || (transport_case == 3 && copy.descriptor.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT);
                            CHECK(copy.descriptor.buft == (direct ? ggml_backend_cuda_moe_cached_bounded_buffer_type(0) : fixture.buft));
                            auto * catalog = direct ? nullptr : source_budget;
                            CHECK(copy.descriptor.source_path == (direct ? MOE_GROUPED_SOURCE_MAPPED : MOE_GROUPED_SOURCE_PAGEABLE_STAGED));
                            CHECK(!direct || copy.descriptor.source_device_alias != nullptr);
                            CHECK(direct ? catalog == nullptr : catalog != nullptr);
                            if (catalog) {
                                std::lock_guard<std::mutex> lock(catalog->mutex);
                                const auto record = catalog->sources.find(copy.descriptor.tensor);
                                CHECK(record != catalog->sources.end() && record->second.device_alias == nullptr);
                            }
                            CHECK(grouped->prepare_source_transport(group->transaction, b, tile_bytes, &transport, &copy.binding, true));
                            copies.push_back(copy);
                            if (copies.size() == 1) { CHECK(grouped->bind_source_transport(transport, transport_execution)); }
                        }
                    }
                    CHECK(groups.size() == 2 && !copies.empty());
                    CUDA_OK(cudaStreamSynchronize(copy_stream));
                    std::vector<std::vector<int32_t>> snapshots;
                    for (size_t i = 0; i < groups.size(); ++i) {
                        const auto * slots = static_cast<int32_t *>(ggml_backend_buffer_get_base(maps[i].get()));
                        snapshots.emplace_back(slots, slots + experts + groups[i]->n_slots);
                    }
                    void * selection_data = nullptr;
                    CUDA_OK(cudaMalloc(&selection_data, sizeof(ggml_cuda_moe_hybrid_selection) + 2 * experts * sizeof(int32_t)));
                    auto * selection = static_cast<ggml_cuda_moe_hybrid_selection *>(selection_data);
                    auto * selected_slots = reinterpret_cast<int32_t *>(selection + 1);
                    for (auto * group : groups) {
                        CHECK(grouped->select_hybrid_group(*group, selection, selected_slots, selected_slots + experts, 0, {}, 0, 1, 0, false, nullptr, 0));
                    }
                    CUDA_OK(cudaStreamSynchronize(copy_stream));
                    CUDA_OK(cudaFree(selection_data));
                    const auto saved_transaction = groups.back()->transaction;
                    groups.back()->transaction.transaction_token = 0;
                    CHECK(!grouped->finish_source_dispatch(&transport_execution));
                    groups.back()->transaction = saved_transaction;
                    for (auto * group : groups) {
                        ggml_cuda_moe_grouped_resource_info info;
                        CHECK(grouped->get_group_resources(group->transaction.acquisition, &info) && info.transaction_active);
                        CHECK(group->authority && group->state == GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_ACTIVE);
                    }
                    CHECK(grouped->finish_source_dispatch(&transport_execution));
                    for (size_t i = 0; i < groups.size(); ++i) {
                        ggml_cuda_moe_grouped_resource_info info;
                        CHECK(grouped->get_group_resources(acquisitions[i], &info) && !info.transaction_active);
                        CHECK(!groups[i]->authority && !groups[i]->transaction.transaction_token &&
                            groups[i]->state == GGML_CUDA_MOE_GRAPH_GROUP_FINISHED);
                    }
                    CHECK(grouped->begin_graph_dispatch(&transport_execution, GGML_CUDA_MOE_GRAPH_DISPATCH_DIRECT));
                    for (size_t i = 0; i < groups.size(); ++i) {
                        const auto previous = residency_tokens[i];
                        auto * slots = static_cast<int32_t *>(ggml_backend_buffer_get_base(maps[i].get()));
                        memset(slots, 0x65, snapshots[i].size() * sizeof(int32_t));
                        CHECK(grouped->prepare_source_group(groups[i], copy_stream, slots, experts, slots + experts, experts, false, &residency_tokens[i]));
                        CHECK(residency_tokens[i] != previous);
                    }
                    CUDA_OK(cudaStreamSynchronize(copy_stream));
                    for (size_t i = 0; i < groups.size(); ++i) {
                        CHECK(memcmp(ggml_backend_buffer_get_base(maps[i].get()), snapshots[i].data(), snapshots[i].size() * sizeof(int32_t)) == 0);
                    }
                    CHECK(grouped->finish_source_dispatch(&transport_execution));
                    for (bool writer : {false, true}) {
                        if (writer) {
                            for (size_t i = 0; i < acquisitions.size(); ++i) {
                                ggml_cuda_moe_grouped_transaction transaction;
                                CHECK(grouped->begin_group_transaction(acquisitions[i], &transaction));
                                CHECK(grouped->end_group_transaction(transaction));
                                memset(ggml_backend_buffer_get_base(maps[i].get()), 0x65, snapshots[i].size() * sizeof(int32_t));
                            }
                        }
                        CHECK(grouped->begin_graph_dispatch(&transport_execution, GGML_CUDA_MOE_GRAPH_DISPATCH_DIRECT));
                        for (size_t i = 0; i < groups.size(); ++i) {
                            const auto previous = residency_tokens[i];
                            auto * slots = static_cast<int32_t *>(ggml_backend_buffer_get_base(maps[i].get()));
                            CHECK(grouped->prepare_source_group(groups[i], copy_stream, slots, experts, slots + experts, experts, false, &residency_tokens[i]));
                            CHECK((residency_tokens[i] != previous) == writer);
                        }
                        CUDA_OK(cudaStreamSynchronize(copy_stream));
                        for (size_t i = 0; i < groups.size(); ++i) {
                            CHECK(memcmp(ggml_backend_buffer_get_base(maps[i].get()), snapshots[i].data(), snapshots[i].size() * sizeof(int32_t)) == 0);
                        }
                        if (!writer) { CHECK(grouped->finish_source_dispatch(&transport_execution)); }
                    }
                    void * destination = nullptr;
                    CUDA_OK(cudaMalloc(&destination, tile_bytes));
                    const auto deadline = []() {
                        return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()) + 5'000'000'000ull;
                    };

                    const auto matches_group = [&](const bank_copy & copy, const ggml_cuda_moe_grouped_transaction & transaction) {
                        std::vector<uint32_t> bindings, banks;
                        for (const auto & candidate : copies) {
                            if (candidate.group != copy.group) { continue; }
                            bindings.push_back(candidate.binding);
                            banks.push_back(candidate.bank);
                        }
                        return grouped->source_transport_matches(transport, bindings.data(), transaction, banks.data(), banks.size());
                    };
                    uint64_t source_identity = 0, full_identity = 0;
                    CHECK(!grouped->source_resource_fingerprint(transport_execution, copy_stream, transport, &source_identity) && !source_identity);
                    CHECK(grouped->bind_source_transport(transport, transport_execution));
                    CHECK(grouped->graph_resource_fingerprint(transport_execution, copy_stream, &full_identity));
                    CHECK(grouped->source_resource_fingerprint(transport_execution, copy_stream, transport, &source_identity) && source_identity == full_identity);
                    ggml_cuda_moe_graph_execution foreign_execution;
                    CHECK(!grouped->bind_source_transport(transport, foreign_execution));
                    CHECK(!grouped->source_resource_fingerprint(foreign_execution, copy_stream, transport, &source_identity) && !source_identity);
                    const auto matches_sources = [&] {
                        source_identity = UINT64_MAX;
                        const bool matched = grouped->source_resource_fingerprint(transport_execution, copy_stream, transport, &source_identity);
                        CHECK(matched ? source_identity == full_identity : source_identity == 0);
                        return matched;
                    };
                    CHECK(matches_sources());
                    for (auto * group : groups) {
                        std::vector<uint32_t> bindings, banks;
                        for (const auto & copy : copies) {
                            if (copy.group != group) { continue; }
                            bindings.push_back(copy.binding);
                            banks.push_back(copy.bank);
                        }
                        CHECK(bindings.size() == group->key.n_banks);
                        CHECK(grouped->source_transport_matches(transport, bindings.data(), group->transaction, banks.data(), banks.size()));
                        CHECK(!grouped->source_transport_matches(transport, bindings.data(), group->transaction, banks.data(), 0));
                        const auto binding = bindings.back();
                        bindings.back() = UINT32_MAX;
                        CHECK(!grouped->source_transport_matches(transport, bindings.data(), group->transaction, banks.data(), banks.size()));
                        bindings.back() = binding;
                        const auto bank = banks.back();
                        banks.back() = group->key.n_banks;
                        CHECK(!grouped->source_transport_matches(transport, bindings.data(), group->transaction, banks.data(), banks.size()));
                        banks.back() = bank;
                        const auto & late = *std::find_if(copies.begin(), copies.end(), [&](const bank_copy & copy) {
                            return copy.group == group && copy.bank == bank;
                        });
                        auto * tensor = const_cast<ggml_tensor *>(late.descriptor.tensor);
                        auto * data = tensor->data;
                        tensor->data = static_cast<uint8_t *>(data) + 1;
                        CHECK(!matches_group(late, group->transaction));
                        CHECK(!matches_sources());
                        tensor->data = data;
                        CHECK(matches_group(late, group->transaction));
                        CHECK(matches_sources());
                    }
                    for (const auto & copy : copies) {
                        CHECK(grouped->source_transport_matches(transport, copy.binding, copy.group->transaction, copy.bank));
                        const void * alias = nullptr;
                        const auto & bank = copy.descriptor;
                        CHECK(grouped->source_transport_alias(transport, copy.binding, bank.tensor, bank.source_data, bank.expert_stride, &alias));
                        CHECK((alias != nullptr) == (bank.source_path == MOE_GROUPED_SOURCE_MAPPED));
                        CHECK(!grouped->source_transport_alias(transport, copy.binding, bank.tensor,
                            static_cast<const uint8_t *>(bank.source_data) + bank.byte_extent - 1, 2, &alias) && !alias);
                        CHECK(grouped->source_transport_alias(transport, copy.binding, bank.tensor,
                            static_cast<const uint8_t *>(bank.source_data) + 1, 33, &alias));
                        CHECK((alias != nullptr) == (bank.source_path == MOE_GROUPED_SOURCE_MAPPED));
                        for (int repeat = 0; repeat < 3; ++repeat) {
                            const auto & bank = copy.descriptor;
                            CHECK(grouped->copy_source_transport(transport, copy.binding, bank.tensor, destination, bank.source_data,
                                bank.expert_stride, copy_stream, deadline()));
                            CUDA_OK(cudaStreamSynchronize(copy_stream));
                            std::vector<uint8_t> observed(bank.expert_stride);
                            CUDA_OK(cudaMemcpy(observed.data(), destination, observed.size(), cudaMemcpyDeviceToHost));
                            CHECK(memcmp(observed.data(), bank.source_data, observed.size()) == 0);
                            (bank.source_path == MOE_GROUPED_SOURCE_MAPPED ? direct_bytes : staged_bytes) += observed.size();
                        }
                    }
                    ggml_cuda_moe_source_transport_stats stats;
                    CHECK(grouped->source_transport_stats(transport, &stats));
                    CHECK(stats.direct_bytes == direct_bytes && stats.catalog_staged_bytes == staged_bytes && stats.null_staged_bytes == 0);
                    const auto & copy = copies.front();
                    auto stale = copy.group->transaction;
                    ++stale.acquisition.resource_generation;
                    CHECK(!grouped->source_transport_matches(transport, copy.binding, stale, copy.bank));
                    CHECK(!matches_group(copy, stale));
                    const auto transaction = copy.group->transaction;
                    copy.group->transaction = stale;
                    CHECK(!matches_sources());
                    copy.group->transaction = transaction;
                    CHECK(matches_sources());
                    std::vector<uint8_t> public_sentinel(ggml_nbytes(fixture.output[1]), 0x6b), public_observed(public_sentinel.size());
                    ggml_backend_tensor_set(fixture.output[1], public_sentinel.data(), 0, public_sentinel.size());
                    auto * tensor = const_cast<ggml_tensor *>(copy.descriptor.tensor);
                    void * original_data = tensor->data;
                    tensor->data = static_cast<uint8_t *>(original_data) + 1;
                    CHECK(!grouped->source_transport_matches(transport, copy.binding, copy.group->transaction, copy.bank));
                    CHECK(!matches_sources());
                    const void * alias = nullptr;
                    CHECK(!grouped->source_transport_alias(transport, copy.binding, tensor,
                        copy.descriptor.source_data, copy.descriptor.expert_stride, &alias) && !alias);
                    CHECK(!grouped->copy_source_transport(transport, copy.binding, tensor, fixture.output[1]->data,
                        copy.descriptor.source_data, copy.descriptor.expert_stride, copy_stream, deadline()));
                    tensor->data = original_data;
                    CHECK(matches_sources());
                    if (transport_case >= 2) {
                        const auto staged = std::find_if(copies.begin(), copies.end(), [](const bank_copy & candidate) {
                            return candidate.descriptor.source_path == MOE_GROUPED_SOURCE_PAGEABLE_STAGED;
                        });
                        CHECK(staged != copies.end());
                        CHECK(staged->descriptor.buft == fixture.buft && source_budget);
                        auto * budget = source_budget;
                        bool read_only = false;
                        {
                            std::lock_guard<std::mutex> lock(budget->mutex);
                            const auto record = budget->sources.find(staged->descriptor.tensor);
                            CHECK(record != budget->sources.end());
                            read_only = record->second.read_only;
                            record->second.read_only = !read_only;
                        }
                        CHECK(!grouped->source_transport_matches(transport, staged->binding, staged->group->transaction, staged->bank));
                        CHECK(!matches_group(*staged, staged->group->transaction));
                        CHECK(!matches_sources());
                        CHECK(!grouped->source_transport_alias(transport, staged->binding, staged->descriptor.tensor,
                            staged->descriptor.source_data, staged->descriptor.expert_stride, &alias) && !alias);
                        CHECK(!grouped->copy_source_transport(transport, staged->binding, staged->descriptor.tensor, fixture.output[1]->data,
                            staged->descriptor.source_data, staged->descriptor.expert_stride, copy_stream, deadline()));
                        {
                            std::lock_guard<std::mutex> lock(budget->mutex);
                            budget->sources.at(staged->descriptor.tensor).read_only = read_only;
                        }
                        CHECK(grouped->source_transport_matches(transport, staged->binding, staged->group->transaction, staged->bank));
                        CHECK(matches_group(*staged, staged->group->transaction));
                        CHECK(matches_sources());
                    }
                    CUDA_OK(cudaStreamSynchronize(copy_stream));
                    ggml_backend_tensor_get(fixture.output[1], public_observed.data(), 0, public_observed.size());
                    CHECK(public_observed == public_sentinel);
                    CHECK(grouped->finish_source_dispatch(&transport_execution));
                    CHECK(!matches_sources());
                    CHECK(grouped->release_source_transport(&transport) && !transport);
                    CUDA_OK(cudaFree(destination));
                    CUDA_OK(cudaStreamDestroy(copy_stream));
                    return std::make_pair(direct_bytes, staged_bytes);
                }();
                CHECK(ggml_backend_sched_moe_source_reset_v1(owner->sched.get()) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK);
                owner->input_buffers.clear(); rebuild(fixture); allocate_inputs(); allocate_regions(); prepare_regions(&certificate);
                set_inputs(fixture, false, 0);
                set_inputs(reference, false, 0);
                CHECK(ggml_backend_sched_graph_compute_ext(owner->oracle.get(), reference.result.get_gf(), &certificate) == GGML_STATUS_SUCCESS);
                CHECK(ggml_backend_sched_graph_compute_ext(owner->sched.get(), fixture.result.get_gf(), &certificate) == GGML_STATUS_SUCCESS);
                const auto recovered = active_grouped_tensor_values(fixture.output[1]);
                const auto expected = active_grouped_tensor_values(reference.output[1]);
                double error = 0, norm = 0;
                CHECK(recovered.size() == expected.size());
                for (size_t i = 0; i < recovered.size(); ++i) {
                    CHECK(std::isfinite(recovered[i]) && std::isfinite(expected[i]));
                    error += (double(recovered[i]) - expected[i]) * (double(recovered[i]) - expected[i]);
                    norm += double(expected[i]) * expected[i];
                }
                CHECK(error / std::max(norm, 1e-30) <= 5e-3);
                ggml_backend_moe_hybrid_state_v1 recovered_state = {};
                recovered_state.struct_size = sizeof(recovered_state);
                CHECK(ggml_backend_sched_moe_hybrid_state_v1(owner->sched.get(), &recovered_state));
                CHECK(recovered_state.window_waits == 1 && recovered_state.cpu_active_jobs == 0 && recovered_state.dispatch_active == 0 &&
                    recovered_state.producer_events == recovered_state.producer_fences && recovered_state.window_fallbacks == 0);
                CHECK(ggml_backend_sched_moe_source_close_v1(owner->sched.get()) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK);
                CHECK(ggml_backend_sched_moe_source_drain_v1(owner->sched.get()) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK);
                fprintf(stderr, "test-moe-cache: canonical transport case=%u actual direct=%llu catalog-staged=%llu stale sentinel/reset/reprepare OK\n",
                    transport_case, (unsigned long long) transport_totals.first, (unsigned long long) transport_totals.second);
            }
            auto * released = owner->sched.release();
            CHECK(ggml_backend_sched_moe_source_free_v1(&released) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK && !released);
        } else { owner->sched.reset(); }
        if (static_profile) {
            const auto retained_state = grouped->state();
            const auto retired_residency = read_residency();
            std::array<ggml_cuda_moe_candidate_group_key, 2> retained_keys;
            std::array<std::array<const ggml_tensor *, 4>, 2> retained_banks;
            std::array<std::array<void *, 4>, 2> retained_payloads = {};
            for (uint32_t layer = 0; layer < retained_keys.size(); ++layer) {
                CHECK(grouped->find_down_group_key(fixture.down[layer], &retained_keys[layer]));
                retained_banks[layer] = {fixture.gate_up[layer], fixture.gate[layer], fixture.up[layer], fixture.down[layer]};
                for (uint32_t bank = 0; bank < retained_banks[layer].size(); ++bank) {
                    if (!retained_banks[layer][bank]) { continue; }
                    retained_payloads[layer][bank] = ggml_cuda_moe_grouped_context_test_access::device_bank_data(
                        *grouped, retained_keys[layer], retained_banks[layer][bank]);
                    CHECK(retained_payloads[layer][bank]);
                }
            }
            const bool previous_debug = ggml_backend_cuda_moe_get_debug_mm();
            ggml_backend_cuda_moe_set_debug_mm(true);
            size_t lane_bytes = 0;
            for (const auto & bank : fixture.tensors) {
                if (bank.status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE) {
                    lane_bytes = std::max(lane_bytes, 2 * bank.tensor->nb[2]);
                }
            }
            CHECK(ggml_cuda_moe_grouped_context_test_access::set_prefill_staging_lane_bytes(*grouped, lane_bytes));
            ggml_backend_sched_reset(owner->oracle.get());
            owner->regions.clear(); owner->descriptors.clear(); owner->metadata.clear();
            owner->graph_nodes.clear(); owner->input_buffers.clear();
            const uint32_t saved_capacity = capacity;
            capacity = 128;
            fixture.build_graph(false, nullptr, nullptr, false, capacity);
            reference.build_graph(false, nullptr, nullptr, false, capacity);
            allocate_inputs();
            ggml_backend_t backends[]{owner->gpu.get(), owner->cpu.get()};
            ggml_backend_sched_ptr prefill(ggml_backend_sched_new(backends, nullptr, 2, 256, false, true));
            CHECK(prefill && ggml_cuda_moe_grouped_context_for_test(owner->gpu.get()) == grouped);
            for (auto & region : fixture.result.get_moe_regions()) { CHECK(region.place(prefill.get(), owner->gpu.get())); }
            for (auto & region : reference.result.get_moe_regions()) { CHECK(region.place(owner->oracle.get(), owner->gpu.get())); }
            CHECK(ggml_backend_sched_alloc_graph(prefill.get(), fixture.result.get_gf()));
            CHECK(ggml_backend_sched_alloc_graph(owner->oracle.get(), reference.result.get_gf()));
            CHECK(read_residency() == retired_residency && grouped->state().generation == retained_state.generation);
            auto prefill_certificate = layer_certificate();
            prefill_certificate.row_semantics = GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL;
            prefill_certificate.n_rows = capacity;
            ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*grouped);
            for (uint32_t step = 0; step < 2; ++step) {
                for (auto * target : {&fixture, &reference}) {
                    set_inputs(*target, false, step);
                    for (auto * logits : target->logits) {
                        std::vector<float> scores(size_t(capacity) * experts, -10.0f);
                        for (uint32_t row = 0; row < capacity; ++row) {
                            for (uint32_t rank = 0; rank < routes; ++rank) {
                                scores[size_t(row) * experts + (row + rank + step) % experts] = 10.0f - float(rank);
                            }
                        }
                        ggml_backend_tensor_set(logits, scores.data(), 0, scores.size() * sizeof(float));
                    }
                }
                CHECK(ggml_backend_sched_graph_compute_ext(owner->oracle.get(), reference.result.get_gf(), &prefill_certificate) == GGML_STATUS_SUCCESS);
                CHECK(ggml_backend_sched_graph_compute_ext(prefill.get(), fixture.result.get_gf(), &prefill_certificate) == GGML_STATUS_SUCCESS);
                const auto actual = active_grouped_tensor_values(fixture.output[1]);
                const auto expected = active_grouped_tensor_values(reference.output[1]);
                CHECK(actual.size() == expected.size());
                double error = 0, norm = 0;
                for (size_t i = 0; i < actual.size(); ++i) {
                    CHECK(std::isfinite(actual[i]) && std::isfinite(expected[i]));
                    const double difference = double(actual[i]) - expected[i];
                    error += difference * difference;
                    norm += double(expected[i]) * expected[i];
                }
                CHECK(norm > 0 && error / norm <= 2e-5);
                const auto telemetry = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*grouped);
                CHECK(telemetry.prefill_grouped == 2 && telemetry.prefill_staged == 0);
                CHECK(telemetry.prefill_bounded_ops == 2 * (signature.layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE ? 3u : 2u));
                CHECK(telemetry.prefill_bounded_waves > telemetry.prefill_bounded_ops);
                CHECK(telemetry.fallback == 0 && telemetry.rollback == 0 && telemetry.prepare_error == 0 && telemetry.finish_error == 0);
                CHECK(grouped->state().generation == retained_state.generation);
                for (uint32_t layer = 0; layer < retained_keys.size(); ++layer) {
                    ggml_cuda_moe_candidate_group_key key;
                    CHECK(grouped->find_down_group_key(fixture.down[layer], &key));
                    CHECK(key.generation == retained_keys[layer].generation && key.group_index == retained_keys[layer].group_index);
                    for (uint32_t bank = 0; bank < retained_banks[layer].size(); ++bank) {
                        if (!retained_banks[layer][bank]) { continue; }
                        CHECK(ggml_cuda_moe_grouped_context_test_access::device_bank_data(*grouped, key, retained_banks[layer][bank]) == retained_payloads[layer][bank]);
                    }
                }
                fprintf(stderr, "test-moe-cache: profile/adaptation=%u source-retirement -> sequential prefill step=%u rows=%u owner/payload retained relative_mse=%.9g OK\n",
                    profile_adaptation, step, capacity, error / norm);
            }
            capacity = saved_capacity;
            ggml_backend_cuda_moe_set_debug_mm(previous_debug);
        }
        owner->model->close_moe_source_owner();
        fprintf(stderr, "test-moe-cache: fidelity scheduler R%u segmented prepare/replay/state/drain OK\n", capacity);
        return;
    }
    if (capacity == 1 && arm == GGML_CUDA_MOE_FIDELITY_SEGMENTED && !no_alias) {
        cudaStream_t stream = nullptr;
        CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        void * storage = nullptr;
        const size_t packet_bytes = sizeof(ggml_cuda_moe_hybrid_selection) +
            size_t(fixture.expert_used) * GGML_CUDA_MOE_HYBRID_PACKET_ARRAYS * sizeof(int32_t);
        CUDA_OK(cudaMalloc(&storage, packet_bytes + sizeof(uint32_t)));
        auto packet = ggml_cuda_moe_hybrid_packet(storage, fixture.expert_used);
        auto * status = reinterpret_cast<uint32_t *>(static_cast<uint8_t *>(storage) + packet_bytes);
        std::shared_ptr<ggml_cuda_moe_graph_plan> plan;
        ggml_cuda_moe_graph_execution execution;
        uint32_t mmids = 0;
        uint64_t fingerprint = 0;
        const uint64_t coverage = grouped->certify_graph_coverage(&owner->graph, &mmids, &fingerprint);
        CHECK(coverage && mmids && fingerprint);
        CHECK(grouped->prepare_graph_execution(&owner->graph, owner->graph.uid, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED,
            &plan, &execution, coverage, owner->graph.nodes, mmids, fingerprint) != GGML_CUDA_MOE_GRAPH_PREPARE_UNAVAILABLE);
        CHECK(execution.resolve_streams([](void * opaque, const ggml_tensor *) {
            return *static_cast<cudaStream_t *>(opaque);
        }, &stream));
        std::vector<uint32_t> strategies;
        for (const auto & descriptor : owner->descriptors) {
            ggml_cuda_moe_graph_binding binding;
            const auto * group = execution.find_group(owner->graph.nodes[descriptor.first_node], &binding);
            CHECK(group != nullptr);
            strategies.push_back(group->strategy);
        }
        for (bool select : {false, true}) {
            CHECK(grouped->begin_graph_dispatch(&execution, GGML_CUDA_MOE_GRAPH_DISPATCH_DIRECT));
            for (const auto & descriptor : owner->descriptors) {
                ggml_cuda_moe_graph_binding binding;
                auto * group = execution.find_group(owner->graph.nodes[descriptor.first_node], &binding);
                ggml_cuda_moe_hybrid_runtime initial = {};
                CHECK(group && grouped->prepare_hybrid_group(group, stream, packet.header, packet.slots, packet.residents,
                    0, packet, 0, 1, select ? nullptr : &initial));
                CHECK(group->hybrid_admission_started == select);
                if (!select) {
                    uint64_t bytes = 0;
                    CHECK(!grouped->begin_hybrid_admission(*group, 0, 0));
                    CHECK(!grouped->complete_hybrid_admission(*group, 0, 0, bytes));
                    CHECK(!grouped->begin_hybrid_admissions(*group, packet, status));
                    CHECK(!grouped->copy_hybrid_admission_bank(*group, packet, 0, storage, status));
                    CHECK(!grouped->complete_hybrid_admissions(*group, packet, status, bytes));
                } else {
                    CHECK(grouped->finish_hybrid_admission(*group, true));
                    CHECK(!group->hybrid_admission_started);
                }
                CHECK(!grouped->finish_hybrid_admission(*group, false));
            }
            CHECK(grouped->finish_graph_dispatch(&execution));
            for (size_t i = 0; i < owner->descriptors.size(); ++i) {
                auto * group = execution.find_group(owner->graph.nodes[owner->descriptors[i].first_node], nullptr);
                CHECK(group && group->strategy == strategies[i] && !group->authority);
                CHECK(group->state == GGML_CUDA_MOE_GRAPH_GROUP_FINISHED && group->transaction.transaction_token == 0);
                CHECK(!group->defer_completion && !group->hybrid_admission_started);
            }
            CHECK(execution.has_explicit_grouped_strategies());
            CUDA_OK(cudaStreamSynchronize(stream));
            CHECK(read_residency() == initial_residency);
        }
        CUDA_OK(cudaFree(storage));
        CUDA_OK(cudaStreamDestroy(stream));
        fprintf(stderr, "test-moe-cache: fidelity prepare-only admission rejection and canonical q0 close OK\n");
    }
    const auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(owner->gpu.get()));
    const auto get = reinterpret_cast<ggml_cuda_moe_fidelity_window_get_v1_t>(
        ggml_backend_reg_get_proc_address(reg, GGML_CUDA_MOE_FIDELITY_WINDOW_V1_PROC_NAME));
    CHECK(get != nullptr);
    const auto * api = get();
    const auto cpu_module = ggml_backend_moe_cpu_module_acquire_v1();
    CHECK(cpu_module);
    const auto cpu_get = reinterpret_cast<ggml_backend_moe_cpu_region_service_v1_t>(
        ggml_backend_reg_get_proc_address(cpu_module, GGML_BACKEND_MOE_CPU_FIDELITY_SERVICE_V1_PROC_NAME));
    CHECK(cpu_get);
    ggml_cuda_moe_fidelity_window_query_v1 query = {};
    query.struct_size = sizeof(query);
    query.graph = &owner->graph;
    query.public_output = fixture.output[1];
    struct numeric_probe {
        const char * name;
        ggml_tensor * actual;
        ggml_tensor * expected;
        bool routes;
    };
    std::vector<numeric_probe> numeric_probes;
    std::vector<ggml_tensor *> public_outputs{query.public_output};
    if (fixture.ordinary_norm_output) { public_outputs.push_back(fixture.ordinary_norm_output); }
    if (fixture.staged) { public_outputs.push_back(fixture.staged); }
    if (numeric_diagnostics) {
        const auto add = [&](const char * name, ggml_tensor * actual, ggml_tensor * expected, bool raw = false) {
            if (actual) {
                CHECK(expected && ggml_are_same_layout(actual, expected));
                numeric_probes.push_back({name, actual, expected, raw});
                public_outputs.push_back(actual);
            }
        };
        add("bf16-projection", fixture.bf16_projection, reference.bf16_projection);
        add("bf16-moe-input", fixture.bf16_prefix, reference.bf16_prefix);
        add("flash-attention", fixture.attention_output, reference.attention_output);
        add("attention-moe-input", fixture.attention_prefix, reference.attention_prefix);
        add("raw-layer0", fixture.result.get_moe_regions()[0].body_output, reference.result.get_moe_regions()[0].body_output, true);
        add("weighted-layer0", fixture.output[0], reference.output[0]);
        add("raw-layer1", fixture.result.get_moe_regions()[1].body_output, reference.result.get_moe_regions()[1].body_output, true);
        if (fixture.route_probe) { public_outputs.push_back(fixture.route_probe); }
    }
    query.public_outputs = public_outputs.data();
    query.n_public_outputs = public_outputs.size();
    query.graph_owner = owner.get();
    query.retain_graph = fidelity_graph_owner::retain;
    query.release_graph = fidelity_graph_owner::release;
    query.arena_generation = generation;
    query.regions = owner->regions.data();
    query.n_regions = owner->regions.size();
    query.source_owner = &source_owner;
    query.cpu_api = cpu_get();
    const auto cpu_requirements_get = reinterpret_cast<ggml_backend_moe_cpu_fidelity_requirements_v1_t>(
        ggml_backend_reg_get_proc_address(cpu_module, GGML_BACKEND_MOE_CPU_FIDELITY_REQUIREMENTS_V1_PROC_NAME));
    CHECK(cpu_requirements_get);
    query.cpu_requirements_api = cpu_requirements_get();
    query.certificate = owner->graph.execution_certificate;
    query.n_threads = 2;
    query.gpu_miss_quota = quota;
    query.no_host_alias = no_alias;
    ggml_cuda_moe_fidelity_window_state_v1 measured = {};
    const auto measured_status = api->measure(owner->gpu.get(), &query, &measured);
    fprintf(stderr, "test-moe-cache: fidelity real measure R%u arm=%u no_alias=%d status=%d storage=%llu/%llu/%llu/%llu\n",
        capacity, arm, no_alias, measured_status, (unsigned long long) measured.storage.device_bytes,
        (unsigned long long) measured.storage.pinned_bytes, (unsigned long long) measured.storage.cpu_bytes,
        (unsigned long long) measured.storage.metadata_bytes);
    CHECK(measured_status == 0 && owner->references == 0);
    if (native_reference) {
        auto no_mapping = query;
        no_mapping.no_host_alias = true;
        ggml_cuda_moe_fidelity_window_state_v1 rejected = {};
        CHECK(api->measure(owner->gpu.get(), &no_mapping, &rejected) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION);
        CHECK(owner->references == 0);
        void * unsupported = nullptr;
        CHECK(api->prepare(owner->gpu.get(), &query, &measured.storage, GGML_CUDA_MOE_FIDELITY_SEGMENTED, &unsupported) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION);
        CHECK(!unsupported && owner->references == 0);
    }
    CHECK(measured.storage.device_runtime_bytes + measured.storage.device_rows_bytes +
        measured.storage.device_resident_body_bytes + measured.storage.device_transfer_body_bytes +
        measured.storage.device_clone_bytes + measured.storage.device_workspace_bytes + measured.storage.device_cublas_bytes +
        measured.storage.device_staged_input_bytes == measured.storage.device_bytes);
    CHECK(measured.storage.pinned_runtime_bytes + measured.storage.pinned_rows_bytes == measured.storage.pinned_bytes);
    CHECK(measured.planned_clone_bytes > 0 && measured.planned_clone_bytes == measured.storage.device_clone_bytes);
    fprintf(stderr, "test-moe-cache: fidelity private clone R%u allocated=%llu planned=%llu\n", capacity,
        (unsigned long long) measured.storage.device_clone_bytes, (unsigned long long) measured.planned_clone_bytes);
    fprintf(stderr, "test-moe-cache: fidelity real device components runtime=%llu rows=%llu resident=%llu transfer=%llu clone=%llu workspace=%llu cublas=%llu staged_input=%llu pinned=%llu/%llu\n",
        (unsigned long long) measured.storage.device_runtime_bytes, (unsigned long long) measured.storage.device_rows_bytes,
        (unsigned long long) measured.storage.device_resident_body_bytes, (unsigned long long) measured.storage.device_transfer_body_bytes,
        (unsigned long long) measured.storage.device_clone_bytes, (unsigned long long) measured.storage.device_workspace_bytes,
        (unsigned long long) measured.storage.device_cublas_bytes,
        (unsigned long long) measured.storage.device_staged_input_bytes,
        (unsigned long long) measured.storage.pinned_runtime_bytes, (unsigned long long) measured.storage.pinned_rows_bytes);
    if (staged_inputs) {
        const size_t bytes = ggml_nbytes(fixture.staged);
        CHECK(measured.storage.device_staged_input_bytes >= bytes && measured.storage.device_staged_input_bytes < bytes + 256);
        auto limits = measured.storage;
        limits.device_bytes -= measured.storage.device_staged_input_bytes;
        void * short_session = nullptr;
        CHECK(api->prepare(owner->gpu.get(), &query, &limits, arm, &short_session) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY);
        CHECK(!short_session && owner->references == 0);
    } else { CHECK(measured.storage.device_staged_input_bytes == 0); }
    if (bf16_columns) {
        const auto projection = std::find_if(owner->graph_nodes.begin(), owner->graph_nodes.end(), [&](const ggml_tensor * node) {
            return node->op == GGML_OP_MUL_MAT && node->src[0] == fixture.bf16_projection_weight;
        });
        CHECK(projection != owner->graph_nodes.end());
        CHECK(measured.storage.device_workspace_bytes >= size_t(ggml_nelements((*projection)->src[1])) * ggml_type_size(GGML_TYPE_BF16));
        CHECK(measured.storage.device_cublas_bytes > 0);
        auto limits = measured.storage;
        limits.device_bytes -= measured.storage.device_cublas_bytes;
        void * short_session = nullptr;
        CHECK(api->prepare(owner->gpu.get(), &query, &limits, arm, &short_session) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY);
        CHECK(short_session == nullptr && owner->references == 0);
    }
    if (flash_attention) {
        auto * context = static_cast<ggml_backend_cuda_context *>(owner->gpu->context);
        ggml_cuda_fattn_resources resources;
        CHECK(ggml_cuda_flash_attn_ext_prepare_resources(*context, fixture.attention_output, resources));
        CHECK(resources.identity != 0 && resources.pool_bytes > 0);
        CHECK(measured.storage.device_workspace_bytes >= resources.pool_bytes);
        ggml_tensor malformed = *fixture.attention_output;
        ggml_tensor malformed_key = *malformed.src[1];
        malformed.src[1] = &malformed_key;
        for (auto type : {static_cast<ggml_type>(4), GGML_TYPE_COUNT}) {
            malformed_key.type = type;
            CHECK(!ggml_cuda_flash_attn_ext_prepare_resources(*context, &malformed, resources));
        }
        malformed_key = *fixture.attention_output->src[1];
        --malformed_key.ne[0];
        CHECK(!ggml_cuda_flash_attn_ext_prepare_resources(*context, &malformed, resources));
        fprintf(stderr, "test-moe-cache: fidelity attention workspace=%zu identity=%llu\n", resources.pool_bytes,
            (unsigned long long) resources.identity);
        auto limits = measured.storage;
        limits.device_bytes -= measured.storage.device_workspace_bytes;
        void * short_session = nullptr;
        CHECK(api->prepare(owner->gpu.get(), &query, &limits, arm, &short_session) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY);
        CHECK(short_session == nullptr && owner->references == 0);
    }
    for (uint32_t bucket = 0; bucket < 4; ++bucket) {
        auto limits = measured.storage;
        uint64_t * sizes[] = {&limits.device_bytes, &limits.pinned_bytes, &limits.cpu_bytes, &limits.metadata_bytes};
        CHECK(*sizes[bucket] > 1);
        --*sizes[bucket];
        void * short_session = nullptr;
        CHECK(api->prepare(owner->gpu.get(), &query, &limits, arm, &short_session) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY);
        CHECK(short_session == nullptr && owner->references == 0);
    }
    void * session = nullptr;
    const auto prepared = api->prepare(owner->gpu.get(), &query, &measured.storage, arm, &session);
    fprintf(stderr, "test-moe-cache: fidelity real prepare R%u arm=%u no_alias=%d status=%d\n", capacity, arm, no_alias, prepared);
    CHECK(prepared == 0 && session && owner->references == 1);
    if (pending_handoff) {
        initial_residency = read_residency();
        for (uint32_t layer = 0; layer < 2; ++layer) {
            for (uint32_t expert = 0; expert < experts; ++expert) {
                CHECK((initial_residency[size_t(layer) * experts + expert] >= 0) == (expert >= experts - routes));
            }
        }
        CHECK(payload_mismatches() == 0);
        fprintf(stderr, "test-moe-cache: reference handoff reconciled canonical maps and source bytes\n");
    }
    ggml_cuda_moe_fidelity_window_state_v1 state = {};
    CHECK(api->state(session, &state));
    CHECK(memcmp(&state.storage, &measured.storage, sizeof(state.storage)) == 0);
    const uint64_t captures = state.captures;
    CHECK(captures == (arm == GGML_CUDA_MOE_FIDELITY_SEGMENTED ? query.n_regions + 1 : 1));
    CHECK(state.protocol_probes == (arm == GGML_CUDA_MOE_FIDELITY_POLL ? 1 : 0));
    CHECK(!no_alias || state.mapped_alias == 0);
    CHECK(state.admission_reserved == 0 && state.admission_aborted == 0 && state.admission_committed == 0);
    CHECK(read_residency() == initial_residency);
    uint64_t epoch = 0;
    std::vector<uint32_t> faults;
    for (uint32_t fault = 0; fault <= GGML_CUDA_MOE_FIDELITY_WINDOW_FAULT_CANCEL_AFTER_COMMIT; ++fault) { faults.push_back(fault); }
    if (native_reference) {
        for (uint32_t fault = GGML_CUDA_MOE_FIDELITY_WINDOW_FAULT_REFERENCE_GROUP_COUNT; fault <= GGML_CUDA_MOE_FIDELITY_WINDOW_FAULT_REFERENCE_BEFORE_B; ++fault) { faults.push_back(fault); }
    }
    if (pending_handoff) { faults = {0}; }
    for (const uint32_t fault : faults) {
        if (fault == GGML_CUDA_MOE_FIDELITY_WINDOW_FAULT_CPU && quota >= experts) { continue; }
        for (uint32_t iteration = 0; iteration < (fault == 0 && !pending_handoff ? 3u : 1u); ++iteration) {
            const uint32_t active_rows = iteration == 1 && (capacity == 4 || (native_reference && capacity > 1)) ? 2 : capacity;
            const uint64_t before_b_guards = state.reference_before_b_guards;
            const uint64_t executed_before = state.executed_selected_bytes;
            const uint64_t transfer_before = state.transfer_bytes;
            const uint64_t accepted_before = state.accepted;
            const uint64_t rejected_before = state.rejected;
            set_inputs(fixture, false, uint32_t(epoch));
            set_inputs(reference, false, uint32_t(epoch));
            CHECK(ggml_backend_sched_graph_compute_ext(owner->oracle.get(), reference.result.get_gf(), &certificate) == GGML_STATUS_SUCCESS);
            std::vector<std::vector<uint8_t>> sentinels;
            for (auto * output : public_outputs) {
                sentinels.emplace_back(ggml_nbytes(output), 0x6b);
                ggml_backend_tensor_set(output, sentinels.back().data(), 0, sentinels.back().size());
            }
            std::vector<uint8_t> effect_sentinel;
            if (external_effect) {
                effect_sentinel.assign(ggml_nbytes(external_effect), 0x53);
                ggml_backend_tensor_set(external_effect, effect_sentinel.data(), 0, effect_sentinel.size());
            }
            ggml_cuda_moe_fidelity_window_replay_v1 replay{++epoch, state.identity, active_rows, fault};
            ggml_cuda_moe_fidelity_sample_v1 sample = {};
            const int32_t status = api->replay(session, &replay, &sample);
            fprintf(stderr, "test-moe-cache: fidelity real R%u active=%u arm=%u no_alias=%d epoch=%llu fault=%u status=%d whole_ns=%llu launch_ns=%llu observe_ns=%llu continuation_ns=%llu cpu_ns=%llu\n",
                capacity, active_rows, arm, no_alias, (unsigned long long) epoch, fault, status,
                (unsigned long long) sample.whole_ns, (unsigned long long) sample.launch_ns,
                (unsigned long long) sample.observation_ns, (unsigned long long) sample.continuation_ns,
                (unsigned long long) sample.cpu_service_ns);
            const bool accepted = fault == 0 || fault == GGML_CUDA_MOE_FIDELITY_WINDOW_FAULT_CANCEL_AFTER_COMMIT;
            CHECK((status == 0) == accepted);
            if (staged_inputs && (accepted || arm == GGML_CUDA_MOE_FIDELITY_POLL)) {
                CHECK(!stage_pending(owner->staged_input.get()));
            }
            if (fault == GGML_CUDA_MOE_FIDELITY_WINDOW_FAULT_GPU) {
                CHECK(status == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_COMPUTE_FAILED);
            }
            if (accepted) {
                if (staged_inputs) {
                    std::vector<uint8_t> actual(ggml_nbytes(fixture.staged));
                    ggml_backend_tensor_get(fixture.staged, actual.data(), 0, actual.size());
                    CHECK(memcmp(actual.data(), stage_api->data(owner->staged_input.get()), actual.size()) == 0);
                }
                if (fault == 0 && iteration == 0) {
                    const auto compare = [&](const numeric_probe & probe) {
                        const auto actual = active_grouped_tensor_values(probe.actual);
                        const auto expected = active_grouped_tensor_values(probe.expected);
                        CHECK(actual.size() == expected.size() && actual.size() % capacity == 0);
                        const size_t count = actual.size() / capacity * active_rows;
                        double error = 0, norm = 0, max_abs = 0;
                        size_t different = 0;
                        for (size_t i = 0; i < count; ++i) {
                            CHECK(std::isfinite(actual[i]) && std::isfinite(expected[i]));
                            const double delta = double(actual[i]) - expected[i];
                            error += delta * delta;
                            norm += double(expected[i]) * expected[i];
                            max_abs = std::max(max_abs, std::abs(delta));
                            different += memcmp(&actual[i], &expected[i], sizeof(float)) != 0;
                        }
                        fprintf(stderr, "test-moe-cache: fidelity numeric %s exact=%d different=%zu/%zu NMSE=%g max_abs=%g expected_l2=%g\n",
                            probe.name, different == 0, different, count, error / std::max(norm, 1e-30), max_abs, norm);
                        if (probe.actual == fixture.attention_output || probe.actual == fixture.attention_prefix) {
                            CHECK(error / std::max(norm, 1e-30) <= 1e-6);
                        }
                        if (probe.routes) {
                            for (uint32_t route = 0; route < active_rows * routes; ++route) {
                                double route_error = 0, route_norm = 0;
                                for (int64_t col = 0; col < probe.actual->ne[0]; ++col) {
                                    const size_t i = size_t(route) * probe.actual->ne[0] + col;
                                    const double delta = double(actual[i]) - expected[i];
                                    route_error += delta * delta;
                                    route_norm += double(expected[i]) * expected[i];
                                }
                                fprintf(stderr, "test-moe-cache: fidelity numeric %s row=%u route=%u NMSE=%g expected_l2=%g\n",
                                    probe.name, route / routes, route % routes, route_error / std::max(route_norm, 1e-30), route_norm);
                            }
                        }
                    };
                    for (const auto & probe : numeric_probes) { compare(probe); }
                    if (cpu_oracle) { fidelity_check_cpu_oracle(*owner, cpu_module, source_owner, active_rows, routes); }
                }
                const auto actual = active_grouped_tensor_values(query.public_output);
                const auto expected = active_grouped_tensor_values(reference.output[1]);
                if (fixture.ordinary_norm_output) {
                    const auto norm_actual = active_grouped_tensor_values(fixture.ordinary_norm_output);
                    const auto norm_expected = active_grouped_tensor_values(reference.ordinary_norm_output);
                    CHECK(norm_actual.size() == norm_expected.size());
                    double norm_error = 0, norm_reference = 0;
                    for (size_t i = 0; i < size_t(active_rows) * signature.n_embd; ++i) {
                        CHECK(std::isfinite(norm_actual[i]) && std::isfinite(norm_expected[i]));
                        const double delta = double(norm_actual[i]) - norm_expected[i];
                        norm_error += delta * delta;
                        norm_reference += double(norm_expected[i]) * norm_expected[i];
                    }
                    fprintf(stderr, "test-moe-cache: source weighted RMS R%u active=%u epoch=%llu NMSE=%g\n",
                        capacity, active_rows, (unsigned long long) epoch, norm_error / std::max(norm_reference, 1e-30));
                    CHECK(norm_error / std::max(norm_reference, 1e-30) <= 1e-10);
                }
                double error = 0, norm = 0;
                for (size_t i = 0; i < size_t(active_rows) * signature.n_embd; ++i) {
                    CHECK(std::isfinite(actual[i]) && std::isfinite(expected[i]));
                    error += (double(actual[i]) - expected[i]) * (double(actual[i]) - expected[i]);
                    norm += double(expected[i]) * expected[i];
                }
                fprintf(stderr, "test-moe-cache: fidelity real R%u active=%u epoch=%llu NMSE=%g\n", capacity, active_rows,
                    (unsigned long long) epoch, error / std::max(norm, 1e-30));
                CHECK(error / std::max(norm, 1e-30) <= 5e-3); // CPU/GPU sanity screen; fault publication is exact below.
            } else {
                for (size_t i = 0; i < public_outputs.size(); ++i) {
                    std::vector<uint8_t> actual(sentinels[i].size());
                    ggml_backend_tensor_get(public_outputs[i], actual.data(), 0, actual.size());
                    CHECK(actual == sentinels[i]);
                }
            }
            if (external_effect) {
                std::vector<uint8_t> actual(effect_sentinel.size());
                ggml_backend_tensor_get(external_effect, actual.data(), 0, actual.size());
                if (accepted) {
                    std::vector<uint8_t> expected(actual.size());
                    ggml_backend_tensor_get(query.public_output, expected.data(), 0, expected.size());
                    CHECK(actual == expected);
                } else if (fault != GGML_CUDA_MOE_FIDELITY_WINDOW_FAULT_CANCEL_BEFORE_COMMIT) {
                    CHECK(actual == effect_sentinel);
                }
            }
            CHECK(api->state(session, &state) && state.active_jobs == 0 && state.retries == 0 && state.fallbacks == 0);
            if (!native_reference && quota >= experts) { CHECK(state.cpu_routes == 0); }
            if (native_reference) {
                CHECK(state.planning_callbacks == 0 && state.transfer_callbacks == 0);
                if (fault == GGML_CUDA_MOE_FIDELITY_WINDOW_FAULT_REFERENCE_BEFORE_B) { CHECK(state.reference_before_b_guards == before_b_guards + 1); }
                if (accepted && capacity == 9) { CHECK(state.cpu_max_group_entries >= 9); }
                if (accepted) { CHECK(state.executed_selected_bytes - executed_before == state.transfer_bytes - transfer_before); }
                if (accepted && pending_handoff) {
                    CHECK(state.resident_routes == 8 && state.transfer_routes == 2 && state.cpu_routes == 10);
                    CHECK(state.executed_selected_bytes > 0 && payload_mismatches() == 0);
                }
            }
            CHECK(state.accepted == accepted_before + accepted && state.rejected == rejected_before + !accepted);
            CHECK(state.published == accepted);
            CHECK(state.captures == captures && state.address_changes == 0);
            CHECK(state.admission_reserved == 0 && state.admission_aborted == 0 && state.admission_committed == 0);
            CHECK(read_residency() == initial_residency);
        }
    }
    ggml_cuda_moe_fidelity_sample_v1 sample = {};
    ggml_cuda_moe_fidelity_window_replay_v1 stale{epoch + 1, state.identity + 1, capacity, 0};
    CHECK(api->replay(session, &stale, &sample) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
    if (staged_inputs) { stage_api->publish(owner->staged_input.get()); }
    stale = {epoch + 1, state.identity, capacity, GGML_CUDA_MOE_FIDELITY_WINDOW_FAULT_TEARDOWN};
    CHECK(api->replay(session, &stale, &sample) != 0);
    CHECK(api->state(session, &state) && state.active_jobs == 0);
    CHECK(state.admission_reserved == 0 && state.admission_aborted == 0 && state.admission_committed == 0);
    CHECK(read_residency() == initial_residency);
    fprintf(stderr, "test-moe-cache: fidelity real lifecycle R%u arm=%u no_alias=%d captures=%llu launches=%llu epochs=%llu accepted=%llu rejected=%llu resident=%llu transfer=%llu cpu_routes=%llu producers=%llu/%llu admissions=%llu/%llu/%llu drains=%llu\n",
        capacity, arm, no_alias, (unsigned long long) state.captures, (unsigned long long) state.launches,
        (unsigned long long) state.epochs, (unsigned long long) state.accepted, (unsigned long long) state.rejected,
        (unsigned long long) state.resident_routes, (unsigned long long) state.transfer_routes,
        (unsigned long long) state.cpu_routes, (unsigned long long) state.producers_completed,
        (unsigned long long) state.producers_expected, (unsigned long long) state.admission_reserved,
        (unsigned long long) state.admission_aborted, (unsigned long long) state.admission_committed,
        (unsigned long long) state.finite_drains);
    CHECK(api->close(session) == 0 && api->drain(session) == 0 && api->destroy(&session) == 0);
    CHECK(!session && owner->references == 0);
    CHECK(read_residency() == initial_residency);
    if (staged_inputs && arm == GGML_CUDA_MOE_FIDELITY_POLL && capacity == 1) {
        CHECK(!stage_pending(owner->staged_input.get()) && external_effect);
        CHECK(api->prepare(owner->gpu.get(), &query, &measured.storage, arm, &session) == 0 && session && owner->references == 1);
        CHECK(api->state(session, &state));
        set_inputs(fixture, false, 0, false);
        std::vector<std::pair<ggml_tensor *, std::vector<uint8_t>>> sentinels;
        for (auto * output : {query.public_output, fixture.staged, external_effect}) {
            sentinels.push_back({output, std::vector<uint8_t>(ggml_nbytes(output), 0x65)});
            ggml_backend_tensor_set(output, sentinels.back().second.data(), 0, sentinels.back().second.size());
        }
        void * input_backing = stage_api->data(owner->staged_input.get());
        std::vector<uint8_t> input_contents(ggml_nbytes(fixture.staged));
        memcpy(input_contents.data(), input_backing, input_contents.size());
        ggml_cuda_moe_fidelity_window_replay_v1 held{1, state.identity, capacity, GGML_CUDA_MOE_FIDELITY_WINDOW_FAULT_GPU};
        ggml_cuda_moe_fidelity_sample_v1 held_sample = {};
        int32_t result = 0;
        std::thread caller([&] { CUDA_OK(cudaSetDevice(device)); result = api->replay(session, &held, &held_sample); });
        const auto expiry = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        do {
            CHECK(api->state(session, &state));
            if (state.cpu_calls) { break; }
            std::this_thread::yield();
        } while (std::chrono::steady_clock::now() < expiry);
        CHECK(api->close(session) == 0);
        caller.join();
        CHECK(result != 0 && held_sample.cpu_service_ns == 0 && owner->references == 1);
        CHECK(api->state(session, &state) && state.cpu_calls == 1 && state.active_jobs == 0);
        CHECK(!stage_pending(owner->staged_input.get()));
        void * retained = session;
        CHECK(api->destroy(&session) != 0 && session == retained && owner->references == 1);
        CHECK(stage_api->data(owner->staged_input.get()) == input_backing);
        CHECK(memcmp(input_contents.data(), input_backing, input_contents.size()) == 0);
        CHECK(api->state(session, &state) && state.address_changes == 0 && state.active_jobs == 0 && !state.published);
        CHECK(memcmp(&state.storage, &measured.storage, sizeof(state.storage)) == 0);
        CHECK(api->replay(session, &held, &held_sample) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
        stage_api->publish(owner->staged_input.get());
        CHECK(api->drain(session) == 0 && !stage_pending(owner->staged_input.get()));
        for (const auto & sentinel : sentinels) {
            std::vector<uint8_t> actual(sentinel.second.size());
            ggml_backend_tensor_get(sentinel.first, actual.data(), 0, actual.size());
            CHECK(actual == sentinel.second);
        }
        CHECK(api->destroy(&session) == 0 && !session && owner->references == 0);
        CHECK(read_residency() == initial_residency);
        CHECK(api->prepare(owner->gpu.get(), &query, &measured.storage, arm, &session) == 0 && api->state(session, &state));
        set_inputs(fixture, false, 1);
        ggml_cuda_moe_fidelity_window_replay_v1 recovered{1, state.identity, capacity, 0};
        CHECK(api->replay(session, &recovered, &sample) == 0 && !stage_pending(owner->staged_input.get()));
        CHECK(api->state(session, &state) && state.accepted == 1 && state.active_jobs == 0);
        std::vector<uint8_t> actual(ggml_nbytes(query.public_output)), effect(actual.size());
        ggml_backend_tensor_get(query.public_output, actual.data(), 0, actual.size());
        ggml_backend_tensor_get(external_effect, effect.data(), 0, effect.size());
        CHECK(actual == effect);
        actual.resize(ggml_nbytes(fixture.staged));
        ggml_backend_tensor_get(fixture.staged, actual.data(), 0, actual.size());
        CHECK(memcmp(actual.data(), stage_api->data(owner->staged_input.get()), actual.size()) == 0);
        CHECK(api->close(session) == 0 && api->drain(session) == 0 && api->destroy(&session) == 0);
        CHECK(!session && owner->references == 0 && read_residency() == initial_residency);
        fprintf(stderr, "test-moe-cache: fidelity held staged input R%u arm=%u owner/backing/retention/publication/recovery OK\n", capacity, arm);
    }
    if (delayed_cancel) {
        CHECK(api->struct_size >= sizeof(*api) && api->set_cpu_test_hook && api->test_snapshot && external_effect);
        struct barrier {
            std::mutex mutex;
            std::condition_variable condition;
            bool entered = false, released = false, timed_out = false, fail = false;
            static void hook(void * opaque, uint32_t phase) {
                if (phase != GGML_BACKEND_MOE_CPU_TEST_PHASE_V1_AFTER_COMMIT) { return; }
                auto & self = *static_cast<barrier *>(opaque);
                std::unique_lock<std::mutex> lock(self.mutex);
                if (self.entered) { return; }
                self.entered = true;
                self.condition.notify_all();
                self.timed_out = !self.condition.wait_for(lock, std::chrono::seconds(30), [&] { return self.released; });
                if (self.fail) { throw 1; }
            }
            void release() {
                std::lock_guard<std::mutex> lock(mutex);
                released = true;
                condition.notify_all();
            }
        };
        for (uint32_t scenario = 0; scenario < 3; ++scenario) {
            barrier paused;
            paused.fail = scenario == 2;
            const bool retain = scenario == 1;
            CHECK(api->prepare(owner->gpu.get(), &query, &measured.storage, arm, &session) == 0 && session && owner->references == 1);
            CHECK(api->state(session, &state));
            CHECK(api->set_cpu_test_hook(session, barrier::hook, &paused) == 0);
            set_inputs(fixture, false, 0);
            std::vector<uint8_t> sentinel(ggml_nbytes(query.public_output), 0x65);
            ggml_backend_tensor_set(query.public_output, sentinel.data(), 0, sentinel.size());
            ggml_backend_tensor_set(external_effect, sentinel.data(), 0, sentinel.size());
            ggml_cuda_moe_fidelity_window_replay_v1 replay{1, state.identity, capacity, 0};
            ggml_cuda_moe_fidelity_sample_v1 delayed_sample = {};
            int32_t result = 0;
            std::thread caller([&] { CUDA_OK(cudaSetDevice(device)); result = api->replay(session, &replay, &delayed_sample); });
            {
                std::unique_lock<std::mutex> lock(paused.mutex);
                const bool entered = paused.condition.wait_for(lock, std::chrono::seconds(5), [&] { return paused.entered; });
                if (!entered) { paused.released = true; paused.condition.notify_all(); lock.unlock(); caller.join(); CHECK(entered); }
            }
            ggml_cuda_moe_fidelity_window_test_snapshot_v1 before = {}, after = {};
            CHECK(api->test_snapshot(session, &before) && before.claimed == 1 && before.cpu_terminal == 0);
            CHECK(api->set_cpu_test_hook(session, nullptr, nullptr) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
            CHECK(api->close(session) == 0);
            const auto expiry = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            do {
                CHECK(api->test_snapshot(session, &after));
                if (after.streams_complete) { break; }
                std::this_thread::yield();
            } while (std::chrono::steady_clock::now() < expiry);
            CHECK(after.streams_complete && after.wake && after.stopped && !after.cpu_terminal);
            CHECK(after.input_hash == before.input_hash && after.control_hash == before.control_hash && after.output_hash == before.output_hash);
            CHECK(after.host_address == before.host_address && after.device_address == before.device_address);
            CHECK(after.imported_bytes == 0 && after.join_cpu_terminal == 0);
            CHECK(after.join_complete == (arm == GGML_CUDA_MOE_FIDELITY_POLL ? 1 : 0));
            if (retain) {
                if (ggml_moe_fidelity_selection().source_pool) {
                    CHECK(api->drain(session) != 0);
                } else {
                    caller.join();
                    CHECK(result != 0 && delayed_sample.cpu_service_ns == 0 && owner->references == 1);
                }
                CHECK(api->state(session, &state) && state.active_jobs == 1);
                void * retained = session;
                CHECK(api->destroy(&session) != 0 && session == retained && owner->references == 1);
                CHECK(api->test_snapshot(session, &after) && !after.cpu_terminal);
                CHECK(after.host_address == before.host_address && after.device_address == before.device_address);
                CHECK(after.input_hash == before.input_hash && after.control_hash == before.control_hash && after.output_hash == before.output_hash);
                CHECK(api->replay(session, &replay, &delayed_sample) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT);
            }
            paused.release();
            if (caller.joinable()) { caller.join(); }
            CHECK(!paused.timed_out && result != 0);
            CHECK(api->drain(session) == 0 && api->state(session, &state) && state.active_jobs == 0 && state.cpu_calls == 1);
            for (auto * output : {query.public_output, external_effect}) {
                std::vector<uint8_t> actual(sentinel.size());
                ggml_backend_tensor_get(output, actual.data(), 0, actual.size());
                CHECK(actual == sentinel);
            }
            CHECK(api->close(session) == 0 && api->drain(session) == 0 && api->destroy(&session) == 0);
            CHECK(!session && owner->references == 0 && read_residency() == initial_residency);
            fprintf(stderr, "test-moe-cache: fidelity delayed CPU R%u arm=%u no_alias=%d scenario=%u terminal/import/effect/retention OK\n",
                capacity, arm, no_alias, scenario);
        }
        CHECK(api->prepare(owner->gpu.get(), &query, &measured.storage, arm, &session) == 0 && api->state(session, &state));
        ggml_cuda_moe_fidelity_window_replay_v1 recovered{1, state.identity, capacity, 0};
        sample = {};
        CHECK(api->replay(session, &recovered, &sample) == 0);
        CHECK(api->state(session, &state) && state.accepted == 1 && state.active_jobs == 0);
        std::vector<uint8_t> recovered_output(ggml_nbytes(query.public_output)), recovered_effect(recovered_output.size());
        ggml_backend_tensor_get(query.public_output, recovered_output.data(), 0, recovered_output.size());
        ggml_backend_tensor_get(external_effect, recovered_effect.data(), 0, recovered_effect.size());
        CHECK(recovered_output == recovered_effect);
        CHECK(api->close(session) == 0 && api->drain(session) == 0 && api->destroy(&session) == 0);
        CHECK(!session && owner->references == 0);
    }
    CHECK(ggml_backend_moe_module_release_v1(cpu_module));
    fprintf(stderr, "test-moe-cache: fidelity real window R%u arm=%u no_alias=%d arithmetic/private/fault/replay/bounds/drain OK\n", capacity, arm, no_alias);
}

static void test_cpu_prefill_partition(int device) {
    ggml_backend_ptr gpu(ggml_backend_cuda_init(device));
    CHECK(gpu);
    auto & cuda = *static_cast<ggml_backend_cuda_context *>(gpu->context);
    const auto stream = ggml_cuda_mmid_execution_stream_for_test(gpu.get());
    auto * reg = ggml_backend_cpu_reg();
    const auto get = reinterpret_cast<ggml_backend_moe_cpu_region_service_v1_t>(
        ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_CPU_REGION_SERVICE_V1_PROC_NAME));
    const auto execute = reinterpret_cast<ggml_backend_moe_cpu_controlled_execute_v1_t>(
        ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_CPU_CONTROLLED_EXECUTE_V1_PROC_NAME));
    CHECK(get && execute);
    const auto * api = get();
    uint32_t cases = 0, total_waves = 0;
    for (const auto type : {GGML_TYPE_Q4_0, GGML_TYPE_Q5_K, GGML_TYPE_IQ4_NL, GGML_TYPE_Q8_0, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_F32}) {
        for (uint32_t rows : {9u, 65u}) {
            for (uint32_t period : {1u, 3u}) {
                constexpr uint32_t width = 256, output_width = 37, experts = 7, routes_per_row = 3;
                const uint32_t routes = rows * routes_per_row;
                ggml_context_ptr cpu_ctx(ggml_init({2 * 1024 * 1024, nullptr, false}));
                ggml_context_ptr gpu_ctx(ggml_init({ggml_tensor_overhead() * 8 + ggml_graph_overhead_custom(8, false), nullptr, true}));
                CHECK(cpu_ctx && gpu_ctx);
                auto * weight = ggml_new_tensor_3d(cpu_ctx.get(), type, width, output_width, experts);
                auto * input = ggml_new_tensor_3d(cpu_ctx.get(), GGML_TYPE_F32, width, period, rows);
                auto * ids = ggml_new_tensor_2d(cpu_ctx.get(), GGML_TYPE_I32, routes_per_row, rows);
                auto * projection = ggml_mul_mat_id(cpu_ctx.get(), weight, input, ids);
                auto * cpu_graph = ggml_new_graph_custom(cpu_ctx.get(), 8, false);
                ggml_build_forward_expand(cpu_graph, projection);
                CHECK(ggml_backend_moe_graph_assign_uid_v1(cpu_graph));
                const auto source = cached_fusion_test_data(weight, 191);
                memcpy(weight->data, source.data(), source.size());
                std::vector<float> activation(ggml_nelements(input));
                for (size_t n = 0; n < activation.size(); ++n) { activation[n] = 0.003f * float(int(n % 23) - 11); }
                std::vector<int32_t> original_ids(routes);
                std::vector<uint32_t> source_rows(routes), destinations(routes);
                for (uint32_t route = 0; route < routes; ++route) {
                    original_ids[route] = route % 5 ? int32_t(route % experts) : 0;
                    source_rows[route] = route / routes_per_row; destinations[route] = route;
                }
                auto model = std::unique_ptr<llama_model>(llama_model_create(LLM_ARCH_QWEN3MOE, llama_model_default_params()));
                CHECK(model && model->record_moe_readable_source(weight, weight->data, ggml_nbytes(weight)));
                ggml_backend_moe_source_owner_v1 owner = {};
                CHECK(model->moe_source_owner_v1(&owner));
                const ggml_tensor * body[] = {projection}, * dynamic[] = {input, ids}, * outputs[] = {projection};
                ggml_backend_moe_cpu_region_source_v1 bank = {weight, weight, weight->data, ggml_nbytes(weight), weight->nb[2], owner.generation};
                ggml_backend_moe_cpu_region_query_v1 query = {};
                query.struct_size = sizeof(query); query.flags = GGML_BACKEND_MOE_CPU_REGION_FLAG_V1_ROUTED_OPERATION;
                query.graph = cpu_graph; query.graph_uid = ggml_backend_moe_graph_uid_v1(cpu_graph); query.graph_generation = 1;
                query.source_generation = owner.generation; query.body_nodes = body; query.n_body_nodes = 1;
                query.activation = input; query.ids = ids; query.dynamic_inputs = dynamic; query.n_dynamic_inputs = 2;
                query.live_outputs = outputs; query.n_live_outputs = 1; query.sources = &bank; query.n_sources = 1;
                query.bucket_rows = rows; query.routes_per_row = routes_per_row;
                query.source_row_capacity = rows; query.scatter_capacity = routes; query.n_lanes = 1; query.n_threads = 2;
                ggml_backend_moe_cpu_service_config_v1 config = {};
                config.struct_size = sizeof(config); config.abi_version = 1; config.source_owner = &owner;
                config.n_threads = 2; config.n_lanes = 1; config.max_regions = 1;
                config.flags = GGML_BACKEND_MOE_CPU_SERVICE_FLAG_V1_ALLOW_UNKNOWN_THREAD_STACK_BYTES |
                    GGML_BACKEND_MOE_CPU_SERVICE_FLAG_V1_ALLOW_UNPROVEN_RUNTIME_ALLOCATIONS;
                ggml_backend_moe_cpu_service_v1_t service = nullptr;
                ggml_backend_moe_cpu_prepared_region_v1_t region = 0;
                ggml_backend_moe_cpu_prepared_requirements_v1 requirements = {};
                requirements.struct_size = sizeof(requirements); requirements.abi_version = 1;
                CHECK(api->create(&config, &service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
                CHECK(api->prepare(service, &query, &requirements, &region) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
                const auto cpu_reference = evaluate_body({projection}, {input, ids}, {projection}, {bank}, {activation.data(), original_ids.data()});
                auto * gpu_weight = ggml_new_tensor_3d(gpu_ctx.get(), type, width, output_width, experts);
                auto * gpu_input = ggml_new_tensor_3d(gpu_ctx.get(), GGML_TYPE_F32, width, period, rows);
                auto * gpu_ids = ggml_new_tensor_2d(gpu_ctx.get(), GGML_TYPE_I32, routes_per_row, rows);
                auto * candidate = ggml_mul_mat_id(gpu_ctx.get(), gpu_weight, gpu_input, gpu_ids);
                auto * reference = ggml_mul_mat_id(gpu_ctx.get(), gpu_weight, gpu_input, gpu_ids);
                if (period > 1) {
                    candidate->nb[2] += candidate->nb[1]; candidate->nb[3] = candidate->nb[2] * rows;
                }
                ggml_set_output(candidate); ggml_set_output(reference);
                auto * graph = ggml_new_graph_custom(gpu_ctx.get(), 8, false);
                ggml_build_forward_expand(graph, candidate); ggml_build_forward_expand(graph, reference);
                auto * allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gpu.get()));
                CHECK(allocator && ggml_gallocr_alloc_graph(allocator, graph));
                ggml_backend_tensor_set(gpu_weight, source.data(), 0, source.size());
                ggml_backend_tensor_set(gpu_input, activation.data(), 0, ggml_nbytes(gpu_input));
                ggml_backend_tensor_set(gpu_ids, original_ids.data(), 0, ggml_nbytes(gpu_ids));
                auto reference_graph = ggml_graph_view(graph, graph->n_nodes - 1, graph->n_nodes);
                CHECK(ggml_backend_graph_compute(gpu.get(), &reference_graph) == GGML_STATUS_SUCCESS);
                std::vector<float> gpu_reference(ggml_nelements(reference)), actual(gpu_reference.size()), cpu_output(gpu_reference.size());
                std::vector<float> storage(ggml_nbytes(candidate) / sizeof(float));
                const auto read_candidate = [&]() {
                    ggml_backend_tensor_get(candidate, storage.data(), 0, ggml_nbytes(candidate));
                    for (uint32_t route = 0; route < routes; ++route) {
                        const size_t offset = (route / routes_per_row * candidate->nb[2] + route % routes_per_row * candidate->nb[1]) / sizeof(float);
                        std::copy_n(storage.begin() + offset, output_width, actual.begin() + route * output_width);
                    }
                };
                ggml_backend_tensor_get(reference, gpu_reference.data(), 0, ggml_nbytes(reference));
                void * resident = nullptr, * staging = nullptr;
                const size_t expert_stride = weight->nb[2];
                CUDA_OK(cudaMalloc(&resident, expert_stride + 512));
                CUDA_OK(cudaMalloc(&staging, expert_stride + 512));
                CUDA_OK(cudaMemcpyAsync(resident, source.data() + (experts - 1) * expert_stride, expert_stride, cudaMemcpyHostToDevice, stream));
                for (uint32_t pattern = 0; pattern < 4; ++pattern) {
                    std::vector<uint8_t> cpu_experts(experts), ownership(routes);
                    for (uint32_t expert = 0; expert < experts; ++expert) {
                        cpu_experts[expert] = pattern == 1 || (pattern == 2 && expert % 2 == 0) || (pattern == 3 && expert == experts - 1);
                    }
                    for (uint32_t route = 0; route < routes; ++route) { ownership[route] = cpu_experts[original_ids[route]]; }
                    std::fill(storage.begin(), storage.end(), -123.0f);
                    ggml_backend_tensor_set(candidate, storage.data(), 0, ggml_nbytes(candidate));
                    auto bad = cpu_experts; bad.back() = 2;
                    const auto prepare = [&](const uint8_t * mask, uint32_t count) {
                        return ggml_cuda_mmid_prefill_prepare_partition(cuda, candidate,
                            reinterpret_cast<const char *>(original_ids.data()), original_ids.size() * sizeof(int32_t), ids->nb[1], mask, count);
                    };
                    CHECK(!prepare(nullptr, experts) && !prepare(cpu_experts.data(), experts - 1) && !prepare(bad.data(), experts));
                    auto invalid_ids = original_ids;
                    invalid_ids.back() = experts;
                    CHECK(!ggml_cuda_mmid_prefill_prepare_partition(cuda, candidate,
                        reinterpret_cast<const char *>(invalid_ids.data()), invalid_ids.size() * sizeof(int32_t), ids->nb[1], cpu_experts.data(), experts));
                    auto prepared = std::unique_ptr<ggml_cuda_mmid_prefill_prepared, decltype(&ggml_cuda_mmid_prefill_free)>(
                        prepare(cpu_experts.data(), experts), &ggml_cuda_mmid_prefill_free);
                    CHECK(prepared);
                    std::fill(cpu_experts.begin(), cpu_experts.end(), 2);
                    std::vector<int32_t> map(experts, -1);
                    map.back() = 0;
                    if (pattern != 1) {
                        CHECK(!ggml_cuda_mmid_prefill_launch_range(cuda, prepared.get(), resident, staging, map.data(), 1, 1, 0, experts));
                        read_candidate();
                        CHECK(std::all_of(actual.begin(), actual.end(), [](float value) { return value == -123.0f; }));
                    }
                    CHECK(ggml_cuda_mmid_prefill_launch_range(cuda, prepared.get(), resident, staging, map.data(), 1, 1, experts - 1, 1));
                    uint32_t waves = 0;
                    for (uint32_t expert = 0; expert + 1 < experts; ++expert) {
                        const auto route = std::find(original_ids.begin(), original_ids.end(), int32_t(expert));
                        if (route == original_ids.end() || ownership[route - original_ids.begin()]) { continue; }
                        CUDA_OK(cudaMemcpyAsync(staging, source.data() + expert * expert_stride, expert_stride, cudaMemcpyHostToDevice, stream));
                        map[expert] = 1;
                        CHECK(ggml_cuda_mmid_prefill_launch_range(cuda, prepared.get(), resident, staging, map.data(), 1, 1, expert, 1));
                        map[expert] = -1; ++waves;
                    }
                    CHECK(ggml_cuda_mmid_prefill_finish(cuda, prepared.get()));
                    std::fill(cpu_output.begin(), cpu_output.end(), std::numeric_limits<float>::quiet_NaN());
                    ggml_backend_moe_cpu_region_binding_v1 binding{sizeof(binding), rows, routes,
                        original_ids.data(), source_rows.data(), destinations.data()};
                    ggml_backend_moe_cpu_dynamic_input_v1 inputs[] = {{activation.data(), ggml_nbytes(input), input->nb[2]}, {}};
                    ggml_backend_moe_cpu_output_v1 output{cpu_output.data(), ggml_nbytes(projection), projection->nb[1]};
                    ggml_backend_moe_cpu_execute_v1 job = {};
                    job.struct_size = sizeof(job); job.epoch = pattern + 1;
                    job.graph_uid = query.graph_uid; job.graph_generation = query.graph_generation; job.source_generation = query.source_generation;
                    job.binding = &binding; job.dynamic_inputs = inputs; job.n_dynamic_inputs = 2; job.outputs = &output; job.n_outputs = 1;
                    ggml_backend_moe_cpu_execute_control_v1 control = {}; control.struct_size = sizeof(control); control.abi_version = 1;
                    ggml_backend_moe_cpu_execute_result_v1 result = {}; result.struct_size = sizeof(result);
                    CHECK(execute(service, region, &job, ownership.data(), routes, &control, &result) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
                    CHECK(result.published_routes == uint32_t(std::count(ownership.begin(), ownership.end(), uint8_t(1))));
                    if (result.published_routes) {
                        read_candidate();
                        const auto unpublished = actual;
                        const size_t last = std::find(ownership.rbegin(), ownership.rend(), uint8_t(1)).base() - ownership.begin() - 1;
                        CHECK(!ggml_cuda_mmid_prefill_join_cpu(cuda, prepared.get(), nullptr, ggml_nbytes(projection), projection->nb[1]));
                        CHECK(!ggml_cuda_mmid_prefill_join_cpu(cuda, prepared.get(), cpu_output.data(), 0, projection->nb[1]));
                        CHECK(!ggml_cuda_mmid_prefill_join_cpu(cuda, prepared.get(), cpu_output.data(), (last + 1) * projection->nb[1] - 1, projection->nb[1]));
                        CHECK(!ggml_cuda_mmid_prefill_join_cpu(cuda, prepared.get(), cpu_output.data(), ggml_nbytes(projection), sizeof(float)));
                        read_candidate();
                        CHECK(actual == unpublished);
                    }
                    CHECK(ggml_cuda_mmid_prefill_join_cpu(cuda, prepared.get(), cpu_output.data(), ggml_nbytes(projection), projection->nb[1]));
                    ggml_backend_synchronize(gpu.get());
                    read_candidate();
                    if (period > 1) {
                        for (uint32_t row = 0; row + 1 < rows; ++row) {
                            const size_t first = (row * candidate->nb[2] + routes_per_row * candidate->nb[1]) / sizeof(float);
                            CHECK(std::all_of(storage.begin() + first, storage.begin() + first + output_width,
                                [](float value) { return value == -123.0f; }));
                        }
                    }
                    double difference = 0, magnitude = 0;
                    for (size_t n = 0; n < actual.size(); ++n) {
                        const float expected = ownership[n / output_width] ? cpu_reference[n] : gpu_reference[n];
                        CHECK(std::isfinite(actual[n]) && std::isfinite(expected));
                        if (!ownership[n / output_width]) { CHECK(std::isnan(cpu_output[n])); }
                        difference += double(actual[n] - expected) * double(actual[n] - expected); magnitude += double(expected) * expected;
                    }
                    const double mse = difference / std::max(magnitude, 1e-30);
                    CHECK(mse < 2e-5);
                    ++cases; total_waves += waves;
                    fprintf(stderr, "test-moe-cache: CPU prefill partition type=%s rows=%u period=%u padded=%u pattern=%u cpu_routes=%u waves=%u relative_mse=%.9g OK\n",
                        ggml_type_name(type), rows, period, unsigned(period > 1), pattern, result.published_routes, waves, mse);
                }
                CUDA_OK(cudaFree(staging)); CUDA_OK(cudaFree(resident));
                ggml_gallocr_free(allocator);
                CHECK(api->destroy_region(service, &region) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
                CHECK(api->close(service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
                CHECK(api->destroy(&service) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
            }
        }
    }
    fprintf(stderr, "test-moe-cache: CPU prefill partition cases=%u bounded_waves=%u actual CPU service and ordinary GPU references OK\n", cases, total_waves);
}

static void test_cpu_prefill_owner(int device) {
    const bool old_debug = ggml_backend_cuda_moe_get_debug_mm();
    ggml_backend_cuda_moe_set_debug_mm(true);
    auto * reg = ggml_backend_cpu_reg();
    const auto get = reinterpret_cast<ggml_backend_moe_cpu_region_service_v1_t>(
        ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_CPU_REGION_SERVICE_V1_PROC_NAME));
    const auto execute = reinterpret_cast<ggml_backend_moe_cpu_controlled_execute_v1_t>(
        ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_CPU_CONTROLLED_EXECUTE_V1_PROC_NAME));
    CHECK(get && execute);
    const auto * api = get();
    uint32_t cases = 0;
    for (const auto type : {GGML_TYPE_Q5_K, GGML_TYPE_IQ4_NL, GGML_TYPE_F32}) {
        for (uint32_t variant = 0; variant < 4; ++variant) {
            const bool pageable = (variant & 1) != 0;
            const bool sparse = (variant & 2) != 0;
            ggml_backend_ptr reference_backend(ggml_backend_cuda_init(device)), backend(ggml_backend_cuda_init(device));
            CHECK(reference_backend && backend);
            auto reference = build_active_grouped_dispatch_graph_types(reference_backend.get(), ggml_backend_cuda_buffer_type(device),
                {type, type, type}, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, false, false, 128, 16, 4, 256, nullptr, false, false, 256);
            auto fixture = build_active_grouped_dispatch_graph_types(backend.get(),
                pageable ? pageable_cached_buffer_type() : ggml_backend_cuda_moe_cached_buffer_type(),
                {type, type, type}, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, false, false, 128, 16, 4, 256, nullptr, false, false, 256);
            ggml_set_output(reference.up_output); ggml_set_output(reference.gate_output);
            initialize_active_grouped_dispatch_graphs({&reference, &fixture});
            if (sparse) {
                const int32_t active[] = {2, 3, 4, 15};
                CHECK(fixture.n_used == 4 && fixture.n_experts == 16);
                std::vector<int32_t> routes(size_t(fixture.n_rows) * fixture.n_used);
                for (size_t route = 0; route < routes.size(); ++route) { routes[route] = active[route % fixture.n_used]; }
                set_active_grouped_dispatch_routes({&reference, &fixture}, routes);
            } else { set_active_grouped_dispatch_logits({&reference, &fixture}, 0); }
            CHECK(ggml_backend_graph_compute(reference_backend.get(), reference.graph) == GGML_STATUS_SUCCESS);
            CHECK(ggml_backend_graph_compute(backend.get(), fixture.graph) == GGML_STATUS_SUCCESS);
            ggml_backend_synchronize(reference_backend.get()); ggml_backend_synchronize(backend.get());
            register_active_grouped_dispatch(backend.get(), fixture, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 5);
            auto * owner = ggml_cuda_moe_grouped_context_for_test(backend.get());
            CHECK(owner);
            CHECK(ggml_cuda_moe_grouped_context_test_access::set_prefill_staging_lane_bytes(*owner, fixture.up_output->src[0]->nb[2] + 4096));
            candidate_stamp_execution(fixture.graph, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
                GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL, fixture.n_rows, 1);
            auto & cuda = *static_cast<ggml_backend_cuda_context *>(backend->context);
            const auto stream = ggml_cuda_mmid_execution_stream_for_test(backend.get());
            auto * original = fixture.up_output;
            CHECK(ggml_is_contiguous(original->src[1]));
            std::vector<float> activation(ggml_nelements(original->src[1]));
            std::vector<int32_t> ids(ggml_nelements(original->src[2]));
            ggml_backend_tensor_get(original->src[1], activation.data(), 0, ggml_nbytes(original->src[1]));
            CHECK(original->src[2]->type == GGML_TYPE_I32 && original->src[2]->ne[2] == 1 && original->src[2]->ne[3] == 1);
            std::vector<uint8_t> ids_storage(ggml_nbytes(original->src[2]));
            ggml_backend_tensor_get(original->src[2], ids_storage.data(), 0, ids_storage.size());
            for (int64_t row = 0; row < original->src[2]->ne[1]; ++row) {
                for (int64_t col = 0; col < original->src[2]->ne[0]; ++col) {
                    memcpy(&ids[row * original->src[2]->ne[0] + col],
                        ids_storage.data() + row * original->src[2]->nb[1] + col * original->src[2]->nb[0], sizeof(int32_t));
                }
            }
            std::vector<int64_t> counts(fixture.n_experts, 0);
            std::vector<int32_t> experts;
            for (const auto id : ids) { CHECK(id >= 0 && uint32_t(id) < fixture.n_experts); if (!counts[id]++) { experts.push_back(id); } }
            ggml_context_ptr cpu_ctx(ggml_init({16 * 1024 * 1024, nullptr, false}));
            CHECK(cpu_ctx);
            auto * weight = ggml_dup_tensor(cpu_ctx.get(), original->src[0]);
            auto * input = ggml_dup_tensor(cpu_ctx.get(), original->src[1]);
            auto * routes = ggml_dup_tensor(cpu_ctx.get(), original->src[2]);
            auto * projection = ggml_mul_mat_id(cpu_ctx.get(), weight, input, routes);
            memcpy(projection->op_params, original->op_params, sizeof(projection->op_params));
            auto * graph = ggml_new_graph_custom(cpu_ctx.get(), 8, false);
            ggml_build_forward_expand(graph, projection);
            CHECK(ggml_backend_moe_graph_assign_uid_v1(graph));
            auto model = std::unique_ptr<llama_model>(llama_model_create(LLM_ARCH_QWEN3MOE, llama_model_default_params()));
            CHECK(model && model->record_moe_readable_source(original->src[0], original->src[0]->data, ggml_nbytes(original->src[0])));
            ggml_backend_moe_source_owner_v1 source_owner = {};
            CHECK(model->moe_source_owner_v1(&source_owner));
            const ggml_tensor * body[] = {projection}, * dynamic[] = {input, routes}, * outputs[] = {projection};
            ggml_backend_moe_cpu_region_source_v1 bank = {weight, original->src[0], original->src[0]->data,
                ggml_nbytes(original->src[0]), original->src[0]->nb[2], source_owner.generation};
            ggml_backend_moe_cpu_region_query_v1 query = {};
            query.struct_size = sizeof(query); query.flags = GGML_BACKEND_MOE_CPU_REGION_FLAG_V1_ROUTED_OPERATION;
            query.graph = graph; query.graph_uid = ggml_backend_moe_graph_uid_v1(graph); query.graph_generation = 1;
            query.source_generation = source_owner.generation; query.body_nodes = body; query.n_body_nodes = 1;
            query.activation = input; query.ids = routes; query.dynamic_inputs = dynamic; query.n_dynamic_inputs = 2;
            query.live_outputs = outputs; query.n_live_outputs = 1; query.sources = &bank; query.n_sources = 1;
            query.bucket_rows = fixture.n_rows; query.routes_per_row = fixture.n_used;
            query.source_row_capacity = fixture.n_rows; query.scatter_capacity = ids.size(); query.n_lanes = 1; query.n_threads = 2;
            ggml_backend_moe_cpu_service_config_v1 config = {};
            config.struct_size = sizeof(config); config.abi_version = 1; config.source_owner = &source_owner;
            config.n_threads = 2; config.n_lanes = 1; config.max_regions = 1;
            config.flags = GGML_BACKEND_MOE_CPU_SERVICE_FLAG_V1_ALLOW_UNKNOWN_THREAD_STACK_BYTES |
                GGML_BACKEND_MOE_CPU_SERVICE_FLAG_V1_ALLOW_UNPROVEN_RUNTIME_ALLOCATIONS;
            ggml_backend_moe_cpu_service_v1_t service = nullptr;
            ggml_backend_moe_cpu_prepared_region_v1_t region = 0;
            ggml_backend_moe_cpu_prepared_requirements_v1 requirements = {};
            requirements.struct_size = sizeof(requirements); requirements.abi_version = 1;
            CHECK(api->create(&config, &service) == 0 && api->prepare(service, &query, &requirements, &region) == 0);
            const auto cpu_reference = evaluate_body({projection}, {input, routes}, {projection}, {bank}, {activation.data(), ids.data()});
            const auto gpu_reference = active_grouped_tensor_values(reference.up_output);
            std::vector<uint32_t> source_rows(ids.size()), destinations(ids.size());
            for (uint32_t route = 0; route < ids.size(); ++route) { source_rows[route] = route / fixture.n_used; destinations[route] = route; }
            ggml_context_ptr private_ctx(ggml_init({8 * ggml_tensor_overhead(), nullptr, true}));
            CHECK(private_ctx);
            ggml_tensor private_weight = *original->src[0];
            auto * private_input = ggml_dup_tensor(private_ctx.get(), original->src[1]);
            auto * private_ids = ggml_dup_tensor(private_ctx.get(), original->src[2]);
            memcpy(private_input->nb, original->src[1]->nb, sizeof(private_input->nb));
            memcpy(private_ids->nb, original->src[2]->nb, sizeof(private_ids->nb));
            auto * private_up = ggml_mul_mat_id(private_ctx.get(), &private_weight, private_input, private_ids);
            memcpy(private_up->op_params, original->op_params, sizeof(private_up->op_params));
            ggml_backend_buffer_ptr private_buffer(ggml_backend_alloc_ctx_tensors(private_ctx.get(), backend.get()));
            CHECK(private_buffer);
            ggml_backend_tensor_set(private_input, activation.data(), 0, ggml_nbytes(private_input));
            ggml_backend_tensor_set(private_ids, ids_storage.data(), 0, ids_storage.size());
            for (uint32_t case_index = 0; case_index < 8; ++case_index) {
                const bool source_frame = case_index >= 4;
                const uint32_t pattern = case_index % 4;
                auto * projection_output = source_frame ? private_up : original;
                candidate_stamp_execution(fixture.graph, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
                    GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL, fixture.n_rows, 1, 0,
                    source_frame ? GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED : GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_NONE);
                auto coverage = candidate_certify_graph(*owner, fixture.graph);
                std::shared_ptr<ggml_cuda_moe_graph_plan> plan;
                ggml_cuda_moe_graph_execution execution;
                CHECK(owner->prepare_graph_execution(fixture.graph, fixture.graph->uid, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED,
                    &plan, &execution, coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint, source_frame) != GGML_CUDA_MOE_GRAPH_PREPARE_UNAVAILABLE);
                CHECK(execution.resolve_streams(candidate_test_graph_stream, stream) && owner->begin_graph_dispatch(&execution, GGML_CUDA_MOE_GRAPH_DISPATCH_DIRECT));
                ggml_cuda_moe_graph_binding first, up, gate, down;
                auto * group = execution.find_group(fixture.readers.front(), &first);
                CHECK(group && execution.find_group(fixture.up_output, &up) == group && execution.find_group(fixture.gate_output, &gate) == group && execution.find_group(fixture.down_output, &down) == group);
                std::vector<int32_t> source_slots(fixture.n_experts), source_owners(fixture.n_experts);
                ggml_cuda_moe_prefill_source_binding source_binding;
                if (source_frame) {
                    CHECK(owner->prepare_source_group(group, stream, source_slots.data(), source_slots.size(),
                        source_owners.data(), source_owners.size(), true, &source_binding.residency_token));
                    ggml_cuda_moe_hybrid_reference_view view;
                    CHECK(owner->get_hybrid_reference_view(*group, &view));
                    source_binding.original = original; source_binding.resource_identity = view.resource_identity;
                    source_binding.slot_for_expert = source_slots.data(); source_binding.expert_for_slot = source_owners.data();
                    source_binding.expert_capacity = source_slots.size(); source_binding.slot_capacity = source_owners.size();
                } else {
                    CHECK(owner->prepare_prefill_group(group, first, fixture.readers.front(), stream, experts.data(), experts.size()) == GGML_CUDA_MOE_GROUPED_DECODE_READY);
                }
                const auto transaction = group->transaction.acquisition;
                ggml_cuda_moe_grouped_resource_info resource;
                CHECK(owner->get_group_resources(transaction, &resource) && resource.transaction_active);
                std::vector<uint8_t> mask(fixture.n_experts), ownership(ids.size());
                uint32_t cpu_count = 0;
                for (uint32_t expert = 0; expert < mask.size(); ++expert) { mask[expert] = pattern == 1 || (pattern == 2 && expert % 2 == 0) || (pattern == 3 && expert + 1 == mask.size()); }
                for (uint32_t route = 0; route < ids.size(); ++route) { ownership[route] = mask[ids[route]]; cpu_count += ownership[route]; }
                std::vector<float> cpu_output(gpu_reference.size(), std::numeric_limits<float>::quiet_NaN()), expected(gpu_reference.size());
                ggml_backend_moe_cpu_region_binding_v1 binding{sizeof(binding), fixture.n_rows, uint32_t(ids.size()), ids.data(), source_rows.data(), destinations.data()};
                ggml_backend_moe_cpu_dynamic_input_v1 inputs[] = {{activation.data(), ggml_nbytes(input), input->nb[2]}, {}};
                ggml_backend_moe_cpu_output_v1 output{cpu_output.data(), ggml_nbytes(projection), projection->nb[1]};
                ggml_backend_moe_cpu_execute_v1 job = {};
                job.struct_size = sizeof(job); job.epoch = case_index + 1; job.graph_uid = query.graph_uid;
                job.graph_generation = query.graph_generation; job.source_generation = query.source_generation;
                job.binding = &binding; job.dynamic_inputs = inputs; job.n_dynamic_inputs = 2; job.outputs = &output; job.n_outputs = 1;
                struct cpu_task {
                    ggml_backend_moe_cpu_controlled_execute_v1_t execute;
                    ggml_backend_moe_cpu_service_v1_t service;
                    ggml_backend_moe_cpu_prepared_region_v1_t region;
                    const ggml_backend_moe_cpu_execute_v1 * job;
                    const std::vector<uint8_t> * ownership;
                    uint32_t expected, calls = 0;
                    bool stop = false, fail = false;
                } task{execute, service, region, &job, &ownership, cpu_count};
                ggml_cuda_moe_prefill_cpu_partition partition;
                partition.experts = mask.data(); partition.n_experts = mask.size(); partition.context = &task;
                partition.output = cpu_output.data(); partition.output_bytes = output.bytes; partition.output_route_stride = output.route_stride;
                partition.canceled = +[](void * data) { return static_cast<cpu_task *>(data)->stop; };
                partition.execute = +[](void * data) {
                    auto & task = *static_cast<cpu_task *>(data); ++task.calls;
                    if (task.fail) { return false; }
                    ggml_backend_moe_cpu_execute_control_v1 control = {};
                    control.struct_size = sizeof(control); control.abi_version = 1;
                    control.abort = +[](void * data) { return static_cast<cpu_task *>(data)->stop; }; control.abort_data = &task;
                    ggml_backend_moe_cpu_execute_result_v1 result = {}; result.struct_size = sizeof(result);
                    return task.execute(task.service, task.region, task.job, task.ownership->data(), task.ownership->size(), &control, &result) == 0 && result.published_routes == task.expected;
                };
                const auto run = [&](const ggml_cuda_moe_graph_binding & up_binding, const ggml_cuda_moe_prefill_cpu_partition & part, const int64_t * rows) {
                    return owner->execute_bounded_prefill(cuda, group, up_binding, projection_output, stream, experts.data(), experts.size(),
                        reinterpret_cast<const char *>(ids.data()), ids.size() * sizeof(int32_t), routes->nb[1], rows, nullptr, nullptr, nullptr, &part,
                        source_frame ? &source_binding : nullptr);
                };
                std::vector<float> sentinel(gpu_reference.size(), -123.0f);
                ggml_backend_tensor_set(projection_output, sentinel.data(), 0, ggml_nbytes(projection_output));
                if (source_frame) {
                    ++source_binding.residency_token;
                    CHECK(run(up, partition, counts.data()) == GGML_CUDA_MOE_GROUPED_DECODE_FALLBACK);
                    --source_binding.residency_token; ++source_binding.resource_identity;
                    CHECK(run(up, partition, counts.data()) == GGML_CUDA_MOE_GROUPED_DECODE_FALLBACK);
                    --source_binding.resource_identity;
                    source_binding.original = fixture.gate_output;
                    CHECK(run(up, partition, counts.data()) == GGML_CUDA_MOE_GROUPED_DECODE_FALLBACK);
                    source_binding.original = original;
                    auto bad_slots = source_slots; bad_slots.front() = group->n_slots;
                    source_binding.slot_for_expert = bad_slots.data();
                    CHECK(run(up, partition, counts.data()) == GGML_CUDA_MOE_GROUPED_DECODE_FALLBACK);
                    source_binding.slot_for_expert = source_slots.data();
                    ggml_tensor bad_ids = *private_ids; bad_ids.nb[1] += sizeof(int32_t);
                    private_up->src[2] = &bad_ids;
                    CHECK(run(up, partition, counts.data()) == GGML_CUDA_MOE_GROUPED_DECODE_FALLBACK);
                    private_up->src[2] = private_ids;
                    void * output_data = private_up->data;
                    private_up->data = static_cast<char *>(ggml_backend_buffer_get_base(private_buffer.get())) + ggml_backend_buffer_get_size(private_buffer.get());
                    CHECK(run(up, partition, counts.data()) == GGML_CUDA_MOE_GROUPED_DECODE_FALLBACK);
                    private_up->data = output_data;
                    CHECK(task.calls == 0 && active_grouped_tensor_values(projection_output) == sentinel);
                }
                auto invalid = partition; --invalid.n_experts;
                CHECK(run(up, invalid, counts.data()) == GGML_CUDA_MOE_GROUPED_DECODE_FALLBACK);
                auto bad_binding = up; ++bad_binding.key.candidate.generation;
                CHECK(run(bad_binding, partition, counts.data()) == GGML_CUDA_MOE_GROUPED_DECODE_FALLBACK);
                auto bad_counts = counts; ++bad_counts.back();
                CHECK(run(up, partition, bad_counts.data()) == GGML_CUDA_MOE_GROUPED_DECODE_FALLBACK);
                if (cpu_count) { invalid = partition; invalid.output_bytes = 0; CHECK(run(up, invalid, counts.data()) == GGML_CUDA_MOE_GROUPED_DECODE_FALLBACK); }
                task.stop = true;
                CHECK(run(up, partition, counts.data()) == GGML_CUDA_MOE_GROUPED_DECODE_ERROR);
                task.stop = false;
                CHECK(task.calls == 0 && active_grouped_tensor_values(projection_output) == sentinel);
                if (cpu_count) {
                    task.fail = true;
                    CHECK(run(up, partition, counts.data()) == GGML_CUDA_MOE_GROUPED_DECODE_ERROR);
                    CHECK(task.calls == 1 && std::all_of(cpu_output.begin(), cpu_output.end(), [](float value) { return std::isnan(value); }));
                    CHECK(owner->get_group_resources(transaction, &resource) && resource.transaction_active);
                    if (pattern == 1) { CHECK(active_grouped_tensor_values(projection_output) == sentinel); }
                    task.fail = false; task.calls = 0;
                    ggml_backend_tensor_set(projection_output, sentinel.data(), 0, ggml_nbytes(projection_output));
                }
                CHECK(run(up, partition, counts.data()) == GGML_CUDA_MOE_GROUPED_DECODE_READY);
                ggml_backend_synchronize(backend.get());
                CHECK(task.calls == uint32_t(cpu_count != 0));
                for (uint32_t route = 0; route < ids.size(); ++route) {
                    for (int64_t feature = 0; feature < original->ne[0]; ++feature) {
                        const size_t index = route * original->ne[0] + feature;
                        expected[index] = ownership[route] ? cpu_reference[index] : gpu_reference[index];
                        if (!ownership[route]) { CHECK(std::isnan(cpu_output[index])); }
                    }
                }
                const auto actual = active_grouped_tensor_values(projection_output);
                double error = 0, norm = 0;
                for (size_t index = 0; index < actual.size(); ++index) { CHECK(std::isfinite(actual[index])); const double delta = actual[index] - expected[index]; error += delta * delta; norm += double(expected[index]) * expected[index]; }
                CHECK(norm > 0 && error / norm < 2e-5);
                CHECK(owner->get_group_resources(transaction, &resource) && resource.transaction_active);
                ggml_tensor gate_node = *fixture.gate_output, gate_weight = *fixture.gate_output->src[0];
                gate_node.src[0] = &gate_weight; gate_node.src[1] = private_input; gate_node.src[2] = private_ids;
                if (source_frame) {
                    CHECK(cudaMemcpyAsync(original->data, private_up->data, ggml_nbytes(original), cudaMemcpyDeviceToDevice, stream) == cudaSuccess);
                    source_binding.original = fixture.gate_output;
                }
                CHECK(owner->execute_bounded_prefill(cuda, group, gate, source_frame ? &gate_node : fixture.gate_output, stream, experts.data(), experts.size(),
                    reinterpret_cast<const char *>(ids.data()), ids.size() * sizeof(int32_t), routes->nb[1], counts.data(), nullptr, nullptr, nullptr, nullptr,
                    source_frame ? &source_binding : nullptr) == GGML_CUDA_MOE_GROUPED_DECODE_READY);
                CHECK(ggml_cuda_moe_router_compute(cuda, fixture.down_output->src[1]));
                ggml_tensor down_node = *fixture.down_output, down_weight = *fixture.down_output->src[0], down_input = *fixture.down_output->src[1];
                down_node.src[0] = &down_weight; down_node.src[1] = &down_input; down_node.src[2] = private_ids;
                source_binding.original = fixture.down_output;
                CHECK(owner->execute_bounded_prefill(cuda, group, down, source_frame ? &down_node : fixture.down_output, stream, experts.data(), experts.size(),
                    reinterpret_cast<const char *>(ids.data()), ids.size() * sizeof(int32_t), routes->nb[1], counts.data(), nullptr, nullptr, nullptr, nullptr,
                    source_frame ? &source_binding : nullptr) == GGML_CUDA_MOE_GROUPED_DECODE_READY);
                if (source_frame) {
                    ggml_backend_synchronize(backend.get());
                    CHECK(owner->finish_source_dispatch(&execution));
                } else { CHECK(owner->finish_prefill_group(group, down, fixture.down_output, stream) && owner->finish_graph_dispatch(&execution)); }
                ggml_backend_synchronize(backend.get());
                CHECK(owner->get_group_resources(transaction, &resource) && !resource.transaction_active);
                ggml_backend_tensor_set(reference.up_output, expected.data(), 0, ggml_nbytes(reference.up_output));
                const auto begin = std::find(reference.graph->nodes, reference.graph->nodes + reference.graph->n_nodes, reference.down_output->src[1]);
                const auto end = std::find(reference.graph->nodes, reference.graph->nodes + reference.graph->n_nodes, reference.down_output);
                CHECK(begin != reference.graph->nodes + reference.graph->n_nodes && end >= begin && end != reference.graph->nodes + reference.graph->n_nodes);
                auto remainder = ggml_graph_view(reference.graph, begin - reference.graph->nodes, end - reference.graph->nodes + 1);
                CHECK(ggml_backend_graph_compute(reference_backend.get(), &remainder) == GGML_STATUS_SUCCESS);
                const auto final_expected = active_grouped_tensor_values(reference.down_output), final_actual = active_grouped_tensor_values(fixture.down_output);
                double final_error = 0, final_norm = 0;
                for (size_t index = 0; index < final_actual.size(); ++index) { CHECK(std::isfinite(final_actual[index])); const double delta = final_actual[index] - final_expected[index]; final_error += delta * delta; final_norm += double(final_expected[index]) * final_expected[index]; }
                CHECK(final_norm > 0 && final_error / final_norm < 2e-5);
                fprintf(stderr, "test-moe-cache: canonical CPU prefill type=%s pageable=%u source_frame=%u sparse=%u pattern=%u cpu_routes=%u relative_mse=%.9g down_relative_mse=%.9g retained through down OK\n",
                    ggml_type_name(type), pageable, source_frame, sparse, pattern, cpu_count, error / norm, final_error / final_norm);
                ++cases;
            }
            CHECK(api->destroy_region(service, &region) == 0 && api->close(service) == 0 && api->destroy(&service) == 0);
        }
    }
    ggml_backend_cuda_moe_set_debug_mm(old_debug);
    fprintf(stderr, "test-moe-cache: canonical CPU prefill cases=%u actual CPU service and canonical staging OK\n", cases);
}

void test_fidelity_fixture(int device, bool benchmark) {
    ggml_backend_ptr gpu(ggml_backend_cuda_init(device));
    auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(gpu.get()));
    auto get_api = reinterpret_cast<ggml_cuda_moe_fidelity_fixture_get_v1_t>(
        ggml_backend_reg_get_proc_address(reg, GGML_CUDA_MOE_FIDELITY_FIXTURE_V1_PROC_NAME));
    CHECK(get_api != nullptr);
    const auto * api = get_api();
    CHECK(api && api->struct_size == sizeof(*api) && api->abi_version == 1);
    const char * names[] = {"disabled", "callback", "poll", "segmented", "memop"};
    uint32_t passed = 0, unavailable = 0;
    for (uint32_t rows : {1u, 4u}) {
        ggml_context_ptr ctx(ggml_init({32 * ggml_tensor_overhead() + ggml_graph_overhead_custom(32, false), nullptr, true}));
        CHECK(ctx != nullptr);
        auto * input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 17, 1, rows);
        auto * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 3, rows);
        auto * bank = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 17, 13, 7);
        auto * output = ggml_mul_mat_id(ctx.get(), bank, input, ids);
        auto * graph = ggml_new_graph_custom(ctx.get(), 32, false);
        ggml_build_forward_expand(graph, output);
        CHECK(ggml_backend_moe_graph_assign_uid_v1(graph));
        for (uint32_t boundaries : {1u, 48u}) {
            ggml_cuda_moe_fidelity_query_v1 query = {sizeof(query), boundaries, graph, output};
            ggml_cuda_moe_fidelity_certificate_v1 certificate = {};
            ggml_cuda_moe_fidelity_storage_v1 storage = {};
            CHECK(api->measure(&query, &certificate, &storage));
            CHECK(certificate.row_capacity == rows && certificate.routes_per_row == 3 && certificate.expert_count == 7 &&
                certificate.input_width == 17 && certificate.output_width == 13 && certificate.boundaries == boundaries);
            auto invalid = query;
            invalid.output = input;
            CHECK(!api->measure(&invalid, &certificate, &storage));
            invalid = query;
            invalid.boundaries = UINT32_MAX;
            CHECK(!api->measure(&invalid, &certificate, &storage));
            CHECK(api->measure(&query, &certificate, &storage));
            ggml_cuda_moe_fidelity_options_v1 options = {};
            options.struct_size = sizeof(options);
            options.arm = GGML_CUDA_MOE_FIDELITY_CALLBACK;
            options.replays = benchmark ? 64 : 4;
            options.warmup = benchmark ? 8 : 1;
            options.limits = storage;
            uint64_t costs[5] = {};
            std::vector<ggml_cuda_moe_fidelity_sample_v1> samples(options.replays);
            ggml_cuda_moe_fidelity_report_v1 report = {};
            report.struct_size = sizeof(report);
            for (uint32_t budget = 0; budget < 3; ++budget) {
                options.limits = storage;
                auto * limit = budget == 0 ? &options.limits.pinned_bytes :
                    budget == 1 ? &options.limits.device_bytes : &options.limits.scratch_bytes;
                CHECK(*limit != 0);
                --*limit;
                CHECK(api->run(gpu.get(), &query, &options, samples.data(), &report) == -2);
            }
            options.limits = storage;
            for (uint32_t repeat = 0; repeat < (benchmark ? 2u : 1u); ++repeat) {
                for (uint32_t work = 0; work < 3; ++work) {
                    options.cpu_delay_us = work == 1 ? 300 : work == 2 ? 50 : 0;
                    options.gpu_delay_us = work == 2 ? 300 : work == 1 ? 50 : 0;
                    options.validate_failures = work == 0;
                    for (uint32_t arm = 1; arm <= 4; ++arm) {
                        options.arm = repeat == 0 ? arm : 5 - arm;
                        const int32_t result = api->run(gpu.get(), &query, &options, samples.data(), &report);
                        if (result == 0) {
                            ++unavailable;
                            fprintf(stderr, "moe-fidelity: arm=%s unavailable=%s\n", names[options.arm], report.reason);
                            continue;
                        }
                        if (result != 1) {
                            fprintf(stderr, "moe-fidelity: arm=%s rows=%u boundaries=%u work=%u failed=%s status=%d\n",
                                names[options.arm], rows, boundaries, work, report.reason, result);
                        }
                        CHECK(result == 1 && report.samples == options.replays);
                        CHECK(report.storage.pinned_bytes == storage.pinned_bytes && report.storage.device_bytes == storage.device_bytes &&
                            report.storage.scratch_bytes == storage.scratch_bytes);
                        const uint64_t graphs = options.arm == GGML_CUDA_MOE_FIDELITY_SEGMENTED ? boundaries + 1 : 1;
                        CHECK(report.captures == graphs && report.graph_launches == report.epochs * graphs);
                        CHECK(report.zero_cpu != 0 && report.accepted >= options.replays && report.finite_drains > report.epochs);
                        if (options.validate_failures) { CHECK(report.rejected == 7); }
                        std::vector<uint64_t> times;
                        uint64_t whole = 0, launch = 0, observed = 0, continuation = 0, cpu = 0, wait = 0, work_ns = 0, commit = 0;
                        uint64_t ready_to_cpu = 0, completion_publish = 0;
                        for (const auto & sample : samples) {
                            times.push_back(sample.whole_ns);
                            whole += sample.whole_ns; launch += sample.launch_ns; observed += sample.observation_ns;
                            continuation += sample.continuation_ns; cpu += sample.cpu_service_ns; wait += sample.gpu_wait_ns;
                            work_ns += sample.gpu_work_ns; commit += sample.gpu_commit_ns;
                            ready_to_cpu += sample.ready_to_cpu_ns; completion_publish += sample.completion_publish_ns;
                        }
                        std::sort(times.begin(), times.end());
                        const auto percentile = [&](uint32_t p) { return times[(times.size() - 1) * p / 100] / 1000.0; };
                        const double scale = 1000.0 * samples.size();
                        costs[options.arm] += whole;
                        fprintf(stderr, "moe-fidelity: arm=%s rows=%u boundaries=%u work=%u repeat=%u replays=%u whole_us=%.3f launch_us=%.3f observation_us=%.3f continuation_us=%.3f cpu_us=%.3f gpu_wait_us=%.3f gpu_work_us=%.3f post_wait_us=%.3f ready_to_cpu_us=%.3f completion_publish_us=%.3f p50_us=%.3f p95_us=%.3f p99_us=%.3f captures=%llu launches=%llu callbacks=%llu waits=%llu epochs=%llu services=%llu zero_cpu=%llu accepted=%llu rejected=%llu resident=%llu transfer=%llu cpu_routes=%llu coordinator_cpu_ms=%.3f supervisor_cpu_ms=%.3f caller_cpu_ms=%.3f pinned=%llu device=%llu scratch=%llu OK\n",
                            names[options.arm], rows, boundaries, work, repeat, options.replays, whole / scale, launch / scale,
                            observed / scale, continuation / scale, cpu / scale, wait / scale, work_ns / scale, commit / scale,
                            ready_to_cpu / scale, completion_publish / scale, percentile(50), percentile(95), percentile(99), (unsigned long long) report.captures,
                            (unsigned long long) report.graph_launches, (unsigned long long) report.callbacks,
                            (unsigned long long) report.wait_nodes, (unsigned long long) report.epochs,
                            (unsigned long long) report.services, (unsigned long long) report.zero_cpu,
                            (unsigned long long) report.accepted, (unsigned long long) report.rejected,
                            (unsigned long long) report.resident_routes, (unsigned long long) report.transfer_routes,
                            (unsigned long long) report.cpu_routes, report.coordinator_cpu_ns / 1e6, report.supervisor_cpu_ns / 1e6, report.caller_cpu_ns / 1e6,
                            (unsigned long long) storage.pinned_bytes, (unsigned long long) storage.device_bytes,
                            (unsigned long long) storage.scratch_bytes);
                        ++passed;
                    }
                }
            }
            options.cpu_delay_us = options.gpu_delay_us = 0;
            options.validate_failures = 1;
            options.ready_memop = 1;
            uint32_t winner = GGML_CUDA_MOE_FIDELITY_CALLBACK;
            for (uint32_t arm = 1; arm <= 4; ++arm) {
                if (costs[arm] && (!costs[winner] || costs[arm] < costs[winner])) { winner = arm; }
            }
            for (uint32_t arm : {winner, uint32_t(GGML_CUDA_MOE_FIDELITY_MEMOP)}) {
                options.arm = arm;
                const auto result = api->run(gpu.get(), &query, &options, samples.data(), &report);
                CHECK(result == 0 || result == 1);
                uint64_t whole = 0, launch = 0, observation = 0;
                if (result == 1) {
                    for (const auto & sample : samples) { whole += sample.whole_ns; launch += sample.launch_ns; observation += sample.observation_ns; }
                }
                const double scale = 1000.0 * samples.size();
                fprintf(stderr, "moe-fidelity: arm=%s rows=%u boundaries=%u ready_memop=1 status=%d whole_us=%.3f launch_us=%.3f observation_us=%.3f reason=%s\n",
                    names[arm], rows, boundaries, result, whole / scale, launch / scale, observation / scale, report.reason);
                if (winner == GGML_CUDA_MOE_FIDELITY_MEMOP) { break; }
            }
        }
    }
    CHECK(passed != 0 || unavailable != 0);
    fprintf(stderr, "test-moe-cache: fidelity fixture passed=%u unavailable=%u mode=%s OK\n", passed, unavailable, benchmark ? "bench" : "check");
}

}  // namespace

void test_hybrid_metadata() {
    {
        llm_graph_result result{16};
        auto * first = ggml_new_tensor_1d(result.get_ctx(), GGML_TYPE_I32, 1);
        auto * registered = ggml_new_tensor_1d(result.get_ctx(), GGML_TYPE_I32, 1);
        ggml_set_input(first);
        CHECK((result.get_inp_tensors() == std::vector<ggml_tensor *>{first}));
        auto * appended = ggml_new_tensor_1d(result.get_ctx(), GGML_TYPE_F32, 1);
        ggml_set_input(appended);
        CHECK((result.get_inp_tensors() == std::vector<ggml_tensor *>{first, appended}));
        auto input = std::make_unique<llm_graph_input_embd>(1);
        input->tokens = registered;
        ggml_set_input(registered);
        result.add_input(std::move(input));
        CHECK((result.get_inp_tensors() == std::vector<ggml_tensor *>{first, registered, appended}));
        result.reset();
        CHECK(result.get_inp_tensors().empty() && result.get_inp_token_tensors().empty());
    }
    if (const char * mode = getenv("GGML_TEST_MOE_FIDELITY")) {
        if (strcmp(mode, "sigmoid-moe") == 0) {
            test_sigmoid_moe_routing();
            return;
        }
        if (strcmp(mode, "route-producers") == 0) {
            test_region_route_producers();
            return;
        }
        if (strcmp(mode, "projection-discovery") == 0) {
            test_projection_discovery();
            return;
        }
        if (strcmp(mode, "body-program") == 0) {
            test_source_body_program();
            return;
        }
        if (strcmp(mode, "allocation") == 0) {
            test_fidelity_allocation_graph();
            return;
        }
    }
    int device = 0;
    CUDA_OK(cudaGetDevice(&device));
    if (const char * mode = getenv("GGML_TEST_MOE_FIDELITY")) {
        if (!strcmp(mode, "cpu-prefill-partition")) { test_cpu_prefill_partition(device); return; }
        if (!strcmp(mode, "cpu-prefill-owner")) { test_cpu_prefill_owner(device); return; }
        if (strcmp(mode, "finalized-routes") == 0) {
            for (uint32_t rows : {2u, 4u}) {
                for (bool pageable : {false, true}) { test_hybrid_row_metadata(device, rows, pageable); }
            }
            return;
        }
        if (!strcmp(mode, "source-backend-sessions")) { test_source_backend_sessions(device); return; }
        if (strcmp(mode, "source-core-images") == 0) {
            CHECK(ggml_moe_fidelity_selection().valid && ggml_moe_fidelity_selection().source_pool && ggml_moe_fidelity_selection().reference);
            const hybrid_layer_signature signature = {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE,
                GGML_TYPE_IQ3_XXS, GGML_TYPE_Q2_0, LLM_FFN_SILU, 2560, n_dim};
            for (uint32_t rows : {1u, 9u}) {
                test_fidelity_real_window(device, rows, GGML_CUDA_MOE_FIDELITY_POLL, false, true, signature,
                    10, 0, 16, false, 48);
            }
            return;
        }
        if (strcmp(mode, "sort-resources") == 0) { test_source_sort_resources(device); return; }
        if (strcmp(mode, "softmax-resources") == 0) { test_source_softmax_resources(device); return; }
        if (strcmp(mode, "routed-scheduler") == 0 || strcmp(mode, "routed-discovery") == 0 || strcmp(mode, "routed-new-op") == 0 || strcmp(mode, "routed-new-op-fallback") == 0) {
            CHECK(ggml_moe_fidelity_selection().valid && ggml_moe_fidelity_selection().source_pool && ggml_moe_fidelity_selection().reference);
            for (const auto types : {std::pair<ggml_type, ggml_type>{GGML_TYPE_IQ2_XS, GGML_TYPE_IQ4_NL},
                    std::pair<ggml_type, ggml_type>{GGML_TYPE_Q5_K, GGML_TYPE_Q5_K}}) {
                const hybrid_layer_signature signature = {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE,
                    types.first, types.second, LLM_FFN_SILU, 256, 256};
                for (uint32_t rows : {2u, 4u}) {
                    for (const auto arm : {std::pair<uint32_t, bool>{GGML_CUDA_MOE_FIDELITY_POLL, false},
                            std::pair<uint32_t, bool>{GGML_CUDA_MOE_FIDELITY_SEGMENTED, false},
                            std::pair<uint32_t, bool>{GGML_CUDA_MOE_FIDELITY_POLL, true}}) {
                        test_fidelity_real_window(device, rows, arm.first, arm.second, true, signature, 10, 0, 16);
                    }
                }
            }
            return;
        }
        if (!strcmp(mode, "routed-profile") || !strcmp(mode, "routed-adapt") || !strcmp(mode, "routed-async") ||
                !strcmp(mode, "routed-statistics") || !strcmp(mode, "routed-statistics-adapt") || !strcmp(mode, "routed-statistics-async")) {
            CHECK(ggml_moe_fidelity_selection().valid && ggml_moe_fidelity_selection().source_pool && ggml_moe_fidelity_selection().reference);
            const hybrid_layer_signature signature = {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE,
                GGML_TYPE_IQ2_XS, GGML_TYPE_IQ4_NL, LLM_FFN_SILU, 256, 256};
            for (uint32_t transport_case : {0u, 3u}) {
                test_fidelity_real_window(device, 5, GGML_CUDA_MOE_FIDELITY_POLL, false, true, signature, 10, 0, 16,
                    false, 0, false, false, false, false, false, false, false, transport_case, false, true,
                    (!strcmp(mode, "routed-async") || !strcmp(mode, "routed-statistics-async")) ? 2 :
                        (!strcmp(mode, "routed-adapt") || !strcmp(mode, "routed-statistics-adapt")), strstr(mode, "statistics") != nullptr);
            }
            return;
        }
        if (strcmp(mode, "source-core-profile") == 0 || strcmp(mode, "source-core-adapt") == 0 || strcmp(mode, "source-core-async") == 0) {
            CHECK(ggml_moe_fidelity_selection().valid && ggml_moe_fidelity_selection().source_pool && ggml_moe_fidelity_selection().reference);
            for (const auto layout : {uint32_t(GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE), uint32_t(GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP)}) {
                const hybrid_layer_signature signature = {layout,
                    layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE ? GGML_TYPE_IQ2_XS : GGML_TYPE_IQ3_XXS,
                    layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE ? GGML_TYPE_IQ4_NL : GGML_TYPE_Q2_0,
                    LLM_FFN_SILU, 256, 256};
                for (uint32_t transport_case : {0u, 3u}) {
                    test_fidelity_real_window(device, 5, GGML_CUDA_MOE_FIDELITY_POLL, false, true, signature, 10, 0, 16,
                        false, 0, false, false, false, false, false, false, false, transport_case, false, true, !strcmp(mode, "source-core-async") ? 2 : !strcmp(mode, "source-core-adapt"));
                }
            }
            return;
        }
        if (strcmp(mode, "source-core-copy-cancel") == 0) {
            CHECK(ggml_moe_fidelity_selection().valid && ggml_moe_fidelity_selection().source_pool && ggml_moe_fidelity_selection().reference);
            const hybrid_layer_signature signature = {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, GGML_TYPE_IQ3_XXS, GGML_TYPE_Q2_0, LLM_FFN_SILU, 256, 256};
            test_fidelity_real_window(device, 5, GGML_CUDA_MOE_FIDELITY_POLL, false, true, signature, 10, 0, 16,
                false, 0, false, false, false, true);
            return;
        }
        if (strcmp(mode, "source-core-transport") == 0) {
            CHECK(ggml_moe_fidelity_selection().valid && ggml_moe_fidelity_selection().source_pool && ggml_moe_fidelity_selection().reference);
            const hybrid_layer_signature signature = {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ4_NL, LLM_FFN_SILU, 256, 256};
            for (uint32_t transport_case : {1u, 2u, 3u}) {
                test_fidelity_real_window(device, 5, GGML_CUDA_MOE_FIDELITY_POLL, false, true, signature, 10, 0, 16,
                    false, 0, false, false, false, false, false, false, false, transport_case);
            }
            return;
        }
        if (!strcmp(mode, "source-core-prefill")) {
            CHECK(ggml_moe_fidelity_selection().valid && ggml_moe_fidelity_selection().source_pool && ggml_moe_fidelity_selection().reference);
            for (const auto type : {GGML_TYPE_Q5_K, GGML_TYPE_IQ4_NL, GGML_TYPE_Q4_K, GGML_TYPE_Q8_0, GGML_TYPE_F32}) {
                const hybrid_layer_signature signature{GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, type, type, LLM_FFN_SILU, 256, 256};
                for (uint32_t rows : {2u, 9u, 65u, 2048u}) {
                    test_fidelity_real_window(device, rows, GGML_CUDA_MOE_FIDELITY_POLL, false, true, signature, 4, 1, 16,
                        false, 0, false, false, true);
                }
            }
            return;
        }
        if (!strcmp(mode, "source-core-generic")) {
            CHECK(ggml_moe_fidelity_selection().valid && ggml_moe_fidelity_selection().source_pool && ggml_moe_fidelity_selection().reference);
            for (const auto & signature : {
                hybrid_layer_signature{GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_TYPE_Q3_K, GGML_TYPE_Q3_K, LLM_FFN_SILU, 256, 256},
                hybrid_layer_signature{GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, LLM_FFN_SILU, 256, 256},
                hybrid_layer_signature{GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, LLM_FFN_SILU, 256, 256},
                hybrid_layer_signature{GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, LLM_FFN_GELU, 256, 256},
                hybrid_layer_signature{GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, LLM_FFN_SILU, 256, 256},
                hybrid_layer_signature{GGML_BACKEND_MOE_CANDIDATE_LAYOUT_UNGATED, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0, LLM_FFN_RELU_SQR, 256, 256},
                hybrid_layer_signature{GGML_BACKEND_MOE_CANDIDATE_LAYOUT_UNGATED, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0, LLM_FFN_RELU_SQR, 256, 256}}) {
                for (uint32_t rows : {1u, 5u}) {
                    test_fidelity_real_window(device, rows, GGML_CUDA_MOE_FIDELITY_POLL, false, true, signature, 10, 0, 16,
                        false, 0, false, false, true);
                }
            }
            return;
        }
        if (strcmp(mode, "source-core") == 0 || strcmp(mode, "source-core-matmul") == 0 || strcmp(mode, "source-core-empty-view") == 0 || strcmp(mode, "source-core-resources") == 0 || strcmp(mode, "source-core-libraries") == 0 ||
                strcmp(mode, "source-core-split-preflight") == 0 || strcmp(mode, "source-core-staged") == 0 || strcmp(mode, "source-core-cancel") == 0 ||
                strcmp(mode, "source-core-fallback") == 0 || strcmp(mode, "source-core-overlap") == 0 || strcmp(mode, "source-core-overlap-cancel") == 0 ||
                strcmp(mode, "source-core-deadline") == 0) {
            CHECK(ggml_moe_fidelity_selection().valid && ggml_moe_fidelity_selection().source_pool && ggml_moe_fidelity_selection().reference);
            const bool staged = strcmp(mode, "source-core-staged") == 0;
            const bool deadline = strcmp(mode, "source-core-deadline") == 0;
            const bool cancel = strcmp(mode, "source-core-cancel") == 0 || strcmp(mode, "source-core-overlap-cancel") == 0 || deadline;
            const bool overlap = strcmp(mode, "source-core-overlap") == 0 || strcmp(mode, "source-core-overlap-cancel") == 0;
            const bool fallback = strcmp(mode, "source-core-fallback") == 0;
            for (const auto layout : {uint32_t(GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE), uint32_t(GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP)}) {
                const hybrid_layer_signature signature = {layout,
                    layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE ? GGML_TYPE_IQ2_XS : GGML_TYPE_IQ3_XXS,
                    layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE ? GGML_TYPE_IQ4_NL : GGML_TYPE_Q2_0,
                    LLM_FFN_SILU, 256, 256};
                if (fallback) {
                    test_fidelity_real_window(device, 5, GGML_CUDA_MOE_FIDELITY_POLL, false, true, signature, 10, 0, 16,
                        false, 0, false, false, false, false, false, false, true);
                    continue;
                }
                for (uint32_t rows : {1u, 5u}) {
                    if (cancel && rows != 5) { continue; }
                    for (uint32_t arm : {uint32_t(GGML_CUDA_MOE_FIDELITY_POLL), uint32_t(GGML_CUDA_MOE_FIDELITY_SEGMENTED)}) {
                        if (deadline && arm != GGML_CUDA_MOE_FIDELITY_POLL) { continue; }
                        test_fidelity_real_window(device, rows, arm, false, true, signature, 10, 0, 16,
                            !cancel && !staged, 0, false, false, false, cancel, staged, false, false, 0, overlap);
                    }
                }
                if (!deadline) {
                    test_fidelity_real_window(device, 5, GGML_CUDA_MOE_FIDELITY_SEGMENTED, true, true, signature, 10, 0, 16,
                        !cancel && !staged, 0, false, false, false, cancel, staged, false, false, 0, overlap);
                }
            }
            return;
        }
        if (strcmp(mode, "real-reference") == 0 || strcmp(mode, "real-reference-staged") == 0 || strcmp(mode, "real-reference-cancel") == 0 || strcmp(mode, "real-reference-transition") == 0) {
            CHECK(ggml_moe_fidelity_selection().valid && ggml_moe_fidelity_selection().reference);
            const bool staged = strcmp(mode, "real-reference-staged") == 0;
            const bool cancel = strcmp(mode, "real-reference-cancel") == 0;
            const bool transition = strcmp(mode, "real-reference-transition") == 0;
            for (const auto layout : {uint32_t(GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE), uint32_t(GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP)}) {
                const hybrid_layer_signature signature = {layout,
                    layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE ? GGML_TYPE_IQ2_XS : GGML_TYPE_IQ3_XXS,
                    layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE ? GGML_TYPE_IQ4_NL : GGML_TYPE_Q2_0,
                    LLM_FFN_SILU, 256, 256};
                for (uint32_t rows : {1u, 5u, 9u}) {
                    if (transition && rows != 1) { continue; }
                    if ((staged || cancel) && rows == 9) { continue; }
                    test_fidelity_real_window(device, rows, GGML_CUDA_MOE_FIDELITY_POLL, false, false, signature,
                        10, 0, 16, false, 0, false, false, false, cancel, staged, transition);
                }
            }
            return;
        }
        if (strcmp(mode, "real-r1-segmented") == 0) {
            test_fidelity_real_window(device, 1, GGML_CUDA_MOE_FIDELITY_SEGMENTED, false);
            return;
        }
        if (strcmp(mode, "real-r5") == 0) {
            test_fidelity_real_window(device, 5, GGML_CUDA_MOE_FIDELITY_SEGMENTED, false);
            return;
        }
        if (strcmp(mode, "real-cancel") == 0) {
            const hybrid_layer_signature signature = {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE,
                GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, LLM_FFN_SILU, n_dim, n_dim};
            for (uint32_t rows : {1u, 4u}) {
                test_fidelity_real_window(device, rows, GGML_CUDA_MOE_FIDELITY_POLL, false, false, signature,
                    n_used, 0, n_experts, false, 0, false, false, false, true);
            }
            test_fidelity_real_window(device, 4, GGML_CUDA_MOE_FIDELITY_SEGMENTED, true, false, signature,
                n_used, 0, n_experts, false, 0, false, false, false, true);
            return;
        }
        if (strcmp(mode, "real-staged-input") == 0) {
            const hybrid_layer_signature signature = {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE,
                GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, LLM_FFN_SILU, n_dim, n_dim};
            for (uint32_t rows : {1u, 4u}) {
                test_fidelity_real_window(device, rows, GGML_CUDA_MOE_FIDELITY_POLL, false, false, signature,
                    n_used, 0, n_experts, false, 0, false, false, false, false, true);
            }
            test_fidelity_real_window(device, 4, GGML_CUDA_MOE_FIDELITY_SEGMENTED, true, false, signature,
                n_used, 0, n_experts, false, 0, false, false, false, false, true);
            return;
        }
        if (strcmp(mode, "real-attention") == 0) {
            const hybrid_layer_signature signature = {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE,
                GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, LLM_FFN_SILU, n_dim, n_dim};
            test_fidelity_real_window(device, 5, GGML_CUDA_MOE_FIDELITY_SEGMENTED, false, false, signature,
                n_used, n_experts, n_experts, false, 0, true, true);
            return;
        }
        if (strcmp(mode, "real-empty") == 0) {
            const hybrid_layer_signature signature = {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE,
                GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, LLM_FFN_SILU, n_dim, n_dim};
            test_fidelity_real_window(device, 1, GGML_CUDA_MOE_FIDELITY_SEGMENTED, false, false, signature, n_used, 0, n_experts, true);
            return;
        }
        if (strcmp(mode, "real-bf16") == 0 || strcmp(mode, "real-bf16-diagnostic") == 0 || strcmp(mode, "real-bf16-gpu") == 0 || strcmp(mode, "real-bf16-cpu-oracle") == 0 ||
                strcmp(mode, "real-q4-wide") == 0 || strcmp(mode, "real-q4-wide-gpu") == 0) {
            const hybrid_layer_signature signature = {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE,
                GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, LLM_FFN_SILU, 2560, n_dim};
            const bool cpu_oracle = strcmp(mode, "real-bf16-cpu-oracle") == 0;
            const bool bf16 = strcmp(mode, "real-bf16") == 0 || strcmp(mode, "real-bf16-diagnostic") == 0 || strcmp(mode, "real-bf16-gpu") == 0 || cpu_oracle;
            const bool gpu = strcmp(mode, "real-bf16-gpu") == 0 || strcmp(mode, "real-q4-wide-gpu") == 0;
            test_fidelity_real_window(device, 5, GGML_CUDA_MOE_FIDELITY_SEGMENTED, false, false, signature,
                n_used, gpu ? n_experts : 1, n_experts, false, bf16 ? 48 : 0, strcmp(mode, "real-bf16") != 0, false, cpu_oracle);
            return;
        }
        if (strcmp(mode, "real-r1-mixed") == 0 || strcmp(mode, "real-r5-mixed") == 0 || strcmp(mode, "real-r5-iq3") == 0) {
            const bool iq3 = strcmp(mode, "real-r5-iq3") == 0;
            const hybrid_layer_signature signature = {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE,
                iq3 ? GGML_TYPE_IQ3_XXS : GGML_TYPE_IQ2_XS, iq3 ? GGML_TYPE_Q2_0 : GGML_TYPE_IQ4_NL, LLM_FFN_SILU, 2560, 640};
            test_fidelity_real_window(device, strcmp(mode, "real-r1-mixed") == 0 ? 1 : 5,
                GGML_CUDA_MOE_FIDELITY_SEGMENTED, false, false, signature, 10, 5, 16);
            return;
        }
        if (strcmp(mode, "real") == 0) {
            for (uint32_t rows : {1u, 2u, 4u}) {
                for (uint32_t arm : {uint32_t(GGML_CUDA_MOE_FIDELITY_SEGMENTED), uint32_t(GGML_CUDA_MOE_FIDELITY_POLL)}) {
                    test_fidelity_real_window(device, rows, arm, false);
                }
            }
            test_fidelity_real_window(device, 4, GGML_CUDA_MOE_FIDELITY_SEGMENTED, true);
            return;
        }
        if (strcmp(mode, "scheduler") == 0) {
            for (uint32_t rows : {2u, 4u, 5u}) {
                test_fidelity_real_window(device, rows, GGML_CUDA_MOE_FIDELITY_SEGMENTED, false, true);
            }
            return;
        }
        test_fidelity_fixture(device, strcmp(mode, "bench") == 0);
        return;
    }
    for (uint32_t rows : {2u, 4u}) {
        for (bool pageable : {false, true}) { test_hybrid_row_metadata(device, rows, pageable); }
    }
    test_hybrid_row_sources();
    for (uint32_t rows : {2u, 4u}) {
        for (bool pageable : {false, true}) { test_hybrid_rows_runtime(device, rows, pageable); }
    }
}

void test_grouped_layer_placement(bool profile_only) {
    if (profile_only) {
        const bool debug = ggml_backend_cuda_moe_get_debug_mm();
        ggml_backend_cuda_moe_set_debug_mm(true);
        for (const bool pageable : {false, true}) {
            for (const auto layout : {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_UNGATED}) {
                run_layers({0}, 0, false, layout, GGML_TYPE_Q4_0, pageable);
            }
        }
        ggml_backend_cuda_moe_set_debug_mm(debug);
        return;
    }
    const bool debug = ggml_backend_cuda_moe_get_debug_mm();
    ggml_backend_cuda_moe_set_debug_mm(true);
    int device = 0;
    CUDA_OK(cudaGetDevice(&device));
    test_hybrid_metadata();
    test_region_descriptor_variants();
    bool window_supported = false;
    bool staged_supported = false;
    {
        ggml_backend_ptr gpu(ggml_backend_cuda_init(device));
        auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(gpu.get()));
        auto probe = reinterpret_cast<ggml_backend_moe_hybrid_window_probe_v1_t>(
            ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_HYBRID_WINDOW_PROBE_V1_PROC_NAME));
        CHECK(probe != nullptr);
        const auto result = probe(gpu.get(), 1000);
        CHECK(result >= 0);
        window_supported = result == 1;
        fprintf(stderr, "test-moe-cache: conditional window probe replays=1000 status=%s\n",
            result == 1 ? "supported" : "unavailable (S5b eager)");
        auto get_stage_api = reinterpret_cast<ggml_staged_input_get_api_t>(
            ggml_backend_reg_get_proc_address(reg, GGML_STAGED_INPUT_PROC));
        if (const auto * api = get_stage_api == nullptr ? nullptr : get_stage_api()) {
            if (void * input = api->create(gpu.get(), n_dim * sizeof(float))) {
                staged_supported = true;
                api->destroy(input);
            }
        }
    }
    const hybrid_layer_signature merged = {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, LLM_FFN_GELU, n_dim, n_dim};
    if (window_supported) {
        hybrid_test_policy window;
        window.window = true;
        window.auto_memory = true;
        window.resident_batch = true;
        window.admission = true;
        window.cpu_pipeline = false;
        window.combine_gpu = false;
        test_hybrid_layers(device, false, false, merged, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, window);
        window.cpu_pipeline = true;
        window.combine_gpu = true;
        window.direct_gather = false;
        test_hybrid_layers(device, false, false, merged, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, window);
        window.direct_gather = true;
        window.window_replays = 32;
        test_hybrid_layers(device, false, false, merged, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, window);
        window.window_replays = 0;
        for (uint32_t phase : {GGML_BACKEND_MOE_HYBRID_TEST_ADMISSION_BANK_COPIED, GGML_BACKEND_MOE_HYBRID_TEST_CPU_JOINED}) {
            window.fail_phase = phase;
            test_hybrid_layers(device, false, false, merged, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, window);
        }
        window.fail_phase = 0;
        window.hits = window.routes;
        test_hybrid_layers(device, false, false, merged, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, window);
        window.hits = 1;
        test_hybrid_layers(device, true, false, merged, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, window);
        window.quota = window.routes;
        test_hybrid_layers(device, true, false, merged, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, window);
        window.quota = 0;
        test_hybrid_layers(device, true, false, merged, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, window);
        window.hits = window.routes;
        test_hybrid_layers(device, true, false, merged, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, window);
        window.quota = window.hits = 1;
        window.admission = false;
        test_hybrid_layers(device, true, false, merged, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, window);
        window.admission = true;
        window.no_host_alias = true;
        window.window_replays = 1000;
        test_hybrid_layers(device, true, false, merged, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, window);
        window.window_replays = 0;
        window.fail_phase = GGML_BACKEND_MOE_HYBRID_TEST_CPU_BEFORE_COMMIT;
        test_hybrid_layers(device, true, false, merged, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, window);
        window.fail_phase = GGML_BACKEND_MOE_HYBRID_TEST_ADMISSION_BANK_COPIED;
        test_hybrid_layers(device, true, false, merged, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, window);
        window.fail_phase = GGML_BACKEND_MOE_HYBRID_TEST_WINDOW_SUBMIT;
        test_hybrid_layers(device, true, false, merged, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, window);
        window.fail_phase = 0;
        test_hybrid_layers(device, true, true, merged, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, window);
        for (uint32_t failure : {1u, 2u, 3u, 4u, 5u, 6u}) {
            window.packet_failure = failure;
            test_hybrid_layers(device, false, false, merged, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, window);
            test_hybrid_layers(device, true, false, merged, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, window);
        }
        window.packet_failure = 0;
        window.quiesce = true;
        window.block_phase = GGML_BACKEND_MOE_HYBRID_TEST_CPU_QUEUED;
        test_hybrid_layers(device, true, false, merged, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, window);
        window.block_phase = GGML_BACKEND_MOE_HYBRID_TEST_CPU_BEFORE_COMMIT;
        window.close_source = true;
        test_hybrid_layers(device, true, false, merged, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, window);
        window.close_source = false;
        window.block_phase = GGML_BACKEND_MOE_HYBRID_TEST_CPU_JOINED;
        test_hybrid_layers(device, false, false, merged, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, window);
        test_hybrid_layers(device, true, false, merged, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, window);
        window.block_phase = GGML_BACKEND_MOE_HYBRID_TEST_BEFORE_PUBLISH;
        test_hybrid_layers(device, true, false, merged, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, window);
        window.quiesce = false;
        window.block_phase = 0;
        window.reset = true;
        test_hybrid_layers(device, true, false, merged, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, window);
        window.reset = false;
        window.routes = 4;
        window.duplicates = true;
        test_hybrid_layers(device, false, false, merged, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, window);
        test_hybrid_layers(device, true, false, merged, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, window);
        window.routes = 3;
        window.duplicates = false;
        test_hybrid_layers(device, false, false,
            {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, LLM_FFN_SILU, 256, 1792},
            GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, window);
        test_hybrid_layers(device, true, false,
            {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, LLM_FFN_SILU, 256, 1792},
            GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, window);
        if (staged_supported) {
            window.staged_input = true;
            test_hybrid_layers(device, true, false, merged, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, window);
            window.boundary_overlap = true;
            window.window_replays = 16;
            test_hybrid_layers(device, true, false, merged, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, window);
            window.window_replays = 0;
            window.fail_phase = GGML_BACKEND_MOE_HYBRID_TEST_CPU_BEFORE_COMMIT;
            test_hybrid_layers(device, true, false, merged, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, window);
            window.fail_phase = 0;
            window.staged_input = false;
            window.boundary_overlap = false;
        }
        window.auto_memory = false;
        window.window_fallback = true;
        test_hybrid_layers(device, true, false, merged, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, window);
        window.auto_memory = true;
        window.window_host_node = true;
        test_hybrid_layers(device, true, false, merged, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, window);
    }
    test_hybrid_layers(device, false, false, merged);
    test_hybrid_layers(device, true, false, merged);
    test_hybrid_layers(device, true, true, merged);
    test_hybrid_layers(device, false, false, {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, LLM_FFN_GELU, n_dim, n_dim});
    for (const auto & signature : {
            hybrid_layer_signature{GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_TYPE_Q5_K, GGML_TYPE_Q5_K, LLM_FFN_SILU, 256, 1792},
            hybrid_layer_signature{GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, LLM_FFN_SILU, 256, 1792},
            hybrid_layer_signature{GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, LLM_FFN_SILU, 256, 1792},
            hybrid_layer_signature{GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_TYPE_Q4_K, GGML_TYPE_Q5_0, LLM_FFN_SILU, 256, 1408},
            hybrid_layer_signature{GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_TYPE_Q4_K, GGML_TYPE_Q8_0, LLM_FFN_SILU, 256, 1408}}) {
        test_hybrid_layers(device, true, false, signature);
        test_hybrid_layers(device, true, true, signature);
        hybrid_test_policy batched;
        batched.quota = 0;
        batched.routes = 4;
        test_hybrid_layers(device, true, false, signature, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, batched);
    }
    test_hybrid_layers(device, true, false,
        {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_TYPE_Q4_K, GGML_TYPE_F32, LLM_FFN_SILU, 256, 1408},
        GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION);
    test_hybrid_layers(device, true, false,
        {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, LLM_FFN_SILU, 512, 6144},
        GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
    const hybrid_layer_signature mixed = {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE,
        GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, LLM_FFN_SILU, 256, 1792};
    hybrid_test_policy policy;
    policy.routes = 4;
    policy.hits = 0;
    policy.quota = 0;
    policy.reset = true;
    test_hybrid_layers(device, true, false, merged, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, policy);
    policy.reset = false;
    policy.quota = policy.routes;
    test_hybrid_layers(device, true, false, mixed, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, policy);
    policy.hits = 4;
    test_hybrid_layers(device, true, false, mixed, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, policy);
    policy.hits = 1;
    policy.quota = 2;
    test_hybrid_layers(device, true, false, mixed, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, policy);
    policy.duplicates = true;
    for (const uint32_t quota : {0u, 1u, 32u}) {
        policy.quota = quota;
        test_hybrid_layers(device, true, false, mixed, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, policy);
    }
    policy.duplicates = false;
    policy.quota = 1;
    policy.overlap = true;
    test_hybrid_layers(device, true, false, mixed, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, policy);
    policy.overlap = false;
    for (const uint32_t phase : {GGML_BACKEND_MOE_HYBRID_TEST_CPU_QUEUED, GGML_BACKEND_MOE_HYBRID_TEST_CPU_ADMITTED,
                                GGML_BACKEND_MOE_HYBRID_TEST_CPU_BEFORE_COMMIT, GGML_BACKEND_MOE_HYBRID_TEST_GPU_ENQUEUED,
                                GGML_BACKEND_MOE_HYBRID_TEST_BEFORE_PUBLISH}) {
        policy.fail_phase = phase;
        test_hybrid_layers(device, true, false, mixed, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, policy);
    }
    policy.fail_phase = 0;
    policy.gpu_pending = true;
    policy.fail_phase = GGML_BACKEND_MOE_HYBRID_TEST_GPU_ENQUEUED;
    for (const uint32_t phase : {GGML_BACKEND_MOE_HYBRID_TEST_CPU_QUEUED, GGML_BACKEND_MOE_HYBRID_TEST_CPU_BEFORE_COMMIT}) {
        policy.block_phase = phase;
        test_hybrid_layers(device, true, false, mixed, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, policy);
    }
    policy.gpu_pending = false;
    policy.fail_phase = 0;
    policy.quiesce = policy.close_source = true;
    for (const uint32_t phase : {GGML_BACKEND_MOE_HYBRID_TEST_CPU_QUEUED, GGML_BACKEND_MOE_HYBRID_TEST_CPU_BEFORE_COMMIT}) {
        policy.block_phase = phase;
        test_hybrid_layers(device, true, false, mixed, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, policy);
    }
    policy = {};
    policy.admission = true;
    policy.routes = 4;
    policy.hits = 1;
    for (const uint32_t quota : {0u, 1u, 32u}) {
        policy.quota = quota;
        test_hybrid_layers(device, true, false, mixed, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, policy);
    }
    for (const ggml_type down_type : {GGML_TYPE_Q5_0, GGML_TYPE_Q8_0}) {
        const hybrid_layer_signature deepseek = {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE,
            GGML_TYPE_Q4_K, down_type, LLM_FFN_SILU, 256, 1408};
        policy.quota = 1;
        test_hybrid_layers(device, true, false, deepseek, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, policy);
    }
    const hybrid_layer_signature merged_mixed = {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP,
        GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, LLM_FFN_SILU, 256, 1792};
    policy.quota = 32;
    policy.hits = 0;
    test_hybrid_layers(device, false, false, merged_mixed, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, policy);
    policy.hits = 4;
    test_hybrid_layers(device, true, false, mixed, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, policy);
    policy.hits = 1;
    policy.quota = 1;
    policy.duplicates = true;
    test_hybrid_layers(device, true, false, mixed, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, policy);
    policy.duplicates = false;
    policy.reset = true;
    test_hybrid_layers(device, true, false, merged, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, policy);
    policy.reset = false;
    for (const uint32_t phase : {GGML_BACKEND_MOE_HYBRID_TEST_ADMISSION_BANK_COPIED,
                                GGML_BACKEND_MOE_HYBRID_TEST_CPU_QUEUED, GGML_BACKEND_MOE_HYBRID_TEST_CPU_BEFORE_COMMIT,
                                GGML_BACKEND_MOE_HYBRID_TEST_BEFORE_PUBLISH}) {
        policy.fail_phase = phase;
        test_hybrid_layers(device, true, false, mixed, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, policy);
    }
    policy.fail_phase = GGML_BACKEND_MOE_HYBRID_TEST_ADMISSION_BANK_COPIED;
    test_hybrid_layers(device, true, false, merged_mixed, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, policy);
    policy.fail_phase = GGML_BACKEND_MOE_HYBRID_TEST_GPU_ENQUEUED;
    policy.block_phase = GGML_BACKEND_MOE_HYBRID_TEST_CPU_QUEUED;
    policy.gpu_pending = true;
    test_hybrid_layers(device, true, false, mixed, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, policy);
    policy.gpu_pending = false;
    policy.fail_phase = 0;
    policy.quiesce = policy.close_source = true;
    test_hybrid_layers(device, true, false, mixed, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, policy);
    policy = {};
    policy.resident_batch = policy.sparse_hits = true;
    policy.routes = 4;
    for (uint32_t hits = 1; hits <= policy.routes; ++hits) {
        policy.hits = hits;
        test_hybrid_layers(device, true, false, mixed, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, policy);
    }
    policy.hits = 3;
    for (const auto & signature : {
            hybrid_layer_signature{GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_TYPE_Q5_K, GGML_TYPE_Q5_K, LLM_FFN_SILU, 256, 1792},
            hybrid_layer_signature{GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, LLM_FFN_SILU, 256, 1792},
            hybrid_layer_signature{GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_TYPE_Q4_K, GGML_TYPE_Q5_0, LLM_FFN_SILU, 256, 1408},
            hybrid_layer_signature{GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_TYPE_Q4_K, GGML_TYPE_Q8_0, LLM_FFN_SILU, 256, 1408},
            merged_mixed, merged}) {
        test_hybrid_layers(device, true, false, signature, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, policy);
    }
    policy.hits = 4;
    policy.duplicates = true;
    test_hybrid_layers(device, true, false, merged_mixed, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, policy);
    policy.hits = 2;
    for (const uint32_t quota : {0u, 1u, 32u}) {
        policy.quota = quota;
        test_hybrid_layers(device, true, false, mixed, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, policy);
    }
    policy.duplicates = false;
    policy.hits = 0;
    for (const uint32_t quota : {0u, 32u}) {
        policy.quota = quota;
        test_hybrid_layers(device, true, false, mixed, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, policy);
    }
    policy.admission = true;
    test_hybrid_layers(device, false, false, merged_mixed, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, policy);
    policy.quota = 1;
    policy.hits = 2;
    policy.overlap = true;
    test_hybrid_layers(device, true, false, mixed, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, policy);
    policy.overlap = false;
    policy.reset = true;
    test_hybrid_layers(device, true, false, merged_mixed, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, policy);
    policy.reset = false;
    for (const uint32_t phase : {GGML_BACKEND_MOE_HYBRID_TEST_CPU_QUEUED, GGML_BACKEND_MOE_HYBRID_TEST_CPU_ADMITTED,
                                GGML_BACKEND_MOE_HYBRID_TEST_CPU_BEFORE_COMMIT, GGML_BACKEND_MOE_HYBRID_TEST_GPU_ENQUEUED,
                                GGML_BACKEND_MOE_HYBRID_TEST_ADMISSION_BANK_COPIED, GGML_BACKEND_MOE_HYBRID_TEST_BEFORE_PUBLISH}) {
        policy.fail_phase = phase;
        test_hybrid_layers(device, true, false, mixed, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, policy);
    }
    policy.fail_phase = GGML_BACKEND_MOE_HYBRID_TEST_GPU_ENQUEUED;
    policy.gpu_pending = true;
    for (const uint32_t phase : {GGML_BACKEND_MOE_HYBRID_TEST_CPU_QUEUED, GGML_BACKEND_MOE_HYBRID_TEST_CPU_BEFORE_COMMIT}) {
        policy.block_phase = phase;
        test_hybrid_layers(device, true, false, mixed, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, policy);
    }
    policy.gpu_pending = false;
    policy.fail_phase = 0;
    policy.quiesce = policy.close_source = true;
    for (const uint32_t phase : {GGML_BACKEND_MOE_HYBRID_TEST_CPU_QUEUED, GGML_BACKEND_MOE_HYBRID_TEST_CPU_BEFORE_COMMIT}) {
        policy.block_phase = phase;
        test_hybrid_layers(device, true, false, mixed, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, policy);
    }
    policy = {};
    policy.resident_batch = policy.sparse_hits = true;
    policy.routes = policy.hits = 4;
    {
        auto packet = policy;
        packet.hits = 1;
        for (uint32_t failure : {1u, 2u, 3u}) {
            packet.packet_failure = failure;
            test_hybrid_layers(device, false, false, mixed, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, packet);
        }
        packet.packet_failure = 0;
        packet.no_host_alias = true;
        packet.quota = 32;
        packet.duplicates = true;
        test_hybrid_layers(device, true, false, merged_mixed, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, packet);
        packet.hits = packet.routes;
        test_hybrid_layers(device, true, false, merged_mixed, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, packet);
    }
    {
        auto transfer_failure = policy;
        transfer_failure.hits = 1;
        transfer_failure.gpu_pending = transfer_failure.io_pending = true;
        transfer_failure.block_phase = GGML_BACKEND_MOE_HYBRID_TEST_CPU_BEFORE_COMMIT;
        transfer_failure.fail_phase = GGML_BACKEND_MOE_HYBRID_TEST_TRANSFER_ENQUEUED;
        transfer_failure.admission = true;
        test_hybrid_layers(device, true, false, mixed, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, transfer_failure);
    }
    const hybrid_layer_signature capacity = {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE,
        GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, LLM_FFN_SILU, 512, 6144};
    uint64_t exact_device_bytes = 0;
    policy.auto_memory = true;
    test_hybrid_layers(device, true, false, capacity, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, policy, &exact_device_bytes);
    CHECK(exact_device_bytes > 0);
    policy.auto_memory = false;
    policy.device_bytes = exact_device_bytes;
    test_hybrid_layers(device, true, false, capacity, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK, policy);
    --policy.device_bytes;
    test_hybrid_layers(device, true, false, capacity, GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY, policy);
    run_layers({ device }, 0);
    run_layers({ device }, bounded_host_budget);
    run_layers({ device }, 0, true);
    run_layers({ device }, 0, false, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_TYPE_Q4_K);
    run_layers({ device }, 0, false, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_UNGATED);
    run_layers({ device }, 0, false, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, GGML_TYPE_Q4_0, false, 1u << 0);
    run_layers({ device }, 0, false, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, GGML_TYPE_Q4_0, false, 1u << 1);
    run_layers({ device }, 0, false, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, GGML_TYPE_Q4_0, false, 0);
    run_layers({ device }, 0, false, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, GGML_TYPE_Q4_0, false,
               all_cached_layers, n_experts);
    for (uint32_t mask : {1u, 2u, all_cached_layers}) {
        run_layers({ device }, 0, false, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, GGML_TYPE_Q4_0, false,
                   mask, n_slots, true);
    }
    test_shared_source_owners(device, 0);
    test_shared_source_owners(device, 1024 * 1024);
    test_shared_source_owners(device, 0, true);
    run_layers({ device }, 0, false, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, GGML_TYPE_Q4_0, true);
    if (ggml_backend_reg_dev_count(ggml_backend_cuda_reg()) >= 3) {
        run_layers({ 0, 2 }, 0, false, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_TYPE_Q4_K, true);
        fprintf(stderr, "test-moe-cache: pageable layer owners 0/2 exact OK (logical device coverage)\n");
    }
    ggml_backend_cuda_moe_set_debug_mm(debug);
    fprintf(stderr, "test-moe-cache: single-device layer fixture OK (not multi-GPU validation)\n");
}

int test_grouped_multigpu() {
    int        count  = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || (status == cudaSuccess && count < 2)) {
        fprintf(stderr, "SKIP: grouped multi-GPU requires two distinct physical CUDA devices; found %d\n", count);
        return 77;
    }
    CUDA_OK(status);
    ggml_backend_dev_props props[2]{};
    auto *                 reg = ggml_backend_cuda_reg();
    if (ggml_backend_reg_dev_count(reg) < 2) {
        fprintf(stderr, "SKIP: CUDA registry exposes fewer than two backends\n");
        return 77;
    }
    for (int i = 0; i < 2; ++i) {
        ggml_backend_dev_get_props(ggml_backend_reg_dev_get(reg, i), &props[i]);
    }
    if (props[0].device_id && props[1].device_id && strcmp(props[0].device_id, props[1].device_id) == 0) {
        fprintf(stderr, "SKIP: logical CUDA backends share physical identity %s\n", props[0].device_id);
        return 77;
    }
    cudaDeviceProp first, second;
    CUDA_OK(cudaGetDeviceProperties(&first, 0));
    CUDA_OK(cudaGetDeviceProperties(&second, 1));
    if (memcmp(&first.uuid, &second.uuid, sizeof(first.uuid)) == 0 ||
        (first.pciDomainID == second.pciDomainID && first.pciBusID == second.pciBusID &&
         first.pciDeviceID == second.pciDeviceID)) {
        fprintf(stderr, "SKIP: grouped multi-GPU devices share a physical GPU\n");
        return 77;
    }
    for (int device = 0; device < 2; ++device) {
        cudaDeviceProp prop;
        CUDA_OK(cudaGetDeviceProperties(&prop, device));
        int peer = 0;
        CUDA_OK(cudaDeviceCanAccessPeer(&peer, device, 1 - device));
        fprintf(stderr, "test-moe-cache: physical device=%d pci=%04x:%02x:%02x cc=%d.%d peer=%d name=%s\n", device,
                prop.pciDomainID, prop.pciBusID, prop.pciDeviceID, prop.major, prop.minor, peer, prop.name);
    }
    const bool debug = ggml_backend_cuda_moe_get_debug_mm();
    ggml_backend_cuda_moe_set_debug_mm(true);
    run_layers({ 0, 1 }, 0);
    run_layers({ 0, 1 }, bounded_host_budget);
    run_layers({ 0, 1 }, 0, false, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, GGML_TYPE_Q4_0, false, 1u << 1);
    run_layers({ 0, 1 }, 0, false, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, GGML_TYPE_Q4_0, false, 1u << 0);
    run_layers({ 0, 1 }, 0, false, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, GGML_TYPE_Q4_0, true);
    run_layers({ 0, 1 }, 0, true);
    run_layers({ 0, 1 }, 0, false, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_TYPE_Q4_K);
    run_layers({ 0, 1 }, 0, false, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_UNGATED);
    ggml_backend_cuda_moe_set_debug_mm(debug);
    fprintf(stderr, "test-moe-cache: two-physical-device grouped layer execution OK\n");
    return 0;
}
