#include "test-moe-cache.h"

struct cached_fusion_test_graph {
    ggml_context_ptr weights;
    ggml_context_ptr auxiliaries;
    ggml_context_ptr nodes;
    ggml_backend_buffer_ptr weight_buffer;
    ggml_backend_buffer_ptr auxiliary_buffer;
    ggml_backend_buffer_ptr node_buffer;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * output = nullptr;
    std::vector<ggml_tensor *> leaves;
    std::vector<ggml_tensor *> cached_weights;
};

static cached_fusion_test_graph build_cached_fusion_test_graph(
        ggml_backend_t backend,
        ggml_backend_buffer_type_t weight_buft) {
    constexpr int64_t N_EXPERTS = 4;
    constexpr int64_t N_USED = 2;
    constexpr int64_t N_TOKENS = 1;
    constexpr int64_t N_IN = 256;
    constexpr int64_t N_OUT = 32;

    ggml_init_params weight_params = {
        /* .mem_size = */ ggml_tensor_overhead() * 64,
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };
    ggml_init_params node_params = {
        /* .mem_size = */ ggml_tensor_overhead() * 512 + ggml_graph_overhead_custom(256, false),
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };

    cached_fusion_test_graph result;
    result.weights.reset(ggml_init(weight_params));
    result.auxiliaries.reset(ggml_init(weight_params));
    result.nodes.reset(ggml_init(node_params));
    CHECK(result.weights != nullptr && result.auxiliaries != nullptr && result.nodes != nullptr);

    auto new_leaf = [&](ggml_context * ctx, ggml_type type, int n_dims, const int64_t * ne, const std::string & name) {
        ggml_tensor * tensor = ggml_new_tensor(ctx, type, n_dims, ne);
        ggml_set_name(tensor, name.c_str());
        result.leaves.push_back(tensor);
        return tensor;
    };
    auto new_weight = [&](ggml_type type, const std::string & name) {
        const int64_t ne[] = {N_IN, N_OUT, N_EXPERTS};
        ggml_tensor * tensor = new_leaf(result.weights.get(), type, 3, ne, name + ".weight");
        result.cached_weights.push_back(tensor);
        return tensor;
    };
    auto new_activation = [&](const std::string & name) {
        const int64_t ne[] = {N_IN, N_USED, N_TOKENS};
        return new_leaf(result.nodes.get(), GGML_TYPE_F32, 3, ne, name + ".input");
    };
    auto new_ids = [&](const std::string & name) {
        const int64_t ne[] = {N_USED, N_TOKENS};
        return new_leaf(result.nodes.get(), GGML_TYPE_I32, 2, ne, name + ".ids");
    };
    auto add_scale = [&](ggml_tensor * mmid, ggml_tensor * ids, const std::string & name) {
        const int64_t ne[] = {N_EXPERTS};
        ggml_tensor * scale = new_leaf(result.auxiliaries.get(), GGML_TYPE_F32, 1, ne, name + ".scale");
        ggml_tensor * selected = ggml_reshape_3d(result.nodes.get(), scale, 1, N_EXPERTS, 1);
        selected = ggml_repeat_4d(result.nodes.get(), selected, 1, N_EXPERTS, N_TOKENS, 1);
        selected = ggml_get_rows(result.nodes.get(), selected, ids);
        return ggml_mul(result.nodes.get(), mmid, selected);
    };
    auto add_bias = [&](ggml_tensor * mmid, ggml_tensor * ids, const std::string & name) {
        const int64_t ne[] = {N_OUT, N_EXPERTS};
        ggml_tensor * bias = new_leaf(result.nodes.get(), GGML_TYPE_F32, 2, ne, name + ".bias");
        return ggml_add_id(result.nodes.get(), mmid, bias, ids);
    };
    auto build_single = [&](ggml_type type, const std::string & name, bool with_scale, bool with_bias) {
        ggml_tensor * weight = new_weight(type, name);
        ggml_tensor * input = new_activation(name);
        ggml_tensor * ids = new_ids(name);
        ggml_tensor * output = ggml_mul_mat_id(result.nodes.get(), weight, input, ids);
        if (with_scale) {
            output = add_scale(output, ids, name);
        }
        if (with_bias) {
            output = add_bias(output, ids, name);
        }
        return output;
    };
    auto build_pair = [&](ggml_type type, const std::string & name, bool with_scale, bool with_bias) {
        ggml_tensor * gate_weight = new_weight(type, name + ".gate");
        ggml_tensor * up_weight = new_weight(type, name + ".up");
        ggml_tensor * input = new_activation(name);
        ggml_tensor * ids = new_ids(name);
        ggml_tensor * gate = ggml_mul_mat_id(result.nodes.get(), gate_weight, input, ids);
        ggml_tensor * up = ggml_mul_mat_id(result.nodes.get(), up_weight, input, ids);
        if (with_scale) {
            gate = add_scale(gate, ids, name + ".gate");
            up = add_scale(up, ids, name + ".up");
        }
        if (with_bias) {
            gate = add_bias(gate, ids, name + ".gate");
            up = add_bias(up, ids, name + ".up");
        }
        return ggml_glu_split(result.nodes.get(), gate, up, GGML_GLU_OP_SWIGLU);
    };

    std::vector<ggml_tensor *> outputs;
    outputs.push_back(build_single(GGML_TYPE_Q4_0, "test.ordinary.q4_0", false, false));
    outputs.push_back(build_single(GGML_TYPE_Q4_K, "test.ordinary.q4_k", false, false));
    outputs.push_back(build_pair(GGML_TYPE_NVFP4, "test.f1.no_bias", true, false));
    outputs.push_back(build_pair(GGML_TYPE_NVFP4, "test.f1.bias", true, true));
    outputs.push_back(build_pair(GGML_TYPE_Q4_0, "test.f2.q4_0", false, true));
    outputs.push_back(build_pair(GGML_TYPE_Q4_K, "test.f2.q4_k", false, true));
    outputs.push_back(build_pair(GGML_TYPE_Q4_0, "test.f3.q4_0", false, false));
    outputs.push_back(build_pair(GGML_TYPE_Q4_K, "test.f3.q4_k", false, false));
    outputs.push_back(build_single(GGML_TYPE_NVFP4, "test.f4.no_bias", true, false));
    outputs.push_back(build_single(GGML_TYPE_NVFP4, "test.f4.bias", true, true));
    outputs.push_back(build_single(GGML_TYPE_Q4_0, "test.f5.q4_0", false, true));
    outputs.push_back(build_single(GGML_TYPE_Q4_K, "test.f5.q4_k", false, true));
    outputs.push_back(build_single(GGML_TYPE_BF16, "test.f5.bf16", false, true));

    result.output = outputs[0];
    for (size_t i = 1; i < outputs.size(); ++i) {
        result.output = ggml_add(result.nodes.get(), result.output, outputs[i]);
    }
    ggml_set_name(result.output, "test.cached_fusion.output");

    result.graph = ggml_new_graph_custom(result.nodes.get(), 256, false);
    ggml_build_forward_expand(result.graph, result.output);
    result.weight_buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(result.weights.get(), weight_buft));
    result.auxiliary_buffer.reset(ggml_backend_alloc_ctx_tensors(result.auxiliaries.get(), backend));
    result.node_buffer.reset(ggml_backend_alloc_ctx_tensors(result.nodes.get(), backend));
    CHECK(result.weight_buffer != nullptr && result.auxiliary_buffer != nullptr && result.node_buffer != nullptr);
    ggml_backend_buffer_set_usage(result.weight_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_set_usage(result.auxiliary_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    return result;
}

void test_cached_mmid_fusion_decline() {
    const bool old_debug_mm = ggml_backend_cuda_moe_get_debug_mm();
    ggml_backend_cuda_moe_set_debug_mm(true);
    ggml_backend_ptr cuda_backend(ggml_backend_cuda_init(0));
    ggml_backend_ptr cpu_backend(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    CHECK(cuda_backend != nullptr && cpu_backend != nullptr);


    cached_fusion_test_graph cuda_graph = build_cached_fusion_test_graph(
        cuda_backend.get(), ggml_backend_cuda_moe_cached_buffer_type());
    cached_fusion_test_graph cpu_graph = build_cached_fusion_test_graph(
        cpu_backend.get(), ggml_backend_cpu_buffer_type());
    CHECK(cuda_graph.leaves.size() == cpu_graph.leaves.size());
    CHECK(cuda_graph.cached_weights.size() == 19);

    for (size_t i = 0; i < cuda_graph.leaves.size(); ++i) {
        CHECK(cuda_graph.leaves[i]->type == cpu_graph.leaves[i]->type);
        CHECK(ggml_nbytes(cuda_graph.leaves[i]) == ggml_nbytes(cpu_graph.leaves[i]));
        std::vector<uint8_t> data = cached_fusion_test_data(cuda_graph.leaves[i], i);
        ggml_backend_tensor_set(cuda_graph.leaves[i], data.data(), 0, data.size());
        ggml_backend_tensor_set(cpu_graph.leaves[i], data.data(), 0, data.size());
    }
    for (ggml_tensor * weight : cuda_graph.cached_weights) {
        CHECK(weight->buffer != nullptr && ggml_backend_buft_is_cuda_moe_cached(weight->buffer->buft));
    }
    const auto disabled = candidate_snapshot(4, nullptr, 0);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v1(cuda_backend.get(), &disabled) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);

    for (int pass = 0; pass < 2; ++pass) {
        CHECK(ggml_backend_graph_compute(cpu_backend.get(), cpu_graph.graph) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_graph_compute(cuda_backend.get(), cuda_graph.graph) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(cpu_backend.get());
        ggml_backend_synchronize(cuda_backend.get());
    }
    CHECK(active_grouped_legacy_op_count(cuda_backend.get()) == 2 * cuda_graph.cached_weights.size());

    std::vector<float> expected(ggml_nelements(cpu_graph.output));
    std::vector<float> actual(ggml_nelements(cuda_graph.output));
    ggml_backend_tensor_get(cpu_graph.output, expected.data(), 0, ggml_nbytes(cpu_graph.output));
    ggml_backend_tensor_get(cuda_graph.output, actual.data(), 0, ggml_nbytes(cuda_graph.output));
    double squared_error = 0.0;
    double squared_expected = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        CHECK(std::isfinite(actual[i]) && std::isfinite(expected[i]));
        const double difference = actual[i] - expected[i];
        squared_error += difference * difference;
        squared_expected += static_cast<double>(expected[i]) * expected[i];
    }
    CHECK(squared_expected > 0.0 && squared_error / squared_expected < 2e-2);

    ggml_backend_cuda_moe_set_debug_mm(old_debug_mm);
    fprintf(stderr, "test-moe-cache: cached MMID F1-F5 decline OK\n");
}

cached_mmid_path_test_graph build_cached_mmid_path_test_graph(
        ggml_backend_t backend,
        ggml_backend_buffer_type_t weight_buft,
        ggml_type weight_type,
        int64_t n_out,
        int64_t n_used,
        int64_t n_tokens,
        int64_t n_experts) {
    constexpr int64_t N_IN = 256;
    const ggml_init_params weight_params = {
        /* .mem_size = */ ggml_tensor_overhead() * 8,
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };
    const ggml_init_params node_params = {
        /* .mem_size = */ ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(32, false),
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };

    cached_mmid_path_test_graph result;
    result.weights.reset(ggml_init(weight_params));
    result.nodes.reset(ggml_init(node_params));
    CHECK(result.weights != nullptr && result.nodes != nullptr);

    const int64_t weight_ne[] = {N_IN, n_out, n_experts};
    ggml_tensor * weight = ggml_new_tensor(result.weights.get(), weight_type, 3, weight_ne);
    ggml_set_name(weight, "test.paths.ffn_up_exps.weight");
    const int64_t input_ne[] = {N_IN, 1, n_tokens};
    ggml_tensor * input = ggml_new_tensor(result.nodes.get(), GGML_TYPE_F32, 3, input_ne);
    ggml_set_name(input, "test.paths.input");
    const int64_t ids_ne[] = {n_used, n_tokens};
    result.ids = ggml_new_tensor(result.nodes.get(), GGML_TYPE_I32, 2, ids_ne);
    ggml_set_name(result.ids, "test.paths.ids");
    result.output = ggml_mul_mat_id(result.nodes.get(), weight, input, result.ids);
    ggml_set_name(result.output, "test.paths.output");
    result.leaves = {weight, input, result.ids};

    result.graph = ggml_new_graph_custom(result.nodes.get(), 32, false);
    ggml_build_forward_expand(result.graph, result.output);
    result.weight_buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(result.weights.get(), weight_buft));
    result.node_buffer.reset(ggml_backend_alloc_ctx_tensors(result.nodes.get(), backend));
    CHECK(result.weight_buffer != nullptr && result.node_buffer != nullptr);
    ggml_backend_buffer_set_usage(result.weight_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    return result;
}

cached_mmid_path_test_graph build_cached_mmid_path_test_graph(
        ggml_backend_t backend,
        ggml_tensor * gate_up,
        ggml_tensor * down,
        int64_t n_used,
        int64_t n_tokens) {
    CHECK(gate_up != nullptr && down != nullptr && gate_up->buffer != nullptr && down->buffer != nullptr &&
        ggml_n_dims(gate_up) == 3 && ggml_n_dims(down) == 3 && gate_up->ne[0] == down->ne[0] &&
        gate_up->ne[1] == 2 * down->ne[0] && gate_up->ne[2] == down->ne[2]);
    const ggml_init_params node_params = {
        /* .mem_size = */ ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(32, false),
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };

    cached_mmid_path_test_graph result;
    result.nodes.reset(ggml_init(node_params));
    CHECK(result.nodes != nullptr);

    const int64_t input_ne[] = {gate_up->ne[0], 1, n_tokens};
    ggml_tensor * input = ggml_new_tensor(result.nodes.get(), GGML_TYPE_F32, 3, input_ne);
    ggml_set_name(input, "test.paths.registered.input");
    const int64_t ids_ne[] = {n_used, n_tokens};
    result.ids = ggml_new_tensor(result.nodes.get(), GGML_TYPE_I32, 2, ids_ne);
    ggml_set_name(result.ids, "test.paths.registered.ids");
    ggml_tensor * hidden = ggml_mul_mat_id(result.nodes.get(), gate_up, input, result.ids);
    hidden = ggml_dup(result.nodes.get(), hidden);
    hidden = ggml_glu(result.nodes.get(), hidden, GGML_GLU_OP_SWIGLU, false);
    result.output = ggml_mul_mat_id(result.nodes.get(), down, hidden, result.ids);
    ggml_set_name(result.output, "test.paths.registered.output");
    result.leaves = {gate_up, down, input, result.ids};

    result.graph = ggml_new_graph_custom(result.nodes.get(), 32, false);
    ggml_build_forward_expand(result.graph, result.output);
    result.node_buffer.reset(ggml_backend_alloc_ctx_tensors(result.nodes.get(), backend));
    CHECK(result.node_buffer != nullptr);
    return result;
}

void initialize_cached_mmid_path_test_graphs(
        cached_mmid_path_test_graph & cuda_graph,
        cached_mmid_path_test_graph & reference_graph) {
    CHECK(cuda_graph.leaves.size() == reference_graph.leaves.size());
    for (size_t i = 0; i < cuda_graph.leaves.size(); ++i) {
        CHECK(cuda_graph.leaves[i]->type == reference_graph.leaves[i]->type);
        CHECK(ggml_nbytes(cuda_graph.leaves[i]) == ggml_nbytes(reference_graph.leaves[i]));
        std::vector<uint8_t> data = cached_fusion_test_data(cuda_graph.leaves[i], i + 31);
        ggml_backend_tensor_set(cuda_graph.leaves[i], data.data(), 0, data.size());
        ggml_backend_tensor_set(reference_graph.leaves[i], data.data(), 0, data.size());
    }
}

std::vector<float> run_cached_mmid_path_test(
        ggml_backend_t cached_backend,
        ggml_backend_t reference_backend,
        cached_mmid_path_test_graph & cuda_graph,
        cached_mmid_path_test_graph & reference_graph,
        const std::vector<int32_t> & ids) {
    CHECK(ids.size() == (size_t) ggml_nelements(cuda_graph.ids));
    ggml_backend_tensor_set(cuda_graph.ids, ids.data(), 0, ids.size() * sizeof(ids[0]));
    ggml_backend_tensor_set(reference_graph.ids, ids.data(), 0, ids.size() * sizeof(ids[0]));
    std::vector<uint8_t> output_zero(ggml_nbytes(cuda_graph.output), 0);
    ggml_backend_tensor_set(cuda_graph.output, output_zero.data(), 0, output_zero.size());
    ggml_backend_tensor_set(reference_graph.output, output_zero.data(), 0, output_zero.size());
    CHECK(ggml_backend_graph_compute(reference_backend, reference_graph.graph) == GGML_STATUS_SUCCESS);
    CHECK(ggml_backend_graph_compute(cached_backend, cuda_graph.graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(reference_backend);
    ggml_backend_synchronize(cached_backend);

    std::vector<float> expected(ggml_nelements(reference_graph.output));
    std::vector<float> actual(ggml_nelements(cuda_graph.output));
    ggml_backend_tensor_get(reference_graph.output, expected.data(), 0, ggml_nbytes(reference_graph.output));
    ggml_backend_tensor_get(cuda_graph.output, actual.data(), 0, ggml_nbytes(cuda_graph.output));
    double squared_error = 0.0;
    double squared_expected = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        CHECK(std::isfinite(actual[i]) && std::isfinite(expected[i]));
        const double difference = actual[i] - expected[i];
        squared_error += difference * difference;
        squared_expected += static_cast<double>(expected[i]) * expected[i];
    }
    const double relative_squared_error = squared_error / squared_expected;
    CHECK(squared_expected > 0.0 && relative_squared_error < 2e-5);
    return actual;
}

void test_cached_mmid_prefill_and_overflow() {
    ggml_backend_ptr cuda_backend(ggml_backend_cuda_init(0));
    ggml_backend_ptr reference_backend(ggml_backend_cuda_init(0));
    CHECK(cuda_backend != nullptr && reference_backend != nullptr);
    const auto disabled = candidate_snapshot(4, nullptr, 0);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v1(cuda_backend.get(), &disabled) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);

    auto cuda_prefill = build_cached_mmid_path_test_graph(
        cuda_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0, 128, 3, 2);
    auto reference_prefill = build_cached_mmid_path_test_graph(
        reference_backend.get(), ggml_backend_cuda_buffer_type(0), GGML_TYPE_Q4_0, 128, 3, 2);
    initialize_cached_mmid_path_test_graphs(cuda_prefill, reference_prefill);
    const std::vector<int32_t> mapped_ids = {0, 1, 2, 1, 2, 3};
    const auto mapped_first = run_cached_mmid_path_test(
        cuda_backend.get(), reference_backend.get(), cuda_prefill, reference_prefill, mapped_ids);
    const auto mapped_second = run_cached_mmid_path_test(
        cuda_backend.get(), reference_backend.get(), cuda_prefill, reference_prefill, mapped_ids);
    CHECK(mapped_first == mapped_second);
    (void) run_cached_mmid_path_test(
        cuda_backend.get(), reference_backend.get(), cuda_prefill, reference_prefill, {0, 1, 2, 3, 4, 5});

    for (ggml_type type : {GGML_TYPE_MXFP4, GGML_TYPE_NVFP4}) {
        auto cuda_fp4 = build_cached_mmid_path_test_graph(
            cuda_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), type, 128, 3, 2);
        auto reference_fp4 = build_cached_mmid_path_test_graph(
            reference_backend.get(), ggml_backend_cuda_buffer_type(0), type, 128, 3, 2);
        initialize_cached_mmid_path_test_graphs(cuda_fp4, reference_fp4);
        const auto fp4_first = run_cached_mmid_path_test(
            cuda_backend.get(), reference_backend.get(), cuda_fp4, reference_fp4, mapped_ids);
        const auto fp4_second = run_cached_mmid_path_test(
            cuda_backend.get(), reference_backend.get(), cuda_fp4, reference_fp4, mapped_ids);
        CHECK(fp4_first == fp4_second);
    }

    for (ggml_type type : {GGML_TYPE_MXFP4, GGML_TYPE_NVFP4}) {
        auto cuda_fp4_overflow = build_cached_mmid_path_test_graph(
            cuda_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), type, 128, 4, 16);
        auto reference_fp4_overflow = build_cached_mmid_path_test_graph(
            reference_backend.get(), ggml_backend_cuda_buffer_type(0), type, 128, 4, 16);
        initialize_cached_mmid_path_test_graphs(cuda_fp4_overflow, reference_fp4_overflow);
        static constexpr int32_t expert_order[] = {5, 2, 7, 1, 6, 0, 4, 3};
        std::vector<int32_t> overflow_ids(4 * 16);
        for (int32_t token = 0; token < 16; ++token) {
            for (int32_t route = 0; route < 4; ++route) {
                overflow_ids[token * 4 + route] = expert_order[(token + route) % 8];
            }
        }
        const auto overflow_output = run_cached_mmid_path_test(
            cuda_backend.get(), reference_backend.get(), cuda_fp4_overflow, reference_fp4_overflow, overflow_ids);
        std::vector<float> overflow_expected(ggml_nelements(reference_fp4_overflow.output));
        ggml_backend_tensor_get(reference_fp4_overflow.output, overflow_expected.data(), 0,
            ggml_nbytes(reference_fp4_overflow.output));
        CHECK(overflow_output == overflow_expected);
    }

    auto cuda_decode = build_cached_mmid_path_test_graph(
        cuda_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0, 128, 6, 1);
    auto reference_decode = build_cached_mmid_path_test_graph(
        reference_backend.get(), ggml_backend_cuda_buffer_type(0), GGML_TYPE_Q4_0, 128, 6, 1);
    initialize_cached_mmid_path_test_graphs(cuda_decode, reference_decode);
    (void) run_cached_mmid_path_test(
        cuda_backend.get(), reference_backend.get(), cuda_decode, reference_decode, {0, 1, 2, 3, 4, 5});

    auto cuda_q4_k = build_cached_mmid_path_test_graph(
        cuda_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_K, 256, 3, 2);
    auto reference_q4_k = build_cached_mmid_path_test_graph(
        reference_backend.get(), ggml_backend_cuda_buffer_type(0), GGML_TYPE_Q4_K, 256, 3, 2);
    initialize_cached_mmid_path_test_graphs(cuda_q4_k, reference_q4_k);
    (void) run_cached_mmid_path_test(
        cuda_backend.get(), reference_backend.get(), cuda_q4_k, reference_q4_k, mapped_ids);

    auto cuda_q4_k_tiny = build_cached_mmid_path_test_graph(
        cuda_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_K, 64, 3, 2);
    auto reference_q4_k_tiny = build_cached_mmid_path_test_graph(
        reference_backend.get(), ggml_backend_cuda_buffer_type(0), GGML_TYPE_Q4_K, 64, 3, 2);
    initialize_cached_mmid_path_test_graphs(cuda_q4_k_tiny, reference_q4_k_tiny);
    (void) run_cached_mmid_path_test(
        cuda_backend.get(), reference_backend.get(), cuda_q4_k_tiny, reference_q4_k_tiny, mapped_ids);

    ggml_backend_ptr transition_backend(ggml_backend_cuda_init(0));
    CHECK(transition_backend != nullptr);
    auto grouped = build_active_grouped_dispatch_graph(
        transition_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP);
    auto grouped_reference = build_active_grouped_dispatch_graph(
        reference_backend.get(), ggml_backend_cuda_buffer_type(0), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP);
    const auto grouped_input = cached_fusion_test_data(grouped.input, 181);
    const float grouped_logits[] = {-1.0f, 3.0f, 0.5f, 9.0f, 2.0f, 8.0f, -2.0f, 1.0f};
    ggml_backend_tensor_set(grouped.input, grouped_input.data(), 0, grouped_input.size());
    ggml_backend_tensor_set(grouped.logits, grouped_logits, 0, sizeof(grouped_logits));
    register_active_grouped_dispatch(
        transition_backend.get(), grouped, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 4);

    auto registered_prefill = build_cached_mmid_path_test_graph(
        transition_backend.get(), grouped.banks[0], grouped.banks[1], 3, 2);
    auto registered_reference = build_cached_mmid_path_test_graph(
        reference_backend.get(), grouped_reference.banks[0], grouped_reference.banks[1], 3, 2);
    initialize_cached_mmid_path_test_graphs(registered_prefill, registered_reference);
    const auto grouped_first = run_active_grouped_dispatch(transition_backend.get(), grouped, 2);
    CHECK(!grouped_first.empty());

    auto * transition_context = ggml_cuda_moe_grouped_context_for_test(transition_backend.get());
    ggml_cuda_moe_candidate_group_key transition_key;
    ggml_cuda_moe_grouped_acquisition overflow_resource;
    CHECK(transition_context != nullptr && transition_context->find_down_group_key(grouped.down, &transition_key));
    CHECK(transition_context->acquire_group_resources(transition_key, &overflow_resource));
    CHECK(ggml_cuda_moe_grouped_context_test_access::has_device_resource(*transition_context, transition_key));
    CHECK(ggml_cuda_moe_grouped_context_test_access::device_resource_complete(*transition_context, transition_key));
    CHECK(grouped.graph->nodes[0] != registered_prefill.graph->nodes[0]);
    const uint64_t overflow_resource_generation = overflow_resource.resource_generation;

    cudaStream_t transition_stream = nullptr;
    CUDA_OK(cudaStreamCreateWithFlags(&transition_stream, cudaStreamNonBlocking));
    const auto grouped_coverage = candidate_certify_graph(*transition_context, grouped.graph);
    std::shared_ptr<ggml_cuda_moe_graph_plan> grouped_plan;
    ggml_cuda_moe_graph_execution grouped_execution;
    CHECK(transition_context->prepare_graph_execution(
        grouped.graph, 800, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, &grouped_plan, &grouped_execution,
        grouped_coverage.epoch, grouped_coverage.nodes,
        grouped_coverage.mmid_count, grouped_coverage.mmid_fingerprint) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
    CHECK(grouped_execution.resolve_streams(candidate_test_graph_stream, transition_stream));
    uint64_t overflow_fingerprint = 0;
    std::vector<std::shared_ptr<void>> overflow_leases;
    CHECK(transition_context->graph_resource_fingerprint(
        grouped_execution, transition_stream, &overflow_fingerprint, &overflow_leases));
    CHECK(overflow_fingerprint != 0 && overflow_leases.size() == 1);
    std::weak_ptr<void> overflow_witness = overflow_leases[0];
    overflow_leases.clear();
    CHECK(!overflow_witness.expired());
    CHECK(ggml_cuda_moe_grouped_context_test_access::set_clock_bound(
        *transition_context, overflow_resource, UINT64_MAX));
    CHECK(run_active_grouped_dispatch(transition_backend.get(), grouped, 2) == grouped_first);
    CHECK(overflow_witness.expired());
    ggml_cuda_moe_grouped_acquisition refreshed_resource;
    CHECK(transition_context->acquire_group_resources(transition_key, &refreshed_resource));
    CHECK(refreshed_resource.resource_generation != overflow_resource_generation);
    CHECK(run_active_grouped_dispatch(transition_backend.get(), grouped, 4) == grouped_first);

    ggml_cuda_moe_grouped_acquisition grouped_resource;
    CHECK(transition_context->acquire_group_resources(transition_key, &grouped_resource));
    const uint64_t grouped_resource_generation = grouped_resource.resource_generation;
    uint64_t grouped_fingerprint = 0;
    std::vector<std::shared_ptr<void>> grouped_leases;
    CHECK(transition_context->graph_resource_fingerprint(
        grouped_execution, transition_stream, &grouped_fingerprint, &grouped_leases));
    CHECK(grouped_fingerprint != 0 && grouped_fingerprint != overflow_fingerprint && grouped_leases.size() == 1);
    std::weak_ptr<void> grouped_witness = grouped_leases[0];
    grouped_leases.clear();
    CHECK(!grouped_witness.expired());

    std::shared_ptr<ggml_cuda_moe_graph_plan> transition_plan;
    ggml_cuda_moe_graph_execution transition_execution;
    CHECK(transition_context->prepare_graph_execution(
        registered_prefill.graph, 801, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &transition_plan, &transition_execution) ==
        GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
    CHECK(transition_execution.size() == 1);
    CHECK(transition_execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_PREFILL_LEGACY);
    CHECK(ggml_cuda_moe_grouped_context_test_access::graph_has_complete_mmid_inventory(*transition_plan));
    CHECK(!ggml_cuda_moe_grouped_context_test_access::graph_group_has_decode_discovery(*transition_plan, 0));
    const ggml_cuda_moe_graph_plan * uncovered_prefill_plan = transition_plan.get();
    CHECK(transition_context->prepare_graph_execution(
        registered_prefill.graph, 802, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &transition_plan, &transition_execution) ==
        GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
    CHECK(transition_plan.get() != uncovered_prefill_plan);
    const auto prefill_coverage = candidate_certify_graph(*transition_context, registered_prefill.graph);
    CHECK(transition_context->prepare_graph_execution(
        registered_prefill.graph, 803, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &transition_plan, &transition_execution,
        prefill_coverage.epoch, prefill_coverage.nodes,
        prefill_coverage.mmid_count, prefill_coverage.mmid_fingerprint) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
    const ggml_cuda_moe_graph_plan * covered_prefill_plan = transition_plan.get();
    CHECK(transition_context->prepare_graph_execution(
        registered_prefill.graph, 804, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &transition_plan, &transition_execution,
        prefill_coverage.epoch, prefill_coverage.nodes,
        prefill_coverage.mmid_count, prefill_coverage.mmid_fingerprint) == GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
    CHECK(transition_plan.get() == covered_prefill_plan);
    CHECK(ggml_cuda_moe_grouped_context_test_access::graph_has_complete_mmid_inventory(*transition_plan));
    CHECK(!ggml_cuda_moe_grouped_context_test_access::graph_group_has_decode_discovery(*transition_plan, 0));
    CHECK(transition_execution.resolve_streams(candidate_test_graph_stream, reinterpret_cast<void *>(uintptr_t{1})));
    CHECK(transition_context->begin_graph_dispatch(&transition_execution, true));
    CHECK(transition_context->get_group_resources(grouped_resource, nullptr));
    CHECK(ggml_cuda_moe_grouped_context_test_access::has_device_resource(*transition_context, transition_key));
    CHECK(!grouped_witness.expired());
    void * first_bank_data = ggml_cuda_moe_grouped_context_test_access::device_bank_data(
        *transition_context, transition_key, grouped.banks[0]);
    CHECK(first_bank_data != nullptr);
    const uint32_t payload_sentinel = 0x5a17c3e9;
    uint32_t payload_probe = 0;
    CUDA_OK(cudaMemcpy(first_bank_data, &payload_sentinel, sizeof(payload_sentinel), cudaMemcpyHostToDevice));
    ggml_cuda_moe_grouped_context_test_access::fail_borrowed_cache_init_after_probe(*transition_context);
    CHECK(!transition_context->acquire_legacy_cache(grouped.banks[0]));
    CUDA_OK(cudaMemcpy(&payload_probe, first_bank_data, sizeof(payload_probe), cudaMemcpyDeviceToHost));
    CHECK(payload_probe == payload_sentinel);
    CHECK(ggml_cuda_moe_grouped_context_test_access::legacy_backing_count(*transition_context, transition_key) == 0);
    auto first_legacy = transition_context->acquire_legacy_cache(grouped.banks[0]);
    CHECK(first_legacy && first_legacy.get() != nullptr && first_legacy.acquisition().registered_source == 1 &&
        first_legacy.acquisition().group_index == transition_key.group_index);
    CUDA_OK(cudaMemcpy(&payload_probe, first_bank_data, sizeof(payload_probe), cudaMemcpyDeviceToHost));
    CHECK(payload_probe == payload_sentinel);
    CHECK(ggml_cuda_moe_cache_slot_ptr(first_legacy.get(), 0) ==
        ggml_cuda_moe_grouped_context_test_access::device_bank_data(*transition_context, transition_key, grouped.banks[0]));
    CHECK(ggml_cuda_moe_grouped_context_test_access::legacy_backing_count(*transition_context, transition_key) == 1);
    const uint64_t legacy_epoch = first_legacy.acquisition().group_authority_epoch;
    CHECK(ggml_cuda_moe_grouped_context_test_access::has_device_resource(*transition_context, transition_key));
    first_legacy = {};
    CHECK(transition_context->finish_graph_dispatch(&transition_execution));

    const auto registered_mapped_first = run_cached_mmid_path_test(
        transition_backend.get(), reference_backend.get(), registered_prefill, registered_reference, mapped_ids);
    const auto registered_mapped_second = run_cached_mmid_path_test(
        transition_backend.get(), reference_backend.get(), registered_prefill, registered_reference, mapped_ids);
    CHECK(registered_mapped_first == registered_mapped_second);
    CHECK(ggml_cuda_moe_grouped_context_test_access::has_device_resource(*transition_context, transition_key));
    auto repeated_legacy = transition_context->acquire_legacy_cache(grouped.banks[0]);
    CHECK(repeated_legacy && repeated_legacy.acquisition().group_authority_epoch == legacy_epoch);
    repeated_legacy = {};
    (void) run_cached_mmid_path_test(
        transition_backend.get(), reference_backend.get(), registered_prefill, registered_reference, {0, 1, 2, 3, 4, 5});
    CHECK(ggml_cuda_moe_grouped_context_test_access::has_device_resource(*transition_context, transition_key));
    CHECK(run_active_grouped_dispatch(transition_backend.get(), grouped, 2) == grouped_first);
    ggml_cuda_moe_grouped_acquisition regrouped_resource;
    CHECK(transition_context->acquire_group_resources(transition_key, &regrouped_resource));
    CHECK(regrouped_resource.resource_generation == grouped_resource_generation && !grouped_witness.expired());
    uint64_t stable_fingerprint = 0;
    CHECK(transition_context->graph_resource_fingerprint(grouped_execution, transition_stream, &stable_fingerprint));
    CHECK(stable_fingerprint == grouped_fingerprint);
    CUDA_OK(cudaStreamDestroy(transition_stream));
    fprintf(stderr, "test-moe-cache: registered grouped prefill stable backing OK\n");
    fprintf(stderr, "test-moe-cache: cached mapped prefill and overflow OK\n");
}

struct routed_separate_chain_test_graph {
    ggml_context_ptr weights;
    ggml_context_ptr nodes;
    ggml_backend_buffer_ptr weight_buffer;
    ggml_backend_buffer_ptr node_buffer;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * logits = nullptr;
    ggml_tensor * ids = nullptr;
    ggml_tensor * route_weights = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * up = nullptr;
    ggml_tensor * hidden = nullptr;
    ggml_tensor * down = nullptr;
    ggml_tensor * weighted = nullptr;
    ggml_tensor * output = nullptr;
    std::array<ggml_tensor *, 3> banks = {};
};

static routed_separate_chain_test_graph build_routed_separate_chain_test_graph(
        ggml_backend_t backend,
        ggml_backend_buffer_type_t weight_buft,
        int64_t n_tokens) {
    constexpr int64_t N_EXPERTS = 64;
    constexpr int64_t N_USED = 6;
    constexpr int64_t N_EMBD = 2048;
    constexpr int64_t N_FF = 1408;
    const ggml_init_params weight_params = {
        /* .mem_size = */ ggml_tensor_overhead() * 8,
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };
    const ggml_init_params node_params = {
        /* .mem_size = */ ggml_tensor_overhead() * 128 + ggml_graph_overhead_custom(128, false),
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };

    routed_separate_chain_test_graph result;
    result.weights.reset(ggml_init(weight_params));
    result.nodes.reset(ggml_init(node_params));
    CHECK(result.weights != nullptr && result.nodes != nullptr);

    result.banks[0] = ggml_new_tensor_3d(result.weights.get(), GGML_TYPE_Q4_K, N_EMBD, N_FF, N_EXPERTS);
    result.banks[1] = ggml_new_tensor_3d(result.weights.get(), GGML_TYPE_Q4_K, N_EMBD, N_FF, N_EXPERTS);
    result.banks[2] = ggml_new_tensor_3d(result.weights.get(), GGML_TYPE_Q8_0, N_FF, N_EMBD, N_EXPERTS);
    ggml_set_name(result.banks[0], "blk.1.ffn_gate_exps.weight");
    ggml_set_name(result.banks[1], "blk.1.ffn_up_exps.weight");
    ggml_set_name(result.banks[2], "blk.1.ffn_down_exps.weight");

    result.input = ggml_new_tensor_3d(result.nodes.get(), GGML_TYPE_F32, N_EMBD, 1, n_tokens);
    result.logits = ggml_new_tensor_2d(result.nodes.get(), GGML_TYPE_F32, N_EXPERTS, n_tokens);
    ggml_set_name(result.input, "test.routed.input");
    ggml_set_name(result.logits, "test.routed.logits");
    ggml_tensor * probs = ggml_soft_max(result.nodes.get(), result.logits);
    result.ids = ggml_argsort_top_k(result.nodes.get(), probs, N_USED);
    ggml_tensor * probs_3d = ggml_reshape_3d(result.nodes.get(), probs, 1, N_EXPERTS, n_tokens);
    result.route_weights = ggml_get_rows(result.nodes.get(), probs_3d, result.ids);

    result.gate = ggml_mul_mat_id(result.nodes.get(), result.banks[0], result.input, result.ids);
    result.up = ggml_mul_mat_id(result.nodes.get(), result.banks[1], result.input, result.ids);
    result.hidden = ggml_swiglu_split(result.nodes.get(), result.gate, result.up);
    result.down = ggml_mul_mat_id(result.nodes.get(), result.banks[2], result.hidden, result.ids);
    result.weighted = ggml_mul(result.nodes.get(), result.down, result.route_weights);

    std::array<ggml_tensor *, N_USED> routes = {};
    for (int64_t route = 0; route < N_USED; ++route) {
        routes[route] = ggml_view_2d(
            result.nodes.get(), result.weighted, N_EMBD, n_tokens, result.weighted->nb[2], route * result.weighted->nb[1]);
    }
    result.output = routes[0];
    for (int64_t route = 1; route < N_USED; ++route) {
        result.output = ggml_add(result.nodes.get(), result.output, routes[route]);
    }
    ggml_set_name(result.output, "test.routed.output");

    result.graph = ggml_new_graph_custom(result.nodes.get(), 128, false);
    ggml_build_forward_expand(result.graph, result.route_weights);
    ggml_build_forward_expand(result.graph, result.output);
    candidate_stamp_execution(result.graph, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
        n_tokens == 1 ? GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT : GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL,
        n_tokens, 1);
    result.weight_buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(result.weights.get(), weight_buft));
    result.node_buffer.reset(ggml_backend_alloc_ctx_tensors(result.nodes.get(), backend));
    CHECK(result.weight_buffer != nullptr && result.node_buffer != nullptr);
    ggml_backend_buffer_set_usage(result.weight_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    return result;
}

static std::vector<float> routed_separate_chain_logits(
        int64_t n_tokens,
        int64_t expert_span,
        int64_t expert_offset) {
    constexpr int64_t N_EXPERTS = 64;
    constexpr int64_t N_USED = 6;
    CHECK(expert_span >= N_USED && expert_span <= N_EXPERTS);
    std::vector<float> result(N_EXPERTS * n_tokens);
    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t expert = 0; expert < N_EXPERTS; ++expert) {
            result[token * N_EXPERTS + expert] = -20.0f - 0.01f * expert;
        }
        for (int64_t route = 0; route < N_USED; ++route) {
            const int64_t expert = (expert_offset + (token * N_USED + route * 11) % expert_span) % N_EXPERTS;
            result[token * N_EXPERTS + expert] = 10.0f - route;
        }
    }
    return result;
}

static std::vector<uint8_t> routed_separate_chain_test_weight_data(
        const ggml_tensor * tensor,
        uint32_t salt) {
    CHECK(ggml_is_quantized(tensor->type));
    std::vector<float> values(ggml_nelements(tensor));
    uint32_t state = 0x9e3779b9u ^ salt;
    for (float & value : values) {
        state = state * 1664525u + 1013904223u;
        value = (static_cast<int32_t>((state >> 8) % 2001) - 1000) * 0.00035f;
    }
    std::vector<uint8_t> bytes(ggml_nbytes(tensor));
    const int64_t nrows = ggml_nelements(tensor) / tensor->ne[0];
    CHECK(ggml_quantize_chunk(
        tensor->type, values.data(), bytes.data(), 0, nrows, tensor->ne[0], nullptr) == bytes.size());
    return bytes;
}

static void initialize_routed_separate_chain_test_weights(
        routed_separate_chain_test_graph & candidate,
        routed_separate_chain_test_graph & reference) {
    for (size_t bank = 0; bank < candidate.banks.size(); ++bank) {
        CHECK(candidate.banks[bank]->type == reference.banks[bank]->type &&
            ggml_nbytes(candidate.banks[bank]) == ggml_nbytes(reference.banks[bank]));
        ggml_tensor expert = *candidate.banks[bank];
        expert.ne[2] = 1;
        expert.ne[3] = 1;
        expert.nb[3] = expert.nb[2];
        CHECK(ggml_nbytes(&expert) == candidate.banks[bank]->nb[2]);
        for (int64_t expert_id = 0; expert_id < candidate.banks[bank]->ne[2]; ++expert_id) {
            const auto bytes = routed_separate_chain_test_weight_data(
                &expert, 311 + 64 * bank + static_cast<uint32_t>(expert_id));
            const size_t offset = expert_id * candidate.banks[bank]->nb[2];
            ggml_backend_tensor_set(candidate.banks[bank], bytes.data(), offset, bytes.size());
            ggml_backend_tensor_set(reference.banks[bank], bytes.data(), offset, bytes.size());
        }
    }
}

static void set_routed_separate_chain_test_inputs(
        routed_separate_chain_test_graph & candidate,
        routed_separate_chain_test_graph & reference,
        size_t pass,
        int64_t expert_span,
        int64_t expert_offset) {
    CHECK(candidate.input->ne[2] == reference.input->ne[2]);
    const auto input = cached_fusion_test_data(candidate.input, 317 + pass);
    const auto logits = routed_separate_chain_logits(candidate.input->ne[2], expert_span, expert_offset);
    ggml_backend_tensor_set(candidate.input, input.data(), 0, input.size());
    ggml_backend_tensor_set(reference.input, input.data(), 0, input.size());
    ggml_backend_tensor_set(candidate.logits, logits.data(), 0, logits.size() * sizeof(float));
    ggml_backend_tensor_set(reference.logits, logits.data(), 0, logits.size() * sizeof(float));
}

static void register_routed_separate_chain_test_graph(
        ggml_backend_t backend,
        const routed_separate_chain_test_graph & graph,
        uint32_t n_slots) {
    const std::array<ggml_backend_moe_candidate_bank_v1, 3> banks = {{
        {graph.banks[0], GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT, 0},
        {graph.banks[1], GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT, 0},
        {graph.banks[2], GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
    }};
    const ggml_backend_moe_candidate_group_v1 group = {
        banks.data(), banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 0, 0,
    };
    const auto snapshot = candidate_snapshot(n_slots, &group, 1);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v1(backend, &snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
}

struct routed_separate_chain_test_result {
    std::vector<int32_t> ids;
    std::vector<float> route_weights;
    std::vector<float> gate;
    std::vector<float> up;
    std::vector<float> hidden;
    std::vector<float> down;
    std::vector<float> weighted;
    std::vector<float> output;
};

static routed_separate_chain_test_result run_routed_separate_chain_test_graph(
        ggml_backend_t backend,
        routed_separate_chain_test_graph & graph) {
    CHECK(ggml_backend_graph_compute(backend, graph.graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
    routed_separate_chain_test_result result;
    result.ids.resize(ggml_nelements(graph.ids));
    CHECK(graph.ids->view_src != nullptr && graph.ids->view_src->type == GGML_TYPE_I32 &&
        graph.ids->nb[0] == sizeof(int32_t) && graph.ids->nb[1] % sizeof(int32_t) == 0);
    std::vector<int32_t> sorted(ggml_nelements(graph.ids->view_src));
    ggml_backend_tensor_get(graph.ids->view_src, sorted.data(), 0, ggml_nbytes(graph.ids->view_src));
    const size_t row_stride = graph.ids->nb[1] / sizeof(int32_t);
    for (int64_t row = 0; row < graph.ids->ne[1]; ++row) {
        for (int64_t route = 0; route < graph.ids->ne[0]; ++route) {
            result.ids[row * graph.ids->ne[0] + route] = sorted[row * row_stride + route];
        }
    }
    result.route_weights.resize(ggml_nelements(graph.route_weights));
    result.gate.resize(ggml_nelements(graph.gate));
    result.up.resize(ggml_nelements(graph.up));
    result.hidden.resize(ggml_nelements(graph.hidden));
    result.down.resize(ggml_nelements(graph.down));
    result.weighted.resize(ggml_nelements(graph.weighted));
    result.output.resize(ggml_nelements(graph.output));
    ggml_backend_tensor_get(graph.route_weights, result.route_weights.data(), 0, ggml_nbytes(graph.route_weights));
    ggml_backend_tensor_get(graph.gate, result.gate.data(), 0, ggml_nbytes(graph.gate));
    ggml_backend_tensor_get(graph.up, result.up.data(), 0, ggml_nbytes(graph.up));
    ggml_backend_tensor_get(graph.hidden, result.hidden.data(), 0, ggml_nbytes(graph.hidden));
    ggml_backend_tensor_get(graph.down, result.down.data(), 0, ggml_nbytes(graph.down));
    ggml_backend_tensor_get(graph.weighted, result.weighted.data(), 0, ggml_nbytes(graph.weighted));
    ggml_backend_tensor_get(graph.output, result.output.data(), 0, ggml_nbytes(graph.output));
    return result;
}

static void check_routed_separate_chain_exact(
        const char * phase,
        const routed_separate_chain_test_result & expected,
        const routed_separate_chain_test_result & actual) {
    CHECK(actual.ids == expected.ids && actual.route_weights == expected.route_weights);
    const auto check_values = [phase, &actual](
            const char * name,
            const std::vector<float> & expected_values,
            const std::vector<float> & actual_values) {
        CHECK(actual_values.size() == expected_values.size());
        for (size_t i = 0; i < actual_values.size(); ++i) {
            if (!std::isfinite(actual_values[i]) || !std::isfinite(expected_values[i])) {
                fprintf(stderr, "test-moe-cache: routed %s %s first_nonfinite=%zu expected=%.9g actual=%.9g\n",
                    phase, name, i, expected_values[i], actual_values[i]);
                if (strcmp(name, "down") == 0) {
                    constexpr size_t N_EMBD = 2048;
                    constexpr size_t N_USED = 6;
                    const size_t row = i / N_EMBD;
                    const size_t token = row / N_USED;
                    const size_t route = row % N_USED;
                    fprintf(stderr, "test-moe-cache: routed %s down token=%zu route=%zu expert=%d element=%zu\n",
                        phase, token, route, actual.ids[token * N_USED + route], i % N_EMBD);
                }
                CHECK(false);
            }
            if (actual_values[i] != expected_values[i]) {
                fprintf(stderr, "test-moe-cache: routed %s %s first_difference=%zu expected=%.9g actual=%.9g\n",
                    phase, name, i, expected_values[i], actual_values[i]);
                CHECK(false);
            }
        }
    };
    check_values("gate", expected.gate, actual.gate);
    check_values("up", expected.up, actual.up);
    check_values("hidden", expected.hidden, actual.hidden);
    check_values("down", expected.down, actual.down);
    check_values("weighted", expected.weighted, actual.weighted);
    check_values("output", expected.output, actual.output);
}

void test_cached_mmid_routed_separate_chain() {
    constexpr uint32_t N_SLOTS = 63;
    const bool old_debug_mm = ggml_backend_cuda_moe_get_debug_mm();
    ggml_backend_cuda_moe_set_debug_mm(true);

    {
        constexpr int64_t N_TOKENS = 512;
        ggml_backend_ptr reference_backend(ggml_backend_cuda_init(0));
        ggml_backend_ptr candidate_backend(ggml_backend_cuda_init(0));
        CHECK(reference_backend != nullptr && candidate_backend != nullptr);
        auto reference = build_routed_separate_chain_test_graph(
            reference_backend.get(), ggml_backend_cuda_buffer_type(0), N_TOKENS);
        auto candidate = build_routed_separate_chain_test_graph(
            candidate_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), N_TOKENS);
        initialize_routed_separate_chain_test_weights(candidate, reference);
        const auto disabled = candidate_snapshot(N_SLOTS, nullptr, 0);
        CHECK(ggml_backend_cuda_moe_candidate_replace_v1(reference_backend.get(), &disabled) ==
            GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        register_routed_separate_chain_test_graph(candidate_backend.get(), candidate, N_SLOTS);

        auto * context = ggml_cuda_moe_grouped_context_for_test(candidate_backend.get());
        CHECK(context != nullptr);
        const auto coverage = candidate_certify_graph(*context, candidate.graph);
        std::shared_ptr<ggml_cuda_moe_graph_plan> plan;
        ggml_cuda_moe_graph_execution execution;
        CHECK(context->prepare_graph_execution(
            candidate.graph, 991, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &plan, &execution,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_PREFILL_LEGACY);

        cudaStream_t cache_stream = nullptr;
        CUDA_OK(cudaStreamCreateWithFlags(&cache_stream, cudaStreamNonBlocking));
        std::array<ggml_cuda_moe_legacy_cache_lease, 3> caches;
        for (size_t bank = 0; bank < caches.size(); ++bank) {
            caches[bank] = context->acquire_legacy_cache(candidate.banks[bank], nullptr, nullptr, cache_stream);
            CHECK(caches[bank] && caches[bank].acquisition().registered_source == 1);
            CHECK(ggml_cuda_moe_cache_n_slots(caches[bank].get()) == static_cast<int>(N_SLOTS));
            ggml_cuda_moe_cache_reset_stats(caches[bank].get());
        }
        const bool overlap = ggml_cuda_moe_cache_can_overlap_staging(caches[0].get());
        CUDA_OK(cudaStreamSynchronize(cache_stream));
        CUDA_OK(cudaStreamDestroy(cache_stream));
        ggml_cuda_moe_candidate_group_key group_key;
        CHECK(context->find_down_group_key(candidate.banks[2], &group_key));
        CHECK(ggml_cuda_moe_grouped_context_test_access::legacy_backing_count(*context, group_key) == 3);

        struct phase_spec {
            const char * name;
            int64_t expert_span;
            int64_t expert_offset;
            std::array<std::array<uint64_t, 3>, 3> cache_stats;
            ggml_cuda_moe_legacy_debug_telemetry telemetry;
        };
        const std::array<phase_spec, 3> phases = {{
            {"overflow-1", 64, 0, {{{63, 63, 0}, {0, 63, 0}, {63, 63, 0}}}, {3, 3, 3, 3, 64, 2}},
            {"narrow", 7, 45, {{{83, 64, 1}, {20, 64, 1}, {83, 64, 1}}}, {6, 3, 3, 3, 64, 4}},
            {"overflow-2", 64, 0, {{{89, 121, 58}, {26, 121, 58}, {152, 121, 58}}}, {9, 6, 6, 6, 64, 6}},
        }};
        for (size_t pass = 0; pass < phases.size(); ++pass) {
            const auto & phase = phases[pass];
            set_routed_separate_chain_test_inputs(
                candidate, reference, pass, phase.expert_span, phase.expert_offset);
            const auto expected = run_routed_separate_chain_test_graph(reference_backend.get(), reference);
            const auto actual = run_routed_separate_chain_test_graph(candidate_backend.get(), candidate);
            CHECK(std::unordered_set<int32_t>(actual.ids.begin(), actual.ids.end()).size() ==
                static_cast<size_t>(phase.expert_span));
            fprintf(stderr, "test-moe-cache: routed %s cache", phase.name);
            for (size_t bank = 0; bank < caches.size(); ++bank) {
                const auto & cache = caches[bank];
                uint64_t hits = 0;
                uint64_t misses = 0;
                uint64_t evictions = 0;
                ggml_cuda_moe_cache_stats(cache.get(), &hits, &misses, &evictions);
                fprintf(stderr, " %llu/%llu/%llu",
                    (unsigned long long) hits, (unsigned long long) misses, (unsigned long long) evictions);
                const auto & expected_stats = phase.cache_stats[!overlap && bank == 0 ? 1 : bank];
                CHECK(hits == expected_stats[0]);
                CHECK(misses == expected_stats[1]);
                CHECK(evictions == expected_stats[2]);
            }
            fprintf(stderr, "\n");
            const auto telemetry = ggml_cuda_moe_grouped_context_test_access::legacy_debug_telemetry(*context, false);
            fprintf(stderr, "test-moe-cache: routed %s legacy %llu/%llu/%llu/%llu/%llu/%llu\n",
                phase.name,
                (unsigned long long) telemetry.ops,
                (unsigned long long) telemetry.staged_ops,
                (unsigned long long) telemetry.split_staged_ops,
                (unsigned long long) telemetry.overflow_ops,
                (unsigned long long) telemetry.unique_experts_max,
                (unsigned long long) telemetry.ids_cache_hits);
            CHECK(telemetry.ops == phase.telemetry.ops);
            CHECK(telemetry.staged_ops == phase.telemetry.staged_ops);
            CHECK(telemetry.split_staged_ops == (overlap ? phase.telemetry.split_staged_ops : 0));
            CHECK(telemetry.overflow_ops == phase.telemetry.overflow_ops);
            CHECK(telemetry.unique_experts_max == phase.telemetry.unique_experts_max);
            CHECK(telemetry.ids_cache_hits == phase.telemetry.ids_cache_hits);
            check_routed_separate_chain_exact(phase.name, expected, actual);
        }
        CHECK(active_grouped_legacy_op_count(candidate_backend.get(), true) == 0);
        fprintf(stderr, "test-moe-cache: routed separate prefill exact overlap=%d OK\n", overlap);
    }

    {
        constexpr uint32_t MULTIWAVE_N_SLOTS = 24;
        constexpr int64_t N_TOKENS = 512;
        ggml_backend_ptr reference_backend(ggml_backend_cuda_init(0));
        ggml_backend_ptr candidate_backend(ggml_backend_cuda_init(0));
        CHECK(reference_backend != nullptr && candidate_backend != nullptr);
        auto reference = build_routed_separate_chain_test_graph(
            reference_backend.get(), ggml_backend_cuda_buffer_type(0), N_TOKENS);
        auto candidate = build_routed_separate_chain_test_graph(
            candidate_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), N_TOKENS);
        initialize_routed_separate_chain_test_weights(candidate, reference);
        const auto disabled = candidate_snapshot(MULTIWAVE_N_SLOTS, nullptr, 0);
        CHECK(ggml_backend_cuda_moe_candidate_replace_v1(reference_backend.get(), &disabled) ==
            GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        register_routed_separate_chain_test_graph(candidate_backend.get(), candidate, MULTIWAVE_N_SLOTS);

        auto * context = ggml_cuda_moe_grouped_context_for_test(candidate_backend.get());
        CHECK(context != nullptr);
        const auto coverage = candidate_certify_graph(*context, candidate.graph);
        std::shared_ptr<ggml_cuda_moe_graph_plan> plan;
        ggml_cuda_moe_graph_execution execution;
        CHECK(context->prepare_graph_execution(
            candidate.graph, 993, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &plan, &execution,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_PREFILL_LEGACY);

        cudaStream_t cache_stream = nullptr;
        CUDA_OK(cudaStreamCreateWithFlags(&cache_stream, cudaStreamNonBlocking));
        std::array<ggml_cuda_moe_legacy_cache_lease, 3> caches;
        for (size_t bank = 0; bank < caches.size(); ++bank) {
            caches[bank] = context->acquire_legacy_cache(candidate.banks[bank], nullptr, nullptr, cache_stream);
            CHECK(caches[bank] && caches[bank].acquisition().registered_source == 1);
            CHECK(ggml_cuda_moe_cache_n_slots(caches[bank].get()) == static_cast<int>(MULTIWAVE_N_SLOTS));
            ggml_cuda_moe_cache_reset_stats(caches[bank].get());
        }
        const bool overlap = ggml_cuda_moe_cache_can_overlap_staging(caches[0].get());
        CUDA_OK(cudaStreamSynchronize(cache_stream));
        CUDA_OK(cudaStreamDestroy(cache_stream));
        ggml_cuda_moe_candidate_group_key group_key;
        CHECK(context->find_down_group_key(candidate.banks[2], &group_key));
        CHECK(ggml_cuda_moe_grouped_context_test_access::legacy_backing_count(*context, group_key) == 3);
        CHECK(ggml_cuda_moe_cache_trailing_padding_bytes_for_test(caches[0].get()) == 0);
        CHECK(ggml_cuda_moe_cache_trailing_padding_bytes_for_test(caches[1].get()) == 0);
        CHECK(ggml_cuda_moe_cache_trailing_padding_bytes_for_test(caches[2].get()) == 136);
        CHECK(ggml_cuda_moe_cache_trailing_padding_zero_for_test(caches[2].get()));

        set_routed_separate_chain_test_inputs(candidate, reference, 17, 64, 0);
        ggml_cuda_moe_grouped_context_test_access::poison_split_staging(*context, 3);
        const auto expected = run_routed_separate_chain_test_graph(reference_backend.get(), reference);
        const auto actual = run_routed_separate_chain_test_graph(candidate_backend.get(), candidate);
        CHECK(ggml_cuda_moe_grouped_context_test_access::split_staging_poison_calls(*context) == (overlap ? 0 : 3));
        CHECK(std::unordered_set<int32_t>(actual.ids.begin(), actual.ids.end()).size() == 64);
        const std::array<std::array<uint64_t, 3>, 3> expected_cache_stats = {{
            {24, 24, 0}, {0, 24, 0}, {24, 24, 0},
        }};
        for (size_t bank = 0; bank < caches.size(); ++bank) {
            uint64_t hits = 0;
            uint64_t misses = 0;
            uint64_t evictions = 0;
            ggml_cuda_moe_cache_stats(caches[bank].get(), &hits, &misses, &evictions);
            const auto & expected_stats = expected_cache_stats[!overlap && bank == 0 ? 1 : bank];
            CHECK(hits == expected_stats[0]);
            CHECK(misses == expected_stats[1]);
            CHECK(evictions == expected_stats[2]);
        }
        const auto telemetry = ggml_cuda_moe_grouped_context_test_access::legacy_debug_telemetry(*context, false);
        CHECK(telemetry.ops == 3 && telemetry.staged_ops == 3 && telemetry.split_staged_ops == (overlap ? 3 : 0));
        CHECK(telemetry.overflow_ops == 3 && telemetry.unique_experts_max == 64 && telemetry.ids_cache_hits == 2);
        check_routed_separate_chain_exact("multiwave", expected, actual);
        CHECK(ggml_cuda_moe_cache_trailing_padding_zero_for_test(caches[2].get()));
        CHECK(active_grouped_legacy_op_count(candidate_backend.get(), true) == 0);
        fprintf(stderr, "test-moe-cache: routed separate cache24 exact overlap=%d OK\n", overlap);
    }

    {
        constexpr int64_t N_TOKENS = 1;
        ggml_backend_ptr reference_backend(ggml_backend_cuda_init(0));
        ggml_backend_ptr candidate_backend(ggml_backend_cuda_init(0));
        CHECK(reference_backend != nullptr && candidate_backend != nullptr);
        auto reference = build_routed_separate_chain_test_graph(
            reference_backend.get(), ggml_backend_cuda_buffer_type(0), N_TOKENS);
        auto candidate = build_routed_separate_chain_test_graph(
            candidate_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), N_TOKENS);
        initialize_routed_separate_chain_test_weights(candidate, reference);
        set_routed_separate_chain_test_inputs(candidate, reference, 0, 64, 0);
        const auto disabled = candidate_snapshot(N_SLOTS, nullptr, 0);
        CHECK(ggml_backend_cuda_moe_candidate_replace_v1(reference_backend.get(), &disabled) ==
            GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        register_routed_separate_chain_test_graph(candidate_backend.get(), candidate, N_SLOTS);

        auto * context = ggml_cuda_moe_grouped_context_for_test(candidate_backend.get());
        CHECK(context != nullptr);
        const auto coverage = candidate_certify_graph(*context, candidate.graph);
        std::shared_ptr<ggml_cuda_moe_graph_plan> plan;
        ggml_cuda_moe_graph_execution execution;
        CHECK(context->prepare_graph_execution(
            candidate.graph, 992, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &plan, &execution,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        fprintf(stderr, "test-moe-cache: routed decode outcome=%u size=%u eligible=%u execution=%u materialization=%u consumer=%u auxiliary=%u\n",
            execution.outcome(), execution.size(),
            ggml_cuda_moe_grouped_context_test_access::graph_group_has_eligible_reason(*plan, 0),
            ggml_cuda_moe_grouped_context_test_access::graph_group_has_execution_reason(*plan, 0),
            ggml_cuda_moe_grouped_context_test_access::graph_group_has_materialization_reason(*plan, 0),
            ggml_cuda_moe_grouped_context_test_access::graph_group_has_consumer_equivalence_reason(*plan, 0),
            ggml_cuda_moe_grouped_context_test_access::graph_group_has_auxiliary_reason(*plan, 0));
        CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED && execution.size() == 1);
        CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_eligible_reason(*plan, 0));

        const auto expected = run_routed_separate_chain_test_graph(reference_backend.get(), reference);
        const auto actual = run_routed_separate_chain_test_graph(candidate_backend.get(), candidate);
        check_routed_separate_chain_exact("decode", expected, actual);
        CHECK(active_grouped_legacy_op_count(candidate_backend.get()) == 0);
        const auto telemetry = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
        fprintf(stderr, "test-moe-cache: routed decode telemetry registered=%llu covered=%llu plan_calls=%llu calls=%llu ready=%llu completed=%llu\n",
            (unsigned long long) telemetry.registered, (unsigned long long) telemetry.covered,
            (unsigned long long) telemetry.plan_calls, (unsigned long long) telemetry.calls,
            (unsigned long long) telemetry.ready, (unsigned long long) telemetry.completed);
        CHECK(telemetry.registered == 1 && telemetry.covered == 1 && telemetry.plan_calls >= 1);
        CHECK(telemetry.calls == 1 && telemetry.ready == 1 && telemetry.completed == 1);
        CHECK(telemetry.fallback == 0 && telemetry.prepare_error == 0 && telemetry.finish_error == 0);
        fprintf(stderr, "test-moe-cache: routed separate decode exact OK\n");
    }

    ggml_backend_cuda_moe_set_debug_mm(old_debug_mm);
}

struct gemma_q4_parity_weights {
    ggml_context_ptr context;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * gate_up = nullptr;
    ggml_tensor * down = nullptr;
};

struct gemma_q4_parity_graphs {
    ggml_context_ptr context;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * gate_input = nullptr;
    ggml_tensor * down_input = nullptr;
    ggml_tensor * ids_storage = nullptr;
    std::array<ggml_tensor *, 3> outputs = {};
    ggml_cgraph * graph = nullptr;
};

struct gemma_q4_routes {
    std::vector<int32_t> padded;
    std::vector<int32_t> unique;
};

static gemma_q4_parity_weights build_gemma_q4_parity_weights(
        ggml_backend_buffer_type_t buft) {
    constexpr int64_t N_EXPERTS = 128;
    constexpr int64_t N_EMBD = 2816;
    constexpr int64_t N_FF = 704;
    const ggml_init_params params = {
        /* .mem_size = */ ggml_tensor_overhead() * 4,
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };

    gemma_q4_parity_weights result;
    result.context.reset(ggml_init(params));
    CHECK(result.context != nullptr);
    const int64_t gate_up_ne[] = {N_EMBD, 2 * N_FF, N_EXPERTS};
    const int64_t down_ne[] = {N_FF, N_EMBD, N_EXPERTS};
    result.gate_up = ggml_new_tensor(result.context.get(), GGML_TYPE_Q4_0, 3, gate_up_ne);
    result.down = ggml_new_tensor(result.context.get(), GGML_TYPE_Q4_0, 3, down_ne);
    ggml_set_name(result.gate_up, "blk.0.ffn_gate_up_exps.weight");
    ggml_set_name(result.down, "blk.0.ffn_down_exps.weight");
    result.buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(result.context.get(), buft));
    CHECK(result.buffer != nullptr);
    ggml_backend_buffer_set_usage(result.buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    CHECK(result.gate_up->nb[2] == 2230272 && result.down->nb[2] == 1115136);
    CHECK(ggml_nbytes(result.gate_up) == 285474816 && ggml_nbytes(result.down) == 142737408);
    return result;
}

static gemma_q4_parity_graphs build_gemma_q4_parity_graphs(
        ggml_backend_t backend,
        ggml_tensor * gate_up,
        ggml_tensor * down,
        int64_t n_rows) {
    constexpr int64_t N_EMBD = 2816;
    constexpr int64_t N_FF = 704;
    constexpr int64_t N_USED = 8;
    constexpr int64_t IDS_STRIDE = 12;
    const ggml_init_params params = {
        /* .mem_size = */ ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false),
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };

    gemma_q4_parity_graphs result;
    result.context.reset(ggml_init(params));
    CHECK(result.context != nullptr);
    result.gate_input = ggml_new_tensor_3d(result.context.get(), GGML_TYPE_F32, N_EMBD, 1, n_rows);
    result.down_input = ggml_new_tensor_3d(result.context.get(), GGML_TYPE_F32, N_FF, N_USED, n_rows);
    result.ids_storage = ggml_new_tensor_2d(result.context.get(), GGML_TYPE_I32, IDS_STRIDE, n_rows);
    ggml_tensor * ids = ggml_view_2d(
        result.context.get(), result.ids_storage, N_USED, n_rows, result.ids_storage->nb[1], 0);
    ggml_set_name(result.gate_input, "test.gemma_q4.gate_input");
    ggml_set_name(result.down_input, "test.gemma_q4.down_input");
    ggml_set_name(result.ids_storage, "test.gemma_q4.ids_storage");
    ggml_set_name(ids, "test.gemma_q4.ids");

    result.outputs[0] = ggml_mul_mat_id(result.context.get(), gate_up, result.gate_input, ids);
    result.outputs[1] = ggml_mul_mat_id(result.context.get(), down, result.down_input, ids);
    ggml_tensor * gate_up_composed = ggml_mul_mat_id(result.context.get(), gate_up, result.gate_input, ids);
    ggml_tensor * gate = ggml_view_3d(
        result.context.get(), gate_up_composed, N_FF, N_USED, n_rows,
        gate_up_composed->nb[1], gate_up_composed->nb[2], 0);
    ggml_tensor * up = ggml_view_3d(
        result.context.get(), gate_up_composed, N_FF, N_USED, n_rows,
        gate_up_composed->nb[1], gate_up_composed->nb[2], N_FF * gate_up_composed->nb[0]);
    ggml_tensor * hidden = ggml_geglu_split(result.context.get(), gate, up);
    result.outputs[2] = ggml_mul_mat_id(result.context.get(), down, hidden, ids);
    ggml_set_name(result.outputs[0], "test.gemma_q4.gate_output");
    ggml_set_name(result.outputs[1], "test.gemma_q4.down_output");
    ggml_set_name(result.outputs[2], "test.gemma_q4.composed_output");

    result.graph = ggml_new_graph_custom(result.context.get(), 64, false);
    for (ggml_tensor * output : result.outputs) {
        ggml_build_forward_expand(result.graph, output);
    }
    result.buffer.reset(ggml_backend_alloc_ctx_tensors(result.context.get(), backend));
    CHECK(result.buffer != nullptr);
    return result;
}

static std::vector<uint8_t> gemma_q4_expert_data(const ggml_tensor * tensor, uint32_t salt) {
    struct q4_0_block {
        ggml_fp16_t d;
        uint8_t qs[16];
    };
    static_assert(sizeof(q4_0_block) == 18);
    constexpr size_t QK4_0_TEST = 32;
    CHECK(tensor != nullptr && tensor->type == GGML_TYPE_Q4_0 && tensor->ne[0] % QK4_0_TEST == 0);
    std::vector<uint8_t> result(ggml_nbytes(tensor), 0);
    const size_t blocks_per_row = tensor->ne[0] / QK4_0_TEST;
    CHECK(tensor->nb[1] == blocks_per_row * sizeof(q4_0_block));
    for (int64_t expert = 0; expert < tensor->ne[2]; ++expert) {
        for (int64_t row = 0; row < tensor->ne[1]; ++row) {
            for (size_t block_index = 0; block_index < blocks_per_row; ++block_index) {
                uint32_t state = salt ^ (uint32_t) expert * 0x9e3779b9u ^ (uint32_t) row * 0x85ebca6bu ^
                    (uint32_t) block_index * 0xc2b2ae35u;
                q4_0_block block;
                block.d = ggml_fp32_to_fp16(0.0005f * (1 + (state % 29)));
                for (size_t byte = 0; byte < sizeof(block.qs); ++byte) {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    block.qs[byte] = (uint8_t) state;
                }
                const size_t offset = (size_t) expert * tensor->nb[2] + (size_t) row * tensor->nb[1] +
                    block_index * sizeof(q4_0_block);
                memcpy(result.data() + offset, &block, sizeof(block));
            }
        }
    }
    return result;
}

static std::vector<float> gemma_q4_input_data(const ggml_tensor * tensor, uint32_t salt) {
    CHECK(tensor != nullptr && tensor->type == GGML_TYPE_F32);
    std::vector<float> result(ggml_nelements(tensor));
    uint32_t state = salt;
    for (float & value : result) {
        state = state * 1664525u + 1013904223u;
        value = ((int32_t) (state % 2001) - 1000) * 0.0005f;
    }
    return result;
}

static gemma_q4_routes gemma_q4_route_data(int64_t n_rows, uint32_t n_unique) {
    constexpr uint32_t N_EXPERTS = 128;
    constexpr uint32_t N_USED = 8;
    constexpr uint32_t IDS_STRIDE = 12;
    CHECK(n_rows > 0 && n_unique >= N_USED && n_unique <= N_EXPERTS);
    std::vector<int32_t> experts = {127, 0};
    for (uint32_t index = 0; experts.size() < n_unique; ++index) {
        const int32_t expert = (int32_t) ((37 * index + 11) % N_EXPERTS);
        if (std::find(experts.begin(), experts.end(), expert) == experts.end()) {
            experts.push_back(expert);
        }
    }

    gemma_q4_routes result;
    result.padded.assign((size_t) IDS_STRIDE * n_rows, -7777777);
    std::array<bool, N_EXPERTS> seen = {};
    for (int64_t row = 0; row < n_rows; ++row) {
        for (uint32_t route = 0; route < N_USED; ++route) {
            const size_t position = (size_t) row * N_USED + route;
            const size_t expert_index = position < experts.size() ? position : (position * 5 + (size_t) row * 3) % experts.size();
            const int32_t expert = experts[expert_index];
            result.padded[(size_t) row * IDS_STRIDE + route] = expert;
            if (!seen[expert]) {
                seen[expert] = true;
                result.unique.push_back(expert);
            }
        }
    }
    CHECK(std::find(result.unique.begin(), result.unique.end(), 0) != result.unique.end());
    CHECK(std::find(result.unique.begin(), result.unique.end(), 127) != result.unique.end());
    CHECK(result.unique.size() == n_unique);
    return result;
}

static void check_gemma_q4_slab(
        const std::vector<uint8_t> & expected,
        size_t expert_stride,
        const void * actual_device,
        int32_t expert,
        const char * boundary) {
    CHECK((size_t) expert < expected.size() / expert_stride);
    std::vector<uint8_t> actual_bytes(expert_stride);
    CUDA_OK(cudaMemcpy(actual_bytes.data(), actual_device, actual_bytes.size(), cudaMemcpyDeviceToHost));
    const uint8_t * expected_bytes = expected.data() + (size_t) expert * expert_stride;
    if (memcmp(expected_bytes, actual_bytes.data(), expert_stride) != 0) {
        fprintf(stderr, "test-moe-cache: first Gemma Q4_0 byte mismatch at %s expert=%d\n", boundary, expert);
    }
    CHECK(memcmp(expected_bytes, actual_bytes.data(), expert_stride) == 0);
}

static void check_gemma_q4_materialization(
        int device,
        uint32_t n_slots,
        const gemma_q4_parity_weights & cached,
        const std::array<const std::vector<uint8_t> *, 2> & expected,
        const std::vector<int32_t> & unique) {
    CHECK((n_slots == 12 && unique.size() > n_slots) || (n_slots == 48 && unique.size() <= n_slots));
    cudaStream_t stream = nullptr;
    CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const std::array<ggml_tensor *, 2> banks = {cached.gate_up, cached.down};
    const std::array<size_t, 3> representative = {0, unique.size() / 2, unique.size() - 1};
    for (size_t bank_index = 0; bank_index < banks.size(); ++bank_index) {
        ggml_tensor * bank = banks[bank_index];
        const auto & bytes = *expected[bank_index];
        CHECK(bytes.size() == ggml_nbytes(bank));
        for (size_t index : representative) {
            const int32_t expert = unique[index];
            CHECK(memcmp(
                static_cast<const char *>(bank->data) + (size_t) expert * bank->nb[2],
                bytes.data() + (size_t) expert * bank->nb[2], bank->nb[2]) == 0);
        }

        std::unique_ptr<ggml_cuda_moe_cache, decltype(&ggml_cuda_moe_cache_free)> cache(
            ggml_cuda_moe_cache_init(device, bank->nb[2], n_slots),
            ggml_cuda_moe_cache_free);
        CHECK(cache != nullptr);
        const char * source = static_cast<const char *>(bank->data);
        cudaStream_t copy_stream = ggml_cuda_moe_cache_copy_stream(cache.get());
        std::vector<int> slots(unique.size(), -1);
        void * misses = nullptr;
        if (n_slots == 48) {
            size_t index = 0;
            for (int32_t expert : unique) {
                const int slot = ggml_cuda_moe_cache_acquire(
                    cache.get(), source + (size_t) expert * bank->nb[2], bank->nb[2],
                    copy_stream, false, false, true);
                CHECK(slot >= 0);
                slots[index++] = slot;
            }
            CUDA_OK(cudaStreamSynchronize(copy_stream));
        } else {
            std::vector<int> primed_slots;
            for (size_t index = 0; index < 4; ++index) {
                const int slot = ggml_cuda_moe_cache_acquire(
                    cache.get(), source + (size_t) unique[index] * bank->nb[2], bank->nb[2],
                    copy_stream, false, false, true);
                CHECK(slot >= 0);
                primed_slots.push_back(slot);
            }
            CUDA_OK(cudaStreamSynchronize(copy_stream));
            ggml_cuda_moe_cache_release_slots(cache.get(), primed_slots.data(), primed_slots.size());

            std::vector<const void *> sources;
            for (int32_t expert : unique) {
                sources.push_back(source + (size_t) expert * bank->nb[2]);
            }
            CUDA_OK(cudaMalloc(&misses, (unique.size() - n_slots) * bank->nb[2]));
            int n_resident = 0;
            int n_wait_classes = 0;
            CHECK(ggml_cuda_moe_cache_prepare_split_staging(
                cache.get(), sources.data(), sources.size(), bank->nb[2], 0, 2,
                slots.data(), nullptr, &n_resident, misses, nullptr, 0, &n_wait_classes, stream, false));
            CHECK(n_resident == (int) n_slots && n_wait_classes == 1);
            CHECK(ggml_cuda_moe_cache_finish_split_staging(cache.get(), stream));
            CUDA_OK(cudaStreamSynchronize(stream));
        }

        std::vector<std::pair<size_t, const void *>> resident;
        std::vector<std::pair<size_t, const void *>> overflow;
        size_t miss_index = 0;
        for (size_t index = 0; index < unique.size(); ++index) {
            if (slots[index] >= 0) {
                resident.push_back({index, ggml_cuda_moe_cache_slot_ptr(cache.get(), slots[index])});
            } else {
                overflow.push_back({index, static_cast<const char *>(misses) + miss_index++ * bank->nb[2]});
            }
        }
        CHECK(resident.size() == std::min<size_t>(n_slots, unique.size()));
        CHECK(resident.size() + overflow.size() == unique.size());
        for (const auto & materialized : {&resident, &overflow}) {
            if (materialized->empty()) {
                continue;
            }
            for (size_t position : std::array<size_t, 3>{0, materialized->size() / 2, materialized->size() - 1}) {
                const auto [index, data] = (*materialized)[position];
                check_gemma_q4_slab(bytes, bank->nb[2], data, unique[index],
                    slots[index] >= 0 ? "resident slot" : "split overflow staging");
            }
        }
        if (n_slots == 48) {
            ggml_cuda_moe_cache_release_slots(cache.get(), slots.data(), slots.size());
        } else {
            CHECK(ggml_cuda_moe_cache_release_split_slots(cache.get(), slots.data(), slots.size(), stream));
            CUDA_OK(cudaStreamSynchronize(stream));
            CUDA_OK(cudaFree(misses));
        }
    }
    CUDA_OK(cudaStreamDestroy(stream));
}

struct direct_source_view_test_state {
    bool descriptor_rejected = false;
    bool quant_mmvq_rejected = false;
    bool bf16_mmq_rejected = false;
    bool generic_rejected = false;
};

static void test_mmid_direct_source_view_case(
        int device,
        ggml_type type,
        int64_t n_tokens,
        ggml_cuda_mmid_consumer expected_consumer,
        direct_source_view_test_state * state) {
    constexpr int64_t logical_experts = 8;
    constexpr int64_t physical_experts = 24;
    constexpr int64_t n_used = 2;
    constexpr int64_t n_out = 256;
    constexpr int64_t physical_begin = physical_experts - logical_experts;
    const std::array<const char *, 3> role_names = {
        "test.direct_view.ffn_gate_exps.weight",
        "test.direct_view.ffn_up_exps.weight",
        "test.direct_view.ffn_down_exps.weight",
    };

    ggml_backend_ptr baseline_backend(ggml_backend_cuda_init(device));
    ggml_backend_ptr physical_backend(ggml_backend_cuda_init(device));
    CHECK(baseline_backend != nullptr && physical_backend != nullptr);
    std::array<cached_mmid_path_test_graph, 3> baseline;
    std::array<cached_mmid_path_test_graph, 3> physical;
    for (size_t role = 0; role < role_names.size(); ++role) {
        baseline[role] = build_cached_mmid_path_test_graph(
            baseline_backend.get(), ggml_backend_cuda_buffer_type(device), type, n_out, n_used, n_tokens, logical_experts);
        physical[role] = build_cached_mmid_path_test_graph(
            physical_backend.get(), ggml_backend_cuda_buffer_type(device), type, n_out, n_used, n_tokens, physical_experts);
        ggml_set_name(baseline[role].leaves[0], role_names[role]);
        ggml_set_name(physical[role].leaves[0], role_names[role]);
    }

    std::vector<int32_t> baseline_ids(n_tokens * n_used);
    std::vector<int32_t> physical_ids(n_tokens * n_used);
    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t route = 0; route < n_used; ++route) {
            const int32_t expert = (3 * token + 5 * route) % logical_experts;
            baseline_ids[token * n_used + route] = expert;
            physical_ids[token * n_used + route] = physical_begin + expert;
        }
    }

    std::array<std::vector<float>, 3> expected;
    std::array<std::vector<float>, 3> actual;
    for (size_t role = 0; role < role_names.size(); ++role) {
        ggml_tensor * baseline_weight = baseline[role].leaves[0];
        ggml_tensor * physical_weight = physical[role].leaves[0];
        CHECK(baseline_weight->nb[2] == physical_weight->nb[2] && baseline_weight->ne[2] == logical_experts &&
            physical_weight->ne[2] == physical_experts);
        auto baseline_bytes = cached_fusion_test_data(baseline_weight, 1201 + role);
        auto physical_bytes = cached_fusion_test_data(physical_weight, 1301 + role);
        for (int64_t expert = 0; expert < logical_experts; ++expert) {
            const size_t source_offset = expert * baseline_weight->nb[2];
            const size_t physical_offset = (physical_begin + expert) * physical_weight->nb[2];
            memcpy(physical_bytes.data() + physical_offset, baseline_bytes.data() + source_offset, baseline_weight->nb[2]);
            CHECK(memcmp(
                physical_bytes.data() + physical_offset,
                baseline_bytes.data() + source_offset,
                baseline_weight->nb[2]) == 0);
        }
        ggml_backend_tensor_set(baseline_weight, baseline_bytes.data(), 0, baseline_bytes.size());
        ggml_backend_tensor_set(physical_weight, physical_bytes.data(), 0, physical_bytes.size());

        const auto input = cached_fusion_test_data(baseline[role].leaves[1], 1401 + role);
        CHECK(input.size() == ggml_nbytes(physical[role].leaves[1]));
        ggml_backend_tensor_set(baseline[role].leaves[1], input.data(), 0, input.size());
        ggml_backend_tensor_set(physical[role].leaves[1], input.data(), 0, input.size());
        ggml_backend_tensor_set(baseline[role].ids, baseline_ids.data(), 0, ggml_nbytes(baseline[role].ids));
        ggml_backend_tensor_set(physical[role].ids, physical_ids.data(), 0, ggml_nbytes(physical[role].ids));

        const auto baseline_capability = native_mmid_capability(
            device, baseline_weight, n_tokens, GGML_CUDA_MMID_MAPPING_DIRECT);
        ggml_tensor logical_physical_weight = *physical_weight;
        logical_physical_weight.ne[2] = logical_experts;
        logical_physical_weight.nb[3] = logical_physical_weight.nb[2] * logical_experts;
        const auto logical_physical_capability = native_mmid_capability(
            device, &logical_physical_weight, n_tokens, GGML_CUDA_MMID_MAPPING_DIRECT);
        CHECK(baseline_capability.reason == GGML_CUDA_MMID_CAPABILITY_OK &&
            baseline_capability.selection == expected_consumer &&
            logical_physical_capability.reason == baseline_capability.reason &&
            logical_physical_capability.selection == baseline_capability.selection);

        const ggml_cuda_mmid_direct_source_view view = {
            physical_weight->type,
            logical_experts,
            physical_experts,
            physical_experts,
            physical_weight->nb[2],
        };
        CHECK(ggml_cuda_mmid_direct_source_view_valid(physical_weight, view, expected_consumer));

        const std::vector<float> baseline_sentinel(ggml_nelements(baseline[role].output), -17001.25f);
        const std::vector<float> physical_sentinel(ggml_nelements(physical[role].output), -18001.5f);
        ggml_backend_tensor_set(
            baseline[role].output, baseline_sentinel.data(), 0, ggml_nbytes(baseline[role].output));
        ggml_backend_tensor_set(
            physical[role].output, physical_sentinel.data(), 0, ggml_nbytes(physical[role].output));
        CHECK(ggml_backend_graph_compute(baseline_backend.get(), baseline[role].graph) == GGML_STATUS_SUCCESS);
        CHECK(ggml_cuda_mmid_direct_source_view_compute_for_test(
            physical_backend.get(), physical[role].output, &view, expected_consumer));
        ggml_backend_synchronize(baseline_backend.get());
        ggml_backend_synchronize(physical_backend.get());
        expected[role] = active_grouped_tensor_values(baseline[role].output);
        actual[role] = active_grouped_tensor_values(physical[role].output);
        CHECK(expected[role] != baseline_sentinel && actual[role] != physical_sentinel &&
            expected[role].size() == actual[role].size());
        double squared_expected = 0.0;
        for (size_t value = 0; value < expected[role].size(); ++value) {
            CHECK(std::isfinite(expected[role][value]) && std::isfinite(actual[role][value]));
            squared_expected += static_cast<double>(expected[role][value]) * expected[role][value];
        }
        CHECK(squared_expected > 0.0 && memcmp(
            expected[role].data(), actual[role].data(), expected[role].size() * sizeof(float)) == 0);

        const auto reject = [&](ggml_cuda_mmid_direct_source_view rejected, ggml_cuda_mmid_consumer consumer) {
            ggml_backend_tensor_set(
                physical[role].output, physical_sentinel.data(), 0, ggml_nbytes(physical[role].output));
            CHECK(!ggml_cuda_mmid_direct_source_view_compute_for_test(
                physical_backend.get(), physical[role].output, &rejected, consumer));
            ggml_backend_synchronize(physical_backend.get());
            CHECK(active_grouped_tensor_values(physical[role].output) == physical_sentinel);
        };
        if (!state->descriptor_rejected) {
            auto rejected = view;
            rejected.logical_n_experts = 0;
            reject(rejected, expected_consumer);
            rejected = view;
            rejected.logical_n_experts = static_cast<int64_t>(INT_MAX) + 1;
            reject(rejected, expected_consumer);
            rejected = view;
            rejected.expert_stride += ggml_type_size(type);
            reject(rejected, expected_consumer);
            rejected = view;
            rejected.source_type = GGML_TYPE_COUNT;
            reject(rejected, expected_consumer);
            rejected = view;
            rejected.physical_n_experts -= 1;
            reject(rejected, expected_consumer);
            rejected = view;
            // The grouped remapper certifies this exclusive device-ID bound.
            rejected.physical_id_upper_bound = physical_experts + 1;
            reject(rejected, expected_consumer);
            state->descriptor_rejected = true;
        }
        if (!state->quant_mmvq_rejected && ggml_is_quantized(type) && n_tokens == 16 &&
                expected_consumer == GGML_CUDA_MMID_CONSUMER_MMQ) {
            CHECK(ggml_cuda_mmid_direct_source_view_valid(
                physical_weight, view, GGML_CUDA_MMID_CONSUMER_MMVQ));
            reject(view, GGML_CUDA_MMID_CONSUMER_MMVQ);
            state->quant_mmvq_rejected = true;
        }
        if (!state->bf16_mmq_rejected && type == GGML_TYPE_BF16 && expected_consumer == GGML_CUDA_MMID_CONSUMER_MMF) {
            reject(view, GGML_CUDA_MMID_CONSUMER_MMQ);
            state->bf16_mmq_rejected = true;
        }
        if (!state->generic_rejected) {
            reject(view, GGML_CUDA_MMID_CONSUMER_GENERIC);
            state->generic_rejected = true;
        }
    }

    CHECK(memcmp(expected[2].data(), actual[2].data(), expected[2].size() * sizeof(float)) == 0);
}

void test_mmid_direct_source_view(int device) {
    struct test_case {
        ggml_type type;
        int64_t n_tokens;
        ggml_cuda_mmid_consumer consumer;
    };
    const std::array<test_case, 10> cases = {{
        {GGML_TYPE_Q4_0, 4, GGML_CUDA_MMID_CONSUMER_MMVQ},
        {GGML_TYPE_Q4_0, 16, GGML_CUDA_MMID_CONSUMER_MMQ},
        {GGML_TYPE_Q4_K, 4, GGML_CUDA_MMID_CONSUMER_MMVQ},
        {GGML_TYPE_Q4_K, 16, GGML_CUDA_MMID_CONSUMER_MMQ},
        {GGML_TYPE_NVFP4, 4, GGML_CUDA_MMID_CONSUMER_MMVQ},
        {GGML_TYPE_NVFP4, 16, GGML_CUDA_MMID_CONSUMER_MMQ},
        {GGML_TYPE_MXFP4, 4, GGML_CUDA_MMID_CONSUMER_MMVQ},
        {GGML_TYPE_MXFP4, 16, GGML_CUDA_MMID_CONSUMER_MMQ},
        {GGML_TYPE_BF16, 4, GGML_CUDA_MMID_CONSUMER_MMF},
        {GGML_TYPE_BF16, 16, GGML_CUDA_MMID_CONSUMER_MMF},
    }};
    cudaDeviceProp properties;
    CUDA_OK(cudaGetDeviceProperties(&properties, device));
    const bool require_cc12 = properties.major == 12;
    direct_source_view_test_state state;
    for (const auto & current : cases) {
        ggml_backend_ptr query_backend(ggml_backend_cuda_init(device));
        CHECK(query_backend != nullptr);
        auto query_graph = build_cached_mmid_path_test_graph(
            query_backend.get(), ggml_backend_cuda_buffer_type(device), current.type, 256, 2, current.n_tokens, 8);
        const auto capability = native_mmid_capability(
            device, query_graph.leaves[0], current.n_tokens, GGML_CUDA_MMID_MAPPING_DIRECT);
        const bool supported = capability.reason == GGML_CUDA_MMID_CAPABILITY_OK &&
            capability.selection == current.consumer;
        if (require_cc12) {
            CHECK(supported);
        }
        if (!supported) {
            fprintf(stderr, "test-moe-cache: skipping direct MMID view type=%s B%lld consumer=%u on CC%d%d\n",
                ggml_type_name(current.type), (long long) current.n_tokens, static_cast<unsigned>(current.consumer),
                properties.major, properties.minor);
            continue;
        }
        test_mmid_direct_source_view_case(
            device, current.type, current.n_tokens, current.consumer, &state);
    }
    if (require_cc12) {
        CHECK(state.descriptor_rejected && state.quant_mmvq_rejected && state.bf16_mmq_rejected && state.generic_rejected);
    }
    fprintf(stderr, "test-moe-cache: direct MMID logical/physical source extent witness OK\n");
}

static std::array<std::vector<float>, 3> run_gemma_q4_graph(
        ggml_backend_t backend,
        gemma_q4_parity_graphs & graphs) {
    CHECK(ggml_backend_graph_compute(backend, graphs.graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
    std::array<std::vector<float>, 3> result;
    for (size_t index = 0; index < result.size(); ++index) {
        result[index].resize(ggml_nelements(graphs.outputs[index]));
        ggml_backend_tensor_get(graphs.outputs[index], result[index].data(), 0, ggml_nbytes(graphs.outputs[index]));
    }
    return result;
}

static void compare_gemma_q4_output(
        const char * boundary,
        const std::vector<float> & expected,
        const std::vector<float> & actual) {
    CHECK(expected.size() == actual.size());
    size_t first_mismatch = expected.size();
    for (size_t index = 0; index < expected.size(); ++index) {
        CHECK(std::isfinite(expected[index]) && std::isfinite(actual[index]));
        if (first_mismatch == expected.size() && expected[index] != actual[index]) {
            first_mismatch = index;
        }
    }
    if (first_mismatch != expected.size()) {
        fprintf(stderr, "test-moe-cache: first Gemma Q4_0 output mismatch at %s index=%zu expected=%.9g actual=%.9g\n",
            boundary, first_mismatch, expected[first_mismatch], actual[first_mismatch]);
    }
    CHECK(first_mismatch == expected.size());
}

static void run_gemma_q4_parity_case(
        ggml_backend_t direct_backend,
        const gemma_q4_parity_weights & direct_weights,
        ggml_backend_t cached_backend,
        const gemma_q4_parity_weights & cached_weights,
        const char * boundary,
        int64_t n_rows,
        uint32_t route_union) {
    const auto routes = gemma_q4_route_data(n_rows, route_union);
    auto direct_graphs = build_gemma_q4_parity_graphs(
        direct_backend, direct_weights.gate_up, direct_weights.down, n_rows);
    auto cached_graphs = build_gemma_q4_parity_graphs(
        cached_backend, cached_weights.gate_up, cached_weights.down, n_rows);
    const auto gate_input = gemma_q4_input_data(direct_graphs.gate_input, 0x1451u);
    const auto down_input = gemma_q4_input_data(direct_graphs.down_input, 0x8d37u);
    for (gemma_q4_parity_graphs * graph : {&direct_graphs, &cached_graphs}) {
        ggml_backend_tensor_set(graph->gate_input, gate_input.data(), 0, ggml_nbytes(graph->gate_input));
        ggml_backend_tensor_set(graph->down_input, down_input.data(), 0, ggml_nbytes(graph->down_input));
        CHECK(ggml_nelements(graph->ids_storage) == (int64_t) routes.padded.size());
        ggml_backend_tensor_set(graph->ids_storage, routes.padded.data(), 0, ggml_nbytes(graph->ids_storage));
    }
    const auto expected = run_gemma_q4_graph(direct_backend, direct_graphs);
    const auto actual = run_gemma_q4_graph(cached_backend, cached_graphs);
    const std::array<const char *, 3> outputs = {"gate_up", "down", "GEGLU/down"};
    for (size_t index = 0; index < outputs.size(); ++index) {
        const std::string name = std::string(boundary) + " " + outputs[index];
        compare_gemma_q4_output(name.c_str(), expected[index], actual[index]);
    }
}

void test_gemma_q4_cached_cuda_parity(int device) {
    ggml_backend_ptr direct_backend(ggml_backend_cuda_init(device));
    CHECK(direct_backend != nullptr);
    auto direct_weights = build_gemma_q4_parity_weights(ggml_backend_cuda_buffer_type(device));
    const auto gate_up_data = gemma_q4_expert_data(direct_weights.gate_up, 0x31a5u);
    const auto down_data = gemma_q4_expert_data(direct_weights.down, 0x7c29u);
    ggml_backend_tensor_set(direct_weights.gate_up, gate_up_data.data(), 0, gate_up_data.size());
    ggml_backend_tensor_set(direct_weights.down, down_data.data(), 0, down_data.size());
    ggml_backend_synchronize(direct_backend.get());

    const auto b16_direct = native_mmid_capability(
        device, direct_weights.gate_up, 16, GGML_CUDA_MMID_MAPPING_DIRECT);
    const auto b16_mapped = native_mmid_capability(
        device, direct_weights.gate_up, 16, GGML_CUDA_MMID_MAPPING_SOURCE_MAP);
    CHECK(b16_direct.reason == GGML_CUDA_MMID_CAPABILITY_OK &&
        b16_direct.selection == GGML_CUDA_MMID_CONSUMER_MMQ &&
        b16_mapped.reason == GGML_CUDA_MMID_CAPABILITY_OK &&
        b16_mapped.selection == GGML_CUDA_MMID_CONSUMER_MMQ);
    const auto b4_direct = native_mmid_capability(
        device, direct_weights.gate_up, 4, GGML_CUDA_MMID_MAPPING_DIRECT);
    const auto b4_mapped = native_mmid_capability(
        device, direct_weights.gate_up, 4, GGML_CUDA_MMID_MAPPING_SOURCE_MAP);
    CHECK(b4_direct.reason == GGML_CUDA_MMID_CAPABILITY_OK &&
        b4_direct.selection == GGML_CUDA_MMID_CONSUMER_MMVQ &&
        b4_mapped.reason == GGML_CUDA_MMID_CAPABILITY_OK &&
        b4_mapped.selection == GGML_CUDA_MMID_CONSUMER_MMQ);
    const auto b112_direct = native_mmid_capability(
        device, direct_weights.gate_up, 112, GGML_CUDA_MMID_MAPPING_DIRECT);
    const auto b112_mapped = native_mmid_capability(
        device, direct_weights.gate_up, 112, GGML_CUDA_MMID_MAPPING_SOURCE_MAP);
    CHECK(b112_direct.reason == GGML_CUDA_MMID_CAPABILITY_OK &&
        b112_direct.selection == GGML_CUDA_MMID_CONSUMER_MMQ &&
        b112_mapped.reason == GGML_CUDA_MMID_CAPABILITY_OK &&
        b112_mapped.selection == GGML_CUDA_MMID_CONSUMER_MMQ);

    for (uint32_t n_slots : {12u, 48u}) {
        ggml_backend_ptr cached_backend(ggml_backend_cuda_init(device));
        CHECK(cached_backend != nullptr);
        const auto disabled = candidate_snapshot(n_slots, nullptr, 0);
        CHECK(ggml_backend_cuda_moe_candidate_replace_v1(cached_backend.get(), &disabled) ==
            GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        auto cached_weights = build_gemma_q4_parity_weights(ggml_backend_cuda_moe_cached_buffer_type());
        ggml_backend_tensor_set(cached_weights.gate_up, gate_up_data.data(), 0, gate_up_data.size());
        ggml_backend_tensor_set(cached_weights.down, down_data.data(), 0, down_data.size());

        const uint32_t b16_union = n_slots == 12 ? 24 : 40;
        const auto materialization_routes = gemma_q4_route_data(16, b16_union);
        check_gemma_q4_materialization(
            device, n_slots, cached_weights, {&gate_up_data, &down_data}, materialization_routes.unique);
        run_gemma_q4_parity_case(
            direct_backend.get(), direct_weights, cached_backend.get(), cached_weights,
            n_slots == 12 ? "cache12 B16" : "cache48 B16", 16, b16_union);
        run_gemma_q4_parity_case(
            direct_backend.get(), direct_weights, cached_backend.get(), cached_weights,
            n_slots == 12 ? "cache12 B4" : "cache48 B4", 4, 24);
        if (n_slots == 48) {
            run_gemma_q4_parity_case(
                direct_backend.get(), direct_weights, cached_backend.get(), cached_weights,
                "cache48 B112 sparse", 112, 96);
            run_gemma_q4_parity_case(
                direct_backend.get(), direct_weights, cached_backend.get(), cached_weights,
                "cache48 B136 sparse fixup", 136, 96);
        }
    }
    fprintf(stderr, "test-moe-cache: opt-in Gemma Q4_0 cached CUDA parity diagnostic OK\n");
}
