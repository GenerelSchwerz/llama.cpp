#include "test-moe-cache.h"

#ifdef __linux__
#include <sys/mman.h>
#endif

static void check_active_grouped_bias_shadows(
        ggml_backend_t backend,
        const active_grouped_dispatch_graph & graph,
        uint32_t expected_resident = 0) {
    auto * context = ggml_cuda_moe_grouped_context_for_test(backend);
    ggml_cuda_moe_candidate_group_key key;
    CHECK(context != nullptr && context->find_down_group_key(graph.down, &key));
    std::vector<const float *> shadows;
    std::vector<std::vector<float>> originals;
    for (ggml_tensor * bias : graph.biases) {
        shadows.push_back(ggml_cuda_moe_grouped_context_test_access::device_auxiliary_data(*context, key, bias));
        CHECK(shadows.back() != nullptr && bias->type == GGML_TYPE_F32 && ggml_n_dims(bias) == 2);
        originals.emplace_back(ggml_nelements(bias));
        ggml_backend_tensor_get(bias, originals.back().data(), 0, ggml_nbytes(bias));
    }
    std::vector<bool> resident_slots(context->state().n_slots, false);
    uint32_t n_resident = 0;
    for (uint32_t expert = 0; expert < graph.n_experts; ++expert) {
        const int32_t slot = ggml_cuda_moe_grouped_context_test_access::device_slot_for_expert(*context, key, expert);
        if (slot < 0) {
            continue;
        }
        CHECK(static_cast<uint32_t>(slot) < resident_slots.size() && !resident_slots[slot]);
        resident_slots[slot] = true;
        ++n_resident;
        for (uint32_t bias_index = 0; bias_index < graph.biases.size(); ++bias_index) {
            ggml_tensor * bias = graph.biases[bias_index];
            std::vector<float> actual(bias->ne[0]);
            CUDA_OK(cudaMemcpy(actual.data(), shadows[bias_index] + slot * bias->ne[0],
                actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK(memcmp(actual.data(), originals[bias_index].data() + expert * bias->ne[0],
                actual.size() * sizeof(float)) == 0);
        }
    }
    if (expected_resident != 0) {
        CHECK(n_resident == expected_resident);
    }
}

static void test_active_grouped_dispatch_types_case(
        const std::array<ggml_type, 3> & types,
        uint32_t layout,
        uint32_t n_slots,
        bool original_direct_down_scale = false,
        bool test_owner_concurrency = false,
        bool original_direct_biases = false,
        uint32_t n_rows = 1,
        uint32_t n_ff = 0,
        bool auxiliary_proof = false) {
    CHECK(!auxiliary_proof || (layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE && n_slots == 3 &&
        test_owner_concurrency && original_direct_biases && n_rows == 1 && n_ff != 0));
    const bool old_debug_mm = ggml_backend_cuda_moe_get_debug_mm();
    ggml_backend_cuda_moe_set_debug_mm(true);
    ggml_backend_ptr reference_backend(ggml_backend_cuda_init(0));
    ggml_backend_ptr first_backend(ggml_backend_cuda_init(0));
    ggml_backend_ptr second_backend(ggml_backend_cuda_init(0));
    CHECK(reference_backend != nullptr && first_backend != nullptr && second_backend != nullptr);
    auto reference = build_active_grouped_dispatch_graph_types(
        reference_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), types, layout, original_direct_down_scale,
        false, n_rows, 8, 2, 256, nullptr, false, original_direct_biases, n_ff, auxiliary_proof);
    auto first = build_active_grouped_dispatch_graph_types(
        first_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), types, layout, original_direct_down_scale,
        false, n_rows, 8, 2, 256, nullptr, false, original_direct_biases, n_ff, auxiliary_proof);
    auto second = build_active_grouped_dispatch_graph_types(
        second_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), types, layout, original_direct_down_scale,
        false, n_rows, 8, 2, 256, nullptr, false, original_direct_biases, n_ff, auxiliary_proof);
    initialize_active_grouped_dispatch_graphs({&reference, &first, &second});
    const auto disabled = candidate_snapshot(n_slots, nullptr, 0);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v1(reference_backend.get(), &disabled) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    register_active_grouped_dispatch(first_backend.get(), first, layout, n_slots, original_direct_down_scale);
    register_active_grouped_dispatch(second_backend.get(), second, layout, n_slots);
    if (original_direct_down_scale) {
        auto * first_context = ggml_cuda_moe_grouped_context_for_test(first_backend.get());
        CHECK(first_context != nullptr);
        const uint64_t reordered_signature = first_context->state().logical_signature;
        register_active_grouped_dispatch(first_backend.get(), first, layout, n_slots);
        CHECK(first_context->state().logical_signature == reordered_signature);
        register_active_grouped_dispatch(first_backend.get(), first, layout, n_slots, true);
        CHECK(first_context->state().logical_signature == reordered_signature);
        for (auto * backend : {first_backend.get(), second_backend.get()}) {
            auto * context = ggml_cuda_moe_grouped_context_for_test(backend);
            ggml_cuda_moe_candidate_group_key key;
            ggml_cuda_moe_candidate_group_info group_info;
            ggml_cuda_moe_candidate_bank_info scale_info;
            const auto & graph = backend == first_backend.get() ? first : second;
            CHECK(context != nullptr && context->find_down_group_key(graph.down, &key));
            CHECK(context->get_group(key, &group_info) && group_info.n_banks == 3);
            CHECK(context->get_bank(key, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_SCALE, &scale_info));
            CHECK(scale_info.tensor == graph.down_scale && scale_info.type == GGML_TYPE_F32);
            CHECK(scale_info.movement == GGML_CUDA_MOE_CANDIDATE_MOVEMENT_PERMANENT_CANDIDATE);
            CHECK(scale_info.index_modes == GGML_CUDA_MOE_CANDIDATE_INDEX_ORIGINAL_DIRECT);
        }
    }
    if (original_direct_biases) {
        for (auto * backend : {first_backend.get(), second_backend.get()}) {
            auto * context = ggml_cuda_moe_grouped_context_for_test(backend);
            const auto & graph = backend == first_backend.get() ? first : second;
            ggml_cuda_moe_candidate_group_key key;
            ggml_cuda_moe_candidate_group_info group_info;
            CHECK(context != nullptr && context->find_down_group_key(graph.down, &key));
            CHECK(context->get_group(key, &group_info) &&
                group_info.n_banks == graph.banks.size() + graph.biases.size());
            for (uint32_t i = 0; i < graph.biases.size(); ++i) {
                ggml_cuda_moe_candidate_bank_info bias_info;
                CHECK(context->get_bank(key, graph.bias_roles[i], &bias_info));
                CHECK(bias_info.tensor == graph.biases[i] && bias_info.type == GGML_TYPE_F32);
                CHECK(bias_info.movement == GGML_CUDA_MOE_CANDIDATE_MOVEMENT_PERMANENT_CANDIDATE);
                CHECK(bias_info.index_modes == GGML_CUDA_MOE_CANDIDATE_INDEX_ORIGINAL_DIRECT);
            }
        }
    }
    if (auxiliary_proof) {
        CHECK(first.biases.size() == 3 && second.biases.size() == first.biases.size());
        CHECK(first.biases[0]->ne[0] == n_ff && first.biases[1]->ne[0] == n_ff && first.biases[2]->ne[0] == 256);
        for (const auto * graph : {&reference, &first, &second}) {
            for (ggml_tensor * bias : graph->biases) {
                CHECK(ggml_backend_buft_is_cuda_moe_cached(ggml_backend_buffer_get_type(bias->buffer)));
                cudaPointerAttributes attributes = {};
                CUDA_OK(cudaPointerGetAttributes(&attributes, bias->data));
                CHECK(attributes.type == cudaMemoryTypeHost);
            }
        }
    }
    check_active_grouped_contract(first_backend.get(), first, n_slots, original_direct_down_scale);
    check_active_grouped_contract(second_backend.get(), second, n_slots);

    std::vector<float> expected;
    std::vector<float> reference_output;
    std::vector<float> first_output;
    std::vector<float> second_output;
    const auto gate_capability = native_mmid_capability(0, first.banks[0], n_rows, GGML_CUDA_MMID_MAPPING_DIRECT);
    const bool f3_skipped = layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE &&
        gate_capability.reason == GGML_CUDA_MMID_CAPABILITY_OK &&
        gate_capability.selection == GGML_CUDA_MMID_CONSUMER_MMVQ &&
        first.banks[0]->type == first.banks[1]->type && ggml_are_same_shape(first.banks[0], first.banks[1]) &&
        ggml_are_same_stride(first.banks[0], first.banks[1]);
    for (int pass = 0; pass < 4; ++pass) {
        if (auxiliary_proof) {
            set_active_grouped_dispatch_logits({&reference, &first, &second}, pass % 3);
        }
        expected = run_active_grouped_dispatch(reference_backend.get(), reference, 0, false);
        const uint64_t expected_clock = auxiliary_proof ? 4 * (pass + 1) : 2 * n_rows * (pass + 2);
        auto current_first = run_active_grouped_dispatch(
            first_backend.get(), first, expected_clock, f3_skipped, original_direct_biases && n_rows == 1);
        auto current_second = run_active_grouped_dispatch(
            second_backend.get(), second, expected_clock, f3_skipped, original_direct_biases && n_rows == 1);
        CHECK(current_first.size() == expected.size() && current_second.size() == expected.size());
        CHECK(memcmp(current_first.data(), expected.data(), expected.size() * sizeof(float)) == 0);
        CHECK(memcmp(current_second.data(), expected.data(), expected.size() * sizeof(float)) == 0);
        if (auxiliary_proof) {
            current_first = run_active_grouped_dispatch(
                first_backend.get(), first, expected_clock + 2, f3_skipped, true);
            current_second = run_active_grouped_dispatch(
                second_backend.get(), second, expected_clock + 2, f3_skipped, true);
            CHECK(memcmp(current_first.data(), expected.data(), expected.size() * sizeof(float)) == 0);
            CHECK(memcmp(current_second.data(), expected.data(), expected.size() * sizeof(float)) == 0);
        }
        if (pass == 0) {
            reference_output = expected;
            first_output = current_first;
            second_output = current_second;
        } else if (!auxiliary_proof || pass == 3) {
            CHECK(expected == reference_output);
            CHECK(current_first == first_output && current_second == second_output);
        }
        if (auxiliary_proof) {
            const uint32_t expected_resident = pass == 0 ? 2 : n_slots;
            check_active_grouped_bias_shadows(first_backend.get(), first, expected_resident);
            check_active_grouped_bias_shadows(second_backend.get(), second, expected_resident);
        }
    }
    if (original_direct_biases && !auxiliary_proof) {
        check_active_grouped_bias_shadows(first_backend.get(), first);
        check_active_grouped_bias_shadows(second_backend.get(), second);
    }
    expected = reference_output;
    CHECK(first_output.size() == expected.size() && second_output.size() == expected.size());
    CHECK(memcmp(first_output.data(), second_output.data(), first_output.size() * sizeof(float)) == 0);
    double squared_error = 0.0;
    double squared_expected = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        CHECK(std::isfinite(expected[i]) && std::isfinite(first_output[i]));
        const double difference = expected[i] - first_output[i];
        squared_error += difference * difference;
        squared_expected += static_cast<double>(expected[i]) * expected[i];
    }
    CHECK(squared_expected > 0.0 && squared_error == 0.0);
    const bool reference_decode = n_rows == 1;
    CHECK(active_grouped_legacy_op_count(reference_backend.get(), reference_decode) == 4 * reference.banks.size());
    CHECK(active_grouped_legacy_op_count(reference_backend.get(), !reference_decode) == 0);
    CHECK(active_grouped_legacy_op_count(first_backend.get(), true) == 0 &&
        active_grouped_legacy_op_count(first_backend.get(), false) == 0);
    CHECK(active_grouped_legacy_op_count(second_backend.get(), true) == 0 &&
        active_grouped_legacy_op_count(second_backend.get(), false) == 0);
    const uint64_t expected_misses = auxiliary_proof ? (grouped_frequency_enabled() ? 7 : 8) : 0;
    check_active_grouped_debug_telemetry(first_backend.get(), first, expected_misses, auxiliary_proof ? 9 : 5);
    check_active_grouped_debug_telemetry(second_backend.get(), second, expected_misses, auxiliary_proof ? 9 : 5);
    if (test_owner_concurrency) {
        std::array<std::vector<float>, 2> concurrent_outputs;
        std::thread first_decode([&]() {
            concurrent_outputs[0] = run_active_grouped_dispatch(
                first_backend.get(), first, auxiliary_proof ? 20 : 12 * n_rows,
                f3_skipped, original_direct_biases && n_rows == 1);
        });
        std::thread second_decode([&]() {
            concurrent_outputs[1] = run_active_grouped_dispatch(
                second_backend.get(), second, auxiliary_proof ? 20 : 12 * n_rows,
                f3_skipped, original_direct_biases && n_rows == 1);
        });
        first_decode.join();
        second_decode.join();
        CHECK(concurrent_outputs[0] == first_output && concurrent_outputs[1] == second_output);
        for (auto * backend : {first_backend.get(), second_backend.get()}) {
            auto * context = ggml_cuda_moe_grouped_context_for_test(backend);
            CHECK(context != nullptr);
            const auto telemetry = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
            CHECK(telemetry.registered == 1 && telemetry.covered == 1 && telemetry.plan_calls == 1);
            CHECK(telemetry.plan_compiles == 0 && telemetry.plan_reuses == 1);
            CHECK(telemetry.calls == 1 && telemetry.ready == 1 && telemetry.completed == 1);
            CHECK(telemetry.admitted_banks == first.banks.size());
            CHECK(telemetry.prepare_error == 0 && telemetry.finish_error == 0);
        }
    }
    if (auxiliary_proof) {
        auto * first_context = ggml_cuda_moe_grouped_context_for_test(first_backend.get());
        auto * second_context = ggml_cuda_moe_grouped_context_for_test(second_backend.get());
        ggml_cuda_moe_candidate_group_key first_key;
        ggml_cuda_moe_candidate_group_key second_key;
        CHECK(first_context != nullptr && second_context != nullptr);
        CHECK(first_context->find_down_group_key(first.down, &first_key));
        CHECK(second_context->find_down_group_key(second.down, &second_key));
        for (uint32_t bias = 0; bias < first.biases.size(); ++bias) {
            CHECK(first.biases[bias]->data != second.biases[bias]->data);
            const float * first_shadow = ggml_cuda_moe_grouped_context_test_access::device_auxiliary_data(
                *first_context, first_key, first.biases[bias]);
            const float * second_shadow = ggml_cuda_moe_grouped_context_test_access::device_auxiliary_data(
                *second_context, second_key, second.biases[bias]);
            CHECK(first_shadow != nullptr && second_shadow != nullptr && first_shadow != second_shadow);
        }

        const uint64_t legacy_before = active_grouped_legacy_op_count(first_backend.get());
        void * saved_bias_data = first.biases[0]->data;
        first.biases[0]->data = static_cast<char *>(saved_bias_data) + sizeof(float);
        const auto coverage = candidate_certify_graph(*first_context, first.graph);
        std::shared_ptr<ggml_cuda_moe_graph_plan> failed_plan;
        ggml_cuda_moe_graph_execution failed_execution;
        CHECK(first_context->prepare_graph_execution(
            first.graph, 971, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &failed_plan, &failed_execution,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(failed_execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR && failed_execution.requires_dispatch());
        CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_auxiliary_reason(*failed_plan, 0));
        CHECK(!first_context->begin_graph_dispatch(&failed_execution, GGML_CUDA_MOE_GRAPH_DISPATCH_DIRECT));
        CHECK(active_grouped_legacy_op_count(first_backend.get()) == legacy_before);
        first.biases[0]->data = saved_bias_data;
        const auto failure_telemetry =
            ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*first_context);
        CHECK(failure_telemetry.registered == 1 && failure_telemetry.covered == 0 &&
            failure_telemetry.plan_calls == 0 && failure_telemetry.calls == 0);
        CHECK(failure_telemetry.fallback == 0 && failure_telemetry.rollback == 0);
    }
    check_active_grouped_legacy_caches(reference_backend.get(), reference, false);
    for (ggml_tensor * bank : first.banks) {
        auto * context = ggml_cuda_moe_grouped_context_for_test(first_backend.get());
        CHECK(context != nullptr && !context->acquire_legacy_cache(bank));
    }
    for (ggml_tensor * bank : second.banks) {
        auto * context = ggml_cuda_moe_grouped_context_for_test(second_backend.get());
        CHECK(context != nullptr && !context->acquire_legacy_cache(bank));
    }

    if (!original_direct_down_scale && types[0] == GGML_TYPE_Q4_0 && types[1] == GGML_TYPE_Q4_0 &&
            layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP && n_slots == 12) {
        auto * context = ggml_cuda_moe_grouped_context_for_test(first_backend.get());
        CHECK(context != nullptr);
        std::array<ggml_backend_moe_candidate_bank_v1, 3> replacement_banks = {};
        for (size_t i = 0; i < first.banks.size(); ++i) {
            replacement_banks[i] = {first.banks[i], first.roles[i], 0};
        }
        const ggml_backend_moe_candidate_group_v1 replacement_group = {
            replacement_banks.data(), static_cast<uint32_t>(first.banks.size()), layout, 0, 0,
        };
        const auto replacement_snapshot = candidate_snapshot(n_slots, &replacement_group, 1);
        CHECK(context->replace(&replacement_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        const auto zero_activity_replacement =
            ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
        CHECK(zero_activity_replacement.registered == 1 && zero_activity_replacement.covered == 0);
        CHECK(zero_activity_replacement.plan_calls == 0 && zero_activity_replacement.calls == 0);
        CHECK(zero_activity_replacement.ready == 0 && zero_activity_replacement.completed == 0);

        cudaStream_t stream = nullptr;
        cudaStream_t wrong_stream = nullptr;
        CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_OK(cudaStreamCreateWithFlags(&wrong_stream, cudaStreamNonBlocking));
        ggml_cuda_moe_candidate_group_key failure_key;
        CHECK(context->find_down_group_key(first.down, &failure_key));
        for (ggml_tensor * bank : first.banks) {
            auto legacy = context->acquire_legacy_cache(bank, nullptr, nullptr, stream);
            CHECK(legacy && legacy.get() != nullptr);
        }
        CHECK(ggml_cuda_moe_grouped_context_test_access::legacy_backing_count(*context, failure_key) == first.banks.size());
        CHECK(ggml_cuda_moe_grouped_context_test_access::device_resource_complete(*context, failure_key));

        std::shared_ptr<ggml_cuda_moe_graph_plan> failure_plan;
        ggml_cuda_moe_graph_execution failure_execution;
        CHECK(context->prepare_graph_execution(first.graph, 701, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &failure_plan, &failure_execution) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(failure_execution.resolve_streams(candidate_test_graph_stream, stream));
        CHECK(context->begin_graph_dispatch(&failure_execution, true));
        ggml_cuda_moe_graph_binding binding;
        auto * failure_group = failure_execution.find_group(first.readers[0], &binding);
        CHECK(failure_group != nullptr);
        CHECK(context->prepare_graph_group(failure_group, binding, first.readers[0], stream) == GGML_CUDA_MOE_GROUPED_DECODE_READY);
        const auto failed_resource = failure_group->transaction.acquisition;
        failure_group->stream = wrong_stream;
        CHECK(!context->finish_graph_dispatch(&failure_execution));
        CHECK(!context->get_group_resources(failed_resource, nullptr));
        CHECK(ggml_cuda_moe_grouped_context_test_access::legacy_backing_count(*context, failure_key) == 0);
        const auto failure_telemetry = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
        CHECK(failure_telemetry.registered == 1 && failure_telemetry.covered == 1);
        CHECK(failure_telemetry.plan_calls == 1 && failure_telemetry.plan_compiles == 1);
        CHECK(failure_telemetry.calls == 1 && failure_telemetry.ready == 1 && failure_telemetry.completed == 0);
        CHECK(failure_telemetry.prepare_error == 0 && failure_telemetry.finish_error == 1);
        CUDA_OK(cudaStreamDestroy(wrong_stream));
        CUDA_OK(cudaStreamDestroy(stream));
        CHECK(run_active_grouped_dispatch(first_backend.get(), first, 2, false) == first_output);

        ggml_cuda_moe_candidate_group_key key;
        ggml_cuda_moe_grouped_acquisition resource;
        ggml_cuda_moe_grouped_transaction held;
        CHECK(context->find_down_group_key(first.down, &key));
        const uint64_t original_graph_uid = first.graph->uid;
        const auto original_certificate = first.graph->execution_certificate;
        candidate_stamp_execution(first.graph, GGML_GRAPH_EXECUTION_DOMAIN_MTP,
            GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, first.n_rows, first.n_rows, 0,
            GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED);
        CHECK(candidate_certify_graph(*context, first.graph).epoch != 0);
        CHECK(context->acquire_group_resources(key, &resource));
        CHECK(context->begin_group_transaction(resource, &held));
        std::atomic<bool> replacement_started{false};
        std::atomic<int32_t> replacement_result{GGML_BACKEND_MOE_CANDIDATE_REPLACE_ERROR};
        std::thread replacement([&]() {
            replacement_started.store(true, std::memory_order_release);
            replacement_result.store(context->replace(&replacement_snapshot), std::memory_order_release);
        });
        while (!replacement_started.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        while (!ggml_cuda_moe_grouped_context_test_access::admission_closed(*context)) {
            std::this_thread::yield();
        }
        std::shared_ptr<ggml_cuda_moe_graph_plan> unavailable_plan;
        ggml_cuda_moe_graph_execution unavailable_execution;
        while (context->prepare_graph_execution(first.graph, 702, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &unavailable_plan, &unavailable_execution) !=
                GGML_CUDA_MOE_GRAPH_PREPARE_UNAVAILABLE) {
            std::this_thread::yield();
        }
        std::vector<float> zero_input(ggml_nelements(first.input));
        ggml_backend_tensor_set(first.input, zero_input.data(), 0, ggml_nbytes(first.input));
        CHECK(ggml_backend_graph_compute(first_backend.get(), first.graph) == GGML_STATUS_FAILED);
        std::vector<float> preserved_output(ggml_nelements(first.output));
        ggml_backend_tensor_get(first.output, preserved_output.data(), 0, ggml_nbytes(first.output));
        CHECK(preserved_output == first_output);
        first.graph->uid = original_graph_uid;
        first.graph->execution_certificate = original_certificate;
        const auto restored_input = cached_fusion_test_data(first.input, 131);
        ggml_backend_tensor_set(first.input, restored_input.data(), 0, restored_input.size());
        CHECK(context->end_group_transaction(held));
        replacement.join();
        CHECK(replacement_result.load(std::memory_order_acquire) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        CHECK(run_active_grouped_dispatch(first_backend.get(), first, 2, false) == first_output);
        const auto replacement_telemetry =
            ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
        CHECK(replacement_telemetry.registered == 2 && replacement_telemetry.covered == 2);
        CHECK(replacement_telemetry.plan_calls == 2 &&
            replacement_telemetry.plan_compiles + replacement_telemetry.plan_reuses == replacement_telemetry.plan_calls);
        CHECK(replacement_telemetry.calls == 2 && replacement_telemetry.ready == 2 && replacement_telemetry.completed == 2);
        CHECK(replacement_telemetry.ready_min == 1 && replacement_telemetry.ready_max == 1);
        CHECK(replacement_telemetry.completed_min == 1 && replacement_telemetry.completed_max == 1);
        CHECK(replacement_telemetry.h2d_banks == 4 * first.banks.size());
        CHECK(replacement_telemetry.required_unsupported >= 1);

        CHECK(run_active_grouped_dispatch(first_backend.get(), first, 4, false) == first_output);
        const auto disabled_snapshot = candidate_snapshot(n_slots, nullptr, 0);
        CHECK(context->replace(&disabled_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        const auto disabled_telemetry = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
        CHECK(disabled_telemetry.registered == 1 && disabled_telemetry.covered == 1);
        CHECK(disabled_telemetry.plan_calls == 1 && disabled_telemetry.calls == 1);
        CHECK(disabled_telemetry.ready == 1 && disabled_telemetry.completed == 1);
        register_active_grouped_dispatch(first_backend.get(), first, layout, n_slots);
        CHECK(run_active_grouped_dispatch(first_backend.get(), first, 2, false) == first_output);
        const ggml_backend_moe_candidate_group_v1 rejected_group = {
            replacement_banks.data(), static_cast<uint32_t>(first.banks.size()), layout, 1, 0,
        };
        const auto rejected_snapshot = candidate_snapshot(n_slots, &rejected_group, 1);
        CHECK(context->replace(&rejected_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_REJECTED);
        const auto rejected_telemetry = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
        CHECK(rejected_telemetry.registered == 1 && rejected_telemetry.covered == 1);
        CHECK(rejected_telemetry.plan_calls == 1 && rejected_telemetry.calls == 1);
        CHECK(rejected_telemetry.ready == 1 && rejected_telemetry.completed == 1);
        CHECK(rejected_telemetry.h2d_banks == 2 * first.banks.size());
        register_active_grouped_dispatch(first_backend.get(), first, layout, n_slots);
        ggml_cuda_moe_grouped_context::log_and_reset_legacy_stats();
        const int32_t shutdown_routes[] = {3, 5};
        ggml_backend_tensor_set(first.ids, shutdown_routes, 0, sizeof(shutdown_routes));
        std::shared_ptr<ggml_cuda_moe_graph_plan> shutdown_plan;
        ggml_cuda_moe_graph_execution shutdown_execution;
        CHECK(context->prepare_graph_execution(first.graph, 703, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &shutdown_plan, &shutdown_execution) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        cudaStream_t shutdown_stream = nullptr;
        CUDA_OK(cudaStreamCreateWithFlags(&shutdown_stream, cudaStreamNonBlocking));
        CHECK(shutdown_execution.resolve_streams(candidate_test_graph_stream, shutdown_stream));
        CHECK(context->begin_graph_dispatch(&shutdown_execution, true));
        ggml_cuda_moe_graph_binding shutdown_binding;
        auto * shutdown_group = shutdown_execution.find_group(first.readers[0], &shutdown_binding);
        CHECK(shutdown_group != nullptr);
        CHECK(context->prepare_graph_group(shutdown_group, shutdown_binding, first.readers[0], shutdown_stream) ==
            GGML_CUDA_MOE_GROUPED_DECODE_READY);
        const auto shutdown_resource = shutdown_group->transaction.acquisition;
        host_barrier barrier;
        CUDA_OK(cudaLaunchHostFunc(shutdown_stream, wait_on_host_barrier, &barrier));
        ggml_cuda_moe_graph_binding shutdown_last_binding;
        CHECK(shutdown_execution.find_group(first.readers.back(), &shutdown_last_binding) == shutdown_group);
        CHECK(context->finish_graph_group(shutdown_group, shutdown_last_binding, first.readers.back(), shutdown_stream));
        CHECK(context->finish_graph_dispatch(&shutdown_execution));
        while (!barrier.entered.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        ggml_cuda_moe_grouped_resource_info shutdown_info;
        CHECK(context->get_group_resources(shutdown_resource, &shutdown_info) && !shutdown_info.transaction_active);
        CHECK(context->find_down_group_key(first.down, &key));
        std::atomic<bool> shutdown_started{false};
        std::atomic<bool> shutdown_done{false};
        std::thread shutdown([&]() {
            shutdown_started.store(true, std::memory_order_release);
            context->shutdown();
            shutdown_done.store(true, std::memory_order_release);
        });
        while (!shutdown_started.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        while (!ggml_cuda_moe_grouped_context_test_access::admission_closed(*context)) {
            std::this_thread::yield();
        }
        ggml_cuda_moe_grouped_acquisition shutdown_probe;
        CHECK(!context->acquire_group_resources(key, &shutdown_probe));
        CHECK(!shutdown_done.load(std::memory_order_acquire));
        std::atomic<bool> logger_started{false};
        std::atomic<bool> logger_done{false};
        ggml_cuda_moe_grouped_debug_telemetry shutdown_telemetry;
        std::thread logger([&]() {
            logger_started.store(true, std::memory_order_release);
            shutdown_telemetry = ggml_cuda_moe_grouped_context::log_and_reset_legacy_stats();
            logger_done.store(true, std::memory_order_release);
        });
        while (!logger_started.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        for (uint32_t attempt = 0; attempt < 100 && !logger_done.load(std::memory_order_acquire); ++attempt) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        CHECK(!logger_done.load(std::memory_order_acquire) && !shutdown_done.load(std::memory_order_acquire));
        barrier.released.store(true, std::memory_order_release);
        logger.join();
        shutdown.join();
        CHECK(logger_done.load(std::memory_order_acquire) && shutdown_done.load(std::memory_order_acquire));
        CUDA_OK(cudaStreamDestroy(shutdown_stream));
        CHECK(shutdown_telemetry.covered == 1 && shutdown_telemetry.plan_calls == 1);
        CHECK(shutdown_telemetry.plan_compiles + shutdown_telemetry.plan_reuses == shutdown_telemetry.plan_calls);
        CHECK(shutdown_telemetry.calls == 1 && shutdown_telemetry.ready == 1 && shutdown_telemetry.completed == 1);
        CHECK(shutdown_telemetry.admitted_banks == first.banks.size());
        CHECK(shutdown_telemetry.fallback == 0 && shutdown_telemetry.rollback == 0);
        CHECK(shutdown_telemetry.prepare_error == 0 && shutdown_telemetry.finish_error == 0);
        CHECK(shutdown_telemetry.h2d_banks == 2 * first.banks.size());
        uint64_t shutdown_h2d_bytes = 0;
        for (const ggml_tensor * bank : first.banks) {
            shutdown_h2d_bytes += 2 * bank->nb[2];
        }
        CHECK(shutdown_telemetry.h2d_bytes == shutdown_h2d_bytes);
        const auto shutdown_reset = ggml_cuda_moe_grouped_context::log_and_reset_legacy_stats();
        CHECK(shutdown_reset.covered == 0 && shutdown_reset.plan_calls == 0);
        CHECK(shutdown_reset.plan_compiles == 0 && shutdown_reset.plan_reuses == 0 && shutdown_reset.calls == 0);
        CHECK(shutdown_reset.ready == 0 && shutdown_reset.ready_min == 0 && shutdown_reset.ready_max == 0);
        CHECK(shutdown_reset.completed == 0 && shutdown_reset.completed_min == 0 && shutdown_reset.completed_max == 0);
        CHECK(shutdown_reset.admitted_banks == 0);
        CHECK(shutdown_reset.fallback == 0 && shutdown_reset.rollback == 0);
        CHECK(shutdown_reset.prepare_error == 0 && shutdown_reset.finish_error == 0);
        CHECK(shutdown_reset.h2d_banks == 0 && shutdown_reset.h2d_bytes == 0);
    }
    ggml_backend_cuda_moe_set_debug_mm(old_debug_mm);
}

static void test_active_grouped_dispatch_case(
        ggml_type type,
        uint32_t layout,
        uint32_t n_slots,
        bool original_direct_down_scale = false,
        bool test_owner_concurrency = false) {
    test_active_grouped_dispatch_types_case(
        {type, type, type}, layout, n_slots, original_direct_down_scale, test_owner_concurrency);
}

static void test_active_grouped_materialization_eligibility() {
#ifdef __linux__
    const bool old_debug_mm = ggml_backend_cuda_moe_get_debug_mm();
    ggml_backend_cuda_moe_set_debug_mm(true);
    ggml_backend_ptr reference_backend(ggml_backend_cuda_init(0));
    ggml_backend_ptr pinned_backend(ggml_backend_cuda_init(0));
    ggml_backend_ptr pageable_backend(ggml_backend_cuda_init(0));
    CHECK(reference_backend != nullptr && pinned_backend != nullptr && pageable_backend != nullptr);
    auto reference = build_active_grouped_dispatch_graph(
        reference_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, true);
    auto pinned = build_active_grouped_dispatch_graph(
        pinned_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, true);
    auto pageable = build_active_grouped_dispatch_graph(
        pageable_backend.get(), file_mmap_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, true);
    initialize_active_grouped_dispatch_graphs({&reference, &pinned, &pageable});

    cudaPointerAttributes attributes = {};
    const cudaError_t pointer_status = cudaPointerGetAttributes(
        &attributes, ggml_backend_buffer_get_base(pageable.weight_buffer.get()));
    CHECK(pointer_status != cudaSuccess || attributes.type == cudaMemoryTypeUnregistered);
    if (pointer_status != cudaSuccess) {
        (void) cudaGetLastError();
    }

    const auto disabled = candidate_snapshot(12, nullptr, 0);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v1(
        reference_backend.get(), &disabled) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    register_active_grouped_dispatch(
        pinned_backend.get(), pinned, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 12);
    register_active_grouped_dispatch(
        pageable_backend.get(), pageable, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 12);

    const auto check_plan = [](
            ggml_backend_t backend,
            const active_grouped_dispatch_graph & graph,
            ggml_cuda_moe_graph_outcome expected_outcome,
            bool materialization) {
        auto * context = ggml_cuda_moe_grouped_context_for_test(backend);
        CHECK(context != nullptr);
        const auto coverage = candidate_certify_graph(*context, graph.graph);
        std::shared_ptr<ggml_cuda_moe_graph_plan> plan;
        ggml_cuda_moe_graph_execution execution;
        CHECK(context->prepare_graph_execution(
            graph.graph, 901, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &plan, &execution,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(plan != nullptr && execution.outcome() == expected_outcome && execution.size() == 1);
        CHECK(execution.find_group(graph.down_output, nullptr) != nullptr);
        CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_materialization_reason(*plan, 0) == materialization);
        CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_eligible_reason(*plan, 0) != materialization);
        ggml_cuda_moe_graph_execution unknown;
        CHECK(context->bind_graph_plan(
            graph.graph, 902, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, *plan, &unknown,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint));
        CHECK(unknown.outcome() == expected_outcome && unknown.find_group(graph.down_output, nullptr) != nullptr);
    };
    check_plan(pinned_backend.get(), pinned, GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED, false);
    check_plan(pageable_backend.get(), pageable, GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_LEGACY, true);

    const auto expected = run_active_grouped_dispatch(reference_backend.get(), reference, 0, false);
    const auto pinned_output = run_active_grouped_dispatch(pinned_backend.get(), pinned, 2, true);
    const auto pageable_output = run_active_grouped_dispatch(pageable_backend.get(), pageable, 0, false);
    CHECK(pinned_output == expected && pageable_output == expected);
    CHECK(active_grouped_legacy_op_count(reference_backend.get()) == reference.banks.size());
    CHECK(active_grouped_legacy_op_count(pinned_backend.get()) == 0);
    CHECK(active_grouped_legacy_op_count(pageable_backend.get()) == pageable.banks.size());

    auto * pinned_context = ggml_cuda_moe_grouped_context_for_test(pinned_backend.get());
    auto * pageable_context = ggml_cuda_moe_grouped_context_for_test(pageable_backend.get());
    ggml_cuda_moe_candidate_group_key pinned_key;
    ggml_cuda_moe_candidate_group_key pageable_key;
    CHECK(pinned_context != nullptr && pageable_context != nullptr);
    CHECK(pinned_context->find_down_group_key(pinned.down, &pinned_key));
    CHECK(pageable_context->find_down_group_key(pageable.down, &pageable_key));
    CHECK(ggml_cuda_moe_grouped_context_test_access::has_device_resource(*pinned_context, pinned_key));
    CHECK(!ggml_cuda_moe_grouped_context_test_access::has_device_resource(*pageable_context, pageable_key));
    const auto pageable_telemetry =
        ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*pageable_context);
    CHECK(pageable_telemetry.registered == 1 && pageable_telemetry.covered == 0);
    CHECK(pageable_telemetry.calls == 0 && pageable_telemetry.ready == 0 && pageable_telemetry.completed == 0);
    CHECK(pageable_telemetry.prepare_error == 0 && pageable_telemetry.finish_error == 0);
    check_active_grouped_legacy_caches(pageable_backend.get(), pageable, true);
    for (ggml_tensor * bank : pinned.banks) {
        CHECK(!pinned_context->acquire_legacy_cache(bank));
    }
    ggml_backend_cuda_moe_set_debug_mm(old_debug_mm);
    fprintf(stderr, "test-moe-cache: grouped mapped-source eligibility OK\n");
#else
    fprintf(stderr, "test-moe-cache: grouped mapped-source eligibility SKIP\n");
#endif
}

static void test_active_grouped_dispatch_staged_legacy() {
    const bool old_debug_mm = ggml_backend_cuda_moe_get_debug_mm();
    ggml_backend_cuda_moe_set_debug_mm(true);
    ggml_backend_ptr reference_backend(ggml_backend_cuda_init(0));
    ggml_backend_ptr candidate_backend(ggml_backend_cuda_init(0));
    CHECK(reference_backend != nullptr && candidate_backend != nullptr);
    auto reference = build_active_grouped_dispatch_graph(
        reference_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE);
    auto candidate = build_active_grouped_dispatch_graph(
        candidate_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE);
    initialize_active_grouped_dispatch_graphs({&reference, &candidate});
    const auto disabled = candidate_snapshot(1, nullptr, 0);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v1(reference_backend.get(), &disabled) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    register_active_grouped_dispatch(candidate_backend.get(), candidate, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 1);

    const auto expected = run_active_grouped_dispatch(reference_backend.get(), reference, 0, false);
    CHECK(!expected.empty());
    const auto actual = run_active_grouped_dispatch(candidate_backend.get(), candidate, 0, false);
    check_active_grouped_exact_output(expected, actual);
    CHECK(active_grouped_legacy_op_count(reference_backend.get()) == reference.banks.size());
    CHECK(active_grouped_legacy_op_count(candidate_backend.get()) == candidate.banks.size());
    check_active_grouped_legacy_caches(reference_backend.get(), reference, false, false);
    check_active_grouped_legacy_caches(candidate_backend.get(), candidate, true, false);

    auto * context = ggml_cuda_moe_grouped_context_for_test(candidate_backend.get());
    ggml_cuda_moe_candidate_group_key key;
    CHECK(context != nullptr && context->find_down_group_key(candidate.down, &key));
    CHECK(ggml_cuda_moe_grouped_context_test_access::has_device_resource(*context, key));
    CHECK(ggml_cuda_moe_grouped_context_test_access::device_resource_complete(*context, key));
    CHECK(ggml_cuda_moe_grouped_context_test_access::legacy_backing_count(*context, key) == candidate.banks.size());
    const auto telemetry = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
    CHECK(telemetry.registered == 1 && telemetry.covered == 0 && telemetry.plan_calls == 0);
    CHECK(telemetry.calls == 0 && telemetry.ready == 0 && telemetry.completed == 0 && telemetry.admitted_banks == 0);
    CHECK(telemetry.decode_grouped == 0 && telemetry.decode_legacy == 1 && telemetry.fallback == 1);
    CHECK(telemetry.rollback == 0 && telemetry.prepare_error == 0 && telemetry.finish_error == 0);
    CHECK(telemetry.h2d_banks == 0 && telemetry.h2d_bytes == 0);
    ggml_backend_cuda_moe_set_debug_mm(old_debug_mm);
}

static void test_active_grouped_inactive_v2_case(uint32_t snapshot_flags, uint32_t group_flags, uint32_t coverage_reason) {
    ggml_backend_ptr backend(ggml_backend_cuda_init(0));
    CHECK(backend != nullptr);
    auto graph = build_active_grouped_dispatch_graph(
        backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE);
    initialize_active_grouped_dispatch_graph(graph, 201);

    const ggml_backend_moe_candidate_group_v2 group = {
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY, group_flags, 0,
    };
    std::array<ggml_backend_moe_candidate_tensor_v2, 3> tensors = {};
    for (uint32_t i = 0; i < tensors.size(); ++i) {
        tensors[i] = {
            graph.banks[i], 0, graph.roles[i], GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE,
            GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_CACHED_BUFFER, 0,
        };
    }
    auto snapshot = candidate_snapshot_v2(12, &group, 1, tensors.data(), tensors.size());
    snapshot.flags = snapshot_flags;
    CHECK(ggml_backend_cuda_moe_candidate_replace_v2(backend.get(), &snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    auto * context = ggml_cuda_moe_grouped_context_for_test(backend.get());
    CHECK(context != nullptr && context->state().accepted && context->state().n_slots == 12 && context->state().n_groups == 0);

    ggml_cuda_moe_graph_plan plan;
    ggml_cuda_moe_graph_execution execution;
    context->compile_graph_plan(graph.graph, 901, &plan, &execution);
    CHECK(execution.size() == 0 && execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR && execution.requires_dispatch());
    CHECK(plan.coverage_diagnostics().cached_mmid == graph.readers.size());
    CHECK(plan.coverage_diagnostics().counts[coverage_reason] == graph.readers.size());

    const auto sentinel = active_grouped_intermediate_sentinel(graph);
    CHECK(ggml_backend_graph_compute(backend.get(), graph.graph) == GGML_STATUS_FAILED);
    ggml_backend_synchronize(backend.get());
    check_active_grouped_intermediates(graph, sentinel, true);
    CHECK(active_grouped_legacy_op_count(backend.get()) == 0);
}

static void test_active_grouped_descriptive_v2() {
    ggml_backend_ptr backend(ggml_backend_cuda_init(0));
    CHECK(backend != nullptr);
    auto graph = build_active_grouped_dispatch_graph(
        backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE);
    initialize_active_grouped_dispatch_graph(graph, 251);

    const ggml_backend_moe_candidate_group_v2 group = {
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_UNGATED, GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_CHUNK,
        GGML_BACKEND_MOE_CANDIDATE_GROUP_V2_FLAG_NONE, 0,
    };
    const auto snapshot = candidate_snapshot_v2(12, &group, 1, nullptr, 0);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v2(backend.get(), &snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    auto * context = ggml_cuda_moe_grouped_context_for_test(backend.get());
    CHECK(context != nullptr && context->state().accepted && context->state().n_slots == 12 && context->state().n_groups == 0);

    ggml_cuda_moe_graph_plan plan;
    ggml_cuda_moe_graph_execution execution;
    context->compile_graph_plan(graph.graph, 9021, &plan, &execution);
    CHECK(execution.size() == 0 && execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR && execution.requires_dispatch());
    CHECK(plan.coverage_diagnostics().cached_mmid == graph.readers.size());
    CHECK(plan.coverage_diagnostics().counts[GGML_CUDA_MOE_GRAPH_COVERAGE_REVERSE_MAP_MISS] == graph.readers.size());

    std::vector<float> sentinel(ggml_nelements(graph.output), -12345.25f);
    ggml_backend_tensor_set(graph.output, sentinel.data(), 0, ggml_nbytes(graph.output));
    ggml_backend_synchronize(backend.get());
    CHECK(ggml_backend_graph_compute(backend.get(), graph.graph) == GGML_STATUS_FAILED);
    ggml_backend_synchronize(backend.get());
    std::vector<float> output(ggml_nelements(graph.output));
    ggml_backend_tensor_get(graph.output, output.data(), 0, ggml_nbytes(graph.output));
    CHECK(output == sentinel);
    CHECK(active_grouped_legacy_op_count(backend.get()) == 0);
}

static void test_active_grouped_flags_only_v2(uint32_t flags) {
    ggml_backend_ptr backend(ggml_backend_cuda_init(0));
    CHECK(backend != nullptr);
    auto graph = build_active_grouped_dispatch_graph(
        backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE);
    initialize_active_grouped_dispatch_graph(graph, 271 + flags);

    auto snapshot = candidate_snapshot_v2(12, nullptr, 0, nullptr, 0);
    snapshot.flags = flags;
    CHECK(ggml_backend_cuda_moe_candidate_replace_v2(backend.get(), &snapshot) ==
        GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    auto * context = ggml_cuda_moe_grouped_context_for_test(backend.get());
    CHECK(context != nullptr && context->state().accepted && context->state().n_groups == 0);

    ggml_cuda_moe_graph_plan plan;
    ggml_cuda_moe_graph_execution execution;
    context->compile_graph_plan(graph.graph, 9022 + flags, &plan, &execution);
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR && execution.requires_dispatch());
    CHECK(execution.rejects_cached_mmid(graph.readers[0]));
    CHECK(!execution.rejects_cached_mmid(graph.ids));

    const auto sentinel = active_grouped_intermediate_sentinel(graph);
    CHECK(ggml_backend_graph_compute(backend.get(), graph.graph) == GGML_STATUS_FAILED);
    ggml_backend_synchronize(backend.get());
    check_active_grouped_intermediates(graph, sentinel, true);
    CHECK(active_grouped_legacy_op_count(backend.get()) == 0);
}

static void test_active_grouped_explicit_empty_case(bool v2) {
    ggml_backend_ptr reference_backend(ggml_backend_cuda_init(0));
    ggml_backend_ptr candidate_backend(ggml_backend_cuda_init(0));
    CHECK(reference_backend != nullptr && candidate_backend != nullptr);
    auto reference = build_active_grouped_dispatch_graph(
        reference_backend.get(), ggml_backend_cuda_buffer_type(0), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE);
    auto candidate = build_active_grouped_dispatch_graph(
        candidate_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE);
    initialize_active_grouped_dispatch_graphs({&reference, &candidate});
    if (v2) {
        const auto disabled = candidate_snapshot_v2(12, nullptr, 0, nullptr, 0);
        CHECK(ggml_backend_cuda_moe_candidate_replace_v2(candidate_backend.get(), &disabled) ==
            GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    } else {
        const auto disabled = candidate_snapshot(12, nullptr, 0);
        CHECK(ggml_backend_cuda_moe_candidate_replace_v1(candidate_backend.get(), &disabled) ==
            GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    }

    auto * context = ggml_cuda_moe_grouped_context_for_test(candidate_backend.get());
    CHECK(context != nullptr);
    const auto coverage = candidate_certify_graph(*context, candidate.graph);
    std::shared_ptr<ggml_cuda_moe_graph_plan> plan;
    ggml_cuda_moe_graph_execution execution;
    CHECK(context->prepare_graph_execution(
        candidate.graph, 903, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &plan, &execution,
        coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
    CHECK(execution.size() == 0 && execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_PREFILL_LEGACY && !execution.requires_dispatch());
    const ggml_cuda_moe_graph_plan * stable_plan = plan.get();
    CHECK(context->prepare_graph_execution(
        candidate.graph, 904, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &plan, &execution,
        coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) == GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
    CHECK(plan.get() == stable_plan && execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_PREFILL_LEGACY);

    const auto expected = run_active_grouped_dispatch(reference_backend.get(), reference, 0, true);
    const auto first = run_active_grouped_dispatch(candidate_backend.get(), candidate, 0, false);
    const auto second = run_active_grouped_dispatch(candidate_backend.get(), candidate, 0, false);
    CHECK(first == expected && second == expected);
    CHECK(active_grouped_legacy_op_count(candidate_backend.get()) == 2 * candidate.readers.size());
}

static void test_active_grouped_empty_policy() {
    test_active_grouped_inactive_v2_case(
        GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_FLAG_NONE,
        GGML_BACKEND_MOE_CANDIDATE_GROUP_V2_FLAG_ACTIVE_LORA,
        GGML_CUDA_MOE_GRAPH_COVERAGE_ACTIVE_LORA);
    test_active_grouped_inactive_v2_case(
        GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_FLAG_TENSOR_OVERRIDES,
        GGML_BACKEND_MOE_CANDIDATE_GROUP_V2_FLAG_NONE,
        GGML_CUDA_MOE_GRAPH_COVERAGE_TENSOR_OVERRIDE);
    test_active_grouped_inactive_v2_case(
        GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_FLAG_INCOMPLETE,
        GGML_BACKEND_MOE_CANDIDATE_GROUP_V2_FLAG_NONE,
        GGML_CUDA_MOE_GRAPH_COVERAGE_INCOMPLETE);
    test_active_grouped_descriptive_v2();
    test_active_grouped_flags_only_v2(GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_FLAG_TENSOR_OVERRIDES);
    test_active_grouped_flags_only_v2(GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_FLAG_INCOMPLETE);

    {
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        CHECK(backend != nullptr);
        auto graph = build_active_grouped_dispatch_graph(
            backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
            GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE);
        initialize_active_grouped_dispatch_graph(graph, 301);
        std::array<ggml_backend_moe_candidate_bank_v1, 3> banks = {};
        for (uint32_t i = 0; i < banks.size(); ++i) {
            banks[i] = {graph.banks[i], graph.roles[i], 0};
        }
        const ggml_backend_moe_candidate_group_v1 group = {
            banks.data(), banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 1, 0,
        };
        const auto snapshot = candidate_snapshot(12, &group, 1);
        CHECK(ggml_backend_cuda_moe_candidate_replace_v1(backend.get(), &snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_REJECTED);
        auto * context = ggml_cuda_moe_grouped_context_for_test(backend.get());
        CHECK(context != nullptr && !context->state().accepted && context->state().n_slots == 12);
        ggml_cuda_moe_graph_plan plan;
        ggml_cuda_moe_graph_execution execution;
        context->compile_graph_plan(graph.graph, 902, &plan, &execution);
        CHECK(execution.size() == 0 && execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR && execution.requires_dispatch());
        std::vector<float> sentinel(ggml_nelements(graph.output), -12345.25f);
        ggml_backend_tensor_set(graph.output, sentinel.data(), 0, ggml_nbytes(graph.output));
        ggml_backend_synchronize(backend.get());
        CHECK(ggml_backend_graph_compute(backend.get(), graph.graph) == GGML_STATUS_FAILED);
        ggml_backend_synchronize(backend.get());
        std::vector<float> output(ggml_nelements(graph.output));
        ggml_backend_tensor_get(graph.output, output.data(), 0, ggml_nbytes(graph.output));
        CHECK(output == sentinel);
        CHECK(active_grouped_legacy_op_count(backend.get()) == 0);
    }
    const bool old_debug_mm = ggml_backend_cuda_moe_get_debug_mm();
    ggml_backend_cuda_moe_set_debug_mm(true);
    test_active_grouped_explicit_empty_case(false);
    test_active_grouped_explicit_empty_case(true);
    ggml_backend_cuda_moe_set_debug_mm(old_debug_mm);
    fprintf(stderr, "test-moe-cache: empty grouped policy production seam OK\n");
}

static void test_active_grouped_inventory_reuse_case(ggml_type second_type, bool second_supported) {
    ggml_backend_ptr backend(ggml_backend_cuda_init(0));
    CHECK(backend != nullptr);
    auto first = build_active_grouped_dispatch_graph(
        backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE);
    auto second = build_active_grouped_dispatch_graph(
        backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), second_type,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE);
    initialize_active_grouped_dispatch_graph(first, 501);
    if (second_supported) {
        initialize_active_grouped_dispatch_graph(second, 601);
    }

    std::array<std::array<ggml_backend_moe_candidate_bank_v1, 3>, 2> banks = {};
    std::array<ggml_backend_moe_candidate_group_v1, 2> groups = {};
    for (uint32_t group_index = 0; group_index < groups.size(); ++group_index) {
        const auto & source = group_index == 0 ? first : second;
        CHECK(source.banks.size() == banks[group_index].size());
        for (uint32_t bank_index = 0; bank_index < source.banks.size(); ++bank_index) {
            banks[group_index][bank_index] = {source.banks[bank_index], source.roles[bank_index], 0};
        }
        groups[group_index] = {
            banks[group_index].data(), static_cast<uint32_t>(banks[group_index].size()),
            GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 0, 0,
        };
    }
    const auto snapshot = candidate_snapshot(12, groups.data(), groups.size());
    CHECK(ggml_backend_cuda_moe_candidate_replace_v1(backend.get(), &snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    auto * context = ggml_cuda_moe_grouped_context_for_test(backend.get());
    CHECK(context != nullptr && context->state().n_groups == groups.size());

    const int32_t combined_nodes = first.graph->n_nodes + second.graph->n_nodes;
    const ggml_init_params graph_params = {
        /* .mem_size = */ ggml_tensor_overhead() * 2 + ggml_graph_overhead_custom(combined_nodes, false),
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };
    ggml_context_ptr graph_context(ggml_init(graph_params));
    CHECK(graph_context != nullptr);
    ggml_cgraph * graph = ggml_new_graph_custom(graph_context.get(), combined_nodes, false);
    CHECK(graph != nullptr);
    for (int32_t node_index = 0; node_index < first.graph->n_nodes; ++node_index) {
        ggml_graph_add_node(graph, first.graph->nodes[node_index]);
    }
    const int32_t tail_index = graph->n_nodes;
    for (int32_t node_index = 0; node_index < second.graph->n_nodes; ++node_index) {
        ggml_graph_add_node(graph, first.graph->nodes[0]);
    }
    candidate_rebuild_graph_uses(graph);
    candidate_stamp_execution(graph, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
        GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 1, 1);

    const auto coverage = candidate_certify_graph(*context, graph);
    std::shared_ptr<ggml_cuda_moe_graph_plan> plan;
    ggml_cuda_moe_graph_execution execution;
    CHECK(context->prepare_graph_execution(
        graph, graph->uid, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &plan, &execution,
        coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
        GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED && execution.size() == 1);
    CHECK(execution.find(first.down_output, nullptr) && !execution.find(second.down_output, nullptr));
    const std::shared_ptr<ggml_cuda_moe_graph_plan> stale_plan = plan;

    for (int32_t node_index = 0; node_index < second.graph->n_nodes; ++node_index) {
        graph->nodes[tail_index + node_index] = second.graph->nodes[node_index];
    }
    candidate_rebuild_graph_uses(graph);
    const auto updated_coverage = candidate_certify_graph(*context, graph);
    CHECK(!context->bind_graph_plan(
        graph, graph->uid, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, *stale_plan, &execution,
        updated_coverage.epoch, updated_coverage.nodes,
        updated_coverage.mmid_count, updated_coverage.mmid_fingerprint));
    CHECK(context->prepare_graph_execution(
        graph, graph->uid, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &plan, &execution,
        updated_coverage.epoch, updated_coverage.nodes,
        updated_coverage.mmid_count, updated_coverage.mmid_fingerprint) ==
        GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
    CHECK(plan != stale_plan);
    if (second_supported) {
        CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED && execution.size() == 2);
        CHECK(execution.find(first.down_output, nullptr) && execution.find(second.down_output, nullptr));
    } else {
        CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR && execution.size() == 2);
        CHECK(!execution.find(first.down_output, nullptr) && !execution.find(second.down_output, nullptr));
    }

    const auto first_sentinel = active_grouped_intermediate_sentinel(first);
    const auto second_sentinel = active_grouped_intermediate_sentinel(second);
    const ggml_status status = ggml_backend_graph_compute(backend.get(), graph);
    ggml_backend_synchronize(backend.get());
    CHECK(status == (second_supported ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED));
    check_active_grouped_intermediates(first, first_sentinel, true);
    check_active_grouped_intermediates(second, second_sentinel, true);
    CHECK(active_grouped_legacy_op_count(backend.get()) == 0);
    if (second_supported) {
        for (const auto * current : {&first, &second}) {
            ggml_cuda_moe_candidate_group_key key;
            CHECK(context->find_down_group_key(current->down, &key));
            CHECK(ggml_cuda_moe_grouped_context_test_access::has_device_resource(*context, key));
        }
    }
}

static void test_active_grouped_in_place_inventory_reuse_case(bool registered) {
    ggml_backend_ptr backend(ggml_backend_cuda_init(0));
    CHECK(backend != nullptr);
    auto first = build_active_grouped_dispatch_graph(
        backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE);
    auto added = build_active_grouped_dispatch_graph(
        backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), registered ? GGML_TYPE_Q8_K : GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE);
    initialize_active_grouped_dispatch_graph(first, 701);

    std::array<std::array<ggml_backend_moe_candidate_bank_v1, 3>, 2> banks = {};
    std::array<ggml_backend_moe_candidate_group_v1, 2> groups = {};
    for (uint32_t group_index = 0; group_index < groups.size(); ++group_index) {
        const auto & source = group_index == 0 ? first : added;
        for (uint32_t bank_index = 0; bank_index < banks[group_index].size(); ++bank_index) {
            banks[group_index][bank_index] = {source.banks[bank_index], source.roles[bank_index], 0};
        }
        groups[group_index] = {
            banks[group_index].data(), static_cast<uint32_t>(banks[group_index].size()),
            GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 0, 0,
        };
    }
    const auto snapshot = candidate_snapshot(12, groups.data(), registered ? 2 : 1);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v1(backend.get(), &snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    auto * context = ggml_cuda_moe_grouped_context_for_test(backend.get());
    CHECK(context != nullptr && context->state().n_groups == (registered ? 2u : 1u));

    std::array<ggml_op, 3> saved_ops = {};
    for (uint32_t reader_index = 0; reader_index < added.readers.size(); ++reader_index) {
        saved_ops[reader_index] = added.readers[reader_index]->op;
        added.readers[reader_index]->op = GGML_OP_NONE;
    }
    const int32_t graph_capacity = first.graph->n_nodes + added.graph->n_nodes;
    const ggml_init_params graph_params = {
        /* .mem_size = */ ggml_tensor_overhead() * 2 + ggml_graph_overhead_custom(graph_capacity, false),
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };
    ggml_context_ptr graph_context(ggml_init(graph_params));
    CHECK(graph_context != nullptr);
    ggml_cgraph * graph = ggml_new_graph_custom(graph_context.get(), graph_capacity, false);
    CHECK(graph != nullptr);
    for (int32_t node_index = 0; node_index < first.graph->n_nodes; ++node_index) {
        ggml_graph_add_node(graph, first.graph->nodes[node_index]);
    }
    ggml_graph_add_node(graph, added.ids->src[0]);
    ggml_graph_add_node(graph, added.ids);
    for (ggml_tensor * reader : added.readers) {
        ggml_graph_add_node(graph, reader);
    }
    candidate_rebuild_graph_uses(graph);
    candidate_stamp_execution(graph, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
        GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 1, 1);
    const uint64_t source_graph_uid = graph->execution_certificate.source_graph_uid;
    std::array<int32_t, 3> saved_use_counts = {};
    for (uint32_t bank_index = 0; bank_index < added.banks.size(); ++bank_index) {
        saved_use_counts[bank_index] = candidate_graph_use_count(graph, added.banks[bank_index]);
        CHECK(saved_use_counts[bank_index] == 1);
    }

    const auto coverage = candidate_certify_graph(*context, graph);
    std::shared_ptr<ggml_cuda_moe_graph_plan> plan;
    ggml_cuda_moe_graph_execution execution;
    CHECK(context->prepare_graph_execution(
        graph, graph->uid, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &plan, &execution,
        coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
        GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED && execution.size() == 1);
    CHECK(execution.find(first.down_output, nullptr) && !execution.find(added.down_output, nullptr));
    const std::shared_ptr<ggml_cuda_moe_graph_plan> stale_plan = plan;
    CHECK(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend.get());
    CHECK(active_grouped_legacy_op_count(backend.get()) == 0);

    for (uint32_t reader_index = 0; reader_index < added.readers.size(); ++reader_index) {
        added.readers[reader_index]->op = saved_ops[reader_index];
    }
    candidate_rebuild_graph_uses(graph);
    candidate_stamp_execution(graph, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
        GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 1, 1, source_graph_uid);
    for (uint32_t bank_index = 0; bank_index < added.banks.size(); ++bank_index) {
        CHECK(candidate_graph_use_count(graph, added.banks[bank_index]) == saved_use_counts[bank_index]);
    }
    CHECK(!context->bind_graph_plan(
        graph, graph->uid, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, *stale_plan, &execution,
        coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint));
    CHECK(execution.size() == 0 && !execution.find(first.down_output, nullptr) && !execution.find(added.down_output, nullptr));
    const auto updated_coverage = candidate_certify_graph(*context, graph);
    CHECK(!context->bind_graph_plan(
        graph, graph->uid, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, *stale_plan, &execution,
        updated_coverage.epoch, updated_coverage.nodes,
        updated_coverage.mmid_count, updated_coverage.mmid_fingerprint));

    const auto first_sentinel = active_grouped_intermediate_sentinel(first);
    const auto added_sentinel = active_grouped_intermediate_sentinel(added);
    ggml_backend_synchronize(backend.get());
    CHECK(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_FAILED);
    ggml_backend_synchronize(backend.get());
    check_active_grouped_intermediates(first, first_sentinel, true);
    check_active_grouped_intermediates(added, added_sentinel, true);
    CHECK(active_grouped_legacy_op_count(backend.get()) == 0);
    for (uint32_t reader_index = 0; reader_index < added.readers.size(); ++reader_index) {
        added.readers[reader_index]->op = GGML_OP_NONE;
    }
    candidate_rebuild_graph_uses(graph);
    candidate_stamp_execution(graph, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
        GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 1, 1, source_graph_uid);
    CHECK(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend.get());
    CHECK(active_grouped_legacy_op_count(backend.get()) == 0);
}

static void test_active_grouped_inventory_reuse() {
    test_active_grouped_inventory_reuse_case(GGML_TYPE_Q4_K, true);
    test_active_grouped_in_place_inventory_reuse_case(true);
    test_active_grouped_in_place_inventory_reuse_case(false);
    fprintf(stderr, "test-moe-cache: complete MMID inventory production seam OK\n");
}

static void check_active_grouped_scale_shadows(
        ggml_cuda_moe_grouped_context & context,
        const ggml_cuda_moe_candidate_group_key & key,
        const active_grouped_dispatch_graph & graph,
        const std::array<int32_t, 2> & experts,
        const std::array<const float *, 3> & expected_pointers) {
    const std::array<ggml_tensor *, 3> scales = {graph.gate_scale, graph.up_scale, graph.down_scale};
    for (uint32_t scale = 0; scale < scales.size(); ++scale) {
        const float * shadow = ggml_cuda_moe_grouped_context_test_access::device_auxiliary_data(context, key, scales[scale]);
        CHECK(shadow != nullptr && shadow == expected_pointers[scale]);
        const auto original = active_grouped_tensor_values(scales[scale]);
        for (int32_t expert : experts) {
            const int32_t slot = ggml_cuda_moe_grouped_context_test_access::device_slot_for_expert(context, key, expert);
            CHECK(slot >= 0);
            float actual = 0.0f;
            CUDA_OK(cudaMemcpy(&actual, shadow + slot, sizeof(actual), cudaMemcpyDeviceToHost));
            CHECK(memcmp(&actual, &original[expert], sizeof(actual)) == 0);
        }
    }
}

void test_active_grouped_nvfp4_scales() {
    const bool old_debug_mm = ggml_backend_cuda_moe_get_debug_mm();
    ggml_backend_cuda_moe_set_debug_mm(true);
    {
        constexpr uint32_t n_slots = 4;
        ggml_backend_ptr reference_backend(ggml_backend_cuda_init(0));
        ggml_backend_ptr candidate_backend(ggml_backend_cuda_init(0));
        CHECK(reference_backend != nullptr && candidate_backend != nullptr);
        auto reference = build_active_grouped_dispatch_graph_types(
            reference_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(),
            {GGML_TYPE_NVFP4, GGML_TYPE_NVFP4, GGML_TYPE_NVFP4},
            GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, false, true, 1);
        auto candidate = build_active_grouped_dispatch_graph_types(
            candidate_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(),
            {GGML_TYPE_NVFP4, GGML_TYPE_NVFP4, GGML_TYPE_NVFP4},
            GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, false, true, 1);
        initialize_active_grouped_dispatch_graphs({&reference, &candidate});
        const auto disabled = candidate_snapshot(n_slots, nullptr, 0);
        CHECK(ggml_backend_cuda_moe_candidate_replace_v1(reference_backend.get(), &disabled) ==
            GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        CHECK(replace_active_grouped_nvfp4_dispatch_v2(candidate_backend.get(), candidate, n_slots) ==
            GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);

        auto * context = ggml_cuda_moe_grouped_context_for_test(candidate_backend.get());
        ggml_cuda_moe_candidate_group_key key;
        CHECK(context != nullptr && context->find_down_group_key(candidate.down, &key));
        check_active_grouped_contract(candidate_backend.get(), candidate, n_slots);
        const std::array<ggml_tensor *, 3> scales = {candidate.gate_scale, candidate.up_scale, candidate.down_scale};
        std::array<const float *, 3> scale_pointers = {};
        for (uint32_t scale = 0; scale < scales.size(); ++scale) {
            scale_pointers[scale] = ggml_cuda_moe_grouped_context_test_access::device_auxiliary_data(
                *context, key, scales[scale]);
            CHECK(scale_pointers[scale] != nullptr);
        }

        ggml_cuda_graph_capture_state_for_test captured = {};
        const std::array<int32_t, 2> first_experts = {
            active_grouped_route(candidate, 0, 0, 0), active_grouped_route(candidate, 0, 0, 1),
        };
        const std::array<int32_t, 2> middle_experts = {
            active_grouped_route(candidate, 1, 0, 0), active_grouped_route(candidate, 1, 0, 1),
        };
        const std::array<int32_t, 2> replacement_experts = {
            active_grouped_route(candidate, 2, 0, 0), active_grouped_route(candidate, 2, 0, 1),
        };
        for (uint32_t pass = 0; pass < 6; ++pass) {
            const uint32_t route_variant = pass >= 4 ? 2 : pass >= 2 ? 1 : 0;
            set_active_grouped_dispatch_logits({&reference, &candidate}, route_variant);
            const auto reference_down_sentinel = active_grouped_tensor_sentinel(reference.down_output);
            const auto candidate_down_sentinel = active_grouped_tensor_sentinel(candidate.down_output);
            const auto expected = run_active_grouped_dispatch(reference_backend.get(), reference, 0, false);
            const auto actual = run_active_grouped_dispatch(
                candidate_backend.get(), candidate, candidate.n_used * (pass + 2), true);
            check_active_grouped_exact_output(expected, actual);
            CHECK(active_grouped_tensor_values(reference.down_output) != reference_down_sentinel);
            CHECK(active_grouped_tensor_values(candidate.down_output) == candidate_down_sentinel);
            for (uint32_t scale = 0; scale < scales.size(); ++scale) {
                CHECK(ggml_cuda_moe_grouped_context_test_access::device_auxiliary_data(
                    *context, key, scales[scale]) == scale_pointers[scale]);
            }

            ggml_cuda_graph_capture_state_for_test state = {};
            CHECK(ggml_cuda_graph_capture_state_query_for_test(candidate_backend.get(), candidate.graph, &state));
            if (pass == 0 || !state.capture_available) {
                CHECK(state.graph == 0 && state.instance == 0 && !state.warmup_complete &&
                    state.moe_resource_fingerprint == 0);
            } else if (pass == 1) {
                CHECK(state.graph != 0 && state.instance != 0 && state.warmup_complete &&
                    state.execution_semantic_key != 0 && state.moe_resource_fingerprint != 0);
                captured = state;
            } else {
                CHECK(state.graph == captured.graph && state.instance == captured.instance &&
                    state.execution_semantic_key == captured.execution_semantic_key &&
                    state.moe_resource_fingerprint == captured.moe_resource_fingerprint);
            }
            if (pass == 0) {
                check_active_grouped_scale_shadows(*context, key, candidate, first_experts, scale_pointers);
            }
            if (pass == 3) {
                check_active_grouped_scale_shadows(*context, key, candidate, middle_experts, scale_pointers);
            }
            if (pass == 5) {
                check_active_grouped_scale_shadows(*context, key, candidate, replacement_experts, scale_pointers);
                for (int32_t expert : grouped_frequency_enabled() ? middle_experts : first_experts) {
                    CHECK(ggml_cuda_moe_grouped_context_test_access::device_slot_for_expert(*context, key, expert) == -1);
                }
            }
        }
        CHECK(active_grouped_legacy_op_count(candidate_backend.get(), true) == 0);
        CHECK(active_grouped_legacy_op_count(candidate_backend.get(), false) == 0);
        const auto telemetry = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
        CHECK(telemetry.registered == 1 && telemetry.covered == 1);
        CHECK(telemetry.plan_calls == 8 && telemetry.plan_compiles == 2 && telemetry.plan_reuses == 6);
        CHECK(telemetry.calls == 7 && telemetry.ready == 7 && telemetry.completed == 7);
        CHECK(telemetry.admitted_banks == 7 * candidate.banks.size());
        CHECK(telemetry.fallback == 0 && telemetry.rollback == 0 &&
            telemetry.prepare_error == 0 && telemetry.finish_error == 0);
        CHECK(telemetry.h2d_banks == 6 * candidate.banks.size());
        uint64_t bytes_per_expert = 0;
        for (const ggml_tensor * bank : candidate.banks) {
            bytes_per_expert += bank->nb[2];
        }
        CHECK(telemetry.h2d_bytes == 6 * bytes_per_expert);
    }

    constexpr uint32_t n_rows = 2;
    for (uint32_t n_slots : {12u, 48u}) {
        ggml_backend_ptr reference_backend(ggml_backend_cuda_init(0));
        ggml_backend_ptr candidate_backend(ggml_backend_cuda_init(0));
        CHECK(reference_backend != nullptr && candidate_backend != nullptr);
        auto reference = build_active_grouped_dispatch_graph_types(
            reference_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(),
            {GGML_TYPE_NVFP4, GGML_TYPE_NVFP4, GGML_TYPE_NVFP4},
            GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, false, true, n_rows);
        auto candidate = build_active_grouped_dispatch_graph_types(
            candidate_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(),
            {GGML_TYPE_NVFP4, GGML_TYPE_NVFP4, GGML_TYPE_NVFP4},
            GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, false, true, n_rows);
        initialize_active_grouped_dispatch_graphs({&reference, &candidate});
        const auto disabled = candidate_snapshot(n_slots, nullptr, 0);
        CHECK(ggml_backend_cuda_moe_candidate_replace_v1(reference_backend.get(), &disabled) ==
            GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        CHECK(replace_active_grouped_nvfp4_dispatch_v2(candidate_backend.get(), candidate, n_slots) ==
            GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);

        auto * context = ggml_cuda_moe_grouped_context_for_test(candidate_backend.get());
        ggml_cuda_moe_candidate_group_key key;
        ggml_cuda_moe_candidate_group_info group_info;
        CHECK(context != nullptr && context->find_down_group_key(candidate.down, &key));
        CHECK(context->get_group(key, &group_info) && group_info.n_banks == 6);
        const std::array<ggml_tensor *, 3> scales = {candidate.gate_scale, candidate.up_scale, candidate.down_scale};
        const std::array<uint32_t, 3> scale_roles = {
            GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_SCALE,
            GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_SCALE,
            GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_SCALE,
        };
        for (uint32_t i = 0; i < candidate.banks.size(); ++i) {
            ggml_cuda_moe_candidate_bank_info bank_info;
            CHECK(context->get_bank(key, candidate.roles[i], &bank_info));
            CHECK(bank_info.tensor == candidate.banks[i] && bank_info.type == GGML_TYPE_NVFP4);
            CHECK(bank_info.encoding == GGML_CUDA_MOE_CANDIDATE_ENCODING_NVFP4_COMPOUND);
            CHECK(bank_info.movement == GGML_CUDA_MOE_CANDIDATE_MOVEMENT_SLOT_BOUND);
            CHECK(bank_info.index_modes == GGML_CUDA_MOE_CANDIDATE_INDEX_GROUP_SLOT_DIRECT);
        }
        for (uint32_t i = 0; i < scales.size(); ++i) {
            ggml_cuda_moe_candidate_bank_info scale_info;
            CHECK(context->get_bank(key, scale_roles[i], &scale_info));
            CHECK(scale_info.tensor == scales[i] && scale_info.type == GGML_TYPE_F32);
            CHECK(scale_info.movement == GGML_CUDA_MOE_CANDIDATE_MOVEMENT_PERMANENT_CANDIDATE);
            CHECK(scale_info.index_modes == GGML_CUDA_MOE_CANDIDATE_INDEX_ORIGINAL_DIRECT);
        }

        check_active_grouped_contract(candidate_backend.get(), candidate, n_slots, false, true);
        std::vector<float> expected;
        std::vector<float> actual;
        ggml_cuda_graph_capture_state_for_test captured = {};
        for (int pass = 0; pass < 4; ++pass) {
            const auto candidate_down_sentinel = active_grouped_tensor_sentinel(candidate.down_output);
            const auto current_expected = run_active_grouped_dispatch(reference_backend.get(), reference, 0, false);
            const auto current_actual = run_active_grouped_dispatch(
                candidate_backend.get(), candidate, n_rows * candidate.n_used * (pass + 2), false);
            check_active_grouped_exact_output(current_expected, current_actual);
            CHECK(active_grouped_tensor_values(candidate.down_output) != candidate_down_sentinel);
            check_active_grouped_routes(reference, n_rows, 0);
            check_active_grouped_routes(candidate, n_rows, 0);
            if (pass == 0) {
                expected = current_expected;
                actual = current_actual;
            } else {
                CHECK(current_expected == expected && current_actual == actual);
            }
            ggml_cuda_graph_capture_state_for_test state = {};
            CHECK(ggml_cuda_graph_capture_state_query_for_test(candidate_backend.get(), candidate.graph, &state));
            if (pass == 0 || !state.capture_available) {
                CHECK(state.graph == 0 && state.instance == 0 && !state.warmup_complete &&
                    state.moe_resource_fingerprint == 0);
            } else if (pass == 1) {
                CHECK(state.graph != 0 && state.instance != 0 && state.warmup_complete &&
                    state.execution_semantic_key != 0 && state.moe_resource_fingerprint != 0);
                captured = state;
            } else {
                CHECK(state.graph == captured.graph && state.instance == captured.instance &&
                    state.execution_semantic_key == captured.execution_semantic_key &&
                    state.moe_resource_fingerprint == captured.moe_resource_fingerprint);
            }
        }
        CHECK(expected.size() == actual.size());
        CHECK(memcmp(expected.data(), actual.data(), expected.size() * sizeof(float)) == 0);
        CHECK(active_grouped_legacy_op_count(reference_backend.get(), true) == 0);
        CHECK(active_grouped_legacy_op_count(reference_backend.get(), false) == 4 * reference.banks.size());
        CHECK(active_grouped_legacy_op_count(candidate_backend.get(), true) == 0);
        CHECK(active_grouped_legacy_op_count(candidate_backend.get(), false) == 0);
        check_active_grouped_debug_telemetry(candidate_backend.get(), candidate);
        check_active_grouped_legacy_caches(reference_backend.get(), reference, false);
    }

    {
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        CHECK(backend != nullptr);
        auto graph = build_active_grouped_dispatch_graph_types(
            backend.get(), ggml_backend_cuda_moe_cached_buffer_type(),
            {GGML_TYPE_NVFP4, GGML_TYPE_NVFP4, GGML_TYPE_NVFP4},
            GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, false, true, n_rows);
        initialize_active_grouped_dispatch_graph(graph, 677);
        CHECK(replace_active_grouped_nvfp4_dispatch_v2(backend.get(), graph, 12) ==
            GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        auto * context = ggml_cuda_moe_grouped_context_for_test(backend.get());
        CHECK(context != nullptr);

        graph.gate_output->ne[2] = n_rows + 1;
        candidate_rebuild_graph_uses(graph.graph);
        auto coverage = candidate_certify_graph(*context, graph.graph);
        std::shared_ptr<ggml_cuda_moe_graph_plan> plan;
        ggml_cuda_moe_graph_execution execution;
        CHECK(context->prepare_graph_execution(
            graph.graph, 850, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &plan, &execution,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR);
        CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_geometry_reason(*plan, 0));
        CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_auxiliary_node_count(*plan, 0) == 0);
        for (uint32_t bank_index = 0; bank_index < 6; ++bank_index) {
            CHECK(!ggml_cuda_moe_grouped_context_test_access::graph_group_has_bank_use(*plan, 0, bank_index));
        }
        const ggml_cuda_moe_graph_plan * geometry_plan = plan.get();
        CHECK(context->prepare_graph_execution(
            graph.graph, 851, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &plan, &execution,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(plan.get() == geometry_plan && execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR);

        graph.gate_output->ne[2] = n_rows;
        candidate_rebuild_graph_uses(graph.graph);
        coverage = candidate_certify_graph(*context, graph.graph);
        CHECK(context->prepare_graph_execution(
            graph.graph, 852, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &plan, &execution,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED);
        CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_auxiliary_node_count(*plan, 0) == 9);

        graph.gate_scale_rows->nb[3]++;
        CHECK(context->prepare_graph_execution(
            graph.graph, 8521, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &plan, &execution,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR);
        graph.gate_scale_rows->nb[3]--;
        CHECK(context->prepare_graph_execution(
            graph.graph, 8522, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &plan, &execution,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED);
        const ggml_cuda_moe_graph_plan * restored_metadata_plan = plan.get();
        CHECK(context->prepare_graph_execution(
            graph.graph, 8523, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &plan, &execution,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(plan.get() == restored_metadata_plan && execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED);

        graph.gate_output->ne[2] = n_rows + 1;
        candidate_rebuild_graph_uses(graph.graph);
        coverage = candidate_certify_graph(*context, graph.graph);
        CHECK(context->prepare_graph_execution(
            graph.graph, 853, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &plan, &execution,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR);
        CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_geometry_reason(*plan, 0));
        CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_auxiliary_node_count(*plan, 0) == 0);
    }

    {
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        CHECK(backend != nullptr);
        auto graph = build_active_grouped_dispatch_graph_types(
            backend.get(), ggml_backend_cuda_moe_cached_buffer_type(),
            {GGML_TYPE_NVFP4, GGML_TYPE_NVFP4, GGML_TYPE_NVFP4},
            GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, false, true, n_rows);
        initialize_active_grouped_dispatch_graph(graph, 689);
        CHECK(replace_active_grouped_nvfp4_dispatch_v2(backend.get(), graph, 12) ==
            GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        auto * context = ggml_cuda_moe_grouped_context_for_test(backend.get());
        CHECK(context != nullptr);

        ggml_tensor * fourth_consumer = ggml_dup(graph.nodes.get(), graph.output);
        fourth_consumer->src[0] = graph.gate_output;
        ggml_graph_add_node(graph.graph, fourth_consumer);
        candidate_rebuild_graph_uses(graph.graph);
        CHECK(candidate_graph_use_count(graph.graph, graph.gate_output) == 2);
        const auto coverage = candidate_certify_graph(*context, graph.graph);
        std::shared_ptr<ggml_cuda_moe_graph_plan> plan;
        ggml_cuda_moe_graph_execution execution;
        CHECK(context->prepare_graph_execution(
            graph.graph, 870, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &plan, &execution,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR && execution.size() == 1 && execution.requires_dispatch());
        CHECK(execution.find_group(graph.gate_output, nullptr) == nullptr);
        const ggml_cuda_moe_graph_plan * fourth_consumer_plan = plan.get();
        CHECK(context->prepare_graph_execution(
            graph.graph, 871, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &plan, &execution,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(plan.get() == fourth_consumer_plan && execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR);
        CHECK(execution.find_group(graph.gate_output, nullptr) == nullptr);

        fourth_consumer->src[0] = graph.output;
        candidate_rebuild_graph_uses(graph.graph);
        CHECK(candidate_graph_use_count(graph.graph, graph.gate_output) == 1);
        CHECK(context->prepare_graph_execution(
            graph.graph, 872, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &plan, &execution,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED && execution.size() == 1);
        CHECK(execution.find_group(graph.gate_output, nullptr) != nullptr);
        const ggml_cuda_moe_graph_plan * exact_consumer_plan = plan.get();
        CHECK(context->prepare_graph_execution(
            graph.graph, 873, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &plan, &execution,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(plan.get() == exact_consumer_plan && execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED);

        fourth_consumer->src[0] = graph.gate_output;
        candidate_rebuild_graph_uses(graph.graph);
        CHECK(context->prepare_graph_execution(
            graph.graph, 874, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &plan, &execution,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR && execution.size() == 1 && execution.requires_dispatch());
        CHECK(execution.find_group(graph.gate_output, nullptr) == nullptr);
    }

    for (int32_t omitted_scale = 0; omitted_scale < 3; ++omitted_scale) {
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        CHECK(backend != nullptr);
        auto graph = build_active_grouped_dispatch_graph_types(
            backend.get(), ggml_backend_cuda_moe_cached_buffer_type(),
            {GGML_TYPE_NVFP4, GGML_TYPE_NVFP4, GGML_TYPE_NVFP4},
            GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, false, true, n_rows);
        initialize_active_grouped_dispatch_graph(graph, 701 + omitted_scale);
        CHECK(replace_active_grouped_nvfp4_dispatch_v2(backend.get(), graph, 12, omitted_scale) ==
            GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        auto * context = ggml_cuda_moe_grouped_context_for_test(backend.get());
        CHECK(context != nullptr);
        const auto coverage = candidate_certify_graph(*context, graph.graph);
        std::shared_ptr<ggml_cuda_moe_graph_plan> plan;
        ggml_cuda_moe_graph_execution execution;
        CHECK(context->prepare_graph_execution(
            graph.graph, 910 + omitted_scale, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &plan, &execution,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR && execution.requires_dispatch());
        CHECK(ggml_backend_graph_compute(backend.get(), graph.graph) == GGML_STATUS_FAILED);
        CHECK(active_grouped_legacy_op_count(backend.get()) == 0);
    }
    ggml_backend_cuda_moe_set_debug_mm(old_debug_mm);
    fprintf(stderr, "test-moe-cache: compound NVFP4 B2 original-ID output scales exact OK\n");
}

static void test_active_grouped_dispatch_generic() {
    for (ggml_type type : {GGML_TYPE_Q3_K, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S}) {
        for (uint32_t layout : {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP}) {
            for (uint32_t n_slots : {12u, 48u}) {
                test_active_grouped_dispatch_case(type, layout, n_slots);
            }
        }
    }
    for (uint32_t n_slots : {12u, 48u}) {
        test_active_grouped_dispatch_types_case(
            {GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_IQ3_S}, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, n_slots);
        test_active_grouped_dispatch_types_case(
            {GGML_TYPE_Q3_K, GGML_TYPE_BF16, GGML_TYPE_IQ3_S}, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, n_slots);
    }
    fprintf(stderr, "test-moe-cache: active grouped Q3 and mixed ordinary/F3 OK\n");
}

static void test_active_grouped_fused_down_scale_extent() {
    const bool old_debug_mm = ggml_backend_cuda_moe_get_debug_mm();
    ggml_backend_cuda_moe_set_debug_mm(true);
    ggml_backend_ptr backend(ggml_backend_cuda_init(0));
    CHECK(backend != nullptr);
    auto graph = build_active_grouped_dispatch_graph(
        backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, true);
    initialize_active_grouped_dispatch_graph(graph, 733);
    register_active_grouped_dispatch(
        backend.get(), graph, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 12);
    auto * context = ggml_cuda_moe_grouped_context_for_test(backend.get());
    CHECK(context != nullptr);

    const auto coverage = candidate_certify_graph(*context, graph.graph);
    std::shared_ptr<ggml_cuda_moe_graph_plan> plan;
    ggml_cuda_moe_graph_execution execution;
    CHECK(context->prepare_graph_execution(
        graph.graph, 880, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &plan, &execution,
        coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
        GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED);

    graph.down_scale_reshape->ne[1]++;
    graph.down_scale_repeat->ne[1]++;
    CHECK(context->prepare_graph_execution(
        graph.graph, 881, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &plan, &execution,
        coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
        GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR);

    graph.down_scale_reshape->ne[1]--;
    graph.down_scale_repeat->ne[1]--;
    CHECK(context->prepare_graph_execution(
        graph.graph, 882, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &plan, &execution,
        coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
        GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED);
    const ggml_cuda_moe_graph_plan * recovered_plan = plan.get();
    CHECK(context->prepare_graph_execution(
        graph.graph, 883, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &plan, &execution,
        coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
        GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
    CHECK(plan.get() == recovered_plan && execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED);
    ggml_backend_cuda_moe_set_debug_mm(old_debug_mm);
}

void test_prefill_resident_biases() {
    const bool old_debug_mm = ggml_backend_cuda_moe_get_debug_mm();
    ggml_backend_cuda_moe_set_debug_mm(true);
    constexpr uint32_t n_slots = 12;
    constexpr uint32_t n_rows = 2;
    constexpr uint32_t n_experts = 128;
    constexpr uint32_t n_used = 8;

    ggml_backend_ptr reference_backend(ggml_backend_cuda_init(0));
    ggml_backend_ptr candidate_backend(ggml_backend_cuda_init(0));
    ggml_backend_ptr parallel_backend(ggml_backend_cuda_init(0));
    ggml_backend_ptr declined_backend(ggml_backend_cuda_init(0));
    CHECK(reference_backend != nullptr && candidate_backend != nullptr &&
        parallel_backend != nullptr && declined_backend != nullptr);
    auto make_graph = [](ggml_backend_t backend, uint32_t rows, uint32_t experts, uint32_t used) {
        return build_active_grouped_dispatch_graph_types(
            backend, ggml_backend_cuda_moe_cached_buffer_type(),
            {GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0},
            GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, false, false,
            rows, experts, used, 64, nullptr, false, true, 96, true);
    };
    auto reference = make_graph(reference_backend.get(), n_rows, n_experts, n_used);
    auto candidate = make_graph(candidate_backend.get(), n_rows, n_experts, n_used);
    auto parallel = make_graph(parallel_backend.get(), n_rows, n_experts, n_used);
    auto declined = make_graph(declined_backend.get(), n_rows, n_experts, n_used);
    for (auto * graph : {&reference, &candidate, &parallel, &declined}) {
        candidate_stamp_execution(graph->graph, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
            GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL, n_rows, 1);
    }
    initialize_active_grouped_dispatch_graphs({&reference, &candidate, &parallel, &declined});

    size_t bias_bytes = 0;
    for (const ggml_tensor * bias : candidate.biases) {
        CHECK(ggml_nbytes(bias) <= SIZE_MAX - bias_bytes);
        bias_bytes += ggml_nbytes(bias);
    }
    CHECK(bias_bytes > 1 && candidate.biases.size() == 3);
    const auto disabled = candidate_snapshot(n_slots, nullptr, 0);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v1(reference_backend.get(), &disabled) ==
        GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v1(candidate_backend.get(), &disabled) ==
        GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v1(parallel_backend.get(), &disabled) ==
        GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v1(declined_backend.get(), &disabled) ==
        GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    auto * candidate_context = ggml_cuda_moe_grouped_context_for_test(candidate_backend.get());
    auto * parallel_context = ggml_cuda_moe_grouped_context_for_test(parallel_backend.get());
    auto * declined_context = ggml_cuda_moe_grouped_context_for_test(declined_backend.get());
    CHECK(candidate_context != nullptr && parallel_context != nullptr && declined_context != nullptr);
    CHECK(ggml_cuda_moe_grouped_context_test_access::set_prefill_resident_budget(*candidate_context, bias_bytes));
    CHECK(ggml_cuda_moe_grouped_context_test_access::set_prefill_resident_budget(*parallel_context, bias_bytes));
    CHECK(ggml_cuda_moe_grouped_context_test_access::set_prefill_resident_budget(*declined_context, bias_bytes - 1));

    register_active_grouped_dispatch(
        candidate_backend.get(), candidate, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, n_slots);
    register_active_grouped_dispatch(
        parallel_backend.get(), parallel, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, n_slots);
    register_active_grouped_dispatch(
        declined_backend.get(), declined, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, n_slots);
    const auto candidate_coverage = candidate_certify_graph(*candidate_context, candidate.graph);
    (void) candidate_certify_graph(*parallel_context, parallel.graph);
    (void) candidate_certify_graph(*declined_context, declined.graph);
    std::shared_ptr<ggml_cuda_moe_graph_plan> prefill_plan;
    ggml_cuda_moe_graph_execution prefill_execution;
    CHECK(candidate_context->prepare_graph_execution(
        candidate.graph, 990, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &prefill_plan, &prefill_execution,
        candidate_coverage.epoch, candidate_coverage.nodes, candidate_coverage.mmid_count,
        candidate_coverage.mmid_fingerprint) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
    CHECK(prefill_plan->outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_PREFILL_LEGACY);
    CHECK(prefill_plan->has_certified_complete_mmid_inventory());
    CHECK(ggml_cuda_moe_grouped_context_test_access::prefill_add_id_witness_count(*prefill_plan) == candidate.biases.size());
    for (ggml_tensor * reader : candidate.readers) {
        CHECK(prefill_execution.find_group(reader, nullptr) != nullptr);
    }

    const auto expected = run_active_grouped_dispatch(reference_backend.get(), reference, 0, false);
    std::array<std::vector<float>, 2> parallel_outputs;
    std::thread first([&]() {
        parallel_outputs[0] = run_active_grouped_dispatch(candidate_backend.get(), candidate, 0, false);
    });
    std::thread second([&]() {
        parallel_outputs[1] = run_active_grouped_dispatch(parallel_backend.get(), parallel, 0, false);
    });
    first.join();
    second.join();
    check_active_grouped_exact_output(expected, parallel_outputs[0]);
    check_active_grouped_exact_output(expected, parallel_outputs[1]);
    check_active_grouped_exact_output(
        expected, run_active_grouped_dispatch(declined_backend.get(), declined, 0, false));
    check_active_grouped_routes(candidate, n_rows, 0);
    CHECK(active_grouped_route(candidate, 0, 0, n_used - 1) >= static_cast<int32_t>(n_slots));

    ggml_cuda_moe_candidate_group_key candidate_key;
    ggml_cuda_moe_candidate_group_key parallel_key;
    ggml_cuda_moe_candidate_group_key declined_key;
    CHECK(candidate_context->find_down_group_key(candidate.down, &candidate_key));
    CHECK(parallel_context->find_down_group_key(parallel.down, &parallel_key));
    CHECK(declined_context->find_down_group_key(declined.down, &declined_key));
    std::array<const float *, 3> candidate_residents = {};
    uint64_t first_resource_generation = 0;
    for (uint32_t i = 0; i < candidate.biases.size(); ++i) {
        size_t extent = 0;
        uint64_t generation = 0;
        candidate_residents[i] = ggml_cuda_moe_grouped_context_test_access::device_prefill_auxiliary_data(
            *candidate_context, candidate_key, candidate.biases[i], &extent, &generation);
        CHECK(candidate_residents[i] != nullptr && extent == ggml_nbytes(candidate.biases[i]) && generation != 0);
        CHECK(candidate_residents[i] != ggml_cuda_moe_grouped_context_test_access::device_auxiliary_data(
            *candidate_context, candidate_key, candidate.biases[i]));
        if (i == 0) {
            first_resource_generation = generation;
        } else {
            CHECK(generation == first_resource_generation);
        }
        std::vector<float> resident(ggml_nelements(candidate.biases[i]));
        CUDA_OK(cudaMemcpy(resident.data(), candidate_residents[i], extent, cudaMemcpyDeviceToHost));
        CHECK(resident == active_grouped_tensor_values(candidate.biases[i]));

        const float * parallel_resident = ggml_cuda_moe_grouped_context_test_access::device_prefill_auxiliary_data(
            *parallel_context, parallel_key, parallel.biases[i]);
        CHECK(parallel_resident != nullptr && parallel_resident != candidate_residents[i]);
        CHECK(ggml_cuda_moe_grouped_context_test_access::device_prefill_auxiliary_data(
            *declined_context, declined_key, declined.biases[i]) == nullptr);
    }

    std::array<const ggml_tensor *, 3> add_nodes = {};
    for (int32_t node_index = 0; node_index < ggml_graph_n_nodes(candidate.graph); ++node_index) {
        const ggml_tensor * node = ggml_graph_node(candidate.graph, node_index);
        if (node->op != GGML_OP_ADD_ID) {
            continue;
        }
        for (uint32_t bias = 0; bias < candidate.biases.size(); ++bias) {
            if (node->src[1] == candidate.biases[bias]) {
                CHECK(add_nodes[bias] == nullptr);
                add_nodes[bias] = node;
            }
        }
    }
    CHECK(std::all_of(add_nodes.begin(), add_nodes.end(), [](const ggml_tensor * node) { return node != nullptr; }));
    const uint32_t routed_expert = active_grouped_route(candidate, 0, 0, 0);
    const size_t first_row_bytes = candidate.biases[0]->ne[0] * sizeof(float);
    const size_t second_row_bytes = candidate.biases[1]->ne[0] * sizeof(float);
    std::vector<float> first_expected(candidate.biases[0]->ne[0]);
    std::vector<float> second_expected(candidate.biases[1]->ne[0]);
    const auto first_original = active_grouped_tensor_values(candidate.biases[0]);
    const auto second_original = active_grouped_tensor_values(candidate.biases[1]);
    memcpy(first_expected.data(), first_original.data() + routed_expert * candidate.biases[0]->ne[0], first_row_bytes);
    memcpy(second_expected.data(), second_original.data() + routed_expert * candidate.biases[1]->ne[0], second_row_bytes);
    CHECK(std::any_of(first_expected.begin(), first_expected.end(), [](float value) { return value != 0.0f; }));

    cudaStream_t first_stream = nullptr;
    cudaStream_t second_stream = nullptr;
    float * staged = nullptr;
    float * first_result = nullptr;
    float * second_result = nullptr;
    CUDA_OK(cudaStreamCreateWithFlags(&first_stream, cudaStreamNonBlocking));
    CUDA_OK(cudaStreamCreateWithFlags(&second_stream, cudaStreamNonBlocking));
    CHECK(first_stream != second_stream);
    CUDA_OK(cudaMalloc(&staged, first_row_bytes));
    CUDA_OK(cudaMalloc(&first_result, first_row_bytes));
    CUDA_OK(cudaMalloc(&second_result, second_row_bytes));
    uint64_t waits_before = 0;
    uint64_t declines_before = 0;
    CHECK(ggml_cuda_moe_grouped_context_test_access::prefill_auxiliary_ordering(
        *candidate_context, candidate_key, &waits_before, &declines_before));
    CHECK(candidate_context->begin_graph_dispatch(&prefill_execution, GGML_CUDA_MOE_GRAPH_DISPATCH_LEGACY));
    const float * first_source = nullptr;
    const float * second_source = nullptr;
    CHECK(candidate_context->prefill_add_id_source(
        prefill_execution, add_nodes[0], first_stream, &first_source));
    CHECK(first_source == candidate_residents[0]);
    CHECK(!candidate_context->prefill_add_id_source(
        prefill_execution, add_nodes[1], second_stream, &second_source));
    CHECK(second_source == nullptr);
    CUDA_OK(cudaMemsetAsync(staged, 0, first_row_bytes, first_stream));
    host_barrier first_barrier;
    CUDA_OK(cudaLaunchHostFunc(first_stream, wait_on_host_barrier, &first_barrier));
    CUDA_OK(cudaMemcpyAsync(staged, first_source + routed_expert * candidate.biases[0]->ne[0],
        first_row_bytes, cudaMemcpyDeviceToDevice, first_stream));
    CHECK(candidate_context->finish_prefill_add_id(
        prefill_execution, add_nodes[0], first_stream, first_source));
    while (!first_barrier.entered.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    CHECK(candidate_context->prefill_add_id_source(
        prefill_execution, add_nodes[1], second_stream, &second_source));
    CHECK(second_source == candidate_residents[1]);
    CUDA_OK(cudaMemcpyAsync(first_result, staged, first_row_bytes, cudaMemcpyDeviceToDevice, second_stream));
    CUDA_OK(cudaMemcpyAsync(second_result, second_source + routed_expert * candidate.biases[1]->ne[0],
        second_row_bytes, cudaMemcpyDeviceToDevice, second_stream));
    CHECK(candidate_context->finish_prefill_add_id(
        prefill_execution, add_nodes[1], second_stream, second_source));
    CHECK(candidate_context->finish_graph_dispatch(&prefill_execution));
    uint64_t waits_after = 0;
    uint64_t declines_after = 0;
    CHECK(ggml_cuda_moe_grouped_context_test_access::prefill_auxiliary_ordering(
        *candidate_context, candidate_key, &waits_after, &declines_after));
    CHECK(waits_after == waits_before + 2 && declines_after == declines_before + 1);
    first_barrier.released.store(true, std::memory_order_release);
    CUDA_OK(cudaStreamSynchronize(second_stream));
    CUDA_OK(cudaStreamSynchronize(first_stream));
    std::vector<float> first_actual(first_expected.size());
    std::vector<float> second_actual(second_expected.size());
    CUDA_OK(cudaMemcpy(first_actual.data(), first_result, first_row_bytes, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(second_actual.data(), second_result, second_row_bytes, cudaMemcpyDeviceToHost));
    CHECK(first_actual == first_expected && second_actual == second_expected);
    CUDA_OK(cudaFree(second_result));
    CUDA_OK(cudaFree(first_result));
    CUDA_OK(cudaFree(staged));
    CUDA_OK(cudaStreamDestroy(second_stream));
    CUDA_OK(cudaStreamDestroy(first_stream));
    check_active_grouped_exact_output(
        expected, run_active_grouped_dispatch(candidate_backend.get(), candidate, 0, false));

    for (uint32_t i = 0; i < candidate.biases.size(); ++i) {
        const auto changed = cached_fusion_test_data(candidate.biases[i], 900 + i);
        ggml_backend_tensor_set(candidate.biases[i], changed.data(), 0, changed.size());
    }
    check_active_grouped_exact_output(
        expected, run_active_grouped_dispatch(candidate_backend.get(), candidate, 0, false));
    check_active_grouped_exact_output(
        expected, run_active_grouped_dispatch(parallel_backend.get(), parallel, 0, false));

    for (uint32_t i = 0; i < declined.biases.size(); ++i) {
        const auto changed = cached_fusion_test_data(declined.biases[i], 950 + i);
        ggml_backend_tensor_set(declined.biases[i], changed.data(), 0, changed.size());
        ggml_backend_tensor_set(reference.biases[i], changed.data(), 0, changed.size());
    }
    const auto declined_expected = run_active_grouped_dispatch(reference_backend.get(), reference, 0, false);
    const auto declined_actual = run_active_grouped_dispatch(declined_backend.get(), declined, 0, false);
    check_active_grouped_exact_output(declined_expected, declined_actual);
    CHECK(declined_actual != expected);

    for (uint32_t i = 0; i < candidate.biases.size(); ++i) {
        const auto changed = cached_fusion_test_data(candidate.biases[i], 900 + i);
        ggml_backend_tensor_set(reference.biases[i], changed.data(), 0, changed.size());
    }
    register_active_grouped_dispatch(
        candidate_backend.get(), candidate, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, n_slots);
    (void) candidate_certify_graph(*candidate_context, candidate.graph);
    ggml_cuda_moe_candidate_group_key refreshed_key;
    CHECK(candidate_context->find_down_group_key(candidate.down, &refreshed_key));
    CHECK(refreshed_key.generation != candidate_key.generation);
    const auto refreshed_expected = run_active_grouped_dispatch(reference_backend.get(), reference, 0, false);
    const auto refreshed_actual = run_active_grouped_dispatch(candidate_backend.get(), candidate, 0, false);
    check_active_grouped_exact_output(refreshed_expected, refreshed_actual);
    CHECK(refreshed_actual != expected);
    size_t refreshed_extent = 0;
    uint64_t refreshed_generation = 0;
    CHECK(ggml_cuda_moe_grouped_context_test_access::device_prefill_auxiliary_data(
        *candidate_context, candidate_key, candidate.biases[0]) == nullptr);
    CHECK(ggml_cuda_moe_grouped_context_test_access::device_prefill_auxiliary_data(
        *candidate_context, refreshed_key, candidate.biases[0], &refreshed_extent, &refreshed_generation) != nullptr);
    CHECK(refreshed_extent == ggml_nbytes(candidate.biases[0]) && refreshed_generation != first_resource_generation);

    {
        ggml_backend_ptr decode_reference_backend(ggml_backend_cuda_init(0));
        ggml_backend_ptr decode_candidate_backend(ggml_backend_cuda_init(0));
        CHECK(decode_reference_backend != nullptr && decode_candidate_backend != nullptr);
        auto decode_reference = make_graph(decode_reference_backend.get(), 1, 8, 2);
        auto decode_candidate = make_graph(decode_candidate_backend.get(), 1, 8, 2);
        initialize_active_grouped_dispatch_graphs({&decode_reference, &decode_candidate});
        const auto decode_disabled = candidate_snapshot(3, nullptr, 0);
        CHECK(ggml_backend_cuda_moe_candidate_replace_v1(decode_reference_backend.get(), &decode_disabled) ==
            GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        register_active_grouped_dispatch(
            decode_candidate_backend.get(), decode_candidate, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 3);
        auto * decode_context = ggml_cuda_moe_grouped_context_for_test(decode_candidate_backend.get());
        CHECK(decode_context != nullptr);
        (void) candidate_certify_graph(*decode_context, decode_candidate.graph);
        const auto decode_expected = run_active_grouped_dispatch(decode_reference_backend.get(), decode_reference, 0, false);
        const auto decode_actual = run_active_grouped_dispatch(
            decode_candidate_backend.get(), decode_candidate, decode_candidate.n_used, true);
        check_active_grouped_exact_output(decode_expected, decode_actual);
        ggml_cuda_moe_candidate_group_key decode_key;
        CHECK(decode_context->find_down_group_key(decode_candidate.down, &decode_key));
        for (ggml_tensor * bias : decode_candidate.biases) {
            const float * resident = ggml_cuda_moe_grouped_context_test_access::device_prefill_auxiliary_data(
                *decode_context, decode_key, bias);
            const float * shadow = ggml_cuda_moe_grouped_context_test_access::device_auxiliary_data(
                *decode_context, decode_key, bias);
            CHECK(resident == nullptr && shadow != nullptr);
        }
    }

    CHECK(ggml_backend_graph_compute(candidate_backend.get(), candidate.graph) == GGML_STATUS_SUCCESS);
    candidate_context->shutdown();
    ggml_backend_synchronize(candidate_backend.get());
    CHECK(ggml_cuda_moe_grouped_context_test_access::device_prefill_auxiliary_data(
        *candidate_context, refreshed_key, candidate.biases[0]) == nullptr);
    check_active_grouped_exact_output(
        expected, run_active_grouped_dispatch(parallel_backend.get(), parallel, 0, false));
    ggml_backend_cuda_moe_set_debug_mm(old_debug_mm);
    fprintf(stderr, "test-moe-cache: bounded prefill resident F32 biases exact OK\n");
}

void test_active_grouped_dispatch() {
    test_prefill_resident_biases();
    test_active_grouped_materialization_eligibility();
    test_active_grouped_dispatch_case(
        GGML_TYPE_Q4_0, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 12, false, true);
    test_active_grouped_dispatch_case(
        GGML_TYPE_Q4_0, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 48);
    test_active_grouped_dispatch_case(
        GGML_TYPE_Q4_K, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 12);
    test_active_grouped_dispatch_case(
        GGML_TYPE_Q4_K, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 48);
    test_active_grouped_dispatch_case(
        GGML_TYPE_Q4_0, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 12);
    test_active_grouped_dispatch_case(
        GGML_TYPE_Q4_0, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 48);
    test_active_grouped_dispatch_case(
        GGML_TYPE_Q4_K, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 12);
    test_active_grouped_dispatch_case(
        GGML_TYPE_Q4_K, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 48);
    test_active_grouped_dispatch_case(
        GGML_TYPE_Q4_0, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 12, true);
    test_active_grouped_dispatch_types_case(
        {GGML_TYPE_MXFP4, GGML_TYPE_MXFP4, GGML_TYPE_MXFP4},
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 24, false, false, true);
    test_active_grouped_dispatch_types_case(
        {GGML_TYPE_MXFP4, GGML_TYPE_MXFP4, GGML_TYPE_MXFP4},
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 24, false, false, true, 2);
    test_active_grouped_dispatch_types_case(
        {GGML_TYPE_MXFP4, GGML_TYPE_MXFP4, GGML_TYPE_MXFP4},
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 24, false, false, true);
    test_active_grouped_dispatch_types_case(
        {GGML_TYPE_MXFP4, GGML_TYPE_MXFP4, GGML_TYPE_MXFP4},
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 3, false, true, true, 1, 512, true);
    fprintf(stderr, "test-moe-cache: active grouped MXFP4 separate/fused output biases exact OK\n");
    test_active_grouped_fused_down_scale_extent();
    fprintf(stderr, "test-moe-cache: Gemma fused original-direct down scale exact OK\n");
    test_active_grouped_dispatch_staged_legacy();
    test_active_grouped_empty_policy();
    test_active_grouped_inventory_reuse();
    test_active_grouped_nvfp4_scales();
    fprintf(stderr, "test-moe-cache: active grouped Q4 ordinary/F3 OK\n");
    test_active_grouped_dispatch_generic();
}
