#include "test-moe-cache.h"

static std::vector<uint8_t> active_grouped_q4k_expert_data(const ggml_tensor * tensor, uint32_t salt) {
    CHECK(tensor->type == GGML_TYPE_Q4_K && tensor->ne[0] > 0 && tensor->ne[1] > 0 && tensor->ne[2] > 0);
    CHECK(tensor->nb[2] == ggml_row_size(tensor->type, tensor->ne[0]) * tensor->ne[1] &&
        ggml_nbytes(tensor) == tensor->nb[2] * tensor->ne[2]);
    std::vector<uint8_t> bytes(ggml_nbytes(tensor));
    std::vector<float> values(tensor->ne[0] * tensor->ne[1]);
    std::vector<uint64_t> fingerprints(tensor->ne[2]);
    std::unordered_set<uint64_t> unique_fingerprints;
    for (uint32_t expert = 0; expert < tensor->ne[2]; ++expert) {
        uint32_t state = 0x9e3779b9u * (expert + 1) ^ 0x85ebca6bu * (salt + 1);
        for (uint32_t value = 0; value < values.size(); ++value) {
            state = 1664525u * state + 1013904223u;
            const int32_t sample = static_cast<int32_t>((state >> 8) & 0xffff) - 32768;
            values[value] = sample * (0.5f + 0.01f * ((expert + salt) % 23)) / 32768.0f;
        }
        uint8_t * slab = bytes.data() + expert * tensor->nb[2];
        CHECK(ggml_quantize_chunk(
            tensor->type, values.data(), slab, 0, tensor->ne[1], tensor->ne[0], nullptr) == tensor->nb[2]);
        uint64_t fingerprint = 1469598103934665603ULL;
        for (size_t byte = 0; byte < tensor->nb[2]; ++byte) {
            fingerprint = (fingerprint ^ slab[byte]) * 1099511628211ULL;
        }
        fingerprints[expert] = fingerprint;
        CHECK(unique_fingerprints.insert(fingerprint).second);
        if (expert >= 17) {
            CHECK(fingerprints[expert] != fingerprints[expert - 17]);
        }
    }
    CHECK(unique_fingerprints.size() == static_cast<size_t>(tensor->ne[2]));
    return bytes;
}

void test_grouped_graph_replay_lifecycle(int device, size_t host_budget, size_t expected_host_nodes, uint32_t n_dim) {
    const bool old_debug_mm = ggml_backend_cuda_moe_get_debug_mm();
    ggml_backend_cuda_moe_set_debug_mm(true);
    ggml_backend_ptr backend(ggml_backend_cuda_init(device));
    CHECK(backend != nullptr);
    auto buft = ggml_backend_cuda_moe_cached_bounded_buffer_type(host_budget);
    CHECK(buft != nullptr);
    auto graph = build_active_grouped_dispatch_graph(
        backend.get(), buft, GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, false, 1, 8, 2, n_dim);
    ggml_backend_cuda_moe_cached_free_buffer_type(buft);
    initialize_active_grouped_dispatch_graph(graph, 177);
    if (host_budget != 0) {
        const ggml_backend_moe_candidate_group_v2 group = {
            GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY, 0, 0,
        };
        std::vector<ggml_backend_moe_candidate_tensor_v2> tensors;
        for (uint32_t i = 0; i < graph.banks.size(); ++i) {
            tensors.push_back({graph.banks[i], 0, graph.roles[i], GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE, 0, 0});
        }
        const auto sources = candidate_snapshot_v2(12, &group, 1, tensors.data(), tensors.size());
        CHECK(ggml_backend_cuda_moe_cached_configure_sources(buft, &sources));
    }
    register_active_grouped_dispatch(
        backend.get(), graph, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 12);
    auto * context = ggml_cuda_moe_grouped_context_for_test(backend.get());
    CHECK(context != nullptr);
    const int32_t routes[] = {3, 5};
    ggml_backend_tensor_set(graph.ids, routes, 0, sizeof(routes));
    ggml_backend_synchronize(backend.get());
    const auto check_payload = [&](const int32_t * remapped, const int32_t * experts) {
        CHECK(remapped != nullptr);
        std::vector<int32_t> slots(graph.n_used);
        CUDA_OK(cudaMemcpy(slots.data(), remapped, slots.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
        ggml_cuda_moe_candidate_group_key key;
        CHECK(context->find_down_group_key(graph.down, &key));
        for (const auto * bank : graph.banks) {
            const auto * data = static_cast<const char *>(ggml_cuda_moe_grouped_context_test_access::device_bank_data(*context, key, bank));
            CHECK(data != nullptr);
            std::vector<uint8_t> bytes(bank->nb[2]);
            for (uint32_t route = 0; route < graph.n_used; ++route) {
                const int32_t slot = slots[route];
                CHECK(slot >= 0 && slot < 12);
                CUDA_OK(cudaMemcpy(bytes.data(), data + slot * bank->nb[2], bytes.size(), cudaMemcpyDeviceToHost));
                CHECK(memcmp(bytes.data(), static_cast<const char *>(bank->data) + experts[route] * bank->nb[2], bytes.size()) == 0);
            }
        }
    };

    cudaStream_t stream = nullptr;
    cudaStream_t previous_stream = nullptr;
    CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_OK(cudaStreamCreateWithFlags(&previous_stream, cudaStreamNonBlocking));
    std::shared_ptr<ggml_cuda_moe_graph_plan> plan;
    auto coverage = candidate_certify_graph(*context, graph.graph);
    const auto prepare_execution = [&](ggml_cuda_moe_graph_execution & execution, ggml_cuda_moe_graph_property_hint hint, cudaStream_t target) {
        CHECK(context->prepare_graph_execution(
            graph.graph, 817, hint, &plan, &execution,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) !=
            GGML_CUDA_MOE_GRAPH_PREPARE_UNAVAILABLE);
        CHECK(execution.resolve_streams(candidate_test_graph_stream, target));
    };
    const auto run_callbacks = [&](ggml_cuda_moe_graph_execution & execution, cudaStream_t target) {
        ggml_cuda_moe_graph_binding first_binding;
        auto * group = execution.find_group(graph.readers.front(), &first_binding);
        CHECK(group != nullptr);
        CHECK(context->prepare_graph_group(group, first_binding, graph.readers.front(), target) ==
            GGML_CUDA_MOE_GROUPED_DECODE_READY);
        const auto acquisition = group->transaction.acquisition;
        ggml_cuda_moe_graph_binding last_binding;
        CHECK(execution.find_group(graph.readers.back(), &last_binding) == group);
        CHECK(context->finish_graph_group(group, last_binding, graph.readers.back(), target));
        return acquisition;
    };

    ggml_cuda_moe_graph_execution direct;
    prepare_execution(direct, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, previous_stream);
    CHECK(context->begin_graph_dispatch(&direct, GGML_CUDA_MOE_GRAPH_DISPATCH_DIRECT));
    CHECK(direct.dispatch_mode() == GGML_CUDA_MOE_GRAPH_DISPATCH_DIRECT);
    run_callbacks(direct, previous_stream);
    CHECK(context->finish_graph_dispatch(&direct));

    ggml_cuda_moe_graph_execution capture;
    prepare_execution(capture, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, stream);
    uint64_t first_fingerprint = 0;
    std::vector<std::shared_ptr<void>> capture_leases;
    CHECK(context->graph_resource_fingerprint(capture, stream, &first_fingerprint, &capture_leases) && first_fingerprint != 0);
    CHECK(context->begin_graph_dispatch(&capture, GGML_CUDA_MOE_GRAPH_DISPATCH_CAPTURE));
    CHECK(context->activate_graph_resources(
        &capture, GGML_CUDA_MOE_GRAPH_DISPATCH_CAPTURE, first_fingerprint));
    cudaGraph_t captured_graph = nullptr;
    cudaGraphExec_t captured_instance = nullptr;
    CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
    const auto capture_acquisition = run_callbacks(capture, stream);
    CUDA_OK(cudaStreamEndCapture(stream, &captured_graph));
    size_t captured_node_count = 0;
    CUDA_OK(cudaGraphGetNodes(captured_graph, nullptr, &captured_node_count));
    std::vector<cudaGraphNode_t> captured_nodes(captured_node_count);
    CUDA_OK(cudaGraphGetNodes(captured_graph, captured_nodes.data(), &captured_node_count));
    size_t host_nodes = 0;
    for (cudaGraphNode_t node : captured_nodes) {
        cudaGraphNodeType type = cudaGraphNodeTypeCount;
        CUDA_OK(cudaGraphNodeGetType(node, &type));
        CHECK(type != cudaGraphNodeTypeWaitEvent && type != cudaGraphNodeTypeEventRecord);
        host_nodes += type == cudaGraphNodeTypeHost;
    }
    CHECK(host_nodes == expected_host_nodes);
    CUDA_OK(cudaGraphInstantiate(&captured_instance, captured_graph, nullptr, nullptr, 0));
    CUDA_OK(cudaGraphLaunch(captured_instance, stream));
    auto * capture_group = capture.find_group(graph.readers.front(), nullptr);
    CHECK(capture_group != nullptr && capture_group->state == GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_ACTIVE &&
        capture_group->defer_completion);
    const int32_t * capture_remapped = capture_group->remapped_ids;
    ggml_cuda_moe_grouped_resource_info capture_info;
    CHECK(context->get_group_resources(capture_acquisition, &capture_info) && capture_info.transaction_active);
    CHECK(context->finish_graph_dispatch(&capture));
    CHECK(context->get_group_resources(capture_acquisition, &capture_info) && !capture_info.transaction_active);
    CUDA_OK(cudaStreamSynchronize(stream));
    check_payload(capture_remapped, routes);

    uint64_t captured_fingerprint = 0;
    CHECK(context->graph_resource_fingerprint(capture, stream, &captured_fingerprint));
    CHECK(captured_fingerprint == first_fingerprint);
    std::vector<std::weak_ptr<void>> capture_witnesses;
    capture_witnesses.reserve(capture_leases.size());
    for (const auto & lease : capture_leases) {
        capture_witnesses.emplace_back(lease);
    }
    capture_leases.clear();
    CHECK(capture_witnesses.size() == 1 && !capture_witnesses[0].expired());

    int32_t * replay_routes = nullptr;
    CUDA_OK(cudaMallocHost(&replay_routes, 2 * sizeof(int32_t)));
    replay_routes[0] = 0;
    replay_routes[1] = 7;
    host_barrier direct_barrier;
    CUDA_OK(cudaLaunchHostFunc(previous_stream, wait_on_host_barrier, &direct_barrier));
    ggml_cuda_moe_graph_execution intervening_direct;
    prepare_execution(intervening_direct, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, previous_stream);
    CHECK(context->begin_graph_dispatch(&intervening_direct, GGML_CUDA_MOE_GRAPH_DISPATCH_DIRECT));
    run_callbacks(intervening_direct, previous_stream);
    CHECK(context->finish_graph_dispatch(&intervening_direct));
    while (!direct_barrier.entered.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    ggml_cuda_moe_graph_execution replay;
    prepare_execution(replay, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, stream);
    CHECK(context->begin_graph_dispatch(&replay, GGML_CUDA_MOE_GRAPH_DISPATCH_REPLAY));
    CHECK(context->activate_graph_resources(
        &replay, GGML_CUDA_MOE_GRAPH_DISPATCH_REPLAY, captured_fingerprint, &capture_witnesses));
    CUDA_OK(cudaMemcpyAsync(graph.ids->data, replay_routes, 2 * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaGraphLaunch(captured_instance, stream));
    cudaEvent_t replay_done = nullptr;
    CUDA_OK(cudaEventCreateWithFlags(&replay_done, cudaEventDisableTiming));
    CUDA_OK(cudaEventRecord(replay_done, stream));
    auto * replay_group = replay.find_group(graph.readers.front(), nullptr);
    CHECK(replay_group != nullptr && replay_group->state == GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_REPLAY);
    ggml_cuda_moe_grouped_resource_info replay_info;
    CHECK(context->get_group_resources(replay_group->transaction.acquisition, &replay_info));
    CHECK(replay_info.transaction_active);
    bool replay_reached = false;
    for (int attempt = 0; attempt < 100; ++attempt) {
        const cudaError_t query = cudaEventQuery(replay_done);
        CHECK(query == cudaSuccess || query == cudaErrorNotReady);
        if (query == cudaSuccess) {
            replay_reached = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const bool replay_waited_for_direct = !replay_reached;
    direct_barrier.released.store(true, std::memory_order_release);
    CUDA_OK(cudaStreamSynchronize(stream));
    check_payload(replay_group->remapped_ids, replay_routes);
    CUDA_OK(cudaFreeHost(replay_routes));

    std::atomic<bool> replacement_started{false};
    std::atomic<bool> replacement_done{false};
    std::thread replacement([&]() {
        replacement_started.store(true, std::memory_order_release);
        register_active_grouped_dispatch(
            backend.get(), graph, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 12);
        replacement_done.store(true, std::memory_order_release);
    });
    while (!replacement_started.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    while (!ggml_cuda_moe_grouped_context_test_access::admission_closed(*context)) {
        std::this_thread::yield();
    }
    CHECK(!replacement_done.load(std::memory_order_acquire));
    CHECK(context->finish_graph_dispatch(&replay));
    replacement.join();
    CHECK(replacement_done.load(std::memory_order_acquire));
    CHECK(capture_witnesses[0].expired());
    CUDA_OK(cudaStreamSynchronize(stream));
    CUDA_OK(cudaStreamSynchronize(previous_stream));
    CHECK(replay_waited_for_direct && cudaEventQuery(replay_done) == cudaSuccess);
    CUDA_OK(cudaEventDestroy(replay_done));
    CUDA_OK(cudaGraphExecDestroy(captured_instance));
    CUDA_OK(cudaGraphDestroy(captured_graph));

    const auto replay_telemetry = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
    CHECK(replay_telemetry.calls == 4 && replay_telemetry.ready == 4 && replay_telemetry.completed == 4);
    CHECK(replay_telemetry.admitted_banks == 4 * graph.banks.size());
    CHECK(replay_telemetry.prepare_error == 0 && replay_telemetry.finish_error == 0);

    coverage = candidate_certify_graph(*context, graph.graph);
    plan.reset();
    ggml_cuda_moe_graph_execution replacement_direct;
    prepare_execution(replacement_direct, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, stream);
    CHECK(context->begin_graph_dispatch(&replacement_direct, GGML_CUDA_MOE_GRAPH_DISPATCH_DIRECT));
    run_callbacks(replacement_direct, stream);
    CHECK(context->finish_graph_dispatch(&replacement_direct));
    CUDA_OK(cudaStreamSynchronize(stream));
    uint64_t replacement_fingerprint = 0;
    std::vector<std::shared_ptr<void>> replacement_leases;
    CHECK(context->graph_resource_fingerprint(
        replacement_direct, stream, &replacement_fingerprint, &replacement_leases));
    CHECK(replacement_fingerprint != captured_fingerprint);
    std::vector<std::weak_ptr<void>> replacement_witnesses;
    replacement_witnesses.reserve(replacement_leases.size());
    for (const auto & lease : replacement_leases) {
        replacement_witnesses.emplace_back(lease);
    }
    replacement_leases.clear();
    CHECK(replacement_witnesses.size() == 1 && !replacement_witnesses[0].expired());

    ggml_cuda_moe_graph_execution replacement_capture;
    prepare_execution(replacement_capture, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, stream);
    CHECK(context->begin_graph_dispatch(&replacement_capture, GGML_CUDA_MOE_GRAPH_DISPATCH_CAPTURE));
    CHECK(context->activate_graph_resources(
        &replacement_capture, GGML_CUDA_MOE_GRAPH_DISPATCH_CAPTURE, replacement_fingerprint));
    const auto replacement_acquisition = run_callbacks(replacement_capture, stream);
    CHECK(context->finish_graph_dispatch(&replacement_capture));
    CUDA_OK(cudaStreamSynchronize(stream));
    CHECK(ggml_cuda_moe_grouped_context_test_access::set_clock_bound(
        *context, replacement_acquisition, UINT64_MAX));

    ggml_cuda_moe_graph_execution overflow_replay;
    prepare_execution(overflow_replay, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, stream);
    CHECK(context->begin_graph_dispatch(&overflow_replay, GGML_CUDA_MOE_GRAPH_DISPATCH_REPLAY));
    CHECK(!context->activate_graph_resources(
        &overflow_replay, GGML_CUDA_MOE_GRAPH_DISPATCH_REPLAY, replacement_fingerprint, &replacement_witnesses));
    CHECK(context->finish_graph_dispatch(&overflow_replay));
    CHECK(context->begin_graph_dispatch(&overflow_replay, GGML_CUDA_MOE_GRAPH_DISPATCH_DIRECT));
    run_callbacks(overflow_replay, stream);
    CHECK(context->finish_graph_dispatch(&overflow_replay));
    CUDA_OK(cudaStreamSynchronize(stream));
    uint64_t refreshed_fingerprint = 0;
    CHECK(context->graph_resource_fingerprint(overflow_replay, stream, &refreshed_fingerprint));
    CHECK(refreshed_fingerprint != replacement_fingerprint);
    CHECK(replacement_witnesses[0].expired());

    ggml_cuda_moe_candidate_group_key key;
    uint64_t clock_bound = 0;
    CHECK(context->find_down_group_key(graph.down, &key));
    CHECK(ggml_cuda_moe_grouped_context_test_access::get_clock_bound(*context, key, &clock_bound));
    CHECK(clock_bound == 2);
    if (n_dim == 1024) {
        CHECK(ggml_cuda_moe_grouped_context_test_access::host_copy_jobs(*context, key) != 0);
    }
    if (host_budget != 0) {
        CHECK(ggml_cuda_moe_grouped_context_test_access::attach_prepack(*context, key, stream));
        CHECK(ggml_cuda_moe_grouped_context_test_access::attach_prepack(*context, key, stream));
        std::vector<std::shared_ptr<void>> pending_leases;
        CHECK(context->graph_resource_fingerprint(overflow_replay, stream, &refreshed_fingerprint, &pending_leases));
        CHECK(pending_leases.size() == 1);
        std::weak_ptr<void> pending_witness = pending_leases[0];
        ggml_backend_tensor_set(graph.ids, routes, 0, sizeof(routes));
        ggml_backend_synchronize(backend.get());

        host_barrier pending_barrier;
        CUDA_OK(cudaLaunchHostFunc(previous_stream, wait_on_host_barrier, &pending_barrier));
        ggml_cuda_moe_graph_execution pending_direct;
        prepare_execution(pending_direct, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, previous_stream);
        CHECK(context->begin_graph_dispatch(&pending_direct, GGML_CUDA_MOE_GRAPH_DISPATCH_DIRECT));
        run_callbacks(pending_direct, previous_stream);
        CHECK(context->finish_graph_dispatch(&pending_direct));
        while (!pending_barrier.entered.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }

        const bool replace_pending = host_budget == 393216;
        std::thread replace_pending_thread;
        if (replace_pending) {
            replace_pending_thread = std::thread([&] {
                register_active_grouped_dispatch(backend.get(), graph, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 12);
            });
            while (ggml_cuda_moe_grouped_context_test_access::has_device_resource(*context, key)) {
                std::this_thread::yield();
            }
        }
        std::atomic<bool> lease_released{false}, shutdown_done{false};
        std::thread release_pending([&] {
            pending_leases.clear();
            lease_released.store(true, std::memory_order_release);
        });
        std::thread shutdown_pending([&] {
            context->shutdown();
            shutdown_done.store(true, std::memory_order_release);
        });
        while (!ggml_cuda_moe_grouped_context_test_access::admission_closed(*context)) {
            std::this_thread::yield();
        }
        for (int attempt = 0; attempt < 100; ++attempt) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        const bool retained = lease_released.load(std::memory_order_acquire) && !pending_witness.expired();
        const bool shutdown_waited = !shutdown_done.load(std::memory_order_acquire);
        pending_barrier.released.store(true, std::memory_order_release);
        release_pending.join();
        if (replace_pending_thread.joinable()) {
            replace_pending_thread.join();
        }
        shutdown_pending.join();
        CHECK(retained && shutdown_waited && pending_witness.expired());
        fprintf(stderr, "test-moe-cache: prepack cross-stream teardown replaced=%d retained=1 waited=1 OK\n", replace_pending);
    }
    CUDA_OK(cudaStreamSynchronize(previous_stream));
    CUDA_OK(cudaStreamDestroy(previous_stream));
    CUDA_OK(cudaStreamDestroy(stream));
    ggml_backend_cuda_moe_set_debug_mm(old_debug_mm);
    fprintf(stderr, "test-moe-cache: grouped graph replay lifecycle host_budget=%zu host_nodes=%zu OK\n", host_budget, host_nodes);
}

void test_early_grouped_graphs() {
    const char * enabled = getenv("GGML_CUDA_MOE_EARLY_ROUTER");
    CHECK(enabled != nullptr && strcmp(enabled, "1") == 0);
    const bool old_debug_mm = ggml_backend_cuda_moe_get_debug_mm();
    ggml_backend_cuda_moe_set_debug_mm(true);
    for (uint32_t rows : {1u, 4u}) {
        for (uint32_t layout : {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE,
                GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_UNGATED}) {
            for (int variant = 0; variant < 21; ++variant) {
                if ((variant == 9 && layout != GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP) ||
                        (variant == 10 && layout != GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE) ||
                        ((variant == 14 || variant >= 17) && rows == 1)) {
                    continue;
                }
                ggml_backend_ptr reference_backend(ggml_backend_cuda_init(0));
                ggml_backend_ptr candidate_backend(ggml_backend_cuda_init(0));
                const ggml_type type = variant == 8 ? GGML_TYPE_Q4_K : variant == 9 ? GGML_TYPE_MXFP4 :
                    variant == 10 ? GGML_TYPE_NVFP4 : variant == 11 ? GGML_TYPE_BF16 :
                    variant == 12 ? GGML_TYPE_Q8_0 : GGML_TYPE_Q4_0;
                const std::array<ggml_type, 3> types = {type, type, type};
                auto reference = build_active_grouped_dispatch_graph_types(reference_backend.get(),
                    ggml_backend_cuda_moe_cached_buffer_type(), types, layout, false, variant == 10, rows, 16, 2, 256,
                    nullptr, false, variant == 9, 0, false, false, variant);
                auto candidate = build_active_grouped_dispatch_graph_types(candidate_backend.get(),
                    ggml_backend_cuda_moe_cached_buffer_type(), types, layout, false, variant == 10, rows, 16, 2, 256,
                    nullptr, false, variant == 9, 0, false, false, variant);
                initialize_active_grouped_dispatch_graphs({&reference, &candidate});
                if (variant == 13 || variant == 14 || variant >= 17) {
                    candidate_stamp_execution(candidate.graph,
                        variant == 20 ? GGML_GRAPH_EXECUTION_DOMAIN_DRAFT :
                        variant == 13 || variant == 18 ? GGML_GRAPH_EXECUTION_DOMAIN_MTP : GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
                        variant == 13 ? GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT :
                        variant == 14 ? GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL :
                        variant == 18 || variant == 20 ? GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL :
                            GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SPECULATIVE,
                        rows, variant == 13 ? rows : 1, 0,
                        variant == 14 ? GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_NONE :
                            GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED);
                }
                const auto disabled = candidate_snapshot(8, nullptr, 0);
                CHECK(ggml_backend_cuda_moe_candidate_replace_v1(reference_backend.get(), &disabled) ==
                    GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
                if (variant == 15) {
                    const ggml_backend_moe_candidate_group_v2 group = {layout, GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_CHUNK, 0, 0};
                    std::vector<ggml_backend_moe_candidate_tensor_v2> tensors;
                    for (size_t bank = 0; bank < candidate.banks.size(); ++bank) {
                        tensors.push_back({candidate.banks[bank], 0, candidate.roles[bank], GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE,
                            GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_CACHED_BUFFER, 0});
                    }
                    const auto snapshot = candidate_snapshot_v2(8, &group, 1, tensors.data(), tensors.size());
                    CHECK(ggml_backend_cuda_moe_candidate_replace_v2(candidate_backend.get(), &snapshot) ==
                        GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
                } else if (variant == 10) {
                    CHECK(replace_active_grouped_nvfp4_dispatch_v2(candidate_backend.get(), candidate, 8) ==
                        GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
                } else {
                    register_active_grouped_dispatch(candidate_backend.get(), candidate, layout,
                        variant == 19 ? 4 : 8);
                }
                if (variant == 13 || variant >= 17) {
                    auto * context = ggml_cuda_moe_grouped_context_for_test(candidate_backend.get());
                    CHECK(context != nullptr);
                    (void) candidate_certify_graph(*context, candidate.graph);
                }
                ggml_context_ptr rebound_nodes;
                ggml_backend_buffer_ptr rebound_buffer;
                if (variant == 15) {
                    CHECK(ggml_backend_graph_compute(reference_backend.get(), reference.graph) == GGML_STATUS_SUCCESS);
                    CHECK(ggml_backend_graph_compute(candidate_backend.get(), candidate.graph) != GGML_STATUS_SUCCESS);
                    ggml_backend_synchronize(reference_backend.get());
                    ggml_backend_synchronize(candidate_backend.get());
                    auto * context = ggml_cuda_moe_grouped_context_for_test(candidate_backend.get());
                    CHECK(context != nullptr && ggml_cuda_moe_grouped_context_test_access::early_bytes(*context) == 0);
                    fprintf(stderr, "early-grouped-test: layout=%u rows=%u variant=%d baseline_route_rejection=expected prediction=unsupported\n",
                        layout, rows, variant);
                    continue;
                }
                for (int pass = 0; pass < 4; ++pass) {
                    if (variant == 5 && pass == 2) {
                        const ggml_init_params params = {ggml_tensor_overhead() * 2, nullptr, true};
                        rebound_nodes.reset(ggml_init(params));
                        auto * replacement = ggml_new_tensor_1d(rebound_nodes.get(), GGML_TYPE_I32, rows);
                        ggml_set_input(replacement);
                        rebound_buffer.reset(ggml_backend_alloc_ctx_tensors(rebound_nodes.get(), candidate_backend.get()));
                        CHECK(rebound_buffer != nullptr);
                        for (int n = 0; n < candidate.graph->n_nodes; ++n) {
                            for (auto & src : candidate.graph->nodes[n]->src) {
                                if (src == candidate.router_rows) {
                                    src = replacement;
                                }
                            }
                        }
                        candidate.router_rows = replacement;
                        ggml_graph_clear(candidate.graph);
                        ggml_build_forward_expand(candidate.graph, candidate.output);
                        candidate_stamp_execution(candidate.graph, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
                            GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, rows, rows);
                    }
                    if (variant == 5 || variant == 16) {
                        std::vector<int32_t> indices(rows);
                        for (uint32_t row = 0; row < rows; ++row) {
                            indices[row] = (row + pass) % (variant == 16 ? 16 : rows);
                        }
                        ggml_backend_tensor_set(reference.router_rows, indices.data(), 0, ggml_nbytes(reference.router_rows));
                        ggml_backend_tensor_set(candidate.router_rows, indices.data(), 0, ggml_nbytes(candidate.router_rows));
                    }
                    std::vector<float> input(ggml_nelements(reference.input));
                    for (size_t i = 0; i < input.size(); ++i) {
                        input[i] = std::cos(float(i * 3 + pass * 11));
                    }
                    ggml_backend_tensor_set(reference.input, input.data(), 0, ggml_nbytes(reference.input));
                    ggml_backend_tensor_set(candidate.input, input.data(), 0, ggml_nbytes(candidate.input));
                    CHECK(ggml_backend_graph_compute(reference_backend.get(), reference.graph) == GGML_STATUS_SUCCESS);
                    CHECK(ggml_backend_graph_compute(candidate_backend.get(), candidate.graph) == GGML_STATUS_SUCCESS);
                    ggml_backend_synchronize(reference_backend.get());
                    ggml_backend_synchronize(candidate_backend.get());
                    std::vector<float> expected(ggml_nelements(reference.output)), actual(expected.size());
                    ggml_backend_tensor_get(reference.output, expected.data(), 0, ggml_nbytes(reference.output));
                    ggml_backend_tensor_get(candidate.output, actual.data(), 0, ggml_nbytes(candidate.output));
                    check_active_grouped_exact_output(expected, actual);
                }
                auto * context = ggml_cuda_moe_grouped_context_for_test(candidate_backend.get());
                CHECK(context != nullptr);
                uint64_t calls = 0;
                const uint64_t bytes = ggml_cuda_moe_grouped_context_test_access::early_bytes(*context, &calls);
                if (variant == 5) {
                    CHECK(ggml_cuda_moe_grouped_context_test_access::early_programs(*context) == 0);
                }
                fprintf(stderr, "early-grouped-test: layout=%u rows=%u variant=%d calls=%llu predicted_bytes=%llu\n",
                    layout, rows, variant, (unsigned long long) calls, (unsigned long long) bytes);
                if (variant == 1 || (variant == 11 && rows > 1)) {
                    CHECK(calls == 0 && bytes == 0);
                    const auto coverage = candidate_certify_graph(*context, candidate.graph);
                    std::shared_ptr<ggml_cuda_moe_graph_plan> plan;
                    auto execution = std::make_unique<ggml_cuda_moe_graph_execution>();
                    CHECK(context->prepare_graph_execution(candidate.graph, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED,
                        &plan, execution.get(), coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
                        GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
                    CHECK(execution->outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_LEGACY);
                    fprintf(stderr, "early-grouped-test: layout=%u rows=%u variant=%d baseline legacy fallback, exact output, prediction=unsupported\n",
                        layout, rows, variant);
                    continue;
                }
                if (variant == 16) {
                    CHECK(calls == 0 && bytes == 0);
                    const auto coverage = candidate_certify_graph(*context, candidate.graph);
                    ggml_cuda_moe_graph_plan plan;
                    auto execution = std::make_unique<ggml_cuda_moe_graph_execution>();
                    context->compile_graph_plan(candidate.graph, 1, &plan, execution.get(),
                        coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint);
                    CHECK(execution->outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED);
                    const auto bind = [&]() {
                        return context->bind_graph_plan(candidate.graph, 1, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN,
                            plan, execution.get(), coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint);
                    };
                    CHECK(bind());
                    auto * table = candidate.ids->src[0];
                    const size_t stride = table->nb[1];
                    table->nb[1] += sizeof(int32_t);
                    CHECK(!bind());
                    table->nb[1] = stride;
                    CHECK(bind());
                    candidate.ids->src[1] = reference.router_rows;
                    CHECK(!bind());
                    candidate.ids->src[1] = candidate.router_rows;
                    CHECK(bind());
                    candidate.ids->view_src = table;
                    CHECK(!bind());
                    candidate.ids->view_src = nullptr;
                    CHECK(bind());
                    ggml_backend_buffer_set_usage(table->buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                    CHECK(!bind());
                    ggml_backend_buffer_set_usage(table->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                    CHECK(bind());
                    fprintf(stderr, "early-grouped-test: token lookup grouped, exact output, malformed bindings rejected, prediction=unsupported\n");
                    continue;
                }
                if (variant == 14 || variant == 19 || variant == 20) {
                    CHECK(calls == 0 && bytes == 0);
                    fprintf(stderr, "early-grouped-test: unsupported domain or host-staged rows, exact output, prediction=excluded\n");
                    continue;
                }
                CHECK(calls == 0 && bytes == 0);
            }
        }
    }
    ggml_backend_cuda_moe_set_debug_mm(old_debug_mm);
}

static void check_active_grouped_capabilities(
        ggml_cuda_moe_grouped_context & context,
        active_grouped_dispatch_graph & graph,
        uint32_t n_slots,
        ggml_cuda_mmid_consumer expected_consumer,
        ggml_cuda_mmid_mapping expected_mapping) {
    (void) candidate_certify_graph(context, graph.graph);
    ggml_cuda_moe_graph_plan plan;
    ggml_cuda_moe_graph_execution execution;
    context.compile_graph_plan(graph.graph, graph.graph->uid, &plan, &execution);
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED && execution.size() == 1);
    for (uint32_t bank = 0; bank < graph.banks.size(); ++bank) {
        const auto capability = ggml_cuda_moe_grouped_context_test_access::graph_bank_capability(plan, 0, bank);
        CHECK(capability.reason == GGML_CUDA_MMID_CAPABILITY_OK &&
            capability.equivalence_reason == GGML_CUDA_MMID_CAPABILITY_OK &&
            capability.consumer == expected_consumer && capability.use_mmq == 1 &&
            capability.mapping == expected_mapping && capability.tensor == graph.banks[bank] &&
            capability.phase == GGML_CUDA_MMID_PHASE_PREFILL &&
            capability.n_experts == graph.n_experts && capability.source_ne[2] == graph.n_experts &&
            capability.grouped_ne[2] == n_slots && capability.top_k == graph.n_used &&
            capability.n_rows == graph.n_rows && capability.n_routes == graph.n_used * graph.n_rows &&
            capability.row_stride == graph.n_experts && capability.n_slots == n_slots);
        for (uint32_t dim = 0; dim < GGML_MAX_DIMS; ++dim) {
            CHECK(capability.source_ne[dim] == graph.banks[bank]->ne[dim] &&
                capability.source_nb[dim] == graph.banks[bank]->nb[dim]);
        }
        CHECK(capability.grouped_ne[0] == capability.source_ne[0] &&
            capability.grouped_ne[1] == capability.source_ne[1] &&
            capability.grouped_nb[2] == capability.source_nb[2] &&
            capability.grouped_nb[3] == capability.grouped_nb[2] * n_slots);
    }
}

void test_active_grouped_lookup_routes(int device) {
    const bool old_debug_mm = ggml_backend_cuda_moe_get_debug_mm();
    ggml_backend_cuda_moe_set_debug_mm(true);
    for (uint32_t layout : {GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE,
            GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_UNGATED}) {
        for (uint32_t rows : {1u, 4u}) {
            ggml_backend_ptr reference_backend(ggml_backend_cuda_init(device));
            ggml_backend_ptr candidate_backend(ggml_backend_cuda_init(device));
            const auto build = [&](ggml_backend_t backend) {
                return build_active_grouped_dispatch_graph_types(backend, ggml_backend_cuda_moe_cached_buffer_type(),
                    {GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0}, layout,
                    false, false, rows, 8, 2, 256, nullptr, false, false, 0, false, true);
            };
            auto reference = build(reference_backend.get());
            auto candidate = build(candidate_backend.get());
            initialize_active_grouped_dispatch_graphs({&reference, &candidate});
            std::vector<int32_t> table(32);
            for (size_t i = 0; i < table.size(); ++i) {
                table[i] = (i + i / 2) % 8;
            }
            for (auto * graph : {&reference, &candidate}) {
                ggml_backend_tensor_set(graph->lookup_table, table.data(), 0, ggml_nbytes(graph->lookup_table));
            }
            const auto disabled = candidate_snapshot(12, nullptr, 0);
            CHECK(ggml_backend_cuda_moe_candidate_replace_v1(reference_backend.get(), &disabled) ==
                GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
            register_active_grouped_dispatch(candidate_backend.get(), candidate, layout, 12);
            auto * context = ggml_cuda_moe_grouped_context_for_test(candidate_backend.get());
            CHECK(context != nullptr);
            (void) candidate_certify_graph(*context, candidate.graph);
            for (uint32_t pass = 0; pass < 4; ++pass) {
                std::vector<int32_t> tokens(rows);
                std::vector<float> input(ggml_nelements(candidate.input));
                for (uint32_t row = 0; row < rows; ++row) {
                    tokens[row] = (row * 3 + pass) % 16;
                }
                for (size_t i = 0; i < input.size(); ++i) {
                    input[i] = 0.01f * (int(i % 31) - 15 + int(pass));
                }
                for (auto * graph : {&reference, &candidate}) {
                    ggml_backend_tensor_set(graph->token_ids, tokens.data(), 0, ggml_nbytes(graph->token_ids));
                    ggml_backend_tensor_set(graph->input, input.data(), 0, ggml_nbytes(graph->input));
                }
                const auto expected = run_active_grouped_dispatch(reference_backend.get(), reference, 0);
                const auto actual = run_active_grouped_dispatch(candidate_backend.get(), candidate, rows * 2 * (pass + 1),
                    layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE);
                check_active_grouped_exact_output(expected, actual);
                std::vector<int32_t> ids(rows * 2);
                ggml_backend_tensor_get(candidate.ids, ids.data(), 0, ggml_nbytes(candidate.ids));
                for (uint32_t row = 0; row < rows; ++row) {
                    CHECK(ids[2 * row] == table[2 * tokens[row]] && ids[2 * row + 1] == table[2 * tokens[row] + 1]);
                }
            }
            const auto telemetry = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
            CHECK(telemetry.calls == 4 && telemetry.completed == 4 && telemetry.fallback == 0 && telemetry.rollback == 0);
            CHECK(telemetry.prepare_error == 0 && telemetry.finish_error == 0);
            CHECK(telemetry.plan_compiles == 1 && telemetry.plan_reuses == 3);
            const auto coverage = candidate_certify_graph(*context, candidate.graph);
            ggml_cuda_moe_graph_plan plan;
            auto execution = std::make_unique<ggml_cuda_moe_graph_execution>();
            context->compile_graph_plan(candidate.graph, 1, &plan, execution.get(),
                coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint);
            CHECK(execution->outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED);
            const auto bind = [&]() {
                const bool unknown = context->bind_graph_plan(candidate.graph, 1, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN,
                    plan, execution.get(), coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint);
                const bool unchanged = context->bind_graph_plan(candidate.graph, 1, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED,
                    plan, execution.get(), coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint);
                CHECK(unknown == unchanged);
                return unknown;
            };
            const auto reject_route = [&]() {
                CHECK(!bind());
                ggml_cuda_moe_graph_plan invalid_plan;
                context->compile_graph_plan(candidate.graph, 2, &invalid_plan, execution.get());
                CHECK(execution->outcome() != GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED);
            };
            CHECK(bind());
            const size_t stride = candidate.lookup_table->nb[1];
            candidate.lookup_table->nb[1] += sizeof(int32_t);
            reject_route();
            candidate.lookup_table->nb[1] = stride;
            CHECK(bind());
            candidate.ids->src[1] = reference.token_ids;
            CHECK(!bind());
            candidate.ids->src[1] = candidate.token_ids;
            CHECK(bind());
            void * token_data = candidate.token_ids->data;
            candidate.token_ids->data = reference.token_ids->data;
            CHECK(!bind());
            candidate.token_ids->data = token_data;
            CHECK(bind());
            candidate.ids->view_src = candidate.lookup_table;
            reject_route();
            candidate.ids->view_src = nullptr;
            CHECK(bind());
            ggml_backend_buffer_set_usage(candidate.lookup_buffer.get(), GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            reject_route();
            ggml_backend_buffer_set_usage(candidate.lookup_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            CHECK(bind());
            for (ggml_tensor * tensor : {candidate.lookup_table, candidate.token_ids}) {
                tensor->type = GGML_TYPE_F32;
                reject_route();
                tensor->type = GGML_TYPE_I32;
                CHECK(bind());
                const int64_t width = tensor->ne[0];
                tensor->ne[0]++;
                reject_route();
                tensor->ne[0] = width;
                CHECK(bind());
            }
            const size_t index_stride = candidate.token_ids->nb[0];
            candidate.token_ids->nb[0] *= 2;
            if (rows > 1) {
                reject_route();
            } else {
                CHECK(!bind());
            }
            candidate.token_ids->nb[0] = index_stride;
            CHECK(bind());
            CHECK(candidate.graph->nodes[0] == candidate.ids);
            std::swap(candidate.graph->nodes[0], candidate.graph->nodes[1]);
            reject_route();
            std::swap(candidate.graph->nodes[0], candidate.graph->nodes[1]);
            CHECK(bind());
            candidate.ids->src[0] = reference.lookup_table;
            CHECK(!bind());
            ggml_cuda_moe_graph_plan rebound_plan;
            context->compile_graph_plan(candidate.graph, 1, &rebound_plan, execution.get(),
                coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint);
            CHECK(execution->outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED);
            CHECK(context->bind_graph_plan(candidate.graph, 1, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN,
                rebound_plan, execution.get(), coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint));
            candidate.ids->src[0] = candidate.lookup_table;
            CHECK(bind());
            fprintf(stderr, "test-moe-cache: lookup route layout=%u rows=%u exact output and binding checks OK\n", layout, rows);
        }
    }
    ggml_backend_cuda_moe_set_debug_mm(old_debug_mm);
}

static void test_active_grouped_multirow_graph_modes_case(
        int device,
        uint32_t n_rows,
        uint32_t n_experts = 8,
        uint32_t n_used = 2,
        uint32_t n_slots = 12,
        bool test_large_transition = false,
        uint32_t layout = GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP,
        ggml_type type = GGML_TYPE_Q4_0,
        ggml_cuda_mmid_consumer expected_consumer = GGML_CUDA_MMID_CONSUMER_MMVQ,
        ggml_cuda_mmid_mapping expected_mapping = GGML_CUDA_MMID_MAPPING_DIRECT) {
    CHECK(n_rows >= 2 && n_slots > 0 && n_used <= n_slots / n_rows);
    const bool old_debug_mm = ggml_backend_cuda_moe_get_debug_mm();
    ggml_backend_cuda_moe_set_debug_mm(true);
    ggml_backend_ptr reference_backend(ggml_backend_cuda_init(device));
    ggml_backend_ptr candidate_backend(ggml_backend_cuda_init(device));
    CHECK(reference_backend != nullptr && candidate_backend != nullptr);
    ggml_backend_cuda_set_decode_boundary_overlap(candidate_backend.get(), true);
    auto reference = build_active_grouped_dispatch_graph(
        reference_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), type,
        layout, false, n_rows, n_experts, n_used, 256);
    auto candidate = build_active_grouped_dispatch_graph(
        candidate_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), type,
        layout, false, n_rows, n_experts, n_used, 256);
    initialize_active_grouped_dispatch_graphs({&reference, &candidate});
    const auto disabled = candidate_snapshot(n_slots, nullptr, 0);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v1(reference_backend.get(), &disabled) ==
        GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    register_active_grouped_dispatch(candidate_backend.get(), candidate, layout, n_slots);
    auto * context = ggml_cuda_moe_grouped_context_for_test(candidate_backend.get());
    CHECK(context != nullptr);
    check_active_grouped_capabilities(*context, candidate, n_slots, expected_consumer, expected_mapping);
    const bool test_transition = test_large_transition || (n_slots == 12 && n_rows == 2);
    cached_mmid_path_test_graph reference_prefill;
    cached_mmid_path_test_graph candidate_prefill;
    std::vector<int32_t> prefill_ids;
    if (test_transition) {
        const int64_t prefill_used = n_experts >= 16 ? 8 : n_used;
        reference_prefill = build_cached_mmid_path_test_graph(
            reference_backend.get(), reference.banks[0], reference.banks[1], prefill_used, 2);
        candidate_prefill = build_cached_mmid_path_test_graph(
            candidate_backend.get(), candidate.banks[0], candidate.banks[1], prefill_used, 2);
        initialize_cached_mmid_path_test_graphs(candidate_prefill, reference_prefill);
        candidate_insert_graph_node(candidate_prefill.graph, 0, candidate.graph->nodes[0]);
        CHECK(candidate_prefill.graph->nodes[0] == candidate.graph->nodes[0]);
        prefill_ids.reserve(2 * prefill_used);
        for (int64_t route = 0; route < prefill_used; ++route) {
            prefill_ids.push_back(static_cast<int32_t>(route));
        }
        for (int64_t route = 0; route < prefill_used; ++route) {
            prefill_ids.push_back(static_cast<int32_t>((route + 5) % n_experts));
        }
    }
    std::vector<float> direct_expected;
    std::vector<float> direct_actual;
    uintptr_t captured_graph = 0;
    uintptr_t captured_instance = 0;
    uint64_t captured_semantic_key = 0;
    uint64_t captured_resource_fingerprint = 0;
    const uint64_t n_routes = n_used * n_rows;
    const bool f3_skipped = layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE &&
        expected_consumer == GGML_CUDA_MMID_CONSUMER_MMVQ &&
        candidate.banks[0]->type == candidate.banks[1]->type &&
        ggml_are_same_shape(candidate.banks[0], candidate.banks[1]) &&
        ggml_are_same_stride(candidate.banks[0], candidate.banks[1]);
    bool capture_available = false;
    bool transition_executed = false;
    uint32_t executed_passes = 0;
    for (uint32_t pass = 0; pass < 4; ++pass) {
        if (pass != 0 && !capture_available) {
            break;
        }
        const uint32_t route_variant = pass >= 2 ? pass - 1 : 0;
        if (pass >= 2) {
            set_active_grouped_dispatch_logits({&reference, &candidate}, route_variant);
        }
        const auto expected = run_active_grouped_dispatch(reference_backend.get(), reference, 0, false);
        const uint64_t clock_pass = transition_executed ? pass - 1 : pass + 1;
        const auto actual = run_active_grouped_dispatch(candidate_backend.get(), candidate, n_routes * clock_pass, f3_skipped);
        ++executed_passes;
        check_active_grouped_exact_output(expected, actual);
        if (pass == 0) {
            direct_expected = expected;
            direct_actual = actual;
        } else if (pass == 1) {
            CHECK(expected == direct_expected && actual == direct_actual);
        }
        check_active_grouped_routes(reference, n_rows, route_variant);
        check_active_grouped_routes(candidate, n_rows, route_variant);

        ggml_cuda_graph_capture_state_for_test reference_graph;
        ggml_cuda_graph_capture_state_for_test candidate_graph;
        CHECK(ggml_cuda_graph_capture_state_query_for_test(reference_backend.get(), reference.graph, &reference_graph));
        CHECK(ggml_cuda_graph_capture_state_query_for_test(candidate_backend.get(), candidate.graph, &candidate_graph));
        CHECK(reference_graph.instance == 0 && reference_graph.graph == 0 &&
            !reference_graph.warmup_complete && reference_graph.moe_resource_fingerprint == 0);
        if (pass == 0) {
            capture_available = candidate_graph.capture_available;
            CHECK(candidate_graph.instance == 0 && candidate_graph.graph == 0 &&
                !candidate_graph.warmup_complete && candidate_graph.moe_resource_fingerprint == 0);
        } else if (pass == 1) {
            CHECK(candidate_graph.instance != 0 && candidate_graph.graph != 0 &&
                candidate_graph.warmup_complete && candidate_graph.moe_resource_fingerprint != 0);
            captured_graph = candidate_graph.graph;
            captured_instance = candidate_graph.instance;
            captured_semantic_key = candidate_graph.execution_semantic_key;
            captured_resource_fingerprint = candidate_graph.moe_resource_fingerprint;
            CHECK(captured_semantic_key != 0 && captured_resource_fingerprint != 0);
        } else {
            CHECK(candidate_graph.graph == captured_graph && candidate_graph.instance == captured_instance &&
                candidate_graph.warmup_complete && candidate_graph.moe_resource_fingerprint != 0 &&
                candidate_graph.execution_semantic_key == captured_semantic_key);
            if (transition_executed) {
                ggml_cuda_moe_candidate_group_key key;
                CHECK(context->find_down_group_key(candidate.down, &key));
                CHECK(ggml_cuda_moe_grouped_context_test_access::graph_clock_active(*context, key));
            }
        }
        if (test_transition && !transition_executed && ((capture_available && pass == 1) || (!capture_available && pass == 0))) {
            ggml_cuda_moe_candidate_group_key key;
            CHECK(context->find_down_group_key(candidate.down, &key));
            if (capture_available) {
                CHECK(ggml_cuda_moe_grouped_context_test_access::graph_clock_active(*context, key));
            }
            (void) candidate_certify_graph(*context, candidate_prefill.graph);
            (void) run_cached_mmid_path_test(
                candidate_backend.get(), reference_backend.get(), candidate_prefill, reference_prefill, prefill_ids);
            CHECK(ggml_cuda_moe_grouped_context_test_access::has_device_resource(*context, key));
            auto lease = context->acquire_legacy_cache(candidate.banks[0]);
            CHECK(lease && lease.get() != nullptr &&
                ggml_cuda_moe_cache_slot_ptr(lease.get(), 0) ==
                    ggml_cuda_moe_grouped_context_test_access::device_bank_data(*context, key, candidate.banks[0]));
            if (capture_available) {
                ggml_cuda_graph_capture_state_for_test retained_graph;
                CHECK(ggml_cuda_graph_capture_state_query_for_test(candidate_backend.get(), candidate.graph, &retained_graph));
                CHECK(retained_graph.graph == captured_graph && retained_graph.instance == captured_instance &&
                    retained_graph.warmup_complete && retained_graph.execution_semantic_key == captured_semantic_key &&
                    retained_graph.moe_resource_fingerprint == captured_resource_fingerprint);
            }
            (void) candidate_certify_graph(*context, candidate.graph);
            transition_executed = true;
        }
    }

    if (transition_executed && !capture_available) {
        set_active_grouped_dispatch_logits({&reference, &candidate}, 1);
        const auto expected = run_active_grouped_dispatch(reference_backend.get(), reference, 0, false);
        const auto actual = run_active_grouped_dispatch(candidate_backend.get(), candidate, n_routes, f3_skipped);
        ++executed_passes;
        check_active_grouped_exact_output(expected, actual);
    }

    const uint64_t transition_legacy_ops = transition_executed ? reference.banks.size() : 0;
    CHECK(active_grouped_legacy_op_count(reference_backend.get(), true) +
        active_grouped_legacy_op_count(reference_backend.get(), false) ==
            executed_passes * reference.banks.size() + transition_legacy_ops);
    CHECK(active_grouped_legacy_op_count(candidate_backend.get(), true) == 0 &&
        active_grouped_legacy_op_count(candidate_backend.get(), false) == transition_legacy_ops);
    const auto telemetry = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
    CHECK(telemetry.registered == 1 && telemetry.covered == 1);
    CHECK(executed_passes == (capture_available ? 4u : transition_executed ? 2u : 1u));
    const uint64_t expected_plan_compiles = 1;
    CHECK(telemetry.plan_calls == executed_passes && telemetry.plan_compiles == expected_plan_compiles &&
        telemetry.plan_reuses == executed_passes - expected_plan_compiles);
    CHECK(telemetry.calls == executed_passes && telemetry.ready == executed_passes && telemetry.completed == executed_passes);
    CHECK(telemetry.ready_min == executed_passes && telemetry.ready_max == executed_passes);
    CHECK(telemetry.completed_min == executed_passes && telemetry.completed_max == executed_passes);
    CHECK(telemetry.admitted_banks == executed_passes * candidate.banks.size());
    CHECK(telemetry.fallback == 0 && telemetry.rollback == 0);
    CHECK(telemetry.prepare_error == 0 && telemetry.finish_error == 0);

    if (capture_available) {
        ggml_backend_buffer_ptr output_buffer(ggml_backend_alloc_buffer(candidate_backend.get(), ggml_nbytes(candidate.output)));
        CHECK(output_buffer != nullptr);
        void * original_data = candidate.output->data;
        auto * original_buffer = candidate.output->buffer;
        for (bool relocated : {true, false}) {
            candidate.output->data = relocated ? ggml_backend_buffer_get_base(output_buffer.get()) : original_data;
            candidate.output->buffer = relocated ? output_buffer.get() : original_buffer;
            candidate_stamp_execution(candidate.graph, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
                GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, n_rows, n_rows);
            (void) candidate_certify_graph(*context, candidate.graph);
            for (uint32_t variant : {1u, 2u}) {
                set_active_grouped_dispatch_logits({&reference, &candidate}, variant);
                const auto expected = run_active_grouped_dispatch(reference_backend.get(), reference, 0, false);
                const auto actual = run_active_grouped_dispatch(candidate_backend.get(), candidate, 0, f3_skipped);
                check_active_grouped_exact_output(expected, actual);
                ggml_cuda_graph_capture_state_for_test state;
                CHECK(ggml_cuda_graph_capture_state_query_for_test(candidate_backend.get(), candidate.graph, &state));
                CHECK(state.instance == captured_instance && state.graph != 0 && state.warmup_complete &&
                    state.moe_resource_fingerprint == captured_resource_fingerprint);
            }
        }
    }

    if (capture_available && test_transition) {
        auto expected = std::vector<float>();
        auto actual = std::vector<float>();
        ggml_cuda_graph_capture_state_for_test invalidated_graph;
        const auto recapture_main = [&]() {
            candidate_stamp_execution(candidate.graph, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
                GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, n_rows, n_rows);
            (void) candidate_certify_graph(*context, candidate.graph);
            for (uint32_t pass = 0; pass < 2; ++pass) {
                expected = run_active_grouped_dispatch(reference_backend.get(), reference, 0, false);
                actual = run_active_grouped_dispatch(candidate_backend.get(), candidate, 0, f3_skipped);
                check_active_grouped_exact_output(expected, actual);
            }
            ggml_cuda_graph_capture_state_for_test state;
            CHECK(ggml_cuda_graph_capture_state_query_for_test(candidate_backend.get(), candidate.graph, &state));
            CHECK(state.graph != 0 && state.instance != 0 && state.warmup_complete && state.moe_resource_fingerprint != 0);
        };

        const auto inventory_shape = [](const ggml_cgraph * graph) {
            std::pair<uint32_t, int32_t> result = {0, -1};
            for (int32_t node_index = 0; node_index < graph->n_nodes; ++node_index) {
                const ggml_tensor * node = graph->nodes[node_index];
                const ggml_tensor * source = node != nullptr && node->op == GGML_OP_MUL_MAT_ID ? node->src[0] : nullptr;
                if (source == nullptr || source->buffer == nullptr ||
                        !ggml_backend_buft_is_cuda_moe_cached(ggml_backend_buffer_get_type(source->buffer))) {
                    continue;
                }
                if (result.second < 0) {
                    result.second = node_index;
                }
                ++result.first;
            }
            return result;
        };
        CHECK(candidate.graph->n_nodes > 1 && candidate.graph->nodes[1]->op == GGML_OP_VIEW);
        candidate_insert_graph_node(candidate_prefill.graph, 1, candidate.graph->nodes[1]);
        const auto certified_inventory = inventory_shape(candidate_prefill.graph);
        const auto stale_coverage = candidate_certify_graph(*context, candidate_prefill.graph);
        CHECK(certified_inventory.first == stale_coverage.mmid_count && stale_coverage.mmid_count == candidate.banks.size());
        ggml_tensor * inventory_padding = candidate_prefill.graph->nodes[1];
        memmove(candidate_prefill.graph->nodes + 1, candidate_prefill.graph->nodes + 2,
            (candidate_prefill.graph->n_nodes - 2) * sizeof(candidate_prefill.graph->nodes[0]));
        candidate_prefill.graph->nodes[candidate_prefill.graph->n_nodes - 1] = inventory_padding;
        candidate_rebuild_graph_uses(candidate_prefill.graph);
        const auto stale_inventory = inventory_shape(candidate_prefill.graph);
        CHECK(candidate_prefill.graph->nodes[0] == candidate.graph->nodes[0] &&
            stale_inventory.first == certified_inventory.first && stale_inventory.second + 1 == certified_inventory.second);
        (void) run_cached_mmid_path_test(
            candidate_backend.get(), reference_backend.get(), candidate_prefill, reference_prefill, prefill_ids);
        CHECK(ggml_cuda_graph_capture_state_query_for_test(candidate_backend.get(), candidate.graph, &invalidated_graph));
        CHECK(invalidated_graph.graph == captured_graph && invalidated_graph.instance == captured_instance &&
            invalidated_graph.warmup_complete && invalidated_graph.execution_semantic_key == captured_semantic_key &&
            invalidated_graph.moe_resource_fingerprint == captured_resource_fingerprint);

        recapture_main();
        candidate_stamp_execution(candidate.graph, GGML_GRAPH_EXECUTION_DOMAIN_DRAFT,
            GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, n_rows, n_rows);
        (void) candidate_certify_graph(*context, candidate.graph);
        expected = run_active_grouped_dispatch(reference_backend.get(), reference, 0, false);
        actual = run_active_grouped_dispatch(candidate_backend.get(), candidate, 0, f3_skipped);
        check_active_grouped_exact_output(expected, actual);
        CHECK(ggml_cuda_graph_capture_state_query_for_test(candidate_backend.get(), candidate.graph, &invalidated_graph));
        CHECK(invalidated_graph.graph == 0 && invalidated_graph.instance == 0 &&
            !invalidated_graph.warmup_complete && invalidated_graph.moe_resource_fingerprint == 0);

        recapture_main();
        register_active_grouped_dispatch(
            candidate_backend.get(), candidate, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, n_slots);
        (void) candidate_certify_graph(*context, candidate.graph);
        expected = run_active_grouped_dispatch(reference_backend.get(), reference, 0, false);
        actual = run_active_grouped_dispatch(candidate_backend.get(), candidate, 0, false);
        check_active_grouped_exact_output(expected, actual);
        CHECK(ggml_cuda_graph_capture_state_query_for_test(candidate_backend.get(), candidate.graph, &invalidated_graph));
        CHECK(invalidated_graph.graph == 0 && invalidated_graph.instance == 0 &&
            !invalidated_graph.warmup_complete && invalidated_graph.moe_resource_fingerprint == 0);
    }
    ggml_backend_cuda_moe_set_debug_mm(old_debug_mm);
    fprintf(stderr, capture_available ?
        "test-moe-cache: active grouped E%u K%u B%u direct/capture/replay dynamic routes OK\n" :
        "test-moe-cache: active grouped E%u K%u B%u direct OK (CUDA graphs unavailable)\n",
        n_experts, n_used, n_rows);
}

static bool test_active_grouped_q4k_eviction_refill_case(int device, uint32_t primary_rows) {
    constexpr uint32_t n_experts = 256;
    constexpr uint32_t n_used = 8;
    constexpr uint32_t n_slots = 132;
    constexpr uint32_t b5_rows = 5;
    CHECK(primary_rows == 8 || primary_rows == 16);

    const bool old_debug_mm = ggml_backend_cuda_moe_get_debug_mm();
    ggml_backend_cuda_moe_set_debug_mm(true);
    ggml_backend_ptr direct_backend(ggml_backend_cuda_init(device));
    ggml_backend_ptr legacy_backend(ggml_backend_cuda_init(device));
    ggml_backend_ptr grouped_backend(ggml_backend_cuda_init(device));
    CHECK(direct_backend != nullptr && legacy_backend != nullptr && grouped_backend != nullptr);

    auto direct_primary = build_active_grouped_dispatch_graph(
        direct_backend.get(), ggml_backend_cuda_buffer_type(device), GGML_TYPE_Q4_K,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, false, primary_rows, n_experts, n_used);
    auto direct_b5 = build_active_grouped_dispatch_graph(
        direct_backend.get(), ggml_backend_cuda_buffer_type(device), GGML_TYPE_Q4_K,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, false, b5_rows, n_experts, n_used, 256, &direct_primary);
    auto legacy_primary = build_active_grouped_dispatch_graph(
        legacy_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_K,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, false, primary_rows, n_experts, n_used);
    auto legacy_b5 = build_active_grouped_dispatch_graph(
        legacy_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_K,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, false, b5_rows, n_experts, n_used, 256, &legacy_primary);
    auto grouped_primary = build_active_grouped_dispatch_graph(
        grouped_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_K,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, false, primary_rows, n_experts, n_used);
    auto grouped_b5 = build_active_grouped_dispatch_graph(
        grouped_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_K,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, false, b5_rows, n_experts, n_used, 256, &grouped_primary);

    std::vector<std::vector<uint8_t>> bank_data;
    bank_data.reserve(direct_primary.banks.size());
    for (uint32_t bank = 0; bank < direct_primary.banks.size(); ++bank) {
        bank_data.push_back(active_grouped_q4k_expert_data(direct_primary.banks[bank], 211 + bank));
    }
    initialize_active_grouped_dispatch_graphs({&direct_primary, &legacy_primary, &grouped_primary}, &bank_data);
    initialize_active_grouped_dispatch_graphs({&direct_b5, &legacy_b5, &grouped_b5}, &bank_data);
    for (const auto * graph : {&direct_primary, &legacy_primary, &grouped_primary, &direct_b5, &legacy_b5, &grouped_b5}) {
        CHECK(graph->output == graph->down_output);
    }

    const auto disabled = candidate_snapshot(n_slots, nullptr, 0);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v1(legacy_backend.get(), &disabled) ==
        GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    register_active_grouped_dispatch(
        grouped_backend.get(), grouped_primary, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, n_slots);
    auto * context = ggml_cuda_moe_grouped_context_for_test(grouped_backend.get());
    CHECK(context != nullptr);
    const auto primary_native = native_mmid_capability(device, direct_primary.down, primary_rows,
        primary_rows == 8 ? GGML_CUDA_MMID_MAPPING_DIRECT : GGML_CUDA_MMID_MAPPING_SOURCE_MAP);
    const auto b5_native = native_mmid_capability(device, direct_b5.down, b5_rows, GGML_CUDA_MMID_MAPPING_DIRECT);
    const bool primary_supported = primary_native.reason == GGML_CUDA_MMID_CAPABILITY_OK &&
        primary_native.selection == (primary_rows == 8 ? GGML_CUDA_MMID_CONSUMER_MMVQ : GGML_CUDA_MMID_CONSUMER_MMQ);
    const bool b5_supported = b5_native.reason == GGML_CUDA_MMID_CAPABILITY_OK &&
        b5_native.selection == GGML_CUDA_MMID_CONSUMER_MMVQ;
    if (primary_rows == 8 && (!primary_supported || !b5_supported)) {
        ggml_backend_cuda_moe_set_debug_mm(old_debug_mm);
        fprintf(stderr, "test-moe-cache: active grouped Q4_K cache132 B%u/B5 skipped (native primary=%u B5=%u)\n",
            primary_rows, primary_supported, b5_supported);
        return false;
    }
    CHECK(primary_supported && b5_supported);
    check_active_grouped_capabilities(
        *context, grouped_primary, n_slots,
        primary_rows == 8 ? GGML_CUDA_MMID_CONSUMER_MMVQ : GGML_CUDA_MMID_CONSUMER_MMQ,
        primary_rows == 8 ? GGML_CUDA_MMID_MAPPING_DIRECT : GGML_CUDA_MMID_MAPPING_SOURCE_MAP);
    check_active_grouped_capabilities(
        *context, grouped_b5, n_slots, GGML_CUDA_MMID_CONSUMER_MMVQ, GGML_CUDA_MMID_MAPPING_DIRECT);

    struct host_cache_state {
        std::vector<int32_t> slot_expert;
        std::vector<int32_t> expert_slot;
        std::vector<uint64_t> last_used;
        std::vector<uint32_t> frequency;
        std::vector<uint64_t> frequency_epoch;
        uint64_t clock = 0;
        uint64_t step = 0;
    } host = {
        std::vector<int32_t>(n_slots, -1),
        std::vector<int32_t>(n_experts, -1),
        std::vector<uint64_t>(n_slots, 0),
        std::vector<uint32_t>(n_experts, 0),
        std::vector<uint64_t>(n_experts, 0),
        0,
        0,
    };
    const auto plan_routes = [&](const std::vector<int32_t> & routes) {
        CHECK(!routes.empty() && routes.size() <= n_slots && routes.size() % n_used == 0);
        const uint64_t current_epoch = host.step >> 4;
        const auto effective_frequency = [&](int32_t expert) {
            if (!grouped_frequency_enabled()) {
                return uint32_t(0);
            }
            CHECK(expert >= 0 && static_cast<uint32_t>(expert) < n_experts);
            CHECK(host.frequency_epoch[expert] <= current_epoch);
            const uint64_t elapsed = current_epoch - host.frequency_epoch[expert];
            return elapsed < 32 ? host.frequency[expert] >> elapsed : 0;
        };
        std::vector<int32_t> unique;
        for (uint32_t route = 0; route < routes.size(); ++route) {
            CHECK(routes[route] >= 0 && static_cast<uint32_t>(routes[route]) < n_experts);
            auto found = std::find(unique.begin(), unique.end(), routes[route]);
            if (found == unique.end()) {
                unique.push_back(routes[route]);
            }
        }
        std::vector<int32_t> unique_slots(unique.size(), -1);
        std::vector<bool> reserved(n_slots, false);
        for (uint32_t unique_index = 0; unique_index < unique.size(); ++unique_index) {
            const int32_t slot = host.expert_slot[unique[unique_index]];
            unique_slots[unique_index] = slot;
            if (slot >= 0) {
                CHECK(static_cast<uint32_t>(slot) < n_slots && host.slot_expert[slot] == unique[unique_index]);
                reserved[slot] = true;
            }
        }
        uint32_t misses = 0;
        for (uint32_t unique_index = 0; unique_index < unique.size(); ++unique_index) {
            if (unique_slots[unique_index] >= 0) {
                continue;
            }
            uint64_t best_age = UINT64_MAX;
            uint32_t best_frequency = UINT32_MAX;
            uint32_t best_slot = UINT32_MAX;
            for (uint32_t slot = 0; slot < n_slots; ++slot) {
                if (reserved[slot]) {
                    continue;
                }
                const uint32_t frequency = host.slot_expert[slot] < 0 ? 0 : effective_frequency(host.slot_expert[slot]);
                const uint64_t age = host.slot_expert[slot] < 0 ? 0 : host.last_used[slot];
                if (frequency < best_frequency ||
                        (frequency == best_frequency && (age < best_age || (age == best_age && slot < best_slot)))) {
                    best_frequency = frequency;
                    best_age = age;
                    best_slot = slot;
                }
            }
            CHECK(best_slot < n_slots);
            unique_slots[unique_index] = best_slot;
            reserved[best_slot] = true;
            ++misses;
        }
        for (uint32_t unique_index = 0; unique_index < unique.size(); ++unique_index) {
            const int32_t expert = unique[unique_index];
            const int32_t slot = unique_slots[unique_index];
            if (host.expert_slot[expert] == slot) {
                continue;
            }
            const int32_t old_expert = host.slot_expert[slot];
            if (old_expert >= 0) {
                host.expert_slot[old_expert] = -1;
            }
            host.slot_expert[slot] = expert;
            host.expert_slot[expert] = slot;
        }
        for (uint32_t unique_index = 0; unique_index < unique.size(); ++unique_index) {
            const int32_t expert = unique[unique_index];
            const uint32_t frequency = effective_frequency(expert);
            host.frequency[expert] = frequency != UINT32_MAX ? frequency + 1 : frequency;
            host.frequency_epoch[expert] = current_epoch;
            host.last_used[unique_slots[unique_index]] = host.clock + unique_index + 1;
        }
        host.clock += routes.size();
        ++host.step;
        return misses;
    };

    struct active_values {
        std::vector<float> gate;
        std::vector<float> up;
        std::vector<float> output;
    };
    const auto run_graph = [&](ggml_backend_t backend, active_grouped_dispatch_graph & graph,
                               uint64_t expected_clock, bool f3_skipped) {
        active_values values;
        values.output = run_active_grouped_dispatch(backend, graph, expected_clock, f3_skipped);
        if (!f3_skipped) {
            values.gate = active_grouped_tensor_values(graph.gate_output);
            values.up = active_grouped_tensor_values(graph.up_output);
        }
        return values;
    };
    const auto check_values = [](const active_values & expected, const active_values & actual) {
        if (!expected.gate.empty() && !actual.gate.empty()) {
            check_active_grouped_exact_output(expected.gate, actual.gate);
            check_active_grouped_exact_output(expected.up, actual.up);
        }
        check_active_grouped_exact_output(expected.output, actual.output);
    };
    ggml_cuda_moe_candidate_group_key group_key;
    CHECK(context->find_down_group_key(grouped_primary.down, &group_key));
    struct capture_state {
        uint32_t passes = 0;
        uintptr_t graph = 0;
        uintptr_t instance = 0;
        uint64_t semantic_key = 0;
        uint64_t resource_fingerprint = 0;
    } primary_capture, b5_capture;
    bool capture_available = false;
    uint64_t total_misses = 0;
    uint32_t total_passes = 0;
    const auto run_step = [&](const std::array<active_grouped_dispatch_graph *, 3> & graphs,
            const std::vector<int32_t> & routes, uint32_t expected_misses, capture_state & captured) {
        const std::vector<int32_t> committed_slots = host.slot_expert;
        const uint32_t misses = plan_routes(routes);
        CHECK(misses == expected_misses);
        total_misses += misses;
        ++total_passes;
        set_active_grouped_dispatch_routes(
            std::vector<active_grouped_dispatch_graph *>(graphs.begin(), graphs.end()), routes);
        const auto direct_capability = native_mmid_capability(
            device, graphs[0]->banks[0], graphs[0]->n_rows, GGML_CUDA_MMID_MAPPING_DIRECT);
        const bool direct_f3_skipped = direct_capability.reason == GGML_CUDA_MMID_CAPABILITY_OK &&
            direct_capability.selection == GGML_CUDA_MMID_CONSUMER_MMVQ;
        const bool grouped_f3_skipped = graphs[2]->n_rows == primary_rows ?
            primary_native.selection == GGML_CUDA_MMID_CONSUMER_MMVQ :
            b5_native.selection == GGML_CUDA_MMID_CONSUMER_MMVQ;
        const auto direct = run_graph(direct_backend.get(), *graphs[0], 0, direct_f3_skipped);
        const auto legacy = run_graph(legacy_backend.get(), *graphs[1], 0, false);
        const auto grouped = run_graph(grouped_backend.get(), *graphs[2], host.clock, grouped_f3_skipped);
        check_values(direct, legacy);
        check_values(direct, grouped);
        for (auto * graph : graphs) {
            CHECK(graph->ids->data == static_cast<const char *>(graph->ids->view_src->data) + graph->ids->view_offs);
            check_active_grouped_routes(*graph, graph->n_rows, routes);
        }
        for (uint32_t expert = 0; expert < n_experts; ++expert) {
            int32_t expected_slot = -1;
            for (uint32_t slot = 0; slot < n_slots; ++slot) {
                if (committed_slots[slot] == static_cast<int32_t>(expert)) {
                    expected_slot = slot;
                    break;
                }
            }
            CHECK(ggml_cuda_moe_grouped_context_test_access::device_slot_for_expert(*context, group_key, expert) == expected_slot);
        }
        for (uint32_t bank = 0; bank < grouped_primary.banks.size(); ++bank) {
            const ggml_tensor * source = grouped_primary.banks[bank];
            const void * device_data = ggml_cuda_moe_grouped_context_test_access::device_bank_data(
                *context, group_key, source);
            CHECK(device_data != nullptr);
            std::vector<uint8_t> physical(n_slots * source->nb[2]);
            CUDA_OK(cudaMemcpy(physical.data(), device_data, physical.size(), cudaMemcpyDeviceToHost));
            for (uint32_t slot = 0; slot < n_slots; ++slot) {
                const int32_t expert = host.slot_expert[slot];
                if (expert < 0) {
                    continue;
                }
                CHECK(memcmp(physical.data() + slot * source->nb[2],
                    static_cast<const uint8_t *>(source->data) + expert * source->nb[2], source->nb[2]) == 0);
            }
        }

        ggml_cuda_graph_capture_state_for_test state;
        CHECK(ggml_cuda_graph_capture_state_query_for_test(grouped_backend.get(), graphs[2]->graph, &state));
        if (total_passes == 1) {
            capture_available = state.capture_available;
        } else {
            CHECK(state.capture_available == capture_available);
        }
        ++captured.passes;
        if (capture_available && captured.passes == 1) {
            CHECK(state.graph == 0 && state.instance == 0 && !state.warmup_complete && state.moe_resource_fingerprint == 0);
        } else if (capture_available && captured.passes == 2) {
            CHECK(state.graph != 0 && state.instance != 0 && state.warmup_complete &&
                state.execution_semantic_key != 0 && state.moe_resource_fingerprint != 0);
            captured.graph = state.graph;
            captured.instance = state.instance;
            captured.semantic_key = state.execution_semantic_key;
            captured.resource_fingerprint = state.moe_resource_fingerprint;
        } else if (capture_available) {
            CHECK(state.graph == captured.graph && state.instance == captured.instance && state.warmup_complete &&
                state.execution_semantic_key == captured.semantic_key &&
                state.moe_resource_fingerprint == captured.resource_fingerprint);
        }
    };

    std::vector<int32_t> wave_a(primary_rows * n_used);
    std::vector<int32_t> wave_b(primary_rows * n_used);
    std::vector<int32_t> wave_c(primary_rows * n_used);
    std::iota(wave_a.begin(), wave_a.end(), 0);
    std::iota(wave_b.begin(), wave_b.end(), 64);
    std::iota(wave_c.begin(), wave_c.end(), 128);
    std::vector<int32_t> b5_a(b5_rows * n_used);
    std::iota(b5_a.begin(), b5_a.end(), 0);
    std::vector<int32_t> mixed(primary_rows * n_used);
    const uint32_t mixed_half = mixed.size() / 2;
    const uint32_t mixed_miss_base = primary_rows == 8 ? 192 : 128;
    for (uint32_t route = 0; route < mixed_half; ++route) {
        mixed[route] = route;
        mixed[mixed_half + route] = mixed_miss_base + route;
    }

    const uint32_t primary_routes = primary_rows * n_used;
    run_step({&direct_primary, &legacy_primary, &grouped_primary}, wave_a, primary_routes, primary_capture);
    run_step({&direct_primary, &legacy_primary, &grouped_primary}, wave_a, 0, primary_capture);
    run_step({&direct_primary, &legacy_primary, &grouped_primary}, wave_a, 0, primary_capture);
    run_step({&direct_primary, &legacy_primary, &grouped_primary}, wave_b, 64, primary_capture);
    run_step({&direct_primary, &legacy_primary, &grouped_primary}, wave_c, 64, primary_capture);
    run_step({&direct_b5, &legacy_b5, &grouped_b5}, b5_a, grouped_frequency_enabled() && primary_rows == 8 ? 0 : 40, b5_capture);
    run_step({&direct_b5, &legacy_b5, &grouped_b5}, b5_a, 0, b5_capture);
    run_step({&direct_b5, &legacy_b5, &grouped_b5}, b5_a, 0, b5_capture);
    run_step({&direct_primary, &legacy_primary, &grouped_primary}, wave_a,
        grouped_frequency_enabled() ? (primary_rows == 8 ? 0 : 84) : (primary_rows == 8 ? 24 : 88), primary_capture);
    run_step({&direct_primary, &legacy_primary, &grouped_primary}, mixed,
        grouped_frequency_enabled() && primary_rows != 8 ? 60 : mixed_half, primary_capture);
    run_step({&direct_primary, &legacy_primary, &grouped_primary}, mixed, 0, primary_capture);
    CHECK(total_passes == 11 && total_misses == (grouped_frequency_enabled() ? (primary_rows == 8 ? 224 : 440) : (primary_rows == 8 ? 288 : 448)));
    CHECK(host.clock == 8 * primary_routes + 3 * b5_rows * n_used);

    CHECK(ggml_cuda_moe_grouped_context_for_test(direct_backend.get()) == nullptr);
    CHECK(active_grouped_legacy_op_count(legacy_backend.get(), true) +
        active_grouped_legacy_op_count(legacy_backend.get(), false) == total_passes * legacy_primary.banks.size());
    CHECK(active_grouped_legacy_op_count(grouped_backend.get(), true) == 0 &&
        active_grouped_legacy_op_count(grouped_backend.get(), false) == 0);
    const auto telemetry = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
    CHECK(telemetry.registered == 1 && telemetry.covered == 1 && telemetry.plan_calls == total_passes &&
        telemetry.plan_compiles == 2 && telemetry.plan_reuses == total_passes - telemetry.plan_compiles &&
        telemetry.calls == total_passes && telemetry.ready == total_passes && telemetry.completed == total_passes &&
        telemetry.ready_min == total_passes && telemetry.ready_max == total_passes &&
        telemetry.completed_min == total_passes && telemetry.completed_max == total_passes &&
        telemetry.admitted_banks == total_passes * grouped_primary.banks.size());
    CHECK(telemetry.h2d_banks == total_misses * grouped_primary.banks.size());
    uint64_t bytes_per_expert = 0;
    for (const ggml_tensor * bank : grouped_primary.banks) {
        bytes_per_expert += bank->nb[2];
    }
    CHECK(telemetry.h2d_bytes == total_misses * bytes_per_expert && telemetry.fallback == 0 &&
        telemetry.rollback == 0 && telemetry.prepare_error == 0 && telemetry.finish_error == 0);
    ggml_backend_cuda_moe_set_debug_mm(old_debug_mm);
    fprintf(stderr, capture_available ?
        "test-moe-cache: active grouped Q4_K cache132 B%u/B5 eviction/refill capture exact OK\n" :
        "test-moe-cache: active grouped Q4_K cache132 B%u/B5 eviction/refill direct exact OK (CUDA graphs unavailable)\n",
        primary_rows);
    return true;
}

static void test_active_grouped_q4k_eviction_refill(int device) {
    const bool b8 = test_active_grouped_q4k_eviction_refill_case(device, 8);
    const bool b16 = test_active_grouped_q4k_eviction_refill_case(device, 16);
    fprintf(stderr, "test-moe-cache: active grouped Q4_K natural cases B8=%s B16=%s\n",
        b8 ? "executed" : "skipped", b16 ? "executed" : "skipped");
}

static void test_active_grouped_same_key_row_transitions(int device, uint32_t n_slots) {
    const bool old_debug_mm = ggml_backend_cuda_moe_get_debug_mm();
    ggml_backend_cuda_moe_set_debug_mm(true);
    ggml_backend_ptr reference_backend(ggml_backend_cuda_init(device));
    ggml_backend_ptr candidate_backend(ggml_backend_cuda_init(device));
    CHECK(reference_backend != nullptr && candidate_backend != nullptr);

    auto candidate_b4 = build_active_grouped_dispatch_graph(
        candidate_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, false, 4);
    auto candidate_b1 = build_active_grouped_dispatch_graph(
        candidate_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, false, 1, 8, 2, 256, &candidate_b4);
    auto candidate_b2 = build_active_grouped_dispatch_graph(
        candidate_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, false, 2, 8, 2, 256, &candidate_b4);
    auto candidate_b3 = build_active_grouped_dispatch_graph(
        candidate_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, false, 3, 8, 2, 256, &candidate_b4);
    auto reference_b1 = build_active_grouped_dispatch_graph(
        reference_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, false, 1);
    auto reference_b2 = build_active_grouped_dispatch_graph(
        reference_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, false, 2);
    auto reference_b3 = build_active_grouped_dispatch_graph(
        reference_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, false, 3);
    auto reference_b4 = build_active_grouped_dispatch_graph(
        reference_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, false, 4);

    initialize_active_grouped_dispatch_graphs({&reference_b1, &candidate_b1});
    initialize_active_grouped_dispatch_graphs({&reference_b2, &candidate_b2});
    initialize_active_grouped_dispatch_graphs({&reference_b3, &candidate_b3});
    initialize_active_grouped_dispatch_graphs({&reference_b4, &candidate_b4});
    const auto disabled = candidate_snapshot(n_slots, nullptr, 0);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v1(reference_backend.get(), &disabled) ==
        GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    register_active_grouped_dispatch(
        candidate_backend.get(), candidate_b4, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, n_slots);

    ggml_tensor * shared_first_node = candidate_b4.graph->nodes[0];
    CHECK(shared_first_node != nullptr && shared_first_node->op != GGML_OP_MUL_MAT_ID);
    candidate_insert_graph_node(candidate_b1.graph, 0, shared_first_node);
    candidate_insert_graph_node(candidate_b2.graph, 0, shared_first_node);
    candidate_insert_graph_node(candidate_b3.graph, 0, shared_first_node);
    CHECK(candidate_b1.graph->nodes[0] == shared_first_node && candidate_b2.graph->nodes[0] == shared_first_node &&
        candidate_b3.graph->nodes[0] == shared_first_node && candidate_b4.graph->nodes[0] == shared_first_node);

    auto * context = ggml_cuda_moe_grouped_context_for_test(candidate_backend.get());
    CHECK(context != nullptr);
    std::array<active_grouped_dispatch_graph *, 4> references = {
        &reference_b1, &reference_b2, &reference_b3, &reference_b4,
    };
    std::array<active_grouped_dispatch_graph *, 4> candidates = {
        &candidate_b1, &candidate_b2, &candidate_b3, &candidate_b4,
    };
    for (const auto * candidate : candidates) {
        CHECK(candidate->banks == candidate_b4.banks && candidate->down == candidate_b4.down);
    }
    std::array<uint64_t, 4> semantic_keys = {};
    for (uint32_t row_index = 0; row_index < candidates.size(); ++row_index) {
        semantic_keys[row_index] = ggml_cuda_moe_execution_semantic_key(candidates[row_index]->graph);
        CHECK(semantic_keys[row_index] != 0);
        for (uint32_t previous = 0; previous < row_index; ++previous) {
            CHECK(semantic_keys[previous] != semantic_keys[row_index]);
        }
    }

    uint32_t executed_passes = 0;
    const auto run_checked = [&](uint32_t n_rows, uint32_t route_variant) {
        CHECK(n_rows >= 1 && n_rows <= 4 && route_variant < 3);
        auto & reference = *references[n_rows - 1];
        auto & candidate = *candidates[n_rows - 1];
        set_active_grouped_dispatch_logits({&reference, &candidate}, route_variant);
        std::vector<float> sentinel(ggml_nelements(candidate.output), -4096.0f - n_rows);
        ggml_backend_tensor_set(candidate.output, sentinel.data(), 0, sentinel.size() * sizeof(float));
        const auto expected = run_active_grouped_dispatch(reference_backend.get(), reference, 0, false);
        const auto actual = run_active_grouped_dispatch(candidate_backend.get(), candidate, 0, false);
        ++executed_passes;
        check_active_grouped_exact_output(expected, actual);
        check_active_grouped_routes(reference, n_rows, route_variant);
        check_active_grouped_routes(candidate, n_rows, route_variant);
    };

    (void) candidate_certify_graph(*context, candidate_b4.graph);
    run_checked(4, 0);
    ggml_cuda_moe_candidate_group_key group_key;
    CHECK(context->find_down_group_key(candidate_b4.down, &group_key));
    ggml_cuda_moe_grouped_acquisition initial_resource;
    ggml_cuda_moe_grouped_resource_info initial_info;
    CHECK(context->acquire_group_resources(group_key, &initial_resource) &&
        context->get_group_resources(initial_resource, &initial_info));
    CHECK(initial_info.down == candidate_b4.down && initial_info.n_slots == n_slots &&
        initial_info.n_banks == candidate_b4.banks.size() && !initial_info.transaction_active);
    const uint64_t resource_generation = initial_resource.resource_generation;
    std::array<void *, 2> bank_data = {
        ggml_cuda_moe_grouped_context_test_access::device_bank_data(*context, group_key, candidate_b4.banks[0]),
        ggml_cuda_moe_grouped_context_test_access::device_bank_data(*context, group_key, candidate_b4.banks[1]),
    };
    CHECK(resource_generation != 0 && bank_data[0] != nullptr && bank_data[1] != nullptr);
    const auto check_resource_identity = [&]() {
        ggml_cuda_moe_grouped_acquisition resource;
        ggml_cuda_moe_grouped_resource_info info;
        CHECK(context->acquire_group_resources(group_key, &resource) && context->get_group_resources(resource, &info));
        CHECK(resource.resource_generation == resource_generation && info.down == candidate_b4.down &&
            info.n_slots == n_slots && info.n_banks == candidate_b4.banks.size() && !info.transaction_active);
        CHECK(ggml_cuda_moe_grouped_context_test_access::device_bank_data(
                *context, group_key, candidate_b4.banks[0]) == bank_data[0] &&
            ggml_cuda_moe_grouped_context_test_access::device_bank_data(
                *context, group_key, candidate_b4.banks[1]) == bank_data[1]);
    };
    ggml_cuda_graph_capture_state_for_test state;
    CHECK(ggml_cuda_graph_capture_state_query_for_test(candidate_backend.get(), candidate_b4.graph, &state));
    const bool capture_available = state.capture_available;
    CHECK(state.graph == 0 && state.instance == 0 && !state.warmup_complete && state.moe_resource_fingerprint == 0 &&
        (!capture_available || state.execution_semantic_key == semantic_keys[3]));

    if (!capture_available) {
        for (uint32_t n_rows : {3u, 2u, 1u, 4u}) {
            (void) candidate_certify_graph(*context, candidates[n_rows - 1]->graph);
            run_checked(n_rows, n_rows % 3);
            check_resource_identity();
        }
    } else {
        run_checked(4, 0);
        CHECK(ggml_cuda_graph_capture_state_query_for_test(candidate_backend.get(), candidate_b4.graph, &state));
        CHECK(state.graph != 0 && state.instance != 0 && state.warmup_complete &&
            state.execution_semantic_key == semantic_keys[3] && state.moe_resource_fingerprint != 0);
        uintptr_t captured_graph = state.graph;
        uintptr_t captured_instance = state.instance;
        run_checked(4, 1);
        CHECK(ggml_cuda_graph_capture_state_query_for_test(candidate_backend.get(), candidate_b4.graph, &state));
        CHECK(state.graph == captured_graph && state.instance == captured_instance && state.warmup_complete &&
            state.execution_semantic_key == semantic_keys[3] && state.moe_resource_fingerprint != 0);
        const uint64_t b4_resource_fingerprint = state.moe_resource_fingerprint;

        uint64_t previous_semantic_key = semantic_keys[3];
        const std::array<uint32_t, 4> transitions = {3, 2, 1, 4};
        for (uint32_t transition = 0; transition < transitions.size(); ++transition) {
            const uint32_t n_rows = transitions[transition];
            auto * graph = candidates[n_rows - 1]->graph;
            const uint64_t semantic_key = semantic_keys[n_rows - 1];
            CHECK(semantic_key != previous_semantic_key);
            (void) candidate_certify_graph(*context, graph);
            const uint32_t direct_variant = (transition + 2) % 3;
            run_checked(n_rows, direct_variant);
            check_resource_identity();
            CHECK(ggml_cuda_graph_capture_state_query_for_test(candidate_backend.get(), graph, &state));
            if (n_rows == 4) {
                CHECK(state.graph == captured_graph && state.instance == captured_instance && state.warmup_complete &&
                    state.execution_semantic_key == semantic_keys[3] &&
                    state.moe_resource_fingerprint == b4_resource_fingerprint);
                run_checked(4, direct_variant);
                CHECK(ggml_cuda_graph_capture_state_query_for_test(candidate_backend.get(), graph, &state));
                CHECK(state.graph != 0 && state.instance != 0 && state.warmup_complete &&
                    state.execution_semantic_key == semantic_keys[3] &&
                    state.moe_resource_fingerprint == b4_resource_fingerprint);
                captured_graph = state.graph;
                captured_instance = state.instance;
                run_checked(4, (direct_variant + 1) % 3);
                CHECK(ggml_cuda_graph_capture_state_query_for_test(candidate_backend.get(), graph, &state));
                CHECK(state.graph == captured_graph && state.instance == captured_instance && state.warmup_complete &&
                    state.execution_semantic_key == semantic_keys[3] &&
                    state.moe_resource_fingerprint == b4_resource_fingerprint);
            } else {
                CHECK(state.graph == 0 && state.instance == 0 && !state.warmup_complete &&
                    state.execution_semantic_key == semantic_key && state.moe_resource_fingerprint == 0);
            }
            previous_semantic_key = semantic_key;
        }
        CHECK(previous_semantic_key == semantic_keys[3]);
    }

    CHECK(active_grouped_legacy_op_count(reference_backend.get(), true) +
        active_grouped_legacy_op_count(reference_backend.get(), false) == executed_passes * reference_b1.banks.size());
    CHECK(active_grouped_legacy_op_count(candidate_backend.get(), true) == 0 &&
        active_grouped_legacy_op_count(candidate_backend.get(), false) == 0);
    const auto telemetry = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
    CHECK(telemetry.registered == 1 && telemetry.covered == 1 && telemetry.plan_calls == executed_passes &&
        telemetry.plan_compiles == 4 && telemetry.plan_reuses == executed_passes - telemetry.plan_compiles &&
        telemetry.calls == executed_passes &&
        telemetry.ready == executed_passes && telemetry.completed == executed_passes &&
        telemetry.admitted_banks == executed_passes * candidate_b1.banks.size());
    CHECK(telemetry.ready_min == executed_passes && telemetry.ready_max == executed_passes &&
        telemetry.completed_min == executed_passes && telemetry.completed_max == executed_passes);
    CHECK(telemetry.fallback == 0 && telemetry.rollback == 0 && telemetry.prepare_error == 0 && telemetry.finish_error == 0);
    ggml_backend_cuda_moe_set_debug_mm(old_debug_mm);
    fprintf(stderr, capture_available ?
        "test-moe-cache: shared-bank cache%u B4/B3/B2/B1 capture transition OK\n" :
        "test-moe-cache: shared-bank cache%u B4/B3/B2/B1 direct transition OK (CUDA graphs unavailable)\n",
        n_slots);
}

static void test_active_grouped_speculative_route_limit_transition(
        int device, uint32_t n_slots, uint32_t domain) {
    CHECK(domain == GGML_GRAPH_EXECUTION_DOMAIN_DRAFT || domain == GGML_GRAPH_EXECUTION_DOMAIN_MTP);
    const bool old_debug_mm = ggml_backend_cuda_moe_get_debug_mm();
    ggml_backend_cuda_moe_set_debug_mm(true);
    const bool sequential_fits = 2 * 8 <= n_slots;
    ggml_backend_ptr reference_backend(ggml_backend_cuda_init(device));
    ggml_backend_ptr candidate_backend(ggml_backend_cuda_init(device));
    CHECK(reference_backend != nullptr && candidate_backend != nullptr);

    auto candidate_b1 = build_active_grouped_dispatch_graph(
        candidate_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, false, 1, 128, 8);
    auto candidate_b2 = build_active_grouped_dispatch_graph(
        candidate_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, false, 2, 128, 8, 256, &candidate_b1);
    auto reference_b1 = build_active_grouped_dispatch_graph(
        reference_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, false, 1, 128, 8);
    auto reference_b2 = build_active_grouped_dispatch_graph(
        reference_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, false, 2, 128, 8);
    initialize_active_grouped_dispatch_graphs({&reference_b1, &candidate_b1});
    initialize_active_grouped_dispatch_graphs({&reference_b2, &candidate_b2});
    const auto disabled = candidate_snapshot(n_slots, nullptr, 0);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v1(reference_backend.get(), &disabled) ==
        GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    register_active_grouped_dispatch(
        candidate_backend.get(), candidate_b1, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, n_slots);
    CHECK(candidate_b2.banks == candidate_b1.banks && candidate_b2.down == candidate_b1.down);

    ggml_tensor * shared_first_node = candidate_b1.graph->nodes[0];
    CHECK(shared_first_node != nullptr && shared_first_node->op != GGML_OP_MUL_MAT_ID);
    candidate_insert_graph_node(candidate_b2.graph, 0, shared_first_node);
    CHECK(candidate_b2.graph->nodes[0] == candidate_b1.graph->nodes[0]);
    candidate_stamp_execution(candidate_b1.graph, domain,
        GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 1, 1, 0,
        GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED);
    candidate_stamp_execution(candidate_b2.graph, domain,
        GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL, 2, 1, 0,
        GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED);
    auto * context = ggml_cuda_moe_grouped_context_for_test(candidate_backend.get());
    CHECK(context != nullptr);

    uint32_t executed_passes = 0;
    const auto run_checked = [&](active_grouped_dispatch_graph & reference,
                                 active_grouped_dispatch_graph & candidate,
                                 uint32_t route_variant) {
        set_active_grouped_dispatch_logits({&reference, &candidate}, route_variant);
        std::vector<float> sentinel(ggml_nelements(candidate.output), -8192.0f - candidate.n_rows);
        ggml_backend_tensor_set(candidate.output, sentinel.data(), 0, sentinel.size() * sizeof(float));
        const auto expected = run_active_grouped_dispatch(reference_backend.get(), reference, 0, false);
        const auto actual = run_active_grouped_dispatch(candidate_backend.get(), candidate, 0, false);
        ++executed_passes;
        check_active_grouped_exact_output(expected, actual);
        check_active_grouped_routes(reference, reference.n_rows, route_variant);
        check_active_grouped_routes(candidate, candidate.n_rows, route_variant);
    };

    (void) candidate_certify_graph(*context, candidate_b1.graph);
    run_checked(reference_b1, candidate_b1, 0);
    ggml_cuda_moe_candidate_group_key group_key;
    ggml_cuda_moe_grouped_acquisition initial_resource;
    CHECK(context->find_down_group_key(candidate_b1.down, &group_key) &&
        context->acquire_group_resources(group_key, &initial_resource));
    const uint64_t resource_generation = initial_resource.resource_generation;
    std::array<void *, 2> bank_data = {
        ggml_cuda_moe_grouped_context_test_access::device_bank_data(*context, group_key, candidate_b1.banks[0]),
        ggml_cuda_moe_grouped_context_test_access::device_bank_data(*context, group_key, candidate_b1.banks[1]),
    };
    CHECK(resource_generation != 0 && bank_data[0] != nullptr && bank_data[1] != nullptr);
    ggml_cuda_graph_capture_state_for_test state;
    CHECK(ggml_cuda_graph_capture_state_query_for_test(candidate_backend.get(), candidate_b1.graph, &state));
    const bool capture_available = state.capture_available;
    uint64_t b1_resource_fingerprint = 0;
    if (capture_available) {
        run_checked(reference_b1, candidate_b1, 0);
        run_checked(reference_b1, candidate_b1, 1);
        CHECK(ggml_cuda_graph_capture_state_query_for_test(candidate_backend.get(), candidate_b1.graph, &state));
        CHECK(state.graph != 0 && state.instance != 0 && state.warmup_complete && state.moe_resource_fingerprint != 0);
        b1_resource_fingerprint = state.moe_resource_fingerprint;
    }

    (void) candidate_certify_graph(*context, candidate_b2.graph);
    run_checked(reference_b2, candidate_b2, 2);
    CHECK(active_grouped_legacy_op_count(candidate_backend.get(), true) +
        active_grouped_legacy_op_count(candidate_backend.get(), false) == 0);
    CHECK(ggml_cuda_graph_capture_state_query_for_test(candidate_backend.get(), candidate_b2.graph, &state));
    if (sequential_fits && capture_available) {
        run_checked(reference_b2, candidate_b2, 0);
        run_checked(reference_b2, candidate_b2, 1);
        CHECK(ggml_cuda_graph_capture_state_query_for_test(candidate_backend.get(), candidate_b2.graph, &state));
        CHECK(state.graph != 0 && state.instance != 0 && state.warmup_complete && state.moe_resource_fingerprint != 0);
    } else {
        CHECK(state.graph == 0 && state.instance == 0 && !state.warmup_complete && state.moe_resource_fingerprint == 0);
    }

    (void) candidate_certify_graph(*context, candidate_b1.graph);
    run_checked(reference_b1, candidate_b1, 2);
    if (capture_available) {
        run_checked(reference_b1, candidate_b1, 2);
        run_checked(reference_b1, candidate_b1, 0);
        CHECK(ggml_cuda_graph_capture_state_query_for_test(candidate_backend.get(), candidate_b1.graph, &state));
        CHECK(state.graph != 0 && state.instance != 0 && state.warmup_complete &&
            state.moe_resource_fingerprint == b1_resource_fingerprint);
    }

    ggml_cuda_moe_grouped_acquisition final_resource;
    CHECK(context->acquire_group_resources(group_key, &final_resource));
    CHECK(final_resource.resource_generation == resource_generation &&
        ggml_cuda_moe_grouped_context_test_access::device_bank_data(
            *context, group_key, candidate_b1.banks[0]) == bank_data[0] &&
        ggml_cuda_moe_grouped_context_test_access::device_bank_data(
            *context, group_key, candidate_b1.banks[1]) == bank_data[1]);
    CHECK(active_grouped_legacy_op_count(reference_backend.get(), true) +
        active_grouped_legacy_op_count(reference_backend.get(), false) == executed_passes * candidate_b1.banks.size());
    CHECK(active_grouped_legacy_op_count(candidate_backend.get(), true) +
        active_grouped_legacy_op_count(candidate_backend.get(), false) == 0);
    const auto telemetry = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
    CHECK(telemetry.registered == 1 && telemetry.covered == 1);
    CHECK(telemetry.plan_calls == executed_passes);
    CHECK(telemetry.plan_compiles == 2 && telemetry.plan_reuses == executed_passes - telemetry.plan_compiles);
    CHECK(telemetry.calls == executed_passes && telemetry.ready == executed_passes && telemetry.completed == executed_passes);
    CHECK(telemetry.admitted_banks == executed_passes * candidate_b1.banks.size());
    CHECK(telemetry.host_staged_calls == (sequential_fits ? 0 : 1) &&
        telemetry.host_staged_ops == (sequential_fits ? 0 : candidate_b1.banks.size()) &&
        telemetry.host_staged_split_ops <= telemetry.host_staged_ops &&
        telemetry.strategy_switches == (sequential_fits ? 0 : 2) &&
        telemetry.required_unsupported == 0);
    CHECK(telemetry.fallback == 0 && telemetry.rollback == 0 && telemetry.prepare_error == 0 && telemetry.finish_error == 0);
    if (domain == GGML_GRAPH_EXECUTION_DOMAIN_DRAFT) {
        const uint64_t legacy_before = active_grouped_legacy_op_count(candidate_backend.get());
        candidate_b1.graph->execution_certificate.n_sequences = 0;
        CHECK(ggml_backend_graph_compute(candidate_backend.get(), candidate_b1.graph) == GGML_STATUS_FAILED);
        CHECK(active_grouped_legacy_op_count(candidate_backend.get()) == legacy_before);
        const auto failure = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
        CHECK(failure.calls == 0 && failure.ready == 0 && failure.completed == 0 &&
            failure.required_unsupported >= 1 && failure.fallback == 0 && failure.rollback == 0);
        candidate_b1.graph->execution_certificate.n_sequences = 1;
    }
    ggml_backend_cuda_moe_set_debug_mm(old_debug_mm);
    fprintf(stderr, capture_available ?
        "test-moe-cache: cache%u domain%u independent/sequential/independent capture transition OK\n" :
        "test-moe-cache: cache%u domain%u independent/sequential/independent direct transition OK (CUDA graphs unavailable)\n",
        n_slots, domain);
}

static void test_active_grouped_host_staged_failure_cleanup(int device) {
    const bool old_debug_mm = ggml_backend_cuda_moe_get_debug_mm();
    ggml_backend_cuda_moe_set_debug_mm(true);
    ggml_backend_ptr reference_backend(ggml_backend_cuda_init(device));
    ggml_backend_ptr candidate_backend(ggml_backend_cuda_init(device));
    CHECK(reference_backend != nullptr && candidate_backend != nullptr);

    auto reference = build_active_grouped_dispatch_graph(
        reference_backend.get(), ggml_backend_cuda_buffer_type(device), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, false, 2, 128, 8);
    auto candidate = build_active_grouped_dispatch_graph(
        candidate_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, false, 2, 128, 8);
    initialize_active_grouped_dispatch_graphs({&reference, &candidate});
    register_active_grouped_dispatch(
        candidate_backend.get(), candidate, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 12);
    candidate_stamp_execution(candidate.graph, GGML_GRAPH_EXECUTION_DOMAIN_MTP,
        GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL, 2, 1, 0,
        GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED);
    auto * context = ggml_cuda_moe_grouped_context_for_test(candidate_backend.get());
    CHECK(context != nullptr);
    const auto coverage = candidate_certify_graph(*context, candidate.graph);
    CHECK(coverage.epoch != 0 && coverage.mmid_count == candidate.banks.size());

    set_active_grouped_dispatch_logits({&reference, &candidate}, 2);
    const auto expected = run_active_grouped_dispatch(reference_backend.get(), reference, 0, false);
    std::vector<float> sentinel(ggml_nelements(candidate.output), -12288.0f);
    ggml_backend_tensor_set(candidate.output, sentinel.data(), 0, ggml_nbytes(candidate.output));
    ggml_cuda_moe_candidate_group_key key;
    CHECK(context->find_down_group_key(candidate.down, &key));
    ggml_cuda_moe_grouped_context_test_access::fail_host_staged_evaluator(*context);
    CHECK(ggml_backend_graph_compute(candidate_backend.get(), candidate.graph) == GGML_STATUS_FAILED);
    ggml_backend_synchronize(candidate_backend.get());
    std::vector<float> failed_output(sentinel.size());
    ggml_backend_tensor_get(candidate.output, failed_output.data(), 0, ggml_nbytes(candidate.output));
    CHECK(failed_output == sentinel && !ggml_cuda_moe_grouped_context_test_access::has_device_resource(*context, key));
    CHECK(active_grouped_legacy_op_count(candidate_backend.get(), true) +
        active_grouped_legacy_op_count(candidate_backend.get(), false) == 0);
    const auto failed = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
    CHECK(failed.registered == 1 && failed.covered == 1 && failed.plan_calls == 1 &&
        failed.calls == 1 && failed.ready == 1 && failed.completed == 0 &&
        failed.host_staged_calls == 1 && failed.host_staged_ops == 0 &&
        failed.finish_error >= 1 && failed.required_unsupported >= 1);

    const auto actual = run_active_grouped_dispatch(candidate_backend.get(), candidate, 0, false);
    check_active_grouped_exact_output(expected, actual);
    ggml_cuda_moe_grouped_acquisition acquisition;
    ggml_cuda_moe_grouped_resource_info info;
    CHECK(context->acquire_group_resources(key, &acquisition) &&
        context->get_group_resources(acquisition, &info) && !info.transaction_active);
    ggml_cuda_graph_capture_state_for_test state = {};
    CHECK(ggml_cuda_graph_capture_state_query_for_test(candidate_backend.get(), candidate.graph, &state));
    CHECK(state.graph == 0 && state.instance == 0 && state.moe_resource_fingerprint == 0);
    const auto recovered = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
    CHECK(recovered.registered == 1 && recovered.covered == 1 && recovered.plan_calls == 1 &&
        recovered.calls == 1 && recovered.ready == 1 && recovered.completed == 1 &&
        recovered.host_staged_calls == 1 && recovered.host_staged_ops == candidate.banks.size() &&
        recovered.required_unsupported == 0 && recovered.prepare_error == 0 && recovered.finish_error == 0);
    CHECK(active_grouped_legacy_op_count(candidate_backend.get(), true) +
        active_grouped_legacy_op_count(candidate_backend.get(), false) == 0);
    ggml_backend_cuda_moe_set_debug_mm(old_debug_mm);
    fprintf(stderr, "test-moe-cache: required host-staged evaluator cleanup OK\n");
}

void test_active_grouped_legacy_phase_telemetry(int device) {
    const bool old_debug_mm = ggml_backend_cuda_moe_get_debug_mm();
    ggml_backend_cuda_moe_set_debug_mm(true);
    ggml_backend_ptr reference_backend(ggml_backend_cuda_init(device));
    ggml_backend_ptr candidate_backend(ggml_backend_cuda_init(device));
    CHECK(reference_backend != nullptr && candidate_backend != nullptr);
    auto reference = build_active_grouped_dispatch_graph(
        reference_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, false, 2, 128, 8);
    auto candidate = build_active_grouped_dispatch_graph(
        candidate_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, false, 2, 128, 8);
    initialize_active_grouped_dispatch_graphs({&reference, &candidate});
    const auto disabled = candidate_snapshot(12, nullptr, 0);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v1(reference_backend.get(), &disabled) ==
        GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    register_active_grouped_dispatch(
        candidate_backend.get(), candidate, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 12);
    auto * context = ggml_cuda_moe_grouped_context_for_test(candidate_backend.get());
    CHECK(context != nullptr);

    struct phase_case {
        uint32_t domain;
        uint32_t row_semantics;
        uint32_t n_sequences;
        bool certified;
        bool is_decode;
    };
    const std::array<phase_case, 6> cases = {{
        {GGML_GRAPH_EXECUTION_DOMAIN_MAIN,  GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 2, true,  true},
        {GGML_GRAPH_EXECUTION_DOMAIN_DRAFT, GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 2, true,  true},
        {GGML_GRAPH_EXECUTION_DOMAIN_MTP,   GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 2, true,  true},
        {GGML_GRAPH_EXECUTION_DOMAIN_MAIN,  GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SPECULATIVE, 1, true,  true},
        {GGML_GRAPH_EXECUTION_DOMAIN_MAIN,  GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL,  1, true,  false},
        {GGML_GRAPH_EXECUTION_DOMAIN_INVALID, GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INVALID,    0, false, false},
    }};
    for (size_t index = 0; index < cases.size(); ++index) {
        const auto & current = cases[index];
        if (current.certified) {
            candidate_stamp_execution(
                candidate.graph, current.domain, current.row_semantics, candidate.n_rows, current.n_sequences);
        } else {
            candidate.graph->uid = ggml_graph_next_uid();
            candidate.graph->execution_certificate = {};
        }
        (void) candidate_certify_graph(*context, candidate.graph);
        set_active_grouped_dispatch_logits({&reference, &candidate}, index % 3);
        const uint64_t decode_before = active_grouped_legacy_op_count(candidate_backend.get(), true);
        const uint64_t prefill_before = active_grouped_legacy_op_count(candidate_backend.get(), false);
        const auto expected = run_active_grouped_dispatch(reference_backend.get(), reference, 0, false);
        const auto actual = run_active_grouped_dispatch(candidate_backend.get(), candidate, 0, false);
        CHECK(expected == actual);
        check_active_grouped_routes(reference, reference.n_rows, index % 3);
        check_active_grouped_routes(candidate, candidate.n_rows, index % 3);
        CHECK(active_grouped_legacy_op_count(candidate_backend.get(), true) ==
            decode_before + (current.is_decode ? candidate.banks.size() : 0));
        CHECK(active_grouped_legacy_op_count(candidate_backend.get(), false) ==
            prefill_before + (current.is_decode ? 0 : candidate.banks.size()));
    }
    ggml_backend_cuda_moe_set_debug_mm(old_debug_mm);
    fprintf(stderr, "test-moe-cache: certified legacy phase telemetry OK\n");
}

static void test_active_grouped_stream_coherence_fallback(int device) {
    const bool old_debug_mm = ggml_backend_cuda_moe_get_debug_mm();
    ggml_backend_cuda_moe_set_debug_mm(true);
    ggml_backend_ptr reference_backend(ggml_backend_cuda_init(device));
    ggml_backend_ptr candidate_backend(ggml_backend_cuda_init(device));
    CHECK(reference_backend != nullptr && candidate_backend != nullptr);
    auto reference = build_active_grouped_dispatch_graph(
        reference_backend.get(), ggml_backend_cuda_buffer_type(device), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, false, 1, 8, 1, 256, nullptr, true);
    auto candidate = build_active_grouped_dispatch_graph(
        candidate_backend.get(), ggml_backend_cuda_moe_cached_buffer_type(), GGML_TYPE_Q4_0,
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, false, 1, 8, 1, 256, nullptr, true);
    initialize_active_grouped_dispatch_graphs({&reference, &candidate});
    register_active_grouped_dispatch(
        candidate_backend.get(), candidate, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 12);
    auto * context = ggml_cuda_moe_grouped_context_for_test(candidate_backend.get());
    CHECK(context != nullptr && candidate.concurrent_root != nullptr && candidate.graph->nodes[0] == candidate.ids->view_src &&
        candidate.graph->nodes[1] == candidate.ids);
    (void) candidate_certify_graph(*context, candidate.graph);
    const auto run_graph = [](ggml_backend_t backend, active_grouped_dispatch_graph & graph) {
        CHECK(ggml_backend_graph_compute(backend, graph.graph) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(backend);
        std::vector<float> output(ggml_nelements(graph.output));
        ggml_backend_tensor_get(graph.output, output.data(), 0, ggml_nbytes(graph.output));
        return output;
    };

    ggml_cuda_graph_capture_state_for_test captured = {};
    ggml_cuda_graph_capture_state_for_test grouped_capture = {};
    for (uint32_t pass = 0; pass < 3; ++pass) {
        const uint32_t route_variant = pass == 2 ? 1 : 0;
        set_active_grouped_dispatch_logits({&reference, &candidate}, route_variant);
        const auto expected = run_graph(reference_backend.get(), reference);
        const auto actual = run_graph(candidate_backend.get(), candidate);
        check_active_grouped_exact_output(expected, actual);
        CHECK(ggml_cuda_graph_capture_state_query_for_test(candidate_backend.get(), candidate.graph, &captured));
        if (pass == 0 && !captured.capture_available) {
            ggml_backend_cuda_moe_set_debug_mm(old_debug_mm);
            fprintf(stderr, "test-moe-cache: grouped stream coherence direct OK (CUDA graphs unavailable)\n");
            return;
        }
        if (pass == 1) {
            CHECK(captured.graph != 0 && captured.instance != 0 && captured.warmup_complete &&
                captured.execution_semantic_key != 0 && captured.moe_resource_fingerprint != 0);
            grouped_capture = captured;
        } else if (pass == 2) {
            CHECK(captured.graph == grouped_capture.graph && captured.instance == grouped_capture.instance &&
                captured.execution_semantic_key == grouped_capture.execution_semantic_key &&
                captured.moe_resource_fingerprint == grouped_capture.moe_resource_fingerprint);
        }
    }
    CHECK(captured.graph != 0 && captured.instance != 0 && captured.warmup_complete &&
        captured.execution_semantic_key != 0 && captured.moe_resource_fingerprint != 0);
    const uintptr_t stale_graph = captured.graph;
    const uintptr_t stale_instance = captured.instance;
    const uint64_t stale_fingerprint = captured.moe_resource_fingerprint;
    const auto grouped_before = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
    CHECK(grouped_before.registered == 1 && grouped_before.covered == 1 && grouped_before.calls == 3 &&
        grouped_before.ready == 3 && grouped_before.completed == 3 &&
        grouped_before.admitted_banks == 3 * candidate.banks.size());
    const uint64_t legacy_before = active_grouped_legacy_op_count(candidate_backend.get());

    const char * previous_graph_opt = getenv("GGML_CUDA_GRAPH_OPT");
    const std::string previous_graph_opt_value = previous_graph_opt != nullptr ? previous_graph_opt : "";
#ifdef _WIN32
    CHECK(_putenv_s("GGML_CUDA_GRAPH_OPT", "1") == 0);
#else
    CHECK(setenv("GGML_CUDA_GRAPH_OPT", "1", 1) == 0);
#endif
    ggml_backend_ptr cpu_backend(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    CHECK(cpu_backend != nullptr);
    ggml_backend_t backends[] = {candidate_backend.get(), cpu_backend.get()};
    ggml_backend_buffer_type_t bufts[] = {ggml_backend_cuda_buffer_type(device), ggml_backend_cpu_buffer_type()};
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, bufts, 2, 128, false, false));
    CHECK(sched != nullptr && ggml_backend_sched_alloc_graph(sched.get(), candidate.graph));
    CHECK(ggml_backend_sched_get_n_splits(sched.get()) == 1);
    ggml_graph_execution_certificate certificate = candidate.graph->execution_certificate;
    certificate.source_graph_uid = 0;
    certificate.split_graph_uid = 0;
    candidate.graph->execution_certificate = {};

    set_active_grouped_dispatch_logits({&reference, &candidate}, 2);
    const auto fallback_expected = run_graph(reference_backend.get(), reference);
    auto required_certificate = certificate;
    required_certificate.flags = GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED;
    required_certificate.domain = GGML_GRAPH_EXECUTION_DOMAIN_MTP;
    CHECK(ggml_backend_sched_graph_compute_ext(sched.get(), candidate.graph, &required_certificate) == GGML_STATUS_FAILED);
    CHECK(active_grouped_legacy_op_count(candidate_backend.get()) == legacy_before);
    const auto required_failure = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
    CHECK(required_failure.registered == 1 && required_failure.covered == 1 && required_failure.plan_calls == 1 &&
        required_failure.calls == 0 && required_failure.ready == 0 && required_failure.completed == 0 &&
        required_failure.required_unsupported >= 1);
    CHECK(ggml_backend_sched_graph_compute_ext(sched.get(), candidate.graph, &certificate) == GGML_STATUS_SUCCESS);
    std::vector<float> fallback_actual(ggml_nelements(candidate.output));
    ggml_backend_tensor_get(candidate.output, fallback_actual.data(), 0, ggml_nbytes(candidate.output));
    check_active_grouped_exact_output(fallback_expected, fallback_actual);
    CHECK(active_grouped_legacy_op_count(candidate_backend.get()) == legacy_before + candidate.banks.size());
    const auto fallback_telemetry = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
    CHECK(fallback_telemetry.registered == 1 && fallback_telemetry.covered == 1 && fallback_telemetry.plan_calls == 1 &&
          fallback_telemetry.plan_compiles + fallback_telemetry.plan_reuses == 1 && fallback_telemetry.calls == 0 &&
          fallback_telemetry.ready == 0 && fallback_telemetry.completed == 0 &&
          fallback_telemetry.admitted_banks == 0 && fallback_telemetry.fallback == 1 &&
          fallback_telemetry.decode_legacy == 1 && fallback_telemetry.decode_grouped == 0 &&
          fallback_telemetry.submitted == 0 && fallback_telemetry.rollback == 0 &&
          fallback_telemetry.prepare_error == 0 && fallback_telemetry.finish_error == 0);
    ggml_cuda_graph_capture_state_for_test invalidated = {};
    CHECK(ggml_cuda_graph_capture_state_query_for_test(candidate_backend.get(), candidate.graph, &invalidated));
    CHECK(stale_graph != 0 && stale_instance != 0 && stale_fingerprint != 0 &&
        invalidated.graph == 0 && invalidated.instance == 0 && invalidated.moe_resource_fingerprint == 0 &&
        !invalidated.warmup_complete);

    ggml_set_name(candidate.concurrent_root, "test.active.parallel_root");
    ggml_backend_sched_reset(sched.get());
    CHECK(ggml_backend_sched_alloc_graph(sched.get(), candidate.graph));
    CHECK(ggml_backend_sched_get_n_splits(sched.get()) == 1);
    ggml_cuda_graph_capture_state_for_test recaptured = {};
    for (uint32_t pass = 0; pass < 3; ++pass) {
        const uint32_t route_variant = pass;
        set_active_grouped_dispatch_logits({&reference, &candidate}, route_variant);
        const auto expected = run_graph(reference_backend.get(), reference);
        CHECK(ggml_backend_sched_graph_compute_ext(sched.get(), candidate.graph, &certificate) == GGML_STATUS_SUCCESS);
        std::vector<float> actual(ggml_nelements(candidate.output));
        ggml_backend_tensor_get(candidate.output, actual.data(), 0, ggml_nbytes(candidate.output));
        check_active_grouped_exact_output(expected, actual);
        ggml_cuda_graph_capture_state_for_test state = {};
        CHECK(ggml_cuda_graph_capture_state_query_for_test(candidate_backend.get(), candidate.graph, &state));
        if (pass == 0) {
            CHECK(state.graph == 0 && state.instance == 0 && !state.warmup_complete && state.moe_resource_fingerprint == 0);
        } else if (pass == 1) {
            CHECK(state.graph != 0 && state.instance != 0 && state.warmup_complete && state.moe_resource_fingerprint != 0);
            recaptured = state;
        } else {
            CHECK(state.graph == recaptured.graph && state.instance == recaptured.instance &&
                state.warmup_complete && state.moe_resource_fingerprint == recaptured.moe_resource_fingerprint);
        }
    }
#ifdef _WIN32
    CHECK(_putenv_s("GGML_CUDA_GRAPH_OPT", previous_graph_opt != nullptr ? previous_graph_opt_value.c_str() : "") == 0);
#else
    CHECK(previous_graph_opt != nullptr ?
        setenv("GGML_CUDA_GRAPH_OPT", previous_graph_opt_value.c_str(), 1) == 0 :
        unsetenv("GGML_CUDA_GRAPH_OPT") == 0);
#endif
    ggml_backend_cuda_moe_set_debug_mm(old_debug_mm);
    fprintf(stderr, "test-moe-cache: grouped stream coherence fallback/capture recovery OK\n");
}

void test_active_grouped_multirow_graph_modes(int device) {
    test_active_grouped_lookup_routes(device);
    test_active_grouped_multirow_graph_modes_case(device, 2);
    test_active_grouped_multirow_graph_modes_case(device, 3);
    test_active_grouped_multirow_graph_modes_case(device, 4);
    test_active_grouped_multirow_graph_modes_case(
        device, 4, 8, 2, 12, false, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE);
    test_active_grouped_multirow_graph_modes_case(
        device, 4, 8, 2, 12, false, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_UNGATED);
    test_active_grouped_multirow_graph_modes_case(device, 4, 128, 8, 48, true);
    test_active_grouped_q4k_eviction_refill(device);
    test_active_grouped_same_key_row_transitions(device, 48);
    test_active_grouped_same_key_row_transitions(device, 138);
    for (uint32_t domain : {GGML_GRAPH_EXECUTION_DOMAIN_DRAFT, GGML_GRAPH_EXECUTION_DOMAIN_MTP}) {
        test_active_grouped_speculative_route_limit_transition(device, 12, domain);
        test_active_grouped_speculative_route_limit_transition(device, 48, domain);
    }
    test_active_grouped_host_staged_failure_cleanup(device);
    test_active_grouped_legacy_phase_telemetry(device);
    test_active_grouped_stream_coherence_fallback(device);
    test_mmid_direct_source_view(device);
}

void test_strided_copy_graph_update(int device, bool enabled) {
    std::array<ggml_backend_ptr, 2> backends;
    std::array<ggml_context_ptr, 2> contexts;
    std::array<ggml_backend_buffer_ptr, 2> buffers;
    std::array<ggml_cuda_graph_capture_state_for_test, 2> states;
    ggml_tensor * output = nullptr;
    for (uint32_t i = 0; i < 2; ++i) {
        const int64_t height = i == 0 ? 128 : 192;
        backends[i].reset(ggml_backend_cuda_init(device));
        if (enabled) {
            ggml_backend_cuda_set_decode_boundary_overlap(backends[i].get(), true);
        }
        const ggml_init_params params = {ggml_tensor_overhead() * 8 + ggml_graph_overhead_custom(8, false), nullptr, true};
        contexts[i].reset(ggml_init(params));
        CHECK(backends[i] != nullptr && contexts[i] != nullptr);
        auto * source = ggml_new_tensor_2d(contexts[i].get(), GGML_TYPE_F32, 512, height);
        auto * view = ggml_view_2d(contexts[i].get(), source, 128, height, source->nb[1], 0);
        auto * destination = ggml_new_tensor_2d(contexts[i].get(), GGML_TYPE_F32, 128, height);
        output = ggml_cpy(contexts[i].get(), view, destination);
        auto * graph = ggml_new_graph_custom(contexts[i].get(), 8, false);
        ggml_build_forward_expand(graph, output);
        buffers[i].reset(ggml_backend_alloc_ctx_tensors(contexts[i].get(), backends[i].get()));
        CHECK(buffers[i] != nullptr);
        std::vector<float> values(512 * height);
        std::iota(values.begin(), values.end(), 0.0f);
        ggml_backend_tensor_set(source, values.data(), 0, values.size() * sizeof(float));
        for (uint32_t pass = 0; pass < 3; ++pass) {
            CHECK(ggml_backend_graph_compute(backends[i].get(), graph) == GGML_STATUS_SUCCESS);
        }
        ggml_backend_synchronize(backends[i].get());
        CHECK(ggml_cuda_graph_capture_state_query_for_test(backends[i].get(), graph, &states[i]));
        if (!states[i].capture_available) {
            fprintf(stderr, "test-moe-cache: strided copy graph update skipped (CUDA graphs unavailable)\n");
            return;
        }
        CHECK(states[i].graph != 0 && states[i].instance != 0 && states[i].warmup_complete);
        size_t count = 0;
        auto captured = reinterpret_cast<cudaGraph_t>(states[i].graph);
        CUDA_OK(cudaGraphGetNodes(captured, nullptr, &count));
        CHECK(count == 1);
        cudaGraphNode_t node;
        CUDA_OK(cudaGraphGetNodes(captured, &node, &count));
        cudaGraphNodeType node_type;
        CUDA_OK(cudaGraphNodeGetType(node, &node_type));
        CHECK(node_type == (enabled ? cudaGraphNodeTypeKernel : cudaGraphNodeTypeMemcpy));
        std::vector<float> actual(128 * height);
        ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
        for (size_t j = 0; j < actual.size(); ++j) {
            CHECK(actual[j] == float((j / 128) * 512 + j % 128));
        }
    }
    if (!enabled) {
        fprintf(stderr, "test-moe-cache: default strided copy retains memcpy graph nodes, exact OK\n");
        return;
    }
    auto instance = reinterpret_cast<cudaGraphExec_t>(states[0].instance);
    auto graph = reinterpret_cast<cudaGraph_t>(states[1].graph);
#if CUDART_VERSION >= 12000
    cudaGraphExecUpdateResultInfo result;
    CUDA_OK(cudaGraphExecUpdate(instance, graph, &result));
    CHECK(result.result == cudaGraphExecUpdateSuccess);
#else
    cudaGraphExecUpdateResult result;
    cudaGraphNode_t error_node;
    CUDA_OK(cudaGraphExecUpdate(instance, graph, &error_node, &result));
    CHECK(result == cudaGraphExecUpdateSuccess);
#endif
    std::vector<float> actual(128 * 192, -1.0f);
    ggml_backend_tensor_set(output, actual.data(), 0, actual.size() * sizeof(float));
    CUDA_OK(cudaGraphLaunch(instance, cudaStreamPerThread));
    CUDA_OK(cudaStreamSynchronize(cudaStreamPerThread));
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    for (size_t i = 0; i < actual.size(); ++i) {
        CHECK(actual[i] == float((i / 128) * 512 + i % 128));
    }
    fprintf(stderr, "test-moe-cache: strided copy graph height update exact OK\n");
}
