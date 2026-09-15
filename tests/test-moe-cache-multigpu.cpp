#include "../src/llama-context.h"
#include "../src/llama-graph.h"
#include "ggml-cuda/moe-cache-host.cuh"
#include "test-moe-cache.h"

namespace {

constexpr int n_dim     = 256;
constexpr int n_experts = 8;
constexpr int n_used    = 2;
constexpr int n_slots   = 4;

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
    ggml_context_ptr                                  weights;
    ggml_backend_buffer_ptr                           weight_buffer;
    ggml_backend_buffer_type_t                        buft = nullptr;
    llm_graph_result                                  result{ 256 };
    ggml_tensor *                                     input      = nullptr;
    ggml_tensor *                                     logits[2]  = {};
    ggml_tensor *                                     ids[2]     = {};
    ggml_tensor *                                     up[2]      = {};
    ggml_tensor *                                     gate[2]    = {};
    ggml_tensor *                                     gate_up[2] = {};
    ggml_tensor *                                     down[2]    = {};
    ggml_tensor *                                     output[2]  = {};
    std::vector<ggml_backend_moe_candidate_group_v2>  groups;
    std::vector<ggml_backend_moe_candidate_tensor_v2> tensors;

    explicit layer_fixture(size_t host_budget, uint32_t layout, ggml_type type) {
        weights.reset(ggml_init({ 32 * ggml_tensor_overhead(), nullptr, true }));
        CHECK(weights != nullptr);
        buft = ggml_backend_cuda_moe_cached_bounded_buffer_type(host_budget);
        CHECK(buft != nullptr);
        for (int layer = 0; layer < 2; ++layer) {
            groups.push_back({ layout, GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY, 0, 0 });
            const auto bank = [&](uint32_t role, int rows) {
                auto * tensor = ggml_new_tensor_3d(weights.get(), type, n_dim, rows, n_experts);
                tensors.push_back({ tensor, (uint32_t) layer, role, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE,
                                    GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_CACHED_BUFFER, 0 });
                return tensor;
            };
            if (layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP) {
                gate_up[layer] = bank(GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, 2 * n_dim);
            } else {
                if (layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE) {
                    gate[layer] = bank(GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT, n_dim);
                }
                up[layer] = bank(GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT, n_dim);
            }
            down[layer] = bank(GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, n_dim);
        }
        weight_buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(weights.get(), buft));
        CHECK(weight_buffer != nullptr);
        ggml_backend_buffer_set_usage(weight_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
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

    void build_graph() {
        result.reset();
        auto * ctx = result.get_ctx();
        input      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_dim, 1);
        ggml_set_input(input);
        auto *              cur = input;
        llama_adapter_loras loras;
        llm_graph_params    params{};
        params.arch                = LLM_ARCH_QWEN3MOE;
        params.hparams.n_embd      = n_dim;
        params.hparams.n_layer_all = 2;
        params.hparams.n_expert    = n_experts;
        params.hparams.n_expert_used_arr.fill(n_used);
        params.hparams.n_head_arr.fill(1);
        params.hparams.n_head_kv_arr.fill(1);
        params.ubatch.n_tokens = 1;
        params.loras           = &loras;
        params.res             = &result;
        llm_graph_context builder(params);
        for (int layer = 0; layer < 2; ++layer) {
            logits[layer] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_experts, 1);
            ggml_set_input(logits[layer]);
            cur = output[layer] =
                builder.build_moe_ffn(cur, nullptr, nullptr, up[layer], nullptr, gate[layer], nullptr, down[layer],
                                      nullptr, nullptr, n_experts, n_used, LLM_FFN_SILU, true, 1.0f,
                                      LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, layer, logits[layer], gate_up[layer]);
            auto & region         = result.get_moe_regions().back();
            region.semantic_group = layer;
            region.domain         = GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY;
            ids[layer]            = region.route;
            ggml_set_output(cur);
            ggml_build_forward_expand(result.get_gf(), cur);
        }
    }

    ~layer_fixture() {
        weight_buffer.reset();
        ggml_backend_cuda_moe_cached_free_buffer_type(buft);
    }

    ggml_backend_moe_candidate_snapshot_v2 manifest() const {
        return candidate_snapshot_v2(n_slots, groups.data(), groups.size(), tensors.data(), tensors.size());
    }

    void set_inputs(int iteration) {
        std::vector<float> values(n_dim);
        for (int i = 0; i < n_dim; ++i) {
            values[i] = 0.001f * ((i + iteration) % 17 - 8);
        }
        ggml_backend_tensor_set(input, values.data(), 0, ggml_nbytes(input));
        for (int layer = 0; layer < 2; ++layer) {
            float scores[n_experts];
            for (int i = 0; i < n_experts; ++i) {
                scores[i] = (i + iteration + 3 * layer) % n_experts;
            }
            ggml_backend_tensor_set(logits[layer], scores, 0, sizeof(scores));
        }
    }
};

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
                ggml_type                type        = GGML_TYPE_Q4_0) {
    layer_fixture                 fixture(host_budget, layout, type);
    std::vector<ggml_backend_ptr> owners;
    std::vector<ggml_backend_t>   backends;
    for (int device : devices) {
        owners.emplace_back(ggml_backend_cuda_init(device));
        CHECK(owners.back() != nullptr);
        backends.push_back(owners.back().get());
    }
    if (devices.size() == 2) {
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
    const auto allocate = [&](bool affinity = true, bool measure = false) {
        CHECK(fixture.result.get_moe_regions().size() == 2);
        for (int layer = 0; layer < 2; ++layer) {
            auto * owner  = backends[layer % devices.size()];
            auto & region = fixture.result.get_moe_regions()[layer];
            if (!affinity) {
                continue;
            }
            CHECK(region.layer == layer && region.down == fixture.down[layer] && region.route == fixture.ids[layer]);
            CHECK(std::find(region.operations.begin(), region.operations.end(), fixture.input) ==
                  region.operations.end());
            CHECK(std::find(region.inputs.begin(), region.inputs.end(), fixture.logits[layer]) != region.inputs.end());
            region.external_route = true;
            CHECK(!region.place(sched.get(), owner));
            CHECK(ggml_backend_sched_get_tensor_backend(sched.get(), region.output) == nullptr);
            region.external_route                = false;
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
            region.output->buffer = fixture.weight_buffer.get();
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
                    ggml_backend_sched_set_tensor_backend(sched.get(), node, backends[1 % devices.size()]);
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
            CHECK(ggml_backend_sched_alloc_graph(sched.get(), fixture.result.get_gf()));
        }
        fprintf(stderr, "test-moe-cache: layer fixture devices=%zu splits=%d host_budget=%zu\n", devices.size(),
                ggml_backend_sched_get_n_splits(sched.get()), host_budget);
        for (int layer = 0; layer < 2; ++layer) {
            if (affinity) {
                CHECK(ggml_backend_sched_get_tensor_backend(sched.get(), fixture.output[layer]) ==
                      backends[layer % devices.size()]);
                for (auto * node : fixture.result.get_moe_regions()[layer].operations) {
                    auto * actual = ggml_backend_sched_get_tensor_backend(sched.get(), node);
                    if (ggml_is_view(node)) {
                        CHECK(actual == ggml_backend_sched_get_tensor_backend(sched.get(), node->view_src));
                    } else {
                        auto * expected = split_route && layer == 1 && node == fixture.ids[layer]->src[0] ?
                                              backends.back() :
                                              backends[layer % devices.size()];
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
    const auto disabled = candidate_snapshot_v2(n_slots, nullptr, 0, nullptr, 0);
    const auto publish  = [&](const ggml_backend_moe_candidate_snapshot_v2 & snapshot) {
        ggml_backend_sched_synchronize(sched.get());
        for (size_t i = 0; i < devices.size(); ++i) {
            CHECK(ggml_backend_cuda_moe_candidate_replace_v2(backends[i], &snapshot) ==
                  GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        }
    };
    const auto compute = [&](int iteration) {
        fixture.set_inputs(iteration);
        const auto certificate = layer_certificate();
        CHECK(ggml_backend_sched_graph_compute_async_ext(sched.get(), fixture.result.get_gf(), &certificate) ==
              GGML_STATUS_SUCCESS);
        ggml_backend_sched_synchronize(sched.get());
        return active_grouped_tensor_values(fixture.output[1]);
    };
    if (devices.size() == 2) {
        allocate(true, true);
        for (int layer = 0; layer < 2; ++layer) {
            CHECK(ggml_backend_sched_get_tensor_backend(sched.get(), fixture.output[layer]) == backends[layer]);
        }
        ggml_backend_sched_reset(sched.get());
        fixture.build_graph();
    }
    publish(disabled);
    allocate(false);
    ggml_backend_sched_reset(sched.get());
    fixture.build_graph();
    allocate();
    std::vector<std::vector<float>> expected;
    for (int iteration = 0; iteration < 8; ++iteration) {
        expected.push_back(compute(iteration));
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
        check_active_grouped_exact_output(expected[iteration], compute(iteration));
    }
    for (size_t i = 0; i < devices.size(); ++i) {
        auto * context = ggml_cuda_moe_grouped_context_for_test(backends[i]);
        CHECK(context != nullptr);
        const auto telemetry = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
        fprintf(stderr, "test-moe-cache: device=%d grouped_completed=%llu legacy_ops=%llu\n", devices[i],
                (unsigned long long) telemetry.completed,
                (unsigned long long) active_grouped_legacy_op_count(backends[i]));
        const bool fallback_owner = split_route && (devices.size() == 1 || i == 1);
        CHECK(telemetry.decode_legacy == (fallback_owner ? 8 : 0));
        CHECK(telemetry.fallback == telemetry.decode_legacy);
        uint64_t expected_completed = 0;
        for (int layer = 0; layer < 2; ++layer) {
            if (layer % devices.size() == i && !(split_route && layer == 1)) {
                expected_completed += 8;
            }
        }
        CHECK(telemetry.completed == expected_completed && telemetry.decode_grouped == expected_completed);
        CHECK(telemetry.ready == expected_completed);
        if (!fallback_owner) {
            CHECK(active_grouped_legacy_op_count(backends[i]) == legacy_before[i]);
            if (getenv("GGML_CUDA_DISABLE_GRAPHS") == nullptr) {
                CHECK(telemetry.captures > 0 && telemetry.replays > 0);
            } else {
                CHECK(telemetry.direct > 0 && telemetry.captures == 0 && telemetry.replays == 0);
            }
        }
        CHECK(telemetry.prepare_error == 0 && telemetry.finish_error == 0);
        CHECK(telemetry.submitted == telemetry.completed);
        for (int layer = 0; layer < 2; ++layer) {
            ggml_cuda_moe_candidate_group_key key;
            CHECK(context->find_down_group_key(fixture.down[layer], &key));
            ggml_cuda_moe_candidate_group_info info;
            CHECK(context->get_group(key, &info) && info.semantic_group_index == (uint32_t) layer);
            if (layer % devices.size() != i) {
                CHECK(!ggml_cuda_moe_grouped_context_test_access::has_device_resource(*context, key));
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
                CHECK(telemetry.submitted == (i == 0 ? 1 : 0) && telemetry.completed == telemetry.submitted);
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
    if (!split_route) {
        if (devices.size() == 2) {
            owners[0].reset();
        }
        auto *                        survivor = backends[devices.size() - 1];
        active_grouped_dispatch_graph shared;
        shared.n_experts = n_experts;
        shared.n_used    = n_used;
        for (const auto & bank : fixture.tensors) {
            if (bank.group_index == 1) {
                shared.banks.push_back(const_cast<ggml_tensor *>(bank.tensor));
                shared.roles.push_back(bank.role);
            }
        }
        auto graph = build_active_grouped_dispatch_graph(survivor, fixture.buft, type, layout, false, 1, n_experts,
                                                         n_used, n_dim, &shared);
        // Fill only graph inputs; the shared immutable weights remain unchanged.
        const auto input = cached_fusion_test_data(graph.input, 71);
        ggml_backend_tensor_set(graph.input, input.data(), 0, input.size());
        set_active_grouped_dispatch_logits({ &graph }, 0);
        CHECK(ggml_backend_cuda_moe_candidate_replace_v2(survivor, &disabled) ==
              GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        const auto reference = run_active_grouped_dispatch(survivor, graph, 0);
        CHECK(ggml_backend_cuda_moe_candidate_replace_v2(survivor, &enabled) ==
              GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        auto * context = ggml_cuda_moe_grouped_context_for_test(survivor);
        CHECK(context != nullptr);
        (void) ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
        const auto legacy_before = active_grouped_legacy_op_count(survivor);
        const auto capability =
            native_mmid_capability(devices.back(), graph.banks[0], graph.n_rows, GGML_CUDA_MMID_MAPPING_DIRECT);
        const char * disable_fusion = getenv("GGML_CUDA_DISABLE_FUSION");
        // F3 writes the GLU result directly and leaves the gate/up sentinels intact.
        const bool   f3_skipped =
            (!disable_fusion || std::atoi(disable_fusion) == 0) &&
            layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE && capability.reason == GGML_CUDA_MMID_CAPABILITY_OK &&
            capability.selection == GGML_CUDA_MMID_CONSUMER_MMVQ && graph.banks[0]->type == graph.banks[1]->type &&
            ggml_are_same_shape(graph.banks[0], graph.banks[1]) && ggml_are_same_stride(graph.banks[0], graph.banks[1]);
        fprintf(stderr, "test-moe-cache: survivor device=%d layout=%u type=%s generation=%llu f3_skipped=%d\n",
                devices.back(), layout, ggml_type_name(type), (unsigned long long) context->state().generation,
                f3_skipped);
        for (int pass = 0; pass < 3; ++pass) {
            check_active_grouped_exact_output(reference, run_active_grouped_dispatch(survivor, graph, 0, f3_skipped));
        }
        ggml_backend_synchronize(survivor);
        const auto telemetry = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
        CHECK(telemetry.decode_grouped == 3 && telemetry.ready == 3 && telemetry.submitted == 3 &&
              telemetry.completed == 3);
        CHECK(telemetry.decode_legacy == 0 && telemetry.fallback == 0 &&
              active_grouped_legacy_op_count(survivor) == legacy_before);
        CHECK(telemetry.prepare_error == 0 && telemetry.finish_error == 0);
        if (getenv("GGML_CUDA_DISABLE_GRAPHS") == nullptr) {
            CHECK(telemetry.captures > 0 && telemetry.replays > 0);
        } else {
            CHECK(telemetry.direct == 3 && telemetry.captures == 0 && telemetry.replays == 0);
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

void test_shared_source_owners(int device, size_t host_budget) {
    ggml_backend_ptr first(ggml_backend_cuda_init(device));
    ggml_backend_ptr second(ggml_backend_cuda_init(device));
    CHECK(first && second);
    auto * buft = ggml_backend_cuda_moe_cached_bounded_buffer_type(host_budget);
    auto   a    = build_active_grouped_dispatch_graph(first.get(), buft, GGML_TYPE_Q4_0,
                                                      GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP);
    auto   b = build_active_grouped_dispatch_graph(second.get(), buft, GGML_TYPE_Q4_0,
                                                   GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, false, 1, n_experts,
                                                   n_used, n_dim, &a);
    initialize_active_grouped_dispatch_graphs({ &a, &b });
    ggml_backend_moe_candidate_group_v2               group{ GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP,
                                                             GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY, 0, 0 };
    std::vector<ggml_backend_moe_candidate_tensor_v2> banks;
    for (size_t i = 0; i < a.banks.size(); ++i) {
        banks.push_back({ a.banks[i], 0, a.roles[i], GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE,
                          GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_CACHED_BUFFER, 0 });
    }
    const auto snapshot = candidate_snapshot_v2(n_slots, &group, 1, banks.data(), banks.size());
    if (host_budget) {
        CHECK(ggml_backend_cuda_moe_cached_configure_sources(buft, &snapshot));
    }
    CHECK(ggml_backend_cuda_moe_candidate_replace_v2(first.get(), &snapshot) ==
          GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v2(second.get(), &snapshot) ==
          GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    const auto expected = run_active_grouped_dispatch(first.get(), a, 0);
    check_active_grouped_exact_output(expected, run_active_grouped_dispatch(second.get(), b, 0));
    const auto generation = ggml_cuda_moe_grouped_context_for_test(second.get())->state().generation;
    CHECK(ggml_backend_cuda_moe_candidate_replace_v2(first.get(), &snapshot) ==
          GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(ggml_cuda_moe_grouped_context_for_test(second.get())->state().generation == generation);
    first.reset();
    check_active_grouped_exact_output(expected, run_active_grouped_dispatch(second.get(), b, 0));
    second.reset();
    if (host_budget) {
        auto * budget = static_cast<moe_host_budget *>(buft->context);
        CHECK(budget->staging_bytes == 0 && budget->staging_peak + budget->source_bytes <= budget->limit);
    }
    ggml_backend_cuda_moe_cached_free_buffer_type(buft);
    fprintf(stderr, "test-moe-cache: shared-source owner replacement/teardown OK (one physical device)\n");
}

}  // namespace

void test_grouped_layer_placement() {
    const bool debug = ggml_backend_cuda_moe_get_debug_mm();
    ggml_backend_cuda_moe_set_debug_mm(true);
    int device = 0;
    CUDA_OK(cudaGetDevice(&device));
    run_layers({ device }, 0);
    run_layers({ device }, 512 * 1024);
    run_layers({ device }, 0, true);
    run_layers({ device }, 0, false, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_TYPE_Q4_K);
    run_layers({ device }, 0, false, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_UNGATED);
    test_shared_source_owners(device, 0);
    test_shared_source_owners(device, 512 * 1024);
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
    run_layers({ 0, 1 }, 512 * 1024);
    run_layers({ 0, 1 }, 0, true);
    run_layers({ 0, 1 }, 0, false, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, GGML_TYPE_Q4_K);
    run_layers({ 0, 1 }, 0, false, GGML_BACKEND_MOE_CANDIDATE_LAYOUT_UNGATED);
    ggml_backend_cuda_moe_set_debug_mm(debug);
    fprintf(stderr, "test-moe-cache: two-physical-device grouped layer execution OK\n");
    return 0;
}
