#include "test-moe-cache.h"

struct candidate_test_graph_streams {
    const ggml_tensor * producer = nullptr;
    cudaStream_t producer_stream = nullptr;
    cudaStream_t reader_stream = nullptr;
};

static cudaStream_t candidate_test_graph_mixed_stream(void * data, const ggml_tensor * node) {
    auto * streams = static_cast<candidate_test_graph_streams *>(data);
    return node == streams->producer ? streams->producer_stream : streams->reader_stream;
}

struct candidate_fused_top_k_route {
    candidate_route route;
    ggml_tensor * softmax = nullptr;
    ggml_tensor * reshaped = nullptr;
    ggml_tensor * weights = nullptr;
};

static candidate_fused_top_k_route candidate_fused_top_k(candidate_test_fixture & fixture, int64_t n_experts, int64_t n_routes) {
    candidate_fused_top_k_route result;
    const int64_t logits_ne[] = {n_experts, 1};
    ggml_tensor * logits = fixture.tensor(GGML_TYPE_F32, 2, logits_ne);
    result.softmax = ggml_soft_max(fixture.ctx, logits);
    fixture.materialize(result.softmax);
    result.softmax->flags |= GGML_TENSOR_FLAG_COMPUTE;
    result.reshaped = ggml_reshape_2d(fixture.ctx, result.softmax, n_experts, 1);
    fixture.materialize(result.reshaped);
    result.reshaped->flags |= GGML_TENSOR_FLAG_COMPUTE;
    result.route.source = result.softmax;
    result.route.root = ggml_argsort(fixture.ctx, result.softmax, GGML_SORT_ORDER_DESC);
    fixture.materialize(result.route.root);
    result.route.root->flags |= GGML_TENSOR_FLAG_COMPUTE;
    result.route.ids = ggml_view_4d(fixture.ctx, result.route.root, n_routes, 1, 1, 1,
        result.route.root->nb[1], result.route.root->nb[2], result.route.root->nb[3], 0);
    fixture.materialize(result.route.ids);
    result.route.ids->flags |= GGML_TENSOR_FLAG_COMPUTE;
    result.weights = ggml_get_rows(fixture.ctx, result.reshaped, result.route.ids);
    fixture.materialize(result.weights);
    result.weights->flags |= GGML_TENSOR_FLAG_COMPUTE;
    return result;
}

static ggml_cgraph * candidate_padded_graph(
        candidate_test_fixture & fixture,
        ggml_tensor * padding,
        uint32_t n_padding,
        std::initializer_list<ggml_tensor *> nodes) {
    ggml_cgraph * graph = ggml_new_graph_custom(fixture.ctx, n_padding + nodes.size(), false);
    CHECK(graph != nullptr);
    for (uint32_t i = 0; i < n_padding; ++i) {
        ggml_tensor * node = ggml_dup(fixture.ctx, padding);
        fixture.materialize(node);
        node->flags |= GGML_TENSOR_FLAG_COMPUTE;
        ggml_graph_add_node(graph, node);
    }
    for (ggml_tensor * node : nodes) {
        ggml_graph_add_node(graph, node);
    }
    candidate_rebuild_graph_uses(graph);
    candidate_stamp_single_row_execution(graph);
    return graph;
}

static void candidate_test_graph_views(
        candidate_test_fixture & fixture,
        ggml_cuda_moe_grouped_context & global_registry,
        const candidate_route & fused_route,
        const candidate_route & separate_route,
        ggml_tensor * fused_gate_up,
        ggml_tensor * fused_down,
        ggml_tensor * separate_gate,
        ggml_tensor * separate_up,
        ggml_tensor * separate_down) {
    ggml_cgraph * split_parent = candidate_graph(fixture, {
        fused_route.root, fused_route.ids, fused_gate_up, fused_down,
        separate_route.root, separate_route.ids, separate_gate, separate_up, separate_down,
    });
    ggml_cgraph split_view = ggml_graph_view(split_parent, 0, 4);
    candidate_stamp_execution(&split_view, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
        GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 1, 1, split_parent->uid);
    const auto split_coverage = candidate_certify_graph(global_registry, &split_view);
    std::shared_ptr<ggml_cuda_moe_graph_plan> split_plan;
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(global_registry.prepare_graph_execution(&split_view, 31, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &split_plan, prepared.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(prepared->size() == 1 && prepared->find(fused_down, nullptr));
    }
    const ggml_cuda_moe_graph_plan * stable_split_plan = split_plan.get();
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(global_registry.prepare_graph_execution(&split_view, 32, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &split_plan, prepared.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(split_plan.get() != stable_split_plan && prepared->find(fused_down, nullptr));
    }

    split_plan.reset();
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(global_registry.prepare_graph_execution(
            &split_view, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &split_plan, prepared.get(),
            split_coverage.epoch, split_coverage.nodes, split_coverage.mmid_count, split_coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(prepared->size() == 1 && prepared->find(fused_down, nullptr));
    }
    stable_split_plan = split_plan.get();
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(global_registry.prepare_graph_execution(
            &split_view, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &split_plan, prepared.get(),
            split_coverage.epoch, split_coverage.nodes, split_coverage.mmid_count, split_coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(split_plan.get() == stable_split_plan && prepared->find(fused_down, nullptr));
    }
    {
        ggml_cgraph exact_callback = ggml_graph_view(&split_view, 0, split_view.n_nodes);
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(!global_registry.bind_graph_plan(
            &exact_callback, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, *split_plan, prepared.get(),
            split_coverage.epoch, split_coverage.nodes, split_coverage.mmid_count, split_coverage.mmid_fingerprint));
        ggml_cuda_moe_graph_plan callback_plan;
        global_registry.compile_graph_plan(&exact_callback, 0, &callback_plan, prepared.get(),
            split_coverage.epoch, split_coverage.nodes, split_coverage.mmid_count, split_coverage.mmid_fingerprint);
        CHECK(prepared->outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_LEGACY);
    }
    {
        ggml_cgraph prefix_callback = ggml_graph_view(&split_view, 0, split_view.n_nodes - 1);
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(!global_registry.bind_graph_plan(
            &prefix_callback, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, *split_plan, prepared.get(),
            split_coverage.epoch, split_coverage.nodes, split_coverage.mmid_count, split_coverage.mmid_fingerprint));
        CHECK(!global_registry.bind_graph_plan(
            &split_view, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, *split_plan, prepared.get(),
            split_coverage.epoch + 1, split_coverage.nodes, split_coverage.mmid_count, split_coverage.mmid_fingerprint));
    }

    ggml_cgraph second_split_view = ggml_graph_view(split_parent, 4, split_parent->n_nodes);
    candidate_stamp_execution(&second_split_view, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
        GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 1, 1, split_parent->uid);
    const auto second_split_coverage = candidate_certify_graph(global_registry, &second_split_view);
    std::shared_ptr<ggml_cuda_moe_graph_plan> second_split_plan;
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(global_registry.prepare_graph_execution(
            &second_split_view, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &second_split_plan, prepared.get(),
            second_split_coverage.epoch, second_split_coverage.nodes,
            second_split_coverage.mmid_count, second_split_coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(prepared->size() == 1 && prepared->find(separate_down, nullptr));
    }
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(global_registry.prepare_graph_execution(
            &second_split_view, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &second_split_plan, prepared.get(),
            second_split_coverage.epoch, second_split_coverage.nodes,
            second_split_coverage.mmid_count, second_split_coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(prepared->find(separate_down, nullptr));
    }

    const std::shared_ptr<ggml_cuda_moe_graph_plan> decode_split_plan = split_plan;
    candidate_set_route_tokens(fused_route, {fused_gate_up, fused_down}, 4);
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(global_registry.prepare_graph_execution(
            &split_view, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &split_plan, prepared.get(),
            split_coverage.epoch, split_coverage.nodes, split_coverage.mmid_count, split_coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(split_plan.get() != decode_split_plan.get() &&
            prepared->outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_LEGACY);
        CHECK(!global_registry.bind_graph_plan(
            &split_view, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, *decode_split_plan, prepared.get(),
            split_coverage.epoch, split_coverage.nodes, split_coverage.mmid_count, split_coverage.mmid_fingerprint));
    }
    candidate_set_route_tokens(fused_route, {fused_gate_up, fused_down}, 1);
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(global_registry.prepare_graph_execution(
            &split_view, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &split_plan, prepared.get(),
            split_coverage.epoch, split_coverage.nodes, split_coverage.mmid_count, split_coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(prepared->find(fused_down, nullptr));
    }

    ggml_tensor * split_consumer = ggml_dup(fixture.ctx, fused_down);
    fixture.materialize(split_consumer);
    split_consumer->flags |= GGML_TENSOR_FLAG_COMPUTE;
    ggml_cgraph * split_consumer_parent = candidate_graph(fixture, {
        fused_route.root, fused_route.ids, fused_gate_up, fused_down, split_consumer,
        separate_route.root, separate_route.ids, separate_gate, separate_up, separate_down,
    });
    ggml_cgraph split_consumer_view = ggml_graph_view(split_consumer_parent, 0, 4);
    candidate_stamp_execution(&split_consumer_view, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
        GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 1, 1, split_consumer_parent->uid);
    const auto split_consumer_coverage = candidate_certify_graph(global_registry, &split_consumer_view);
    split_plan.reset();
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(global_registry.prepare_graph_execution(
            &split_consumer_view, 35, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &split_plan, prepared.get(),
            split_consumer_coverage.epoch, split_consumer_coverage.nodes,
            split_consumer_coverage.mmid_count, split_consumer_coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(prepared->size() == 1 && !prepared->find(fused_down, nullptr));
    }
    stable_split_plan = split_plan.get();
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(global_registry.prepare_graph_execution(
            &split_consumer_view, 36, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &split_plan, prepared.get(),
            split_consumer_coverage.epoch, split_consumer_coverage.nodes,
            split_consumer_coverage.mmid_count, split_consumer_coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(split_plan.get() == stable_split_plan && !prepared->find(fused_down, nullptr));
    }
    split_consumer->src[0] = nullptr;
    candidate_rebuild_graph_uses(split_consumer_parent);
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(global_registry.prepare_graph_execution(
            &split_consumer_view, 37, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &split_plan, prepared.get(),
            split_consumer_coverage.epoch, split_consumer_coverage.nodes,
            split_consumer_coverage.mmid_count, split_consumer_coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(split_plan.get() != stable_split_plan && prepared->find(fused_down, nullptr));
    }
}

struct candidate_graph_holder {
    void reset() {
        plan.reset();
        nodes = nullptr;
        epoch = 0;
        mmid_fingerprint = 0;
        n_nodes = 0;
        mmid_count = 0;
    }

    std::shared_ptr<ggml_cuda_moe_graph_plan> plan;
    const void * nodes = nullptr;
    uint64_t epoch = 0;
    uint64_t mmid_fingerprint = 0;
    int32_t n_nodes = 0;
    uint32_t mmid_count = 0;
};

struct candidate_graph_holder_context {
    explicit candidate_graph_holder_context(ggml_cuda_moe_grouped_context & registry) : registry(registry) {}

    candidate_graph_holder & holder(const void * key) {
        auto & result = holders[key];
        if (result == nullptr) {
            result = std::make_unique<candidate_graph_holder>();
        }
        return *result;
    }

    void certify(ggml_cgraph * graph) {
        ggml_cuda_moe_graph_span span;
        CHECK(graph != nullptr && ggml_cuda_moe_graph_span_bounds(graph->nodes, graph->n_nodes, &span));
        const void * key = graph->nodes[0];
        auto existing = holders.find(key);
        if (existing != holders.end()) {
            existing->second->reset();
        }
        uint32_t mmid_count = 0;
        uint64_t mmid_fingerprint = 0;
        const uint64_t epoch = registry.certify_graph_coverage(graph, &mmid_count, &mmid_fingerprint);
        auto & current = holder(key);
        current.reset();
        CHECK(epoch != 0);
        current.nodes = graph->nodes;
        current.epoch = epoch;
        current.mmid_fingerprint = mmid_fingerprint;
        current.n_nodes = graph->n_nodes;
        current.mmid_count = mmid_count;
    }

    bool recover(ggml_cgraph * graph, candidate_graph_holder & holder) {
        holder.reset();
        uint64_t epoch = 0;
        uint32_t mmid_count = 0;
        uint64_t mmid_fingerprint = 0;
        if (!registry.recover_graph_coverage(graph, &epoch, &mmid_count, &mmid_fingerprint)) {
            return false;
        }
        holder.nodes = graph->nodes;
        holder.epoch = epoch;
        holder.mmid_fingerprint = mmid_fingerprint;
        holder.n_nodes = graph->n_nodes;
        holder.mmid_count = mmid_count;
        return true;
    }

    void evict(const void * key) {
        holders.erase(key);
    }

    ggml_cuda_moe_grouped_context & registry;
    std::unordered_map<const void *, std::unique_ptr<candidate_graph_holder>> holders;
};

static ggml_cuda_moe_graph_prepare_result candidate_prepare_graph_holder(
        candidate_graph_holder_context & holder_context,
        ggml_cgraph * graph,
        uint64_t graph_uid,
        ggml_cuda_moe_graph_property_hint property_hint,
        ggml_cuda_moe_graph_execution * execution) {
    auto & holder = holder_context.holder(graph->nodes[0]);
    if (holder.epoch == 0 || holder.nodes != graph->nodes || holder.n_nodes != graph->n_nodes) {
        holder_context.recover(graph, holder);
    }
    std::shared_ptr<ggml_cuda_moe_graph_plan> local_plan;
    auto * plan = &local_plan;
    uint64_t coverage_epoch = 0;
    uint64_t coverage_mmid_fingerprint = 0;
    const void * coverage_nodes = nullptr;
    uint32_t coverage_mmid_count = 0;
    if (holder.epoch != 0 && holder.nodes == graph->nodes && holder.n_nodes == graph->n_nodes) {
        plan = &holder.plan;
        coverage_epoch = holder.epoch;
        coverage_mmid_fingerprint = holder.mmid_fingerprint;
        coverage_nodes = holder.nodes;
        coverage_mmid_count = holder.mmid_count;
    }
    return holder_context.registry.prepare_graph_execution(
        graph, graph_uid, property_hint, plan, execution, coverage_epoch, coverage_nodes,
        coverage_mmid_count, coverage_mmid_fingerprint);
}

static void candidate_test_graph_holder_coverage(
        candidate_test_fixture & fixture,
        const ggml_backend_moe_candidate_snapshot_v1 & snapshot,
        const candidate_route & fused_route,
        const candidate_route & separate_route,
        ggml_tensor * fused_gate_up,
        ggml_tensor * fused_down,
        ggml_tensor * separate_gate,
        ggml_tensor * separate_up,
        ggml_tensor * separate_down) {
    ggml_cuda_moe_grouped_context registry(&fixture.owner, 0);
    CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    candidate_graph_holder_context holder_context(registry);

    ggml_cgraph * parent = candidate_graph(fixture, {
        fused_route.root, fused_route.ids, fused_gate_up, fused_down,
        separate_route.root, separate_route.ids, separate_gate, separate_up, separate_down,
    });
    ggml_cgraph first = ggml_graph_view(parent, 0, 4);
    ggml_cgraph suffix = ggml_graph_view(parent, 4, parent->n_nodes);
    candidate_stamp_execution(&first, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
        GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 1, 1, parent->uid);
    candidate_stamp_execution(&suffix, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
        GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 1, 1, parent->uid);

    holder_context.certify(&first);
    auto & first_holder = holder_context.holder(first.nodes[0]);
    {
        auto execution = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(candidate_prepare_graph_holder(
            holder_context, &first, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, execution.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(execution->find(fused_down, nullptr));
    }
    const uint64_t first_epoch = first_holder.epoch;
    const auto first_plan = first_holder.plan;

    holder_context.certify(&suffix);
    CHECK(first_holder.epoch == first_epoch && first_holder.plan == first_plan);
    {
        auto execution = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(candidate_prepare_graph_holder(
            holder_context, &suffix, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, execution.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(execution->find(separate_down, nullptr));
    }

    const void * suffix_key = suffix.nodes[0];
    const uint64_t suffix_epoch = holder_context.holder(suffix_key).epoch;
    holder_context.evict(suffix_key);
    CHECK(holder_context.holders.find(suffix_key) == holder_context.holders.end());
    {
        auto execution = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(candidate_prepare_graph_holder(
            holder_context, &suffix, 100, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, execution.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(holder_context.holder(suffix_key).epoch == suffix_epoch && execution->find(separate_down, nullptr));
    }
    const auto recovered_plan = holder_context.holder(suffix_key).plan;
    {
        auto execution = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(candidate_prepare_graph_holder(
            holder_context, &suffix, 101, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, execution.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(holder_context.holder(suffix_key).plan == recovered_plan);
    }

    const auto old_suffix_plan = holder_context.holder(suffix_key).plan;
    holder_context.certify(&suffix);
    auto & suffix_holder = holder_context.holder(suffix_key);
    CHECK(suffix_holder.epoch > suffix_epoch && suffix_holder.plan == nullptr);
    {
        auto execution = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(!registry.bind_graph_plan(
            &suffix, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, *old_suffix_plan, execution.get(),
            suffix_holder.epoch, suffix.nodes));
        CHECK(candidate_prepare_graph_holder(
            holder_context, &suffix, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, execution.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
    }

    const auto pre_replace_plan = suffix_holder.plan;
    CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    {
        auto execution = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(candidate_prepare_graph_holder(
            holder_context, &suffix, 102, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, execution.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(suffix_holder.plan != pre_replace_plan && execution->find(separate_down, nullptr));
    }

    candidate_set_route_tokens(separate_route, {separate_gate, separate_up, separate_down}, 4);
    {
        auto execution = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(candidate_prepare_graph_holder(
            holder_context, &suffix, 103, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, execution.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(execution->outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_LEGACY);
    }
    candidate_set_route_tokens(separate_route, {separate_gate, separate_up, separate_down}, 1);
    {
        auto execution = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(candidate_prepare_graph_holder(
            holder_context, &suffix, 104, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, execution.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(execution->find(separate_down, nullptr));
    }

    const candidate_route disjoint_route = candidate_top_k_route(fixture, 4, 2);
    ggml_tensor * disjoint_gate_up = candidate_mmid(fixture, fused_gate_up->src[0], disjoint_route.ids);
    ggml_tensor * disjoint_down = candidate_mmid(fixture, fused_down->src[0], disjoint_route.ids);
    ggml_cgraph * disjoint = candidate_graph(fixture, {
        disjoint_route.root, disjoint_route.ids, disjoint_gate_up, disjoint_down,
    });
    holder_context.certify(disjoint);
    {
        auto execution = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(candidate_prepare_graph_holder(
            holder_context, disjoint, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, execution.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
    }
    auto & disjoint_holder = holder_context.holder(disjoint->nodes[0]);
    const uint64_t disjoint_epoch = disjoint_holder.epoch;
    const auto disjoint_plan = disjoint_holder.plan;
    const uint64_t retained_suffix_epoch = suffix_holder.epoch;
    const auto stale_suffix_plan = suffix_holder.plan;

    holder_context.certify(parent);
    CHECK(suffix_holder.epoch == retained_suffix_epoch && suffix_holder.plan == stale_suffix_plan);
    CHECK(disjoint_holder.epoch == disjoint_epoch && disjoint_holder.plan == disjoint_plan);
    CHECK(stale_suffix_plan != nullptr);
    holder_context.evict(suffix_key);
    for (uint64_t uid = 105; uid < 107; ++uid) {
        auto execution = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(candidate_prepare_graph_holder(
            holder_context, &suffix, uid, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, execution.get()) ==
            (uid == 105 ? GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED : GGML_CUDA_MOE_GRAPH_PREPARE_REUSED));
        const auto & recovered_suffix = holder_context.holder(suffix_key);
        CHECK(execution->find(separate_down, nullptr) && recovered_suffix.epoch == retained_suffix_epoch &&
            recovered_suffix.plan != nullptr);
    }
}

void test_legacy_owner_leases() {
    candidate_test_fixture fixture;
    const int64_t gate_ne[] = {64, 32, 4};
    const int64_t gate_up_ne[] = {64, 64, 4};
    const int64_t down_ne[] = {32, 64, 4};
    ggml_tensor * gate = fixture.tensor(GGML_TYPE_BF16, 3, gate_ne);
    ggml_tensor * up = fixture.tensor(GGML_TYPE_BF16, 3, gate_ne);
    ggml_tensor * gate_up = fixture.tensor(GGML_TYPE_BF16, 3, gate_up_ne);
    ggml_tensor * down = fixture.tensor(GGML_TYPE_BF16, 3, down_ne);
    ggml_tensor * unsupported = fixture.tensor(GGML_TYPE_BF16, 3, gate_up_ne);
    std::array<ggml_backend_moe_candidate_bank_v1, 2> banks = {{
        {gate_up, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, 0},
        {down, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
    }};
    ggml_backend_moe_candidate_group_v1 group = {banks.data(), banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 0, 0};
    auto snapshot = candidate_snapshot(12, &group, 1);

    auto first = std::make_unique<ggml_cuda_moe_grouped_context>(&fixture.owner);
    ggml_cuda_moe_grouped_context second(&fixture.owner);
    CHECK(first->replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(second.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);

    auto first_lease = first->acquire_legacy_cache(gate_up);
    auto second_lease = second.acquire_legacy_cache(gate_up);
    CHECK(first_lease && second_lease);
    CHECK(first_lease.get() == nullptr && second_lease.get() == nullptr);
    CHECK(first_lease.acquisition().owner != second_lease.acquisition().owner);
    CHECK(first_lease.acquisition().tensor == gate_up && first_lease.acquisition().candidate_generation == 1);
    CHECK(first_lease.acquisition().authority_epoch != 0 && first_lease.acquisition().group_index == 0);
    CHECK(first_lease.acquisition().role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT);
    CHECK(first_lease.acquisition().n_slots == 12 && first_lease.acquisition().registered_source == 1);
    CHECK(!second.acquire_legacy_cache(gate_up, &first_lease.acquisition()));

    auto moved_lease = std::move(first_lease);
    CHECK(!first_lease && moved_lease && moved_lease.get() == nullptr);
    const auto stale = moved_lease.acquisition();

    std::atomic<bool> replacement_started{false};
    std::atomic<bool> replacement_done{false};
    std::thread replacement_thread([&]() {
        replacement_started.store(true, std::memory_order_release);
        CHECK(first->replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        replacement_done.store(true, std::memory_order_release);
    });
    while (!replacement_started.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    for (;;) {
        auto rejected = first->acquire_legacy_cache(gate_up);
        if (!rejected) {
            break;
        }
        std::this_thread::yield();
    }
    CHECK(!replacement_done.load(std::memory_order_acquire));
    moved_lease = {};
    replacement_thread.join();
    CHECK(replacement_done.load(std::memory_order_acquire));
    CHECK(first->state().generation == 2 && second.state().generation == 1);
    CHECK(!first->acquire_legacy_cache(gate_up, &stale));

    auto current = first->acquire_legacy_cache(gate_up);
    CHECK(current && current.get() == nullptr);
    CHECK(current.acquisition().candidate_generation == 2 && current.acquisition().authority_epoch > stale.authority_epoch);
    auto wrong_generation = current.acquisition();
    wrong_generation.candidate_generation--;
    CHECK(!first->acquire_legacy_cache(gate_up, &wrong_generation));
    auto wrong_epoch = current.acquisition();
    wrong_epoch.authority_epoch--;
    CHECK(!first->acquire_legacy_cache(gate_up, &wrong_epoch));
    current = {};

    std::array<ggml_backend_moe_candidate_bank_v1, 3> separate_banks = {{
        {gate, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT, 0},
        {up, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT, 0},
        {down, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
    }};
    ggml_backend_moe_candidate_group_v1 separate_group = {
        separate_banks.data(), separate_banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 0, 0,
    };
    auto separate_snapshot = candidate_snapshot(12, &separate_group, 1);
    CHECK(first->replace(&separate_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);

    auto gate_lease = first->acquire_legacy_cache(gate);
    auto up_lease = first->acquire_legacy_cache(up);
    auto down_lease = first->acquire_legacy_cache(down);
    CHECK(gate_lease && up_lease && down_lease);
    CHECK(gate_lease.get() == nullptr && up_lease.get() == nullptr && down_lease.get() == nullptr);
    CHECK(gate_lease.acquisition().group_index == 0 && gate_lease.acquisition().role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT);
    CHECK(up_lease.acquisition().group_index == 0 && up_lease.acquisition().role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT);
    CHECK(down_lease.acquisition().group_index == 0 && down_lease.acquisition().role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT);
    CHECK(gate_lease.acquisition().registered_source == 1 && up_lease.acquisition().registered_source == 1 && down_lease.acquisition().registered_source == 1);

    const auto stale_data = gate_lease.acquisition();
    gate_lease = {};
    void * gate_data = gate->data;
    gate->data = static_cast<uint8_t *>(gate_data) + 1;
    CHECK(!first->acquire_legacy_cache(gate, &stale_data));
    CHECK(!first->acquire_legacy_cache(gate));
    gate->data = gate_data;
    gate_lease = first->acquire_legacy_cache(gate);
    CHECK(gate_lease && gate_lease.get() == nullptr);
    CHECK(gate_lease.acquisition().authority_epoch > stale_data.authority_epoch);
    gate_lease = {};

    const auto stale_stride = up_lease.acquisition();
    up_lease = {};
    const size_t up_stride = up->nb[1];
    up->nb[1]++;
    CHECK(!first->acquire_legacy_cache(up, &stale_stride));
    CHECK(!first->acquire_legacy_cache(up));
    up->nb[1] = up_stride;
    up_lease = first->acquire_legacy_cache(up);
    CHECK(up_lease && up_lease.get() == nullptr);
    CHECK(up_lease.acquisition().authority_epoch > stale_stride.authority_epoch);
    up_lease = {};
    down_lease = {};

    separate_group.flags = 1;
    separate_snapshot.n_slots = 7;
    CHECK(first->replace(&separate_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_REJECTED);
    separate_group.flags = 0;
    auto rejected_lease = first->acquire_legacy_cache(gate);
    CHECK(rejected_lease && rejected_lease.get() == nullptr);
    CHECK(rejected_lease.acquisition().n_slots == 7 && rejected_lease.acquisition().registered_source == 0);
    CHECK(rejected_lease.acquisition().group_index == UINT32_MAX);
    CHECK(rejected_lease.acquisition().role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID);
    rejected_lease = {};

    auto disabled_snapshot = candidate_snapshot(9, nullptr, 0);
    CHECK(first->replace(&disabled_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    auto disabled_lease = first->acquire_legacy_cache(gate);
    CHECK(disabled_lease && disabled_lease.get() == nullptr);
    CHECK(disabled_lease.acquisition().n_slots == 9 && disabled_lease.acquisition().registered_source == 0);
    CHECK(disabled_lease.acquisition().group_index == UINT32_MAX);
    CHECK(disabled_lease.acquisition().role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID);
    disabled_lease = {};

    auto unsupported_lease = first->acquire_legacy_cache(unsupported);
    CHECK(unsupported_lease && unsupported_lease.get() == nullptr);
    CHECK(unsupported_lease.acquisition().group_index == UINT32_MAX);
    CHECK(unsupported_lease.acquisition().role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID);
    CHECK(unsupported_lease.acquisition().registered_source == 0 && unsupported_lease.acquisition().n_slots == 9);
    unsupported_lease = {};
    second_lease = {};

    auto null_cache_owner = std::make_unique<ggml_cuda_moe_grouped_context>(&fixture.owner);
    auto null_cache_snapshot = candidate_snapshot(0, nullptr, 0);
    CHECK(null_cache_owner->replace(&null_cache_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    auto null_cache_operation = null_cache_owner->begin_legacy_operation();
    CHECK(null_cache_operation && !null_cache_owner->acquire_legacy_cache(down));
    auto replacement_snapshot = candidate_snapshot(9, nullptr, 0);
    std::atomic<bool> null_replacement_started{false};
    std::atomic<bool> null_replacement_done{false};
    std::thread null_replacement_thread([&]() {
        null_replacement_started.store(true, std::memory_order_release);
        CHECK(null_cache_owner->replace(&replacement_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        null_replacement_done.store(true, std::memory_order_release);
    });
    while (!null_replacement_started.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    for (;;) {
        auto rejected = null_cache_owner->begin_legacy_operation();
        if (!rejected) {
            break;
        }
        std::this_thread::yield();
    }
    CHECK(!null_replacement_done.load(std::memory_order_acquire));
    null_cache_operation = {};
    null_replacement_thread.join();
    CHECK(null_replacement_done.load(std::memory_order_acquire));

    CHECK(null_cache_owner->replace(&null_cache_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    null_cache_operation = null_cache_owner->begin_legacy_operation();
    CHECK(null_cache_operation && !null_cache_owner->acquire_legacy_cache(down));
    auto * null_cache_context = null_cache_owner.get();
    std::atomic<bool> null_shutdown_started{false};
    std::atomic<bool> null_shutdown_done{false};
    std::thread null_shutdown_thread([&]() {
        null_shutdown_started.store(true, std::memory_order_release);
        null_cache_context->shutdown();
        null_shutdown_done.store(true, std::memory_order_release);
    });
    while (!null_shutdown_started.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    for (;;) {
        auto rejected = null_cache_context->begin_legacy_operation();
        if (!rejected) {
            break;
        }
        std::this_thread::yield();
    }
    CHECK(!null_shutdown_done.load(std::memory_order_acquire));
    null_cache_operation = {};
    null_shutdown_thread.join();
    CHECK(null_shutdown_done.load(std::memory_order_acquire));
    null_cache_owner.reset();

    auto terminal = std::make_unique<ggml_cuda_moe_grouped_context>(&fixture.owner);
    CHECK(terminal->replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    auto terminal_lease = terminal->acquire_legacy_cache(down);
    CHECK(terminal_lease && terminal_lease.get() == nullptr);
    auto * terminal_context = terminal.get();
    std::atomic<bool> shutdown_started{false};
    std::atomic<bool> shutdown_done{false};
    std::thread shutdown_thread([&]() {
        shutdown_started.store(true, std::memory_order_release);
        terminal_context->shutdown();
        shutdown_done.store(true, std::memory_order_release);
        terminal.reset();
    });
    while (!shutdown_started.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    for (;;) {
        auto rejected = terminal_context->acquire_legacy_cache(down);
        if (!rejected) {
            break;
        }
        std::this_thread::yield();
    }
    CHECK(!shutdown_done.load(std::memory_order_acquire));
    terminal_lease = {};
    shutdown_thread.join();
    CHECK(shutdown_done.load(std::memory_order_acquire));

    fprintf(stderr, "test-moe-cache: legacy owner leases OK\n");
}

void test_grouped_context_resources() {
    candidate_test_fixture fixture;
    const int64_t gate_up_ne[] = {64, 64, 4};
    const int64_t down_ne[] = {32, 64, 4};
    ggml_tensor * gate_up = fixture.tensor(GGML_TYPE_BF16, 3, gate_up_ne);
    ggml_tensor * down = fixture.tensor(GGML_TYPE_BF16, 3, down_ne);
    ggml_tensor * gate_up_peer = fixture.tensor(GGML_TYPE_BF16, 3, gate_up_ne);
    ggml_tensor * down_peer = fixture.tensor(GGML_TYPE_BF16, 3, down_ne);
    std::array<ggml_backend_moe_candidate_bank_v1, 2> banks = {{
        {gate_up, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, 0},
        {down, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
    }};
    std::array<ggml_backend_moe_candidate_bank_v1, 2> peer_banks = {{
        {gate_up_peer, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, 0},
        {down_peer, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
    }};
    ggml_backend_moe_candidate_group_v1 group = {banks.data(), banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 0, 0};
    ggml_backend_moe_candidate_group_v1 peer_group = {peer_banks.data(), peer_banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 0, 0};
    std::array<ggml_backend_moe_candidate_group_v1, 2> groups = {group, peer_group};
    auto snapshot = candidate_snapshot(12, groups.data(), groups.size());

    auto first = std::make_unique<ggml_cuda_moe_grouped_context>(&fixture.owner);
    ggml_cuda_moe_grouped_context second(&fixture.owner);
    CHECK(first->replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    snapshot.n_slots = 48;
    CHECK(second.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);

    ggml_cuda_moe_candidate_group_key first_key;
    ggml_cuda_moe_candidate_group_key peer_key;
    ggml_cuda_moe_candidate_group_key second_key;
    CHECK(first->find_down_group_key(down, &first_key));
    CHECK(first->find_down_group_key(down_peer, &peer_key));
    CHECK(second.find_down_group_key(down, &second_key));
    ggml_cuda_moe_grouped_acquisition first_acquisition;
    ggml_cuda_moe_grouped_acquisition peer_acquisition;
    ggml_cuda_moe_grouped_acquisition repeated_acquisition;
    ggml_cuda_moe_grouped_acquisition second_acquisition;
    CHECK(first->acquire_group_resources(first_key, &first_acquisition));
    CHECK(first->acquire_group_resources(first_key, &repeated_acquisition));
    CHECK(first->acquire_group_resources(peer_key, &peer_acquisition));
    CHECK(second.acquire_group_resources(second_key, &second_acquisition));
    CHECK(first_acquisition.resource_generation == 1);
    CHECK(repeated_acquisition.resource_generation == first_acquisition.resource_generation);
    CHECK(peer_acquisition.resource_generation == 2);
    CHECK(second_acquisition.resource_generation == 1);

    ggml_cuda_moe_grouped_resource_info resource_info;
    CHECK(first->get_group_resources(first_acquisition, &resource_info));
    CHECK(resource_info.acquisition.candidate.generation == 1 && resource_info.n_slots == 12);
    CHECK(resource_info.down == down && resource_info.layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP && resource_info.n_banks == 2);
    ggml_cuda_moe_grouped_transaction inactive_transaction;
    inactive_transaction.acquisition = first_acquisition;
    CHECK(!first->get_group_resource_bank(inactive_transaction, 0, nullptr));
    ggml_cuda_moe_grouped_transaction first_transaction;
    ggml_cuda_moe_grouped_transaction repeated_transaction;
    CHECK(first->begin_group_transaction(first_acquisition, &first_transaction));
    CHECK(!first->begin_group_transaction(first_acquisition, &repeated_transaction));
    CHECK(repeated_transaction.transaction_token == 0);
    CHECK(first->get_group_resources(first_acquisition, &resource_info) && resource_info.transaction_active == 1);
    bool found_gate_up = false;
    bool found_down = false;
    for (uint32_t i = 0; i < resource_info.n_banks; ++i) {
        ggml_cuda_moe_grouped_bank_descriptor descriptor;
        CHECK(first->get_group_resource_bank(first_transaction, i, &descriptor));
        CHECK(descriptor.buffer == descriptor.tensor->buffer && descriptor.buft == descriptor.tensor->buffer->buft);
        CHECK(descriptor.source_data == descriptor.tensor->data && descriptor.buffer_base == fixture.storage);
        CHECK(descriptor.buffer_size == candidate_test_fixture::BUFFER_SIZE && descriptor.byte_extent == ggml_nbytes(descriptor.tensor));
        CHECK(descriptor.data_offset == static_cast<uint64_t>(static_cast<const uint8_t *>(descriptor.source_data) - static_cast<const uint8_t *>(descriptor.buffer_base)));
        CHECK(descriptor.expert_stride == descriptor.tensor->nb[2] && descriptor.alignment == ggml_backend_buffer_get_alignment(descriptor.buffer));
        CHECK(descriptor.encoding == GGML_CUDA_MOE_CANDIDATE_ENCODING_PLAIN);
        CHECK(descriptor.movement == GGML_CUDA_MOE_CANDIDATE_MOVEMENT_SLOT_BOUND);
        CHECK(descriptor.index_modes == (GGML_CUDA_MOE_CANDIDATE_INDEX_GROUP_SLOT_DIRECT | GGML_CUDA_MOE_CANDIDATE_INDEX_ORIGINAL_SOURCE_MAP));
        for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
            CHECK(descriptor.ne[dim] == descriptor.tensor->ne[dim] && descriptor.nb[dim] == descriptor.tensor->nb[dim]);
        }
        if (descriptor.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT) {
            CHECK(descriptor.tensor == gate_up && descriptor.type == GGML_TYPE_BF16);
            found_gate_up = true;
        } else if (descriptor.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT) {
            CHECK(descriptor.tensor == down && descriptor.type == GGML_TYPE_BF16);
            found_down = true;
        } else {
            CHECK(false);
        }
    }
    CHECK(found_gate_up && found_down);

    auto wrong_transaction = first_transaction;
    ++wrong_transaction.acquisition.resource_generation;
    CHECK(!first->end_group_transaction(wrong_transaction));
    const uint64_t logical_signature = first->state().logical_signature;
    CHECK(first->end_group_transaction(first_transaction));
    CHECK(first->begin_group_transaction(first_acquisition, &repeated_transaction));
    CHECK(repeated_transaction.transaction_token > first_transaction.transaction_token);
    CHECK(!first->end_group_transaction(first_transaction));
    CHECK(!first->get_group_resource_bank(first_transaction, 0, nullptr));
    CHECK(first->get_group_resource_bank(repeated_transaction, 0, nullptr));
    CHECK(first->end_group_transaction(repeated_transaction));
    CHECK(!first->end_group_transaction(repeated_transaction));

    ggml_cuda_moe_grouped_transaction held_transaction;
    CHECK(first->begin_group_transaction(first_acquisition, &held_transaction));
    auto * first_context = first.get();
    std::atomic<bool> replacement_started{false};
    std::atomic<bool> replacement_done{false};
    std::atomic<int32_t> replacement_result{-1};
    std::thread replacement_thread([&]() {
        replacement_started.store(true, std::memory_order_release);
        replacement_result.store(first_context->replace(&snapshot), std::memory_order_release);
        replacement_done.store(true, std::memory_order_release);
    });
    while (!replacement_started.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    for (;;) {
        ggml_cuda_moe_grouped_transaction peer_transaction;
        if (!first->begin_group_transaction(peer_acquisition, &peer_transaction)) {
            break;
        }
        CHECK(first->end_group_transaction(peer_transaction));
        std::this_thread::yield();
    }
    CHECK(!replacement_done.load(std::memory_order_acquire));
    CHECK(first->state().generation == 1 && first->state().n_slots == 12);
    CHECK(first->get_group_resources(first_acquisition, &resource_info) && resource_info.transaction_active == 1);
    CHECK(!first->acquire_group_resources(peer_key, &repeated_acquisition));
    CHECK(repeated_acquisition.resource_generation == 0);
    CHECK(first->get_group_resource_bank(held_transaction, 0, nullptr));
    CHECK(first->end_group_transaction(held_transaction));
    replacement_thread.join();
    CHECK(replacement_done.load(std::memory_order_acquire));
    CHECK(replacement_result.load(std::memory_order_acquire) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);

    CHECK(first->state().generation == 2 && first->state().n_slots == 48);
    CHECK(first->state().logical_signature == logical_signature);
    CHECK(!first->get_group_resources(first_acquisition, nullptr));
    CHECK(!first->get_group_resources(peer_acquisition, nullptr));
    CHECK(!first->end_group_transaction(held_transaction));
    CHECK(!first->get_group_resource_bank(held_transaction, 0, nullptr));
    CHECK(second.get_group_resources(second_acquisition, &resource_info));
    CHECK(resource_info.n_slots == 48 && resource_info.acquisition.candidate.generation == 1);

    ggml_cuda_moe_candidate_group_key replacement_key;
    ggml_cuda_moe_grouped_acquisition replacement_acquisition;
    CHECK(first->find_down_group_key(down, &replacement_key));
    CHECK(first->acquire_group_resources(replacement_key, &replacement_acquisition));
    CHECK(replacement_acquisition.resource_generation == 3);
    auto disabled = candidate_snapshot(12, nullptr, 0);
    CHECK(first->replace(&disabled) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(first->state().generation == 3 && first->state().accepted == 1 && first->state().n_groups == 0);
    CHECK(!first->get_group_resources(replacement_acquisition, nullptr));
    CHECK(!first->acquire_group_resources(replacement_key, &repeated_acquisition));

    snapshot.n_slots = 12;
    CHECK(first->replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(first->state().generation == 4 && first->state().logical_signature == logical_signature);
    CHECK(first->find_down_group_key(down, &replacement_key));
    CHECK(first->acquire_group_resources(replacement_key, &replacement_acquisition));
    CHECK(replacement_acquisition.resource_generation == 4);
    ggml_cuda_moe_grouped_transaction shutdown_transaction;
    CHECK(first->begin_group_transaction(replacement_acquisition, &shutdown_transaction));
    std::atomic<bool> shutdown_started{false};
    std::atomic<bool> shutdown_done{false};
    std::thread shutdown_thread([&]() {
        shutdown_started.store(true, std::memory_order_release);
        first_context->shutdown();
        shutdown_done.store(true, std::memory_order_release);
    });
    while (!shutdown_started.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    while (first->get_group_resources(replacement_acquisition, nullptr)) {
        std::this_thread::yield();
    }
    CHECK(!shutdown_done.load(std::memory_order_acquire));
    CHECK(first->get_group_resource_bank(shutdown_transaction, 0, nullptr));
    CHECK(first->end_group_transaction(shutdown_transaction));
    shutdown_thread.join();
    CHECK(shutdown_done.load(std::memory_order_acquire));
    first.reset();
    CHECK(second.find_down_group_key(down, nullptr));
    CHECK(second.get_group_resources(second_acquisition, &resource_info));
    ggml_cuda_moe_grouped_transaction second_transaction;
    CHECK(second.begin_group_transaction(second_acquisition, &second_transaction));
    CHECK(second.end_group_transaction(second_transaction));

    fprintf(stderr, "test-moe-cache: grouped context resources OK\n");
}

void test_grouped_graph_preflight(bool benchmark) {
    CHECK(sizeof(ggml_cuda_moe_graph_plan) <= 128 * 1024);
    CHECK(ggml_cuda_moe_grouped_context_test_access::graph_reader_witness_size() <= 640);
    CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_record_size() <= 4160);
    CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_observation_size() <= 4096);
    fprintf(stderr, "test-moe-cache: graph witness sizes plan=%zu reader=%zu group=%zu observation=%zu\n",
        sizeof(ggml_cuda_moe_graph_plan),
        ggml_cuda_moe_grouped_context_test_access::graph_reader_witness_size(),
        ggml_cuda_moe_grouped_context_test_access::graph_group_record_size(),
        ggml_cuda_moe_grouped_context_test_access::graph_group_observation_size());

    candidate_test_fixture fixture;
    const int64_t gate_ne[] = {256, 256, 4};
    const int64_t fused_ne[] = {256, 512, 4};
    const int64_t ids_ne[] = {2, 1};
    const int64_t fused_bias_ne[] = {512, 4};

    ggml_tensor * fused_gate_up = fixture.tensor(GGML_TYPE_Q4_0, 3, fused_ne);
    ggml_tensor * fused_down = fixture.tensor(GGML_TYPE_Q4_0, 3, gate_ne);
    ggml_tensor * separate_gate = fixture.tensor(GGML_TYPE_Q4_K, 3, gate_ne);
    ggml_tensor * separate_up = fixture.tensor(GGML_TYPE_Q4_K, 3, gate_ne);
    ggml_tensor * separate_down = fixture.tensor(GGML_TYPE_Q4_K, 3, gate_ne);
    ggml_tensor * fused_bias = fixture.tensor(GGML_TYPE_F32, 2, fused_bias_ne);
    ggml_tensor * external_ids = fixture.tensor(GGML_TYPE_I32, 2, ids_ne);
    const candidate_route fused_route = candidate_top_k_route(fixture, 4, 2);
    const candidate_route fused_route_other = candidate_top_k_route(fixture, 4, 2);
    const candidate_route separate_route = candidate_top_k_route(fixture, 4, 2);

    std::array<ggml_backend_moe_candidate_bank_v1, 2> fused_banks = {{
        {fused_gate_up, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, 0},
        {fused_down, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
    }};
    std::array<ggml_backend_moe_candidate_bank_v1, 3> separate_banks = {{
        {separate_gate, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT, 0},
        {separate_up, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT, 0},
        {separate_down, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
    }};
    std::array<ggml_backend_moe_candidate_group_v1, 2> groups = {{
        {fused_banks.data(), fused_banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 0, 0},
        {separate_banks.data(), separate_banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 0, 0},
    }};
    auto snapshot = candidate_snapshot(12, groups.data(), groups.size());
    ggml_cuda_moe_grouped_context registry(&fixture.owner, 0);
    CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);

    ggml_tensor * fused_gate_up_node = candidate_mmid(fixture, fused_gate_up, fused_route.ids);
    ggml_tensor * fused_down_node = candidate_mmid(fixture, fused_down, fused_route.ids);
    ggml_tensor * separate_up_node = candidate_mmid(fixture, separate_up, separate_route.ids);
    ggml_tensor * separate_gate_node = candidate_mmid(fixture, separate_gate, separate_route.ids);
    ggml_tensor * separate_down_node = candidate_mmid(fixture, separate_down, separate_route.ids);
    ggml_cgraph * complete_graph = candidate_graph(fixture, {
        fused_route.root, fused_route.ids, separate_route.root, separate_route.ids,
        fused_gate_up_node, fused_down_node, separate_up_node, separate_gate_node, separate_down_node,
    });

    candidate_test_graph_views(fixture, registry, fused_route, separate_route,
        fused_gate_up_node, fused_down_node, separate_gate_node, separate_up_node, separate_down_node);
    candidate_test_graph_holder_coverage(fixture, snapshot, fused_route, separate_route,
        fused_gate_up_node, fused_down_node, separate_gate_node, separate_up_node, separate_down_node);

    ggml_cuda_moe_graph_plan plan;
    ggml_cuda_moe_graph_execution execution;
    auto reused = std::make_unique<ggml_cuda_moe_graph_execution>();
    registry.compile_graph_plan(complete_graph, 41, &plan, &execution);
    CHECK(plan.size() == 2 && execution.size() == 2);
    CHECK(!execution.has_stream_grouped_candidate());
    CHECK(!execution.has_coherent_grouped_streams());
    CHECK(execution.resolve_streams(candidate_test_graph_stream, reinterpret_cast<void *>(uintptr_t{1})));
    CHECK(execution.has_stream_grouped_candidate());
    CHECK(execution.has_coherent_grouped_streams());
    CHECK(plan.registry_generation() == 1 && plan.graph_uid() == 41 && plan.graph_node_count() == 9);
    ggml_cuda_moe_graph_binding binding;
    CHECK(execution.find(fused_gate_up_node, &binding));
    CHECK(binding.key.candidate.generation == 1 && binding.key.candidate.group_index == 0);
    CHECK(binding.key.ids.tensor == fused_route.ids && binding.key.ids.data == fused_route.ids->data && binding.key.ids.buffer == fused_route.ids->buffer);
    CHECK(binding.key.layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP && binding.key.n_banks == 2);
    CHECK(binding.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT && binding.bank_index == 0 && binding.slot_index == 0);
    CHECK(execution.find(fused_down_node, &binding) && binding.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT &&
        binding.bank_index == 1 && binding.slot_index == 1);
    CHECK(execution.find(separate_up_node, &binding) && binding.key.candidate.group_index == 1 && binding.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT);
    CHECK(execution.find(separate_gate_node, &binding) && binding.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT);
    CHECK(execution.find(separate_down_node, &binding) && binding.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT && binding.key.n_banks == 3);
    CHECK(!execution.find(fused_bias, nullptr));

    const int64_t route_weight_ne[] = {1, 4, 1};
    ggml_tensor * route_weight_source = fixture.tensor(GGML_TYPE_F32, 3, route_weight_ne);
    ggml_tensor * route_weights = ggml_get_rows(fixture.ctx, route_weight_source, fused_route.ids);
    fixture.materialize(route_weights);
    route_weights->flags |= GGML_TENSOR_FLAG_COMPUTE;
    ggml_tensor * weighted_gate_up = candidate_mmid(fixture, fused_gate_up, fused_route.ids);
    ggml_tensor * weighted_down = candidate_mmid(fixture, fused_down, fused_route.ids);
    ggml_tensor * weighted_output = ggml_mul(fixture.ctx, weighted_down, route_weights);
    fixture.materialize(weighted_output);
    weighted_output->flags |= GGML_TENSOR_FLAG_COMPUTE;
    ggml_cgraph * weighted_graph = candidate_graph(fixture, {
        fused_route.root, fused_route.ids, route_weights, weighted_gate_up, weighted_down, weighted_output,
    });
    registry.compile_graph_plan(weighted_graph, 410, &plan, &execution);
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED && execution.size() == 1);
    CHECK(execution.find(weighted_gate_up, nullptr) && execution.find(weighted_down, nullptr));
    CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_eligible_reason(plan, 0));

    {
        ggml_cuda_moe_graph_execution stream_execution;
        registry.compile_graph_plan(complete_graph, 41, &plan, &stream_execution);
        candidate_test_graph_streams streams = {
            fused_route.root,
            reinterpret_cast<cudaStream_t>(uintptr_t{2}),
            reinterpret_cast<cudaStream_t>(uintptr_t{1}),
        };
        CHECK(stream_execution.resolve_streams(candidate_test_graph_mixed_stream, &streams));
        CHECK(stream_execution.has_stream_grouped_candidate());
        CHECK(!stream_execution.has_coherent_grouped_streams());
        CHECK(!registry.begin_graph_dispatch(&stream_execution, true));
        CHECK(stream_execution.find_authority(fused_gate_up_node) == nullptr);
        CHECK(stream_execution.find_authority(separate_gate_node) == nullptr);
        CHECK(registry.begin_graph_dispatch(&stream_execution, GGML_CUDA_MOE_GRAPH_DISPATCH_LEGACY));
        const auto * fused_authority = stream_execution.find_authority(fused_gate_up_node);
        const auto * separate_authority = stream_execution.find_authority(separate_gate_node);
        const auto * fused_dispatch = stream_execution.find_group(fused_gate_up_node, nullptr);
        CHECK(fused_authority != nullptr && fused_authority->authority() == GGML_CUDA_MOE_GROUP_AUTHORITY_LEGACY);
        CHECK(fused_dispatch != nullptr && fused_dispatch->state == GGML_CUDA_MOE_GRAPH_GROUP_WHOLE_LEGACY &&
            fused_dispatch->transaction.transaction_token == 0);
        CHECK(separate_authority != nullptr && separate_authority->authority() == GGML_CUDA_MOE_GROUP_AUTHORITY_LEGACY);
        CHECK(registry.finish_graph_dispatch(&stream_execution));
    }

    CHECK(registry.bind_graph_plan(complete_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, plan, reused.get()));
    CHECK(reused->size() == 2 && reused->find(separate_down_node, nullptr));
    CHECK(!registry.bind_graph_plan(complete_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, plan, reused.get()) && reused->size() == 0);
    CHECK(registry.bind_graph_plan(complete_graph, 42, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, plan, reused.get()) && reused->size() == 2);
    CHECK(!registry.bind_graph_plan(complete_graph, 0, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, plan, reused.get()) && reused->size() == 0);
    ggml_cgraph * reordered_graph = candidate_graph(fixture, {
        fused_route.root, fused_route.ids, separate_route.root, separate_route.ids,
        fused_gate_up_node, fused_down_node, separate_gate_node, separate_up_node, separate_down_node,
    });
    CHECK(execution.find(fused_gate_up_node, &binding) && binding.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT);
    CHECK(execution.find(fused_down_node, &binding) && binding.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT);
    CHECK(!registry.bind_graph_plan(reordered_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, plan, reused.get()) && reused->size() == 0);
    CHECK(registry.bind_graph_plan(complete_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, plan, reused.get()));
    ggml_tensor * stale_gate_up_node = candidate_mmid(fixture, fused_gate_up, fused_route.ids);
    ggml_cgraph * stale_graph = candidate_graph(fixture, {
        fused_route.root, fused_route.ids, separate_route.root, separate_route.ids,
        stale_gate_up_node, fused_down_node, separate_up_node, separate_gate_node, separate_down_node,
    });
    CHECK(!registry.bind_graph_plan(stale_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, plan, reused.get()) && reused->size() == 0);
    CHECK(registry.bind_graph_plan(complete_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, plan, reused.get()));

    std::shared_ptr<ggml_cuda_moe_graph_plan> cached_plan;
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(registry.prepare_graph_execution(complete_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &cached_plan, prepared.get()) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(cached_plan != nullptr && prepared->size() == 2);
    }
    const ggml_cuda_moe_graph_plan * first_cached_plan = cached_plan.get();
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(registry.prepare_graph_execution(complete_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, &cached_plan, prepared.get()) == GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(cached_plan.get() == first_cached_plan && prepared->find(separate_down_node, nullptr));
    }
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(registry.prepare_graph_execution(complete_graph, 42, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &cached_plan, prepared.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(cached_plan.get() != first_cached_plan && prepared->find(fused_down_node, nullptr));
    }
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(registry.prepare_graph_execution(complete_graph, 42, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &cached_plan, prepared.get()) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(prepared->size() == 2);
    }
    {
        std::weak_ptr<ggml_cuda_moe_graph_plan> lifetime;
        {
            std::shared_ptr<ggml_cuda_moe_graph_plan> lifetime_plan;
            auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
            CHECK(registry.prepare_graph_execution(complete_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &lifetime_plan, prepared.get()) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
            lifetime = lifetime_plan;
            lifetime_plan.reset();
            CHECK(!lifetime.expired() && prepared->find(fused_down_node, nullptr));
        }
        CHECK(lifetime.expired());
    }
    {
        const ggml_cuda_moe_graph_plan * current_plan = cached_plan.get();
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(registry.prepare_graph_execution(reordered_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, &cached_plan, prepared.get()) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(cached_plan.get() != current_plan && prepared->find(fused_gate_up_node, nullptr));
    }
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(registry.prepare_graph_execution(complete_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, &cached_plan, prepared.get()) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(prepared->size() == 2);
    }

    const candidate_route transition_fused_route = candidate_top_k_route(fixture, 4, 2, 4);
    const candidate_route transition_separate_route = candidate_top_k_route(fixture, 4, 2, 4);
    ggml_tensor * transition_fused_gate_up = candidate_mmid(fixture, fused_gate_up, transition_fused_route.ids);
    ggml_tensor * transition_fused_down = candidate_mmid(fixture, fused_down, transition_fused_route.ids);
    ggml_tensor * transition_separate_gate = candidate_mmid(fixture, separate_gate, transition_separate_route.ids);
    ggml_tensor * transition_separate_up = candidate_mmid(fixture, separate_up, transition_separate_route.ids);
    ggml_tensor * transition_separate_down = candidate_mmid(fixture, separate_down, transition_separate_route.ids);
    ggml_cgraph * transition_graph = candidate_graph(fixture, {
        transition_fused_route.root, transition_fused_route.ids, transition_separate_route.root, transition_separate_route.ids,
        transition_fused_gate_up, transition_fused_down, transition_separate_gate, transition_separate_up, transition_separate_down,
    });
    const auto transition_coverage = candidate_certify_graph(registry, transition_graph);
    std::shared_ptr<ggml_cuda_moe_graph_plan> transition_plan;
    const auto prepare_transition = [&](uint64_t uid, ggml_cuda_moe_graph_property_hint hint, ggml_cuda_moe_graph_execution * prepared) {
        return registry.prepare_graph_execution(
            transition_graph, uid, hint, &transition_plan, prepared,
            transition_coverage.epoch, transition_coverage.nodes,
            transition_coverage.mmid_count, transition_coverage.mmid_fingerprint);
    };
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(prepare_transition(200, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, prepared.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(prepared->size() == 2 && !prepared->find(transition_fused_gate_up, nullptr) && !prepared->find(transition_separate_gate, nullptr));
    }
    const std::shared_ptr<ggml_cuda_moe_graph_plan> warmup_plan = transition_plan;
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(prepare_transition(2001, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, prepared.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(transition_plan.get() == warmup_plan.get() && !prepared->find(transition_fused_gate_up, nullptr));
    }
    candidate_set_route_tokens(transition_fused_route, {transition_fused_gate_up, transition_fused_down}, 1);
    candidate_set_route_tokens(transition_separate_route, {transition_separate_gate, transition_separate_up, transition_separate_down}, 1);
    candidate_stamp_execution(transition_graph, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
        GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 1, 1);
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(prepare_transition(201, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, prepared.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(transition_plan.get() != warmup_plan.get() && prepared->find(transition_fused_gate_up, nullptr) &&
            prepared->find(transition_separate_down, nullptr));
    }
    const ggml_cuda_moe_graph_plan * decode_plan = transition_plan.get();
    for (uint64_t uid = 202; uid < 234; ++uid) {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(prepare_transition(uid, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, prepared.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(transition_plan.get() == decode_plan && prepared->find(transition_fused_down, nullptr));
    }
    candidate_set_route_tokens(transition_fused_route, {transition_fused_gate_up, transition_fused_down}, 4);
    candidate_set_route_tokens(transition_separate_route, {transition_separate_gate, transition_separate_up, transition_separate_down}, 4);
    transition_graph->execution_certificate = {};
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(prepare_transition(234, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, prepared.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(!prepared->find(transition_fused_gate_up, nullptr) && !prepared->find(transition_separate_gate, nullptr));
    }
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(prepare_transition(2341, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, prepared.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(!prepared->find(transition_fused_gate_up, nullptr));
    }
    candidate_set_route_tokens(transition_fused_route, {transition_fused_gate_up, transition_fused_down}, 1);
    candidate_set_route_tokens(transition_separate_route, {transition_separate_gate, transition_separate_up, transition_separate_down}, 1);
    candidate_stamp_execution(transition_graph, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
        GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 1, 1);
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(prepare_transition(235, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, prepared.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(prepared->find(transition_fused_gate_up, nullptr) && prepared->find(transition_separate_down, nullptr));
    }

    ggml_tensor * transition_consumer = ggml_dup(fixture.ctx, transition_fused_gate_up);
    fixture.materialize(transition_consumer);
    transition_consumer->flags |= GGML_TENSOR_FLAG_COMPUTE;
    ggml_cgraph * mutation_graph = candidate_graph(fixture, {
        transition_fused_route.root, transition_fused_route.ids, transition_separate_route.root, transition_separate_route.ids,
        transition_fused_gate_up, transition_consumer, transition_fused_down,
        transition_separate_gate, transition_separate_up, transition_separate_down,
    });
    auto mutation_coverage = candidate_certify_graph(registry, mutation_graph);
    std::shared_ptr<ggml_cuda_moe_graph_plan> mutation_plan;
    const auto prepare_mutation = [&](uint64_t uid, ggml_cuda_moe_graph_property_hint hint, ggml_cuda_moe_graph_execution * prepared) {
        return registry.prepare_graph_execution(
            mutation_graph, uid, hint, &mutation_plan, prepared,
            mutation_coverage.epoch, mutation_coverage.nodes,
            mutation_coverage.mmid_count, mutation_coverage.mmid_fingerprint);
    };
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(prepare_mutation(240, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, prepared.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(prepared->find(transition_fused_gate_up, nullptr) && prepared->find(transition_separate_down, nullptr));
    }
    const ggml_op saved_consumer_op = transition_consumer->op;
    ggml_tensor * saved_consumer_src[3] = {transition_consumer->src[0], transition_consumer->src[1], transition_consumer->src[2]};
    transition_consumer->op = GGML_OP_ADD_ID;
    transition_consumer->src[0] = transition_fused_gate_up;
    transition_consumer->src[1] = fused_bias;
    transition_consumer->src[2] = transition_fused_route.ids;
    candidate_rebuild_graph_uses(mutation_graph);
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(prepare_mutation(241, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, prepared.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(!prepared->find(transition_fused_gate_up, nullptr) && !prepared->find(transition_separate_down, nullptr));
        CHECK(prepared->outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR);
    }
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(prepare_mutation(2411, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, prepared.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(!prepared->find(transition_fused_gate_up, nullptr));
    }
    transition_consumer->op = saved_consumer_op;
    memcpy(transition_consumer->src, saved_consumer_src, sizeof(saved_consumer_src));
    candidate_rebuild_graph_uses(mutation_graph);
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(prepare_mutation(242, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, prepared.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(prepared->find(transition_fused_gate_up, nullptr));
    }
    transition_consumer->op = GGML_OP_MUL_MAT_ID;
    transition_consumer->src[0] = fused_gate_up;
    transition_consumer->src[1] = transition_fused_gate_up->src[1];
    transition_consumer->src[2] = transition_fused_route.ids;
    candidate_rebuild_graph_uses(mutation_graph);
    mutation_coverage = candidate_certify_graph(registry, mutation_graph);
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(prepare_mutation(243, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, prepared.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(!prepared->find(transition_fused_gate_up, nullptr) && !prepared->find(transition_separate_down, nullptr));
        CHECK(prepared->outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR);
    }
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(prepare_mutation(2431, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, prepared.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(!prepared->find(transition_fused_gate_up, nullptr));
    }
    transition_consumer->op = saved_consumer_op;
    memcpy(transition_consumer->src, saved_consumer_src, sizeof(saved_consumer_src));
    candidate_rebuild_graph_uses(mutation_graph);
    mutation_coverage = candidate_certify_graph(registry, mutation_graph);
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(prepare_mutation(244, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, prepared.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(prepared->find(transition_fused_gate_up, nullptr));
    }
    transition_fused_route.ids->view_offs = sizeof(int32_t);
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(prepare_mutation(245, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, prepared.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(!prepared->find(transition_fused_gate_up, nullptr));
    }
    transition_fused_route.ids->view_offs = 0;
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(prepare_mutation(246, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, prepared.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(prepared->find(transition_fused_gate_up, nullptr));
    }
    {
        const ggml_cuda_moe_graph_plan * current_plan = cached_plan.get();
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(registry.prepare_graph_execution(stale_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, &cached_plan, prepared.get()) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(cached_plan.get() != current_plan && prepared->find(stale_gate_up_node, nullptr));
    }
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(registry.prepare_graph_execution(complete_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, &cached_plan, prepared.get()) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(prepared->size() == 2);
    }
    ggml_cuda_moe_grouped_context other_registry(&fixture.owner, 0);
    CHECK(other_registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(!other_registry.bind_graph_plan(complete_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, plan, reused.get()) && reused->size() == 0);
    {
        std::shared_ptr<ggml_cuda_moe_graph_plan> foreign_plan = cached_plan;
        const ggml_cuda_moe_graph_plan * original_owner_plan = foreign_plan.get();
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(other_registry.prepare_graph_execution(complete_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, &foreign_plan, prepared.get()) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(foreign_plan.get() != original_owner_plan && prepared->size() == 2);
    }

    fused_gate_up_node->src[0] = separate_gate;
    CHECK(!registry.bind_graph_plan(complete_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, plan, reused.get()) && reused->size() == 0);
    fused_gate_up_node->src[0] = fused_gate_up;
    CHECK(registry.bind_graph_plan(complete_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, plan, reused.get()));

    CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    CHECK(!registry.bind_graph_plan(complete_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, plan, reused.get()) && reused->size() == 0);
    const ggml_cuda_moe_graph_plan * generation_one_plan = cached_plan.get();
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(registry.prepare_graph_execution(complete_graph, 41, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, &cached_plan, prepared.get()) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(cached_plan.get() != generation_one_plan && cached_plan->registry_generation() == 2 && prepared->size() == 2);
    }
    registry.compile_graph_plan(complete_graph, 43, &plan, &execution);
    CHECK(plan.registry_generation() == 2 && execution.size() == 2);

    ggml_cuda_moe_candidate_group_key held_key;
    ggml_cuda_moe_grouped_acquisition held_acquisition;
    ggml_cuda_moe_grouped_transaction held_transaction;
    CHECK(registry.find_down_group_key(fused_down, &held_key));
    CHECK(registry.acquire_group_resources(held_key, &held_acquisition));
    CHECK(registry.begin_group_transaction(held_acquisition, &held_transaction));
    std::atomic<bool> replacement_started{false};
    std::atomic<bool> replacement_done{false};
    std::thread replacement_thread([&]() {
        replacement_started.store(true, std::memory_order_release);
        CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        replacement_done.store(true, std::memory_order_release);
    });
    while (!replacement_started.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    do {
        std::this_thread::yield();
    } while (registry.bind_graph_plan(complete_graph, 43, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, plan, reused.get()));
    CHECK(reused->size() == 0 && !replacement_done.load(std::memory_order_acquire));
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(registry.prepare_graph_execution(complete_graph, 43, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, &cached_plan, prepared.get()) == GGML_CUDA_MOE_GRAPH_PREPARE_UNAVAILABLE);
        CHECK(cached_plan == nullptr && prepared->size() == 0);
    }
    registry.compile_graph_plan(complete_graph, 43, &plan, &execution);
    CHECK(plan.size() == 0 && execution.size() == 0 && plan.registry_generation() == 0);
    CHECK(registry.end_group_transaction(held_transaction));
    replacement_thread.join();
    CHECK(replacement_done.load(std::memory_order_acquire));
    CHECK(!registry.bind_graph_plan(complete_graph, 43, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, plan, reused.get()) && reused->size() == 0);

    registry.compile_graph_plan(complete_graph, 44, &plan, &execution);
    CHECK(execution.size() == 2);
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(registry.prepare_graph_execution(complete_graph, 44, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &cached_plan, prepared.get()) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(cached_plan != nullptr && prepared->size() == 2);
    }
    auto disabled = candidate_snapshot(12, nullptr, 0);
    CHECK(registry.replace(&disabled) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    registry.compile_graph_plan(complete_graph, 44, &plan, &execution);
    CHECK(plan.size() == 0 && execution.size() == 0 && plan.registry_generation() != 0);
    {
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(registry.prepare_graph_execution(complete_graph, 44, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, &cached_plan, prepared.get()) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(cached_plan != nullptr && prepared->size() == 0);
        const ggml_cuda_moe_graph_plan * disabled_plan = cached_plan.get();
        CHECK(registry.prepare_graph_execution(complete_graph, 44, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, &cached_plan, prepared.get()) == GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(cached_plan.get() == disabled_plan && prepared->size() == 0);
    }
    CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);

    const candidate_fused_top_k_route evaluator_route = candidate_fused_top_k(fixture, 4, 2);
    ggml_tensor * evaluator_gate_up_node = candidate_mmid(fixture, fused_gate_up, evaluator_route.route.ids);
    ggml_tensor * evaluator_down_node = candidate_mmid(fixture, fused_down, evaluator_route.route.ids);
    ggml_cgraph * evaluator_graph = candidate_graph(fixture, {
        evaluator_route.softmax, evaluator_route.reshaped, evaluator_route.route.root, evaluator_route.route.ids,
        evaluator_route.weights, evaluator_gate_up_node, evaluator_down_node,
    });
    registry.compile_graph_plan(evaluator_graph, 45, &plan, &execution);
    CHECK(plan.size() == 1 && execution.size() == 1);

    ggml_tensor * external_gate_up_node = candidate_mmid(fixture, fused_gate_up, external_ids);
    ggml_tensor * external_down_node = candidate_mmid(fixture, fused_down, external_ids);
    ggml_cgraph * external_graph = candidate_graph(fixture, {external_gate_up_node, external_down_node});
    registry.compile_graph_plan(external_graph, 46, &plan, &execution);
    CHECK(plan.size() == 1 && execution.size() == 1 && !execution.find(external_gate_up_node, nullptr));
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_LEGACY);

    ggml_tensor * copied_ids = fixture.tensor(GGML_TYPE_I32, 2, ids_ne);
    copied_ids->op = GGML_OP_CPY;
    copied_ids->src[0] = external_ids;
    copied_ids->flags |= GGML_TENSOR_FLAG_COMPUTE;
    ggml_tensor * copied_gate_up_node = candidate_mmid(fixture, fused_gate_up, copied_ids);
    ggml_tensor * copied_down_node = candidate_mmid(fixture, fused_down, copied_ids);
    ggml_cgraph * copied_graph = candidate_graph(fixture, {copied_ids, copied_gate_up_node, copied_down_node});
    registry.compile_graph_plan(copied_graph, 47, &plan, &execution);
    CHECK(plan.size() == 1 && execution.size() == 1 && !execution.find(copied_gate_up_node, nullptr));

    const candidate_route offset_route = candidate_top_k_route(fixture, 4, 2, 1, sizeof(int32_t));
    ggml_tensor * offset_gate_up_node = candidate_mmid(fixture, fused_gate_up, offset_route.ids);
    ggml_tensor * offset_down_node = candidate_mmid(fixture, fused_down, offset_route.ids);
    ggml_cgraph * offset_graph = candidate_graph(fixture, {offset_route.root, offset_route.ids, offset_gate_up_node, offset_down_node});
    registry.compile_graph_plan(offset_graph, 48, &plan, &execution);
    CHECK(plan.size() == 1 && execution.size() == 1 && !execution.find(offset_gate_up_node, nullptr));

    ggml_tensor * transformed_ids = ggml_reshape_2d(fixture.ctx, fused_route_other.ids, 2, 1);
    fixture.materialize(transformed_ids);
    transformed_ids->flags |= GGML_TENSOR_FLAG_COMPUTE;
    ggml_tensor * transformed_gate_up_node = candidate_mmid(fixture, fused_gate_up, transformed_ids);
    ggml_tensor * transformed_down_node = candidate_mmid(fixture, fused_down, transformed_ids);
    ggml_cgraph * transformed_graph = candidate_graph(fixture, {
        fused_route_other.root, fused_route_other.ids, transformed_ids, transformed_gate_up_node, transformed_down_node,
    });
    registry.compile_graph_plan(transformed_graph, 49, &plan, &execution);
    CHECK(plan.size() == 1 && execution.size() == 1 && !execution.find(transformed_gate_up_node, nullptr));

    const candidate_route wrong_axis_route = candidate_top_k_route(fixture, 5, 2);
    ggml_tensor * wrong_axis_gate_up_node = candidate_mmid(fixture, fused_gate_up, wrong_axis_route.ids);
    ggml_tensor * wrong_axis_down_node = candidate_mmid(fixture, fused_down, wrong_axis_route.ids);
    ggml_cgraph * wrong_axis_graph = candidate_graph(fixture, {
        wrong_axis_route.root, wrong_axis_route.ids, wrong_axis_gate_up_node, wrong_axis_down_node,
    });
    registry.compile_graph_plan(wrong_axis_graph, 50, &plan, &execution);
    CHECK(plan.size() == 1 && execution.size() == 1 && !execution.find(wrong_axis_gate_up_node, nullptr));

    ggml_tensor * order_gate_up_node = candidate_mmid(fixture, fused_gate_up, fused_route_other.ids);
    ggml_tensor * order_down_node = candidate_mmid(fixture, fused_down, fused_route_other.ids);
    ggml_cgraph * producer_order_graph = candidate_graph(fixture, {
        fused_route_other.ids, fused_route_other.root, order_gate_up_node, order_down_node,
    });
    registry.compile_graph_plan(producer_order_graph, 51, &plan, &execution);
    CHECK(plan.size() == 1 && execution.size() == 1 && !execution.find(order_gate_up_node, nullptr));

    const int64_t padding_ne[] = {1};
    ggml_tensor * padding_input = fixture.tensor(GGML_TYPE_F32, 1, padding_ne);
    ggml_tensor * padding_node = ggml_dup(fixture.ctx, padding_input);
    fixture.materialize(padding_node);
    padding_node->flags |= GGML_TENSOR_FLAG_COMPUTE;
    const candidate_route padded_route = candidate_top_k_route(fixture, 4, 2);
    ggml_tensor * padded_gate_up_node = candidate_mmid(fixture, fused_gate_up, padded_route.ids);
    ggml_tensor * padded_down_node = candidate_mmid(fixture, fused_down, padded_route.ids);
    constexpr uint32_t n_padding_nodes = 32768;
    ggml_cgraph * padded_graph = candidate_padded_graph(fixture, padding_node, n_padding_nodes, {
        padded_route.root, padded_route.ids, separate_route.root, separate_route.ids,
        padded_gate_up_node, padded_down_node, separate_up_node, separate_gate_node, separate_down_node,
    });
    const auto padded_coverage = candidate_certify_graph(registry, padded_graph);
    ggml_cuda_moe_graph_plan padded_plan;
    ggml_cuda_moe_graph_execution padded_execution;
    registry.compile_graph_plan(
        padded_graph, 100, &padded_plan, &padded_execution,
        padded_coverage.epoch, padded_coverage.nodes, padded_coverage.mmid_count, padded_coverage.mmid_fingerprint);
    CHECK(padded_plan.size() == 2 && padded_execution.size() == 2);
    CHECK(padded_plan.graph_node_count() == static_cast<int32_t>(n_padding_nodes + 9));
    CHECK(registry.bind_graph_plan(
        padded_graph, 101, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, padded_plan, &padded_execution,
        padded_coverage.epoch, padded_coverage.nodes, padded_coverage.mmid_count, padded_coverage.mmid_fingerprint) &&
        padded_execution.size() == 2);

    const auto complete_coverage = candidate_certify_graph(registry, complete_graph);
    registry.compile_graph_plan(
        complete_graph, 52, &plan, &execution,
        complete_coverage.epoch, complete_coverage.nodes,
        complete_coverage.mmid_count, complete_coverage.mmid_fingerprint);
    CHECK(plan.size() == 2 && execution.size() == 2);
    const auto bind_complete_unknown = [&]() {
        return registry.bind_graph_plan(
            complete_graph, 52, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, plan, reused.get(),
            complete_coverage.epoch, complete_coverage.nodes,
            complete_coverage.mmid_count, complete_coverage.mmid_fingerprint);
    };
    void * saved_ids_data = fused_route.ids->data;
    fused_route.ids->data = fused_route_other.ids->data;
    CHECK(!bind_complete_unknown() && reused->size() == 0);
    fused_route.ids->data = saved_ids_data;
    CHECK(bind_complete_unknown());
    ggml_backend_buffer_t saved_ids_buffer = fused_route.ids->buffer;
    fused_route.ids->buffer = nullptr;
    CHECK(!bind_complete_unknown() && reused->size() == 0);
    fused_route.ids->buffer = saved_ids_buffer;
    CHECK(bind_complete_unknown());
    const size_t saved_ids_stride = fused_route.ids->nb[0];
    fused_route.ids->nb[0] += sizeof(int32_t);
    CHECK(!bind_complete_unknown() && reused->size() == 0);
    fused_route.ids->nb[0] = saved_ids_stride;
    CHECK(bind_complete_unknown());
    const int64_t saved_activation_tokens = fused_gate_up_node->src[1]->ne[2];
    fused_gate_up_node->src[1]->ne[2] = 2;
    CHECK(!bind_complete_unknown() && reused->size() == 0);
    fused_gate_up_node->src[1]->ne[2] = saved_activation_tokens;
    CHECK(bind_complete_unknown());
    const int64_t saved_output_tokens = fused_gate_up_node->ne[2];
    fused_gate_up_node->ne[2] = 2;
    CHECK(!bind_complete_unknown() && reused->size() == 0);
    fused_gate_up_node->ne[2] = saved_output_tokens;
    CHECK(bind_complete_unknown());
    ggml_tensor * saved_weight = fused_gate_up_node->src[0];
    fused_gate_up_node->src[0] = separate_gate;
    CHECK(!bind_complete_unknown() && reused->size() == 0);
    fused_gate_up_node->src[0] = saved_weight;
    CHECK(bind_complete_unknown());
    ggml_tensor * saved_view_src = fused_route.ids->view_src;
    fused_route.ids->view_src = fused_route_other.root;
    CHECK(!bind_complete_unknown() && reused->size() == 0);
    fused_route.ids->view_src = saved_view_src;
    CHECK(bind_complete_unknown());
    ggml_tensor * saved_view_source = fused_route.ids->src[0];
    fused_route.ids->src[0] = fused_route_other.root;
    CHECK(!bind_complete_unknown() && reused->size() == 0);
    fused_route.ids->src[0] = saved_view_source;
    CHECK(bind_complete_unknown());
    fused_route.ids->view_offs = sizeof(int32_t);
    CHECK(!bind_complete_unknown() && reused->size() == 0);
    fused_route.ids->view_offs = 0;
    CHECK(bind_complete_unknown());
    const size_t saved_root_stride = fused_route.root->nb[1];
    fused_route.root->nb[1] += sizeof(int32_t);
    CHECK(!bind_complete_unknown() && reused->size() == 0);
    fused_route.root->nb[1] = saved_root_stride;
    CHECK(bind_complete_unknown());
    ggml_tensor * saved_root_source = fused_route.root->src[0];
    fused_route.root->src[0] = fused_route_other.source;
    CHECK(!bind_complete_unknown() && reused->size() == 0);
    fused_route.root->src[0] = saved_root_source;
    CHECK(bind_complete_unknown());
    const int32_t sort_asc = GGML_SORT_ORDER_ASC;
    memcpy(fused_route.root->op_params, &sort_asc, sizeof(sort_asc));
    CHECK(!bind_complete_unknown() && reused->size() == 0);
    const int32_t sort_desc = GGML_SORT_ORDER_DESC;
    memcpy(fused_route.root->op_params, &sort_desc, sizeof(sort_desc));
    CHECK(bind_complete_unknown());
    ggml_cgraph * producer_reordered_graph = candidate_graph(fixture, {
        fused_route.root, fused_route.ids, separate_route.ids, separate_route.root,
        fused_gate_up_node, fused_down_node, separate_up_node, separate_gate_node, separate_down_node,
    });
    CHECK(!registry.bind_graph_plan(producer_reordered_graph, 52, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, plan, reused.get()) && reused->size() == 0);
    CHECK(bind_complete_unknown());

    ggml_tensor * mixed_gate_up_node = candidate_mmid(fixture, fused_gate_up, fused_route.ids);
    ggml_tensor * mixed_down_node = candidate_mmid(fixture, fused_down, fused_route_other.ids);
    ggml_cgraph * mixed_graph = candidate_graph(fixture, {
        fused_route.root, fused_route.ids, fused_route_other.root, fused_route_other.ids, mixed_gate_up_node, mixed_down_node,
    });
    registry.compile_graph_plan(mixed_graph, 53, &plan, &execution);
    CHECK(plan.size() == 1 && execution.size() == 1 && !execution.find(mixed_gate_up_node, nullptr));

    ggml_tensor * missing_gate_up_node = candidate_mmid(fixture, fused_gate_up, fused_route.ids);
    ggml_cgraph * missing_graph = candidate_graph(fixture, {fused_route.root, fused_route.ids, missing_gate_up_node});
    registry.compile_graph_plan(missing_graph, 54, &plan, &execution);
    CHECK(plan.size() == 1 && execution.size() == 1 && !execution.find(missing_gate_up_node, nullptr));
    {
        std::shared_ptr<ggml_cuda_moe_graph_plan> missing_plan;
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(registry.prepare_graph_execution(missing_graph, 540, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &missing_plan, prepared.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        const std::shared_ptr<ggml_cuda_moe_graph_plan> first_missing_plan = missing_plan;
        CHECK(registry.prepare_graph_execution(missing_graph, 541, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &missing_plan, prepared.get()) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(missing_plan.get() != first_missing_plan.get() && !prepared->find(missing_gate_up_node, nullptr));
    }
    ggml_cgraph * complete_missing_graph = candidate_graph(fixture, {
        fused_route.root, fused_route.ids, separate_route.root, separate_route.ids,
        missing_gate_up_node, separate_up_node, separate_gate_node, separate_down_node,
    });
    const auto complete_missing_coverage = candidate_certify_graph(registry, complete_missing_graph);
    {
        std::shared_ptr<ggml_cuda_moe_graph_plan> missing_plan;
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(registry.prepare_graph_execution(
            complete_missing_graph, 542, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &missing_plan, prepared.get(),
            complete_missing_coverage.epoch, complete_missing_coverage.nodes,
            complete_missing_coverage.mmid_count, complete_missing_coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        const ggml_cuda_moe_graph_plan * stable_missing_plan = missing_plan.get();
        CHECK(!prepared->find(missing_gate_up_node, nullptr) && !prepared->find(separate_down_node, nullptr));
        CHECK(prepared->outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR);
        CHECK(registry.prepare_graph_execution(
            complete_missing_graph, 543, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &missing_plan, prepared.get(),
            complete_missing_coverage.epoch, complete_missing_coverage.nodes,
            complete_missing_coverage.mmid_count, complete_missing_coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(missing_plan.get() == stable_missing_plan && !prepared->find(missing_gate_up_node, nullptr));
    }

    ggml_tensor * duplicate_gate_up_node = candidate_mmid(fixture, fused_gate_up, fused_route.ids);
    ggml_tensor * duplicate_gate_up_peer = candidate_mmid(fixture, fused_gate_up, fused_route.ids);
    ggml_tensor * duplicate_down_node = candidate_mmid(fixture, fused_down, fused_route.ids);
    ggml_cgraph * duplicate_graph = candidate_graph(fixture, {
        fused_route.root, fused_route.ids, duplicate_gate_up_node, duplicate_gate_up_peer, duplicate_down_node,
    });
    registry.compile_graph_plan(duplicate_graph, 55, &plan, &execution);
    CHECK(plan.size() == 1 && execution.size() == 1 && !execution.find(duplicate_gate_up_node, nullptr));

    for (int64_t n_sequences : {2, 4}) {
        const candidate_route multi_sequence_route = candidate_top_k_route(fixture, 4, 2, n_sequences);
        ggml_tensor * multi_sequence_gate_up = candidate_mmid(fixture, fused_gate_up, multi_sequence_route.ids);
        ggml_tensor * multi_sequence_down = candidate_mmid(fixture, fused_down, multi_sequence_route.ids);
        ggml_cgraph * multi_sequence_graph = candidate_graph(fixture, {
            multi_sequence_route.root, multi_sequence_route.ids, multi_sequence_gate_up, multi_sequence_down,
        });
        registry.compile_graph_plan(multi_sequence_graph, 54 + n_sequences, &plan, &execution);
        CHECK(multi_sequence_route.ids->ne[0] == 2 && multi_sequence_route.ids->ne[1] == n_sequences);
        CHECK(plan.size() == 1 && execution.size() == 1 && !execution.find(multi_sequence_gate_up, nullptr));
        CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_PREFILL_LEGACY);
        CHECK(!registry.begin_graph_dispatch(&execution, GGML_CUDA_MOE_GRAPH_DISPATCH_CAPTURE));
        CHECK(registry.begin_graph_dispatch(&execution, GGML_CUDA_MOE_GRAPH_DISPATCH_DIRECT));
        CHECK(execution.dispatch_mode() == GGML_CUDA_MOE_GRAPH_DISPATCH_DIRECT);
        CHECK(registry.finish_graph_dispatch(&execution));
    }

    ggml_tensor * wrong_source_node = candidate_mmid(fixture, separate_gate, fused_route.ids);
    ggml_tensor * correct_down_node = candidate_mmid(fixture, fused_down, fused_route.ids);
    ggml_cgraph * wrong_source_graph = candidate_graph(fixture, {
        fused_route.root, fused_route.ids, wrong_source_node, correct_down_node,
    });
    registry.compile_graph_plan(wrong_source_graph, 57, &plan, &execution);
    CHECK(plan.size() == 2 && execution.size() == 2 && !execution.find(wrong_source_node, nullptr));

    ggml_tensor * auxiliary_gate_up_node = candidate_mmid(fixture, fused_gate_up, fused_route.ids);
    ggml_tensor * auxiliary_down_node = candidate_mmid(fixture, fused_down, fused_route.ids);
    const int64_t auxiliary_ne[] = {auxiliary_gate_up_node->ne[0], auxiliary_gate_up_node->ne[1], auxiliary_gate_up_node->ne[2]};
    ggml_tensor * auxiliary_out = fixture.tensor(GGML_TYPE_F32, 3, auxiliary_ne);
    auxiliary_out->op = GGML_OP_ADD_ID;
    auxiliary_out->src[0] = auxiliary_gate_up_node;
    auxiliary_out->src[1] = fused_bias;
    auxiliary_out->src[2] = fused_route.ids;
    auxiliary_out->flags |= GGML_TENSOR_FLAG_COMPUTE;
    ggml_cgraph * auxiliary_graph = candidate_graph(fixture, {
        fused_route.root, fused_route.ids, auxiliary_gate_up_node, auxiliary_out, auxiliary_down_node,
    });
    registry.compile_graph_plan(auxiliary_graph, 58, &plan, &execution);
    CHECK(plan.size() == 1 && execution.size() == 1 && !execution.find(auxiliary_gate_up_node, nullptr));
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR);
    CHECK(!registry.begin_graph_dispatch(&execution, GGML_CUDA_MOE_GRAPH_DISPATCH_DIRECT));

    std::array<ggml_backend_moe_candidate_bank_v1, 3> auxiliary_banks = {{
        {fused_gate_up, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, 0},
        {fused_down, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
        {fused_bias, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_BIAS, 0},
    }};
    ggml_backend_moe_candidate_group_v1 auxiliary_group = {
        auxiliary_banks.data(), auxiliary_banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 0, 0,
    };
    auto auxiliary_snapshot = candidate_snapshot(12, &auxiliary_group, 1);
    CHECK(registry.replace(&auxiliary_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    ggml_cgraph * auxiliary_registered_graph = candidate_graph(fixture, {
        fused_route.root, fused_route.ids, auxiliary_gate_up_node, auxiliary_down_node,
    });
    registry.compile_graph_plan(auxiliary_registered_graph, 59, &plan, &execution);
    CHECK(plan.size() == 1 && execution.size() == 1 && !execution.find(auxiliary_gate_up_node, nullptr));
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR);
    CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_auxiliary_reason(plan, 0));

    registry.compile_graph_plan(auxiliary_graph, 60, &plan, &execution);
    CHECK(plan.size() == 1 && execution.size() == 1);
    CHECK(execution.find(auxiliary_gate_up_node, nullptr) && execution.find(auxiliary_down_node, nullptr));
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED);
    auxiliary_out->src[2] = fused_route_other.ids;
    candidate_rebuild_graph_uses(auxiliary_graph);
    registry.compile_graph_plan(auxiliary_graph, 61, &plan, &execution);
    CHECK(plan.size() == 1 && execution.size() == 1 && !execution.find(auxiliary_gate_up_node, nullptr));
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR);
    CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_auxiliary_reason(plan, 0));
    auxiliary_out->src[2] = fused_route.ids;
    candidate_rebuild_graph_uses(auxiliary_graph);
    registry.compile_graph_plan(auxiliary_graph, 62, &plan, &execution);
    CHECK(execution.find(auxiliary_gate_up_node, nullptr));

    const int64_t separate_bias_ne[] = {256, 4};
    const int64_t separate_scale_ne[] = {4};
    ggml_tensor * separate_gate_bias = fixture.tensor(GGML_TYPE_F32, 2, separate_bias_ne);
    ggml_tensor * separate_up_bias = fixture.tensor(GGML_TYPE_F32, 2, separate_bias_ne);
    ggml_tensor * separate_down_bias = fixture.tensor(GGML_TYPE_F32, 2, separate_bias_ne);
    ggml_tensor * separate_down_scale = fixture.tensor(GGML_TYPE_F32, 1, separate_scale_ne);
    std::array<ggml_backend_moe_candidate_bank_v1, 6> separate_bias_banks = {{
        {separate_gate, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT, 0},
        {separate_up, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT, 0},
        {separate_down, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
        {separate_gate_bias, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_BIAS, 0},
        {separate_up_bias, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_BIAS, 0},
        {separate_down_bias, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_BIAS, 0},
    }};
    const ggml_backend_moe_candidate_group_v1 separate_bias_group = {
        separate_bias_banks.data(), separate_bias_banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 0, 0,
    };
    const auto separate_bias_snapshot = candidate_snapshot(12, &separate_bias_group, 1);
    CHECK(registry.replace(&separate_bias_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);

    ggml_tensor * biased_gate_reader = candidate_mmid(fixture, separate_gate, separate_route.ids);
    ggml_tensor * biased_up_reader = candidate_mmid(fixture, separate_up, separate_route.ids);
    ggml_tensor * biased_down_reader = candidate_mmid(fixture, separate_down, separate_route.ids);
    ggml_tensor * biased_gate_out = ggml_add_id(fixture.ctx, biased_gate_reader, separate_gate_bias, separate_route.ids);
    ggml_tensor * biased_up_out = ggml_add_id(fixture.ctx, biased_up_reader, separate_up_bias, separate_route.ids);
    ggml_tensor * biased_down_out = ggml_add_id(fixture.ctx, biased_down_reader, separate_down_bias, separate_route.ids);
    ggml_tensor * biased_extra_consumer = ggml_dup(fixture.ctx, biased_gate_out);
    for (ggml_tensor * node : {biased_gate_out, biased_up_out, biased_down_out, biased_extra_consumer}) {
        fixture.materialize(node);
        node->flags |= GGML_TENSOR_FLAG_COMPUTE;
    }
    ggml_cgraph * separate_bias_graph = candidate_graph(fixture, {
        separate_route.root, separate_route.ids,
        biased_gate_reader, biased_gate_out, biased_extra_consumer,
        biased_up_reader, biased_up_out, biased_down_reader, biased_down_out,
    });
    const auto separate_bias_coverage = candidate_certify_graph(registry, separate_bias_graph);
    registry.compile_graph_plan(
        separate_bias_graph, 63, &plan, &execution,
        separate_bias_coverage.epoch, separate_bias_coverage.nodes,
        separate_bias_coverage.mmid_count, separate_bias_coverage.mmid_fingerprint);
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED);
    CHECK(execution.find(biased_gate_reader, nullptr) && execution.find(biased_up_reader, nullptr) &&
        execution.find(biased_down_reader, nullptr));
    ggml_cuda_moe_graph_execution separate_bias_reused;
    const auto bind_separate_bias = [&](uint64_t graph_uid) {
        return registry.bind_graph_plan(
            separate_bias_graph, graph_uid, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, plan, &separate_bias_reused,
            separate_bias_coverage.epoch, separate_bias_coverage.nodes,
            separate_bias_coverage.mmid_count, separate_bias_coverage.mmid_fingerprint);
    };
    CHECK(bind_separate_bias(64));

    biased_gate_out->src[1] = separate_up_bias;
    candidate_rebuild_graph_uses(separate_bias_graph);
    CHECK(!bind_separate_bias(65));
    biased_gate_out->src[1] = separate_gate_bias;
    candidate_rebuild_graph_uses(separate_bias_graph);
    CHECK(bind_separate_bias(66));

    biased_gate_out->src[2] = fused_route_other.ids;
    candidate_rebuild_graph_uses(separate_bias_graph);
    CHECK(!bind_separate_bias(67));
    biased_gate_out->src[2] = separate_route.ids;
    candidate_rebuild_graph_uses(separate_bias_graph);

    biased_gate_out->src[0] = biased_up_reader;
    candidate_rebuild_graph_uses(separate_bias_graph);
    CHECK(!bind_separate_bias(671));
    biased_gate_out->src[0] = biased_gate_reader;
    candidate_rebuild_graph_uses(separate_bias_graph);
    biased_gate_out->src[3] = separate_gate_bias;
    candidate_rebuild_graph_uses(separate_bias_graph);
    CHECK(!bind_separate_bias(672));
    biased_gate_out->src[3] = nullptr;
    candidate_rebuild_graph_uses(separate_bias_graph);
    const ggml_type saved_bias_output_type = biased_gate_out->type;
    biased_gate_out->type = GGML_TYPE_BF16;
    CHECK(!bind_separate_bias(673));
    biased_gate_out->type = saved_bias_output_type;
    const int64_t saved_bias_output_ne = biased_gate_out->ne[0];
    biased_gate_out->ne[0]--;
    CHECK(!bind_separate_bias(674));
    biased_gate_out->ne[0] = saved_bias_output_ne;
    const size_t saved_bias_output_nb = biased_gate_out->nb[1];
    biased_gate_out->nb[1] += sizeof(float);
    CHECK(!bind_separate_bias(675));
    biased_gate_out->nb[1] = saved_bias_output_nb;

    const ggml_type saved_bias_type = separate_gate_bias->type;
    separate_gate_bias->type = GGML_TYPE_BF16;
    CHECK(!bind_separate_bias(68));
    separate_gate_bias->type = saved_bias_type;
    const int64_t saved_bias_ne = separate_gate_bias->ne[0];
    separate_gate_bias->ne[0]--;
    CHECK(!bind_separate_bias(69));
    separate_gate_bias->ne[0] = saved_bias_ne;
    const size_t saved_bias_nb = separate_gate_bias->nb[1];
    separate_gate_bias->nb[1] += sizeof(float);
    CHECK(!bind_separate_bias(70));
    separate_gate_bias->nb[1] = saved_bias_nb;
    void * saved_bias_data = separate_gate_bias->data;
    separate_gate_bias->data = static_cast<char *>(separate_gate_bias->data) + sizeof(float);
    CHECK(!bind_separate_bias(71));
    separate_gate_bias->data = saved_bias_data;

    std::swap(separate_bias_graph->nodes[2], separate_bias_graph->nodes[3]);
    CHECK(!bind_separate_bias(72));
    std::swap(separate_bias_graph->nodes[2], separate_bias_graph->nodes[3]);
    biased_extra_consumer->src[0] = biased_gate_reader;
    candidate_rebuild_graph_uses(separate_bias_graph);
    CHECK(!bind_separate_bias(73));
    ggml_cuda_moe_graph_plan extra_consumer_plan;
    registry.compile_graph_plan(separate_bias_graph, 731, &extra_consumer_plan, &execution);
    CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_auxiliary_reason(extra_consumer_plan, 0));
    biased_extra_consumer->src[0] = biased_gate_out;
    candidate_rebuild_graph_uses(separate_bias_graph);
    CHECK(bind_separate_bias(74));

    separate_bias_banks[3].role = GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_BIAS;
    separate_bias_banks[4].role = GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_BIAS;
    const auto wrong_bias_role_snapshot = candidate_snapshot(12, &separate_bias_group, 1);
    CHECK(registry.replace(&wrong_bias_role_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    registry.compile_graph_plan(separate_bias_graph, 75, &plan, &execution);
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR && !execution.find(biased_gate_reader, nullptr));
    CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_auxiliary_reason(plan, 0));
    separate_bias_banks[3].role = GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_BIAS;
    separate_bias_banks[4].role = GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_BIAS;
    CHECK(registry.replace(&separate_bias_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);

    std::array<ggml_backend_moe_candidate_bank_v1, 7> scale_bias_banks;
    std::copy(separate_bias_banks.begin(), separate_bias_banks.end(), scale_bias_banks.begin());
    scale_bias_banks.back() = {separate_down_scale, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_SCALE, 0};
    const ggml_backend_moe_candidate_group_v1 scale_bias_group = {
        scale_bias_banks.data(), scale_bias_banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE, 0, 0,
    };
    const auto scale_bias_snapshot = candidate_snapshot(12, &scale_bias_group, 1);
    CHECK(registry.replace(&scale_bias_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    registry.compile_graph_plan(separate_bias_graph, 76, &plan, &execution);
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR && !execution.find(biased_gate_reader, nullptr));
    CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_descriptor_reason(plan, 0));

    CHECK(registry.replace(&separate_bias_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    const candidate_route biased_prefill_route = candidate_top_k_route(fixture, 4, 2, 2);
    ggml_tensor * biased_prefill_gate = candidate_mmid(fixture, separate_gate, biased_prefill_route.ids);
    ggml_tensor * biased_prefill_up = candidate_mmid(fixture, separate_up, biased_prefill_route.ids);
    ggml_tensor * biased_prefill_down = candidate_mmid(fixture, separate_down, biased_prefill_route.ids);
    ggml_tensor * biased_prefill_gate_out = ggml_add_id(
        fixture.ctx, biased_prefill_gate, separate_gate_bias, biased_prefill_route.ids);
    ggml_tensor * biased_prefill_up_out = ggml_add_id(
        fixture.ctx, biased_prefill_up, separate_up_bias, biased_prefill_route.ids);
    ggml_tensor * biased_prefill_down_out = ggml_add_id(
        fixture.ctx, biased_prefill_down, separate_down_bias, biased_prefill_route.ids);
    for (ggml_tensor * node : {biased_prefill_gate_out, biased_prefill_up_out, biased_prefill_down_out}) {
        fixture.materialize(node);
        node->flags |= GGML_TENSOR_FLAG_COMPUTE;
    }
    ggml_cgraph * biased_prefill_graph = candidate_graph(fixture, {
        biased_prefill_route.root, biased_prefill_route.ids,
        biased_prefill_gate, biased_prefill_gate_out,
        biased_prefill_up, biased_prefill_up_out,
        biased_prefill_down, biased_prefill_down_out,
    });
    registry.compile_graph_plan(biased_prefill_graph, 77, &plan, &execution);
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_PREFILL_LEGACY && execution.size() == 1);
    CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_prefill_reason(plan, 0));

    const auto check_biased_execution_unknown_reuse = [&](ggml_cgraph * graph, uint64_t graph_uid) {
        const auto coverage = candidate_certify_graph(registry, graph);
        std::shared_ptr<ggml_cuda_moe_graph_plan> reusable_plan;
        ggml_cuda_moe_graph_execution reusable_execution;
        CHECK(registry.prepare_graph_execution(
            graph, graph_uid, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &reusable_plan, &reusable_execution,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        CHECK(reusable_execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_LEGACY && reusable_execution.size() == 1);
        CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_execution_reason(*reusable_plan, 0));
        const auto * compiled_plan = reusable_plan.get();
        CHECK(registry.prepare_graph_execution(
            graph, graph_uid + 1, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &reusable_plan, &reusable_execution,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
        CHECK(reusable_plan.get() == compiled_plan &&
            reusable_execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_LEGACY);
    };
    for (uint32_t domain : {GGML_GRAPH_EXECUTION_DOMAIN_DRAFT, GGML_GRAPH_EXECUTION_DOMAIN_MTP}) {
        candidate_stamp_execution(separate_bias_graph, domain,
            GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 1, 1);
        check_biased_execution_unknown_reuse(separate_bias_graph, 780 + domain * 2);
    }

    const candidate_route biased_overslot_route = candidate_top_k_route(fixture, 4, 4, 4);
    ggml_tensor * biased_overslot_gate = candidate_mmid(fixture, separate_gate, biased_overslot_route.ids);
    ggml_tensor * biased_overslot_up = candidate_mmid(fixture, separate_up, biased_overslot_route.ids);
    ggml_tensor * biased_overslot_down = candidate_mmid(fixture, separate_down, biased_overslot_route.ids);
    ggml_tensor * biased_overslot_gate_out = ggml_add_id(
        fixture.ctx, biased_overslot_gate, separate_gate_bias, biased_overslot_route.ids);
    ggml_tensor * biased_overslot_up_out = ggml_add_id(
        fixture.ctx, biased_overslot_up, separate_up_bias, biased_overslot_route.ids);
    ggml_tensor * biased_overslot_down_out = ggml_add_id(
        fixture.ctx, biased_overslot_down, separate_down_bias, biased_overslot_route.ids);
    for (ggml_tensor * node : {biased_overslot_gate_out, biased_overslot_up_out, biased_overslot_down_out}) {
        fixture.materialize(node);
        node->flags |= GGML_TENSOR_FLAG_COMPUTE;
    }
    ggml_cgraph * biased_overslot_graph = candidate_graph(fixture, {
        biased_overslot_route.root, biased_overslot_route.ids,
        biased_overslot_gate, biased_overslot_gate_out,
        biased_overslot_up, biased_overslot_up_out,
        biased_overslot_down, biased_overslot_down_out,
    });
    candidate_stamp_execution(biased_overslot_graph, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
        GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 4, 4);
    check_biased_execution_unknown_reuse(biased_overslot_graph, 790);

    {
        candidate_test_fixture oversized_fixture;
        constexpr int64_t n_experts = 1025;
        const int64_t oversized_gate_up_ne[] = {32, 64, n_experts};
        const int64_t oversized_down_ne[] = {32, 32, n_experts};
        const int64_t oversized_scale_ne[] = {n_experts};
        ggml_tensor * oversized_gate_up = oversized_fixture.tensor(GGML_TYPE_Q4_0, 3, oversized_gate_up_ne);
        ggml_tensor * oversized_down = oversized_fixture.tensor(GGML_TYPE_Q4_0, 3, oversized_down_ne);
        ggml_tensor * oversized_scale = oversized_fixture.tensor(GGML_TYPE_F32, 1, oversized_scale_ne);
        const candidate_route oversized_route = candidate_top_k_route(oversized_fixture, n_experts, 2);
        ggml_tensor * oversized_gate_up_node = candidate_mmid(oversized_fixture, oversized_gate_up, oversized_route.ids);
        ggml_tensor * oversized_down_node = candidate_mmid(oversized_fixture, oversized_down, oversized_route.ids);
        ggml_tensor * scale = ggml_reshape_3d(oversized_fixture.ctx, oversized_scale, 1, n_experts, 1);
        oversized_fixture.materialize(scale);
        scale->flags |= GGML_TENSOR_FLAG_COMPUTE;
        scale = ggml_repeat_4d(oversized_fixture.ctx, scale, 1, n_experts, 1, 1);
        oversized_fixture.materialize(scale);
        scale->flags |= GGML_TENSOR_FLAG_COMPUTE;
        scale = ggml_get_rows(oversized_fixture.ctx, scale, oversized_route.ids);
        oversized_fixture.materialize(scale);
        scale->flags |= GGML_TENSOR_FLAG_COMPUTE;
        ggml_tensor * oversized_output = ggml_mul(oversized_fixture.ctx, oversized_down_node, scale);
        oversized_fixture.materialize(oversized_output);
        oversized_output->flags |= GGML_TENSOR_FLAG_COMPUTE;
        ggml_cgraph * oversized_graph = candidate_graph(oversized_fixture, {
            oversized_route.root, oversized_route.ids, oversized_gate_up_node, oversized_down_node,
            scale->src[0]->src[0], scale->src[0], scale, oversized_output,
        });
        std::array<ggml_backend_moe_candidate_bank_v1, 3> oversized_banks = {{
            {oversized_gate_up, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, 0},
            {oversized_down, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
            {oversized_scale, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_SCALE, 0},
        }};
        const ggml_backend_moe_candidate_group_v1 oversized_group = {
            oversized_banks.data(), oversized_banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 0, 0,
        };
        const auto oversized_snapshot = candidate_snapshot(12, &oversized_group, 1);
        CHECK(registry.replace(&oversized_snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        CHECK(registry.state().n_groups == 1 && registry.state().permanent_candidate_bytes == 4100);
        registry.compile_graph_plan(oversized_graph, 591, &plan, &execution);
        CHECK(plan.size() == 1 && execution.size() == 1 && !execution.find(oversized_gate_up_node, nullptr));
        fprintf(stderr, "test-moe-cache: oversized original-direct auxiliary decline OK\n");
    }

    CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    registry.compile_graph_plan(complete_graph, 60, &plan, &execution);
    CHECK(execution.resolve_streams(candidate_test_graph_stream, reinterpret_cast<void *>(uintptr_t{1})));
    CHECK(registry.begin_graph_dispatch(&execution, false));
    const auto * legacy_authority = execution.find_authority(fused_gate_up_node);
    CHECK(legacy_authority != nullptr && legacy_authority->authority() == GGML_CUDA_MOE_GROUP_AUTHORITY_LEGACY);
    auto legacy_before = registry.acquire_legacy_cache(fused_gate_up, nullptr, legacy_authority);
    CHECK(legacy_before);
    const auto legacy_before_state = legacy_before.acquisition();
    legacy_before = {};
    CHECK(registry.finish_graph_dispatch(&execution));

    CHECK(registry.begin_graph_dispatch(&execution, true));
    const auto * grouped_authority = execution.find_authority(fused_gate_up_node);
    CHECK(grouped_authority != nullptr && grouped_authority->authority() == GGML_CUDA_MOE_GROUP_AUTHORITY_GROUPED);
    CHECK(!registry.acquire_legacy_cache(fused_gate_up, nullptr, grouped_authority));
    ggml_cuda_moe_candidate_group_key authority_key;
    ggml_cuda_moe_grouped_acquisition authority_resource;
    CHECK(registry.find_down_group_key(fused_down, &authority_key));
    CHECK(registry.acquire_group_resources(authority_key, &authority_resource));
    CHECK(!registry.finish_graph_dispatch(&execution));

    registry.compile_graph_plan(missing_graph, 61, &plan, &execution);
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR);
    CHECK(!registry.begin_graph_dispatch(&execution, true));
    CHECK(registry.get_group_resources(authority_resource, nullptr));
    CHECK(registry.begin_graph_dispatch(&execution, false));
    CHECK(!registry.get_group_resources(authority_resource, nullptr));
    auto legacy_after = registry.acquire_legacy_cache(fused_gate_up);
    CHECK(legacy_after && legacy_after.acquisition().group_authority_epoch > legacy_before_state.group_authority_epoch);
    legacy_after = {};
    CHECK(registry.finish_graph_dispatch(&execution));

    auto finalization_registry = std::make_unique<ggml_cuda_moe_grouped_context>(&fixture.owner, 0);
    CHECK(finalization_registry->replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    ggml_cuda_moe_graph_plan finalization_plan;
    ggml_cuda_moe_graph_execution finalization_execution;
    finalization_registry->compile_graph_plan(complete_graph, 62, &finalization_plan, &finalization_execution);
    CHECK(finalization_execution.resolve_streams(candidate_test_graph_stream, reinterpret_cast<void *>(uintptr_t{1})));
    CHECK(finalization_registry->begin_graph_dispatch(&finalization_execution, true));
    ggml_cuda_moe_candidate_group_key failed_keys[2];
    ggml_cuda_moe_grouped_acquisition failed_resources[2];
    ggml_cuda_moe_grouped_transaction failed_transactions[2];
    CHECK(finalization_registry->find_down_group_key(fused_down, &failed_keys[0]));
    CHECK(finalization_registry->find_down_group_key(separate_down, &failed_keys[1]));
    for (uint32_t i = 0; i < 2; ++i) {
        CHECK(finalization_registry->acquire_group_resources(failed_keys[i], &failed_resources[i]));
        CHECK(finalization_registry->begin_group_transaction(failed_resources[i], &failed_transactions[i]));
    }
    auto * failed_fused = finalization_execution.find_group(fused_gate_up_node, nullptr);
    auto * failed_separate = finalization_execution.find_group(separate_gate_node, nullptr);
    CHECK(failed_fused != nullptr && failed_separate != nullptr && failed_fused != failed_separate);
    failed_fused->transaction = failed_transactions[0];
    failed_fused->state = GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_ACTIVE;
    failed_separate->transaction = failed_transactions[1];
    failed_separate->state = GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_ACTIVE;
    CHECK(!finalization_registry->finish_graph_dispatch(&finalization_execution));
    for (uint32_t i = 0; i < 2; ++i) {
        CHECK(!finalization_registry->get_group_resources(failed_resources[i], nullptr));
        CHECK(!finalization_registry->end_group_transaction(failed_transactions[i]));
    }
    CHECK(finalization_execution.find_authority(fused_gate_up_node) == nullptr);
    CHECK(finalization_execution.find_authority(separate_gate_node) == nullptr);
    CHECK(finalization_registry->replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    finalization_registry->shutdown();
    finalization_registry.reset();

    registry.compile_graph_plan(complete_graph, 62, &plan, &execution);
    CHECK(execution.resolve_streams(candidate_test_graph_stream, reinterpret_cast<void *>(uintptr_t{1})));
    CHECK(registry.begin_graph_dispatch(&execution, true));
    std::atomic<bool> authority_replacement_started{false};
    std::atomic<bool> authority_replacement_done{false};
    std::thread authority_replacement([&]() {
        authority_replacement_started.store(true, std::memory_order_release);
        CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        authority_replacement_done.store(true, std::memory_order_release);
    });
    while (!authority_replacement_started.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    do {
        std::this_thread::yield();
    } while (registry.bind_graph_plan(complete_graph, 62, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, plan, reused.get()));
    CHECK(!authority_replacement_done.load(std::memory_order_acquire));
    CHECK(!registry.finish_graph_dispatch(&execution));
    authority_replacement.join();
    CHECK(authority_replacement_done.load(std::memory_order_acquire));

    {
        ggml_cuda_moe_graph_plan scoped_plan;
        ggml_cuda_moe_graph_execution scoped_execution;
        registry.compile_graph_plan(complete_graph, 63, &scoped_plan, &scoped_execution);
        CHECK(scoped_execution.resolve_streams(candidate_test_graph_stream, reinterpret_cast<void *>(uintptr_t{1})));
        CHECK(registry.begin_graph_dispatch(&scoped_execution, true));
    }
    CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);

    auto terminal = std::make_unique<ggml_cuda_moe_grouped_context>(&fixture.owner, 0);
    CHECK(terminal->replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
    ggml_cuda_moe_graph_plan terminal_plan;
    ggml_cuda_moe_graph_execution terminal_execution;
    ggml_cuda_moe_graph_execution terminal_probe;
    terminal->compile_graph_plan(complete_graph, 64, &terminal_plan, &terminal_execution);
    CHECK(terminal_execution.resolve_streams(candidate_test_graph_stream, reinterpret_cast<void *>(uintptr_t{1})));
    CHECK(terminal->begin_graph_dispatch(&terminal_execution, true));
    std::atomic<bool> authority_shutdown_started{false};
    std::atomic<bool> authority_shutdown_done{false};
    std::thread authority_shutdown([&]() {
        authority_shutdown_started.store(true, std::memory_order_release);
        terminal->shutdown();
        authority_shutdown_done.store(true, std::memory_order_release);
    });
    while (!authority_shutdown_started.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    do {
        std::this_thread::yield();
    } while (terminal->bind_graph_plan(complete_graph, 64, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, terminal_plan, &terminal_probe));
    CHECK(!authority_shutdown_done.load(std::memory_order_acquire));
    CHECK(!terminal->finish_graph_dispatch(&terminal_execution));
    authority_shutdown.join();
    CHECK(authority_shutdown_done.load(std::memory_order_acquire));
    terminal.reset();

    if (benchmark) {
        CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
        std::shared_ptr<ggml_cuda_moe_graph_plan> benchmark_plan;
        auto prepared = std::make_unique<ggml_cuda_moe_graph_execution>();
        CHECK(registry.prepare_graph_execution(
            complete_graph, 60, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &benchmark_plan, prepared.get(),
            complete_coverage.epoch, complete_coverage.nodes,
            complete_coverage.mmid_count, complete_coverage.mmid_fingerprint) == GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        const ggml_cuda_moe_graph_plan * stable_plan = benchmark_plan.get();
        constexpr uint32_t n_reuses = 200000;
        uint32_t reused_count = 0;
        const auto begin = std::chrono::steady_clock::now();
        for (uint32_t i = 0; i < n_reuses; ++i) {
            reused_count += registry.prepare_graph_execution(
                complete_graph, 61 + i, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &benchmark_plan, prepared.get(),
                complete_coverage.epoch, complete_coverage.nodes,
                complete_coverage.mmid_count, complete_coverage.mmid_fingerprint) ==
                GGML_CUDA_MOE_GRAPH_PREPARE_REUSED ? 1 : 0;
        }
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin).count();
        CHECK(reused_count == n_reuses && benchmark_plan.get() == stable_plan && prepared->size() == 2);
        fprintf(stderr, "test-moe-cache: grouped graph plan %.1f ns/reuse with fresh UIDs, recompiles=0/%u\n",
            static_cast<double>(elapsed) / n_reuses, n_reuses);

        std::shared_ptr<ggml_cuda_moe_graph_plan> padded_benchmark_plan;
        CHECK(registry.prepare_graph_execution(
            padded_graph, 100, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &padded_benchmark_plan, prepared.get(),
            padded_coverage.epoch, padded_coverage.nodes,
            padded_coverage.mmid_count, padded_coverage.mmid_fingerprint) ==
            GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
        const ggml_cuda_moe_graph_plan * stable_padded_plan = padded_benchmark_plan.get();
        constexpr uint32_t n_padded_reuses = 100000;
        uint32_t padded_reused_count = 0;
        const auto padded_begin = std::chrono::steady_clock::now();
        for (uint32_t i = 0; i < n_padded_reuses; ++i) {
            padded_reused_count += registry.prepare_graph_execution(
                padded_graph, 101 + i, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &padded_benchmark_plan, prepared.get(),
                padded_coverage.epoch, padded_coverage.nodes,
                padded_coverage.mmid_count, padded_coverage.mmid_fingerprint) ==
                GGML_CUDA_MOE_GRAPH_PREPARE_REUSED ? 1 : 0;
        }
        const auto padded_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - padded_begin).count();
        const double padded_ns = static_cast<double>(padded_elapsed) / n_padded_reuses;
        CHECK(padded_reused_count == n_padded_reuses && padded_benchmark_plan.get() == stable_padded_plan && prepared->size() == 2);
        fprintf(stderr, "test-moe-cache: grouped padded graph plan %.1f ns/reuse with %u-node inventory scan, recompiles=0/%u\n",
            padded_ns, n_padding_nodes + 9, n_padded_reuses);
        CHECK(padded_ns < 100000.0);
    }

    fprintf(stderr, "test-moe-cache: grouped graph preflight OK\n");
}

void test_grouped_graph_mixed_phase() {
    candidate_test_fixture fixture;
    const int64_t gate_up_ne[] = {256, 512, 4};
    const int64_t down_ne[] = {256, 256, 4};
    ggml_tensor * prefill_gate_up = fixture.cached_tensor(GGML_TYPE_Q4_0, 3, gate_up_ne);
    ggml_tensor * prefill_down = fixture.cached_tensor(GGML_TYPE_Q4_0, 3, down_ne);
    ggml_tensor * decode_gate_up = fixture.cached_tensor(GGML_TYPE_Q4_0, 3, gate_up_ne);
    ggml_tensor * decode_down = fixture.cached_tensor(GGML_TYPE_Q4_0, 3, down_ne);
    ggml_tensor * late_gate_up = fixture.cached_tensor(GGML_TYPE_Q4_0, 3, gate_up_ne);
    ggml_tensor * late_down = fixture.cached_tensor(GGML_TYPE_Q4_0, 3, down_ne);
    ggml_tensor * uncovered = fixture.cached_tensor(GGML_TYPE_Q4_0, 3, gate_up_ne);
    std::array<ggml_backend_moe_candidate_bank_v1, 2> prefill_banks = {{
        {prefill_gate_up, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, 0},
        {prefill_down, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
    }};
    std::array<ggml_backend_moe_candidate_bank_v1, 2> decode_banks = {{
        {decode_gate_up, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, 0},
        {decode_down, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
    }};
    std::array<ggml_backend_moe_candidate_bank_v1, 2> late_banks = {{
        {late_gate_up, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT, 0},
        {late_down, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT, 0},
    }};
    std::array<ggml_backend_moe_candidate_group_v1, 3> groups = {{
        {prefill_banks.data(), prefill_banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 0, 0},
        {decode_banks.data(), decode_banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 0, 0},
        {late_banks.data(), late_banks.size(), GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP, 0, 0},
    }};
    const auto snapshot = candidate_snapshot(12, groups.data(), groups.size());
    ggml_cuda_moe_grouped_context registry(&fixture.owner, 0);
    CHECK(registry.replace(&snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);

    const candidate_route prefill_route = candidate_top_k_route(fixture, 4, 2, 4);
    const candidate_route decode_route = candidate_top_k_route(fixture, 4, 2);
    const candidate_route late_route = candidate_top_k_route(fixture, 4, 2);
    ggml_tensor * prefill_gate_up_reader = candidate_mmid(fixture, prefill_gate_up, prefill_route.ids);
    ggml_tensor * prefill_down_reader = candidate_mmid(fixture, prefill_down, prefill_route.ids);
    ggml_tensor * decode_gate_up_reader = candidate_mmid(fixture, decode_gate_up, decode_route.ids);
    ggml_tensor * decode_down_reader = candidate_mmid(fixture, decode_down, decode_route.ids);
    ggml_tensor * late_gate_up_reader = candidate_mmid(fixture, late_gate_up, late_route.ids);
    ggml_tensor * late_down_reader = candidate_mmid(fixture, late_down, late_route.ids);
    late_gate_up_reader->op = GGML_OP_DUP;
    late_down_reader->op = GGML_OP_DUP;
    ggml_tensor * cross_phase_down_reader = candidate_mmid(fixture, prefill_down, decode_route.ids);
    ggml_tensor * uncovered_reader = candidate_mmid(fixture, uncovered, prefill_route.ids);
    ggml_cgraph * mixed_graph = candidate_graph(fixture, {
        prefill_route.root, prefill_route.ids, decode_route.root, decode_route.ids, late_route.root, late_route.ids,
        prefill_gate_up_reader, prefill_down_reader, decode_gate_up_reader, decode_down_reader,
        late_gate_up_reader, late_down_reader,
    });
    const auto mixed_coverage = candidate_certify_graph(registry, mixed_graph);
    CHECK(mixed_coverage.mmid_count == 4);
    auto stale_plan = std::make_shared<ggml_cuda_moe_graph_plan>();
    ggml_cuda_moe_graph_execution execution;
    registry.compile_graph_plan(
        mixed_graph, 701, stale_plan.get(), &execution, mixed_coverage.epoch, mixed_coverage.nodes,
        mixed_coverage.mmid_count, mixed_coverage.mmid_fingerprint);
    CHECK(stale_plan->size() == 2 && execution.size() == 2);
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_PREFILL_LEGACY && execution.requires_dispatch());
    CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_prefill_reason(*stale_plan, 0));
    CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_execution_reason(*stale_plan, 1));
    CHECK(execution.find(prefill_gate_up_reader, nullptr) && execution.find(prefill_down_reader, nullptr));
    CHECK(execution.find(decode_gate_up_reader, nullptr) && execution.find(decode_down_reader, nullptr));
    CHECK(registry.begin_graph_dispatch(&execution, true));
    const auto * prefill_authority = execution.find_authority(prefill_gate_up_reader);
    const auto * decode_authority = execution.find_authority(decode_gate_up_reader);
    CHECK(prefill_authority != nullptr && prefill_authority->authority() == GGML_CUDA_MOE_GROUP_AUTHORITY_LEGACY);
    CHECK(decode_authority != nullptr && decode_authority->authority() == GGML_CUDA_MOE_GROUP_AUTHORITY_LEGACY);
    CHECK(registry.finish_graph_dispatch(&execution));
    ggml_cuda_moe_graph_execution reused;
    CHECK(registry.bind_graph_plan(
        mixed_graph, 702, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, *stale_plan, &reused,
        mixed_coverage.epoch, mixed_coverage.nodes, mixed_coverage.mmid_count, mixed_coverage.mmid_fingerprint));
    CHECK(reused.find(prefill_down_reader, nullptr) && reused.find(decode_down_reader, nullptr));

    late_gate_up_reader->op = GGML_OP_MUL_MAT_ID;
    late_down_reader->op = GGML_OP_MUL_MAT_ID;
    candidate_rebuild_graph_uses(mixed_graph);
    CHECK(!registry.bind_graph_plan(
        mixed_graph, 702, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, *stale_plan, &reused,
        mixed_coverage.epoch, mixed_coverage.nodes, mixed_coverage.mmid_count, mixed_coverage.mmid_fingerprint));
    CHECK(reused.size() == 0 && !reused.find(prefill_down_reader, nullptr) && !reused.find(decode_down_reader, nullptr) &&
        !reused.find(late_gate_up_reader, nullptr) && !reused.find(late_down_reader, nullptr));
    const ggml_cuda_moe_graph_plan * stale_plan_ptr = stale_plan.get();
    CHECK(registry.prepare_graph_execution(
        mixed_graph, 7021, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, &stale_plan, &reused,
        mixed_coverage.epoch, mixed_coverage.nodes, mixed_coverage.mmid_count, mixed_coverage.mmid_fingerprint) ==
        GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
    CHECK(stale_plan.get() != stale_plan_ptr && stale_plan->size() == 3 && reused.size() == 3 &&
        stale_plan->coverage_diagnostics().cached_mmid == 6);
    CHECK(reused.find(prefill_gate_up_reader, nullptr) && reused.find(prefill_down_reader, nullptr) &&
        reused.find(decode_gate_up_reader, nullptr) && reused.find(decode_down_reader, nullptr) &&
        reused.find(late_gate_up_reader, nullptr) && reused.find(late_down_reader, nullptr));

    ggml_cuda_moe_graph_plan plan;

    ggml_cgraph * incomplete_prefill_graph = candidate_graph(fixture, {
        prefill_route.root, prefill_route.ids, decode_route.root, decode_route.ids,
        prefill_gate_up_reader, decode_gate_up_reader, decode_down_reader,
    });
    const auto incomplete_prefill_coverage = candidate_certify_graph(registry, incomplete_prefill_graph);
    registry.compile_graph_plan(
        incomplete_prefill_graph, 703, &plan, &execution,
        incomplete_prefill_coverage.epoch, incomplete_prefill_coverage.nodes,
        incomplete_prefill_coverage.mmid_count, incomplete_prefill_coverage.mmid_fingerprint);
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR);
    CHECK(execution.requires_dispatch());
    CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_missing_role_reason(plan, 0));
    CHECK(execution.rejects_cached_mmid(prefill_gate_up_reader) && execution.rejects_cached_mmid(decode_down_reader));

    ggml_cgraph * cross_phase_group_graph = candidate_graph(fixture, {
        prefill_route.root, prefill_route.ids, decode_route.root, decode_route.ids,
        prefill_gate_up_reader, cross_phase_down_reader,
    });
    const auto cross_phase_group_coverage = candidate_certify_graph(registry, cross_phase_group_graph);
    registry.compile_graph_plan(
        cross_phase_group_graph, 704, &plan, &execution,
        cross_phase_group_coverage.epoch, cross_phase_group_coverage.nodes,
        cross_phase_group_coverage.mmid_count, cross_phase_group_coverage.mmid_fingerprint);
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR && execution.requires_dispatch());

    ggml_cgraph * uncovered_graph = candidate_graph(fixture, {
        prefill_route.root, prefill_route.ids, decode_route.root, decode_route.ids,
        prefill_gate_up_reader, prefill_down_reader, decode_gate_up_reader, decode_down_reader, uncovered_reader,
    });
    const auto uncovered_coverage = candidate_certify_graph(registry, uncovered_graph);
    registry.compile_graph_plan(
        uncovered_graph, 705, &plan, &execution, uncovered_coverage.epoch, uncovered_coverage.nodes,
        uncovered_coverage.mmid_count, uncovered_coverage.mmid_fingerprint);
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR && execution.rejects_cached_mmid(uncovered_reader));

    ggml_cgraph * pure_prefill_graph = candidate_graph(fixture, {
        prefill_route.root, prefill_route.ids, prefill_gate_up_reader, prefill_down_reader,
    });
    const auto pure_prefill_coverage = candidate_certify_graph(registry, pure_prefill_graph);
    registry.compile_graph_plan(
        pure_prefill_graph, 706, &plan, &execution, pure_prefill_coverage.epoch, pure_prefill_coverage.nodes,
        pure_prefill_coverage.mmid_count, pure_prefill_coverage.mmid_fingerprint);
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_PREFILL_LEGACY && execution.size() == 1);
    CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_prefill_reason(plan, 0));

    ggml_cgraph * pure_decode_graph = candidate_graph(fixture, {
        decode_route.root, decode_route.ids, decode_gate_up_reader, decode_down_reader,
    });
    const auto pure_decode_coverage = candidate_certify_graph(registry, pure_decode_graph);
    registry.compile_graph_plan(
        pure_decode_graph, 707, &plan, &execution, pure_decode_coverage.epoch, pure_decode_coverage.nodes,
        pure_decode_coverage.mmid_count, pure_decode_coverage.mmid_fingerprint);
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_DECODE_GROUPED && execution.size() == 1);
    CHECK(ggml_cuda_moe_grouped_context_test_access::graph_group_has_eligible_reason(plan, 0));
    CHECK(execution.resolve_streams(candidate_test_graph_stream, reinterpret_cast<void *>(uintptr_t{1})));
    CHECK(execution.has_stream_grouped_candidate());

    ggml_cgraph * incomplete_decode_graph = candidate_graph(fixture, {
        decode_route.root, decode_route.ids, decode_gate_up_reader,
    });
    const auto incomplete_decode_coverage = candidate_certify_graph(registry, incomplete_decode_graph);
    registry.compile_graph_plan(
        incomplete_decode_graph, 708, &plan, &execution,
        incomplete_decode_coverage.epoch, incomplete_decode_coverage.nodes,
        incomplete_decode_coverage.mmid_count, incomplete_decode_coverage.mmid_fingerprint);
    CHECK(execution.outcome() == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR && execution.rejects_cached_mmid(decode_gate_up_reader));

    fprintf(stderr, "test-moe-cache: mixed phase graph repair OK\n");
}
